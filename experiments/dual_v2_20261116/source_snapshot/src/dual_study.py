"""Prospective dual-model ablation. No human labels or automatic failed-job reruns."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import statistics
from uuid import uuid4

from src.ai_judge import parse_four, prompt_for
from src.annotation import agreement_with_ci
from src.app import run
from src.artifacts import provenance, save_result
from src.client import QwenClient, error_details
from src.config import PROJECT_ROOT, Settings
from src.experiments import dump, load_jobs, make_plan
from src.metrics import bootstrap_mean_interval
from src.network_guard import JournalRecorder, append_json, preflight, provider_failure
from src.nodes import RUBRIC_VERSION
from src.schemas import DIMENSIONS, WEIGHTS
from src.telemetry import CALL_CONTEXT, ObservedModel, token_summary
from src.utils import essay_digest


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []


def view(path: Path, value: dict) -> None:
    """Only derived views may be refreshed; plans, results, journals stay immutable."""
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def prepare(dataset: Path, destination: Path) -> dict:
    settings = Settings(model="qwen3.8-27b", max_rounds=2, max_attempts=1,
        timeout_seconds=180, retry_delay_seconds=0, stop_policy="fixed")
    experiment = make_plan(dataset, "ablation", settings, repeats=1, seed=20261116)
    judge_bound = 4 * len(experiment["samples"])
    if len(experiment["samples"]) != 10 or experiment["max_call_attempts"] + judge_bound != 350:
        raise ValueError("This authorization is exactly ten samples and at most 350 inference attempts")
    policy = {"version": "dual-model-v2", "prepared_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scheme": "dual_model_agreement_no_human_review",
        "human_review_required": False, "rubric_version": RUBRIC_VERSION,
        "primary_comparison": "all_feedback_vs_no_feedback", "max_new_inference_attempts": 350,
        "workflow_call_bound": 310, "second_judge_call_bound": 40,
        "first_model": settings.model, "second_model": "kimi-k3",
        "judge_settings": Settings(model="kimi-k3", temperature=0, max_rounds=1,
            max_attempts=1, timeout_seconds=180, max_output_tokens=4096).model_dump(),
        "retry_policy": "No hidden HTTP retry; no failed/started task replay. Explicit resume skips archived failures.",
        "network_policy": "Reuse shared verified HTTPS connection pools; preflight before each sample block; pause batch after any provider failure. Max four already dispatched calls may finish.",
        "interpretation": "AI model agreement and second-model assessed text changes; not accuracy, human agreement or objective quality. Model identity and single/joint prompt layout are confounded.",
        "provenance": experiment["provenance"]}
    destination.mkdir(parents=True, exist_ok=False)
    dump(destination / "batch_plan.json", policy)
    (destination / "ablation").mkdir()
    dump(destination / "ablation/plan.json", experiment)
    snapshot = destination / "source_snapshot"
    for folder in ("src", "tests", "docs"):
        shutil.copytree(PROJECT_ROOT / folder, snapshot / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("requirements.txt", "requirements-dev.txt", "requirements-lock.txt", "pytest.ini", "README.md"):
        if (PROJECT_ROOT / name).is_file():
            shutil.copy2(PROJECT_ROOT / name, snapshot / name)
    view(destination / "status.json", {"status": "prepared", "calls_used": 0, "human_review": "not_in_protocol"})
    return policy


def count_calls(directory: Path) -> list[dict]:
    return [c for file in directory.glob("journals/*.jsonl") for c in lines(file)]


def judge_items(directory: Path, samples: list[dict], jobs: list[dict]) -> list[dict]:
    path = directory / "judge_plan.json"
    if path.exists():
        return read(path)["items"]
    unique = {}
    def add(topic, essay, target):
        unique.setdefault((topic, essay), []).append(target)
    for sample in samples:
        add(sample["topic"], sample["essay"], {"kind": "original", "sample_id": sample["sample_id"]})
    for job in jobs:
        result = job["result"]
        if result.get("essay"):
            add(result["topic"], result["essay"], {"kind": "final", "job_id": job["job_id"],
                "sample_id": job["sample_id"], "variant": job["variant"],
                "workflow_success": result.get("final_score") is not None})
    items = [{"blind_id": "ai-" + uuid4().hex[:16], "topic": topic, "essay": essay,
              "essay_sha256": essay_digest(essay), "targets": targets} for (topic, essay), targets in unique.items()]
    if len(items) > 40:
        raise ValueError("Judge text bound exceeded")
    random.Random(20261116).shuffle(items)
    dump(path, {"items": items, "prompt_layout": "joint_four_dimensions", "rubric_version": RUBRIC_VERSION})
    return items


def execute(directory: Path, *, execute_live: bool, max_calls: int, resume: bool = False,
            checker=None, client_factory=None, runner=None) -> dict:
    directory = Path(directory)
    checker, client_factory, runner = checker or preflight, client_factory or QwenClient, runner or run
    policy = read(directory / "batch_plan.json")
    if not execute_live or type(max_calls) is not int or max_calls != 350:
        raise ValueError("Explicit live execution with the authorized limit 350 is required")
    current = provenance()
    for key in ("source_sha256", "dependencies", "python"):
        if current[key] != policy["provenance"][key]:
            raise ValueError("Source/runtime changed after planning; do not mix versions")
    if policy["rubric_version"] != RUBRIC_VERSION:
        raise ValueError("Rubric changed")
    if (directory / "sessions.jsonl").exists() and not resume:
        raise ValueError("Already started; explicit --resume required, no replay")
    lock = directory / "execution.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("Exclusive batch execution; stale lock requires crash/journal inspection.\n")
    clients = []
    state = {"status": "running", "human_review": "not_in_protocol"}
    try:
        append_json(directory / "sessions.jsonl", {"time": datetime.now(timezone.utc).isoformat(), "resume": resume})
        for name in ("started", "journals", "judge_results"):
            (directory / name).mkdir(exist_ok=True)
        plan, completed = load_jobs(directory / "ablation")
        if (plan["max_call_attempts"] != 310 or len(plan["jobs"]) != 40
                or plan["base_settings"]["model"] != policy["first_model"]
                or policy["second_model"] != "kimi-k3"
                or policy["judge_settings"]["model"] != policy["second_model"]):
            raise ValueError("Authorized model/budget policy changed")
        done = {j["job_id"] for j in completed}
        # Saved result but missing index is ambiguous: never replay automatically.
        pending = [j for j in plan["jobs"] if j["job_id"] not in done]
        for job in pending:
            if (directory / "started" / (job["job_id"] + ".json")).exists():
                state.update(status="paused_ambiguous_interrupted_task", task=job["job_id"])
                return state
        initial_calls = count_calls(directory)
        if len(initial_calls) > max_calls:
            raise ValueError("Existing journal exceeds authorized budget")
        settings = Settings.model_validate(plan["base_settings"])
        checked = checker(trust_env=settings.trust_env)
        append_json(directory / "network_checks.jsonl", checked)
        if not checked["ready"]:
            state["status"] = "paused_network_preflight"
            return state
        samples = {s["sample_id"]: s for s in plan["samples"]}
        if pending:
            generator = client_factory(settings)
            evaluator = client_factory(settings.model_copy(update={"temperature": settings.evaluator_temperature}))
            clients.extend([generator, evaluator])
            last_sample = None
            for job in pending:
                if job["sample_id"] != last_sample and last_sample is not None:
                    checked = checker(trust_env=settings.trust_env)
                    append_json(directory / "network_checks.jsonl", checked)
                    if not checked["ready"]:
                        state["status"] = "paused_network_preflight"
                        return state
                last_sample = job["sample_id"]
                if len(count_calls(directory)) + job["max_call_attempts"] > max_calls - policy["second_judge_call_bound"]:
                    raise ValueError("Workflow budget exhausted")
                dump(directory / "started" / (job["job_id"] + ".json"), {"task": job["job_id"], "bound": job["max_call_attempts"]})
                recorder = JournalRecorder(directory / "journals" / (job["job_id"] + ".jsonl"))
                sample = samples[job["sample_id"]]
                result = runner(sample["topic"], essay=sample["essay"], mode=job["mode"],
                    settings=Settings.model_validate(job["settings"]), generator=generator, evaluator=evaluator, recorder=recorder)
                if len(result["calls"]) != len(recorder.snapshot()) or len(result["calls"]) > job["max_call_attempts"]:
                    raise ValueError("Recorder or per-job bound mismatch")
                result["evidence_type"] = "live_dual_model_exploration_no_human_validation"
                result["experiment"] = {k: job[k] for k in ("job_id", "sample_id", "repeat", "variant")}
                path = save_result(result, directory / "ablation" / job["job_id"])
                append_json(directory / "ablation/index.jsonl", {**result["experiment"], "result": str(path.relative_to(directory / "ablation"))})
                print(job["job_id"], job["sample_id"], job["variant"], result.get("exit_reason") or "evaluated", flush=True)
                view(directory / "status.json", {"status": "running", "latest_job": job["job_id"], "calls_used": len(count_calls(directory))})
                if provider_failure(result["calls"]):
                    state.update(status="paused_provider_failure", task=job["job_id"])
                    return state
        _, completed = load_jobs(directory / "ablation")
        items = judge_items(directory, plan["samples"], completed)
        if (len(items) > 40 or len({i["blind_id"] for i in items}) != len(items)
                or any(essay_digest(i["essay"]) != i["essay_sha256"] for i in items)):
            raise ValueError("Frozen judge items changed")
        judge_settings = Settings.model_validate(policy["judge_settings"])
        judge = client_factory(judge_settings)
        clients.append(judge)
        for item in items:
            result_path = directory / "judge_results" / (item["blind_id"] + ".json")
            if result_path.exists():
                continue
            start_path = directory / "started" / (item["blind_id"] + ".json")
            if start_path.exists():
                state.update(status="paused_ambiguous_interrupted_task", task=item["blind_id"])
                return state
            if len(count_calls(directory)) >= max_calls:
                raise ValueError("Total inference budget exhausted")
            dump(start_path, {"task": item["blind_id"], "bound": 1})
            recorder = JournalRecorder(directory / "journals" / (item["blind_id"] + ".jsonl"))
            model = ObservedModel(judge, "second_ai_judge", judge_settings.model, recorder)
            row = {"blind_id": item["blind_id"], "rater_type": "ai", "model_requested": judge_settings.model,
                   "essay_sha256": item["essay_sha256"], "human_attested": False}
            raw = None
            token = CALL_CONTEXT.set({"item_id": item["blind_id"], "attempt": 1, "stage": "four_dimension_judge"})
            try:
                raw = model.invoke(prompt_for(item))
                row.update(status="ok", judgments=parse_four(raw, item["essay"]))
            except Exception as exc:
                row.update(status="provider_error" if raw is None else "parse_error",
                           error_details=error_details(exc) if raw is None else {"error": "invalid_json_or_nonverbatim_evidence"})
            finally:
                CALL_CONTEXT.reset(token)
            row["raw_response"] = raw
            dump(result_path, row)
            print(item["blind_id"], row["status"], flush=True)
            if row["status"] == "provider_error":
                state.update(status="paused_provider_failure", task=item["blind_id"])
                return state
        state["status"] = "completed"
        return state
    except BaseException:
        state["status"] = "paused_local_error_or_interrupt"
        raise
    finally:
        for client in clients:
            client.close()
        state.update(calls_used=len(count_calls(directory)), updated_at=datetime.now(timezone.utc).isoformat())
        view(directory / "status.json", state)
        lock.unlink()
        analyze(directory)


def analyze(directory: Path) -> dict:
    directory = Path(directory)
    plan, jobs = load_jobs(directory / "ablation")
    judge_plan = read(directory / "judge_plan.json") if (directory / "judge_plan.json").exists() else {"items": []}
    originals, finals, first_scores, paired = {}, {}, {}, []
    judge_rows = {p.stem: read(p) for p in (directory / "judge_results").glob("*.json")}
    for item in judge_plan["items"]:
        row = judge_rows.get(item["blind_id"], {})
        if row.get("status") != "ok":
            continue
        scores = {d: row["judgments"][d]["score"] for d in DIMENSIONS}
        scores["weighted_total"] = sum(WEIGHTS[d] * scores[d] for d in DIMENSIONS)
        for target in item["targets"]:
            if target["kind"] == "original":
                originals[target["sample_id"]] = scores
            else:
                finals[target["job_id"]] = scores
    for job in jobs:
        result = job["result"]
        if result.get("final_score") is None:
            continue
        scores = {d: result[d + "_result"]["judgment"]["score"] for d in DIMENSIONS}
        scores["weighted_total"] = result["final_score"]
        first_scores[job["job_id"]] = scores
        if job["job_id"] in finals:
            paired.append(job)
    agreement = {}
    original_agreement = {}
    for dimension in (*DIMENSIONS, "weighted_total"):
        # One averaged pair per independent original, avoiding 40 pseudoreplicates.
        pairs = []
        for sample in plan["samples"]:
            subset = [j for j in paired if j["sample_id"] == sample["sample_id"]]
            if subset:
                pairs.append((statistics.mean(first_scores[j["job_id"]][dimension] for j in subset),
                              statistics.mean(finals[j["job_id"]][dimension] for j in subset)))
        agreement[dimension] = agreement_with_ci(pairs)
        original_pairs = []
        for sample in plan["samples"]:
            sid = sample["sample_id"]
            if sid not in originals:
                continue
            observations = []
            for job in jobs:
                if job["sample_id"] != sid:
                    continue
                first_round = (job["result"].get("history") or [{}])[0]
                if dimension == "weighted_total":
                    value = first_round.get("final_score")
                else:
                    value = next((r["judgment"]["score"] for r in first_round.get("results", [])
                                  if r["dimension"] == dimension and r["status"] == "ok"), None)
                if value is not None:
                    observations.append(value)
            if observations:
                original_pairs.append((statistics.mean(observations), originals[sid][dimension]))
        original_agreement[dimension] = agreement_with_ci(original_pairs)
    def changes(values):
        return {"n": len(values), "mean_delta": statistics.mean(values) if values else None,
                "ci95_sample_bootstrap": bootstrap_mean_interval(values),
                "improved": sum(x > 1e-12 for x in values), "tied": sum(abs(x) <= 1e-12 for x in values),
                "regressed": sum(x < -1e-12 for x in values), "tie_tolerance": 1e-12}
    variants = {}
    for variant in ("no_rewrite", "no_feedback", "relevance_feedback", "all_feedback"):
        subset = [j for j in jobs if j["variant"] == variant]
        success = [j for j in subset if j["job_id"] in first_scores]
        delta = [finals[j["job_id"]]["weighted_total"] - originals[j["sample_id"]]["weighted_total"]
                 for j in success if j["job_id"] in finals and j["sample_id"] in originals]
        variants[variant] = {"attempted": len(subset), "workflow_success": len(success),
            "workflow_failed": len(subset) - len(success), "not_attempted": len(plan["samples"]) - len(subset),
            "ai_changes_vs_original": changes(delta)}
    comparisons = {}
    lookup = {(j["sample_id"], j["variant"]): j for j in jobs}
    for left, right in [("all_feedback", "no_feedback"), ("all_feedback", "relevance_feedback"),
                        ("no_feedback", "no_rewrite"), ("relevance_feedback", "no_rewrite"), ("all_feedback", "no_rewrite")]:
        valid, failed, missing, unrated = [], 0, 0, 0
        for sample in plan["samples"]:
            a, b = lookup.get((sample["sample_id"], left)), lookup.get((sample["sample_id"], right))
            if a is None or b is None:
                missing += 1
            elif a["job_id"] not in first_scores or b["job_id"] not in first_scores:
                failed += 1
            elif a["job_id"] not in finals or b["job_id"] not in finals:
                unrated += 1
            else:
                valid.append({"sample_id": sample["sample_id"],
                    "first_model_delta": first_scores[a["job_id"]]["weighted_total"] - first_scores[b["job_id"]]["weighted_total"],
                    "second_model_delta": finals[a["job_id"]]["weighted_total"] - finals[b["job_id"]]["weighted_total"]})
        comparisons[left + "_vs_" + right] = {"per_sample": valid,
            "first_model": changes([x["first_model_delta"] for x in valid]),
            "second_model": changes([x["second_model_delta"] for x in valid]),
            "failed_workflow_pairs": failed, "missing_pairs": missing, "unrated_successful_pairs": unrated}
    calls = count_calls(directory)
    report = {"protocol": "dual-model-v2", "human_review": "cancelled_by_user_not_in_protocol",
        "evidence_type": "small_exploratory_dual_model_comparison", "workflow_attempted": len(jobs),
        "workflow_planned": len(plan["jobs"]), "judge_attempted": len(judge_rows),
        "judge_successful": sum(r["status"] == "ok" for r in judge_rows.values()),
        "judge_failed": sum(r["status"] != "ok" for r in judge_rows.values()),
        "model_agreement_by_original_cluster_not_accuracy": agreement,
        "original_text_model_agreement_not_accuracy": original_agreement,
        "agreement_definitions": {"original_text_model_agreement_not_accuracy":
            "One pair per original: mean of first-round first-model observations across conditions versus one second-model original rating. Effective counts differ after failures.",
            "model_agreement_by_original_cluster_not_accuracy":
            "One pair per original cluster: average scores across successful final condition texts. Not forty independent samples; excludes unsuccessful/unrated workflows."},
        "successful_final_text_pairs": len(paired), "variants": variants, "comparisons": comparisons,
        "telemetry": token_summary(calls), "total_authorized_inference_attempts": 350,
        "warning": "No human review or ground truth. Models differ in identity and prompt layout. N=10 synthetic dev originals, one rewrite realization. Retain failures and negative deltas; no full-system validation or speedup claim."}
    view(directory / "DUAL_MODEL_REPORT.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--dataset", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("run"); p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--execute", action="store_true"); p.add_argument("--max-calls", type=int, required=True); p.add_argument("--resume", action="store_true")
    p = sub.add_parser("analyze"); p.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        policy = prepare(args.dataset, args.output_dir)
        print(json.dumps({"status": "prepared", "bound": policy["max_new_inference_attempts"]}))
    elif args.command == "run":
        print(json.dumps(execute(args.run_dir, execute_live=args.execute, max_calls=args.max_calls, resume=args.resume)))
    else:
        report = analyze(args.run_dir)
        print(json.dumps({"calls": report["telemetry"]["call_count"], "jobs": report["workflow_attempted"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
