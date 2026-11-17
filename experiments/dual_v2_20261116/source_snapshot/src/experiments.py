"""EssayLoop experiment plans, bounded execution and offline analysis.

Planning is free. Live runs require --execute and an explicit call budget.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

from src.app import run
from src.artifacts import provenance, save_result
from src.config import Settings
from src.costs import calculate_cost
from src.datasets import load_dataset
from src.metrics import bootstrap_mean_interval, reliability_report
from src.schemas import DIMENSIONS, WEIGHTS
from src.plans import variants_for, call_bound, sample_digest, validate_plan
from src.telemetry import token_summary


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)


def make_plan(dataset, kind, settings, repeats=3, seed=42, split="dev"):
    if kind not in {"reliability", "ablation", "efficiency", "early_exit"}:
        raise ValueError("Unknown experiment kind")
    if not 1 <= repeats <= 100:
        raise ValueError("repeats must be 1..100")
    samples, digest = load_dataset(dataset)
    samples = [s for s in samples if s["split"] == split]
    if not samples:
        raise ValueError("No samples in requested split")
    variants = variants_for(kind)
    if kind in {"ablation", "early_exit"} and settings.max_rounds < 2:
        raise ValueError("Revision experiments need at least two rounds")
    rng, jobs = random.Random(seed), []
    pairs = [(s["sample_id"], repeat) for s in samples for repeat in range(repeats)]
    rng.shuffle(pairs)
    for sample_id, repeat in pairs:
        order = list(variants)
        rng.shuffle(order)
        for variant in order:
            config = dict(variants[variant])
            mode = config.pop("mode")
            effective = Settings.model_validate({**settings.model_dump(), **config})
            limit = call_bound(mode, effective)
            jobs.append({"job_id": f"job-{len(jobs):06d}", "sample_id": sample_id, "repeat": repeat,
                         "variant": variant, "mode": mode, "settings": effective.model_dump(),
                         "max_call_attempts": limit})
    return {"format_version": 1, "kind": kind, "seed": seed, "split": split, "repeats": repeats,
            "dataset_sha256": digest, "selected_samples_sha256": sample_digest(samples),
            "base_settings": settings.model_dump(), "samples": samples, "jobs": jobs,
            "max_call_attempts": sum(j["max_call_attempts"] for j in jobs), "provenance": provenance(),
            "status": "planned_not_executed", "note": "Randomized within paired sample/repeat blocks; API nondeterminism remains. Labels never enter model prompts."}


def execute_plan(plan, destination, *, execute=False, demo=False, max_calls=None):
    validate_plan(plan)
    if demo and execute:
        raise ValueError("Choose demo OR explicit live execution")
    if not demo and not execute:
        raise ValueError("Live calls require --execute")
    if type(max_calls) is not int or max_calls < plan["max_call_attempts"]:
        raise ValueError("Explicit max_calls must cover the conservative attempt bound in the plan")
    if plan["provenance"]["source_sha256"] != provenance()["source_sha256"]:
        raise ValueError("Source changed after planning; create a fresh plan")
    current = provenance()
    if any(plan["provenance"].get(k) != current.get(k) for k in ("python", "dependencies")):
        raise ValueError("Runtime changed after planning; create a fresh plan")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    dump(destination / "plan.json", plan)
    calls_used = 0
    samples = {s["sample_id"]: s for s in plan["samples"]}
    for job in plan["jobs"]:
        sample = samples[job["sample_id"]]
        injected = {}
        if demo:
            from src.demo import DemoGenerator, DemoEvaluator
            injected = {"generator": DemoGenerator(), "evaluator": DemoEvaluator()}
        # Each result is persisted immediately; failures are retained, not silently dropped.
        result = run(sample["topic"], essay=sample["essay"], mode=job["mode"],
                     settings=Settings.model_validate(job["settings"]), **injected)
        result["evidence_type"] = "scripted_demo_not_experiment" if demo else "live_api_run_unvalidated_quality"
        result["experiment"] = {k: job[k] for k in ("job_id", "sample_id", "repeat", "variant")}
        calls_used += len(result["calls"])
        if len(result["calls"]) > job["max_call_attempts"] or calls_used > max_calls:
            raise RuntimeError("Internal call-bound invariant violated")
        path = save_result(result, destination / job["job_id"])
        with (destination / "index.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps({**result["experiment"], "result": str(path.relative_to(destination))}) + "\n")
        print(f"{job['job_id']} {job['variant']} {result.get('exit_reason') or 'evaluated'}", file=sys.stderr)
        if execute and any(c.get("status") == "provider_error" for c in result["calls"]):
            dump(destination / "completion.json", {"status": "stopped_on_provider_error",
                "completed_jobs": sum(1 for _ in (destination / "index.jsonl").open(encoding="utf-8")),
                "demo": False, "call_attempts_used": calls_used,
                "note": "Failure retained. Use the guarded dual_study runner for resumable new batches."})
            return
    dump(destination / "completion.json", {"completed_jobs": len(plan["jobs"]), "demo": demo, "call_attempts_used": calls_used})


def load_jobs(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8-sig"))
    validate_plan(plan)
    planned = {j["job_id"]: j for j in plan["jobs"]}
    seen = set()
    jobs = []
    index = directory / "index.jsonl"
    for line in index.read_text(encoding="utf-8-sig").splitlines() if index.exists() else []:
        entry = json.loads(line)
        job_id = entry["job_id"]
        if job_id in seen or job_id not in planned:
            raise ValueError("Duplicate or unplanned result in index")
        seen.add(job_id)
        if any(entry[k] != planned[job_id][k] for k in ("sample_id", "repeat", "variant")):
            raise ValueError("Result index differs from plan")
        path = (directory / entry["result"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Result path escapes experiment directory")
        entry["result"] = json.loads(path.read_text(encoding="utf-8-sig"))
        jobs.append(entry)
    return plan, jobs


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    return ordered[math.floor(k)] + (ordered[math.ceil(k)] - ordered[math.floor(k)]) * (k - math.floor(k))


def export_blind(directory, destination, seed=42):
    _, jobs = load_jobs(directory)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows, mapping = [], {}
    for entry in jobs:
        result = entry["result"]
        if result.get("final_score") is None or not result.get("essay"):
            continue
        blind_id = "b-" + hashlib.sha256((str(directory) + entry["job_id"]).encode()).hexdigest()[:20]
        rows.append({"blind_id": blind_id, "topic": result["topic"], "essay": result["essay"],
                     "scores": {d: None for d in DIMENSIONS}})
        mapping[blind_id] = {"job_id": entry["job_id"], "essay_sha256": hashlib.sha256(result["essay"].encode()).hexdigest()}
    random.Random(seed).shuffle(rows)
    (destination / "annotation.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    dump(destination / "private_mapping.json", mapping)
    return {"annotation_count": len(rows), "warning": "Give annotators ONLY annotation.jsonl and the rubric; keep mapping/results private. Fill all scores; do not invent labels."}


def human_scores(annotation_path, mapping_path, jobs):
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8-sig"))
    by_job = {e["job_id"]: e for e in jobs}
    scores = {}
    seen = set()
    for line in Path(annotation_path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        blind_id = row["blind_id"]
        if blind_id in seen or blind_id not in mapping:
            raise ValueError("Duplicate or unknown blind_id")
        seen.add(blind_id)
        reference = mapping[blind_id]
        entry = by_job[reference["job_id"]]
        if hashlib.sha256(entry["result"]["essay"].encode()).hexdigest() != reference["essay_sha256"]:
            raise ValueError("Annotated essay differs from experiment artifact")
        if row.get("essay") != entry["result"]["essay"] or row.get("topic") != entry["result"]["topic"]:
            raise ValueError("Annotation text changed")
        values = row["scores"]
        if set(values) != set(DIMENSIONS) or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
            raise ValueError("Every annotation needs four finite numeric scores in [0,1]")
        scores[reference["job_id"]] = sum(values[d] * WEIGHTS[d] for d in DIMENSIONS)
    return scores


def analyze(directory, *, annotation_path=None, mapping_path=None, prices=None, human_job_scores=None):
    plan, jobs = load_jobs(directory)
    annotations = human_scores(annotation_path, mapping_path, jobs) if annotation_path and mapping_path else {}
    if human_job_scores is not None:
        if annotation_path or mapping_path:
            raise ValueError('Choose annotation files or validated human scores, not both')
        annotations = human_job_scores
    if bool(annotation_path) != bool(mapping_path):
        raise ValueError("Annotations and private mapping must be supplied together")
    variants = {}
    for variant in sorted({j["variant"] for j in plan["jobs"]}):
        subset = [e for e in jobs if e["variant"] == variant]
        successful = [e for e in subset if e["result"].get("final_score") is not None]
        latency = [e["result"]["telemetry"]["elapsed_seconds"] for e in subset]
        calls = [c for e in subset for c in e["result"].get("calls", [])]
        variants[variant] = {"planned": sum(j["variant"] == variant for j in plan["jobs"]),
                             "completed": len(subset), "successful": len(successful),
                             "p50_seconds": percentile(latency, .5), "p95_seconds": percentile(latency, .95),
                             "call_attempts": len(calls), "tokens": token_summary(calls),
                             "failed": len(subset) - len(successful),
                             "success_rate_completed": len(successful) / len(subset) if subset else None,
                             "annotated_jobs": sum(e["job_id"] in annotations for e in successful), "mean_rounds": statistics.mean(len(e["result"]["history"]) for e in subset) if subset else None,
                             "cost": calculate_cost(calls, prices) if prices else None,
                             "model_mean_score_descriptive_only": statistics.mean(e["result"]["final_score"] for e in successful) if successful else None}
    comparisons = {}
    baseline = {"efficiency": "serial", "ablation": "no_rewrite", "early_exit": "fixed"}.get(plan["kind"])
    comparison_pairs = [(v, baseline) for v in variants if baseline and v != baseline]
    if plan["kind"] == "ablation":
        comparison_pairs.extend([("all_feedback", "no_feedback"), ("all_feedback", "relevance_feedback")])
    for variant, baseline in comparison_pairs:
        pairs = []
        pair_records = []
        failed_pair_count = missing_pair_count = unannotated_pair_count = 0
        for sample in plan["samples"]:
            diffs = []
            for repeat in range(plan["repeats"]):
                selected = {e["variant"]: e for e in jobs if e["sample_id"] == sample["sample_id"] and e["repeat"] == repeat}
                if variant not in selected or baseline not in selected:
                    missing_pair_count += 1
                    continue
                a, b = selected[variant], selected[baseline]
                if a["result"].get("final_score") is None or b["result"].get("final_score") is None:
                    failed_pair_count += 1
                    continue
                if plan["kind"] == "efficiency":
                    diffs.append(b["result"]["telemetry"]["elapsed_seconds"] - a["result"]["telemetry"]["elapsed_seconds"])
                elif a["job_id"] in annotations and b["job_id"] in annotations:
                    diffs.append(annotations[a["job_id"]] - annotations[b["job_id"]])
                else:
                    unannotated_pair_count += 1
            if diffs:
                pairs.append(statistics.mean(diffs))
                pair_records.append({"sample_id": sample["sample_id"], "paired_repeats": len(diffs),
                                     "delta": statistics.mean(diffs)})
        improved = sum(x > 1e-12 for x in pairs)
        tied = sum(abs(x) <= 1e-12 for x in pairs)
        regressed = sum(x < -1e-12 for x in pairs)
        comparisons[variant + "_vs_" + baseline] = {"paired_samples": len(pairs),
            "metric": "serial_minus_parallel_seconds" if plan["kind"] == "efficiency" else "human_score_delta",
            "mean_delta": statistics.mean(pairs) if pairs else None,
            "ci95_sample_bootstrap": bootstrap_mean_interval(pairs),
            "improved_count": improved, "tied_count": tied, "regressed_count": regressed,
            "tie_tolerance": 1e-12, "per_sample_deltas": pair_records,
            "failed_pairs": failed_pair_count, "missing_pairs": missing_pair_count,
            "unannotated_successful_pairs": unannotated_pair_count,
            "improved_fraction": improved / len(pairs) if pairs else None,
            "tied_fraction": tied / len(pairs) if pairs else None,
            "regressed_fraction": regressed / len(pairs) if pairs else None,
            "efficiency": efficiency_comparison(jobs, plan, variant, baseline, prices)}
    return {"kind": plan["kind"], "complete": len(jobs) == len(plan["jobs"]),
            "human_annotation_coverage": {"annotated_jobs": len(annotations), "completed_jobs": len(jobs)}, "planned_jobs": len(plan["jobs"]), "completed_jobs": len(jobs),
            "evidence_types": sorted({e["result"].get("evidence_type", "unknown") for e in jobs}),
            "variants": variants, "comparisons": comparisons,
            "reliability": reliability_report(jobs, plan["samples"], plan["repeats"]) if plan["kind"] == "reliability" else None,
            "warning": "Scripted demos prove wiring only. Self-judged score changes are not quality evidence. Human comparisons require genuine blind labels. Missing pairs are excluded and counts reported."}



def efficiency_comparison(jobs, plan, variant, baseline, prices=None):
    """Pair within sample/repeat, then average repeats before sample bootstrap."""
    lookup = {(e["sample_id"], e["repeat"], e["variant"]): e for e in jobs}
    collected = {k: [] for k in ("baseline_minus_variant_seconds", "baseline_over_variant_speedup",
                                "baseline_minus_variant_calls", "baseline_minus_variant_tokens", "baseline_minus_variant_cost")}
    for sample in plan["samples"]:
        local = {key: [] for key in collected}
        for repeat in range(plan["repeats"]):
            a = lookup.get((sample["sample_id"], repeat, variant))
            b = lookup.get((sample["sample_id"], repeat, baseline))
            if not a or not b or a["result"].get("final_score") is None or b["result"].get("final_score") is None:
                continue
            a, b = a["result"], b["result"]
            at, bt = a["telemetry"]["elapsed_seconds"], b["telemetry"]["elapsed_seconds"]
            local["baseline_minus_variant_seconds"].append(bt - at)
            if at > 0:
                local["baseline_over_variant_speedup"].append(bt / at)
            local["baseline_minus_variant_calls"].append(len(b["calls"]) - len(a["calls"]))
            if a["telemetry"].get("total_tokens") is not None and b["telemetry"].get("total_tokens") is not None:
                local["baseline_minus_variant_tokens"].append(b["telemetry"]["total_tokens"] - a["telemetry"]["total_tokens"])
            if prices:
                ac, bc = calculate_cost(a["calls"], prices)["complete_cost"], calculate_cost(b["calls"], prices)["complete_cost"]
                if ac is not None and bc is not None:
                    local["baseline_minus_variant_cost"].append(bc - ac)
        for key, values in local.items():
            if values:
                collected[key].append(statistics.mean(values))
    return {key: {"paired_samples": len(values), "mean": statistics.mean(values) if values else None,
                  "ci95_sample_bootstrap": bootstrap_mean_interval(values)} for key, values in collected.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="EssayLoop reproducible experiment tools; planning and analysis are offline")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--kind", choices=["reliability", "ablation", "efficiency", "early_exit"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="qwen-plus")
    p.add_argument("--evaluator-model")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--threshold", type=float, default=.85)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--max-output-tokens", type=int, default=2048)
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--max-calls", type=int, required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--mapping", type=Path)
    p.add_argument("--prices", type=Path)
    p = sub.add_parser("export-blind")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            if args.output.exists():
                raise ValueError("Output already exists")
            plan = make_plan(args.dataset, args.kind, Settings(model=args.model, evaluator_model=args.evaluator_model,
                             max_rounds=args.max_rounds, max_attempts=args.max_attempts, threshold=args.threshold,
                             timeout_seconds=args.timeout, max_output_tokens=args.max_output_tokens),
                             args.repeats, args.seed, args.split)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            dump(args.output, plan)
            print(json.dumps({"jobs": len(plan["jobs"]), "max_call_attempts": plan["max_call_attempts"], "status": plan["status"]}))
        elif args.command == "run":
            execute_plan(json.loads(args.plan.read_text(encoding="utf-8-sig")), args.output_dir,
                         execute=args.execute, demo=args.demo, max_calls=args.max_calls)
        elif args.command == "export-blind":
            print(json.dumps(export_blind(args.run_dir, args.output_dir)))
        else:
            if args.output.exists():
                raise ValueError("Report output already exists")
            prices = json.loads(args.prices.read_text(encoding="utf-8-sig")) if args.prices else None
            result = analyze(args.run_dir, annotation_path=args.annotations, mapping_path=args.mapping, prices=prices)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            dump(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        print(f"Experiment configuration/artifact error ({type(exc).__name__}); no guessed fallback values.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
