"""Validate executable plans before any model client is created."""
import hashlib
import json
import re
from src.config import Settings
from src.datasets import Sample

KINDS = ("reliability", "ablation", "efficiency", "early_exit")


def variants_for(kind):
    return {
        "reliability": {"judge": {"mode": "evaluate"}},
        "efficiency": {"serial": {"mode": "evaluate", "execution": "serial"},
                       "parallel": {"mode": "evaluate", "execution": "parallel"}},
        "ablation": {"no_rewrite": {"mode": "evaluate"},
                     "no_feedback": {"mode": "optimize", "feedback_mode": "none", "stop_policy": "fixed"},
                     "relevance_feedback": {"mode": "optimize", "feedback_mode": "relevance", "stop_policy": "fixed"},
                     "all_feedback": {"mode": "optimize", "feedback_mode": "all", "stop_policy": "fixed"}},
        "early_exit": {"threshold": {"mode": "optimize", "stop_policy": "threshold"},
                       "fixed": {"mode": "optimize", "stop_policy": "fixed"}},
    }[kind]


def call_bound(mode, settings):
    # Plans always supply an initial essay. Round one therefore has no generation.
    calls = 4 if mode == "evaluate" else 5 * settings.max_rounds - 1
    return calls * settings.max_attempts


def sample_digest(samples):
    payload = json.dumps(samples, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("format_version") != 1 or plan.get("kind") not in KINDS:
        raise ValueError("Unsupported plan format or experiment kind")
    if type(plan.get("repeats")) is not int or not 1 <= plan["repeats"] <= 100:
        raise ValueError("Invalid repeats")
    if type(plan.get("seed")) is not int or plan.get("split") not in ("dev", "test"):
        raise ValueError("Invalid seed or split")
    settings = Settings.model_validate(plan["base_settings"])
    if plan["kind"] in ("ablation", "early_exit") and settings.max_rounds < 2:
        raise ValueError("Revision experiments need at least two rounds")
    samples = [Sample.model_validate(s).model_dump() for s in plan["samples"]]
    ids = {s["sample_id"] for s in samples}
    if not samples or len(ids) != len(samples) or any(s["split"] != plan["split"] for s in samples):
        raise ValueError("Invalid sample IDs or split")
    if not re.fullmatch(r"[0-9a-f]{64}", plan.get("dataset_sha256", "")):
        raise ValueError("Invalid original dataset fingerprint")
    if plan.get("selected_samples_sha256") != sample_digest(samples):
        raise ValueError("Embedded sample content changed; create a fresh plan")
    variants = variants_for(plan["kind"])
    expected = {(sample, repeat, variant) for sample in ids for repeat in range(plan["repeats"]) for variant in variants}
    seen, total = set(), 0
    for i, job in enumerate(plan["jobs"]):
        key = (job["sample_id"], job["repeat"], job["variant"])
        if type(job["repeat"]) is not int or key not in expected or key in seen or job["job_id"] != f"job-{i:06d}":
            raise ValueError("Duplicate, missing or unexpected paired job")
        seen.add(key)
        config = dict(variants[job["variant"]])
        mode = config.pop("mode")
        effective = Settings.model_validate({**settings.model_dump(), **config})
        if job["mode"] != mode or job["settings"] != effective.model_dump():
            raise ValueError("Job settings differ from the declared experiment variant")
        bound = call_bound(mode, effective)
        if type(job["max_call_attempts"]) is not int or job["max_call_attempts"] != bound:
            raise ValueError("Job call bound was changed")
        total += bound
    if seen != expected or type(plan.get("max_call_attempts")) is not int or plan["max_call_attempts"] != total:
        raise ValueError("Incomplete jobs or invalid total call bound")
    if not isinstance(plan.get("provenance", {}).get("source_sha256"), dict):
        raise ValueError("Source provenance is required")
    return plan
