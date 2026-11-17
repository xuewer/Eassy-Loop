"""Integrity checks use local fixtures only, not experimental evidence."""
import copy
import json
import pytest
from src.config import Settings
from src.datasets import Sample
from src.experiments import make_plan, execute_plan, analyze
from src.plans import validate_plan
from src.telemetry import ObservedModel, Recorder


def plan(tmp_path, kind="ablation"):
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps({"sample_id":"s1", "topic":"AI", "essay":"AI changes work.",
        "split":"dev", "source":"synthetic test fixture"}) + "\n", encoding="utf-8")
    return make_plan(p, kind, Settings(max_rounds=2, max_attempts=1, retry_delay_seconds=0), repeats=1)


@pytest.mark.parametrize("mutation", ["budget", "job_budget", "settings", "duplicate", "sample", "job_path", "variant"])
def test_edited_plans_rejected_before_any_execution(tmp_path, mutation):
    p = plan(tmp_path)
    if mutation == "budget": p["max_call_attempts"] = 0
    elif mutation == "job_budget": p["jobs"][0]["max_call_attempts"] = 0
    elif mutation == "settings": p["jobs"][0]["settings"]["max_rounds"] = 20
    elif mutation == "duplicate": p["jobs"].append(copy.deepcopy(p["jobs"][0]))
    elif mutation == "sample": p["samples"][0]["essay"] = "Changed content"
    elif mutation == "job_path": p["jobs"][0]["job_id"] = "../outside"
    else: p["jobs"][0]["variant"] = "unknown"
    with pytest.raises(ValueError):
        execute_plan(p, tmp_path/"run", demo=True, max_calls=1000)
    assert not (tmp_path/"run").exists()


def test_source_change_requires_fresh_plan(tmp_path):
    p = plan(tmp_path)
    p["provenance"]["source_sha256"] = {}
    with pytest.raises(ValueError, match="Source changed"):
        execute_plan(p, tmp_path/"run", demo=True, max_calls=1000)


def test_ambiguous_live_demo_flags_rejected(tmp_path):
    p = plan(tmp_path)
    with pytest.raises(ValueError, match="OR"):
        execute_plan(p, tmp_path/"run", demo=True, execute=True, max_calls=1000)


def test_paired_metrics_preserve_missing_usage_and_direct_feedback_baseline(tmp_path):
    p = plan(tmp_path)
    execute_plan(p, tmp_path/"run", demo=True, max_calls=p["max_call_attempts"])
    report = analyze(tmp_path/"run")
    assert report["complete"]
    comparison = report["comparisons"]["all_feedback_vs_no_feedback"]
    assert comparison["mean_delta"] is None
    assert comparison["efficiency"]["baseline_minus_variant_tokens"]["mean"] is None
    assert report["variants"]["all_feedback"]["tokens"]["total_tokens"] is None
    assert comparison["efficiency"]["baseline_minus_variant_calls"]["paired_samples"] == 1


def test_partial_experiment_is_reported_not_filled(tmp_path):
    p = plan(tmp_path, "efficiency")
    execute_plan(p, tmp_path/"run", demo=True, max_calls=p["max_call_attempts"])
    index = tmp_path/"run"/"index.jsonl"
    first = index.read_text(encoding="utf-8").splitlines()[0]
    index.write_text(first+"\n", encoding="utf-8")
    report = analyze(tmp_path/"run")
    assert report["completed_jobs"] == 1
    assert report["complete"] is False
    assert report["comparisons"]["parallel_vs_serial"]["paired_samples"] == 0


@pytest.mark.parametrize("bad", [True, "0.5", -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_human_labels_rejected(bad):
    scores = dict.fromkeys(("relevance", "grammar", "structure", "depth"), .5)
    scores["depth"] = bad
    with pytest.raises(ValueError):
        Sample.model_validate({"sample_id":"x", "topic":"AI", "essay":"text", "split":"dev",
            "source":"test", "human_scores":scores})


def test_metadata_failure_does_not_discard_successful_model_output():
    class BrokenMetadata:
        def invoke(self, prompt): return "valid output"
        def take_metadata(self): raise RuntimeError("secret must not be logged")
    recorder = Recorder()
    assert ObservedModel(BrokenMetadata(), "evaluator", "fake", recorder).invoke("test") == "valid output"
    call = recorder.snapshot()[0]
    assert call["metadata_error"] == "usage_collection_failed"
    assert call["input_tokens"] is None
    assert "secret" not in json.dumps(call)
