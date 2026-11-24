"""Offline synthetic fixtures; not experiment evidence."""
import json
from pathlib import Path
import pytest
from src import dual_study
from src.app import run
from src.client import ProviderError
from src.demo import DemoEvaluator, DemoGenerator
from src.network_guard import preflight
from src.schemas import DIMENSIONS


@pytest.mark.parametrize("status", [302, 429, 503])
def test_preflight_rejects_redirect_throttle_or_server_error(status):
    sleeps = []
    result = preflight(probe=lambda *a, **kw: {"status": "http_response_received", "http_status": status}, sleeper=sleeps.append)
    assert not result["ready"] and len(result["checks"]) == 3 and sleeps == [1, 2]


def test_preflight_requires_two_consecutive_http_responses():
    events = iter([{"status": "transport_failed", "connection_cause": "dns_resolution_failed"},
                   {"status": "http_response_received", "http_status": 400},
                   {"status": "http_response_received", "http_status": 405}])
    result = preflight(probe=lambda *a, **kw: next(events), sleeper=lambda _: None)
    assert result["ready"] and len(result["checks"]) == 3


def test_preflight_does_not_retry_bad_certificate():
    result = preflight(probe=lambda *a, **kw: {"status": "transport_failed", "connection_cause": "tls_certificate_error"},
                       sleeper=lambda _: pytest.fail("certificate errors should stop"))
    assert not result["ready"] and len(result["checks"]) == 1


def batch(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("".join(json.dumps({"sample_id": f"s{i}", "topic": f"Topic {i}",
        "essay": f"Topic {i} changes work.", "source": "synthetic offline fixture", "split": "dev"}) + "\n" for i in range(10)), encoding="utf-8")
    directory = tmp_path / "batch"
    dual_study.prepare(dataset, directory)
    return directory


class FakeModel:
    def __init__(self, settings, fail=False):
        self.settings, self.fail = settings, fail
        self.generator, self.evaluator = DemoGenerator(), DemoEvaluator()
        self.calls = 0
    def invoke(self, prompt):
        self.calls += 1
        if self.fail:
            raise ProviderError("SECRET_DO_NOT_LOG", category="connection_error", connection_cause="dns_resolution_failed", retryable=True)
        if self.settings.model == "kimi-k3":
            data = json.loads(prompt.split("INPUT_JSON:\n")[1])
            return json.dumps({"judgments": [{"dimension": d, "score": .5, "rationale": "Synthetic test",
                "evidence": [data["essay"][:8]], "suggestions": ["Review the reasoning"]} for d in DIMENSIONS]})
        return self.evaluator.invoke(prompt) if prompt.startswith("DIMENSION:") else self.generator.invoke(prompt)
    def close(self):
        pass


def ready(**kwargs):
    return {"ready": True, "note": "synthetic offline fixture"}


def test_failed_preflight_does_not_create_clients_or_spend_calls(tmp_path):
    directory = batch(tmp_path)
    result = dual_study.execute(directory, execute_live=True, max_calls=350,
        checker=lambda **kw: {"ready": False}, client_factory=lambda _: pytest.fail("No model construction"))
    assert result["status"] == "paused_network_preflight"
    assert result["calls_used"] == 0 and not (directory / "execution.lock").exists()


def test_provider_failure_pauses_batch_and_resume_never_replays_failed_task(tmp_path):
    directory = batch(tmp_path)
    first = dual_study.execute(directory, execute_live=True, max_calls=350, checker=ready,
        client_factory=lambda settings: FakeModel(settings, fail=True))
    _, jobs = dual_study.load_jobs(directory / "ablation")
    assert first["status"] == "paused_provider_failure" and len(jobs) == 1
    assert first["calls_used"] == 4  # already dispatched four-agent superstep
    failed_id = jobs[0]["job_id"]
    original_bytes = (directory / "ablation" / failed_id / "result.json").read_bytes()
    second = dual_study.execute(directory, execute_live=True, max_calls=350, resume=True, checker=ready,
        client_factory=lambda settings: FakeModel(settings, fail=True))
    _, jobs = dual_study.load_jobs(directory / "ablation")
    assert second["status"] == "paused_provider_failure" and len(jobs) == 2
    assert (directory / "ablation" / failed_id / "result.json").read_bytes() == original_bytes
    assert "SECRET_DO_NOT_LOG" not in (directory / "DUAL_MODEL_REPORT.json").read_text()


def test_complete_dual_batch_respects_bound_and_reports_ai_not_human(tmp_path):
    directory = batch(tmp_path)
    result = dual_study.execute(directory, execute_live=True, max_calls=350, checker=ready, client_factory=FakeModel)
    report = dual_study.analyze(directory)
    assert result["status"] == "completed" and result["calls_used"] <= 350
    assert report["workflow_attempted"] == 40 and report["successful_final_text_pairs"] == 40
    assert report["human_review"] == "cancelled_by_user_not_in_protocol"
    assert report["model_agreement_by_original_cluster_not_accuracy"]["weighted_total"]["n"] == 10
    assert report["comparisons"]["all_feedback_vs_no_feedback"]["second_model"]["n"] == 10
    before = dual_study.count_calls(directory)
    resumed = dual_study.execute(directory, execute_live=True, max_calls=350, resume=True, checker=ready, client_factory=FakeModel)
    assert resumed["calls_used"] == len(before)
    with pytest.raises(ValueError, match="explicit --resume"):
        dual_study.execute(directory, execute_live=True, max_calls=350)


def test_ambiguous_interruption_blocks_replay(tmp_path):
    directory = batch(tmp_path)
    (directory / "started").mkdir()
    dual_study.dump(directory / "started/job-000000.json", {"task": "job-000000"})
    result = dual_study.execute(directory, execute_live=True, max_calls=350, checker=ready,
        client_factory=lambda _: pytest.fail("ambiguous task must not contact models"))
    assert result["status"] == "paused_ambiguous_interrupted_task" and result["calls_used"] == 0
