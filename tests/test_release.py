import json

from src.app import main, run
from src.artifacts import summary
from src.config import Settings
from src.demo import DemoEvaluator
from src.experiments import execute_plan, load_jobs, make_plan


def test_cli_run_id_matches_artifact_directory(tmp_path):
    output = tmp_path / "chosen-run"
    assert main(["--demo", "--output-dir", str(output)]) == 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    brief = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["run_id"] == brief["run_id"] == output.name
    assert brief["project"] == "Eassy Loop"


def test_unchanged_rewrite_is_reported_separately_from_successful_call():
    essay = "A local shop should collect only data needed to serve its customers."

    class EchoGenerator:
        def invoke(self, prompt):
            return essay

    result = run(
        "Customer privacy", essay=essay,
        generator=EchoGenerator(), evaluator=DemoEvaluator(),
        settings=Settings(max_rounds=2, stop_policy="fixed", retry_delay_seconds=0),
    )
    assert result["rewrites_completed"] == 1
    assert result["generation_history"][0]["text_changed"] is False
    assert summary(result)["rewrites_changed_text"] == 0


def test_archived_windows_result_paths_load_portably(tmp_path):
    dataset = tmp_path / "samples.jsonl"
    dataset.write_text(json.dumps({
        "sample_id": "one", "topic": "Privacy", "essay": "A shop needs little customer data.",
        "split": "dev", "source": "synthetic test fixture",
    }) + "\n", encoding="utf-8")
    plan = make_plan(dataset, "reliability", Settings(max_attempts=1), repeats=1)
    output = tmp_path / "experiment"
    execute_plan(plan, output, demo=True, max_calls=plan["max_call_attempts"])
    index_path = output / "index.jsonl"
    entry = json.loads(index_path.read_text(encoding="utf-8"))
    entry["result"] = entry["result"].replace("/", chr(92))
    index_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    _, jobs = load_jobs(output)
    assert len(jobs) == 1 and jobs[0]["result"]["final_score"] is not None

    entry["result"] = ".." + chr(92) + "outside.json"
    index_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="escapes"):
        load_jobs(output)
