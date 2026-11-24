import json
import threading
from collections import Counter
import httpx
import pytest
from pydantic import ValidationError
from src.app import run
from src.client import ProviderError, QwenClient
from src.config import Settings
from src.nodes import Nodes
from src.schemas import DIMENSIONS, EvaluationInput
from src.utils import parse_judgment

ESSAY = "AI changes work. Workers need training."


def judgment(d="relevance", score=0.8, **updates):
    return {"dimension": d, "score": score, "rationale": "Reasoned assessment", "evidence": ["AI changes work."],
            "suggestions": [f"Improve {d}"], **updates}


class FakeEvaluator:
    def __init__(self, scores=(0.8,), failure=None, barrier=None):
        self.scores = scores
        self.failure = failure
        self.barrier = barrier
        self.calls = Counter()
        self.lock = threading.Lock()

    def invoke(self, prompt):
        d = prompt.splitlines()[0].split(": ")[1]
        with self.lock:
            attempt = self.calls[d]
            self.calls[d] += 1
        if self.barrier:
            self.barrier.wait(timeout=10)
        if self.failure and d == "grammar":
            if isinstance(self.failure, Exception):
                raise self.failure
            return self.failure
        return json.dumps(judgment(d, self.scores[min(attempt, len(self.scores) - 1)]))


class FakeGenerator:
    def __init__(self, failure=None):
        self.prompts = []
        self.failure = failure

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.failure:
            raise self.failure
        return ESSAY


def settings(**updates):
    return Settings(retry_delay_seconds=0, **updates)


@pytest.mark.parametrize("score", [0, 0.85, 1])
def test_valid_score(score):
    assert parse_judgment(json.dumps(judgment(score=score)), "relevance", ESSAY).score == score


@pytest.mark.parametrize("score", [-0.1, 1.1, "0.8", True, float("nan"), float("inf")])
def test_reject_invalid_score(score):
    with pytest.raises(ValueError):
        parse_judgment(json.dumps(judgment(score=score)), "relevance", ESSAY)


@pytest.mark.parametrize("updates", [{"dimension": "grammar"}, {"evidence": ["Invented quote"]},
                                         {"evidence": []}, {"suggestions": [" "]}, {"rationale": " "}, {"extra": 1}])
def test_invalid_contract(updates):
    with pytest.raises(ValueError):
        parse_judgment(json.dumps(judgment(**updates)), "relevance", ESSAY)


def test_fenced_json():
    fence = chr(96) * 3
    assert parse_judgment(fence + "json" + chr(10) + json.dumps(judgment()) + chr(10) + fence, "relevance", ESSAY).score == 0.8


@pytest.mark.parametrize("raw", ['{"score": 0.1, "score": 0.9}', "Score: 0.8", "{}"])
def test_malformed_json(raw):
    with pytest.raises(ValueError):
        parse_judgment(raw, "relevance", ESSAY)


@pytest.mark.parametrize("kwargs", [{"topic": " "}, {"topic": "x", "essay": " "}])
def test_blank_input(kwargs):
    with pytest.raises(ValidationError):
        EvaluationInput(**kwargs)


@pytest.mark.parametrize("kwargs", [{"max_rounds": 0}, {"threshold": 2}, {"timeout_seconds": 0}, {"max_attempts": 0}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValidationError):
        Settings(**kwargs)


def test_parallel_barrier_and_single_aggregate():
    evaluator = FakeEvaluator(barrier=threading.Barrier(4))
    generator = FakeGenerator()
    result = run("AI", essay=ESSAY, mode="evaluate", generator=generator, evaluator=evaluator, settings=settings())
    assert len(result["history"]) == 1
    assert result["final_score"] == pytest.approx(0.8)
    assert set(evaluator.calls) == set(DIMENSIONS)
    assert all(n == 1 for n in evaluator.calls.values())
    assert generator.prompts == []
    assert result["exit_reason"] is None


@pytest.mark.parametrize("score,reason", [(0.85, "threshold_met"), (0.2, "max_rounds")])
def test_first_round_exit(score, reason):
    generator = FakeGenerator()
    result = run("AI", generator=generator, evaluator=FakeEvaluator((score,)), settings=settings(max_rounds=1))
    assert result["exit_reason"] == reason
    assert len(generator.prompts) == 1
    assert len(result["history"]) == 1


def test_full_feedback_and_fresh_round_scores():
    generator = FakeGenerator()
    evaluator = FakeEvaluator((0.2, 0.9))
    result = run("AI", essay=ESSAY, generator=generator, evaluator=evaluator, settings=settings())
    assert result["exit_reason"] == "threshold_met"
    assert len(result["history"]) == 2
    assert result["history"][0]["final_score"] == pytest.approx(0.2)
    assert result["history"][1]["final_score"] == pytest.approx(0.9)
    assert len(generator.prompts) == 1
    for d in DIMENSIONS:
        assert f"Improve {d}" in generator.prompts[0]
        assert evaluator.calls[d] == 2
    assert "AI" in generator.prompts[0]
    assert all(r["round_id"] == 2 for r in result["history"][1]["results"])


def test_maximum_three_rounds():
    generator = FakeGenerator()
    result = run("AI", generator=generator, evaluator=FakeEvaluator((0.1,)), settings=settings())
    assert result["exit_reason"] == "max_rounds"
    assert len(result["history"]) == 3
    assert len(generator.prompts) == 3


@pytest.mark.parametrize("failure,status,attempts", [("not json", "parse_error", 2),
    (TimeoutError(), "provider_error", 2), (ProviderError("HTTP 401"), "provider_error", 1)])
def test_failed_dimension_stops_without_default_score(failure, status, attempts):
    generator = FakeGenerator()
    evaluator = FakeEvaluator(failure=failure)
    result = run("AI", essay=ESSAY, generator=generator, evaluator=evaluator, settings=settings())
    assert result["exit_reason"] == "evaluation_failed"
    assert result["final_score"] is None
    assert result["grammar_result"]["status"] == status
    assert result["grammar_result"]["judgment"] is None
    assert len(result["grammar_result"]["attempts"]) == attempts
    assert len(result["history"]) == 1
    assert generator.prompts == []


def test_generation_failure_stops_before_evaluation():
    evaluator = FakeEvaluator()
    result = run("AI", generator=FakeGenerator(TimeoutError()), evaluator=evaluator, settings=settings())
    assert result["exit_reason"] == "generation_failed"
    assert result["final_score"] is None
    assert not evaluator.calls
    assert len(result["generation_attempts"]) == 2


def test_parse_retry_recovers_and_preserves_attempts():
    class Recover:
        calls = 0
        def invoke(self, prompt):
            self.calls += 1
            return "bad" if self.calls == 1 else json.dumps(judgment())
    nodes = Nodes(FakeGenerator(), Recover(), settings())
    result = nodes.evaluate({"essay": ESSAY, "topic": "AI", "current_round": 1}, "relevance")["relevance_result"]
    assert result["status"] == "ok"
    assert [a["status"] for a in result["attempts"]] == ["parse_error", "ok"]


def test_stale_round_rejected():
    nodes = Nodes(FakeGenerator(), FakeEvaluator(), settings())
    state = {"essay": ESSAY, "topic": "AI", "current_round": 1}
    for d in DIMENSIONS:
        state.update(nodes.evaluate(state, d))
    state["current_round"] = 2
    result = nodes.aggregate(state)
    assert result["exit_reason"] == "evaluation_failed"
    assert result["final_score"] is None


def test_missing_dimension_rejected():
    nodes = Nodes(FakeGenerator(), FakeEvaluator(), settings())
    result = nodes.aggregate({"essay": ESSAY, "current_round": 1})
    assert result["exit_reason"] == "evaluation_failed"


def test_evaluate_requires_essay():
    with pytest.raises(ValueError):
        run("AI", mode="evaluate", evaluator=FakeEvaluator())


def test_missing_key_and_no_import_side_effect(monkeypatch):
    import src.client as module
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        QwenClient(settings())


@pytest.mark.parametrize("http_status,retryable", [(401, False), (429, True), (500, True)])
def test_provider_http_errors(monkeypatch, http_status, retryable):
    import src.client as module
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-key")
    client = QwenClient(settings())
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(http_status)))
    try:
        with pytest.raises(ProviderError) as caught:
            client.invoke("test")
        assert caught.value.retryable is retryable
    finally:
        client.close()


def test_provider_success(monkeypatch):
    import src.client as module
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-key")
    client = QwenClient(settings())
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"finish_reason": "stop", "message": {"content": ESSAY}}]})))
    try:
        assert client.invoke("test") == ESSAY
    finally:
        client.close()


@pytest.mark.parametrize("response", [{"choices": []}, {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": " "}}]}])
def test_provider_malformed_or_truncated_response(monkeypatch, response):
    import src.client as module
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-key")
    client = QwenClient(settings())
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)))
    try:
        with pytest.raises(ProviderError) as caught:
            client.invoke("test")
        assert caught.value.retryable
    finally:
        client.close()


def test_provider_network_timeout(monkeypatch):
    import src.client as module
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-key")
    def timeout(request):
        raise httpx.ReadTimeout("offline timeout", request=request)
    client = QwenClient(settings())
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(timeout))
    try:
        with pytest.raises(ProviderError) as caught:
            client.invoke("test")
        assert caught.value.retryable
        assert "offline-test-key" not in str(caught.value)
    finally:
        client.close()


def test_configuration_precedes_client_construction(monkeypatch):
    import src.client as module
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(module, "load_environment", lambda: monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-test-key"))
    client = QwenClient(settings())
    client.close()


def test_cli_input_error_does_not_require_key(capsys):
    from src.app import main
    assert main(["--topic", "AI", "--mode", "evaluate"]) == 2
    assert "error" in capsys.readouterr().err.lower()


def test_changed_essay_rejected():
    nodes = Nodes(FakeGenerator(), FakeEvaluator(), settings())
    state = {"essay": ESSAY, "topic": "AI", "current_round": 1}
    for d in DIMENSIONS:
        state.update(nodes.evaluate(state, d))
    state["essay"] = ESSAY + " Changed."
    assert nodes.aggregate(state)["exit_reason"] == "evaluation_failed"
