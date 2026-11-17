"""Offline regression tests for error visibility without credential disclosure."""
import json
import httpx
import pytest
from src.app import run
from src.client import ProviderError, QwenClient, error_details
from src.config import Settings
from src.client import response_diagnostics
from src.artifacts import save_result


@pytest.mark.parametrize("payload,expected", [
    ({"error": {"code": "AccessDenied.Unpurchased"}}, "AccessDenied.Unpurchased"),
    ({"code": "Arrearage"}, "Arrearage"),
    ({"error": {"code": "AccessDenied.Workspace", "message": "PRIVATE_SECRET"}}, "AccessDenied.Workspace"),
    ({"error": []}, None),
    ([], None),
])
def test_safe_provider_code(payload, expected):
    response = httpx.Response(403, json=payload)
    diagnostics = response_diagnostics(response)
    assert diagnostics == {"provider_code": expected, "response_format": "json"}
    assert "PRIVATE_SECRET" not in json.dumps(diagnostics)


def test_html_refusal_classification():
    response = httpx.Response(403, text="<html>PRIVATE_SECRET</html>",
                              headers={"content-type": "text/html"})
    assert response_diagnostics(response) == {"provider_code": None, "response_format": "html"}


def test_provider_code_survives_failure_report(monkeypatch):
    import src.client as module
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-secret")
    client = QwenClient(Settings(retry_delay_seconds=0))
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        403, json={"error": {"code": "AccessDenied", "message": "PRIVATE_SECRET"}})))
    try:
        result = run("AI", generator=client, evaluator=client, settings=Settings(retry_delay_seconds=0))
        attempt = result["generation_attempts"][0]
        assert attempt["provider_code"] == "AccessDenied"
        assert attempt["response_format"] == "json"
        assert "PRIVATE_SECRET" not in json.dumps(result)
    finally:
        client.close()


@pytest.mark.parametrize("category,status,retryable", [
    ("authentication_error",401,False), ("permission_error",403,False),
    ("not_found",404,False), ("rate_limited",429,True), ("server_error",503,True)])
def test_http_diagnostics_in_workflow(monkeypatch, category, status, retryable):
    import src.client as module
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-secret")
    client = QwenClient(Settings(retry_delay_seconds=0))
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        status, json={"error":{"message":"offline-secret PRIVATE_PAYLOAD"}})))
    try:
        result = run("AI", generator=client, evaluator=client,
                     settings=Settings(retry_delay_seconds=0))
        attempt = result["generation_attempts"][0]
        assert attempt["error"] == category
        assert attempt["http_status"] == status
        assert attempt["retryable"] is retryable
        assert attempt["hint"]
        assert len(result["generation_attempts"]) == (2 if retryable else 1)
        assert "offline-secret" not in json.dumps(result)
        assert "PRIVATE_PAYLOAD" not in json.dumps(result)
    finally:
        client.close()


@pytest.mark.parametrize("exc_type,category", [(httpx.ProxyError,"proxy_error"),
    (httpx.ConnectError,"connection_error"), (httpx.ReadTimeout,"timeout")])
def test_transport_diagnostics(monkeypatch, exc_type, category):
    import src.client as module
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-secret")
    def failure(request):
        raise exc_type("PRIVATE_SECRET", request=request)
    client = QwenClient(Settings())
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(failure))
    try:
        with pytest.raises(ProviderError) as caught:
            client.invoke("test")
        details = error_details(caught.value)
        assert details["error"] == category
        assert details["retryable"]
        assert "PRIVATE_SECRET" not in json.dumps(details)
    finally:
        client.close()


def test_unknown_exception_message_is_never_serialized():
    details = error_details(RuntimeError("PRIVATE_SECRET"))
    assert details["error"] == "unexpected_error"
    assert "PRIVATE_SECRET" not in json.dumps(details)


def test_ignore_environment_is_explicit(monkeypatch):
    import src.client as module
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-secret")
    original = httpx.Client
    observed = {}
    def factory(**kwargs):
        observed.update(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(module.httpx, "Client", factory)
    client = QwenClient(Settings(trust_env=False))
    try:
        assert observed["trust_env"] is False
        assert observed.get("verify", True) is True
    finally:
        client.close()


@pytest.mark.parametrize("code", ["sk-secret", "BearerSecret", "bad code with spaces", "x" * 97, "prefix-offline-secret"])
def test_code_secret_redaction(code):
    response = httpx.Response(403, json={"error": {"code": code}})
    assert response_diagnostics(response, "offline-secret")["provider_code"] == "redacted"


def test_reason_is_controlled_label():
    response = httpx.Response(403, json={"error": {"code": "AccountDenied", "message": "Account balance overdue PRIVATE_SECRET"}})
    result = response_diagnostics(response)
    assert result["provider_reason"] == "account_or_billing"
    assert "PRIVATE_SECRET" not in json.dumps(result)


@pytest.mark.parametrize("exc_type,phase", [
    (httpx.ConnectTimeout, "connect"), (httpx.ReadTimeout, "read"),
    (httpx.WriteTimeout, "write"), (httpx.PoolTimeout, "pool"),
])
def test_timeout_phase_is_safe_and_actionable(monkeypatch, exc_type, phase):
    import src.client as module
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "offline-secret")

    def failure(request):
        raise exc_type("PRIVATE_TIMEOUT_DETAIL", request=request)

    client = QwenClient(Settings(retry_delay_seconds=0))
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(failure))
    try:
        with pytest.raises(ProviderError) as caught:
            client.invoke("test")
        details = error_details(caught.value)
        assert details["error"] == "timeout"
        assert details["timeout_phase"] == phase
        assert details["retryable"] is True
        assert details["hint"]
        assert "PRIVATE_TIMEOUT_DETAIL" not in json.dumps(details)
    finally:
        client.close()


def test_failure_reason_is_in_summary_and_saved_for_future_runs(tmp_path):
    result = {"run_id": "test", "mode": "optimize", "current_round": 1,
              "exit_reason": "generation_failed", "generation_attempts": [
                  {"attempt": 1, "status": "provider_error", "error": "timeout",
                   "timeout_phase": "read", "retryable": True, "hint": "Safe read-timeout hint"}],
              "generation_history": [{"is_rewrite": False}], "calls": [], "telemetry": {}, "provenance": {}}
    path = save_result(result, tmp_path / "run")
    saved = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
    assert saved["failures"][0]["stage"] == "generate"
    assert saved["failures"][0]["timeout_phase"] == "read"
    assert saved["failures"][0]["hint"] == "Safe read-timeout hint"
