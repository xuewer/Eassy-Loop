import errno
import json
import socket
import ssl
import httpx
import pytest
from src.client import QwenClient
from src.config import Settings
from src.app import run
from src.artifacts import summary
from src.diagnostics import check_connection, main
from src.transport import connection_reason


@pytest.mark.parametrize("cause,reason", [
    (socket.gaierror(-2, "PRIVATE_DETAIL"), "dns_resolution_failed"),
    (ssl.SSLCertVerificationError(1, "PRIVATE_DETAIL"), "tls_certificate_error"),
    (ssl.SSLError(1, "PRIVATE_DETAIL"), "tls_handshake_error"),
    (ConnectionRefusedError(errno.ECONNREFUSED, "PRIVATE_DETAIL"), "connection_refused"),
    (ConnectionResetError(errno.ECONNRESET, "PRIVATE_DETAIL"), "connection_reset"),
    (OSError(errno.ENETUNREACH, "PRIVATE_DETAIL"), "network_unreachable"),
    (RuntimeError("PRIVATE_DETAIL"), "connection_failed"),
])
def test_connection_cause_survives_workflow_without_private_text(monkeypatch, cause, reason):
    import src.client as module
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.invalid/v1")
    def failure(request):
        raise httpx.ConnectError("PRIVATE_WRAPPER", request=request) from cause
    client = QwenClient(Settings(max_attempts=1))
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(failure))
    try:
        result = run("AI", generator=client, evaluator=client, settings=Settings(max_attempts=1))
        attempt = result["generation_attempts"][0]
        assert attempt["connection_cause"] == reason
        assert attempt["retryable"] is (reason != "tls_certificate_error")
        assert summary(result)["failures"][0]["connection_cause"] == reason
        assert "PRIVATE_" not in json.dumps(result)
    finally:
        client.close()


def test_cyclic_exception_chain_is_bounded():
    cause = RuntimeError("private")
    cause.__cause__ = cause
    assert connection_reason(cause) == "connection_failed"


def test_head_check_has_no_authorization_no_body_no_redirects(monkeypatch):
    import src.diagnostics as module
    original = httpx.Client
    captured = {}
    def handler(request):
        assert request.method == "HEAD"
        assert "authorization" not in request.headers
        assert request.content == b""
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(405)
    def factory(**kwargs):
        captured.update(kwargs)
        return original(**kwargs, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(module.httpx, "Client", factory)
    result = check_connection("https://example.invalid/v1", trust_env=False)
    assert result["status"] == "http_response_received"
    assert result["http_status"] == 405
    assert captured["trust_env"] is False
    assert captured["follow_redirects"] is False
    assert captured.get("verify", True) is True


def test_default_diagnostics_do_not_connect_or_print_environment_values(monkeypatch, capsys):
    import src.diagnostics as module
    monkeypatch.setattr(module, "load_environment", lambda: None)
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.invalid/private-path")
    monkeypatch.setenv("HTTPS_PROXY", "http://private-user:private-pass@proxy.invalid:9999")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "PRIVATE_KEY")
    def denied(*args, **kwargs):
        raise AssertionError("No network check requested")
    monkeypatch.setattr(module, "check_connection", denied)
    assert main([]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["network_check"]["status"] == "not_requested"
    assert report["environment_settings_present"]["HTTPS_PROXY"] is True
    assert all(x not in output for x in ("private-user", "private-pass", "private-path", "PRIVATE_KEY"))
