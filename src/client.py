"""OpenAI-compatible DashScope client with safe, actionable diagnostics."""
import os
import re
import threading
from typing import Protocol
import httpx
from src.config import Settings, load_environment
from src.transport import CONNECTION_HINTS, connection_reason


class TextModel(Protocol):
    def invoke(self, prompt: str) -> str: ...


def safe_provider_code(value: object, secret: str = "") -> str | None:
    """Retain bounded symbolic codes without exposing credentials."""
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,95}", value):
        return "redacted"
    if (secret and secret in value) or value.lower().startswith(("sk-", "bearer")):
        return "redacted"
    return value


def response_diagnostics(response: httpx.Response, secret: str = "") -> dict:
    """Expose symbolic codes and controlled reason labels, never raw messages."""
    content_type = response.headers.get("content-type", "").lower()
    response_format = "html" if "text/html" in content_type else "other"
    code = None
    reason = None
    try:
        body = response.json()
        response_format = "json"
        if isinstance(body, dict):
            nested = body.get("error")
            error = nested if isinstance(nested, dict) else body
            code = safe_provider_code(error.get("code", body.get("code")), secret)
            message = error.get("message", "")
            if isinstance(message, str):
                lower = message[:8192].lower()
                clues = [
                    ("account_or_billing", ("arrear", "overdue", "billing", "balance", "payment")),
                    ("service_not_enabled", ("not activated", "not enabled", "not opened", "not purchased", "unpurchase")),
                    ("region_restriction", ("region", "country", "geographic")),
                    ("model_access", ("model access", "access to the model", "access to this model")),
                    ("policy_restriction", ("policy", "risk control", "security")),
                ]
                reason = next((label for label, phrases in clues if any(x in lower for x in phrases)), None)
    except (ValueError, UnicodeError):
        pass
    result = {"provider_code": code, "response_format": response_format}
    if reason:
        result["provider_reason"] = reason
    return result


# Values are application-owned; raw response bodies and exception text are never logged.
ERROR_HINTS = {
    "timeout": "Network timeout: check connectivity, proxy and timeout settings.",
    "proxy_error": "Proxy connection failed: check environment proxy configuration.",
    "connection_error": "Connection failed: check DNS, TLS certificates, network and proxy configuration.",
    "transport_error": "HTTP transport failed: check connectivity and proxy configuration.",
    "authentication_error": "Check DASHSCOPE_API_KEY and whether its region matches QWEN_BASE_URL.",
    "permission_error": "Check model access permissions, account status and endpoint region.",
    "not_found": "Check the API base URL and model identifier.",
    "rate_limited": "Request was rate limited; check provider limits and available quota.",
    "server_error": "Provider returned a server error; retry later.",
    "http_error": "Provider rejected the request; check model parameters and endpoint configuration.",
    "output_truncated": "Output reached the token limit; shorten the requested output or adjust the limit.",
    "response_blocked": "Provider did not complete the response; review input and provider policies.",
    "invalid_response": "Provider returned an unexpected response format; verify compatible API endpoint.",
    "provider_error": "Provider call failed; inspect configuration without sharing credentials.",
}

TIMEOUT_HINTS = {
    "connect": "Connection establishment timed out; check endpoint connectivity, TLS and proxy settings.",
    "read": "Timed out waiting for response data; check provider load, proxy and timeout settings. The server may already have processed the request.",
    "write": "Timed out sending request data; check network and proxy connectivity.",
    "pool": "Timed out acquiring a pooled connection; check concurrent requests and connection availability.",
}


class ProviderError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False, *,
                 category: str = "provider_error", http_status: int | None = None,
                 provider_code: str | None = None, response_format: str | None = None,
                 provider_reason: str | None = None, timeout_phase: str | None = None,
                 connection_cause: str | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.category = category if category in ERROR_HINTS else "provider_error"
        self.http_status = http_status
        self.provider_code = safe_provider_code(provider_code, os.getenv("DASHSCOPE_API_KEY", "").strip())
        self.provider_reason = provider_reason if provider_reason in {"account_or_billing", "service_not_enabled", "region_restriction", "model_access", "policy_restriction"} else None
        self.response_format = response_format if response_format in ("json", "html", "other") else None
        self.timeout_phase = timeout_phase if self.category == "timeout" and timeout_phase in TIMEOUT_HINTS else None
        self.connection_cause = connection_cause if connection_cause in CONNECTION_HINTS else None


def error_details(exc: Exception) -> dict:
    """Serialize only controlled metadata, never str(exc), URLs or response payloads."""
    if isinstance(exc, ProviderError):
        return {"error": exc.category, "http_status": exc.http_status,
                "retryable": exc.retryable, "hint": CONNECTION_HINTS.get(exc.connection_cause, TIMEOUT_HINTS.get(exc.timeout_phase, ERROR_HINTS[exc.category])),
                "provider_code": exc.provider_code, "response_format": exc.response_format,
                "provider_reason": exc.provider_reason, "timeout_phase": exc.timeout_phase,
                "connection_cause": exc.connection_cause}
    if isinstance(exc, (TimeoutError, ConnectionError)):
        category = "timeout" if isinstance(exc, TimeoutError) else "connection_error"
        return {"error": category, "retryable": True, "hint": ERROR_HINTS[category]}
    return {"error": "unexpected_error", "retryable": False,
            "hint": "Unexpected local failure; inspect the injected model implementation."}


class QwenClient:
    def __init__(self, settings: Settings):
        load_environment()
        key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not key:
            raise ValueError("DASHSCOPE_API_KEY is required; configure the project .env")
        base = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip().rstrip("/")
        url = httpx.URL(base)
        if url.scheme != "https" or not url.host or url.username or url.password or url.query or url.fragment:
            raise ValueError("QWEN_BASE_URL must be an HTTPS API base URL")
        self._local = threading.local()
        self._api_key = key
        self.settings = settings
        self.url = base + "/chat/completions"
        # Four evaluator workers can share one pool. Keep connections across jobs
        # in the guarded batch runner, avoiding needless DNS/TLS handshakes.
        # No transport retries: every attempted request must remain budgeted.
        self.http = httpx.Client(timeout=httpx.Timeout(settings.timeout_seconds,
                                 connect=min(15.0, settings.timeout_seconds), pool=15.0),
                                 limits=httpx.Limits(max_connections=4, max_keepalive_connections=4,
                                                     keepalive_expiry=120.0),
                                 trust_env=settings.trust_env, follow_redirects=False,
                                 headers={"Authorization": f"Bearer {key}"})

    def take_metadata(self) -> dict:
        value = getattr(self._local, "metadata", {})
        self._local.metadata = {}
        return value

    def invoke(self, prompt: str) -> str:
        self._local.metadata = {}
        try:
            response = self.http.post(self.url, json={
                "model": self.settings.model,
                "temperature": self.settings.temperature,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.settings.max_output_tokens,
            })
        except httpx.TransportError as exc:
            timeout_phase = next((phase for kind, phase in (
                (httpx.ConnectTimeout, "connect"), (httpx.ReadTimeout, "read"),
                (httpx.WriteTimeout, "write"), (httpx.PoolTimeout, "pool")) if isinstance(exc, kind)), None)
            category = ("timeout" if isinstance(exc, httpx.TimeoutException) else
                        "proxy_error" if isinstance(exc, httpx.ProxyError) else
                        "connection_error" if isinstance(exc, httpx.ConnectError) else "transport_error")
            cause = connection_reason(exc) if isinstance(exc, (httpx.ConnectError, httpx.ProxyError)) else None
            raise ProviderError(ERROR_HINTS[category], retryable=cause != "tls_certificate_error", category=category,
                                timeout_phase=timeout_phase, connection_cause=cause) from None
        try:
            payload = response.json()
            usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
            usage = usage if isinstance(usage, dict) else {}
            def count(name):
                v = usage.get(name)
                return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
            self._local.metadata = {"input_tokens": count("prompt_tokens"),
                                    "output_tokens": count("completion_tokens"),
                                    "endpoint_host": httpx.URL(self.url).host}
        except (ValueError, UnicodeError):
            pass
        if response.status_code >= 400:
            status = response.status_code
            category = ({401: "authentication_error", 403: "permission_error", 404: "not_found",
                         429: "rate_limited"}.get(status, "server_error" if status >= 500 else "http_error"))
            raise ProviderError(ERROR_HINTS[category], retryable=status == 429 or status >= 500,
                                category=category, http_status=status, **response_diagnostics(response, self._api_key))
        try:
            choice = response.json()["choices"][0]
            finish = choice.get("finish_reason")
            if finish == "length":
                raise ProviderError(ERROR_HINTS["output_truncated"], retryable=True, category="output_truncated")
            if finish == "content_filter":
                raise ProviderError(ERROR_HINTS["response_blocked"], category="response_blocked")
            if finish != "stop":
                raise ValueError("Incomplete response")
            text = choice["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty response")
            return text
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError(ERROR_HINTS["invalid_response"], retryable=True, category="invalid_response") from None

    def close(self) -> None:
        self.http.close()
