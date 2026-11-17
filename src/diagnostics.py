"""Local configuration report and optional unauthenticated HEAD connectivity check."""
import argparse
import json
import os
import httpx
from src.client import ProviderError, ERROR_HINTS, error_details
from src.config import load_environment
from src.transport import connection_reason


def configuration_report(*, trust_env=True):
    base = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip().rstrip("/")
    url = httpx.URL(base)
    if url.scheme != "https" or not url.host or url.username or url.password or url.query or url.fragment:
        raise ValueError("Invalid HTTPS API base URL")
    return base, {"endpoint_host": url.host, "endpoint_port": url.port or 443,
                  "endpoint_source": "environment_or_dotenv" if os.getenv("QWEN_BASE_URL") else "default",
                  "trust_env": trust_env, "tls_verification": True,
                  "environment_settings_present": {key: bool(os.getenv(key)) for key in
                    ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR")},
                  "note": "Presence flags do not prove a proxy is used. No credential or environment values are reported."}


def check_connection(base, *, trust_env=True, timeout=10):
    """No bearer token, no text, no model body, no redirects and no retries."""
    try:
        with httpx.Client(timeout=timeout, trust_env=trust_env, follow_redirects=False) as client:
            response = client.head(base + "/chat/completions")
        return {"status": "http_response_received", "http_status": response.status_code,
                "note": "An HTTP response was received for an unauthenticated HEAD request; this does not validate API credentials, model access or successful generation."}
    except httpx.TransportError as exc:
        phase = next((phase for kind, phase in ((httpx.ConnectTimeout, "connect"),
            (httpx.ReadTimeout, "read"), (httpx.WriteTimeout, "write"),
            (httpx.PoolTimeout, "pool")) if isinstance(exc, kind)), None)
        category = ("timeout" if isinstance(exc, httpx.TimeoutException) else
                    "proxy_error" if isinstance(exc, httpx.ProxyError) else
                    "connection_error" if isinstance(exc, httpx.ConnectError) else "transport_error")
        cause = connection_reason(exc) if category in ("connection_error", "proxy_error") else None
        return {"status": "transport_failed", **error_details(ProviderError(ERROR_HINTS[category],
            category=category, timeout_phase=phase, connection_cause=cause, retryable=cause != "tls_certificate_error"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Eassy Loop connection diagnostics; no model calls")
    parser.add_argument("--check-network", action="store_true", help="Send one unauthenticated HEAD request, without model inference")
    parser.add_argument("--ignore-env-proxy", action="store_true", help="Ignore HTTPX environment proxies and custom CA paths; TLS remains enabled")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args(argv)
    if not 0 < args.timeout <= 60:
        parser.error("timeout must be greater than 0 and at most 60")
    try:
        load_environment()
        base, report = configuration_report(trust_env=not args.ignore_env_proxy)
        report["network_check"] = check_connection(base, trust_env=not args.ignore_env_proxy, timeout=args.timeout) if args.check_network else {"status": "not_requested"}
    except (ValueError, OSError, httpx.InvalidURL):
        print(json.dumps({"status": "configuration_error", "hint": "Check API base URL and certificate configuration; private values are not printed."}))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["network_check"]["status"] == "transport_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
