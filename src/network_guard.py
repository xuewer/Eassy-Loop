"""Bounded non-inference preflight and durable request journals, without secrets."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from src.config import load_environment
from src.diagnostics import check_connection, configuration_report
from src.telemetry import Recorder


def append_json(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class JournalRecorder(Recorder):
    """Persist completed attempts even when a later graph or save fails."""
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.journal_lock = threading.Lock()

    def add(self, record: dict) -> None:
        with self.journal_lock:
            append_json(self.path, record)
            super().add(record)


def preflight(*, trust_env: bool = True, probe=None, sleeper=time.sleep) -> dict:
    """At most three unauthenticated HEADs with 1/2s backoff; no POST retry.

    An HTTP response proves only this route is reachable. Auth failures,
    unsupported models, and provider overload may still affect inference.
    """
    load_environment()
    base, configuration = configuration_report(trust_env=trust_env)
    probe = probe or check_connection
    checks = []
    successes = 0
    for index in range(3):
        check = probe(base, trust_env=trust_env, timeout=10)
        checks.append(check)
        reachable = check.get("status") == "http_response_received"
        status = check.get("http_status")
        # 400/401/403/404/405 are valid unauthenticated connectivity responses;
        # redirects, throttling and server errors should not launch a batch.
        usable = (reachable and type(status) is int and 200 <= status < 500
                  and not 300 <= status < 400 and status != 429)
        successes = successes + 1 if usable else 0
        if successes >= 2:
            break
        if check.get("connection_cause") in {"tls_certificate_error", "tls_handshake_error"}:
            break
        if index < 2:
            sleeper(2 ** index)
    return {"time": datetime.now(timezone.utc).isoformat(), "ready": successes >= 2,
            "configuration": configuration, "checks": checks,
            "note": "Two consecutive non-redirect HTTP responses; short connectivity observation only, not a stability, credential or model-access guarantee. TLS verified; no OS DNS/proxy edits."}


def provider_failure(calls: list[dict]) -> bool:
    return any(c.get("status") == "provider_error" for c in calls)
