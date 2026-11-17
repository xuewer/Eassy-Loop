"""Thread-safe request tracing. Usage is observed, never estimated from text."""
from contextvars import ContextVar
import hashlib
import threading
import time
from uuid import uuid4
from src.client import error_details

CALL_CONTEXT = ContextVar("essayloop_call_context", default={})


class Recorder:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def add(self, record):
        with self.lock:
            self.calls.append(record)

    def snapshot(self):
        with self.lock:
            return sorted(self.calls, key=lambda c: c["started_monotonic"])


class ObservedModel:
    def __init__(self, model, role, model_name, recorder):
        self.model = model
        self.role = role
        self.model_name = model_name
        self.recorder = recorder

    def invoke(self, prompt):
        start = time.perf_counter()
        record = {"call_id": uuid4().hex, "role": self.role, "model": self.model_name,
                  "started_monotonic": start, "started_unix": time.time(),
                  "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                  **CALL_CONTEXT.get(), "status": "ok"}
        try:
            return self.model.invoke(prompt)
        except Exception as exc:
            record.update(status="provider_error", **error_details(exc))
            raise
        finally:
            record["latency_seconds"] = time.perf_counter() - start
            record["ended_monotonic"] = time.perf_counter()
            getter = getattr(self.model, "take_metadata", None)
            try:
                metadata = getter() if callable(getter) else {}
            except Exception:
                metadata = {}
                record["metadata_error"] = "usage_collection_failed"
            record.update(metadata or {})
            record.setdefault("input_tokens", None)
            record.setdefault("output_tokens", None)
            self.recorder.add(record)


def token_summary(calls):
    complete = bool(calls) and all(c.get("input_tokens") is not None and c.get("output_tokens") is not None for c in calls)
    return {"call_count": len(calls), "provider_failures": sum(c["status"] != "ok" for c in calls),
            "usage_complete": complete,
            "calls_missing_usage": sum(c.get("input_tokens") is None or c.get("output_tokens") is None for c in calls),
            "observed_input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
            "observed_output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
            "total_tokens": sum((c.get("input_tokens") or 0) + (c.get("output_tokens") or 0) for c in calls) if complete else None,
            "cost": None, "cost_note": "No price assumed. Use an explicit dated price table for offline cost calculation."}
