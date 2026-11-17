"""Lossless run artifacts; fresh directories prevent accidental result replacement."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.config import PROJECT_ROOT

PROJECT_NAME = "Eassy Loop"


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:12]


def provenance() -> dict:
    files = [*sorted((PROJECT_ROOT / "src").glob("*.py")),
             PROJECT_ROOT / "requirements-lock.txt", PROJECT_ROOT / "requirements.txt"]
    hashes = {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files if p.is_file()}
    try:
        commit = subprocess.run(["git", "--no-optional-locks", "rev-parse", "HEAD"],
                                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        commit = ""
    versions = {}
    for name in ("langgraph", "pydantic", "httpx", "python-dotenv"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"project": PROJECT_NAME, "python": platform.python_version(),
            "dependencies": versions, "git_commit": commit or None, "source_sha256": hashes,
            "note": "Commit may not include local edits; source hashes describe the executed files. No environment values are saved."}


def summary(result: dict) -> dict:
    failures = []
    fields = ("attempt", "status", "error", "http_status", "timeout_phase", "connection_cause",
              "provider_code", "provider_reason", "retryable", "hint")
    if result.get("exit_reason") == "generation_failed":
        attempts = result.get("generation_attempts") or []
        generation = (result.get("generation_history") or [{}])[-1]
        if attempts:
            failures.append({"stage": "rewrite" if generation.get("is_rewrite") else "generate",
                             **{k: attempts[-1].get(k) for k in fields}})
    dimensions = {}
    for d in ("relevance", "grammar", "structure", "depth"):
        value = result.get(d + "_result")
        if value and value.get("status") != "ok" and value.get("attempts"):
            failures.append({"stage": d, **{k: value["attempts"][-1].get(k) for k in fields}})
        dimensions[d] = {"status": value.get("status"),
                         "score": (value.get("judgment") or {}).get("score"),
                         "error": (value.get("attempts") or [{}])[-1].get("error")} if value else None
    return {"project": PROJECT_NAME, "evidence_type": result.get("evidence_type", "unknown"),
            "run_id": result.get("run_id"), "mode": result.get("mode"),
            "current_round": result.get("current_round"), "final_score": result.get("final_score"),
            "exit_reason": result.get("exit_reason"), "failures": failures, "dimensions": dimensions,
            "rewrites_completed": result.get("rewrites_completed", 0),
            "rewrites_changed_text": sum(
                item.get("status") == "ok" and item.get("is_rewrite", False)
                and item.get("input_essay_hash") != item.get("output_essay_hash")
                for item in result.get("generation_history", [])
            ),
            "telemetry": result.get("telemetry")}


def save_result(result: dict, directory: Path) -> Path:
    """Directory must not exist. JSON is UTF-8 and never passes through a terminal."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    result_path = directory / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(summary(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "calls.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False, allow_nan=False) + "\n"
                                                   for c in result.get("calls", [])), encoding="utf-8")
    if result.get("essay"):
        (directory / "final_essay.txt").write_text(result["essay"], encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(result.get("provenance", {}), indent=2, allow_nan=False), encoding="utf-8")
    return result_path
