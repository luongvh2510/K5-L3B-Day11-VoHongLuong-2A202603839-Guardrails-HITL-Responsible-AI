"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    MAX_TEXT = 500  # keep the log readable and bounded

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}      # key -> start time (monotonic seconds)
        self._inputs: dict[str, dict] = {}     # key -> {"input": ..., "timestamp": ...}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        """request_id identifies one exchange; fall back to the user if none is given."""
        return request_id or user_id

    def _clean(self, text: str) -> str:
        """Never let the audit trail itself become a leak: mask PII / secrets, cap length."""
        from guardrails.output_guardrails import content_filter

        cleaned = content_filter(text or "")["redacted"]
        if len(cleaned) > self.MAX_TEXT:
            cleaned = cleaned[: self.MAX_TEXT] + "...[truncated]"
        return cleaned

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store the (cleaned) input and a start time, keyed by request_id/user_id."""
        key = self._key(user_id, request_id)
        # Clean FIRST, start the clock LAST: cleaning (and its one-time import) must not
        # be counted as request latency.
        self._inputs[key] = {"input": self._clean(text), "timestamp": utc_now_iso()}
        self._open[key] = time.perf_counter()

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete the entry with output, layer decision and latency; append to self.logs."""
        key = self._key(user_id, request_id)
        started = self._open.pop(key, None)
        pending = self._inputs.pop(key, {})
        latency_ms = round((time.perf_counter() - started) * 1000, 2) if started is not None else None

        entry = {
            "timestamp": pending.get("timestamp") or utc_now_iso(),
            "request_id": request_id,
            "user_id": user_id,
            "input": pending.get("input"),   # None if record_input was never called
            "output": self._clean(text),
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": latency_ms,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
