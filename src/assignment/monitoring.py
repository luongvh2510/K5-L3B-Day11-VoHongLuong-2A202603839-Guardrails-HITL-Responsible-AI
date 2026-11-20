"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def record_request(
        self,
        *,
        blocked: bool = False,
        rate_limited: bool = False,
        judge_checked: bool = False,
        judge_failed: bool = False,
    ) -> None:
        """Update every counter for one finished request (call once per request)."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
        if rate_limited:
            self.rate_limit_hits += 1
        if judge_checked:
            self.judge_checks += 1
            if judge_failed:
                self.judge_fails += 1

    def check_metrics(self) -> list[Alert]:
        """Compute rates and return an Alert for every threshold that is reached.

        The alert list is rebuilt on each call, so calling this after every request
        (and again on export) never produces duplicate alerts for the same problem.
        """
        snap = self.snapshot()
        alerts: list[Alert] = []

        if snap["block_rate"] >= self.block_rate_threshold:
            alerts.append(Alert(
                metric="block_rate",
                value=snap["block_rate"],
                threshold=self.block_rate_threshold,
                message=(
                    f"High block rate: {snap['block_rate']:.0%} of "
                    f"{self.total_requests} requests blocked "
                    f"(threshold {self.block_rate_threshold:.0%}) - possible attack wave."
                ),
            ))

        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            alerts.append(Alert(
                metric="rate_limit_hits",
                value=float(self.rate_limit_hits),
                threshold=float(self.rate_limit_hit_threshold),
                message=(
                    f"{self.rate_limit_hits} requests were rate limited "
                    f"(threshold {self.rate_limit_hit_threshold}) - possible flooding."
                ),
            ))

        # Only meaningful if a judge actually ran (judge_checks == 0 -> rate is 0.0).
        if self.judge_checks and snap["judge_fail_rate"] >= self.judge_fail_rate_threshold:
            alerts.append(Alert(
                metric="judge_fail_rate",
                value=snap["judge_fail_rate"],
                threshold=self.judge_fail_rate_threshold,
                message=(
                    f"Judge rejected {snap['judge_fail_rate']:.0%} of responses "
                    f"(threshold {self.judge_fail_rate_threshold:.0%}) - model may be unsafe."
                ),
            ))

        self.alerts = alerts
        for a in alerts:
            print(f"[ALERT] {a.message}")
        return alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON, by default ``<repo>/outputs/metrics.json``.

        Uses ``default_metrics_path()`` so running from ``src/`` does not create ``src/outputs/``.
        """
        self.check_metrics()  # make sure the file reflects the latest counters
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
