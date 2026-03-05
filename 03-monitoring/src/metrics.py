"""
metrics.py — SRE-grade metrics collection for the RAG pipeline.

Tracks:
  - P50 / P95 latency (not just averages)
  - Cost per request in USD
  - Citation coverage % (grounded vs. declined answers)
  - Failure rate
  - Request volume over time

Stores all metrics in a local SQLite database for persistence.
"""

import os
import json
import sqlite3
import numpy as np
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "metrics.db")


def _get_conn() -> sqlite3.Connection:
    """Returns a SQLite connection, creating the DB if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            latency_ms  REAL NOT NULL,
            prompt_tokens   INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            cost_usd    REAL DEFAULT 0.0,
            is_grounded INTEGER DEFAULT 1,   -- 1=cited, 0=declined
            is_error    INTEGER DEFAULT 0,   -- 1=exception
            model       TEXT,
            provider    TEXT,
            question    TEXT,
            answer_preview TEXT
        )
    """)
    conn.commit()


class MetricsRecorder:
    """Records per-request metrics to SQLite and computes aggregates."""

    def record(
        self,
        session_id: str,
        latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        is_grounded: bool = True,
        is_error: bool = False,
        model: str = "",
        provider: str = "groq",
        question: str = "",
        answer_preview: str = "",
    ):
        """Write one request's metrics to the DB."""
        conn = _get_conn()
        conn.execute("""
            INSERT INTO requests
              (session_id, timestamp, latency_ms, prompt_tokens, completion_tokens,
               total_tokens, cost_usd, is_grounded, is_error, model, provider,
               question, answer_preview)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            session_id,
            datetime.utcnow().isoformat(),
            latency_ms,
            prompt_tokens,
            completion_tokens,
            prompt_tokens + completion_tokens,
            cost_usd,
            1 if is_grounded else 0,
            1 if is_error else 0,
            model,
            provider,
            question,
            answer_preview[:500],
        ))
        conn.commit()
        conn.close()

    # ── Aggregate Queries ─────────────────────────────────────────────

    def get_latency_percentiles(self, hours: int = 24) -> Dict[str, float]:
        """Return P50, P75, P95, P99 latency over the last N hours."""
        rows = self._fetch_recent(hours, "latency_ms")
        if not rows:
            return {"p50": 0, "p75": 0, "p95": 0, "p99": 0, "count": 0}
        arr = np.array(rows)
        return {
            "p50": round(float(np.percentile(arr, 50)), 1),
            "p75": round(float(np.percentile(arr, 75)), 1),
            "p95": round(float(np.percentile(arr, 95)), 1),
            "p99": round(float(np.percentile(arr, 99)), 1),
            "mean": round(float(np.mean(arr)), 1),
            "count": len(arr),
        }

    def get_cost_summary(self, hours: int = 24) -> Dict[str, float]:
        """Return total and average cost over the last N hours."""
        rows = self._fetch_recent(hours, "cost_usd")
        if not rows:
            return {"total_usd": 0, "avg_usd": 0, "count": 0}
        arr = np.array(rows)
        return {
            "total_usd": round(float(np.sum(arr)), 6),
            "avg_usd": round(float(np.mean(arr)), 6),
            "count": len(arr),
        }

    def get_citation_coverage(self, hours: int = 24) -> Dict[str, Any]:
        """Return citation coverage % (grounded responses / total)."""
        conn = _get_conn()
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(is_grounded) as grounded,
                SUM(is_error) as errors
            FROM requests WHERE timestamp >= ?
        """, (since,)).fetchone()
        conn.close()
        total = row["total"] or 0
        grounded = row["grounded"] or 0
        errors = row["errors"] or 0
        return {
            "total_requests": total,
            "grounded": grounded,
            "declined": total - grounded - errors,
            "errors": errors,
            "citation_pct": round(grounded / total * 100, 1) if total > 0 else 0.0,
            "failure_rate_pct": round(errors / total * 100, 1) if total > 0 else 0.0,
        }

    def get_recent_requests(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        """Return recent requests for trace inspector view."""
        conn = _get_conn()
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        rows = conn.execute("""
            SELECT * FROM requests WHERE timestamp >= ?
            ORDER BY id DESC LIMIT ?
        """, (since, limit)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_timeseries(self, hours: int = 24, bucket_minutes: int = 30) -> List[Dict]:
        """Return time-bucketed latency and cost for charting."""
        conn = _get_conn()
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        rows = conn.execute("""
            SELECT timestamp, latency_ms, cost_usd, is_grounded
            FROM requests WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (since,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_all_stats(self) -> Dict[str, Any]:
        """Single call returning all key metrics for the dashboard."""
        return {
            "latency": self.get_latency_percentiles(),
            "cost": self.get_cost_summary(),
            "quality": self.get_citation_coverage(),
        }

    # ── Helpers ───────────────────────────────────────────────────────
    def _fetch_recent(self, hours: int, column: str) -> List[float]:
        conn = _get_conn()
        since = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        rows = conn.execute(
            f"SELECT {column} FROM requests WHERE timestamp >= ? AND is_error = 0",
            (since,)
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
