"""Append-only JSONL task journal (reliability / observability).

Every high-level orchestrator flow records one line per finished task:
timestamp, kind, status, duration, output path and error text.  The journal
lives in ``output/tasks.jsonl`` and can be inspected with::

    python veo_generator.py tasks [--limit N] [--json]
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

_JOURNAL_LOCK = threading.Lock()


class TaskJournal:
    """Thread-safe append-only JSONL log of generation tasks."""

    def __init__(self, path: str):
        self.path = path

    def record(
        self,
        kind: str,
        status: str,
        *,
        prompt: Optional[str] = None,
        output: Optional[str] = None,
        duration_s: float = 0.0,
        error: Optional[str] = None,
        extra: Optional[dict] = None,
    ) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kind": kind,            # video | image | audio | chain | batch | bridge
            "status": status,        # ok | failed
            "duration_s": round(duration_s, 2),
        }
        if prompt is not None:
            entry["prompt"] = prompt[:200]
        if output:
            entry["output"] = output
        if error:
            entry["error"] = str(error)[:300]
        if extra:
            entry.update(extra)
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            line = json.dumps(entry, ensure_ascii=False)
            with _JOURNAL_LOCK:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:
            # Journaling must never break the main workflow.
            pass

    def load(self, limit: int = 50) -> list[dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
        except OSError:
            return []
        entries: list[dict] = []
        for ln in lines[-limit:] if limit else lines:
            try:
                entries.append(json.loads(ln))
            except json.JSONDecodeError:
                continue  # skip corrupted lines
        return entries

    def stats(self) -> dict:
        """Aggregate counters useful for quick health checks."""
        entries = self.load(limit=0)
        by_kind: dict[str, dict[str, int]] = {}
        total_ok = total_failed = 0
        durations: list[float] = []
        for e in entries:
            k = e.get("kind", "?")
            bucket = by_kind.setdefault(k, {"ok": 0, "failed": 0})
            st = e.get("status", "failed")
            bucket[st] = bucket.get(st, 0) + 1
            if st == "ok":
                total_ok += 1
            else:
                total_failed += 1
            d = e.get("duration_s")
            if isinstance(d, (int, float)):
                durations.append(float(d))
        return {
            "total": len(entries),
            "ok": total_ok,
            "failed": total_failed,
            "by_kind": by_kind,
            "avg_duration_s": round(sum(durations) / len(durations), 2) if durations else 0.0,
        }
