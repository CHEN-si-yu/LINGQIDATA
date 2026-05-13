from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Callable

from .settings import ProjectPaths


# ── Timing record ──────────────────────────────────────────────────────────

@dataclass
class FactorTiming:
    name: str
    category: str
    start: float = 0.0
    end: float = 0.0
    elapsed: float = 0.0
    rows: int = 0
    non_null_rows: int = 0
    error: str | None = None


class TimingCollector:
    """Collect per-factor timing in-process. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[FactorTiming] = []

    def add(self, record: FactorTiming) -> None:
        with self._lock:
            self.records.append(record)

    def save(self, path: Path) -> None:
        with self._lock:
            payload = {
                "built_at": datetime.now().isoformat(timespec="seconds"),
                "total_factors": len(self.records),
                "total_elapsed": sum(r.elapsed for r in self.records),
                "records": [
                    {
                        "name": r.name,
                        "category": r.category,
                        "elapsed": round(r.elapsed, 2),
                        "rows": r.rows,
                        "non_null_rows": r.non_null_rows,
                        "error": r.error,
                    }
                    for r in sorted(self.records, key=lambda r: r.elapsed, reverse=True)
                ],
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )


# Global timing collector
_timing = TimingCollector()


def get_timing_collector() -> TimingCollector:
    return _timing


# ── Sub-progress tracker (multiprocessing-safe) ────────────────────────────

# Stages that _build_daily_financial goes through
STAGE_PIVOT = "pivot"
STAGE_FFILL = "ffill"
STAGE_STACK = "stack"
STAGE_FILTER = "filter"


class SubProgress:
    """Shared-memory progress state for a single factor build step.

    Uses multiprocessing.Manager for cross-process visibility.
    """

    def __init__(self) -> None:
        manager = Manager()
        self._state = manager.dict()
        self._state["factor"] = ""
        self._state["stage"] = ""
        self._state["current"] = 0
        self._state["total"] = 0
        self._state["elapsed"] = 0.0

    def update(self, factor: str, stage: str, current: int, total: int) -> None:
        self._state["factor"] = factor
        self._state["stage"] = stage
        self._state["current"] = current
        self._state["total"] = total
        self._state["elapsed"] = time.time()

    def snapshot(self) -> dict[str, Any]:
        return dict(self._state)

    def clear(self) -> None:
        self._state["factor"] = ""
        self._state["stage"] = ""
        self._state["current"] = 0
        self._state["total"] = 0


# Callback type
ProgressCallback = Callable[[str, int, int], None] | None
