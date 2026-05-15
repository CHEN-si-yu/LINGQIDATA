from __future__ import annotations

import time
from multiprocessing import Manager
from typing import Any, Callable

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
