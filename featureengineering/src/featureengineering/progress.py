from __future__ import annotations

from typing import Callable

# ── Stage constants ───────────────────────────────────────────────────────────
# Stages that _build_daily_financial goes through

STAGE_PIVOT = "pivot"
STAGE_FFILL = "ffill"
STAGE_STACK = "stack"
STAGE_FILTER = "filter"

# Callback type
ProgressCallback = Callable[[str, int, int], None] | None
