"""Rich-based live dashboard for factor build progress visualisation.

Usage::

    from featureengineering.dashboard import FactorBuildDashboard

    dash = FactorBuildDashboard(total_factors=580, max_workers=16)
    with dash:
        for factor in factors:
            dash.update_worker(slot, name="mom_20", stage="computing",
                               pct=0.67, elapsed=45.2)
            ...
        dash.mark_done(name, action="rebuild", elapsed=12.3)
    dash.print_summary(results)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Rich is an optional dependency — degrade gracefully when unavailable.
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich import box

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

# ── colour palette ──────────────────────────────────────────────────────────
_COLOUR_REBUILD = "#FF6B6B"     # warm red
_COLOUR_INCREMENTAL = "#FFD93D"  # amber
_COLOUR_SKIP = "#6BCB77"         # soft green
_COLOUR_ERROR = "#FF0000"        # bright red
_COLOUR_COMPUTING = "#4D96FF"    # blue
_COLOUR_WRITING = "#9B59B6"      # purple
_COLOUR_LOADING = "#FF8C32"      # orange
_COLOUR_BAR_FINISHED = "#6BCB77"
_COLOUR_BAR_ACTIVE = "#4D96FF"
_COLOUR_MUTED = "#888888"
_COLOUR_HEADER = "#FFFFFF"


@dataclass
class WorkerSlot:
    """Live state for one worker slot."""

    name: str = ""
    stage: str = "---"
    pct: float = 0.0
    elapsed: float = 0.0
    done: bool = False


@dataclass
class BuildStats:
    """Running counters updated by the main process."""

    rebuild: int = 0
    incremental: int = 0
    skip: int = 0
    error: int = 0
    completed: int = 0
    total: int = 0
    t_start: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t_start

    @property
    def remaining(self) -> float:
        if self.completed == 0:
            return 0.0
        rate = self.completed / max(self.elapsed, 0.001)
        pending = self.total - self.completed
        return pending / rate if rate > 0 else 0.0


class FactorBuildDashboard:
    """Rich live dashboard for factor build progress.

    Renders a full-terminal dashboard with:
    - Header: status, elapsed, ETA
    - Counters: rebuild / incremental / skip / error
    - Worker slots: one row per concurrent worker with stage + mini bar
    - Overall progress bar
    - Recent completions list
    """

    def __init__(
        self,
        total_factors: int,
        max_workers: int = 16,
        title: str = "Factor Build",
        refresh_per_second: int = 4,
    ) -> None:
        import sys as _sys

        self._stats = BuildStats(total=total_factors)
        self._max_workers = max_workers
        self._title = title
        self._slots: dict[int, WorkerSlot] = {
            i: WorkerSlot() for i in range(max_workers)
        }
        self._results: list[dict[str, Any]] = []

        # Only enable the ANSI-escape Live display when stdout is a real
        # terminal.  When stdout is piped (e.g. parent process streaming
        # subprocess output line-by-line), Rich's cursor-movement codes
        # would never emit a newline, causing the parent to block forever
        # in readline().  Fall back to plain-text progress in that case.
        _tty = _sys.stdout.isatty()

        if _RICH_AVAILABLE and _tty:
            self._console = Console()
            self._live = Live(
                self._render(),
                console=self._console,
                refresh_per_second=refresh_per_second,
                screen=False,
                auto_refresh=False,
            )
        else:
            self._console = None
            self._live = None

    # ── public API ──────────────────────────────────────────────────────

    def __enter__(self) -> "FactorBuildDashboard":
        if self._live is not None:
            self._live.start()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._live is not None:
            self._live.stop()

    def update_worker(
        self,
        slot: int,
        *,
        name: str = "",
        stage: str = "",
        pct: float = 0.0,
        elapsed: float = 0.0,
    ) -> None:
        """Update the display for one worker slot."""
        s = self._slots.get(slot)
        if s is None:
            return
        if name:
            s.name = name
        if stage:
            s.stage = stage
        s.pct = max(0.0, min(1.0, pct))
        s.elapsed = elapsed
        s.done = stage in ("done", "error")
        if self._live is not None:
            self._live.refresh()

    def mark_done(
        self,
        name: str,
        action: str,
        elapsed: float,
        slot: int | None = None,
    ) -> None:
        """Record a completed factor and update counters."""
        self._stats.completed += 1
        if action == "rebuild":
            self._stats.rebuild += 1
        elif action == "incremental":
            self._stats.incremental += 1
        elif action == "skip":
            self._stats.skip += 1
        elif action == "error":
            self._stats.error += 1

        self._results.append({
            "name": name,
            "action": action,
            "elapsed": elapsed,
        })

        if slot is not None:
            s = self._slots.get(slot)
            if s is not None:
                s.done = True
                s.name = name
                s.stage = action
                s.pct = 1.0
                s.elapsed = elapsed

        if self._live is not None:
            self._live.refresh()

    def refresh(self) -> None:
        """Force a display refresh."""
        if self._live is not None:
            self._live.refresh()

    def print_summary(self) -> None:
        """Print a post-build summary table to the console."""
        if self._console is None:
            self._print_plain_summary()
        else:
            self._print_rich_summary()

    # ── rendering ───────────────────────────────────────────────────────

    def _render(self) -> Layout:
        """Build the rich Layout tree."""
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="counters", size=3),
            Layout(name="body"),
            Layout(name="progress", size=3),
        )
        layout["body"].split_row(
            Layout(name="workers", ratio=2),
            Layout(name="recent", ratio=1),
        )
        layout["header"].update(self._render_header())
        layout["counters"].update(self._render_counters())
        layout["workers"].update(self._render_workers())
        layout["recent"].update(self._render_recent())
        layout["progress"].update(self._render_progress())
        return layout

    def _render_header(self) -> Panel:
        st = self._stats
        status = "● RUNNING" if st.completed < st.total else "● DONE"
        status_colour = _COLOUR_BAR_ACTIVE if st.completed < st.total else _COLOUR_SKIP

        eta_str = f"{st.remaining:.0f}s" if st.remaining < 120 else f"{st.remaining / 60:.1f}m"

        text = Text()
        text.append("█" * 60 + "\n", style=_COLOUR_MUTED)
        title = Text(f"  {self._title}  ", style=f"bold {_COLOUR_HEADER}")
        text.append(title)
        text.append(f"    {status}", style=f"bold {status_colour}")
        text.append(
            f"    Elapsed: {st.elapsed:.0f}s    "
            f"ETA: ~{eta_str}    "
            f"{st.completed}/{st.total}\n",
            style=_COLOUR_MUTED,
        )
        text.append("█" * 60, style=_COLOUR_MUTED)
        return Panel(text, style=_COLOUR_MUTED, box=box.SIMPLE)

    def _render_counters(self) -> Panel:
        st = self._stats
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row(
            Text("◆", style=_COLOUR_REBUILD),
            Text(f"Rebuild: {st.rebuild}", style=f"bold {_COLOUR_REBUILD}"),
            Text("▲", style=_COLOUR_INCREMENTAL),
            Text(f"Incremental: {st.incremental}",
                 style=f"bold {_COLOUR_INCREMENTAL}"),
            Text("✓", style=_COLOUR_SKIP),
            Text(f"Skip: {st.skip}", style=f"bold {_COLOUR_SKIP}"),
            Text("✗", style=_COLOUR_ERROR),
            Text(f"Error: {st.error}", style=f"bold {_COLOUR_ERROR}"),
        )
        return Panel(table, box=box.SIMPLE, border_style=_COLOUR_MUTED)

    def _render_workers(self) -> Panel:
        """Render worker slot table — one row per slot."""
        table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1),
                       border_style=_COLOUR_MUTED)
        table.add_column("#", width=4, style=_COLOUR_MUTED)
        table.add_column("factor", width=28, style=_COLOUR_HEADER)
        table.add_column("stage", width=14)
        table.add_column("progress", width=20)
        table.add_column("time", width=6, justify="right", style=_COLOUR_MUTED)

        active_slots = sorted(
            [s for s in self._slots.values() if s.name],
            key=lambda s: (s.done, -s.elapsed),
        )
        idle_count = self._max_workers - len(active_slots)

        for i, s in enumerate(active_slots):
            stage_style = _stage_colour(s.stage)
            if s.done:
                bar = _bar_str(1.0, 16, finished=True)
            else:
                bar = _bar_str(s.pct, 16)

            table.add_row(
                f"[{i + 1}]",
                s.name[:27],
                Text(s.stage, style=stage_style),
                bar,
                f"{s.elapsed:.0f}s" if s.elapsed > 0 else "",
            )

        for j in range(idle_count):
            i = len(active_slots) + j
            table.add_row(
                f"[{i + 1}]", "---", Text("idle", style=_COLOUR_MUTED),
                _bar_str(0.0, 16, idle=True), ""
            )

        return Panel(table, title="Workers", box=box.SIMPLE,
                      border_style=_COLOUR_MUTED)

    def _render_recent(self) -> Panel:
        """Render the 10 most recently completed factors."""
        table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1),
                       border_style=_COLOUR_MUTED)
        table.add_column("factor", width=22, style=_COLOUR_HEADER)
        table.add_column("action", width=14)
        table.add_column("time", width=8, justify="right", style=_COLOUR_MUTED)

        for entry in self._results[-10:]:
            action = entry["action"]
            table.add_row(
                entry["name"][:21],
                Text(action, style=_action_colour(action)),
                f"{entry['elapsed']:.1f}s" if entry["elapsed"] > 0 else "",
            )

        if not self._results:
            table.add_row("---", Text("waiting", style=_COLOUR_MUTED), "")

        return Panel(table, title="Recent", box=box.SIMPLE,
                      border_style=_COLOUR_MUTED)

    def _render_progress(self) -> Panel:
        st = self._stats
        pct = st.completed / max(st.total, 1)
        bar = _bar_str(pct, 50)
        text = Text()
        text.append(bar)
        text.append(
            f"  {pct * 100:.0f}%  ({st.completed}/{st.total})",
            style=_COLOUR_MUTED,
        )
        return Panel(text, box=box.SIMPLE, border_style=_COLOUR_MUTED)

    # ── plain-text fallback ─────────────────────────────────────────────

    def _print_plain_summary(self) -> None:
        """When rich is not available, print a simple text summary."""
        st = self._stats
        width = 64
        print()
        print("#" * width)
        print(f"  BUILD COMPLETE — {st.elapsed:.0f}s")
        print(f"  {st.total} factors: "
              f"◆ {st.rebuild} rebuild  "
              f"▲ {st.incremental} incremental  "
              f"✓ {st.skip} skip  "
              f"✗ {st.error} error")
        print("#" * width)
        if self._results:
            header = f"  {'Factor':<32} {'Action':<14} {'Time':>8}"
            sep = f"  {'-'*32} {'-'*14} {'-'*8}"
            print(header)
            print(sep)
            for r in self._results[-20:]:
                action_str = r["action"].upper() if r["action"] == "error" else r["action"]
                time_str = f"{r['elapsed']:.1f}s" if r["elapsed"] > 0 else ""
                print(f"  {r['name']:<32} {action_str:<14} {time_str:>8}")
        print()

    def _print_rich_summary(self) -> None:
        """Print a rich formatted summary table."""
        if self._console is None:
            return
        st = self._stats
        self._console.print()
        table = Table(
            title=f"Build Complete — {st.elapsed:.0f}s",
            box=box.ROUNDED,
            border_style=_COLOUR_MUTED,
        )
        table.add_column("Factor", style=_COLOUR_HEADER, width=32)
        table.add_column("Action", width=14)
        table.add_column("Time", justify="right", width=10, style=_COLOUR_MUTED)

        for r in self._results:
            action = r["action"]
            time_str = f"{r['elapsed']:.1f}s" if r["elapsed"] > 0 else ""
            table.add_row(
                r["name"],
                Text(action, style=_action_colour(action)),
                time_str,
            )

        table.add_row()
        table.add_row(
            f"Total: {st.total}",
            Text(
                f"◆ {st.rebuild}  ▲ {st.incremental}  "
                f"✓ {st.skip}  ✗ {st.error}",
                style="bold",
            ),
            f"{st.elapsed:.0f}s",
        )
        self._console.print(table)
        self._console.print()


# ── helpers ─────────────────────────────────────────────────────────────────

def _bar_str(
    pct: float, width: int, idle: bool = False, finished: bool = False,
) -> Text:
    """Build a mini progress-bar ``Text`` object."""
    filled = int(pct * width)
    if idle:
        bar = "─" * width
        return Text(bar, style=_COLOUR_MUTED)
    if finished:
        bar = "█" * width
        return Text(bar, style=_COLOUR_BAR_FINISHED)
    bar = "█" * filled + "░" * (width - filled)
    return Text(bar, style=_COLOUR_BAR_ACTIVE)


def _stage_colour(stage: str) -> str:
    """Return a rich style string for a worker stage."""
    mapping = {
        "init": _COLOUR_MUTED,
        "loading": _COLOUR_LOADING,
        "computing": _COLOUR_COMPUTING,
        "writing": _COLOUR_WRITING,
        "done": _COLOUR_SKIP,
        "error": _COLOUR_ERROR,
        "rebuild": _COLOUR_REBUILD,
        "incremental": _COLOUR_INCREMENTAL,
        "skip": _COLOUR_SKIP,
        "ffill": _COLOUR_LOADING,
        "index": _COLOUR_WRITING,
        "pivot": _COLOUR_LOADING,
        "stack": _COLOUR_WRITING,
        "filter": _COLOUR_COMPUTING,
    }
    return mapping.get(stage, _COLOUR_MUTED)


def _action_colour(action: str) -> str:
    """Return a rich style string for a build action."""
    mapping = {
        "rebuild": _COLOUR_REBUILD,
        "incremental": _COLOUR_INCREMENTAL,
        "skip": _COLOUR_SKIP,
        "error": _COLOUR_ERROR,
    }
    return mapping.get(action, _COLOUR_MUTED)
