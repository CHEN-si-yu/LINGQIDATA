"""
因子评估进度可视化

基于 Rich 的实时评估仪表盘，展示:
- 总体进度 (因子 × target 组合数)
- 每 worker 的当前处理状态
- 实时的 ICIR / 多空收益统计
- ETA 预估
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


_COLOUR_HEADER = "#FFFFFF"
_COLOUR_MUTED = "#888888"
_COLOUR_BAR_ACTIVE = "#4D96FF"
_COLOUR_BAR_FINISHED = "#6BCB77"
_COLOUR_GOOD = "#6BCB77"
_COLOUR_WARN = "#FFD93D"
_COLOUR_BAD = "#FF6B6B"
_COLOUR_INFO = "#4D96FF"


@dataclass
class EvalStats:
    """Running counters for evaluation progress."""

    completed: int = 0
    total: int = 0
    t_start: float = field(default_factory=time.perf_counter)
    running_icir_sum: float = 0.0
    running_icir_count: int = 0
    good_count: int = 0   # |ICIR| >= 0.5
    warn_count: int = 0   # 0.1 <= |ICIR| < 0.5
    bad_count: int = 0    # |ICIR| < 0.1
    failed_count: int = 0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t_start

    @property
    def pct(self) -> float:
        return self.completed / max(self.total, 1)

    @property
    def remaining(self) -> float:
        if self.completed == 0:
            return 0.0
        rate = self.completed / max(self.elapsed, 0.001)
        pending = self.total - self.completed
        return pending / rate if rate > 0 else 0.0

    @property
    def avg_icir(self) -> float:
        if self.icir_count == 0:
            return 0.0
        return self.running_icir_sum / self.icir_count


class EvalDashboard:
    """因子评估实时仪表盘"""

    def __init__(
        self,
        total_combinations: int,
        max_workers: int = 16,
        refresh_per_second: int = 4,
    ) -> None:
        import sys as _sys

        self._stats = EvalStats(total=total_combinations)
        self._max_workers = max_workers
        self._slots: dict[int, dict] = {i: {"name": "", "progress": 0.0} for i in range(max_workers)}
        self._recent: list[dict] = []

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

    def __enter__(self) -> "EvalDashboard":
        if self._live is not None:
            self._live.start()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._live is not None:
            self._live.stop()

    def update_worker(self, slot: int, name: str = "", pct: float = 0.0) -> None:
        s = self._slots.get(slot)
        if s is None:
            return
        if name:
            s["name"] = name
        s["progress"] = pct
        if self._live is not None:
            self._live.refresh()

    def mark_done(self, name: str, icir: float | None = None, ok: bool = True) -> None:
        self._stats.completed += 1
        if not ok:
            self._stats.failed_count += 1
        elif icir is not None:
            self._stats.running_icir_sum += abs(icir)
            self._stats.icir_count += 1
            if abs(icir) >= 0.5:
                self._stats.good_count += 1
            elif abs(icir) >= 0.1:
                self._stats.warn_count += 1
            else:
                self._stats.bad_count += 1

        self._recent.append({"name": name, "icir": icir, "ok": ok})
        if len(self._recent) > 20:
            self._recent = self._recent[-20:]

        if self._live is not None:
            self._live.refresh()

    def refresh(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def print_summary(self) -> None:
        if self._console is not None:
            self._console.print(self._render_summary())
        else:
            self._print_plain_summary()

    def _render(self) -> Layout:
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="progress", size=3),
        )
        layout["body"].split_row(
            Layout(name="workers", ratio=2),
            Layout(name="recent", ratio=1),
        )
        layout["header"].update(self._render_header())
        layout["workers"].update(self._render_workers())
        layout["recent"].update(self._render_recent())
        layout["progress"].update(self._render_progress())
        return layout

    def _render_header(self) -> Panel:
        st = self._stats
        status = "● RUNNING" if st.completed < st.total else "● DONE"
        status_style = _COLOUR_BAR_ACTIVE if st.completed < st.total else _COLOUR_BAR_FINISHED
        eta = f"{st.remaining:.0f}s" if st.remaining < 120 else f"{st.remaining / 60:.1f}m"

        text = Text()
        text.append("█" * 60 + "\n", style=_COLOUR_MUTED)
        text.append("  Factor Evaluation  ", style=f"bold {_COLOUR_HEADER}")
        text.append(f"    {status}", style=f"bold {status_style}")
        text.append(
            f"    Elapsed: {st.elapsed:.0f}s    ETA: ~{eta}    "
            f"{st.completed}/{st.total}\n",
            style=_COLOUR_MUTED,
        )
        text.append("█" * 60, style=_COLOUR_MUTED)
        return Panel(text, style=_COLOUR_MUTED, box=box.SIMPLE)

    def _render_workers(self) -> Panel:
        table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1), border_style=_COLOUR_MUTED)
        table.add_column("#", width=4, style=_COLOUR_MUTED)
        table.add_column("factor", width=30, style=_COLOUR_HEADER)
        table.add_column("progress", width=20)

        for i in range(min(16, self._max_workers)):
            s = self._slots.get(i, {"name": "", "progress": 0.0})
            name = s["name"][:29] if s["name"] else "---"
            pct = s["progress"]
            bar = _bar_str(pct, 16)
            table.add_row(f"[{i+1}]", name, bar)

        return Panel(table, title="Workers", box=box.SIMPLE, border_style=_COLOUR_MUTED)

    def _render_recent(self) -> Panel:
        table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1), border_style=_COLOUR_MUTED)
        table.add_column("factor", width=22, style=_COLOUR_HEADER)
        table.add_column("ICIR", width=10)
        table.add_column("status", width=8)

        for entry in self._recent[-10:]:
            icir_str = f"{entry['icir']:+.4f}" if entry["icir"] is not None else "N/A"
            status = "OK" if entry["ok"] else "ERR"
            s_style = _COLOUR_GOOD if entry["ok"] else _COLOUR_BAD
            table.add_row(entry["name"][:21], icir_str, Text(status, style=s_style))

        if not self._recent:
            table.add_row("---", "---", Text("waiting", style=_COLOUR_MUTED))

        return Panel(table, title="Recent", box=box.SIMPLE, border_style=_COLOUR_MUTED)

    def _render_progress(self) -> Panel:
        st = self._stats
        bar = _bar_str(st.pct, 50)
        text = Text()
        text.append(bar)
        text.append(f"  {st.pct*100:.0f}%", style=_COLOUR_MUTED)
        text.append(
            f"  ✓{st.good_count}  ~{st.warn_count}  ✗{st.bad_count}  ✖{st.failed_count}",
            style=_COLOUR_MUTED,
        )
        text.append(f"  avg|ICIR|={st.avg_icir:.3f}", style=_COLOUR_INFO)
        return Panel(text, box=box.SIMPLE, border_style=_COLOUR_MUTED)

    def _render_summary(self) -> Panel:
        st = self._stats
        table = Table(title=f"Evaluation Complete — {st.elapsed:.0f}s", box=box.ROUNDED)
        table.add_column("Metric", style=_COLOUR_HEADER)
        table.add_column("Value", justify="right")
        table.add_row("Total combinations", str(st.total))
        table.add_row("Completed", str(st.completed))
        table.add_row("Good (|ICIR| ≥ 0.5)", str(st.good_count))
        table.add_row("Warning (0.1 ≤ |ICIR| < 0.5)", str(st.warn_count))
        table.add_row("Bad (|ICIR| < 0.1)", str(st.bad_count))
        table.add_row("Failed", str(st.failed_count))
        table.add_row("Avg |ICIR|", f"{st.avg_icir:.4f}")
        table.add_row("Elapsed", f"{st.elapsed:.0f}s")
        return table

    def _print_plain_summary(self) -> None:
        st = self._stats
        print(f"\n{'='*60}")
        print(f"  Evaluation Complete — {st.elapsed:.0f}s")
        print(f"  Total: {st.total}  Good: {st.good_count}  "
              f"Warn: {st.warn_count}  Bad: {st.bad_count}  Failed: {st.failed_count}")
        print(f"  Avg |ICIR|: {st.avg_icir:.4f}")
        print(f"{'='*60}")


def _bar_str(pct: float, width: int, idle: bool = False, finished: bool = False) -> Text:
    filled = int(pct * width)
    if idle:
        return Text("─" * width, style=_COLOUR_MUTED)
    if finished:
        return Text("█" * width, style=_COLOUR_BAR_FINISHED)
    return Text("█" * filled + "░" * (width - filled), style=_COLOUR_BAR_ACTIVE)
