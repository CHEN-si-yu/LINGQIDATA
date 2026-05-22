from __future__ import annotations

"""Display and reporting utilities for factor build status, plans, and results."""

# ── Colour / marker constants ─────────────────────────────────────────────────

_COLOR_MAP = {"rebuild": "\033[33m", "incremental": "\033[36m",
              "skip": "\033[32m", "error": "\033[31m"}
_RESET = "\033[0m"

_STATUS_MARKERS = {
    "ok": "✓", "stale": "△", "future": "▶", "error": "✗", "empty": "○",
}


def action_colored(action: str) -> str:
    c = _COLOR_MAP.get(action, "")
    return f"{c}{action}{_RESET}"


# ── Status table ──────────────────────────────────────────────────────────────

def print_status_table(
    date_info: dict[str, dict[str, str | None]],
    title: str = "Factor status",
) -> None:
    """Print a multi-column status table with aligned columns."""
    if not date_info:
        print("No factor files found.")
        return

    effective_end = next(iter(date_info.values()))["effective_end"]
    print(f"\n{'='*72}")
    print(f"  {title}  (effective end: {effective_end})")
    print(f"{'='*72}")
    print(f"  {'Factor':<38} {'Last date':>10}  Status")
    print(f"  {'-'*38} {'-'*10}  {'-'*6}")
    counts: dict[str, int] = {}
    for name, info in date_info.items():
        last = info["last_date"] or "---"
        status = info["status"]
        marker = _STATUS_MARKERS.get(status, "?")
        print(f"  {name:<38} {last:>10}  {marker}  {status}")
        counts[status] = counts.get(status, 0) + 1
    print(f"{'='*72}")
    total = sum(counts.values())
    detail = "  ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"  Total: {total}  |  {detail}")


# ── Plan summary ──────────────────────────────────────────────────────────────

def print_plan_summary(plan: dict[str, tuple[str, str | None]],
                       title: str = "Build plan") -> None:
    """Print a compact build-plan summary."""
    counts: dict[str, int] = {}
    for action, _ in plan.values():
        counts[action] = counts.get(action, 0) + 1
    parts = "  ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"{title}:  {parts}  (total: {len(plan)})")


def print_plan_detail(plan: dict[str, tuple[str, str | None]],
                      force: bool = False) -> None:
    """Print per-factor build plan with color-coded actions."""
    print_plan_summary(plan)
    print()
    action_order = {"rebuild": 0, "incremental": 1, "skip": 2}
    for name in sorted(plan, key=lambda n: (action_order.get(plan[n][0], 9), n)):
        action, reason = plan[name]
        print(f"  {action:<13} {name:<35}  ({reason})")
    print()


# ── Post-build report ─────────────────────────────────────────────────────────

def print_post_build_report(
    date_info: dict[str, dict[str, str | None]],
) -> None:
    """Print a factor date health report after a build completes."""
    if not date_info:
        return
    counts = {"ok": 0, "stale": 0, "future": 0, "error": 0}
    effective_end = next(iter(date_info.values()))["effective_end"]
    print(f"\n{'='*64}")
    print(f"Factor last-date report  (effective end: {effective_end})")
    print(f"{'='*64}")
    print(f"{'Factor':<35} {'Last date':>10}  Status")
    print(f"{'-'*35} {'-'*10}  {'-'*6}")
    for name, info in date_info.items():
        last = info["last_date"] or "---"
        status = info["status"]
        marker = {"ok": "", "stale": "!", "future": ">>", "error": "ERR"}.get(status, "?")
        print(f"{name:<35} {last:>10}  {marker:<4} {status}")
        counts[status] = counts.get(status, 0) + 1
    print(f"{'='*64}")
    total = sum(counts.values())
    print(f"Total: {total}  |  ok: {counts['ok']}  stale: {counts['stale']}  "
          f"future: {counts['future']}  error: {counts['error']}")
