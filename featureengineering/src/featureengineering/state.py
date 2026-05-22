from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class BuildState:
    """Tracks build progress so interrupted runs can resume near-instantly.

    On every factor completion the state is flushed to disk.  On restart
    *completed* factors are skipped without touching source parquet files,
    which makes resume effectively free for large factor sets.
    """

    def __init__(self, path: Path, force: bool = False) -> None:
        self._path = path
        self._force = force
        self._data: dict = {}
        self._dirty = False

    # ── load / save ──────────────────────────────────────────────────────────

    def load(self) -> dict:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}
        else:
            self._data = {}

        # Invalidate if force mode changed
        if self._data.get("force") != self._force:
            self._data = {}

        # Auto-recovery: if the previous run crashed after completing phase1,
        # finalize the state so the next run doesn't see a stuck build.
        if self._data.get("status") == "running":
            phase1 = self._data.get("phase1", {})
            if phase1.get("status") == "completed":
                self._data["status"] = "completed_with_errors"
                self._data["finished_at"] = datetime.now(timezone.utc).isoformat()
                errors = phase1.get("errors", [])
                logger.warning(
                    "Recovered stuck build state (%s) — phase1 had completed. "
                    "Errors: %d factors (%s). Marking as completed_with_errors.",
                    self._data.get("build_id", "unknown"),
                    len(errors),
                    ", ".join(errors) if errors else "none",
                )
                self.save()

        return self._data

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            return
        self._data["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = self._path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self._path)
        except (OSError, PermissionError):
            logger.warning("Cannot write build state to %s — permission denied.", self._path)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self, total: int) -> None:
        """Initialise a fresh build record."""
        self._data = {
            "version": 1,
            "build_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": "",
            "force": self._force,
            "status": "running",
            "phase1": {"status": "pending", "total": total, "done": 0,
                       "errors": []},
            "completed": {},
        }
        self.save()

    def mark_phase(self, phase: str, status: str) -> None:
        if phase in self._data:
            self._data[phase]["status"] = status
        self.save()

    def finish(self, status: str = "completed") -> None:
        self._data["status"] = status
        self._data["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.save()

    # ── per-factor tracking ──────────────────────────────────────────────────

    def is_completed(self, factor_name: str) -> bool:
        return factor_name in self._data.get("completed", {})

    def mark_factor(self, factor_name: str, action: str, phase: str) -> None:
        self._data["completed"][factor_name] = {
            "action": action,
            "time": datetime.now(timezone.utc).isoformat(),
        }
        phase_data = self._data.get(phase)
        if phase_data:
            phase_data["done"] = phase_data.get("done", 0) + 1
            if action == "error":
                phase_data.setdefault("errors", []).append(factor_name)
        self.save()

    # ── query ────────────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        return self._data.get("status") == "running"

    @property
    def errors(self) -> list[str]:
        """Return error list from all phases."""
        result: list[str] = []
        for key in ("phase1", "phase2"):
            phase = self._data.get(key)
            if phase and isinstance(phase, dict):
                result.extend(phase.get("errors", []))
        return result

    @property
    def completed_count(self) -> int:
        return len(self._data.get("completed", {}))


def resolve_state_path(project_root: Path, state_file: str = "data/build_state.json") -> Path:
    return (project_root / state_file).resolve()
