#!/usr/bin/env python3
"""
Delete the latest day's data from all .fea factor files.

This script removes the most recent date from every .fea file in data/factors/
and data/targets/, and updates the corresponding manifest (.json) and done (.done)
files to match the new state.

Purpose: prepare data for testing build_factors.py incremental-update logic.
After running this script with --go, re-running build_factors.py should detect
that the latest day is missing and perform an incremental rebuild.

Usage:
    python delete.py              # dry-run — delete the MAX date in each file
    python delete.py --go         # actually delete
    python delete.py --date 20260602 --go   # delete a specific date (skip if absent)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _find_fea_files(root: Path) -> list[Path]:
    """Return all .fea files under data/factors/ and data/targets/, sorted."""
    fea_files: list[Path] = []
    for subdir in ("factors", "targets"):
        d = root / "data" / subdir
        if d.is_dir():
            fea_files.extend(sorted(d.glob("*.fea")))
    return fea_files


def _find_manifest(fea_path: Path, root: Path) -> Optional[Path]:
    """Return the companion manifest .json path, or None if it doesn't exist."""
    if fea_path.parent.name == "targets":
        manifest = fea_path.with_suffix(".json")
    else:
        manifest = root / "data" / "manifests" / f"{fea_path.stem}.json"
    return manifest if manifest.is_file() else None


def _find_done_file(fea_path: Path, root: Path) -> Optional[Path]:
    """Return the companion .done file path, or None.

    .done files only exist for factors, not targets.
    """
    if fea_path.parent.name == "targets":
        return None
    done = root / "data" / "manifests" / f"{fea_path.stem}.done"
    return done if done.is_file() else None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _read_with_date_index(fea_path: Path) -> tuple[pd.DataFrame, pd.Index]:
    """Read a .fea file and return (DataFrame, date_index).

    Ensures the index is named "Date" and contains YYYYMMDD strings.
    """
    df = pd.read_feather(fea_path)

    # Case 1: index is already named "Date" (normal path)
    if isinstance(df.index.name, str) and df.index.name.lower() == "date":
        date_idx = df.index.astype(str)
        date_idx.name = "Date"
        df.index = date_idx
        return df, date_idx

    # Case 2: index is set but has a different name — try to detect date-like index
    sample = str(df.index[0]) if len(df) > 0 else ""
    if sample.isdigit() and len(sample) == 8:
        date_idx = df.index.astype(str)
        date_idx.name = "Date"
        df.index = date_idx
        return df, date_idx

    # Case 3: "Date" is a column, not the index
    if "Date" in df.columns:
        df = df.set_index("Date")
        date_idx = df.index.astype(str)
        date_idx.name = "Date"
        df.index = date_idx
        return df, date_idx

    # Fallback: look for the first column that looks like YYYYMMDD dates
    for col in df.columns:
        sample = str(df.iloc[0][col]) if len(df) > 0 else ""
        if sample.isdigit() and len(sample) == 8:
            df = df.set_index(col)
            date_idx = df.index.astype(str)
            date_idx.name = "Date"
            df.index = date_idx
            return df, date_idx

    raise ValueError(
        f"Cannot identify Date index/column in {fea_path}. "
        f"Index name={df.index.name!r}, columns={list(df.columns[:5])}"
    )


def _recompute_manifest_fields(df: pd.DataFrame) -> dict:
    """Return the manifest fields that change when rows are removed."""
    dates = df.index.astype(str)
    last_date = str(dates[-1])
    last_day_stock_count = int(df.loc[last_date].notna().sum())

    total_cells = len(df) * len(df.columns)
    non_null_cells = int(df.notna().sum().sum())

    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "non_null_cells": non_null_cells,
        "coverage_ratio": float(non_null_cells / total_cells) if total_cells else 0.0,
        "last_date": last_date,
        "last_day_stock_count": last_day_stock_count,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }


def _resolve_date_to_delete(date_idx: pd.Index, specified_date: str | None) -> tuple[str, bool]:
    """Resolve which date to delete for this file.

    Args:
        date_idx: The sorted Date index of the file.
        specified_date: The --date argument (may be None).

    Returns:
        (date_to_delete, is_default): The date string to delete, and whether it
        was auto-resolved from the file's max date (True) or user-specified (False).
    """
    max_date = str(date_idx[-1])

    if specified_date is None:
        # Default mode: delete each file's own maximum date
        return max_date, True
    else:
        # Explicit mode: delete the specified date (will be checked for existence later)
        return specified_date, False


def process_one(
    fea_path: Path,
    specified_date: str | None,
    root: Path,
    *,
    dry_run: bool = True,
) -> dict:
    """Delete the latest day from a single .fea file.

    If *specified_date* is None, the file's own maximum date is deleted.
    Otherwise only that exact date is deleted (skipped if not present).

    Returns a status dict summarising what happened.
    """
    rel = fea_path.relative_to(root)
    result = {
        "file": str(rel),
        "status": "skipped",
        "reason": "",
        "old_rows": 0,
        "new_rows": 0,
        "old_last_date": "",
        "new_last_date": "",
    }

    # ---- read ---------------------------------------------------------------
    try:
        df, date_idx = _read_with_date_index(fea_path)
    except Exception as exc:
        result["status"] = "error"
        result["reason"] = f"Read error: {exc}"
        return result

    result["old_rows"] = len(df)
    result["old_last_date"] = str(date_idx[-1])

    # ---- resolve which date to delete ---------------------------------------
    date_to_delete, is_default = _resolve_date_to_delete(date_idx, specified_date)

    # ---- check existence ----------------------------------------------------
    if date_to_delete not in date_idx:
        # Build a clear skip message
        if is_default:
            # Should never happen (max date is always in the index), but be safe
            result["reason"] = "No data to delete (empty index?)"
        else:
            result["reason"] = (
                f"Date {date_to_delete} not in [{date_idx[0]}, {date_idx[-1]}]"
            )
        result["new_rows"] = result["old_rows"]
        result["new_last_date"] = result["old_last_date"]
        return result

    # ---- drop the row -------------------------------------------------------
    mask = date_idx != date_to_delete
    df_new = df.loc[mask].copy()

    new_idx = df_new.index.astype(str)
    if len(new_idx) == 0:
        result["status"] = "error"
        result["reason"] = (
            f"Refusing to delete the *only* row ({date_to_delete}). "
            f"The file would become empty."
        )
        return result

    result["new_rows"] = len(df_new)
    result["new_last_date"] = str(new_idx[-1])

    if dry_run:
        result["status"] = "would_delete"
        return result

    # ---- write back (.fea) --------------------------------------------------
    tmp = fea_path.with_suffix(".fea.tmp")
    try:
        df_new.to_feather(tmp)
        tmp.replace(fea_path)
    except Exception as exc:
        result["status"] = "error"
        result["reason"] = f"Write error: {exc}"
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return result

    # ---- update manifest (.json) --------------------------------------------
    manifest_path = _find_manifest(fea_path, root)
    if manifest_path is not None:
        try:
            _update_manifest(manifest_path, df_new)
        except Exception as exc:
            result["reason"] = f"Manifest update error: {exc}"
            # .fea was already written — don't roll back, just report

    # ---- update done file (.done) -------------------------------------------
    done_path = _find_done_file(fea_path, root)
    if done_path is not None:
        try:
            _update_done(done_path, action="strip_latest_day")
        except Exception:
            pass  # non-critical

    result["status"] = "deleted"
    return result


def _update_manifest(manifest_path: Path, df: pd.DataFrame) -> None:
    """Patch a manifest JSON in-place with recomputed stats from *df*."""
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    manifest.update(_recompute_manifest_fields(df))
    # Ensure list-typed keys are preserved as lists
    manifest["dependencies"] = list(manifest.get("dependencies", []))
    manifest["index_names"] = list(manifest.get("index_names", ["Date"]))

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def _update_done(done_path: Path, *, action: str) -> None:
    """Overwrite the .done file with the new action."""
    payload = {
        "name": done_path.stem,
        "action": action,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    done_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Delete the latest day from all .fea factor files "
                    "(for testing incremental rebuild)."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date to delete in YYYYMMDD format. "
             "If omitted, each file's own maximum date is deleted.",
    )
    parser.add_argument(
        "--go",
        action="store_true",
        help="Actually perform the deletion (default: dry-run)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Project root directory (default: directory containing this script)",
    )
    args = parser.parse_args(argv)

    # Resolve root
    if args.root:
        root = Path(args.root).resolve()
    else:
        root = Path(__file__).resolve().parent

    if not root.is_dir():
        print(f"ERROR: root directory does not exist: {root}", file=sys.stderr)
        raise SystemExit(1)

    # Discover .fea files
    fea_files = _find_fea_files(root)
    if not fea_files:
        print(
            f"ERROR: no .fea files found under {root}/data/factors/ or "
            f"{root}/data/targets/",
            file=sys.stderr,
        )
        raise SystemExit(1)

    mode = "DRY RUN" if not args.go else "LIVE"
    specified_date: str | None = args.date

    width = 72
    print("=" * width)
    print(f"  delete.py  —  {mode}")
    if specified_date:
        print(f"  Date to delete : {specified_date}  (explicit)")
    else:
        print(f"  Date to delete : <max date per file>  (auto)")
    print(f"  Project root   : {root}")
    print(f"  .fea files     : {len(fea_files)}")
    print("=" * width)
    print()

    # Process each file
    stats: dict[str, int] = {}

    for i, fea_path in enumerate(fea_files, 1):
        rel = fea_path.relative_to(root)
        res = process_one(fea_path, specified_date, root, dry_run=not args.go)

        stats[res["status"]] = stats.get(res["status"], 0) + 1

        if res["status"] in ("deleted", "would_delete"):
            print(
                f"[{i:4d}/{len(fea_files)}] {res['status']:>12s}  {rel!s:60s}  "
                f"rows: {res['old_rows']} → {res['new_rows']}, "
                f"latest: {res['old_last_date']} → {res['new_last_date']}"
            )
        elif res["status"] == "error":
            print(
                f"[{i:4d}/{len(fea_files)}] {res['status']:>12s}  {rel!s:60s}  "
                f"{res['reason']}"
            )
        # skipped: only print first 5, then every 100th to keep output manageable
        elif i <= 5 or i % 100 == 0:
            print(
                f"[{i:4d}/{len(fea_files)}] {res['status']:>12s}  {rel!s:60s}  "
                f"{res['reason']}"
            )

    # Summary
    print()
    print("=" * width)
    print(f"  SUMMARY ({mode})")
    for k in ("deleted", "would_delete", "skipped", "error"):
        v = stats.get(k, 0)
        if v:
            print(f"    {k}: {v}")
    print("=" * width)

    if not args.go:
        print()
        print("  This was a DRY RUN.  Re-run with --go to actually delete.")
        print()

    # Exit non-zero on errors
    if stats.get("error", 0) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
