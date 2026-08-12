#!/usr/bin/env python3
"""Deduplicate a FreqAI ``historic_predictions.pkl`` store on ``date_pred``.

FreqAI persists rolling predictions per pair in
``user_data/models/<identifier>/historic_predictions.pkl`` (a ``dict`` mapping a
pair to a ``pandas.DataFrame`` keyed on the candle timestamp copied into
``date_pred``). After a brutal stop (SIGKILL/OOM/power loss), FreqAI's clean-exit
save is skipped and, on restart, its backfill can append the same ``date_pred``
more than once. FreqAI then merges predictions onto the candle frame with
``validate="m:1"`` and raises ``MergeError`` (or, on builds without that guard,
silently row-explodes the merge), which stops that pair from being analyzed.

This tool removes the duplicate ``date_pred`` rows, keeping the most informative
row per timestamp so a real prediction is never discarded in favour of a
backfill placeholder (placeholders are all-zero or all-NaN except ``date_pred``).

Run it inside the freqtrade container so it uses the same pandas that wrote the
file; the host pandas may be unable to unpickle it. Stop the bot first: a running
bot holds the store in memory and would re-persist the corrupt state.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

DEFAULTS: dict[str, Any] = {
    "user_data": "/freqtrade/user_data",
    "quarantine_tag": "corrupt",
    "quarantine_tie_break_limit": 99,
}

# ``date`` and ``date_pred`` are timestamps, never prediction content; excluding
# them keeps the informativeness score focused on actual prediction values.
_CONTENT_EXCLUDE = ("date_pred", "date")


def _informative_score(frame: pd.DataFrame) -> pd.Series:
    """Count informative cells per row (non-null and, for numerics, non-zero).

    FreqAI backfill placeholders are all-zero or all-NaN except ``date_pred``, so
    a plain non-null count would rank a zero-filled placeholder as high as a real
    prediction. Treating zero as non-informative lets a real row always win.
    """
    columns = [column for column in frame.columns if column not in _CONTENT_EXCLUDE]
    if not columns:
        return pd.Series(0, index=frame.index, dtype="int64")
    block = frame[columns]
    numeric = block.apply(pd.to_numeric, errors="coerce")
    is_numeric = numeric.notna()
    informative = (is_numeric & numeric.ne(0)) | (block.notna() & ~is_numeric)
    return informative.sum(axis=1).astype("int64")


def deduplicate_pair(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return the frame with unique ``date_pred`` and the removed-row count."""
    if frame is None or frame.empty or "date_pred" not in frame.columns:
        return frame, 0
    normalized = pd.to_datetime(frame["date_pred"], utc=True, errors="coerce")
    if not normalized.duplicated().any():
        return frame, 0
    work = frame.reset_index(drop=True)
    work = work.assign(
        _dp=normalized.to_numpy(),
        _score=_informative_score(work).to_numpy(),
        _order=work.index.to_numpy(),
    )
    # Sort so the kept row per timestamp is the most informative, breaking ties
    # by the latest original position (most recent write).
    work = work.sort_values(
        ["_dp", "_score", "_order"], kind="stable", na_position="last"
    )
    kept = work.drop_duplicates(subset="_dp", keep="last")
    kept = kept.sort_values("_dp", kind="stable", na_position="last")
    removed = len(frame) - len(kept)
    result = kept.drop(columns=["_dp", "_score", "_order"]).reset_index(drop=True)
    return result, removed


def deduplicate_store(
    store: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]], int]:
    """Deduplicate every pair; return the new store, a report, and total removed."""
    new_store: dict[str, pd.DataFrame] = {}
    report: list[dict[str, Any]] = []
    total_removed = 0
    for pair in sorted(store):
        frame = store[pair]
        rows_before = 0 if frame is None else len(frame)
        deduplicated, removed = deduplicate_pair(frame)
        new_store[pair] = deduplicated
        total_removed += removed
        report.append(
            {
                "pair": pair,
                "rows_before": rows_before,
                "removed": removed,
                "rows_after": rows_before - removed,
            }
        )
        if removed:
            after = deduplicated
            duplicated = pd.to_datetime(
                after["date_pred"], utc=True, errors="coerce"
            ).duplicated()
            if duplicated.any():
                raise AssertionError(f"[{pair}] duplicate date_pred remain after dedup")
    return new_store, report, total_removed


def _quarantine_original(path: Path, now: datetime) -> Path:
    """Copy the pre-dedup file aside as ``<name>.corrupt-<stamp>`` (kept, not lost)."""
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    base = f"{path.name}.{DEFAULTS['quarantine_tag']}-{stamp}"
    for index in range(DEFAULTS["quarantine_tie_break_limit"] + 1):
        suffix = "" if index == 0 else f"-{index}"
        candidate = path.with_name(f"{base}{suffix}")
        if not candidate.exists():
            shutil.copy2(path, candidate)
            return candidate
    raise FileExistsError(path)


def _atomic_write_pickle(store: dict[str, pd.DataFrame], path: Path) -> None:
    """Write the store atomically (temp + fsync + os.replace), preserving mode."""
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        existing_mode = path.stat().st_mode
        payload = pickle.dumps(store, protocol=pickle.HIGHEST_PROTOCOL)
        file_descriptor = os.open(
            temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666
        )
        with os.fdopen(file_descriptor, mode="wb") as write_file:
            os.fchmod(write_file.fileno(), stat.S_IMODE(existing_mode))
            write_file.write(payload)
            write_file.flush()
            os.fsync(write_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def load_store(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as read_file:
        store = pickle.load(read_file)
    if not isinstance(store, dict):
        raise TypeError(f"{path}: expected a dict of DataFrames, got {type(store)!r}")
    return store


def find_targets(
    user_data: Path, identifier: str | None, path: str | None
) -> list[Path]:
    if path is not None:
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(target)
        return [target]
    if identifier is not None:
        target = user_data / "models" / identifier / "historic_predictions.pkl"
        if not target.is_file():
            raise FileNotFoundError(target)
        return [target]
    return sorted(user_data.glob("models/*/historic_predictions.pkl"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--user-data",
        default=DEFAULTS["user_data"],
        help="user_data directory (container path); default %(default)s",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--identifier", help="repair only models/<identifier>/")
    selector.add_argument("--path", help="repair one explicit historic_predictions.pkl")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write changes; without it the tool only reports (dry-run)",
    )
    return parser.parse_args(argv)


def _print_report(
    path: Path, report: list[dict[str, Any]], removed: int, apply: bool
) -> None:
    mode = "apply" if apply else "dry-run"
    print(f"# {path} | mode={mode} | pandas={pd.__version__}")
    print(f"{'pair':<24}{'rows_before':>12}{'removed':>10}{'rows_after':>12}")
    for row in report:
        if row["removed"]:
            print(
                f"{row['pair']:<24}{row['rows_before']:>12}"
                f"{row['removed']:>10}{row['rows_after']:>12}"
            )
    print(f"# total duplicate rows removed: {removed}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    user_data = Path(args.user_data)
    targets = find_targets(user_data, args.identifier, args.path)
    if not targets:
        print(f"# no historic_predictions.pkl found under {user_data}/models/")
        return 0
    changed_files = 0
    for target in targets:
        store = load_store(target)
        new_store, report, removed = deduplicate_store(store)
        _print_report(target, report, removed, args.apply)
        if removed and args.apply:
            quarantine = _quarantine_original(target, datetime.now(timezone.utc))
            _atomic_write_pickle(new_store, target)
            print(
                f"# quarantined original to {quarantine.name}; wrote deduplicated store"
            )
            changed_files += 1
        elif removed:
            print("# dry-run: re-run with --apply to write changes")
    print(f"# files scanned: {len(targets)} | files changed: {changed_files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
