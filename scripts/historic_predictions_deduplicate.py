#!/usr/bin/env python3
"""Deduplicate a FreqAI ``historic_predictions.pkl`` store on ``date_pred``.

FreqAI persists rolling predictions per pair in
``user_data/models/<identifier>/historic_predictions.pkl`` (a ``dict`` mapping a
pair to a ``pandas.DataFrame`` keyed on the candle timestamp copied into
``date_pred``). After a brutal stop (SIGKILL/OOM/power loss), FreqAI's clean-exit
save is skipped and, on restart, its backfill can leave the store with the same
``date_pred`` appearing more than once. FreqAI then merges predictions onto the
candle frame with a left join, so duplicate keys silently row-explode the merge;
only builds that add a ``validate="m:1"`` guard raise ``MergeError`` instead,
which stops that pair from being analyzed.

This tool removes the duplicate ``date_pred`` rows, keeping the most informative
row per timestamp: rows are ranked by informative cells (non-null and, for
numerics, non-zero), then by non-null count, then by most-recent write. A real
prediction is therefore never discarded in favour of an all-NaN placeholder; a
real all-zero row and a zero-filled placeholder are byte-identical, so collapsing
them loses nothing. FreqAI also mirrors the store to
``historic_predictions.backup.pkl`` on every clean save and falls back to it
only when the primary is truncated (loading it raises ``EOFError``), so the tool
deduplicates the backup as well.

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
    "models_dirname": "models",
    "store_filename": "historic_predictions.pkl",
    "backup_filename": "historic_predictions.backup.pkl",
    "quarantine_tag": "corrupt",
    "quarantine_tie_break_limit": 99,
}

# ``date`` and ``date_pred`` are timestamps, never prediction content; excluding
# them keeps the informativeness score focused on actual prediction values.
_CONTENT_EXCLUDE = ("date_pred", "date")


def _content_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in _CONTENT_EXCLUDE]


def _informative_score(frame: pd.DataFrame) -> pd.Series:
    """Count informative cells per row (non-null and, for numerics, non-zero).

    FreqAI backfill placeholders are all-zero or all-NaN except ``date_pred``, so
    a plain non-null count would rank a zero-filled placeholder as high as a real
    prediction. Treating zero as non-informative lets a real row outrank a zero
    placeholder; ties against an all-NaN placeholder are then broken by the
    non-null count (see ``_nonnull_count``).
    """
    columns = _content_columns(frame)
    if not columns:
        return pd.Series(0, index=frame.index, dtype="int64")
    block = frame[columns]
    numeric = block.apply(pd.to_numeric, errors="coerce")
    is_numeric = numeric.notna()
    informative = (is_numeric & numeric.ne(0)) | (block.notna() & ~is_numeric)
    return informative.sum(axis=1).astype("int64")


def _nonnull_count(frame: pd.DataFrame) -> pd.Series:
    """Count non-null content cells per row (tie-break below informativeness).

    A real all-zero row has non-null zeros; an all-NaN placeholder does not, so
    this keeps the real row when both score zero on informativeness.
    """
    columns = _content_columns(frame)
    if not columns:
        return pd.Series(0, index=frame.index, dtype="int64")
    return frame[columns].notna().sum(axis=1).astype("int64")


def deduplicate_pair(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return the frame with unique non-NaT ``date_pred`` and the removed count.

    Only rows with a valid (non-NaT) ``date_pred`` can duplicate FreqAI's
    many-to-one merge key, so NaT rows are preserved untouched.
    """
    if frame is None or frame.empty or "date_pred" not in frame.columns:
        return frame, 0
    normalized = pd.to_datetime(frame["date_pred"], utc=True, errors="coerce")
    valid = normalized.notna()
    if not normalized[valid].duplicated().any():
        return frame, 0
    work = frame.reset_index(drop=True)
    work = work.assign(
        _dp=normalized.to_numpy(),
        _valid=valid.to_numpy(),
        _score=_informative_score(work).to_numpy(),
        _nonnull=_nonnull_count(work).to_numpy(),
        _order=work.index.to_numpy(),
    )
    # Among duplicate timestamps keep the most informative row, breaking ties by
    # non-null count (a real all-zero row beats an all-NaN placeholder) then by
    # the latest original position; NaT rows are carried through untouched.
    contested = work[work["_valid"]].sort_values(
        ["_dp", "_score", "_nonnull", "_order"], kind="stable"
    )
    kept_contested = contested.drop_duplicates(subset="_dp", keep="last")
    kept = pd.concat([kept_contested, work[~work["_valid"]]])
    kept = kept.sort_values("_dp", kind="stable", na_position="last")
    removed = len(frame) - len(kept)
    result = kept.drop(columns=["_dp", "_valid", "_score", "_nonnull", "_order"]).reset_index(
        drop=True
    )
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
            after_normalized = pd.to_datetime(deduplicated["date_pred"], utc=True, errors="coerce")
            if after_normalized[after_normalized.notna()].duplicated().any():
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
    """Write the store atomically (temp + fsync + os.replace), preserving mode/owner.

    Uses the stdlib ``pickle``: FreqAI writes the store with joblib's vendored
    cloudpickle, but a plain ``dict`` of DataFrames pickles to a standard stream
    that ``pickle`` round-trips and FreqAI's ``cloudpickle.load`` reads back;
    standalone ``cloudpickle`` is not importable in the freqtrade image.
    """
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        existing_stat = path.stat()
        file_descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(file_descriptor, mode="wb") as write_file:
            temporary_stat = os.fstat(write_file.fileno())
            if (
                temporary_stat.st_uid != existing_stat.st_uid
                or temporary_stat.st_gid != existing_stat.st_gid
            ):
                # Best-effort: a non-root process on a cross-uid bind mount lacks
                # CAP_CHOWN; the store was never chowned before, so failure here
                # must not abort the repair.
                try:
                    os.fchown(
                        write_file.fileno(),
                        existing_stat.st_uid
                        if temporary_stat.st_uid != existing_stat.st_uid
                        else -1,
                        existing_stat.st_gid
                        if temporary_stat.st_gid != existing_stat.st_gid
                        else -1,
                    )
                except PermissionError:
                    pass
            os.fchmod(write_file.fileno(), stat.S_IMODE(existing_stat.st_mode))
            pickle.dump(store, write_file, protocol=pickle.HIGHEST_PROTOCOL)
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


def find_targets(user_data: Path, identifier: str | None, path: str | None) -> list[Path]:
    if path is not None:
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(target)
        return [target]
    filenames = (DEFAULTS["store_filename"], DEFAULTS["backup_filename"])
    models = user_data / DEFAULTS["models_dirname"]
    if identifier is not None:
        directory = models / identifier
        existing = [directory / name for name in filenames if (directory / name).is_file()]
        if not existing:
            raise FileNotFoundError(directory / DEFAULTS["store_filename"])
        return existing
    found: list[Path] = []
    for name in filenames:
        found.extend(models.glob(f"*/{name}"))
    return sorted(found)


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


def _print_report(path: Path, report: list[dict[str, Any]], removed: int, apply: bool) -> None:
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
        print(
            f"# no {DEFAULTS['store_filename']} found under "
            f"{user_data}/{DEFAULTS['models_dirname']}/",
            file=sys.stderr,
        )
        return 0
    changed_files = 0
    skipped_files = 0
    for target in targets:
        try:
            store = load_store(target)
        except (OSError, EOFError, pickle.UnpicklingError, TypeError) as error:
            skipped_files += 1
            print(f"# {target}: skipped unreadable store: {error!r}", file=sys.stderr)
            continue
        new_store, report, removed = deduplicate_store(store)
        _print_report(target, report, removed, args.apply)
        if removed and args.apply:
            quarantine = _quarantine_original(target, datetime.now(timezone.utc))
            _atomic_write_pickle(new_store, target)
            print(f"# quarantined original to {quarantine.name}; wrote deduplicated store")
            changed_files += 1
        elif removed:
            print("# dry-run: re-run with --apply to write changes", file=sys.stderr)
    print(
        f"# files scanned: {len(targets)} | files changed: {changed_files} "
        f"| files skipped: {skipped_files}"
    )
    return 1 if skipped_files else 0


if __name__ == "__main__":
    sys.exit(main())
