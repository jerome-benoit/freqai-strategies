#!/usr/bin/env python3
"""Tests for the historic_predictions deduplication tool.

The dedup logic is pure pandas and version-agnostic, so these run under any
pandas 2.x/3.x. Runnable with pytest or directly (``python <this file>``); the
container ships pandas but not necessarily pytest.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_MODULE_PATH = Path(__file__).resolve().parent.parent / "historic_predictions_deduplicate.py"
_spec = importlib.util.spec_from_file_location("hp_dedup", _MODULE_PATH)
assert _spec and _spec.loader
hp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hp)


def _row(date_pred: str, extrema: float, do_predict: int, close: float) -> dict:
    return {
        "date_pred": pd.Timestamp(date_pred, tz="UTC"),
        "&s-extrema": extrema,
        "do_predict": do_predict,
        "close_price": close,
    }


def _placeholder(date_pred: str, *, zero_filled: bool) -> dict:
    value = 0 if zero_filled else float("nan")
    return {
        "date_pred": pd.Timestamp(date_pred, tz="UTC"),
        "&s-extrema": value,
        "do_predict": value,
        "close_price": value,
    }


def test_real_then_zero_placeholder_keeps_real() -> None:
    frame = pd.DataFrame(
        [
            _row("2026-08-12 12:10:00", 0.9, 1, 100.0),
            _placeholder("2026-08-12 12:15:00", zero_filled=True),
            _row("2026-08-12 12:15:00", 0.7, -1, 101.0),
        ]
    )
    result, removed = hp.deduplicate_pair(frame)
    assert removed == 1
    assert not result["date_pred"].duplicated().any()
    kept = result[result["date_pred"] == pd.Timestamp("2026-08-12 12:15:00", tz="UTC")]
    assert kept["&s-extrema"].iloc[0] == 0.7  # real row kept, not the zero placeholder


def test_nan_placeholder_dropped() -> None:
    frame = pd.DataFrame(
        [
            _placeholder("2026-08-12 12:15:00", zero_filled=False),
            _row("2026-08-12 12:15:00", 0.6, 1, 100.0),
        ]
    )
    result, removed = hp.deduplicate_pair(frame)
    assert removed == 1
    assert result["&s-extrema"].iloc[0] == 0.6


def test_two_real_rows_keeps_most_recent() -> None:
    frame = pd.DataFrame(
        [
            _row("2026-08-12 12:15:00", 0.5, 1, 100.0),
            _row("2026-08-12 12:15:00", 0.7, 1, 100.0),
        ]
    )
    result, removed = hp.deduplicate_pair(frame)
    assert removed == 1
    assert result["&s-extrema"].iloc[0] == 0.7  # latest write among equal completeness


def test_triple_dup_real_in_middle() -> None:
    frame = pd.DataFrame(
        [
            _placeholder("2026-08-12 12:15:00", zero_filled=True),
            _row("2026-08-12 12:15:00", 0.8, 1, 100.0),
            _placeholder("2026-08-12 12:15:00", zero_filled=False),
        ]
    )
    result, removed = hp.deduplicate_pair(frame)
    assert removed == 2
    assert result["&s-extrema"].iloc[0] == 0.8


def test_clean_pair_is_noop() -> None:
    frame = pd.DataFrame(
        [
            _row("2026-08-12 12:10:00", 0.1, 1, 100.0),
            _row("2026-08-12 12:15:00", 0.2, 1, 101.0),
        ]
    )
    result, removed = hp.deduplicate_pair(frame)
    assert removed == 0
    assert result.equals(frame)


def test_sorted_ascending_after_dedup() -> None:
    frame = pd.DataFrame(
        [
            _row("2026-08-12 12:20:00", 0.3, 1, 100.0),
            _row("2026-08-12 12:15:00", 0.5, 1, 100.0),
            _row("2026-08-12 12:15:00", 0.7, 1, 100.0),
        ]
    )
    result, removed = hp.deduplicate_pair(frame)
    assert removed == 1
    assert list(result["date_pred"]) == sorted(result["date_pred"])


def test_idempotent() -> None:
    frame = pd.DataFrame(
        [
            _placeholder("2026-08-12 12:15:00", zero_filled=True),
            _row("2026-08-12 12:15:00", 0.7, 1, 100.0),
        ]
    )
    once, _ = hp.deduplicate_pair(frame)
    twice, removed2 = hp.deduplicate_pair(once)
    assert removed2 == 0
    assert twice.equals(once)


def test_edge_cases_do_not_crash() -> None:
    empty, removed = hp.deduplicate_pair(pd.DataFrame())
    assert removed == 0 and empty.empty
    no_key = pd.DataFrame([{"&s-extrema": 0.1}])
    result, removed = hp.deduplicate_pair(no_key)
    assert removed == 0 and result.equals(no_key)
    nat = pd.DataFrame(
        [
            {
                "date_pred": pd.NaT,
                "&s-extrema": 0.1,
                "do_predict": 1,
                "close_price": 1.0,
            },
            {
                "date_pred": pd.NaT,
                "&s-extrema": 0.2,
                "do_predict": 1,
                "close_price": 1.0,
            },
        ]
    )
    result, removed = hp.deduplicate_pair(nat)
    assert removed == 0  # distinct NaT rows are preserved (never match the merge)
    assert len(result) == 2


def test_store_roundtrip_and_atomic_write(tmp_path: Path) -> None:
    store = {
        "SUI/USD:USD": pd.DataFrame(
            [
                _row("2026-08-12 12:10:00", 0.9, 1, 100.0),
                _placeholder("2026-08-12 12:15:00", zero_filled=True),
                _row("2026-08-12 12:15:00", 0.7, -1, 101.0),
            ]
        ),
        "XRP/USD:USD": pd.DataFrame([_row("2026-08-12 12:10:00", 0.1, 1, 1.0)]),
    }
    path = tmp_path / "historic_predictions.pkl"
    with path.open("wb") as handle:
        pickle.dump(store, handle)
    loaded = hp.load_store(path)
    new_store, report, removed = hp.deduplicate_store(loaded)
    assert removed == 1
    hp._atomic_write_pickle(new_store, path)
    reloaded = hp.load_store(path)
    for frame in reloaded.values():
        assert not frame["date_pred"].duplicated().any()
    assert {row["pair"] for row in report} == set(store)


def test_real_all_zero_row_beats_nan_placeholder() -> None:
    frame = pd.DataFrame(
        [
            _row("2026-08-12 12:15:00", 0.0, 0, 0.0),  # real, all-zero, written first
            _placeholder("2026-08-12 12:15:00", zero_filled=False),  # all-NaN, later
        ]
    )
    result, removed = hp.deduplicate_pair(frame)
    assert removed == 1
    assert result["&s-extrema"].notna().all()  # NaN placeholder dropped, real kept


def test_distinct_nat_rows_preserved_with_real_dup() -> None:
    frame = pd.DataFrame(
        [
            {"date_pred": pd.NaT, "&s-extrema": 0.11, "do_predict": 1, "close_price": 10.0},
            {"date_pred": pd.NaT, "&s-extrema": 0.22, "do_predict": 1, "close_price": 20.0},
            _row("2026-08-12 12:15:00", 0.3, 1, 30.0),
            _row("2026-08-12 12:15:00", 0.4, 1, 30.0),
        ]
    )
    result, removed = hp.deduplicate_pair(frame)
    assert removed == 1  # only the real duplicate collapses; both NaT rows kept
    assert len(result) == 3
    assert {0.11, 0.22} <= set(result["&s-extrema"])


def _dup_store() -> dict:
    return {
        "SUI/USD:USD": pd.DataFrame(
            [
                _row("2026-08-12 12:10:00", 0.9, 1, 100.0),
                _placeholder("2026-08-12 12:15:00", zero_filled=True),
                _row("2026-08-12 12:15:00", 0.7, -1, 101.0),
            ]
        )
    }


def test_find_targets_glob_includes_backup_excludes_quarantine(tmp_path: Path) -> None:
    model = tmp_path / "models" / "id1"
    model.mkdir(parents=True)
    (model / "historic_predictions.pkl").touch()
    (model / "historic_predictions.backup.pkl").touch()
    (model / "historic_predictions.pkl.original-20260812T000000000000Z").touch()
    names = {p.name for p in hp.find_targets(tmp_path, None, None)}
    assert names == {"historic_predictions.pkl", "historic_predictions.backup.pkl"}
    assert {p.name for p in hp.find_targets(tmp_path, "id1", None)} == names


def test_find_targets_path_mode_single_file(tmp_path: Path) -> None:
    target = tmp_path / "historic_predictions.pkl"
    target.touch()
    assert hp.find_targets(tmp_path, None, str(target)) == [target]


def test_quarantine_original_unique_names(tmp_path: Path) -> None:
    target = tmp_path / "historic_predictions.pkl"
    target.write_bytes(b"payload")
    now = datetime(2026, 8, 12, tzinfo=timezone.utc)
    first = hp._quarantine_original(target, now)
    second = hp._quarantine_original(target, now)  # same stamp -> -1 suffix
    assert first.exists() and second.exists() and first != second
    assert first.read_bytes() == b"payload"


def test_print_report_values() -> None:
    report = [{"pair": "SUI/USD:USD", "rows_before": 3, "removed": 1, "rows_after": 2}]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        hp._print_report(Path("/x/historic_predictions.pkl"), report, 1, apply=True)
    out = buffer.getvalue()
    assert "mode=apply" in out
    assert "SUI/USD:USD" in out
    assert "# total duplicate rows removed: 1" in out


def test_main_dry_run_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "historic_predictions.pkl"
    with path.open("wb") as handle:
        pickle.dump(_dup_store(), handle)
    before = path.read_bytes()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = hp.main(["--path", str(path)])
    assert code == 0
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.original-*"))
    assert "files changed: 0" in buffer.getvalue()


def test_main_apply_dedups_and_quarantines(tmp_path: Path) -> None:
    path = tmp_path / "historic_predictions.pkl"
    with path.open("wb") as handle:
        pickle.dump(_dup_store(), handle)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = hp.main(["--path", str(path), "--apply"])
    assert code == 0
    quarantined = list(tmp_path.glob("historic_predictions.pkl.original-*"))
    assert len(quarantined) == 1
    for frame in hp.load_store(path).values():
        assert not frame["date_pred"].duplicated().any()
    assert "files changed: 1" in buffer.getvalue()
    replay = io.StringIO()
    with contextlib.redirect_stdout(replay):
        hp.main(["--path", str(path), "--apply"])
    assert "files changed: 0" in replay.getvalue()  # idempotent


def test_main_skips_unreadable_target(tmp_path: Path) -> None:
    models = tmp_path / "models"
    bad = models / "id1"
    bad.mkdir(parents=True)
    (bad / "historic_predictions.pkl").write_bytes(b"not a pickle")
    good = models / "id2"
    good.mkdir(parents=True)
    with (good / "historic_predictions.pkl").open("wb") as handle:
        pickle.dump(_dup_store(), handle)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = hp.main(["--user-data", str(tmp_path), "--apply"])
    assert code == 1
    assert "files skipped: 1" in out.getvalue()
    assert "files changed: 1" in out.getvalue()
    assert "skipped unreadable store" in err.getvalue()
    for frame in hp.load_store(good / "historic_predictions.pkl").values():
        assert not frame["date_pred"].duplicated().any()


def _run_all() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    import tempfile

    for test in tests:
        try:
            if "tmp_path" in inspect.signature(test).parameters:
                with tempfile.TemporaryDirectory() as directory:
                    test(Path(directory))
            else:
                test()
            print(f"PASS {test.__name__}")
        except Exception as error:  # noqa: BLE001 - self-test reporter
            failures += 1
            print(f"FAIL {test.__name__}: {error!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
