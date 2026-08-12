#!/usr/bin/env python3
"""Tests for the historic_predictions deduplication tool.

The dedup logic is pure pandas and version-agnostic, so these run under any
pandas 2.x/3.x. Runnable with pytest or directly (``python <this file>``); the
container ships pandas but not necessarily pytest.
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import pandas as pd

_MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "historic_predictions_deduplicate.py"
)
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
    assert removed == 1  # NaT rows collapse (they never match the merge anyway)


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


def _run_all() -> int:
    tests = [
        value for name, value in sorted(globals().items()) if name.startswith("test_")
    ]
    failures = 0
    import tempfile

    for test in tests:
        try:
            if "tmp_path" in test.__code__.co_varnames:
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
