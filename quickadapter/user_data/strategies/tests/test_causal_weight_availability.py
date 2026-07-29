"""Regression tests for the causal label-weight availability invariants (#131).

`_compute_knn_pivot_sigma_availability` proves leak-free causal availability by
assuming the Zigzag confirmation geometry: successive pivots are at least
`_ZIGZAG_MIN_CONFIRMATION_SLOPES + 1` candles apart, and the earliest possible
successor position (`first_future`) never exceeds the actual next pivot. These
tests lock both against the real `_zigzag`, so a future Zigzag change that
weakens either assumption fails here instead of silently leaking future info.

Requires the freqtrade runtime stack (talib). Run in the container:
    python -m pytest user_data/strategies/tests/ -q
or directly:
    python user_data/strategies/tests/test_causal_weight_availability.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Utils  # noqa: E402  (needs the strategies dir on sys.path)

_PIVOT_SPACING = Utils._ZIGZAG_MIN_CONFIRMATION_SLOPES + 1


def _make_ohlc(rng: np.random.Generator, n: int) -> pd.DataFrame:
    log_return = rng.normal(0.0, 0.01, n).cumsum()
    close = 100.0 * np.exp(log_return)
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.004, n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.004, n)))
    volume = rng.uniform(1e3, 1e4, n)
    return pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume})


def _real_pivot_series(seed: int, series: int = 120):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(series):
        n = int(rng.integers(250, 1000))
        result = Utils._zigzag(
            _make_ohlc(rng, n),
            natr_period=int(rng.integers(10, 20)),
            natr_multiplier=float(rng.uniform(6.0, 12.0)),
        )
        idx = np.asarray(result.indices, dtype=np.int64)
        if idx.size >= 2:
            out.append((idx, result.known_at_positions, n))
    return out


def test_min_pivot_spacing_covers_confirmation_bound() -> None:
    """Real consecutive pivots are never closer than `_PIVOT_SPACING`."""
    worst = min(
        int(np.diff(idx).min()) for idx, _known_at, _n in _real_pivot_series(20260729)
    )
    assert worst >= _PIVOT_SPACING, worst


def test_first_future_bound_is_never_understated() -> None:
    """`first_future` <= actual next pivot, so possible futures are over-counted.

    Ordinary group: `confirmation + 1`. Initial-orientation replay group (shared
    confirmation watermark): `position + _PIVOT_SPACING`. Either bound must not
    exceed the real next pivot, or the availability predicate would understate
    availability and leak future information.
    """
    for idx, known_at_positions, _n in _real_pivot_series(20260729):
        confirmation_0 = known_at_positions[idx[0]]
        for k in range(idx.size - 1):
            confirmation_k = int(known_at_positions[idx[k]])
            first_future = (
                idx[k] + _PIVOT_SPACING
                if confirmation_k == confirmation_0
                else confirmation_k + 1
            )
            assert first_future <= idx[k + 1], (k, first_future, int(idx[k + 1]))


def test_knn_sigma_availability_within_bounds() -> None:
    """Availability stays in `[pivot confirmation, n]` on real pivots."""
    for idx, known_at_positions, n in _real_pivot_series(1, series=30):
        availability = Utils._compute_knn_pivot_sigma_availability(
            idx, known_at_positions, 4, 0.2, 0.5, 2.0, n
        )
        assert availability.shape == idx.shape
        assert np.all(availability >= known_at_positions[idx])
        assert np.all(availability <= n)


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
    print("ALL PASS")
