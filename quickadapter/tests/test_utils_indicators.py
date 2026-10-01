"""Indicator kernel contracts: extrema counting, log returns, smoothing and the callback hash."""

import functools
import hashlib
import itertools
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from qa_support import QaTestCase, ohlcv_frame
from Utils import (
    MA_MODES,
    PRICE_MODES,
    _fractal_dimension,
    alligator,
    bottom_log_return,
    calculate_min_extrema,
    calculate_n_extrema,
    calculate_quantile,
    calculate_zero_lag,
    ewo,
    find_fractals,
    frama,
    get_callable_sha256,
    get_distance,
    get_ma_fn,
    get_price_fn,
    get_zl_ma_fn,
    midpoint,
    price_retracement_percent,
    smma,
    top_log_return,
    zlema,
)

# The tolerance tiers below are the ones section 5.6 of the design fixes for these
# quantities: midpoint and get_distance are rtol=1e-9/atol=0.0, calculate_quantile is the
# scalars tier, and ewo(normalize=True) is percent-scaled so it takes rtol=1e-9/atol=1e-12.
_QUANTILE_PLACES = 12


def sawtooth(peaks: int) -> pd.Series:
    """Return a 0/1/0 repeating series with `peaks` interior peaks, each above a zero trough."""
    return pd.Series([float(value) for value in [0.0, 1.0, 0.0] * peaks])


class UtilsIndicatorsTest(QaTestCase):
    def test_the_callback_hash_is_the_digest_of_the_bytecode(self):
        def alpha(value):
            return value

        digest = get_callable_sha256(alpha)
        self.assertEqual(digest, hashlib.sha256(alpha.__code__.co_code).hexdigest())
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, digest.lower())
        int(digest, 16)

        # Determinism is asserted against a DISTINCT closure, not against a second
        # evaluation of the same call: the hash is a pure function of the bytecode, so
        # comparing it to itself would pass even for an implementation returning a
        # constant digest for every callable.
        def beta(value):
            return value * 2

        self.assertEqual(get_callable_sha256(beta), get_callable_sha256(beta))
        self.assertNotEqual(get_callable_sha256(beta), digest)

    def test_different_bytecode_hashes_differently(self):
        # Same-shaped but structurally different bodies: arity and the constant/opcode
        # sequence both differ, so co_code differs. Two identical bodies would collide,
        # which is what test_strategy_sizing pins from the throttle side.
        def unary(value):
            return value

        def binary(left, right):
            return left + right

        def increment(value):
            return value + 1

        digests = {
            get_callable_sha256(unary),
            get_callable_sha256(binary),
            get_callable_sha256(increment),
        }
        self.assertEqual(len(digests), 3)

    def test_a_bound_method_hashes_as_its_unbound_function(self):
        class Holder:
            def method(self):
                return 1

        holder = Holder()
        self.assertEqual(get_callable_sha256(holder.method), get_callable_sha256(Holder.method))
        self.assertEqual(
            get_callable_sha256(functools.partial(holder.method)),
            get_callable_sha256(Holder.method),
        )

    def test_a_partial_hashes_as_the_callable_it_wraps(self):
        def gamma(value):
            return value

        self.assertEqual(
            get_callable_sha256(functools.partial(gamma, 1)), get_callable_sha256(gamma)
        )

    def test_a_callable_object_hashes_as_its_dunder_call(self):
        class Invoker:
            def __call__(self):
                return 1

        self.assertEqual(get_callable_sha256(Invoker()), get_callable_sha256(Invoker.__call__))
        self.assertEqual(
            get_callable_sha256(functools.partial(Invoker())),
            get_callable_sha256(Invoker.__call__),
        )

    def test_a_non_callable_is_refused(self):
        for value in (None, 5, "not callable", [1], 1.5):
            with (
                self.subTest(value=repr(value)),
                self.assertRaisesRegex(ValueError, "must be callable"),
            ):
                get_callable_sha256(value)

    def test_a_callable_without_a_code_object_is_refused(self):
        # throttle_callback (QuickAdapterV3.py:1397) keys every callback on this digest, so a
        # C-level callable must be rejected rather than silently keyed on an empty string.
        for builtin in (len, print, np.mean):
            with (
                self.subTest(builtin=builtin.__name__),
                self.assertRaisesRegex(ValueError, "unable to retrieve code object"),
            ):
                get_callable_sha256(builtin)

    def test_the_minimum_extrema_count_is_the_rounded_window_count_scaled(self):
        # int(round(length / fit_live_predictions_candles)) * min_extrema, with the default
        # min_extrema of 2. Values read from the implementation.
        for length, fit_live, expected in (
            (100, 5, 40),
            (7, 3, 4),
            (1000, 2, 1000),
            (5, 5, 2),
            (13, 7, 4),
            (100, 3, 66),
            (1, 1, 2),
            (0, 5, 0),
            (9, 4, 4),
        ):
            with self.subTest(length=length, fit_live=fit_live):
                self.assertEqual(calculate_min_extrema(length, fit_live), expected)

    def test_the_minimum_extrema_count_rounds_half_to_even(self):
        # Python's round is banker's rounding, not half-up: 5/2 = 2.5 rounds down to 2 and
        # 7/2 = 3.5 rounds up to 4. Pinned because half-up is the common assumption.
        self.assertEqual(calculate_min_extrema(5, 2), 4)
        self.assertEqual(calculate_min_extrema(7, 2), 8)

    def test_the_minimum_extrema_count_is_a_multiple_of_the_extrema_multiplier(self):
        for length in range(0, 40):
            for fit_live in range(1, 9):
                for min_extrema in (1, 2, 3, 5):
                    with self.subTest(length=length, fit_live=fit_live, min_extrema=min_extrema):
                        count = calculate_min_extrema(length, fit_live, min_extrema)
                        self.assertEqual(count % min_extrema, 0)
                        self.assertGreaterEqual(count, 0)

    def test_the_extrema_count_adds_peaks_and_troughs(self):
        # sawtooth(5) is 0,1,0,0,1,0,...: scipy finds interior peaks at 1,4,7,10,13 and
        # interior troughs at 2,5,8,11, so 5 + 4 = 9 only if both polarities are counted.
        series = sawtooth(5)
        self.assertEqual(calculate_n_extrema(series), 9)
        self.assertEqual(calculate_n_extrema(series), 2 * 5 - 1)

    def test_the_extrema_count_separates_peaks_from_troughs(self):
        # A peaks-only shape and a troughs-only shape each contribute, so the count is a sum
        # of two independent detections rather than a doubled count of one of them.
        for peaks in range(1, 6):
            series = sawtooth(peaks)
            with self.subTest(peaks=peaks):
                expected_peaks = peaks
                expected_troughs = peaks - 1
                self.assertEqual(calculate_n_extrema(series), expected_peaks + expected_troughs)

    def test_the_extrema_count_ignores_series_without_an_interior_turn(self):
        for series in (
            pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
            pd.Series([5.0, 4.0, 3.0, 2.0, 1.0]),
            pd.Series([1.0] * 5),
            pd.Series([1.0]),
            pd.Series([], dtype=float),
        ):
            with self.subTest(series=series.to_numpy().tolist()):
                self.assertEqual(calculate_n_extrema(series), 0)

    def test_the_log_returns_are_the_closed_form_on_a_positive_series(self):
        frame = ohlcv_frame([100.0, 102, 101, 103, 105, 104, 106, 108])
        close = frame.get("close")
        for period in (1, 2, 3, 5):
            with self.subTest(period=period):
                reference_top = close.rolling(period, min_periods=period).max().shift(1)
                reference_bottom = close.rolling(period, min_periods=period).min().shift(1)
                expected_top = np.log(close / reference_top)
                expected_bottom = np.log(close / reference_bottom)
                assert_allclose(
                    top_log_return(frame, period).to_numpy(),
                    expected_top.to_numpy(),
                    rtol=1e-12,
                    atol=0.0,
                    equal_nan=True,
                )
                assert_allclose(
                    bottom_log_return(frame, period).to_numpy(),
                    expected_bottom.to_numpy(),
                    rtol=1e-12,
                    atol=0.0,
                    equal_nan=True,
                )

    def test_a_non_positive_price_yields_a_missing_log_return_not_an_infinity(self):
        # safe_log_ratio replaces a non-positive operand with the NaN fallback, so a zero or
        # negative price leaves a hole in the series rather than emitting -inf. assertAlmostEqual
        # cannot express this; the positions are checked with np.isnan and the no-infinity
        # invariant separately.
        frame = ohlcv_frame([100.0, 0.0, 102, 0.0, 103, 0.0, 105, 106, 107, 108, 109, 110])
        for indicator in (top_log_return, bottom_log_return):
            with self.subTest(indicator=indicator.__name__):
                result = indicator(frame, 3).to_numpy()
                self.assertFalse(np.isinf(result).any())
                self.assertTrue(np.isnan(result).any())
                close = frame.get("close").to_numpy()
                self.assertTrue(np.isnan(result[close <= 0.0]).all())

    def test_a_fully_non_positive_price_leaves_the_whole_series_missing(self):
        for values in ([0.0] * 12, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]):
            frame = ohlcv_frame(values)
            with self.subTest(values=values):
                for indicator in (top_log_return, bottom_log_return, price_retracement_percent):
                    result = indicator(frame, 3).to_numpy()
                    self.assertTrue(np.isnan(result).all())
                    self.assertFalse(np.isinf(result).any())

    def test_a_period_below_one_is_refused_by_the_log_return_indicators(self):
        frame = ohlcv_frame([100.0, 101, 102, 103])
        for indicator in (top_log_return, bottom_log_return, price_retracement_percent):
            for period in (0, -1):
                with (
                    self.subTest(indicator=indicator.__name__, period=period),
                    self.assertRaises(ValueError),
                ):
                    indicator(frame, period)

    def test_the_retracement_is_the_log_position_within_the_rolling_range(self):
        closes = [100.0, 102, 101, 103, 105, 104, 106, 108]
        frame = ohlcv_frame(closes)
        close = frame.get("close")
        low = close.rolling(3, min_periods=3).min().shift(1)
        high = close.rolling(3, min_periods=3).max().shift(1)
        expected = np.log(close / low) / np.log(high / low)
        assert_allclose(
            price_retracement_percent(frame, 3).to_numpy(),
            expected.to_numpy(),
            rtol=1e-12,
            atol=0.0,
            equal_nan=True,
        )

    def test_a_flat_range_retraces_to_the_bottom_rather_than_dividing_by_zero(self):
        # A constant close makes the log(high/low) denominator exactly 0.0; the result is
        # pinned to 0.0 instead of a division, and stays on the series index.
        frame = ohlcv_frame([100.0] * 8)
        result = price_retracement_percent(frame, 3)
        self.assertTrue(result.index.equals(frame.index))
        assert_allclose(
            result.to_numpy()[3:],
            np.zeros(5),
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(np.isinf(result.to_numpy()).any())

    def test_a_near_zero_range_retraces_to_the_bottom(self):
        # np.isclose on the denominator admits any range smaller than 1e-8 in log space,
        # so a ramp whose total range is 1e-7 still reads as flat.
        for span in (0.0, 1e-9, 1e-7):
            frame = ohlcv_frame([100.0 + span * step for step in range(8)])
            with self.subTest(span=span):
                result = price_retracement_percent(frame, 3).to_numpy()[3:]
                assert_allclose(result, np.zeros(5), rtol=0.0, atol=0.0)

    def test_an_empty_sample_has_no_quantile(self):
        # Documented empty-input return; assertAlmostEqual(nan, nan) would fail, so the
        # branch is asserted through np.isnan.
        self.assertTrue(np.isnan(calculate_quantile(np.array([], dtype=float), 1.0)))
        self.assertTrue(np.isnan(calculate_quantile(np.array([np.nan, np.nan]), 1.0)))

    def test_a_quantile_is_the_share_of_the_sample_at_or_below_the_value(self):
        sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for value, expected in (
            (0.0, 0.0),
            (1.0, 0.1),
            (2.0, 0.3),
            (3.0, 0.5),
            (5.0, 0.9),
            (6.0, 1.0),
        ):
            with self.subTest(value=value):
                self.assertAlmostEqual(
                    calculate_quantile(sample, value), expected, places=_QUANTILE_PLACES
                )

    def test_a_quantile_rises_with_the_value_and_stays_within_the_unit_interval(self):
        sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 5.0, 1.0, 4.0])
        scores = [float(calculate_quantile(sample, value)) for value in np.linspace(-1.0, 6.0, 25)]
        for lower, upper in itertools.pairwise(scores):
            self.assertLessEqual(lower, upper)
        for score in scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_a_nan_in_the_sample_is_omitted_from_the_quantile(self):
        self.assertAlmostEqual(
            float(calculate_quantile(np.array([1.0, np.nan, 3.0]), 2.0)),
            0.5,
            places=_QUANTILE_PLACES,
        )

    def test_a_multidimensional_sample_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must be 1-dimensional"):
            calculate_quantile(np.array([[1.0, 2.0], [3.0, 4.0]]), 3.0)

    def test_the_fractal_lists_name_the_peak_and_trough_positions(self):
        highs = [1.0, 2, 3, 2, 1, 2, 3, 2, 1]
        frame = pd.DataFrame({"high": highs, "low": [value - 1.0 for value in highs]})
        fractal_highs, fractal_lows = find_fractals(frame, 2)
        self.assertEqual(fractal_highs, [2, 6])
        self.assertEqual(fractal_lows, [4])
        for position in fractal_highs + fractal_lows:
            self.assertIn(position, frame.index.tolist())
        self.assertFalse(set(fractal_highs) & set(fractal_lows))

    def test_the_fractal_masks_span_the_whole_frame(self):
        highs = [1.0, 2, 3, 2, 1, 2, 3, 2, 1]
        frame = pd.DataFrame({"high": highs, "low": [value - 1.0 for value in highs]})
        fractal_highs, fractal_lows = find_fractals(frame, 2)
        high_mask = frame.index.isin(fractal_highs)
        low_mask = frame.index.isin(fractal_lows)
        assert_array_equal(
            high_mask, np.array([False, False, True, False, False, False, True, False, False])
        )
        assert_array_equal(
            low_mask, np.array([False, False, False, False, True, False, False, False, False])
        )
        self.assertEqual(high_mask.dtype, np.bool_)
        self.assertEqual(high_mask.shape, (len(frame),))

    def test_a_monotone_series_carries_no_fractal(self):
        frame = pd.DataFrame(
            {"high": [1.0, 2, 3, 4, 5, 6, 7, 8, 9], "low": [0.0, 1, 2, 3, 4, 5, 6, 7, 8]},
        )
        fractal_highs, fractal_lows = find_fractals(frame, 2)
        self.assertEqual(fractal_highs, [])
        self.assertEqual(fractal_lows, [])
        self.assertFalse(frame.index.isin(fractal_highs + fractal_lows).any())

    def test_a_series_shorter_than_two_periods_plus_one_carries_no_fractal(self):
        highs = [1.0, 2, 3, 2, 1]
        frame = pd.DataFrame({"high": highs, "low": [value - 1.0 for value in highs]})
        for length in range(0, 5):
            with self.subTest(length=length):
                self.assertEqual(find_fractals(frame.iloc[:length], 2), ([], []))

    def test_a_fractal_is_only_looked_for_inside_the_frame(self):
        # The detector skips the first and last `period` bars, so widening the period can only
        # remove candidates, never add them.
        highs = [1.0, 2, 3, 2, 1, 2, 3, 2, 1]
        frame = pd.DataFrame({"high": highs, "low": [value - 1.0 for value in highs]})
        counts = [
            sum(len(part) for part in find_fractals(frame, period)) for period in (1, 2, 3, 4)
        ]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[-1], 0)

    def test_a_tied_neighbour_is_not_a_fractal(self):
        # The comparison against each neighbour is strict, so a plateau registers no fractal
        # at any bar of it. In 1,2,3,3,2,1,2,3,2,1 the peak at index 2 ties with its right
        # neighbour at index 3, so neither is a fractal and only index 7 qualifies.
        highs = [1.0, 2, 3, 3, 2, 1, 2, 3, 2, 1]
        frame = pd.DataFrame({"high": highs, "low": [value - 1.0 for value in highs]})
        fractal_highs, _ = find_fractals(frame, 2)
        self.assertEqual(fractal_highs, [7])
        for plateau in (2, 3):
            with self.subTest(plateau=plateau):
                self.assertNotIn(plateau, fractal_highs)

    def test_each_moving_average_name_resolves_to_its_own_callable(self):
        callables = [get_ma_fn(name) for name in MA_MODES]
        for name, function in zip(MA_MODES, callables, strict=True):
            with self.subTest(name=name):
                self.assertTrue(callable(function))
                self.assertIs(get_ma_fn(name), function)
        self.assertEqual(len(set(callables)), len(MA_MODES))
        self.assertEqual(len({get_zl_ma_fn(name) for name in MA_MODES}), len(MA_MODES))

    def test_an_unknown_moving_average_name_falls_back_to_the_simple_average(self):
        # The registry is a .get with the SMA default, so an unrecognised name is not an
        # error: it silently resolves to the first entry. Pinned because a raise here would
        # break every caller that forwards a user-supplied mamode.
        sma = get_ma_fn(MA_MODES[0])
        for name in ("bogus", "", "SMA", "sma ", None, 0):
            with self.subTest(name=repr(name)):
                self.assertIs(get_ma_fn(name), sma)

    def test_each_price_name_resolves_to_its_own_callable(self):
        callables = [get_price_fn(name) for name in PRICE_MODES]
        for name, function in zip(PRICE_MODES, callables, strict=True):
            with self.subTest(name=name):
                self.assertTrue(callable(function))
                self.assertIs(get_price_fn(name), function)
        self.assertEqual(len(set(callables)), len(PRICE_MODES))

    def test_an_unknown_price_name_falls_back_to_the_close(self):
        frame = ohlcv_frame([10.0, 11.0, 12.0])
        close = frame.get("close")
        for name in ("bogus", "", "CLOSE", None):
            with self.subTest(name=repr(name)):
                pd.testing.assert_series_equal(get_price_fn(name)(frame), close)

    def test_a_constant_series_smooths_to_the_constant(self):
        # zlema leads by the half-period lag it removes, smma and frama by period - 1 bars;
        # past that warmup every bar of a constant series must read back as that constant.
        series = pd.Series([7.0] * 20)
        for period in (2, 4, 6, 10):
            for indicator, warmup in ((zlema, (period - 1) // 2), (smma, period - 1)):
                result = indicator(series, period)
                self.assertEqual(int(result.isna().sum()), warmup)
                assert_allclose(
                    result.to_numpy()[warmup:], np.full(20 - warmup, 7.0), rtol=0.0, atol=0.0
                )
            smoothed = frama(ohlcv_frame([7.0] * 20), period)
            self.assertEqual(int(smoothed.isna().sum()), period - 1)
            assert_allclose(
                smoothed.to_numpy()[period - 1 :],
                np.full(21 - period, 7.0),
                rtol=0.0,
                atol=0.0,
            )

    def test_a_missing_input_value_never_produces_an_infinity(self):
        # The gap is absorbed: ewm and the smma seed skip missing bars, so from `period` on
        # every output is a finite number. Nothing here may become an infinity.
        series = pd.Series([1.0, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
        for indicator in (zlema, smma):
            for period in (1, 2, 3, 5, 6):
                with self.subTest(indicator=indicator.__name__, period=period):
                    result = indicator(series, period).to_numpy()
                    self.assertFalse(np.isinf(result).any())
                    self.assertTrue(np.isfinite(result[period:]).all())

    def test_frama_degrades_to_missing_after_a_gap_in_its_input(self):
        # _fractal_dimension returns nan once its window contains a nan high, and the
        # recursion skips on nan, so the tail stays missing rather than diverging: the
        # output after the gap is all nan and never an infinity.
        frame = ohlcv_frame([7.0] * 20)
        frame.loc[8, "high"] = np.nan
        result = frama(frame, 6).to_numpy()
        self.assertFalse(np.isinf(result).any())
        self.assertTrue(np.isnan(result[:5]).all())
        assert_allclose(result[5:9], np.full(4, 7.0), rtol=0.0, atol=0.0)
        self.assertTrue(np.isnan(result[9:]).all())

    def test_frama_survives_an_intact_series_across_the_same_window(self):
        # The counterpart to the gap case: without the nan the recursion runs to the end, so
        # the missing tail above is caused by the gap and not by the period.
        result = frama(ohlcv_frame([7.0] * 20), 6).to_numpy()
        self.assertTrue(np.isnan(result[:5]).all())
        assert_allclose(result[5:], np.full(15, 7.0), rtol=0.0, atol=0.0)

    def test_frama_reproduces_close_because_the_dimension_is_one_up_to_rounding(self):
        # The real contract, asserted rather than implied. HL1 and HL2 each span a HALF of the
        # window while HL3 spans all of it, so HL1 + HL2 <= 2 * HL3, D <= 1, and the clip
        # returns exactly 1.0 on every path. alpha is therefore exp(-4.6 * 0) == 1.0 and
        # frama reproduces close after its seed. This pins the DEGENERACY as a fact: a change
        # that made the dimension vary, or the -4.6 scale observable, would have to update
        # this test rather than pass unnoticed.
        closes = np.array([100.0 + i for i in range(30)])
        result = frama(ohlcv_frame(closes.tolist()), 6).to_numpy()
        self.assertTrue(np.isnan(result[:5]).all())
        assert_allclose(result[5], closes[:6].mean(), rtol=0.0, atol=1e-9)
        assert_allclose(result[6:], closes[6:], rtol=0.0, atol=1e-9)

    def test_the_fractal_dimension_is_one_within_rounding_on_every_window(self):
        # Asserted directly on the helper. The name says "within rounding" on purpose: the
        # clip's lower bound is not a floor in practice, because `np.clip` is a no-op for
        # values already at or above it, and the ratio lands on 1.0 + O(1e-16) for some
        # windows. Asserting `== 1.0` here would be pinning a rounding accident of these
        # three fixtures rather than the contract.
        trending_high = 100.0 + np.arange(16) * 0.5
        trending_low = trending_high - 0.8
        noisy = np.array([100.0 + i for i in range(16)])
        plateaus = np.array(
            [1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 9.0, 9.0, 9.0, 3.0, 3.0, 3.0, 4.0]
        )
        for highs, lows in (
            (trending_high, trending_low),
            (noisy, noisy - 0.3),
            (plateaus, plateaus - 1.0),
        ):
            with self.subTest(highs=highs[:3].tolist()):
                self.assertAlmostEqual(_fractal_dimension(highs, lows, 16), 1.0, places=12)

    def test_the_fractal_dimension_can_exceed_one_by_rounding(self):
        # The counterexample that makes the case above honest. This window is not synthetic:
        # the ratio computes to 1.0000000000000002, the clip leaves it untouched because it is
        # already above the lower bound, and alpha becomes 0.999999999999999. Anything that
        # branches on `alpha == 1.0` or promises bit-exact `close` is wrong on this input.
        highs = np.array([100.2, 101.2, 102.7, 101.6, 102.8, 102.5, 102.5, 102.8, 100.9, 100.3])
        lows = np.array([99.9, 100.9, 101.6, 101.1, 102.4, 102.3, 101.6, 101.7, 100.6, 99.9])
        dimension = _fractal_dimension(highs, lows, 10)
        self.assertGreater(dimension, 1.0)
        self.assertLessEqual(dimension, 1.0 + 1e-15)
        self.assertLess(np.exp(-4.6 * (dimension - 1.0)), 1.0)

    def test_the_zero_lag_frama_de_lags_before_the_fractal_dimension(self):
        # zero_lag replaces high, low and close with calculate_zero_lag first, which makes the
        # seed the mean of the de-lagged closes rather than the raw ones. Ignoring the flag
        # leaves a different seed, so the first finite bar alone distinguishes the two.
        frame = ohlcv_frame([float(value) for value in range(1, 13)])
        de_lagged = frama(frame, 4, zero_lag=True).to_numpy()
        raw = frama(frame, 4, zero_lag=False).to_numpy()
        self.assertFalse(np.array_equal(de_lagged, raw, equal_nan=True))
        closes = frame.get("close")
        self.assertAlmostEqual(
            float(de_lagged[3]),
            float(calculate_zero_lag(closes, 4).iloc[:4].mean()),
            places=9,
        )
        self.assertAlmostEqual(float(raw[3]), float(closes.iloc[:4].mean()), places=9)

    def test_a_framed_gap_stops_the_fractal_dimension_at_the_window_edge(self):
        # _fractal_dimension returns 1.0 when either half-range is zero, which pins alpha at
        # exp(0) = 1 and makes the recursion follow close exactly. A constant frame is the
        # only shape that reaches that branch.
        result = frama(ohlcv_frame([7.0] * 12), 4).to_numpy()
        self.assertFalse(np.isinf(result).any())
        assert_allclose(result[3:], np.full(9, 7.0), rtol=0.0, atol=0.0)

    def test_the_smma_follows_its_own_recursion(self):
        # Seeded with the mean of the first `period` bars, then alpha = 1/period per bar.
        # Pinning the recursion catches a wrong alpha, which a constant or a NaN series
        # cannot distinguish because both collapse to the same constant.
        series = pd.Series([1.0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13])
        for period in (2, 3, 4, 5):
            with self.subTest(period=period):
                values = series.to_numpy()
                expected = np.full(len(values), np.nan)
                seed = float(np.mean(values[:period]))
                expected[period - 1] = seed
                previous = seed
                for index in range(period, len(values)):
                    previous += (values[index] - previous) / period
                    expected[index] = previous
                assert_allclose(
                    smma(series, period).to_numpy(), expected, rtol=1e-12, atol=0.0, equal_nan=True
                )

    def test_the_zero_lag_ema_follows_its_own_recursion(self):
        # alpha = 2/(period+1) applied to 2*series - series.shift(lag) with adjust=False, so
        # the first defined bar seeds the recursion rather than the whole history weighting it.
        series = pd.Series([1.0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13])
        values = series.to_numpy()
        for period in (2, 3, 5, 7):
            with self.subTest(period=period):
                lag = (period - 1) // 2
                alpha = 2 / (period + 1)
                expected = np.full(len(values), np.nan)
                previous = None
                for index in range(len(values)):
                    if index < lag:
                        continue
                    de_lagged = 2 * values[index] - values[index - lag]
                    previous = (
                        de_lagged
                        if previous is None
                        else alpha * de_lagged + (1 - alpha) * previous
                    )
                    expected[index] = previous
                assert_allclose(
                    zlema(series, period).to_numpy(), expected, rtol=1e-12, atol=0.0, equal_nan=True
                )

    def test_the_smma_never_looks_ahead(self):
        # The purge depends on causality: changing a later bar must leave every earlier
        # output bit-identical, NaN warmup included.
        series = pd.Series([1.0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13])
        for period in (2, 3, 4, 6):
            perturbed = series.copy()
            perturbed.iloc[9] = 1000.0
            for indicator in (smma, zlema):
                with self.subTest(indicator=indicator.__name__, period=period):
                    before = indicator(series, period).to_numpy()
                    after = indicator(perturbed, period).to_numpy()
                    assert_array_equal(before[:9], after[:9])
                    self.assertEqual(before.shape, after.shape)
                    self.assertNotEqual(before[9], after[9])

    def test_the_smma_keeps_the_index_and_is_empty_when_the_series_is_shorter(self):
        series = pd.Series([7.0] * 20)
        result = smma(series, 25)
        self.assertEqual(len(result), 20)
        self.assertTrue(result.index.equals(series.index))
        self.assertTrue(result.isna().all())

    def test_the_smma_shifts_by_its_offset(self):
        series = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8])
        shifted = smma(series, 3, offset=2)
        assert_array_equal(shifted.to_numpy()[2:], smma(series, 3).to_numpy()[:-2])

    def test_a_non_positive_smma_period_is_refused(self):
        series = pd.Series([1.0, 2, 3])
        for period in (0, -3):
            with self.subTest(period=period), self.assertRaisesRegex(ValueError, "must be > 0"):
                smma(series, period)

    def test_an_odd_fractal_dimension_period_is_refused(self):
        frame = ohlcv_frame([1.0, 2, 3, 4, 5, 6])
        for period in (3, 5, 7):
            with self.subTest(period=period), self.assertRaisesRegex(ValueError, "must be even"):
                frama(frame, period)

    def test_a_fractal_dimension_period_below_two_is_refused(self):
        # The `period < 2` clause is a separate contract from the evenness clause, and the
        # existing case only drives odd periods, which the evenness clause already catches.
        # Zero is even. Without its own case the clause can be dropped and the caller gets
        # numpy's internal "zero-size array to reduction operation maximum which has no
        # identity" instead of the documented domain error.
        frame = ohlcv_frame([1.0, 2, 3, 4, 5, 6])
        for period in (0, -4):
            with (
                self.subTest(period=period),
                self.assertRaisesRegex(ValueError, "must be an even integer >= 2"),
            ):
                frama(frame, period)

    def test_the_alligator_lines_are_the_shifted_smoothed_averages_of_the_median_price(self):
        # The multipliers are asymmetric so that median, close, average, typical and
        # weighted-close are five different series: a symmetric frame collapses them all
        # onto the close and cannot tell the pricemode argument from being ignored.
        close = pd.Series(np.linspace(10.0, 40.0, 30))
        frame = pd.DataFrame(
            {"open": close, "high": close * 1.5, "low": close * 0.8, "close": close},
        )
        for pricemode in PRICE_MODES:
            prices = get_price_fn(pricemode)(frame)
            with self.subTest(pricemode=pricemode):
                jaw, teeth, lips = alligator(frame, 6, 4, 3, 2, 1, 1, pricemode=pricemode)
                for line, period, offset in ((jaw, 6, 2), (teeth, 4, 1), (lips, 3, 1)):
                    assert_array_equal(
                        line.to_numpy(), smma(prices, period, offset=offset).to_numpy()
                    )

    def test_the_alligator_lines_are_de_lagged_before_they_are_smoothed(self):
        # zero_lag reaches smma, so each line smooths calculate_zero_lag(prices, period) with
        # its own period rather than the shared one.
        close = pd.Series(np.linspace(10.0, 40.0, 30))
        frame = pd.DataFrame(
            {"open": close, "high": close * 1.5, "low": close * 0.8, "close": close},
        )
        median = (frame.get("high") + frame.get("low")) / 2
        causal = alligator(frame, 6, 4, 3, 2, 1, 1, pricemode="median")
        de_lagged = alligator(frame, 6, 4, 3, 2, 1, 1, pricemode="median", zero_lag=True)
        for line, causal_line, period, offset in zip(
            de_lagged, causal, (6, 4, 3), (2, 1, 1), strict=True
        ):
            with self.subTest(period=period):
                assert_array_equal(
                    line.to_numpy(),
                    smma(calculate_zero_lag(median, period), period, offset=offset).to_numpy(),
                )
                self.assertFalse(
                    np.array_equal(line.to_numpy(), causal_line.to_numpy(), equal_nan=True)
                )

    def test_the_oscillator_is_the_gap_between_its_two_averages(self):
        frame = ohlcv_frame([100.0, 101, 102, 101.5, 103, 104, 103.5, 105, 106, 105.5, 107, 108])
        close = frame.get("close")
        for mamode in MA_MODES:
            for ma1_length, ma2_length in ((3, 6), (2, 8)):
                with self.subTest(mamode=mamode, ma1=ma1_length, ma2=ma2_length):
                    average = get_ma_fn(mamode)
                    expected = average(close, ma1_length) - average(close, ma2_length)
                    assert_allclose(
                        ewo(frame, ma1_length, ma2_length, mamode=mamode),
                        expected,
                        rtol=1e-12,
                        atol=0.0,
                        equal_nan=True,
                    )

    def test_the_oscillator_waits_for_its_slowest_average(self):
        # The longer period dominates the warmup: the gap is missing for exactly as many bars
        # as the slow average needs, so a shallow crossover cannot appear before the data
        # supports it.
        frame = ohlcv_frame([100.0, 101, 102, 101.5, 103, 104, 103.5, 105, 106, 105.5, 107, 108])
        close = frame.get("close")
        for ma1_length, ma2_length in ((2, 6), (3, 8), (4, 10)):
            with self.subTest(ma1=ma1_length, ma2=ma2_length):
                raw = ewo(frame, ma1_length, ma2_length)
                slow = get_ma_fn(MA_MODES[0])(close, ma2_length)
                assert_array_equal(np.isnan(raw), np.isnan(slow))
                self.assertFalse(np.isinf(raw).any())

    def test_a_normalised_oscillator_is_the_raw_gap_as_a_percent_of_price(self):
        frame = ohlcv_frame([100.0, 101, 102, 101.5, 103, 104, 103.5, 105, 106, 105.5, 107, 108])
        raw = ewo(frame, 3, 6)
        close = frame.get("close").to_numpy()
        assert_allclose(
            ewo(frame, 3, 6, normalize=True),
            raw / close * 100.0,
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        )

    def test_the_zero_lag_oscillator_is_not_the_causal_one(self):
        frame = ohlcv_frame([100.0, 101, 102, 101.5, 103, 104, 103.5, 105, 106, 105.5, 107, 108])
        causal = ewo(frame, 3, 6)
        differs = False
        for mamode in MA_MODES:
            with self.subTest(mamode=mamode):
                zero_lagged = ewo(frame, 3, 6, zero_lag=True, mamode=mamode)
                self.assertEqual(zero_lagged.shape, causal.shape)
                self.assertFalse(np.isinf(zero_lagged).any())
                differs |= not np.array_equal(zero_lagged, causal, equal_nan=True)
        self.assertTrue(differs)

    def test_the_zero_lag_oscillator_averages_the_de_lagged_series(self):
        # ma_fn becomes get_zl_ma_fn(mamode), i.e. the registry average applied to the
        # zero-lagged input. The expectation is built from get_ma_fn and calculate_zero_lag
        # separately, so a get_zl_ma_fn that forwarded the raw series would not reproduce it.
        # The ema name is excluded because zero_lag substitutes zlema for it outright.
        frame = ohlcv_frame([100.0, 101, 102, 101.5, 103, 104, 103.5, 105, 106, 105.5, 107, 108])
        close = frame.get("close")
        registry_modes = tuple(name for name in MA_MODES if name != MA_MODES[1])
        for mamode in registry_modes:
            for ma1_length, ma2_length in ((3, 6), (4, 9)):
                with self.subTest(mamode=mamode, ma1=ma1_length, ma2=ma2_length):
                    average = get_ma_fn(mamode)
                    fast = average(calculate_zero_lag(close, ma1_length), timeperiod=ma1_length)
                    slow = average(calculate_zero_lag(close, ma2_length), timeperiod=ma2_length)
                    assert_allclose(
                        ewo(frame, ma1_length, ma2_length, zero_lag=True, mamode=mamode),
                        fast - slow,
                        rtol=1e-12,
                        atol=0.0,
                        equal_nan=True,
                    )

    def test_the_zero_lag_ema_oscillator_is_the_zero_lag_ema_gap(self):
        # The ema branch of zero_lag substitutes zlema itself rather than going through
        # get_zl_ma_fn, so it leads the shared registry path.
        frame = ohlcv_frame([100.0, 101, 102, 101.5, 103, 104, 103.5, 105, 106, 105.5, 107, 108])
        close = frame.get("close")
        expected = zlema(close, 3).to_numpy() - zlema(close, 6).to_numpy()
        assert_allclose(
            ewo(frame, 3, 6, zero_lag=True, mamode=MA_MODES[1]),
            expected,
            rtol=1e-12,
            atol=0.0,
            equal_nan=True,
        )

    def test_the_oscillator_of_a_constant_series_collapses_to_zero(self):
        frame = ohlcv_frame([7.0] * 20)
        for mamode in MA_MODES:
            with self.subTest(mamode=mamode):
                raw = ewo(frame, 3, 6, mamode=mamode)
                self.assertFalse(np.isinf(raw).any())
                assert_allclose(
                    raw[np.isfinite(raw)],
                    np.zeros(int(np.isfinite(raw).sum())),
                    rtol=0.0,
                    atol=0.0,
                )

    def test_the_distance_is_the_absolute_gap_and_the_midpoint_brackets_it(self):
        self.assertEqual(get_distance(-3.5, 1.25), 4.75)
        self.assertEqual(get_distance(2.5, 2.5), 0.0)
        assert_allclose(midpoint(0.1, 0.3), 0.2, rtol=1e-9, atol=0.0)
        left = pd.Series([1.0, 4.0])
        right = pd.Series([3.0, 2.0])
        assert_allclose(
            get_distance(left, right).to_numpy(), np.array([2.0, 2.0]), rtol=1e-9, atol=0.0
        )
        assert_allclose(midpoint(left, right).to_numpy(), np.array([2.0, 3.0]), rtol=1e-9, atol=0.0)
        # The TypeVar admits pd.Series or float, and Python's true division makes an int pair
        # come back as a float: two of the four production call sites get that float back.
        self.assertIsInstance(midpoint(2, 3), float)
        assert_allclose(midpoint(2, 3), 2.5, rtol=1e-9, atol=0.0)
        self.assertIsInstance(midpoint(left, right), pd.Series)


if __name__ == "__main__":
    unittest.main()
