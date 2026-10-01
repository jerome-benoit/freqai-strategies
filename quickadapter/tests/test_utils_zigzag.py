"""Zigzag pivot confirmation, label registration and soft extrema; requires the Freqtrade QA image."""

import logging
import unittest

import numpy as np
import pandas as pd
from qa_support import QaTestCase, ohlcv_frame
from Utils import (
    _LABEL_GENERATORS,
    _ZIGZAG_CONFIRMATION_ALPHA,
    _ZIGZAG_MIN_CONFIRMATION_SLOPES,
    EXTREMA_COLUMN,
    LABEL_COLUMNS,
    LabelData,
    TrendDirection,
    ZigzagResult,
    _adapt_label_generator,
    _zigzag,
    generate_label_data,
    register_label_generator,
    soft_extremum,
    zigzag,
)

LOGGER = logging.getLogger("test-utils-zigzag")

# natr_period 5 keeps the ATR warm-up short enough that a 45-candle first leg still leaves
# room for the five consecutive slopes the confirmation test demands.
NATR_PERIOD = 5
NATR_MULTIPLIER = 2.0
PARAMS = {"natr_period": NATR_PERIOD, "natr_multiplier": NATR_MULTIPLIER}
FAKE_LABEL_COLUMN = "&s-zigzag-fake"

# A clean fall then a clean rise: the peak at index 0 and the trough at index 29.
V_LEGS = ((100.0, 80.0, 30), (80.0, 120.0, 31))
# Fall, rise, fall: peak / trough / peak at 0, 29 and 69.
W_LEGS = ((100.0, 80.0, 30), (80.0, 110.0, 40), (110.0, 90.0, 31))
# A 45-candle rise into the peak at index 44, reused by the wobble and reversal cases.
RISE_TO_PEAK = ((100.0, 160.0, 45),)
# A deep candle after a monotone run in each direction. Each clears exactly one of the two
# initial amplitude thresholds, so the seed pivot comes from that move alone.
DIP_AFTER_RISE = ((100.0, 130.0, 20), (95.0, 95.0, 1), (95.0, 120.0, 20))
RISE_AFTER_FALL = ((130.0, 95.0, 20), (135.0, 135.0, 1), (135.0, 110.0, 20))
# A confirmed peak followed by a confirmed trough, used for the rebound threshold cases.
PEAK_AND_TROUGH = ((100.0, 160.0, 45), (160.0, 120.0, 16))
# The dip depth in the middle of the rising leg of W_LEGS, in price units.
WIGGLE_DEPTH = 3.0
WIGGLE_SLICE = slice(15, 18)

METRIC_FIELDS = {
    "amplitude": "amplitudes",
    "amplitude_threshold_ratio": "amplitude_threshold_ratios",
    "volume_rate": "volume_rates",
    "speed": "speeds",
    "efficiency_ratio": "efficiency_ratios",
    "volume_weighted_efficiency_ratio": "volume_weighted_efficiency_ratios",
}


def price_path(*legs: tuple[float, float, int]) -> np.ndarray:
    """Join (start, stop, candles) linear legs into one close series."""
    return np.concatenate([np.linspace(start, stop, candles) for start, stop, candles in legs])


def zigzag_result(closes: np.ndarray) -> ZigzagResult:
    """Run ``_zigzag`` on a close series under the shared NATR parameters."""
    return _zigzag(ohlcv_frame(closes), **PARAMS)


def zigzag_on(*legs: tuple[float, float, int]) -> ZigzagResult:
    """Run ``_zigzag`` on a price path assembled from ``legs``."""
    return zigzag_result(price_path(*legs))


def nested_bar_frame(second_close: float, second_spread: float) -> pd.DataFrame:
    """Return a frame whose first two wide bars clear both initial thresholds on one candle."""
    closes = np.concatenate(
        [
            [100.0, second_close],
            np.linspace(second_close, 100.0, 6),
            np.linspace(100.0, 80.0, 10),
            np.linspace(80.0, 120.0, 15),
        ]
    )
    spreads = np.full(len(closes), 0.001)
    spreads[0] = 0.10
    spreads[1] = second_spread
    return ohlcv_frame(closes, spread=spreads)


def wiggly_closes() -> np.ndarray:
    """Return the W path with a counter-trend dip inside its rising leg."""
    fall, rise, decline = (np.linspace(start, stop, candles) for start, stop, candles in W_LEGS)
    rise = rise.copy()
    rise[WIGGLE_SLICE] -= WIGGLE_DEPTH
    return np.concatenate([fall, rise, decline])


def soft_extremum_oracle(values: np.ndarray, alpha: float) -> float:
    """Return the softmax-weighted mean a finite ``alpha`` is defined to produce."""
    finite = np.isfinite(values)
    scaled = alpha * values[finite]
    weights = np.exp(scaled - np.nanmax(scaled))
    return float(np.average(values[finite], weights=weights))


def two_positional_recorder(calls: list) -> callable:
    """Return a canonical two-positional generator that records every dispatch."""

    def generator(dataframe, params):
        calls.append((len(dataframe), dict(params)))
        return LabelData(series=pd.Series(0.0, index=dataframe.index), indices=[], metrics={})

    return generator


def three_positional_recorder(calls: list) -> callable:
    """Return a canonical three-positional generator that records the logger it receives."""

    def generator(dataframe, params, logger):
        calls.append((len(dataframe), dict(params), logger))
        return LabelData(series=pd.Series(1.0, index=dataframe.index), indices=[], metrics={})

    return generator


def three_positional_with_default(dataframe, params, logger=None):
    """A canonical three-positional generator whose ``logger`` carries a default."""
    return LabelData(series=pd.Series(0.0, index=dataframe.index), indices=[], metrics={})


# --- generator signatures the registry must reject ---------------------------------------


def var_positional_generator(*args):
    """``*args`` is rejected: the positional arity cannot be inspected."""


def var_keyword_generator(**kwargs):
    """``**kwargs`` is rejected: the generator could swallow the logger silently."""


def keyword_only_logger_generator(dataframe, params, *, logger=None):
    """A keyword-only ``logger`` cannot be filled positionally at dispatch."""


def misnamed_third_generator(dataframe, params, log):
    """A required third positional parameter that is not ``logger``."""


def misnamed_optional_third_generator(dataframe, params, log=None):
    """An optional third positional parameter that is not ``logger``."""


def one_positional_generator(dataframe):
    """Too few required positionals: the params mapping would never reach the generator."""


def four_positional_generator(dataframe, params, logger, extra):
    """Too many required positionals: dispatch could never supply the fourth."""


class UtilsZigzagTest(QaTestCase):
    def test_a_v_shaped_series_marks_its_two_turning_points(self):
        result = zigzag_on(*V_LEGS)

        self.assertEqual(result.indices, [0, 29])
        # The direction names the pivot's role, not the leg that follows it: index 0 is the
        # peak the fall was measured from (UP) and index 29 the trough (DOWN).
        self.assertEqual(result.directions, [TrendDirection.UP, TrendDirection.DOWN])
        self.assertNotIn(TrendDirection.NEUTRAL, result.directions)
        np.testing.assert_allclose(
            result.values_log,
            [np.log(100.0 * 1.001), np.log(80.0 * 0.999)],
            rtol=1e-12,
            atol=0.0,
        )

    def test_three_turning_points_alternate_peak_trough_peak(self):
        closes = price_path(*W_LEGS)

        result = zigzag_result(closes)

        self.assertEqual(result.indices, [0, 29, 69])
        self.assertEqual(
            result.directions,
            [TrendDirection.UP, TrendDirection.DOWN, TrendDirection.UP],
        )
        # Each pivot lands exactly on the turning point of the leg it closes, not one
        # candle late: the peaks are the tops of their legs and the trough their bottom.
        # The legs share their endpoint, so leg k starts one candle before its own share.
        starts = np.cumsum([0] + [candles for _, _, candles in W_LEGS])
        fall = closes[: starts[1]]
        rise = closes[starts[1] - 1 : starts[2]]
        decline = closes[starts[2] - 1 : starts[3]]
        self.assertEqual(float(fall[0]), 100.0)
        self.assertEqual(float(rise[0]), float(fall[-1]))
        self.assertEqual(float(decline[0]), float(rise[-1]))
        self.assertEqual(result.indices[0], int(np.argmax(fall)))
        self.assertEqual(result.indices[1], starts[1] - 1 + int(np.argmin(rise)))
        self.assertEqual(result.indices[2], starts[2] - 1 + int(np.argmax(decline)))

    def test_pivot_metrics_are_backfilled_from_the_next_pivot(self):
        result = zigzag_on(*W_LEGS)
        first, second, last = result.indices

        # A leg is measurable only once the pivot closing it is known, so the final pivot
        # carries no amplitude, speed or efficiency ratio.
        self.assertTrue(np.isnan(result.amplitudes[-1]))
        self.assertTrue(np.isnan(result.speeds[-1]))
        self.assertTrue(np.isnan(result.efficiency_ratios[-1]))
        self.assertTrue(np.isnan(result.volume_weighted_efficiency_ratios[-1]))
        self.assertTrue(np.all(np.isfinite(result.amplitudes[:-1])))

        fall = abs(np.log(80.0 * 0.999) - np.log(100.0 * 1.001))
        rise = abs(np.log(110.0 * 1.001) - np.log(80.0 * 0.999))
        self.assertEqual(result.amplitudes[0], fall)
        self.assertEqual(result.amplitudes[1], rise)
        # speed is amplitude per candle of the leg the pivot closes.
        self.assertEqual(result.speeds[0], fall / (second - first))
        self.assertEqual(result.speeds[1], rise / (last - second))
        # Both legs are monotone, so the travelled path equals the net move.
        self.assertEqual(result.efficiency_ratios[:2], [1.0, 1.0])
        self.assertEqual(result.volume_weighted_efficiency_ratios[:2], [1.0, 1.0])
        # Constant volume makes the average volume per candle equal its median.
        self.assertEqual(result.volume_rates[:2], [1.0, 1.0])
        self.assertTrue(np.all(np.asarray(result.amplitude_threshold_ratios[:-1]) > 0.0))

    def test_the_volume_rate_compares_the_leg_average_to_its_median(self):
        closes = price_path(*W_LEGS)
        # A volume that rises and falls inside every leg, so the average and the median of a
        # leg differ and the ratio is not the identity.
        volumes = 10.0 + 5.0 * np.sin(np.arange(len(closes)) / 4.0)

        result = _zigzag(ohlcv_frame(closes, volumes), **PARAMS)

        self.assertEqual(result.indices, [0, 29, 69])
        for index in (0, 1):
            window = volumes[result.indices[index] : result.indices[index + 1] + 1]
            with self.subTest(pivot=result.indices[index]):
                self.assertEqual(
                    result.volume_rates[index], float(window.mean() / np.median(window))
                )
        self.assertNotEqual(result.volume_rates[0], 1.0)

    def test_the_efficiency_ratio_measures_directness_of_the_leg(self):
        closes = wiggly_closes()
        result = zigzag_result(closes)

        self.assertEqual(result.indices, [0, 29, 69])
        log_closes = np.log(closes)
        for index, previous in ((0, 0), (1, 29)):
            # The ratio is the net move over the whole travelled path, so a leg containing a
            # counter-trend dip scores strictly below the monotone case.
            start, end = previous, result.indices[index + 1]
            path = float(np.abs(np.diff(log_closes[start : end + 1])).sum())
            net = abs(float(log_closes[end] - log_closes[start]))
            with self.subTest(pivot=result.indices[index]):
                np.testing.assert_allclose(
                    result.efficiency_ratios[index], net / path, rtol=1e-12, atol=0.0
                )
        self.assertEqual(result.efficiency_ratios[0], 1.0)
        self.assertLess(result.efficiency_ratios[1], 1.0)

    def test_the_public_tuple_mirrors_the_result_fields(self):
        frame = ohlcv_frame(price_path(*W_LEGS))

        result = _zigzag(frame, **PARAMS)

        published = result.as_tuple()
        self.assertEqual(
            published,
            (
                result.indices,
                result.values_log,
                result.directions,
                result.amplitudes,
                result.amplitude_threshold_ratios,
                result.volume_rates,
                result.speeds,
                result.efficiency_ratios,
                result.volume_weighted_efficiency_ratios,
            ),
        )
        # known_at_positions stays outside the tuple: it is a per-row array rather than a
        # per-pivot metric, and the tuple is the per-pivot channel.
        self.assertEqual(len(published), 9)
        self.assertEqual(zigzag(frame, **PARAMS), published)

    def test_a_rising_series_starts_from_a_trough_and_a_falling_one_from_a_peak(self):
        rising = zigzag_on((100.0, 200.0, 60))
        falling = zigzag_on((200.0, 100.0, 60))

        # The first significant move is what orients the scan, so the seed pivot is the
        # opposite extreme: a rise is confirmed from the trough, a fall from the peak.
        self.assertEqual(rising.indices, [0])
        self.assertEqual(rising.directions, [TrendDirection.DOWN])
        self.assertEqual(falling.indices, [0])
        self.assertEqual(falling.directions, [TrendDirection.UP])

    def test_a_flat_series_never_produces_a_pivot(self):
        flat = zigzag_on((100.0, 100.0, 60))

        self.assertEqual(flat.indices, [])
        self.assertEqual(flat.directions, [])
        self.assertEqual(flat.amplitudes, [])
        # NEUTRAL is the pre-orientation sentinel only: with no price move at all no pivot
        # is ever confirmed, so no NEUTRAL label is ever emitted.
        self.assertEqual(TrendDirection.NEUTRAL, 0)
        self.assertEqual(TrendDirection.UP, 1)
        self.assertEqual(TrendDirection.DOWN, -1)

    def test_a_short_series_yields_no_pivot_but_a_full_known_at_row(self):
        frame = ohlcv_frame(price_path((100.0, 90.0, 3)))

        result = _zigzag(frame, **PARAMS)

        self.assertEqual(result.indices, [])
        np.testing.assert_array_equal(result.known_at_positions, np.full(3, 3, dtype=np.int64))

    def test_a_single_candle_wobble_does_not_confirm_a_reversal(self):
        # One candle down 12.5 % that recovers above the peak within three candles. The dip
        # clears the amplitude threshold by a wide margin, so only the slope test refuses it.
        wobble = zigzag_on(*RISE_TO_PEAK, (140.0, 165.0, 4), (165.0, 180.0, 15))

        self.assertEqual(wobble.indices, [0])
        self.assertEqual(wobble.directions, [TrendDirection.DOWN])

    def test_a_sustained_reversal_is_confirmed_and_labels_the_peak(self):
        sustained = zigzag_on(*RISE_TO_PEAK, (160.0, 120.0, 16), (120.0, 180.0, 16))

        self.assertEqual(sustained.indices, [0, 44, 60])
        self.assertEqual(
            sustained.directions,
            [TrendDirection.DOWN, TrendDirection.UP, TrendDirection.DOWN],
        )

    def test_the_confirmation_boundary_is_the_binomial_minimum(self):
        # The test is a one-sided Binomial(0.5): with every slope in agreement the p-value
        # is 2**-m, so m slopes confirm exactly when 2**-m <= alpha.
        self.assertEqual(_ZIGZAG_CONFIRMATION_ALPHA, 0.05)
        self.assertLessEqual(2.0**-_ZIGZAG_MIN_CONFIRMATION_SLOPES, _ZIGZAG_CONFIRMATION_ALPHA)
        self.assertGreater(2.0 ** (1 - _ZIGZAG_MIN_CONFIRMATION_SLOPES), _ZIGZAG_CONFIRMATION_ALPHA)
        self.assertEqual(_ZIGZAG_MIN_CONFIRMATION_SLOPES, 5)

        rise = price_path(*RISE_TO_PEAK)
        for declining_candles in range(1, _ZIGZAG_MIN_CONFIRMATION_SLOPES):
            tail = np.linspace(160.0, 160.0 - 1.5 * declining_candles, declining_candles + 1)[1:]
            with self.subTest(declining_candles=declining_candles):
                self.assertEqual(zigzag_result(np.concatenate([rise, tail])).indices, [0])

        confirmed_candles = _ZIGZAG_MIN_CONFIRMATION_SLOPES
        tail = np.linspace(160.0, 160.0 - 1.5 * confirmed_candles, confirmed_candles + 1)[1:]
        confirmed = zigzag_result(np.concatenate([rise, tail]))
        self.assertEqual(confirmed.indices, [0, 44])
        self.assertEqual(confirmed.directions, [TrendDirection.DOWN, TrendDirection.UP])

    def test_a_decline_below_the_amplitude_threshold_is_not_confirmed(self):
        rise = price_path(*RISE_TO_PEAK)
        # Both dips run eight declining candles, well past the five-slope minimum, so only
        # the amplitude gate separates them: a 1.5 % fall is inside the NATR threshold and a
        # 15 % fall is outside it.
        shallow = 160.0 - 0.3 * np.arange(1, 9)
        deep = 160.0 - 3.0 * np.arange(1, 9)

        refused = np.concatenate([rise, shallow, np.linspace(shallow[-1], 175.0, 14)])
        confirmed = np.concatenate([rise, deep, np.linspace(deep[-1], 175.0, 14)])

        self.assertEqual(zigzag_result(refused).indices, [0])
        self.assertEqual(zigzag_result(confirmed).indices[:2], [0, 44])

    def test_a_rebound_below_the_amplitude_threshold_is_not_confirmed(self):
        # The mirror of the decline case: eight rising candles off the confirmed trough at
        # index 60 clear the slope minimum but stay inside the NATR amplitude threshold.
        approach = price_path(*PEAK_AND_TROUGH)
        shallow = 120.0 + 0.4 * np.arange(1, 9)
        deep = 120.0 + 1.0 * np.arange(1, 9)

        refused = np.concatenate([approach, shallow, np.linspace(shallow[-1], 100.0, 14)])
        confirmed = np.concatenate([approach, deep, np.linspace(deep[-1], 100.0, 14)])

        self.assertEqual(zigzag_result(refused).indices, [0, 44])
        self.assertEqual(zigzag_result(confirmed).indices, [0, 44, 60])

    def test_the_first_clear_initial_threshold_orients_the_scan(self):
        # Only one of the two thresholds is cleared at a time here, so the seed pivot is the
        # extreme that the clearing move started from.
        after_rise = zigzag_on(*DIP_AFTER_RISE)
        after_fall = zigzag_on(*RISE_AFTER_FALL)

        self.assertEqual(after_rise.indices, [0, 19, 25])
        self.assertEqual(
            after_rise.directions, [TrendDirection.DOWN, TrendDirection.UP, TrendDirection.DOWN]
        )
        self.assertEqual(after_fall.indices, [0, 19, 25])
        self.assertEqual(
            after_fall.directions, [TrendDirection.UP, TrendDirection.DOWN, TrendDirection.UP]
        )

    def test_the_larger_of_two_simultaneous_moves_orients_the_scan(self):
        # Two nested wide bars clear the high threshold and the low threshold on the same
        # candle without setting a new high or a new low, so the scan compares the two moves
        # instead of taking whichever it happens to test first.
        peak_first = _zigzag(nested_bar_frame(99.5, 0.0950), **PARAMS)
        trough_first = _zigzag(nested_bar_frame(100.0, 0.0905), **PARAMS)

        self.assertEqual(peak_first.indices[0], 0)
        self.assertEqual(peak_first.directions[0], TrendDirection.UP)
        self.assertEqual(trough_first.indices[0], 0)
        self.assertEqual(trough_first.directions[0], TrendDirection.DOWN)

    def test_the_label_series_is_aligned_to_the_input_index(self):
        frame = ohlcv_frame(price_path(*W_LEGS))

        label = generate_label_data(frame, EXTREMA_COLUMN, PARAMS, LOGGER)

        self.assertTrue(label.series.index.equals(frame.index))
        self.assertEqual(label.series.dtype, np.float64)
        self.assertEqual(sorted(set(label.series.tolist())), [-1.0, 0.0, 1.0])
        self.assertEqual(label.indices, [0, 29, 69])
        for position, direction in zip(label.indices, [1.0, -1.0, 1.0], strict=True):
            self.assertEqual(label.series.iloc[position], direction)
        self.assertEqual(int((label.series != 0.0).sum()), len(label.indices))

    def test_pivot_indices_are_positions_not_index_labels(self):
        # `LabelData.indices` is documented as "positions of detected pivots in series", and
        # `compute_label_weights` consumes it that way: it casts to int and bounds the result
        # by the row count. Every other fixture in this suite uses a zero-based RangeIndex,
        # where a label and a position coincide, so returning `df.index` instead of positions
        # would pass everywhere. These three indexes are what make the two distinguishable.
        closes = price_path(*W_LEGS)
        expected = generate_label_data(ohlcv_frame(closes), EXTREMA_COLUMN, PARAMS, LOGGER)
        indexes = {
            "shifted_int": pd.RangeIndex(500, 500 + len(closes)),
            "datetime": pd.date_range("2024-01-01", periods=len(closes), freq="5min"),
            "string": pd.Index([f"candle-{i}" for i in range(len(closes))]),
        }
        for name, index in indexes.items():
            with self.subTest(index=name):
                frame = ohlcv_frame(closes).set_axis(index)
                label = generate_label_data(frame, EXTREMA_COLUMN, PARAMS, LOGGER)
                self.assertEqual(label.indices, expected.indices)
                self.assertEqual(label.indices, sorted(set(label.indices)))
                np.testing.assert_array_equal(label.series.to_numpy(), expected.series.to_numpy())
                self.assertTrue(label.series.index.equals(index))

    def test_label_weights_survive_an_index_that_is_not_a_position(self):
        # The consumer side of the same contract. `compute_label_weights` casts the pivot
        # indices to int and bounds them by n_values, so index labels raise TypeError on the
        # DatetimeIndex that FreqAI actually passes, and are dropped as out-of-range on a
        # shifted integer index, leaving every weight at zero with only a warning.
        from Utils import compute_label_weights

        closes = price_path(*W_LEGS)
        expected = generate_label_data(ohlcv_frame(closes), EXTREMA_COLUMN, PARAMS, LOGGER)
        for name, index in (
            ("shifted_int", pd.RangeIndex(500, 500 + len(closes))),
            ("datetime", pd.date_range("2024-01-01", periods=len(closes), freq="5min")),
        ):
            with self.subTest(index=name):
                frame = ohlcv_frame(closes).set_axis(index)
                label = generate_label_data(frame, EXTREMA_COLUMN, PARAMS, LOGGER)
                weights = compute_label_weights(
                    len(closes),
                    label.indices,
                    {"efficiency_ratio": [1.0] * len(label.indices)},
                    {"strategy": "uniform"},
                    logger=LOGGER,
                )
                self.assertEqual(int((weights > 0).sum()), len(expected.indices))

    def test_the_label_metrics_mirror_the_pivot_metrics(self):
        frame = ohlcv_frame(price_path(*W_LEGS))
        result = _zigzag(frame, **PARAMS)

        label = generate_label_data(frame, EXTREMA_COLUMN, PARAMS, LOGGER)

        self.assertEqual(list(label.metrics), list(METRIC_FIELDS))
        for name, field in METRIC_FIELDS.items():
            with self.subTest(metric=name):
                self.assertEqual(len(label.metrics[name]), len(label.indices))
                np.testing.assert_allclose(
                    label.metrics[name], getattr(result, field), equal_nan=True
                )

    def test_the_label_lookahead_is_the_confirmation_distance(self):
        frame = ohlcv_frame(price_path(*W_LEGS))
        result = _zigzag(frame, **PARAMS)

        label = generate_label_data(frame, EXTREMA_COLUMN, PARAMS, LOGGER)

        lookahead = label.known_at_lookahead
        self.assertIsNotNone(lookahead)
        self.assertTrue(lookahead.index.equals(frame.index))
        self.assertEqual(lookahead.dtype, np.int64)
        # The lookahead is a distance in candles, not an absolute position.
        np.testing.assert_array_equal(
            lookahead.to_numpy(),
            result.known_at_positions - np.arange(len(frame), dtype=np.int64),
        )
        self.assertTrue(np.all(np.diff(result.known_at_positions) >= 0))
        # Each pivot stamps its confirmation position over the rows it resolves, and never
        # over rows a later pivot has already stamped: 5 for the orientation candle (the
        # ATR warm-up, which no earlier confirmation could undercut), then 37, 84 and
        # finally n for the tail the scan never resolved.
        n = len(frame)
        expected = np.concatenate(
            [
                np.full(1, 5),
                np.full(37, 37),
                np.full(47, 84),
                np.full(n - 85, n),
            ]
        )
        self.assertEqual(n, 101)
        np.testing.assert_array_equal(result.known_at_positions, expected)
        # Every row is resolved no earlier than itself, and the unresolved tail, marked by
        # known_at == n, still has to burn off one candle of lookahead per row.
        self.assertTrue(np.all(result.known_at_positions >= np.arange(len(frame), dtype=np.int64)))
        unresolved = result.known_at_positions == len(frame)
        self.assertGreaterEqual(int(unresolved.sum()), 2)
        self.assertTrue(np.all(lookahead.to_numpy()[unresolved] > 0))
        self.assertEqual(
            np.diff(lookahead.to_numpy()[unresolved]).tolist(),
            [-1] * (int(unresolved.sum()) - 1),
        )

    def test_an_unregistered_label_column_raises_a_key_error_listing_the_registry(self):
        frame = ohlcv_frame(price_path(*W_LEGS))

        self.assertEqual(LABEL_COLUMNS, (EXTREMA_COLUMN,))
        self.assertIn(EXTREMA_COLUMN, _LABEL_GENERATORS)

        with self.assertRaises(KeyError) as caught:
            generate_label_data(frame, "&s-absent", PARAMS, LOGGER)

        message = str(caught.exception)
        self.assertIn("No label generator registered for column '&s-absent'", message)
        self.assertIn(f"Available columns: {[EXTREMA_COLUMN]}", message)

    def test_a_two_positional_generator_round_trips_and_never_sees_the_logger(self):
        frame = ohlcv_frame(price_path(*W_LEGS))
        calls: list = []
        self.addCleanup(_LABEL_GENERATORS.pop, FAKE_LABEL_COLUMN, None)
        register_label_generator(FAKE_LABEL_COLUMN, two_positional_recorder(calls))

        label = generate_label_data(frame, FAKE_LABEL_COLUMN, PARAMS, LOGGER)

        # Dispatch always passes three arguments; the adapter is what drops the logger.
        self.assertEqual(calls, [(len(frame), PARAMS)])
        self.assertTrue(label.series.index.equals(frame.index))

    def test_the_label_generator_registry_is_restored_after_every_case(self):
        # A second case is run to completion inside this one, so the restore is observed
        # without depending on the order unittest happens to run the methods in.
        class RegisteringCase(QaTestCase):
            def runTest(self):
                register_label_generator(FAKE_LABEL_COLUMN, two_positional_recorder([]))
                self.assertIn(FAKE_LABEL_COLUMN, _LABEL_GENERATORS)

        outcome = RegisteringCase().run()

        self.assertTrue(outcome.wasSuccessful())
        self.assertNotIn(FAKE_LABEL_COLUMN, _LABEL_GENERATORS)
        self.assertEqual(list(_LABEL_GENERATORS), [EXTREMA_COLUMN])

    def test_the_two_canonical_generator_shapes_are_accepted(self):
        calls: list = []
        two = two_positional_recorder(calls)
        three = three_positional_recorder([])
        frame = ohlcv_frame(price_path(*W_LEGS))

        # Two positionals are wrapped so dispatch can still pass three arguments; three
        # positionals named logger pass through untouched, with or without a default.
        adapted = _adapt_label_generator(two)
        self.assertIsNot(adapted, two)
        adapted(frame, PARAMS, LOGGER)
        self.assertEqual(calls, [(len(frame), PARAMS)])
        self.assertIs(_adapt_label_generator(three), three)
        self.assertIs(
            _adapt_label_generator(three_positional_with_default),
            three_positional_with_default,
        )

    def test_a_three_positional_generator_receives_the_logger(self):
        frame = ohlcv_frame(price_path(*W_LEGS))
        calls: list = []
        self.addCleanup(_LABEL_GENERATORS.pop, FAKE_LABEL_COLUMN, None)
        register_label_generator(FAKE_LABEL_COLUMN, three_positional_recorder(calls))

        label = generate_label_data(frame, FAKE_LABEL_COLUMN, PARAMS, LOGGER)

        self.assertEqual(calls, [(len(frame), PARAMS, LOGGER)])
        self.assertEqual(set(label.series.tolist()), {1.0})

    def test_every_other_generator_signature_is_rejected(self):
        rejected = (
            (var_positional_generator, "*args"),
            (var_keyword_generator, "**kwargs"),
            (keyword_only_logger_generator, "keyword-only ``logger`` is not supported"),
            (misnamed_third_generator, "third positional parameter is named 'log'"),
            (misnamed_optional_third_generator, "third positional parameter is named 'log'"),
            (one_positional_generator, "1 required positional parameter(s); expected at least 2"),
            (four_positional_generator, "4 required positional parameter(s); expected 2"),
        )
        for generator, reason in rejected:
            with self.subTest(generator=generator.__name__):
                with self.assertRaises(ValueError) as caught:
                    _adapt_label_generator(generator)
                self.assertIn(reason, str(caught.exception))

    def test_a_rejected_generator_is_never_registered(self):
        frame = ohlcv_frame(price_path(*W_LEGS))

        with self.assertRaises(ValueError):
            register_label_generator(FAKE_LABEL_COLUMN, misnamed_third_generator)

        self.assertNotIn(FAKE_LABEL_COLUMN, _LABEL_GENERATORS)
        with self.assertRaises(KeyError):
            generate_label_data(frame, FAKE_LABEL_COLUMN, PARAMS, LOGGER)

    def test_alpha_zero_returns_the_plain_mean(self):
        values = np.array([1.0, 2.0, 3.0, 9.0])

        self.assertEqual(soft_extremum(pd.Series(values), 0.0), float(np.mean(values)))

    def test_the_plain_mean_skips_nan_but_keeps_infinities(self):
        # The alpha-0 branch calls np.nanmean directly, so - unlike the weighted branch,
        # which runs through nan_average's finite mask - it does not strip +/-inf.
        self.assertEqual(soft_extremum(pd.Series([1.0, np.nan, 3.0, 5.0]), 0.0), 3.0)
        self.assertEqual(soft_extremum(pd.Series([1.0, np.inf]), 0.0), np.inf)

    def test_a_negligible_alpha_is_treated_as_zero(self):
        # The gate is np.isclose(alpha, 0.0), so an alpha far below its tolerance disables
        # the softmax outright instead of merely flattening it.
        values = np.array([1.0, 2.0, 3.0, 9.0])
        plain = float(np.mean(values))

        for alpha in (0.0, 1e-12, -1e-12, 1e-9):
            with self.subTest(alpha=alpha):
                self.assertEqual(soft_extremum(pd.Series(values), alpha), plain)

    def test_a_finite_alpha_returns_the_softmax_weighted_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        series = pd.Series(values)

        for alpha in (-12.0, -1.0, 0.5, 1.0, 3.0, 12.0):
            with self.subTest(alpha=alpha):
                np.testing.assert_allclose(
                    soft_extremum(series, alpha),
                    soft_extremum_oracle(values, alpha),
                    rtol=1e-12,
                    atol=0.0,
                )

        # The sign of alpha picks which end is favoured, and the magnitude concentrates the
        # weight: a soft minimum sits below the mean, a soft maximum above it.
        self.assertLess(soft_extremum(series, -12.0), soft_extremum(series, -1.0))
        self.assertLess(soft_extremum(series, 1.0), soft_extremum(series, 12.0))
        self.assertLess(soft_extremum(series, -12.0), float(np.mean(values)))
        self.assertGreater(soft_extremum(series, 12.0), float(np.mean(values)))

    def test_the_weighted_mean_ignores_missing_values(self):
        values = np.array([1.0, np.nan, 3.0, 5.0])

        np.testing.assert_allclose(
            soft_extremum(pd.Series(values), 1.0),
            soft_extremum_oracle(values, 1.0),
            rtol=1e-12,
            atol=0.0,
        )

    def test_empty_or_all_non_finite_input_returns_nan(self):
        for values in ([], [np.nan], [np.nan, np.nan, np.nan], [np.inf, -np.inf]):
            for alpha in (0.0, 1.0, 12.0):
                with self.subTest(values=values, alpha=alpha):
                    self.assertTrue(np.isnan(soft_extremum(pd.Series(values, dtype=float), alpha)))

    def test_a_non_finite_softmax_scale_falls_back_to_the_argmax_element(self):
        # alpha * 2 overflows, so the shift constant is infinite and no softmax can be
        # formed; the fallback is the element the scaled series maximises.
        with np.errstate(over="ignore"):
            self.assertEqual(soft_extremum(pd.Series([1.0, 2.0, 0.5]), 1e308), 2.0)
            self.assertEqual(soft_extremum(pd.Series([-5.0, 2.0, 0.5]), 1e308), 2.0)
            # Every scaled entry is +inf here, so the argmax is the first position.
            self.assertEqual(soft_extremum(pd.Series([1.0, 2.0, 0.5]), np.inf), 1.0)


if __name__ == "__main__":
    unittest.main()
