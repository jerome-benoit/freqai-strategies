"""Candle-cache, threshold and take-profit feature contracts; requires the Freqtrade QA image."""

import datetime
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import QuickAdapterV3 as strategy_module
from EnumErrors import enum_error_message
from numpy.testing import assert_allclose
from qa_support import PAIR, QaTestCase
from QuickAdapterV3 import QuickAdapterV3

LONG = QuickAdapterV3._TRADE_LONG
SHORT = QuickAdapterV3._TRADE_SHORT
DIRECT = QuickAdapterV3._INTERPOLATION_DIRECT
INVERSE = QuickAdapterV3._INTERPOLATION_INVERSE
FIRST_CANDLE = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
TIMEFRAME = "5m"
MIN_FRACTION = 0.2
MAX_FRACTION = 0.8
MULTIPLIER = 3.0
STEP_MINUTES = 5


def mid_rank(window, value):
    """The rank ``calculate_quantile`` reports, from scipy's ``kind="mean"`` convention.

    ``percentileofscore(..., kind="mean")`` scores a value as the mean of the ranks
    it spans, i.e. ``(strictly_below + 0.5 * tied) / n``, and ``calculate_quantile``
    divides that by 100. Stated here rather than called, so the expectations below
    rest on the documented convention instead of on the helper under test.
    """
    below = sum(1 for candidate in window if candidate < value)
    tied = sum(1 for candidate in window if candidate == value)
    return (below + 0.5 * tied) / len(window)


def interpolated_fraction(quantile, exponent, low=MIN_FRACTION, high=MAX_FRACTION):
    """The natr-multiplier fraction a candle of quantile rank ``quantile`` is priced at."""
    shaped = quantile**exponent
    return low + (high - low) * shaped


def candles(
    closes,
    natr=None,
    multiplier=None,
    opens=None,
    highs=None,
    lows=None,
    start=FIRST_CANDLE,
):
    """An OHLC frame keyed by ``date``, carrying the optional label-NATR columns."""
    closes = np.asarray(closes, dtype=float)
    count = closes.size
    frame = pd.DataFrame(
        {
            "date": [start + datetime.timedelta(minutes=STEP_MINUTES * i) for i in range(count)],
            "open": closes - 0.5 if opens is None else np.asarray(opens, dtype=float),
            "high": closes + 1.0 if highs is None else np.asarray(highs, dtype=float),
            "low": closes - 1.0 if lows is None else np.asarray(lows, dtype=float),
            "close": closes,
        }
    )
    if natr is not None:
        frame["natr_label_period_candles"] = np.asarray(natr, dtype=float)
    if multiplier is not None:
        frame["label_natr_multiplier"] = np.full(count, float(multiplier))
    return frame


def strategy(label_period_candles=4, label_natr_multiplier=MULTIPLIER):
    """A candle-cache strategy whose label parameters come from the pair, not the frame."""
    model = object.__new__(QuickAdapterV3)
    model._candle_deviation_cache = {}
    model._candle_threshold_cache = {}
    model._cached_df_signature = {}
    model._label_params = {
        PAIR: {
            "label_period_candles": label_period_candles,
            "label_natr_multiplier": label_natr_multiplier,
        }
    }
    model._label_defaults = (24, 5.0)
    model.freqai_info = {}
    model.config = {
        "timeframe": TIMEFRAME,
        "exit_pricing": {"trade_natr_method": "quantile_interpolation"},
    }
    return model


def trade(open_rate=100.0, is_short=False, open_candle_offset=1):
    """A trade opened ``open_candle_offset`` candles after the frame's first candle."""
    return SimpleNamespace(
        pair=PAIR,
        open_rate=float(open_rate),
        is_short=is_short,
        open_date_utc=FIRST_CANDLE + datetime.timedelta(minutes=STEP_MINUTES * open_candle_offset),
    )


class StrategyFeaturesTest(QaTestCase):
    def deviation_of(
        self,
        model,
        frame,
        candle_idx,
        exponent=1.5,
        direction=DIRECT,
        low=MIN_FRACTION,
        high=MAX_FRACTION,
    ):
        """The candle deviation the strategy derives, from its own quantile and multiplier."""
        natr = frame["natr_label_period_candles"]
        candle_idx = QuickAdapterV3._normalize_candle_idx(len(natr), candle_idx)
        label_period_candles = model._label_params[PAIR]["label_period_candles"]
        window = natr.iloc[: candle_idx + 1].to_numpy()[-label_period_candles:]
        value = natr.iloc[candle_idx]
        quantile = mid_rank(window, value)
        if direction == DIRECT:
            fraction = interpolated_fraction(quantile, exponent, low, high)
        else:
            fraction = high - (high - low) * quantile**exponent
        multiplier = model._label_params[PAIR]["label_natr_multiplier"]
        return float(value) / 100.0 * multiplier * fraction

    def test_a_signature_ignores_the_feature_columns(self):
        base = candles([100.0, 101.0, 102.0, 103.0])
        extended = base.copy()
        extended["%-rsi-14"] = [1.0, 2.0, 3.0, 4.0]
        extended["%-day_of_week"] = 0.25
        self.assertEqual(QuickAdapterV3._df_signature(extended), QuickAdapterV3._df_signature(base))
        self.assertEqual(QuickAdapterV3._df_signature(extended), (4, base["date"].iloc[-1]))

    def test_a_signature_separates_a_grown_frame_from_its_prefix(self):
        base = candles([100.0, 101.0, 102.0, 103.0])
        grown = candles([104.0], start=FIRST_CANDLE + datetime.timedelta(minutes=3 * STEP_MINUTES))
        self.assertNotEqual(QuickAdapterV3._df_signature(grown), QuickAdapterV3._df_signature(base))
        self.assertEqual(
            QuickAdapterV3._df_signature(grown),
            (1, FIRST_CANDLE + datetime.timedelta(minutes=3 * STEP_MINUTES)),
        )

    def test_a_signature_separates_frames_whose_last_date_differs(self):
        base = candles([100.0, 101.0, 102.0, 103.0])
        refiled = candles(
            [100.0, 101.0, 102.0, 999.0],
            start=FIRST_CANDLE + datetime.timedelta(minutes=STEP_MINUTES),
        )
        self.assertNotEqual(
            QuickAdapterV3._df_signature(refiled), QuickAdapterV3._df_signature(base)
        )
        self.assertEqual(
            QuickAdapterV3._df_signature(refiled)[0], QuickAdapterV3._df_signature(base)[0]
        )

    def test_an_empty_or_dateless_frame_carries_no_candle(self):
        self.assertEqual(QuickAdapterV3._df_signature(pd.DataFrame()), (0, None))
        self.assertEqual(
            QuickAdapterV3._df_signature(pd.DataFrame({"close": [1.0, 2.0]})), (2, None)
        )
        self.assertEqual(
            QuickAdapterV3._df_signature(pd.DataFrame({"date": [], "close": []})), (0, None)
        )

    def test_an_unchanged_frame_reuses_the_cached_deviation(self):
        model = strategy()
        frame = candles([100.0, 101.0, 102.0, 103.0], natr=[1.0, 2.0, 3.0, 4.0])
        counted = mock.Mock(side_effect=strategy_module.calculate_quantile)
        with mock.patch.object(strategy_module, "calculate_quantile", counted):
            first = model._calculate_candle_deviation(frame, PAIR, MIN_FRACTION, MAX_FRACTION)
            second = model._calculate_candle_deviation(frame, PAIR, MIN_FRACTION, MAX_FRACTION)
        self.assertEqual(counted.call_count, 1)
        self.assertEqual(second, first)
        self.assertEqual(len(model._candle_deviation_cache), 1)

    def test_a_frame_that_grew_recomputes_the_deviation_and_threshold(self):
        model = strategy()
        frame = candles(
            [100.0, 101.0, 102.0, 103.0], natr=[1.0, 2.0, 3.0, 4.0], multiplier=MULTIPLIER
        )
        stale_deviation = model._calculate_candle_deviation(frame, PAIR, MIN_FRACTION, MAX_FRACTION)
        stale_threshold = model._calculate_candle_threshold(
            frame, PAIR, LONG, MIN_FRACTION, MAX_FRACTION
        )
        grown = pd.concat(
            [
                frame,
                candles(
                    [104.0],
                    natr=[5.0],
                    multiplier=MULTIPLIER,
                    start=FIRST_CANDLE + datetime.timedelta(minutes=4 * STEP_MINUTES),
                ),
            ],
            ignore_index=True,
        )
        deviation = model._calculate_candle_deviation(grown, PAIR, MIN_FRACTION, MAX_FRACTION)
        threshold = model._calculate_candle_threshold(grown, PAIR, LONG, MIN_FRACTION, MAX_FRACTION)
        assert_allclose(deviation, self.deviation_of(model, grown, -1), rtol=1e-9, atol=1e-12)
        assert_allclose(threshold, 104.0 * (1.0 + deviation), rtol=1e-9, atol=1e-12)
        self.assertNotEqual(deviation, stale_deviation)
        self.assertNotEqual(threshold, stale_threshold)
        grown_signature = QuickAdapterV3._df_signature(grown)
        self.assertEqual(model._cached_df_signature[PAIR], grown_signature)
        self.assertEqual(
            [key[1] for key in model._candle_deviation_cache],
            [grown_signature],
        )
        self.assertEqual(
            [key[1] for key in model._candle_threshold_cache],
            [grown_signature],
        )

    def test_a_matching_signature_keeps_the_pair_caches(self):
        model = strategy()
        frame = candles([100.0, 101.0, 102.0, 103.0], natr=[1.0, 2.0, 3.0, 4.0])
        signature = QuickAdapterV3._df_signature(frame)
        model._calculate_candle_deviation(frame, PAIR, MIN_FRACTION, MAX_FRACTION)
        model._calculate_candle_threshold(frame, PAIR, LONG, MIN_FRACTION, MAX_FRACTION)
        model._invalidate_pair_caches(PAIR, signature)
        self.assertEqual(len(model._candle_deviation_cache), 1)
        self.assertEqual(len(model._candle_threshold_cache), 1)
        self.assertEqual(model._cached_df_signature[PAIR], signature)

    def test_invalidating_a_pair_drops_its_caches_and_signature_only(self):
        model = strategy()
        frame = candles([100.0, 101.0, 102.0, 103.0], natr=[1.0, 2.0, 3.0, 4.0])
        other = "ETH/USDT"
        model._calculate_candle_deviation(frame, PAIR, MIN_FRACTION, MAX_FRACTION)
        model._calculate_candle_threshold(frame, PAIR, LONG, MIN_FRACTION, MAX_FRACTION)
        signature = QuickAdapterV3._df_signature(frame)
        model._cached_df_signature[other] = signature
        model._candle_deviation_cache[(other, signature, 0.0, 1.0, -1, DIRECT, 1.0)] = 0.5
        model._candle_threshold_cache[(other, signature, LONG, -1, 0.0, 1.0)] = 0.5
        model._invalidate_pair_caches(PAIR, None)
        self.assertEqual({key[0] for key in model._candle_deviation_cache}, {other})
        self.assertEqual({key[0] for key in model._candle_threshold_cache}, {other})
        self.assertNotIn(PAIR, model._cached_df_signature)
        self.assertEqual(model._cached_df_signature[other], signature)

    def test_the_deviation_follows_the_quantile_rank_of_the_candle(self):
        model = strategy(label_period_candles=5)
        natr = [1.0, 2.0, 3.0, 4.0, 5.0]
        frame = candles([100.0, 101.0, 102.0, 103.0, 104.0], natr=natr)
        # Mid-rank of the top value in a k-length strictly increasing window is (k - 0.5) / k.
        for candle_idx, quantile in enumerate((0.5, 0.75, 2.5 / 3, 0.875, 0.9)):
            with self.subTest(candle_idx=candle_idx):
                deviation = model._calculate_candle_deviation(
                    frame, PAIR, MIN_FRACTION, MAX_FRACTION, candle_idx=candle_idx
                )
                assert_allclose(
                    deviation,
                    natr[candle_idx] / 100.0 * MULTIPLIER * (0.2 + 0.6 * quantile**1.5),
                    rtol=1e-9,
                    atol=1e-12,
                )
                assert_allclose(
                    deviation, self.deviation_of(model, frame, candle_idx), rtol=1e-9, atol=1e-12
                )

    def test_the_deviation_window_is_the_last_label_period_candles(self):
        natr = [1.0, 2.0, 3.0, 4.0, 5.0]
        frame = candles([100.0, 101.0, 102.0, 103.0, 104.0], natr=natr)
        expected = {}
        for label_period_candles in (1, 2, 5):
            model = strategy(label_period_candles=label_period_candles)
            expected[label_period_candles] = [
                model._calculate_candle_deviation(
                    frame, PAIR, MIN_FRACTION, MAX_FRACTION, candle_idx=idx
                )
                for idx in range(len(natr))
            ]
        for label_period_candles, deviations in expected.items():
            for candle_idx, deviation in enumerate(deviations):
                with self.subTest(label_period_candles=label_period_candles, candle_idx=candle_idx):
                    model = strategy(label_period_candles=label_period_candles)
                    assert_allclose(
                        deviation,
                        self.deviation_of(model, frame, candle_idx),
                        rtol=1e-9,
                        atol=1e-12,
                    )
        self.assertNotEqual(expected[1][-1], expected[2][-1])
        self.assertNotEqual(expected[2][-1], expected[5][-1])

    def test_the_inverse_direction_mirrors_the_direct_one(self):
        model = strategy()
        frame = candles([100.0, 101.0, 102.0, 103.0, 104.0], natr=[1.0, 2.0, 3.0, 4.0, 5.0])
        for candle_idx in range(5):
            with self.subTest(candle_idx=candle_idx):
                direct = model._calculate_candle_deviation(
                    frame, PAIR, MIN_FRACTION, MAX_FRACTION, candle_idx=candle_idx
                )
                inverse = model._calculate_candle_deviation(
                    frame,
                    PAIR,
                    MIN_FRACTION,
                    MAX_FRACTION,
                    candle_idx=candle_idx,
                    interpolation_direction=INVERSE,
                )
                natr = float(frame["natr_label_period_candles"].iloc[candle_idx])
                quantile = mid_rank(
                    frame["natr_label_period_candles"].to_numpy()[: candle_idx + 1][
                        -model._label_params[PAIR]["label_period_candles"] :
                    ],
                    natr,
                )
                expected_inverse = (
                    natr
                    / 100.0
                    * MULTIPLIER
                    * (MAX_FRACTION - (MAX_FRACTION - MIN_FRACTION) * quantile**1.5)
                )
                assert_allclose(inverse, expected_inverse, rtol=1e-9, atol=1e-12)
                # The two directions sum to a multiple of the candle's NATR that does not
                # depend on its quantile rank: (natr / 100) * multiplier * (low + high).
                assert_allclose(
                    (direct + inverse) / MULTIPLIER,
                    natr / 100.0 * (MIN_FRACTION + MAX_FRACTION),
                    rtol=1e-12,
                    atol=0.0,
                )
                assert_allclose(
                    inverse,
                    self.deviation_of(model, frame, candle_idx, direction=INVERSE),
                    rtol=1e-9,
                    atol=1e-12,
                )

    def test_the_quantile_exponent_shapes_the_interpolation(self):
        model = strategy()
        frame = candles([100.0, 101.0, 102.0, 103.0], natr=[1.0, 2.0, 3.0, 4.0])
        shaped = {}
        for exponent in (0.5, 1.0, 1.5, 2.0):
            shaped[exponent] = model._calculate_candle_deviation(
                frame, PAIR, MIN_FRACTION, MAX_FRACTION, quantile_exponent=exponent
            )
            self.assertEqual(len(model._candle_deviation_cache), len(shaped))
        self.assertEqual(len(set(shaped.values())), len(shaped))
        for exponent, deviation in shaped.items():
            with self.subTest(exponent=exponent):
                assert_allclose(
                    deviation,
                    self.deviation_of(model, frame, -1, exponent=exponent),
                    rtol=1e-9,
                    atol=1e-12,
                )
        # Exponent 2 on the top rank of a four-value window (7/8) is the exact rational 49/64.
        assert_allclose(shaped[2.0], 0.12 * (0.2 + 0.6 * 0.765625), rtol=1e-9, atol=1e-12)
        assert_allclose(shaped[1.0], 0.12 * (0.2 + 0.6 * 0.875), rtol=1e-9, atol=1e-12)

    def test_an_unknown_interpolation_direction_is_refused(self):
        model = strategy()
        frame = candles([100.0, 101.0, 102.0, 103.0], natr=[1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ValueError) as raised:
            model._calculate_candle_deviation(
                frame, PAIR, MIN_FRACTION, MAX_FRACTION, interpolation_direction="sideways"
            )
        self.assertEqual(
            str(raised.exception),
            enum_error_message(
                "interpolation_direction", "sideways", QuickAdapterV3._INTERPOLATION_DIRECTIONS
            ),
        )

    def test_a_with_trend_candle_is_thresholded_from_its_close(self):
        # Long + bullish and short + bearish are with-trend: the base is the close itself.
        for side, closes, opens in ((LONG, [110.0], [100.0]), (SHORT, [100.0], [110.0])):
            with self.subTest(side=side):
                model = strategy(label_period_candles=1)
                frame = candles(
                    closes,
                    opens=opens,
                    highs=[112.0],
                    lows=[99.0],
                    natr=[1.0],
                    multiplier=MULTIPLIER,
                )
                threshold = model._calculate_candle_threshold(frame, PAIR, side, 0.0, 1.0)
                deviation = model._calculate_candle_deviation(frame, PAIR, 0.0, 1.0)
                assert_allclose(
                    threshold,
                    110.0 * (1 + deviation) if side == LONG else 100.0 * (1 - deviation),
                    rtol=1e-9,
                    atol=1e-12,
                )
                self.assertNotAlmostEqual(
                    threshold,
                    QuickAdapterV3.weighted_close(frame.iloc[0]) * (1 + deviation)
                    if side == LONG
                    else QuickAdapterV3.weighted_close(frame.iloc[0]) * (1 - deviation),
                    delta=1e-6,
                )

    def test_an_adverse_candle_is_thresholded_from_its_weighted_close(self):
        for side, closes, opens in ((LONG, [100.0], [110.0]), (SHORT, [110.0], [100.0])):
            with self.subTest(side=side):
                model = strategy(label_period_candles=1)
                frame = candles(
                    closes,
                    opens=opens,
                    highs=[112.0],
                    lows=[99.0],
                    natr=[1.0],
                    multiplier=MULTIPLIER,
                )
                threshold = model._calculate_candle_threshold(frame, PAIR, side, 0.0, 1.0)
                deviation = model._calculate_candle_deviation(frame, PAIR, 0.0, 1.0)
                weighted_close = QuickAdapterV3.weighted_close(frame.iloc[0])
                self.assertNotAlmostEqual(weighted_close, float(closes[0]))
                assert_allclose(
                    threshold,
                    weighted_close * (1 + deviation)
                    if side == LONG
                    else weighted_close * (1 - deviation),
                    rtol=1e-9,
                    atol=1e-12,
                )

    def test_an_unknown_side_is_refused(self):
        model = strategy(label_period_candles=1)
        frame = candles([110.0], opens=[100.0], natr=[1.0], multiplier=MULTIPLIER)
        with self.assertRaises(ValueError) as raised:
            model._calculate_candle_threshold(frame, PAIR, "sideways", 0.0, 1.0)
        self.assertEqual(
            str(raised.exception),
            enum_error_message("side", "sideways", QuickAdapterV3._TRADE_DIRECTIONS),
        )

    def test_an_unpriceable_candle_is_refused_and_never_memoised(self):
        model = strategy(label_period_candles=1)
        flat = candles(
            [110.0], opens=[100.0], highs=[112.0], lows=[99.0], natr=[0.0], multiplier=MULTIPLIER
        )
        self.assertEqual(model._calculate_candle_deviation(flat, PAIR, 0.0, 1.0), 0.0)
        for side in (LONG, SHORT):
            with self.subTest(zero_volatility=side):
                self.assertTrue(
                    np.isnan(model._calculate_candle_threshold(flat, PAIR, side, 0.0, 1.0))
                )
                # Checked HERE, while this frame's signature is still the live one. Every
                # probe below rotates the timestamp on purpose, so a refusal that memoised
                # would be wiped by the next invalidation and never observable at the end.
                # Inserting a memo into this branch left the suite green for that reason.
                self.assertEqual(model._candle_threshold_cache, {})

        # Every probe below needs a signature the cache has not seen, because the memoised
        # entry of an earlier frame shadows the branch under test: same-signature frames
        # share one cache entry, whatever their values are.
        def unreadable_candle(offset, natr):
            return candles(
                [110.0],
                opens=[100.0],
                highs=[112.0],
                lows=[99.0],
                natr=[natr],
                multiplier=MULTIPLIER,
                start=FIRST_CANDLE + datetime.timedelta(minutes=STEP_MINUTES * offset),
            )

        # Each refusal is checked IMMEDIATELY. Asserting only at the end cannot work here:
        # every probe rotates the frame timestamp on purpose, and
        # `_invalidate_pair_caches` wipes the pair's entries on a signature change, so by the
        # final assertion only the last probe's signature survives and an earlier refusal
        # that had memoised is invisible. Making the two NaN refusal branches memoise left
        # the suite green for exactly that reason.
        for offset, natr in ((1, np.nan), (2, -1.0)):
            with self.subTest(natr=natr):
                deviation = model._calculate_candle_deviation(
                    unreadable_candle(offset, natr), PAIR, 0.0, 1.0
                )
                self.assertTrue(np.isnan(deviation))
                self.assertEqual(model._candle_deviation_cache, {})
                self.assertEqual(model._candle_threshold_cache, {})
        for offset, column in ((3, "close"), (4, "open")):
            holed = unreadable_candle(offset, 1.0)
            holed.loc[0, column] = np.nan
            for side in (LONG, SHORT):
                with self.subTest(missing=column, side=side):
                    self.assertTrue(
                        np.isnan(model._calculate_candle_threshold(holed, PAIR, side, 0.0, 1.0))
                    )
                    # Only the THRESHOLD cache must stay empty here: computing a threshold
                    # legitimately memoises the deviation it derives, and that deviation is
                    # readable because the hole is in the price, not in the NATR.
                    self.assertEqual(model._candle_threshold_cache, {})
        # And the legitimate memoisation from the flat candle is still there, so the two
        # assertions above are not passing because the cache is simply never written to.
        self.assertEqual(len(model._candle_deviation_cache), 1)

    def test_the_weighted_close_lies_between_the_low_and_the_high(self):
        candle = candles([110.0], opens=[100.0], highs=[120.0], lows=[95.0]).iloc[0]
        for weight in (0.0, 0.5, 1.0, 2.0, 5.0):
            with self.subTest(weight=weight):
                value = QuickAdapterV3.weighted_close(candle, weight)
                self.assertGreater(value, float(candle["low"]))
                self.assertLess(value, float(candle["high"]))
                assert_allclose(
                    value,
                    (candle["high"] + candle["low"] + weight * candle["close"]) / (2.0 + weight),
                    rtol=1e-9,
                    atol=1e-9,
                )

    def test_the_default_weight_pulls_the_midpoint_towards_the_close(self):
        # Close 100 sits below the 105.5 midpoint of the 99/112 range, so the default weight
        # of 2.0 has to move the value down from the midpoint without reaching the close.
        candle = candles([100.0], opens=[110.0], highs=[112.0], lows=[99.0]).iloc[0]
        midpoint = QuickAdapterV3.weighted_close(candle, 0.0)
        weighted = QuickAdapterV3.weighted_close(candle)
        assert_allclose(midpoint, 105.5, rtol=1e-9, atol=1e-9)
        assert_allclose(weighted, (112.0 + 99.0 + 2.0 * 100.0) / 4.0, rtol=1e-9, atol=1e-9)
        self.assertLess(weighted, midpoint)
        self.assertGreater(weighted, float(candle["close"]))
        heavier = QuickAdapterV3.weighted_close(candle, 10.0)
        self.assertLess(heavier, weighted)
        self.assertGreater(heavier, float(candle["close"]))

    def test_each_exit_stage_maps_to_its_natr_fraction(self):
        model = strategy()
        frame = candles([100.0, 100.0, 100.0], natr=[1.0, 4.0, 8.0], multiplier=MULTIPLIER)
        long_trade = trade()
        # One candle of duration, a two-value trade NATR window and multiplier 3.0 make the
        # distance open_rate * 0.07 * 3.0 * fraction * log10(10), and log10(10) is exactly 1.
        for stage, params in sorted(QuickAdapterV3.partial_exit_stages.items()):
            with self.subTest(stage=stage):
                fraction = params[0]
                self.assertIsInstance(fraction, float)
                distance = 100.0 * 0.07 * MULTIPLIER * fraction
                target = model.get_take_profit_target(frame, long_trade, stage)
                self.assertIsNotNone(target)
                assert_allclose(target[0], 100.0 + distance, rtol=1e-9, atol=1e-9)
                assert_allclose(target[1], distance, rtol=1e-9, atol=1e-9)
        for stage in (QuickAdapterV3._FINAL_EXIT_STAGE, 99, -1):
            with self.subTest(final_stage=stage):
                target = model.get_take_profit_target(frame, long_trade, stage)
                assert_allclose(
                    target[0],
                    100.0 * (1 + 0.07 * MULTIPLIER * QuickAdapterV3._FINAL_EXIT_STAGE_PARAMS[0]),
                    rtol=1e-9,
                    atol=1e-9,
                )
        stages = [
            model.get_take_profit_target(frame, long_trade, stage)[1]
            for stage in (0, 1, 2, QuickAdapterV3._FINAL_EXIT_STAGE)
        ]
        self.assertEqual(stages, sorted(stages))

    def test_a_short_take_profit_target_flips_the_sign(self):
        model = strategy()
        frame = candles([100.0, 100.0, 100.0], natr=[1.0, 4.0, 8.0], multiplier=MULTIPLIER)
        long_target = model.get_take_profit_target(frame, trade(), 1)
        short_target = model.get_take_profit_target(frame, trade(is_short=True), 1)
        assert_allclose(short_target[0], 2 * 100.0 - long_target[0], rtol=1e-9, atol=1e-9)
        self.assertEqual(short_target[1], long_target[1])
        self.assertLess(short_target[0], 100.0)

    def test_a_target_that_would_equal_the_open_rate_is_nudged_off_it(self):
        model = strategy()
        frame = candles([100.0, 100.0, 100.0], natr=[0.0, 1e-20, 2e-20], multiplier=MULTIPLIER)
        long_target = model.get_take_profit_target(frame, trade(), 0)
        self.assertGreater(long_target[1], 0.0)
        self.assertEqual(long_target[0], math.nextafter(100.0, math.inf))
        short_target = model.get_take_profit_target(frame, trade(is_short=True), 0)
        self.assertGreater(short_target[1], 0.0)
        self.assertEqual(short_target[0], math.nextafter(100.0, 0.0))

    def test_a_non_positive_take_profit_target_is_refused(self):
        model = strategy()
        healthy = candles([100.0, 100.0, 100.0], natr=[1.0, 4.0, 8.0], multiplier=MULTIPLIER)
        long_trade = trade()
        self.assertIsNone(
            model.get_take_profit_target(
                healthy.drop(columns=["natr_label_period_candles"]), long_trade, 0
            )
        )
        self.assertIsNone(
            model.get_take_profit_target(
                candles(
                    [100.0, 100.0, 100.0], natr=[np.nan, np.nan, np.nan], multiplier=MULTIPLIER
                ),
                long_trade,
                0,
            )
        )
        self.assertIsNone(
            model.get_take_profit_target(
                candles([100.0, 100.0, 100.0], natr=[0.0, 0.0, 0.0], multiplier=MULTIPLIER),
                long_trade,
                0,
            )
        )
        self.assertIsNone(model.get_take_profit_target(healthy, trade(open_candle_offset=2), 0))
        underwater = model.get_take_profit_target(
            candles([100.0, 100.0, 100.0], natr=[1.0, 400.0, 800.0], multiplier=MULTIPLIER),
            trade(open_rate=1e-9, is_short=True),
            QuickAdapterV3._FINAL_EXIT_STAGE,
        )
        self.assertIsNone(underwater)

    def test_a_candle_index_is_addressable_by_position_or_by_timestamp(self):
        frame = candles([100.0, 101.0, 102.0, 103.0, 104.0]).set_index("date")
        for position, candle_date in enumerate(frame.index):
            with self.subTest(candle_date=candle_date):
                positional = frame.index.get_loc(candle_date)
                self.assertEqual(positional, position)
                self.assertEqual(
                    QuickAdapterV3._normalize_candle_idx(len(frame), positional),
                    QuickAdapterV3._normalize_candle_idx(len(frame), position),
                )
                self.assertEqual(
                    QuickAdapterV3._normalize_candle_idx(len(frame), position - len(frame)),
                    position,
                )
        self.assertEqual(
            QuickAdapterV3._normalize_candle_idx(len(frame), frame.index.get_loc(frame.index[-1])),
            QuickAdapterV3._normalize_candle_idx(len(frame), -1),
        )

    def test_a_candle_index_is_clamped_to_the_frame(self):
        for length, idx, expected in (
            (0, -1, 0),
            (0, 7, 0),
            (5, -1, 4),
            (5, -5, 0),
            (5, -6, 0),
            (5, 0, 0),
            (5, 4, 4),
            (5, 99, 4),
            (1, -1, 0),
            (1, 1, 0),
        ):
            with self.subTest(length=length, idx=idx):
                self.assertEqual(QuickAdapterV3._normalize_candle_idx(length, idx), expected)

    def test_the_standard_features_scale_the_weekday_and_hour_into_the_unit_interval(self):
        model = object.__new__(QuickAdapterV3)
        dates = [
            datetime.datetime(2026, 1, 5, 0, 0, tzinfo=datetime.UTC),
            datetime.datetime(2026, 1, 11, 23, 55, tzinfo=datetime.UTC),
        ]
        frame = pd.DataFrame({"date": dates, "close": [1.0, 2.0]})
        result = model.feature_engineering_standard(frame, {})
        self.assertIs(result, frame)
        self.assertEqual(list(result.columns), ["date", "close", "%-day_of_week", "%-hour_of_day"])
        assert_allclose(result["%-day_of_week"].to_numpy(), [1.0 / 7.0, 1.0], rtol=1e-12, atol=0.0)
        assert_allclose(
            result["%-hour_of_day"].to_numpy(), [1.0 / 25.0, 0.96], rtol=1e-12, atol=0.0
        )
        for column in ("%-day_of_week", "%-hour_of_day"):
            self.assertTrue(result[column].between(0.0, 1.0).all())

    def test_the_standard_features_read_epoch_millisecond_dates(self):
        model = object.__new__(QuickAdapterV3)
        milliseconds = [
            int(datetime.datetime(2026, 1, 5, 0, 0, tzinfo=datetime.UTC).timestamp() * 1000),
            int(datetime.datetime(2026, 1, 11, 23, 55, tzinfo=datetime.UTC).timestamp() * 1000),
        ]
        frame = pd.DataFrame({"date": pd.Series(milliseconds, dtype="int64"), "close": [1.0, 2.0]})
        result = model.feature_engineering_standard(frame, {})
        assert_allclose(result["%-day_of_week"].to_numpy(), [1.0 / 7.0, 1.0], rtol=1e-12, atol=0.0)
        assert_allclose(
            result["%-hour_of_day"].to_numpy(), [1.0 / 25.0, 0.96], rtol=1e-12, atol=0.0
        )

    def test_a_malformed_date_column_is_refused(self):
        model = object.__new__(QuickAdapterV3)
        frame = pd.DataFrame({"date": pd.Series([7, 8], dtype="int64"), "close": [1.0, 2.0]})
        with self.assertRaises(ValueError) as raised:
            model.feature_engineering_standard(frame, {})
        self.assertIn("outside the expected epoch-ms range", str(raised.exception))

    def test_an_iso_string_is_recognised_only_when_it_parses(self):
        for value in ("2026-01-01", "2026-01-01T00:00:00+00:00", "2026-01-01 00:00:00"):
            with self.subTest(value=value):
                self.assertTrue(QuickAdapterV3.is_isoformat(value))
        for value in ("", "not-a-date", "2026-13-01", 20260101, None, b"2026-01-01", object()):
            with self.subTest(value=value):
                self.assertFalse(QuickAdapterV3.is_isoformat(value))

    def test_a_timedelta_is_formatted_with_its_sign(self):
        for delta, expected in (
            (datetime.timedelta(0), "0:00:00:00"),
            (datetime.timedelta(minutes=5), "0:00:05:00"),
            (datetime.timedelta(hours=1, minutes=2, seconds=3), "0:01:02:03"),
            (datetime.timedelta(days=2, hours=3), "2:03:00:00"),
            (datetime.timedelta(minutes=-5, seconds=-7), "-0:00:05:07"),
        ):
            with self.subTest(delta=delta):
                self.assertEqual(QuickAdapterV3._td_format(delta), expected)

    def test_a_timedelta_format_reports_an_unusable_pattern(self):
        self.assertEqual(
            QuickAdapterV3._td_format(datetime.timedelta(seconds=90), "{m}:{s}"), "1:30"
        )
        with self.assertRaises(ValueError) as raised:
            QuickAdapterV3._td_format(datetime.timedelta(seconds=1), "{bogus}")
        self.assertIn("Invalid pattern value '{bogus}'", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, KeyError)


if __name__ == "__main__":
    unittest.main()
