"""Final take-profit state machine contracts; requires the Freqtrade QA image."""

import datetime
import unittest

import numpy as np
from qa_support import QaTestCase
from QuickAdapterV3 import QuickAdapterV3

LONG = QuickAdapterV3._TRADE_LONG
SHORT = QuickAdapterV3._TRADE_SHORT
TIMEFRAME = "5m"
CANDLE = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.UTC)


def later(minutes: int) -> datetime.datetime:
    return CANDLE + datetime.timedelta(minutes=minutes)


def state(**overrides):
    """Build a valid final take-profit state, then apply the caller's overrides."""
    base = QuickAdapterV3._build_final_take_profit_state(
        exit_stage=2,
        trade_direction=LONG,
        current_rate=100.0,
        take_profit_distance=10.0,
        retracement_fraction=0.5,
        candle_date=CANDLE,
        timeframe=TIMEFRAME,
    )
    assert base is not None
    base.update(overrides)
    return base


class CandleDateTest(QaTestCase):
    def test_an_iso_string_is_parsed_as_utc(self):
        self.assertEqual(QuickAdapterV3._as_utc_candle_date("2026-01-01T12:00:00+00:00"), CANDLE)

    def test_an_offset_timestamp_is_converted_to_utc(self):
        offset = datetime.datetime(
            2026, 1, 1, 14, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2))
        )
        self.assertEqual(QuickAdapterV3._as_utc_candle_date(offset), CANDLE)

    def test_a_naive_timestamp_is_refused(self):
        self.assertIsNone(QuickAdapterV3._as_utc_candle_date(datetime.datetime(2026, 1, 1, 12, 0)))

    def test_a_malformed_string_is_refused(self):
        self.assertIsNone(QuickAdapterV3._as_utc_candle_date("not-a-date"))

    def test_a_non_timestamp_value_is_refused(self):
        for value in (None, 42, object()):
            with self.subTest(value=type(value).__name__):
                self.assertIsNone(QuickAdapterV3._as_utc_candle_date(value))

    def test_alignment_requires_an_exact_timeframe_boundary(self):
        self.assertTrue(QuickAdapterV3._is_candle_date_aligned(CANDLE, TIMEFRAME))
        self.assertFalse(
            QuickAdapterV3._is_candle_date_aligned(
                CANDLE + datetime.timedelta(minutes=1), TIMEFRAME
            )
        )

    def test_alignment_refuses_a_missing_date_or_timeframe(self):
        for candle, timeframe in ((None, TIMEFRAME), (CANDLE, ""), (CANDLE, None), (CANDLE, 5)):
            with self.subTest(candle=candle, timeframe=timeframe):
                self.assertFalse(QuickAdapterV3._is_candle_date_aligned(candle, timeframe))


class RetracementDistanceTest(QaTestCase):
    def _normalize(self, best_rate, retracement, direction):
        return QuickAdapterV3._normalize_final_take_profit_retracement_distance(
            best_rate=best_rate, retracement_distance=retracement, trade_direction=direction
        )

    def test_a_long_boundary_sits_below_the_best_rate(self):
        self.assertEqual(self._normalize(100.0, 5.0, LONG), 5.0)

    def test_a_short_boundary_sits_above_the_best_rate(self):
        self.assertEqual(self._normalize(100.0, 5.0, SHORT), 5.0)

    def test_a_zero_distance_is_nudged_to_one_ulp(self):
        distance = self._normalize(100.0, 0.0, LONG)
        self.assertGreater(distance, 0.0)
        self.assertEqual(100.0 - distance, float(np.nextafter(100.0, 0.0)))

    def test_a_distance_wider_than_the_best_rate_is_refused(self):
        self.assertIsNone(self._normalize(100.0, 150.0, LONG))

    def test_invalid_inputs_are_refused(self):
        for best_rate, retracement, direction in (
            (np.nan, 1.0, LONG),
            (0.0, 1.0, LONG),
            (-1.0, 1.0, LONG),
            (100.0, np.inf, LONG),
            (100.0, -1.0, LONG),
            (100.0, 1.0, "sideways"),
        ):
            with self.subTest(best=best_rate, distance=retracement, direction=direction):
                self.assertIsNone(self._normalize(best_rate, retracement, direction))


class StateBuilderTest(QaTestCase):
    def _build(self, **overrides):
        arguments = {
            "exit_stage": 2,
            "trade_direction": LONG,
            "current_rate": 100.0,
            "take_profit_distance": 10.0,
            "retracement_fraction": 0.5,
            "candle_date": CANDLE,
            "timeframe": TIMEFRAME,
        }
        arguments.update(overrides)
        return QuickAdapterV3._build_final_take_profit_state(**arguments)

    def test_a_valid_input_produces_a_usable_state(self):
        built = self._build()
        self.assertIsNotNone(built)
        self.assertEqual(built["version"], QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_VERSION)
        self.assertEqual(built["exit_stage"], 2)
        self.assertEqual(built["best_rate"], 100.0)
        self.assertEqual(built["retracement_distance"], 5.0)
        self.assertIsNone(built["trigger_candle_date"])

    def test_the_retracement_is_the_fraction_of_the_take_profit_distance(self):
        for fraction in (0.1, 0.5, 1.0):
            with self.subTest(fraction=fraction):
                self.assertAlmostEqual(
                    self._build(retracement_fraction=fraction)["retracement_distance"],
                    10.0 * fraction,
                )

    def test_a_boolean_exit_stage_is_refused(self):
        self.assertIsNone(self._build(exit_stage=True))

    def test_a_negative_exit_stage_is_refused(self):
        self.assertIsNone(self._build(exit_stage=-1))

    def test_a_fraction_outside_the_half_open_range_is_refused(self):
        for fraction in (0.0, -0.1, 1.0001, np.nan):
            with self.subTest(fraction=fraction):
                self.assertIsNone(self._build(retracement_fraction=fraction))

    def test_a_misaligned_candle_date_is_refused(self):
        self.assertIsNone(self._build(candle_date=CANDLE + datetime.timedelta(minutes=1)))
        self.assertIsNone(self._build(candle_date=None))

    def test_a_non_positive_rate_or_distance_is_refused(self):
        for rate, distance in ((0.0, 10.0), (-1.0, 10.0), (100.0, 0.0), (100.0, -1.0)):
            with self.subTest(rate=rate, distance=distance):
                self.assertIsNone(self._build(current_rate=rate, take_profit_distance=distance))

    def test_an_unknown_direction_is_refused(self):
        self.assertIsNone(self._build(trade_direction="sideways"))


class BoundaryTest(QaTestCase):
    def test_a_long_boundary_is_below_the_best_rate(self):
        self.assertEqual(QuickAdapterV3._final_take_profit_boundary(state()), 95.0)

    def test_a_short_boundary_is_above_the_best_rate(self):
        short = state(trade_direction=SHORT, best_rate=100.0, retracement_distance=5.0)
        self.assertEqual(QuickAdapterV3._final_take_profit_boundary(short), 105.0)

    def test_validity_requires_a_boundary_on_the_favourable_side(self):
        self.assertTrue(QuickAdapterV3._is_valid_final_take_profit_boundary(state()))
        self.assertFalse(
            QuickAdapterV3._is_valid_final_take_profit_boundary(
                state(best_rate=100.0, retracement_distance=150.0)
            )
        )

    def test_a_non_finite_best_rate_is_refused(self):
        self.assertFalse(
            QuickAdapterV3._is_valid_final_take_profit_boundary(state(best_rate=np.nan))
        )


class AdvanceStateTest(QaTestCase):
    def test_a_long_best_rate_ratchets_upwards_only(self):
        current = state()
        minute = 5
        for rate, expected in ((99.0, 100.0), (101.0, 101.0), (98.0, 101.0)):
            with self.subTest(rate=rate):
                _, _, changed = QuickAdapterV3._advance_final_take_profit_state(
                    current, current_rate=rate, candle_date=later(minute)
                )
                self.assertTrue(changed)
                self.assertEqual(current["best_rate"], expected)
                minute += 5

    def test_a_short_best_rate_ratchets_downwards_only(self):
        current = state(trade_direction=SHORT, best_rate=100.0, retracement_distance=5.0)
        minute = 5
        for rate, expected in ((101.0, 100.0), (99.0, 99.0), (102.0, 99.0)):
            with self.subTest(rate=rate):
                QuickAdapterV3._advance_final_take_profit_state(
                    current, current_rate=rate, candle_date=later(minute)
                )
                self.assertEqual(current["best_rate"], expected)
                minute += 5

    def test_the_retracement_distance_stays_absolute_while_the_boundary_travels(self):
        current = state()
        QuickAdapterV3._advance_final_take_profit_state(
            current, current_rate=110.0, candle_date=later(5)
        )
        self.assertEqual(current["retracement_distance"], 5.0)
        self.assertEqual(QuickAdapterV3._final_take_profit_boundary(current), 105.0)

    def test_a_repeat_on_the_same_candle_is_a_no_op(self):
        current = state()
        before = QuickAdapterV3._final_take_profit_boundary(current)
        boundary, should_exit, changed = QuickAdapterV3._advance_final_take_profit_state(
            current, current_rate=50.0, candle_date=CANDLE
        )
        self.assertEqual(boundary, before)
        self.assertFalse(should_exit)
        self.assertFalse(changed)
        self.assertEqual(current["best_rate"], 100.0)

    def test_a_candle_going_backwards_is_refused(self):
        current = state()
        QuickAdapterV3._advance_final_take_profit_state(
            current, current_rate=110.0, candle_date=later(5)
        )
        _, should_exit, changed = QuickAdapterV3._advance_final_take_profit_state(
            current, current_rate=120.0, candle_date=CANDLE
        )
        self.assertFalse(should_exit)
        self.assertFalse(changed)
        self.assertEqual(current["best_rate"], 110.0)

    def test_an_invalid_rate_leaves_the_state_untouched(self):
        for rate in (0.0, -1.0, np.nan):
            with self.subTest(rate=rate):
                current = state()
                _, should_exit, changed = QuickAdapterV3._advance_final_take_profit_state(
                    current, current_rate=rate, candle_date=later(5)
                )
                self.assertFalse(should_exit)
                self.assertFalse(changed)
                self.assertEqual(current["best_rate"], 100.0)

    def test_a_long_exit_triggers_at_or_below_the_boundary(self):
        for rate, expected in ((95.0, True), (94.0, True), (95.1, False)):
            with self.subTest(rate=rate):
                current = state()
                _, should_exit, _ = QuickAdapterV3._advance_final_take_profit_state(
                    current, current_rate=rate, candle_date=later(5)
                )
                self.assertIs(should_exit, expected)

    def test_a_short_exit_triggers_at_or_above_the_boundary(self):
        for rate, expected in ((105.0, True), (106.0, True), (104.9, False)):
            with self.subTest(rate=rate):
                current = state(trade_direction=SHORT, best_rate=100.0, retracement_distance=5.0)
                _, should_exit, _ = QuickAdapterV3._advance_final_take_profit_state(
                    current, current_rate=rate, candle_date=later(5)
                )
                self.assertIs(should_exit, expected)

    def test_a_trigger_latches_and_refuses_to_re_arm(self):
        current = state()
        _, should_exit, _ = QuickAdapterV3._advance_final_take_profit_state(
            current, current_rate=90.0, candle_date=later(5)
        )
        self.assertTrue(should_exit)
        self.assertIsNotNone(current["trigger_candle_date"])

        boundary, second_exit, changed = QuickAdapterV3._advance_final_take_profit_state(
            current, current_rate=120.0, candle_date=later(10)
        )
        self.assertTrue(second_exit)
        self.assertFalse(changed)
        self.assertEqual(current["best_rate"], 100.0)
        self.assertEqual(QuickAdapterV3._final_take_profit_boundary(current), boundary)

    def test_the_boundary_candle_advances_only_when_the_best_rate_moves(self):
        current = state()
        first_boundary_candle = current["boundary_candle_date"]
        QuickAdapterV3._advance_final_take_profit_state(
            current, current_rate=99.0, candle_date=later(5)
        )
        self.assertEqual(current["boundary_candle_date"], first_boundary_candle)
        QuickAdapterV3._advance_final_take_profit_state(
            current, current_rate=105.0, candle_date=later(10)
        )
        self.assertNotEqual(current["boundary_candle_date"], first_boundary_candle)


if __name__ == "__main__":
    unittest.main()
