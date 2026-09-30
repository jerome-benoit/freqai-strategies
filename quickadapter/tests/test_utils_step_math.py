"""Step and period algebra; requires the Freqtrade QA image."""

import unittest

import numpy as np
from qa_support import QaTestCase
from Utils import (
    ceil_to_step,
    floor_to_step,
    get_min_max_label_period_candles,
    largest_divisor_to_step,
    round_to_step,
)


class StepTest(QaTestCase):
    def test_round_to_step_snaps_to_the_nearest_multiple(self):
        # 8 and 4 are not multiples of 3; the nearest ones are 9 and 3 respectively.
        for value, expected in ((7, 6), (8, 9), (5, 6), (4, 3), (0, 0), (9, 9)):
            with self.subTest(value=value):
                self.assertEqual(round_to_step(value, 3), expected)
                self.assertEqual(round_to_step(value, 3) % 3, 0)

    def test_floor_and_ceil_bracket_the_value(self):
        for value in (4, 5, 6, 7, 8):
            with self.subTest(value=value):
                low = floor_to_step(value, 3)
                high = ceil_to_step(value, 3)
                self.assertLessEqual(low, value)
                self.assertGreaterEqual(high, value)
                self.assertLessEqual(value - low, 3)
                self.assertLessEqual(high - value, 3)

    def test_an_exact_multiple_is_its_own_floor_and_ceiling(self):
        self.assertEqual(floor_to_step(9, 3), 9)
        self.assertEqual(ceil_to_step(9, 3), 9)
        self.assertEqual(round_to_step(9, 3), 9)

    def test_a_non_positive_step_is_refused(self):
        for step in (0, -1):
            with self.subTest(step=step):
                for function in (round_to_step, ceil_to_step, floor_to_step):
                    with self.assertRaises(ValueError):
                        function(5, step)

    def test_a_non_numeric_value_is_refused(self):
        for value in ("5", None, np.nan):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                round_to_step(value, 3)

    def test_an_exact_half_rounds_to_even(self):
        # The step algebra uses Python's round, which is half-to-EVEN, not half-up:
        # 2.5 goes to 2 and 3.5 to 4. Pinned because half-up is the common assumption.
        self.assertEqual(round_to_step(2.5, 1), 2)
        self.assertEqual(round_to_step(3.5, 1), 4)
        self.assertEqual(round_to_step(0.5, 1), 0)
        self.assertEqual(round_to_step(1.5, 1), 2)


class LargestDivisorTest(QaTestCase):
    def test_it_returns_the_largest_multiple_of_the_step_that_divides_the_integer(self):
        # 12 is a multiple of 4, so 12 comes back; 60 is a multiple of both 3 and 4.
        for integer, step, expected in ((12, 4, 12), (12, 12, 12), (60, 3, 60), (60, 4, 60)):
            with self.subTest(integer=integer, step=step):
                self.assertEqual(largest_divisor_to_step(integer, step), expected)

    def test_it_returns_none_when_no_multiple_of_the_step_divides_the_integer(self):
        # 12 has divisors 1, 2, 3, 4, 6, 12; none of those is a multiple of 5.
        for integer, step in ((12, 5), (13, 5), (60, 7)):
            with self.subTest(integer=integer, step=step):
                self.assertIsNone(largest_divisor_to_step(integer, step))

    def test_a_step_of_one_always_yields_the_integer_itself(self):
        self.assertEqual(largest_divisor_to_step(12, 1), 12)
        self.assertEqual(largest_divisor_to_step(13, 1), 13)

    def test_a_non_positive_or_non_integer_argument_is_refused(self):
        for integer, step in ((0, 5), (-3, 5), (12, 0), (12, -2), (1.5, 5), (12, 2.5)):
            with self.subTest(integer=integer, step=step), self.assertRaises(ValueError):
                largest_divisor_to_step(integer, step)


class LabelPeriodRangeTest(QaTestCase):
    def _range(self, fit_live, step, low, high, **kwargs):
        return get_min_max_label_period_candles(fit_live, step, low, high, **kwargs)

    def test_a_range_the_budget_can_carry_keeps_both_endpoints(self):
        # A horizon of 300/3 = 100 leaves room for the requested 12..24 on a step of 3.
        self.assertEqual(self._range(300, 3, 12, 24), (12, 24, 3))

    def test_endpoints_are_snapped_onto_the_step(self):
        # With a 150-candle budget and a step of 7, 10 rounds up to 14 and 40 down to 35.
        self.assertEqual(self._range(150, 7, 10, 40), (14, 35, 7))

    def test_the_horizon_cap_shrinks_the_maximum_below_the_requested_one(self):
        # A 12-candle budget allows a horizon of ceil(12/3) = 4, floored to 3 on the step,
        # so the requested maximum of 24 is capped to 3 and both endpoints collapse.
        self.assertEqual(self._range(12, 3, 12, 24), (3, 3, 1))

    def test_a_reversed_range_is_refused(self):
        with self.assertRaisesRegex(ValueError, "min must be <= max"):
            self._range(300, 3, 24, 12)

    def test_a_range_narrower_than_the_step_collapses_to_the_step_of_one(self):
        low, high, step = self._range(300, 3, 12, 13)
        self.assertEqual(step, 1)
        self.assertEqual((low, high), (12, 13))


if __name__ == "__main__":
    unittest.main()
