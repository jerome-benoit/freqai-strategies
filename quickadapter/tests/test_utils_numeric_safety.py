"""Numeric safety primitives; requires the Freqtrade QA image."""

import logging
import math
import unittest

import numpy as np
import pandas as pd
from qa_support import QaTestCase
from Utils import (
    FiniteSample,
    _is_finite_value,
    finite_sample,
    is_finite_number,
    safe_divide,
    safe_log_ratio,
)

LOGGER = logging.getLogger("test-numeric-safety")


class FiniteSampleTest(QaTestCase):
    def test_the_partition_invariant_holds(self):
        for values in ([1.0, np.nan, 2.0], [np.nan] * 4, [1.0, 2.0, 3.0], []):
            with self.subTest(values=str(values)):
                sample = finite_sample(np.array(values))
                self.assertEqual(sample.dropped_count, sample.total_count - sample.finite_count)

    def test_non_finite_entries_are_dropped(self):
        sample = finite_sample(np.array([1.0, np.nan, np.inf, -np.inf, 2.0]))
        np.testing.assert_array_equal(sample.values, np.array([1.0, 2.0]))
        self.assertEqual(sample.total_count, 5)
        self.assertEqual(sample.finite_count, 2)
        self.assertEqual(sample.dropped_count, 3)

    def test_the_result_is_flattened_to_one_dimension(self):
        sample = finite_sample(np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(sample.values.ndim, 1)
        self.assertEqual(sample.total_count, 4)

    def test_positive_only_drops_zero_and_signed_zero_strictly(self):
        sample = finite_sample(np.array([1.0, 0.0, -0.0, -1.0, 2.0]), positive_only=True)
        np.testing.assert_array_equal(sample.values, np.array([1.0, 2.0]))

    def test_without_positive_only_zero_survives(self):
        sample = finite_sample(np.array([1.0, 0.0, -1.0]))
        np.testing.assert_array_equal(sample.values, np.array([1.0, 0.0, -1.0]))

    def test_integer_input_is_coerced_to_float(self):
        sample = finite_sample(np.array([1, 2, 3]))
        self.assertEqual(sample.values.dtype, np.dtype(np.float64))

    def test_the_dataclass_does_not_enforce_the_invariant_when_built_directly(self):
        # Documented escape hatch: only the factory guarantees values are finite.
        direct = FiniteSample(
            values=np.array([np.nan]), total_count=1, finite_count=1, dropped_count=0
        )
        self.assertEqual(direct.dropped_count, 0)


class SafeDivideTest(QaTestCase):
    def test_a_plain_division_is_exact(self):
        self.assertAlmostEqual(safe_divide(6.0, 3.0), 2.0)

    def test_a_zero_denominator_falls_back(self):
        self.assertTrue(math.isnan(safe_divide(1.0, 0.0, fallback=np.nan)))
        self.assertEqual(safe_divide(1.0, 0.0, fallback=-1.0), -1.0)

    def test_a_subnormal_denominator_passes_through(self):
        # Satoshi-scale price quotes divide fine; the guard is on exactly zero, not on
        # magnitude, so a 1e-8 denominator still yields its true quotient.
        self.assertAlmostEqual(safe_divide(1.0, 1e-8), 1e8)

    def test_non_finite_operands_fall_back(self):
        for numerator, denominator in ((np.nan, 1.0), (1.0, np.nan), (np.inf, 1.0), (1.0, -np.inf)):
            with self.subTest(numerator=numerator, denominator=denominator):
                self.assertTrue(math.isnan(safe_divide(numerator, denominator)))

    def test_an_overflowing_result_is_coerced_to_the_fallback(self):
        result = safe_divide(1.0, 1e-320, fallback=-1.0)
        self.assertEqual(result, -1.0)

    def test_element_wise_results_keep_their_shape(self):
        result = safe_divide(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 2.0]), fallback=0.0)
        np.testing.assert_array_equal(result, np.array([1.0, 0.0, 1.5]))

    def test_a_series_input_produces_an_indexed_series(self):
        result = safe_divide(pd.Series([1.0, 2.0], index=["a", "b"]), pd.Series([1.0, 0.0]))
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(list(result.index), ["a", "b"])
        self.assertTrue(math.isnan(result["b"]))

    def test_a_zero_dimensional_result_is_a_python_float(self):
        self.assertIsInstance(safe_divide(np.array(4.0), np.array(2.0)), float)

    def test_the_logger_records_how_many_results_were_replaced(self):
        with self.assertLogs(LOGGER, level="DEBUG") as captured:
            safe_divide(np.array([1.0, 1.0]), np.array([0.0, 2.0]), logger=LOGGER)
        self.assertTrue(any("Replaced 1 invalid" in line for line in captured.output))


class SafeLogRatioTest(QaTestCase):
    def test_a_plain_ratio_gives_its_logarithm(self):
        self.assertAlmostEqual(safe_log_ratio(4.0, 2.0), math.log(2.0), places=12)

    def test_a_ratio_of_unity_is_exactly_zero(self):
        self.assertAlmostEqual(safe_log_ratio(1e-8, 1e-8), 0.0, places=12)

    def test_a_wide_ratio_survives_where_direct_division_would_underflow(self):
        # 1e-320 / 1e300 underflows to zero, so log of the ratio would be -inf. The
        # difference of logarithms keeps it finite, which is the whole point of the form.
        result = safe_log_ratio(1e-320, 1e300, fallback=np.nan)
        self.assertTrue(np.isfinite(result))
        self.assertAlmostEqual(result, math.log(1e-320) - math.log(1e300), places=9)

    def test_a_non_positive_operand_falls_back(self):
        for numerator, denominator in ((0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)):
            with self.subTest(numerator=numerator, denominator=denominator):
                self.assertTrue(math.isnan(safe_log_ratio(numerator, denominator)))

    def test_non_finite_operands_fall_back(self):
        for numerator, denominator in ((np.nan, 1.0), (1.0, np.nan), (np.inf, 1.0)):
            with self.subTest(numerator=numerator, denominator=denominator):
                self.assertTrue(math.isnan(safe_log_ratio(numerator, denominator)))

    def test_element_wise_results_keep_their_shape(self):
        result = safe_log_ratio(
            np.array([4.0, 1.0, 0.0]), np.array([2.0, 0.0, 2.0]), fallback=np.nan
        )
        self.assertAlmostEqual(result[0], math.log(2.0), places=12)
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))

    def test_a_series_input_produces_an_indexed_series(self):
        result = safe_log_ratio(pd.Series([4.0], index=["a"]), pd.Series([2.0]))
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(list(result.index), ["a"])


class FinitenessTest(QaTestCase):
    def test_finite_numbers_are_accepted(self):
        for value in (1, -1, 0, 1.5, np.int64(3), np.float64(2.5)):
            with self.subTest(value=repr(value)):
                self.assertTrue(is_finite_number(value))

    def test_booleans_are_rejected_despite_being_integers(self):
        for value in (True, False, np.bool_(True)):
            with self.subTest(value=repr(value)):
                self.assertFalse(is_finite_number(value))

    def test_non_finite_numbers_are_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=repr(value)):
                self.assertFalse(is_finite_number(value))

    def test_non_numeric_values_are_rejected(self):
        for value in ("1.0", None, object(), [1.0], {"a": 1}):
            with self.subTest(value=repr(value)):
                self.assertFalse(is_finite_number(value))

    def test_a_python_int_beyond_the_numpy_range_is_still_finite(self):
        # np.isfinite raises above 2**64, so the int branch must short-circuit.
        huge = 2**70
        self.assertTrue(is_finite_number(huge))
        self.assertTrue(_is_finite_value(huge))

    def test_a_multi_element_array_is_treated_as_non_finite(self):
        # np.isfinite returns a non-scalar here, and bool() on it raises, which the
        # guard turns into "not finite" rather than letting it escape.
        self.assertFalse(_is_finite_value(np.array([1.0, 2.0])))

    def test_a_single_element_array_is_a_de_facto_scalar(self):
        # bool() of a one-element array does not raise, so it is reported as finite.
        # Asserted because it is the boundary the guard's rationale glosses over.
        self.assertTrue(_is_finite_value(np.array([1.0])))

    def test_an_empty_array_is_treated_as_non_finite(self):
        self.assertFalse(_is_finite_value(np.array([])))


if __name__ == "__main__":
    unittest.main()
