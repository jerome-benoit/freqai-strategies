"""Sample weight composition and support contracts; requires the Freqtrade QA image."""

import logging
import unittest
from typing import ClassVar

import numpy as np
import pandas as pd
from qa_support import QaTestCase
from Utils import (
    LabelWeightSupportError,
    _effective_sample_size,
    compose_label_lookahead,
    compose_sample_weights,
    compute_label_weights,
    nan_average,
    non_zero_diff,
    sanitize_and_renormalize,
    summarize_label_weight_support,
)

LOGGER = logging.getLogger("test-label-weights")
CONTEXT = "[BTC/USDT] test"


class SanitizeAndRenormalizeTest(QaTestCase):
    def _sanitize(self, values, drop_mask=None):
        return sanitize_and_renormalize(values, drop_mask, logger=LOGGER, context=CONTEXT)

    def test_the_output_always_has_unit_mean(self):
        for values in ([1.0, 1.0, 1.0], [0.1, 0.9, 5.0], [3.0, 1.0], [1e-6, 1e6]):
            with self.subTest(values=values):
                self.assertAlmostEqual(float(np.mean(self._sanitize(np.array(values)))), 1.0)

    def test_non_finite_and_non_positive_entries_become_exactly_zero(self):
        result = self._sanitize(np.array([1.0, np.nan, np.inf, -np.inf, 0.0, -2.0, 3.0]))
        np.testing.assert_array_equal(result[[1, 2, 3, 4, 5]], np.zeros(5))

    def test_the_surviving_entries_keep_their_relative_proportions(self):
        result = self._sanitize(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(result[1] / result[0]), 2.0)
        self.assertAlmostEqual(float(result[2] / result[0]), 3.0)

    def test_a_dropped_row_is_exactly_zero(self):
        result = self._sanitize(np.array([1.0, 2.0, 3.0]), np.array([False, True, False]))
        self.assertEqual(result[1], 0.0)
        self.assertGreater(result[0], 0.0)
        self.assertAlmostEqual(float(np.mean(result)), 1.0)

    def test_a_full_collapse_falls_back_to_uniform_weights_on_survivors(self):
        # The fallback is rescaled over ALL n rows, dropped ones included, so the mean stays
        # 1 and the surviving row carries the full mass: 3 here, not 1.
        result = self._sanitize(np.array([0.0, 0.0, 0.0]), np.array([True, False, True]))
        np.testing.assert_allclose(result, np.array([0.0, 3.0, 0.0]), rtol=1e-12, atol=0.0)
        self.assertAlmostEqual(float(np.mean(result)), 1.0)

    def test_a_collapse_with_no_survivor_ignores_the_mask_to_preserve_unit_mean(self):
        result = self._sanitize(np.array([0.0, 0.0]), np.array([True, True]))
        np.testing.assert_allclose(result, np.ones(2), rtol=1e-12, atol=0.0)

    def test_a_mismatched_drop_mask_is_refused(self):
        with self.assertRaisesRegex(ValueError, "drop_mask shape"):
            self._sanitize(np.array([1.0, 1.0]), np.array([True]))

    def test_a_non_boolean_drop_mask_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not boolean"):
            self._sanitize(np.array([1.0, 1.0]), np.array([1, 0]))

    def test_an_empty_input_stays_empty(self):
        self.assertEqual(self._sanitize(np.array([])).size, 0)


class EffectiveSampleSizeTest(QaTestCase):
    def test_uniform_weights_give_the_full_sample_size(self):
        self.assertAlmostEqual(_effective_sample_size(np.ones(10)), 10.0)

    def test_ess_follows_kishs_formula(self):
        weights = np.array([1.0, 1.0, 2.0, 2.0])
        expected = float(weights.sum() ** 2 / np.square(weights).sum())
        self.assertAlmostEqual(_effective_sample_size(weights), expected, places=10)

    def test_ess_never_exceeds_the_number_of_positive_entries(self):
        weights = np.array([1.0, 0.01, 0.01, 0.01, 5.0])
        positive = int((weights > 0).sum())
        self.assertLessEqual(_effective_sample_size(weights), positive)

    def test_degenerate_input_yields_zero(self):
        for weights in (
            np.array([]),
            np.zeros(4),
            np.array([-1.0, -2.0]),
            np.array([np.nan, np.inf]),
        ):
            with self.subTest(weights=weights.tolist()):
                self.assertEqual(_effective_sample_size(weights), 0.0)

    def test_non_positive_and_non_finite_entries_leave_the_rest_intact(self):
        # The filter keeps strictly positive finite entries only, so a mixed vector reports
        # the ESS of its positive part rather than zero.
        self.assertAlmostEqual(_effective_sample_size(np.array([-1.0, 2.0])), 1.0)
        self.assertAlmostEqual(_effective_sample_size(np.array([np.nan, 1.0, 1.0])), 2.0)


class ComposeSampleWeightsTest(QaTestCase):
    def _compose(self, base, labels, on_collapse="raise"):
        return compose_sample_weights(
            base, labels, logger=LOGGER, context=CONTEXT, on_collapse=on_collapse
        )

    def test_absent_label_weights_return_the_sanitized_base(self):
        base = np.array([1.0, 2.0, 3.0])
        result = self._compose(base, None)
        self.assertAlmostEqual(float(np.mean(result)), 1.0)
        np.testing.assert_allclose(result / result[0], base / base[0], rtol=1e-12, atol=0.0)

    def test_the_composition_has_unit_mean(self):
        result = self._compose(np.array([1.0, 1.0, 1.0, 1.0]), np.array([3.0, 1.0, 2.0, 1.0]))
        self.assertAlmostEqual(float(np.mean(result)), 1.0)

    def test_rows_with_a_non_positive_label_weight_are_exactly_zero(self):
        result = self._compose(np.ones(4), np.array([1.0, 0.0, -1.0, np.nan]))
        np.testing.assert_array_equal(result[[1, 2, 3]], np.zeros(3))
        self.assertGreater(result[0], 0.0)

    def test_surviving_rows_keep_the_base_to_label_product(self):
        base = np.array([1.0, 2.0])
        labels = np.array([4.0, 1.0])
        result = self._compose(base, labels)
        # The product is 4 and 2, so the second row is half the first.
        self.assertAlmostEqual(float(result[1] / result[0]), 0.5)

    def test_a_shape_mismatch_is_a_hard_failure(self):
        with self.assertRaisesRegex(ValueError, "label_weights shape"):
            self._compose(np.ones(4), np.ones(3))

    def test_every_row_dropped_is_a_support_error_not_a_bare_value_error(self):
        with self.assertRaises(LabelWeightSupportError):
            self._compose(np.ones(4), np.zeros(4))

    def test_a_collapse_on_survivors_raises_under_the_raise_policy(self):
        base = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([1.0, 1.0, np.nan, np.nan])
        with self.assertRaises(LabelWeightSupportError):
            self._compose(base, labels, on_collapse="raise")

    def test_a_collapse_falls_back_to_base_weights_under_the_fallback_policy(self):
        base = np.array([0.0, 0.0, 1.0, 1.0])
        labels = np.array([1.0, 1.0, np.nan, np.nan])
        result = self._compose(base, labels, on_collapse="fallback")
        self.assertAlmostEqual(float(np.mean(result)), 1.0)
        self.assertGreater(result[0], 0.0)
        self.assertEqual(result[2], 0.0)

    def test_the_support_error_is_a_value_error(self):
        self.assertTrue(issubclass(LabelWeightSupportError, ValueError))


class SupportSummaryTest(QaTestCase):
    def test_a_shape_mismatch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "label_weights shape"):
            summarize_label_weight_support(np.ones(4), np.ones(3))

    def test_the_summary_counts_positive_label_weights(self):
        summary = summarize_label_weight_support(np.array([1.0, 0.0, 2.0, np.nan]), np.ones(4))
        self.assertEqual(summary.total_rows, 4)
        self.assertEqual(summary.positive_label_weight_count, 2)
        self.assertAlmostEqual(summary.positive_label_weight_fraction, 0.5)

    def test_the_summary_reports_kish_ess_on_the_composed_weights(self):
        summary = summarize_label_weight_support(np.ones(4), np.array([1.0, 1.0, 1.0, 3.0]))
        self.assertAlmostEqual(
            summary.effective_sample_size,
            _effective_sample_size(np.array([1.0, 1.0, 1.0, 3.0])),
            places=12,
        )

    def test_pivot_equivalent_rows_are_the_ones_near_the_surviving_maximum(self):
        # 5.0 dominates, and 0.5 is at the 10% pivot threshold while 0.1 is not.
        summary = summarize_label_weight_support(np.array([5.0, 0.5, 0.1, 0.0]), np.ones(4))
        self.assertEqual(summary.pivot_equivalent_count, 2)

    def test_an_empty_input_reports_zeroes(self):
        summary = summarize_label_weight_support(np.array([]), np.array([]))
        self.assertEqual(summary.total_rows, 0)
        self.assertEqual(summary.positive_label_weight_count, 0)
        self.assertEqual(summary.positive_label_weight_fraction, 0.0)
        self.assertEqual(summary.effective_sample_size, 0.0)


class NanAverageTest(QaTestCase):
    def test_an_unweighted_average_ignores_non_finite_values(self):
        self.assertAlmostEqual(nan_average(np.array([1.0, np.nan, 3.0])), 2.0)

    def test_a_weighted_average_follows_the_weights(self):
        values = np.array([0.0, 10.0])
        weights = np.array([3.0, 1.0])
        self.assertAlmostEqual(nan_average(values, weights), 2.5)

    def test_empty_and_all_non_finite_inputs_yield_nan(self):
        for values in (np.array([]), np.array([np.nan, np.inf])):
            with self.subTest(values=values.tolist()):
                self.assertTrue(np.isnan(nan_average(values)))

    def test_a_shape_mismatch_yields_nan(self):
        self.assertTrue(np.isnan(nan_average(np.ones(4), np.ones(3))))

    def test_a_zero_weight_sum_yields_nan(self):
        self.assertTrue(np.isnan(nan_average(np.array([1.0, 2.0]), np.array([0.0, 0.0]))))

    def test_rows_with_a_non_finite_weight_are_excluded(self):
        result = nan_average(np.array([1.0, 100.0, 3.0]), np.array([1.0, np.nan, 1.0]))
        self.assertAlmostEqual(result, 2.0)


class NonZeroDiffTest(QaTestCase):
    def test_a_zero_difference_becomes_the_float_epsilon(self):
        result = non_zero_diff(pd.Series([1.0, 2.0]), pd.Series([1.0, 5.0]))
        self.assertEqual(result.iloc[0], np.finfo(float).eps)
        self.assertEqual(result.iloc[1], -3.0)

    def test_nan_propagates_unchanged(self):
        result = non_zero_diff(pd.Series([np.nan]), pd.Series([1.0]))
        self.assertTrue(np.isnan(result.iloc[0]))


class ComposeLabelLookaheadTest(QaTestCase):
    """The right kernel boundary, which `compose_label_lookahead` has to hold shut.

    A centered smoothing kernel needs future candles that do not exist yet, so the last
    `kernel_half_width` rows must stay unavailable until the frame grows. The bound is
    `right_edge_start = max(0, n - kernel_half_width)` and it was untested: widening it by a
    single row kept the whole suite green while changing the reported lookahead for that row
    from `n - row` to 1, on every combination the function is reachable with.
    """

    def test_the_last_kernel_half_width_rows_stay_unavailable(self):
        n, half = 300, 5
        # A known_at of 0 everywhere, so the available position equals the row itself. The
        # centered rolling max then returns `last row of the window` for every row, which
        # would leave the final rows available immediately: at row n-1 the window ends at
        # n-1, giving a lookahead of 0. The clamp holds those rows to `n` instead, so the
        # reported lookahead is exactly `n - row` and decays to 1 at the last row.
        result = compose_label_lookahead(
            pd.Series(np.zeros(n, dtype=np.int64)), half, method="gaussian", mode="zero"
        )

        # The clamped window, and only it, is pinned to `n`.
        np.testing.assert_array_equal(result.to_numpy()[n - half :], np.arange(half, 0, -1))
        self.assertEqual(int(result.iloc[-1]), 1)
        # One row earlier the rolling max already returns 5, not the 4 the clamp would give,
        # so the boundary is exactly where the clamp starts and not one row off.
        self.assertEqual(int(result.iloc[n - half - 1]), half)

    def test_a_zero_half_width_returns_the_input_untouched(self):
        values = np.array([0, 3, 1, 7], dtype=np.int64)
        result = compose_label_lookahead(pd.Series(values), 0, method="gaussian", mode="zero")
        np.testing.assert_array_equal(result.to_numpy(), values)


class MetricWeightingStrategyTest(QaTestCase):
    """The per-metric weighting strategies, which no test exercised.

    Only the `uniform` strategy was ever selected, so `weights = np.asarray(metrics[strategy])`
    never ran: replacing it with `np.ones` left the whole suite green while every configured
    swing importance silently became uniform. The amplitude strategy is reachable from a
    real config and is the alternative the shipped template documents.
    """

    METRICS: ClassVar[dict[str, list[float]]] = {
        "amplitude": [0.2, 0.7],
        "speed": [1.0, 4.0],
        "efficiency_ratio": [0.5, 0.9],
    }

    def _weights(self, strategy: str) -> np.ndarray:
        return compute_label_weights(
            10, [2, 8], self.METRICS, {"strategy": strategy}, logger=LOGGER
        )

    def test_each_metric_strategy_places_that_metric_on_the_pivots(self):
        for strategy, values in self.METRICS.items():
            with self.subTest(strategy=strategy):
                weights = self._weights(strategy)
                self.assertEqual(weights[2], values[0])
                self.assertEqual(weights[8], values[1])
                # Everything that is not a pivot carries no label weight at all.
                off_pivot = np.delete(weights, [2, 8])
                np.testing.assert_array_equal(off_pivot, np.zeros(8))

    def test_a_metric_strategy_keeps_the_relative_importance_of_the_pivots(self):
        # The part the consumer actually sees. `compose_sample_weights` is what turns these
        # into the training sample weights, so it is the ratio there that must survive a
        # uniformisation of the metric — mean, and the fact that weights are positive, hold
        # either way.
        for strategy, expected in (("amplitude", 3.5), ("speed", 4.0), ("efficiency_ratio", 1.8)):
            with self.subTest(strategy=strategy):
                weights = self._weights(strategy)
                labels = np.zeros(10)
                labels[[2, 8]] = weights[[2, 8]]
                composed = compose_sample_weights(
                    np.ones(10), labels, logger=LOGGER, context=CONTEXT
                )
                self.assertAlmostEqual(float(composed[8] / composed[2]), expected, places=9)

    def test_uniform_still_ignores_the_metrics(self):
        np.testing.assert_array_equal(
            compute_label_weights(10, [2, 8], self.METRICS, {"strategy": "uniform"}, logger=LOGGER)[
                [2, 8]
            ],
            np.ones(2),
        )


if __name__ == "__main__":
    unittest.main()
