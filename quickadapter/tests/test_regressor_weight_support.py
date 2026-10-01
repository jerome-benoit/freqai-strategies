"""Training weight support gating; requires the Freqtrade QA image."""

import logging
import unittest

import numpy as np
from qa_support import QaTestCase
from Utils import LabelWeightSupportError

from quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 import QuickAdapterRegressorV3

RAISE = QuickAdapterRegressorV3._SUPPORT_POLICY_RAISE
FALLBACK = QuickAdapterRegressorV3._SUPPORT_POLICY_FALLBACK
NONE_STRATEGY = "none"
NON_NONE_STRATEGY = "zigzag"
CONTEXT = "[BTC/USDT] test"


def config(**overrides):
    base = {
        "support_policy": RAISE,
        "min_pivot_equivalent_count": 1,
        "min_positive_label_weight_fraction": 0.0,
        "min_effective_sample_size": 0.0,
        "strategy": NONE_STRATEGY,
    }
    base.update(overrides)
    return base


class EvalWeightsTest(QaTestCase):
    """Eval splits bypass support gating: routine test splits trip the thresholds by construction."""

    def _compose(self, base, labels):
        return QuickAdapterRegressorV3._compose_eval_weights(base, labels, context=CONTEXT)

    def test_eval_weights_are_proportional_to_the_label_weights(self):
        # Asserting only the unit mean is satisfied by any positive vector, including one
        # that has thrown the label weights away and binarised the positives to all-ones —
        # which has mean 1.0 too and shipped green. The uneven fixture is the whole point:
        # measured, the labels [3, 1, 2, 1] give [1.714, 0.571, 1.143, 0.571] while the
        # binarised version gives [1, 1, 1, 1].
        result = self._compose(np.ones(4), np.array([3.0, 1.0, 2.0, 1.0]))

        self.assertAlmostEqual(float(np.mean(result)), 1.0)
        np.testing.assert_allclose(
            result / result[0], np.array([1.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0]), rtol=1e-12
        )
        self.assertFalse(np.allclose(result, np.ones(4)), "the labels were discarded")

    def test_eval_weights_ignore_the_support_policy_entirely(self):
        # Every training threshold is unreachable here, and nothing is raised.
        result = self._compose(np.ones(4), np.array([1.0, 1.0, 1.0, 1.0]))
        self.assertEqual(result.size, 4)

    def test_an_all_dropped_composition_falls_back_to_base_weights(self):
        base = np.array([1.0, 2.0, 3.0, 4.0])
        with self.assertLogs(
            "quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3", "WARNING"
        ):
            result = self._compose(base, np.zeros(4))
        self.assertAlmostEqual(float(np.mean(result)), 1.0)

    def test_a_shape_mismatch_is_a_hard_failure_not_a_support_condition(self):
        with self.assertRaisesRegex(ValueError, "label_weights shape"):
            self._compose(np.ones(4), np.ones(3))

    def test_the_fallback_preserves_the_base_proportions(self):
        base = np.array([1.0, 1.0, 3.0])
        with self.assertLogs(
            "quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3", "WARNING"
        ):
            result = self._compose(base, np.zeros(3))
        self.assertAlmostEqual(float(result[2] / result[0]), 3.0)


class SupportPolicyTest(QaTestCase):
    def _apply(self, policy, reasons=("because",)):
        return QuickAdapterRegressorV3._apply_support_policy(
            np.array([1.0, 3.0]), context=CONTEXT, policy=policy, reasons=list(reasons)
        )

    def test_the_raise_policy_aborts_with_the_reasons(self):
        with self.assertRaisesRegex(ValueError, r"label weighting support failed \(because\)"):
            self._apply(RAISE)

    def test_the_fallback_policy_returns_sanitized_base_weights(self):
        with self.assertLogs(
            "quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3", "WARNING"
        ):
            result = self._apply(FALLBACK)
        self.assertAlmostEqual(float(np.mean(result)), 1.0)
        self.assertAlmostEqual(float(result[1] / result[0]), 3.0)

    def test_every_reason_is_reported_not_just_the_first(self):
        with self.assertRaisesRegex(ValueError, "first.*second.*third"):
            self._apply(RAISE, reasons=("first", "second", "third"))


class EnforceSupportTest(QaTestCase):
    def _enforce(self, label_weights, sample_weights, **config_overrides):
        return QuickAdapterRegressorV3._enforce_train_weight_support(
            np.ones(len(label_weights)),
            label_weights,
            sample_weights,
            config(**config_overrides),
            context=CONTEXT,
        )

    def test_weights_passing_every_threshold_are_returned_unchanged(self):
        sample = np.array([1.0, 1.0, 1.0, 1.0])
        result = self._enforce(np.ones(4), sample, min_effective_sample_size=4.0)
        np.testing.assert_allclose(result, sample, rtol=1e-12, atol=0.0)

    def test_a_threshold_is_inclusive_at_the_boundary(self):
        # Four equal positive label weights give four pivots, a positive fraction of
        # exactly 1.0 and an ESS of exactly 4.0, so all three pass at their minimum.
        result = self._enforce(
            np.ones(4),
            np.ones(4),
            min_pivot_equivalent_count=4,
            min_positive_label_weight_fraction=1.0,
            min_effective_sample_size=4.0,
        )
        self.assertEqual(result.size, 4)

    def test_one_ulp_below_a_threshold_is_refused(self):
        with self.assertRaisesRegex(ValueError, "effective_sample_size"):
            self._enforce(np.ones(4), np.ones(4), min_effective_sample_size=4.5)

    def test_too_few_pivots_is_refused(self):
        with self.assertRaisesRegex(ValueError, "pivot_equivalent_count"):
            self._enforce(np.ones(4), np.ones(4), min_pivot_equivalent_count=5)

    def test_too_small_a_positive_fraction_is_refused(self):
        # Two of four rows carry a positive label weight, and only those two are pivots.
        with self.assertRaisesRegex(ValueError, "positive_label_weight_fraction"):
            self._enforce(
                np.array([1.0, 0.0, 1.0, 0.0]),
                np.ones(4),
                min_positive_label_weight_fraction=0.75,
            )

    def test_all_three_reasons_accumulate_into_one_failure(self):
        with self.assertRaisesRegex(
            ValueError, "pivot_equivalent_count.*positive_label_weight_fraction"
        ):
            self._enforce(
                np.array([1.0, 0.0, 1.0, 0.0]),
                np.ones(4),
                min_pivot_equivalent_count=3,
                min_positive_label_weight_fraction=0.75,
                min_effective_sample_size=4.5,
            )

    def test_the_fallback_policy_turns_a_support_failure_into_base_weights(self):
        with self.assertLogs(
            "quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3", "WARNING"
        ) as captured:
            result = self._enforce(
                np.ones(4),
                np.ones(4),
                support_policy=FALLBACK,
                min_effective_sample_size=4.5,
            )
        self.assertAlmostEqual(float(np.mean(result)), 1.0)
        # assertLogs alone only proves THAT a warning fired. Under `fallback` the operator
        # has no exception to read, so the warning is the whole diagnostic: it must name the
        # threshold that actually failed, not merely announce that something did.
        self.assertEqual(len(captured.records), 1)
        self.assertIn("effective_sample_size=4 < min_effective_sample_size=4.5", captured.output[0])
        self.assertNotIn("pivot_equivalent_count=0", captured.output[0])


class ComposeTrainWeightsTest(QaTestCase):
    def _compose(self, base, labels, **config_overrides):
        return QuickAdapterRegressorV3._compose_train_weights_with_support(
            base, labels, config(**config_overrides), context=CONTEXT
        )

    def test_a_none_strategy_with_no_label_weights_returns_base_weights(self):
        result = self._compose(np.array([1.0, 3.0]), None)
        self.assertAlmostEqual(float(result[1] / result[0]), 3.0)

    def test_a_configured_strategy_with_no_label_weights_is_gated_by_the_policy(self):
        # A non-"none" strategy that produced no pivots must not silently return base
        # weights: under "raise" it aborts, naming the strategy and the absence of pivots.
        with self.assertRaisesRegex(ValueError, "no label weights available"):
            self._compose(np.ones(4), None, strategy=NON_NONE_STRATEGY)

    def test_the_fallback_policy_handles_a_missing_label_weighting_gracefully(self):
        with self.assertLogs(
            "quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3", "WARNING"
        ):
            result = self._compose(
                np.array([1.0, 3.0]), None, strategy=NON_NONE_STRATEGY, support_policy=FALLBACK
            )
        self.assertAlmostEqual(float(result[1] / result[0]), 3.0)

    def test_a_composed_set_that_passes_support_is_returned(self):
        result = self._compose(
            np.ones(4), np.array([1.0, 1.0, 1.0, 1.0]), min_effective_sample_size=4.0
        )
        self.assertEqual(result.size, 4)

    def test_a_composed_set_that_breaches_support_routes_through_the_policy(self):
        with self.assertRaisesRegex(ValueError, "effective_sample_size"):
            self._compose(np.ones(4), np.ones(4), min_effective_sample_size=4.5)

    def test_an_all_dropped_composition_routes_through_the_policy(self):
        with self.assertRaisesRegex(ValueError, "all rows dropped"):
            self._compose(np.ones(4), np.zeros(4))

    def test_an_all_dropped_composition_falls_back_under_the_fallback_policy(self):
        with self.assertLogs(
            "quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3", "WARNING"
        ):
            result = self._compose(np.ones(4), np.zeros(4), support_policy=FALLBACK)
        self.assertAlmostEqual(float(np.mean(result)), 1.0)

    def test_the_support_error_is_what_the_train_path_catches(self):
        self.assertTrue(issubclass(LabelWeightSupportError, ValueError))
        with self.assertRaises(LabelWeightSupportError):
            from Utils import compose_sample_weights

            compose_sample_weights(
                np.ones(4), np.zeros(4), logger=logging.getLogger("t"), context=CONTEXT
            )


if __name__ == "__main__":
    unittest.main()
