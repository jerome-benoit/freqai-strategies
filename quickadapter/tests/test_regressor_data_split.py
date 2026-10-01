"""Causal purge and chronological split contracts; requires the Freqtrade QA image."""

import unittest

import numpy as np
import pandas as pd
from qa_support import QaTestCase, model_config, temporary_directory
from Utils import (
    LABEL_COLUMNS,
    label_known_at_lookahead_column_name,
    label_weight_known_at_lookahead_column_name,
)

from quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 import (
    QuickAdapterRegressorV3,
    SampleWeightInputs,
)

PAIR = "BTC/USDT"
LABEL = LABEL_COLUMNS[0]
LOOKAHEAD = label_known_at_lookahead_column_name(LABEL)
WEIGHT_LOOKAHEAD = label_weight_known_at_lookahead_column_name(LABEL)
WEIGHTING_CONFIG = {
    "support_policy": "fallback",
    "min_pivot_equivalent_count": 1,
    "min_positive_label_weight_fraction": 0.0,
    "min_effective_sample_size": 1.0,
}


def regressor(**attrs: object) -> QuickAdapterRegressorV3:
    """Build a regressor without running __init__, injecting only what the code under test reads."""
    model = object.__new__(QuickAdapterRegressorV3)
    model.ft_params = attrs.pop("ft_params", {"label_period_candles": 1})
    model.data_split_parameters = attrs.pop(
        "data_split_parameters", {"test_size": 0, "shuffle": False}
    )
    model._causal_mode = attrs.pop("_causal_mode", True)
    model._label_horizon_candles = lambda pair=None: 0
    for name, value in attrs.items():
        setattr(model, name, value)
    return model


def frame(index: pd.Index, **columns: object) -> pd.DataFrame:
    return pd.DataFrame({**columns}, index=index)


class IndexAlignmentTest(QaTestCase):
    def test_non_unique_unfiltered_index_is_refused(self):
        duplicated = pd.Index([0, 1, 1])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            QuickAdapterRegressorV3._validate_index_alignment(frame(duplicated), frame(duplicated))

    def test_filtered_index_outside_unfiltered_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must be a subset"):
            QuickAdapterRegressorV3._validate_index_alignment(
                frame(pd.Index([0, 9])), frame(pd.Index([0, 1, 2]))
            )

    def test_row_positions_are_the_unfiltered_row_ordinals(self):
        unfiltered = frame(pd.Index([10, 11, 12, 13]))
        positions = QuickAdapterRegressorV3._row_positions(frame(pd.Index([12, 10])), unfiltered)
        np.testing.assert_array_equal(positions.to_numpy(), np.array([2, 0]))
        self.assertEqual(list(positions.index), [12, 10])

    def test_row_positions_preserve_the_filtered_row_order(self):
        unfiltered = frame(pd.Index([0, 1, 2, 3]))
        positions = QuickAdapterRegressorV3._row_positions(frame(pd.Index([3, 1])), unfiltered)
        self.assertEqual(list(positions.index), [3, 1])


class KnownAtLookaheadTest(QaTestCase):
    def test_returns_none_when_no_lookahead_column_is_emitted(self):
        unfiltered = frame(pd.Index([0, 1]))
        self.assertIsNone(QuickAdapterRegressorV3._known_at_lookahead(unfiltered, unfiltered))

    def test_single_column_is_returned_as_int64(self):
        unfiltered = frame(pd.Index([0, 1]), **{LOOKAHEAD: [3, 7]})
        result = QuickAdapterRegressorV3._known_at_lookahead(unfiltered, unfiltered)
        self.assertEqual(result.dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(result.to_numpy(), np.array([3, 7]))

    def test_weight_availability_lags_the_label_by_a_row_wise_max(self):
        unfiltered = frame(pd.Index([0, 1]), **{LOOKAHEAD: [3, 1], WEIGHT_LOOKAHEAD: [9, 1]})
        result = QuickAdapterRegressorV3._known_at_lookahead(unfiltered, unfiltered)
        np.testing.assert_array_equal(result.to_numpy(), np.array([9, 1]))

    def test_a_column_holding_nan_is_skipped_entirely(self):
        unfiltered = frame(
            pd.Index([0, 1]),
            **{LOOKAHEAD: [3.0, np.nan], WEIGHT_LOOKAHEAD: [9, 9]},
        )
        result = QuickAdapterRegressorV3._known_at_lookahead(unfiltered, unfiltered)
        np.testing.assert_array_equal(result.to_numpy(), np.array([9, 9]))

    def test_result_is_aligned_to_the_filtered_index(self):
        unfiltered = frame(pd.Index([0, 1, 2]), **{LOOKAHEAD: [5, 6, 7]})
        result = QuickAdapterRegressorV3._known_at_lookahead(frame(pd.Index([2, 0])), unfiltered)
        self.assertEqual(list(result.index), [2, 0])
        np.testing.assert_array_equal(result.to_numpy(), np.array([7, 5]))


class FilterTrainByMaskTest(QaTestCase):
    def setUp(self):
        super().setUp()
        self.features = frame(pd.Index([0, 1, 2]), f=[1.0, 2.0, 3.0])
        self.labels = frame(pd.Index([0, 1, 2]), y=[10.0, 20.0, 30.0])
        self.weights = np.array([1.0, 2.0, 3.0])
        self.label_weights = np.array([0.1, 0.2, 0.3])

    def test_rows_are_sliced_in_lockstep_across_all_four_carriers(self):
        mask = np.array([True, False, True])
        features, labels, weights, label_weights = QuickAdapterRegressorV3._filter_train_by_mask(
            self.features, self.labels, self.weights, mask, "ctx", self.label_weights
        )
        self.assertEqual(list(features.index), [0, 2])
        self.assertEqual(list(labels.index), [0, 2])
        np.testing.assert_array_equal(weights, np.array([1.0, 3.0]))
        np.testing.assert_array_equal(label_weights, np.array([0.1, 0.3]))

    def test_absent_label_weights_stay_absent(self):
        _, _, _, label_weights = QuickAdapterRegressorV3._filter_train_by_mask(
            self.features, self.labels, self.weights, np.array([True, True, True]), "ctx"
        )
        self.assertIsNone(label_weights)

    def test_an_entirely_removed_training_set_is_refused(self):
        with self.assertRaisesRegex(ValueError, "causal guard removed all train rows"):
            QuickAdapterRegressorV3._filter_train_by_mask(
                self.features, self.labels, self.weights, np.array([False, False, False]), "ctx"
            )

    def test_a_single_surviving_row_is_enough(self):
        features, _, weights, _ = QuickAdapterRegressorV3._filter_train_by_mask(
            self.features, self.labels, self.weights, np.array([False, True, False]), "ctx"
        )
        self.assertEqual(list(features.index), [1])
        np.testing.assert_array_equal(weights, np.array([2.0]))


class ShuffleSplitRowsTest(QaTestCase):
    def test_weights_travel_with_their_shuffled_rows(self):
        features = frame(pd.Index([0, 1, 2, 3]), f=[10.0, 20.0, 30.0, 40.0])
        labels = frame(pd.Index([0, 1, 2, 3]), y=[1.0, 2.0, 3.0, 4.0])
        base = np.array([1.0, 2.0, 3.0, 4.0])
        label_weights = np.array([0.1, 0.2, 0.3, 0.4])

        shuffled, shuffled_labels, shuffled_base, shuffled_label_weights = (
            QuickAdapterRegressorV3._shuffle_split_rows(
                features, labels, base, label_weights, seed=7
            )
        )
        for position, index in enumerate(shuffled.index):
            with self.subTest(row=index):
                self.assertEqual(float(shuffled["f"].iloc[position]), 10.0 * (index + 1))
                self.assertEqual(float(shuffled_labels["y"].iloc[position]), float(index + 1))
                self.assertEqual(float(shuffled_base[position]), float(index + 1))
                self.assertAlmostEqual(float(shuffled_label_weights[position]), (index + 1) / 10.0)

    def test_the_same_seed_reproduces_the_same_order(self):
        features = frame(pd.Index(range(8)), f=np.arange(8.0))
        labels = frame(pd.Index(range(8)), y=np.arange(8.0))
        base = np.arange(8.0)
        first, *_ = QuickAdapterRegressorV3._shuffle_split_rows(
            features, labels, base, None, seed=11
        )
        second, *_ = QuickAdapterRegressorV3._shuffle_split_rows(
            features, labels, base, None, seed=11
        )
        self.assertEqual(list(first.index), list(second.index))

    def test_a_different_seed_may_reorder_the_rows(self):
        features = frame(pd.Index(range(8)), f=np.arange(8.0))
        labels = frame(pd.Index(range(8)), y=np.arange(8.0))
        base = np.arange(8.0)
        orders = {
            tuple(
                QuickAdapterRegressorV3._shuffle_split_rows(
                    features, labels, base, None, seed=seed
                )[0].index
            )
            for seed in range(8)
        }
        self.assertGreater(len(orders), 1)


class KnownBeforePositionMaskTest(QaTestCase):
    def setUp(self):
        super().setUp()
        self.unfiltered = frame(pd.Index(range(10)))

    def test_a_row_known_exactly_at_the_cutoff_is_excluded(self):
        model = regressor()
        model._label_horizon_candles = lambda pair=None: 0
        unfiltered = frame(pd.Index(range(10)), **{LOOKAHEAD: np.zeros(10, dtype=np.int64)})
        features = frame(pd.Index([0, 3, 4, 5]))
        mask = model._known_before_position_mask(features, unfiltered, PAIR, cutoff_position=4)
        # A zero lookahead makes availability the row position itself, so the row AT the cutoff
        # is the boundary case: the comparison is strict, so it must be excluded.
        np.testing.assert_array_equal(mask, np.array([True, True, False, False]))

    def test_the_cutoff_is_exclusive_on_both_horizons(self):
        model = regressor()
        model._label_horizon_candles = lambda pair=None: 0
        unfiltered = frame(pd.Index(range(10)), **{LOOKAHEAD: np.zeros(10, dtype=np.int64)})
        features = frame(pd.Index([0, 3]))
        for cutoff, expected in ((3, [True, False]), (4, [True, True]), (5, [True, True])):
            with self.subTest(cutoff=cutoff):
                mask = model._known_before_position_mask(features, unfiltered, PAIR, cutoff)
                np.testing.assert_array_equal(mask, np.array(expected))

    def test_the_position_fallback_applies_when_no_column_is_emitted(self):
        model = regressor()
        model._label_horizon_candles = lambda pair=None: 2
        features = frame(pd.Index([0, 1, 2]))
        mask = model._known_before_position_mask(features, self.unfiltered, PAIR, cutoff_position=5)
        np.testing.assert_array_equal(mask, np.array([0 + 2 < 5, 1 + 2 < 5, 2 + 2 < 5]))

    def test_the_fallback_horizon_shifts_the_cutoff(self):
        wide = regressor()
        wide._label_horizon_candles = lambda pair=None: 0
        narrow = regressor()
        narrow._label_horizon_candles = lambda pair=None: 3
        features = frame(pd.Index([0, 1, 2]))
        loose = wide._known_before_position_mask(features, self.unfiltered, PAIR, 3)
        strict = narrow._known_before_position_mask(features, self.unfiltered, PAIR, 3)
        self.assertEqual(loose.sum(), 3)
        self.assertEqual(strict.sum(), 0)


class ValidationSplitTest(QaTestCase):
    def setUp(self):
        super().setUp()
        self.unfiltered = frame(pd.Index(range(12)))
        self.features = frame(pd.Index(range(8)), f=np.arange(8.0))
        self.weights = SampleWeightInputs(
            base=np.arange(1.0, 9.0), label=None, label_weighting_config=WEIGHTING_CONFIG
        )
        self.data_dictionary = {
            "train_features": self.features,
            "train_labels": frame(pd.Index(range(8)), y=np.arange(8.0)),
            "train_weights": np.ones(8),
            "test_features": frame(pd.Index(range(8, 12)), f=np.arange(8.0, 12.0)),
            "test_labels": frame(pd.Index(range(8, 12)), y=np.arange(8.0, 12.0)),
            "test_weights": np.ones(4),
        }

    def _model(self, **attrs: object) -> QuickAdapterRegressorV3:
        return regressor(**attrs)

    def _split(self, model: QuickAdapterRegressorV3, **kwargs: object) -> dict:
        return model._add_validation_split(
            dict(self.data_dictionary),
            self.features,
            self.weights,
            self.unfiltered,
            PAIR,
            **kwargs,
        )

    def test_a_zero_test_size_yields_empty_validation_frames(self):
        model = self._model(data_split_parameters={"test_size": 0, "shuffle": False})
        result = self._split(model)
        self.assertTrue(result["validation_features"].empty)
        self.assertTrue(result["validation_labels"].empty)
        self.assertEqual(len(result["validation_weights"]), 0)

    def test_a_zero_test_size_leaves_the_training_rows_untouched(self):
        model = self._model(data_split_parameters={"test_size": 0, "shuffle": False})
        result = self._split(model)
        self.assertEqual(list(result["train_features"].index), list(range(8)))

    def test_a_shuffled_configuration_is_refused(self):
        for key, source in (
            ("shuffle", {"data_split_parameters": {"test_size": 2, "shuffle": True}}),
            ("shuffle_after_split", {"ft_params": {"shuffle_after_split": True}}),
            ("reverse_train_test_order", {"ft_params": {"reverse_train_test_order": True}}),
        ):
            with self.subTest(flag=key):
                # A validation split must actually be requested: the guard sits behind the
                # test_size == 0 early return, so a zero-size configuration never reaches it.
                model = self._model(
                    **{"data_split_parameters": {"test_size": 2, "shuffle": False}, **source}
                )
                with self.assertRaisesRegex(ValueError, "Independent holdout evaluation requires"):
                    self._split(model)

    def test_a_zero_size_configuration_returns_before_the_shuffle_guard(self):
        model = self._model(
            data_split_parameters={"test_size": 0, "shuffle": True},
            ft_params={"shuffle_after_split": True},
        )
        result = self._split(model)
        self.assertTrue(result["validation_features"].empty)

    def test_the_chronological_tail_is_reserved_not_sampled(self):
        model = self._model(data_split_parameters={"test_size": 2, "shuffle": False})
        result = self._split(model)
        self.assertEqual(list(result["validation_features"].index), [6, 7])
        self.assertEqual(list(result["train_features"].index), [0, 1, 2, 3, 4, 5])

    def test_the_purge_leaves_no_overlap_with_the_validation_tail(self):
        model = self._model(data_split_parameters={"test_size": 2, "shuffle": False})
        result = self._split(model)
        self.assertTrue(
            set(result["train_features"].index).isdisjoint(result["validation_features"].index)
        )

    def test_a_wider_horizon_purges_more_training_rows(self):
        kept = {}
        for horizon in (0, 1, 3):
            with self.subTest(horizon=horizon):
                model = self._model(data_split_parameters={"test_size": 2, "shuffle": False})
                model._label_horizon_candles = lambda pair=None, h=horizon: h
                result = self._split(model)
                kept[horizon] = len(result["train_features"])
        self.assertGreater(kept[0], kept[1])
        self.assertGreater(kept[1], kept[3])

    def test_the_weights_are_recomposed_on_the_split_rows_not_sliced(self):
        model = self._model(data_split_parameters={"test_size": 2, "shuffle": False})
        result = self._split(model)
        self.assertEqual(len(result["train_weights"]), len(result["train_features"]))
        self.assertEqual(len(result["validation_weights"]), len(result["validation_features"]))
        # A recomposition renormalises each subset to mean 1, so the recomposed train weights
        # cannot equal the raw prefix of the base weights the caller passed in.
        self.assertAlmostEqual(float(np.mean(result["train_weights"])), 1.0)
        self.assertAlmostEqual(float(np.mean(result["validation_weights"])), 1.0)

    def test_holdout_rows_available_before_the_window_end_are_kept(self):
        model = self._model(data_split_parameters={"test_size": 2, "shuffle": False})
        result = self._split(model)
        self.assertEqual(list(result["test_features"].index), [8, 9, 10, 11])
        self.assertNotIn("holdout_purged_empty", result)

    def test_a_fully_purged_holdout_is_flagged_rather_than_scored(self):
        # A lookahead of 4 makes row 8 available at position 12, which is the window end and
        # therefore not strictly before it: every holdout row is purged.
        unfiltered = self.unfiltered.assign(**{LOOKAHEAD: np.full(12, 4, dtype=np.int64)})
        model = self._model(data_split_parameters={"test_size": 2, "shuffle": False})
        result = model._add_validation_split(
            dict(self.data_dictionary), self.features, self.weights, unfiltered, PAIR
        )
        self.assertTrue(result["test_features"].empty)
        self.assertTrue(result.get("holdout_purged_empty"))

    def test_a_partially_purged_holdout_keeps_the_available_rows(self):
        unfiltered = self.unfiltered.assign(**{LOOKAHEAD: np.full(12, 2, dtype=np.int64)})
        model = self._model(data_split_parameters={"test_size": 2, "shuffle": False})
        result = model._add_validation_split(
            dict(self.data_dictionary), self.features, self.weights, unfiltered, PAIR
        )
        # Rows 8 and 9 are available at 10 and 11, both strictly before the window end.
        self.assertEqual(list(result["test_features"].index), [8, 9])
        self.assertNotIn("holdout_purged_empty", result)

    def test_a_lookahead_column_replaces_the_position_fallback(self):
        unfiltered = self.unfiltered.assign(**{LOOKAHEAD: np.full(12, 4, dtype=np.int64)})
        model = self._model(data_split_parameters={"test_size": 2, "shuffle": False})
        result = model._add_validation_split(
            dict(self.data_dictionary),
            self.features,
            self.weights,
            unfiltered,
            PAIR,
        )
        # Validation starts at position 6; a lookahead of 4 keeps only positions 0 and 1.
        self.assertEqual(list(result["validation_features"].index), [6, 7])
        self.assertEqual(list(result["train_features"].index), [0, 1])


class ConstructionContractTest(QaTestCase):
    """The fixtures above bypass __init__; this pins the invariants they assume."""

    def test_a_real_regressor_exposes_the_attributes_the_purge_fixtures_inject(self):
        with temporary_directory() as temp:
            model = QuickAdapterRegressorV3(config=model_config(temp))
        for attribute in ("ft_params", "data_split_parameters", "data_provider"):
            with self.subTest(attribute=attribute):
                self.assertTrue(hasattr(model, attribute))

    def test_a_real_regressor_derives_its_label_horizon_from_ft_params(self):
        with temporary_directory() as temp:
            model = QuickAdapterRegressorV3(config=model_config(temp))
        self.assertEqual(model._label_horizon_candles(), 1)

    def test_sample_weight_inputs_refuse_a_two_dimensional_base(self):
        with self.assertRaisesRegex(ValueError, "must be 1-D"):
            SampleWeightInputs(
                base=np.ones((2, 2)), label=None, label_weighting_config=WEIGHTING_CONFIG
            )

    def test_sample_weight_inputs_refuse_mismatched_label_weights(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            SampleWeightInputs(
                base=np.ones(4), label=np.ones(3), label_weighting_config=WEIGHTING_CONFIG
            )

    def test_sample_weight_inputs_refuse_an_incomplete_weighting_config(self):
        with self.assertRaisesRegex(KeyError, "min_effective_sample_size"):
            SampleWeightInputs(
                base=np.ones(4),
                label=None,
                label_weighting_config={"support_policy": "fallback"},
            )

    def test_sample_weight_inputs_refuse_an_unknown_support_policy(self):
        with self.assertRaisesRegex(ValueError, "support_policy"):
            SampleWeightInputs(
                base=np.ones(4),
                label=None,
                label_weighting_config={**WEIGHTING_CONFIG, "support_policy": "ignore"},
            )


if __name__ == "__main__":
    unittest.main()
