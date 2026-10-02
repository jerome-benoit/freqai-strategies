"""Causal purge and chronological split contracts; requires the Freqtrade QA image."""

import unittest

import numpy as np
import pandas as pd
from freqtrade.exceptions import DependencyException
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

    def test_a_lone_column_holding_a_nan_leaves_no_usable_lookahead(self):
        # The skip is per COLUMN, and the fixture above keeps a clean weight column, so the
        # surviving values coincide whether or not the guard runs. It is observable only
        # when the ONLY emitted column carries a NaN: the caller must then get None and fall
        # back to the position-based purge, where without the guard the NaN reaches
        # astype(np.int64) and raises IntCastingNaNError out of the function.
        unfiltered = frame(pd.Index([0, 1]), **{LOOKAHEAD: [3.0, np.nan]})
        self.assertIsNone(QuickAdapterRegressorV3._known_at_lookahead(unfiltered, unfiltered))

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

    def test_an_oversized_validation_count_is_refused(self):
        # test_size is bounded by the rows remaining after the outer holdout, and the bound
        # is `>=`: eight training rows admit at most a seven-row validation tail. Removing the
        # guard, or relaxing it to `>`, previously left the suite green.
        for test_size in (8, 9):
            with self.subTest(test_size=test_size):
                model = self._model(
                    data_split_parameters={"test_size": test_size, "shuffle": False}
                )
                with self.assertRaisesRegex(DependencyException, "is not smaller than the 8"):
                    self._split(model)

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
        # The ratios are the discriminating assertion: `base` is a strict ramp and the label
        # weights are None, so a recomposition is a pure rescale and the weights reproduce
        # the BASE ratios. np.ones has ratio 1.0 everywhere and cannot pass this, which is
        # the whole point of the contract the production comment spells out.
        for weights, expected in (
            (result["train_weights"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            (result["validation_weights"], [1.0, 8.0 / 7.0]),
        ):
            with self.subTest(weights=list(weights)):
                self.assertAlmostEqual(float(np.mean(weights)), 1.0)
                np.testing.assert_allclose(
                    np.asarray(weights) / weights[0], np.array(expected), rtol=1e-12, atol=0.0
                )

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


class TrainTestSplitCausalPurgeTest(QaTestCase):
    """The OUTER split, which is the shipped default.

    `causal_mode` defaults to True, the split method defaults to `train_test_split`
    and `config-template.json` ships `"test_size": 0.333`, so
    `_make_train_test_split_datasets` with a non-zero test size is the path a stock
    configuration takes. Every other test drives `_add_validation_split` or `test_size: 0`
    instead, which left the whole causal purge block — and both of its boundary
    comparisons — with no coverage at all: relaxing `<` to `<=` on either of them kept the
    suite green.
    """

    ROWS = 40
    TEST_ROWS = 10

    class _Kitchen:
        pair = PAIR

        @staticmethod
        def build_data_dictionary(
            train_features, test_features, train_labels, test_labels, train_weights, test_weights
        ):
            return {
                "train_features": train_features,
                "test_features": test_features,
                "train_labels": train_labels,
                "test_labels": test_labels,
                "train_weights": train_weights,
                "test_weights": test_weights,
            }

    def setUp(self):
        super().setUp()
        self.index = pd.Index(range(self.ROWS))
        # `_known_at_lookahead` reads the availability columns from the UNFILTERED frame, so
        # that is where they have to live. With no availability column present it returns
        # None, the guard logs once and skips its second comparison entirely — so the horizon
        # boundary and the known_at boundary are exercised from separate fixtures below.
        self.unfiltered = frame(self.index)
        self.features = frame(self.index, f=np.arange(self.ROWS, dtype=float))
        self.labels = frame(self.index, y=np.arange(self.ROWS, dtype=float))
        self.weights = SampleWeightInputs(
            base=np.ones(self.ROWS),
            label=None,
            label_weighting_config=WEIGHTING_CONFIG,
        )

    def _split(self, unfiltered: pd.DataFrame, horizon: int = 0, **params):
        model = regressor(
            data_split_parameters={"test_size": self.TEST_ROWS, "shuffle": False, **params},
            _label_horizon_candles=lambda pair=None: horizon,
        )
        return model._make_train_test_split_datasets(
            self.features, self.labels, self.weights, self._Kitchen(), unfiltered
        )

    def test_the_horizon_boundary_purges_the_training_row_exactly_at_the_cutoff(self):
        # train_test_split without shuffle puts the LAST 10 rows in the test set, so
        # first_test_position is 30. With a horizon of 3 the strict comparison keeps a
        # training row only while position < 27, so row 26 is the last survivor and row 27 is
        # purged. `<=` would keep row 27 as well, which is the off-by-one the guard exists
        # to prevent: that row's label horizon reaches into the test window.
        result = self._split(self.unfiltered, horizon=3)
        kept = list(result["train_features"].index)
        self.assertEqual(max(kept), 26)
        self.assertNotIn(27, kept)
        self.assertEqual(list(result["test_features"].index), list(range(30, 40)))

    def test_a_zero_horizon_purges_nothing_by_position(self):
        # The floor of the same comparison: with no horizon the last training row sits
        # immediately before the test window and is kept, so row 29 must survive.
        result = self._split(self.unfiltered, horizon=0)
        self.assertEqual(list(result["train_features"].index), list(range(30)))

    def test_a_row_known_exactly_at_the_first_test_row_is_purged(self):
        # The second, independent boundary. A label that becomes available at position
        # position + lookahead must be strictly before the first test row, so a training row
        # at position 25 with a lookahead of 5 is available exactly at 30 and is purged.
        # Without the availability column the whole branch is skipped and this never bites.
        def with_lookahead(row: int, value: int):
            values = [0] * self.ROWS
            values[row] = value
            return frame(self.index, **{LOOKAHEAD: values, WEIGHT_LOOKAHEAD: [0] * self.ROWS})

        unfiltered = with_lookahead(25, 5)
        result = self._split(unfiltered, horizon=0)
        kept = list(result["train_features"].index)
        self.assertNotIn(25, kept, "available exactly at the first test row, so it must be purged")
        self.assertIn(24, kept)
        # A row one candle earlier is available at 29, strictly inside the training window.
        kept = list(self._split(with_lookahead(24, 4), horizon=0)["train_features"].index)
        self.assertIn(24, kept)

    def test_the_horizon_and_the_availability_masks_are_combined_not_replaced(self):
        # Pinning the two boundaries one at a time proves nothing about how they combine,
        # which is exactly how the first version of this class went wrong: the horizon case
        # carried no availability column and the availability case used a zero horizon, so
        # `keep_mask &=` never ran with both constraints live. Rewriting the conjunction as an
        # assignment — `keep_mask = train_known_at_position < ...` — discards the horizon mask
        # and left the suite green.
        #
        # Both directions are needed, because either one alone is satisfied by a mask that
        # simply overwrites the other.
        zero_lookahead = frame(
            self.index, **{LOOKAHEAD: [0] * self.ROWS, WEIGHT_LOOKAHEAD: [0] * self.ROWS}
        )

        # Direction 1 — the horizon mask removes rows the availability mask would keep. With
        # a horizon of 3 the horizon keeps rows below 27 while availability, at a zero
        # lookahead, keeps everything below the first test row at 30. The conjunction must
        # give 26; the assignment gives 29.
        kept = list(self._split(zero_lookahead, horizon=3)["train_features"].index)
        self.assertEqual(max(kept), 26)
        for row in (27, 28, 29):
            with self.subTest(discarded_by="horizon", row=row):
                self.assertNotIn(row, kept)

        # Direction 2 — the availability mask removes a row the horizon kept. A lookahead of 6
        # on row 20 makes it available at 26, inside the window, so it survives; a lookahead
        # of 10 on row 20 makes it available exactly at 30 and it must go even though the
        # horizon would have kept it.
        for value, expected in ((6, True), (10, False)):
            lookahead = [0] * self.ROWS
            lookahead[20] = value
            unfiltered = frame(
                self.index, **{LOOKAHEAD: lookahead, WEIGHT_LOOKAHEAD: [0] * self.ROWS}
            )
            with self.subTest(discarded_by="availability", lookahead=value):
                self.assertEqual(
                    20 in self._split(unfiltered, horizon=0)["train_features"].index, expected
                )

    def test_the_causal_guard_is_skipped_outside_causal_mode(self):
        # The counterpart, so the boundary tests above are not passing for the wrong reason:
        # with causal_mode off the same row survives.
        lookahead = [0] * self.ROWS
        lookahead[25] = 5
        unfiltered = frame(self.index, **{LOOKAHEAD: lookahead, WEIGHT_LOOKAHEAD: [0] * self.ROWS})
        model = regressor(
            data_split_parameters={"test_size": self.TEST_ROWS, "shuffle": False},
            _causal_mode=False,
            _label_horizon_candles=lambda pair=None: 0,
        )
        result = model._make_train_test_split_datasets(
            self.features, self.labels, self.weights, self._Kitchen(), unfiltered
        )
        self.assertEqual(list(result["train_features"].index), list(range(30)))


class TimeSeriesSplitCausalPurgeTest(QaTestCase):
    """Final-fold gap and label-availability boundaries for chronological splitting."""

    ROWS = 40
    N_SPLITS = 3
    TEST_SIZE = 5
    GAP = 2

    class _Kitchen:
        pair = PAIR

        @staticmethod
        def build_data_dictionary(
            train_features, test_features, train_labels, test_labels, train_weights, test_weights
        ):
            return {
                "train_features": train_features,
                "test_features": test_features,
                "train_labels": train_labels,
                "test_labels": test_labels,
                "train_weights": train_weights,
                "test_weights": test_weights,
            }

    def setUp(self):
        super().setUp()
        self.index = pd.Index(range(self.ROWS))
        self.unfiltered = frame(self.index)
        self.features = frame(self.index, f=np.arange(self.ROWS, dtype=float))
        self.labels = frame(self.index, y=np.arange(self.ROWS, dtype=float))
        self.weights = SampleWeightInputs(
            base=np.ones(self.ROWS),
            label=None,
            label_weighting_config=WEIGHTING_CONFIG,
        )

    def _split(self, unfiltered: pd.DataFrame, horizon: int = 0, **params):
        model = regressor(
            data_split_parameters={
                "method": "timeseries_split",
                "n_splits": self.N_SPLITS,
                "test_size": self.TEST_SIZE,
                # The code refuses a gap smaller than the horizon under causal_mode, so the
                # horizon cases widen the gap to match rather than tripping that guard.
                "gap": max(self.GAP, horizon),
                **params,
            },
            _label_horizon_candles=lambda pair=None: horizon,
        )
        return model._make_timeseries_split_datasets(
            self.features, self.labels, self.weights, self._Kitchen(), unfiltered
        )

    def test_the_last_fold_preserves_the_gap_and_the_chronological_tail(self):
        # With no emitted availability, these cases exercise only the splitter gap.
        # A horizon of three widens a zero requested gap to three, ending training at row 31.
        # The availability-column purge is exercised separately below.
        for horizon, gap, expected_train in (
            (0, 2, list(range(33))),
            (3, 3, list(range(32))),
            (3, 0, list(range(32))),
        ):
            with self.subTest(horizon=horizon, gap=gap):
                result = self._split(self.unfiltered, horizon=horizon, gap=gap)
                self.assertEqual(list(result["train_features"].index), expected_train)
                self.assertEqual(list(result["test_features"].index), [35, 36, 37, 38, 39])

    def test_a_row_known_at_or_after_the_first_test_row_is_purged(self):
        # The boundary this block exists for, and the one coverage could not see at all.
        #
        # `gap=2` removes the two rows before the test fold from the training set BEFORE the
        # availability comparison runs, so the last row that comparison can act on is
        # `first_test - gap - 1` = 32. Measured: train ends at 32, test is 35..39. The lookahead
        # is placed there for that reason — on row 33 or 34 it is indistinguishable from no
        # lookahead at all, because those rows are already gone before the comparison.
        last_train = self.ROWS - self.TEST_SIZE - self.GAP - 1
        first_test = self.ROWS - self.TEST_SIZE

        def with_lookahead(row: int, value: int):
            values = [0] * self.ROWS
            values[row] = value
            return frame(self.index, **{LOOKAHEAD: values, WEIGHT_LOOKAHEAD: [0] * self.ROWS})

        # Available at last_train, +1, +2 — all strictly before the cutoff — and exactly at it.
        for value, expected in ((0, True), (1, True), (2, True), (3, False)):
            with self.subTest(lookahead=value):
                kept = list(
                    self._split(with_lookahead(last_train, value), horizon=0)[
                        "train_features"
                    ].index
                )
                self.assertEqual(
                    last_train in kept,
                    expected,
                    f"lookahead {value} lands at {last_train + value}, cutoff {first_test}",
                )

    def test_the_availability_comparison_is_strict_at_the_cutoff(self):
        # The isolating case: a single row whose label becomes knowable exactly on the first
        # test row. Relaxing `<` to `<=` keeps it, and nothing else in the suite says so.
        last_train = self.ROWS - self.TEST_SIZE - self.GAP - 1
        values = [0] * self.ROWS
        values[last_train] = self.GAP + 1
        unfiltered = frame(self.index, **{LOOKAHEAD: values, WEIGHT_LOOKAHEAD: [0] * self.ROWS})

        kept = list(self._split(unfiltered, horizon=0)["train_features"].index)

        self.assertNotIn(last_train, kept)


if __name__ == "__main__":
    unittest.main()
