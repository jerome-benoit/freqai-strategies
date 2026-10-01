"""Regressor refit, alias and Optuna-range plumbing contracts; requires the Freqtrade QA image."""

import unittest
from typing import Any

import numpy as np
import pandas as pd
from EnumErrors import enum_error_message
from qa_support import QaTestCase
from Utils import (
    _EARLY_STOPPING_ROUNDS_DEFAULT,
    _NGBOOST_DISTRIBUTIONS,
    _REGRESSOR_SPEC_BY_NAME,
    REGRESSORS,
    _apply_verbosity_alias,
    _build_int_range,
    _optuna_suggest_int_from_range,
    _pop_early_stopping_rounds,
    get_ngboost_dist,
    get_refit_model_training_parameters,
    make_test_set_and_weights,
    resolve_optuna_model_parameters,
)


class _Booster:
    def __init__(self, rounds: int) -> None:
        self._rounds = rounds

    def num_boosted_rounds(self) -> int:
        return self._rounds


class _XGBoostModel:
    def __init__(self, rounds: int) -> None:
        self._booster = _Booster(rounds)

    def get_booster(self) -> _Booster:
        return self._booster


class _LightGBMBooster:
    def __init__(self, current_iteration: int) -> None:
        self._current_iteration = current_iteration

    def current_iteration(self) -> int:
        return self._current_iteration


class _LightGBMModel:
    def __init__(self, n_estimators: int, best_iteration: int | None = None) -> None:
        if best_iteration is not None:
            self.best_iteration_ = best_iteration
        self.n_estimators_ = n_estimators

    @property
    def booster_(self) -> _LightGBMBooster:
        return _LightGBMBooster(self._current_iteration)

    def warm_start(self, current_iteration: int) -> "_LightGBMModel":
        model = _LightGBMModel(self.n_estimators_, getattr(self, "best_iteration_", None))
        model._current_iteration = current_iteration
        return model


class _HistGradientBoostingModel:
    def __init__(self, n_iter: int) -> None:
        self.n_iter_ = n_iter


class _NGBoostModel:
    def __init__(self, base_models: int) -> None:
        self.base_models = list(range(base_models))


class _CatBoostModel:
    def __init__(self, tree_count: int) -> None:
        self.tree_count_ = tree_count


class _RecordingTrial:
    def __init__(self) -> None:
        self.int_calls: list[tuple[str, int, int, bool]] = []

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
        self.int_calls.append((name, low, high, log))
        return low if low == high else (low + high) // 2


def recording_trial() -> _RecordingTrial:
    return _RecordingTrial()


def validation_split() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    features = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    targets = pd.DataFrame({"&-s_close": [0.1, 0.2, 0.3]})
    weights = np.array([1.0, 2.0, 3.0])
    return features, targets, weights


class UtilsRegressorPlumbingTest(QaTestCase):
    def test_exactly_one_iteration_parameter_survives_a_refit_for_every_regressor(self):
        # The alias table is the reason CatBoost does not abort on a duplicate
        # iteration parameter: every alias must be gone, and the canonical one set.
        cases = {
            "xgboost": (_XGBoostModel(77), 77),
            "lightgbm": (_LightGBMModel(250), 250),
            "histgradientboostingregressor": (_HistGradientBoostingModel(30), 30),
            "ngboost": (_NGBoostModel(17), 17),
            "catboost": (_CatBoostModel(120), 120),
        }
        self.assertEqual(sorted(cases), sorted(REGRESSORS))

        for regressor, (model, expected) in cases.items():
            with self.subTest(regressor=regressor):
                spec = _REGRESSOR_SPEC_BY_NAME[regressor]
                training_parameters = dict.fromkeys(sorted(spec.iteration_aliases), 999)

                refit_parameters = get_refit_model_training_parameters(
                    regressor, model, training_parameters
                )

                iteration_keys = [key for key in refit_parameters if key in spec.iteration_aliases]
                self.assertEqual(iteration_keys, [spec.iteration_param])
                self.assertEqual(refit_parameters[spec.iteration_param], expected)

    def test_a_refit_drops_every_alias_even_when_they_agree_with_each_other(self):
        spec = _REGRESSOR_SPEC_BY_NAME["catboost"]
        self.assertEqual(
            sorted(spec.iteration_aliases),
            ["iterations", "n_estimators", "num_boost_round", "num_trees"],
        )
        training_parameters = dict.fromkeys(sorted(spec.iteration_aliases), 120)

        refit_parameters = get_refit_model_training_parameters(
            "catboost", _CatBoostModel(120), training_parameters
        )

        self.assertEqual(refit_parameters, {"iterations": 120})

    def test_a_refit_keeps_every_parameter_that_is_not_an_iteration_alias(self):
        training_parameters = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 999,
            "num_boost_round": 999,
        }

        refit_parameters = get_refit_model_training_parameters(
            "xgboost", _XGBoostModel(77), training_parameters
        )

        self.assertEqual(
            refit_parameters,
            {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 77},
        )

    def test_a_refit_preserves_a_degenerate_capacity_by_clamping_to_one(self):
        # A fit that added no rounds over the warm start, and a fit that added none
        # at all, both degrade to a single round rather than to zero or a raise.
        warm_start = _XGBoostModel(77)
        self.assertEqual(
            get_refit_model_training_parameters("xgboost", _XGBoostModel(77), {}, warm_start),
            {"n_estimators": 1},
        )
        self.assertEqual(
            get_refit_model_training_parameters("catboost", _CatBoostModel(0), {}),
            {"iterations": 1},
        )
        self.assertEqual(
            get_refit_model_training_parameters("ngboost", _NGBoostModel(0), {}),
            {"n_estimators": 1},
        )

    def test_a_refit_preserves_only_the_rounds_a_warm_start_has_not_fitted_yet(self):
        self.assertEqual(
            get_refit_model_training_parameters(
                "xgboost", _XGBoostModel(77), {}, _XGBoostModel(20)
            ),
            {"n_estimators": 57},
        )
        self.assertEqual(
            get_refit_model_training_parameters(
                "lightgbm",
                _LightGBMModel(250).warm_start(90),
                {},
                _LightGBMModel(250).warm_start(90),
            ),
            {"n_estimators": 160},
        )

    def test_a_lightgbm_refit_prefers_a_positive_best_iteration_over_n_estimators(self):
        # best_iteration_ == 0 and == -1 both mean "no early stopping ran", so the
        # configured n_estimators_ is the only capacity information left.
        self.assertEqual(
            get_refit_model_training_parameters("lightgbm", _LightGBMModel(250, 40), {}),
            {"n_estimators": 40},
        )
        for unused_best_iteration in (0, -1):
            with self.subTest(best_iteration=unused_best_iteration):
                self.assertEqual(
                    get_refit_model_training_parameters(
                        "lightgbm", _LightGBMModel(250, unused_best_iteration), {}
                    ),
                    {"n_estimators": 250},
                )

    def test_a_refit_does_not_mutate_or_share_the_caller_parameters(self):
        nested = [1, 2]
        training_parameters: dict[str, Any] = {"custom": nested, "n_estimators": 999}

        refit_parameters = get_refit_model_training_parameters(
            "xgboost", _XGBoostModel(77), training_parameters
        )

        self.assertEqual(training_parameters, {"custom": [1, 2], "n_estimators": 999})
        self.assertIsNot(refit_parameters["custom"], nested)
        refit_parameters["custom"].append(3)
        self.assertEqual(nested, [1, 2])

    def test_a_histgradientboosting_refit_disables_early_stopping(self):
        refit_parameters = get_refit_model_training_parameters(
            "histgradientboostingregressor",
            _HistGradientBoostingModel(30),
            {"early_stopping": True, "early_stopping_rounds": 17},
        )

        self.assertEqual(
            refit_parameters,
            {"early_stopping": False, "early_stopping_rounds": 17, "max_iter": 30},
        )

    def test_a_refit_rejects_an_unknown_regressor_with_the_canonical_enum_error(self):
        with self.assertRaises(ValueError) as caught:
            get_refit_model_training_parameters("randomforest", _CatBoostModel(1), {})

        self.assertEqual(
            str(caught.exception),
            enum_error_message("regressor", "randomforest", REGRESSORS),
        )

    def test_early_stopping_rounds_are_returned_and_consumed_when_an_eval_set_exists(self):
        training_parameters = {"early_stopping_rounds": 17}

        rounds = _pop_early_stopping_rounds(training_parameters, True)

        self.assertEqual(rounds, 17)
        self.assertEqual(training_parameters, {})

    def test_early_stopping_rounds_fall_back_to_the_default_when_the_key_is_absent(self):
        training_parameters: dict[str, Any] = {}

        rounds = _pop_early_stopping_rounds(training_parameters, True)

        self.assertEqual(rounds, _EARLY_STOPPING_ROUNDS_DEFAULT)
        self.assertEqual(rounds, 50)
        self.assertEqual(training_parameters, {})

    def test_early_stopping_rounds_are_removed_without_an_eval_set(self):
        # The removal, not the return value, is the contract: a refit must not carry
        # an early-stopping parameter when there is nothing to stop on.
        training_parameters = {"early_stopping_rounds": 17}

        rounds = _pop_early_stopping_rounds(training_parameters, False)

        self.assertIsNone(rounds)
        self.assertEqual(training_parameters, {})

    def test_a_missing_early_stopping_key_without_an_eval_set_leaves_the_parameters_alone(self):
        training_parameters = {"max_depth": 6}

        rounds = _pop_early_stopping_rounds(training_parameters, False)

        self.assertIsNone(rounds)
        self.assertEqual(training_parameters, {"max_depth": 6})

    def test_verbosity_is_renamed_to_verbose_when_verbose_is_absent(self):
        training_parameters = {"verbosity": 2, "max_depth": 6}

        _apply_verbosity_alias(training_parameters)

        self.assertEqual(training_parameters, {"verbose": 2, "max_depth": 6})

    def test_an_explicit_verbose_wins_and_the_verbosity_alias_is_still_removed(self):
        training_parameters = {"verbosity": 2, "verbose": 0}

        _apply_verbosity_alias(training_parameters)

        self.assertEqual(training_parameters, {"verbose": 0})

    def test_a_verbosity_of_zero_is_renamed_because_it_is_set_rather_than_truthy(self):
        training_parameters = {"verbosity": 0}

        _apply_verbosity_alias(training_parameters)

        self.assertEqual(training_parameters, {"verbose": 0})

    def test_verbosity_is_dropped_without_a_value_when_it_is_explicitly_none(self):
        training_parameters = {"verbosity": None, "max_depth": 6}

        _apply_verbosity_alias(training_parameters)

        self.assertEqual(training_parameters, {"max_depth": 6})

    def test_verbosity_aliasing_leaves_a_verbose_only_mapping_untouched(self):
        training_parameters = {"verbose": 1}

        _apply_verbosity_alias(training_parameters)

        self.assertEqual(training_parameters, {"verbose": 1})

    def test_lossguide_resolves_to_an_unbounded_xgboost_depth(self):
        resolved = resolve_optuna_model_parameters(
            "xgboost", {"grow_policy": "lossguide", "max_depth": 6, "max_leaves": 64}
        )

        self.assertEqual(resolved, {"grow_policy": "lossguide", "max_depth": 0, "max_leaves": 64})

    def test_only_lossguide_forces_the_depth_to_zero(self):
        cases: tuple[tuple[str, dict[str, Any]], ...] = (
            ("xgboost", {"grow_policy": "depthwise", "max_depth": 6}),
            ("xgboost", {"max_depth": 6}),
            ("xgboost", {"grow_policy": "LossGuide", "max_depth": 6}),
            ("lightgbm", {"grow_policy": "lossguide", "max_depth": 6}),
            ("catboost", {"grow_policy": "lossguide", "max_depth": 6}),
            ("ngboost", {"grow_policy": "lossguide", "max_depth": 6}),
        )
        for regressor, params in cases:
            with self.subTest(regressor=regressor, params=params):
                resolved = resolve_optuna_model_parameters(regressor, dict(params))

                self.assertEqual(resolved, params)

    def test_the_l2_regularization_zero_suggestion_maps_onto_l2_regularization(self):
        # The raw suggestion is a boolean because the search space spans both zero
        # and a continuous range; the estimator only accepts the numeric parameter.
        resolved = resolve_optuna_model_parameters(
            "histgradientboostingregressor",
            {"l2_regularization_zero": True, "l2_regularization": 3.0, "max_leaf_nodes": 31},
        )

        self.assertEqual(resolved, {"l2_regularization": 0.0, "max_leaf_nodes": 31})

    def test_a_rejected_l2_regularization_zero_suggestion_is_dropped_not_forwarded(self):
        resolved = resolve_optuna_model_parameters(
            "histgradientboostingregressor", {"l2_regularization_zero": False}
        )

        self.assertEqual(resolved, {})

    def test_an_absent_l2_regularization_zero_leaves_the_numeric_parameter_alone(self):
        resolved = resolve_optuna_model_parameters(
            "histgradientboostingregressor", {"l2_regularization": 3.0}
        )

        self.assertEqual(resolved, {"l2_regularization": 3.0})

    def test_the_l2_regularization_zero_mapping_belongs_to_histgradientboosting_alone(self):
        resolved = resolve_optuna_model_parameters("xgboost", {"l2_regularization_zero": True})

        self.assertEqual(resolved, {"l2_regularization_zero": True})

    def test_resolving_optuna_parameters_does_not_mutate_the_caller_mapping(self):
        params = {"grow_policy": "lossguide", "max_depth": 6}

        resolve_optuna_model_parameters("xgboost", params)

        self.assertEqual(params, {"grow_policy": "lossguide", "max_depth": 6})

    def test_a_usable_range_is_rounded_outward_to_whole_iterations(self):
        self.assertEqual(_build_int_range((1.2, 5.9)), (2, 5))
        self.assertEqual(_build_int_range((3.0, 7.0)), (3, 7))
        self.assertEqual(_build_int_range((4.0, 4.0)), (4, 4))

    def test_an_inverted_range_collapses_onto_its_rounded_midpoint(self):
        # A space-reduction pass can invert a range; collapsing keeps the study
        # running at one value instead of handing Optuna an empty range.
        cases: tuple[tuple[tuple[float, float], tuple[int, int]], ...] = (
            ((10.0, 2.0), (6, 6)),
            ((5.0, 2.0), (4, 4)),
            ((4.0, 1.0), (2, 2)),
            ((2.6, 2.4), (2, 2)),
        )
        for frange, expected in cases:
            with self.subTest(frange=frange):
                self.assertEqual(_build_int_range(frange), expected)

    def test_the_declared_minimum_is_enforced_on_both_ends_of_a_range(self):
        self.assertEqual(_build_int_range((2.0, 6.0), 5), (5, 6))
        self.assertEqual(_build_int_range((3.0, 7.0), 2), (3, 7))
        self.assertEqual(_build_int_range((0.2, 0.8), 5), (5, 5))

    def test_the_declared_minimum_is_enforced_on_a_collapsed_range(self):
        self.assertEqual(_build_int_range((0.0, -3.0)), (1, 1))
        self.assertEqual(_build_int_range((0.0, -3.0), 2), (2, 2))
        self.assertEqual(_build_int_range((10.0, 2.0), 1), (6, 6))

    def test_an_inverted_range_is_suggested_as_its_midpoint(self):
        trial = recording_trial()

        value = _optuna_suggest_int_from_range(trial, "max_depth", (10.0, 2.0))

        self.assertEqual(value, 6)
        self.assertEqual(trial.int_calls, [("max_depth", 6, 6, False)])

    def test_a_suggestion_never_falls_below_the_declared_minimum(self):
        cases: tuple[tuple[str, tuple[float, float], int, int], ...] = (
            ("max_depth", (0.0, -3.0), 2, 2),
            ("max_depth", (0.2, 0.8), 5, 5),
            ("learning_rate", (0.005, 0.3), 1, 1),
            ("max_leaves", (16.0, 32.0), 2, 24),
        )
        for name, frange, min_val, expected in cases:
            with self.subTest(name=name, frange=frange, min_val=min_val):
                trial = recording_trial()

                value = _optuna_suggest_int_from_range(
                    trial, name, frange, min_val=min_val, log=True
                )

                low, high = _build_int_range(frange, min_val=min_val)
                self.assertGreaterEqual(low, min_val)
                self.assertGreaterEqual(high, low)
                self.assertEqual(trial.int_calls, [(name, low, high, True)])
                self.assertGreaterEqual(value, min_val)
                self.assertEqual(value, expected)

    def test_every_distribution_name_resolves_to_its_ngboost_class(self):
        from ngboost.distns import Exponential, Laplace, LogNormal, Normal, T

        cases = (
            ("normal", Normal),
            ("lognormal", LogNormal),
            ("exponential", Exponential),
            ("laplace", Laplace),
            ("t", T),
        )
        self.assertEqual(tuple(_NGBOOST_DISTRIBUTIONS), tuple(name for name, _ in cases))

        for dist_name, dist_class in cases:
            with self.subTest(dist_name=dist_name):
                self.assertIs(get_ngboost_dist(dist_name), dist_class)

    def test_an_unknown_distribution_raises_through_the_canonical_enum_error(self):
        with self.assertRaises(ValueError) as caught:
            get_ngboost_dist("poisson")

        self.assertEqual(
            str(caught.exception),
            enum_error_message("dist_name", "poisson", _NGBOOST_DISTRIBUTIONS),
        )

    def test_a_positive_test_size_wraps_the_validation_split_for_model_fit(self):
        features, targets, weights = validation_split()

        eval_set, eval_weights = make_test_set_and_weights(features, targets, weights, 0.25)

        self.assertEqual(len(eval_set), 1)
        self.assertEqual(len(eval_weights), 1)
        np.testing.assert_array_equal(eval_set[0][0].to_numpy(), features.to_numpy())
        np.testing.assert_array_equal(eval_set[0][1].to_numpy(), targets.to_numpy())
        self.assertEqual(eval_weights[0].ndim, 1)
        self.assertEqual(eval_weights[0].shape, (len(features),))
        # eval_set[0] must be the caller's own frames: fit_regressor ravel y but not X,
        # and a copy here would silently drop the caller's index alignment.
        self.assertIs(eval_set[0][0], features)
        self.assertIs(eval_set[0][1], targets)
        self.assertIs(eval_weights[0], weights)

    def test_a_non_positive_test_size_suppresses_evaluation_entirely(self):
        features, targets, weights = validation_split()

        for test_size in (0.0, -0.5):
            with self.subTest(test_size=test_size):
                self.assertEqual(
                    make_test_set_and_weights(features, targets, weights, test_size),
                    (None, None),
                )

    def test_the_weight_vector_is_passed_through_without_reshaping_or_validation(self):
        # The helper never inspects the weights: alignment is the caller's contract
        # (both call sites pass a vector already indexed to the validation rows), so
        # this pins that a mismatched shape is forwarded rather than repaired.
        features, targets, _ = validation_split()

        eval_set, eval_weights = make_test_set_and_weights(features, targets, np.ones((2, 3)), 0.25)

        self.assertEqual(eval_weights[0].shape, (2, 3))
        self.assertIs(eval_set[0][0], features)


if __name__ == "__main__":
    unittest.main()
