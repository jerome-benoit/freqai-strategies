"""Label-method configuration, validation and Optuna sampler/pool contracts.

Covers the config-to-object boundary: which misconfigurations are fatal and which
are downgraded with a warning, plus the hermetic Optuna sampler mapping and the
label candle pool that feeds label selection. Requires the Freqtrade QA image.
"""

import json
import random
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import QuickAdapterV3 as strategy_module
from LabelTransformer import (
    CUSTOM_THRESHOLD_METHODS,
    PREDICTION_METHODS,
    SKIMAGE_THRESHOLD_METHODS,
)
from qa_support import PAIR, QaTestCase, model_config, temporary_directory
from QuickAdapterV3 import QuickAdapterV3
from Utils import _OPTUNA_LABEL_SELECTION_SCHEMA_VERSION, _OPTUNA_NAMESPACES, enum_error_message

import quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 as regressor_module
from quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 import (
    QuickAdapterRegressorV3,
)

REGRESSOR_LOGGER = "quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3"
# Two active objectives, neither collinear with the other, so both standardized
# distance metrics have a positive-definite covariance to work with.
REFERENCE_MATRIX = np.array(
    [
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
    ]
)
# Rank-deficient: the second column is exactly twice the first.
COLLINEAR_MATRIX = np.array(
    [
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ]
)
NO_ENTRY_SIGNALS = -2.0, 2.0


def label_regressor(**attributes: object) -> QuickAdapterRegressorV3:
    """Build a regressor without running __init__, injecting only what the code under test reads."""
    model = object.__new__(QuickAdapterRegressorV3)
    model.ft_params = attributes.pop("ft_params", {})
    for name, value in attributes.items():
        setattr(model, name, value)
    return model


def candle_regressor(
    pairs: list[str], label_frequency_candles: int, seed: int
) -> QuickAdapterRegressorV3:
    """Build a regressor wired for the label candle pool, with a seeded shuffle RNG."""
    return label_regressor(
        pairs=list(pairs),
        ft_params={"label_frequency_candles": label_frequency_candles},
        _optuna_label_candle_pool_full_cache={},
        _optuna_label_shuffle_rng=random.Random(seed),
        _optuna_label_candle_pool=[],
        _optuna_label_candle={},
        _optuna_label_candles=dict.fromkeys(pairs, 0),
        _optuna_label_incremented_pairs=[],
    )


def optuna_config(**overrides: object) -> dict:
    """The subset of freqai.optuna_hyperopt that the sampler and pruner constructors read."""
    return {
        "sampler": "tpe",
        "label_sampler": "tpe",
        "n_startup_trials": 15,
        "n_jobs": 1,
        "seed": 1,
        "min_resource": 3,
        **overrides,
    }


def min_max_config(**overrides: object) -> dict:
    """A col_prediction_config with every key min_max_pred reads, plus the ``method`` gate."""
    return {
        "method": PREDICTION_METHODS[1],  # "thresholding"
        "selection_method": "rank_extrema",
        "threshold_method": SKIMAGE_THRESHOLD_METHODS[0],  # "mean"
        "outlier_quantile": 0.999,
        "soft_extremum_alpha": 12.0,
        "keep_fraction": 1.0,
        **overrides,
    }


def floored_pseudo_inverse(reference_matrix: np.ndarray) -> np.ndarray:
    """Reproduce the eigenvalue-floored inverse the mahalanobis branch is specified to build."""
    covariance = np.atleast_2d(np.cov(reference_matrix, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = eigenvalues[-1] * np.sqrt(np.finfo(float).eps)
    return (eigenvectors / np.maximum(eigenvalues, floor)) @ eigenvectors.T


def json_roundtrip(metadata: dict) -> object:
    """Round-trip through json so a numpy value in the marker cannot hide behind dict equality."""
    return json.loads(json.dumps(metadata))


class RegressorLabelConfigTest(QaTestCase):
    def test_unknown_label_method_raises_with_the_canonical_enum_message(self):
        with self.assertRaises(ValueError) as caught:
            label_regressor()._resolve_label_method_config("kde")
        self.assertEqual(
            str(caught.exception),
            enum_error_message("label_method", "kde", QuickAdapterRegressorV3._SELECTION_METHODS),
        )

    def test_category_defaults_are_resolved_per_method(self):
        expected = {
            "topsis": {
                "category": "distance",
                "method": "topsis",
                "distance_metric": "euclidean",
            },
            "kmeans": {
                "category": "cluster",
                "method": "kmeans",
                "distance_metric": "euclidean",
                "selection_method": "topsis",
                "trial_selection_method": "topsis",
            },
            "knn": {
                "category": "density",
                "method": "knn",
                "distance_metric": "minkowski",
                "aggregation": "power_mean",
                "n_neighbors": 5,
                "aggregation_param": 1.0,
            },
            "medoid": {
                "category": "density",
                "method": "medoid",
                "distance_metric": "euclidean",
            },
        }
        for method, config in expected.items():
            with self.subTest(method=method):
                resolved = label_regressor()._resolve_label_method_config(method)
                self.assertEqual(resolved, config)

    def test_distance_category_accepts_an_aggregate_metric(self):
        resolved = label_regressor(ft_params={"label_distance_metric": "weighted_sum"})
        self.assertEqual(
            resolved._resolve_label_method_config("topsis")["distance_metric"],
            "weighted_sum",
        )

    def test_bad_distance_metric_warns_and_substitutes_the_category_default(self):
        cases = {
            "topsis": ("label_distance_metric", "jaccard", "euclidean"),
            "kmeans": ("label_cluster_metric", "power_mean", "euclidean"),
            "knn": ("label_density_metric", "weighted_sum", "minkowski"),
            "medoid": ("label_density_metric", "hellinger", "euclidean"),
        }
        for method, (key, bad_metric, expected_default) in cases.items():
            with self.subTest(method=method):
                model = label_regressor(ft_params={key: bad_metric})
                with self.assertLogs(REGRESSOR_LOGGER, level="WARNING") as logs:
                    resolved = model._resolve_label_method_config(method)
                self.assertEqual(resolved["distance_metric"], expected_default)
                self.assertIn(f"Invalid {key} value {bad_metric!r}", "\n".join(logs.output))

    def test_cluster_and_cluster_trial_selection_methods_are_fatal(self):
        for key in ("label_cluster_selection_method", "label_cluster_trial_selection_method"):
            with self.subTest(key=key):
                model = label_regressor(ft_params={key: "kalman"})
                with self.assertRaises(ValueError) as caught:
                    model._resolve_label_method_config("kmeans")
                self.assertEqual(
                    str(caught.exception),
                    enum_error_message(key, "kalman", QuickAdapterRegressorV3._DISTANCE_METHODS),
                )

    def test_aggregate_metric_is_rejected_for_cluster_and_density_categories(self):
        aggregates = set(QuickAdapterRegressorV3._POWER_MEAN_MAP) | {"weighted_sum"}
        for aggregate in aggregates:
            with self.subTest(aggregate=aggregate):
                with self.assertRaises(ValueError) as caught:
                    QuickAdapterRegressorV3._validate_label_selection_metric(
                        aggregate,
                        ctx="label_cluster_metric",
                        default="euclidean",
                        aggregate_allowed=False,
                        mode="raise",
                    )
                self.assertEqual(
                    str(caught.exception),
                    enum_error_message(
                        "label_cluster_metric",
                        aggregate,
                        tuple(
                            candidate
                            for candidate in QuickAdapterRegressorV3._DISTANCE_METRICS
                            if candidate
                            in QuickAdapterRegressorV3._CLUSTER_DENSITY_DISTANCE_METRICS_SET
                        ),
                    ),
                )

    def test_probability_metrics_are_rejected_for_cluster_and_density_categories(self):
        for probability_metric in QuickAdapterRegressorV3._PROBABILITY_DISTANCE_METRICS_SET:
            with self.subTest(metric=probability_metric):
                self.assertEqual(
                    QuickAdapterRegressorV3._validate_label_selection_metric(
                        probability_metric,
                        ctx="label_density_metric",
                        default="minkowski",
                        aggregate_allowed=False,
                        mode="none",
                    ),
                    "minkowski",
                )

    def test_knn_density_aggregation_and_neighbour_bounds_are_fatal(self):
        cases = {
            "label_density_aggregation": ("median", "supported values are power_mean"),
            "label_density_n_neighbors": (0, "must be int >= 1"),
            "label_density_aggregation_param": (1.5, "must be in \\[0, 1\\]"),
        }
        for key, (value, fragment) in cases.items():
            with self.subTest(key=key):
                ft_params = {"label_density_aggregation": "quantile"}
                ft_params[key] = value
                model = label_regressor(ft_params=ft_params)
                with self.assertRaisesRegex(ValueError, fragment):
                    model._resolve_label_method_config("knn")

    def test_knn_aggregation_param_defaults_follow_the_aggregation(self):
        expected = {
            "power_mean": 1.0,
            "quantile": 0.5,
            "min": None,
            "max": None,
        }
        for aggregation, default in expected.items():
            with self.subTest(aggregation=aggregation):
                model = label_regressor(ft_params={"label_density_aggregation": aggregation})
                resolved = model._resolve_label_method_config("knn")
                self.assertEqual(resolved["aggregation"], aggregation)
                self.assertEqual(resolved["aggregation_param"], default)

    def test_knn_aggregation_param_is_taken_from_the_config_when_present(self):
        model = label_regressor(
            ft_params={
                "label_density_aggregation": "quantile",
                "label_density_aggregation_param": 0.25,
            }
        )
        self.assertEqual(model._resolve_label_method_config("knn")["aggregation_param"], 0.25)

    def test_resolve_p_order_applies_the_minkowski_power_default_per_category(self):
        cases = {
            "minkowski": 2.0,
            "power_mean": 1.0,
            "euclidean": None,
            "cityblock": None,
        }
        for metric, expected in cases.items():
            with self.subTest(metric=metric):
                self.assertEqual(
                    QuickAdapterRegressorV3._resolve_p_order(metric, None, ctx="label_p_order"),
                    expected,
                )

    def test_resolve_p_order_prefers_the_configured_value_over_the_default(self):
        for metric in ("minkowski", "power_mean", "euclidean"):
            with self.subTest(metric=metric):
                self.assertEqual(
                    QuickAdapterRegressorV3._resolve_p_order(metric, 7.5, ctx="label_p_order"),
                    7.5,
                )

    def test_resolve_p_order_refuses_a_non_positive_minkowski_power(self):
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            QuickAdapterRegressorV3._resolve_p_order(
                "minkowski", -1.0, ctx="label_p_order", mode="raise"
            )
        for mode in ("warn", "none"):
            with self.subTest(mode=mode):
                if mode == "warn":
                    with self.assertLogs(REGRESSOR_LOGGER, level="WARNING"):
                        resolved = QuickAdapterRegressorV3._resolve_p_order(
                            "minkowski", 0.0, ctx="label_p_order", mode=mode
                        )
                else:
                    resolved = QuickAdapterRegressorV3._resolve_p_order(
                        "minkowski", 0.0, ctx="label_p_order", mode=mode
                    )
                self.assertIsNone(resolved)

    def test_label_weights_are_normalized_to_a_unit_sum(self):
        cases = {
            (1.0, 3.0): [0.25, 0.75],
            (2.0, 2.0): [0.5, 0.5],
            (0.1, 0.2, 0.7): [0.1, 0.2, 0.7],
            (1e-9, 1e-9): [0.5, 0.5],
            (0.0, 5.0): [0.0, 1.0],
            # Dividing by the maximum before summing is what keeps a weight vector
            # that would overflow on summation from collapsing to zero.
            (1e308, 1e308): [0.5, 0.5],
            (1e-320, 1e-320): [0.5, 0.5],
        }
        for weights, expected in cases.items():
            with self.subTest(weights=weights):
                resolved = QuickAdapterRegressorV3._validate_label_weights(
                    list(weights), len(weights), ctx="label_weights"
                )
                np.testing.assert_allclose(resolved, np.array(expected), rtol=1e-12, atol=0.0)
                np.testing.assert_allclose(resolved.sum(), 1.0, rtol=1e-12, atol=0.0)

    def test_absent_label_weights_resolve_to_the_uniform_vector(self):
        for n_objectives in (1, 2, 4, 7):
            with self.subTest(n_objectives=n_objectives):
                resolved = QuickAdapterRegressorV3._validate_label_weights(
                    None, n_objectives, ctx="label_weights"
                )
                np.testing.assert_allclose(
                    resolved,
                    np.full(n_objectives, 1.0 / n_objectives),
                    rtol=1e-12,
                    atol=0.0,
                )

    def test_label_weights_refuse_non_finite_and_non_positive_vectors(self):
        cases = {
            (np.inf, 1.0): "contains non-finite values",
            (np.nan, 1.0): "contains non-finite values",
            (-1.0, 2.0): "contains negative values",
            (0.0, 0.0): "sum is zero",
        }
        for weights, fragment in cases.items():
            with (
                self.subTest(weights=weights),
                self.assertRaisesRegex(ValueError, fragment),
            ):
                QuickAdapterRegressorV3._validate_label_weights(
                    list(weights), 2, ctx="label_weights"
                )

    def test_label_weights_refuse_a_wrong_length_or_a_non_vector(self):
        with self.assertRaisesRegex(ValueError, "must contain 2 weights"):
            QuickAdapterRegressorV3._validate_label_weights([1.0], 2, ctx="label_weights")
        with self.assertRaisesRegex(ValueError, "must be a one-dimensional vector"):
            QuickAdapterRegressorV3._validate_label_weights([[1.0, 2.0]], 2, ctx="label_weights")
        with self.assertRaisesRegex(ValueError, "must be a list, tuple, or array"):
            QuickAdapterRegressorV3._validate_label_weights("0.5,0.5", 2, ctx="label_weights")

    def test_downgraded_label_weights_fall_back_to_uniform(self):
        for mode in ("warn", "none"):
            for weights in ([np.inf, 1.0], [-1.0, 1.0], [1.0]):
                with self.subTest(mode=mode, weights=weights):
                    if mode == "warn":
                        with self.assertLogs(REGRESSOR_LOGGER, level="WARNING"):
                            resolved = QuickAdapterRegressorV3._validate_label_weights(
                                weights, 2, ctx="label_weights", mode=mode
                            )
                    else:
                        resolved = QuickAdapterRegressorV3._validate_label_weights(
                            weights, 2, ctx="label_weights", mode=mode
                        )
                    np.testing.assert_allclose(resolved, np.full(2, 0.5), rtol=1e-12, atol=0.0)

    def test_label_weights_never_mutate_the_caller_array(self):
        caller = np.array([1.0, 3.0])
        resolved = QuickAdapterRegressorV3._validate_label_weights(caller, 2, ctx="label_weights")
        np.testing.assert_array_equal(caller, np.array([1.0, 3.0]))
        self.assertFalse(np.shares_memory(caller, resolved))

    def test_metric_weights_support_refuses_metrics_that_cannot_carry_weights(self):
        unsupported = QuickAdapterRegressorV3._UNSUPPORTED_WEIGHTS_METRICS_SET
        self.assertEqual(unsupported, {"mahalanobis", "seuclidean", "jensenshannon"})
        for metric in unsupported:
            with self.subTest(metric=metric):
                with self.assertRaisesRegex(ValueError, "does not support custom weights"):
                    QuickAdapterRegressorV3._validate_metric_weights_support(
                        metric, ctx="label_distance_metric", mode="raise"
                    )
                with self.assertLogs(REGRESSOR_LOGGER, level="WARNING"):
                    self.assertIsNone(
                        QuickAdapterRegressorV3._validate_metric_weights_support(
                            metric, ctx="label_distance_metric", mode="warn"
                        )
                    )
                self.assertIsNone(
                    QuickAdapterRegressorV3._validate_metric_weights_support(
                        metric, ctx="label_distance_metric", mode="none"
                    )
                )

    def test_metric_weights_support_passes_a_weight_capable_metric_through(self):
        weight_capable = QuickAdapterRegressorV3._SCIPY_METRICS_SET - {
            "seuclidean",
            "mahalanobis",
            "jensenshannon",
        }
        for metric in weight_capable:
            with self.subTest(metric=metric):
                self.assertEqual(
                    QuickAdapterRegressorV3._validate_metric_weights_support(
                        metric, ctx="label_distance_metric", mode="raise"
                    ),
                    metric,
                )

    def test_prepare_distance_kwargs_carries_weights_only_for_a_weight_capable_metric(self):
        weights = np.array([0.25, 0.75])
        with self.assertLogs(REGRESSOR_LOGGER, level="WARNING"):
            refused = QuickAdapterRegressorV3._prepare_distance_kwargs(
                "mahalanobis",
                weights,
                mode="warn",
                reference_matrix=REFERENCE_MATRIX,
            )
        self.assertNotIn("w", refused)
        accepted = QuickAdapterRegressorV3._prepare_distance_kwargs(
            "euclidean", weights, reference_matrix=REFERENCE_MATRIX
        )
        np.testing.assert_array_equal(accepted["w"], weights)
        self.assertEqual(
            QuickAdapterRegressorV3._prepare_distance_kwargs(
                "euclidean", reference_matrix=REFERENCE_MATRIX
            ),
            {},
        )

    def test_prepare_distance_kwargs_builds_the_seuclidean_variance_vector(self):
        kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
            "seuclidean", reference_matrix=REFERENCE_MATRIX
        )
        self.assertEqual(sorted(kwargs), ["V"])
        np.testing.assert_allclose(
            kwargs["V"],
            np.var(REFERENCE_MATRIX, axis=0, ddof=1),
            rtol=1e-12,
            atol=0.0,
        )

    def test_prepare_distance_kwargs_builds_the_mahalanobis_inverse_covariance(self):
        kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
            "mahalanobis", reference_matrix=REFERENCE_MATRIX
        )
        self.assertEqual(sorted(kwargs), ["VI"])
        np.testing.assert_allclose(
            kwargs["VI"], floored_pseudo_inverse(REFERENCE_MATRIX), rtol=1e-12, atol=0.0
        )
        np.testing.assert_allclose(kwargs["VI"], kwargs["VI"].T, rtol=1e-12, atol=0.0)

    def test_prepare_distance_kwargs_floors_a_collinear_eigenvalue_instead_of_inverting_it(self):
        kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
            "mahalanobis", reference_matrix=COLLINEAR_MATRIX
        )
        np.testing.assert_allclose(
            kwargs["VI"], floored_pseudo_inverse(COLLINEAR_MATRIX), rtol=1e-12, atol=0.0
        )
        self.assertTrue(np.all(np.isfinite(kwargs["VI"])))
        np.testing.assert_allclose(kwargs["VI"], kwargs["VI"].T, rtol=1e-12, atol=0.0)
        # A plain pseudo-inverse would drop the degenerate direction instead of flooring it.
        self.assertFalse(
            np.allclose(
                kwargs["VI"], np.linalg.pinv(np.cov(COLLINEAR_MATRIX, rowvar=False, ddof=1))
            )
        )

    def test_prepare_distance_kwargs_refuses_a_degenerate_reference_front(self):
        cases = {
            "single_row": (
                REFERENCE_MATRIX[:1],
                "finite full front with at least two rows",
            ),
            "non_finite": (
                np.array([[0.0, 0.0], [1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]]),
                "finite full front with at least two rows",
            ),
            "constant_objective": (
                np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
                "positive active-objective variances",
            ),
        }
        for name, (matrix, fragment) in cases.items():
            for metric in ("seuclidean", "mahalanobis"):
                with (
                    self.subTest(case=name, metric=metric),
                    self.assertRaisesRegex(ValueError, fragment),
                ):
                    QuickAdapterRegressorV3._prepare_distance_kwargs(
                        metric, reference_matrix=matrix
                    )

    def test_prepare_distance_kwargs_validates_the_minkowski_power_only_for_minkowski(self):
        self.assertEqual(
            QuickAdapterRegressorV3._prepare_distance_kwargs(
                "minkowski", p=3.0, reference_matrix=REFERENCE_MATRIX
            ),
            {"p": 3.0},
        )
        self.assertEqual(
            QuickAdapterRegressorV3._prepare_distance_kwargs(
                "minkowski", reference_matrix=REFERENCE_MATRIX
            ),
            {},
        )
        self.assertEqual(
            QuickAdapterRegressorV3._prepare_distance_kwargs(
                "euclidean", p=3.0, reference_matrix=REFERENCE_MATRIX
            ),
            {},
        )
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            QuickAdapterRegressorV3._prepare_distance_kwargs(
                "minkowski", p=-1.0, mode="raise", reference_matrix=REFERENCE_MATRIX
            )

    def test_selection_metadata_is_json_native_and_defaults_to_the_distance_category(self):
        metadata = label_regressor()._optuna_label_selection_metadata()
        self.assertEqual(metadata["schema_version"], _OPTUNA_LABEL_SELECTION_SCHEMA_VERSION)
        self.assertIsNone(metadata["label_weights"])
        self.assertIsNone(metadata["label_p_order"])
        self.assertEqual(
            metadata["method_config"],
            {
                "category": "distance",
                "method": "compromise_programming",
                "distance_metric": "euclidean",
            },
        )
        self.assertIsInstance(json_roundtrip(metadata), dict)

    def test_selection_metadata_coerces_numpy_weights_and_p_order_to_plain_floats(self):
        metadata = label_regressor(
            ft_params={
                "label_method": "kmeans",
                "label_weights": np.array([0.25, 0.75]),
                "label_p_order": np.float64(3.0),
            }
        )._optuna_label_selection_metadata()
        self.assertEqual(metadata["label_weights"], [0.25, 0.75])
        self.assertIs(type(metadata["label_weights"][0]), float)
        self.assertIs(type(metadata["label_p_order"]), float)
        self.assertEqual(metadata["label_p_order"], 3.0)
        self.assertEqual(metadata["method_config"]["category"], "cluster")
        self.assertIsInstance(json_roundtrip(metadata), dict)

    def test_selection_metadata_refuses_non_finite_weights_and_p_order(self):
        cases = {
            "label_weights": (
                {"label_weights": [0.5, np.inf]},
                "label_weights contains non-finite",
            ),
            "label_p_order": (
                {"label_p_order": float("nan")},
                "label_p_order is non-finite",
            ),
        }
        for name, (ft_params, fragment) in cases.items():
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, fragment):
                label_regressor(ft_params=ft_params)._optuna_label_selection_metadata()

    def test_optuna_create_sampler_maps_every_enum_to_its_sampler(self):
        model = label_regressor(_optuna_config=optuna_config())
        for sampler, expected_type in (
            ("tpe", "TPESampler"),
            ("nsgaii", "NSGAIISampler"),
            ("nsgaiii", "NSGAIIISampler"),
        ):
            with self.subTest(sampler=sampler):
                self.assertEqual(type(model.optuna_create_sampler(sampler)).__name__, expected_type)

    def test_optuna_create_sampler_defaults_to_the_configured_sampler(self):
        model = label_regressor(_optuna_config=optuna_config())
        self.assertEqual(type(model.optuna_create_sampler()).__name__, "TPESampler")
        self.assertEqual(type(model.optuna_create_sampler(None)).__name__, "TPESampler")

    def test_optuna_create_sampler_reads_the_tpe_startup_trials_and_parallel_constant_liar(self):
        single = label_regressor(_optuna_config=optuna_config(n_startup_trials=9, n_jobs=1))
        parallel = label_regressor(_optuna_config=optuna_config(n_startup_trials=9, n_jobs=4))
        self.assertEqual(single.optuna_create_sampler("tpe")._n_startup_trials, 9)
        self.assertFalse(single.optuna_create_sampler("tpe")._constant_liar)
        self.assertTrue(parallel.optuna_create_sampler("tpe")._constant_liar)

    def test_optuna_create_sampler_resolves_auto_through_the_patched_optunahub(self):
        model = label_regressor(_optuna_config=optuna_config(seed=42))
        with mock.patch.object(
            regressor_module.optunahub, "load_module", return_value=mock.MagicMock()
        ) as load_module:
            model.optuna_create_sampler("auto")
        load_module.assert_called_once_with("samplers/auto_sampler")

    def test_optuna_create_sampler_reports_a_missing_sampler_through_the_enum_message(self):
        model = label_regressor(_optuna_config=optuna_config(sampler=None))
        with self.assertRaises(ValueError) as caught:
            model.optuna_create_sampler(None)
        self.assertEqual(
            str(caught.exception),
            enum_error_message(
                "optuna sampler",
                None,
                QuickAdapterRegressorV3._OPTUNA_SAMPLERS,
            ),
        )

    def test_optuna_create_sampler_refuses_an_unmapped_sampler_by_exhaustiveness(self):
        model = label_regressor(_optuna_config=optuna_config())
        with self.assertRaises(AssertionError):
            model.optuna_create_sampler("cmaes")

    def test_optuna_create_pruner_follows_the_single_objective_flag(self):
        model = label_regressor(_optuna_config=optuna_config(min_resource=9))
        for flag in (True, 1):
            with self.subTest(flag=flag):
                pruner = model.optuna_create_pruner(flag)
                self.assertEqual(type(pruner).__name__, "HyperbandPruner")
                self.assertEqual(pruner._min_resource, 9)
        for flag in (False, 0):
            with self.subTest(flag=flag):
                self.assertEqual(type(model.optuna_create_pruner(flag)).__name__, "NopPruner")

    def test_optuna_samplers_by_namespace_keeps_the_two_namespaces_apart(self):
        model = label_regressor(
            _optuna_config=optuna_config(sampler="nsgaii", label_sampler="nsgaiii")
        )
        hp_allowed, hp_sampler = model.optuna_samplers_by_namespace(_OPTUNA_NAMESPACES.hp)
        label_allowed, label_sampler = model.optuna_samplers_by_namespace(_OPTUNA_NAMESPACES.label)
        self.assertEqual(hp_sampler, "nsgaii")
        self.assertEqual(label_sampler, "nsgaiii")
        self.assertEqual(hp_allowed, frozenset({"tpe", "auto"}))
        self.assertEqual(label_allowed, frozenset({"tpe", "auto", "nsgaii", "nsgaiii"}))
        self.assertTrue(hp_allowed < label_allowed)

    def test_optuna_samplers_by_namespace_refuses_an_unknown_namespace(self):
        model = label_regressor(_optuna_config=optuna_config())
        with self.assertRaises(ValueError) as caught:
            model.optuna_samplers_by_namespace("hp_label")
        self.assertEqual(
            str(caught.exception),
            enum_error_message("namespace", "hp_label", _OPTUNA_NAMESPACES),
        )

    def test_candle_pool_full_is_centred_on_the_label_frequency(self):
        for frequency, expected in (
            (2, [1, 2, 3]),
            (3, [2, 3, 4]),
            (4, [2, 3, 4, 5, 6]),
            (6, [3, 4, 5, 6, 7, 8, 9]),
            (9, [5, 6, 7, 8, 9, 10, 11, 12, 13]),
        ):
            with self.subTest(frequency=frequency):
                model = candle_regressor([PAIR], frequency, seed=1)
                self.assertEqual(model._optuna_label_candle_pool_full, expected)
                self.assertEqual(model._optuna_label_candle_pool_full_cache[frequency], expected)

    def test_init_candle_pool_preserves_the_full_pool_and_is_seed_reproducible(self):
        pairs = [PAIR, "ETH/USDT"]
        first = candle_regressor(pairs, 8, seed=3)
        second = candle_regressor(pairs, 8, seed=3)
        other_seed = candle_regressor(pairs, 8, seed=4)
        first.init_optuna_label_candle_pool()
        second.init_optuna_label_candle_pool()
        other_seed.init_optuna_label_candle_pool()
        self.assertEqual(first._optuna_label_candle_pool, second._optuna_label_candle_pool)
        self.assertNotEqual(first._optuna_label_candle_pool, other_seed._optuna_label_candle_pool)
        self.assertEqual(
            sorted(first._optuna_label_candle_pool),
            sorted(first._optuna_label_candle_pool_full),
        )
        # The cache backs the property, so re-initialising must not have consumed it.
        self.assertEqual(
            first._optuna_label_candle_pool_full_cache[8],
            first._optuna_label_candle_pool_full,
        )

    def test_candle_pool_fires_once_per_pair_and_rearms_after_the_budget(self):
        pairs = [PAIR, "ETH/USDT", "SOL/USDT"]
        model = candle_regressor(pairs, 6, seed=11)
        full = set(model._optuna_label_candle_pool_full)
        model.init_optuna_label_candle_pool()

        for pair in pairs:
            model.set_optuna_label_candle(pair)
            self.assertIn(model._optuna_label_candle[pair], full)

        self.assertEqual(len(model._optuna_label_candle), len(pairs))
        self.assertEqual(
            len(set(model._optuna_label_candle.values())),
            len(pairs),
            "each pair must hold a distinct candle while the pool still has the budget",
        )
        self.assertEqual(len(model._optuna_label_candle_pool), len(full) - len(pairs))

        # A further call for an already-assigned pair releases its old candle back
        # into the pool rather than drawing a second one for that pair.
        released = model._optuna_label_candle[pairs[0]]
        model.set_optuna_label_candle(pairs[0])
        self.assertEqual(len(model._optuna_label_candle), len(pairs))
        self.assertEqual(len(model._optuna_label_candle_pool), len(full) - len(pairs))
        self.assertEqual(
            set(model._optuna_label_candle_pool),
            full - set(model._optuna_label_candle.values()),
        )
        self.assertIn(released, model._optuna_label_candle_pool)

    def test_candle_pool_draws_the_last_available_candle(self):
        model = candle_regressor([PAIR], 4, seed=1)
        model._optuna_label_candle_pool_full_cache = {4: [1, 2, 3]}
        model._optuna_label_candle_pool = [1, 2, 3]
        model.set_optuna_label_candle(PAIR)
        self.assertEqual(model._optuna_label_candle[PAIR], 3)
        self.assertEqual(sorted(model._optuna_label_candle_pool), [1, 2])

    def test_candle_pool_reserves_a_candle_another_pair_still_needs(self):
        pairs = [PAIR, "ETH/USDT"]
        model = candle_regressor(pairs, 4, seed=1)
        model._optuna_label_candle_pool_full_cache = {4: [1, 2, 3]}
        # The first pair has taken candle 3 and burned one candle against it, so it
        # still needs 2. The second pair must not be dealt the same value.
        model._optuna_label_candle[PAIR] = 3
        model._optuna_label_candles[PAIR] = 1
        model._optuna_label_candle_pool = [1, 2]
        model.set_optuna_label_candle("ETH/USDT")
        self.assertEqual(model._optuna_label_candle["ETH/USDT"], 1)
        self.assertEqual(sorted(model._optuna_label_candle_pool), [2])

    def test_candle_pool_reshuffles_after_refilling_available_candles(self):
        model = candle_regressor([PAIR], 6, seed=1)
        model._optuna_label_candle_pool_full_cache = {6: [3, 4, 5, 6, 7, 8, 9]}
        # Candle 3 is spent and 9 is unclaimed, so the refill puts 9 back and
        # reshuffles: the pool must not come back in sorted order.
        model._optuna_label_candle_pool = [3, 4, 5, 6, 7, 8]
        model._optuna_label_candle = {PAIR: 3}
        model.set_optuna_label_candle(PAIR)
        self.assertEqual(model._optuna_label_candle[PAIR], 8)
        self.assertEqual(sorted(model._optuna_label_candle_pool), [3, 4, 5, 6, 7, 9])
        self.assertNotEqual(
            model._optuna_label_candle_pool,
            sorted(model._optuna_label_candle_pool),
            "a refilled pool must be reshuffled, not left in sorted order",
        )

    def test_candle_pool_rearms_after_being_drained(self):
        model = candle_regressor([PAIR], 2, seed=1)
        full = model._optuna_label_candle_pool_full
        self.assertEqual(full, [1, 2, 3])
        model.init_optuna_label_candle_pool()
        for _ in range(len(full)):
            model.set_optuna_label_candle(PAIR)
            self.assertIn(model._optuna_label_candle[PAIR], full)

        model._optuna_label_candle_pool = []
        with self.assertLogs(REGRESSOR_LOGGER, level="WARNING") as logs:
            model.set_optuna_label_candle(PAIR)
        self.assertIn("pool is empty, reinitializing", "\n".join(logs.output))
        self.assertEqual(len(model._optuna_label_candle_pool), len(full) - 1)
        self.assertEqual(
            set(model._optuna_label_candle_pool) | {model._optuna_label_candle[PAIR]},
            set(full),
        )

    def test_init_candle_pool_refuses_a_pool_emptied_by_shuffle(self):
        model = candle_regressor([PAIR], 4, seed=1)
        model._optuna_label_shuffle_rng = mock.MagicMock()
        model._optuna_label_shuffle_rng.shuffle.side_effect = lambda pool: pool.clear()
        with self.assertRaisesRegex(RuntimeError, "pool became empty after shuffle"):
            model.init_optuna_label_candle_pool()

    def test_init_candle_pool_refuses_an_empty_initial_pool(self):
        model = candle_regressor([PAIR], 4, seed=1)
        model._optuna_label_candle_pool_full_cache = {4: []}
        with self.assertRaisesRegex(RuntimeError, "initial pool is empty"):
            model.init_optuna_label_candle_pool()

    def test_construction_arms_the_pool_without_tripping_the_rearm_warning(self):
        # __init__ initialises the pool itself; only set_optuna_label_candle re-arms
        # a drained one, so a clean construction must log no re-arm warning.
        with (
            temporary_directory() as temp,
            mock.patch.object(
                regressor_module.logger, "warning", wraps=regressor_module.logger.warning
            ) as warning,
        ):
            QuickAdapterRegressorV3(
                config=model_config(
                    temp,
                    freqai={
                        "data_split_parameters": {
                            "test_size": 0.2,
                            "shuffle": False,
                        }
                    },
                )
            )
        self.assertNotIn(
            mock.call(f"[{PAIR}] Optuna label candle pool is empty, reinitializing"),
            warning.mock_calls,
        )

    def test_min_max_pred_rounds_the_threshold_window_to_label_period_multiples(self):
        model = label_regressor(ft_params={"label_period_candles": 4})
        model._label_defaults = (18, 10.5)
        predictions = pd.DataFrame({"&s-extrema": [float(i) for i in range(100)]})
        cases = {
            (5, 3): 6,
            (5, 1): 5,
            (7, 4): 8,
            (5, 2): 4,
            (100, 7): 98,
        }
        for (fit_candles, period), window in cases.items():
            with self.subTest(fit_candles=fit_candles, label_period_candles=period):
                minimum, maximum = model.min_max_pred(
                    "&s-extrema", min_max_config(), predictions, fit_candles, period
                )
                self.assertEqual(maximum, 99.0)
                self.assertEqual(minimum, float(len(predictions) - window))

    def test_min_max_pred_falls_back_to_the_configured_label_period(self):
        model = label_regressor(ft_params={"label_period_candles": 4})
        model._label_defaults = (18, 10.5)
        predictions = pd.DataFrame({"&s-extrema": [float(i) for i in range(100)]})
        for unusable in (None, 0, -2):
            with self.subTest(label_period_candles=unusable):
                self.assertEqual(
                    model.min_max_pred("&s-extrema", min_max_config(), predictions, 5, unusable),
                    (92.0, 99.0),
                )

    def test_min_max_pred_returns_the_no_entry_signal_pair_unchanged(self):
        model = label_regressor(ft_params={"label_period_candles": 4})
        model._label_defaults = (18, 10.5)
        predictions = pd.DataFrame({"&s-extrema": [float(i) for i in range(100)]})
        self.assertEqual(
            model.min_max_pred(
                "&s-extrema",
                min_max_config(threshold_method="rolling_quantile"),
                predictions,
                5,
                3,
            ),
            NO_ENTRY_SIGNALS,
        )
        self.assertEqual(
            model.min_max_pred("&s-other", min_max_config(), predictions, 5, 3),
            NO_ENTRY_SIGNALS,
        )

    def test_min_max_pred_leaves_the_method_none_gate_to_its_caller(self):
        # ``fit_live_predictions`` skips a ``method="none"`` column before it ever
        # calls min_max_pred, so min_max_pred itself is indifferent to the method
        # key: it dispatches on threshold_method alone.
        model = label_regressor(ft_params={"label_period_candles": 4})
        model._label_defaults = (18, 10.5)
        predictions = pd.DataFrame({"&s-extrema": [float(i) for i in range(100)]})
        self.assertEqual(
            model.min_max_pred("&s-extrema", min_max_config(), predictions, 5, 3),
            model.min_max_pred(
                "&s-extrema",
                min_max_config(method=PREDICTION_METHODS[0]),
                predictions,
                5,
                3,
            ),
        )

    def test_min_max_pred_dispatches_every_supported_threshold_method(self):
        model = label_regressor(ft_params={"label_period_candles": 4})
        model._label_defaults = (18, 10.5)
        values = [float(i) * 0.5 for i in range(40)]
        predictions = pd.DataFrame({"&s-extrema": values})
        for threshold_method in (*SKIMAGE_THRESHOLD_METHODS, *CUSTOM_THRESHOLD_METHODS):
            with self.subTest(threshold_method=threshold_method):
                minimum, maximum = model.min_max_pred(
                    "&s-extrema",
                    min_max_config(threshold_method=threshold_method),
                    predictions,
                    20,
                    2,
                )
                self.assertLess(minimum, maximum)
                self.assertGreaterEqual(maximum, max(values[-20:]))

    def test_strategy_optuna_load_best_params_tolerates_selection_metadata_drift(self):
        strategy = object.__new__(QuickAdapterV3)
        strategy.models_full_path = Path("/nonexistent/models")
        strategy.pairs = [PAIR]
        with mock.patch.object(
            strategy_module, "optuna_load_best_params", return_value=None
        ) as loader:
            strategy.optuna_load_best_params(PAIR, _OPTUNA_NAMESPACES.label)
        loader.assert_called_once()
        arguments, keywords = loader.call_args
        self.assertEqual(arguments[1:3], (PAIR, _OPTUNA_NAMESPACES.label))
        self.assertEqual(keywords, {"pairs": [PAIR]})
        self.assertNotIn("expected_selection_metadata", keywords)
        self.assertNotIn("expected_objective_identity", keywords)

    def test_regressor_optuna_load_best_params_pins_the_expected_selection_metadata(self):
        model = label_regressor(full_path=Path("/nonexistent/models"), pairs=[PAIR])
        with mock.patch.object(
            regressor_module, "optuna_load_best_params", return_value=None
        ) as loader:
            model.optuna_load_best_params(PAIR, _OPTUNA_NAMESPACES.label)
            _, keywords = loader.call_args
            self.assertEqual(
                keywords["expected_selection_metadata"],
                model._optuna_label_selection_metadata(),
            )
            self.assertIsNone(keywords["expected_objective_identity"])
            model.optuna_load_best_params(PAIR, _OPTUNA_NAMESPACES.hp)
            _, keywords = loader.call_args
            self.assertIsNone(keywords["expected_selection_metadata"])
            self.assertEqual(
                keywords["expected_objective_identity"],
                QuickAdapterRegressorV3._OPTUNA_HP_OBJECTIVE_IDENTITY,
            )

    def test_construction_from_the_shared_config_pins_the_config_boundary(self):
        with temporary_directory() as temp:
            model = QuickAdapterRegressorV3(config=model_config(temp))
        self.assertEqual(model.pairs, [PAIR])
        self.assertEqual(model._optuna_config["sampler"], "tpe")
        self.assertEqual(model._optuna_config["label_sampler"], "tpe")
        # test_size == 0 disables HPO, so the candle pool is never armed at construction.
        self.assertFalse(model._optuna_hyperopt)
        self.assertEqual(model._optuna_label_candle_pool, [])
        self.assertEqual(model._optuna_label_candle_pool_full, [1, 2, 3])
        self.assertEqual(model._optuna_label_candle, {})
        self.assertEqual(model._optuna_label_candles, {PAIR: 0})
        self.assertEqual(model._optuna_label_incremented_pairs, [])
        self.assertEqual(
            model._optuna_label_selection_metadata()["method_config"]["method"],
            QuickAdapterRegressorV3.LABEL_METHOD_DEFAULT,
        )

    def test_construction_arms_the_candle_pool_when_hyperopt_is_enabled(self):
        with temporary_directory() as temp:
            model = QuickAdapterRegressorV3(
                config=model_config(
                    temp,
                    freqai={"data_split_parameters": {"test_size": 0.2, "shuffle": False}},
                )
            )
        self.assertTrue(model._optuna_hyperopt)
        # The one pair is dealt its candle during __init__, so the pool is short by one.
        self.assertEqual(model._optuna_label_candle_pool_full, [1, 2, 3])
        self.assertEqual(sorted(model._optuna_label_candle), [PAIR])
        self.assertIn(model._optuna_label_candle[PAIR], model._optuna_label_candle_pool_full)
        self.assertEqual(
            sorted(model._optuna_label_candle_pool),
            sorted(set(model._optuna_label_candle_pool_full) - {model._optuna_label_candle[PAIR]}),
        )


if __name__ == "__main__":
    unittest.main()
