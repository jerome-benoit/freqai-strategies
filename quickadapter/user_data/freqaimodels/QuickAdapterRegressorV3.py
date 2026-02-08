import copy
import json
import logging
import random
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import AbstractSet, Any, Callable, Final, Literal, Optional, Union, cast

import numpy as np
import optuna
import optunahub
import pandas as pd
import scipy as sp
import skimage
import sklearn
from datasieve.pipeline import Pipeline
from datasieve.transforms import SKLearnWrapper
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from numpy.typing import NDArray
from optuna.study.study import ObjectiveFuncType
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn_extra.cluster import KMedoids

from LabelTransformer import (
    CUSTOM_THRESHOLD_METHODS,
    EXTREMA_SELECTION_METHODS,
    PREDICTION_METHODS,
    SKIMAGE_THRESHOLD_METHODS,
    THRESHOLD_METHODS,
    CustomThresholdMethod,
    ExtremaSelectionMethod,
    LabelTransformer,
    SkimageThresholdMethod,
    ThresholdMethod,
    get_label_column_config,
)

from Utils import (
    DEFAULT_FIT_LIVE_PREDICTIONS_CANDLES,
    DEFAULTS_LABEL_PREDICTION,
    LABEL_COLUMNS,
    REGRESSORS,
    Regressor,
    eval_set_and_weights,
    fit_regressor,
    format_dict,
    format_number,
    get_label_defaults,
    get_label_pipeline_config,
    get_label_prediction_config,
    get_min_max_label_period_candles,
    get_optuna_study_model_parameters,
    migrate_config,
    soft_extremum,
    zigzag,
)

OptunaSampler = Literal["tpe", "auto", "nsgaii", "nsgaiii"]
OptunaNamespace = Literal["hp", "label"]
ScalerType = Literal["minmax", "maxabs", "standard", "robust"]
DensityAggregation = Literal["power_mean", "quantile", "min", "max"]
DistanceMethod = Literal["compromise_programming", "topsis"]
ClusterMethod = Literal["kmeans", "kmeans2", "kmedoids"]
DensityMethod = Literal["knn", "medoid"]
SelectionMethod = Union[DistanceMethod, ClusterMethod, DensityMethod]
ValidationMode = Literal["warn", "raise", "none"]
warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class QuickAdapterRegressorV3(BaseRegressionModel):
    """
    The following freqaimodel is released to sponsors of the non-profit FreqAI open-source project.
    If you find the FreqAI project useful, please consider supporting it by becoming a sponsor.
    We use sponsor money to help stimulate new features and to pay for running these public
    experiments, with a an objective of helping the community make smarter choices in their
    ML journey.

    This freqaimodel is experimental (as with all models released to sponsors). Do *not* expect
    returns. The goal is to demonstrate gratitude to people who support the project and to
    help them find a good starting point for their own creativity.

    If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

    https://github.com/sponsors/robcaulk
    """

    version = "3.11.1"

    _TEST_SIZE: Final[float] = 0.1

    _SQRT_2: Final[float] = np.sqrt(2.0)

    _OPTUNA_LABEL_N_OBJECTIVES: Final[int] = 7
    _OPTUNA_LABEL_DIRECTIONS: Final[tuple[optuna.study.StudyDirection, ...]] = (
        optuna.study.StudyDirection.MAXIMIZE,
    ) * _OPTUNA_LABEL_N_OBJECTIVES
    _OPTUNA_STORAGE_BACKENDS: Final[tuple[str, ...]] = ("file", "sqlite")
    _OPTUNA_SAMPLERS: Final[tuple[OptunaSampler, ...]] = (
        "tpe",
        "auto",
        "nsgaii",
        "nsgaiii",
    )
    _OPTUNA_HPO_SAMPLERS: Final[tuple[OptunaSampler, ...]] = _OPTUNA_SAMPLERS[:2]
    _OPTUNA_LABEL_SAMPLERS: Final[tuple[OptunaSampler, ...]] = (
        _OPTUNA_SAMPLERS[1],  # "auto"
        _OPTUNA_SAMPLERS[0],  # "tpe"
        _OPTUNA_SAMPLERS[2],  # "nsgaii"
        _OPTUNA_SAMPLERS[3],  # "nsgaiii"
    )
    _OPTUNA_NAMESPACES: Final[tuple[OptunaNamespace, ...]] = ("hp", "label")

    _SCALER_TYPES: Final[tuple[ScalerType, ...]] = (
        "minmax",
        "maxabs",
        "standard",
        "robust",
    )

    SCALER_DEFAULT: Final[ScalerType] = _SCALER_TYPES[0]  # "minmax"
    RANGE_DEFAULT: Final[tuple[float, float]] = (-1.0, 1.0)

    _DISTANCE_METHODS: Final[tuple[DistanceMethod, ...]] = (
        "compromise_programming",
        "topsis",
    )
    _CLUSTER_METHODS: Final[tuple[ClusterMethod, ...]] = (
        "kmeans",
        "kmeans2",
        "kmedoids",
    )
    _DENSITY_METHODS: Final[tuple[DensityMethod, ...]] = ("knn", "medoid")

    _SELECTION_CATEGORIES: Final[dict[str, tuple[SelectionMethod, ...]]] = {
        "distance": _DISTANCE_METHODS,
        "cluster": _CLUSTER_METHODS,
        "density": _DENSITY_METHODS,
    }

    _SELECTION_METHODS: Final[tuple[SelectionMethod, ...]] = (
        *_DISTANCE_METHODS,
        *_CLUSTER_METHODS,
        *_DENSITY_METHODS,
    )

    _DISTANCE_METRICS: Final[tuple[str, ...]] = (
        "euclidean",
        "minkowski",
        "chebyshev",
        "cityblock",
        "sqeuclidean",
        "seuclidean",
        "mahalanobis",
        "jensenshannon",
        "hellinger",
        "shellinger",
        "harmonic_mean",
        "geometric_mean",
        "arithmetic_mean",
        "quadratic_mean",
        "cubic_mean",
        "power_mean",
        "weighted_sum",
    )

    _UNSUPPORTED_WEIGHTS_METRICS: Final[tuple[str, ...]] = (
        _DISTANCE_METRICS[6],  # "mahalanobis"
        _DISTANCE_METRICS[5],  # "seuclidean"
        _DISTANCE_METRICS[7],  # "jensenshannon"
    )

    _DENSITY_AGGREGATIONS: Final[tuple[DensityAggregation, ...]] = (
        "power_mean",
        "quantile",
        "min",
        "max",
    )

    _POWER_MEAN_MAP: Final[dict[str, float]] = {
        "harmonic_mean": -1.0,
        "geometric_mean": 0.0,
        "arithmetic_mean": 1.0,
        "quadratic_mean": 2.0,
        "cubic_mean": 3.0,
    }

    FIT_LIVE_PREDICTIONS_CANDLES_DEFAULT: Final[int] = (
        DEFAULT_FIT_LIVE_PREDICTIONS_CANDLES
    )
    MIN_LABEL_PERIOD_CANDLES_DEFAULT: Final[int] = 12
    MAX_LABEL_PERIOD_CANDLES_DEFAULT: Final[int] = 24
    MIN_LABEL_NATR_MULTIPLIER_DEFAULT: Final[float] = 9.0
    MAX_LABEL_NATR_MULTIPLIER_DEFAULT: Final[float] = 12.0

    LABEL_METHOD_DEFAULT: Final[str] = _SELECTION_METHODS[0]  # "compromise_programming"

    LABEL_DISTANCE_METRIC_DEFAULT: Final[str] = _DISTANCE_METRICS[0]  # "euclidean"

    LABEL_CLUSTER_METRIC_DEFAULT: Final[str] = _DISTANCE_METRICS[0]  # "euclidean"
    LABEL_CLUSTER_SELECTION_METHOD_DEFAULT: Final[DistanceMethod] = _DISTANCE_METHODS[
        1
    ]  # "topsis"
    LABEL_CLUSTER_TRIAL_SELECTION_METHOD_DEFAULT: Final[DistanceMethod] = (
        _DISTANCE_METHODS[1]  # "topsis"
    )

    LABEL_DENSITY_N_NEIGHBORS_DEFAULT: Final[int] = 5
    LABEL_DENSITY_AGGREGATION_DEFAULT: Final[DensityAggregation] = (
        _DENSITY_AGGREGATIONS[0]  # "power_mean"
    )

    OPTUNA_N_JOBS_DEFAULT: Final[int] = 1
    OPTUNA_N_STARTUP_TRIALS_DEFAULT: Final[int] = 15
    OPTUNA_N_TRIALS_DEFAULT: Final[int] = 50
    OPTUNA_TIMEOUT_DEFAULT: Final[int] = 7200
    OPTUNA_MIN_RESOURCE_DEFAULT: Final[int] = 3
    OPTUNA_LABEL_CANDLES_STEP_DEFAULT: Final[int] = 1
    OPTUNA_SPACE_REDUCTION_DEFAULT: Final[bool] = False
    OPTUNA_SPACE_FRACTION_DEFAULT: Final[float] = 0.4
    OPTUNA_SEED_DEFAULT: Final[int] = 1

    _DATA_SPLIT_METHODS: Final[tuple[str, ...]] = (
        "train_test_split",
        "timeseries_split",
    )
    DATA_SPLIT_METHOD_DEFAULT: Final[str] = _DATA_SPLIT_METHODS[0]
    TIMESERIES_N_SPLITS_DEFAULT: Final[int] = 5
    TIMESERIES_GAP_DEFAULT: Final[int] = 0
    TIMESERIES_MAX_TRAIN_SIZE_DEFAULT: Final[int | None] = None

    @staticmethod
    @lru_cache(maxsize=None)
    def _extrema_selection_methods_set() -> set[ExtremaSelectionMethod]:
        return set(EXTREMA_SELECTION_METHODS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _custom_threshold_methods_set() -> set[CustomThresholdMethod]:
        return set(CUSTOM_THRESHOLD_METHODS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _skimage_threshold_methods_set() -> set[SkimageThresholdMethod]:
        return set(SKIMAGE_THRESHOLD_METHODS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _threshold_methods_set() -> set[ThresholdMethod]:
        return set(THRESHOLD_METHODS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _optuna_namespaces_set() -> set[OptunaNamespace]:
        return set(QuickAdapterRegressorV3._OPTUNA_NAMESPACES)

    @staticmethod
    @lru_cache(maxsize=None)
    def _optuna_hpo_samplers_set() -> set[OptunaSampler]:
        return set(QuickAdapterRegressorV3._OPTUNA_HPO_SAMPLERS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _optuna_label_samplers_set() -> set[OptunaSampler]:
        return set(QuickAdapterRegressorV3._OPTUNA_LABEL_SAMPLERS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _scaler_types_set() -> set[ScalerType]:
        return set(QuickAdapterRegressorV3._SCALER_TYPES)

    @staticmethod
    @lru_cache(maxsize=None)
    def _scipy_metrics_set() -> set[str]:
        return set(QuickAdapterRegressorV3._DISTANCE_METRICS[:8])

    @staticmethod
    @lru_cache(maxsize=None)
    def _unsupported_weights_metrics_set() -> set[str]:
        return set(QuickAdapterRegressorV3._UNSUPPORTED_WEIGHTS_METRICS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _distance_methods_set() -> set[DistanceMethod]:
        return set(QuickAdapterRegressorV3._DISTANCE_METHODS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _selection_methods_set() -> set[str]:
        return set(QuickAdapterRegressorV3._SELECTION_METHODS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _distance_metrics_set() -> set[str]:
        return set(QuickAdapterRegressorV3._DISTANCE_METRICS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _density_aggregations_set() -> set[str]:
        return set(QuickAdapterRegressorV3._DENSITY_AGGREGATIONS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _power_mean_metrics_set() -> set[str]:
        return set(QuickAdapterRegressorV3._POWER_MEAN_MAP.keys())

    @staticmethod
    def _get_selection_category(method: str) -> Optional[str]:
        for (
            category,
            methods,
        ) in QuickAdapterRegressorV3._SELECTION_CATEGORIES.items():
            if method in methods:
                return category
        return None

    @staticmethod
    def _get_label_p_order_default(distance_metric: str) -> Optional[float]:
        if (
            distance_metric == QuickAdapterRegressorV3._DISTANCE_METRICS[1]
        ):  # "minkowski"
            return 2.0
        elif (
            distance_metric == QuickAdapterRegressorV3._DISTANCE_METRICS[15]
        ):  # "power_mean"
            return 1.0
        return None

    @staticmethod
    def _get_label_density_metric_default(method: DensityMethod) -> Optional[str]:
        if method == QuickAdapterRegressorV3._DENSITY_METHODS[1]:  # "medoid"
            return QuickAdapterRegressorV3._DISTANCE_METRICS[0]  # "euclidean"
        elif method == QuickAdapterRegressorV3._DENSITY_METHODS[0]:  # "knn"
            return QuickAdapterRegressorV3._DISTANCE_METRICS[1]  # "minkowski"
        return None

    @staticmethod
    def _get_label_density_aggregation_param_default(
        aggregation: DensityAggregation,
    ) -> Optional[float]:
        if (
            aggregation == QuickAdapterRegressorV3._DENSITY_AGGREGATIONS[0]
        ):  # "power_mean"
            return 1.0
        elif (
            aggregation == QuickAdapterRegressorV3._DENSITY_AGGREGATIONS[1]
        ):  # "quantile"
            return 0.5
        return None

    @staticmethod
    def _validate_minkowski_p(
        p: Optional[float],
        *,
        ctx: str,
        mode: ValidationMode = "raise",
    ) -> Optional[float]:
        if p is None:
            return None
        if mode == "none":
            return float(p) if (np.isfinite(p) and p > 0) else None

        if not np.isfinite(p):
            msg = f"Invalid {ctx} value {p!r}: must be finite"
            if mode == "raise":
                raise ValueError(msg)
            logger.warning(f"{msg}, using default")
            return None

        if p <= 0:
            msg = f"Invalid {ctx} value {p!r}: must be > 0"
            if mode == "raise":
                raise ValueError(msg)
            logger.warning(f"{msg}, using default")
            return None

        return float(p)

    @staticmethod
    def _prepare_distance_kwargs(
        distance_metric: str,
        weights: Optional[NDArray[np.floating]] = None,
        p: Optional[float] = None,
        mode: ValidationMode = "none",
        metric_ctx: str = "distance_metric",
        p_ctx: str = "p",
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}

        if weights is not None:
            validated_metric = QuickAdapterRegressorV3._validate_metric_weights_support(
                distance_metric, ctx=metric_ctx, mode=mode
            )
            if validated_metric is not None:
                kwargs["w"] = weights

        if (
            distance_metric == QuickAdapterRegressorV3._DISTANCE_METRICS[1]
        ):  # "minkowski"
            validated_p = QuickAdapterRegressorV3._validate_minkowski_p(
                p, ctx=p_ctx, mode=mode
            )
            if validated_p is not None:
                kwargs["p"] = validated_p

        return kwargs

    @staticmethod
    def _validate_quantile_q(
        q: Optional[float], *, ctx: str, mode: ValidationMode = "raise"
    ) -> Optional[float]:
        if q is None:
            return None
        if mode == "none":
            return float(q) if (np.isfinite(q) and 0.0 <= q <= 1.0) else None

        if not np.isfinite(q):
            msg = f"Invalid {ctx} value {q!r}: must be finite"
            if mode == "raise":
                raise ValueError(msg)
            logger.warning(f"{msg}, using default")
            return None

        if q < 0.0 or q > 1.0:
            msg = f"Invalid {ctx} value {q!r}: must be in [0, 1]"
            if mode == "raise":
                raise ValueError(msg)
            logger.warning(f"{msg}, using default")
            return None

        return float(q)

    @staticmethod
    def _validate_power_mean_p(
        p: Optional[float], *, ctx: str, mode: ValidationMode = "raise"
    ) -> Optional[float]:
        if p is None:
            return None
        if mode == "none":
            return float(p) if np.isfinite(p) else None

        if not np.isfinite(p):
            msg = f"Invalid {ctx} value {p!r}: must be finite"
            if mode == "raise":
                raise ValueError(msg)
            logger.warning(f"{msg}, using default")
            return None

        return float(p)

    @staticmethod
    def _validate_metric_weights_support(
        metric: str, *, ctx: str, mode: ValidationMode = "warn"
    ) -> Optional[str]:
        if metric not in QuickAdapterRegressorV3._unsupported_weights_metrics_set():
            return metric

        if mode == "none":
            return None

        msg = f"Invalid {ctx} value {metric!r}: does not support custom weights"
        if mode == "raise":
            raise ValueError(msg)
        logger.warning(f"{msg}, using uniform weights")
        return None

    @staticmethod
    def _validate_label_weights(
        weights: Any,
        n_objectives: int,
        *,
        ctx: str,
        mode: ValidationMode = "raise",
    ) -> Optional[NDArray[np.floating]]:
        uniform_weights = np.full(n_objectives, 1.0 / n_objectives)

        if weights is None:
            return uniform_weights

        if mode == "none":
            if not isinstance(weights, (list, tuple, np.ndarray)):
                return uniform_weights
            try:
                np_weights = np.asarray(weights, dtype=float)
            except (ValueError, TypeError):
                return uniform_weights
            if np_weights.size != n_objectives:
                return uniform_weights
            if not np.all(np.isfinite(np_weights)):
                return uniform_weights
            if np.any(np_weights < 0):
                return uniform_weights
            weights_sum = np.nansum(np_weights)
            if np.isclose(weights_sum, 0.0):
                return uniform_weights
            return np_weights / weights_sum

        if not isinstance(weights, (list, tuple, np.ndarray)):
            msg = (
                f"Invalid {ctx} {type(weights).__name__!r}: "
                f"must be a list, tuple, or array"
            )
            if mode == "raise":
                raise ValueError(msg)
            logger.warning(f"{msg}, using uniform weights")
            return uniform_weights

        np_weights = np.asarray(weights, dtype=float)

        if np_weights.size != n_objectives:
            msg = (
                f"Invalid {ctx} (length={np_weights.size}): "
                f"must match number of objectives ({n_objectives})"
            )
            if mode == "raise":
                raise ValueError(msg)
            logger.warning(f"{msg}, using uniform weights")
            return uniform_weights

        if not np.all(np.isfinite(np_weights)):
            msg = f"Invalid {ctx} value: contains non-finite values"
            if mode == "raise":
                raise ValueError(msg)
            logger.warning(f"{msg}, using uniform weights")
            return uniform_weights

        if np.any(np_weights < 0):
            msg = f"Invalid {ctx} value: contains negative values"
            if mode == "raise":
                raise ValueError(msg)
            logger.warning(f"{msg}, using uniform weights")
            return uniform_weights

        weights_sum = np.nansum(np_weights)
        if np.isclose(weights_sum, 0.0):
            msg = f"Invalid {ctx} value: sum is zero"
            if mode == "raise":
                raise ValueError(msg)
            logger.warning(f"{msg}, using uniform weights")
            return uniform_weights

        return np_weights / weights_sum

    @staticmethod
    def _validate_enum_value(
        value: str,
        valid_set: AbstractSet[str],
        valid_options: tuple[str, ...],
        *,
        ctx: str,
        mode: ValidationMode = "raise",
        default: Optional[str] = None,
    ) -> Optional[str]:
        if value in valid_set:
            return value

        if mode == "none":
            return default

        msg = (
            f"Invalid {ctx} {value!r}: supported values are {', '.join(valid_options)}"
        )
        if mode == "raise":
            raise ValueError(msg)
        logger.warning(f"{msg}, using {default!r}")
        return default

    @staticmethod
    def _prepare_knn_kwargs(
        distance_metric: str,
        *,
        weights: Optional[NDArray[np.floating]] = None,
        p: Optional[float] = None,
        mode: ValidationMode = "warn",
        p_ctx: str = "label_density_p",
    ) -> dict[str, Any]:
        knn_kwargs: dict[str, Any] = {}

        if distance_metric == QuickAdapterRegressorV3._DISTANCE_METRICS[1]:
            validated_p = QuickAdapterRegressorV3._validate_minkowski_p(
                p, ctx=p_ctx, mode=mode
            )
            if validated_p is not None:
                knn_kwargs["p"] = validated_p
            if weights is not None:
                knn_kwargs["metric_params"] = {"w": weights}

        return knn_kwargs

    @staticmethod
    def _resolve_p_order(
        distance_metric: str,
        label_p_order: Optional[float],
        *,
        ctx: str,
        mode: ValidationMode = "raise",
    ) -> Optional[float]:
        p = (
            label_p_order
            if label_p_order is not None
            else QuickAdapterRegressorV3._get_label_p_order_default(distance_metric)
        )
        if (
            distance_metric == QuickAdapterRegressorV3._DISTANCE_METRICS[1]
        ):  # "minkowski"
            p = QuickAdapterRegressorV3._validate_minkowski_p(p, ctx=ctx, mode=mode)
        return p

    def _resolve_label_method_config(self, label_method: str) -> dict[str, Any]:
        QuickAdapterRegressorV3._validate_enum_value(
            label_method,
            QuickAdapterRegressorV3._selection_methods_set(),
            QuickAdapterRegressorV3._SELECTION_METHODS,
            ctx="label_method",
        )

        category = QuickAdapterRegressorV3._get_selection_category(label_method)
        config: dict[str, Any] = {
            "category": category,
            "method": label_method,
        }

        if category == "distance":
            distance_metric = self.ft_params.get(
                "label_distance_metric",
                QuickAdapterRegressorV3.LABEL_DISTANCE_METRIC_DEFAULT,
            )
            QuickAdapterRegressorV3._validate_enum_value(
                distance_metric,
                QuickAdapterRegressorV3._distance_metrics_set(),
                QuickAdapterRegressorV3._DISTANCE_METRICS,
                ctx="label_distance_metric",
            )
            config["distance_metric"] = distance_metric
        elif category == "cluster":
            distance_metric = self.ft_params.get(
                "label_cluster_metric",
                QuickAdapterRegressorV3.LABEL_CLUSTER_METRIC_DEFAULT,
            )
            QuickAdapterRegressorV3._validate_enum_value(
                distance_metric,
                QuickAdapterRegressorV3._distance_metrics_set(),
                QuickAdapterRegressorV3._DISTANCE_METRICS,
                ctx="label_cluster_metric",
            )
            config["distance_metric"] = distance_metric

            selection_method = self.ft_params.get(
                "label_cluster_selection_method",
                QuickAdapterRegressorV3.LABEL_CLUSTER_SELECTION_METHOD_DEFAULT,
            )
            QuickAdapterRegressorV3._validate_enum_value(
                selection_method,
                QuickAdapterRegressorV3._distance_methods_set(),
                QuickAdapterRegressorV3._DISTANCE_METHODS,
                ctx="label_cluster_selection_method",
            )
            config["selection_method"] = selection_method

            trial_selection_method = self.ft_params.get(
                "label_cluster_trial_selection_method",
                QuickAdapterRegressorV3.LABEL_CLUSTER_TRIAL_SELECTION_METHOD_DEFAULT,
            )
            QuickAdapterRegressorV3._validate_enum_value(
                trial_selection_method,
                QuickAdapterRegressorV3._distance_methods_set(),
                QuickAdapterRegressorV3._DISTANCE_METHODS,
                ctx="label_cluster_trial_selection_method",
            )
            config["trial_selection_method"] = trial_selection_method
        elif category == "density":
            density_method = cast(DensityMethod, label_method)
            distance_metric = self.ft_params.get(
                "label_density_metric",
                QuickAdapterRegressorV3._get_label_density_metric_default(
                    density_method
                ),
            )
            QuickAdapterRegressorV3._validate_enum_value(
                distance_metric,
                QuickAdapterRegressorV3._distance_metrics_set(),
                QuickAdapterRegressorV3._DISTANCE_METRICS,
                ctx="label_density_metric",
            )
            config["distance_metric"] = distance_metric

            if density_method == QuickAdapterRegressorV3._DENSITY_METHODS[0]:  # "knn"
                aggregation = cast(
                    DensityAggregation,
                    self.ft_params.get(
                        "label_density_aggregation",
                        QuickAdapterRegressorV3.LABEL_DENSITY_AGGREGATION_DEFAULT,
                    ),
                )
                QuickAdapterRegressorV3._validate_enum_value(
                    aggregation,
                    QuickAdapterRegressorV3._density_aggregations_set(),
                    QuickAdapterRegressorV3._DENSITY_AGGREGATIONS,
                    ctx="label_density_aggregation",
                )
                config["aggregation"] = aggregation

                n_neighbors = self.ft_params.get(
                    "label_density_n_neighbors",
                    QuickAdapterRegressorV3.LABEL_DENSITY_N_NEIGHBORS_DEFAULT,
                )
                if not isinstance(n_neighbors, int) or n_neighbors < 1:
                    raise ValueError(
                        f"Invalid label_density_n_neighbors value {n_neighbors!r}: must be int >= 1"
                    )
                config["n_neighbors"] = n_neighbors

                aggregation_param = self.ft_params.get(
                    "label_density_aggregation_param",
                    QuickAdapterRegressorV3._get_label_density_aggregation_param_default(
                        aggregation
                    ),
                )

                if aggregation_param is not None:
                    if (
                        aggregation == QuickAdapterRegressorV3._DENSITY_AGGREGATIONS[1]
                    ):  # "quantile"
                        QuickAdapterRegressorV3._validate_quantile_q(
                            aggregation_param,
                            ctx="label_density_aggregation_param",
                        )
                    elif (
                        aggregation == QuickAdapterRegressorV3._DENSITY_AGGREGATIONS[0]
                    ):  # "power_mean"
                        QuickAdapterRegressorV3._validate_power_mean_p(
                            aggregation_param,
                            ctx="label_density_aggregation_param",
                        )

                config["aggregation_param"] = aggregation_param

        return config

    _CONFIG_KEY_TO_TUNABLE_SUFFIX: Final[dict[str, str]] = {
        "distance_metric": "metric",
    }

    @staticmethod
    def _log_label_method_config(config: dict[str, Any]) -> None:
        category = config.get("category", "")
        for key, value in config.items():
            if key in ("category", "method"):
                continue
            suffix = QuickAdapterRegressorV3._CONFIG_KEY_TO_TUNABLE_SUFFIX.get(key, key)
            tunable_name = f"label_{category}_{suffix}"
            if isinstance(value, float):
                formatted_value = format_number(value)
            else:
                formatted_value = value
            logger.info(f"  {tunable_name}: {formatted_value}")

    @property
    def _optuna_config(self) -> dict[str, Any]:
        optuna_default_config = {
            "enabled": False,
            "n_jobs": min(
                self.config.get("freqai", {})
                .get("optuna_hyperopt", {})
                .get("n_jobs", QuickAdapterRegressorV3.OPTUNA_N_JOBS_DEFAULT),
                max(int(self.max_system_threads / 4), 1),
            ),
            "sampler": QuickAdapterRegressorV3._OPTUNA_HPO_SAMPLERS[0],  # "tpe"
            "storage": QuickAdapterRegressorV3._OPTUNA_STORAGE_BACKENDS[0],  # "file"
            "continuous": True,
            "warm_start": True,
            "n_startup_trials": QuickAdapterRegressorV3.OPTUNA_N_STARTUP_TRIALS_DEFAULT,
            "n_trials": QuickAdapterRegressorV3.OPTUNA_N_TRIALS_DEFAULT,
            "timeout": QuickAdapterRegressorV3.OPTUNA_TIMEOUT_DEFAULT,
            "label_sampler": QuickAdapterRegressorV3._OPTUNA_LABEL_SAMPLERS[
                0
            ],  # "auto"
            "label_candles_step": QuickAdapterRegressorV3.OPTUNA_LABEL_CANDLES_STEP_DEFAULT,
            "space_reduction": QuickAdapterRegressorV3.OPTUNA_SPACE_REDUCTION_DEFAULT,
            "space_fraction": QuickAdapterRegressorV3.OPTUNA_SPACE_FRACTION_DEFAULT,
            "min_resource": QuickAdapterRegressorV3.OPTUNA_MIN_RESOURCE_DEFAULT,
            "seed": QuickAdapterRegressorV3.OPTUNA_SEED_DEFAULT,
        }
        optuna_hyperopt = self.config.get("freqai", {}).get("optuna_hyperopt", {})
        return {
            **optuna_default_config,
            **optuna_hyperopt,
        }

    @property
    def _min_label_period_candles(self) -> int:
        return self.ft_params.get(
            "min_label_period_candles",
            QuickAdapterRegressorV3.MIN_LABEL_PERIOD_CANDLES_DEFAULT,
        )

    @property
    def _max_label_period_candles(self) -> int:
        return self.ft_params.get(
            "max_label_period_candles",
            QuickAdapterRegressorV3.MAX_LABEL_PERIOD_CANDLES_DEFAULT,
        )

    @property
    def _min_label_natr_multiplier(self) -> float:
        return self.ft_params.get(
            "min_label_natr_multiplier",
            QuickAdapterRegressorV3.MIN_LABEL_NATR_MULTIPLIER_DEFAULT,
        )

    @property
    def _max_label_natr_multiplier(self) -> float:
        return self.ft_params.get(
            "max_label_natr_multiplier",
            QuickAdapterRegressorV3.MAX_LABEL_NATR_MULTIPLIER_DEFAULT,
        )

    @property
    def _label_frequency_candles(self) -> int:
        default_label_frequency_candles = max(2, 2 * len(self.pairs))

        label_frequency_candles = self.config.get("feature_parameters", {}).get(
            "label_frequency_candles"
        )

        if label_frequency_candles is None:
            label_frequency_candles = default_label_frequency_candles
        elif isinstance(label_frequency_candles, str):
            if label_frequency_candles == "auto":
                label_frequency_candles = default_label_frequency_candles
            else:
                logger.warning(
                    f"Invalid label_frequency_candles value {label_frequency_candles!r}: only 'auto' is supported for string values, using default {default_label_frequency_candles!r}"
                )
                label_frequency_candles = default_label_frequency_candles
        elif isinstance(label_frequency_candles, (int, float)):
            if label_frequency_candles >= 2 and label_frequency_candles <= 10000:
                label_frequency_candles = int(label_frequency_candles)
            else:
                logger.warning(
                    f"Invalid label_frequency_candles value {label_frequency_candles!r}: must be in range [2, 10000], using default {default_label_frequency_candles!r}"
                )
                label_frequency_candles = default_label_frequency_candles
        else:
            logger.warning(
                f"Invalid label_frequency_candles value {label_frequency_candles!r}: expected int, float, or 'auto', using default {default_label_frequency_candles!r}"
            )
            label_frequency_candles = default_label_frequency_candles

        return label_frequency_candles

    @property
    def label_pipeline(self) -> dict[str, Any]:
        label_pipeline_raw = self.freqai_info.get("label_pipeline")
        if not isinstance(label_pipeline_raw, dict):
            label_pipeline_raw = {}
        return get_label_pipeline_config(label_pipeline_raw, logger)

    @property
    def label_prediction(self) -> dict[str, Any]:
        label_prediction_raw = self.freqai_info.get("label_prediction")
        if not isinstance(label_prediction_raw, dict):
            label_prediction_raw = {}
        return get_label_prediction_config(label_prediction_raw, logger)

    @property
    def _label_defaults(self) -> tuple[int, float]:
        return get_label_defaults(self.ft_params, logger)

    @property
    def _optuna_label_candle_pool_full(self) -> list[int]:
        label_frequency_candles = self._label_frequency_candles
        cache_key = label_frequency_candles
        if cache_key not in self._optuna_label_candle_pool_full_cache:
            half_label_frequency_candles = int(label_frequency_candles / 2)
            self._optuna_label_candle_pool_full_cache[cache_key] = [
                max(1, label_frequency_candles + offset)
                for offset in range(
                    -half_label_frequency_candles, half_label_frequency_candles + 1
                )
            ]
        return copy.deepcopy(self._optuna_label_candle_pool_full_cache[cache_key])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        migrate_config(self.config, logger)
        self.pairs: list[str] = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.pairs:
            raise ValueError(
                "Invalid configuration: 'pair_whitelist' must be defined in exchange section and StaticPairList must be configured in pairlists"
            )
        if (
            not isinstance(self.freqai_info.get("identifier"), str)
            or not self.freqai_info.get("identifier", "").strip()
        ):
            raise ValueError(
                "Invalid freqai configuration: 'identifier' must be a non-empty string"
            )
        self._optuna_hyperopt: Optional[bool] = (
            self.freqai_info.get("enabled", False)
            and self._optuna_config.get("enabled")
            and self.data_split_parameters.get(
                "test_size", QuickAdapterRegressorV3._TEST_SIZE
            )
            > 0
        )
        self._optuna_hp_value: dict[str, float] = {}
        self._optuna_label_values: dict[str, list[float | int]] = {}
        self._optuna_hp_params: dict[str, dict[str, Any]] = {}
        self._optuna_label_params: dict[str, dict[str, Any]] = {}
        self._optuna_label_candle_pool_full_cache: dict[int, list[int]] = {}
        self._optuna_label_shuffle_rng = random.Random(
            self._optuna_config.get("seed", QuickAdapterRegressorV3.OPTUNA_SEED_DEFAULT)
        )
        self.init_optuna_label_candle_pool()
        self._optuna_label_candle: dict[str, int] = {}
        self._optuna_label_candles: dict[str, int] = {}
        self._optuna_label_incremented_pairs: list[str] = []
        default_label_period_candles, default_label_natr_multiplier = (
            self._label_defaults
        )
        for pair in self.pairs:
            self._optuna_hp_value[pair] = -1
            self._optuna_label_values[pair] = [
                -1
            ] * QuickAdapterRegressorV3._OPTUNA_LABEL_N_OBJECTIVES
            self._optuna_hp_params[pair] = (
                self.optuna_load_best_params(
                    pair, QuickAdapterRegressorV3._OPTUNA_NAMESPACES[0]
                )  # "hp"
                or {}
            )
            self._optuna_label_params[pair] = (
                self.optuna_load_best_params(
                    pair, QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]
                )  # "label"
                or {
                    "label_period_candles": self.ft_params.get(
                        "label_period_candles",
                        default_label_period_candles,
                    ),
                    "label_natr_multiplier": float(
                        self.ft_params.get(
                            "label_natr_multiplier",
                            default_label_natr_multiplier,
                        )
                    ),
                }
            )
            self.set_optuna_label_candle(pair)
            self._optuna_label_candles[pair] = 0

        self.regressor: Regressor = self.freqai_info.get("regressor", REGRESSORS[0])
        if self.regressor not in set(REGRESSORS):
            self.regressor = REGRESSORS[0]
            self.freqai_info["regressor"] = self.regressor
        self._log_model_configuration()

    def _log_model_configuration(self) -> None:
        logger.info("=" * 60)
        logger.info("QuickAdapterRegressor Model Configuration")
        logger.info("=" * 60)

        logger.info(f"Model Version: {self.version}")
        logger.info(f"Regressor: {self.regressor}")

        logger.info("Optuna Hyperopt Configuration:")
        optuna_config = self._optuna_config
        logger.info(f"  enabled: {optuna_config.get('enabled')}")
        if optuna_config.get("enabled"):
            logger.info(f"  n_jobs: {optuna_config.get('n_jobs')}")
            logger.info(f"  sampler: {optuna_config.get('sampler')}")
            logger.info(f"  storage: {optuna_config.get('storage')}")
            logger.info(f"  continuous: {optuna_config.get('continuous')}")
            logger.info(f"  warm_start: {optuna_config.get('warm_start')}")
            logger.info(f"  n_startup_trials: {optuna_config.get('n_startup_trials')}")
            logger.info(f"  n_trials: {optuna_config.get('n_trials')}")
            logger.info(f"  timeout: {optuna_config.get('timeout')}")
            logger.info(f"  space_reduction: {optuna_config.get('space_reduction')}")
            logger.info(
                f"  space_fraction: {format_number(optuna_config.get('space_fraction'))}"
            )
            logger.info(f"  min_resource: {optuna_config.get('min_resource')}")
            logger.info(f"  seed: {optuna_config.get('seed')}")

            logger.info(f"  label_sampler: {optuna_config.get('label_sampler')}")
            logger.info(
                f"  label_candles_step: {optuna_config.get('label_candles_step')}"
            )
            label_method = self.ft_params.get(
                "label_method", QuickAdapterRegressorV3.LABEL_METHOD_DEFAULT
            )
            logger.info(f"  label_method: {label_method}")

            label_config = self._resolve_label_method_config(label_method)
            QuickAdapterRegressorV3._log_label_method_config(label_config)

            label_weights = self.ft_params.get("label_weights")
            if label_weights is not None:
                formatted_label_weights = [format_number(w) for w in label_weights]
                logger.info(f"  label_weights: [{', '.join(formatted_label_weights)}]")
            else:
                logger.info(
                    "  label_weights: [1.0, ...] * n_objectives, l1 normalized (default)"
                )

            label_p_order_config = self.ft_params.get("label_p_order")
            if label_p_order_config is not None:
                logger.info(
                    f"  label_p_order: {format_number(float(label_p_order_config))}"
                )
            else:
                distance_metric = label_config["distance_metric"]
                if distance_metric in {
                    QuickAdapterRegressorV3._DISTANCE_METRICS[1],  # "minkowski"
                    QuickAdapterRegressorV3._DISTANCE_METRICS[15],  # "power_mean"
                }:
                    label_p_order_default = (
                        QuickAdapterRegressorV3._get_label_p_order_default(
                            distance_metric
                        )
                    )
                    logger.info(
                        f"  label_p_order: {format_number(label_p_order_default)} (default for {distance_metric})"
                    )

        label_pipeline = self.label_pipeline
        label_prediction = self.label_prediction
        for label_col in LABEL_COLUMNS:
            logger.info(f"Label Configuration [{label_col}]:")

            col_pipeline = get_label_column_config(
                label_col, label_pipeline["default"], label_pipeline["columns"]
            )
            logger.info("  Pipeline:")
            logger.info(f"    standardization: {col_pipeline['standardization']}")
            logger.info(
                f"    robust_quantiles: ({format_number(col_pipeline['robust_quantiles'][0])}, {format_number(col_pipeline['robust_quantiles'][1])})"
            )
            logger.info(
                f"    mmad_scaling_factor: {format_number(col_pipeline['mmad_scaling_factor'])}"
            )
            logger.info(f"    normalization: {col_pipeline['normalization']}")
            logger.info(
                f"    minmax_range: ({format_number(col_pipeline['minmax_range'][0])}, {format_number(col_pipeline['minmax_range'][1])})"
            )
            logger.info(
                f"    sigmoid_scale: {format_number(col_pipeline['sigmoid_scale'])}"
            )
            logger.info(f"    gamma: {format_number(col_pipeline['gamma'])}")

            col_prediction = get_label_column_config(
                label_col, label_prediction["default"], label_prediction["columns"]
            )
            logger.info("  Prediction:")
            logger.info(f"    method: {col_prediction['method']}")
            logger.info(f"    selection_method: {col_prediction['selection_method']}")
            logger.info(f"    threshold_method: {col_prediction['threshold_method']}")
            logger.info(
                f"    outlier_quantile: {format_number(col_prediction['outlier_quantile'])}"
            )
            logger.info(
                f"    soft_extremum_alpha: {format_number(col_prediction['soft_extremum_alpha'])}"
            )
            logger.info(
                f"    keep_fraction: {format_number(col_prediction['keep_fraction'])}"
            )

        default_label_period_candles, default_label_natr_multiplier = (
            self._label_defaults
        )
        label_period_candles = self.ft_params.get(
            "label_period_candles", default_label_period_candles
        )
        label_natr_multiplier = float(
            self.ft_params.get("label_natr_multiplier", default_label_natr_multiplier)
        )
        logger.info("Label Hyperparameters:")
        logger.info(
            f"  fit_live_predictions_candles: {self.freqai_info.get('fit_live_predictions_candles', QuickAdapterRegressorV3.FIT_LIVE_PREDICTIONS_CANDLES_DEFAULT)}"
        )
        if self._optuna_hyperopt:
            logger.info(
                f"  label_period_candles: {label_period_candles} (initial value)"
            )
            logger.info(
                f"  label_natr_multiplier: {format_number(label_natr_multiplier)} (initial value)"
            )
        logger.info(f"  label_frequency_candles: {self._label_frequency_candles}")
        logger.info(f"  min_label_period_candles: {self._min_label_period_candles}")
        logger.info(f"  max_label_period_candles: {self._max_label_period_candles}")
        logger.info(
            f"  min_label_natr_multiplier: {format_number(self._min_label_natr_multiplier)}"
        )
        logger.info(
            f"  max_label_natr_multiplier: {format_number(self._max_label_natr_multiplier)}"
        )

        if self._optuna_hyperopt:
            logger.info("Label Parameters:")
            for pair in self.pairs:
                params = self._optuna_label_params.get(pair, {})
                if params:
                    logger.info(
                        f"  {pair}: label_period_candles={params.get('label_period_candles')}, "
                        f"label_natr_multiplier={format_number(params.get('label_natr_multiplier'))}"
                    )
        else:
            logger.info("Label Parameters:")
            logger.info(f"  label_period_candles: {label_period_candles}")
            logger.info(
                f"  label_natr_multiplier: {format_number(label_natr_multiplier)}"
            )

        scaler = self.ft_params.get("scaler", QuickAdapterRegressorV3.SCALER_DEFAULT)
        feature_range = self.ft_params.get(
            "range", QuickAdapterRegressorV3.RANGE_DEFAULT
        )
        logger.info("Feature Parameters Configuration:")
        logger.info(f"  scaler: {scaler}")
        logger.info(
            f"  range: ({format_number(feature_range[0])}, {format_number(feature_range[1])})"
        )

        logger.info("=" * 60)

    def get_optuna_params(
        self, pair: str, namespace: OptunaNamespace
    ) -> dict[str, Any]:
        if namespace == QuickAdapterRegressorV3._OPTUNA_NAMESPACES[0]:  # "hp"
            params = self._optuna_hp_params.get(pair, {})
        elif namespace == QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]:  # "label"
            params = self._optuna_label_params.get(pair, {})
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {', '.join(QuickAdapterRegressorV3._OPTUNA_NAMESPACES)}"
            )
        return params

    def set_optuna_params(
        self, pair: str, namespace: OptunaNamespace, params: dict[str, Any]
    ) -> None:
        if namespace == QuickAdapterRegressorV3._OPTUNA_NAMESPACES[0]:  # "hp"
            self._optuna_hp_params[pair] = params
        elif namespace == QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]:  # "label"
            self._optuna_label_params[pair] = params
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {', '.join(QuickAdapterRegressorV3._OPTUNA_NAMESPACES)}"
            )

    def get_optuna_value(self, pair: str, namespace: OptunaNamespace) -> float:
        if namespace == QuickAdapterRegressorV3._OPTUNA_NAMESPACES[0]:  # "hp"
            value = self._optuna_hp_value.get(pair, np.nan)
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {QuickAdapterRegressorV3._OPTUNA_NAMESPACES[0]!r}"  # "hp"
            )
        return value

    def set_optuna_value(
        self, pair: str, namespace: OptunaNamespace, value: float
    ) -> None:
        if namespace == QuickAdapterRegressorV3._OPTUNA_NAMESPACES[0]:  # "hp"
            self._optuna_hp_value[pair] = value
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {QuickAdapterRegressorV3._OPTUNA_NAMESPACES[0]!r}"  # "hp"
            )

    def get_optuna_values(
        self, pair: str, namespace: OptunaNamespace
    ) -> list[float | int]:
        if namespace == QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]:  # "label"
            values = self._optuna_label_values.get(
                pair, [np.nan] * QuickAdapterRegressorV3._OPTUNA_LABEL_N_OBJECTIVES
            )
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]}"  # "label"
            )
        return values

    def set_optuna_values(
        self, pair: str, namespace: OptunaNamespace, values: list[float | int]
    ) -> None:
        if namespace == QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]:  # "label"
            self._optuna_label_values[pair] = values
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]}"  # "label"
            )

    def init_optuna_label_candle_pool(self) -> None:
        optuna_label_candle_pool_full = self._optuna_label_candle_pool_full
        if len(optuna_label_candle_pool_full) == 0:
            raise RuntimeError(
                "Failed to initialize optuna label candle pool: initial pool is empty"
            )
        self._optuna_label_candle_pool = optuna_label_candle_pool_full
        self._optuna_label_shuffle_rng.shuffle(self._optuna_label_candle_pool)
        if len(self._optuna_label_candle_pool) == 0:
            raise RuntimeError(
                "Failed to initialize optuna label candle pool: pool became empty after shuffle"
            )

    def set_optuna_label_candle(self, pair: str) -> None:
        if len(self._optuna_label_candle_pool) == 0:
            logger.warning(
                f"[{pair}] Optuna label candle pool is empty, reinitializing"
            )
            logger.debug(
                f"[{pair}] Optuna label candle pool state: "
                f"pool={self._optuna_label_candle_pool}, "
                f"pool_full={self._optuna_label_candle_pool_full}, "
                f"candle={list(self._optuna_label_candle.values())}, "
                f"candles={list(self._optuna_label_candles.values())}, "
                f"incremented_pairs={self._optuna_label_incremented_pairs}"
            )
            self.init_optuna_label_candle_pool()
        optuna_label_candle_pool = copy.deepcopy(self._optuna_label_candle_pool)
        for p in self.pairs:
            if p == pair:
                continue
            optuna_label_candle = self._optuna_label_candle.get(p)
            optuna_label_candles = self._optuna_label_candles.get(p)
            if optuna_label_candle is not None and optuna_label_candles is not None:
                if (
                    self._optuna_label_incremented_pairs
                    and p not in self._optuna_label_incremented_pairs
                ):
                    optuna_label_candles += 1
                remaining_candles = optuna_label_candle - optuna_label_candles
                if remaining_candles in optuna_label_candle_pool:
                    optuna_label_candle_pool.remove(remaining_candles)
        optuna_label_candle = optuna_label_candle_pool.pop()
        self._optuna_label_candle[pair] = optuna_label_candle
        self._optuna_label_candle_pool.remove(optuna_label_candle)
        optuna_label_available_candles = (
            set(self._optuna_label_candle_pool_full)
            - set(self._optuna_label_candle_pool)
            - set(self._optuna_label_candle.values())
        )
        if len(optuna_label_available_candles) > 0:
            self._optuna_label_candle_pool.extend(
                sorted(optuna_label_available_candles)
            )
            self._optuna_label_shuffle_rng.shuffle(self._optuna_label_candle_pool)

    def define_data_pipeline(self, threads: int = -1) -> Pipeline:
        scaler = self.ft_params.get("scaler", QuickAdapterRegressorV3.SCALER_DEFAULT)

        QuickAdapterRegressorV3._validate_enum_value(
            scaler,
            QuickAdapterRegressorV3._scaler_types_set(),
            QuickAdapterRegressorV3._SCALER_TYPES,
            ctx="scaler",
        )

        feature_range = self.ft_params.get(
            "range", QuickAdapterRegressorV3.RANGE_DEFAULT
        )

        if not isinstance(feature_range, (list, tuple)) or len(feature_range) != 2:
            raise ValueError(
                f"Invalid range {type(feature_range).__name__!r}: "
                f"must be a list or tuple of 2 numbers"
            )
        min_val, max_val = float(feature_range[0]), float(feature_range[1])
        if min_val >= max_val:
            raise ValueError(f"Invalid range [{min_val}, {max_val}]: min must be < max")
        feature_range = (min_val, max_val)

        if (
            scaler == QuickAdapterRegressorV3.SCALER_DEFAULT
            and feature_range == QuickAdapterRegressorV3.RANGE_DEFAULT
        ):
            return super().define_data_pipeline(threads)

        pipeline = super().define_data_pipeline(threads)

        if scaler == QuickAdapterRegressorV3._SCALER_TYPES[1]:  # "maxabs"
            scaler_obj = SKLearnWrapper(MaxAbsScaler())
        elif scaler == QuickAdapterRegressorV3._SCALER_TYPES[2]:  # "standard"
            scaler_obj = SKLearnWrapper(StandardScaler())
        elif scaler == QuickAdapterRegressorV3._SCALER_TYPES[3]:  # "robust"
            scaler_obj = SKLearnWrapper(RobustScaler())
        else:  # "minmax"
            scaler_obj = SKLearnWrapper(MinMaxScaler(feature_range=feature_range))

        steps = [
            (name, scaler_obj)
            if name in ("scaler", "post-pca-scaler")
            else (name, transformer)
            for name, transformer in pipeline.steps
        ]

        return Pipeline(steps)

    def define_label_pipeline(self, threads: int = -1) -> Pipeline:
        return Pipeline(
            [
                (
                    "label_transformer",
                    LabelTransformer(label_transformer=self.label_pipeline),
                ),
            ]
        )

    def train(
        self, unfiltered_df: pd.DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Filter the training data and train a model to it.

        Supports two data split methods:
        - 'train_test_split' (default): Delegates to BaseRegressionModel.train()
        - 'timeseries_split': Uses TimeSeriesSplit for temporal validation

        :param unfiltered_df: Full dataframe for the current training period
        :param pair: Trading pair being trained
        :param dk: FreqaiDataKitchen object containing configuration
        :return: Trained model
        """
        method = self.data_split_parameters.get(
            "method", self.DATA_SPLIT_METHOD_DEFAULT
        )

        if method not in self._DATA_SPLIT_METHODS:
            raise ValueError(
                f"Invalid data_split_parameters.method value {method!r}: "
                f"supported values are {', '.join(self._DATA_SPLIT_METHODS)}"
            )

        logger.info(f"Using data split method: {method}")

        if method == self.DATA_SPLIT_METHOD_DEFAULT:
            return super().train(unfiltered_df, pair, dk, **kwargs)

        elif method == "timeseries_split":
            from time import time

            logger.info(
                f"-------------------- Starting training {pair} --------------------"
            )

            start_time = time()

            features_filtered, labels_filtered = dk.filter_features(
                unfiltered_df,
                dk.training_features_list,
                dk.label_list,
                training_filter=True,
            )

            start_date = unfiltered_df["date"].iloc[0].strftime("%Y-%m-%d")
            end_date = unfiltered_df["date"].iloc[-1].strftime("%Y-%m-%d")
            logger.info(
                f"-------------------- Training on data from {start_date} to "
                f"{end_date} --------------------"
            )

            dd = self._make_timeseries_split_datasets(
                features_filtered, labels_filtered, dk
            )

            if (
                not self.freqai_info.get("fit_live_predictions_candles", 0)
                or not self.live
            ):
                dk.fit_labels()

            dd = self._apply_pipelines(dd, dk, pair)

            logger.info(
                f"Training model on {len(dk.data_dictionary['train_features'].columns)} features"
            )
            logger.info(f"Training model on {len(dd['train_features'])} data points")

            model = self.fit(dd, dk)

            end_time = time()

            logger.info(
                f"-------------------- Done training {pair} "
                f"({end_time - start_time:.2f} secs) --------------------"
            )

            return model

    def _apply_pipelines(
        self,
        dd: dict,
        dk: FreqaiDataKitchen,
        pair: str,
    ) -> dict:
        """
        Apply feature and label pipelines to train/test data.

        This helper reduces code duplication between train() methods that need
        custom data splitting but share the same pipeline application logic.

        :param dd: data_dictionary with train/test features/labels/weights
        :param dk: FreqaiDataKitchen instance
        :param pair: Trading pair (for error messages)
        :return: data_dictionary with transformed features/labels
        """
        from freqtrade.exceptions import DependencyException

        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)
        dk.label_pipeline = self.define_label_pipeline(threads=dk.thread_count)

        (dd["train_features"], dd["train_labels"], dd["train_weights"]) = (
            dk.feature_pipeline.fit_transform(
                dd["train_features"], dd["train_labels"], dd["train_weights"]
            )
        )

        dd["train_labels"], _, _ = dk.label_pipeline.fit_transform(dd["train_labels"])

        if self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.1) != 0:
            if dd["test_labels"].shape[0] == 0:
                raise DependencyException(
                    f"{pair}: test set is empty after filtering. "
                    f"This is usually caused by overly strict SVM thresholds or insufficient data. "
                    f"Try reducing 'test_size' or relaxing your SVM conditions."
                )
            else:
                (dd["test_features"], dd["test_labels"], dd["test_weights"]) = (
                    dk.feature_pipeline.transform(
                        dd["test_features"], dd["test_labels"], dd["test_weights"]
                    )
                )
                dd["test_labels"], _, _ = dk.label_pipeline.transform(dd["test_labels"])

        dk.data_dictionary = dd

        return dd

    @staticmethod
    def _compute_timeseries_min_samples(
        n_splits: int, gap: int, test_size: int | None
    ) -> int:
        """
        Compute minimum samples required for TimeSeriesSplit.

        sklearn's TimeSeriesSplit requires: n_samples >= n_splits + 1 + gap + test_size.
        When test_size is None, sklearn computes it as n_samples // (n_splits + 1).

        :param n_splits: Number of folds
        :param gap: Gap between train and test sets
        :param test_size: Fixed test size per fold (None for dynamic sizing)
        :return: Minimum number of samples required
        """
        if test_size is not None:
            return n_splits * (test_size + gap) + test_size
        return n_splits + 1 + gap

    def _make_timeseries_split_datasets(
        self,
        filtered_dataframe: pd.DataFrame,
        labels: pd.DataFrame,
        dk: FreqaiDataKitchen,
    ) -> dict:
        """
        Split data using TimeSeriesSplit for temporal validation.

        Uses the LAST fold from TimeSeriesSplit to create a single train/test split
        that respects temporal ordering. Delegates weight calculation and dictionary
        building to FreqaiDataKitchen to maintain consistency with FreqAI conventions.

        :param filtered_dataframe: Feature data to split
        :param labels: Label data to split
        :param dk: FreqaiDataKitchen instance for weight calculation and data building
        :return: data_dictionary with train/test features/labels/weights
        """
        n_splits = self.data_split_parameters.get(
            "n_splits", self.TIMESERIES_N_SPLITS_DEFAULT
        )
        gap = self.data_split_parameters.get("gap", self.TIMESERIES_GAP_DEFAULT)
        max_train_size = self.data_split_parameters.get(
            "max_train_size", self.TIMESERIES_MAX_TRAIN_SIZE_DEFAULT
        )

        if n_splits < 2:
            raise ValueError(
                f"Invalid data_split_parameters.n_splits value {n_splits!r}: must be >= 2"
            )

        test_size_param = self.data_split_parameters.get("test_size", None)
        test_size: int | None = None
        if test_size_param is not None:
            test_size = int(len(filtered_dataframe) * test_size_param)

        min_samples = self._compute_timeseries_min_samples(n_splits, gap, test_size)
        if len(filtered_dataframe) < min_samples:
            raise ValueError(
                f"Dataset size ({len(filtered_dataframe)}) too small for "
                f"n_splits={n_splits}, gap={gap}, test_size={test_size}. "
                f"Minimum required: {min_samples}"
            )

        if gap == 0:
            logger.warning(
                "TimeSeriesSplit with gap=0 risks look-ahead bias. "
                "Consider setting gap >= label_period_candles to prevent data leakage."
            )

        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap,
            max_train_size=max_train_size,
            test_size=test_size,
        )
        splits = list(tscv.split(filtered_dataframe))
        train_idx, test_idx = splits[-1]

        train_features = filtered_dataframe.iloc[train_idx]
        test_features = filtered_dataframe.iloc[test_idx]
        train_labels = labels.iloc[train_idx]
        test_labels = labels.iloc[test_idx]

        feat_dict = self.freqai_info.get("feature_parameters", {})
        if feat_dict.get("weight_factor", 0) > 0:
            total_weights = dk.set_weights_higher_recent(len(train_idx) + len(test_idx))
            train_weights = total_weights[: len(train_idx)]
            test_weights = total_weights[len(train_idx) :]
        else:
            train_weights = np.ones(len(train_idx))
            test_weights = np.ones(len(test_idx))

        return dk.build_data_dictionary(
            train_features,
            test_features,
            train_labels,
            test_labels,
            train_weights,
            test_weights,
        )

    def fit(
        self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        :param dk: the FreqaiDataKitchen object
        """

        X = data_dictionary.get("train_features")
        y = data_dictionary.get("train_labels")
        train_weights = data_dictionary.get("train_weights")

        X_test = data_dictionary.get("test_features")
        y_test = data_dictionary.get("test_labels")
        test_weights = data_dictionary.get("test_weights")

        model_training_parameters = copy.deepcopy(self.model_training_parameters)

        start_time = time.time()
        if self._optuna_hyperopt:
            self.optuna_optimize(
                pair=dk.pair,
                namespace=QuickAdapterRegressorV3._OPTUNA_NAMESPACES[0],  # "hp"
                objective=lambda trial: hp_objective(
                    trial,
                    self.regressor,
                    X,
                    y,
                    train_weights,
                    X_test,
                    y_test,
                    test_weights,
                    self.data_split_parameters.get(
                        "test_size", QuickAdapterRegressorV3._TEST_SIZE
                    ),
                    self.get_optuna_params(
                        dk.pair, QuickAdapterRegressorV3._OPTUNA_NAMESPACES[0]
                    ),  # "hp"
                    model_training_parameters,
                    self._optuna_config.get(
                        "space_reduction",
                        QuickAdapterRegressorV3.OPTUNA_SPACE_REDUCTION_DEFAULT,
                    ),
                    self._optuna_config.get(
                        "space_fraction",
                        QuickAdapterRegressorV3.OPTUNA_SPACE_FRACTION_DEFAULT,
                    ),
                    dk.data_path,
                ),
                direction=optuna.study.StudyDirection.MINIMIZE,
            )

            optuna_hp_params = self.get_optuna_params(
                dk.pair, QuickAdapterRegressorV3._OPTUNA_NAMESPACES[0]
            )  # "hp"
            if optuna_hp_params:
                model_training_parameters = {
                    **model_training_parameters,
                    **optuna_hp_params,
                }

        eval_set, eval_weights = eval_set_and_weights(
            X_test,
            y_test,
            test_weights,
            self.data_split_parameters.get(
                "test_size", QuickAdapterRegressorV3._TEST_SIZE
            ),
        )

        model = fit_regressor(
            regressor=self.regressor,
            X=X,
            y=y,
            train_weights=train_weights,
            eval_set=eval_set,
            eval_weights=eval_weights,
            model_training_parameters=model_training_parameters,
            init_model=self.get_init_model(dk.pair),
            model_path=dk.data_path,
        )
        time_spent = time.time() - start_time
        self.dd.update_metric_tracker("fit_time", time_spent, dk.pair)

        return model

    def optuna_throttle_callback(
        self,
        pair: str,
        namespace: OptunaNamespace,
        callback: Callable[[], Optional[optuna.study.Study]],
    ) -> None:
        if namespace not in {QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]}:  # "label"
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]}"  # "label"
            )
        if not callable(callback):
            raise ValueError(
                f"Invalid callback value {type(callback).__name__!r}: must be callable"
            )
        self._optuna_label_candles[pair] += 1
        if pair not in self._optuna_label_incremented_pairs:
            self._optuna_label_incremented_pairs.append(pair)
        optuna_label_remaining_candles = self._optuna_label_candle.get(
            pair, 0
        ) - self._optuna_label_candles.get(pair, 0)
        if optuna_label_remaining_candles <= 0:
            try:
                callback()
            except Exception as e:
                logger.error(
                    f"[{pair}] Optuna {namespace} callback execution failed: {e!r}",
                    exc_info=True,
                )
            finally:
                self.set_optuna_label_candle(pair)
                self._optuna_label_candles[pair] = 0
        else:
            logger.debug(
                f"[{pair}] Optuna {namespace} callback throttled, still {optuna_label_remaining_candles} candles to go"
            )
        if len(self._optuna_label_incremented_pairs) >= len(self.pairs):
            self._optuna_label_incremented_pairs = []

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:
        warmed_up = True

        fit_live_predictions_candles = self.freqai_info.get(
            "fit_live_predictions_candles",
            QuickAdapterRegressorV3.FIT_LIVE_PREDICTIONS_CANDLES_DEFAULT,
        )

        if self._optuna_hyperopt:
            self.optuna_throttle_callback(
                pair=pair,
                namespace=QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1],  # "label"
                callback=lambda: self.optuna_optimize(
                    pair=pair,
                    namespace=QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1],  # "label"
                    objective=lambda trial: label_objective(
                        trial,
                        self.data_provider.get_pair_dataframe(
                            pair=pair, timeframe=self.config.get("timeframe")
                        ),
                        fit_live_predictions_candles,
                        self._optuna_config.get(
                            "label_candles_step",
                            QuickAdapterRegressorV3.OPTUNA_LABEL_CANDLES_STEP_DEFAULT,
                        ),
                        min_label_period_candles=self._min_label_period_candles,
                        max_label_period_candles=self._max_label_period_candles,
                        min_label_natr_multiplier=self._min_label_natr_multiplier,
                        max_label_natr_multiplier=self._max_label_natr_multiplier,
                    ),
                    directions=list(QuickAdapterRegressorV3._OPTUNA_LABEL_DIRECTIONS),
                ),
            )

        if self.live:
            if not hasattr(self, "exchange_candles"):
                self.exchange_candles = len(self.dd.model_return_values[pair].index)
            candles_diff = len(self.dd.historic_predictions[pair].index) - (
                fit_live_predictions_candles + self.exchange_candles
            )
            if candles_diff < 0:
                logger.warning(
                    f"[{pair}] Fit live predictions not warmed up yet, still {abs(candles_diff)} candles to go"
                )
                warmed_up = False

        pred_df = (
            self.dd.historic_predictions[pair]
            .iloc[-fit_live_predictions_candles:]
            .reset_index(drop=True)
        )

        di_values = pred_df.get("DI_values")
        dk.data["DI_value_mean"] = di_values.mean()
        dk.data["DI_value_std"] = di_values.std(ddof=1)

        label_prediction = self.label_prediction
        for label_col in dk.label_list:
            col_prediction_config = get_label_column_config(
                label_col, label_prediction["default"], label_prediction["columns"]
            )
            method = col_prediction_config.get("method")
            if method == PREDICTION_METHODS[0]:  # "none"
                continue
            elif method == PREDICTION_METHODS[1]:  # "thresholding"
                if not warmed_up:
                    min_pred, max_pred = -2.0, 2.0
                    f = [0.0, 0.0, 0.0]
                    cutoff = 2.0
                else:
                    min_pred, max_pred = self.min_max_pred(
                        label_col,
                        col_prediction_config,
                        pred_df,
                        fit_live_predictions_candles,
                        self.get_optuna_params(
                            pair, QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]
                        ).get("label_period_candles"),  # "label"
                    )
                    f = sp.stats.weibull_min.fit(
                        pd.to_numeric(di_values, errors="coerce").dropna(), floc=0
                    )
                    outlier_quantile = col_prediction_config.get(
                        "outlier_quantile",
                        DEFAULTS_LABEL_PREDICTION["outlier_quantile"],
                    )
                    cutoff = sp.stats.weibull_min.ppf(outlier_quantile, *f)
                dk.data["extra_returns_per_train"][f"{label_col}_minima_threshold"] = (
                    min_pred
                )
                dk.data["extra_returns_per_train"][f"{label_col}_maxima_threshold"] = (
                    max_pred
                )
                dk.data["extra_returns_per_train"]["DI_value_param1"] = f[0]
                dk.data["extra_returns_per_train"]["DI_value_param2"] = f[1]
                dk.data["extra_returns_per_train"]["DI_value_param3"] = f[2]
                dk.data["extra_returns_per_train"]["DI_cutoff"] = cutoff

        dk.data["labels_mean"], dk.data["labels_std"] = {}, {}
        for label_col in dk.label_list + dk.unique_class_list:
            pred_label = pred_df.get(label_col)
            if pred_label is None or pred_label.dtype == object:
                continue
            if not warmed_up:
                f = [0.0, 0.0]
            else:
                f = sp.stats.norm.fit(pred_label)
            dk.data["labels_mean"][label_col], dk.data["labels_std"][label_col] = (
                f[0],
                f[1],
            )

        dk.data["extra_returns_per_train"]["label_period_candles"] = (
            self.get_optuna_params(
                pair, QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]
            ).get("label_period_candles")  # "label"
        )
        dk.data["extra_returns_per_train"]["label_natr_multiplier"] = (
            self.get_optuna_params(
                pair,
                QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1],  # "label"
            ).get("label_natr_multiplier")
        )

        hp_rmse = QuickAdapterRegressorV3.optuna_validate_value(
            self.get_optuna_value(pair, QuickAdapterRegressorV3._OPTUNA_NAMESPACES[0])
        )  # "hp"
        dk.data["extra_returns_per_train"]["hp_rmse"] = (
            hp_rmse if hp_rmse is not None else np.inf
        )

    @staticmethod
    def optuna_validate_value(value: Any) -> Optional[float]:
        return value if isinstance(value, (int, float)) and np.isfinite(value) else None

    def min_max_pred(
        self,
        label_col: str,
        col_prediction_config: dict[str, Any],
        pred_df: pd.DataFrame,
        fit_live_predictions_candles: int,
        label_period_candles: Optional[int],
    ) -> tuple[float, float]:
        if label_period_candles is None or label_period_candles <= 0:
            label_period_candles = int(
                self.ft_params.get("label_period_candles", self._label_defaults[0])
            )
        thresholds_candles = (
            max(2, int(fit_live_predictions_candles / label_period_candles))
            * label_period_candles
        )

        pred_label = pred_df.get(label_col)
        if pred_label is None:
            return -2.0, 2.0
        pred_label = pred_label.iloc[-thresholds_candles:].copy()

        selection_method = col_prediction_config["selection_method"]
        threshold_method = col_prediction_config["threshold_method"]
        keep_fraction = col_prediction_config["keep_fraction"]

        if threshold_method == CUSTOM_THRESHOLD_METHODS[0]:  # "median"
            return QuickAdapterRegressorV3.median_min_max(
                pred_label, selection_method, keep_fraction
            )
        elif threshold_method == CUSTOM_THRESHOLD_METHODS[1]:  # "soft_extremum"
            return QuickAdapterRegressorV3.soft_extremum_min_max(
                pred_label,
                col_prediction_config["soft_extremum_alpha"],
                selection_method,
                keep_fraction,
            )
        elif (
            threshold_method in QuickAdapterRegressorV3._skimage_threshold_methods_set()
        ):
            return QuickAdapterRegressorV3.skimage_min_max(
                pred_label,
                threshold_method,
                selection_method,
                keep_fraction,
            )
        return -2.0, 2.0

    @staticmethod
    def _get_extrema_indices(
        pred_label: pd.Series,
    ) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        minima_indices = sp.signal.find_peaks(-pred_label)[0]
        maxima_indices = sp.signal.find_peaks(pred_label)[0]
        logger.debug(
            f"Extrema detection | find_peaks detected: "
            f"{minima_indices.size} minima, {maxima_indices.size} maxima, "
            f"total={minima_indices.size + maxima_indices.size}"
        )
        return minima_indices, maxima_indices

    @staticmethod
    def _calculate_n_kept_extrema(size: int, keep_fraction: float) -> int:
        return max(1, int(round(size * keep_fraction))) if size > 0 else 0

    @staticmethod
    def _get_ranked_peaks(
        pred_label: pd.Series,
        minima_indices: NDArray[np.intp],
        maxima_indices: NDArray[np.intp],
        keep_fraction: float = 1.0,
    ) -> tuple[pd.Series, pd.Series]:
        n_kept_minima = QuickAdapterRegressorV3._calculate_n_kept_extrema(
            minima_indices.size, keep_fraction
        )
        n_kept_maxima = QuickAdapterRegressorV3._calculate_n_kept_extrema(
            maxima_indices.size, keep_fraction
        )

        pred_label_minima = (
            pred_label.loc[
                pred_label.iloc[minima_indices].nsmallest(n_kept_minima).index
            ]
            if n_kept_minima > 0
            else pd.Series(dtype=float)
        )
        pred_label_maxima = (
            pred_label.loc[
                pred_label.iloc[maxima_indices].nlargest(n_kept_maxima).index
            ]
            if n_kept_maxima > 0
            else pd.Series(dtype=float)
        )

        logger.debug(
            f"Extrema filtering | rank_peaks: kept {n_kept_minima}/{minima_indices.size} minima, "
            f"{n_kept_maxima}/{maxima_indices.size} maxima with keep_fraction={keep_fraction}"
        )
        return pred_label_minima, pred_label_maxima

    @staticmethod
    def _get_ranked_extrema(
        pred_label: pd.Series,
        n_minima: int,
        n_maxima: int,
        keep_fraction: float = 1.0,
    ) -> tuple[pd.Series, pd.Series]:
        n_kept_minima = QuickAdapterRegressorV3._calculate_n_kept_extrema(
            n_minima, keep_fraction
        )
        n_kept_maxima = QuickAdapterRegressorV3._calculate_n_kept_extrema(
            n_maxima, keep_fraction
        )

        pred_label_minima = (
            pred_label.nsmallest(n_kept_minima)
            if n_kept_minima > 0
            else pd.Series(dtype=float)
        )
        pred_label_maxima = (
            pred_label.nlargest(n_kept_maxima)
            if n_kept_maxima > 0
            else pd.Series(dtype=float)
        )

        logger.debug(
            f"Extrema filtering | rank_extrema: kept {n_kept_minima}/{n_minima} minima, "
            f"{n_kept_maxima}/{n_maxima} maxima with keep_fraction={keep_fraction}"
        )
        return pred_label_minima, pred_label_maxima

    @staticmethod
    def get_pred_min_max(
        pred_label: pd.Series,
        selection_method: ExtremaSelectionMethod,
        keep_fraction: float = 1.0,
    ) -> tuple[pd.Series, pd.Series]:
        pred_label = (
            pd.to_numeric(pred_label, errors="coerce")
            .where(np.isfinite, np.nan)
            .dropna()
        )
        if pred_label.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        if selection_method == EXTREMA_SELECTION_METHODS[0]:  # "rank_extrema"
            minima_indices, maxima_indices = (
                QuickAdapterRegressorV3._get_extrema_indices(pred_label)
            )
            pred_label_minima, pred_label_maxima = (
                QuickAdapterRegressorV3._get_ranked_extrema(
                    pred_label,
                    minima_indices.size,
                    maxima_indices.size,
                    keep_fraction,
                )
            )

        elif selection_method == EXTREMA_SELECTION_METHODS[1]:  # "rank_peaks"
            minima_indices, maxima_indices = (
                QuickAdapterRegressorV3._get_extrema_indices(pred_label)
            )
            pred_label_minima, pred_label_maxima = (
                QuickAdapterRegressorV3._get_ranked_peaks(
                    pred_label, minima_indices, maxima_indices, keep_fraction
                )
            )

        elif selection_method == EXTREMA_SELECTION_METHODS[2]:  # "partition"
            eps = np.finfo(float).eps

            pred_label_maxima = pred_label[pred_label > eps]
            pred_label_minima = pred_label[pred_label < -eps]
        else:
            raise ValueError(
                f"Invalid selection_method value {selection_method!r}: "
                f"supported values are {', '.join(EXTREMA_SELECTION_METHODS)}"
            )

        return pred_label_minima, pred_label_maxima

    @staticmethod
    def safe_min_pred(pred_label: pd.Series) -> float:
        try:
            pred_label_minimum = pred_label.min()
        except Exception:
            pred_label_minimum = None
        if (
            pred_label_minimum is not None
            and isinstance(pred_label_minimum, (int, float, np.number))
            and np.isfinite(pred_label_minimum)
        ):
            return float(pred_label_minimum)
        return -2.0

    @staticmethod
    def safe_max_pred(pred_label: pd.Series) -> float:
        try:
            pred_label_maximum = pred_label.max()
        except Exception:
            pred_label_maximum = None
        if (
            pred_label_maximum is not None
            and isinstance(pred_label_maximum, (int, float, np.number))
            and np.isfinite(pred_label_maximum)
        ):
            return float(pred_label_maximum)
        return 2.0

    @staticmethod
    def soft_extremum_min_max(
        pred_label: pd.Series,
        alpha: float,
        selection_method: ExtremaSelectionMethod,
        keep_fraction: float = 1.0,
    ) -> tuple[float, float]:
        if alpha < 0:
            raise ValueError(f"Invalid alpha value {alpha!r}: must be >= 0")
        pred_label_minima, pred_label_maxima = QuickAdapterRegressorV3.get_pred_min_max(
            pred_label, selection_method, keep_fraction
        )
        soft_minimum = soft_extremum(pred_label_minima, alpha=-alpha)
        if not np.isfinite(soft_minimum):
            soft_minimum = QuickAdapterRegressorV3.safe_min_pred(pred_label)
        soft_maximum = soft_extremum(pred_label_maxima, alpha=alpha)
        if not np.isfinite(soft_maximum):
            soft_maximum = QuickAdapterRegressorV3.safe_max_pred(pred_label)
        return soft_minimum, soft_maximum

    @staticmethod
    def median_min_max(
        pred_label: pd.Series,
        selection_method: ExtremaSelectionMethod,
        keep_fraction: float = 1.0,
    ) -> tuple[float, float]:
        pred_label_minima, pred_label_maxima = QuickAdapterRegressorV3.get_pred_min_max(
            pred_label, selection_method, keep_fraction
        )

        if pred_label_minima.empty:
            min_val = np.nan
        else:
            min_val = np.nanmedian(pred_label_minima.to_numpy())
        if not np.isfinite(min_val):
            min_val = QuickAdapterRegressorV3.safe_min_pred(pred_label)

        if pred_label_maxima.empty:
            max_val = np.nan
        else:
            max_val = np.nanmedian(pred_label_maxima.to_numpy())
        if not np.isfinite(max_val):
            max_val = QuickAdapterRegressorV3.safe_max_pred(pred_label)

        return min_val, max_val

    @staticmethod
    def skimage_min_max(
        pred_label: pd.Series,
        method: SkimageThresholdMethod,
        selection_method: ExtremaSelectionMethod,
        keep_fraction: float = 1.0,
    ) -> tuple[float, float]:
        pred_label_minima, pred_label_maxima = QuickAdapterRegressorV3.get_pred_min_max(
            pred_label, selection_method, keep_fraction
        )

        try:
            threshold_func = getattr(skimage.filters, f"threshold_{method}")
        except AttributeError:
            raise ValueError(
                f"Invalid skimage threshold method value {method!r}: "
                f"supported values are {', '.join(SKIMAGE_THRESHOLD_METHODS)}"
            )

        min_func = QuickAdapterRegressorV3.apply_skimage_threshold
        max_func = QuickAdapterRegressorV3.apply_skimage_threshold

        min_val = min_func(pred_label_minima, threshold_func)
        if not np.isfinite(min_val):
            min_val = QuickAdapterRegressorV3.safe_min_pred(pred_label)

        max_val = max_func(pred_label_maxima, threshold_func)
        if not np.isfinite(max_val):
            max_val = QuickAdapterRegressorV3.safe_max_pred(pred_label)

        return min_val, max_val

    @staticmethod
    def apply_skimage_threshold(
        series: pd.Series, threshold_func: Callable[[NDArray[np.floating]], float]
    ) -> float:
        values = series.to_numpy()

        if values.size == 0:
            return np.nan
        if (
            values.size == 1
            or np.unique(values).size < 3
            or np.allclose(values, values[0])
        ):
            return np.nanmedian(values)
        try:
            return threshold_func(values)
        except Exception as e:
            logger.warning(
                f"Threshold function {threshold_func.__name__} failed on series {series.name}: {e!r}, falling back to median",
                exc_info=True,
            )
            return np.nanmedian(values)

    @staticmethod
    def _hellinger_distance(
        matrix: NDArray[np.floating],
        reference_point: NDArray[np.floating],
        *,
        weights: Optional[NDArray[np.floating]] = None,
        standardized: bool = False,
    ) -> NDArray[np.floating]:
        if standardized:
            variances = np.nanvar(np.sqrt(matrix), axis=0, ddof=1)
            if np.any(variances <= 0):
                raise ValueError(
                    "Invalid data for shellinger metric: "
                    "requires non-zero variance for all objectives"
                )
            weights = 1.0 / variances
        elif weights is None:
            weights = np.ones(matrix.shape[1])

        return (
            np.sqrt(
                np.nansum(
                    weights * (np.sqrt(matrix) - np.sqrt(reference_point)) ** 2, axis=1
                )
            )
            / QuickAdapterRegressorV3._SQRT_2
        )

    @staticmethod
    def _power_mean_distance(
        matrix: NDArray[np.floating],
        reference_point: NDArray[np.floating],
        distance_metric: str,
        *,
        weights: Optional[NDArray[np.floating]] = None,
        p: Optional[float] = None,
        mode: ValidationMode = "none",
        p_ctx: str = "p",
    ) -> NDArray[np.floating]:
        power = (
            QuickAdapterRegressorV3._POWER_MEAN_MAP[distance_metric]
            if distance_metric in QuickAdapterRegressorV3._power_mean_metrics_set()
            else (
                QuickAdapterRegressorV3._validate_power_mean_p(p, ctx=p_ctx, mode=mode)
                or 1.0
            )
        )
        if weights is None:
            weights = np.ones(matrix.shape[1])

        return sp.stats.pmean(
            reference_point.flatten() if reference_point.ndim > 1 else reference_point,
            p=power,
            weights=weights,
        ) - sp.stats.pmean(matrix, p=power, weights=weights, axis=1)

    @staticmethod
    def _weighted_sum_distance(
        matrix: NDArray[np.floating],
        reference_point: NDArray[np.floating],
        *,
        weights: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        return (reference_point - matrix) @ weights

    @staticmethod
    def _compromise_programming_scores(
        normalized_matrix: NDArray[np.floating],
        distance_metric: str,
        *,
        weights: Optional[NDArray[np.floating]] = None,
        p: Optional[float] = None,
    ) -> NDArray[np.floating]:
        n_samples, n_objectives = normalized_matrix.shape

        if n_samples == 0:
            return np.array([])
        if n_samples == 1:
            return np.array([0.0])

        if weights is None:
            weights = np.ones(n_objectives)

        ideal_point = np.ones(n_objectives)

        if distance_metric in QuickAdapterRegressorV3._scipy_metrics_set():
            return sp.spatial.distance.cdist(
                normalized_matrix,
                ideal_point.reshape(1, -1),
                metric=distance_metric,
                **QuickAdapterRegressorV3._prepare_distance_kwargs(
                    distance_metric,
                    weights=weights,
                    p=p,
                    mode="warn",
                    metric_ctx="label_distance_metric",
                    p_ctx="label_distance_p",
                ),
            ).flatten()

        if distance_metric in {
            QuickAdapterRegressorV3._DISTANCE_METRICS[8],  # "hellinger"
            QuickAdapterRegressorV3._DISTANCE_METRICS[9],  # "shellinger"
        }:
            return QuickAdapterRegressorV3._hellinger_distance(
                normalized_matrix,
                ideal_point,
                weights=weights,
                standardized=(
                    distance_metric
                    == QuickAdapterRegressorV3._DISTANCE_METRICS[9]  # "shellinger"
                ),
            )

        if distance_metric in (
            QuickAdapterRegressorV3._power_mean_metrics_set()
            | {QuickAdapterRegressorV3._DISTANCE_METRICS[15]}  # "power_mean"
        ):
            return QuickAdapterRegressorV3._power_mean_distance(
                normalized_matrix,
                ideal_point,
                distance_metric,
                weights=weights,
                p=p,
                mode="warn",
                p_ctx="label_distance_p",
            )

        if (
            distance_metric == QuickAdapterRegressorV3._DISTANCE_METRICS[16]
        ):  # "weighted_sum"
            assert weights is not None
            return QuickAdapterRegressorV3._weighted_sum_distance(
                normalized_matrix,
                ideal_point,
                weights=weights,
            )

        raise ValueError(
            f"Invalid distance_metric value {distance_metric!r} for {QuickAdapterRegressorV3._DISTANCE_METHODS[0]}: "
            f"supported values are {', '.join(QuickAdapterRegressorV3._DISTANCE_METRICS)}"
        )

    @staticmethod
    def _pairwise_distance_sums(
        matrix: NDArray[np.floating],
        distance_metric: str,
        *,
        weights: Optional[NDArray[np.floating]] = None,
        p: Optional[float] = None,
    ) -> NDArray[np.floating]:
        if matrix.ndim != 2:
            raise ValueError(
                f"Invalid matrix (shape={matrix.shape}, ndim={matrix.ndim}): "
                f"must be 2-dimensional"
            )
        if matrix.shape[1] == 0:
            raise ValueError(
                f"Invalid matrix (shape={matrix.shape}): must have at least one feature"
            )

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "Invalid matrix: must contain only finite values (no NaN or inf)"
            )

        n = matrix.shape[0]
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([0.0])

        pdist_kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
            distance_metric,
            weights=weights,
            p=p,
            mode="warn",
            metric_ctx="label_density_metric",
            p_ctx="label_density_p",
        )

        pairwise_distances_vector = sp.spatial.distance.pdist(
            matrix, metric=distance_metric, **pdist_kwargs
        )

        sums = np.zeros(n, dtype=float)

        idx_i, idx_j = np.triu_indices(n, k=1)
        np.add.at(sums, idx_i, pairwise_distances_vector)
        np.add.at(sums, idx_j, pairwise_distances_vector)

        return sums

    @staticmethod
    def _topsis_scores(
        normalized_matrix: NDArray[np.floating],
        distance_metric: str,
        *,
        weights: Optional[NDArray[np.floating]] = None,
        p: Optional[float] = None,
    ) -> NDArray[np.floating]:
        n_samples, n_objectives = normalized_matrix.shape

        if n_samples == 0:
            return np.array([])
        if n_samples == 1:
            return np.array([0.5])

        if weights is None:
            weights = np.ones(n_objectives)

        ideal_point = np.ones(n_objectives)
        anti_ideal_point = np.zeros(n_objectives)

        if distance_metric in QuickAdapterRegressorV3._scipy_metrics_set():
            cdist_kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
                distance_metric=distance_metric,
                weights=weights,
                p=p,
                mode="warn",
                metric_ctx="label_distance_metric",
                p_ctx="label_distance_p",
            )

            dist_to_ideal = sp.spatial.distance.cdist(
                normalized_matrix,
                ideal_point.reshape(1, -1),
                metric=distance_metric,
                **cdist_kwargs,
            ).flatten()
            dist_to_anti_ideal = sp.spatial.distance.cdist(
                normalized_matrix,
                anti_ideal_point.reshape(1, -1),
                metric=distance_metric,
                **cdist_kwargs,
            ).flatten()
        elif distance_metric in {
            QuickAdapterRegressorV3._DISTANCE_METRICS[8],  # "hellinger"
            QuickAdapterRegressorV3._DISTANCE_METRICS[9],  # "shellinger"
        }:
            dist_to_ideal = QuickAdapterRegressorV3._hellinger_distance(
                normalized_matrix,
                ideal_point,
                weights=weights,
                standardized=(
                    distance_metric
                    == QuickAdapterRegressorV3._DISTANCE_METRICS[9]  # "shellinger"
                ),
            )
            dist_to_anti_ideal = QuickAdapterRegressorV3._hellinger_distance(
                normalized_matrix,
                anti_ideal_point,
                weights=weights,
                standardized=(
                    distance_metric
                    == QuickAdapterRegressorV3._DISTANCE_METRICS[9]  # "shellinger"
                ),
            )
        elif distance_metric in (
            QuickAdapterRegressorV3._power_mean_metrics_set()
            | {QuickAdapterRegressorV3._DISTANCE_METRICS[15]}  # "power_mean"
        ):
            dist_to_ideal = np.abs(
                QuickAdapterRegressorV3._power_mean_distance(
                    normalized_matrix,
                    ideal_point,
                    distance_metric,
                    weights=weights,
                    p=p,
                    mode="warn",
                    p_ctx="label_distance_p",
                )
            )
            dist_to_anti_ideal = np.abs(
                QuickAdapterRegressorV3._power_mean_distance(
                    normalized_matrix,
                    anti_ideal_point,
                    distance_metric,
                    weights=weights,
                    p=p,
                    mode="warn",
                    p_ctx="label_distance_p",
                )
            )
        elif (
            distance_metric == QuickAdapterRegressorV3._DISTANCE_METRICS[16]
        ):  # "weighted_sum"
            assert weights is not None
            dist_to_ideal = np.abs(
                QuickAdapterRegressorV3._weighted_sum_distance(
                    normalized_matrix,
                    ideal_point,
                    weights=weights,
                )
            )
            dist_to_anti_ideal = np.abs(
                QuickAdapterRegressorV3._weighted_sum_distance(
                    normalized_matrix,
                    anti_ideal_point,
                    weights=weights,
                )
            )
        else:
            raise ValueError(
                f"Invalid distance_metric value {distance_metric!r} for {QuickAdapterRegressorV3._DISTANCE_METHODS[1]}: "
                f"supported values are {', '.join(QuickAdapterRegressorV3._DISTANCE_METRICS)}"
            )

        denominator = dist_to_ideal + dist_to_anti_ideal
        zero_mask = np.isclose(denominator, 0.0)
        denominator[zero_mask] = 1.0
        scores = dist_to_ideal / denominator
        scores[zero_mask] = 0.5

        return scores

    @staticmethod
    def _calculate_trial_distance_to_ideal(
        normalized_matrix: NDArray[np.floating],
        trial_index: int,
        ideal_point_2d: NDArray[np.floating],
        distance_metric: str,
        *,
        weights: Optional[NDArray[np.floating]] = None,
        p: Optional[float] = None,
    ) -> float:
        cdist_kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
            distance_metric=distance_metric,
            weights=weights,
            p=p,
            mode="warn",
            metric_ctx="label_cluster_metric",
            p_ctx="label_cluster_p",
        )

        return sp.spatial.distance.cdist(
            normalized_matrix[[trial_index]],
            ideal_point_2d,
            metric=distance_metric,
            **cdist_kwargs,
        ).item()

    def _select_best_trial_from_cluster(
        self,
        normalized_matrix: NDArray[np.floating],
        trial_selection_method: DistanceMethod,
        best_cluster_indices: NDArray[np.intp],
        ideal_point_2d: NDArray[np.floating],
        distance_metric: str,
        *,
        weights: Optional[NDArray[np.floating]] = None,
        p: Optional[float] = None,
    ) -> tuple[int, float]:
        if best_cluster_indices.size == 1:
            best_trial_index = best_cluster_indices[0]
            best_trial_distance = (
                QuickAdapterRegressorV3._calculate_trial_distance_to_ideal(
                    normalized_matrix,
                    best_trial_index,
                    ideal_point_2d,
                    distance_metric,
                    weights=weights,
                    p=p,
                )
            )
            return best_trial_index, best_trial_distance

        if (
            trial_selection_method == QuickAdapterRegressorV3._DISTANCE_METHODS[0]
        ):  # "compromise_programming"
            scores = QuickAdapterRegressorV3._compromise_programming_scores(
                normalized_matrix[best_cluster_indices],
                distance_metric,
                weights=weights,
                p=p,
            )
        elif (
            trial_selection_method == QuickAdapterRegressorV3._DISTANCE_METHODS[1]
        ):  # "topsis"
            scores = QuickAdapterRegressorV3._topsis_scores(
                normalized_matrix[best_cluster_indices],
                distance_metric,
                weights=weights,
                p=p,
            )
        else:
            raise ValueError(
                f"Invalid trial_selection_method value {trial_selection_method!r}: "
                f"supported values are {', '.join(QuickAdapterRegressorV3._DISTANCE_METHODS)}"
            )

        min_score_position = np.nanargmin(scores)
        best_trial_index = best_cluster_indices[min_score_position]
        best_trial_distance = (
            QuickAdapterRegressorV3._calculate_trial_distance_to_ideal(
                normalized_matrix,
                best_trial_index,
                ideal_point_2d,
                distance_metric,
                weights=weights,
                p=p,
            )
        )
        return best_trial_index, best_trial_distance

    def _cluster_based_selection(
        self,
        normalized_matrix: NDArray[np.floating],
        cluster_method: ClusterMethod,
        *,
        distance_metric: str,
        selection_method: DistanceMethod,
        trial_selection_method: DistanceMethod,
        weights: Optional[NDArray[np.floating]] = None,
        p: Optional[float] = None,
    ) -> NDArray[np.floating]:
        n_samples, n_objectives = normalized_matrix.shape

        if n_samples == 0:
            return np.array([])
        if n_samples == 1:
            return np.array([0.0])

        ideal_point_2d = np.ones((1, n_objectives))

        n_clusters = QuickAdapterRegressorV3._get_n_clusters(normalized_matrix)

        if cluster_method in {
            QuickAdapterRegressorV3._CLUSTER_METHODS[0],  # "kmeans"
            QuickAdapterRegressorV3._CLUSTER_METHODS[1],  # "kmeans2"
        }:
            if (
                cluster_method == QuickAdapterRegressorV3._CLUSTER_METHODS[0]
            ):  # "kmeans"
                kmeans = sklearn.cluster.KMeans(
                    n_clusters=n_clusters, random_state=42, n_init=10
                )
                cluster_labels = kmeans.fit_predict(normalized_matrix)
                cluster_centers = kmeans.cluster_centers_
            else:  # kmeans2
                cluster_centers, cluster_labels = sp.cluster.vq.kmeans2(
                    normalized_matrix, n_clusters, rng=42, minit="++"
                )

            if (
                selection_method == QuickAdapterRegressorV3._DISTANCE_METHODS[0]
            ):  # "compromise_programming"
                cluster_center_scores = (
                    QuickAdapterRegressorV3._compromise_programming_scores(
                        cluster_centers,
                        distance_metric,
                        p=p,
                    )
                )
            elif (
                selection_method == QuickAdapterRegressorV3._DISTANCE_METHODS[1]
            ):  # "topsis"
                cluster_center_scores = QuickAdapterRegressorV3._topsis_scores(
                    cluster_centers,
                    distance_metric,
                    p=p,
                )
            else:
                raise ValueError(
                    f"Invalid selection_method value {selection_method!r}: "
                    f"supported values are {', '.join(QuickAdapterRegressorV3._DISTANCE_METHODS)}"
                )
            ordered_cluster_indices = np.argsort(cluster_center_scores)

            best_cluster_indices = None
            for cluster_index in ordered_cluster_indices:
                cluster_indices = np.flatnonzero(cluster_labels == cluster_index)
                if cluster_indices.size > 0:
                    best_cluster_indices = cluster_indices
                    break

            trial_distances = np.full(n_samples, np.inf)
            if best_cluster_indices is not None and best_cluster_indices.size > 0:
                best_trial_index, best_trial_distance = (
                    self._select_best_trial_from_cluster(
                        normalized_matrix,
                        trial_selection_method,
                        best_cluster_indices,
                        ideal_point_2d,
                        distance_metric,
                        weights=weights,
                        p=p,
                    )
                )
                trial_distances[best_trial_index] = best_trial_distance
            return trial_distances

        elif (
            cluster_method == QuickAdapterRegressorV3._CLUSTER_METHODS[2]
        ):  # "kmedoids"
            kmedoids_kwargs: dict[str, Any] = {
                "metric": distance_metric,
                "random_state": 42,
                "init": "k-medoids++",
                "method": "pam",
            }
            kmedoids = KMedoids(n_clusters=n_clusters, **kmedoids_kwargs)
            cluster_labels = kmedoids.fit_predict(normalized_matrix)
            medoid_indices = kmedoids.medoid_indices_

            if (
                selection_method == QuickAdapterRegressorV3._DISTANCE_METHODS[0]
            ):  # "compromise_programming"
                medoid_scores = QuickAdapterRegressorV3._compromise_programming_scores(
                    normalized_matrix[medoid_indices],
                    distance_metric,
                    p=p,
                )
            elif (
                selection_method == QuickAdapterRegressorV3._DISTANCE_METHODS[1]
            ):  # "topsis"
                medoid_scores = QuickAdapterRegressorV3._topsis_scores(
                    normalized_matrix[medoid_indices],
                    distance_metric,
                    p=p,
                )
            else:
                raise ValueError(
                    f"Invalid selection_method value {selection_method!r}: "
                    f"supported values are {', '.join(QuickAdapterRegressorV3._DISTANCE_METHODS)}"
                )
            best_medoid_score_position = np.nanargmin(medoid_scores)
            best_medoid_index = medoid_indices[best_medoid_score_position]
            cluster_index = cluster_labels[best_medoid_index]
            best_cluster_indices = np.flatnonzero(cluster_labels == cluster_index)

            trial_distances = np.full(n_samples, np.inf)
            if best_cluster_indices is not None and best_cluster_indices.size > 0:
                best_trial_index, best_trial_distance = (
                    self._select_best_trial_from_cluster(
                        normalized_matrix,
                        trial_selection_method,
                        best_cluster_indices,
                        ideal_point_2d,
                        distance_metric,
                        weights=weights,
                        p=p,
                    )
                )
                trial_distances[best_trial_index] = best_trial_distance
            return trial_distances

        else:
            raise ValueError(
                f"Invalid cluster_method value {cluster_method!r}: "
                f"supported values are {', '.join(QuickAdapterRegressorV3._CLUSTER_METHODS)}"
            )

    @staticmethod
    def _knn_based_selection(
        normalized_matrix: NDArray[np.floating],
        aggregation: DensityAggregation,
        *,
        distance_metric: str,
        n_neighbors: int,
        weights: Optional[NDArray[np.floating]] = None,
        p: Optional[float] = None,
        aggregation_param: Optional[float] = None,
    ) -> NDArray[np.floating]:
        n_samples, _ = normalized_matrix.shape

        if n_samples == 0:
            return np.array([])
        if n_samples == 1:
            return np.array([0.0])

        knn_kwargs = QuickAdapterRegressorV3._prepare_knn_kwargs(
            distance_metric,
            weights=weights,
            p=p,
            mode="raise",
        )

        nbrs = sklearn.neighbors.NearestNeighbors(
            n_neighbors=min(n_neighbors, n_samples - 1) + 1,
            metric=distance_metric,
            **knn_kwargs,
        ).fit(normalized_matrix)
        distances, _ = nbrs.kneighbors(normalized_matrix)
        neighbor_distances = distances[:, 1:]

        if neighbor_distances.shape[1] < 1:
            return np.full(n_samples, np.inf)

        if (
            aggregation == QuickAdapterRegressorV3._DENSITY_AGGREGATIONS[0]
        ):  # "power_mean"
            power = (
                aggregation_param
                if aggregation_param is not None
                else QuickAdapterRegressorV3._get_label_density_aggregation_param_default(
                    aggregation
                )
            )
            if power is None:
                power = 1.0
            power = QuickAdapterRegressorV3._validate_power_mean_p(
                power,
                ctx="label_density_aggregation_param",
            )
            assert power is not None
            return np.asarray(sp.stats.pmean(neighbor_distances, p=power, axis=1))
        elif (
            aggregation == QuickAdapterRegressorV3._DENSITY_AGGREGATIONS[1]
        ):  # "quantile"
            quantile = (
                aggregation_param
                if aggregation_param is not None
                else QuickAdapterRegressorV3._get_label_density_aggregation_param_default(
                    aggregation
                )
            )
            if quantile is None:
                quantile = 0.5
            quantile = QuickAdapterRegressorV3._validate_quantile_q(
                quantile,
                ctx="label_density_aggregation_param",
            )
            assert quantile is not None
            return np.asarray(np.nanquantile(neighbor_distances, quantile, axis=1))
        elif aggregation == QuickAdapterRegressorV3._DENSITY_AGGREGATIONS[2]:  # "min"
            return np.nanmin(neighbor_distances, axis=1)
        elif aggregation == QuickAdapterRegressorV3._DENSITY_AGGREGATIONS[3]:  # "max"
            return np.nanmax(neighbor_distances, axis=1)
        else:
            raise ValueError(
                f"Invalid aggregation value {aggregation!r}: "
                f"supported values are {', '.join(QuickAdapterRegressorV3._DENSITY_AGGREGATIONS)}"
            )

    @staticmethod
    def _normalize_objective_values(
        objective_values_matrix: NDArray[np.floating],
        directions: list[optuna.study.StudyDirection],
    ) -> NDArray[np.floating]:
        if objective_values_matrix.ndim != 2:
            raise ValueError(
                f"Invalid objective_values_matrix (shape={objective_values_matrix.shape}, "
                f"ndim={objective_values_matrix.ndim}): must be 2-dimensional"
            )

        n_samples, n_objectives = objective_values_matrix.shape
        if n_samples == 0 or n_objectives == 0:
            raise ValueError(
                "Invalid objective_values_matrix: must have at least one sample and one objective"
            )

        if len(directions) != n_objectives:
            raise ValueError(
                f"Invalid directions: length ({len(directions)}) must match number of objectives ({n_objectives})"
            )

        normalized_matrix = np.zeros_like(objective_values_matrix, dtype=float)

        for i in range(n_objectives):
            current_column = objective_values_matrix[:, i]
            current_direction = directions[i]

            is_neg_inf_mask = np.isneginf(current_column)
            is_pos_inf_mask = np.isposinf(current_column)

            if current_direction == optuna.study.StudyDirection.MAXIMIZE:
                normalized_matrix[is_neg_inf_mask, i] = 0.0
                normalized_matrix[is_pos_inf_mask, i] = 1.0
            else:
                normalized_matrix[is_neg_inf_mask, i] = 1.0
                normalized_matrix[is_pos_inf_mask, i] = 0.0

            is_finite_mask = np.isfinite(current_column)

            if np.any(is_finite_mask):
                finite_col = current_column[is_finite_mask]
                finite_min_val = np.min(finite_col)
                finite_max_val = np.max(finite_col)
                finite_range_val = finite_max_val - finite_min_val

                if np.isclose(finite_range_val, 0.0):
                    if np.any(is_pos_inf_mask) and np.any(is_neg_inf_mask):
                        normalized_matrix[is_finite_mask, i] = 0.5
                    elif np.any(is_pos_inf_mask):
                        normalized_matrix[is_finite_mask, i] = (
                            0.0
                            if current_direction == optuna.study.StudyDirection.MAXIMIZE
                            else 1.0
                        )
                    elif np.any(is_neg_inf_mask):
                        normalized_matrix[is_finite_mask, i] = (
                            1.0
                            if current_direction == optuna.study.StudyDirection.MAXIMIZE
                            else 0.0
                        )
                    else:
                        normalized_matrix[is_finite_mask, i] = 0.5
                else:
                    if current_direction == optuna.study.StudyDirection.MAXIMIZE:
                        normalized_matrix[is_finite_mask, i] = (
                            finite_col - finite_min_val
                        ) / finite_range_val
                    else:
                        normalized_matrix[is_finite_mask, i] = (
                            finite_max_val - finite_col
                        ) / finite_range_val

        if not np.all(np.isfinite(normalized_matrix)):
            raise ValueError(
                "Invalid normalized_matrix: must contain only finite values after normalization"
            )

        return normalized_matrix

    @staticmethod
    def _get_n_clusters(
        matrix: NDArray[np.floating],
        *,
        min_n_clusters: int = 2,
        max_n_clusters: int = 10,
    ) -> int:
        n_samples = matrix.shape[0]
        if n_samples <= 1:
            return 1
        n_uniques = np.unique(matrix, axis=0).shape[0]
        upper_bound = min(max_n_clusters, n_uniques, n_samples)
        if upper_bound < 2:
            return 1
        lower_bound = min(min_n_clusters, upper_bound)
        if n_uniques <= 3:
            return min(n_uniques, upper_bound)
        n_clusters = int(round((np.log2(n_uniques) + np.sqrt(n_uniques)) / 2.0))
        return min(max(lower_bound, n_clusters), upper_bound)

    def _calculate_distances(
        self,
        normalized_matrix: NDArray[np.floating],
        selection_method: SelectionMethod,
    ) -> NDArray[np.floating]:
        if normalized_matrix.ndim != 2:
            raise ValueError(
                f"Invalid normalized_matrix (shape={normalized_matrix.shape}, "
                f"ndim={normalized_matrix.ndim}): must be 2-dimensional"
            )

        n_samples, n_objectives = normalized_matrix.shape
        if n_samples == 0 or n_objectives == 0:
            raise ValueError(
                "Invalid normalized_matrix: must have at least one sample and one objective"
            )
        if not np.all(np.isfinite(normalized_matrix)):
            raise ValueError(
                "Invalid normalized_matrix: must contain only finite values (no NaN or inf)"
            )

        label_config = self._resolve_label_method_config(selection_method)
        method = label_config["method"]
        category = label_config["category"]

        label_p_order = self.ft_params.get("label_p_order")
        label_weights = self.ft_params.get("label_weights")
        weights = QuickAdapterRegressorV3._validate_label_weights(
            label_weights,
            n_objectives,
            ctx="label_weights",
            mode="raise",
        )

        if n_samples == 1:
            if method in {
                QuickAdapterRegressorV3._SELECTION_METHODS[6],  # "medoid"
                QuickAdapterRegressorV3._SELECTION_METHODS[2],  # "kmeans"
                QuickAdapterRegressorV3._SELECTION_METHODS[3],  # "kmeans2"
                QuickAdapterRegressorV3._SELECTION_METHODS[4],  # "kmedoids"
                QuickAdapterRegressorV3._SELECTION_METHODS[5],  # "knn"
            }:
                return np.array([0.0])

        if category == "distance":
            distance_metric = label_config["distance_metric"]
            p = QuickAdapterRegressorV3._resolve_p_order(
                distance_metric,
                label_p_order,
                ctx=f"label_p_order for {method}",
                mode="none",
            )

            if (
                method == QuickAdapterRegressorV3._DISTANCE_METHODS[0]
            ):  # "compromise_programming"
                return QuickAdapterRegressorV3._compromise_programming_scores(
                    normalized_matrix,
                    distance_metric,
                    weights=weights,
                    p=p,
                )
            if method == QuickAdapterRegressorV3._DISTANCE_METHODS[1]:  # "topsis"
                return QuickAdapterRegressorV3._topsis_scores(
                    normalized_matrix,
                    distance_metric,
                    weights=weights,
                    p=p,
                )

        if category == "cluster":
            cluster_metric = label_config["distance_metric"]
            cluster_selection_method = label_config["selection_method"]
            trial_selection_method = label_config["trial_selection_method"]

            p = QuickAdapterRegressorV3._resolve_p_order(
                cluster_metric,
                label_p_order,
                ctx=f"label_p_order for {method}",
                mode="none",
            )
            return self._cluster_based_selection(
                normalized_matrix,
                method,
                distance_metric=cluster_metric,
                selection_method=cluster_selection_method,
                trial_selection_method=trial_selection_method,
                weights=weights,
                p=p,
            )

        if category == "density":
            density_method = cast(DensityMethod, method)
            density_metric = label_config["distance_metric"]
            p = QuickAdapterRegressorV3._resolve_p_order(
                density_metric,
                label_p_order,
                ctx=f"label_p_order for {density_method}",
                mode="none",
            )

            if density_method == QuickAdapterRegressorV3._DENSITY_METHODS[0]:  # "knn"
                knn_n_neighbors = int(label_config["n_neighbors"])
                knn_aggregation = cast(DensityAggregation, label_config["aggregation"])
                if (
                    knn_aggregation
                    not in QuickAdapterRegressorV3._density_aggregations_set()
                ):
                    raise ValueError(
                        f"Invalid aggregation value in label_config {knn_aggregation!r}: "
                        f"supported values are {', '.join(QuickAdapterRegressorV3._DENSITY_AGGREGATIONS)}"
                    )
                knn_aggregation_param = label_config["aggregation_param"]
                return QuickAdapterRegressorV3._knn_based_selection(
                    normalized_matrix,
                    knn_aggregation,
                    distance_metric=density_metric,
                    n_neighbors=knn_n_neighbors,
                    weights=weights,
                    p=p,
                    aggregation_param=knn_aggregation_param,
                )

            if (
                density_method == QuickAdapterRegressorV3._DENSITY_METHODS[1]
            ):  # "medoid"
                return QuickAdapterRegressorV3._pairwise_distance_sums(
                    normalized_matrix,
                    density_metric,
                    weights=weights,
                    p=p,
                )

        raise ValueError(
            f"Invalid label_method value {selection_method!r}: "
            f"supported values are {', '.join(QuickAdapterRegressorV3._SELECTION_METHODS)}"
        )

    def _get_multi_objective_study_best_trial(
        self, namespace: OptunaNamespace, study: optuna.study.Study
    ) -> Optional[optuna.trial.FrozenTrial]:
        if namespace not in {QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]}:  # "label"
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]}"  # "label"
            )
        n_objectives = len(study.directions)
        if n_objectives < 2:
            raise ValueError(
                f"Multi-objective study must have at least 2 objectives, but got {n_objectives}"
            )
        if not QuickAdapterRegressorV3.optuna_study_has_best_trials(study):
            return None

        label_method = self.ft_params.get(
            "label_method", QuickAdapterRegressorV3.LABEL_METHOD_DEFAULT
        )  # "compromise_programming"

        best_trials = [
            trial
            for trial in study.best_trials
            if (
                isinstance(trial.values, list)
                and len(trial.values) == n_objectives
                and all(
                    isinstance(value, (int, float))
                    and (np.isfinite(value) or np.isinf(value))
                    for value in trial.values
                )
            )
        ]
        if not best_trials:
            return None

        objective_values_matrix = np.array(
            [trial.values for trial in best_trials], dtype=float
        )
        normalized_matrix = QuickAdapterRegressorV3._normalize_objective_values(
            objective_values_matrix, study.directions
        )

        trial_distances = self._calculate_distances(
            normalized_matrix,
            selection_method=label_method,
        )

        return best_trials[np.nanargmin(trial_distances)]

    def optuna_optimize(
        self,
        pair: str,
        namespace: OptunaNamespace,
        objective: ObjectiveFuncType,
        direction: Optional[optuna.study.StudyDirection] = None,
        directions: Optional[list[optuna.study.StudyDirection]] = None,
    ) -> Optional[optuna.study.Study]:
        if direction is not None and directions is not None:
            raise ValueError(
                "Cannot specify both 'direction' and 'directions'. Use one or the other"
            )
        is_study_single_objective = direction is not None and directions is None
        if (
            not is_study_single_objective
            and isinstance(directions, list)
            and len(directions) < 2
        ):
            raise ValueError(
                "Multi-objective study must have at least 2 objectives specified"
            )

        study = self.optuna_create_study(
            pair=pair,
            namespace=namespace,
            direction=direction,
            directions=directions,
        )
        if not study:
            return

        if self._optuna_config.get("warm_start"):
            self.optuna_enqueue_previous_best_params(pair, namespace, study)

        objective_type = "single" if is_study_single_objective else "multi"
        logger.info(
            f"[{pair}] Optuna {namespace} {objective_type} objective hyperopt started"
        )
        start_time = time.time()
        try:
            study.optimize(
                objective,
                n_trials=self._optuna_config.get(
                    "n_trials", QuickAdapterRegressorV3.OPTUNA_N_TRIALS_DEFAULT
                ),
                n_jobs=self._optuna_config.get(
                    "n_jobs", QuickAdapterRegressorV3.OPTUNA_N_JOBS_DEFAULT
                ),
                timeout=self._optuna_config.get(
                    "timeout", QuickAdapterRegressorV3.OPTUNA_TIMEOUT_DEFAULT
                ),
                gc_after_trial=True,
            )
        except Exception as e:
            time_spent = time.time() - start_time
            logger.error(
                f"[{pair}] Optuna {namespace} {objective_type} objective hyperopt failed ({time_spent:.2f} secs): {e!r}",
                exc_info=True,
            )
            return

        time_spent = time.time() - start_time
        if is_study_single_objective:
            if not QuickAdapterRegressorV3.optuna_study_has_best_trial(study):
                logger.error(
                    f"[{pair}] Optuna {namespace} {objective_type} objective hyperopt failed ({time_spent:.2f} secs): no study best trial found"
                )
                return
            self.set_optuna_value(pair, namespace, study.best_value)
            self.set_optuna_params(pair, namespace, study.best_params)
            study_best_results = {
                "value": self.get_optuna_value(pair, namespace),
                **self.get_optuna_params(pair, namespace),
            }
            metric_log_msg = ""
        else:
            try:
                best_trial = self._get_multi_objective_study_best_trial(
                    namespace, study
                )
            except Exception as e:
                logger.error(
                    f"[{pair}] Optuna {namespace} {objective_type} objective hyperopt failed ({time_spent:.2f} secs): {e!r}",
                    exc_info=True,
                )
                best_trial = None
            if not best_trial:
                logger.error(
                    f"[{pair}] Optuna {namespace} {objective_type} objective hyperopt failed ({time_spent:.2f} secs): no study best trial found"
                )
                return
            self.set_optuna_values(pair, namespace, best_trial.values)
            self.set_optuna_params(pair, namespace, best_trial.params)
            study_best_results = {
                "values": self.get_optuna_values(pair, namespace),
                **self.get_optuna_params(pair, namespace),
            }
            label_config = self._resolve_label_method_config(
                self.ft_params.get(
                    "label_method", QuickAdapterRegressorV3.LABEL_METHOD_DEFAULT
                )
            )
            metric_log_msg = f" ({format_dict(label_config, style='params')})"
        logger.info(
            f"[{pair}] Optuna {namespace} {objective_type} objective hyperopt completed"
            f"{metric_log_msg} ({time_spent:.2f} secs)"
        )
        if study_best_results:
            logger.info(
                f"[{pair}] Optuna {namespace} {objective_type} objective hyperopt best params: {format_dict(study_best_results, style='dict')}"
            )
        if not self.optuna_validate_params(pair, namespace, study):
            logger.warning(
                f"[{pair}] Optuna {namespace} {objective_type} objective hyperopt best params found has invalid optimization target value(s)"
            )
        self.optuna_save_best_params(pair, namespace)
        return study

    def optuna_create_storage(self, pair: str) -> optuna.storages.BaseStorage:
        storage_dir = self.full_path
        storage_filename = f"optuna-{pair.split('/')[0]}"
        storage_backend = self._optuna_config.get("storage")
        if (
            storage_backend == QuickAdapterRegressorV3._OPTUNA_STORAGE_BACKENDS[0]
        ):  # "file"
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(
                    f"{storage_dir}/{storage_filename}.log"
                )
            )
        elif (
            storage_backend == QuickAdapterRegressorV3._OPTUNA_STORAGE_BACKENDS[1]
        ):  # "sqlite"
            storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{storage_dir}/{storage_filename}.sqlite",
                heartbeat_interval=60,
                failed_trial_callback=optuna.storages.RetryFailedTrialCallback(
                    max_retry=3
                ),
            )
        else:
            raise ValueError(
                f"Invalid optuna storage_backend value {storage_backend!r}: "
                f"supported values are {', '.join(QuickAdapterRegressorV3._OPTUNA_STORAGE_BACKENDS)}"
            )
        return storage

    def optuna_create_pruner(
        self, is_single_objective: bool
    ) -> optuna.pruners.BasePruner:
        if is_single_objective:
            return optuna.pruners.HyperbandPruner(
                min_resource=self._optuna_config.get(
                    "min_resource", QuickAdapterRegressorV3.OPTUNA_MIN_RESOURCE_DEFAULT
                )
            )
        else:
            return optuna.pruners.NopPruner()

    def optuna_create_sampler(
        self, sampler: Optional[OptunaSampler] = None
    ) -> optuna.samplers.BaseSampler:
        if sampler is None:
            sampler = self._optuna_config.get(
                "sampler",
            )
        if sampler == QuickAdapterRegressorV3._OPTUNA_SAMPLERS[0]:  # "tpe"
            return optuna.samplers.TPESampler(
                n_startup_trials=self._optuna_config.get(
                    "n_startup_trials",
                    QuickAdapterRegressorV3.OPTUNA_N_STARTUP_TRIALS_DEFAULT,
                ),
                multivariate=True,
                group=True,
                constant_liar=self._optuna_config.get(
                    "n_jobs", QuickAdapterRegressorV3.OPTUNA_N_JOBS_DEFAULT
                )
                > 1,
                seed=self._optuna_config.get(
                    "seed", QuickAdapterRegressorV3.OPTUNA_SEED_DEFAULT
                ),
            )
        elif sampler == QuickAdapterRegressorV3._OPTUNA_SAMPLERS[1]:  # "auto"
            return optunahub.load_module("samplers/auto_sampler").AutoSampler(
                seed=self._optuna_config.get(
                    "seed", QuickAdapterRegressorV3.OPTUNA_SEED_DEFAULT
                )
            )
        elif sampler == QuickAdapterRegressorV3._OPTUNA_SAMPLERS[2]:  # "nsgaii"
            return optuna.samplers.NSGAIISampler(
                seed=self._optuna_config.get(
                    "seed", QuickAdapterRegressorV3.OPTUNA_SEED_DEFAULT
                ),
            )
        elif sampler == QuickAdapterRegressorV3._OPTUNA_SAMPLERS[3]:  # "nsgaiii"
            return optuna.samplers.NSGAIIISampler(
                seed=self._optuna_config.get(
                    "seed", QuickAdapterRegressorV3.OPTUNA_SEED_DEFAULT
                ),
            )
        else:
            raise ValueError(
                f"Invalid optuna sampler value {sampler!r}: "
                f"supported values are {', '.join(QuickAdapterRegressorV3._OPTUNA_SAMPLERS)}"
            )

    @lru_cache(maxsize=8)
    def optuna_samplers_by_namespace(
        self, namespace: OptunaNamespace
    ) -> tuple[set[OptunaSampler], OptunaSampler]:
        if namespace == QuickAdapterRegressorV3._OPTUNA_NAMESPACES[0]:  # "hp"
            return (
                QuickAdapterRegressorV3._optuna_hpo_samplers_set(),
                self._optuna_config.get(
                    "sampler", QuickAdapterRegressorV3._OPTUNA_HPO_SAMPLERS[0]
                ),
            )
        elif namespace == QuickAdapterRegressorV3._OPTUNA_NAMESPACES[1]:  # "label"
            return (
                QuickAdapterRegressorV3._optuna_label_samplers_set(),
                self._optuna_config.get(
                    "label_sampler", QuickAdapterRegressorV3._OPTUNA_LABEL_SAMPLERS[0]
                ),
            )
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {', '.join(QuickAdapterRegressorV3._OPTUNA_NAMESPACES)}"
            )

    def optuna_create_study(
        self,
        pair: str,
        namespace: OptunaNamespace,
        direction: Optional[optuna.study.StudyDirection] = None,
        directions: Optional[list[optuna.study.StudyDirection]] = None,
    ) -> Optional[optuna.study.Study]:
        if direction is not None and directions is not None:
            raise ValueError(
                "Cannot specify both 'direction' and 'directions'. Use one or the other"
            )

        is_study_single_objective = direction is not None and directions is None
        if not is_study_single_objective:
            if directions is None or len(directions) < 2:
                raise ValueError(
                    "Multi-objective study must have at least 2 objectives specified"
                )

        identifier = self.freqai_info.get("identifier")
        study_name = f"{identifier}-{pair}-{namespace}"

        try:
            storage = self.optuna_create_storage(pair)
        except Exception as e:
            logger.error(
                f"[{pair}] Optuna {namespace} storage creation failed for study {study_name}: {e!r}",
                exc_info=True,
            )
            return None

        continuous = self._optuna_config.get("continuous")
        if continuous:
            QuickAdapterRegressorV3.optuna_delete_study(
                pair, namespace, study_name, storage
            )

        samplers, sampler = self.optuna_samplers_by_namespace(namespace)
        if sampler not in samplers:
            raise ValueError(
                f"Invalid optuna {namespace} sampler value {sampler!r}: "
                f"supported values are {', '.join(samplers)}"
            )

        try:
            return optuna.create_study(
                study_name=study_name,
                sampler=self.optuna_create_sampler(sampler),
                pruner=self.optuna_create_pruner(is_study_single_objective),
                direction=direction,
                directions=directions,
                storage=storage,
                load_if_exists=not continuous,
            )
        except Exception as e:
            logger.error(
                f"[{pair}] Optuna {namespace} study creation failed for study {study_name}: {e!r}",
                exc_info=True,
            )
            return None

    def optuna_validate_params(
        self, pair: str, namespace: OptunaNamespace, study: Optional[optuna.study.Study]
    ) -> bool:
        if not study:
            return False
        n_objectives = len(study.directions)
        if n_objectives > 1:
            best_values = self.get_optuna_values(pair, namespace)
            return (
                isinstance(best_values, list)
                and len(best_values) == n_objectives
                and all(
                    QuickAdapterRegressorV3.optuna_validate_value(value) is not None
                    for value in best_values
                )
            )
        else:
            best_value = self.get_optuna_value(pair, namespace)
            return QuickAdapterRegressorV3.optuna_validate_value(best_value) is not None

    def optuna_enqueue_previous_best_params(
        self, pair: str, namespace: OptunaNamespace, study: Optional[optuna.study.Study]
    ) -> None:
        if not study:
            return
        if not self.optuna_validate_params(pair, namespace, study):
            return
        best_params = self.get_optuna_params(pair, namespace)
        if not best_params:
            return

        try:
            study.enqueue_trial(best_params)
        except Exception as e:
            logger.warning(
                f"[{pair}] Optuna {namespace} failed to enqueue previous best params: {e!r}",
                exc_info=True,
            )

    def optuna_save_best_params(self, pair: str, namespace: OptunaNamespace) -> None:
        best_params_path = Path(
            self.full_path / f"optuna-{namespace}-best-params-{pair.split('/')[0]}.json"
        )
        try:
            with best_params_path.open("w", encoding="utf-8") as write_file:
                json.dump(self.get_optuna_params(pair, namespace), write_file, indent=4)
        except Exception as e:
            logger.error(
                f"[{pair}] Optuna {namespace} failed to save best params: {e!r}",
                exc_info=True,
            )
            raise

    def optuna_load_best_params(
        self, pair: str, namespace: OptunaNamespace
    ) -> Optional[dict[str, Any]]:
        best_params_path = Path(
            self.full_path / f"optuna-{namespace}-best-params-{pair.split('/')[0]}.json"
        )
        if best_params_path.is_file():
            with best_params_path.open("r", encoding="utf-8") as read_file:
                return json.load(read_file)
        return None

    @staticmethod
    def optuna_delete_study(
        pair: str,
        namespace: OptunaNamespace,
        study_name: str,
        storage: optuna.storages.BaseStorage,
    ) -> None:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except Exception as e:
            logger.warning(
                f"[{pair}] Optuna {namespace} study {study_name} deletion failed: {e!r}",
                exc_info=True,
            )

    @staticmethod
    def optuna_load_study(
        study_name: str, storage: optuna.storages.BaseStorage
    ) -> Optional[optuna.study.Study]:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except Exception:
            study = None
        return study

    @staticmethod
    def optuna_study_has_best_trial(study: Optional[optuna.study.Study]) -> bool:
        if not study:
            return False
        try:
            _ = study.best_trial
            return True
        except (ValueError, KeyError):
            return False

    @staticmethod
    def optuna_study_has_best_trials(study: Optional[optuna.study.Study]) -> bool:
        if not study:
            return False
        try:
            _ = study.best_trials
            return True
        except (ValueError, KeyError):
            return False


def hp_objective(
    trial: optuna.trial.Trial,
    regressor: Regressor,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_weights: NDArray[np.floating],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    test_weights: NDArray[np.floating],
    test_size: float,
    model_training_best_parameters: dict[str, Any],
    model_training_parameters: dict[str, Any],
    space_reduction: bool,
    space_fraction: float,
    model_path: Optional[Path] = None,
) -> float:
    study_model_parameters = get_optuna_study_model_parameters(
        trial,
        regressor,
        model_training_best_parameters,
        model_training_parameters,
        space_reduction,
        space_fraction,
    )
    model_training_parameters = {**model_training_parameters, **study_model_parameters}

    eval_set, eval_weights = eval_set_and_weights(
        X_test, y_test, test_weights, test_size
    )

    model = fit_regressor(
        regressor=regressor,
        X=X,
        y=y,
        train_weights=train_weights,
        eval_set=eval_set,
        eval_weights=eval_weights,
        model_training_parameters=model_training_parameters,
        model_path=model_path,
        trial=trial,
    )
    y_pred = model.predict(X_test)

    return sklearn.metrics.root_mean_squared_error(
        y_test, y_pred, sample_weight=test_weights
    )


def label_objective(
    trial: optuna.trial.Trial,
    df: pd.DataFrame,
    fit_live_predictions_candles: int,
    candles_step: int,
    min_label_period_candles: int = QuickAdapterRegressorV3.MIN_LABEL_PERIOD_CANDLES_DEFAULT,
    max_label_period_candles: int = QuickAdapterRegressorV3.MAX_LABEL_PERIOD_CANDLES_DEFAULT,
    min_label_natr_multiplier: float = QuickAdapterRegressorV3.MIN_LABEL_NATR_MULTIPLIER_DEFAULT,
    max_label_natr_multiplier: float = QuickAdapterRegressorV3.MAX_LABEL_NATR_MULTIPLIER_DEFAULT,
) -> tuple[int, float, float, float, float, float, float]:
    min_label_period_candles, max_label_period_candles, candles_step = (
        get_min_max_label_period_candles(
            fit_live_predictions_candles,
            candles_step,
            min_label_period_candles=min_label_period_candles,
            max_label_period_candles=max_label_period_candles,
            min_label_period_candles_fallback=min_label_period_candles,
            max_label_period_candles_fallback=max_label_period_candles,
        )
    )

    label_period_candles = trial.suggest_int(
        "label_period_candles",
        min_label_period_candles,
        max_label_period_candles,
        step=candles_step,
    )
    label_natr_multiplier = trial.suggest_float(
        "label_natr_multiplier",
        min_label_natr_multiplier,
        max_label_natr_multiplier,
        step=0.05,
    )

    df = df.iloc[
        -(
            max(2, int(fit_live_predictions_candles / label_period_candles))
            * label_period_candles
        ) :
    ]

    if df.empty:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    (
        pivots_indices,
        _,
        _,
        pivots_amplitudes,
        pivots_amplitude_threshold_ratios,
        pivots_volume_rates,
        pivots_speeds,
        pivots_efficiency_ratios,
        pivots_volume_weighted_efficiency_ratios,
    ) = zigzag(
        df,
        natr_period=label_period_candles,
        natr_multiplier=label_natr_multiplier,
    )

    median_amplitude = np.nanmedian(np.asarray(pivots_amplitudes, dtype=float))
    if not np.isfinite(median_amplitude):
        median_amplitude = 0.0

    median_amplitude_threshold_ratio = np.nanmedian(
        np.asarray(pivots_amplitude_threshold_ratios, dtype=float)
    )
    if not np.isfinite(median_amplitude_threshold_ratio):
        median_amplitude_threshold_ratio = 0.0

    median_volume_rate = np.nanmedian(np.asarray(pivots_volume_rates, dtype=float))
    if not np.isfinite(median_volume_rate):
        median_volume_rate = 0.0

    median_speed = np.nanmedian(np.asarray(pivots_speeds, dtype=float))
    if not np.isfinite(median_speed):
        median_speed = 0.0

    median_efficiency_ratio = np.nanmedian(
        np.asarray(pivots_efficiency_ratios, dtype=float)
    )
    if not np.isfinite(median_efficiency_ratio):
        median_efficiency_ratio = 0.0

    median_volume_weighted_efficiency_ratio = np.nanmedian(
        np.asarray(pivots_volume_weighted_efficiency_ratios, dtype=float)
    )
    if not np.isfinite(median_volume_weighted_efficiency_ratio):
        median_volume_weighted_efficiency_ratio = 0.0

    return (
        len(pivots_indices),
        median_amplitude,
        median_amplitude_threshold_ratio,
        median_volume_rate,
        median_speed,
        median_efficiency_ratio,
        median_volume_weighted_efficiency_ratio,
    )
