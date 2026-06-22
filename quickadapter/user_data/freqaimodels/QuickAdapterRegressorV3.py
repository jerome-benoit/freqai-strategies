import copy
import logging
import math
import random
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Callable,
    ClassVar,
    Final,
    Literal,
    Optional,
    Union,
    assert_never,
    cast,
)

import numpy as np
import optuna
import optunahub
import pandas as pd
import scipy as sp
import skimage
import sklearn
from datasieve.pipeline import Pipeline
from datasieve.transforms import SKLearnWrapper
from freqtrade.exceptions import DependencyException
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from numpy.typing import NDArray
from optuna.study.study import ObjectiveFuncType
from sklearn.model_selection import TimeSeriesSplit, train_test_split
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
    LABEL_WEIGHT_SUPPORT_POLICIES,
    PREDICTION_METHODS,
    SKIMAGE_THRESHOLD_METHODS,
    THRESHOLD_METHODS,
    CustomThresholdMethod,
    ExtremaSelectionMethod,
    LabelTransformer,
    LabelWeightSupportPolicy,
    SkimageThresholdMethod,
    ThresholdMethod,
    get_label_column_config,
)

from Utils import (
    DEFAULT_FIT_LIVE_PREDICTIONS_CANDLES,
    DEFAULTS_LABEL_PREDICTION,
    LABEL_COLUMNS,
    LabelWeightSupportError,
    REGRESSORS,
    Regressor,
    WEIGHT_STRATEGIES,
    _OPTUNA_NAMESPACES,
    _OPTUNA_LABEL_SELECTION_SCHEMA_VERSION,
    OptunaNamespace,
    compose_sample_weights,
    ensure_datetime_series,
    make_test_set_and_weights,
    fit_regressor,
    finite_sample,
    format_dict,
    format_number,
    get_causal_mode,
    get_label_defaults,
    get_label_horizon_candles,
    get_label_pipeline_config,
    get_label_prediction_config,
    get_label_weighting_config,
    get_min_max_label_period_candles,
    get_optuna_study_model_parameters,
    label_known_at_column_name,
    label_weight_column_name,
    migrate_config,
    optuna_load_best_params,
    optuna_save_best_params,
    sanitize_and_renormalize,
    safe_distribution_fit,
    summarize_label_weight_support,
    soft_extremum,
    zigzag,
)

OptunaSampler = Literal["tpe", "auto", "nsgaii", "nsgaiii"]
ScalerType = Literal["minmax", "maxabs", "standard", "robust"]
DensityAggregation = Literal["power_mean", "quantile", "min", "max"]
DistanceMethod = Literal["compromise_programming", "topsis"]
ClusterMethod = Literal["kmeans", "kmeans2", "kmedoids"]
DensityMethod = Literal["knn", "medoid"]
SelectionMethod = Union[DistanceMethod, ClusterMethod, DensityMethod]
ValidationMode = Literal["warn", "raise", "none"]
SplitFn = Callable[
    [pd.DataFrame, pd.DataFrame, "SampleWeightInputs", pd.DataFrame], dict[str, Any]
]
warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

_KNOWN_AT_NONE_LOGGED: set[tuple[str, str]] = set()


def _log_known_at_none_once(pair: str, context: str) -> None:
    key = (pair, context)
    if key in _KNOWN_AT_NONE_LOGGED:
        return
    _KNOWN_AT_NONE_LOGGED.add(key)
    logger.info(
        f"[{pair}] {context}: no <label>_known_at_index column present; "
        "causal guards use position-based purge only (label-aware filtering disabled)"
    )


@dataclass(frozen=True, slots=True)
class SampleWeightInputs:
    base: NDArray[np.floating]
    label: NDArray[np.floating] | None
    label_weighting_config: dict[str, Any]

    _REQUIRED_LABEL_WEIGHTING_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "support_policy",
            "min_pivot_equivalent_count",
            "min_positive_label_weight_fraction",
            "min_effective_sample_size",
        }
    )

    def __post_init__(self) -> None:
        if self.base.ndim != 1:
            raise ValueError(
                f"SampleWeightInputs.base: must be 1-D (ndim={self.base.ndim})"
            )
        if self.label is not None and self.base.shape != self.label.shape:
            raise ValueError(
                f"SampleWeightInputs.label: shape {self.label.shape} "
                f"!= base shape {self.base.shape}"
            )
        missing = (
            self._REQUIRED_LABEL_WEIGHTING_KEYS - self.label_weighting_config.keys()
        )
        if missing:
            raise KeyError(
                f"SampleWeightInputs.label_weighting_config: missing required keys "
                f"{sorted(missing)}"
            )
        policy = self.label_weighting_config["support_policy"]
        if policy not in LABEL_WEIGHT_SUPPORT_POLICIES:
            raise ValueError(
                f"SampleWeightInputs.label_weighting_config.support_policy: "
                f"{policy!r} not in {LABEL_WEIGHT_SUPPORT_POLICIES}"
            )


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

    version = "3.12.0"

    _TEST_SIZE: Final[float] = 0.1
    # Substituted whenever the Weibull DI cutoff (``weibull_min.ppf``) is
    # non-finite (cold start or degenerate fit). Preserves the prior
    # pre-warm-up heuristic for the outlier-quantile cutoff scale.
    _DI_CUTOFF_DEFAULT: Final[float] = 2.0
    _SKLEARN_TRAIN_TEST_SPLIT_KEYS: Final[frozenset[str]] = frozenset(
        {"test_size", "train_size", "random_state", "shuffle", "stratify"}
    )

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

    _PROBABILITY_DISTANCE_METRICS: Final[tuple[str, ...]] = (
        "jensenshannon",
        "hellinger",
        "shellinger",
    )

    # Absolute tolerance (rtol=0) for constant-column detection in
    # `_non_constant_objective_indices`; valid on the [0,1]-normalized
    # output of `_normalize_objective_values`.
    _NON_CONSTANT_OBJECTIVE_ATOL: Final[float] = 1e-8

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
        return set(_OPTUNA_NAMESPACES)

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
    def _probability_distance_metrics_set() -> set[str]:
        return set(QuickAdapterRegressorV3._PROBABILITY_DISTANCE_METRICS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _label_selection_distance_metrics_set() -> set[str]:
        return (
            QuickAdapterRegressorV3._distance_metrics_set()
            - QuickAdapterRegressorV3._probability_distance_metrics_set()
        )

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
    @lru_cache(maxsize=None)
    def _aggregate_distance_metrics_set() -> set[str]:
        """Aggregate metrics: distance metrics computed by reduction over
        objective coordinates rather than SciPy/sklearn pairwise routines.

        Computed as the complement of SciPy and probability metrics over
        ``_DISTANCE_METRICS``: ``harmonic_mean``, ``geometric_mean``,
        ``arithmetic_mean``, ``quadratic_mean``, ``cubic_mean``,
        ``power_mean``, ``weighted_sum``. Accepted by
        ``compromise_programming``/``topsis`` via
        ``_calculate_trial_distance_to_ideal``; rejected by
        cluster/density categories that route to
        ``pairwise_distances``/``KMeans``/``KMedoids``/``NearestNeighbors``.
        """
        return (
            QuickAdapterRegressorV3._distance_metrics_set()
            - QuickAdapterRegressorV3._scipy_metrics_set()
            - QuickAdapterRegressorV3._probability_distance_metrics_set()
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def _cluster_density_distance_metrics_set() -> set[str]:
        """SciPy-compatible non-probability metrics.

        Accepted by cluster/density categories that route to
        ``pairwise_distances``/``KMeans``/``KMedoids``/``NearestNeighbors``;
        rejected by the aggregate set and by probability metrics.
        """
        return (
            QuickAdapterRegressorV3._scipy_metrics_set()
            - QuickAdapterRegressorV3._probability_distance_metrics_set()
        )

    @staticmethod
    def _coerce_int(value: Any, name: str, *, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(
                f"Invalid data_split_parameters.{name} value {value!r}: "
                f"must be int >= {minimum}"
            )
        return value

    @staticmethod
    def _coerce_optional_int(value: Any, name: str, *, minimum: int) -> Optional[int]:
        if value is None:
            return None
        return QuickAdapterRegressorV3._coerce_int(value, name, minimum=minimum)

    @staticmethod
    def _validate_index_alignment(
        filtered_dataframe: pd.DataFrame,
        unfiltered_df: pd.DataFrame,
    ) -> None:
        if not unfiltered_df.index.is_unique:
            raise ValueError(
                "unfiltered_df.index must be unique for causal split guards"
            )
        if not filtered_dataframe.index.isin(unfiltered_df.index).all():
            raise ValueError(
                "filtered_dataframe.index must be a subset of unfiltered_df.index"
            )

    @staticmethod
    def _row_positions(
        filtered_dataframe: pd.DataFrame,
        unfiltered_df: pd.DataFrame,
    ) -> pd.Series:
        QuickAdapterRegressorV3._validate_index_alignment(
            filtered_dataframe, unfiltered_df
        )
        positions = pd.Series(
            np.arange(len(unfiltered_df), dtype=np.int64), index=unfiltered_df.index
        )
        return positions.loc[filtered_dataframe.index]

    @staticmethod
    def _known_at_index(
        filtered_dataframe: pd.DataFrame,
        unfiltered_df: pd.DataFrame,
    ) -> pd.Series | None:
        """Per-row leak boundary across all registered labels.

        Returns the row-wise ``max`` of every present
        ``<label>_known_at_index`` column. A label whose column is missing
        or contains any NaN is skipped (silently — labels can opt in by
        emitting the column). Returns ``None`` only when no label exposes
        a usable column, in which case the caller falls back to the
        position-based purge.
        """
        QuickAdapterRegressorV3._validate_index_alignment(
            filtered_dataframe, unfiltered_df
        )
        series_list: list[pd.Series] = []
        for label_col in LABEL_COLUMNS:
            known_at_col = label_known_at_column_name(label_col)
            if known_at_col not in unfiltered_df.columns:
                continue
            known_at = unfiltered_df.loc[filtered_dataframe.index, known_at_col]
            if known_at.isna().any():
                continue
            series_list.append(pd.to_numeric(known_at, errors="raise"))
        if not series_list:
            return None
        if len(series_list) == 1:
            return series_list[0]
        return pd.concat(series_list, axis=1).max(axis=1).astype(np.int64)

    @staticmethod
    def _filter_train_by_mask(
        train_features: pd.DataFrame,
        train_labels: pd.DataFrame,
        train_weights: NDArray[np.floating],
        keep_mask: NDArray[np.bool_],
        context: str,
        train_label_weights: NDArray[np.floating] | None = None,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        NDArray[np.floating],
        NDArray[np.floating] | None,
    ]:
        removed = int((~keep_mask).sum())
        if removed:
            logger.info(f"{context}: removed {removed} causal-unsafe train rows")
        if not keep_mask.any():
            raise ValueError(f"{context}: causal guard removed all train rows")
        return (
            train_features.loc[keep_mask],
            train_labels.loc[keep_mask],
            train_weights[keep_mask],
            None if train_label_weights is None else train_label_weights[keep_mask],
        )

    @staticmethod
    def _shuffle_split_rows(
        features: pd.DataFrame,
        labels: pd.DataFrame,
        base_weights: NDArray[np.floating],
        label_weights: NDArray[np.floating] | None,
        seed: int,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        NDArray[np.floating],
        NDArray[np.floating] | None,
    ]:
        shuffled_features = features.sample(frac=1, random_state=seed)
        order = features.index.get_indexer(shuffled_features.index)
        if (order < 0).any():
            raise ValueError(
                f"_shuffle_split_rows: unable to align shuffled feature rows "
                f"to sample weights (missing={int((order < 0).sum())} rows)"
            )
        shuffled_labels = labels.loc[shuffled_features.index]
        shuffled_label_weights = None if label_weights is None else label_weights[order]
        return (
            shuffled_features,
            shuffled_labels,
            base_weights[order],
            shuffled_label_weights,
        )

    @staticmethod
    def _compose_eval_weights(
        base_weights: NDArray[np.floating],
        label_weights: NDArray[np.floating] | None,
        *,
        context: str,
    ) -> NDArray[np.floating]:
        """Compose eval (test/val) sample weights, bypassing ``support_policy``.

        Support thresholds are training-fit invariants and routine test/val
        splits trip them by construction. With ``on_collapse="fallback"``,
        the label-derived ``drop_mask`` propagates on collapse-on-survivors
        so framework-side early-stopping that consumes eval sample weights
        sees the row-survival pattern of the training-time label weighting.
        The all-dropped path raises ``LabelWeightSupportError`` and falls
        back to base weights only; no survival pattern exists to propagate.
        Shape-parity ``ValueError`` is left uncaught: a hard contract
        failure, not a support condition.
        """
        try:
            return compose_sample_weights(
                base_weights,
                label_weights,
                logger=logger,
                on_collapse="fallback",
            )
        except LabelWeightSupportError as exc:
            logger.warning(
                "%s: label-weighted eval weights failed (%s); using base weights",
                context,
                exc,
            )
            return compose_sample_weights(base_weights, None, logger=logger)

    @staticmethod
    def _apply_support_policy(
        base_weights: NDArray[np.floating],
        *,
        context: str,
        policy: LabelWeightSupportPolicy,
        reasons: list[str],
    ) -> NDArray[np.floating]:
        reason_text = "; ".join(reasons)
        match policy:
            case "raise":
                raise ValueError(
                    f"{context}: label weighting support failed ({reason_text}); "
                    "support_policy='raise'"
                )
            case "fallback":
                logger.warning(
                    "%s: label weighting support failed (%s); "
                    "falling back to sanitized base weights (support_policy='fallback')",
                    context,
                    reason_text,
                )
                return compose_sample_weights(base_weights, None, logger=logger)
            case _:
                assert_never(policy)

    @staticmethod
    def _compose_train_weights_with_support(
        base_weights: NDArray[np.floating],
        label_weights: NDArray[np.floating] | None,
        label_weighting_config: dict[str, Any],
        *,
        context: str,
    ) -> NDArray[np.floating]:
        policy = cast(
            LabelWeightSupportPolicy, label_weighting_config["support_policy"]
        )
        if label_weights is None:
            # Non-"none" label-weighting strategy with no available label
            # weights (zigzag produced zero pivots): the support policy
            # governs the contract -- ``raise`` raises, ``fallback``
            # warns. A direct return to base weights would bypass the
            # policy silently.
            strategy = label_weighting_config.get("strategy", WEIGHT_STRATEGIES[0])
            if strategy != WEIGHT_STRATEGIES[0]:  # "none"
                return QuickAdapterRegressorV3._apply_support_policy(
                    base_weights,
                    context=context,
                    policy=policy,
                    reasons=[
                        f"label_weighting.strategy={strategy!r} configured but "
                        f"no label weights available (no pivots detected)"
                    ],
                )
            return compose_sample_weights(base_weights, None, logger=logger)

        try:
            composed = compose_sample_weights(
                base_weights, label_weights, logger=logger
            )
        except LabelWeightSupportError as exc:
            return QuickAdapterRegressorV3._apply_support_policy(
                base_weights,
                context=context,
                policy=policy,
                reasons=[str(exc)],
            )

        summary = summarize_label_weight_support(label_weights, composed)
        reasons: list[str] = []
        min_pivot_equivalent_count = label_weighting_config[
            "min_pivot_equivalent_count"
        ]
        min_positive_label_weight_fraction = label_weighting_config[
            "min_positive_label_weight_fraction"
        ]
        min_effective_sample_size = label_weighting_config["min_effective_sample_size"]
        if summary.pivot_equivalent_count < min_pivot_equivalent_count:
            reasons.append(
                f"pivot_equivalent_count={summary.pivot_equivalent_count} "
                f"< min_pivot_equivalent_count={min_pivot_equivalent_count}"
            )
        if summary.positive_label_weight_fraction < min_positive_label_weight_fraction:
            reasons.append(
                f"positive_label_weight_fraction={summary.positive_label_weight_fraction:.6g} "
                f"< min_positive_label_weight_fraction={min_positive_label_weight_fraction:.6g} "
                f"({summary.positive_label_weight_count}/{summary.total_rows} rows)"
            )
        if summary.effective_sample_size < min_effective_sample_size:
            reasons.append(
                f"effective_sample_size={summary.effective_sample_size:.6g} "
                f"< min_effective_sample_size={min_effective_sample_size:.6g}"
            )
        if reasons:
            return QuickAdapterRegressorV3._apply_support_policy(
                base_weights, context=context, policy=policy, reasons=reasons
            )
        logger.debug(
            "%s: label weighting support passed "
            "(pivot_equivalent_count=%d, positive_label_weight_fraction=%.6g, "
            "effective_sample_size=%.6g)",
            context,
            summary.pivot_equivalent_count,
            summary.positive_label_weight_fraction,
            summary.effective_sample_size,
        )
        return composed

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
    def _validate_label_selection_metric(
        metric: str,
        *,
        ctx: str,
        default: str,
        aggregate_allowed: bool,
        mode: ValidationMode = "warn",
    ) -> str:
        if aggregate_allowed:
            valid_metrics = (
                QuickAdapterRegressorV3._label_selection_distance_metrics_set()
            )
        else:
            # Cluster/density paths route the metric to SciPy/sklearn APIs
            # (pairwise_distances, KMeans, KMedoids, NearestNeighbors) which
            # reject `_aggregate_distance_metrics_set()`; restrict the
            # valid set to SciPy-compatible non-probability metrics.
            valid_metrics = (
                QuickAdapterRegressorV3._cluster_density_distance_metrics_set()
            )
        valid_options = tuple(
            candidate
            for candidate in QuickAdapterRegressorV3._DISTANCE_METRICS
            if candidate in valid_metrics
        )
        resolved_metric = QuickAdapterRegressorV3._validate_enum_value(
            metric,
            valid_metrics,
            valid_options,
            ctx=ctx,
            mode=mode,
            default=default,
        )
        return cast(str, resolved_metric)

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
            distance_metric = QuickAdapterRegressorV3._validate_label_selection_metric(
                distance_metric,
                ctx="label_distance_metric",
                mode="warn",
                default=QuickAdapterRegressorV3.LABEL_DISTANCE_METRIC_DEFAULT,
                aggregate_allowed=True,
            )
            config["distance_metric"] = distance_metric
        elif category == "cluster":
            distance_metric = self.ft_params.get(
                "label_cluster_metric",
                QuickAdapterRegressorV3.LABEL_CLUSTER_METRIC_DEFAULT,
            )
            distance_metric = QuickAdapterRegressorV3._validate_label_selection_metric(
                distance_metric,
                ctx="label_cluster_metric",
                mode="warn",
                default=QuickAdapterRegressorV3.LABEL_CLUSTER_METRIC_DEFAULT,
                aggregate_allowed=False,
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
            density_metric_default = (
                QuickAdapterRegressorV3._get_label_density_metric_default(
                    density_method
                )
            )
            distance_metric = self.ft_params.get(
                "label_density_metric",
                density_metric_default,
            )
            distance_metric = QuickAdapterRegressorV3._validate_label_selection_metric(
                distance_metric,
                ctx="label_density_metric",
                mode="warn",
                default=density_metric_default,
                aggregate_allowed=False,
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

    def _optuna_label_selection_metadata(self) -> dict[str, Any]:
        """Build the label-namespace selection metadata for ``set_user_attr``.

        Must return JSON-serializable values only (str/int/float/bool/None
        and nested dicts/lists thereof); the dict-equality check at
        ``optuna_create_study`` (idempotent ``set_user_attr`` write) breaks
        on numpy arrays.
        """
        label_method = self.ft_params.get(
            "label_method", QuickAdapterRegressorV3.LABEL_METHOD_DEFAULT
        )
        label_weights = self.ft_params.get("label_weights")
        label_p_order = self.ft_params.get("label_p_order")
        if label_weights is not None and not all(
            math.isfinite(float(w)) for w in label_weights
        ):
            raise ValueError(
                f"label_weights contains non-finite values: {label_weights!r}"
            )
        if label_p_order is not None and not math.isfinite(float(label_p_order)):
            raise ValueError(f"label_p_order is non-finite: {label_p_order!r}")
        return {
            "schema_version": _OPTUNA_LABEL_SELECTION_SCHEMA_VERSION,
            "method_config": self._resolve_label_method_config(label_method),
            "label_weights": (
                [float(w) for w in label_weights] if label_weights is not None else None
            ),
            "label_p_order": (
                float(label_p_order) if label_p_order is not None else None
            ),
        }

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

        label_frequency_candles = self.ft_params.get("label_frequency_candles")

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
    def label_weighting(self) -> dict[str, Any]:
        label_weighting_raw = self.freqai_info.get("label_weighting")
        if not isinstance(label_weighting_raw, dict):
            label_weighting_raw = {}
        return get_label_weighting_config(label_weighting_raw, logger)

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
    def _causal_mode(self) -> bool:
        return get_causal_mode(self.ft_params, logger)

    def _label_horizon_candles(self, pair: str | None = None) -> int:
        if pair is None:
            return get_label_horizon_candles(self.ft_params, logger)
        label_params = self.get_optuna_params(pair, _OPTUNA_NAMESPACES.label)
        return get_label_horizon_candles({**self.ft_params, **label_params}, logger)

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
                self.optuna_load_best_params(pair, _OPTUNA_NAMESPACES.hp) or {}
            )
            self._optuna_label_params[pair] = self.optuna_load_best_params(
                pair, _OPTUNA_NAMESPACES.label
            ) or {
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

        logger.info("Optuna Hyperopt:")
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
            logger.info(f"Label [{label_col}]:")

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
            if col_prediction["method"] == PREDICTION_METHODS[0]:  # "none"
                logger.warning(
                    f"  Prediction method is 'none' for label [{label_col}]: "
                    f"minima_threshold/maxima_threshold will not be computed and "
                    f"entry signals based on them will never trigger."
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
        logger.info("Feature Parameters:")
        logger.info(f"  scaler: {scaler}")
        logger.info(
            f"  range: ({format_number(feature_range[0])}, {format_number(feature_range[1])})"
        )

        logger.info("=" * 60)

    def get_optuna_params(
        self, pair: str, namespace: OptunaNamespace
    ) -> dict[str, Any]:
        if namespace == _OPTUNA_NAMESPACES.hp:
            params = self._optuna_hp_params.get(pair, {})
        elif namespace == _OPTUNA_NAMESPACES.label:
            params = self._optuna_label_params.get(pair, {})
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {', '.join(_OPTUNA_NAMESPACES)}"
            )
        return params

    def set_optuna_params(
        self, pair: str, namespace: OptunaNamespace, params: dict[str, Any]
    ) -> None:
        if namespace == _OPTUNA_NAMESPACES.hp:
            self._optuna_hp_params[pair] = params
        elif namespace == _OPTUNA_NAMESPACES.label:
            self._optuna_label_params[pair] = params
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {', '.join(_OPTUNA_NAMESPACES)}"
            )

    def get_optuna_value(self, pair: str, namespace: OptunaNamespace) -> float:
        if namespace == _OPTUNA_NAMESPACES.hp:
            value = self._optuna_hp_value.get(pair, np.nan)
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {_OPTUNA_NAMESPACES.hp!r}"
            )
        return value

    def set_optuna_value(
        self, pair: str, namespace: OptunaNamespace, value: float
    ) -> None:
        if namespace == _OPTUNA_NAMESPACES.hp:
            self._optuna_hp_value[pair] = value
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {_OPTUNA_NAMESPACES.hp!r}"
            )

    def get_optuna_values(
        self, pair: str, namespace: OptunaNamespace
    ) -> list[float | int]:
        if namespace == _OPTUNA_NAMESPACES.label:
            values = self._optuna_label_values.get(
                pair, [np.nan] * QuickAdapterRegressorV3._OPTUNA_LABEL_N_OBJECTIVES
            )
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {_OPTUNA_NAMESPACES.label}"
            )
        return values

    def set_optuna_values(
        self, pair: str, namespace: OptunaNamespace, values: list[float | int]
    ) -> None:
        if namespace == _OPTUNA_NAMESPACES.label:
            self._optuna_label_values[pair] = values
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {_OPTUNA_NAMESPACES.label}"
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
        """Train a model with per-row sample weights.

        Dispatches on ``data_split_parameters.method``:
        - ``train_test_split``: random sklearn split.
        - ``timeseries_split``: chronological final-fold split.
        Both paths build per-row weights via ``_build_sample_weight_inputs``
        before splitting. After split + causal-guard filtering, train weights
        compose through ``_compose_train_weights_with_support`` (gated by
        ``support_policy``) and eval weights through ``_compose_eval_weights``
        (bypasses ``support_policy``). ``_train_common`` then feeds them to
        ``model.fit(sample_weight=...)``.
        """
        method = self.data_split_parameters.get(
            "method", QuickAdapterRegressorV3.DATA_SPLIT_METHOD_DEFAULT
        )

        match method:
            case "train_test_split":
                split_builder = self._make_train_test_split_datasets
            case "timeseries_split":
                split_builder = self._make_timeseries_split_datasets
            case _:
                raise ValueError(
                    f"Invalid data_split_parameters.method value {method!r}: "
                    f"supported values are "
                    f"{', '.join(QuickAdapterRegressorV3._DATA_SPLIT_METHODS)}"
                )

        def split_fn(
            features: pd.DataFrame,
            labels: pd.DataFrame,
            weights: SampleWeightInputs,
            unfiltered: pd.DataFrame,
        ) -> dict[str, Any]:
            return split_builder(features, labels, weights, dk, unfiltered)

        logger.info(f"Using data split method: {method}")
        return self._train_common(unfiltered_df, pair, dk, split_fn, **kwargs)

    def _make_train_test_split_datasets(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        weights: SampleWeightInputs,
        dk: FreqaiDataKitchen,
        unfiltered_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Train/test split via sklearn's ``train_test_split``.

        Routes ``data_split_parameters`` to sklearn through a whitelist of
        sklearn-recognized keys; project-custom keys (``method``,
        ``n_splits``, ``gap``, ``max_train_size``) are filtered out.
        ``shuffle`` and ``test_size`` default to ``False`` and ``_TEST_SIZE``
        respectively when absent from ``data_split_parameters``. Honors
        ``feature_parameters.shuffle_after_split`` (deterministic when
        ``random_state`` is set) and ``feature_parameters.reverse_train_test_order``.
        Per-row sample weights are sliced positionally and propagate to both
        train and test sets.
        """
        feat_dict = self.ft_params
        dsp = dict(self.data_split_parameters)
        dsp.setdefault("shuffle", False)
        dsp.setdefault("test_size", QuickAdapterRegressorV3._TEST_SIZE)
        causal_mode = self._causal_mode
        if causal_mode and dsp.get("shuffle", False):
            raise ValueError(
                "feature_parameters.causal_mode=True is incompatible with "
                "data_split_parameters.shuffle=True"
            )
        if causal_mode and feat_dict.get("shuffle_after_split", False):
            raise ValueError(
                "feature_parameters.causal_mode=True is incompatible with "
                "feature_parameters.shuffle_after_split=True"
            )
        if causal_mode and feat_dict.get("reverse_train_test_order", False):
            raise ValueError(
                "feature_parameters.causal_mode=True is incompatible with "
                "feature_parameters.reverse_train_test_order=True"
            )
        sklearn_kwargs = {
            k: v
            for k, v in dsp.items()
            if k in QuickAdapterRegressorV3._SKLEARN_TRAIN_TEST_SPLIT_KEYS
        }
        test_size = dsp["test_size"]
        if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
            raise ValueError(
                f"Invalid data_split_parameters.test_size value {test_size!r}: "
                f"must be int or float"
            )

        if test_size != 0:
            if weights.label is None:
                (
                    train_features,
                    test_features,
                    train_labels,
                    test_labels,
                    train_base_weights,
                    test_base_weights,
                ) = train_test_split(features, labels, weights.base, **sklearn_kwargs)
                train_label_weights = None
                test_label_weights = None
            else:
                (
                    train_features,
                    test_features,
                    train_labels,
                    test_labels,
                    train_base_weights,
                    test_base_weights,
                    train_label_weights,
                    test_label_weights,
                ) = train_test_split(
                    features, labels, weights.base, weights.label, **sklearn_kwargs
                )
            if causal_mode:
                row_positions = QuickAdapterRegressorV3._row_positions(
                    features, unfiltered_df
                )
                first_test_position = int(row_positions.loc[test_features.index].min())
                label_horizon_candles = self._label_horizon_candles(dk.pair)
                train_positions = row_positions.loc[train_features.index]
                keep_mask = (
                    train_positions.to_numpy(dtype=np.int64)
                    < first_test_position - label_horizon_candles
                )
                known_at_index = QuickAdapterRegressorV3._known_at_index(
                    features, unfiltered_df
                )
                if known_at_index is not None:
                    known_at_train = known_at_index.loc[train_features.index]
                    keep_mask &= (
                        known_at_train.to_numpy(dtype=np.int64) < first_test_position
                    )
                else:
                    _log_known_at_none_once(dk.pair, "train_test_split causal guard")
                (
                    train_features,
                    train_labels,
                    train_base_weights,
                    train_label_weights,
                ) = QuickAdapterRegressorV3._filter_train_by_mask(
                    train_features,
                    train_labels,
                    train_base_weights,
                    keep_mask,
                    f"[{dk.pair}] train_test_split causal guard",
                    train_label_weights=train_label_weights,
                )
        else:
            train_features = features
            train_labels = labels
            train_base_weights = weights.base
            train_label_weights = weights.label
            test_features = features.iloc[:0]
            test_labels = labels.iloc[:0]
            test_base_weights = weights.base[:0]
            test_label_weights = None if weights.label is None else weights.label[:0]

        if feat_dict.get("shuffle_after_split", False):
            parent_seed = sklearn_kwargs.get("random_state")
            shuffle_rng = (
                random.Random(parent_seed)
                if parent_seed is not None
                else random.Random()
            )
            train_features, train_labels, train_base_weights, train_label_weights = (
                QuickAdapterRegressorV3._shuffle_split_rows(
                    train_features,
                    train_labels,
                    train_base_weights,
                    train_label_weights,
                    shuffle_rng.randint(0, 2**31 - 1),
                )
            )
            if test_size != 0:
                test_features, test_labels, test_base_weights, test_label_weights = (
                    QuickAdapterRegressorV3._shuffle_split_rows(
                        test_features,
                        test_labels,
                        test_base_weights,
                        test_label_weights,
                        shuffle_rng.randint(0, 2**31 - 1),
                    )
                )

        train_weights = QuickAdapterRegressorV3._compose_train_weights_with_support(
            train_base_weights,
            train_label_weights,
            weights.label_weighting_config,
            context=f"[{dk.pair}] train_test_split:train",
        )
        if test_size != 0:
            test_weights = QuickAdapterRegressorV3._compose_eval_weights(
                test_base_weights,
                test_label_weights,
                context=f"[{dk.pair}] train_test_split:test",
            )
        else:
            test_weights = test_base_weights

        if feat_dict.get("reverse_train_test_order", False):
            return dk.build_data_dictionary(
                test_features,
                train_features,
                test_labels,
                train_labels,
                test_weights,
                train_weights,
            )
        return dk.build_data_dictionary(
            train_features,
            test_features,
            train_labels,
            test_labels,
            train_weights,
            test_weights,
        )

    def _build_sample_weight_inputs(
        self,
        features_filtered: pd.DataFrame,
        unfiltered_df: pd.DataFrame,
        dk: FreqaiDataKitchen,
    ) -> SampleWeightInputs:
        """Build per-row base and label weight vectors aligned to features_filtered.index.

        Multiplies freqtrade's per-row base weights (recency-decayed via
        ``dk.set_weights_higher_recent`` when ``feature_parameters.weight_factor > 0``,
        else ones) with the label importance weight column produced by
        ``compute_label_weights`` and stored on ``unfiltered_df`` under
        ``label_weight_column_name(LABEL_COLUMNS[0])``. Alignment runs before
        any shuffle/split on ``features_filtered.index`` (a subset of
        ``unfiltered_df.index``) to avoid post-hoc reindex against shuffled
        data. The weight column is absent when ``label_weighting.strategy``
        is ``'none'`` (no per-label importance applied); in that case the
        final split stage composes base-only sample weights.
        """
        if not unfiltered_df.index.is_unique:
            raise ValueError(
                "unfiltered_df.index must be unique for label-based weight "
                "alignment; received non-unique index"
            )
        if not features_filtered.index.isin(unfiltered_df.index).all():
            raise ValueError(
                "features_filtered.index must be a subset of "
                "unfiltered_df.index (filter_features should preserve original "
                "row labels)"
            )
        if LABEL_COLUMNS[0] not in dk.label_list:
            raise ValueError(
                f"LABEL_COLUMNS[0]={LABEL_COLUMNS[0]!r} is not in "
                f"dk.label_list={dk.label_list!r}: project label constant "
                f"diverged from freqtrade's runtime label list"
            )
        n_rows = len(features_filtered)
        feat_dict = self.ft_params
        weight_factor = feat_dict.get("weight_factor", 0)
        if (
            not isinstance(weight_factor, bool)
            and isinstance(weight_factor, (int, float))
            and weight_factor > 0
        ):
            base_weights = np.asarray(dk.set_weights_higher_recent(n_rows), dtype=float)
        else:
            base_weights = np.ones(n_rows, dtype=float)

        label_weighting = self.label_weighting
        label_weighting_config = get_label_column_config(
            LABEL_COLUMNS[0], label_weighting["default"], label_weighting["columns"]
        )
        weight_col = label_weight_column_name(LABEL_COLUMNS[0])
        if weight_col in unfiltered_df.columns:
            label_weights = unfiltered_df.loc[
                features_filtered.index, weight_col
            ].to_numpy(dtype=float)
            logger.debug(f"label weight column active: {weight_col!r}")
        else:
            label_weights = None
            logger.debug(
                f"label weight column absent ({weight_col!r}); using base weights only"
            )
        return SampleWeightInputs(
            base=base_weights,
            label=label_weights,
            label_weighting_config=label_weighting_config,
        )

    def _train_common(
        self,
        unfiltered_df: pd.DataFrame,
        pair: str,
        dk: FreqaiDataKitchen,
        split_fn: SplitFn,
        **kwargs,
    ) -> Any:
        logger.info(
            f"-------------------- Starting training {pair} --------------------"
        )
        start_time = time.time()
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )
        weights = self._build_sample_weight_inputs(features_filtered, unfiltered_df, dk)
        dates = ensure_datetime_series(unfiltered_df["date"])
        start_date = dates.iloc[0].strftime("%Y-%m-%d")
        end_date = dates.iloc[-1].strftime("%Y-%m-%d")
        logger.info(
            f"-------------------- Training on data from {start_date} to "
            f"{end_date} --------------------"
        )
        dd = split_fn(features_filtered, labels_filtered, weights, unfiltered_df)
        if not self.freqai_info.get("fit_live_predictions_candles", 0) or not self.live:
            dk.fit_labels()
        dd = self._apply_pipelines(dd, dk, pair)
        if len(dd["train_features"]) != len(dd["train_weights"]):
            raise RuntimeError(
                f"Pipeline broke shape invariant: "
                f"len(train_features)={len(dd['train_features'])} != "
                f"len(train_weights)={len(dd['train_weights'])}"
            )
        logger.info(f"Training model on {len(dd['train_features'].columns)} features")
        logger.info(f"Training model on {len(dd['train_features'])} data points")
        model = self.fit(dd, dk, **kwargs)
        end_time = time.time()
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
        """Apply feature and label pipelines; renormalize weights post-transform."""
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)
        dk.label_pipeline = self.define_label_pipeline(threads=dk.thread_count)

        (dd["train_features"], dd["train_labels"], dd["train_weights"]) = (
            dk.feature_pipeline.fit_transform(
                dd["train_features"], dd["train_labels"], dd["train_weights"]
            )
        )
        dd["train_weights"] = sanitize_and_renormalize(
            dd["train_weights"],
            logger=logger,
            context="post_feature_pipeline:train",
        )
        dd["train_labels"], _, _ = dk.label_pipeline.fit_transform(dd["train_labels"])

        if (
            self.data_split_parameters.get(
                "test_size", QuickAdapterRegressorV3._TEST_SIZE
            )
            != 0
        ):
            if dd["test_labels"].shape[0] == 0:
                method = self.data_split_parameters.get(
                    "method", QuickAdapterRegressorV3.DATA_SPLIT_METHOD_DEFAULT
                )
                if (
                    method == QuickAdapterRegressorV3._DATA_SPLIT_METHODS[1]
                ):  # timeseries_split
                    n_splits = self.data_split_parameters.get(
                        "n_splits", QuickAdapterRegressorV3.TIMESERIES_N_SPLITS_DEFAULT
                    )
                    gap = self.data_split_parameters.get(
                        "gap", QuickAdapterRegressorV3.TIMESERIES_GAP_DEFAULT
                    )
                    max_train_size = self.data_split_parameters.get("max_train_size")
                    test_size = self.data_split_parameters.get("test_size")
                    error_msg = (
                        f"[{pair}] test set is empty after filtering. "
                        f"Possible causes: n_splits too high, gap too large, "
                        f"max_train_size too restrictive, or insufficient data. "
                        f"Current parameters: n_splits={n_splits}, gap={gap}, "
                        f"max_train_size={max_train_size}, test_size={test_size}. "
                        f"Try reducing n_splits/gap or increasing data period."
                    )
                else:
                    test_size = self.data_split_parameters.get(
                        "test_size", QuickAdapterRegressorV3._TEST_SIZE
                    )
                    error_msg = (
                        f"[{pair}] test set is empty after filtering. "
                        f"Possible causes: overly strict SVM thresholds or insufficient data. "
                        f"Current test_size={test_size}. "
                        f"Try reducing test_size or relaxing SVM conditions."
                    )
                raise DependencyException(error_msg)
            else:
                (dd["test_features"], dd["test_labels"], dd["test_weights"]) = (
                    dk.feature_pipeline.transform(
                        dd["test_features"], dd["test_labels"], dd["test_weights"]
                    )
                )
                dd["test_weights"] = sanitize_and_renormalize(
                    dd["test_weights"],
                    logger=logger,
                    context="post_feature_pipeline:test",
                )
                dd["test_labels"], _, _ = dk.label_pipeline.transform(dd["test_labels"])

        return dd

    def _make_timeseries_split_datasets(
        self,
        filtered_dataframe: pd.DataFrame,
        labels: pd.DataFrame,
        weights: SampleWeightInputs,
        dk: FreqaiDataKitchen,
        unfiltered_df: pd.DataFrame,
    ) -> dict:
        """Chronological train/test split using sklearn's TimeSeriesSplit final fold.

        ``n_splits`` controls train/test proportions (higher = larger train).
        ``gap`` excludes samples between train and test; when 0, auto-derived
        from ``label_period_candles``. ``max_train_size`` enables sliding
        window mode. ``weights`` is sliced positionally via ``train_idx`` /
        ``test_idx``.
        """
        feat_dict = self.ft_params
        causal_mode = self._causal_mode
        if feat_dict.get("shuffle_after_split", False):
            raise ValueError(
                "feature_parameters.shuffle_after_split=True is incompatible "
                "with data_split_parameters.method='timeseries_split': "
                "chronological split must preserve temporal ordering"
            )
        if causal_mode and self.data_split_parameters.get("shuffle", False):
            raise ValueError(
                "feature_parameters.causal_mode=True is incompatible with "
                "data_split_parameters.shuffle=True"
            )
        if causal_mode and feat_dict.get("reverse_train_test_order", False):
            raise ValueError(
                "feature_parameters.causal_mode=True is incompatible with "
                "feature_parameters.reverse_train_test_order=True"
            )
        n_splits = QuickAdapterRegressorV3._coerce_int(
            self.data_split_parameters.get(
                "n_splits", QuickAdapterRegressorV3.TIMESERIES_N_SPLITS_DEFAULT
            ),
            "n_splits",
            minimum=2,
        )
        raw_gap = self.data_split_parameters.get("gap", None)
        gap = QuickAdapterRegressorV3._coerce_int(
            raw_gap
            if raw_gap is not None
            else QuickAdapterRegressorV3.TIMESERIES_GAP_DEFAULT,
            "gap",
            minimum=0,
        )
        max_train_size = QuickAdapterRegressorV3._coerce_optional_int(
            self.data_split_parameters.get(
                "max_train_size",
                QuickAdapterRegressorV3.TIMESERIES_MAX_TRAIN_SIZE_DEFAULT,
            ),
            "max_train_size",
            minimum=1,
        )

        test_size = self.data_split_parameters.get("test_size", None)
        if test_size is not None:
            if (
                not isinstance(test_size, bool)
                and isinstance(test_size, float)
                and 0 < test_size < 1
            ):
                test_size = int(len(filtered_dataframe) * test_size)
            elif not (
                not isinstance(test_size, bool)
                and isinstance(test_size, int)
                and test_size >= 1
            ):
                raise ValueError(
                    f"Invalid data_split_parameters.test_size value {test_size!r}: "
                    f"must be float in (0, 1) as fraction, int >= 1 as count, or None"
                )
            if test_size < 1:
                raise ValueError(
                    f"Computed test_size ({test_size}) is too small. "
                    f"Increase test_size or provide more data."
                )

        if causal_mode:
            label_horizon_candles = self._label_horizon_candles(dk.pair)
            if raw_gap is None or gap == 0:
                gap = label_horizon_candles
                logger.info(
                    f"[{dk.pair}] TimeSeriesSplit gap auto-set from label_horizon_candles: {gap}"
                )
            elif gap < label_horizon_candles:
                raise ValueError(
                    f"data_split_parameters.gap={gap!r} is smaller than "
                    f"label_horizon_candles={label_horizon_candles!r} while "
                    "feature_parameters.causal_mode=True"
                )
        elif gap == 0:
            gap = self.get_optuna_params(
                dk.pair,
                _OPTUNA_NAMESPACES.label,
            ).get("label_period_candles")
            logger.info(
                f"[{dk.pair}] TimeSeriesSplit gap auto-set from label_period_candles: {gap}"
            )

        tscv = TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap,
            max_train_size=max_train_size,
            test_size=test_size,
        )
        folds = list(tscv.split(filtered_dataframe))
        if not folds:
            raise ValueError(
                f"TimeSeriesSplit yielded no folds for {len(filtered_dataframe)} "
                f"samples (n_splits={n_splits}, gap={gap}, "
                f"max_train_size={max_train_size}, test_size={test_size})"
            )
        train_idx, test_idx = folds[-1]

        train_features = filtered_dataframe.iloc[train_idx]
        test_features = filtered_dataframe.iloc[test_idx]
        train_labels = labels.iloc[train_idx]
        test_labels = labels.iloc[test_idx]
        train_base_weights = weights.base[train_idx]
        test_base_weights = weights.base[test_idx]
        train_label_weights = (
            None if weights.label is None else weights.label[train_idx]
        )
        test_label_weights = None if weights.label is None else weights.label[test_idx]
        test_weights = QuickAdapterRegressorV3._compose_eval_weights(
            test_base_weights,
            test_label_weights,
            context=f"[{dk.pair}] timeseries_split:test",
        )

        if causal_mode:
            row_positions = QuickAdapterRegressorV3._row_positions(
                filtered_dataframe, unfiltered_df
            )
            first_test_position = int(row_positions.iloc[test_idx].min())
            known_at_index = QuickAdapterRegressorV3._known_at_index(
                filtered_dataframe, unfiltered_df
            )
            if known_at_index is not None:
                known_at_train = known_at_index.iloc[train_idx]
                keep_mask = (
                    known_at_train.to_numpy(dtype=np.int64) < first_test_position
                )
                (
                    train_features,
                    train_labels,
                    train_base_weights,
                    train_label_weights,
                ) = QuickAdapterRegressorV3._filter_train_by_mask(
                    train_features,
                    train_labels,
                    train_base_weights,
                    keep_mask,
                    f"[{dk.pair}] timeseries_split causal guard",
                    train_label_weights=train_label_weights,
                )
            else:
                _log_known_at_none_once(dk.pair, "timeseries_split causal guard")

        train_weights = QuickAdapterRegressorV3._compose_train_weights_with_support(
            train_base_weights,
            train_label_weights,
            weights.label_weighting_config,
            context=f"[{dk.pair}] timeseries_split:train",
        )

        if feat_dict.get("reverse_train_test_order", False):
            return dk.build_data_dictionary(
                test_features,
                train_features,
                test_labels,
                train_labels,
                test_weights,
                train_weights,
            )
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
                namespace=_OPTUNA_NAMESPACES.hp,
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
                    self.get_optuna_params(dk.pair, _OPTUNA_NAMESPACES.hp),
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

            optuna_hp_params = self.get_optuna_params(dk.pair, _OPTUNA_NAMESPACES.hp)
            if optuna_hp_params:
                model_training_parameters = {
                    **model_training_parameters,
                    **optuna_hp_params,
                }

        eval_set, eval_weights = make_test_set_and_weights(
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
        if namespace not in {_OPTUNA_NAMESPACES.label}:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {_OPTUNA_NAMESPACES.label}"
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
                namespace=_OPTUNA_NAMESPACES.label,
                callback=lambda: self.optuna_optimize(
                    pair=pair,
                    namespace=_OPTUNA_NAMESPACES.label,
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
        if di_values is not None:
            dk.data["DI_value_mean"] = di_values.mean()
            dk.data["DI_value_std"] = di_values.std(ddof=1)
        else:
            dk.data["DI_value_mean"] = 0.0
            dk.data["DI_value_std"] = 0.0

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
                    cutoff = QuickAdapterRegressorV3._DI_CUTOFF_DEFAULT
                else:
                    min_pred, max_pred = self.min_max_pred(
                        label_col,
                        col_prediction_config,
                        pred_df,
                        fit_live_predictions_candles,
                        self.get_optuna_params(pair, _OPTUNA_NAMESPACES.label).get(
                            "label_period_candles"
                        ),
                    )
                    di_sample = finite_sample(
                        []
                        if di_values is None
                        else pd.to_numeric(di_values, errors="coerce"),
                        positive_only=True,
                    )
                    f = safe_distribution_fit(
                        di_sample,
                        sp.stats.weibull_min.fit,
                        # Intentionally non-ppf-able; the ``weibull_min.ppf``
                        # downstream returns NaN on degenerate scale and the
                        # ``np.isfinite(cutoff)`` guard substitutes
                        # ``_DI_CUTOFF_DEFAULT``.
                        fallback=(0.0, 0.0, 0.0),
                        context=f"di_values_weibull_fit:{pair}",
                        logger=logger,
                        min_count=2,
                        require_variance=True,
                        floc=0,
                    )
                    outlier_quantile = col_prediction_config.get(
                        "outlier_quantile",
                        DEFAULTS_LABEL_PREDICTION["outlier_quantile"],
                    )
                    cutoff = sp.stats.weibull_min.ppf(outlier_quantile, *f)
                    if not np.isfinite(cutoff):
                        logger.warning(
                            "[%s] DI_values Weibull cutoff is invalid "
                            "(params=%r, quantile=%r); using fallback %r",
                            pair,
                            f,
                            outlier_quantile,
                            QuickAdapterRegressorV3._DI_CUTOFF_DEFAULT,
                        )
                        cutoff = QuickAdapterRegressorV3._DI_CUTOFF_DEFAULT
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
                sample = finite_sample(pred_label)
                if sample.finite_count == 0:
                    fallback = (0.0, 0.0)
                else:
                    sample_mean = float(np.mean(sample.values))
                    sample_std = float(np.std(sample.values, ddof=0))
                    fallback = (
                        sample_mean if np.isfinite(sample_mean) else 0.0,
                        sample_std if np.isfinite(sample_std) else 0.0,
                    )
                f = safe_distribution_fit(
                    sample,
                    sp.stats.norm.fit,
                    fallback=fallback,
                    context=f"label_norm_fit:{pair}:{label_col}",
                    logger=logger,
                    min_count=2,
                    require_variance=True,
                )
            dk.data["labels_mean"][label_col], dk.data["labels_std"][label_col] = (
                f[0],
                f[1],
            )

        dk.data["extra_returns_per_train"]["label_period_candles"] = (
            self.get_optuna_params(pair, _OPTUNA_NAMESPACES.label).get(
                "label_period_candles"
            )
        )
        dk.data["extra_returns_per_train"]["label_natr_multiplier"] = (
            self.get_optuna_params(
                pair,
                _OPTUNA_NAMESPACES.label,
            ).get("label_natr_multiplier")
        )

        hp_rmse = QuickAdapterRegressorV3.optuna_validate_value(
            self.get_optuna_value(pair, _OPTUNA_NAMESPACES.hp)
        )
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
            eps = 10 * np.finfo(float).eps

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

                if finite_range_val < 10 * np.finfo(float).eps:
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
    def _non_constant_objective_indices(
        normalized_matrix: NDArray[np.floating],
    ) -> NDArray[np.intp]:
        if normalized_matrix.ndim != 2:
            raise ValueError(
                f"Invalid normalized_matrix (shape={normalized_matrix.shape}, "
                f"ndim={normalized_matrix.ndim}): must be 2-dimensional"
            )
        if not np.all(np.isfinite(normalized_matrix)):
            raise ValueError(
                "Invalid normalized_matrix: must contain only finite values (no NaN or inf)"
            )
        non_constant_mask = np.array(
            [
                # rtol=0: pure absolute tolerance on [0,1]-normalized columns;
                # any finite rtol would leak column magnitude into the threshold.
                not np.allclose(
                    normalized_matrix[:, column_index],
                    normalized_matrix[0, column_index],
                    rtol=0.0,
                    atol=QuickAdapterRegressorV3._NON_CONSTANT_OBJECTIVE_ATOL,
                )
                for column_index in range(normalized_matrix.shape[1])
            ],
            dtype=bool,
        )
        return np.flatnonzero(non_constant_mask)

    @staticmethod
    def _select_lowest_number_trial(
        trials: list[optuna.trial.FrozenTrial],
    ) -> optuna.trial.FrozenTrial:
        return min(trials, key=lambda trial: trial.number)

    @staticmethod
    def _select_best_trial_by_distance(
        trials: list[optuna.trial.FrozenTrial],
        distances: NDArray[np.floating],
    ) -> optuna.trial.FrozenTrial:
        if distances.size != len(trials):
            raise ValueError(
                f"Invalid trial distances length {distances.size}: "
                f"must match trials length {len(trials)}"
            )

        candidates = [
            (float(distance), trial.number, trial)
            for trial, distance in zip(trials, distances)
            if np.isfinite(distance)
        ]
        if not candidates:
            logger.warning(
                "_select_best_trial_by_distance: all %d candidate distances "
                "are non-finite; falling back to lowest trial number",
                len(trials),
            )
            return QuickAdapterRegressorV3._select_lowest_number_trial(trials)
        return min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[2]

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
        objective_indices: Optional[NDArray[np.intp]] = None,
        original_n_objectives: Optional[int] = None,
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
        if (
            label_weights is not None
            and original_n_objectives is not None
            and objective_indices is not None
            and original_n_objectives != n_objectives
        ):
            try:
                label_weights_array = np.asarray(label_weights, dtype=float)
            except (ValueError, TypeError):
                label_weights_array = None
            if label_weights_array is not None and label_weights_array.ndim == 1:
                if label_weights_array.size != original_n_objectives:
                    raise ValueError(
                        f"Invalid label_weights size {label_weights_array.size}: "
                        f"must match original objective count "
                        f"{original_n_objectives}"
                    )
                sliced_weights = label_weights_array[objective_indices]
                if np.all(sliced_weights == 0.0):
                    # All user-positive weights project onto dropped
                    # (constant) objectives; uniform fallback keeps
                    # selection deterministic and avoids
                    # ``_validate_label_weights`` raising on sum-zero.
                    # Negative or non-finite slices flow through to the
                    # validator.
                    logger.warning(
                        "label_weights sliced to non-constant objectives "
                        "is all-zero (indices=%s, original=%s); "
                        "falling back to uniform weights",
                        objective_indices.tolist(),
                        label_weights_array.tolist(),
                    )
                    label_weights = None
                else:
                    label_weights = sliced_weights
                    logger.debug(
                        "label_weights sliced to non-constant objectives "
                        "(indices=%s, original_size=%d, sliced_size=%d)",
                        objective_indices.tolist(),
                        label_weights_array.size,
                        sliced_weights.size,
                    )
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
        if namespace not in {_OPTUNA_NAMESPACES.label}:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {_OPTUNA_NAMESPACES.label}"
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
        original_n_objectives = normalized_matrix.shape[1]
        non_constant_objective_indices = (
            QuickAdapterRegressorV3._non_constant_objective_indices(normalized_matrix)
        )
        if non_constant_objective_indices.size == 0:
            return QuickAdapterRegressorV3._select_lowest_number_trial(best_trials)
        normalized_matrix = normalized_matrix[:, non_constant_objective_indices]

        trial_distances = self._calculate_distances(
            normalized_matrix,
            selection_method=label_method,
            objective_indices=non_constant_objective_indices,
            original_n_objectives=original_n_objectives,
        )

        return QuickAdapterRegressorV3._select_best_trial_by_distance(
            best_trials, trial_distances
        )

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
        if namespace == _OPTUNA_NAMESPACES.hp:
            return (
                QuickAdapterRegressorV3._optuna_hpo_samplers_set(),
                self._optuna_config.get(
                    "sampler", QuickAdapterRegressorV3._OPTUNA_HPO_SAMPLERS[0]
                ),
            )
        elif namespace == _OPTUNA_NAMESPACES.label:
            return (
                QuickAdapterRegressorV3._optuna_label_samplers_set(),
                self._optuna_config.get(
                    "label_sampler", QuickAdapterRegressorV3._OPTUNA_LABEL_SAMPLERS[0]
                ),
            )
        else:
            raise ValueError(
                f"Invalid namespace value {namespace!r}: "
                f"supported values are {', '.join(_OPTUNA_NAMESPACES)}"
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
        elif namespace == _OPTUNA_NAMESPACES.label:
            existing_study = QuickAdapterRegressorV3.optuna_load_study(
                study_name, storage
            )
            if existing_study is not None:
                existing_selection_metadata = existing_study.user_attrs.get(
                    "selection_metadata"
                )
                existing_schema_version = (
                    existing_selection_metadata.get("schema_version")
                    if isinstance(existing_selection_metadata, dict)
                    else None
                )
                target_version = _OPTUNA_LABEL_SELECTION_SCHEMA_VERSION
                if (
                    isinstance(existing_schema_version, bool)
                    or not isinstance(existing_schema_version, (int, np.integer))
                    or existing_schema_version != target_version
                ):
                    version_repr = (
                        "none"
                        if existing_schema_version is None
                        else f"v{existing_schema_version}"
                    )
                    logger.warning(
                        f"[{pair}] Optuna {namespace} study {study_name}: "
                        f"selection schema {version_repr} incompatible "
                        f"with v{target_version}; resetting study"
                    )
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
            study = optuna.create_study(
                study_name=study_name,
                sampler=self.optuna_create_sampler(sampler),
                pruner=self.optuna_create_pruner(is_study_single_objective),
                direction=direction,
                directions=directions,
                storage=storage,
                load_if_exists=not continuous,
            )
            if namespace == _OPTUNA_NAMESPACES.label:
                new_selection_metadata = self._optuna_label_selection_metadata()
                existing_selection_metadata = study.user_attrs.get("selection_metadata")
                if existing_selection_metadata != new_selection_metadata:
                    if isinstance(existing_selection_metadata, dict):
                        logger.warning(
                            f"[{pair}] Optuna {namespace} study {study_name}: "
                            f"selection_metadata change detected "
                            f"(stored: {existing_selection_metadata!r}, "
                            f"current: {new_selection_metadata!r})"
                        )
                    study.set_user_attr("selection_metadata", new_selection_metadata)
            return study
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
        optuna_save_best_params(
            self.full_path,
            pair,
            namespace,
            self.get_optuna_params(pair, namespace),
            logger,
            selection_metadata=self._optuna_label_selection_metadata()
            if namespace == _OPTUNA_NAMESPACES.label
            else None,
        )

    def optuna_load_best_params(
        self, pair: str, namespace: OptunaNamespace
    ) -> Optional[dict[str, Any]]:
        expected = (
            self._optuna_label_selection_metadata()
            if namespace == _OPTUNA_NAMESPACES.label
            else None
        )
        return optuna_load_best_params(
            self.full_path,
            pair,
            namespace,
            logger,
            expected_selection_metadata=expected,
        )

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

    eval_set, eval_weights = make_test_set_and_weights(
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
        logger=logger,
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
