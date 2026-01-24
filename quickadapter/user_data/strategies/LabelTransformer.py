import logging
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np
import scipy as sp
from datasieve.transforms.base_transform import (
    ArrayOrNone,
    BaseTransform,
    ListOrNone,
)
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from Utils import get_column_config

logger = logging.getLogger(__name__)

WeightStrategy = Literal[
    "none",
    "amplitude",
    "amplitude_threshold_ratio",
    "volume_rate",
    "speed",
    "efficiency_ratio",
    "volume_weighted_efficiency_ratio",
    "combined",
]

CombinedMetric = Literal[
    "amplitude",
    "amplitude_threshold_ratio",
    "volume_rate",
    "speed",
    "efficiency_ratio",
    "volume_weighted_efficiency_ratio",
]
COMBINED_METRICS: Final[tuple[CombinedMetric, ...]] = (
    "amplitude",
    "amplitude_threshold_ratio",
    "volume_rate",
    "speed",
    "efficiency_ratio",
    "volume_weighted_efficiency_ratio",
)

CombinedAggregation = Literal[
    "arithmetic_mean",
    "geometric_mean",
    "harmonic_mean",
    "quadratic_mean",
    "weighted_median",
    "softmax",
]
COMBINED_AGGREGATIONS: Final[tuple[CombinedAggregation, ...]] = (
    "arithmetic_mean",
    "geometric_mean",
    "harmonic_mean",
    "quadratic_mean",
    "weighted_median",
    "softmax",
)

WEIGHT_STRATEGIES: Final[tuple[WeightStrategy, ...]] = (
    "none",
    "amplitude",
    "amplitude_threshold_ratio",
    "volume_rate",
    "speed",
    "efficiency_ratio",
    "volume_weighted_efficiency_ratio",
    "combined",
)

StandardizationType = Literal["none", "zscore", "robust", "mmad", "power_yj"]
STANDARDIZATION_TYPES: Final[tuple[StandardizationType, ...]] = (
    "none",  # 0 - w
    "zscore",  # 1 - (w - μ) / σ
    "robust",  # 2 - (w - median) / IQR
    "mmad",  # 3 - (w - median) / (MAD · k)
    "power_yj",  # 4 - YJ(w) (standardized)
)

NormalizationType = Literal["maxabs", "minmax", "sigmoid", "none"]
NORMALIZATION_TYPES: Final[tuple[NormalizationType, ...]] = (
    "maxabs",  # 0 - w / max(|w|)
    "minmax",  # 1 - low + (w - min) / (max - min) · (high - low)
    "sigmoid",  # 2 - 2·σ(scale · w) - 1
    "none",  # 3 - w
)

# Default configuration for label weighting (sample weight computation in strategy)
DEFAULTS_LABEL_WEIGHTING: Final[dict[str, Any]] = {
    "strategy": WEIGHT_STRATEGIES[0],  # "none"
    "metric_coefficients": {},
    "aggregation": COMBINED_AGGREGATIONS[0],  # "arithmetic_mean"
    "softmax_temperature": 1.0,
}

# Default configuration for label pipeline transformation (standardization/normalization)
DEFAULTS_LABEL_PIPELINE: Final[dict[str, Any]] = {
    "standardization": STANDARDIZATION_TYPES[0],  # "none"
    "robust_quantiles": (0.25, 0.75),
    "mmad_scaling_factor": 1.4826,
    "normalization": NORMALIZATION_TYPES[0],  # "maxabs"
    "minmax_range": (-1.0, 1.0),
    "sigmoid_scale": 1.0,
    "gamma": 1.0,
}

# Legacy: Combined defaults (deprecated, for backward compatibility)
DEFAULTS_LABEL_TRANSFORMER: Final[dict[str, Any]] = {
    **DEFAULTS_LABEL_WEIGHTING,
    **DEFAULTS_LABEL_PIPELINE,
}

DEFAULTS_PIPELINE_TRANSFORM = DEFAULTS_LABEL_PIPELINE

SmoothingMethod = Literal[
    "none", "gaussian", "kaiser", "triang", "smm", "sma", "savgol", "gaussian_filter1d"
]
SMOOTHING_METHODS: Final[tuple[SmoothingMethod, ...]] = (
    "none",
    "gaussian",
    "kaiser",
    "triang",
    "smm",
    "sma",
    "savgol",
    "gaussian_filter1d",
)

SmoothingMode = Literal["mirror", "constant", "nearest", "wrap", "interp"]
SMOOTHING_MODES: Final[tuple[SmoothingMode, ...]] = (
    "mirror",
    "constant",
    "nearest",
    "wrap",
    "interp",
)

DEFAULTS_LABEL_SMOOTHING: Final[dict[str, Any]] = {
    "method": SMOOTHING_METHODS[1],  # "gaussian"
    "window_candles": 5,
    "beta": 8.0,
    "polyorder": 3,
    "mode": SMOOTHING_MODES[0],  # "mirror"
    "sigma": 1.0,
}

PredictionMethod = Literal["none", "thresholding"]
PREDICTION_METHODS: Final[tuple[PredictionMethod, ...]] = (
    "none",
    "thresholding",
)

ExtremaSelectionMethod = Literal["rank_extrema", "rank_peaks", "partition"]
EXTREMA_SELECTION_METHODS: Final[tuple[ExtremaSelectionMethod, ...]] = (
    "rank_extrema",
    "rank_peaks",
    "partition",
)

ThresholdMethod = Literal[
    "mean",
    "isodata",
    "li",
    "minimum",
    "otsu",
    "triangle",
    "yen",
    "median",
    "soft_extremum",
]
SKIMAGE_THRESHOLD_METHODS: Final[tuple[str, ...]] = (
    "mean",
    "isodata",
    "li",
    "minimum",
    "otsu",
    "triangle",
    "yen",
)
CUSTOM_THRESHOLD_METHODS: Final[tuple[str, ...]] = (
    "median",
    "soft_extremum",
)
THRESHOLD_METHODS: Final[tuple[ThresholdMethod, ...]] = (
    *SKIMAGE_THRESHOLD_METHODS,
    *CUSTOM_THRESHOLD_METHODS,
)

DEFAULTS_LABEL_PREDICTION: Final[dict[str, Any]] = {
    "method": PREDICTION_METHODS[0],  # "none"
    "selection_method": EXTREMA_SELECTION_METHODS[0],  # "rank_extrema"
    "threshold_method": SKIMAGE_THRESHOLD_METHODS[0],  # "mean"
    "outlier_quantile": 0.999,
    "soft_alpha": 12.0,
    "keep_fraction": 1.0,
}


@dataclass
class _ColumnState:
    """Fitted state for a single column's transformation pipeline."""

    config: dict[str, Any]
    # Phase 1: Standardization
    standard_scaler: StandardScaler | None = None
    robust_scaler: RobustScaler | None = None
    power_transformer: PowerTransformer | None = None
    median: float = 0.0
    mad: float = 1.0
    # Phase 2: Normalization
    minmax_scaler: MinMaxScaler | None = None
    maxabs_scaler: MaxAbsScaler | None = None


@dataclass
class _LabelTransformerConfig:
    """
    Configuration for LabelTransformer with per-label settings.

    Supports two configuration formats:

    1. Flat format (legacy, applies same config to all labels):
       {
           "standardization": "robust",
           "normalization": "minmax",
           "gamma": 1.5
       }

    2. Per-label format (new, allows different configs per label):
       {
           "default": {
               "standardization": "none",
               "normalization": "maxabs"
           },
           "columns": {
               "&s-extrema": {
                   "standardization": "robust",
                   "normalization": "minmax",
                   "gamma": 1.5
               },
               "&s-threshold_*": {
                   "standardization": "zscore"
               }
           }
       }

    The "columns" mapping supports glob patterns (*, ?) for matching label names.
    """

    default: dict[str, Any] = field(
        default_factory=lambda: DEFAULTS_PIPELINE_TRANSFORM.copy()
    )
    columns: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "_LabelTransformerConfig":
        """
        Parse configuration dict into LabelTransformerConfig.

        Detects format automatically:
        - If "default" or "columns" keys exist -> per-label format
        - Otherwise -> flat format (used as default for all labels)
        """
        if "default" in config or "columns" in config:
            # Per-label format
            default = {**DEFAULTS_PIPELINE_TRANSFORM, **config.get("default", {})}
            columns = config.get("columns", {})
            return cls(default=default, columns=columns)
        else:
            # Flat format - use entire config as default
            # Filter out keys that are not pipeline transform keys (e.g., strategy, aggregation)
            pipeline_keys = set(DEFAULTS_PIPELINE_TRANSFORM.keys())
            filtered_config = {k: v for k, v in config.items() if k in pipeline_keys}
            default = {**DEFAULTS_PIPELINE_TRANSFORM, **filtered_config}
            return cls(default=default, columns={})

    def get_column_config(self, column_name: str) -> dict[str, Any]:
        """Get the configuration for a specific column."""
        return get_column_config(column_name, self.default, self.columns)


class LabelTransformer(BaseTransform):
    """
    Datasieve-compatible transformer for FreqAI label pipelines.

    Applies standardization, normalization, and gamma correction to labels.
    Supports per-column configuration via the "columns" mapping with glob patterns.

    Pipeline stages:
    1. Standardization: zscore, robust, mmad, power_yj, or none
    2. Normalization: maxabs, minmax, sigmoid, or none
    3. Post-processing: gamma power transform

    Example configuration:
    ```python
    LabelTransformer(label_transformer={
        "default": {
            "standardization": "none",
            "normalization": "maxabs",
            "gamma": 1.0
        },
        "columns": {
            "&s-extrema": {
                "standardization": "robust",
                "normalization": "minmax",
                "minmax_range": (-1.0, 1.0),
                "gamma": 1.5
            }
        }
    })
    ```
    """

    _STANDARDIZATION_SCALERS: dict[str, str] = {
        "zscore": "standard_scaler",
        "robust": "robust_scaler",
        "power_yj": "power_transformer",
    }
    _NORMALIZATION_SCALERS: dict[str, str] = {
        "maxabs": "maxabs_scaler",
        "minmax": "minmax_scaler",
    }

    def __init__(self, *, label_transformer: dict[str, Any]) -> None:
        super().__init__(name="LabelTransformer")
        self._config = _LabelTransformerConfig.from_dict(label_transformer)
        self._column_states: dict[str, _ColumnState] = {}
        self._fitted_columns: list[str] = []
        self._fitted = False

    # -------------------------------------------------------------------------
    # Static transform methods (operate on values with config)
    # -------------------------------------------------------------------------

    @staticmethod
    def _apply_scaler(
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        scaler: Any,
        inverse: bool = False,
    ) -> NDArray[np.floating]:
        if values[mask].size == 0:
            return values
        out = values.copy()
        method = scaler.inverse_transform if inverse else scaler.transform
        out[mask] = method(values[mask].reshape(-1, 1)).flatten()
        return out

    @staticmethod
    def _apply_mmad(
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        median: float,
        mad: float,
        k: float,
        inverse: bool = False,
    ) -> NDArray[np.floating]:
        if values[mask].size == 0:
            return values
        out = values.copy()
        if inverse:
            out[mask] = values[mask] * (mad * k) + median
        else:
            out[mask] = (values[mask] - median) / (mad * k)
        return out

    @staticmethod
    def _apply_sigmoid(
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        scale: float,
        inverse: bool = False,
    ) -> NDArray[np.floating]:
        if values[mask].size == 0:
            return values
        if not np.isfinite(scale) or np.isclose(scale, 0.0):
            return values
        out = values.copy()
        if inverse:
            out[mask] = sp.special.logit((values[mask] + 1.0) / 2.0) / scale
        else:
            out[mask] = 2.0 * sp.special.expit(scale * values[mask]) - 1.0
        return out

    @staticmethod
    def _apply_gamma(
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        gamma: float,
        inverse: bool = False,
    ) -> NDArray[np.floating]:
        if np.isclose(gamma, 1.0) or not np.isfinite(gamma) or gamma <= 0:
            return values
        out = values.copy()
        exp = 1.0 / gamma if inverse else gamma
        out[mask] = np.sign(values[mask]) * np.power(np.abs(values[mask]), exp)
        return out

    # -------------------------------------------------------------------------
    # Per-column transform methods
    # -------------------------------------------------------------------------

    def _standardize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        state: _ColumnState,
        inverse: bool = False,
    ) -> NDArray[np.floating]:
        method = state.config["standardization"]
        if method == "none":
            return values
        if method == "mmad":
            return self._apply_mmad(
                values,
                mask,
                state.median,
                state.mad,
                state.config["mmad_scaling_factor"],
                inverse=inverse,
            )

        scaler_attr = self._STANDARDIZATION_SCALERS.get(method)
        if scaler_attr is None:
            raise ValueError(
                f"Invalid standardization value {method!r}: "
                f"supported values are {', '.join(STANDARDIZATION_TYPES)}"
            )
        scaler = getattr(state, scaler_attr, None)
        if scaler is None:
            raise RuntimeError(f"{scaler_attr} not fitted")
        return self._apply_scaler(values, mask, scaler, inverse=inverse)

    def _normalize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        state: _ColumnState,
        inverse: bool = False,
    ) -> NDArray[np.floating]:
        method = state.config["normalization"]
        if method == "sigmoid":
            return self._apply_sigmoid(
                values, mask, state.config["sigmoid_scale"], inverse=inverse
            )
        if method == "none":
            return values

        scaler_attr = self._NORMALIZATION_SCALERS.get(method)
        if scaler_attr is None:
            raise ValueError(
                f"Invalid normalization value {method!r}: "
                f"supported values are {', '.join(NORMALIZATION_TYPES)}"
            )
        scaler = getattr(state, scaler_attr, None)
        if scaler is None:
            raise RuntimeError(f"{scaler_attr} not fitted")
        return self._apply_scaler(values, mask, scaler, inverse=inverse)

    # -------------------------------------------------------------------------
    # Fitting methods
    # -------------------------------------------------------------------------

    def _fit_standardization(
        self, values: NDArray[np.floating], state: _ColumnState
    ) -> None:
        method = state.config["standardization"]
        if method == "none":
            return
        if method == "zscore":
            state.standard_scaler = StandardScaler()
            state.standard_scaler.fit(values.reshape(-1, 1))
            return
        if method == "robust":
            q = state.config["robust_quantiles"]
            state.robust_scaler = RobustScaler(quantile_range=(q[0] * 100, q[1] * 100))
            state.robust_scaler.fit(values.reshape(-1, 1))
            return
        if method == "mmad":
            state.median = float(np.median(values))
            mad = np.median(np.abs(values - state.median))
            state.mad = (
                float(mad) if np.isfinite(mad) and not np.isclose(mad, 0.0) else 1.0
            )
            return
        if method == "power_yj":
            state.power_transformer = PowerTransformer(
                method="yeo-johnson", standardize=True
            )
            state.power_transformer.fit(values.reshape(-1, 1))
            return

        raise ValueError(
            f"Invalid standardization value {method!r}: "
            f"supported values are {', '.join(STANDARDIZATION_TYPES)}"
        )

    def _fit_normalization(
        self, values: NDArray[np.floating], state: _ColumnState
    ) -> None:
        method = state.config["normalization"]
        if method == "maxabs":
            state.maxabs_scaler = MaxAbsScaler()
            state.maxabs_scaler.fit(values.reshape(-1, 1))
            return
        if method == "minmax":
            state.minmax_scaler = MinMaxScaler(
                feature_range=state.config["minmax_range"]
            )
            state.minmax_scaler.fit(values.reshape(-1, 1))
            return
        if method in ("sigmoid", "none"):
            return

        raise ValueError(
            f"Invalid normalization value {method!r}: "
            f"supported values are {', '.join(NORMALIZATION_TYPES)}"
        )

    def _fit_column(
        self, column_name: str, values: NDArray[np.floating]
    ) -> _ColumnState:
        """Fit transformation pipeline for a single column."""
        config = self._config.get_column_config(column_name)
        state = _ColumnState(config=config)

        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            logger.warning(
                f"Column {column_name!r}: no finite values found, using fallback [0.0, 1.0]"
            )
            fit_values = np.array([0.0, 1.0])
        else:
            fit_values = finite_values

        # Fit standardization
        self._fit_standardization(fit_values, state)

        # Standardize for normalization fitting
        finite_mask = np.ones(len(fit_values), dtype=bool)
        standardized = self._standardize(fit_values, finite_mask, state, inverse=False)

        # Fit normalization
        self._fit_normalization(standardized, state)

        return state

    def _transform_column(
        self,
        values: NDArray[np.floating],
        state: _ColumnState,
        inverse: bool = False,
    ) -> NDArray[np.floating]:
        """Transform a single column using its fitted state."""
        mask = np.isfinite(values)

        if inverse:
            # Inverse order: gamma -> normalize -> standardize
            degamma = self._apply_gamma(
                values, mask, state.config["gamma"], inverse=True
            )
            denorm = self._normalize(degamma, mask, state, inverse=True)
            return self._standardize(denorm, mask, state, inverse=True)
        else:
            # Forward order: standardize -> normalize -> gamma
            standardized = self._standardize(values, mask, state, inverse=False)
            normalized = self._normalize(standardized, mask, state, inverse=False)
            return self._apply_gamma(
                normalized, mask, state.config["gamma"], inverse=False
            )

    # -------------------------------------------------------------------------
    # BaseTransform interface
    # -------------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: ArrayOrNone = None,
        sample_weight: ArrayOrNone = None,
        feature_list: ListOrNone = None,
        **kwargs,
    ) -> tuple[ArrayLike, ArrayOrNone, ArrayOrNone, ListOrNone]:
        arr = np.asarray(X, dtype=float)

        # Handle 1D vs 2D input
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        n_columns = arr.shape[1]

        # Determine column names
        if feature_list is not None and len(feature_list) == n_columns:
            column_names = list(feature_list)
        else:
            column_names = [f"column_{i}" for i in range(n_columns)]

        # Fit each column
        self._column_states = {}
        for i, col_name in enumerate(column_names):
            col_values = arr[:, i]
            self._column_states[col_name] = self._fit_column(col_name, col_values)

        self._fitted_columns = column_names
        self._fitted = True

        return X, y, sample_weight, feature_list

    def transform(
        self,
        X: ArrayLike,
        y: ArrayOrNone = None,
        sample_weight: ArrayOrNone = None,
        feature_list: ListOrNone = None,
        outlier_check: bool = False,
        **kwargs,
    ) -> tuple[ArrayLike, ArrayOrNone, ArrayOrNone, ListOrNone]:
        if not self._fitted:
            raise RuntimeError("LabelTransformer must be fitted before transform")

        arr = np.asarray(X, dtype=float)
        was_1d = arr.ndim == 1
        if was_1d:
            arr = arr.reshape(-1, 1)

        n_columns = arr.shape[1]

        # Determine column names (must match fitted columns)
        if feature_list is not None and len(feature_list) == n_columns:
            column_names = list(feature_list)
        else:
            column_names = self._fitted_columns

        if len(column_names) != n_columns:
            raise ValueError(
                f"Column count mismatch: fitted on {len(self._fitted_columns)} columns, "
                f"got {n_columns}"
            )

        # Transform each column
        result = np.empty_like(arr)
        for i, col_name in enumerate(column_names):
            if col_name not in self._column_states:
                raise ValueError(f"Column {col_name!r} was not present during fitting")
            result[:, i] = self._transform_column(
                arr[:, i], self._column_states[col_name]
            )

        if was_1d:
            result = result.flatten()

        return result, y, sample_weight, feature_list

    def fit_transform(
        self,
        X: ArrayLike,
        y: ArrayOrNone = None,
        sample_weight: ArrayOrNone = None,
        feature_list: ListOrNone = None,
        **kwargs,
    ) -> tuple[ArrayLike, ArrayOrNone, ArrayOrNone, ListOrNone]:
        self.fit(X, y, sample_weight, feature_list, **kwargs)
        return self.transform(X, y, sample_weight, feature_list, **kwargs)

    def inverse_transform(
        self,
        X: ArrayLike,
        y: ArrayOrNone = None,
        sample_weight: ArrayOrNone = None,
        feature_list: ListOrNone = None,
        **kwargs,
    ) -> tuple[ArrayLike, ArrayOrNone, ArrayOrNone, ListOrNone]:
        if not self._fitted:
            raise RuntimeError(
                "LabelTransformer must be fitted before inverse_transform"
            )

        arr = np.asarray(X, dtype=float)
        was_1d = arr.ndim == 1
        if was_1d:
            arr = arr.reshape(-1, 1)

        n_columns = arr.shape[1]

        # Determine column names
        if feature_list is not None and len(feature_list) == n_columns:
            column_names = list(feature_list)
        else:
            column_names = self._fitted_columns

        if len(column_names) != n_columns:
            raise ValueError(
                f"Column count mismatch: fitted on {len(self._fitted_columns)} columns, "
                f"got {n_columns}"
            )

        # Inverse transform each column
        result = np.empty_like(arr)
        for i, col_name in enumerate(column_names):
            if col_name not in self._column_states:
                raise ValueError(f"Column {col_name!r} was not present during fitting")
            result[:, i] = self._transform_column(
                arr[:, i], self._column_states[col_name], inverse=True
            )

        if was_1d:
            result = result.flatten()

        return result, y, sample_weight, feature_list
