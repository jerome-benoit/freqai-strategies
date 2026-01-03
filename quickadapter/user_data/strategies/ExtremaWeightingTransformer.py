from typing import Any, Final, Literal

import numpy as np
import scipy as sp
from datasieve.transforms.base_transform import BaseTransform
from numpy.typing import NDArray

WeightStrategy = Literal[
    "none",
    "amplitude",
    "amplitude_threshold_ratio",
    "volume_rate",
    "speed",
    "efficiency_ratio",
    "volume_weighted_efficiency_ratio",
    "hybrid",
]
WEIGHT_STRATEGIES: Final[tuple[WeightStrategy, ...]] = (
    "none",
    "amplitude",
    "amplitude_threshold_ratio",
    "volume_rate",
    "speed",
    "efficiency_ratio",
    "volume_weighted_efficiency_ratio",
    "hybrid",
)

WeightSource = Literal[
    "amplitude",
    "amplitude_threshold_ratio",
    "volume_rate",
    "speed",
    "efficiency_ratio",
    "volume_weighted_efficiency_ratio",
]
WEIGHT_SOURCES: Final[tuple[WeightSource, ...]] = (
    "amplitude",
    "amplitude_threshold_ratio",
    "volume_rate",
    "speed",
    "efficiency_ratio",
    "volume_weighted_efficiency_ratio",
)

WeightAggregation = Literal["weighted_sum", "geometric_mean"]
WEIGHT_AGGREGATIONS: Final[tuple[WeightAggregation, ...]] = (
    "weighted_sum",
    "geometric_mean",
)

StandardizationType = Literal["none", "zscore", "robust", "mmad"]
STANDARDIZATION_TYPES: Final[tuple[StandardizationType, ...]] = (
    "none",  # 0 - No standardization
    "zscore",  # 1 - (w - μ) / σ
    "robust",  # 2 - (w - median) / IQR
    "mmad",  # 3 - (w - median) / MAD
)

NormalizationType = Literal["minmax", "sigmoid", "none"]
NORMALIZATION_TYPES: Final[tuple[NormalizationType, ...]] = (
    "minmax",  # 0 - (w - min) / (max - min)
    "sigmoid",  # 1 - 1 / (1 + exp(-scale × w))
    "none",  # 2 - w (identity)
)

DEFAULTS_EXTREMA_WEIGHTING: Final[dict[str, Any]] = {
    "strategy": WEIGHT_STRATEGIES[0],  # "none"
    "source_weights": {s: 1.0 for s in WEIGHT_SOURCES},
    "aggregation": WEIGHT_AGGREGATIONS[0],  # "weighted_sum"
    # Phase 1: Standardization
    "standardization": STANDARDIZATION_TYPES[0],  # "none"
    "robust_quantiles": (0.25, 0.75),
    "mmad_scaling_factor": 1.4826,
    # Phase 2: Normalization
    "normalization": NORMALIZATION_TYPES[0],  # "minmax"
    "minmax_range": (-1.0, 1.0),
    "sigmoid_scale": 1.0,
    # Phase 3: Post-processing
    "gamma": 1.0,
}


class ExtremaWeightingTransformer(BaseTransform):
    def __init__(self, *, extrema_weighting: dict[str, Any]) -> None:
        self.extrema_weighting = extrema_weighting or {}
        self._fitted = False
        self._mean: float = 0.0
        self._std: float = 1.0
        self._min: float = 0.0
        self._max: float = 1.0
        self._median: float = 0.0
        self._iqr: float = 1.0
        self._mad: float = 1.0
        self._n_train: int = 0

    def _get_config(
        self,
    ) -> tuple[
        StandardizationType,
        NormalizationType,
        float,
        tuple[float, float],
        float,
        float,
    ]:
        """Extract and validate configuration parameters."""
        config = self.extrema_weighting
        standardization: StandardizationType = config.get(
            "standardization", DEFAULTS_EXTREMA_WEIGHTING["standardization"]
        )
        normalization: NormalizationType = config.get(
            "normalization", DEFAULTS_EXTREMA_WEIGHTING["normalization"]
        )
        gamma = float(config.get("gamma", DEFAULTS_EXTREMA_WEIGHTING["gamma"]))
        minmax_range: tuple[float, float] = config.get(
            "minmax_range", DEFAULTS_EXTREMA_WEIGHTING["minmax_range"]
        )
        sigmoid_scale = float(
            config.get("sigmoid_scale", DEFAULTS_EXTREMA_WEIGHTING["sigmoid_scale"])
        )
        mmad_scaling_factor = float(
            config.get(
                "mmad_scaling_factor", DEFAULTS_EXTREMA_WEIGHTING["mmad_scaling_factor"]
            )
        )
        return (
            standardization,
            normalization,
            gamma,
            minmax_range,
            sigmoid_scale,
            mmad_scaling_factor,
        )

    def _standardize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        method: StandardizationType,
        mmad_scaling_factor: float,
    ) -> NDArray[np.floating]:
        """Apply standardization to non-zero values using fitted statistics."""
        if method == STANDARDIZATION_TYPES[0]:  # "none"
            return values
        out = values.copy()
        if method == STANDARDIZATION_TYPES[1]:  # "zscore"
            out[mask] = (values[mask] - self._mean) / self._std
        elif method == STANDARDIZATION_TYPES[2]:  # "robust"
            out[mask] = (values[mask] - self._median) / self._iqr
        elif method == STANDARDIZATION_TYPES[3]:  # "mmad"
            out[mask] = (values[mask] - self._median) / (
                self._mad * mmad_scaling_factor
            )
        else:
            raise ValueError(
                f"Unsupported standardization: {method!r}. "
                f"Supported: {', '.join(STANDARDIZATION_TYPES)}"
            )
        return out

    def _normalize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        method: NormalizationType,
        minmax_range: tuple[float, float],
        sigmoid_scale: float,
    ) -> NDArray[np.floating]:
        """Apply normalization to non-zero values using fitted statistics."""
        if method == NORMALIZATION_TYPES[2]:  # "none"
            return values
        out = values.copy()
        if method == NORMALIZATION_TYPES[0]:  # "minmax"
            denom = self._max - self._min
            low, high = minmax_range
            out[mask] = low + (values[mask] - self._min) / denom * (high - low)
        elif method == NORMALIZATION_TYPES[1]:  # "sigmoid"
            out[mask] = sp.special.expit(sigmoid_scale * values[mask])
        else:
            raise ValueError(
                f"Unsupported normalization: {method!r}. "
                f"Supported: {', '.join(NORMALIZATION_TYPES)}"
            )
        return out

    def _apply_gamma(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        gamma: float,
    ) -> NDArray[np.floating]:
        """Apply gamma correction to non-zero values."""
        if np.isclose(gamma, 1.0) or not np.isfinite(gamma) or gamma <= 0:
            return values
        out = values.copy()
        out[mask] = np.power(np.abs(values[mask]), gamma) * np.sign(values[mask])
        return out

    def _inverse_standardize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        method: StandardizationType,
        mmad_scaling_factor: float,
    ) -> NDArray[np.floating]:
        """Inverse standardization."""
        if method == STANDARDIZATION_TYPES[0]:  # "none"
            return values
        out = values.copy()
        if method == STANDARDIZATION_TYPES[1]:  # "zscore"
            out[mask] = values[mask] * self._std + self._mean
        elif method == STANDARDIZATION_TYPES[2]:  # "robust"
            out[mask] = values[mask] * self._iqr + self._median
        elif method == STANDARDIZATION_TYPES[3]:  # "mmad"
            out[mask] = values[mask] * (self._mad * mmad_scaling_factor) + self._median
        return out

    def _inverse_normalize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        method: NormalizationType,
        minmax_range: tuple[float, float],
        sigmoid_scale: float,
    ) -> NDArray[np.floating]:
        """Inverse normalization."""
        if method == NORMALIZATION_TYPES[2]:  # "none"
            return values
        out = values.copy()
        if method == NORMALIZATION_TYPES[0]:  # "minmax"
            low, high = minmax_range
            denom = self._max - self._min
            out[mask] = self._min + (values[mask] - low) / (high - low) * denom
        elif method == NORMALIZATION_TYPES[1]:  # "sigmoid"
            clipped = np.clip(values[mask], 1e-7, 1.0 - 1e-7)
            out[mask] = -np.log(1.0 / clipped - 1.0) / sigmoid_scale
        return out

    def _inverse_gamma(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        gamma: float,
    ) -> NDArray[np.floating]:
        """Inverse gamma correction."""
        if np.isclose(gamma, 1.0) or not np.isfinite(gamma) or gamma <= 0:
            return values
        out = values.copy()
        out[mask] = np.power(np.abs(values[mask]), 1.0 / gamma) * np.sign(values[mask])
        return out

    def fit(
        self,
        X: NDArray[np.floating],
        y: Any = None,
        sample_weight: Any = None,
        feature_list: Any = None,
        **kwargs: Any,
    ) -> "ExtremaWeightingTransformer":
        """Fit transformer on training labels (non-zero values only)."""
        values = np.asarray(X, dtype=float).ravel()
        nonzero_mask = values != 0.0
        nonzero_values = values[nonzero_mask & np.isfinite(values)]

        if nonzero_values.size == 0:
            self._fitted = True
            return self

        self._n_train = int(nonzero_values.size)
        self._mean = float(np.mean(nonzero_values))
        std = float(np.std(nonzero_values))
        self._std = std if np.isfinite(std) and std > 0 else 1.0
        self._min = float(np.min(nonzero_values))
        self._max = float(np.max(nonzero_values))
        if np.isclose(self._max, self._min):
            self._max = self._min + 1.0
        self._median = float(np.median(nonzero_values))
        q1, q3 = (
            float(np.percentile(nonzero_values, 25)),
            float(np.percentile(nonzero_values, 75)),
        )
        iqr = q3 - q1
        self._iqr = iqr if np.isfinite(iqr) and iqr > 0 else 1.0
        mad = float(np.median(np.abs(nonzero_values - self._median)))
        self._mad = mad if np.isfinite(mad) and mad > 0 else 1.0

        self._fitted = True
        return self

    def transform(
        self,
        X: NDArray[np.floating],
        y: Any = None,
        sample_weight: Any = None,
        feature_list: Any = None,
        outlier_check: bool = False,
        **kwargs: Any,
    ) -> tuple[NDArray[np.floating], Any, Any, Any]:
        """Transform labels using fitted statistics."""
        if not self._fitted:
            raise RuntimeError(
                "ExtremaWeightingTransformer must be fitted before transform"
            )

        (
            standardization,
            normalization,
            gamma,
            minmax_range,
            sigmoid_scale,
            mmad_scaling_factor,
        ) = self._get_config()

        arr = np.asarray(X, dtype=float)
        nonzero_mask = arr != 0.0

        # Phase 1: Standardization
        standardized = self._standardize(
            arr, nonzero_mask, standardization, mmad_scaling_factor
        )
        # Phase 2: Normalization
        normalized = self._normalize(
            standardized,
            nonzero_mask,
            normalization,
            minmax_range,
            sigmoid_scale,
        )
        # Phase 3: Gamma correction
        result = self._apply_gamma(normalized, nonzero_mask, gamma)

        return result, y, sample_weight, feature_list

    def fit_transform(
        self,
        X: NDArray[np.floating],
        y: Any = None,
        sample_weight: Any = None,
        feature_list: Any = None,
        **kwargs: Any,
    ) -> tuple[NDArray[np.floating], Any, Any, Any]:
        """Fit and transform in one step."""
        self.fit(X, y, sample_weight, feature_list, **kwargs)
        return self.transform(X, y, sample_weight, feature_list, **kwargs)

    def inverse_transform(
        self,
        X: NDArray[np.floating],
        y: Any = None,
        sample_weight: Any = None,
        feature_list: Any = None,
        **kwargs: Any,
    ) -> tuple[NDArray[np.floating], Any, Any, Any]:
        """Inverse transform to recover original scale."""
        if not self._fitted:
            raise RuntimeError(
                "ExtremaWeightingTransformer must be fitted before inverse_transform"
            )

        (
            standardization,
            normalization,
            gamma,
            minmax_range,
            sigmoid_scale,
            mmad_scaling_factor,
        ) = self._get_config()

        arr = np.asarray(X, dtype=float)
        nonzero_mask = arr != 0.0

        # Inverse in reverse order: gamma -> normalization -> standardization
        degamma = self._inverse_gamma(arr, nonzero_mask, gamma)
        denormalized = self._inverse_normalize(
            degamma, nonzero_mask, normalization, minmax_range, sigmoid_scale
        )
        destandardized = self._inverse_standardize(
            denormalized, nonzero_mask, standardization, mmad_scaling_factor
        )

        return destandardized, y, sample_weight, feature_list
