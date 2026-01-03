from typing import Any, Final, Literal

import numpy as np
import scipy as sp
from datasieve.transforms.base_transform import (
    ArrayOrNone,
    BaseTransform,
    ListOrNone,
)
from numpy.typing import ArrayLike, NDArray

WeightStrategy = Literal[
    "none",
    "amplitude",
    "amplitude_threshold_ratio",
    "volume_rate",
    "speed",
    "efficiency_ratio",
    "volume_weighted_efficiency_ratio",
]
WEIGHT_STRATEGIES: Final[tuple[WeightStrategy, ...]] = (
    "none",
    "amplitude",
    "amplitude_threshold_ratio",
    "volume_rate",
    "speed",
    "efficiency_ratio",
    "volume_weighted_efficiency_ratio",
)

StandardizationType = Literal["none", "zscore", "robust", "mmad"]
STANDARDIZATION_TYPES: Final[tuple[StandardizationType, ...]] = (
    "none",  # 0 - w (identity)
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
        super().__init__(name="ExtremaWeightingTransformer")
        self.extrema_weighting = {**DEFAULTS_EXTREMA_WEIGHTING, **extrema_weighting}
        self._fitted = False
        self._mean = 0.0
        self._std = 1.0
        self._min = 0.0
        self._max = 1.0
        self._median = 0.0
        self._iqr = 1.0
        self._mad = 1.0

    def _standardize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating]:
        method = self.extrema_weighting["standardization"]
        if method == STANDARDIZATION_TYPES[0]:  # "none"
            return values
        out = values.copy()
        if method == STANDARDIZATION_TYPES[1]:  # "zscore"
            out[mask] = (values[mask] - self._mean) / self._std
        elif method == STANDARDIZATION_TYPES[2]:  # "robust"
            out[mask] = (values[mask] - self._median) / self._iqr
        elif method == STANDARDIZATION_TYPES[3]:  # "mmad"
            mmad_scaling_factor = self.extrema_weighting["mmad_scaling_factor"]
            out[mask] = (values[mask] - self._median) / (
                self._mad * mmad_scaling_factor
            )
        else:
            raise ValueError(
                f"Invalid standardization {method!r}. "
                f"Supported: {', '.join(STANDARDIZATION_TYPES)}"
            )
        return out

    def _normalize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating]:
        method = self.extrema_weighting["normalization"]
        if method == NORMALIZATION_TYPES[2]:  # "none"
            return values
        out = values.copy()
        if method == NORMALIZATION_TYPES[0]:  # "minmax"
            minmax_range = self.extrema_weighting["minmax_range"]
            value_range = self._max - self._min
            low, high = minmax_range
            scale_range = high - low

            if (
                not np.isfinite(value_range)
                or np.isclose(value_range, 0.0)
                or not np.isfinite(scale_range)
                or np.isclose(scale_range, 0.0)
            ):
                return values

            out[mask] = low + (values[mask] - self._min) / value_range * scale_range
        elif method == NORMALIZATION_TYPES[1]:  # "sigmoid"
            sigmoid_scale = self.extrema_weighting["sigmoid_scale"]
            out[mask] = sp.special.expit(sigmoid_scale * values[mask])
        else:
            raise ValueError(
                f"Invalid normalization {method!r}. "
                f"Supported: {', '.join(NORMALIZATION_TYPES)}"
            )
        return out

    def _apply_gamma(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating]:
        gamma = self.extrema_weighting["gamma"]
        if np.isclose(gamma, 1.0) or not np.isfinite(gamma) or gamma <= 0:
            return values
        out = values.copy()
        out[mask] = np.sign(values[mask]) * np.power(np.abs(values[mask]), gamma)
        return out

    def _inverse_standardize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating]:
        method = self.extrema_weighting["standardization"]
        if method == STANDARDIZATION_TYPES[0]:  # "none"
            return values
        out = values.copy()
        if method == STANDARDIZATION_TYPES[1]:  # "zscore"
            out[mask] = values[mask] * self._std + self._mean
        elif method == STANDARDIZATION_TYPES[2]:  # "robust"
            out[mask] = values[mask] * self._iqr + self._median
        elif method == STANDARDIZATION_TYPES[3]:  # "mmad"
            mmad_scaling_factor = self.extrema_weighting["mmad_scaling_factor"]
            out[mask] = values[mask] * (self._mad * mmad_scaling_factor) + self._median
        else:
            raise ValueError(
                f"Invalid standardization {method!r}. "
                f"Supported: {', '.join(STANDARDIZATION_TYPES)}"
            )
        return out

    def _inverse_normalize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating]:
        method = self.extrema_weighting["normalization"]
        if method == NORMALIZATION_TYPES[2]:  # "none"
            return values
        out = values.copy()
        if method == NORMALIZATION_TYPES[0]:  # "minmax"
            minmax_range = self.extrema_weighting["minmax_range"]
            low, high = minmax_range
            value_range = self._max - self._min
            scale_range = high - low

            if (
                not np.isfinite(value_range)
                or np.isclose(value_range, 0.0)
                or not np.isfinite(scale_range)
                or np.isclose(scale_range, 0.0)
            ):
                return values

            out[mask] = self._min + (values[mask] - low) / scale_range * value_range
        elif method == NORMALIZATION_TYPES[1]:  # "sigmoid"
            sigmoid_scale = self.extrema_weighting["sigmoid_scale"]
            out[mask] = sp.special.logit(values[mask]) / sigmoid_scale
        else:
            raise ValueError(
                f"Invalid normalization {method!r}. "
                f"Supported: {', '.join(NORMALIZATION_TYPES)}"
            )
        return out

    def _inverse_gamma(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating]:
        gamma = self.extrema_weighting["gamma"]
        if np.isclose(gamma, 1.0) or not np.isfinite(gamma) or gamma <= 0:
            return values
        out = values.copy()
        out[mask] = np.power(np.abs(values[mask]), 1.0 / gamma) * np.sign(values[mask])
        return out

    def fit(
        self,
        X: ArrayLike,
        y: ArrayOrNone = None,
        sample_weight: ArrayOrNone = None,
        feature_list: ListOrNone = None,
        **kwargs,
    ) -> tuple[ArrayLike, ArrayOrNone, ArrayOrNone, ListOrNone]:
        values = np.asarray(X, dtype=float)
        non_zero_finite_values = values[np.isfinite(values) & ~np.isclose(values, 0.0)]

        if non_zero_finite_values.size == 0:
            self._mean = 0.0
            self._std = 1.0
            self._min = 0.0
            self._max = 1.0
            self._median = 0.0
            self._iqr = 1.0
            self._mad = 1.0
            self._fitted = True
            return X, y, sample_weight, feature_list

        robust_quantiles = self.extrema_weighting["robust_quantiles"]

        self._mean = np.mean(non_zero_finite_values)
        std = np.std(non_zero_finite_values, ddof=1)
        self._std = std if np.isfinite(std) and not np.isclose(std, 0.0) else 1.0
        self._min = np.min(non_zero_finite_values)
        self._max = np.max(non_zero_finite_values)
        if np.isclose(self._max, self._min):
            self._max = self._min + 1.0
        self._median = np.median(non_zero_finite_values)
        q1, q3 = (
            np.quantile(non_zero_finite_values, robust_quantiles[0]),
            np.quantile(non_zero_finite_values, robust_quantiles[1]),
        )
        iqr = q3 - q1
        self._iqr = iqr if np.isfinite(iqr) and not np.isclose(iqr, 0.0) else 1.0
        mad = np.median(np.abs(non_zero_finite_values - self._median))
        self._mad = mad if np.isfinite(mad) and not np.isclose(mad, 0.0) else 1.0

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
            raise RuntimeError(
                "ExtremaWeightingTransformer must be fitted before transform"
            )

        arr = np.asarray(X, dtype=float)
        mask = np.isfinite(arr) & ~np.isclose(arr, 0.0)

        standardized = self._standardize(arr, mask)
        normalized = self._normalize(standardized, mask)
        gammaized = self._apply_gamma(normalized, mask)

        return gammaized, y, sample_weight, feature_list

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
                "ExtremaWeightingTransformer must be fitted before inverse_transform"
            )

        arr = np.asarray(X, dtype=float)
        mask = np.isfinite(arr) & ~np.isclose(arr, 0.0)

        degammaized = self._inverse_gamma(arr, mask)
        denormalized = self._inverse_normalize(degammaized, mask)
        destandardized = self._inverse_standardize(denormalized, mask)

        return destandardized, y, sample_weight, feature_list
