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

DEFAULTS_EXTREMA_WEIGHTING: Final[dict[str, Any]] = {
    "strategy": WEIGHT_STRATEGIES[0],  # "none"
    # Phase 1: Standardization
    "standardization": STANDARDIZATION_TYPES[0],  # "none"
    "robust_quantiles": (0.25, 0.75),
    "mmad_scaling_factor": 1.4826,
    # Phase 2: Normalization
    "normalization": NORMALIZATION_TYPES[0],  # "maxabs"
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
        # Phase 1: Standardization
        self._standard_scaler: StandardScaler | None = None
        self._robust_scaler: RobustScaler | None = None
        self._power_transformer: PowerTransformer | None = None
        self._median = 0.0
        self._mad = 1.0
        # Phase 2: Normalization
        self._minmax_scaler: MinMaxScaler | None = None
        self._maxabs_scaler: MaxAbsScaler | None = None

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
            if self._standard_scaler is None:
                raise RuntimeError("StandardScaler not fitted")
            if values[mask].size > 0:
                out[mask] = self._standard_scaler.transform(
                    values[mask].reshape(-1, 1)
                ).flatten()
        elif method == STANDARDIZATION_TYPES[2]:  # "robust"
            if self._robust_scaler is None:
                raise RuntimeError("RobustScaler not fitted")
            if values[mask].size > 0:
                out[mask] = self._robust_scaler.transform(
                    values[mask].reshape(-1, 1)
                ).flatten()
        elif method == STANDARDIZATION_TYPES[3]:  # "mmad"
            if values[mask].size > 0:
                mmad_scaling_factor = self.extrema_weighting["mmad_scaling_factor"]
                out[mask] = (values[mask] - self._median) / (
                    self._mad * mmad_scaling_factor
                )
        elif method == STANDARDIZATION_TYPES[4]:  # "power_yj"
            if self._power_transformer is None:
                raise RuntimeError("PowerTransformer not fitted")
            if values[mask].size > 0:
                out[mask] = self._power_transformer.transform(
                    values[mask].reshape(-1, 1)
                ).flatten()
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
        if method == NORMALIZATION_TYPES[3]:  # "none"
            return values
        out = values.copy()
        if method == NORMALIZATION_TYPES[0]:  # "maxabs"
            if self._maxabs_scaler is None:
                raise RuntimeError("MaxAbsScaler not fitted")
            if values[mask].size > 0:
                out[mask] = self._maxabs_scaler.transform(
                    values[mask].reshape(-1, 1)
                ).flatten()
        elif method == NORMALIZATION_TYPES[1]:  # "minmax"
            if self._minmax_scaler is None:
                raise RuntimeError("MinMaxScaler not fitted")
            if values[mask].size > 0:
                out[mask] = self._minmax_scaler.transform(
                    values[mask].reshape(-1, 1)
                ).flatten()
        elif method == NORMALIZATION_TYPES[2]:  # "sigmoid"
            sigmoid_scale = self.extrema_weighting["sigmoid_scale"]
            out[mask] = 2.0 * sp.special.expit(sigmoid_scale * values[mask]) - 1.0
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
            if self._standard_scaler is None:
                raise RuntimeError("StandardScaler not fitted")
            if values[mask].size > 0:
                out[mask] = self._standard_scaler.inverse_transform(
                    values[mask].reshape(-1, 1)
                ).flatten()
        elif method == STANDARDIZATION_TYPES[2]:  # "robust"
            if self._robust_scaler is None:
                raise RuntimeError("RobustScaler not fitted")
            if values[mask].size > 0:
                out[mask] = self._robust_scaler.inverse_transform(
                    values[mask].reshape(-1, 1)
                ).flatten()
        elif method == STANDARDIZATION_TYPES[3]:  # "mmad"
            if values[mask].size > 0:
                mmad_scaling_factor = self.extrema_weighting["mmad_scaling_factor"]
                out[mask] = (
                    values[mask] * (self._mad * mmad_scaling_factor) + self._median
                )
        elif method == STANDARDIZATION_TYPES[4]:  # "power_yj"
            if self._power_transformer is None:
                raise RuntimeError("PowerTransformer not fitted")
            if values[mask].size > 0:
                out[mask] = self._power_transformer.inverse_transform(
                    values[mask].reshape(-1, 1)
                ).flatten()
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
        if method == NORMALIZATION_TYPES[3]:  # "none"
            return values
        out = values.copy()
        if method == NORMALIZATION_TYPES[0]:  # "maxabs"
            if self._maxabs_scaler is None:
                raise RuntimeError("MaxAbsScaler not fitted")
            if values[mask].size > 0:
                out[mask] = self._maxabs_scaler.inverse_transform(
                    values[mask].reshape(-1, 1)
                ).flatten()
        elif method == NORMALIZATION_TYPES[1]:  # "minmax"
            if self._minmax_scaler is None:
                raise RuntimeError("MinMaxScaler not fitted")
            if values[mask].size > 0:
                out[mask] = self._minmax_scaler.inverse_transform(
                    values[mask].reshape(-1, 1)
                ).flatten()
        elif method == NORMALIZATION_TYPES[2]:  # "sigmoid"
            sigmoid_scale = self.extrema_weighting["sigmoid_scale"]
            out[mask] = sp.special.logit((values[mask] + 1.0) / 2.0) / sigmoid_scale
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
        out[mask] = np.sign(values[mask]) * np.power(np.abs(values[mask]), 1.0 / gamma)
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
        finite_values = values[np.isfinite(values)]

        standardization = self.extrema_weighting["standardization"]
        normalization = self.extrema_weighting["normalization"]

        if finite_values.size == 0:
            if standardization == STANDARDIZATION_TYPES[1]:  # "zscore"
                self._standard_scaler = StandardScaler()
                self._standard_scaler.fit([[0.0], [1.0]])
            elif standardization == STANDARDIZATION_TYPES[2]:  # "robust"
                self._robust_scaler = RobustScaler(
                    quantile_range=(
                        self.extrema_weighting["robust_quantiles"][0] * 100,
                        self.extrema_weighting["robust_quantiles"][1] * 100,
                    )
                )
                self._robust_scaler.fit([[0.0], [1.0]])
            elif standardization == STANDARDIZATION_TYPES[3]:  # "mmad"
                self._median = 0.0
                self._mad = 1.0
            elif standardization == STANDARDIZATION_TYPES[4]:  # "power_yj"
                self._power_transformer = PowerTransformer(
                    method="yeo-johnson", standardize=True
                )
                self._power_transformer.fit(np.array([[0.0], [1.0]]))

            if normalization == NORMALIZATION_TYPES[0]:  # "maxabs"
                self._maxabs_scaler = MaxAbsScaler()
                self._maxabs_scaler.fit([[0.0], [1.0]])
            elif normalization == NORMALIZATION_TYPES[1]:  # "minmax"
                self._minmax_scaler = MinMaxScaler(
                    feature_range=self.extrema_weighting["minmax_range"]
                )
                self._minmax_scaler.fit(np.array([[0.0], [1.0]]))

            self._fitted = True
            return X, y, sample_weight, feature_list

        if standardization == STANDARDIZATION_TYPES[1]:  # "zscore"
            self._standard_scaler = StandardScaler()
            self._standard_scaler.fit(finite_values.reshape(-1, 1))
        elif standardization == STANDARDIZATION_TYPES[2]:  # "robust"
            self._robust_scaler = RobustScaler(
                quantile_range=(
                    self.extrema_weighting["robust_quantiles"][0] * 100,
                    self.extrema_weighting["robust_quantiles"][1] * 100,
                )
            )
            self._robust_scaler.fit(finite_values.reshape(-1, 1))
        elif standardization == STANDARDIZATION_TYPES[3]:  # "mmad"
            self._median = np.median(finite_values)
            mad = np.median(np.abs(finite_values - self._median))
            self._mad = mad if np.isfinite(mad) and not np.isclose(mad, 0.0) else 1.0
        elif standardization == STANDARDIZATION_TYPES[4]:  # "power_yj"
            self._power_transformer = PowerTransformer(
                method="yeo-johnson", standardize=True
            )
            self._power_transformer.fit(finite_values.reshape(-1, 1))

        finite_mask = np.ones(len(finite_values), dtype=bool)
        standardized_values = self._standardize(finite_values, finite_mask)

        if normalization == NORMALIZATION_TYPES[0]:  # "maxabs"
            self._maxabs_scaler = MaxAbsScaler()
            self._maxabs_scaler.fit(standardized_values.reshape(-1, 1))
        elif normalization == NORMALIZATION_TYPES[1]:  # "minmax"
            self._minmax_scaler = MinMaxScaler(
                feature_range=self.extrema_weighting["minmax_range"]
            )
            self._minmax_scaler.fit(standardized_values.reshape(-1, 1))

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
        mask = np.isfinite(arr)

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
        mask = np.isfinite(arr)

        degammaized = self._inverse_gamma(arr, mask)
        denormalized = self._inverse_normalize(degammaized, mask)
        destandardized = self._inverse_standardize(denormalized, mask)

        return destandardized, y, sample_weight, feature_list
