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
    _STANDARDIZATION_SCALERS: dict[str, str] = {
        "zscore": "_standard_scaler",
        "robust": "_robust_scaler",
        "power_yj": "_power_transformer",
    }
    _NORMALIZATION_SCALERS: dict[str, str] = {
        "maxabs": "_maxabs_scaler",
        "minmax": "_minmax_scaler",
    }

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

    def _apply_scaler(
        self,
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

    def _apply_mmad(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        inverse: bool = False,
    ) -> NDArray[np.floating]:
        if values[mask].size == 0:
            return values
        out = values.copy()
        k = self.extrema_weighting["mmad_scaling_factor"]
        if inverse:
            out[mask] = values[mask] * (self._mad * k) + self._median
        else:
            out[mask] = (values[mask] - self._median) / (self._mad * k)
        return out

    def _apply_sigmoid(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
        inverse: bool = False,
    ) -> NDArray[np.floating]:
        if values[mask].size == 0:
            return values
        scale = self.extrema_weighting["sigmoid_scale"]
        if not np.isfinite(scale) or np.isclose(scale, 0.0):
            return values
        out = values.copy()
        if inverse:
            out[mask] = sp.special.logit((values[mask] + 1.0) / 2.0) / scale
        else:
            out[mask] = 2.0 * sp.special.expit(scale * values[mask]) - 1.0
        return out

    def _standardize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating]:
        method = self.extrema_weighting["standardization"]
        if method == STANDARDIZATION_TYPES[0]:  # "none"
            return values
        if method == STANDARDIZATION_TYPES[3]:  # "mmad"
            return self._apply_mmad(values, mask, inverse=False)

        scaler_attr = self._STANDARDIZATION_SCALERS.get(method)
        if scaler_attr is None:
            raise ValueError(
                f"Invalid standardization {method!r}. "
                f"Supported: {', '.join(STANDARDIZATION_TYPES)}"
            )
        scaler = getattr(self, scaler_attr)
        if scaler is None:
            raise RuntimeError(f"{scaler_attr[1:]} not fitted")
        return self._apply_scaler(values, mask, scaler, inverse=False)

    def _normalize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating]:
        method = self.extrema_weighting["normalization"]
        if method == NORMALIZATION_TYPES[2]:  # "sigmoid"
            return self._apply_sigmoid(values, mask, inverse=False)
        if method == NORMALIZATION_TYPES[3]:  # "none"
            return values

        scaler_attr = self._NORMALIZATION_SCALERS.get(method)
        if scaler_attr is None:
            raise ValueError(
                f"Invalid normalization {method!r}. "
                f"Supported: {', '.join(NORMALIZATION_TYPES)}"
            )
        scaler = getattr(self, scaler_attr)
        if scaler is None:
            raise RuntimeError(f"{scaler_attr[1:]} not fitted")
        return self._apply_scaler(values, mask, scaler, inverse=False)

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
        if method == STANDARDIZATION_TYPES[3]:  # "mmad"
            return self._apply_mmad(values, mask, inverse=True)

        scaler_attr = self._STANDARDIZATION_SCALERS.get(method)
        if scaler_attr is None:
            raise ValueError(
                f"Invalid standardization {method!r}. "
                f"Supported: {', '.join(STANDARDIZATION_TYPES)}"
            )
        scaler = getattr(self, scaler_attr)
        if scaler is None:
            raise RuntimeError(f"{scaler_attr[1:]} not fitted")
        return self._apply_scaler(values, mask, scaler, inverse=True)

    def _inverse_normalize(
        self,
        values: NDArray[np.floating],
        mask: NDArray[np.bool_],
    ) -> NDArray[np.floating]:
        method = self.extrema_weighting["normalization"]
        if method == NORMALIZATION_TYPES[2]:  # "sigmoid"
            return self._apply_sigmoid(values, mask, inverse=True)
        if method == NORMALIZATION_TYPES[3]:  # "none"
            return values

        scaler_attr = self._NORMALIZATION_SCALERS.get(method)
        if scaler_attr is None:
            raise ValueError(
                f"Invalid normalization {method!r}. "
                f"Supported: {', '.join(NORMALIZATION_TYPES)}"
            )
        scaler = getattr(self, scaler_attr)
        if scaler is None:
            raise RuntimeError(f"{scaler_attr[1:]} not fitted")
        return self._apply_scaler(values, mask, scaler, inverse=True)

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

    def _fit_standardization(self, values: NDArray[np.floating]) -> None:
        method = self.extrema_weighting["standardization"]
        if method == STANDARDIZATION_TYPES[0]:  # "none"
            return
        if method == STANDARDIZATION_TYPES[1]:  # "zscore"
            self._standard_scaler = StandardScaler()
            self._standard_scaler.fit(values.reshape(-1, 1))
            return
        if method == STANDARDIZATION_TYPES[2]:  # "robust"
            q = self.extrema_weighting["robust_quantiles"]
            self._robust_scaler = RobustScaler(quantile_range=(q[0] * 100, q[1] * 100))
            self._robust_scaler.fit(values.reshape(-1, 1))
            return
        if method == STANDARDIZATION_TYPES[3]:  # "mmad"
            self._median = float(np.median(values))
            mad = np.median(np.abs(values - self._median))
            self._mad = (
                float(mad) if np.isfinite(mad) and not np.isclose(mad, 0.0) else 1.0
            )
            return
        if method == STANDARDIZATION_TYPES[4]:  # "power_yj"
            self._power_transformer = PowerTransformer(
                method="yeo-johnson", standardize=True
            )
            self._power_transformer.fit(values.reshape(-1, 1))
            return

        raise ValueError(
            f"Invalid standardization {method!r}. Supported: {', '.join(STANDARDIZATION_TYPES)}"
        )

    def _fit_normalization(self, values: NDArray[np.floating]) -> None:
        method = self.extrema_weighting["normalization"]
        if method == NORMALIZATION_TYPES[0]:  # "maxabs"
            self._maxabs_scaler = MaxAbsScaler()
            self._maxabs_scaler.fit(values.reshape(-1, 1))
            return
        if method == NORMALIZATION_TYPES[1]:  # "minmax"
            self._minmax_scaler = MinMaxScaler(
                feature_range=self.extrema_weighting["minmax_range"]
            )
            self._minmax_scaler.fit(values.reshape(-1, 1))
            return
        if method == NORMALIZATION_TYPES[2]:  # "sigmoid"
            return
        if method == NORMALIZATION_TYPES[3]:  # "none"
            return

        raise ValueError(
            f"Invalid normalization {method!r}. Supported: {', '.join(NORMALIZATION_TYPES)}"
        )

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

        fit_values = finite_values if finite_values.size > 0 else np.array([0.0, 1.0])

        self._fit_standardization(fit_values)

        finite_mask = np.ones(len(fit_values), dtype=bool)
        standardized_fit_values = self._standardize(fit_values, finite_mask)
        self._fit_normalization(standardized_fit_values)

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
