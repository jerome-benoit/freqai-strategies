import copy
import functools
import hashlib
import math
from enum import IntEnum
from functools import lru_cache
from logging import Logger
from typing import Any, Callable, Final, Literal, Optional, TypeVar, Union

import numpy as np
import optuna
import pandas as pd
import scipy as sp
import talib.abstract as ta
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from technical import qtpylib

T = TypeVar("T", pd.Series, float)


WeightStrategy = Literal[
    "none",
    "amplitude",
    "amplitude_threshold_ratio",
    "volume",
    "speed",
    "efficiency_ratio",
    "hybrid",
]
WEIGHT_STRATEGIES: Final[tuple[WeightStrategy, ...]] = (
    "none",
    "amplitude",
    "amplitude_threshold_ratio",
    "volume",
    "speed",
    "efficiency_ratio",
    "hybrid",
)

HybridWeightSource = Literal[
    "amplitude",
    "amplitude_threshold_ratio",
    "volume",
    "speed",
    "efficiency_ratio",
]
HYBRID_WEIGHT_SOURCES: Final[tuple[HybridWeightSource, ...]] = (
    "amplitude",
    "amplitude_threshold_ratio",
    "volume",
    "speed",
    "efficiency_ratio",
)

HybridAggregation = Literal["weighted_sum", "geometric_mean"]
HYBRID_AGGREGATIONS: Final[tuple[HybridAggregation, ...]] = (
    "weighted_sum",
    "geometric_mean",
)

EXTREMA_COLUMN: Final = "&s-extrema"
MAXIMA_THRESHOLD_COLUMN: Final = "&s-maxima_threshold"
MINIMA_THRESHOLD_COLUMN: Final = "&s-minima_threshold"

StandardizationType = Literal["none", "zscore", "robust", "mmad"]
STANDARDIZATION_TYPES: Final[tuple[StandardizationType, ...]] = (
    "none",  # 0 - No standardization
    "zscore",  # 1 - (w - μ) / σ
    "robust",  # 2 - (w - median) / IQR
    "mmad",  # 3 - (w - median) / MAD
)

NormalizationType = Literal["minmax", "sigmoid", "softmax", "l1", "l2", "rank", "none"]
NORMALIZATION_TYPES: Final[tuple[NormalizationType, ...]] = (
    "minmax",  # 0 - (w - min) / (max - min)
    "sigmoid",  # 1 - 1 / (1 + exp(-scale × w))
    "softmax",  # 2 - exp(w/T) / Σexp(w/T)
    "l1",  # 3 - w / Σ|w|
    "l2",  # 4 - w / ||w||₂
    "rank",  # 5 - (rank(w) - 1) / (n - 1)
    "none",  # 6 - w (identity)
)

RankMethod = Literal["average", "min", "max", "dense", "ordinal"]
RANK_METHODS: Final[tuple[RankMethod, ...]] = (
    "average",
    "min",
    "max",
    "dense",
    "ordinal",
)

SmoothingKernel = Literal["gaussian", "kaiser", "triang"]
SmoothingMethod = Union[
    SmoothingKernel, Literal["smm", "sma", "savgol", "nadaraya_watson"]
]
SMOOTHING_METHODS: Final[tuple[SmoothingMethod, ...]] = (
    "gaussian",
    "kaiser",
    "triang",
    "smm",
    "sma",
    "savgol",
    "nadaraya_watson",
)

SmoothingMode = Literal["mirror", "constant", "nearest", "wrap", "interp"]
SMOOTHING_MODES: Final[tuple[SmoothingMode, ...]] = (
    "mirror",
    "constant",
    "nearest",
    "wrap",
    "interp",
)


DEFAULTS_EXTREMA_SMOOTHING: Final[dict[str, Any]] = {
    "method": SMOOTHING_METHODS[0],  # "gaussian"
    "window": 5,
    "beta": 8.0,
    "polyorder": 3,
    "mode": SMOOTHING_MODES[0],  # "mirror"
    "bandwidth": 1.0,
}

DEFAULTS_EXTREMA_WEIGHTING: Final[dict[str, Any]] = {
    "strategy": WEIGHT_STRATEGIES[0],  # "none"
    "source_weights": {s: 1.0 for s in HYBRID_WEIGHT_SOURCES},
    "aggregation": HYBRID_AGGREGATIONS[0],  # "weighted_sum"
    "aggregation_normalization": NORMALIZATION_TYPES[6],  # "none"
    # Phase 1: Standardization
    "standardization": STANDARDIZATION_TYPES[0],  # "none"
    "robust_quantiles": (0.25, 0.75),
    "mmad_scaling_factor": 1.4826,
    # Phase 2: Normalization
    "normalization": NORMALIZATION_TYPES[0],  # "minmax"
    "minmax_range": (0.0, 1.0),
    "sigmoid_scale": 1.0,
    "softmax_temperature": 1.0,
    "rank_method": RANK_METHODS[0],  # "average"
    # Phase 3: Post-processing
    "gamma": 1.0,
}

DEFAULT_EXTREMA_WEIGHT: Final[float] = 1.0


def get_distance(p1: T, p2: T) -> T:
    return abs(p1 - p2)


def midpoint(value1: T, value2: T) -> T:
    """Calculate the midpoint (geometric center) between two values."""
    return (value1 + value2) / 2


def non_zero_diff(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """Returns the difference of two series and replaces zeros with epsilon."""
    diff = s1 - s2
    return diff.where(diff != 0, np.finfo(float).eps)


@lru_cache(maxsize=8)
def get_odd_window(window: int) -> int:
    if window < 1:
        raise ValueError("Window size must be greater than 0")
    return window if window % 2 == 1 else window + 1


@lru_cache(maxsize=8)
def get_gaussian_std(window: int) -> float:
    return (window - 1) / 6.0 if window > 1 else 0.5


@lru_cache(maxsize=8)
def get_savgol_params(
    window: int, polyorder: int, mode: SmoothingMode
) -> tuple[int, int, str]:
    if window <= polyorder:
        window = polyorder + 1
    window = get_odd_window(window)
    return window, polyorder, mode


def nadaraya_watson(
    series: pd.Series, bandwidth: float, mode: SmoothingMode
) -> pd.Series:
    return pd.Series(
        gaussian_filter1d(
            series.to_numpy(),
            sigma=bandwidth,
            mode=mode,  # type: ignore
        ),
        index=series.index,
    )


@lru_cache(maxsize=8)
def _calculate_coeffs(
    window: int,
    win_type: SmoothingKernel,
    std: float,
    beta: float,
) -> NDArray[np.floating]:
    if win_type == SMOOTHING_METHODS[0]:  # "gaussian"
        coeffs = sp.signal.windows.gaussian(M=window, std=std, sym=True)
    elif win_type == SMOOTHING_METHODS[1]:  # "kaiser"
        coeffs = sp.signal.windows.kaiser(M=window, beta=beta, sym=True)
    elif win_type == SMOOTHING_METHODS[2]:  # "triang"
        coeffs = sp.signal.windows.triang(M=window, sym=True)
    else:
        raise ValueError(f"Unknown window type: {win_type}")
    return coeffs / np.sum(coeffs)


def zero_phase_filter(
    series: pd.Series,
    window: int,
    win_type: SmoothingKernel,
    std: float,
    beta: float,
) -> pd.Series:
    if len(series) == 0:
        return series
    if len(series) < window:
        return series
    values = series.to_numpy()
    b = _calculate_coeffs(window=window, win_type=win_type, std=std, beta=beta)
    a = 1.0
    filtered_values = sp.signal.filtfilt(b, a, values)
    return pd.Series(filtered_values, index=series.index)


def smooth_extrema(
    series: pd.Series,
    method: SmoothingMethod = DEFAULTS_EXTREMA_SMOOTHING["method"],
    window: int = DEFAULTS_EXTREMA_SMOOTHING["window"],
    beta: float = DEFAULTS_EXTREMA_SMOOTHING["beta"],
    polyorder: int = DEFAULTS_EXTREMA_SMOOTHING["polyorder"],
    mode: SmoothingMode = DEFAULTS_EXTREMA_SMOOTHING["mode"],
    bandwidth: float = DEFAULTS_EXTREMA_SMOOTHING["bandwidth"],
) -> pd.Series:
    n = len(series)
    if n == 0:
        return series
    if window < 3:
        window = 3
    if n < window:
        return series
    if beta <= 0 or not np.isfinite(beta):
        beta = 1.0

    odd_window = get_odd_window(window)
    std = get_gaussian_std(odd_window)

    if method == SMOOTHING_METHODS[0]:  # "gaussian"
        return zero_phase_filter(
            series=series,
            window=odd_window,
            win_type=SMOOTHING_METHODS[0],
            std=std,
            beta=beta,
        )
    elif method == SMOOTHING_METHODS[1]:  # "kaiser"
        return zero_phase_filter(
            series=series,
            window=odd_window,
            win_type=SMOOTHING_METHODS[1],
            std=std,
            beta=beta,
        )
    elif method == SMOOTHING_METHODS[2]:  # "triang"
        return zero_phase_filter(
            series=series,
            window=odd_window,
            win_type=SMOOTHING_METHODS[2],
            std=std,
            beta=beta,
        )
    elif method == SMOOTHING_METHODS[3]:  # "smm" (Simple Moving Median)
        return series.rolling(window=odd_window, center=True).median()
    elif method == SMOOTHING_METHODS[4]:  # "sma" (Simple Moving Average)
        return series.rolling(window=odd_window, center=True).mean()
    elif method == SMOOTHING_METHODS[5]:  # "savgol" (Savitzky-Golay)
        w, p, m = get_savgol_params(odd_window, polyorder, mode)
        if n < w:
            return series
        return pd.Series(
            sp.signal.savgol_filter(
                series.to_numpy(),
                window_length=w,
                polyorder=p,
                mode=m,  # type: ignore
            ),
            index=series.index,
        )
    elif method == SMOOTHING_METHODS[6]:  # "nadaraya_watson"
        return nadaraya_watson(series, bandwidth, mode)
    else:
        return zero_phase_filter(
            series=series,
            window=odd_window,
            win_type=SMOOTHING_METHODS[0],
            std=std,
            beta=beta,
        )


def _standardize_zscore(weights: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Z-score standardization: (w - μ) / σ
    Returns: mean≈0, std≈1
    """
    if weights.size == 0:
        return weights

    weights = weights.astype(float, copy=False)

    if np.isnan(weights).any():
        return np.zeros_like(weights, dtype=float)

    if weights.size == 1 or np.allclose(weights, weights[0]):
        return np.zeros_like(weights, dtype=float)

    try:
        z_scores = sp.stats.zscore(weights, ddof=1, nan_policy="raise")
    except Exception:
        return np.zeros_like(weights, dtype=float)

    if np.isnan(z_scores).any() or not np.isfinite(z_scores).all():
        return np.zeros_like(weights, dtype=float)

    return z_scores


def _standardize_robust(
    weights: NDArray[np.floating],
    quantiles: tuple[float, float] = DEFAULTS_EXTREMA_WEIGHTING["robust_quantiles"],
) -> NDArray[np.floating]:
    """
    Robust standardization: (w - median) / IQR
    Returns: median≈0, IQR≈1 (outlier-resistant)
    """
    weights = weights.astype(float, copy=False)
    if np.isnan(weights).any():
        return np.zeros_like(weights, dtype=float)

    median = np.nanmedian(weights)
    q1, q3 = np.nanquantile(weights, quantiles)
    iqr = q3 - q1

    if np.isclose(iqr, 0.0):
        return np.zeros_like(weights, dtype=float)

    return (weights - median) / iqr


def _standardize_mmad(
    weights: NDArray[np.floating],
    scaling_factor: float = DEFAULTS_EXTREMA_WEIGHTING["mmad_scaling_factor"],
) -> NDArray[np.floating]:
    """
    MMAD standardization: (w - median) / MAD
    Returns: median≈0, MAD≈1 (outlier-resistant)
    """
    weights = weights.astype(float, copy=False)
    if np.isnan(weights).any():
        return np.zeros_like(weights, dtype=float)

    median = np.nanmedian(weights)
    mad = np.nanmedian(np.abs(weights - median))

    if np.isclose(mad, 0.0):
        return np.zeros_like(weights, dtype=float)

    return (weights - median) / (scaling_factor * mad)


def standardize_weights(
    weights: NDArray[np.floating],
    method: StandardizationType = STANDARDIZATION_TYPES[0],
    robust_quantiles: tuple[float, float] = DEFAULTS_EXTREMA_WEIGHTING[
        "robust_quantiles"
    ],
    mmad_scaling_factor: float = DEFAULTS_EXTREMA_WEIGHTING["mmad_scaling_factor"],
) -> NDArray[np.floating]:
    """
    Phase 1: Standardize weights (centering/scaling, not [0,1] mapping).
    Methods: "none", "zscore", "robust", "mmad"
    """
    if weights.size == 0:
        return weights

    if method == STANDARDIZATION_TYPES[0]:  # "none"
        return weights

    elif method == STANDARDIZATION_TYPES[1]:  # "zscore"
        return _standardize_zscore(weights)

    elif method == STANDARDIZATION_TYPES[2]:  # "robust"
        return _standardize_robust(weights, quantiles=robust_quantiles)

    elif method == STANDARDIZATION_TYPES[3]:  # "mmad"
        return _standardize_mmad(weights, scaling_factor=mmad_scaling_factor)

    else:
        raise ValueError(f"Unknown standardization method: {method}")


def _normalize_sigmoid(
    weights: NDArray[np.floating],
    scale: float = DEFAULTS_EXTREMA_WEIGHTING["sigmoid_scale"],
) -> NDArray[np.floating]:
    """
    Sigmoid normalization: 1 / (1 + exp(-scale × w))
    Returns: [0, 1] with soft compression
    """
    weights = weights.astype(float, copy=False)
    if np.isnan(weights).any():
        return np.full_like(weights, DEFAULT_EXTREMA_WEIGHT, dtype=float)

    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0

    return sp.special.expit(scale * weights)


def _normalize_minmax(
    weights: NDArray[np.floating],
    range: tuple[float, float] = (0.0, 1.0),
) -> NDArray[np.floating]:
    """
    MinMax normalization: range_min + [(w - min) / (max - min)] × (range_max - range_min)
    Returns: [range_min, range_max]
    """
    weights = weights.astype(float, copy=False)
    if np.isnan(weights).any():
        return np.full_like(weights, DEFAULT_EXTREMA_WEIGHT, dtype=float)

    w_min = np.min(weights)
    w_max = np.max(weights)

    if not (np.isfinite(w_min) and np.isfinite(w_max)):
        return np.full_like(weights, DEFAULT_EXTREMA_WEIGHT, dtype=float)

    w_range = w_max - w_min
    if np.isclose(w_range, 0.0):
        return np.full_like(weights, midpoint(range[0], range[1]), dtype=float)

    return range[0] + ((weights - w_min) / w_range) * (range[1] - range[0])


def _normalize_l1(weights: NDArray[np.floating]) -> NDArray[np.floating]:
    """L1 normalization: w / Σ|w|  →  Σ|w| = 1"""
    weights_sum = np.nansum(np.abs(weights))
    if weights_sum <= 0 or not np.isfinite(weights_sum):
        return np.full_like(weights, DEFAULT_EXTREMA_WEIGHT, dtype=float)
    return weights / weights_sum


def _normalize_l2(weights: NDArray[np.floating]) -> NDArray[np.floating]:
    """L2 normalization: w / ||w||₂  →  ||w||₂ = 1"""
    weights = weights.astype(float, copy=False)
    if np.isnan(weights).any():
        return np.full_like(weights, DEFAULT_EXTREMA_WEIGHT, dtype=float)

    l2_norm = np.linalg.norm(weights, ord=2)

    if l2_norm <= 0 or not np.isfinite(l2_norm):
        return np.full_like(weights, DEFAULT_EXTREMA_WEIGHT, dtype=float)

    return weights / l2_norm


def _normalize_softmax(
    weights: NDArray[np.floating],
    temperature: float = DEFAULTS_EXTREMA_WEIGHTING["softmax_temperature"],
) -> NDArray[np.floating]:
    """Softmax normalization: exp(w/T) / Σexp(w/T)  →  Σw = 1, range [0,1]"""
    weights = weights.astype(float, copy=False)
    if np.isnan(weights).any():
        return np.full_like(weights, DEFAULT_EXTREMA_WEIGHT, dtype=float)
    if not np.isclose(temperature, 1.0) and temperature > 0:
        weights = weights / temperature
    return sp.special.softmax(weights)


def _normalize_rank(
    weights: NDArray[np.floating],
    method: RankMethod = DEFAULTS_EXTREMA_WEIGHTING["rank_method"],
) -> NDArray[np.floating]:
    """Rank normalization: [rank(w) - 1] / (n - 1)  →  [0, 1] uniformly distributed"""
    weights = weights.astype(float, copy=False)
    if np.isnan(weights).any():
        return np.full_like(weights, DEFAULT_EXTREMA_WEIGHT, dtype=float)

    ranks = sp.stats.rankdata(weights, method=method)
    n = len(weights)
    if n <= 1:
        return np.full_like(weights, DEFAULT_EXTREMA_WEIGHT, dtype=float)

    return (ranks - 1) / (n - 1)


def _impute_weights(
    weights: NDArray[np.floating],
    *,
    finite_mask: NDArray[np.bool_] | None = None,
    default_weight: float = DEFAULT_EXTREMA_WEIGHT,
) -> NDArray[np.floating]:
    weights = weights.astype(float, copy=False)
    if finite_mask is None:
        finite_mask = np.isfinite(weights)

    if not finite_mask.any():
        return np.full_like(weights, default_weight, dtype=float)

    median_weight = np.nanmedian(weights[finite_mask])
    if not np.isfinite(median_weight):
        median_weight = default_weight

    weights_out = weights.astype(float, copy=True)
    weights_out[~finite_mask] = median_weight
    return weights_out


def normalize_weights(
    weights: NDArray[np.floating],
    # Phase 1: Standardization
    standardization: StandardizationType = DEFAULTS_EXTREMA_WEIGHTING[
        "standardization"
    ],
    robust_quantiles: tuple[float, float] = DEFAULTS_EXTREMA_WEIGHTING[
        "robust_quantiles"
    ],
    mmad_scaling_factor: float = DEFAULTS_EXTREMA_WEIGHTING["mmad_scaling_factor"],
    # Phase 2: Normalization
    normalization: NormalizationType = DEFAULTS_EXTREMA_WEIGHTING["normalization"],
    minmax_range: tuple[float, float] = DEFAULTS_EXTREMA_WEIGHTING["minmax_range"],
    sigmoid_scale: float = DEFAULTS_EXTREMA_WEIGHTING["sigmoid_scale"],
    softmax_temperature: float = DEFAULTS_EXTREMA_WEIGHTING["softmax_temperature"],
    rank_method: RankMethod = DEFAULTS_EXTREMA_WEIGHTING["rank_method"],
    # Phase 3: Post-processing
    gamma: float = DEFAULTS_EXTREMA_WEIGHTING["gamma"],
) -> NDArray[np.floating]:
    """
    3-phase weight normalization:
    1. Standardization: zscore (w-μ)/σ | robust (w-median)/IQR | mmad (w-median)/MAD | none
    2. Normalization: minmax, sigmoid, softmax, l1, l2, rank, none
    3. Post-processing: gamma correction w^γ
    """
    if weights.size == 0:
        return weights

    weights_finite_mask = np.isfinite(weights)
    if not weights_finite_mask.any():
        return np.full_like(weights, DEFAULT_EXTREMA_WEIGHT, dtype=float)

    weights = _impute_weights(
        weights,
        finite_mask=weights_finite_mask,
        default_weight=DEFAULT_EXTREMA_WEIGHT,
    )

    # Phase 1: Standardization
    standardized_weights = standardize_weights(
        weights,
        method=standardization,
        robust_quantiles=robust_quantiles,
        mmad_scaling_factor=mmad_scaling_factor,
    )

    # Phase 2: Normalization
    if normalization == NORMALIZATION_TYPES[6]:  # "none"
        normalized_weights = standardized_weights
    elif normalization == NORMALIZATION_TYPES[0]:  # "minmax"
        normalized_weights = _normalize_minmax(standardized_weights, range=minmax_range)
    elif normalization == NORMALIZATION_TYPES[1]:  # "sigmoid"
        normalized_weights = _normalize_sigmoid(
            standardized_weights, scale=sigmoid_scale
        )
    elif normalization == NORMALIZATION_TYPES[2]:  # "softmax"
        normalized_weights = _normalize_softmax(
            standardized_weights, temperature=softmax_temperature
        )
    elif normalization == NORMALIZATION_TYPES[3]:  # "l1"
        normalized_weights = _normalize_l1(standardized_weights)
    elif normalization == NORMALIZATION_TYPES[4]:  # "l2"
        normalized_weights = _normalize_l2(standardized_weights)
    elif normalization == NORMALIZATION_TYPES[5]:  # "rank"
        normalized_weights = _normalize_rank(standardized_weights, method=rank_method)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    # Phase 3: Post-processing
    if not np.isclose(gamma, 1.0) and np.isfinite(gamma) and gamma > 0:
        normalized_weights = np.power(np.abs(normalized_weights), gamma) * np.sign(
            normalized_weights
        )

    if not np.isfinite(normalized_weights).all():
        return np.full_like(weights, DEFAULT_EXTREMA_WEIGHT, dtype=float)

    return normalized_weights


def _weights_array_to_series(
    index: pd.Index,
    indices: list[int],
    weights: NDArray[np.floating],
    default_weight: float = DEFAULT_EXTREMA_WEIGHT,
) -> pd.Series:
    weights_series = pd.Series(default_weight, index=index)

    if len(indices) == 0 or weights.size == 0:
        return weights_series

    if len(indices) != weights.size:
        raise ValueError(
            f"Length mismatch: {len(indices)} indices but {weights.size} weights"
        )

    mask = pd.Index(indices).isin(index)
    if not np.any(mask):
        return weights_series

    valid_indices = [idx for idx, is_valid in zip(indices, mask) if is_valid]
    weights_series.loc[valid_indices] = weights[mask]
    return weights_series


def calculate_hybrid_extrema_weights(
    extrema: pd.Series,
    indices: list[int],
    amplitudes: list[float],
    amplitude_threshold_ratios: list[float],
    volumes: list[float],
    speeds: list[float],
    efficiency_ratios: list[float],
    source_weights: dict[str, float],
    aggregation: HybridAggregation = DEFAULTS_EXTREMA_WEIGHTING["aggregation"],
    aggregation_normalization: NormalizationType = DEFAULTS_EXTREMA_WEIGHTING[
        "aggregation_normalization"
    ],
    # Phase 1: Standardization
    standardization: StandardizationType = DEFAULTS_EXTREMA_WEIGHTING[
        "standardization"
    ],
    robust_quantiles: tuple[float, float] = DEFAULTS_EXTREMA_WEIGHTING[
        "robust_quantiles"
    ],
    mmad_scaling_factor: float = DEFAULTS_EXTREMA_WEIGHTING["mmad_scaling_factor"],
    # Phase 2: Normalization
    normalization: NormalizationType = DEFAULTS_EXTREMA_WEIGHTING["normalization"],
    minmax_range: tuple[float, float] = DEFAULTS_EXTREMA_WEIGHTING["minmax_range"],
    sigmoid_scale: float = DEFAULTS_EXTREMA_WEIGHTING["sigmoid_scale"],
    softmax_temperature: float = DEFAULTS_EXTREMA_WEIGHTING["softmax_temperature"],
    rank_method: RankMethod = DEFAULTS_EXTREMA_WEIGHTING["rank_method"],
    # Phase 3: Post-processing
    gamma: float = DEFAULTS_EXTREMA_WEIGHTING["gamma"],
) -> pd.Series:
    n = len(indices)
    if n == 0:
        return pd.Series(DEFAULT_EXTREMA_WEIGHT, index=extrema.index)

    if not isinstance(source_weights, dict):
        source_weights = {}

    weights_by_source: dict[HybridWeightSource, NDArray[np.floating]] = {
        "amplitude": np.asarray(amplitudes, dtype=float),
        "amplitude_threshold_ratio": np.asarray(
            amplitude_threshold_ratios, dtype=float
        ),
        "volume": np.asarray(volumes, dtype=float),
        "speed": np.asarray(speeds, dtype=float),
        "efficiency_ratio": np.asarray(efficiency_ratios, dtype=float),
    }

    enabled_sources: list[HybridWeightSource] = []
    source_weights_values: list[float] = []
    for source in HYBRID_WEIGHT_SOURCES:
        source_weight = source_weights.get(source)
        if source_weight is None:
            continue
        if (
            not isinstance(source_weight, (int, float))
            or not np.isfinite(source_weight)
            or source_weight < 0
        ):
            continue
        enabled_sources.append(source)
        source_weights_values.append(float(source_weight))

    if len(enabled_sources) == 0:
        enabled_sources = list(HYBRID_WEIGHT_SOURCES)
        source_weights_values = [1.0 for _ in enabled_sources]

    if any(weights_by_source[s].size != n for s in enabled_sources):
        raise ValueError(
            f"Length mismatch: hybrid {n} indices but inconsistent weights lengths"
        )

    np_source_weights: NDArray[np.floating] = np.asarray(
        source_weights_values, dtype=float
    )
    source_weights_sum = np.nansum(np.abs(np_source_weights))
    if not np.isfinite(source_weights_sum) or source_weights_sum <= 0:
        return pd.Series(DEFAULT_EXTREMA_WEIGHT, index=extrema.index)
    np_source_weights = np_source_weights / source_weights_sum

    normalized_source_weights: list[NDArray[np.floating]] = []
    for source in enabled_sources:
        normalized_source_weights.append(
            normalize_weights(
                weights_by_source[source],
                standardization=standardization,
                robust_quantiles=robust_quantiles,
                mmad_scaling_factor=mmad_scaling_factor,
                normalization=normalization,
                minmax_range=minmax_range,
                sigmoid_scale=sigmoid_scale,
                softmax_temperature=softmax_temperature,
                rank_method=rank_method,
                gamma=gamma,
            )
        )

    if aggregation == HYBRID_AGGREGATIONS[0]:  # "weighted_sum"
        combined_source_weights = np.zeros(n, dtype=float)
        for source_weight, values in zip(np_source_weights, normalized_source_weights):
            combined_source_weights = combined_source_weights + source_weight * values
    elif aggregation == HYBRID_AGGREGATIONS[1]:  # "geometric_mean"
        combined_source_weights = sp.stats.gmean(
            np.vstack([np.abs(values) for values in normalized_source_weights]),
            axis=0,
            weights=np_source_weights,
        )
    else:
        raise ValueError(f"Unknown hybrid aggregation method: {aggregation}")

    if aggregation_normalization != NORMALIZATION_TYPES[6]:  # "none"
        combined_source_weights = normalize_weights(
            combined_source_weights,
            standardization=STANDARDIZATION_TYPES[0],
            robust_quantiles=robust_quantiles,
            mmad_scaling_factor=mmad_scaling_factor,
            normalization=aggregation_normalization,
            minmax_range=minmax_range,
            sigmoid_scale=sigmoid_scale,
            softmax_temperature=softmax_temperature,
            rank_method=rank_method,
            gamma=1.0,
        )

    if (
        combined_source_weights.size == 0
        or not np.isfinite(combined_source_weights).all()
    ):
        return pd.Series(DEFAULT_EXTREMA_WEIGHT, index=extrema.index)

    return _weights_array_to_series(
        index=extrema.index,
        indices=indices,
        weights=combined_source_weights,
        default_weight=np.nanmedian(combined_source_weights),
    )


def calculate_extrema_weights(
    extrema: pd.Series,
    indices: list[int],
    weights: NDArray[np.floating],
    # Phase 1: Standardization
    standardization: StandardizationType = DEFAULTS_EXTREMA_WEIGHTING[
        "standardization"
    ],
    robust_quantiles: tuple[float, float] = DEFAULTS_EXTREMA_WEIGHTING[
        "robust_quantiles"
    ],
    mmad_scaling_factor: float = DEFAULTS_EXTREMA_WEIGHTING["mmad_scaling_factor"],
    # Phase 2: Normalization
    normalization: NormalizationType = DEFAULTS_EXTREMA_WEIGHTING["normalization"],
    minmax_range: tuple[float, float] = DEFAULTS_EXTREMA_WEIGHTING["minmax_range"],
    sigmoid_scale: float = DEFAULTS_EXTREMA_WEIGHTING["sigmoid_scale"],
    softmax_temperature: float = DEFAULTS_EXTREMA_WEIGHTING["softmax_temperature"],
    rank_method: RankMethod = DEFAULTS_EXTREMA_WEIGHTING["rank_method"],
    # Phase 3: Post-processing
    gamma: float = DEFAULTS_EXTREMA_WEIGHTING["gamma"],
) -> pd.Series:
    """
    Calculate normalized weights for extrema points.
    Returns: Series with weights at extrema indices (rest filled with default).
    """
    if len(indices) == 0 or len(weights) == 0:
        return pd.Series(DEFAULT_EXTREMA_WEIGHT, index=extrema.index)

    if len(indices) != len(weights):
        raise ValueError(
            f"Length mismatch: {len(indices)} indices but {len(weights)} weights"
        )

    normalized_weights = normalize_weights(
        weights,
        standardization=standardization,
        robust_quantiles=robust_quantiles,
        mmad_scaling_factor=mmad_scaling_factor,
        normalization=normalization,
        minmax_range=minmax_range,
        sigmoid_scale=sigmoid_scale,
        softmax_temperature=softmax_temperature,
        rank_method=rank_method,
        gamma=gamma,
    )

    if normalized_weights.size == 0 or np.allclose(
        normalized_weights, normalized_weights[0]
    ):
        normalized_weights = np.full_like(normalized_weights, DEFAULT_EXTREMA_WEIGHT)

    return _weights_array_to_series(
        index=extrema.index,
        indices=indices,
        weights=normalized_weights,
        default_weight=np.nanmedian(normalized_weights),
    )


def compute_extrema_weights(
    extrema: pd.Series,
    indices: list[int],
    amplitudes: list[float],
    amplitude_threshold_ratios: list[float],
    volumes: list[float],
    speeds: list[float],
    efficiency_ratios: list[float],
    source_weights: dict[str, float],
    strategy: WeightStrategy = DEFAULTS_EXTREMA_WEIGHTING["strategy"],
    aggregation: HybridAggregation = DEFAULTS_EXTREMA_WEIGHTING["aggregation"],
    aggregation_normalization: NormalizationType = DEFAULTS_EXTREMA_WEIGHTING[
        "aggregation_normalization"
    ],
    # Phase 1: Standardization
    standardization: StandardizationType = DEFAULTS_EXTREMA_WEIGHTING[
        "standardization"
    ],
    robust_quantiles: tuple[float, float] = DEFAULTS_EXTREMA_WEIGHTING[
        "robust_quantiles"
    ],
    mmad_scaling_factor: float = DEFAULTS_EXTREMA_WEIGHTING["mmad_scaling_factor"],
    # Phase 2: Normalization
    normalization: NormalizationType = DEFAULTS_EXTREMA_WEIGHTING["normalization"],
    minmax_range: tuple[float, float] = DEFAULTS_EXTREMA_WEIGHTING["minmax_range"],
    sigmoid_scale: float = DEFAULTS_EXTREMA_WEIGHTING["sigmoid_scale"],
    softmax_temperature: float = DEFAULTS_EXTREMA_WEIGHTING["softmax_temperature"],
    rank_method: RankMethod = DEFAULTS_EXTREMA_WEIGHTING["rank_method"],
    # Phase 3: Post-processing
    gamma: float = DEFAULTS_EXTREMA_WEIGHTING["gamma"],
) -> pd.Series:
    if len(indices) == 0 or strategy == WEIGHT_STRATEGIES[0]:  # "none"
        return pd.Series(DEFAULT_EXTREMA_WEIGHT, index=extrema.index)

    if (
        strategy
        in {
            WEIGHT_STRATEGIES[1],
            WEIGHT_STRATEGIES[2],
            WEIGHT_STRATEGIES[3],
            WEIGHT_STRATEGIES[4],
            WEIGHT_STRATEGIES[5],
        }
    ):  # "amplitude" / "amplitude_threshold_ratio" / "volume" / "speed" / "efficiency_ratio"
        if strategy == WEIGHT_STRATEGIES[1]:  # "amplitude"
            weights = np.asarray(amplitudes, dtype=float)
        elif strategy == WEIGHT_STRATEGIES[2]:  # "amplitude_threshold_ratio"
            weights = np.asarray(amplitude_threshold_ratios, dtype=float)
        elif strategy == WEIGHT_STRATEGIES[3]:  # "volume"
            weights = np.asarray(volumes, dtype=float)
        elif strategy == WEIGHT_STRATEGIES[4]:  # "speed"
            weights = np.asarray(speeds, dtype=float)
        elif strategy == WEIGHT_STRATEGIES[5]:  # "efficiency_ratio"
            weights = np.asarray(efficiency_ratios, dtype=float)
        else:
            weights = np.asarray([], dtype=float)

        if weights.size == 0:
            return pd.Series(DEFAULT_EXTREMA_WEIGHT, index=extrema.index)

        return calculate_extrema_weights(
            extrema=extrema,
            indices=indices,
            weights=weights,
            standardization=standardization,
            robust_quantiles=robust_quantiles,
            mmad_scaling_factor=mmad_scaling_factor,
            normalization=normalization,
            minmax_range=minmax_range,
            sigmoid_scale=sigmoid_scale,
            softmax_temperature=softmax_temperature,
            rank_method=rank_method,
            gamma=gamma,
        )

    if strategy == WEIGHT_STRATEGIES[6]:  # "hybrid"
        return calculate_hybrid_extrema_weights(
            extrema=extrema,
            indices=indices,
            amplitudes=amplitudes,
            amplitude_threshold_ratios=amplitude_threshold_ratios,
            volumes=volumes,
            speeds=speeds,
            efficiency_ratios=efficiency_ratios,
            source_weights=source_weights,
            aggregation=aggregation,
            aggregation_normalization=aggregation_normalization,
            standardization=standardization,
            robust_quantiles=robust_quantiles,
            mmad_scaling_factor=mmad_scaling_factor,
            normalization=normalization,
            minmax_range=minmax_range,
            sigmoid_scale=sigmoid_scale,
            softmax_temperature=softmax_temperature,
            rank_method=rank_method,
            gamma=gamma,
        )

    raise ValueError(f"Unknown extrema weighting strategy: {strategy}")


def _apply_weights(extrema: pd.Series, weights: pd.Series) -> pd.Series:
    if weights.empty:
        return extrema
    if np.allclose(weights.to_numpy(dtype=float), DEFAULT_EXTREMA_WEIGHT):
        return extrema
    return extrema * weights


def get_weighted_extrema(
    extrema: pd.Series,
    indices: list[int],
    amplitudes: list[float],
    amplitude_threshold_ratios: list[float],
    volumes: list[float],
    speeds: list[float],
    efficiency_ratios: list[float],
    source_weights: dict[str, float],
    strategy: WeightStrategy = DEFAULTS_EXTREMA_WEIGHTING["strategy"],
    aggregation: HybridAggregation = DEFAULTS_EXTREMA_WEIGHTING["aggregation"],
    aggregation_normalization: NormalizationType = DEFAULTS_EXTREMA_WEIGHTING[
        "aggregation_normalization"
    ],
    # Phase 1: Standardization
    standardization: StandardizationType = DEFAULTS_EXTREMA_WEIGHTING[
        "standardization"
    ],
    robust_quantiles: tuple[float, float] = DEFAULTS_EXTREMA_WEIGHTING[
        "robust_quantiles"
    ],
    mmad_scaling_factor: float = DEFAULTS_EXTREMA_WEIGHTING["mmad_scaling_factor"],
    # Phase 2: Normalization
    normalization: NormalizationType = DEFAULTS_EXTREMA_WEIGHTING["normalization"],
    minmax_range: tuple[float, float] = DEFAULTS_EXTREMA_WEIGHTING["minmax_range"],
    sigmoid_scale: float = DEFAULTS_EXTREMA_WEIGHTING["sigmoid_scale"],
    softmax_temperature: float = DEFAULTS_EXTREMA_WEIGHTING["softmax_temperature"],
    rank_method: RankMethod = DEFAULTS_EXTREMA_WEIGHTING["rank_method"],
    # Phase 3: Post-processing
    gamma: float = DEFAULTS_EXTREMA_WEIGHTING["gamma"],
) -> tuple[pd.Series, pd.Series]:
    weights = compute_extrema_weights(
        extrema=extrema,
        indices=indices,
        amplitudes=amplitudes,
        amplitude_threshold_ratios=amplitude_threshold_ratios,
        volumes=volumes,
        speeds=speeds,
        efficiency_ratios=efficiency_ratios,
        source_weights=source_weights,
        strategy=strategy,
        aggregation=aggregation,
        aggregation_normalization=aggregation_normalization,
        standardization=standardization,
        robust_quantiles=robust_quantiles,
        mmad_scaling_factor=mmad_scaling_factor,
        normalization=normalization,
        minmax_range=minmax_range,
        sigmoid_scale=sigmoid_scale,
        softmax_temperature=softmax_temperature,
        rank_method=rank_method,
        gamma=gamma,
    )

    weighted_extrema = _apply_weights(extrema, weights)
    return weighted_extrema, weights


def get_callable_sha256(fn: Callable[..., Any]) -> str:
    if not callable(fn):
        raise ValueError("fn must be callable")
    code = getattr(fn, "__code__", None)
    if code is None and isinstance(fn, functools.partial):
        fn = fn.func
        code = getattr(fn, "__code__", None)
        if code is None and hasattr(fn, "__func__"):
            code = getattr(fn.__func__, "__code__", None)
    if code is None and hasattr(fn, "__func__"):
        code = getattr(fn.__func__, "__code__", None)
    if code is None and hasattr(fn, "__call__"):
        code = getattr(fn.__call__, "__code__", None)
    if code is None:
        raise ValueError("Unable to retrieve code object from fn")
    return hashlib.sha256(code.co_code).hexdigest()


@lru_cache(maxsize=128)
def format_number(value: int | float, significant_digits: int = 5) -> str:
    if not isinstance(value, (int, float)):
        return str(value)

    if np.isposinf(value):
        return "+∞"
    if np.isneginf(value):
        return "-∞"
    if np.isnan(value):
        return "NaN"

    if value == int(value):
        return str(int(value))

    abs_value = abs(value)

    if abs_value >= 1.0:
        precision = significant_digits
    else:
        if abs_value == 0:
            return "0"
        order_of_magnitude = math.floor(math.log10(abs_value))
        leading_zeros = abs(order_of_magnitude) - 1
        precision = leading_zeros + significant_digits
    precision = max(0, int(precision))

    formatted_value = f"{value:.{precision}f}"

    if "." in formatted_value:
        formatted_value = formatted_value.rstrip("0").rstrip(".")

    return formatted_value


@lru_cache(maxsize=128)
def calculate_min_extrema(
    length: int, fit_live_predictions_candles: int, min_extrema: int = 2
) -> int:
    return int(round(length / fit_live_predictions_candles) * min_extrema)


def calculate_n_extrema(series: pd.Series) -> int:
    return sp.signal.find_peaks(-series)[0].size + sp.signal.find_peaks(series)[0].size


def top_change_percent(dataframe: pd.DataFrame, period: int) -> pd.Series:
    """
    Percentage change of the current close relative to the top close price in the previous `period` bars.

    :param dataframe: OHLCV DataFrame
    :param period: The previous period window size to look back (>=1)
    :return: The top change percentage series
    """
    if period < 1:
        raise ValueError("period must be greater than or equal to 1")

    previous_close_top = (
        dataframe.get("close").rolling(period, min_periods=period).max().shift(1)
    )

    return (dataframe.get("close") - previous_close_top) / previous_close_top


def bottom_change_percent(dataframe: pd.DataFrame, period: int) -> pd.Series:
    """
    Percentage change of the current close relative to the bottom close price in the previous `period` bars.

    :param dataframe: OHLCV DataFrame
    :param period: The previous period window size to look back (>=1)
    :return: The bottom change percentage series
    """
    if period < 1:
        raise ValueError("period must be greater than or equal to 1")

    previous_close_bottom = (
        dataframe.get("close").rolling(period, min_periods=period).min().shift(1)
    )

    return (dataframe.get("close") - previous_close_bottom) / previous_close_bottom


def price_retracement_percent(dataframe: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate the percentage retracement of the current close within the high/low close price range
    of the previous `period` bars.

    :param dataframe: OHLCV DataFrame
    :param period: Window size for calculating historical closes high/low (>=1)
    :return: Retracement percentage series
    """
    if period < 1:
        raise ValueError("period must be greater than or equal to 1")

    previous_close_low = (
        dataframe.get("close").rolling(period, min_periods=period).min().shift(1)
    )
    previous_close_high = (
        dataframe.get("close").rolling(period, min_periods=period).max().shift(1)
    )

    return (dataframe.get("close") - previous_close_low) / (
        non_zero_diff(previous_close_high, previous_close_low)
    )


# VWAP bands
def vwapb(
    dataframe: pd.DataFrame, window: int = 20, std_factor: float = 1.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    vwap = qtpylib.rolling_vwap(dataframe, window=window)
    rolling_std = vwap.rolling(window=window, min_periods=window).std(ddof=1)
    vwap_low = vwap - (rolling_std * std_factor)
    vwap_high = vwap + (rolling_std * std_factor)
    return vwap_low, vwap, vwap_high


def calculate_zero_lag(series: pd.Series, period: int) -> pd.Series:
    """Applies a zero lag filter to reduce MA lag."""
    lag = max((period - 1) / 2, 0)
    if lag == 0:
        return series
    return 2 * series - series.shift(int(lag))


@lru_cache(maxsize=8)
def get_ma_fn(
    mamode: str,
) -> Callable[
    [pd.Series | NDArray[np.floating], int], pd.Series | NDArray[np.floating]
]:
    mamodes: dict[
        str,
        Callable[
            [pd.Series | NDArray[np.floating], int], pd.Series | NDArray[np.floating]
        ],
    ] = {
        "sma": ta.SMA,
        "ema": ta.EMA,
        "wma": ta.WMA,
        "dema": ta.DEMA,
        "tema": ta.TEMA,
        "trima": ta.TRIMA,
        "kama": ta.KAMA,
        "t3": ta.T3,
    }
    return mamodes.get(mamode, mamodes["sma"])


@lru_cache(maxsize=8)
def get_zl_ma_fn(
    mamode: str,
) -> Callable[
    [pd.Series | NDArray[np.floating], int], pd.Series | NDArray[np.floating]
]:
    ma_fn = get_ma_fn(mamode)
    return lambda series, timeperiod: ma_fn(
        calculate_zero_lag(series, timeperiod), timeperiod=timeperiod
    )


def zlema(series: pd.Series, period: int) -> pd.Series:
    """Ehlers' Zero Lag EMA."""
    lag = max((period - 1) / 2, 0)
    alpha = 2 / (period + 1)
    zl_series = 2 * series - series.shift(int(lag))
    return zl_series.ewm(alpha=alpha, adjust=False).mean()


def _fractal_dimension(
    highs: NDArray[np.floating], lows: NDArray[np.floating], period: int
) -> float:
    """Original fractal dimension computation implementation per Ehlers' paper."""
    if period % 2 != 0:
        raise ValueError("period must be even")

    half_period = period // 2

    H1 = np.max(highs[:half_period])
    L1 = np.min(lows[:half_period])

    H2 = np.max(highs[half_period:])
    L2 = np.min(lows[half_period:])

    H3 = np.max(highs)
    L3 = np.min(lows)

    HL1 = H1 - L1
    HL2 = H2 - L2
    HL3 = H3 - L3

    if (HL1 + HL2) == 0 or HL3 == 0:
        return 1.0

    D = (np.log(HL1 + HL2) - np.log(HL3)) / np.log(2)
    return np.clip(D, 1.0, 2.0)


def frama(df: pd.DataFrame, period: int = 16, zero_lag: bool = False) -> pd.Series:
    """
    Original FRAMA implementation per Ehlers' paper with optional zero lag.
    """
    if period % 2 != 0:
        raise ValueError("period must be even")

    n = len(df)

    highs = df.get("high")
    lows = df.get("low")
    closes = df.get("close")

    if zero_lag:
        highs = calculate_zero_lag(highs, period=period)
        lows = calculate_zero_lag(lows, period=period)
        closes = calculate_zero_lag(closes, period=period)

    fd = pd.Series(np.nan, index=closes.index)
    for i in range(period, n):
        window_highs = highs.iloc[i - period : i]
        window_lows = lows.iloc[i - period : i]
        fd.iloc[i] = _fractal_dimension(
            window_highs.to_numpy(), window_lows.to_numpy(), period
        )

    alpha = np.exp(-4.6 * (fd - 1)).clip(0.01, 1)

    frama = pd.Series(np.nan, index=closes.index)
    frama.iloc[period - 1] = closes.iloc[:period].mean()
    for i in range(period, n):
        if pd.isna(frama.iloc[i - 1]) or pd.isna(alpha.iloc[i]):
            continue
        frama.iloc[i] = (
            alpha.iloc[i] * closes.iloc[i] + (1 - alpha.iloc[i]) * frama.iloc[i - 1]
        )

    return frama


def smma(series: pd.Series, period: int, zero_lag=False, offset=0) -> pd.Series:
    """
    SMoothed Moving Average (SMMA).

    https://www.sierrachart.com/index.php?page=doc/StudiesReference.php&ID=173&Name=Moving_Average_-_Smoothed
    """
    if period <= 0:
        raise ValueError("period must be greater than 0")
    n = len(series)
    if n < period:
        return pd.Series(index=series.index, dtype=float)

    if zero_lag:
        series = calculate_zero_lag(series, period=period)

    values = series.to_numpy()

    smma_values = np.full(n, np.nan)
    smma_values[period - 1] = np.nanmean(values[:period])
    for i in range(period, n):
        smma_values[i] = (smma_values[i - 1] * (period - 1) + values[i]) / period

    smma = pd.Series(smma_values, index=series.index)

    if offset != 0:
        smma = smma.shift(offset)

    return smma


@lru_cache(maxsize=8)
def get_price_fn(pricemode: str) -> Callable[[pd.DataFrame], pd.Series]:
    pricemodes = {
        "average": ta.AVGPRICE,
        "median": ta.MEDPRICE,
        "typical": ta.TYPPRICE,
        "weighted-close": ta.WCLPRICE,
        "close": lambda df: df.get("close"),
    }
    return pricemodes.get(pricemode, pricemodes["close"])


def ewo(
    dataframe: pd.DataFrame,
    ma1_length: int = 5,
    ma2_length: int = 34,
    pricemode: str = "close",
    mamode: str = "sma",
    zero_lag: bool = False,
    normalize: bool = False,
) -> pd.Series:
    """
    Calculate the Elliott Wave Oscillator (EWO) using two moving averages.
    """
    prices = get_price_fn(pricemode)(dataframe)

    if zero_lag:
        if mamode == "ema":

            def ma_fn(series, timeperiod):
                return zlema(series, period=timeperiod)
        else:
            ma_fn = get_zl_ma_fn(mamode)
    else:
        ma_fn = get_ma_fn(mamode)

    ma1 = ma_fn(prices, timeperiod=ma1_length)
    ma2 = ma_fn(prices, timeperiod=ma2_length)
    madiff = ma1 - ma2
    if normalize:
        madiff = (madiff / prices) * 100.0
    return madiff


def alligator(
    df: pd.DataFrame,
    jaw_period: int = 13,
    teeth_period: int = 8,
    lips_period: int = 5,
    jaw_shift: int = 8,
    teeth_shift: int = 5,
    lips_shift: int = 3,
    pricemode: str = "median",
    zero_lag: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bill Williams' Alligator indicator lines.
    """
    prices = get_price_fn(pricemode)(df)

    jaw = smma(prices, period=jaw_period, zero_lag=zero_lag, offset=jaw_shift)
    teeth = smma(prices, period=teeth_period, zero_lag=zero_lag, offset=teeth_shift)
    lips = smma(prices, period=lips_period, zero_lag=zero_lag, offset=lips_shift)

    return jaw, teeth, lips


def find_fractals(df: pd.DataFrame, period: int = 2) -> tuple[list[int], list[int]]:
    n = len(df)
    if n < 2 * period + 1:
        return [], []

    highs = df.get("high").to_numpy()
    lows = df.get("low").to_numpy()

    indices = df.index.tolist()

    fractal_highs = []
    fractal_lows = []

    for i in range(period, n - period):
        is_high_fractal = all(
            highs[i] > highs[i - j] and highs[i] > highs[i + j]
            for j in range(1, period + 1)
        )
        is_low_fractal = all(
            lows[i] < lows[i - j] and lows[i] < lows[i + j]
            for j in range(1, period + 1)
        )

        if is_high_fractal:
            fractal_highs.append(indices[i])
        if is_low_fractal:
            fractal_lows.append(indices[i])

    return fractal_highs, fractal_lows


def calculate_quantile(values: NDArray[np.floating], value: float) -> float:
    if values.size == 0:
        return np.nan

    first_value = values[0]
    if np.allclose(values, first_value):
        return (
            0.5
            if np.isclose(value, first_value)
            else (0.0 if value < first_value else 1.0)
        )

    return np.sum(values <= value) / values.size


class TrendDirection(IntEnum):
    NEUTRAL = 0
    UP = 1
    DOWN = -1


def zigzag(
    df: pd.DataFrame,
    natr_period: int = 14,
    natr_ratio: float = 9.0,
) -> tuple[
    list[int],
    list[float],
    list[TrendDirection],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    n = len(df)
    if df.empty or n < natr_period:
        return (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    natr_values = (ta.NATR(df, timeperiod=natr_period).bfill() / 100.0).to_numpy()

    indices: list[int] = df.index.tolist()
    thresholds: NDArray[np.floating] = natr_values * natr_ratio
    closes = df.get("close").to_numpy()
    log_closes = np.log(closes)
    highs = df.get("high").to_numpy()
    lows = df.get("low").to_numpy()
    volumes = df.get("volume").to_numpy()

    state: TrendDirection = TrendDirection.NEUTRAL

    pivots_indices: list[int] = []
    pivots_values: list[float] = []
    pivots_directions: list[TrendDirection] = []
    pivots_amplitudes: list[float] = []
    pivots_amplitude_threshold_ratios: list[float] = []
    pivots_volumes: list[float] = []
    pivots_durations: list[float] = []
    pivots_speeds: list[float] = []
    pivots_efficiency_ratios: list[float] = []
    last_pivot_pos: int = -1

    candidate_pivot_pos: int = -1
    candidate_pivot_value: float = np.nan

    volatility_quantile_cache: dict[int, float] = {}

    def calculate_volatility_quantile(pos: int) -> float:
        if pos not in volatility_quantile_cache:
            pos_plus_1 = pos + 1
            start_pos = max(0, pos_plus_1 - natr_period)
            end_pos = min(pos_plus_1, n)
            if start_pos >= end_pos:
                volatility_quantile_cache[pos] = np.nan
            else:
                volatility_quantile_cache[pos] = calculate_quantile(
                    natr_values[start_pos:end_pos], natr_values[pos]
                )

        return volatility_quantile_cache[pos]

    def calculate_slopes_ok_threshold(
        pos: int,
        min_threshold: float = 0.75,
        max_threshold: float = 0.95,
    ) -> float:
        volatility_quantile = calculate_volatility_quantile(pos)
        if np.isnan(volatility_quantile):
            return midpoint(min_threshold, max_threshold)

        return max_threshold - (max_threshold - min_threshold) * volatility_quantile

    def update_candidate_pivot(pos: int, value: float):
        nonlocal candidate_pivot_pos, candidate_pivot_value
        if 0 <= pos < n:
            candidate_pivot_pos = pos
            candidate_pivot_value = value

    def reset_candidate_pivot():
        nonlocal candidate_pivot_pos, candidate_pivot_value
        candidate_pivot_pos = -1
        candidate_pivot_value = np.nan

    def calculate_pivot_amplitude_and_threshold_ratio(
        *,
        previous_pos: int,
        previous_value: float,
        current_pos: int,
        current_value: float,
    ) -> tuple[float, float]:
        if previous_pos < 0 or current_pos < 0:
            return np.nan, np.nan
        if previous_pos >= n or current_pos >= n:
            return np.nan, np.nan

        if np.isclose(previous_value, 0.0):
            return np.nan, np.nan

        amplitude = abs(current_value - previous_value) / abs(previous_value)

        start_pos = min(previous_pos, current_pos)
        end_pos = max(previous_pos, current_pos) + 1
        median_threshold = np.nanmedian(thresholds[start_pos:end_pos])

        if (
            np.isfinite(median_threshold)
            and median_threshold > 0
            and np.isfinite(amplitude)
        ):
            amplitude_threshold_ratio = amplitude / median_threshold
        else:
            amplitude_threshold_ratio = np.nan

        return amplitude, amplitude_threshold_ratio

    def calculate_pivot_volume(
        *,
        previous_pos: int,
        current_pos: int,
    ) -> float:
        if previous_pos < 0 or current_pos < 0:
            return np.nan
        if previous_pos >= n or current_pos >= n:
            return np.nan

        start_pos = min(previous_pos, current_pos)
        end_pos = max(previous_pos, current_pos) + 1
        return np.nansum(volumes[start_pos:end_pos])

    def calculate_pivot_duration(
        *,
        previous_pos: int,
        current_pos: int,
    ) -> float:
        if previous_pos < 0 or current_pos < 0:
            return np.nan
        if previous_pos >= n or current_pos >= n:
            return np.nan

        return abs(current_pos - previous_pos)

    def calculate_pivot_efficiency_ratio(
        *,
        previous_pos: int,
        current_pos: int,
    ) -> float:
        if previous_pos < 0 or current_pos < 0:
            return np.nan
        if previous_pos >= n or current_pos >= n:
            return np.nan

        start_pos = min(previous_pos, current_pos)
        end_pos = max(previous_pos, current_pos) + 1
        if (end_pos - start_pos) < 2:
            return np.nan

        closes_slice = closes[start_pos:end_pos]
        close_diffs = np.diff(closes_slice)
        path_length = float(np.nansum(np.abs(close_diffs)))
        net_move = float(abs(closes_slice[-1] - closes_slice[0]))

        if not (np.isfinite(path_length) and np.isfinite(net_move)):
            return np.nan
        if np.isclose(path_length, 0.0):
            return np.nan

        return net_move / path_length

    def add_pivot(pos: int, value: float, direction: TrendDirection):
        nonlocal last_pivot_pos
        if pivots_indices and indices[pos] == pivots_indices[-1]:
            return
        pivots_indices.append(indices[pos])
        pivots_values.append(value)
        pivots_directions.append(direction)

        if len(pivots_values) > 1 and last_pivot_pos >= 0:
            amplitude, amplitude_threshold_ratio = (
                calculate_pivot_amplitude_and_threshold_ratio(
                    previous_pos=last_pivot_pos,
                    previous_value=pivots_values[-2],
                    current_pos=pos,
                    current_value=value,
                )
            )
            volume = calculate_pivot_volume(
                previous_pos=last_pivot_pos,
                current_pos=pos,
            )
            duration = calculate_pivot_duration(
                previous_pos=last_pivot_pos,
                current_pos=pos,
            )
            if np.isfinite(amplitude) and np.isfinite(duration) and duration > 0.0:
                speed = amplitude / duration
            else:
                speed = np.nan
            efficiency_ratio = calculate_pivot_efficiency_ratio(
                previous_pos=last_pivot_pos,
                current_pos=pos,
            )
        else:
            amplitude = np.nan
            amplitude_threshold_ratio = np.nan
            volume = np.nan
            duration = np.nan
            speed = np.nan
            efficiency_ratio = np.nan

        pivots_amplitudes.append(amplitude)
        pivots_amplitude_threshold_ratios.append(amplitude_threshold_ratio)
        pivots_volumes.append(volume)
        pivots_durations.append(duration)
        pivots_speeds.append(speed)
        pivots_efficiency_ratios.append(efficiency_ratio)

        last_pivot_pos = pos
        reset_candidate_pivot()

    slope_ok_cache: dict[tuple[int, int, TrendDirection, float], bool] = {}

    def get_slope_ok(
        pos: int,
        candidate_pivot_pos: int,
        direction: TrendDirection,
        min_slope: float,
    ) -> bool:
        cache_key = (
            pos,
            candidate_pivot_pos,
            direction,
            min_slope,
        )

        if cache_key in slope_ok_cache:
            return slope_ok_cache[cache_key]

        if pos <= candidate_pivot_pos:
            slope_ok_cache[cache_key] = False
            return slope_ok_cache[cache_key]

        log_candidate_pivot_close = log_closes[candidate_pivot_pos]
        log_current_close = log_closes[pos]

        log_slope_close = (log_current_close - log_candidate_pivot_close) / (
            pos - candidate_pivot_pos
        )

        if direction == TrendDirection.UP:
            slope_ok_cache[cache_key] = log_slope_close > min_slope
        elif direction == TrendDirection.DOWN:
            slope_ok_cache[cache_key] = log_slope_close < -min_slope
        else:
            slope_ok_cache[cache_key] = False

        return slope_ok_cache[cache_key]

    def is_pivot_confirmed(
        pos: int,
        candidate_pivot_pos: int,
        direction: TrendDirection,
        min_slope: float = np.finfo(float).eps,
        alpha: float = 0.05,
    ) -> bool:
        start_pos = min(candidate_pivot_pos + 1, n)
        end_pos = min(pos + 1, n)
        n_slopes = max(0, end_pos - start_pos)

        if n_slopes < 1:
            return False

        slopes_ok: list[bool] = []
        for i in range(start_pos, end_pos):
            slopes_ok.append(
                get_slope_ok(
                    pos=i,
                    candidate_pivot_pos=candidate_pivot_pos,
                    direction=direction,
                    min_slope=min_slope,
                )
            )

        slopes_ok_threshold = calculate_slopes_ok_threshold(candidate_pivot_pos)
        n_slopes_ok = sum(slopes_ok)
        binomtest = sp.stats.binomtest(
            k=n_slopes_ok, n=n_slopes, p=0.5, alternative="greater"
        )

        return (
            binomtest.pvalue <= alpha
            and (n_slopes_ok / n_slopes) >= slopes_ok_threshold
        )

    start_pos = 0
    initial_high_pos = start_pos
    initial_low_pos = start_pos
    initial_high = highs[initial_high_pos]
    initial_low = lows[initial_low_pos]
    for i in range(start_pos + 1, n):
        current_high = highs[i]
        current_low = lows[i]
        if current_high > initial_high:
            initial_high, initial_high_pos = current_high, i
        if current_low < initial_low:
            initial_low, initial_low_pos = current_low, i

        initial_move_from_high = (initial_high - current_low) / initial_high
        initial_move_from_low = (current_high - initial_low) / initial_low
        is_initial_high_move_significant: bool = (
            initial_move_from_high >= thresholds[initial_high_pos]
        )
        is_initial_low_move_significant: bool = (
            initial_move_from_low >= thresholds[initial_low_pos]
        )
        if is_initial_high_move_significant and is_initial_low_move_significant:
            if initial_move_from_high > initial_move_from_low:
                add_pivot(initial_high_pos, initial_high, TrendDirection.UP)
                state = TrendDirection.DOWN
                break
            else:
                add_pivot(initial_low_pos, initial_low, TrendDirection.DOWN)
                state = TrendDirection.UP
                break
        else:
            if is_initial_high_move_significant:
                add_pivot(initial_high_pos, initial_high, TrendDirection.UP)
                state = TrendDirection.DOWN
                break
            elif is_initial_low_move_significant:
                add_pivot(initial_low_pos, initial_low, TrendDirection.DOWN)
                state = TrendDirection.UP
                break
    else:
        return (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    for i in range(last_pivot_pos + 1, n):
        current_high = highs[i]
        current_low = lows[i]

        if state == TrendDirection.UP:
            if np.isnan(candidate_pivot_value) or current_high > candidate_pivot_value:
                update_candidate_pivot(i, current_high)
            if (
                candidate_pivot_value - current_low
            ) / candidate_pivot_value >= thresholds[
                candidate_pivot_pos
            ] and is_pivot_confirmed(i, candidate_pivot_pos, TrendDirection.DOWN):
                add_pivot(candidate_pivot_pos, candidate_pivot_value, TrendDirection.UP)
                state = TrendDirection.DOWN

        elif state == TrendDirection.DOWN:
            if np.isnan(candidate_pivot_value) or current_low < candidate_pivot_value:
                update_candidate_pivot(i, current_low)
            if (
                current_high - candidate_pivot_value
            ) / candidate_pivot_value >= thresholds[
                candidate_pivot_pos
            ] and is_pivot_confirmed(i, candidate_pivot_pos, TrendDirection.UP):
                add_pivot(
                    candidate_pivot_pos, candidate_pivot_value, TrendDirection.DOWN
                )
                state = TrendDirection.UP

    return (
        pivots_indices,
        pivots_values,
        pivots_directions,
        pivots_amplitudes,
        pivots_amplitude_threshold_ratios,
        pivots_volumes,
        pivots_durations,
        pivots_speeds,
        pivots_efficiency_ratios,
    )


Regressor = Literal["xgboost", "lightgbm"]
REGRESSORS: Final[tuple[Regressor, ...]] = ("xgboost", "lightgbm")


def get_optuna_callbacks(
    trial: optuna.trial.Trial, regressor: Regressor
) -> list[
    Union[
        optuna.integration.XGBoostPruningCallback,
        optuna.integration.LightGBMPruningCallback,
    ]
]:
    callbacks: list[
        Union[
            optuna.integration.XGBoostPruningCallback,
            optuna.integration.LightGBMPruningCallback,
        ]
    ]
    if regressor == REGRESSORS[0]:  # "xgboost"
        callbacks = [
            optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")
        ]
    elif regressor == REGRESSORS[1]:  # "lightgbm"
        callbacks = [optuna.integration.LightGBMPruningCallback(trial, "rmse")]
    else:
        raise ValueError(
            f"Unsupported regressor model: {regressor} (supported: {', '.join(REGRESSORS)})"
        )
    return callbacks


def fit_regressor(
    regressor: Regressor,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_weights: NDArray[np.floating],
    eval_set: Optional[list[tuple[pd.DataFrame, pd.DataFrame]]],
    eval_weights: Optional[list[NDArray[np.floating]]],
    model_training_parameters: dict[str, Any],
    init_model: Any = None,
    callbacks: Optional[
        list[
            Union[
                optuna.integration.XGBoostPruningCallback,
                optuna.integration.LightGBMPruningCallback,
            ]
        ]
    ] = None,
    trial: Optional[optuna.trial.Trial] = None,
) -> Any:
    if regressor == REGRESSORS[0]:  # "xgboost"
        from xgboost import XGBRegressor

        model_training_parameters.setdefault("random_state", 1)

        if trial is not None:
            model_training_parameters["random_state"] = (
                model_training_parameters["random_state"] + trial.number
            )

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            callbacks=callbacks,
            **model_training_parameters,
        )
        model.fit(
            X=X,
            y=y,
            sample_weight=train_weights,
            eval_set=eval_set,
            sample_weight_eval_set=eval_weights,
            xgb_model=init_model,
        )
    elif regressor == REGRESSORS[1]:  # "lightgbm"
        from lightgbm import LGBMRegressor

        model_training_parameters.setdefault("seed", 1)

        if trial is not None:
            model_training_parameters["seed"] = (
                model_training_parameters["seed"] + trial.number
            )

        model = LGBMRegressor(objective="regression", **model_training_parameters)
        model.fit(
            X=X,
            y=y,
            sample_weight=train_weights,
            eval_set=eval_set,
            eval_sample_weight=eval_weights,
            eval_metric="rmse",
            init_model=init_model,
            callbacks=callbacks,
        )
    else:
        raise ValueError(
            f"Unsupported regressor model: {regressor} (supported: {', '.join(REGRESSORS)})"
        )
    return model


def get_optuna_study_model_parameters(
    trial: optuna.trial.Trial,
    regressor: Regressor,
    model_training_best_parameters: dict[str, Any],
    space_reduction: bool,
    expansion_ratio: float,
) -> dict[str, Any]:
    if regressor not in set(REGRESSORS):
        raise ValueError(
            f"Unsupported regressor model: {regressor} (supported: {', '.join(REGRESSORS)})"
        )
    if not isinstance(expansion_ratio, (int, float)) or not (
        0.0 <= expansion_ratio <= 1.0
    ):
        raise ValueError(
            f"expansion_ratio must be a float between 0 and 1, got {expansion_ratio}"
        )
    default_ranges: dict[str, tuple[float, float]] = {
        "n_estimators": (100, 2000),
        "learning_rate": (1e-3, 0.5),
        "min_child_weight": (1e-8, 100.0),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (1e-8, 100.0),
        "reg_lambda": (1e-8, 100.0),
        "max_depth": (3, 13),
        "gamma": (1e-8, 10.0),
        "num_leaves": (8, 256),
        "min_split_gain": (1e-8, 10.0),
        "min_child_samples": (10, 100),
    }

    log_scaled_params = {
        "learning_rate",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "gamma",
        "min_split_gain",
    }

    ranges = copy.deepcopy(default_ranges)
    if space_reduction and model_training_best_parameters:
        for param, (default_min, default_max) in default_ranges.items():
            center_value = model_training_best_parameters.get(param)

            center_value = center_value or midpoint(default_min, default_max)
            if not isinstance(center_value, (int, float)) or not np.isfinite(
                center_value
            ):
                continue

            if param in log_scaled_params:
                new_min = center_value / (1 + expansion_ratio)
                new_max = center_value * (1 + expansion_ratio)
            else:
                margin = (default_max - default_min) * expansion_ratio / 2
                new_min = center_value - margin
                new_max = center_value + margin

            param_min = max(default_min, new_min)
            param_max = min(default_max, new_max)

            if param_min < param_max:
                ranges[param] = (param_min, param_max)

    study_model_parameters = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            int(ranges["n_estimators"][0]),
            int(ranges["n_estimators"][1]),
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            ranges["learning_rate"][0],
            ranges["learning_rate"][1],
            log=True,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            ranges["min_child_weight"][0],
            ranges["min_child_weight"][1],
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample", ranges["subsample"][0], ranges["subsample"][1]
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            ranges["colsample_bytree"][0],
            ranges["colsample_bytree"][1],
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", ranges["reg_alpha"][0], ranges["reg_alpha"][1], log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", ranges["reg_lambda"][0], ranges["reg_lambda"][1], log=True
        ),
    }
    if regressor == REGRESSORS[0]:  # "xgboost"
        study_model_parameters.update(
            {
                "max_depth": trial.suggest_int(
                    "max_depth",
                    int(ranges["max_depth"][0]),
                    int(ranges["max_depth"][1]),
                ),
                "gamma": trial.suggest_float(
                    "gamma", ranges["gamma"][0], ranges["gamma"][1], log=True
                ),
            }
        )
    elif regressor == REGRESSORS[1]:  # "lightgbm"
        study_model_parameters.update(
            {
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    int(ranges["num_leaves"][0]),
                    int(ranges["num_leaves"][1]),
                ),
                "min_split_gain": trial.suggest_float(
                    "min_split_gain",
                    ranges["min_split_gain"][0],
                    ranges["min_split_gain"][1],
                    log=True,
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples",
                    int(ranges["min_child_samples"][0]),
                    int(ranges["min_child_samples"][1]),
                ),
            }
        )
    return study_model_parameters


@lru_cache(maxsize=128)
def largest_divisor_to_step(integer: int, step: int) -> Optional[int]:
    if not isinstance(integer, int) or integer <= 0:
        raise ValueError("integer must be a positive integer")
    if not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")

    if step == 1 or integer % step == 0:
        return integer

    best_divisor: Optional[int] = None
    max_divisor = int(math.isqrt(integer))
    for i in range(1, max_divisor + 1):
        if integer % i != 0:
            continue
        j = integer // i
        if j % step == 0:
            return j
        if i % step == 0:
            best_divisor = i

    return best_divisor


def soft_extremum(series: pd.Series, alpha: float) -> float:
    np_array = series.to_numpy()
    if np_array.size == 0:
        return np.nan
    if np.isclose(alpha, 0.0):
        return np.nanmean(np_array)
    scaled_np_array = alpha * np_array
    max_scaled_np_array = np.max(scaled_np_array)
    if np.isinf(max_scaled_np_array):
        return np_array[np.argmax(scaled_np_array)]
    shifted_exponentials = np.exp(scaled_np_array - max_scaled_np_array)
    numerator = np.sum(np_array * shifted_exponentials)
    denominator = np.sum(shifted_exponentials)
    if denominator == 0:
        return np.max(np_array)
    return numerator / denominator


@lru_cache(maxsize=8)
def get_min_max_label_period_candles(
    fit_live_predictions_candles: int,
    candles_step: int,
    min_label_period_candles: int = 12,
    max_label_period_candles: int = 24,
    min_label_period_candles_fallback: int = 12,
    max_label_period_candles_fallback: int = 24,
    max_period_candles: int = 48,
    max_horizon_fraction: float = 1.0 / 3.0,
) -> tuple[int, int, int]:
    if min_label_period_candles > max_label_period_candles:
        raise ValueError(
            "min_label_period_candles must be less than or equal to max_label_period_candles"
        )

    capped_period_candles = max(1, floor_to_step(max_period_candles, candles_step))
    capped_horizon_candles = max(
        1,
        floor_to_step(
            max(1, math.ceil(fit_live_predictions_candles * max_horizon_fraction)),
            candles_step,
        ),
    )
    max_label_period_candles = min(
        max_label_period_candles, capped_period_candles, capped_horizon_candles
    )

    if min_label_period_candles > max_label_period_candles:
        fallback_high = min(
            max_label_period_candles_fallback,
            capped_period_candles,
            capped_horizon_candles,
        )
        return (
            min(min_label_period_candles_fallback, fallback_high),
            fallback_high,
            1,
        )

    if candles_step <= (max_label_period_candles - min_label_period_candles):
        low = ceil_to_step(min_label_period_candles, candles_step)
        high = floor_to_step(max_label_period_candles, candles_step)
        if low > high:
            low, high, candles_step = (
                min_label_period_candles,
                max_label_period_candles,
                1,
            )
    else:
        low, high, candles_step = min_label_period_candles, max_label_period_candles, 1

    return low, high, candles_step


@lru_cache(maxsize=128)
def round_to_step(value: float | int, step: int) -> int:
    """
    Round a value to the nearest multiple of a given step.
    :param value: The value to round.
    :param step: The step size to round to (must be a positive integer).
    :return: The rounded value.
    :raises ValueError: If step is not a positive integer or value is not finite.
    """
    if not isinstance(value, (int, float)):
        raise ValueError("value must be an integer or float")
    if not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")
    if isinstance(value, (int, np.integer)):
        q, r = divmod(value, step)
        twice_r = r * 2
        if twice_r < step:
            return q * step
        if twice_r > step:
            return (q + 1) * step
        return int(round(value / step) * step)
    if not np.isfinite(value):
        raise ValueError("value must be finite")
    return int(round(float(value) / step) * step)


@lru_cache(maxsize=128)
def ceil_to_step(value: float | int, step: int) -> int:
    if not isinstance(value, (int, float)):
        raise ValueError("value must be an integer or float")
    if not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")
    if isinstance(value, (int, np.integer)):
        return int(-(-int(value) // step) * step)
    if not np.isfinite(value):
        raise ValueError("value must be finite")
    return int(math.ceil(float(value) / step) * step)


@lru_cache(maxsize=128)
def floor_to_step(value: float | int, step: int) -> int:
    if not isinstance(value, (int, float)):
        raise ValueError("value must be an integer or float")
    if not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")
    if isinstance(value, (int, np.integer)):
        return int((int(value) // step) * step)
    if not np.isfinite(value):
        raise ValueError("value must be finite")
    return int(math.floor(float(value) / step) * step)


def validate_range(
    min_val: float | int,
    max_val: float | int,
    logger: Logger,
    *,
    name: str,
    default_min: float | int,
    default_max: float | int,
    allow_equal: bool = False,
    non_negative: bool = True,
    finite_only: bool = True,
) -> tuple[float | int, float | int]:
    min_name = f"min_{name}"
    max_name = f"max_{name}"

    if not isinstance(default_min, (int, float)) or not isinstance(
        default_max, (int, float)
    ):
        raise ValueError(f"{name}: defaults must be numeric")
    if default_min > default_max or (not allow_equal and default_min == default_max):
        raise ValueError(
            f"{name}: invalid defaults ordering {default_min} >= {default_max}"
        )

    def _validate_component(
        value: float | int | None, name: str, default_value: float | int
    ) -> float | int:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or (finite_only and not np.isfinite(value))
            or (non_negative and value < 0)
        ):
            logger.warning(
                f"{name}: invalid value {value!r}, using default {default_value}"
            )
            return default_value
        return value

    sanitized_min = _validate_component(min_val, min_name, default_min)
    sanitized_max = _validate_component(max_val, max_name, default_max)

    ordering_ok = (
        (sanitized_min < sanitized_max)
        if not allow_equal
        else (sanitized_min <= sanitized_max)
    )
    if not ordering_ok:
        logger.warning(
            f"{name}: invalid ordering ({min_name}={sanitized_min}, {max_name}={sanitized_max}); using defaults ({default_min}, {default_max})"
        )
        sanitized_min, sanitized_max = default_min, default_max

    if sanitized_min != min_val or sanitized_max != max_val:
        logger.warning(
            f"{name}: sanitized {min_name}={sanitized_min}, {max_name}={sanitized_max} (defaults=({default_min}, {default_max}))"
        )

    return sanitized_min, sanitized_max


def get_label_defaults(
    feature_parameters: dict[str, Any],
    logger: Logger,
    *,
    default_min_label_period_candles: int = 12,
    default_max_label_period_candles: int = 24,
    default_min_label_natr_ratio: float = 9.0,
    default_max_label_natr_ratio: float = 12.0,
) -> tuple[float, int]:
    min_label_natr_ratio = feature_parameters.get(
        "min_label_natr_ratio", default_min_label_natr_ratio
    )
    max_label_natr_ratio = feature_parameters.get(
        "max_label_natr_ratio", default_max_label_natr_ratio
    )
    min_label_natr_ratio, max_label_natr_ratio = validate_range(
        min_label_natr_ratio,
        max_label_natr_ratio,
        logger,
        name="label_natr_ratio",
        default_min=default_min_label_natr_ratio,
        default_max=default_max_label_natr_ratio,
        allow_equal=False,
        non_negative=True,
        finite_only=True,
    )
    default_label_natr_ratio = float(
        midpoint(min_label_natr_ratio, max_label_natr_ratio)
    )

    min_label_period_candles = feature_parameters.get(
        "min_label_period_candles", default_min_label_period_candles
    )
    max_label_period_candles = feature_parameters.get(
        "max_label_period_candles", default_max_label_period_candles
    )
    min_label_period_candles, max_label_period_candles = validate_range(
        min_label_period_candles,
        max_label_period_candles,
        logger,
        name="label_period_candles",
        default_min=default_min_label_period_candles,
        default_max=default_max_label_period_candles,
        allow_equal=True,
        non_negative=True,
        finite_only=True,
    )
    default_label_period_candles = int(
        round(midpoint(min_label_period_candles, max_label_period_candles))
    )

    return default_label_natr_ratio, default_label_period_candles
