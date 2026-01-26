import copy
import functools
import hashlib
import math
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache, singledispatch
from logging import Logger
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import numpy as np
import optuna
import pandas as pd
import scipy as sp
import talib.abstract as ta
from LabelTransformer import (
    COMBINED_AGGREGATIONS,
    COMBINED_METRICS,
    DEFAULTS_LABEL_PIPELINE,
    DEFAULTS_LABEL_PREDICTION,
    DEFAULTS_LABEL_SMOOTHING,
    DEFAULTS_LABEL_WEIGHTING,
    EXTREMA_SELECTION_METHODS,
    NORMALIZATION_TYPES,
    PREDICTION_METHODS,
    SMOOTHING_METHODS,
    SMOOTHING_MODES,
    STANDARDIZATION_TYPES,
    THRESHOLD_METHODS,
    WEIGHT_STRATEGIES,
    CombinedAggregation,
    CombinedMetric,
    SmoothingMethod,
    SmoothingMode,
)
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.stats import percentileofscore
from technical import qtpylib

if TYPE_CHECKING:
    from xgboost.callback import TrainingCallback as XGBoostTrainingCallback
else:
    XGBoostTrainingCallback = object

T = TypeVar("T", pd.Series, float)


@dataclass(frozen=True, slots=True)
class _EnumValidator:
    valid_values: tuple[str, ...]

    def __call__(self, value: Any) -> bool:
        return value in self.valid_values

    def message(self, param: str) -> str:
        return f"supported values are {', '.join(self.valid_values)}"


@dataclass(frozen=True, slots=True)
class _NumericValidator:
    min_value: float | None = None
    max_value: float | None = None
    min_exclusive: bool = False
    max_exclusive: bool = False
    require_int: bool = False

    def __call__(self, value: Any) -> bool:
        if self.require_int and not isinstance(value, int):
            return False
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            return False
        if self.min_value is not None:
            if self.min_exclusive and value <= self.min_value:
                return False
            if not self.min_exclusive and value < self.min_value:
                return False
        if self.max_value is not None:
            if self.max_exclusive and value >= self.max_value:
                return False
            if not self.max_exclusive and value > self.max_value:
                return False
        return True

    def message(self, param: str) -> str:
        parts = []
        if self.require_int:
            parts.append("must be an integer")
        else:
            parts.append("must be a finite number")
        if self.min_value is not None:
            op = ">" if self.min_exclusive else ">="
            parts.append(f"{op} {self.min_value}")
        if self.max_value is not None:
            op = "<" if self.max_exclusive else "<="
            parts.append(f"{op} {self.max_value}")
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class _RangeValidator:
    min_bound: float | None = None
    max_bound: float | None = None

    def __call__(self, value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return False
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in value):
            return False
        if value[0] >= value[1]:
            return False
        if self.min_bound is not None and value[0] < self.min_bound:
            return False
        if self.max_bound is not None and value[1] > self.max_bound:
            return False
        return True

    def message(self, param: str) -> str:
        if self.min_bound is not None and self.max_bound is not None:
            return f"must be (low, high) with {self.min_bound} <= low < high <= {self.max_bound}"
        return "must be (low, high) with low < high"


@dataclass(frozen=True, slots=True)
class _DictValidator:
    valid_keys: tuple[str, ...] | None = None

    def __call__(self, value: Any) -> bool:
        return isinstance(value, dict)

    def message(self, param: str) -> str:
        return "must be a mapping"


_Validator = _EnumValidator | _NumericValidator | _RangeValidator | _DictValidator


@dataclass(frozen=True, slots=True)
class _ParamSpec:
    validator: _Validator
    output_type: type | None = None


def _validate_params(
    config: dict[str, Any],
    logger: Logger,
    config_name: str,
    specs: dict[str, _ParamSpec],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for param, spec in specs.items():
        value = config.get(param, defaults[param])
        if not spec.validator(value):
            logger.warning(
                f"Invalid {config_name} {param} value {value!r}: "
                f"{spec.validator.message(param)}, using default {defaults[param]!r}"
            )
            value = defaults[param]
        elif isinstance(spec.validator, _DictValidator) and spec.validator.valid_keys:
            invalid_keys = set(value.keys()) - set(spec.validator.valid_keys)
            if invalid_keys:
                logger.warning(
                    f"Invalid {config_name} {param} keys {sorted(invalid_keys)!r}, "
                    f"valid keys: {', '.join(spec.validator.valid_keys)}"
                )
                value = {
                    k: v for k, v in value.items() if k in spec.validator.valid_keys
                }
        if spec.output_type is not None:
            if spec.output_type is tuple and isinstance(value, (list, tuple)):
                value = (value[0], value[1])
            else:
                value = spec.output_type(value)
        result[param] = value
    return result


_WEIGHTING_SPECS: Final[dict[str, _ParamSpec]] = {
    "strategy": _ParamSpec(_EnumValidator(WEIGHT_STRATEGIES)),
    "metric_coefficients": _ParamSpec(_DictValidator(COMBINED_METRICS)),
    "aggregation": _ParamSpec(_EnumValidator(COMBINED_AGGREGATIONS)),
    "softmax_temperature": _ParamSpec(
        _NumericValidator(min_value=0, min_exclusive=True)
    ),
}

_PIPELINE_SPECS: Final[dict[str, _ParamSpec]] = {
    "standardization": _ParamSpec(_EnumValidator(STANDARDIZATION_TYPES)),
    "robust_quantiles": _ParamSpec(
        _RangeValidator(min_bound=0, max_bound=1), output_type=tuple
    ),
    "mmad_scaling_factor": _ParamSpec(
        _NumericValidator(min_value=0, min_exclusive=True)
    ),
    "normalization": _ParamSpec(_EnumValidator(NORMALIZATION_TYPES)),
    "minmax_range": _ParamSpec(_RangeValidator(), output_type=tuple),
    "sigmoid_scale": _ParamSpec(_NumericValidator(min_value=0, min_exclusive=True)),
    "gamma": _ParamSpec(
        _NumericValidator(min_value=0, max_value=10, min_exclusive=True)
    ),
}

_SMOOTHING_SPECS: Final[dict[str, _ParamSpec]] = {
    "method": _ParamSpec(_EnumValidator(SMOOTHING_METHODS)),
    "window_candles": _ParamSpec(
        _NumericValidator(min_value=1, require_int=True), output_type=int
    ),
    "beta": _ParamSpec(
        _NumericValidator(min_value=0, min_exclusive=True), output_type=float
    ),
    "polyorder": _ParamSpec(
        _NumericValidator(min_value=0, require_int=True), output_type=int
    ),
    "mode": _ParamSpec(_EnumValidator(SMOOTHING_MODES)),
    "sigma": _ParamSpec(
        _NumericValidator(min_value=0, min_exclusive=True), output_type=float
    ),
}

_PREDICTION_SPECS: Final[dict[str, _ParamSpec]] = {
    "method": _ParamSpec(_EnumValidator(PREDICTION_METHODS)),
    "selection_method": _ParamSpec(_EnumValidator(EXTREMA_SELECTION_METHODS)),
    "threshold_method": _ParamSpec(_EnumValidator(THRESHOLD_METHODS)),
    "outlier_quantile": _ParamSpec(
        _NumericValidator(
            min_value=0, max_value=1, min_exclusive=True, max_exclusive=True
        ),
        output_type=float,
    ),
    "soft_extremum_alpha": _ParamSpec(
        _NumericValidator(min_value=0), output_type=float
    ),
    "keep_fraction": _ParamSpec(
        _NumericValidator(min_value=0, max_value=1, min_exclusive=True),
        output_type=float,
    ),
}


EXTREMA_COLUMN: Final = "&s-extrema"
LABEL_COLUMNS: Final[tuple[str, ...]] = (EXTREMA_COLUMN,)


@dataclass
class LabelData:
    series: pd.Series
    indices: list[int]
    metrics: dict[str, list[float]]


LabelGenerator = Callable[[pd.DataFrame, dict[str, Any]], LabelData]
_LABEL_GENERATORS: dict[str, LabelGenerator] = {}


def register_label_generator(label_column: str, generator: LabelGenerator) -> None:
    _LABEL_GENERATORS[label_column] = generator


def _generate_extrema_label(
    dataframe: pd.DataFrame,
    params: dict[str, Any],
) -> LabelData:
    natr_period = params.get("natr_period", 14)
    natr_multiplier = params.get("natr_multiplier", 9.0)

    (
        pivots_indices,
        _,
        pivots_directions,
        pivots_amplitudes,
        pivots_amplitude_threshold_ratios,
        pivots_volume_rates,
        pivots_speeds,
        pivots_efficiency_ratios,
        pivots_volume_weighted_efficiency_ratios,
    ) = zigzag(
        dataframe,
        natr_period=natr_period,
        natr_multiplier=natr_multiplier,
    )

    series = pd.Series(0.0, index=dataframe.index)
    if pivots_indices:
        series.loc[pivots_indices] = pivots_directions

    metrics: dict[str, list[float]] = {
        "amplitude": pivots_amplitudes,
        "amplitude_threshold_ratio": pivots_amplitude_threshold_ratios,
        "volume_rate": pivots_volume_rates,
        "speed": pivots_speeds,
        "efficiency_ratio": pivots_efficiency_ratios,
        "volume_weighted_efficiency_ratio": pivots_volume_weighted_efficiency_ratios,
    }

    return LabelData(series=series, indices=pivots_indices, metrics=metrics)


register_label_generator(EXTREMA_COLUMN, _generate_extrema_label)


def generate_label_data(
    dataframe: pd.DataFrame,
    label_column: str,
    params: dict[str, Any],
) -> LabelData:
    generator = _LABEL_GENERATORS.get(label_column)
    if generator is None:
        raise KeyError(
            f"No label generator registered for column '{label_column}'. "
            f"Available columns: {list(_LABEL_GENERATORS.keys())}"
        )
    return generator(dataframe, params)


MAXIMA_COLUMN: Final = "maxima"
MINIMA_COLUMN: Final = "minima"
SMOOTHED_EXTREMA_COLUMN: Final = "smoothed_extrema"

SmoothingKernel = Literal["gaussian", "kaiser", "triang"]
SMOOTHING_KERNELS: Final[tuple[SmoothingKernel, ...]] = (
    "gaussian",
    "kaiser",
    "triang",
)

TradePriceTarget = Literal[
    "moving_average", "quantile_interpolation", "weighted_average"
]
TRADE_PRICE_TARGETS: Final[tuple[TradePriceTarget, ...]] = (
    "moving_average",
    "quantile_interpolation",
    "weighted_average",
)


DEFAULT_LABEL_WEIGHT: Final[float] = 1.0

DEFAULT_FIT_LIVE_PREDICTIONS_CANDLES: Final[int] = 100


ValidateParamsFn = Callable[[dict[str, Any], Logger, str], dict[str, Any]]


_MISSING: Final = object()


def _get_path(config: dict[str, Any], path: str) -> Any:
    keys = path.split(".")
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _delete_path(config: dict[str, Any], path: str) -> bool:
    keys = path.split(".")
    current = config
    for key in keys[:-1]:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    if isinstance(current, dict) and keys[-1] in current:
        del current[keys[-1]]
        return True
    return False


# Order matters: section renames before key moves (e.g. extrema_weighting.gamma -> label_weighting.gamma -> label_pipeline.gamma)
CONFIG_MIGRATIONS: Final[tuple[tuple[str, str], ...]] = (
    ("freqai.extrema_weighting", "freqai.label_weighting"),
    ("freqai.extrema_smoothing", "freqai.label_smoothing"),
    ("freqai.predictions_extrema", "freqai.label_prediction"),
    ("freqai.label_smoothing.window", "freqai.label_smoothing.window_candles"),
    (
        "freqai.label_prediction.thresholds_smoothing",
        "freqai.label_prediction.threshold_smoothing_method",
    ),
    (
        "freqai.label_prediction.threshold_smoothing_method",
        "freqai.label_prediction.threshold_method",
    ),
    (
        "freqai.label_prediction.threshold_outlier",
        "freqai.label_prediction.outlier_threshold_quantile",
    ),
    (
        "freqai.label_prediction.outlier_threshold_quantile",
        "freqai.label_prediction.outlier_quantile",
    ),
    (
        "freqai.label_prediction.extrema_fraction",
        "freqai.label_prediction.keep_extrema_fraction",
    ),
    (
        "freqai.label_prediction.keep_extrema_fraction",
        "freqai.label_prediction.keep_fraction",
    ),
    (
        "freqai.label_prediction.thresholds_alpha",
        "freqai.label_prediction.soft_extremum_alpha",
    ),
    ("exit_pricing.trade_price_target", "exit_pricing.trade_price_target_method"),
    (
        "reversal_confirmation.lookback_period",
        "reversal_confirmation.lookback_period_candles",
    ),
    ("reversal_confirmation.decay_ratio", "reversal_confirmation.decay_fraction"),
    (
        "reversal_confirmation.min_natr_ratio_percent",
        "reversal_confirmation.min_natr_multiplier_fraction",
    ),
    (
        "reversal_confirmation.max_natr_ratio_percent",
        "reversal_confirmation.max_natr_multiplier_fraction",
    ),
    (
        "freqai.feature_parameters.min_label_natr_ratio",
        "freqai.feature_parameters.min_label_natr_multiplier",
    ),
    (
        "freqai.feature_parameters.max_label_natr_ratio",
        "freqai.feature_parameters.max_label_natr_multiplier",
    ),
    (
        "freqai.feature_parameters.label_natr_ratio",
        "freqai.feature_parameters.label_natr_multiplier",
    ),
    ("freqai.optuna_hyperopt.expansion_ratio", "freqai.optuna_hyperopt.space_fraction"),
    (
        "freqai.label_weighting.standardization",
        "freqai.label_pipeline.standardization",
    ),
    (
        "freqai.label_weighting.robust_quantiles",
        "freqai.label_pipeline.robust_quantiles",
    ),
    (
        "freqai.label_weighting.mmad_scaling_factor",
        "freqai.label_pipeline.mmad_scaling_factor",
    ),
    ("freqai.label_weighting.normalization", "freqai.label_pipeline.normalization"),
    ("freqai.label_weighting.minmax_range", "freqai.label_pipeline.minmax_range"),
    ("freqai.label_weighting.sigmoid_scale", "freqai.label_pipeline.sigmoid_scale"),
    ("freqai.label_weighting.gamma", "freqai.label_pipeline.gamma"),
)


def migrate_config(config: dict[str, Any], logger: Logger) -> None:
    for old_path, new_path in CONFIG_MIGRATIONS:
        old_value = _get_path(config, old_path)
        if old_value is _MISSING:
            continue

        old_section = old_path.rsplit(".", 1)[0] if "." in old_path else ""
        new_section = new_path.rsplit(".", 1)[0] if "." in new_path else ""
        new_key = new_path.rsplit(".", 1)[-1]

        new_value = _get_path(config, new_path)
        if new_value is _MISSING:
            _set_path(config, new_path, old_value)
            _delete_path(config, old_path)
            if old_section == new_section:
                logger.warning(f"{old_path} is deprecated, use {new_key} instead")
            else:
                logger.warning(f"{old_path} has moved to {new_path}")
        else:
            _delete_path(config, old_path)
            if old_section == new_section:
                logger.warning(
                    f"{new_section} has both {new_key} and deprecated {old_path.rsplit('.', 1)[-1]}, using {new_key}"
                )
            else:
                logger.warning(
                    f"{new_section} has {new_key} and deprecated {old_path}, using {new_path}"
                )


def _get_label_config(
    config: dict[str, Any],
    logger: Logger,
    config_name: str,
    validate_fn: ValidateParamsFn,
    defaults_dict: dict[str, Any],
) -> dict[str, Any]:
    if "default" in config or "columns" in config:
        default_config = config.get("default", {})
        if not isinstance(default_config, dict):
            logger.warning(
                f"Invalid {config_name} default value {default_config!r}: must be a mapping, using defaults"
            )
            default_config = {}

        validated_default = validate_fn(
            default_config, logger, f"{config_name}.default"
        )

        columns_config = config.get("columns", {})
        if not isinstance(columns_config, dict):
            logger.warning(
                f"Invalid {config_name} columns value {columns_config!r}: must be a mapping, ignoring"
            )
            columns_config = {}

        validated_columns: dict[str, dict[str, Any]] = {}
        for col_pattern, col_config in columns_config.items():
            if not isinstance(col_config, dict):
                logger.warning(
                    f"Invalid {config_name} columns[{col_pattern!r}] value {col_config!r}: must be a mapping, ignoring"
                )
                continue
            validated_col: dict[str, Any] = {}
            for key, value in col_config.items():
                if key in defaults_dict:
                    temp = {key: value}
                    validated = validate_fn(
                        temp, logger, f"{config_name}.columns[{col_pattern!r}]"
                    )
                    validated_col[key] = validated[key]
                else:
                    logger.warning(
                        f"Unknown {config_name}.columns[{col_pattern!r}] key {key!r}, ignoring"
                    )
            if validated_col:
                validated_columns[col_pattern] = validated_col

        return {"default": validated_default, "columns": validated_columns}
    else:
        validated_default = validate_fn(config, logger, config_name)
        return {"default": validated_default, "columns": {}}


def _validate_weighting_params(
    config: dict[str, Any],
    logger: Logger,
    config_name: str = "label_weighting",
) -> dict[str, Any]:
    return _validate_params(
        config, logger, config_name, _WEIGHTING_SPECS, DEFAULTS_LABEL_WEIGHTING
    )


def get_label_weighting_config(
    config: dict[str, Any],
    logger: Logger,
) -> dict[str, Any]:
    return _get_label_config(
        config,
        logger,
        "label_weighting",
        _validate_weighting_params,
        DEFAULTS_LABEL_WEIGHTING,
    )


def _validate_pipeline_params(
    config: dict[str, Any],
    logger: Logger,
    config_name: str = "label_pipeline",
) -> dict[str, Any]:
    return _validate_params(
        config, logger, config_name, _PIPELINE_SPECS, DEFAULTS_LABEL_PIPELINE
    )


def get_label_pipeline_config(
    config: dict[str, Any],
    logger: Logger,
) -> dict[str, Any]:
    return _get_label_config(
        config,
        logger,
        "label_pipeline",
        _validate_pipeline_params,
        DEFAULTS_LABEL_PIPELINE,
    )


def _validate_smoothing_params(
    config: dict[str, Any],
    logger: Logger,
    config_name: str = "label_smoothing",
) -> dict[str, Any]:
    return _validate_params(
        config, logger, config_name, _SMOOTHING_SPECS, DEFAULTS_LABEL_SMOOTHING
    )


def get_label_smoothing_config(
    config: dict[str, Any],
    logger: Logger,
) -> dict[str, Any]:
    return _get_label_config(
        config,
        logger,
        "label_smoothing",
        _validate_smoothing_params,
        DEFAULTS_LABEL_SMOOTHING,
    )


def _validate_prediction_params(
    config: dict[str, Any],
    logger: Logger,
    config_name: str = "label_prediction",
) -> dict[str, Any]:
    return _validate_params(
        config, logger, config_name, _PREDICTION_SPECS, DEFAULTS_LABEL_PREDICTION
    )


def get_label_prediction_config(
    config: dict[str, Any],
    logger: Logger,
) -> dict[str, Any]:
    return _get_label_config(
        config,
        logger,
        "label_prediction",
        _validate_prediction_params,
        DEFAULTS_LABEL_PREDICTION,
    )


def get_distance(p1: T, p2: T) -> T:
    return abs(p1 - p2)


def midpoint(value1: T, value2: T) -> T:
    """Calculate the midpoint between two values."""
    return (value1 + value2) / 2


def nan_average(
    values: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan

    if weights is None:
        return float(np.nanmean(values))

    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    if not mask.any():
        return np.nan

    return float(np.average(values[mask], weights=weights[mask]))


def non_zero_diff(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """Returns the difference of two series and replaces zeros with epsilon."""
    diff = s1 - s2
    return diff.where(diff != 0, np.finfo(float).eps)


@lru_cache(maxsize=8)
def get_odd_window(window: int) -> int:
    if window < 1:
        raise ValueError(f"Invalid window value {window!r}: must be > 0")
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


@lru_cache(maxsize=8)
def _calculate_coeffs(
    window: int,
    win_type: SmoothingKernel,
    std: float,
    beta: float,
) -> NDArray[np.floating]:
    if win_type == SMOOTHING_KERNELS[0]:  # "gaussian"
        coeffs = sp.signal.windows.gaussian(M=window, std=std, sym=True)
    elif win_type == SMOOTHING_KERNELS[1]:  # "kaiser"
        coeffs = sp.signal.windows.kaiser(M=window, beta=beta, sym=True)
    elif win_type == SMOOTHING_KERNELS[2]:  # "triang"
        coeffs = sp.signal.windows.triang(M=window, sym=True)
    else:
        raise ValueError(
            f"Invalid window type value {win_type!r}: "
            f"supported values are {', '.join(SMOOTHING_KERNELS)}"
        )
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

    values = series.to_numpy(dtype=float)
    if values.size <= 1:
        return series

    b = _calculate_coeffs(window=window, win_type=win_type, std=std, beta=beta)
    a = np.array([1.0], dtype=float)

    filtered_values = sp.signal.filtfilt(b, a, values)
    return pd.Series(filtered_values, index=series.index)


def smooth_label(
    series: pd.Series,
    method: SmoothingMethod = DEFAULTS_LABEL_SMOOTHING["method"],
    window_candles: int = DEFAULTS_LABEL_SMOOTHING["window_candles"],
    beta: float = DEFAULTS_LABEL_SMOOTHING["beta"],
    polyorder: int = DEFAULTS_LABEL_SMOOTHING["polyorder"],
    mode: SmoothingMode = DEFAULTS_LABEL_SMOOTHING["mode"],
    sigma: float = DEFAULTS_LABEL_SMOOTHING["sigma"],
) -> pd.Series:
    n = len(series)
    if n == 0:
        return series

    if window_candles < 3:
        window_candles = 3
    if n < window_candles:
        return series
    if beta <= 0 or not np.isfinite(beta):
        beta = 1.0

    odd_window = get_odd_window(window_candles)
    std = get_gaussian_std(odd_window)

    if method == SMOOTHING_METHODS[0]:  # "none"
        return series
    elif method == SMOOTHING_METHODS[1]:  # "gaussian"
        return zero_phase_filter(
            series=series,
            window=odd_window,
            win_type=SMOOTHING_KERNELS[0],  # "gaussian"
            std=std,
            beta=beta,
        )
    elif method == SMOOTHING_METHODS[2]:  # "kaiser"
        return zero_phase_filter(
            series=series,
            window=odd_window,
            win_type=SMOOTHING_KERNELS[1],  # "kaiser"
            std=std,
            beta=beta,
        )
    elif method == SMOOTHING_METHODS[3]:  # "triang"
        return zero_phase_filter(
            series=series,
            window=odd_window,
            win_type=SMOOTHING_KERNELS[2],  # "triang"
            std=std,
            beta=beta,
        )
    elif method == SMOOTHING_METHODS[4]:  # "smm" (Simple Moving Median)
        return series.rolling(window=odd_window, center=True, min_periods=1).median()
    elif method == SMOOTHING_METHODS[5]:  # "sma" (Simple Moving Average)
        return series.rolling(window=odd_window, center=True, min_periods=1).mean()
    elif method == SMOOTHING_METHODS[6]:  # "savgol" (Savitzky-Golay)
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
    elif method == SMOOTHING_METHODS[7]:  # "gaussian_filter1d"
        return pd.Series(
            gaussian_filter1d(
                series.to_numpy(),
                sigma=sigma,
                mode=mode,  # type: ignore
            ),
            index=series.index,
        )
    else:
        return zero_phase_filter(
            series=series,
            window=odd_window,
            win_type=SMOOTHING_KERNELS[0],  # "gaussian"
            std=std,
            beta=beta,
        )


def _impute_weights(
    weights: NDArray[np.floating],
    *,
    default_weight: float = DEFAULT_LABEL_WEIGHT,
) -> NDArray[np.floating]:
    weights = weights.astype(float, copy=True)

    if weights.size == 0:
        return np.full_like(weights, default_weight, dtype=float)

    # Weights computed by `zigzag` can be NaN on boundary pivots
    if not np.isfinite(weights[0]):
        weights[0] = 0.0
    if not np.isfinite(weights[-1]):
        weights[-1] = 0.0

    finite_mask = np.isfinite(weights)
    if not finite_mask.any():
        return np.full_like(weights, default_weight, dtype=float)

    median_weight = np.nanmedian(weights[finite_mask])
    if not np.isfinite(median_weight):
        median_weight = default_weight

    weights[~finite_mask] = median_weight

    return weights


def _build_weights_array(
    n_extrema: int,
    indices: list[int],
    weights: NDArray[np.floating],
    default_weight: float = DEFAULT_LABEL_WEIGHT,
) -> NDArray[np.floating]:
    if len(indices) == 0 or weights.size == 0:
        return np.full(n_extrema, DEFAULT_LABEL_WEIGHT, dtype=float)

    if len(indices) != weights.size:
        raise ValueError(
            f"Invalid indices/weights values: length mismatch, got {len(indices)} indices but {weights.size} weights"
        )

    weights_array = np.full(n_extrema, default_weight, dtype=float)

    indices_array = np.array(indices)
    mask = (indices_array >= 0) & (indices_array < n_extrema)

    if not np.any(mask):
        return weights_array

    valid_indices = indices_array[mask]
    weights_array[valid_indices] = weights[mask]
    return weights_array


def _parse_metric_coefficients(
    metric_coefficients: dict[str, Any],
) -> dict[CombinedMetric, float]:
    out: dict[CombinedMetric, float] = {}
    for metric in COMBINED_METRICS:
        value = metric_coefficients.get(metric)
        if not isinstance(value, (int, float)):
            continue
        if not np.isfinite(value) or value <= 0:
            continue
        out[metric] = float(value)

    return out


def _aggregate_metrics(
    stacked_metrics: NDArray[np.floating],
    coefficients: NDArray[np.floating],
    aggregation: CombinedAggregation,
    softmax_temperature: float,
) -> NDArray[np.floating]:
    if aggregation == COMBINED_AGGREGATIONS[0]:  # "arithmetic_mean"
        return np.asarray(
            sp.stats.pmean(stacked_metrics.T, p=1.0, weights=coefficients, axis=1)
        )
    elif aggregation == COMBINED_AGGREGATIONS[1]:  # "geometric_mean"
        return np.asarray(
            sp.stats.pmean(stacked_metrics.T, p=0.0, weights=coefficients, axis=1)
        )
    elif aggregation == COMBINED_AGGREGATIONS[2]:  # "harmonic_mean"
        return np.asarray(
            sp.stats.pmean(stacked_metrics.T, p=-1.0, weights=coefficients, axis=1)
        )
    elif aggregation == COMBINED_AGGREGATIONS[3]:  # "quadratic_mean"
        return np.asarray(
            sp.stats.pmean(stacked_metrics.T, p=2.0, weights=coefficients, axis=1)
        )
    elif aggregation == COMBINED_AGGREGATIONS[4]:  # "weighted_median"
        return np.array(
            [
                np.quantile(
                    stacked_metrics[:, i],
                    0.5,
                    weights=coefficients,
                    method="inverted_cdf",
                )
                for i in range(stacked_metrics.shape[1])
            ]
        )
    elif aggregation == COMBINED_AGGREGATIONS[5]:  # "softmax"
        scaled_metrics = stacked_metrics / softmax_temperature
        softmax_weights = sp.special.softmax(scaled_metrics, axis=0)
        combined_weights = softmax_weights * coefficients[:, np.newaxis]
        combined_weights = combined_weights / np.sum(
            combined_weights, axis=0, keepdims=True
        )
        return np.sum(stacked_metrics * combined_weights, axis=0)
    else:
        raise ValueError(
            f"Invalid aggregation value {aggregation!r}: supported values are {', '.join(COMBINED_AGGREGATIONS)}"
        )


def _compute_combined_label_weights(
    metrics: dict[str, list[float]],
    metric_coefficients: dict[str, Any],
    aggregation: CombinedAggregation,
    softmax_temperature: float,
) -> NDArray[np.floating]:
    if len(metrics) == 0:
        return np.asarray([], dtype=float)

    coefficients = _parse_metric_coefficients(metric_coefficients)
    if len(coefficients) == 0:
        coefficients = {k: DEFAULT_LABEL_WEIGHT for k in metrics.keys()}

    imputed_metrics: list[NDArray[np.floating]] = []
    coefficients_list: list[float] = []

    for metric_name, metric_values in metrics.items():
        if metric_name not in coefficients:
            continue
        coefficient = coefficients[metric_name]
        values_array = np.asarray(metric_values, dtype=float)
        if values_array.size == 0:
            continue
        imputed_metrics.append(_impute_weights(weights=values_array))
        coefficients_list.append(float(coefficient))

    if len(imputed_metrics) == 0:
        return np.asarray([], dtype=float)

    stacked_metrics = np.vstack(imputed_metrics)
    coefficients_array = np.asarray(coefficients_list, dtype=float)

    return _aggregate_metrics(
        stacked_metrics, coefficients_array, aggregation, softmax_temperature
    )


def compute_label_weights(
    n_values: int,
    indices: list[int],
    metrics: dict[str, list[float]],
    weighting_config: dict[str, Any],
) -> NDArray[np.floating]:
    label_weighting = {**DEFAULTS_LABEL_WEIGHTING, **weighting_config}
    strategy = label_weighting["strategy"]

    if len(indices) == 0 or strategy == WEIGHT_STRATEGIES[0]:  # "none"
        return np.full(n_values, DEFAULT_LABEL_WEIGHT, dtype=float)

    weights: Optional[NDArray[np.floating]] = None

    if strategy in metrics:
        weights = np.asarray(metrics[strategy], dtype=float)
    elif strategy == WEIGHT_STRATEGIES[7]:  # "combined"
        weights = _compute_combined_label_weights(
            metrics=metrics,
            metric_coefficients=label_weighting["metric_coefficients"],
            aggregation=label_weighting["aggregation"],
            softmax_temperature=label_weighting["softmax_temperature"],
        )
    else:
        raise ValueError(
            f"Invalid weighting strategy value {strategy!r}: "
            f"supported values are {', '.join(WEIGHT_STRATEGIES)} or metric names {', '.join(metrics.keys())}"
        )

    weights = _impute_weights(
        weights=weights,
    )

    return _build_weights_array(
        n_extrema=n_values,
        indices=indices,
        weights=weights,
        default_weight=float(np.nanmedian(weights)),
    )


def _apply_label_weights(
    values: NDArray[np.floating], weights: NDArray[np.floating]
) -> NDArray[np.floating]:
    if weights.size == 0:
        return values

    if not np.isfinite(weights).all():
        return values

    if np.allclose(weights, weights[0]):
        return values

    if np.allclose(weights, DEFAULT_LABEL_WEIGHT):
        return values

    return values * weights


def apply_label_weighting(
    label: pd.Series,
    indices: list[int],
    metrics: dict[str, list[float]],
    weighting_config: dict[str, Any],
) -> tuple[pd.Series, pd.Series]:
    label_values = label.to_numpy(dtype=float)
    label_index = label.index
    n_values = len(label_values)

    weights = compute_label_weights(
        n_values=n_values,
        indices=indices,
        metrics=metrics,
        weighting_config=weighting_config,
    )

    return pd.Series(
        _apply_label_weights(label_values, weights), index=label_index
    ), pd.Series(weights, index=label_index)


def get_callable_sha256(fn: Callable[..., Any]) -> str:
    if not callable(fn):
        raise ValueError(f"Invalid fn value {type(fn).__name__!r}: must be callable")
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
        raise ValueError(
            f"Invalid fn value: unable to retrieve code object, got {type(fn).__name__!r}"
        )
    return hashlib.sha256(code.co_code).hexdigest()


_SCIENTIFIC_THRESHOLD_HIGH = 1e12
_SCIENTIFIC_THRESHOLD_LOW = 1e-6


@lru_cache(maxsize=128)
def format_number(value: int | float, significant_digits: int = 5) -> str:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        value = float(value)

    if np.isposinf(value):
        return "+∞"
    if np.isneginf(value):
        return "-∞"
    if np.isnan(value):
        return "NaN"

    abs_value = abs(value)

    if abs_value >= _SCIENTIFIC_THRESHOLD_HIGH or (
        0 < abs_value <= _SCIENTIFIC_THRESHOLD_LOW
    ):
        return f"{value:.{significant_digits - 1}e}"

    if abs_value == 0:
        return "0"

    magnitude = math.floor(math.log10(abs_value))
    precision = significant_digits - 1 - magnitude

    if precision < 0:
        factor = 10 ** (-precision)
        rounded = round(value / factor) * factor
        return f"{rounded:.0f}"

    formatted = f"{value:.{precision}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


_MAX_STR_LEN = 50
_MAX_ITEMS = 10
_MAX_DEPTH = 2


class _FormatContext:
    __slots__ = ("quote_strings", "sig_digits", "seen")

    def __init__(self, quote_strings: bool, sig_digits: int):
        self.quote_strings = quote_strings
        self.sig_digits = sig_digits
        self.seen: set[int] = set()


@singledispatch
def _format_value(value: Any, ctx: _FormatContext, depth: int) -> str:
    return repr(value)


@_format_value.register(type(None))
def _(value: None, ctx: _FormatContext, depth: int) -> str:
    return "None"


@_format_value.register(bool)
def _(value: bool, ctx: _FormatContext, depth: int) -> str:
    return str(value)


@_format_value.register(int)
def _(value: int, ctx: _FormatContext, depth: int) -> str:
    return format_number(float(value), significant_digits=ctx.sig_digits)


@_format_value.register(float)
def _(value: float, ctx: _FormatContext, depth: int) -> str:
    return format_number(value, significant_digits=ctx.sig_digits)


@_format_value.register(np.integer)
def _(value: np.integer, ctx: _FormatContext, depth: int) -> str:
    return format_number(float(value), significant_digits=ctx.sig_digits)


@_format_value.register(np.floating)
def _(value: np.floating, ctx: _FormatContext, depth: int) -> str:
    return format_number(float(value), significant_digits=ctx.sig_digits)


@_format_value.register(np.bool_)
def _(value: np.bool_, ctx: _FormatContext, depth: int) -> str:
    return str(bool(value))


@_format_value.register(str)
def _(value: str, ctx: _FormatContext, depth: int) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    if len(escaped) > _MAX_STR_LEN:
        escaped = escaped[:_MAX_STR_LEN] + "..."
    if ctx.quote_strings:
        escaped = escaped.replace("'", "\\'")
        return f"'{escaped}'"
    return escaped


def _format_collection(
    value: list | tuple | set,
    ctx: _FormatContext,
    depth: int,
    brackets: tuple[str, str],
    empty: str,
    trailing_comma: bool = False,
) -> str:
    if not value:
        return empty
    obj_id = id(value)
    if obj_id in ctx.seen:
        return f"{brackets[0]}<circular>{brackets[1]}"
    if depth >= _MAX_DEPTH:
        return f"{brackets[0]}...{brackets[1]}"
    ctx.seen.add(obj_id)
    items_iter = sorted(value, key=str) if isinstance(value, set) else value
    items = [_format_value(v, ctx, depth + 1) for v in list(items_iter)[:_MAX_ITEMS]]
    if len(value) > _MAX_ITEMS:
        items.append(f"...+{len(value) - _MAX_ITEMS}")
    content = ", ".join(items)
    if trailing_comma and len(value) == 1 and len(items) == 1:
        content += ","
    ctx.seen.discard(obj_id)
    return f"{brackets[0]}{content}{brackets[1]}"


@_format_value.register(list)
def _(value: list, ctx: _FormatContext, depth: int) -> str:
    return _format_collection(value, ctx, depth, ("[", "]"), "[]")


@_format_value.register(tuple)
def _(value: tuple, ctx: _FormatContext, depth: int) -> str:
    return _format_collection(value, ctx, depth, ("(", ")"), "()", trailing_comma=True)


@_format_value.register(set)
def _(value: set, ctx: _FormatContext, depth: int) -> str:
    return _format_collection(value, ctx, depth, ("{", "}"), "set()")


@_format_value.register(dict)
def _(value: dict, ctx: _FormatContext, depth: int) -> str:
    obj_id = id(value)
    if obj_id in ctx.seen:
        return "{<circular>}"
    if depth >= _MAX_DEPTH:
        return "{...}"
    if not value:
        return "{}"
    ctx.seen.add(obj_id)
    sep = ": " if ctx.quote_strings else "="
    items = [
        f"{k}{sep}{_format_value(v, ctx, depth + 1)}"
        for k, v in list(value.items())[:_MAX_ITEMS]
    ]
    if len(value) > _MAX_ITEMS:
        items.append(f"...+{len(value) - _MAX_ITEMS}")
    ctx.seen.discard(obj_id)
    return f"{{{', '.join(items)}}}"


@_format_value.register(np.ndarray)
def _(value: np.ndarray, ctx: _FormatContext, depth: int) -> str:
    return f"array{value.shape}"


def format_dict(
    d: dict[str, Any],
    style: Literal["dict", "params"] = "dict",
    significant_digits: int = 5,
) -> str:
    if not d:
        return "{}" if style == "dict" else ""

    ctx = _FormatContext(quote_strings=(style == "dict"), sig_digits=significant_digits)
    sep = ": " if style == "dict" else "="
    items = [f"{k}{sep}{_format_value(v, ctx, 0)}" for k, v in d.items()]
    joined = ", ".join(items)

    return f"{{{joined}}}" if style == "dict" else joined


@lru_cache(maxsize=128)
def calculate_min_extrema(
    length: int, fit_live_predictions_candles: int, min_extrema: int = 2
) -> int:
    return int(round(length / fit_live_predictions_candles) * min_extrema)


def calculate_n_extrema(series: pd.Series) -> int:
    return sp.signal.find_peaks(-series)[0].size + sp.signal.find_peaks(series)[0].size


def top_log_return(dataframe: pd.DataFrame, period: int) -> pd.Series:
    """
    Logarithmic return from rolling maximum: log(close / rolling_max).

    Measures distance below the highest close in previous `period` bars.
    Returns ≤ 0 (e.g., -0.10 ≈ -9.5% below peak). Zero when at peak.

    :param dataframe: OHLCV DataFrame with 'close' column
    :param period: Lookback window (>=1)
    :return: Log return series (≤ 0)
    """
    if period < 1:
        raise ValueError(f"Invalid period value {period!r}: must be >= 1")

    previous_close_top = (
        dataframe.get("close").rolling(period, min_periods=period).max().shift(1)
    )

    return np.log(dataframe.get("close") / previous_close_top)


def bottom_log_return(dataframe: pd.DataFrame, period: int) -> pd.Series:
    """
    Logarithmic return from rolling minimum: log(close / rolling_min).

    Measures distance above the lowest close in previous `period` bars.
    Returns ≥ 0 (e.g., +0.10 ≈ +10.5% above bottom). Zero when at bottom.

    :param dataframe: OHLCV DataFrame with 'close' column
    :param period: Lookback window (>=1)
    :return: Log return series (≥ 0)
    """
    if period < 1:
        raise ValueError(f"Invalid period value {period!r}: must be >= 1")

    previous_close_bottom = (
        dataframe.get("close").rolling(period, min_periods=period).min().shift(1)
    )

    return np.log(dataframe.get("close") / previous_close_bottom)


def price_retracement_percent(dataframe: pd.DataFrame, period: int) -> pd.Series:
    """
    Normalized position (0-1) of close within rolling high/low range, using log scale.

    Formula: log(close / low) / log(high / low)

    Returns 0 at bottom, 1 at top, 0.5 at geometric midpoint (not arithmetic).
    Example: range [100, 200] → midpoint at ~141, not 150.

    :param dataframe: OHLCV DataFrame with 'close' column
    :param period: Lookback window (>=1)
    :return: Normalized position (0 to 1)
    """
    if period < 1:
        raise ValueError(f"Invalid period value {period!r}: must be >= 1")

    previous_close_low = (
        dataframe.get("close").rolling(period, min_periods=period).min().shift(1)
    )
    previous_close_high = (
        dataframe.get("close").rolling(period, min_periods=period).max().shift(1)
    )
    denominator = np.log(previous_close_high / previous_close_low)
    return (np.log(dataframe.get("close") / previous_close_low) / denominator).where(
        ~np.isclose(denominator, 0.0), 0.0
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
        raise ValueError(f"Invalid period value {period!r}: must be even")

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
        raise ValueError(f"Invalid period value {period!r}: must be even")

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
        raise ValueError(f"Invalid period value {period!r}: must be > 0")
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
    """Return the quantile (0-1) of value within values.

    Uses percentileofscore(kind='mean') for unbiased estimation.
    Returns np.nan if values is empty. NaN values are ignored.
    """
    if values.size == 0:
        return np.nan

    return percentileofscore(values, value, kind="mean", nan_policy="omit") / 100.0


class TrendDirection(IntEnum):
    NEUTRAL = 0
    UP = 1
    DOWN = -1


def zigzag(
    df: pd.DataFrame,
    natr_period: int = 14,
    natr_multiplier: float = 9.0,
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
    thresholds: NDArray[np.floating] = natr_values * natr_multiplier
    closes_log = np.log(df.get("close").to_numpy())
    highs_log = np.log(df.get("high").to_numpy())
    lows_log = np.log(df.get("low").to_numpy())
    volumes = df.get("volume").to_numpy()

    state: TrendDirection = TrendDirection.NEUTRAL

    pivots_indices: list[int] = []
    pivots_values_log: list[float] = []
    pivots_directions: list[TrendDirection] = []
    pivots_amplitudes: list[float] = []
    pivots_amplitude_threshold_ratios: list[float] = []
    pivots_volume_rates: list[float] = []
    pivots_speeds: list[float] = []
    pivots_efficiency_ratios: list[float] = []
    pivots_volume_weighted_efficiency_ratios: list[float] = []
    last_pivot_pos: int = -1

    candidate_pivot_pos: int = -1
    candidate_pivot_value_log: float = np.nan

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

    def update_candidate_pivot(pos: int, value_log: float):
        nonlocal candidate_pivot_pos, candidate_pivot_value_log
        if 0 <= pos < n:
            candidate_pivot_pos = pos
            candidate_pivot_value_log = value_log

    def reset_candidate_pivot():
        nonlocal candidate_pivot_pos, candidate_pivot_value_log
        candidate_pivot_pos = -1
        candidate_pivot_value_log = np.nan

    def calculate_pivot_metrics(
        *,
        previous_pos: int,
        previous_value_log: float,
        current_pos: int,
        current_value_log: float,
    ) -> tuple[float, float, float]:
        if previous_pos < 0 or current_pos < 0:
            return np.nan, np.nan, np.nan
        if previous_pos >= n or current_pos >= n:
            return np.nan, np.nan, np.nan

        if not np.isfinite(previous_value_log) or not np.isfinite(current_value_log):
            return np.nan, np.nan, np.nan

        amplitude = abs(current_value_log - previous_value_log)
        if not (np.isfinite(amplitude) and amplitude >= 0):
            return np.nan, np.nan, np.nan

        start_pos = min(previous_pos, current_pos)
        end_pos = max(previous_pos, current_pos) + 1
        median_threshold_log = np.nanmedian(np.log1p(thresholds[start_pos:end_pos]))

        amplitude_threshold_ratio = (
            amplitude / (amplitude + median_threshold_log)
            if np.isfinite(median_threshold_log) and median_threshold_log > 0
            else np.nan
        )

        duration = calculate_pivot_duration(
            previous_pos=previous_pos,
            current_pos=current_pos,
        )

        if np.isfinite(duration) and duration > 0:
            speed = amplitude / duration
        else:
            speed = np.nan

        return (
            amplitude,
            amplitude_threshold_ratio,
            speed,
        )

    def calculate_pivot_duration(
        *,
        previous_pos: int,
        current_pos: int,
    ) -> float:
        if previous_pos < 0 or current_pos < 0:
            return np.nan
        if previous_pos >= n or current_pos >= n:
            return np.nan

        return float(abs(current_pos - previous_pos))

    def calculate_pivot_volume_rate(
        *,
        previous_pos: int,
        current_pos: int,
    ) -> float:
        if previous_pos < 0 or current_pos < 0:
            return np.nan
        if previous_pos >= n or current_pos >= n:
            return np.nan

        duration = calculate_pivot_duration(
            previous_pos=previous_pos,
            current_pos=current_pos,
        )
        if not np.isfinite(duration) or duration == 0:
            return np.nan

        start_pos = min(previous_pos, current_pos)
        end_pos = max(previous_pos, current_pos) + 1
        avg_volume_per_candle = np.nansum(volumes[start_pos:end_pos]) / duration
        median_volume = np.nanmedian(volumes[start_pos:end_pos])
        if (
            np.isfinite(avg_volume_per_candle)
            and avg_volume_per_candle >= 0
            and np.isfinite(median_volume)
            and median_volume > 0
        ):
            return avg_volume_per_candle / (avg_volume_per_candle + median_volume)
        return np.nan

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

        path_length = np.nansum(np.abs(np.diff(closes_log[start_pos:end_pos])))
        net_move = abs(closes_log[end_pos - 1] - closes_log[start_pos])

        if not (np.isfinite(path_length) and np.isfinite(net_move)):
            return np.nan
        if np.isclose(path_length, 0.0):
            return np.nan

        return net_move / path_length

    def calculate_pivot_volume_weighted_efficiency_ratio(
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

        volumes_slice = volumes[start_pos + 1 : end_pos]
        total_volume = np.nansum(volumes_slice)
        if not np.isfinite(total_volume) or np.isclose(total_volume, 0.0):
            return np.nan

        vw_close_diffs = np.diff(closes_log[start_pos:end_pos]) * (
            volumes_slice / total_volume
        )
        vw_path_length = np.nansum(np.abs(vw_close_diffs))
        vw_net_move = abs(np.nansum(vw_close_diffs))

        if not (np.isfinite(vw_path_length) and np.isfinite(vw_net_move)):
            return np.nan
        if np.isclose(vw_path_length, 0.0):
            return np.nan

        return vw_net_move / vw_path_length

    def add_pivot(pos: int, value_log: float, direction: TrendDirection):
        nonlocal last_pivot_pos
        if pivots_indices and indices[pos] == pivots_indices[-1]:
            return

        if (
            pivots_values_log
            and last_pivot_pos >= 0
            and len(pivots_values_log) == len(pivots_amplitudes)
        ):
            amplitude, amplitude_threshold_ratio, speed = calculate_pivot_metrics(
                previous_pos=last_pivot_pos,
                previous_value_log=pivots_values_log[-1],
                current_pos=pos,
                current_value_log=value_log,
            )
            volume_rate = calculate_pivot_volume_rate(
                previous_pos=last_pivot_pos,
                current_pos=pos,
            )
            efficiency_ratio = calculate_pivot_efficiency_ratio(
                previous_pos=last_pivot_pos,
                current_pos=pos,
            )
            volume_weighted_efficiency_ratio = (
                calculate_pivot_volume_weighted_efficiency_ratio(
                    previous_pos=last_pivot_pos,
                    current_pos=pos,
                )
            )

            pivots_amplitudes[-1] = amplitude
            pivots_amplitude_threshold_ratios[-1] = amplitude_threshold_ratio
            pivots_volume_rates[-1] = volume_rate
            pivots_speeds[-1] = speed
            pivots_efficiency_ratios[-1] = efficiency_ratio
            pivots_volume_weighted_efficiency_ratios[-1] = (
                volume_weighted_efficiency_ratio
            )

        pivots_indices.append(indices[pos])
        pivots_values_log.append(value_log)
        pivots_directions.append(direction)

        pivots_amplitudes.append(np.nan)
        pivots_amplitude_threshold_ratios.append(np.nan)
        pivots_volume_rates.append(np.nan)
        pivots_speeds.append(np.nan)
        pivots_efficiency_ratios.append(np.nan)
        pivots_volume_weighted_efficiency_ratios.append(np.nan)

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

        candidate_pivot_close_log = closes_log[candidate_pivot_pos]
        current_close_log = closes_log[pos]

        slope_close_log = (current_close_log - candidate_pivot_close_log) / (
            pos - candidate_pivot_pos
        )

        if direction == TrendDirection.UP:
            slope_ok_cache[cache_key] = slope_close_log > min_slope
        elif direction == TrendDirection.DOWN:
            slope_ok_cache[cache_key] = slope_close_log < -min_slope
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
    initial_high_log = highs_log[initial_high_pos]
    initial_low_log = lows_log[initial_low_pos]
    for i in range(start_pos + 1, n):
        if highs_log[i] > initial_high_log:
            initial_high_log, initial_high_pos = highs_log[i], i
        if lows_log[i] < initial_low_log:
            initial_low_log, initial_low_pos = lows_log[i], i

        initial_move_from_high = abs(lows_log[i] - highs_log[initial_high_pos])
        initial_move_from_low = abs(highs_log[i] - lows_log[initial_low_pos])
        is_initial_high_move_significant: bool = initial_move_from_high >= np.log1p(
            thresholds[initial_high_pos]
        )
        is_initial_low_move_significant: bool = initial_move_from_low >= np.log1p(
            thresholds[initial_low_pos]
        )
        if is_initial_high_move_significant and is_initial_low_move_significant:
            if initial_move_from_high > initial_move_from_low:
                add_pivot(initial_high_pos, initial_high_log, TrendDirection.UP)
                state = TrendDirection.DOWN
                break
            else:
                add_pivot(initial_low_pos, initial_low_log, TrendDirection.DOWN)
                state = TrendDirection.UP
                break
        else:
            if is_initial_high_move_significant:
                add_pivot(initial_high_pos, initial_high_log, TrendDirection.UP)
                state = TrendDirection.DOWN
                break
            elif is_initial_low_move_significant:
                add_pivot(initial_low_pos, initial_low_log, TrendDirection.DOWN)
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
        if state == TrendDirection.UP:
            if (
                np.isnan(candidate_pivot_value_log)
                or highs_log[i] > highs_log[candidate_pivot_pos]
            ):
                update_candidate_pivot(i, highs_log[i])
            move_down = abs(lows_log[i] - candidate_pivot_value_log)
            if move_down >= np.log1p(
                thresholds[candidate_pivot_pos]
            ) and is_pivot_confirmed(i, candidate_pivot_pos, TrendDirection.DOWN):
                add_pivot(
                    candidate_pivot_pos,
                    highs_log[candidate_pivot_pos],
                    TrendDirection.UP,
                )
                state = TrendDirection.DOWN

        elif state == TrendDirection.DOWN:
            if (
                np.isnan(candidate_pivot_value_log)
                or lows_log[i] < lows_log[candidate_pivot_pos]
            ):
                update_candidate_pivot(i, lows_log[i])
            move_up = abs(highs_log[i] - candidate_pivot_value_log)
            if move_up >= np.log1p(
                thresholds[candidate_pivot_pos]
            ) and is_pivot_confirmed(i, candidate_pivot_pos, TrendDirection.UP):
                add_pivot(
                    candidate_pivot_pos,
                    lows_log[candidate_pivot_pos],
                    TrendDirection.DOWN,
                )
                state = TrendDirection.UP

    return (
        pivots_indices,
        pivots_values_log,
        pivots_directions,
        pivots_amplitudes,
        pivots_amplitude_threshold_ratios,
        pivots_volume_rates,
        pivots_speeds,
        pivots_efficiency_ratios,
        pivots_volume_weighted_efficiency_ratios,
    )


Regressor = Literal[
    "xgboost", "lightgbm", "histgradientboostingregressor", "ngboost", "catboost"
]
REGRESSORS: Final[tuple[Regressor, ...]] = (
    "xgboost",
    "lightgbm",
    "histgradientboostingregressor",
    "ngboost",
    "catboost",
)

RegressorCallback = Union[Callable[..., Any], XGBoostTrainingCallback]

_EARLY_STOPPING_ROUNDS_DEFAULT: Final[int] = 50

_CATBOOST_GPU_RSM_LOSS_FUNCTIONS: Final[tuple[str, ...]] = (
    "PairLogit",
    "PairLogitPairwise",
)

_CATBOOST_GPU_PAIRWISE_LOSS_FUNCTIONS: Final[tuple[str, ...]] = (
    "YetiRank",
    "PairLogitPairwise",
    "QueryCrossEntropy",
)


def get_ngboost_dist(dist_name: str) -> type:
    from ngboost.distns import Exponential, Laplace, LogNormal, Normal, T

    dist_map = {
        "normal": Normal,
        "lognormal": LogNormal,
        "exponential": Exponential,
        "laplace": Laplace,
        "t": T,
    }

    if dist_name not in dist_map:
        raise ValueError(
            f"Invalid dist_name {dist_name!r}: supported values are {', '.join(dist_map.keys())}"
        )

    return dist_map[dist_name]


def fit_regressor(
    regressor: Regressor,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_weights: NDArray[np.floating],
    eval_set: Optional[list[tuple[pd.DataFrame, pd.DataFrame]]],
    eval_weights: Optional[list[NDArray[np.floating]]],
    model_training_parameters: dict[str, Any],
    init_model: Any = None,
    callbacks: Optional[list[RegressorCallback]] = None,
    model_path: Optional[Path] = None,
    trial: Optional[optuna.trial.Trial] = None,
) -> Any:
    """Fit a regressor model."""
    fit_callbacks = list(callbacks) if callbacks else []

    has_eval_set = (
        eval_set is not None
        and len(eval_set) > 0
        and eval_weights is not None
        and len(eval_weights) > 0
    )
    if not has_eval_set:
        eval_set = None
        eval_weights = None

    if regressor == REGRESSORS[0]:  # "xgboost"
        from xgboost import XGBRegressor
        from xgboost.callback import EarlyStopping

        model_training_parameters.setdefault("random_state", 1)

        early_stopping_rounds = None
        if has_eval_set:
            early_stopping_rounds = model_training_parameters.pop(
                "early_stopping_rounds", _EARLY_STOPPING_ROUNDS_DEFAULT
            )
        else:
            model_training_parameters.pop("early_stopping_rounds", None)

        if early_stopping_rounds is not None:
            fit_callbacks.append(
                EarlyStopping(
                    rounds=early_stopping_rounds,
                    metric_name="rmse",
                    data_name="validation_0",
                    save_best=True,
                )
            )

        if trial is not None:
            model_training_parameters["random_state"] = (
                model_training_parameters["random_state"] + trial.number
            )
            if has_eval_set:
                fit_callbacks.append(
                    optuna.integration.XGBoostPruningCallback(
                        trial, "validation_0-rmse"
                    )
                )

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            callbacks=fit_callbacks if fit_callbacks else None,
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
        from lightgbm import LGBMRegressor, early_stopping

        model_training_parameters.setdefault("seed", 1)

        early_stopping_rounds = None
        if has_eval_set:
            early_stopping_rounds = model_training_parameters.pop(
                "early_stopping_rounds", _EARLY_STOPPING_ROUNDS_DEFAULT
            )
        else:
            model_training_parameters.pop("early_stopping_rounds", None)

        if early_stopping_rounds is not None:
            fit_callbacks.append(
                early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    first_metric_only=True,
                    verbose=False,
                )
            )

        if trial is not None:
            model_training_parameters["seed"] = (
                model_training_parameters["seed"] + trial.number
            )
            if has_eval_set:
                fit_callbacks.append(
                    optuna.integration.LightGBMPruningCallback(
                        trial, "rmse", valid_name="valid_0"
                    )
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
            callbacks=fit_callbacks if fit_callbacks else None,
        )
    elif regressor == REGRESSORS[2]:  # "histgradientboostingregressor"
        from sklearn.ensemble import HistGradientBoostingRegressor

        model_training_parameters.setdefault("random_state", 1)
        model_training_parameters.setdefault("loss", "squared_error")
        model_training_parameters.pop("early_stopping", None)
        model_training_parameters.pop("n_jobs", None)
        model_training_parameters.pop("l2_regularization_zero", None)

        early_stopping_rounds = model_training_parameters.pop(
            "early_stopping_rounds", None
        )
        if "n_iter_no_change" not in model_training_parameters:
            if early_stopping_rounds is not None:
                model_training_parameters["n_iter_no_change"] = early_stopping_rounds
            else:
                model_training_parameters["n_iter_no_change"] = (
                    _EARLY_STOPPING_ROUNDS_DEFAULT
                )

        verbosity = model_training_parameters.pop("verbosity", None)
        if "verbose" not in model_training_parameters and verbosity is not None:
            model_training_parameters["verbose"] = verbosity

        if trial is not None:
            model_training_parameters["random_state"] = (
                model_training_parameters["random_state"] + trial.number
            )

        X_val = None
        y_val = None
        if has_eval_set:
            X_val, y_val = eval_set[0]
            y_val = y_val.to_numpy().ravel()

        sample_weight_val = None
        if eval_weights is not None and len(eval_weights) > 0:
            sample_weight_val = eval_weights[0]

        model = HistGradientBoostingRegressor(
            early_stopping=True,
            scoring="neg_root_mean_squared_error",
            **model_training_parameters,
        )
        model.fit(
            X=X,
            y=y.to_numpy().ravel(),
            sample_weight=train_weights,
            X_val=X_val,
            y_val=y_val,
            sample_weight_val=sample_weight_val,
        )
    elif regressor == REGRESSORS[3]:  # "ngboost"
        from ngboost import NGBRegressor
        from sklearn.tree import DecisionTreeRegressor

        model_training_parameters.setdefault("random_state", 1)

        verbosity = model_training_parameters.pop("verbosity", None)
        if "verbose" not in model_training_parameters and verbosity is not None:
            model_training_parameters["verbose"] = verbosity

        model_training_parameters.pop("n_jobs", None)

        early_stopping_rounds = None
        if has_eval_set:
            early_stopping_rounds = model_training_parameters.pop(
                "early_stopping_rounds", _EARLY_STOPPING_ROUNDS_DEFAULT
            )
        else:
            model_training_parameters.pop("early_stopping_rounds", None)

        if trial is not None:
            model_training_parameters["random_state"] = (
                model_training_parameters["random_state"] + trial.number
            )

        dist = model_training_parameters.pop("dist", "lognormal")

        X_val = None
        Y_val = None
        val_sample_weight = None
        if has_eval_set:
            X_val, Y_val = eval_set[0]
            Y_val = Y_val.to_numpy().ravel()
            if eval_weights is not None and len(eval_weights) > 0:
                val_sample_weight = eval_weights[0]

        model = NGBRegressor(
            Dist=get_ngboost_dist(dist),
            Base=DecisionTreeRegressor(
                criterion="friedman_mse",
                max_depth=model_training_parameters.pop("max_depth", None),
                min_samples_split=model_training_parameters.pop("min_samples_split", 2),
                min_samples_leaf=model_training_parameters.pop("min_samples_leaf", 1),
            ),
            **model_training_parameters,
        )

        model.fit(
            X=X,
            Y=y.to_numpy().ravel(),
            sample_weight=train_weights,
            X_val=X_val,
            Y_val=Y_val,
            val_sample_weight=val_sample_weight,
            early_stopping_rounds=early_stopping_rounds,
        )
    elif regressor == REGRESSORS[4]:  # "catboost"
        from catboost import CatBoostRegressor, Pool

        model_training_parameters.setdefault("random_seed", 1)
        model_training_parameters.setdefault("loss_function", "RMSE")

        if model_path is not None and "train_dir" not in model_training_parameters:
            if trial is not None:
                trial_path = model_path / f"hp_trial_{trial.number}"
                trial_path.mkdir(parents=True, exist_ok=True)
                model_training_parameters["train_dir"] = str(
                    trial_path / "catboost_info"
                )
            else:
                model_training_parameters["train_dir"] = str(
                    model_path / "catboost_info"
                )

        task_type = model_training_parameters.get("task_type", "CPU")
        loss_function = model_training_parameters.get("loss_function", "RMSE")
        if task_type == "GPU":
            model_training_parameters.setdefault("max_ctr_complexity", 4)
            model_training_parameters.pop("n_jobs", None)
            if loss_function not in _CATBOOST_GPU_RSM_LOSS_FUNCTIONS:
                model_training_parameters.pop("rsm", None)
        else:
            n_jobs = model_training_parameters.pop("n_jobs", None)
            if n_jobs is not None:
                model_training_parameters.setdefault("thread_count", n_jobs)
            model_training_parameters.setdefault("max_ctr_complexity", 2)

        early_stopping_rounds = None
        if has_eval_set:
            early_stopping_rounds = model_training_parameters.pop(
                "early_stopping_rounds", _EARLY_STOPPING_ROUNDS_DEFAULT
            )
        else:
            model_training_parameters.pop("early_stopping_rounds", None)

        verbosity = model_training_parameters.pop("verbosity", None)
        if "verbose" not in model_training_parameters and verbosity is not None:
            model_training_parameters["verbose"] = verbosity

        if trial is not None:
            model_training_parameters["random_seed"] = (
                model_training_parameters["random_seed"] + trial.number
            )

        pruning_callback = None
        if trial is not None and has_eval_set and task_type != "GPU":
            pruning_callback = optuna.integration.CatBoostPruningCallback(trial, "RMSE")
            fit_callbacks.append(pruning_callback)

        model = CatBoostRegressor(**model_training_parameters)

        model.fit(
            Pool(data=X, label=y, weight=train_weights),
            eval_set=(
                Pool(
                    data=eval_set[0][0],
                    label=eval_set[0][1],
                    weight=eval_weights[0],
                )
                if has_eval_set and eval_set is not None and eval_weights is not None
                else None
            ),
            early_stopping_rounds=early_stopping_rounds
            if early_stopping_rounds is not None and has_eval_set
            else None,
            use_best_model=True
            if early_stopping_rounds is not None and has_eval_set
            else False,
            callbacks=fit_callbacks if fit_callbacks else None,
        )

        if pruning_callback is not None:
            pruning_callback.check_pruned()
    else:
        raise ValueError(
            f"Invalid regressor value {regressor!r}: supported values are {', '.join(REGRESSORS)}"
        )
    return model


def eval_set_and_weights(
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    test_weights: NDArray[np.floating],
    test_size: float,
) -> tuple[
    Optional[list[tuple[pd.DataFrame, pd.DataFrame]]],
    Optional[list[NDArray[np.floating]]],
]:
    if test_size <= 0:
        return None, None

    return [(X_test, y_test)], [test_weights]


def _build_int_range(
    frange: tuple[float, float],
    min_val: int = 1,
) -> tuple[int, int]:
    lo, hi = math.ceil(frange[0]), math.floor(frange[1])
    if lo > hi:
        lo = hi = max(min_val, int(round((frange[0] + frange[1]) / 2)))
    return max(min_val, lo), max(min_val, hi)


def _optuna_suggest_int_from_range(
    trial: optuna.trial.Trial,
    name: str,
    frange: tuple[float, float],
    *,
    min_val: int = 1,
    log: bool = False,
) -> int:
    int_range = _build_int_range(frange, min_val=min_val)
    return trial.suggest_int(name, int_range[0], int_range[1], log=log)


def get_optuna_study_model_parameters(
    trial: optuna.trial.Trial,
    regressor: Regressor,
    model_training_best_parameters: dict[str, Any],
    model_training_parameters: dict[str, Any],
    space_reduction: bool,
    space_fraction: float,
) -> dict[str, Any]:
    if regressor not in set(REGRESSORS):
        raise ValueError(
            f"Invalid regressor value {regressor!r}: supported values are {', '.join(REGRESSORS)}"
        )
    if not isinstance(space_fraction, (int, float)) or not (
        0.0 <= space_fraction <= 1.0
    ):
        raise ValueError(
            f"Invalid space_fraction: must be in range [0, 1], got {space_fraction!r}"
        )

    def _build_ranges(
        default_ranges: dict[str, tuple[float, float]],
        log_scaled_params: set[str],
    ) -> dict[str, tuple[float, float]]:
        ranges = copy.deepcopy(default_ranges)
        if space_reduction and model_training_best_parameters:
            for param, (default_min, default_max) in default_ranges.items():
                center_value = model_training_best_parameters.get(param)
                if center_value is None:
                    # Use geometric mean for log-scaled params
                    if (
                        param in log_scaled_params
                        and default_min > 0
                        and default_max > 0
                    ):
                        center_value = math.sqrt(default_min * default_max)
                    else:
                        center_value = midpoint(default_min, default_max)
                if not isinstance(center_value, (int, float)) or not np.isfinite(
                    center_value
                ):
                    continue
                if param in log_scaled_params:
                    if center_value <= 0:
                        continue
                    # Proportional reduction in log-space
                    factor = math.pow(default_max / default_min, space_fraction / 2)
                    new_min = center_value / factor
                    new_max = center_value * factor
                else:
                    margin = (default_max - default_min) * space_fraction / 2
                    new_min = center_value - margin
                    new_max = center_value + margin
                param_min = max(default_min, new_min)
                param_max = min(default_max, new_max)
                if param_min < param_max:
                    ranges[param] = (param_min, param_max)
        return ranges

    if regressor == REGRESSORS[0]:  # "xgboost"
        # Parameter order: boosting -> tree structure -> leaf constraints ->
        #                  sampling -> regularization -> binning
        default_ranges: dict[str, tuple[float, float]] = {
            # Boosting/Training
            "n_estimators": (50, 3000),
            "learning_rate": (0.005, 0.3),
            # Tree structure
            "max_depth": (3, 10),
            "max_leaves": (16, 512),
            # Leaf constraints
            "min_child_weight": (1.0, 200.0),
            # Sampling
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "colsample_bylevel": (0.5, 1.0),
            "colsample_bynode": (0.5, 1.0),
            # Regularization
            "reg_alpha": (1e-8, 10.0),
            "reg_lambda": (1e-8, 10.0),
            "gamma": (1e-8, 1.0),
            # Binning
            "max_bin": (63, 255),
        }
        log_scaled_params = {
            "n_estimators",
            "learning_rate",
            "min_child_weight",
            "max_leaves",
            "reg_alpha",
            "reg_lambda",
            "gamma",
        }

        ranges = _build_ranges(default_ranges, log_scaled_params)

        booster = trial.suggest_categorical("booster", ["gbtree", "dart"])
        grow_policy = trial.suggest_categorical(
            "grow_policy", ["depthwise", "lossguide"]
        )

        params: dict[str, Any] = {
            # Boosting/Training
            "booster": booster,
            "n_estimators": _optuna_suggest_int_from_range(
                trial, "n_estimators", ranges["n_estimators"], min_val=1, log=True
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                ranges["learning_rate"][0],
                ranges["learning_rate"][1],
                log=True,
            ),
            # Tree structure
            "grow_policy": grow_policy,
            **(
                {
                    "max_depth": 0,
                    "max_leaves": _optuna_suggest_int_from_range(
                        trial, "max_leaves", ranges["max_leaves"], min_val=2, log=True
                    ),
                }
                if grow_policy == "lossguide"
                else {
                    "max_depth": _optuna_suggest_int_from_range(
                        trial, "max_depth", ranges["max_depth"], min_val=1
                    ),
                }
            ),
            # Leaf constraints
            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                ranges["min_child_weight"][0],
                ranges["min_child_weight"][1],
                log=True,
            ),
            # Sampling
            "subsample": trial.suggest_float(
                "subsample", ranges["subsample"][0], ranges["subsample"][1]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                ranges["colsample_bytree"][0],
                ranges["colsample_bytree"][1],
            ),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel",
                ranges["colsample_bylevel"][0],
                ranges["colsample_bylevel"][1],
            ),
            "colsample_bynode": trial.suggest_float(
                "colsample_bynode",
                ranges["colsample_bynode"][0],
                ranges["colsample_bynode"][1],
            ),
            # Regularization
            "reg_alpha": trial.suggest_float(
                "reg_alpha", ranges["reg_alpha"][0], ranges["reg_alpha"][1], log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", ranges["reg_lambda"][0], ranges["reg_lambda"][1], log=True
            ),
            "gamma": trial.suggest_float(
                "gamma", ranges["gamma"][0], ranges["gamma"][1], log=True
            ),
            # Binning
            "max_bin": _optuna_suggest_int_from_range(
                trial, "max_bin", ranges["max_bin"], min_val=2
            ),
        }

        if booster == "dart":
            params["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"]
            )
            params["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]
            )
            params["rate_drop"] = trial.suggest_float("rate_drop", 0.0, 0.5)
            params["skip_drop"] = trial.suggest_float("skip_drop", 0.0, 0.7)
            params["one_drop"] = trial.suggest_categorical("one_drop", [False, True])

        return params

    elif regressor == REGRESSORS[1]:  # "lightgbm"
        # Parameter order: boosting -> tree structure -> leaf constraints ->
        #                  sampling -> regularization -> binning
        default_ranges: dict[str, tuple[float, float]] = {
            # Boosting/Training
            "n_estimators": (50, 3000),
            "learning_rate": (0.005, 0.3),
            # Tree structure
            "num_leaves": (8, 512),
            # Leaf constraints
            "min_child_weight": (1e-5, 10.0),
            "min_child_samples": (1, 200),
            "min_split_gain": (1e-8, 1.0),
            # Sampling
            "subsample": (0.4, 1.0),
            "subsample_freq": (1, 7),
            "colsample_bytree": (0.4, 1.0),
            # Regularization
            "reg_alpha": (1e-8, 10.0),
            "reg_lambda": (1e-8, 10.0),
            # Binning
            "max_bin": (63, 255),
        }
        log_scaled_params = {
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "min_child_weight",
            "min_split_gain",
            "reg_alpha",
            "reg_lambda",
        }

        ranges = _build_ranges(default_ranges, log_scaled_params)

        boosting_type = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])

        params: dict[str, Any] = {
            # Boosting/Training
            "boosting_type": boosting_type,
            "n_estimators": _optuna_suggest_int_from_range(
                trial, "n_estimators", ranges["n_estimators"], min_val=1, log=True
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                ranges["learning_rate"][0],
                ranges["learning_rate"][1],
                log=True,
            ),
            # Tree structure
            "num_leaves": _optuna_suggest_int_from_range(
                trial, "num_leaves", ranges["num_leaves"], min_val=2, log=True
            ),
            # Leaf constraints
            "min_child_weight": trial.suggest_float(
                "min_child_weight",
                ranges["min_child_weight"][0],
                ranges["min_child_weight"][1],
                log=True,
            ),
            "min_child_samples": _optuna_suggest_int_from_range(
                trial, "min_child_samples", ranges["min_child_samples"], min_val=1
            ),
            "min_split_gain": trial.suggest_float(
                "min_split_gain",
                ranges["min_split_gain"][0],
                ranges["min_split_gain"][1],
                log=True,
            ),
            # Sampling
            "subsample": trial.suggest_float(
                "subsample", ranges["subsample"][0], ranges["subsample"][1]
            ),
            "subsample_freq": _optuna_suggest_int_from_range(
                trial, "subsample_freq", ranges["subsample_freq"], min_val=1
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                ranges["colsample_bytree"][0],
                ranges["colsample_bytree"][1],
            ),
            # Regularization
            "reg_alpha": trial.suggest_float(
                "reg_alpha", ranges["reg_alpha"][0], ranges["reg_alpha"][1], log=True
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", ranges["reg_lambda"][0], ranges["reg_lambda"][1], log=True
            ),
            # Binning
            "max_bin": _optuna_suggest_int_from_range(
                trial, "max_bin", ranges["max_bin"], min_val=2
            ),
        }

        if boosting_type == "dart":
            params["xgboost_dart_mode"] = trial.suggest_categorical(
                "xgboost_dart_mode", [False, True]
            )
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.0, 0.5)
            params["skip_drop"] = trial.suggest_float("skip_drop", 0.0, 0.7)
            params["max_drop"] = trial.suggest_int("max_drop", 10, 100)
            params["uniform_drop"] = trial.suggest_categorical(
                "uniform_drop", [False, True]
            )

        return params

    elif regressor == REGRESSORS[2]:  # "histgradientboostingregressor"
        # Parameter order: boosting -> tree structure -> leaf constraints ->
        #                  sampling -> regularization -> binning -> early stopping
        default_ranges: dict[str, tuple[float, float]] = {
            # Boosting/Training
            "max_iter": (100, 2000),
            "learning_rate": (0.01, 0.3),
            # Tree structure
            "max_leaf_nodes": (15, 255),
            # Leaf constraints
            "min_samples_leaf": (5, 150),
            # Sampling
            "max_features": (0.5, 1.0),
            # Regularization
            "l2_regularization": (1e-8, 10.0),
            # Binning
            "max_bins": (63, 255),
            # Early stopping
            "n_iter_no_change": (5, 20),
            "tol": (1e-7, 1e-3),
        }
        log_scaled_params = {
            "max_iter",
            "learning_rate",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
            "tol",
        }

        ranges = _build_ranges(default_ranges, log_scaled_params)

        l2_regularization_zero = trial.suggest_categorical(
            "l2_regularization_zero", [False, True]
        )
        if l2_regularization_zero:
            l2_regularization = 0.0
        else:
            l2_regularization = trial.suggest_float(
                "l2_regularization",
                ranges["l2_regularization"][0],
                ranges["l2_regularization"][1],
                log=True,
            )

        max_depth = trial.suggest_categorical(
            "max_depth", [None, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
        )

        max_leaf_nodes_range = ranges["max_leaf_nodes"]
        if isinstance(max_depth, int) and max_depth > 0:
            max_leaf_nodes_range = (
                max_leaf_nodes_range[0],
                min(max_leaf_nodes_range[1], float(2**max_depth)),
            )

        return {
            # Boosting/Training
            "max_iter": _optuna_suggest_int_from_range(
                trial, "max_iter", ranges["max_iter"], min_val=1, log=True
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                ranges["learning_rate"][0],
                ranges["learning_rate"][1],
                log=True,
            ),
            # Tree structure
            "max_depth": max_depth,
            "max_leaf_nodes": _optuna_suggest_int_from_range(
                trial, "max_leaf_nodes", max_leaf_nodes_range, min_val=2, log=True
            ),
            # Leaf constraints
            "min_samples_leaf": _optuna_suggest_int_from_range(
                trial,
                "min_samples_leaf",
                ranges["min_samples_leaf"],
                min_val=1,
                log=True,
            ),
            # Sampling
            "max_features": trial.suggest_float(
                "max_features",
                ranges["max_features"][0],
                ranges["max_features"][1],
            ),
            # Regularization
            "l2_regularization": l2_regularization,
            # Binning
            "max_bins": _optuna_suggest_int_from_range(
                trial, "max_bins", ranges["max_bins"], min_val=2
            ),
            # Early stopping
            "n_iter_no_change": _optuna_suggest_int_from_range(
                trial, "n_iter_no_change", ranges["n_iter_no_change"], min_val=1
            ),
            "tol": trial.suggest_float(
                "tol",
                ranges["tol"][0],
                ranges["tol"][1],
                log=True,
            ),
        }

    elif regressor == REGRESSORS[3]:  # "ngboost"
        # Parameter order: boosting -> tree structure -> sampling -> early stopping -> distribution
        default_ranges: dict[str, tuple[float, float]] = {
            # Boosting/Training
            "n_estimators": (200, 2000),
            "learning_rate": (0.001, 0.3),
            # Tree structure
            "max_depth": (2, 6),
            "min_samples_split": (2, 20),
            "min_samples_leaf": (1, 8),
            # Sampling
            "minibatch_frac": (0.6, 1.0),
            "col_sample": (0.4, 1.0),
            # Early stopping
            "tol": (1e-5, 1e-3),
        }
        log_scaled_params = {
            "n_estimators",
            "learning_rate",
            "tol",
        }

        ranges = _build_ranges(default_ranges, log_scaled_params)

        return {
            # Boosting/Training
            "n_estimators": _optuna_suggest_int_from_range(
                trial, "n_estimators", ranges["n_estimators"], min_val=1, log=True
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                ranges["learning_rate"][0],
                ranges["learning_rate"][1],
                log=True,
            ),
            # Tree structure
            "max_depth": _optuna_suggest_int_from_range(
                trial, "max_depth", ranges["max_depth"], min_val=1
            ),
            "min_samples_split": _optuna_suggest_int_from_range(
                trial, "min_samples_split", ranges["min_samples_split"], min_val=2
            ),
            "min_samples_leaf": _optuna_suggest_int_from_range(
                trial, "min_samples_leaf", ranges["min_samples_leaf"], min_val=1
            ),
            # Sampling
            "minibatch_frac": trial.suggest_float(
                "minibatch_frac",
                ranges["minibatch_frac"][0],
                ranges["minibatch_frac"][1],
            ),
            "col_sample": trial.suggest_float(
                "col_sample",
                ranges["col_sample"][0],
                ranges["col_sample"][1],
            ),
            # Early stopping
            "tol": trial.suggest_float(
                "tol",
                ranges["tol"][0],
                ranges["tol"][1],
                log=True,
            ),
            # Distribution
            "dist": trial.suggest_categorical("dist", ["normal", "lognormal"]),
        }

    elif regressor == REGRESSORS[4]:  # "catboost"
        # Parameter order: boosting -> tree structure -> regularization -> sampling
        task_type = model_training_parameters.get("task_type", "CPU")
        loss_function = model_training_parameters.get("loss_function", "RMSE")

        if (
            task_type == "GPU"
            and loss_function in _CATBOOST_GPU_PAIRWISE_LOSS_FUNCTIONS
        ):
            max_depth = 8
        elif task_type == "GPU":
            max_depth = 12
        else:  # CPU
            max_depth = 10

        if task_type == "GPU":
            default_ranges: dict[str, tuple[float, float]] = {
                # Boosting/Training
                "iterations": (100, 2000),
                "learning_rate": (0.001, 0.3),
                # Tree structure
                "depth": (4, max_depth),
                "min_data_in_leaf": (1, 20),
                "border_count": (128, 255),
                "max_ctr_complexity": (2, 6),
                # Regularization
                "l2_leaf_reg": (1, 10),
                "model_size_reg": (0.0, 1.0),
                # Sampling/Randomization
                "bagging_temperature": (0, 10),
                "random_strength": (1, 20),
                "rsm": (0.5, 1.0),
                "subsample": (0.6, 1.0),
            }
            bootstrap_options = ["Bayesian", "Bernoulli"]
            boosting_type_options = ["Plain"]
        else:  # CPU
            default_ranges: dict[str, tuple[float, float]] = {
                # Boosting/Training
                "iterations": (100, 2000),
                "learning_rate": (0.001, 0.3),
                # Tree structure
                "depth": (4, max_depth),
                "min_data_in_leaf": (1, 20),
                # Regularization
                "l2_leaf_reg": (1, 10),
                "model_size_reg": (0.0, 1.0),
                # Sampling/Randomization
                "bagging_temperature": (0, 10),
                "random_strength": (1, 20),
                "rsm": (0.5, 1.0),
                "subsample": (0.6, 1.0),
            }
            bootstrap_options = ["Bayesian", "Bernoulli", "MVS"]
            boosting_type_options = ["Plain", "Ordered"]

        log_scaled_params = {
            "iterations",
            "learning_rate",
            "l2_leaf_reg",
            "random_strength",
        }

        ranges = _build_ranges(default_ranges, log_scaled_params)

        boosting_type = trial.suggest_categorical(
            "boosting_type", boosting_type_options
        )
        bootstrap_type = trial.suggest_categorical("bootstrap_type", bootstrap_options)
        grow_policy = trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
        )
        if boosting_type == "Ordered" and grow_policy != "SymmetricTree":
            raise optuna.TrialPruned(
                "Ordered boosting is not supported for nonsymmetric trees"
            )

        params: dict[str, Any] = {
            # Boosting/Training
            "boosting_type": boosting_type,
            "iterations": _optuna_suggest_int_from_range(
                trial, "iterations", ranges["iterations"], min_val=1, log=True
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                ranges["learning_rate"][0],
                ranges["learning_rate"][1],
                log=True,
            ),
            # Tree structure
            "depth": _optuna_suggest_int_from_range(
                trial, "depth", ranges["depth"], min_val=1
            ),
            "min_data_in_leaf": _optuna_suggest_int_from_range(
                trial, "min_data_in_leaf", ranges["min_data_in_leaf"], min_val=1
            ),
            "grow_policy": grow_policy,
            # Regularization
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg",
                ranges["l2_leaf_reg"][0],
                ranges["l2_leaf_reg"][1],
                log=True,
            ),
            "model_size_reg": trial.suggest_float(
                "model_size_reg",
                ranges["model_size_reg"][0],
                ranges["model_size_reg"][1],
            ),
            # Sampling/Randomization
            "bootstrap_type": bootstrap_type,
            "leaf_estimation_method": trial.suggest_categorical(
                "leaf_estimation_method", ["Newton", "Gradient"]
            ),
            "random_strength": trial.suggest_float(
                "random_strength",
                ranges["random_strength"][0],
                ranges["random_strength"][1],
                log=True,
            ),
        }

        if task_type == "CPU" or loss_function in _CATBOOST_GPU_RSM_LOSS_FUNCTIONS:
            params["rsm"] = trial.suggest_float(
                "rsm",
                ranges["rsm"][0],
                ranges["rsm"][1],
            )

        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature",
                ranges["bagging_temperature"][0],
                ranges["bagging_temperature"][1],
            )

        if bootstrap_type in ["Bernoulli", "MVS"]:
            params["subsample"] = trial.suggest_float(
                "subsample",
                ranges["subsample"][0],
                ranges["subsample"][1],
            )

        if task_type == "GPU":
            params["border_count"] = _optuna_suggest_int_from_range(
                trial, "border_count", ranges["border_count"], min_val=1
            )
            params["max_ctr_complexity"] = _optuna_suggest_int_from_range(
                trial,
                "max_ctr_complexity",
                ranges["max_ctr_complexity"],
                min_val=1,
            )

        return params
    else:
        raise ValueError(
            f"Invalid regressor value {regressor!r}: supported values are {', '.join(REGRESSORS)}"
        )


@lru_cache(maxsize=128)
def largest_divisor_to_step(integer: int, step: int) -> Optional[int]:
    if not isinstance(integer, int) or integer <= 0:
        raise ValueError(
            f"Invalid integer value {integer!r}: must be a positive integer"
        )
    if not isinstance(step, int) or step <= 0:
        raise ValueError(f"Invalid step value {step!r}: must be a positive integer")

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
    values = series.to_numpy()
    if values.size == 0:
        return np.nan
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.nan
    if np.isclose(alpha, 0.0):
        return float(np.nanmean(values))
    scaled_values = alpha * values
    max_scaled_values = np.nanmax(scaled_values)
    if not np.isfinite(max_scaled_values):
        return values[np.nanargmax(scaled_values)]
    shifted_exponentials = np.exp(scaled_values - max_scaled_values)
    sum_exponentials = np.nansum(shifted_exponentials)
    if not np.isfinite(sum_exponentials) or sum_exponentials <= 0.0:
        return values[np.nanargmax(scaled_values)]
    return nan_average(values, weights=shifted_exponentials)


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
            f"Invalid label_period_candles range: min must be <= max, "
            f"got min={min_label_period_candles!r}, max={max_label_period_candles!r}"
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
        raise ValueError(f"Invalid value {value!r}: must be an integer or float")
    if not isinstance(step, int) or step <= 0:
        raise ValueError(f"Invalid step value {step!r}: must be a positive integer")
    if isinstance(value, (int, np.integer)):
        q, r = divmod(value, step)
        twice_r = r * 2
        if twice_r < step:
            return q * step
        if twice_r > step:
            return (q + 1) * step
        return int(round(value / step) * step)
    if not np.isfinite(value):
        raise ValueError(f"Invalid value {value!r}: must be finite")
    return int(round(float(value) / step) * step)


@lru_cache(maxsize=128)
def ceil_to_step(value: float | int, step: int) -> int:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Invalid value {value!r}: must be an integer or float")
    if not isinstance(step, int) or step <= 0:
        raise ValueError(f"Invalid step value {step!r}: must be a positive integer")
    if isinstance(value, (int, np.integer)):
        return int(-(-int(value) // step) * step)
    if not np.isfinite(value):
        raise ValueError(f"Invalid value {value!r}: must be finite")
    return int(math.ceil(float(value) / step) * step)


@lru_cache(maxsize=128)
def floor_to_step(value: float | int, step: int) -> int:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Invalid value {value!r}: must be an integer or float")
    if not isinstance(step, int) or step <= 0:
        raise ValueError(f"Invalid step value {step!r}: must be a positive integer")
    if isinstance(value, (int, np.integer)):
        return int((int(value) // step) * step)
    if not np.isfinite(value):
        raise ValueError(f"Invalid value {value!r}: must be finite")
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
        raise ValueError(
            f"Invalid {name}: defaults must be numeric, "
            f"got min={type(default_min).__name__!r}, max={type(default_max).__name__!r}"
        )
    if default_min > default_max or (not allow_equal and default_min == default_max):
        raise ValueError(
            f"Invalid {name}: defaults ordering must have min < max, "
            f"got min={default_min!r}, max={default_max!r}"
        )

    def _validate_component(
        value: float | int | None, name: str, default_value: float | int
    ) -> float | int:
        constraints = []
        if finite_only:
            constraints.append("finite")
        if non_negative:
            constraints.append("non-negative")
        constraints.append("numeric")
        constraint_str = " ".join(constraints)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or (finite_only and not np.isfinite(value))
            or (non_negative and value < 0)
        ):
            logger.warning(
                f"Invalid {name} {value!r}: must be {constraint_str}, using default {default_value!r}"
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
            f"Invalid {name} ordering: must have {min_name} < {max_name}, "
            f"got {min_name}={sanitized_min!r}, {max_name}={sanitized_max!r}, "
            f"using defaults {default_min!r}, {default_max!r}"
        )
        sanitized_min, sanitized_max = default_min, default_max

    if sanitized_min != min_val or sanitized_max != max_val:
        logger.warning(
            f"Invalid {name} range ({min_name}={min_val!r}, {max_name}={max_val!r}), using ({sanitized_min!r}, {sanitized_max!r})"
        )

    return sanitized_min, sanitized_max


def get_label_defaults(
    feature_parameters: dict[str, Any],
    logger: Logger,
    *,
    default_min_label_period_candles: int = 12,
    default_max_label_period_candles: int = 24,
    default_min_label_natr_multiplier: float = 9.0,
    default_max_label_natr_multiplier: float = 12.0,
) -> tuple[int, float]:
    min_label_natr_multiplier = feature_parameters.get(
        "min_label_natr_multiplier", default_min_label_natr_multiplier
    )
    max_label_natr_multiplier = feature_parameters.get(
        "max_label_natr_multiplier", default_max_label_natr_multiplier
    )
    min_label_natr_multiplier, max_label_natr_multiplier = validate_range(
        min_label_natr_multiplier,
        max_label_natr_multiplier,
        logger,
        name="label_natr_multiplier",
        default_min=default_min_label_natr_multiplier,
        default_max=default_max_label_natr_multiplier,
        allow_equal=False,
        non_negative=True,
        finite_only=True,
    )
    default_label_natr_multiplier = float(
        midpoint(min_label_natr_multiplier, max_label_natr_multiplier)
    )
    feature_parameters.setdefault(
        "label_natr_multiplier", default_label_natr_multiplier
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

    return default_label_period_candles, default_label_natr_multiplier
