import copy
import functools
import hashlib
import math
from enum import IntEnum
from functools import lru_cache
from logging import Logger
from typing import Any, Callable, Literal, Optional, TypeVar

import numpy as np
import optuna
import pandas as pd
import scipy as sp
import talib.abstract as ta
from numpy.typing import NDArray
from technical import qtpylib

T = TypeVar("T", pd.Series, float)


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
    # Assuming window = 6 * std + 1 => std = (window - 1) / 6
    return (window - 1) / 6.0 if window > 1 else 0.5


@lru_cache(maxsize=8)
def _calculate_coeffs(
    window: int,
    win_type: Literal["gaussian", "kaiser", "triang"],
    std: float,
    beta: float,
) -> NDArray[np.floating]:
    if win_type == "gaussian":
        coeffs = sp.signal.windows.gaussian(M=window, std=std, sym=True)
    elif win_type == "kaiser":
        coeffs = sp.signal.windows.kaiser(M=window, beta=beta, sym=True)
    elif win_type == "triang":
        coeffs = sp.signal.windows.triang(M=window, sym=True)
    else:
        raise ValueError(f"Unknown window type: {win_type}")
    return coeffs / np.sum(coeffs)


def zero_phase(
    series: pd.Series,
    window: int,
    win_type: Literal["gaussian", "kaiser", "triang"],
    std: float,
    beta: float,
) -> pd.Series:
    if len(series) == 0:
        return series
    if len(series) < window:
        raise ValueError("Series length must be greater than or equal to window size")
    values = series.to_numpy()
    b = _calculate_coeffs(window=window, win_type=win_type, std=std, beta=beta)
    a = 1.0
    filtered_values = sp.signal.filtfilt(b, a, values)
    return pd.Series(filtered_values, index=series.index)


def smooth_extrema(
    series: pd.Series, method: str, window: int, beta: float
) -> pd.Series:
    std = get_gaussian_std(window)
    odd_window = get_odd_window(window)
    smoothing_methods: dict[str, pd.Series] = {
        "gaussian": zero_phase(
            series=series,
            window=window,
            win_type="gaussian",
            std=std,
            beta=beta,
        ),
        "kaiser": zero_phase(
            series=series,
            window=window,
            win_type="kaiser",
            std=std,
            beta=beta,
        ),
        "triang": zero_phase(
            series=series,
            window=window,
            win_type="triang",
            std=std,
            beta=beta,
        ),
        "smm": series.rolling(window=odd_window, center=True).median(),
        "sma": series.rolling(window=odd_window, center=True).mean(),
    }
    return smoothing_methods.get(
        method,
        smoothing_methods["gaussian"],
    )


def get_callable_sha256(fn: Callable) -> str:
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
            ma_fn = lambda series, timeperiod: zlema(series, period=timeperiod)
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
) -> tuple[list[int], list[float], list[TrendDirection], list[float]]:
    n = len(df)
    if df.empty or n < natr_period:
        return [], [], [], []

    natr_values = (ta.NATR(df, timeperiod=natr_period).bfill() / 100.0).to_numpy()

    indices: list[int] = df.index.tolist()
    thresholds: NDArray[np.floating] = natr_values * natr_ratio
    closes = df.get("close").to_numpy()
    log_closes = np.log(closes)
    highs = df.get("high").to_numpy()
    lows = df.get("low").to_numpy()

    state: TrendDirection = TrendDirection.NEUTRAL

    pivots_indices: list[int] = []
    pivots_values: list[float] = []
    pivots_directions: list[TrendDirection] = []
    pivots_thresholds: list[float] = []
    last_pivot_pos: int = -1

    candidate_pivot_pos: int = -1
    candidate_pivot_value: float = np.nan

    volatility_quantile_cache: dict[int, float] = {}

    def calculate_volatility_quantile(pos: int) -> float:
        if pos not in volatility_quantile_cache:
            start_pos = max(0, pos + 1 - natr_period)
            end_pos = min(pos + 1, n)
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

    def add_pivot(pos: int, value: float, direction: TrendDirection):
        nonlocal last_pivot_pos
        if pivots_indices and indices[pos] == pivots_indices[-1]:
            return
        pivots_indices.append(indices[pos])
        pivots_values.append(value)
        pivots_directions.append(direction)
        pivots_thresholds.append(thresholds[pos])
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
        is_initial_high_move_significant = (
            initial_move_from_high >= thresholds[initial_high_pos]
        )
        is_initial_low_move_significant = (
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
        return [], [], [], []

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

    return pivots_indices, pivots_values, pivots_directions, pivots_thresholds


regressors = {"xgboost", "lightgbm"}


def get_optuna_callbacks(
    trial: optuna.trial.Trial, regressor: str
) -> list[Callable[[optuna.trial.Trial, str], None]]:
    if regressor == "xgboost":
        callbacks = [
            optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")
        ]
    elif regressor == "lightgbm":
        callbacks = [optuna.integration.LightGBMPruningCallback(trial, "rmse")]
    else:
        raise ValueError(
            f"Unsupported regressor model: {regressor} (supported: {', '.join(regressors)})"
        )
    return callbacks


def fit_regressor(
    regressor: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_weights: NDArray[np.floating],
    eval_set: Optional[list[tuple[pd.DataFrame, pd.DataFrame]]],
    eval_weights: Optional[list[NDArray[np.floating]]],
    model_training_parameters: dict[str, Any],
    init_model: Any = None,
    callbacks: Optional[list[Callable[[optuna.trial.Trial, str], None]]] = None,
    trial: Optional[optuna.trial.Trial] = None,
) -> Any:
    if regressor == "xgboost":
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
    elif regressor == "lightgbm":
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
            f"Unsupported regressor model: {regressor} (supported: {', '.join(regressors)})"
        )
    return model


def get_optuna_study_model_parameters(
    trial: optuna.trial.Trial,
    regressor: str,
    model_training_best_parameters: dict[str, Any],
    space_reduction: bool,
    expansion_ratio: float,
) -> dict[str, Any]:
    if regressor not in regressors:
        raise ValueError(
            f"Unsupported regressor model: {regressor} (supported: {', '.join(regressors)})"
        )
    if not isinstance(expansion_ratio, (int, float)) or not (
        0.0 <= expansion_ratio <= 1.0
    ):
        raise ValueError(
            f"expansion_ratio must be a float between 0 and 1, got {expansion_ratio}"
        )
    default_ranges = {
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

            if center_value is None:
                center_value = midpoint(default_min, default_max)
            elif not isinstance(center_value, (int, float)) or not np.isfinite(
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
    if regressor == "xgboost":
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
    elif regressor == "lightgbm":
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
        return np.mean(np_array)
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
        ok = True
        if not isinstance(value, (int, float)):
            ok = False
        elif isinstance(value, bool):
            ok = False
        elif finite_only and not np.isfinite(value):
            ok = False
        elif non_negative and value < 0:
            ok = False
        if not ok:
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
