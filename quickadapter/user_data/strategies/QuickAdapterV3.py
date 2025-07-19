import json
import logging
from functools import lru_cache, reduce, cached_property
import datetime
import math
from pathlib import Path
import talib.abstract as ta
from pandas import DataFrame, Series, isna
from typing import Any, Callable, Optional
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import stoploss_from_absolute
from technical.pivots_points import pivots_points
from freqtrade.persistence import Trade
import numpy as np
import pandas_ta as pta
import scipy as sp

from Utils import (
    TrendDirection,
    alligator,
    bottom_change_percent,
    calculate_quantile,
    get_zl_ma_fn,
    zero_phase,
    zigzag,
    ewo,
    non_zero_diff,
    price_retracement_percent,
    vwapb,
    top_change_percent,
    get_distance,
    get_gaussian_window,
    get_odd_window,
    derive_gaussian_std_from_window,
    zlema,
)

debug = False

logger = logging.getLogger(__name__)

EXTREMA_COLUMN = "&s-extrema"
MAXIMA_THRESHOLD_COLUMN = "&s-maxima_threshold"
MINIMA_THRESHOLD_COLUMN = "&s-minima_threshold"


class QuickAdapterV3(IStrategy):
    """
    The following freqtrade strategy is released to sponsors of the non-profit FreqAI open-source project.
    If you find the FreqAI project useful, please consider supporting it by becoming a sponsor.
    We use sponsor money to help stimulate new features and to pay for running these public
    experiments, with a an objective of helping the community make smarter choices in their
    ML journey.

    This strategy is experimental (as with all strategies released to sponsors). Do *not* expect
    returns. The goal is to demonstrate gratitude to people who support the project and to
    help them find a good starting point for their own creativity.

    If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

    https://github.com/sponsors/robcaulk
    """

    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "3.3.99"

    timeframe = "5m"

    stoploss = -0.02
    use_custom_stoploss = True

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.011
    trailing_only_offset_is_reached = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "limit",
        "force_exit": "limit",
        "force_entry": "limit",
        "stoploss": "limit",
        "stoploss_on_exchange": False,
        "stoploss_on_exchange_interval": 60,
        "stoploss_on_exchange_limit_ratio": 0.99,
    }

    position_adjustment_enable = True

    # {stage: (natr_ratio_percent, stake_percent)}
    partial_exit_stages: dict[int, tuple[float, float]] = {
        0: (0.4, 0.35),
        1: (0.7, 0.75),
        2: (0.9, 0.5),
    }

    timeframe_minutes = timeframe_to_minutes(timeframe)
    minimal_roi = {str(timeframe_minutes * 864): -1}

    process_only_new_candles = True

    @cached_property
    def can_short(self) -> bool:
        return self.is_short_allowed()

    @cached_property
    def plot_config(self) -> dict[str, Any]:
        return {
            "main_plot": {},
            "subplots": {
                "accuracy": {
                    "hp_rmse": {"color": "#c28ce3", "type": "line"},
                    "train_rmse": {"color": "#a3087a", "type": "line"},
                },
                "extrema": {
                    MAXIMA_THRESHOLD_COLUMN: {"color": "#e6be0b", "type": "line"},
                    EXTREMA_COLUMN: {"color": "#f53580", "type": "line"},
                    MINIMA_THRESHOLD_COLUMN: {"color": "#4ae747", "type": "line"},
                },
                "min_max": {
                    "maxima": {"color": "#0dd6de", "type": "bar"},
                    "minima": {"color": "#e3970b", "type": "bar"},
                },
            },
        }

    @cached_property
    def protections(self) -> list[dict[str, Any]]:
        fit_live_predictions_candles = self.freqai_info.get(
            "fit_live_predictions_candles", 100
        )
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 2},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": fit_live_predictions_candles,
                "trade_limit": self.config.get("max_open_trades"),
                "stop_duration_candles": fit_live_predictions_candles,
                "max_allowed_drawdown": 0.2,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": fit_live_predictions_candles,
                "trade_limit": 1,
                "stop_duration_candles": fit_live_predictions_candles,
                "only_per_pair": True,
            },
        ]

    use_exit_signal = True

    @cached_property
    def startup_candle_count(self) -> int:
        # Match the predictions warmup period
        return self.freqai_info.get("fit_live_predictions_candles", 100)

    @cached_property
    def max_open_trades_per_side(self) -> int:
        max_open_trades = self.config.get("max_open_trades")
        if max_open_trades < 0:
            return -1
        if self.is_short_allowed():
            if max_open_trades % 2 == 1:
                max_open_trades += 1
            return int(max_open_trades / 2)
        else:
            return max_open_trades

    def bot_start(self, **kwargs) -> None:
        self.pairs = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.pairs:
            raise ValueError(
                "FreqAI strategy requires StaticPairList method defined in pairlists configuration and 'pair_whitelist' defined in exchange section configuration"
            )
        if (
            not isinstance(self.freqai_info.get("identifier"), str)
            or self.freqai_info.get("identifier").strip() == ""
        ):
            raise ValueError(
                "FreqAI strategy requires 'identifier' defined in the freqai section configuration"
            )
        self.models_full_path = Path(
            self.config.get("user_data_dir")
            / "models"
            / self.freqai_info.get("identifier")
        )
        self._label_params: dict[str, dict[str, Any]] = {}
        for pair in self.pairs:
            self._label_params[pair] = (
                self.optuna_load_best_params(pair, "label")
                if self.optuna_load_best_params(pair, "label")
                else {
                    "label_period_candles": self.freqai_info["feature_parameters"].get(
                        "label_period_candles", 50
                    ),
                    "label_natr_ratio": float(
                        self.freqai_info["feature_parameters"].get(
                            "label_natr_ratio", 6.0
                        )
                    ),
                }
            )
        self._throttle_modulo = max(
            1,
            int(
                round(
                    (timeframe_to_minutes(self.config.get("timeframe")) * 60)
                    / self.config.get("internals", {}).get("process_throttle_secs", 5)
                )
            ),
        )

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        highs = dataframe.get("high")
        lows = dataframe.get("low")
        closes = dataframe.get("close")
        volumes = dataframe.get("volume")

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-aroonosc-period"] = ta.AROONOSC(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=period)
        dataframe["%-er-period"] = pta.er(closes, length=period)
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-trix-period"] = ta.TRIX(dataframe, timeperiod=period)
        dataframe["%-cmf-period"] = pta.cmf(
            highs,
            lows,
            closes,
            volumes,
            length=period,
        )
        dataframe["%-tcp-period"] = top_change_percent(dataframe, period=period)
        dataframe["%-bcp-period"] = bottom_change_percent(dataframe, period=period)
        dataframe["%-prp-period"] = price_retracement_percent(dataframe, period=period)
        dataframe["%-cti-period"] = pta.cti(closes, length=period)
        dataframe["%-chop-period"] = pta.chop(
            highs,
            lows,
            closes,
            length=period,
        )
        dataframe["%-linearreg_angle-period"] = ta.LINEARREG_ANGLE(
            dataframe, timeperiod=period
        )
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-natr-period"] = ta.NATR(dataframe, timeperiod=period)
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        highs = dataframe.get("high")
        lows = dataframe.get("low")
        opens = dataframe.get("open")
        closes = dataframe.get("close")
        volumes = dataframe.get("volume")

        dataframe["%-close_pct_change"] = closes.pct_change()
        dataframe["%-raw_volume"] = volumes
        dataframe["%-obv"] = ta.OBV(dataframe)
        label_period_candles = self.get_label_period_candles(str(metadata.get("pair")))
        dataframe["%-atr_label_period_candles"] = ta.ATR(
            dataframe, timeperiod=label_period_candles
        )
        dataframe["%-natr_label_period_candles"] = ta.NATR(
            dataframe, timeperiod=label_period_candles
        )
        dataframe["%-ewo"] = ewo(
            dataframe=dataframe,
            pricemode="close",
            mamode="ema",
            zero_lag=True,
            normalize=True,
        )
        psar = ta.SAR(dataframe, acceleration=0.02, maximum=0.2)
        dataframe["%-diff_to_psar"] = closes - psar
        kc = pta.kc(
            highs,
            lows,
            closes,
            length=14,
            scalar=2,
        )
        dataframe["kc_lowerband"] = kc["KCLe_14_2.0"]
        dataframe["kc_middleband"] = kc["KCBe_14_2.0"]
        dataframe["kc_upperband"] = kc["KCUe_14_2.0"]
        dataframe["%-kc_width"] = (
            dataframe["kc_upperband"] - dataframe["kc_lowerband"]
        ) / dataframe["kc_middleband"]
        (
            dataframe["bb_upperband"],
            dataframe["bb_middleband"],
            dataframe["bb_lowerband"],
        ) = ta.BBANDS(
            ta.TYPPRICE(dataframe),
            timeperiod=14,
            nbdevup=2.2,
            nbdevdn=2.2,
        )
        dataframe["%-bb_width"] = (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        ) / dataframe["bb_middleband"]
        dataframe["%-ibs"] = (closes - lows) / non_zero_diff(highs, lows)
        dataframe["jaw"], dataframe["teeth"], dataframe["lips"] = alligator(
            dataframe, pricemode="median", zero_lag=True
        )
        dataframe["%-dist_to_jaw"] = get_distance(closes, dataframe["jaw"])
        dataframe["%-dist_to_teeth"] = get_distance(closes, dataframe["teeth"])
        dataframe["%-dist_to_lips"] = get_distance(closes, dataframe["lips"])
        dataframe["%-spread_jaw_teeth"] = dataframe["jaw"] - dataframe["teeth"]
        dataframe["%-spread_teeth_lips"] = dataframe["teeth"] - dataframe["lips"]
        dataframe["zlema_50"] = zlema(closes, period=50)
        dataframe["zlema_12"] = zlema(closes, period=12)
        dataframe["zlema_26"] = zlema(closes, period=26)
        dataframe["%-distzlema50"] = get_distance(closes, dataframe["zlema_50"])
        dataframe["%-distzlema12"] = get_distance(closes, dataframe["zlema_12"])
        dataframe["%-distzlema26"] = get_distance(closes, dataframe["zlema_26"])
        macd = ta.MACD(dataframe)
        dataframe["%-macd"] = macd["macd"]
        dataframe["%-macdsignal"] = macd["macdsignal"]
        dataframe["%-macdhist"] = macd["macdhist"]
        dataframe["%-dist_to_macdsignal"] = get_distance(
            dataframe["%-macd"], dataframe["%-macdsignal"]
        )
        dataframe["%-dist_to_zerohist"] = get_distance(0, dataframe["%-macdhist"])
        # VWAP bands
        (
            dataframe["vwap_lowerband"],
            dataframe["vwap_middleband"],
            dataframe["vwap_upperband"],
        ) = vwapb(dataframe, 20, 1.0)
        dataframe["%-vwap_width"] = (
            dataframe["vwap_upperband"] - dataframe["vwap_lowerband"]
        ) / dataframe["vwap_middleband"]
        dataframe["%-dist_to_vwap_upperband"] = get_distance(
            closes, dataframe["vwap_upperband"]
        )
        dataframe["%-dist_to_vwap_middleband"] = get_distance(
            closes, dataframe["vwap_middleband"]
        )
        dataframe["%-dist_to_vwap_lowerband"] = get_distance(
            closes, dataframe["vwap_lowerband"]
        )
        dataframe["%-body"] = closes - opens
        dataframe["%-tail"] = (np.minimum(opens, closes) - lows).clip(lower=0)
        dataframe["%-wick"] = (highs - np.maximum(opens, closes)).clip(lower=0)
        pp = pivots_points(dataframe)
        dataframe["r1"] = pp["r1"]
        dataframe["s1"] = pp["s1"]
        dataframe["r2"] = pp["r2"]
        dataframe["s2"] = pp["s2"]
        dataframe["r3"] = pp["r3"]
        dataframe["s3"] = pp["s3"]
        dataframe["%-dist_to_r1"] = get_distance(closes, dataframe["r1"])
        dataframe["%-dist_to_r2"] = get_distance(closes, dataframe["r2"])
        dataframe["%-dist_to_r3"] = get_distance(closes, dataframe["r3"])
        dataframe["%-dist_to_s1"] = get_distance(closes, dataframe["s1"])
        dataframe["%-dist_to_s2"] = get_distance(closes, dataframe["s2"])
        dataframe["%-dist_to_s3"] = get_distance(closes, dataframe["s3"])
        dataframe["%-raw_close"] = closes
        dataframe["%-raw_open"] = opens
        dataframe["%-raw_low"] = lows
        dataframe["%-raw_high"] = highs
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, **kwargs) -> DataFrame:
        dates = dataframe.get("date")

        dataframe["%-day_of_week"] = (dates.dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dates.dt.hour + 1) / 25
        return dataframe

    def get_label_period_candles(self, pair: str) -> int:
        label_period_candles = self._label_params.get(pair, {}).get(
            "label_period_candles"
        )
        if label_period_candles and isinstance(label_period_candles, int):
            return label_period_candles
        return self.freqai_info["feature_parameters"].get("label_period_candles", 50)

    def set_label_period_candles(self, pair: str, label_period_candles: int) -> None:
        if isinstance(label_period_candles, int):
            self._label_params[pair]["label_period_candles"] = label_period_candles

    def get_label_natr_ratio(self, pair: str) -> float:
        label_natr_ratio = self._label_params.get(pair, {}).get("label_natr_ratio")
        if label_natr_ratio and isinstance(label_natr_ratio, float):
            return label_natr_ratio
        return float(
            self.freqai_info["feature_parameters"].get("label_natr_ratio", 6.0)
        )

    def set_label_natr_ratio(self, pair: str, label_natr_ratio: float) -> None:
        if isinstance(label_natr_ratio, float) and np.isfinite(label_natr_ratio):
            self._label_params[pair]["label_natr_ratio"] = label_natr_ratio

    def get_entry_natr_ratio(self, pair: str, percent: float) -> float:
        return self.get_label_natr_ratio(pair) * percent

    def get_stoploss_natr_ratio(self, pair: str, percent: float) -> float:
        return self.get_label_natr_ratio(pair) * percent

    def get_take_profit_natr_ratio(self, pair: str, percent: float) -> float:
        return self.get_label_natr_ratio(pair) * percent

    @staticmethod
    def td_format(
        delta: datetime.timedelta, pattern: str = "{sign}{d}:{h:02d}:{m:02d}:{s:02d}"
    ) -> str:
        negative_duration = delta.total_seconds() < 0
        delta = abs(delta)
        duration = {"d": delta.days}
        duration["h"], remainder = divmod(delta.seconds, 3600)
        duration["m"], duration["s"] = divmod(remainder, 60)
        duration["ms"] = delta.microseconds // 1000
        duration["sign"] = "-" if negative_duration else ""
        try:
            return pattern.format(**duration)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid pattern '{pattern}': {e}")

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        pair = str(metadata.get("pair"))
        label_period_candles = self.get_label_period_candles(pair)
        label_natr_ratio = self.get_label_natr_ratio(pair)
        pivots_indices, _, pivots_directions, _ = zigzag(
            dataframe,
            natr_period=label_period_candles,
            natr_ratio=label_natr_ratio,
        )
        label_period = datetime.timedelta(
            minutes=len(dataframe) * timeframe_to_minutes(self.config.get("timeframe"))
        )
        dataframe[EXTREMA_COLUMN] = 0
        if len(pivots_indices) == 0:
            logger.warning(
                f"{pair}: no extrema to label (label_period={QuickAdapterV3.td_format(label_period)} / {label_period_candles=} / {label_natr_ratio=:.2f})"
            )
        else:
            for pivot_idx, pivot_dir in zip(pivots_indices, pivots_directions):
                dataframe.at[pivot_idx, EXTREMA_COLUMN] = pivot_dir
            dataframe["minima"] = np.where(
                dataframe[EXTREMA_COLUMN] == TrendDirection.DOWN, -1, 0
            )
            dataframe["maxima"] = np.where(
                dataframe[EXTREMA_COLUMN] == TrendDirection.UP, 1, 0
            )
            logger.info(
                f"{pair}: labeled {len(pivots_indices)} extrema (label_period={QuickAdapterV3.td_format(label_period)} / {label_period_candles=} / {label_natr_ratio=:.2f})"
            )
        dataframe[EXTREMA_COLUMN] = self.smooth_extrema(
            dataframe[EXTREMA_COLUMN],
            self.freqai_info.get("extrema_smoothing_window", 5),
        )
        if debug:
            logger.info(f"{dataframe[EXTREMA_COLUMN].to_numpy()=}")
            n_minima: int = sp.signal.find_peaks(-dataframe[EXTREMA_COLUMN])[0].size
            n_maxima: int = sp.signal.find_peaks(dataframe[EXTREMA_COLUMN])[0].size
            n_extrema: int = n_minima + n_maxima
            logger.info(f"{n_extrema=}")
        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        dataframe["DI_catch"] = np.where(
            dataframe.get("DI_values") > dataframe.get("DI_cutoff"),
            0,
            1,
        )

        pair = str(metadata.get("pair"))

        self.set_label_period_candles(
            pair, dataframe.get("label_period_candles").iloc[-1]
        )
        self.set_label_natr_ratio(pair, dataframe.get("label_natr_ratio").iloc[-1])

        dataframe["natr_label_period_candles"] = ta.NATR(
            dataframe, timeperiod=self.get_label_period_candles(pair)
        )

        dataframe["minima_threshold"] = dataframe.get(MINIMA_THRESHOLD_COLUMN)
        dataframe["maxima_threshold"] = dataframe.get(MAXIMA_THRESHOLD_COLUMN)

        return dataframe

    def populate_entry_trend(
        self, df: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        enter_long_conditions = [
            df.get("do_predict") == 1,
            df.get("DI_catch") == 1,
            df.get(EXTREMA_COLUMN) < df.get("minima_threshold"),
        ]

        df.loc[
            reduce(lambda x, y: x & y, enter_long_conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "long")

        enter_short_conditions = [
            df.get("do_predict") == 1,
            df.get("DI_catch") == 1,
            df.get(EXTREMA_COLUMN) > df.get("maxima_threshold"),
        ]

        df.loc[
            reduce(lambda x, y: x & y, enter_short_conditions),
            ["enter_short", "enter_tag"],
        ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        return df

    def get_trade_entry_date(self, trade: Trade) -> datetime.datetime:
        return timeframe_to_prev_date(self.config.get("timeframe"), trade.open_date_utc)

    def get_trade_duration_candles(self, df: DataFrame, trade: Trade) -> Optional[int]:
        """
        Get the number of candles since the trade entry.
        :param df: DataFrame with the current data
        :param trade: Trade object
        :return: Number of candles since the trade entry
        """
        entry_date = self.get_trade_entry_date(trade)
        dates = df.get("date")
        if dates is None or dates.empty:
            return None
        current_date = dates.iloc[-1]
        if isna(current_date):
            return None
        trade_duration_minutes = (current_date - entry_date).total_seconds() / 60.0
        return int(
            trade_duration_minutes / timeframe_to_minutes(self.config.get("timeframe"))
        )

    @staticmethod
    def is_trade_duration_valid(trade_duration: Optional[int | float]) -> bool:
        return isinstance(trade_duration, (int, float)) and not (
            isna(trade_duration) or trade_duration <= 0
        )

    def get_trade_weighted_interpolation_natr(
        self, df: DataFrame, trade: Trade
    ) -> Optional[float]:
        label_natr = df.get("natr_label_period_candles")
        if label_natr is None or label_natr.empty:
            return None
        dates = df.get("date")
        if dates is None or dates.empty:
            return None
        entry_date = self.get_trade_entry_date(trade)
        trade_label_natr = label_natr[dates >= entry_date]
        if trade_label_natr.empty:
            return None
        entry_natr = trade_label_natr.iloc[0]
        if isna(entry_natr) or entry_natr < 0:
            return None
        if len(trade_label_natr) == 1:
            return entry_natr
        current_natr = trade_label_natr.iloc[-1]
        if isna(current_natr) or current_natr < 0:
            return None
        median_natr = trade_label_natr.median()

        np_trade_label_natr = trade_label_natr.to_numpy()
        entry_quantile = calculate_quantile(np_trade_label_natr, entry_natr)
        current_quantile = calculate_quantile(np_trade_label_natr, current_natr)
        median_quantile = calculate_quantile(np_trade_label_natr, median_natr)

        if isna(entry_quantile) or isna(current_quantile) or isna(median_quantile):
            return None

        def calculate_weight(
            quantile: float,
            min_weight: float = 0.0,
            max_weight: float = 1.0,
            steepness: float = 1.5,
        ) -> float:
            normalized_distance_from_center = abs(quantile - 0.5) * 2.0
            return (
                min_weight
                + (max_weight - min_weight) * normalized_distance_from_center**steepness
            )

        entry_weight = calculate_weight(entry_quantile)
        current_weight = calculate_weight(current_quantile)
        median_weight = calculate_weight(median_quantile)

        total_weight = entry_weight + current_weight + median_weight
        if np.isclose(total_weight, 0):
            return None
        entry_weight /= total_weight
        current_weight /= total_weight
        median_weight /= total_weight

        return (
            entry_natr * entry_weight
            + current_natr * current_weight
            + median_natr * median_weight
        )

    def get_trade_interpolation_natr(
        self, df: DataFrame, trade: Trade
    ) -> Optional[float]:
        label_natr = df.get("natr_label_period_candles")
        if label_natr is None or label_natr.empty:
            return None
        dates = df.get("date")
        if dates is None or dates.empty:
            return None
        entry_date = self.get_trade_entry_date(trade)
        trade_label_natr = label_natr[dates >= entry_date]
        if trade_label_natr.empty:
            return None
        entry_natr = trade_label_natr.iloc[0]
        if isna(entry_natr) or entry_natr < 0:
            return None
        if len(trade_label_natr) == 1:
            return entry_natr
        current_natr = trade_label_natr.iloc[-1]
        if isna(current_natr) or current_natr < 0:
            return None
        trade_volatility_quantile = calculate_quantile(
            trade_label_natr.to_numpy(), entry_natr
        )
        if isna(trade_volatility_quantile):
            return None
        return np.interp(
            trade_volatility_quantile,
            [0.0, 1.0],
            [current_natr, entry_natr],
        )

    def get_trade_moving_average_natr(
        self, df: DataFrame, pair: str, trade_duration_candles: int
    ) -> Optional[float]:
        if not QuickAdapterV3.is_trade_duration_valid(trade_duration_candles):
            return None
        label_natr = df.get("natr_label_period_candles")
        if label_natr is None or label_natr.empty:
            return None
        if trade_duration_candles >= 2:
            zl_kama = get_zl_ma_fn("kama")
            try:
                trade_kama_natr_values = zl_kama(
                    label_natr, timeperiod=trade_duration_candles
                )
                trade_kama_natr_values = trade_kama_natr_values[
                    ~np.isnan(trade_kama_natr_values)
                ]
                if trade_kama_natr_values.size > 0:
                    return trade_kama_natr_values[-1]
            except Exception as e:
                logger.error(
                    f"Failed to calculate KAMA for pair {pair}: {str(e)}", exc_info=True
                )
        return label_natr.iloc[-1]

    def get_trade_natr(
        self, df: DataFrame, trade: Trade, trade_duration_candles: int
    ) -> Optional[float]:
        trade_price_target = self.config.get("exit_pricing", {}).get(
            "trade_price_target", "moving_average"
        )
        if trade_price_target == "interpolation":
            return self.get_trade_interpolation_natr(df, trade)
        elif trade_price_target == "weighted_interpolation":
            return self.get_trade_weighted_interpolation_natr(df, trade)
        elif trade_price_target == "moving_average":
            return self.get_trade_moving_average_natr(
                df, trade.pair, trade_duration_candles
            )
        else:
            raise ValueError(
                f"Invalid trade_price_target: {trade_price_target}. Expected 'interpolation', 'weighted_interpolation' or 'moving_average'."
            )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_stoploss_log_factor(trade_duration_candles: int) -> float:
        return 1 / math.log10(3.75 + 0.25 * trade_duration_candles)

    def get_stoploss_distance(
        self,
        df: DataFrame,
        trade: Trade,
        current_rate: float,
        natr_ratio_percent: float,
    ) -> Optional[float]:
        trade_duration_candles = self.get_trade_duration_candles(df, trade)
        if not QuickAdapterV3.is_trade_duration_valid(trade_duration_candles):
            return None
        trade_natr = self.get_trade_natr(df, trade, trade_duration_candles)
        if isna(trade_natr) or trade_natr < 0:
            return None
        return (
            current_rate
            * (trade_natr / 100.0)
            * self.get_stoploss_natr_ratio(trade.pair, natr_ratio_percent)
            * QuickAdapterV3.get_stoploss_log_factor(trade_duration_candles)
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_take_profit_log_factor(trade_duration_candles: int) -> float:
        return math.log10(9.75 + 0.25 * trade_duration_candles)

    def get_take_profit_distance(
        self, df: DataFrame, trade: Trade, natr_ratio_percent: float
    ) -> Optional[float]:
        trade_duration_candles = self.get_trade_duration_candles(df, trade)
        if not QuickAdapterV3.is_trade_duration_valid(trade_duration_candles):
            return None
        trade_natr = self.get_trade_natr(df, trade, trade_duration_candles)
        if isna(trade_natr) or trade_natr < 0:
            return None
        return (
            trade.open_rate
            * (trade_natr / 100.0)
            * self.get_take_profit_natr_ratio(trade.pair, natr_ratio_percent)
            * QuickAdapterV3.get_take_profit_log_factor(trade_duration_candles)
        )

    def throttle_callback(
        self,
        pair: str,
        current_time: datetime.datetime,
        callback: Callable[[], None],
    ) -> None:
        if hash(pair + str(current_time)) % self._throttle_modulo == 0:
            try:
                callback()
            except Exception as e:
                logger.error(
                    f"Error executing callback for {pair}: {str(e)}", exc_info=True
                )

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[float]:
        df, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            return None

        stoploss_distance = self.get_stoploss_distance(df, trade, current_rate, 0.9)
        if isna(stoploss_distance) or stoploss_distance <= 0:
            return None
        return stoploss_from_absolute(
            current_rate + (1 if trade.is_short else -1) * stoploss_distance,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float] | tuple[Optional[float], Optional[str]]:
        if trade.has_open_orders:
            return None

        exit_stage: int = trade.get_custom_data("exit_stage", 0)
        if exit_stage not in self.partial_exit_stages:
            return None

        df, _ = self.dp.get_analyzed_dataframe(
            trade.pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            return None

        natr_ratio_percent, stake_percent = self.partial_exit_stages[exit_stage]

        take_profit_distance = self.get_take_profit_distance(
            df, trade, natr_ratio_percent
        )
        if isna(take_profit_distance) or take_profit_distance <= 0:
            return None

        take_profit_price = (
            trade.open_rate + (-1 if trade.is_short else 1) * take_profit_distance
        )
        previous_take_profit_price = trade.get_custom_data("take_profit_price", None)
        if (
            previous_take_profit_price is None
            or previous_take_profit_price != take_profit_price
        ):
            trade.set_custom_data(key="take_profit_price", value=take_profit_price)

        trade_partial_exit = (trade.is_short and current_rate <= take_profit_price) or (
            not trade.is_short and current_rate >= take_profit_price
        )
        if not trade_partial_exit:
            self.throttle_callback(
                pair=trade.pair,
                current_time=current_time,
                callback=lambda: logger.info(
                    f"Trade {trade.trade_direction} for {trade.pair}: open price {trade.open_rate}, current price {current_rate}, partial exit stage {exit_stage} price {take_profit_price}"
                ),
            )
        if trade_partial_exit:
            trade_stake_amount = trade.stake_amount * stake_percent
            trade.set_custom_data(key="exit_stage", value=exit_stage + 1)
            return (
                -trade_stake_amount,
                f"take_profit_{trade.trade_direction}_{exit_stage}",
            )

        return None

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        df, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            return None

        last_candle = df.iloc[-1]
        if last_candle.get("do_predict") == 2:
            return "model_expired"
        if last_candle.get("DI_catch") == 0:
            return "outlier_detected"

        entry_tag = trade.enter_tag

        if (
            entry_tag == "short"
            and last_candle.get("do_predict") == 1
            and last_candle.get(EXTREMA_COLUMN) < last_candle.get("minima_threshold")
        ):
            return "minima_detected_short"
        if (
            entry_tag == "long"
            and last_candle.get("do_predict") == 1
            and last_candle.get(EXTREMA_COLUMN) > last_candle.get("maxima_threshold")
        ):
            return "maxima_detected_long"

        start_partial_exit_stage = list(self.partial_exit_stages.keys())[0]
        end_partial_exit_stage = list(self.partial_exit_stages.keys())[-1]
        final_exit_stage = end_partial_exit_stage + 1
        exit_stage: int = trade.get_custom_data("exit_stage", 0)
        if self.position_adjustment_enable:
            if exit_stage == start_partial_exit_stage:
                return None
            secure_take_profit_distance = self.get_take_profit_distance(
                df,
                trade,
                min(
                    self.config.get("exit_pricing", {}).get(
                        "trade_secure_percent", 0.2
                    ),
                    self.partial_exit_stages[start_partial_exit_stage][0] * 0.95,
                ),
            )
            if isna(secure_take_profit_distance) or secure_take_profit_distance <= 0:
                return None
            secure_take_profit_price = (
                trade.open_rate
                + (-1 if trade.is_short else 1) * secure_take_profit_distance
            )
            secure_trade_exit = (
                trade.is_short and current_rate <= secure_take_profit_price
            ) or (not trade.is_short and current_rate >= secure_take_profit_price)
            if not secure_trade_exit:
                self.throttle_callback(
                    pair=pair,
                    current_time=current_time,
                    callback=lambda: logger.info(
                        f"Trade {trade.trade_direction} for {pair}: open price {trade.open_rate}, current price {current_rate}, secure final exit stage {final_exit_stage} price {secure_take_profit_price}"
                    ),
                )
            if secure_trade_exit:
                trade.set_custom_data(
                    key="take_profit_price", value=secure_take_profit_price
                )
                if exit_stage < final_exit_stage:
                    trade.set_custom_data(key="exit_stage", value=final_exit_stage)
                return f"secure_take_profit_{trade.trade_direction}_{final_exit_stage}"
            if exit_stage == final_exit_stage:
                natr_ratio_percent = 1.0
            else:
                return None
        else:
            if start_partial_exit_stage <= exit_stage <= end_partial_exit_stage:
                trade.set_custom_data(key="exit_stage", value=final_exit_stage)
            natr_ratio_percent = 0.7

        take_profit_distance = self.get_take_profit_distance(
            df, trade, natr_ratio_percent
        )
        if isna(take_profit_distance) or take_profit_distance <= 0:
            return None
        take_profit_price = (
            trade.open_rate + (-1 if trade.is_short else 1) * take_profit_distance
        )
        previous_take_profit_price = trade.get_custom_data("take_profit_price", None)
        if (
            previous_take_profit_price is None
            or previous_take_profit_price != take_profit_price
        ):
            trade.set_custom_data(key="take_profit_price", value=take_profit_price)
        trade_exit = (trade.is_short and current_rate <= take_profit_price) or (
            not trade.is_short and current_rate >= take_profit_price
        )
        if not trade_exit:
            self.throttle_callback(
                pair=pair,
                current_time=current_time,
                callback=lambda: logger.info(
                    f"Trade {trade.trade_direction} for {pair}: open price {trade.open_rate}, current price {current_rate}, final exit stage {final_exit_stage} price {take_profit_price}"
                ),
            )
        if trade_exit:
            return f"take_profit_{trade.trade_direction}_{final_exit_stage}"

        return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime.datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        if Trade.get_open_trade_count() >= self.config.get("max_open_trades"):
            return False
        max_open_trades_per_side = self.max_open_trades_per_side
        if max_open_trades_per_side >= 0:
            open_trades = Trade.get_open_trades()
            trades_per_side = sum(1 for trade in open_trades if trade.enter_tag == side)
            if trades_per_side >= max_open_trades_per_side:
                return False

        df, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            return False
        last_candle = df.iloc[-1]
        last_candle_close = last_candle.get("close")
        last_candle_high = last_candle.get("high")
        last_candle_low = last_candle.get("low")
        last_candle_weighted_close_price = (
            last_candle_high + last_candle_low + 2 * last_candle_close
        ) / 4.0
        last_candle_natr = last_candle.get("natr_label_period_candles")
        if isna(last_candle_natr) or last_candle_natr < 0:
            return False
        lower_bound = 0
        upper_bound = 0
        price_deviation = (last_candle_natr / 100.0) * self.get_entry_natr_ratio(
            pair, 0.01
        )
        if side == "long":
            lower_bound = last_candle_low * (1 - price_deviation)
            upper_bound = last_candle_weighted_close_price * (1 + price_deviation)
        elif side == "short":
            lower_bound = last_candle_weighted_close_price * (1 - price_deviation)
            upper_bound = last_candle_high * (1 + price_deviation)
        if lower_bound < 0:
            logger.info(
                f"User denied {side} entry for {pair}: calculated lower bound {lower_bound} is below zero"
            )
            return False
        if lower_bound <= rate <= upper_bound:
            return True
        else:
            logger.info(
                f"User denied {side} entry for {pair}: rate {rate} outside bounds [{lower_bound}, {upper_bound}]"
            )
        return False

    def is_short_allowed(self) -> bool:
        trading_mode = self.config.get("trading_mode")
        if trading_mode == "margin" or trading_mode == "futures":
            return True
        elif trading_mode == "spot":
            return False
        else:
            raise ValueError(f"Invalid trading_mode: {trading_mode}")

    def smooth_extrema(
        self,
        series: Series,
        window: int,
    ) -> Series:
        extrema_smoothing = self.freqai_info.get("extrema_smoothing", "gaussian")
        extrema_smoothing_zero_phase = self.freqai_info.get(
            "extrema_smoothing_zero_phase", True
        )
        std = derive_gaussian_std_from_window(window)
        extrema_smoothing_beta = float(
            self.freqai_info.get("extrema_smoothing_beta", 8.0)
        )
        if debug:
            logger.info(
                f"{extrema_smoothing=}, {extrema_smoothing_zero_phase=}, {window=}, {std=}, {extrema_smoothing_beta=}"
            )
        gaussian_window = get_gaussian_window(std, True)
        odd_window = get_odd_window(window)
        smoothing_methods: dict[str, Series] = {
            "gaussian": zero_phase(
                series=series,
                window=window,
                win_type="gaussian",
                std=std,
                beta=extrema_smoothing_beta,
            )
            if extrema_smoothing_zero_phase
            else series.rolling(
                window=gaussian_window,
                win_type="gaussian",
                center=True,
            ).mean(std=std),
            "kaiser": zero_phase(
                series=series,
                window=window,
                win_type="kaiser",
                std=std,
                beta=extrema_smoothing_beta,
            )
            if extrema_smoothing_zero_phase
            else series.rolling(
                window=odd_window,
                win_type="kaiser",
                center=True,
            ).mean(beta=extrema_smoothing_beta),
            "triang": zero_phase(
                series=series,
                window=window,
                win_type="triang",
                std=std,
                beta=extrema_smoothing_beta,
            )
            if extrema_smoothing_zero_phase
            else series.rolling(
                window=odd_window, win_type="triang", center=True
            ).mean(),
            "smm": series.rolling(window=odd_window, center=True).median(),
            "sma": series.rolling(window=odd_window, center=True).mean(),
        }
        return smoothing_methods.get(
            extrema_smoothing,
            smoothing_methods["gaussian"],
        )

    def optuna_load_best_params(
        self, pair: str, namespace: str
    ) -> Optional[dict[str, Any]]:
        best_params_path = Path(
            self.models_full_path
            / f"optuna-{namespace}-best-params-{pair.split('/')[0]}.json"
        )
        if best_params_path.is_file():
            with best_params_path.open("r", encoding="utf-8") as read_file:
                return json.load(read_file)
        return None
