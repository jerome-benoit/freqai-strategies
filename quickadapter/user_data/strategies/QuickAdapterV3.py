import datetime
import hashlib
import json
import logging
import math
from functools import cached_property, lru_cache, reduce
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
import pandas_ta as pta
import talib.abstract as ta
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_absolute
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series, isna
from scipy.stats import t
from technical.pivots_points import pivots_points

from Utils import (
    TrendDirection,
    alligator,
    bottom_change_percent,
    calculate_n_extrema,
    calculate_quantile,
    ewo,
    format_number,
    get_callable_sha256,
    get_distance,
    get_label_defaults,
    get_zl_ma_fn,
    non_zero_diff,
    price_retracement_percent,
    smooth_extrema,
    top_change_percent,
    validate_range,
    vwapb,
    zigzag,
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
        return "3.3.164"

    timeframe = "5m"

    stoploss = -0.02
    use_custom_stoploss = True

    default_exit_thresholds: dict[str, float] = {
        "k_decl_v": 0.6,
        "k_decl_a": 0.4,
    }

    default_exit_thresholds_calibration: dict[str, float] = {
        "decline_quantile": 0.90,
    }

    default_reversal_confirmation: dict[str, int | float] = {
        "lookback_period": 0,
        "decay_ratio": 0.5,
        "min_natr_ratio_percent": 0.0099,
        "max_natr_ratio_percent": 0.33,
    }

    position_adjustment_enable = True

    # {stage: (natr_ratio_percent, stake_percent)}
    partial_exit_stages: dict[int, tuple[float, float]] = {
        0: (0.4858, 0.4),
        1: (0.6180, 0.3),
        2: (0.7640, 0.2),
    }

    timeframe_minutes = timeframe_to_minutes(timeframe)
    minimal_roi = {str(timeframe_minutes * 864): -1}

    # FreqAI is crashing if minimal_roi is a property
    # @property
    # def minimal_roi(self) -> dict[str, Any]:
    #     timeframe_minutes = timeframe_to_minutes(self.config.get("timeframe", "5m"))
    #     fit_live_predictions_candles = int(
    #         self.config.get("freqai", {}).get("fit_live_predictions_candles", 100)
    #     )
    #     return {str(timeframe_minutes * fit_live_predictions_candles): -1}

    # @minimal_roi.setter
    # def minimal_roi(self, value: dict[str, Any]) -> None:
    #     pass

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
        fit_live_predictions_candles = int(
            self.config.get("freqai", {}).get("fit_live_predictions_candles", 100)
        )
        estimated_trade_duration_candles = int(
            self.config.get("estimated_trade_duration_candles", 48)
        )
        stoploss_guard_lookback_period_candles = int(fit_live_predictions_candles / 2)
        stoploss_guard_trade_limit = max(
            1,
            int(
                round(
                    (
                        stoploss_guard_lookback_period_candles
                        / estimated_trade_duration_candles
                    )
                    * 0.5
                )
            ),
        )
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": fit_live_predictions_candles,
                "trade_limit": 2 * self.config.get("max_open_trades"),
                "stop_duration_candles": fit_live_predictions_candles,
                "max_allowed_drawdown": 0.2,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": stoploss_guard_lookback_period_candles,
                "trade_limit": stoploss_guard_trade_limit,
                "stop_duration_candles": stoploss_guard_lookback_period_candles,
                "only_per_pair": True,
            },
        ]

    use_exit_signal = True

    @cached_property
    def startup_candle_count(self) -> int:
        # Match the predictions warmup period
        return self.config.get("freqai", {}).get("fit_live_predictions_candles", 100)

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
        self.pairs: list[str] = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.pairs:
            raise ValueError(
                "FreqAI strategy requires StaticPairList method defined in pairlists configuration and 'pair_whitelist' defined in exchange section configuration"
            )
        if (
            not isinstance(self.freqai_info.get("identifier"), str)
            or not self.freqai_info.get("identifier", "").strip()
        ):
            raise ValueError(
                "FreqAI strategy requires 'identifier' defined in the freqai section configuration"
            )
        self.models_full_path = Path(
            self.config.get("user_data_dir")
            / "models"
            / self.freqai_info.get("identifier")
        )
        feature_parameters = self.freqai_info.get("feature_parameters", {})
        self._default_label_natr_ratio, self._default_label_period_candles = (
            get_label_defaults(feature_parameters, logger)
        )
        self._label_params: dict[str, dict[str, Any]] = {}
        for pair in self.pairs:
            self._label_params[pair] = (
                self.optuna_load_best_params(pair, "label")
                if self.optuna_load_best_params(pair, "label")
                else {
                    "label_period_candles": feature_parameters.get(
                        "label_period_candles",
                        self._default_label_period_candles,
                    ),
                    "label_natr_ratio": float(
                        feature_parameters.get(
                            "label_natr_ratio",
                            self._default_label_natr_ratio,
                        )
                    ),
                }
            )
        self._init_reversal_confirmation_defaults()
        self._candle_duration_secs = int(
            timeframe_to_minutes(self.config.get("timeframe")) * 60
        )
        self.last_candle_start_secs: dict[str, Optional[int]] = {}
        process_throttle_secs = self.config.get("internals", {}).get(
            "process_throttle_secs", 5
        )
        self._max_history_size = int(12 * 60 * 60 / process_throttle_secs)
        self._pnl_momentum_window_size = int(30 * 60 / process_throttle_secs)
        self._exit_thresholds_calibration: dict[str, float] = {
            **QuickAdapterV3.default_exit_thresholds_calibration,
            **self.config.get("exit_pricing", {}).get("thresholds_calibration", {}),
        }

    def _init_reversal_confirmation_defaults(self) -> None:
        reversal_confirmation = self.config.get("reversal_confirmation", {})
        lookback_period = reversal_confirmation.get(
            "lookback_period",
            QuickAdapterV3.default_reversal_confirmation["lookback_period"],
        )
        decay_ratio = reversal_confirmation.get(
            "decay_ratio", QuickAdapterV3.default_reversal_confirmation["decay_ratio"]
        )
        min_natr_ratio_percent = reversal_confirmation.get(
            "min_natr_ratio_percent",
            QuickAdapterV3.default_reversal_confirmation["min_natr_ratio_percent"],
        )
        max_natr_ratio_percent = reversal_confirmation.get(
            "max_natr_ratio_percent",
            QuickAdapterV3.default_reversal_confirmation["max_natr_ratio_percent"],
        )

        if not isinstance(lookback_period, int) or lookback_period < 0:
            logger.warning(
                f"reversal_confirmation: invalid lookback_period {lookback_period!r}, using default {QuickAdapterV3.default_reversal_confirmation['lookback_period']}"
            )
            lookback_period = QuickAdapterV3.default_reversal_confirmation[
                "lookback_period"
            ]

        if not isinstance(decay_ratio, (int, float)) or not (
            0.0 < float(decay_ratio) <= 1.0
        ):
            logger.warning(
                f"reversal_confirmation: invalid decay_ratio {decay_ratio!r}, using default {QuickAdapterV3.default_reversal_confirmation['decay_ratio']}"
            )
            decay_ratio = QuickAdapterV3.default_reversal_confirmation["decay_ratio"]
        else:
            decay_ratio = float(decay_ratio)

        min_natr_ratio_percent, max_natr_ratio_percent = validate_range(
            min_natr_ratio_percent,
            max_natr_ratio_percent,
            logger,
            name="natr_ratio_percent",
            default_min=QuickAdapterV3.default_reversal_confirmation[
                "min_natr_ratio_percent"
            ],
            default_max=QuickAdapterV3.default_reversal_confirmation[
                "max_natr_ratio_percent"
            ],
            allow_equal=False,
            non_negative=True,
            finite_only=True,
        )

        self._reversal_lookback_period = int(lookback_period)
        self._reversal_decay_ratio = float(decay_ratio)
        self._reversal_min_natr_ratio_percent = float(min_natr_ratio_percent)
        self._reversal_max_natr_ratio_percent = float(max_natr_ratio_percent)

        logger.debug(
            "reversal_confirmation: lookback_period=%s, decay_ratio=%s, natr_ratio_percent_range=(%s, %s)",
            self._reversal_lookback_period,
            format_number(self._reversal_decay_ratio),
            format_number(self._reversal_min_natr_ratio_percent),
            format_number(self._reversal_max_natr_ratio_percent),
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

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
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
        return self.freqai_info.get("feature_parameters", {}).get(
            "label_period_candles",
            self._default_label_period_candles,
        )

    def set_label_period_candles(self, pair: str, label_period_candles: int) -> None:
        if isinstance(label_period_candles, int):
            self._label_params[pair]["label_period_candles"] = label_period_candles

    def get_label_natr_ratio(self, pair: str) -> float:
        label_natr_ratio = self._label_params.get(pair, {}).get("label_natr_ratio")
        if label_natr_ratio and isinstance(label_natr_ratio, float):
            return label_natr_ratio
        return float(
            self.freqai_info.get("feature_parameters", {}).get(
                "label_natr_ratio",
                self._default_label_natr_ratio,
            )
        )

    def set_label_natr_ratio(self, pair: str, label_natr_ratio: float) -> None:
        if isinstance(label_natr_ratio, float) and np.isfinite(label_natr_ratio):
            self._label_params[pair]["label_natr_ratio"] = label_natr_ratio

    def get_label_natr_ratio_percent(self, pair: str, percent: float) -> float:
        if not isinstance(percent, float) or not (0.0 <= percent <= 1.0):
            raise ValueError(
                f"Invalid percent value: {percent}. It should be a float between 0 and 1"
            )
        return self.get_label_natr_ratio(pair) * percent

    @staticmethod
    @lru_cache(maxsize=128)
    def td_format(
        delta: datetime.timedelta, pattern: str = "{sign}{d}:{h:02d}:{m:02d}:{s:02d}"
    ) -> str:
        negative_duration = delta.total_seconds() < 0
        delta = abs(delta)
        duration: dict[str, Any] = {"d": delta.days}
        duration["h"], remainder = divmod(delta.seconds, 3600)
        duration["m"], duration["s"] = divmod(remainder, 60)
        duration["ms"] = delta.microseconds // 1000
        duration["sign"] = "-" if negative_duration else ""
        try:
            return pattern.format(**duration)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid pattern '{pattern}': {repr(e)}")

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
            dataframe.loc[pivots_indices, EXTREMA_COLUMN] = pivots_directions
            dataframe["minima"] = np.where(
                dataframe[EXTREMA_COLUMN] == TrendDirection.DOWN, -1, 0
            )
            dataframe["maxima"] = np.where(
                dataframe[EXTREMA_COLUMN] == TrendDirection.UP, 1, 0
            )
            logger.info(
                f"{pair}: labeled {len(pivots_indices)} extrema (label_period={QuickAdapterV3.td_format(label_period)} / {label_period_candles=} / {label_natr_ratio=:.2f})"
            )
        dataframe[EXTREMA_COLUMN] = smooth_extrema(
            dataframe[EXTREMA_COLUMN],
            str(self.freqai_info.get("extrema_smoothing", "gaussian")),
            int(self.freqai_info.get("extrema_smoothing_window", 5)),
            float(self.freqai_info.get("extrema_smoothing_beta", 8.0)),
        )
        if debug:
            extrema = dataframe[EXTREMA_COLUMN]
            logger.info(f"{extrema.to_numpy()=}")
            n_extrema: int = calculate_n_extrema(extrema)
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
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        enter_long_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get("DI_catch") == 1,
            dataframe.get(EXTREMA_COLUMN) < dataframe.get("minima_threshold"),
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_long_conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "long")

        enter_short_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get("DI_catch") == 1,
            dataframe.get(EXTREMA_COLUMN) > dataframe.get("maxima_threshold"),
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_short_conditions),
            ["enter_short", "enter_tag"],
        ] = (1, "short")

        return dataframe

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        return dataframe

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
    @lru_cache(maxsize=128)
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
            weighting_exponent: float = 1.5,
        ) -> float:
            normalized_distance_from_center = abs(quantile - 0.5) * 2.0
            return (
                min_weight
                + (max_weight - min_weight)
                * normalized_distance_from_center**weighting_exponent
            )

        entry_weight = calculate_weight(entry_quantile)
        current_weight = calculate_weight(current_quantile)
        median_weight = calculate_weight(median_quantile)

        total_weight = entry_weight + current_weight + median_weight
        if np.isclose(total_weight, 0.0):
            return np.mean([entry_natr, current_natr, median_natr])
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
            trade_volatility_quantile = 0.5
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
                trade_kama_natr_values = np.asarray(
                    zl_kama(label_natr, timeperiod=trade_duration_candles)
                )
                trade_kama_natr_values = trade_kama_natr_values[
                    np.isfinite(trade_kama_natr_values)
                ]
                if trade_kama_natr_values.size > 0:
                    return trade_kama_natr_values[-1]
            except Exception as e:
                logger.warning(
                    f"Failed to calculate trade NATR KAMA for pair {pair}: {repr(e)}. Falling back to last trade NATR value",
                    exc_info=True,
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
    def get_trade_exit_stage(trade: Trade) -> int:
        exit_side = "buy" if trade.is_short else "sell"
        try:
            return sum(
                1
                for order in trade.orders
                if order.side == exit_side and order.status in {"open", "closed"}
            )
        except Exception:
            return 0

    @staticmethod
    @lru_cache(maxsize=128)
    def get_stoploss_factor(trade_duration_candles: int) -> float:
        return 2.75 / (1.2675 + math.atan(0.25 * trade_duration_candles))

    def get_stoploss_distance(
        self,
        df: DataFrame,
        trade: Trade,
        current_rate: float,
        natr_ratio_percent: float,
    ) -> Optional[float]:
        if not (0.0 <= natr_ratio_percent <= 1.0):
            raise ValueError(
                f"natr_ratio_percent must be in [0, 1], got {natr_ratio_percent}"
            )
        trade_duration_candles = self.get_trade_duration_candles(df, trade)
        if not QuickAdapterV3.is_trade_duration_valid(trade_duration_candles):
            return None
        trade_natr = self.get_trade_natr(df, trade, trade_duration_candles)
        if isna(trade_natr) or trade_natr < 0:
            return None
        return (
            current_rate
            * (trade_natr / 100.0)
            * self.get_label_natr_ratio_percent(trade.pair, natr_ratio_percent)
            * QuickAdapterV3.get_stoploss_factor(
                trade_duration_candles + int(round(trade.nr_of_successful_exits**1.5))
            )
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_take_profit_factor(trade_duration_candles: int) -> float:
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
            * self.get_label_natr_ratio_percent(trade.pair, natr_ratio_percent)
            * QuickAdapterV3.get_take_profit_factor(trade_duration_candles)
        )

    def throttle_callback(
        self,
        pair: str,
        current_time: datetime.datetime,
        callback: Callable[[], None],
    ) -> None:
        if not callable(callback):
            raise ValueError("callback must be callable")
        timestamp = int(current_time.timestamp())
        candle_duration_secs = max(1, int(self._candle_duration_secs))
        candle_start_secs = (timestamp // candle_duration_secs) * candle_duration_secs
        callback_hash = get_callable_sha256(callback)
        key = hashlib.sha256(f"{pair}\x00{callback_hash}".encode()).hexdigest()
        if candle_start_secs != self.last_candle_start_secs.get(key):
            self.last_candle_start_secs[key] = candle_start_secs
            try:
                callback()
            except Exception as e:
                logger.error(
                    f"Error executing callback for {pair}: {repr(e)}", exc_info=True
                )

            threshold_secs = 10 * candle_duration_secs
            keys_to_remove = [
                key
                for key, ts in self.last_candle_start_secs.items()
                if ts is not None and timestamp - ts > threshold_secs
            ]
            for key in keys_to_remove:
                del self.last_candle_start_secs[key]

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        df, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            return None

        stoploss_distance = self.get_stoploss_distance(df, trade, current_rate, 0.7860)
        if isna(stoploss_distance) or stoploss_distance <= 0:
            return None
        return stoploss_from_absolute(
            current_rate + (1 if trade.is_short else -1) * stoploss_distance,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    @staticmethod
    def can_take_profit(
        trade: Trade, current_rate: float, take_profit_price: float
    ) -> bool:
        return (trade.is_short and current_rate <= take_profit_price) or (
            not trade.is_short and current_rate >= take_profit_price
        )

    def get_take_profit_price(
        self, df: DataFrame, trade: Trade, exit_stage: int
    ) -> Optional[float]:
        natr_ratio_percent = (
            self.partial_exit_stages[exit_stage][0]
            if exit_stage in self.partial_exit_stages
            else 1.0
        )
        take_profit_distance = self.get_take_profit_distance(
            df, trade, natr_ratio_percent
        )
        if isna(take_profit_distance) or take_profit_distance <= 0:
            return None

        take_profit_price = (
            trade.open_rate + (-1 if trade.is_short else 1) * take_profit_distance
        )
        self.safe_append_trade_take_profit_price(trade, take_profit_price, exit_stage)

        return take_profit_price

    @staticmethod
    def _get_trade_history(trade: Trade) -> dict[str, list[float | tuple[int, float]]]:
        return trade.get_custom_data(
            "history", {"unrealized_pnl": [], "take_profit_price": []}
        )

    @staticmethod
    def get_trade_unrealized_pnl_history(trade: Trade) -> list[float]:
        history = QuickAdapterV3._get_trade_history(trade)
        return history.get("unrealized_pnl", [])

    @staticmethod
    def get_trade_take_profit_price_history(
        trade: Trade,
    ) -> list[float | tuple[int, float]]:
        history = QuickAdapterV3._get_trade_history(trade)
        return history.get("take_profit_price", [])

    def append_trade_unrealized_pnl(self, trade: Trade, pnl: float) -> list[float]:
        history = QuickAdapterV3._get_trade_history(trade)
        pnl_history = history.setdefault("unrealized_pnl", [])
        pnl_history.append(pnl)
        if len(pnl_history) > self._max_history_size:
            pnl_history = pnl_history[-self._max_history_size :]
            history["unrealized_pnl"] = pnl_history
        trade.set_custom_data("history", history)
        return pnl_history

    def safe_append_trade_unrealized_pnl(self, trade: Trade, pnl: float) -> list[float]:
        trade_unrealized_pnl_history = QuickAdapterV3.get_trade_unrealized_pnl_history(
            trade
        )
        previous_unrealized_pnl = (
            trade_unrealized_pnl_history[-1] if trade_unrealized_pnl_history else None
        )
        if previous_unrealized_pnl is None or not np.isclose(
            previous_unrealized_pnl, pnl
        ):
            trade_unrealized_pnl_history = self.append_trade_unrealized_pnl(trade, pnl)
        return trade_unrealized_pnl_history

    def append_trade_take_profit_price(
        self, trade: Trade, take_profit_price: float, exit_stage: int
    ) -> list[float | tuple[int, float]]:
        history = QuickAdapterV3._get_trade_history(trade)
        price_history = history.setdefault("take_profit_price", [])
        price_history.append((exit_stage, take_profit_price))
        if len(price_history) > self._max_history_size:
            price_history = price_history[-self._max_history_size :]
            history["take_profit_price"] = price_history
        trade.set_custom_data("history", history)
        return price_history

    def safe_append_trade_take_profit_price(
        self, trade: Trade, take_profit_price: float, exit_stage: int
    ) -> list[float | tuple[int, float]]:
        trade_take_profit_price_history = (
            QuickAdapterV3.get_trade_take_profit_price_history(trade)
        )
        previous_take_profit_entry = (
            trade_take_profit_price_history[-1]
            if trade_take_profit_price_history
            else None
        )
        previous_exit_stage = None
        previous_take_profit_price = None
        if isinstance(previous_take_profit_entry, tuple):
            previous_exit_stage = (
                previous_take_profit_entry[0] if previous_take_profit_entry else None
            )
            previous_take_profit_price = (
                previous_take_profit_entry[1] if previous_take_profit_entry else None
            )
        elif isinstance(previous_take_profit_entry, float):
            previous_exit_stage = -1
            previous_take_profit_price = previous_take_profit_entry
        if (
            previous_take_profit_price is None
            or (previous_exit_stage is not None and previous_exit_stage != exit_stage)
            or not np.isclose(previous_take_profit_price, take_profit_price)
        ):
            trade_take_profit_price_history = self.append_trade_take_profit_price(
                trade, take_profit_price, exit_stage
            )
        return trade_take_profit_price_history

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float] | tuple[Optional[float], Optional[str]]:
        if trade.has_open_orders:
            return None

        trade_exit_stage = QuickAdapterV3.get_trade_exit_stage(trade)
        if trade_exit_stage not in self.partial_exit_stages:
            return None

        df, _ = self.dp.get_analyzed_dataframe(
            pair=trade.pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            return None

        trade_take_profit_price = self.get_take_profit_price(
            df, trade, trade_exit_stage
        )
        if isna(trade_take_profit_price):
            return None

        trade_partial_exit = QuickAdapterV3.can_take_profit(
            trade, current_rate, trade_take_profit_price
        )
        if not trade_partial_exit:
            self.throttle_callback(
                pair=trade.pair,
                current_time=current_time,
                callback=lambda: logger.info(
                    f"Trade {trade.trade_direction} {trade.pair} stage {trade_exit_stage} | "
                    f"Take Profit: {format_number(trade_take_profit_price)}, Rate: {format_number(current_rate)}"
                ),
            )
        if trade_partial_exit:
            trade_stake_percent = self.partial_exit_stages[trade_exit_stage][1]
            trade_partial_stake_amount = trade.stake_amount * trade_stake_percent
            return (
                -trade_partial_stake_amount,
                f"take_profit_{trade.trade_direction}_{trade_exit_stage}",
            )

        return None

    @staticmethod
    def weighted_close(series: Series, weight: float = 2.0) -> float:
        return (
            series.get("high") + series.get("low") + weight * series.get("close")
        ) / (2.0 + weight)

    @staticmethod
    def _normalize_candle_idx(length: int, idx: int) -> int:
        """
        Normalize a candle index against a sequence length:
        - supports negative indexing (Python-like),
        - clamps to [0, length-1].
        """
        if length <= 0:
            return 0
        if idx < 0:
            idx = length + idx
        return max(0, min(idx, length - 1))

    def _calculate_candle_deviation(
        self,
        df: DataFrame,
        pair: str,
        min_natr_ratio_percent: float,
        max_natr_ratio_percent: float,
        candle_idx: int = -1,
        interpolation_direction: Literal["direct", "inverse"] = "direct",
        quantile_exponent: float = 1.5,
    ) -> Optional[float]:
        label_natr_series = df.get("natr_label_period_candles")
        if label_natr_series is None or label_natr_series.empty:
            return None

        candle_idx = QuickAdapterV3._normalize_candle_idx(
            len(label_natr_series), candle_idx
        )

        label_natr_values = label_natr_series.iloc[: candle_idx + 1].to_numpy()
        if label_natr_values.size == 0:
            return None
        candle_label_natr_value = label_natr_values[-1]
        if isna(candle_label_natr_value) or candle_label_natr_value < 0:
            return None
        label_period_candles = self.get_label_period_candles(pair)
        candle_label_natr_value_quantile = calculate_quantile(
            label_natr_values[-label_period_candles:], candle_label_natr_value
        )
        if isna(candle_label_natr_value_quantile):
            return None

        if interpolation_direction == "direct":
            natr_ratio_percent = (
                min_natr_ratio_percent
                + (max_natr_ratio_percent - min_natr_ratio_percent)
                * candle_label_natr_value_quantile**quantile_exponent
            )
        elif interpolation_direction == "inverse":
            natr_ratio_percent = (
                max_natr_ratio_percent
                - (max_natr_ratio_percent - min_natr_ratio_percent)
                * candle_label_natr_value_quantile**quantile_exponent
            )
        else:
            raise ValueError(
                f"Invalid interpolation_direction: {interpolation_direction}. Expected 'direct' or 'inverse'"
            )
        return (candle_label_natr_value / 100.0) * self.get_label_natr_ratio_percent(
            pair, natr_ratio_percent
        )

    def calculate_candle_threshold(
        self,
        df: DataFrame,
        pair: str,
        side: str,
        min_natr_ratio_percent: float,
        max_natr_ratio_percent: float,
        candle_idx: int = -1,
    ) -> float:
        current_deviation = self._calculate_candle_deviation(
            df,
            pair,
            min_natr_ratio_percent=min_natr_ratio_percent,
            max_natr_ratio_percent=max_natr_ratio_percent,
            candle_idx=candle_idx,
            interpolation_direction="direct",
        )
        if isna(current_deviation) or current_deviation <= 0:
            return np.nan

        candle_idx = QuickAdapterV3._normalize_candle_idx(len(df), candle_idx)

        candle = df.iloc[candle_idx]
        candle_close = candle.get("close")
        candle_open = candle.get("open")
        if isna(candle_close) or isna(candle_open):
            return np.nan
        is_candle_bullish: bool = candle_close > candle_open
        is_candle_bearish: bool = candle_close < candle_open

        if side == "long":
            base_price = (
                QuickAdapterV3.weighted_close(candle)
                if is_candle_bearish
                else candle_close
            )
            return base_price * (1 + current_deviation)
        elif side == "short":
            base_price = (
                QuickAdapterV3.weighted_close(candle)
                if is_candle_bullish
                else candle_close
            )
            return base_price * (1 - current_deviation)

        raise ValueError(f"Invalid side: {side}. Expected 'long' or 'short'")

    def reversal_confirmed(
        self,
        df: DataFrame,
        pair: str,
        side: str,
        order: Literal["entry", "exit"],
        rate: float,
        lookback_period: int,
        decay_ratio: float,
        min_natr_ratio_percent: float,
        max_natr_ratio_percent: float,
    ) -> bool:
        """Confirm a directional reversal using a volatility-adaptive current-candle
        threshold and optionally a backward confirmation chain with geometric decay.

        Overview
        --------
        1. Compute a deviation-based threshold on the latest candle (-1). The current
           rate must strictly break it (long: rate > threshold; short: rate < threshold).
        2. If lookback_period > 0, for each k = 1..lookback_period:
             - Decay (min_natr_ratio_percent, max_natr_ratio_percent) by (decay_ratio ** k),
               clamped to [0, 1].
             - Recompute the threshold on candle index -(k+1).
             - Require close[-k] to have strictly broken that historical threshold.
        3. If an intermediate close or threshold is non-finite, chain evaluation aborts
           and the function falls back to step 1 result only (permissive fallback).

        Parameters
        ----------
        df : DataFrame
            Must contain 'open', 'close' and the NATR label series used indirectly.
        pair : str
            Trading pair identifier.
        side : {'long','short'}
            Direction to confirm.
        order : {'entry','exit'}
            Context (affects log wording only).
        rate : float
            Candidate execution price; must break the current threshold.
        lookback_period : int
            Number of historical confirmation steps requested; truncated to history.
        decay_ratio : float
            Geometric decay factor per step (0 < decay_ratio <= 1); 1.0 disables decay.
        min_natr_ratio_percent : float
            Lower bound fraction (e.g. 0.009 = 0.9%).
        max_natr_ratio_percent : float
            Upper bound fraction (>= lower bound).

        Returns
        -------
        bool
            True iff the current threshold is broken AND (lookback chain succeeded OR
            a permissive fallback occurred). False otherwise.

        Fallback Semantics
        ------------------
        Missing / non-finite intermediate data ⇒ stop chain; return current candle result.
        This may yield True on partial history, weakening strict multi-candle guarantees.

        Rejection Conditions
        --------------------
        Empty dataframe, invalid side/order, negative lookback, decay_ratio outside (0,1],
        failure to break current threshold, or failed historical step comparison.

        Complexity
        ----------
        O(lookback_period) threshold computations.

        Logging
        -------
        Logs rejection reasons (invalid decay_ratio, threshold not broken, failed step).
        Fallback aborts are silent.

        Limitations
        -----------
        No validation of min/max ordering beyond usage; no strict mode; partial data may
        still confirm. Rate finiteness not explicitly validated.
        """
        if df.empty:
            return False
        if side not in {"long", "short"}:
            return False
        if order not in {"entry", "exit"}:
            return False

        trade_direction = side

        max_lookback_period = max(0, len(df) - 1)
        if lookback_period > max_lookback_period:
            lookback_period = max_lookback_period
        if not isinstance(decay_ratio, (int, float)):
            logger.info(
                f"User denied {trade_direction} {order} for {pair}: invalid decay_ratio type"
            )
            return False
        if not (0.0 < decay_ratio <= 1.0):
            logger.info(
                f"User denied {trade_direction} {order} for {pair}: invalid decay_ratio {decay_ratio}, must be in (0, 1]"
            )
            return False

        current_threshold = self.calculate_candle_threshold(
            df,
            pair,
            side,
            min_natr_ratio_percent=min_natr_ratio_percent,
            max_natr_ratio_percent=max_natr_ratio_percent,
            candle_idx=-1,
        )
        current_ok = np.isfinite(current_threshold) and (
            (side == "long" and rate > current_threshold)
            or (side == "short" and rate < current_threshold)
        )
        if order == "exit":
            if side == "long":
                trade_direction = "short"
            if side == "short":
                trade_direction = "long"
        if not current_ok:
            logger.info(
                f"User denied {trade_direction} {order} for {pair}: rate {format_number(rate)} did not break threshold {format_number(current_threshold)}"
            )
            return False

        if lookback_period == 0:
            return current_ok

        for k in range(1, lookback_period + 1):
            close_k = df.iloc[-k].get("close")
            if not isinstance(close_k, (int, float)) or not np.isfinite(close_k):
                return current_ok

            decay_factor = decay_ratio**k
            decayed_min_natr_ratio_percent = max(
                0.0, min(1.0, min_natr_ratio_percent * decay_factor)
            )
            decayed_max_natr_ratio_percent = max(
                decayed_min_natr_ratio_percent,
                min(1.0, max_natr_ratio_percent * decay_factor),
            )

            threshold_k = self.calculate_candle_threshold(
                df,
                pair,
                side,
                min_natr_ratio_percent=decayed_min_natr_ratio_percent,
                max_natr_ratio_percent=decayed_max_natr_ratio_percent,
                candle_idx=-(k + 1),
            )
            if not isinstance(threshold_k, (int, float)) or not np.isfinite(
                threshold_k
            ):
                return current_ok

            if (side == "long" and not (close_k > threshold_k)) or (
                side == "short" and not (close_k < threshold_k)
            ):
                logger.info(
                    f"User denied {trade_direction} {order} for {pair}: "
                    f"close_k[{-k}] {format_number(close_k)} "
                    f"did not break threshold_k[{-(k + 1)}] {format_number(threshold_k)} "
                    f"(decayed natr_ratio_percent: min={format_number(decayed_min_natr_ratio_percent)}, max={format_number(decayed_max_natr_ratio_percent)})"
                )
                return False

        return True

    @staticmethod
    def get_pnl_momentum(
        unrealized_pnl_history: Sequence[float], window_size: int
    ) -> tuple[float, float, float, float, float, float, float, float]:
        unrealized_pnl_history = np.asarray(unrealized_pnl_history)

        velocity = np.diff(unrealized_pnl_history)
        velocity_std = np.std(velocity, ddof=1) if velocity.size > 1 else 0.0
        acceleration = np.diff(velocity)
        acceleration_std = (
            np.std(acceleration, ddof=1) if acceleration.size > 1 else 0.0
        )

        mean_velocity = np.mean(velocity) if velocity.size > 0 else 0.0
        mean_acceleration = np.mean(acceleration) if acceleration.size > 0 else 0.0

        if window_size > 0 and len(unrealized_pnl_history) > window_size:
            recent_unrealized_pnl_history = unrealized_pnl_history[-window_size:]
        else:
            recent_unrealized_pnl_history = unrealized_pnl_history

        recent_velocity = np.diff(recent_unrealized_pnl_history)
        recent_velocity_std = (
            np.std(recent_velocity, ddof=1) if recent_velocity.size > 1 else 0.0
        )
        recent_acceleration = np.diff(recent_velocity)
        recent_acceleration_std = (
            np.std(recent_acceleration, ddof=1) if recent_acceleration.size > 1 else 0.0
        )

        recent_mean_velocity = (
            np.mean(recent_velocity) if recent_velocity.size > 0 else 0.0
        )
        recent_mean_acceleration = (
            np.mean(recent_acceleration) if recent_acceleration.size > 0 else 0.0
        )

        return (
            mean_velocity,
            velocity_std,
            mean_acceleration,
            acceleration_std,
            recent_mean_velocity,
            recent_velocity_std,
            recent_mean_acceleration,
            recent_acceleration_std,
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _zscore(mean: float, std: float) -> float:
        if not np.isfinite(mean) or not np.isfinite(std):
            return np.nan
        if np.isclose(std, 0.0):
            return np.nan
        return mean / std

    @staticmethod
    @lru_cache(maxsize=128)
    def is_isoformat(string: str) -> bool:
        if not isinstance(string, str):
            return False
        try:
            datetime.datetime.fromisoformat(string)
        except (ValueError, TypeError):
            return False
        return True

    def _get_exit_thresholds(
        self,
        hist_len: int,
        std_v_global: float,
        std_a_global: float,
        std_v_recent: float,
        std_a_recent: float,
        min_alpha: float = 0.05,
    ) -> dict[str, float]:
        q_decl = float(self._exit_thresholds_calibration.get("decline_quantile"))

        recent_hist_len = min(hist_len, self._pnl_momentum_window_size)

        n_v_global = max(0, hist_len - 1)
        n_a_global = max(0, hist_len - 2)
        n_v_recent = max(0, recent_hist_len - 1)
        n_a_recent = max(0, recent_hist_len - 2)

        if hist_len <= 0:
            alpha_len = 1.0
        else:
            alpha_len = recent_hist_len / hist_len
        alpha_len = max(min_alpha, alpha_len)

        def volatility_adjusted_alpha(
            alpha_base: float,
            sigma_global: float,
            sigma_recent: float,
            gamma: float = 1.25,
            min_alpha: float = 0.05,
        ) -> float:
            if not (np.isfinite(sigma_global) and np.isfinite(sigma_recent)):
                return alpha_base
            if sigma_global <= 0 and sigma_recent <= 0:
                return alpha_base
            sigma_total = sigma_global + sigma_recent
            if sigma_total <= 0:
                return alpha_base
            ratio = sigma_global / sigma_total
            alpha_vol = alpha_base * (ratio**gamma)
            return max(min_alpha, alpha_vol)

        alpha_v = volatility_adjusted_alpha(
            alpha_len, std_v_global, std_v_recent, min_alpha=min_alpha
        )
        alpha_a = volatility_adjusted_alpha(
            alpha_len, std_a_global, std_a_recent, min_alpha=min_alpha
        )
        n_eff_v = alpha_v * n_v_recent + (1.0 - alpha_v) * n_v_global
        n_eff_a = alpha_a * n_a_recent + (1.0 - alpha_a) * n_a_global

        def effective_k(
            q: float,
            n_eff: float,
            default_k: float,
        ) -> float:
            if not (0.0 < q < 1.0) or np.isclose(q, 0.0) or np.isclose(q, 1.0):
                return default_k
            try:
                if n_eff < 2:
                    return default_k
                df_eff = max(n_eff - 1.0, 1.0)
                k = float(t.ppf(q, df_eff)) / math.sqrt(n_eff)
                if not np.isfinite(k):
                    return default_k
                return k
            except Exception:
                return default_k

        k_decl_v = effective_k(
            q_decl, n_eff_v, QuickAdapterV3.default_exit_thresholds["k_decl_v"]
        )
        k_decl_a = effective_k(
            q_decl, n_eff_a, QuickAdapterV3.default_exit_thresholds["k_decl_a"]
        )

        if debug:
            logger.info(
                (
                    "hist_len=%s recent_len=%s | alpha_len=%s | q_decl=%s | "
                    "n_v_(global,recent)=(%s,%s) n_a_(global,recent)=(%s,%s) | "
                    "std_v_(global,recent)=(%s,%s) std_a_(global,recent)=(%s,%s) | "
                    "alpha_(v,a)=(%s,%s) | n_eff_(v,a)=(%s,%s) | "
                    "k_decl_(v,a)=(%s,%s)"
                ),
                hist_len,
                recent_hist_len,
                format_number(alpha_len),
                format_number(q_decl),
                n_v_global,
                n_v_recent,
                n_a_global,
                n_a_recent,
                format_number(std_v_global),
                format_number(std_v_recent),
                format_number(std_a_global),
                format_number(std_a_recent),
                format_number(alpha_v),
                format_number(alpha_a),
                format_number(n_eff_v),
                format_number(n_eff_a),
                format_number(k_decl_v),
                format_number(k_decl_a),
            )

        return {
            "k_decl_v": k_decl_v,
            "k_decl_a": k_decl_a,
        }

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        self.safe_append_trade_unrealized_pnl(trade, current_profit)

        df, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            return None

        last_candle = df.iloc[-1]
        if last_candle.get("do_predict") == 2:
            return "model_expired"
        if last_candle.get("DI_catch") == 0:
            last_candle_date = last_candle.get("date")
            last_outlier_date_isoformat = trade.get_custom_data("last_outlier_date")
            last_outlier_date = (
                datetime.datetime.fromisoformat(last_outlier_date_isoformat)
                if QuickAdapterV3.is_isoformat(last_outlier_date_isoformat)
                else None
            )
            if last_outlier_date != last_candle_date:
                n_outliers = trade.get_custom_data("n_outliers", 0)
                n_outliers += 1
                logger.warning(
                    f"{pair}: detected new predictions outlier ({n_outliers=}) on trade {trade.id}"
                )
                trade.set_custom_data("n_outliers", n_outliers)
                trade.set_custom_data("last_outlier_date", last_candle_date.isoformat())

        if (
            trade.trade_direction == "short"
            and last_candle.get("do_predict") == 1
            and last_candle.get("DI_catch") == 1
            and last_candle.get(EXTREMA_COLUMN) < last_candle.get("minima_threshold")
            and self.reversal_confirmed(
                df,
                pair,
                "long",
                "exit",
                current_rate,
                self._reversal_lookback_period,
                self._reversal_decay_ratio,
                self._reversal_min_natr_ratio_percent,
                self._reversal_max_natr_ratio_percent,
            )
        ):
            return "minima_detected_short"
        if (
            trade.trade_direction == "long"
            and last_candle.get("do_predict") == 1
            and last_candle.get("DI_catch") == 1
            and last_candle.get(EXTREMA_COLUMN) > last_candle.get("maxima_threshold")
            and self.reversal_confirmed(
                df,
                pair,
                "short",
                "exit",
                current_rate,
                self._reversal_lookback_period,
                self._reversal_decay_ratio,
                self._reversal_min_natr_ratio_percent,
                self._reversal_max_natr_ratio_percent,
            )
        ):
            return "maxima_detected_long"

        trade_exit_stage = QuickAdapterV3.get_trade_exit_stage(trade)
        if trade_exit_stage in self.partial_exit_stages:
            return None

        trade_take_profit_price = self.get_take_profit_price(
            df, trade, trade_exit_stage
        )
        if isna(trade_take_profit_price):
            return None
        trade_take_profit_exit = QuickAdapterV3.can_take_profit(
            trade, current_rate, trade_take_profit_price
        )

        if not trade_take_profit_exit:
            self.throttle_callback(
                pair=pair,
                current_time=current_time,
                callback=lambda: logger.info(
                    f"Trade {trade.trade_direction} {trade.pair} stage {trade_exit_stage} | "
                    f"Take Profit: {format_number(trade_take_profit_price)}, Rate: {format_number(current_rate)}"
                ),
            )
            return None

        trade_unrealized_pnl_history = QuickAdapterV3.get_trade_unrealized_pnl_history(
            trade
        )
        (
            _,
            trade_global_pnl_velocity_std,
            _,
            trade_global_pnl_acceleration_std,
            trade_recent_pnl_velocity,
            trade_recent_pnl_velocity_std,
            trade_recent_pnl_acceleration,
            trade_recent_pnl_acceleration_std,
        ) = QuickAdapterV3.get_pnl_momentum(
            trade_unrealized_pnl_history, self._pnl_momentum_window_size
        )

        z_recent_v = QuickAdapterV3._zscore(
            trade_recent_pnl_velocity, trade_recent_pnl_velocity_std
        )
        z_recent_a = QuickAdapterV3._zscore(
            trade_recent_pnl_acceleration, trade_recent_pnl_acceleration_std
        )

        trade_hist_len = len(trade_unrealized_pnl_history)
        trade_exit_thresholds = self._get_exit_thresholds(
            hist_len=trade_hist_len,
            std_v_global=trade_global_pnl_velocity_std,
            std_a_global=trade_global_pnl_acceleration_std,
            std_v_recent=trade_recent_pnl_velocity_std,
            std_a_recent=trade_recent_pnl_acceleration_std,
        )
        k_decl_v = trade_exit_thresholds.get("k_decl_v")
        k_decl_a = trade_exit_thresholds.get("k_decl_a")

        decl_checks: list[bool] = []
        if np.isfinite(z_recent_v):
            decl_checks.append(z_recent_v <= -k_decl_v)
        if np.isfinite(z_recent_a):
            decl_checks.append(z_recent_a <= -k_decl_a)
        if len(decl_checks) == 0:
            trade_recent_pnl_declining = True
        else:
            trade_recent_pnl_declining = all(decl_checks)

        trade_exit = trade_take_profit_exit and trade_recent_pnl_declining

        if not trade_exit:
            self.throttle_callback(
                pair=pair,
                current_time=current_time,
                callback=lambda: logger.info(
                    f"Trade {trade.trade_direction} {trade.pair} stage {trade_exit_stage} | "
                    f"Take Profit: {format_number(trade_take_profit_price)}, Rate: {format_number(current_rate)} | "
                    f"Declining: {trade_recent_pnl_declining} "
                    f"(zV:{format_number(z_recent_v)}<=-k:{format_number(-k_decl_v)}, zA:{format_number(z_recent_a)}<=-k:{format_number(-k_decl_a)})"
                ),
            )

        if trade_exit:
            return f"take_profit_{trade.trade_direction}_{trade_exit_stage}"

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
        if side not in {"long", "short"}:
            return False
        if side == "short" and not self.can_short:
            logger.info(f"User denied short entry for {pair}: shorting not allowed")
            return False
        if Trade.get_open_trade_count() >= self.config.get("max_open_trades"):
            return False
        max_open_trades_per_side = self.max_open_trades_per_side
        if max_open_trades_per_side >= 0:
            open_trades = Trade.get_open_trades()
            trades_per_side = sum(
                1 for trade in open_trades if trade.trade_direction == side
            )
            if trades_per_side >= max_open_trades_per_side:
                return False

        df, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.config.get("timeframe")
        )
        if df.empty:
            logger.info(f"User denied {side} entry for {pair}: dataframe is empty")
            return False
        if self.reversal_confirmed(
            df,
            pair,
            side,
            "entry",
            rate,
            self._reversal_lookback_period,
            self._reversal_decay_ratio,
            self._reversal_min_natr_ratio_percent,
            self._reversal_max_natr_ratio_percent,
        ):
            return True
        return False

    def is_short_allowed(self) -> bool:
        trading_mode = self.config.get("trading_mode")
        if trading_mode in {"margin", "futures"}:
            return True
        elif trading_mode == "spot":
            return False
        else:
            raise ValueError(f"Invalid trading_mode: {trading_mode}")

    def leverage(
        self,
        pair: str,
        current_time: datetime.datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs: Any,
    ) -> float:
        """
        Customize leverage for each new trade. This method is only called in trading modes
        which allow leverage (margin / futures). The strategy is expected to return a
        leverage value between 1.0 and max_leverage.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which will be between 1.0 and max_leverage.
        """
        return min(self.config.get("leverage", proposed_leverage), max_leverage)

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
