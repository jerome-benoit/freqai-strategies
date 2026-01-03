import datetime
import hashlib
import json
import logging
import math
from functools import cached_property, lru_cache, reduce
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Literal,
    Optional,
    Sequence,
)

import numpy as np
import pandas_ta as pta
import talib.abstract as ta
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.persistence import Trade
from freqtrade.strategy import AnnotationType, stoploss_from_absolute
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series, isna
from scipy.stats import pearsonr, t
from technical.pivots_points import pivots_points

from user_data.strategies.Utils import (
    DEFAULT_FIT_LIVE_PREDICTIONS_CANDLES,
    DEFAULTS_EXTREMA_SMOOTHING,
    EXTREMA_COLUMN,
    MAXIMA_THRESHOLD_COLUMN,
    MINIMA_THRESHOLD_COLUMN,
    SMOOTHING_METHODS,
    SMOOTHING_MODES,
    TRADE_PRICE_TARGETS,
    alligator,
    bottom_change_percent,
    calculate_quantile,
    ewo,
    format_number,
    get_callable_sha256,
    get_distance,
    get_extrema_weighting_config,
    get_label_defaults,
    get_weighted_extrema,
    get_zl_ma_fn,
    nan_average,
    non_zero_diff,
    price_retracement_percent,
    smooth_extrema,
    top_change_percent,
    update_config_value,
    validate_range,
    vwapb,
    zigzag,
    zlema,
)

TradeDirection = Literal["long", "short"]
InterpolationDirection = Literal["direct", "inverse"]
OrderType = Literal["entry", "exit"]
TradingMode = Literal["spot", "margin", "futures"]

DfSignature = tuple[int, Optional[datetime.datetime]]
CandleDeviationCacheKey = tuple[
    str, DfSignature, float, float, int, InterpolationDirection, float
]
CandleThresholdCacheKey = tuple[str, DfSignature, str, int, float, float]

logger = logging.getLogger(__name__)


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

    _TRADE_DIRECTIONS: Final[tuple[TradeDirection, ...]] = ("long", "short")
    _INTERPOLATION_DIRECTIONS: Final[tuple[InterpolationDirection, ...]] = (
        "direct",
        "inverse",
    )
    _ORDER_TYPES: Final[tuple[OrderType, ...]] = ("entry", "exit")
    _TRADING_MODES: Final[tuple[TradingMode, ...]] = ("spot", "margin", "futures")

    def version(self) -> str:
        return "3.9.2"

    timeframe = "5m"
    timeframe_minutes = timeframe_to_minutes(timeframe)

    stoploss = -0.025
    use_custom_stoploss = True

    default_exit_thresholds: ClassVar[dict[str, float]] = {
        "t_decl_v": 0.675,
        "t_decl_a": 0.675,
    }

    default_exit_thresholds_calibration: ClassVar[dict[str, float]] = {
        "decline_quantile": 0.75,
    }

    default_reversal_confirmation: ClassVar[dict[str, int | float]] = {
        "lookback_period_candles": 0,
        "decay_fraction": 0.5,
        "min_natr_multiplier_fraction": 0.0095,
        "max_natr_multiplier_fraction": 0.075,
    }

    position_adjustment_enable = True

    # {stage: (natr_multiplier_fraction, stake_percent, color)}
    partial_exit_stages: ClassVar[dict[int, tuple[float, float, str]]] = {
        0: (0.4858, 0.4, "lime"),
        1: (0.6180, 0.3, "yellow"),
        2: (0.7640, 0.2, "coral"),
    }

    # (natr_multiplier_fraction, stake_percent, color)
    _FINAL_EXIT_STAGE: Final[tuple[float, float, str]] = (1.0, 1.0, "deepskyblue")

    _CUSTOM_STOPLOSS_NATR_MULTIPLIER_FRACTION: Final[float] = 0.7860

    _ANNOTATION_LINE_OFFSET_CANDLES: Final[int] = 10

    _PLOT_EXTREMA_MIN_EPS: Final[float] = 0.01

    minimal_roi = {str(timeframe_minutes * 864): -1}

    # FreqAI is crashing if minimal_roi is a property
    # @property
    # def minimal_roi(self) -> dict[str, Any]:
    #     timeframe_minutes = self.get_timeframe_minutes()
    #     fit_live_predictions_candles = int(
    #         self.config.get("freqai", {}).get(
    #             "fit_live_predictions_candles", DEFAULT_FIT_LIVE_PREDICTIONS_CANDLES
    #         )
    #     )
    #     return {str(timeframe_minutes * fit_live_predictions_candles): -1}

    # @minimal_roi.setter
    # def minimal_roi(self, value: dict[str, Any]) -> None:
    #     pass

    process_only_new_candles = True

    @staticmethod
    @lru_cache(maxsize=None)
    def _trade_directions_set() -> set[TradeDirection]:
        return set(QuickAdapterV3._TRADE_DIRECTIONS)

    @staticmethod
    @lru_cache(maxsize=None)
    def _order_types_set() -> set[OrderType]:
        return set(QuickAdapterV3._ORDER_TYPES)

    @lru_cache(maxsize=None)
    def get_timeframe_minutes(self) -> int:
        return timeframe_to_minutes(self.config.get("timeframe"))

    @property
    def can_short(self) -> bool:
        return self.is_short_allowed()

    @cached_property
    def plot_config(self) -> dict[str, Any]:
        return {
            "main_plot": {},
            "subplots": {
                "accuracy": {
                    "hp_rmse": {"color": "violet", "type": "line"},
                },
                "extrema": {
                    MAXIMA_THRESHOLD_COLUMN: {"color": "blue", "type": "line"},
                    MINIMA_THRESHOLD_COLUMN: {"color": "cyan", "type": "line"},
                    EXTREMA_COLUMN: {"color": "orange", "type": "line"},
                },
                "min_max": {
                    "maxima": {"color": "red", "type": "bar"},
                    "minima": {"color": "green", "type": "bar"},
                },
            },
        }

    @property
    def protections(self) -> list[dict[str, Any]]:
        fit_live_predictions_candles = int(
            self.config.get("freqai", {}).get(
                "fit_live_predictions_candles", DEFAULT_FIT_LIVE_PREDICTIONS_CANDLES
            )
        )
        protections = self.config.get("custom_protections", {})
        trade_duration_candles = int(protections.get("trade_duration_candles", 72))
        lookback_period_fraction = float(
            protections.get("lookback_period_fraction", 0.5)
        )

        lookback_period_candles = max(
            1, int(round(fit_live_predictions_candles * lookback_period_fraction))
        )

        cooldown = protections.get("cooldown", {})
        cooldown_stop_duration_candles = int(cooldown.get("stop_duration_candles", 4))
        stoploss_stop_duration_candles = max(
            cooldown_stop_duration_candles, trade_duration_candles
        )
        drawdown_stop_duration_candles = max(
            stoploss_stop_duration_candles,
            fit_live_predictions_candles,
        )
        max_open_trades = int(self.config.get("max_open_trades", 0))
        stoploss_trade_limit = min(
            max(
                2,
                int(round(lookback_period_candles / max(1, trade_duration_candles))),
            ),
            max(2, int(round(max_open_trades * 0.75))),
        )

        protections_list = []

        if cooldown.get("enabled", True):
            protections_list.append(
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": cooldown_stop_duration_candles,
                }
            )

        drawdown = protections.get("drawdown", {})
        if drawdown.get("enabled", True):
            protections_list.append(
                {
                    "method": "MaxDrawdown",
                    "lookback_period_candles": lookback_period_candles,
                    "trade_limit": 2 * max_open_trades,
                    "stop_duration_candles": drawdown_stop_duration_candles,
                    "max_allowed_drawdown": float(
                        drawdown.get("max_allowed_drawdown", 0.2)
                    ),
                }
            )

        stoploss = protections.get("stoploss", {})
        if stoploss.get("enabled", True):
            protections_list.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": lookback_period_candles,
                    "trade_limit": stoploss_trade_limit,
                    "stop_duration_candles": stoploss_stop_duration_candles,
                    "only_per_pair": True,
                }
            )

        return protections_list

    use_exit_signal = True

    @property
    def startup_candle_count(self) -> int:
        # Match the predictions warmup period
        return self.config.get("freqai", {}).get(
            "fit_live_predictions_candles", DEFAULT_FIT_LIVE_PREDICTIONS_CANDLES
        )

    @property
    def max_open_trades_per_side(self) -> int:
        max_open_trades = self.config.get("max_open_trades", 0)
        if max_open_trades < 0:
            return -1
        if self.is_short_allowed():
            if max_open_trades % 2 == 1:
                max_open_trades += 1
            return int(max_open_trades / 2)
        else:
            return max_open_trades

    @property
    def extrema_weighting(self) -> dict[str, Any]:
        extrema_weighting = self.freqai_info.get("extrema_weighting", {})
        if not isinstance(extrema_weighting, dict):
            extrema_weighting = {}
        return get_extrema_weighting_config(extrema_weighting, logger)

    @property
    def extrema_smoothing(self) -> dict[str, Any]:
        extrema_smoothing = self.freqai_info.get("extrema_smoothing", {})
        if not isinstance(extrema_smoothing, dict):
            extrema_smoothing = {}
        method = extrema_smoothing.get("method", DEFAULTS_EXTREMA_SMOOTHING["method"])
        if method not in set(SMOOTHING_METHODS):
            logger.warning(
                f"Invalid extrema_smoothing method {method!r}, supported: {', '.join(SMOOTHING_METHODS)}, using default {SMOOTHING_METHODS[0]!r}"
            )
            method = SMOOTHING_METHODS[0]

        window_candles = update_config_value(
            extrema_smoothing,
            new_key="window_candles",
            old_key="window",
            default=DEFAULTS_EXTREMA_SMOOTHING["window_candles"],
            logger=logger,
            new_path="freqai.extrema_smoothing.window_candles",
            old_path="freqai.extrema_smoothing.window",
        )
        if not isinstance(window_candles, int) or window_candles < 3:
            logger.warning(
                f"Invalid extrema_smoothing window_candles {window_candles!r}: must be an integer >= 3, using default {DEFAULTS_EXTREMA_SMOOTHING['window_candles']!r}"
            )
            window_candles = int(DEFAULTS_EXTREMA_SMOOTHING["window_candles"])

        beta = extrema_smoothing.get("beta", DEFAULTS_EXTREMA_SMOOTHING["beta"])
        if not isinstance(beta, (int, float)) or not np.isfinite(beta) or beta <= 0:
            logger.warning(
                f"Invalid extrema_smoothing beta {beta!r}: must be a finite number > 0, using default {DEFAULTS_EXTREMA_SMOOTHING['beta']!r}"
            )
            beta = DEFAULTS_EXTREMA_SMOOTHING["beta"]

        polyorder = extrema_smoothing.get(
            "polyorder", DEFAULTS_EXTREMA_SMOOTHING["polyorder"]
        )
        if not isinstance(polyorder, int) or polyorder < 1:
            logger.warning(
                f"Invalid extrema_smoothing polyorder {polyorder!r}: must be an integer >= 1, using default {DEFAULTS_EXTREMA_SMOOTHING['polyorder']!r}"
            )
            polyorder = DEFAULTS_EXTREMA_SMOOTHING["polyorder"]

        mode = str(extrema_smoothing.get("mode", DEFAULTS_EXTREMA_SMOOTHING["mode"]))
        if mode not in set(SMOOTHING_MODES):
            logger.warning(
                f"Invalid extrema_smoothing mode {mode!r}, supported: {', '.join(SMOOTHING_MODES)}, using default {SMOOTHING_MODES[0]!r}"
            )
            mode = SMOOTHING_MODES[0]

        sigma = extrema_smoothing.get("sigma", DEFAULTS_EXTREMA_SMOOTHING["sigma"])
        if not isinstance(sigma, (int, float)) or sigma <= 0 or not np.isfinite(sigma):
            logger.warning(
                f"Invalid extrema_smoothing sigma {sigma!r}: must be a finite number > 0, using default {DEFAULTS_EXTREMA_SMOOTHING['sigma']!r}"
            )
            sigma = DEFAULTS_EXTREMA_SMOOTHING["sigma"]

        return {
            "method": method,
            "window_candles": window_candles,
            "beta": beta,
            "polyorder": polyorder,
            "mode": mode,
            "sigma": sigma,
        }

    @property
    def trade_price_target_method(self) -> str:
        exit_pricing = self.config.get("exit_pricing", {})
        trade_price_target_method = update_config_value(
            exit_pricing,
            new_key="trade_price_target_method",
            old_key="trade_price_target",
            default=TRADE_PRICE_TARGETS[0],  # "moving_average"
            logger=logger,
            new_path="exit_pricing.trade_price_target_method",
            old_path="exit_pricing.trade_price_target",
        )
        if trade_price_target_method not in set(TRADE_PRICE_TARGETS):
            logger.warning(
                f"Invalid trade_price_target_method {trade_price_target_method!r}. "
                f"Supported: {', '.join(TRADE_PRICE_TARGETS)}. "
                f"Using default {TRADE_PRICE_TARGETS[0]!r}"
            )
            trade_price_target_method = TRADE_PRICE_TARGETS[0]
        return str(trade_price_target_method)

    @property
    def reversal_confirmation(self) -> dict[str, int | float]:
        reversal_confirmation = self.config.get("reversal_confirmation", {})

        lookback_period_candles = update_config_value(
            reversal_confirmation,
            new_key="lookback_period_candles",
            old_key="lookback_period",
            default=QuickAdapterV3.default_reversal_confirmation[
                "lookback_period_candles"
            ],
            logger=logger,
            new_path="reversal_confirmation.lookback_period_candles",
            old_path="reversal_confirmation.lookback_period",
        )
        decay_fraction = update_config_value(
            reversal_confirmation,
            new_key="decay_fraction",
            old_key="decay_ratio",
            default=QuickAdapterV3.default_reversal_confirmation["decay_fraction"],
            logger=logger,
            new_path="reversal_confirmation.decay_fraction",
            old_path="reversal_confirmation.decay_ratio",
        )

        min_natr_multiplier_fraction = update_config_value(
            reversal_confirmation,
            new_key="min_natr_multiplier_fraction",
            old_key="min_natr_ratio_percent",
            default=QuickAdapterV3.default_reversal_confirmation[
                "min_natr_multiplier_fraction"
            ],
            logger=logger,
            new_path="reversal_confirmation.min_natr_multiplier_fraction",
            old_path="reversal_confirmation.min_natr_ratio_percent",
        )
        max_natr_multiplier_fraction = update_config_value(
            reversal_confirmation,
            new_key="max_natr_multiplier_fraction",
            old_key="max_natr_ratio_percent",
            default=QuickAdapterV3.default_reversal_confirmation[
                "max_natr_multiplier_fraction"
            ],
            logger=logger,
            new_path="reversal_confirmation.max_natr_multiplier_fraction",
            old_path="reversal_confirmation.max_natr_ratio_percent",
        )

        if not isinstance(lookback_period_candles, int) or lookback_period_candles < 0:
            logger.warning(
                f"Invalid reversal_confirmation lookback_period_candles {lookback_period_candles!r}: must be >= 0, using default {QuickAdapterV3.default_reversal_confirmation['lookback_period_candles']!r}"
            )
            lookback_period_candles = QuickAdapterV3.default_reversal_confirmation[
                "lookback_period_candles"
            ]

        if not isinstance(decay_fraction, (int, float)) or not (
            0.0 < decay_fraction <= 1.0
        ):
            logger.warning(
                f"Invalid reversal_confirmation decay_fraction {decay_fraction!r}: must be in range (0, 1], using default {QuickAdapterV3.default_reversal_confirmation['decay_fraction']!r}"
            )
            decay_fraction = QuickAdapterV3.default_reversal_confirmation[
                "decay_fraction"
            ]

        min_natr_multiplier_fraction, max_natr_multiplier_fraction = validate_range(
            min_natr_multiplier_fraction,
            max_natr_multiplier_fraction,
            logger,
            name="natr_multiplier_fraction",
            default_min=QuickAdapterV3.default_reversal_confirmation[
                "min_natr_multiplier_fraction"
            ],
            default_max=QuickAdapterV3.default_reversal_confirmation[
                "max_natr_multiplier_fraction"
            ],
            allow_equal=False,
            non_negative=True,
            finite_only=True,
        )

        return {
            "lookback_period_candles": int(lookback_period_candles),
            "decay_fraction": float(decay_fraction),
            "min_natr_multiplier_fraction": float(min_natr_multiplier_fraction),
            "max_natr_multiplier_fraction": float(max_natr_multiplier_fraction),
        }

    @property
    def _label_defaults(self) -> tuple[int, float]:
        feature_parameters = self.freqai_info.get("feature_parameters", {})
        return get_label_defaults(feature_parameters, logger)

    def bot_start(self, **kwargs) -> None:
        self.pairs: list[str] = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.pairs:
            raise ValueError(
                "Invalid configuration: FreqAI strategy requires StaticPairList method in pairlists and 'pair_whitelist' in exchange section"
            )
        if (
            not isinstance(self.freqai_info.get("identifier"), str)
            or not self.freqai_info.get("identifier", "").strip()
        ):
            raise ValueError(
                "Invalid freqai configuration: 'identifier' must be defined in freqai section"
            )
        self.models_full_path = Path(
            self.config.get("user_data_dir")
            / "models"
            / self.freqai_info.get("identifier")
        )
        feature_parameters = self.freqai_info.get("feature_parameters", {})
        default_label_period_candles, default_label_natr_multiplier = (
            self._label_defaults
        )
        self._label_params: dict[str, dict[str, Any]] = {}
        for pair in self.pairs:
            self._label_params[pair] = (
                self.optuna_load_best_params(pair, "label")
                if self.optuna_load_best_params(pair, "label")
                else {
                    "label_period_candles": feature_parameters.get(
                        "label_period_candles",
                        default_label_period_candles,
                    ),
                    "label_natr_multiplier": float(
                        feature_parameters.get(
                            "label_natr_multiplier",
                            default_label_natr_multiplier,
                        )
                    ),
                }
            )
        self._candle_duration_secs = int(self.get_timeframe_minutes() * 60)
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
        self._candle_deviation_cache: dict[CandleDeviationCacheKey, float] = {}
        self._candle_threshold_cache: dict[CandleThresholdCacheKey, float] = {}
        self._cached_df_signature: dict[str, DfSignature] = {}

        self._log_strategy_configuration()

    def _log_strategy_configuration(self) -> None:
        logger.info("=" * 60)
        logger.info("QuickAdapter Strategy Configuration")
        logger.info("=" * 60)

        logger.info("Extrema Weighting:")
        logger.info(f"  strategy: {self.extrema_weighting['strategy']}")
        formatted_source_weights = {
            k: format_number(v)
            for k, v in self.extrema_weighting["source_weights"].items()
        }
        logger.info(f"  source_weights: {formatted_source_weights}")
        logger.info(f"  aggregation: {self.extrema_weighting['aggregation']}")
        logger.info(f"  standardization: {self.extrema_weighting['standardization']}")
        logger.info(
            f"  robust_quantiles: ({format_number(self.extrema_weighting['robust_quantiles'][0])}, {format_number(self.extrema_weighting['robust_quantiles'][1])})"
        )
        logger.info(
            f"  mmad_scaling_factor: {format_number(self.extrema_weighting['mmad_scaling_factor'])}"
        )
        logger.info(f"  normalization: {self.extrema_weighting['normalization']}")
        logger.info(
            f"  minmax_range: ({format_number(self.extrema_weighting['minmax_range'][0])}, {format_number(self.extrema_weighting['minmax_range'][1])})"
        )
        logger.info(
            f"  sigmoid_scale: {format_number(self.extrema_weighting['sigmoid_scale'])}"
        )
        logger.info(f"  gamma: {format_number(self.extrema_weighting['gamma'])}")

        logger.info("Extrema Smoothing:")
        logger.info(f"  method: {self.extrema_smoothing['method']}")
        logger.info(f"  window_candles: {self.extrema_smoothing['window_candles']}")
        logger.info(f"  beta: {format_number(self.extrema_smoothing['beta'])}")
        logger.info(f"  polyorder: {self.extrema_smoothing['polyorder']}")
        logger.info(f"  mode: {self.extrema_smoothing['mode']}")
        logger.info(f"  sigma: {format_number(self.extrema_smoothing['sigma'])}")

        logger.info("Reversal Confirmation:")
        logger.info(
            f"  lookback_period_candles: {self.reversal_confirmation['lookback_period_candles']}"
        )
        logger.info(
            f"  decay_fraction: {format_number(self.reversal_confirmation['decay_fraction'])}"
        )
        logger.info(
            f"  min_natr_multiplier_fraction: {format_number(self.reversal_confirmation['min_natr_multiplier_fraction'])}"
        )
        logger.info(
            f"  max_natr_multiplier_fraction: {format_number(self.reversal_confirmation['max_natr_multiplier_fraction'])}"
        )

        logger.info("Exit Pricing:")
        logger.info(f"  trade_price_target_method: {self.trade_price_target_method}")
        logger.info(f"  thresholds_calibration: {self._exit_thresholds_calibration}")

        logger.info("Custom Stoploss:")
        logger.info(
            f"  natr_multiplier_fraction: {format_number(QuickAdapterV3._CUSTOM_STOPLOSS_NATR_MULTIPLIER_FRACTION)}"
        )

        logger.info("Partial Exit Stages:")
        for stage, (
            natr_multiplier_fraction,
            stake_percent,
            color,
        ) in QuickAdapterV3.partial_exit_stages.items():
            logger.info(
                f"  stage {stage}: natr_multiplier_fraction={format_number(natr_multiplier_fraction)}, stake_percent={format_number(stake_percent)}, color={color}"
            )

        final_stage = max(QuickAdapterV3.partial_exit_stages.keys(), default=-1) + 1
        logger.info(
            f"Final Exit Stage: stage {final_stage}: natr_multiplier_fraction={format_number(QuickAdapterV3._FINAL_EXIT_STAGE[0])}, stake_percent={format_number(QuickAdapterV3._FINAL_EXIT_STAGE[1])}, color={QuickAdapterV3._FINAL_EXIT_STAGE[2]}"
        )

        logger.info("Protections:")
        if self.protections:
            for protection in self.protections:
                method = protection.get("method", "Unknown")
                logger.info(f"  {method}:")
                for key, value in protection.items():
                    if key != "method":
                        if isinstance(value, bool):
                            logger.info(f"    {key}: {value}")
                        elif isinstance(value, (int, float)):
                            logger.info(f"    {key}: {format_number(value)}")
                        else:
                            logger.info(f"    {key}: {value}")
        else:
            logger.info("  No protections enabled")

        logger.info("=" * 60)

    @staticmethod
    def _df_signature(df: DataFrame) -> DfSignature:
        n = len(df)
        if n == 0:
            return (0, None)
        dates = df.get("date")
        return (n, dates.iloc[-1] if dates is not None and not dates.empty else None)

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
        dataframe["%-diff_to_psar"] = closes - ta.SAR(
            dataframe, acceleration=0.02, maximum=0.2
        )
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
            self._label_defaults[0],
        )

    def set_label_period_candles(self, pair: str, label_period_candles: int) -> None:
        if isinstance(label_period_candles, int):
            self._label_params[pair]["label_period_candles"] = label_period_candles

    def get_label_natr_multiplier(self, pair: str) -> float:
        label_natr_multiplier = self._label_params.get(pair, {}).get(
            "label_natr_multiplier"
        )
        if label_natr_multiplier and isinstance(label_natr_multiplier, float):
            return label_natr_multiplier
        feature_parameters = self.freqai_info.get("feature_parameters", {})
        return float(
            feature_parameters.get("label_natr_multiplier", self._label_defaults[1])
        )

    def set_label_natr_multiplier(
        self, pair: str, label_natr_multiplier: float
    ) -> None:
        if isinstance(label_natr_multiplier, float) and np.isfinite(
            label_natr_multiplier
        ):
            self._label_params[pair]["label_natr_multiplier"] = label_natr_multiplier

    def get_label_natr_multiplier_fraction(self, pair: str, fraction: float) -> float:
        if not isinstance(fraction, float) or not (0.0 <= fraction <= 1.0):
            raise ValueError(
                f"Invalid fraction {fraction!r}: must be a float in range [0, 1]"
            )
        return self.get_label_natr_multiplier(pair) * fraction

    @staticmethod
    @lru_cache(maxsize=128)
    def _td_format(
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
            raise ValueError(
                f"Invalid pattern {pattern!r}: failed to format with {e!r}"
            )

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        pair = str(metadata.get("pair"))
        label_period_candles = self.get_label_period_candles(pair)
        label_natr_multiplier = self.get_label_natr_multiplier(pair)
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
            natr_period=label_period_candles,
            natr_multiplier=label_natr_multiplier,
        )
        label_period = datetime.timedelta(
            minutes=len(dataframe) * self.get_timeframe_minutes()
        )
        dataframe[EXTREMA_COLUMN] = 0.0
        dataframe["minima"] = 0.0
        dataframe["maxima"] = 0.0

        if len(pivots_indices) == 0:
            logger.warning(
                f"[{pair}] No extrema to label | label_period: {QuickAdapterV3._td_format(label_period)} | label_period_candles: {label_period_candles} | label_natr_multiplier: {format_number(label_natr_multiplier)}"
            )
        else:
            logger.info(
                f"[{pair}] Labeled {len(pivots_indices)} extrema | label_period: {QuickAdapterV3._td_format(label_period)} | label_period_candles: {label_period_candles} | label_natr_multiplier: {format_number(label_natr_multiplier)}"
            )
            dataframe.loc[pivots_indices, EXTREMA_COLUMN] = pivots_directions

        extrema_direction = dataframe[EXTREMA_COLUMN]

        weighted_extrema, _ = get_weighted_extrema(
            extrema=extrema_direction,
            indices=pivots_indices,
            amplitudes=pivots_amplitudes,
            amplitude_threshold_ratios=pivots_amplitude_threshold_ratios,
            volume_rates=pivots_volume_rates,
            speeds=pivots_speeds,
            efficiency_ratios=pivots_efficiency_ratios,
            volume_weighted_efficiency_ratios=pivots_volume_weighted_efficiency_ratios,
            source_weights=self.extrema_weighting["source_weights"],
            strategy=self.extrema_weighting["strategy"],
            aggregation=self.extrema_weighting["aggregation"],
        )

        plot_eps = weighted_extrema.abs().where(weighted_extrema.ne(0.0)).min()
        if not np.isfinite(plot_eps):
            plot_eps = 0.0
        plot_eps = max(float(plot_eps) * 0.5, QuickAdapterV3._PLOT_EXTREMA_MIN_EPS)
        dataframe["maxima"] = (
            weighted_extrema.where(extrema_direction.gt(0), 0.0)
            .clip(lower=0.0)
            .mask(extrema_direction.gt(0) & weighted_extrema.eq(0.0), plot_eps)
        )
        dataframe["minima"] = (
            weighted_extrema.where(extrema_direction.lt(0), 0.0)
            .clip(upper=0.0)
            .mask(extrema_direction.lt(0) & weighted_extrema.eq(0.0), -plot_eps)
        )

        dataframe[EXTREMA_COLUMN] = smooth_extrema(
            weighted_extrema,
            self.extrema_smoothing["method"],
            self.extrema_smoothing["window_candles"],
            self.extrema_smoothing["beta"],
            self.extrema_smoothing["polyorder"],
            self.extrema_smoothing["mode"],
            self.extrema_smoothing["sigma"],
        )

        return dataframe

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        di_values = dataframe.get("DI_values")
        di_cutoff = dataframe.get("DI_cutoff")
        if di_values is not None and di_cutoff is not None:
            dataframe["DI_catch"] = np.where(di_values > di_cutoff, 0, 1)
        else:
            dataframe["DI_catch"] = 1

        pair = str(metadata.get("pair"))

        label_period_candles_series = dataframe.get("label_period_candles")
        if label_period_candles_series is not None:
            self.set_label_period_candles(pair, label_period_candles_series.iloc[-1])
        label_natr_multiplier_series = dataframe.get("label_natr_multiplier")
        if label_natr_multiplier_series is not None:
            self.set_label_natr_multiplier(pair, label_natr_multiplier_series.iloc[-1])

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
        ] = (1, QuickAdapterV3._TRADE_DIRECTIONS[0])  # "long"

        enter_short_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get("DI_catch") == 1,
            dataframe.get(EXTREMA_COLUMN) > dataframe.get("maxima_threshold"),
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_short_conditions),
            ["enter_short", "enter_tag"],
        ] = (1, QuickAdapterV3._TRADE_DIRECTIONS[1])  # "short"

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
        return int(
            ((current_date - entry_date).total_seconds() / 60.0)
            / self.get_timeframe_minutes()
        )

    def get_trade_annotation_line_start_date(
        self, dataframe: DataFrame, trade: Trade, offset_candles: Optional[int] = None
    ) -> datetime.datetime:
        if offset_candles is None:
            offset_candles = QuickAdapterV3._ANNOTATION_LINE_OFFSET_CANDLES

        trade_duration_candles = self.get_trade_duration_candles(dataframe, trade)

        offset_candles_remaining = max(
            0,
            offset_candles
            - (trade_duration_candles if trade_duration_candles is not None else 0),
        )

        offset_timedelta = datetime.timedelta(
            minutes=offset_candles_remaining * self.get_timeframe_minutes()
        )

        return trade.open_date_utc - offset_timedelta

    @staticmethod
    @lru_cache(maxsize=128)
    def is_trade_duration_valid(trade_duration: Optional[int | float]) -> bool:
        return isinstance(trade_duration, (int, float)) and not (
            isna(trade_duration) or trade_duration <= 0
        )

    def get_trade_weighted_average_natr(
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

        trade_label_natr_values = trade_label_natr.to_numpy()
        entry_quantile = calculate_quantile(trade_label_natr_values, entry_natr)
        current_quantile = calculate_quantile(trade_label_natr_values, current_natr)
        median_quantile = calculate_quantile(trade_label_natr_values, median_natr)

        if isna(entry_quantile) or isna(current_quantile) or isna(median_quantile):
            return None

        def calculate_weight(
            quantile: float,
            min_weight: float = 0.0,
            max_weight: float = 1.0,
            weighting_exponent: float = 1.5,
        ) -> float:
            return (
                min_weight
                + (max_weight - min_weight)
                * (abs(quantile - 0.5) * 2.0) ** weighting_exponent
            )

        entry_weight = calculate_weight(entry_quantile)
        current_weight = calculate_weight(current_quantile)
        median_weight = calculate_weight(median_quantile)

        total_weight = entry_weight + current_weight + median_weight
        if np.isclose(total_weight, 0.0):
            return np.nanmean([entry_natr, current_natr, median_natr])
        return nan_average(
            np.array([entry_natr, current_natr, median_natr]),
            weights=np.array([entry_weight, current_weight, median_weight]),
        )

    def get_trade_quantile_interpolation_natr(
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
                    zl_kama(label_natr, timeperiod=trade_duration_candles), dtype=float
                )
                trade_kama_natr_values = trade_kama_natr_values[
                    np.isfinite(trade_kama_natr_values)
                ]
                if trade_kama_natr_values.size > 0:
                    return trade_kama_natr_values[-1]
            except Exception as e:
                logger.warning(
                    f"[{pair}] Failed to calculate trade NATR KAMA: {e!r}, falling back to last trade NATR value",
                    exc_info=True,
                )
        return label_natr.iloc[-1]

    def get_trade_natr(
        self, df: DataFrame, trade: Trade, trade_duration_candles: int
    ) -> Optional[float]:
        trade_price_target_methods: dict[str, Callable[[], Optional[float]]] = {
            # 0 - "moving_average"
            TRADE_PRICE_TARGETS[0]: lambda: self.get_trade_moving_average_natr(
                df, trade.pair, trade_duration_candles
            ),
            # 1 - "quantile_interpolation"
            TRADE_PRICE_TARGETS[1]: lambda: self.get_trade_quantile_interpolation_natr(
                df, trade
            ),
            # 2 - "weighted_average"
            TRADE_PRICE_TARGETS[2]: lambda: self.get_trade_weighted_average_natr(
                df, trade
            ),
        }
        trade_price_target_method_fn = trade_price_target_methods.get(
            self.trade_price_target_method
        )
        if trade_price_target_method_fn is None:
            raise ValueError(
                f"Invalid trade_price_target_method {self.trade_price_target_method!r}. "
                f"Supported: {', '.join(TRADE_PRICE_TARGETS)}"
            )
        return trade_price_target_method_fn()

    @staticmethod
    def get_trade_exit_stage(trade: Trade) -> int:
        n_open_orders = 0
        if trade.has_open_orders:
            n_open_orders = sum(
                1
                for open_order in trade.open_orders
                if open_order.side == ("buy" if trade.is_short else "sell")
            )
        return trade.nr_of_successful_exits + n_open_orders

    @staticmethod
    @lru_cache(maxsize=128)
    def get_stoploss_factor(trade_duration_candles: int) -> float:
        return 2.75 / (1.2675 + math.atan(0.25 * trade_duration_candles))

    def get_stoploss_distance(
        self,
        df: DataFrame,
        trade: Trade,
        current_rate: float,
        natr_multiplier_fraction: float,
    ) -> Optional[float]:
        if not (0.0 <= natr_multiplier_fraction <= 1.0):
            raise ValueError(
                f"Invalid natr_multiplier_fraction {natr_multiplier_fraction!r}: must be in range [0, 1]"
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
            * self.get_label_natr_multiplier_fraction(
                trade.pair, natr_multiplier_fraction
            )
            * QuickAdapterV3.get_stoploss_factor(
                trade_duration_candles + int(round(trade.nr_of_successful_exits**1.5))
            )
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def get_take_profit_factor(trade_duration_candles: int) -> float:
        return math.log10(9.75 + 0.25 * trade_duration_candles)

    def get_take_profit_distance(
        self, df: DataFrame, trade: Trade, natr_multiplier_fraction: float
    ) -> Optional[float]:
        if not (0.0 <= natr_multiplier_fraction <= 1.0):
            raise ValueError(
                f"Invalid natr_multiplier_fraction {natr_multiplier_fraction!r}: must be in range [0, 1]"
            )
        trade_duration_candles = self.get_trade_duration_candles(df, trade)
        if not QuickAdapterV3.is_trade_duration_valid(trade_duration_candles):
            return None
        trade_natr = self.get_trade_natr(df, trade, trade_duration_candles)
        if isna(trade_natr) or trade_natr < 0:
            return None
        return (
            trade.open_rate
            * (trade_natr / 100.0)
            * self.get_label_natr_multiplier_fraction(
                trade.pair, natr_multiplier_fraction
            )
            * QuickAdapterV3.get_take_profit_factor(trade_duration_candles)
        )

    def throttle_callback(
        self,
        pair: str,
        current_time: datetime.datetime,
        callback: Callable[[], None],
    ) -> None:
        if not callable(callback):
            raise ValueError(f"Invalid callback {callback!r}: must be callable")
        timestamp = int(current_time.timestamp())
        candle_duration_secs = max(1, int(self._candle_duration_secs))
        candle_start_secs = (timestamp // candle_duration_secs) * candle_duration_secs
        key = hashlib.sha256(
            f"{pair}\x00{get_callable_sha256(callback)}".encode()
        ).hexdigest()
        if candle_start_secs != self.last_candle_start_secs.get(key):
            self.last_candle_start_secs[key] = candle_start_secs
            try:
                callback()
            except Exception as e:
                logger.error(
                    f"[{pair}] Callback execution failed: {e!r}", exc_info=True
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

        stoploss_distance = self.get_stoploss_distance(
            df,
            trade,
            current_rate,
            QuickAdapterV3._CUSTOM_STOPLOSS_NATR_MULTIPLIER_FRACTION,
        )
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
        natr_multiplier_fraction = (
            QuickAdapterV3.partial_exit_stages[exit_stage][0]
            if exit_stage in QuickAdapterV3.partial_exit_stages
            else QuickAdapterV3._FINAL_EXIT_STAGE[0]
        )
        take_profit_distance = self.get_take_profit_distance(
            df, trade, natr_multiplier_fraction
        )
        if isna(take_profit_distance) or take_profit_distance <= 0:
            return None

        take_profit_price = (
            trade.open_rate + (-1 if trade.is_short else 1) * take_profit_distance
        )

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
        if trade_exit_stage not in QuickAdapterV3.partial_exit_stages:
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

        self.safe_append_trade_take_profit_price(
            trade, trade_take_profit_price, trade_exit_stage
        )

        trade_partial_exit = QuickAdapterV3.can_take_profit(
            trade, current_rate, trade_take_profit_price
        )
        if not trade_partial_exit:
            self.throttle_callback(
                pair=trade.pair,
                current_time=current_time,
                callback=lambda: logger.info(
                    f"[{trade.pair}] Trade {trade.trade_direction} stage {trade_exit_stage} | "
                    f"Take Profit: {format_number(trade_take_profit_price)}, Rate: {format_number(current_rate)}"
                ),
            )
        if trade_partial_exit:
            if min_stake is None:
                min_stake = 0.0
            if min_stake > trade.stake_amount:
                return None
            trade_stake_percent = QuickAdapterV3.partial_exit_stages[trade_exit_stage][
                1
            ]
            trade_partial_stake_amount = trade_stake_percent * trade.stake_amount
            remaining_stake_amount = trade.stake_amount - trade_partial_stake_amount
            if remaining_stake_amount < min_stake:
                initial_trade_partial_stake_amount = trade_partial_stake_amount
                trade_partial_stake_amount = trade.stake_amount - min_stake
                logger.info(
                    f"[{trade.pair}] Trade {trade.trade_direction} stage {trade_exit_stage} | "
                    f"Partial stake amount adjusted from {format_number(initial_trade_partial_stake_amount)} to {format_number(trade_partial_stake_amount)} to respect min_stake {format_number(min_stake)}"
                )
            return (
                -trade_partial_stake_amount,
                f"take_profit_{trade.trade_direction}_{trade_exit_stage}",
            )

        return None

    @staticmethod
    def weighted_close(series: Series, weight: float = 2.0) -> float:
        return float(
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
        return min(max(0, idx), length - 1)

    def _calculate_candle_deviation(
        self,
        df: DataFrame,
        pair: str,
        min_natr_multiplier_fraction: float,
        max_natr_multiplier_fraction: float,
        candle_idx: int = -1,
        interpolation_direction: InterpolationDirection = "direct",
        quantile_exponent: float = 1.5,
    ) -> float:
        df_signature = QuickAdapterV3._df_signature(df)
        prev_df_signature = self._cached_df_signature.get(pair)
        if prev_df_signature != df_signature:
            self._candle_deviation_cache = {
                k: v for k, v in self._candle_deviation_cache.items() if k[0] != pair
            }
            self._cached_df_signature[pair] = df_signature
        cache_key: CandleDeviationCacheKey = (
            pair,
            df_signature,
            float(min_natr_multiplier_fraction),
            float(max_natr_multiplier_fraction),
            candle_idx,
            interpolation_direction,
            float(quantile_exponent),
        )
        if cache_key in self._candle_deviation_cache:
            return self._candle_deviation_cache[cache_key]
        label_natr_series = df.get("natr_label_period_candles")
        if label_natr_series is None or label_natr_series.empty:
            return np.nan

        candle_idx = QuickAdapterV3._normalize_candle_idx(
            len(label_natr_series), candle_idx
        )

        label_natr_values = label_natr_series.iloc[: candle_idx + 1].to_numpy()
        if label_natr_values.size == 0:
            return np.nan
        candle_label_natr_value = label_natr_values[-1]
        if isna(candle_label_natr_value) or candle_label_natr_value < 0:
            return np.nan
        label_period_candles = self.get_label_period_candles(pair)
        candle_label_natr_value_quantile = calculate_quantile(
            label_natr_values[-label_period_candles:], candle_label_natr_value
        )
        if isna(candle_label_natr_value_quantile):
            return np.nan

        if (
            interpolation_direction == QuickAdapterV3._INTERPOLATION_DIRECTIONS[0]
        ):  # "direct"
            natr_multiplier_fraction = (
                min_natr_multiplier_fraction
                + (max_natr_multiplier_fraction - min_natr_multiplier_fraction)
                * candle_label_natr_value_quantile**quantile_exponent
            )
        elif (
            interpolation_direction == QuickAdapterV3._INTERPOLATION_DIRECTIONS[1]
        ):  # "inverse"
            natr_multiplier_fraction = (
                max_natr_multiplier_fraction
                - (max_natr_multiplier_fraction - min_natr_multiplier_fraction)
                * candle_label_natr_value_quantile**quantile_exponent
            )
        else:
            raise ValueError(
                f"Invalid interpolation_direction {interpolation_direction!r}. "
                f"Supported: {', '.join(QuickAdapterV3._INTERPOLATION_DIRECTIONS)}"
            )
        candle_deviation = (
            candle_label_natr_value / 100.0
        ) * self.get_label_natr_multiplier_fraction(pair, natr_multiplier_fraction)
        self._candle_deviation_cache[cache_key] = candle_deviation
        return self._candle_deviation_cache[cache_key]

    def _calculate_candle_threshold(
        self,
        df: DataFrame,
        pair: str,
        side: TradeDirection,
        min_natr_multiplier_fraction: float,
        max_natr_multiplier_fraction: float,
        candle_idx: int = -1,
    ) -> float:
        df_signature = QuickAdapterV3._df_signature(df)
        prev_df_signature = self._cached_df_signature.get(pair)
        if prev_df_signature != df_signature:
            self._candle_threshold_cache = {
                k: v for k, v in self._candle_threshold_cache.items() if k[0] != pair
            }
            self._cached_df_signature[pair] = df_signature
        cache_key: CandleThresholdCacheKey = (
            pair,
            df_signature,
            side,
            candle_idx,
            float(min_natr_multiplier_fraction),
            float(max_natr_multiplier_fraction),
        )
        if cache_key in self._candle_threshold_cache:
            return self._candle_threshold_cache[cache_key]
        current_deviation = self._calculate_candle_deviation(
            df,
            pair,
            min_natr_multiplier_fraction=min_natr_multiplier_fraction,
            max_natr_multiplier_fraction=max_natr_multiplier_fraction,
            candle_idx=candle_idx,
            interpolation_direction=QuickAdapterV3._INTERPOLATION_DIRECTIONS[
                0
            ],  # "direct"
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

        if side == QuickAdapterV3._TRADE_DIRECTIONS[0]:  # "long"
            base_price = (
                QuickAdapterV3.weighted_close(candle)
                if is_candle_bearish
                else candle_close
            )
            candle_threshold = base_price * (1 + current_deviation)
        elif side == QuickAdapterV3._TRADE_DIRECTIONS[1]:  # "short"
            base_price = (
                QuickAdapterV3.weighted_close(candle)
                if is_candle_bullish
                else candle_close
            )
            candle_threshold = base_price * (1 - current_deviation)
        else:
            raise ValueError(
                f"Invalid side {side!r}, supported: {', '.join(QuickAdapterV3._TRADE_DIRECTIONS)}"
            )
        self._candle_threshold_cache[cache_key] = candle_threshold
        return self._candle_threshold_cache[cache_key]

    def reversal_confirmed(
        self,
        df: DataFrame,
        pair: str,
        side: TradeDirection,
        order: OrderType,
        rate: float,
        lookback_period_candles: int,
        decay_fraction: float,
        min_natr_multiplier_fraction: float,
        max_natr_multiplier_fraction: float,
    ) -> bool:
        """Confirm a directional reversal using a volatility-adaptive current-candle
        threshold and optionally a backward confirmation chain with geometric decay.

        Overview
        --------
        1. Compute a deviation-based threshold on the latest candle (-1). The current
           rate must strictly break it (long: rate > threshold; short: rate < threshold).
        2. If lookback_period_candles > 0, for each k = 1..lookback_period_candles:
             - Decay (min_natr_multiplier_fraction, max_natr_multiplier_fraction) by
               (decay_fraction ** k), clamped to [0, 1].
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
        lookback_period_candles : int
            Number of historical confirmation steps requested; truncated to history.
        decay_fraction : float
            Geometric decay factor per step (0 < decay_fraction <= 1); 1.0 disables decay.
        min_natr_multiplier_fraction : float
            Lower-bound fraction (e.g. 0.009 = 0.9%).
        max_natr_multiplier_fraction : float
            Upper-bound fraction (>= lower bound).

        Returns
        -------
        bool
            True iff the current threshold is broken AND (lookback chain succeeded OR
            a permissive fallback occurred). False otherwise.

        Fallback Semantics
        ------------------
        Missing / non-finite intermediate data -> stop chain; return current candle result.
        This may yield True on partial history, weakening strict multi-candle guarantees.

        Rejection Conditions
        --------------------
        Empty dataframe, invalid side/order, non-finite rate, negative lookback,
        decay_fraction outside (0,1], invalid min/max ordering, failure to break current
        threshold, or failed historical step comparison.

        Complexity
        ----------
        O(lookback_period_candles) threshold computations.

        Logging
        -------
        Logs rejection reasons (invalid decay_fraction, threshold not broken, failed step).
        Fallback aborts are silent.

        Limitations
        -----------
        No strict mode; partial data may still confirm.
        """
        if df.empty:
            return False
        if side not in QuickAdapterV3._trade_directions_set():
            return False
        if order not in QuickAdapterV3._order_types_set():
            return False
        if not isinstance(rate, (int, float)) or not np.isfinite(rate):
            return False
        if (
            not isinstance(min_natr_multiplier_fraction, (int, float))
            or not isinstance(max_natr_multiplier_fraction, (int, float))
            or not np.isfinite(min_natr_multiplier_fraction)
            or not np.isfinite(max_natr_multiplier_fraction)
            or min_natr_multiplier_fraction < 0
            or max_natr_multiplier_fraction < 0
            or min_natr_multiplier_fraction > max_natr_multiplier_fraction
        ):
            return False

        trade_direction = side

        max_lookback_period_candles = max(0, len(df) - 1)
        lookback_period_candles = min(
            lookback_period_candles, max_lookback_period_candles
        )
        if not isinstance(decay_fraction, (int, float)):
            logger.debug(
                f"[{pair}] Denied {trade_direction} {order}: invalid decay_fraction type"
            )
            return False
        if not (0.0 < decay_fraction <= 1.0):
            logger.debug(
                f"[{pair}] Denied {trade_direction} {order}: invalid decay_fraction {decay_fraction}, must be in (0, 1]"
            )
            return False

        current_threshold = self._calculate_candle_threshold(
            df,
            pair,
            side,
            min_natr_multiplier_fraction=min_natr_multiplier_fraction,
            max_natr_multiplier_fraction=max_natr_multiplier_fraction,
            candle_idx=-1,
        )
        current_ok = np.isfinite(current_threshold) and (
            (
                side == QuickAdapterV3._TRADE_DIRECTIONS[0] and rate > current_threshold
            )  # "long"
            or (
                side == QuickAdapterV3._TRADE_DIRECTIONS[1] and rate < current_threshold
            )  # "short"
        )
        if order == QuickAdapterV3._ORDER_TYPES[1]:  # "exit"
            if side == QuickAdapterV3._TRADE_DIRECTIONS[0]:  # "long"
                trade_direction = QuickAdapterV3._TRADE_DIRECTIONS[1]  # "short"
            if side == QuickAdapterV3._TRADE_DIRECTIONS[1]:  # "short"
                trade_direction = QuickAdapterV3._TRADE_DIRECTIONS[0]  # "long"
        if not current_ok:
            logger.debug(
                f"[{pair}] Denied {trade_direction} {order}: rate {format_number(rate)} did not break threshold {format_number(current_threshold)}"
            )
            return False

        if lookback_period_candles == 0:
            return current_ok

        for k in range(1, lookback_period_candles + 1):
            close_k = df.iloc[-k].get("close")
            if not isinstance(close_k, (int, float)) or not np.isfinite(close_k):
                return current_ok

            decay_factor = decay_fraction**k
            decayed_min_natr_multiplier_fraction = max(
                0.0, min(1.0, min_natr_multiplier_fraction * decay_factor)
            )
            decayed_max_natr_multiplier_fraction = max(
                decayed_min_natr_multiplier_fraction,
                min(1.0, max_natr_multiplier_fraction * decay_factor),
            )

            threshold_k = self._calculate_candle_threshold(
                df,
                pair,
                side,
                min_natr_multiplier_fraction=decayed_min_natr_multiplier_fraction,
                max_natr_multiplier_fraction=decayed_max_natr_multiplier_fraction,
                candle_idx=-(k + 1),
            )
            if not isinstance(threshold_k, (int, float)) or not np.isfinite(
                threshold_k
            ):
                return current_ok

            if (
                side == QuickAdapterV3._TRADE_DIRECTIONS[0]
                and not (close_k > threshold_k)  # "long"
            ) or (
                side == QuickAdapterV3._TRADE_DIRECTIONS[1]
                and not (close_k < threshold_k)  # "short"
            ):
                logger.debug(
                    f"[{pair}] Denied {trade_direction} {order}: "
                    f"close_k[{-k}] {format_number(close_k)} "
                    f"did not break threshold_k[{-(k + 1)}] {format_number(threshold_k)} "
                    f"(decayed natr_multiplier_fraction: min={format_number(decayed_min_natr_multiplier_fraction)}, max={format_number(decayed_max_natr_multiplier_fraction)})"
                )
                return False

        return True

    @staticmethod
    def get_pnl_momentum(
        unrealized_pnl_history: Sequence[float], window_size: int
    ) -> tuple[
        tuple[float, ...],
        float,
        float,
        tuple[float, ...],
        float,
        float,
    ]:
        """Compute velocity and acceleration from PnL history.

        Velocity is the first derivative of PnL, acceleration is the second.

        Args:
            unrealized_pnl_history: PnL values sequence.
            window_size: Recent window size (0 = no windowing).

        Returns:
            (velocity_values, velocity_mean, velocity_std,
             acceleration_values, acceleration_mean, acceleration_std)
        """
        unrealized_pnl_history_array = np.asarray(unrealized_pnl_history, dtype=float)

        if window_size > 0 and len(unrealized_pnl_history_array) > window_size:
            unrealized_pnl_history_array = unrealized_pnl_history_array[-window_size:]

        velocity = np.diff(unrealized_pnl_history_array)
        velocity_mean = np.nanmean(velocity) if velocity.size > 0 else 0.0
        velocity_std = np.nanstd(velocity, ddof=1) if velocity.size > 1 else 0.0

        acceleration = np.diff(velocity)
        acceleration_mean = np.nanmean(acceleration) if acceleration.size > 0 else 0.0
        acceleration_std = (
            np.nanstd(acceleration, ddof=1) if acceleration.size > 1 else 0.0
        )

        return (
            tuple(velocity.tolist()),
            velocity_mean,
            velocity_std,
            tuple(acceleration.tolist()),
            acceleration_mean,
            acceleration_std,
        )

    @staticmethod
    @lru_cache(maxsize=128)
    def _t_statistic(mean: float, std: float, n: int) -> float:
        """Compute t-statistic for H₀: μ = 0.

        Formula: t = mean * √n / std

        Args:
            mean: Sample mean.
            std: Sample standard deviation (ddof=1).
            n: Sample size.

        Returns:
            t-statistic, or NaN if n < 2 or std ≈ 0.
        """
        if n < 2:
            return np.nan
        if not np.isfinite(mean) or not np.isfinite(std):
            return np.nan
        if np.isclose(std, 0.0):
            return np.nan
        return mean * math.sqrt(n) / std

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

    @staticmethod
    @lru_cache(maxsize=128)
    def _effective_df(x: tuple[float, ...]) -> float:
        """Compute effective degrees of freedom with Bartlett's autocorrelation correction.

        Formula: df_eff = (n - 1) * (1 - ρ₁) / (1 + ρ₁), where ρ₁ is lag-1 autocorrelation.

        Args:
            x: Observations tuple.

        Returns:
            Effective df (≥ 1). Falls back to n - 1 if n < 4 or on error.
        """
        n = len(x)
        if n < 4:
            return max(1.0, n - 1)

        x_arr = np.asarray(x, dtype=float)
        x_centered = x_arr - np.nanmean(x_arr)

        try:
            rho1, _ = pearsonr(x_centered[:-1], x_centered[1:])
        except Exception:
            return n - 1

        if not np.isfinite(rho1):
            return n - 1

        # Clamp to avoid division by zero or negative n_eff
        rho1 = np.clip(rho1, -0.99, 0.99)
        correction_factor = (1 - rho1) / (1 + rho1)

        n_eff = n * correction_factor
        df_eff = max(1.0, n_eff - 1)

        return df_eff

    @staticmethod
    @lru_cache(maxsize=128)
    def _t_critical(q: float, df: float, default_t: float) -> float:
        """Compute critical t-value from Student's t-distribution.

        Args:
            q: Quantile in (0, 1), e.g. 0.75.
            df: Degrees of freedom (can be fractional).
            default_t: Fallback value on error.

        Returns:
            t.ppf(q, df), or default_t if invalid inputs.
        """
        if not (0.0 < q < 1.0):
            return default_t
        if df < 1:
            return default_t
        try:
            t_crit = float(t.ppf(q, df))
            if not np.isfinite(t_crit):
                return default_t
            return t_crit
        except Exception:
            return default_t

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
                    f"[{pair}] Detected new predictions outlier ({n_outliers=}) on trade {trade.id}"
                )
                trade.set_custom_data("n_outliers", n_outliers)
                trade.set_custom_data("last_outlier_date", last_candle_date.isoformat())

        if (
            trade.trade_direction == QuickAdapterV3._TRADE_DIRECTIONS[1]  # "short"
            and last_candle.get("do_predict") == 1
            and last_candle.get("DI_catch") == 1
            and last_candle.get(EXTREMA_COLUMN) < last_candle.get("minima_threshold")
            and self.reversal_confirmed(
                df,
                pair,
                QuickAdapterV3._TRADE_DIRECTIONS[0],  # "long"
                QuickAdapterV3._ORDER_TYPES[1],  # "exit"
                current_rate,
                self.reversal_confirmation["lookback_period_candles"],
                self.reversal_confirmation["decay_fraction"],
                self.reversal_confirmation["min_natr_multiplier_fraction"],
                self.reversal_confirmation["max_natr_multiplier_fraction"],
            )
        ):
            return "minima_detected_short"
        if (
            trade.trade_direction == QuickAdapterV3._TRADE_DIRECTIONS[0]  # "long"
            and last_candle.get("do_predict") == 1
            and last_candle.get("DI_catch") == 1
            and last_candle.get(EXTREMA_COLUMN) > last_candle.get("maxima_threshold")
            and self.reversal_confirmed(
                df,
                pair,
                QuickAdapterV3._TRADE_DIRECTIONS[1],  # "short"
                QuickAdapterV3._ORDER_TYPES[1],  # "exit"
                current_rate,
                self.reversal_confirmation["lookback_period_candles"],
                self.reversal_confirmation["decay_fraction"],
                self.reversal_confirmation["min_natr_multiplier_fraction"],
                self.reversal_confirmation["max_natr_multiplier_fraction"],
            )
        ):
            return "maxima_detected_long"

        trade_exit_stage = QuickAdapterV3.get_trade_exit_stage(trade)
        if trade_exit_stage in QuickAdapterV3.partial_exit_stages:
            return None

        trade_take_profit_price = self.get_take_profit_price(
            df, trade, trade_exit_stage
        )
        if isna(trade_take_profit_price):
            return None

        self.safe_append_trade_take_profit_price(
            trade, trade_take_profit_price, trade_exit_stage
        )

        trade_take_profit_exit = QuickAdapterV3.can_take_profit(
            trade, current_rate, trade_take_profit_price
        )

        if not trade_take_profit_exit:
            self.throttle_callback(
                pair=pair,
                current_time=current_time,
                callback=lambda: logger.info(
                    f"[{pair}] Trade {trade.trade_direction} stage {trade_exit_stage} | "
                    f"Take Profit: {format_number(trade_take_profit_price)}, Rate: {format_number(current_rate)}"
                ),
            )
            return None

        trade_unrealized_pnl_history = QuickAdapterV3.get_trade_unrealized_pnl_history(
            trade
        )
        (
            trade_recent_velocity_values,
            trade_recent_velocity_mean,
            trade_recent_velocity_std,
            trade_recent_acceleration_values,
            trade_recent_acceleration_mean,
            trade_recent_acceleration_std,
        ) = QuickAdapterV3.get_pnl_momentum(
            trade_unrealized_pnl_history, self._pnl_momentum_window_size
        )

        q_decl = self._exit_thresholds_calibration.get("decline_quantile")

        n_trade_recent_velocity = len(trade_recent_velocity_values)
        n_trade_recent_acceleration = len(trade_recent_acceleration_values)

        t_trade_recent_velocity = QuickAdapterV3._t_statistic(
            trade_recent_velocity_mean,
            trade_recent_velocity_std,
            n_trade_recent_velocity,
        )
        t_trade_recent_acceleration = QuickAdapterV3._t_statistic(
            trade_recent_acceleration_mean,
            trade_recent_acceleration_std,
            n_trade_recent_acceleration,
        )

        df_eff_trade_recent_velocity = QuickAdapterV3._effective_df(
            trade_recent_velocity_values
        )
        df_eff_trade_recent_acceleration = QuickAdapterV3._effective_df(
            trade_recent_acceleration_values
        )

        t_crit_trade_recent_velocity = QuickAdapterV3._t_critical(
            q_decl,
            df_eff_trade_recent_velocity,
            QuickAdapterV3.default_exit_thresholds["t_decl_v"],
        )
        t_crit_trade_recent_acceleration = QuickAdapterV3._t_critical(
            q_decl,
            df_eff_trade_recent_acceleration,
            QuickAdapterV3.default_exit_thresholds["t_decl_a"],
        )

        # Declining if t_stat ≤ -t_crit (one-sided test for μ < 0)
        decl_checks: list[bool] = []
        if np.isfinite(t_trade_recent_velocity):
            decl_checks.append(t_trade_recent_velocity <= -t_crit_trade_recent_velocity)
        if np.isfinite(t_trade_recent_acceleration):
            decl_checks.append(
                t_trade_recent_acceleration <= -t_crit_trade_recent_acceleration
            )
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
                    f"[{pair}] Trade {trade.trade_direction} stage {trade_exit_stage} | "
                    f"Take Profit: {format_number(trade_take_profit_price)}, Rate: {format_number(current_rate)} | "
                    f"Declining: {trade_recent_pnl_declining} "
                    f"(tV:{format_number(t_trade_recent_velocity)}<=-t:{format_number(-t_crit_trade_recent_velocity)}, tA:{format_number(t_trade_recent_acceleration)}<=-t:{format_number(-t_crit_trade_recent_acceleration)})"
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
        if side not in QuickAdapterV3._trade_directions_set():
            return False
        if (
            side == QuickAdapterV3._TRADE_DIRECTIONS[1] and not self.can_short
        ):  # "short"
            logger.info(
                f"[{pair}] Denied short {QuickAdapterV3._ORDER_TYPES[0]}: shorting not allowed"
            )
            return False
        if Trade.get_open_trade_count() >= self.config.get("max_open_trades", 0):
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
            logger.info(
                f"[{pair}] Denied {side} {QuickAdapterV3._ORDER_TYPES[0]}: dataframe is empty"
            )
            return False
        if self.reversal_confirmed(
            df,
            pair,
            side,
            QuickAdapterV3._ORDER_TYPES[0],  # "entry"
            rate,
            self.reversal_confirmation["lookback_period_candles"],
            self.reversal_confirmation["decay_fraction"],
            self.reversal_confirmation["min_natr_multiplier_fraction"],
            self.reversal_confirmation["max_natr_multiplier_fraction"],
        ):
            return True
        return False

    def is_short_allowed(self) -> bool:
        trading_mode = self.config.get("trading_mode")
        if trading_mode in {
            QuickAdapterV3._TRADING_MODES[1],
            QuickAdapterV3._TRADING_MODES[2],
        }:  # margin, futures
            return True
        elif trading_mode == QuickAdapterV3._TRADING_MODES[0]:  # "spot"
            return False
        else:
            raise ValueError(
                f"Invalid trading_mode {trading_mode!r}. "
                f"Supported: {', '.join(QuickAdapterV3._TRADING_MODES)}"
            )

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

    def plot_annotations(
        self,
        pair: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        dataframe: DataFrame,
        **kwargs: Any,
    ) -> list[AnnotationType]:
        """
        Plot annotations.

        :param pair: Pair that's currently being plotted
        :param start_date: Start date of the chart range
        :param end_date: End date of the chart range
        :param dataframe: DataFrame with analyzed data for this pair
        :param **kwargs: Additional arguments
        :return: List of annotations to display on the chart
        """
        annotations: list[AnnotationType] = []

        open_trades = Trade.get_trades_proxy(pair=pair, is_open=True)

        for trade in open_trades:
            if trade.open_date_utc > end_date:
                continue

            trade_annotation_line_start_date = (
                self.get_trade_annotation_line_start_date(dataframe, trade)
            )

            trade_exit_stage = self.get_trade_exit_stage(trade)

            for take_profit_stage, (_, _, color) in self.partial_exit_stages.items():
                if take_profit_stage < trade_exit_stage:
                    continue

                partial_take_profit_price = self.get_take_profit_price(
                    dataframe, trade, take_profit_stage
                )

                if isna(partial_take_profit_price):
                    continue

                take_profit_line_annotation: AnnotationType = {
                    "type": "line",
                    "start": max(trade_annotation_line_start_date, start_date),
                    "end": end_date,
                    "y_start": partial_take_profit_price,
                    "y_end": partial_take_profit_price,
                    "color": color,
                    "line_style": "solid",
                    "width": 1,
                    "label": f"Take Profit {take_profit_stage}",
                    "z_level": 10 + take_profit_stage,
                }
                annotations.append(take_profit_line_annotation)

            final_stage = max(self.partial_exit_stages.keys(), default=-1) + 1
            final_take_profit_price = self.get_take_profit_price(
                dataframe, trade, final_stage
            )

            if not isna(final_take_profit_price):
                take_profit_line_annotation: AnnotationType = {
                    "type": "line",
                    "start": max(trade_annotation_line_start_date, start_date),
                    "end": end_date,
                    "y_start": final_take_profit_price,
                    "y_end": final_take_profit_price,
                    "color": QuickAdapterV3._FINAL_EXIT_STAGE[2],
                    "line_style": "solid",
                    "width": 1,
                    "label": f"Take Profit {final_stage}",
                    "z_level": 10 + final_stage,
                }
                annotations.append(take_profit_line_annotation)

        return annotations

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
