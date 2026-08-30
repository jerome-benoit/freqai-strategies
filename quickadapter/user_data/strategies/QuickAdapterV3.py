import datetime
import hashlib
import logging
import math
from collections.abc import Callable
from functools import cached_property, lru_cache, reduce
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Final,
    Literal,
    TypedDict,
)

import numpy as np
import pandas_ta as pta
import talib.abstract as ta
from freqtrade.enums import TRADE_MODES
from freqtrade.exchange import (
    timeframe_to_minutes,
    timeframe_to_prev_date,
)
from freqtrade.persistence import Trade
from freqtrade.strategy import AnnotationType, stoploss_from_absolute
from freqtrade.strategy.interface import IStrategy
from LabelTransformer import (
    COMBINED_AGGREGATIONS,
    FILL_METHODS,
    SMOOTHING_METHOD_MODES,
    SMOOTHING_METHODS,
    SMOOTHING_MODES,
    WEIGHT_STRATEGIES,
    get_label_column_config,
)
from pandas import DataFrame, Series, isna, to_numeric
from technical.pivots_points import pivots_points
from Utils import (
    _CACHE_MAXSIZE_LARGE,
    _OPTUNA_NAMESPACES,
    EXTREMA_COLUMN,
    EXTREMA_DIRECTION_COLUMN,
    EXTREMA_DIRECTION_SMOOTHED_COLUMN,
    EXTREMA_WEIGHT_COLUMN,
    EXTREMA_WEIGHT_SMOOTHED_COLUMN,
    LABEL_COLUMNS,
    TRADE_NATR_METHODS,
    OptunaNamespace,
    alligator,
    bottom_log_return,
    calculate_quantile,
    compose_label_lookahead,
    compute_label_weight_imputation_dependency_mask,
    compute_label_weight_known_at_lookahead,
    compute_label_weights,
    ensure_datetime_series,
    enum_error_message,
    ewo,
    format_dict,
    format_number,
    generate_label_data,
    get_callable_sha256,
    get_causal_mode,
    get_custom_protections_config,
    get_distance,
    get_exit_pricing_config,
    get_fit_live_predictions_candles,
    get_label_defaults,
    get_label_horizon_candles,
    get_label_smoothing_config,
    get_label_weighting_config,
    get_reversal_confirmation_config,
    get_smoothing_kernel_half_width,
    get_zl_ma_fn,
    is_finite_number,
    label_known_at_lookahead_column_name,
    label_weight_column_name,
    label_weight_known_at_lookahead_column_name,
    migrate_config,
    nan_average,
    non_zero_diff,
    optuna_load_best_params,
    price_retracement_percent,
    safe_divide,
    smooth,
    top_log_return,
    vwapb,
    weight_fill_radius,
    zlema,
)

TradeDirection = Literal["long", "short"]
InterpolationDirection = Literal["direct", "inverse"]
OrderType = Literal["entry", "exit"]
TradingMode = Literal["spot", "margin", "futures"]

DfSignature = tuple[int, datetime.datetime | None]
CandleDeviationCacheKey = tuple[str, DfSignature, float, float, int, InterpolationDirection, float]
CandleThresholdCacheKey = tuple[str, DfSignature, str, int, float, float]


class _FinalTakeProfitState(TypedDict):
    version: int
    exit_stage: int
    trade_direction: TradeDirection
    best_rate: float
    retracement_distance: float
    boundary_candle_date: str
    last_candle_date: str
    trigger_candle_date: str | None
    timeframe: str


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
    _TRADE_LONG: Final[str] = _TRADE_DIRECTIONS[0]
    _TRADE_SHORT: Final[str] = _TRADE_DIRECTIONS[1]
    _TRADE_DIRECTIONS_SET: Final[frozenset[TradeDirection]] = frozenset(_TRADE_DIRECTIONS)
    _INTERPOLATION_DIRECTIONS: Final[tuple[InterpolationDirection, ...]] = (
        "direct",
        "inverse",
    )
    _INTERPOLATION_DIRECT: Final[str] = _INTERPOLATION_DIRECTIONS[0]
    _INTERPOLATION_INVERSE: Final[str] = _INTERPOLATION_DIRECTIONS[1]
    _ORDER_TYPES: Final[tuple[OrderType, ...]] = ("entry", "exit")
    _ORDER_ENTRY: Final[str] = _ORDER_TYPES[0]
    _ORDER_EXIT: Final[str] = _ORDER_TYPES[1]
    _ORDER_TYPES_SET: Final[frozenset[OrderType]] = frozenset(_ORDER_TYPES)
    _TRADING_MODES: Final[tuple[TradingMode, ...]] = ("spot", "margin", "futures")
    _TRADING_MODE_SPOT: Final[str] = _TRADING_MODES[0]
    _TRADING_MODE_MARGIN: Final[str] = _TRADING_MODES[1]
    _TRADING_MODE_FUTURES: Final[str] = _TRADING_MODES[2]
    _SMOOTHING_SMM: Final[str] = SMOOTHING_METHODS[5]
    _SMOOTHING_SAVGOL: Final[str] = SMOOTHING_METHODS[7]
    _FILL_EPSILON: Final[str] = FILL_METHODS[1]
    _FILL_GAUSSIAN: Final[str] = FILL_METHODS[2]
    _FILL_EPSILON_GAUSSIAN: Final[str] = FILL_METHODS[3]
    _WEIGHT_NONE: Final[str] = WEIGHT_STRATEGIES[0]

    _CUSTOM_STOPLOSS_NATR_MULTIPLIER_FRACTION: Final[float] = 0.7860

    _ANNOTATION_LINE_OFFSET_CANDLES: Final[int] = 10

    def version(self) -> str:
        return "3.13.0-rc.7"

    timeframe = "5m"
    timeframe_minutes = timeframe_to_minutes(timeframe)

    stoploss = -0.025
    use_custom_stoploss = True

    position_adjustment_enable = True

    # {stage: (natr_multiplier_fraction, stake_percent, color)}
    partial_exit_stages: ClassVar[dict[int, tuple[float, float, str]]] = {
        0: (0.4858, 0.4, "lime"),
        1: (0.6180, 0.3, "yellow"),
        2: (0.7640, 0.2, "coral"),
    }

    # (natr_multiplier_fraction, stake_percent, color)
    _FINAL_EXIT_STAGE_PARAMS: Final[tuple[float, float, str]] = (
        1.0,
        1.0,
        "deepskyblue",
    )

    # Final full-exit stage, derived from the configured partial exits.
    _FINAL_EXIT_STAGE: Final[int] = max(partial_exit_stages.keys(), default=-1) + 1

    _TAKE_PROFIT_ORDER_TAG_PREFIX: Final[str] = "take_profit_"
    _FINAL_TAKE_PROFIT_STATE_KEY: Final[str] = "final_take_profit_state"
    _FINAL_TAKE_PROFIT_STATE_VERSION: Final[int] = 3
    _FINAL_TAKE_PROFIT_SUPPORTED_STATE_VERSIONS: Final[range] = range(
        1, _FINAL_TAKE_PROFIT_STATE_VERSION + 1
    )

    # Rounding margin so the sized partial-exit remainder clears freqtrade's
    # strict ``remaining < min_exit_stake`` guard.
    _PARTIAL_EXIT_MIN_STAKE_MARGIN: Final[float] = 1e-3

    # FreqAI is crashing if ``minimal_roi`` is a property
    minimal_roi: ClassVar[dict[str, int]] = {str(timeframe_minutes * 864): -1}

    process_only_new_candles = True

    def __init__(self, config: dict[str, Any], *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        migrate_config(self.config, logger)

    @cached_property
    def timeframe_minutes(self) -> int:
        return timeframe_to_minutes(self.config.get("timeframe"))

    @cached_property
    def is_trade_runmode(self) -> bool:
        # True in live and dry-run (``runmode`` in ``TRADE_MODES``), mirroring the
        # regressor's ``self.live`` gate.
        return self.config.get("runmode") in TRADE_MODES

    @property
    def can_short(self) -> bool:
        return self.is_short_allowed()

    @cached_property
    def plot_config(self) -> dict[str, Any]:
        return {
            "main_plot": {},
            "subplots": {
                "accuracy": {
                    "holdout_rmse": {"color": "violet", "type": "line"},
                },
                "extrema": {
                    f"{EXTREMA_COLUMN}_maxima_threshold": {
                        "color": "blue",
                        "type": "line",
                    },
                    f"{EXTREMA_COLUMN}_minima_threshold": {
                        "color": "cyan",
                        "type": "line",
                    },
                    EXTREMA_COLUMN: {"color": "orange", "type": "line"},
                },
                "direction": {
                    EXTREMA_DIRECTION_COLUMN: {"color": "steelblue", "type": "bar"},
                    EXTREMA_DIRECTION_SMOOTHED_COLUMN: {
                        "color": "orange",
                        "type": "line",
                    },
                },
                "weight": {
                    EXTREMA_WEIGHT_COLUMN: {"color": "steelblue", "type": "bar"},
                    EXTREMA_WEIGHT_SMOOTHED_COLUMN: {
                        "color": "orange",
                        "type": "line",
                    },
                },
            },
        }

    @cached_property
    def _fit_live_predictions_candles(self) -> int:
        return get_fit_live_predictions_candles(self.config.get("freqai"), logger)

    @staticmethod
    def _is_unlimited_max_open_trades(max_open_trades: float) -> bool:
        return max_open_trades == -1 or max_open_trades == math.inf

    @cached_property
    def protections(self) -> list[dict[str, Any]]:
        fit_live_predictions_candles = self._fit_live_predictions_candles
        protections = get_custom_protections_config(self.config.get("custom_protections"), logger)
        trade_duration_candles = protections["trade_duration_candles"]
        lookback_period_fraction = protections["lookback_period_fraction"]

        lookback_period_candles = max(
            1, round(fit_live_predictions_candles * lookback_period_fraction)
        )

        cooldown = protections["cooldown"]
        cooldown_stop_duration_candles = cooldown["stop_duration_candles"]
        stoploss_stop_duration_candles = max(cooldown_stop_duration_candles, trade_duration_candles)
        drawdown_stop_duration_candles = max(
            stoploss_stop_duration_candles,
            fit_live_predictions_candles,
        )
        max_open_trades = self.config.get("max_open_trades", 0)
        unlimited_max_open_trades = QuickAdapterV3._is_unlimited_max_open_trades(max_open_trades)
        estimated_trade_limit = max(
            2,
            round(lookback_period_candles / max(1, trade_duration_candles)),
        )
        if unlimited_max_open_trades:
            stoploss_trade_limit = estimated_trade_limit
            drawdown_trade_limit = 2 * estimated_trade_limit
        else:
            max_open_trades = int(max_open_trades)
            stoploss_trade_limit = min(
                estimated_trade_limit,
                max(2, round(max_open_trades * 0.75)),
            )
            drawdown_trade_limit = 2 * max_open_trades

        protections_list = []

        if cooldown["enabled"]:
            protections_list.append(
                {
                    "method": "CooldownPeriod",
                    "stop_duration_candles": cooldown_stop_duration_candles,
                }
            )

        drawdown = protections["drawdown"]
        if drawdown["enabled"]:
            protections_list.append(
                {
                    "method": "MaxDrawdown",
                    "lookback_period_candles": lookback_period_candles,
                    "trade_limit": drawdown_trade_limit,
                    "stop_duration_candles": drawdown_stop_duration_candles,
                    "max_allowed_drawdown": drawdown["max_allowed_drawdown"],
                }
            )

        stoploss = protections["stoploss"]
        if stoploss["enabled"]:
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
        return self._fit_live_predictions_candles

    @property
    def max_open_trades_per_side(self) -> int:
        max_open_trades = self.config.get("max_open_trades", 0)
        if QuickAdapterV3._is_unlimited_max_open_trades(max_open_trades):
            return -1
        if self.is_short_allowed():
            if max_open_trades % 2 == 1:
                max_open_trades += 1
            return int(max_open_trades / 2)
        else:
            return max_open_trades

    @cached_property
    def label_weighting(self) -> dict[str, Any]:
        return get_label_weighting_config(self.freqai_info.get("label_weighting"), logger)

    @cached_property
    def label_smoothing(self) -> dict[str, Any]:
        return get_label_smoothing_config(self.freqai_info.get("label_smoothing"), logger)

    @cached_property
    def exit_pricing(self) -> dict[str, str | float]:
        return get_exit_pricing_config(self.config.get("exit_pricing"), logger)

    @property
    def trade_natr_method(self) -> str:
        return str(self.exit_pricing["trade_natr_method"])

    @property
    def final_take_profit_retracement_fraction(self) -> float:
        return float(self.exit_pricing["final_take_profit_retracement_fraction"])

    @cached_property
    def reversal_confirmation(self) -> dict[str, int | float]:
        return get_reversal_confirmation_config(self.config.get("reversal_confirmation"), logger)

    @cached_property
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
            self.config.get("user_data_dir") / "models" / self.freqai_info.get("identifier")
        )
        feature_parameters = self.freqai_info.get("feature_parameters", {})
        if get_causal_mode(feature_parameters, logger):
            label_smoothing = self.label_smoothing
            for label_col in LABEL_COLUMNS:
                col_smoothing_config = get_label_column_config(
                    label_col, label_smoothing["default"], label_smoothing["columns"]
                )
                if (
                    col_smoothing_config["method"] in SMOOTHING_METHOD_MODES
                    and col_smoothing_config["mode"] == SMOOTHING_MODES[3]
                ):  # "wrap"
                    raise ValueError(
                        "label_smoothing.mode='wrap' is incompatible with "
                        "feature_parameters.causal_mode=true"
                    )
        default_label_period_candles, default_label_natr_multiplier = self._label_defaults
        self._label_params: dict[str, dict[str, Any]] = {}
        load_persisted_label_params = self.is_trade_runmode
        for pair in self.pairs:
            label_best_params = (
                self.optuna_load_best_params(pair, _OPTUNA_NAMESPACES.label)
                if load_persisted_label_params
                else None
            )
            self._label_params[pair] = (
                label_best_params
                if label_best_params
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
        self._candle_duration_secs = int(self.timeframe_minutes * 60)
        self.last_candle_start_secs: dict[str, int | None] = {}
        self._max_take_profit_history_size = max(1, int(12 * 60 / self.timeframe_minutes))
        self._candle_deviation_cache: dict[CandleDeviationCacheKey, float] = {}
        self._candle_threshold_cache: dict[CandleThresholdCacheKey, float] = {}
        self._cached_df_signature: dict[str, DfSignature] = {}

        self._log_strategy_configuration()

    def _log_strategy_configuration(self) -> None:
        logger.info("=" * 60)
        logger.info("QuickAdapter Strategy Configuration")
        logger.info("=" * 60)

        label_weighting = self.label_weighting
        label_smoothing = self.label_smoothing
        for label_col in LABEL_COLUMNS:
            logger.info(f"Label [{label_col}]:")

            col_weighting = get_label_column_config(
                label_col, label_weighting["default"], label_weighting["columns"]
            )
            logger.info("  Weighting:")
            logger.info(f"    strategy: {col_weighting['strategy']}")
            logger.info(
                f"    metric_coefficients: {format_dict(col_weighting['metric_coefficients'], style='dict')}"
            )
            logger.info(f"    aggregation: {col_weighting['aggregation']}")
            if col_weighting["aggregation"] == COMBINED_AGGREGATIONS[5]:  # "softmax"
                logger.info(
                    f"    softmax_temperature: {format_number(col_weighting['softmax_temperature'])}"
                )
            fill_method = col_weighting["fill_method"]
            logger.info(f"    fill_method: {fill_method}")
            if fill_method in (
                QuickAdapterV3._FILL_EPSILON,
                QuickAdapterV3._FILL_EPSILON_GAUSSIAN,
            ):
                logger.info(f"    fill_epsilon: {format_number(col_weighting['fill_epsilon'])}")
                logger.info(f"    fill_epsilon_baseline: {col_weighting['fill_epsilon_baseline']}")
            if fill_method in (
                QuickAdapterV3._FILL_GAUSSIAN,
                QuickAdapterV3._FILL_EPSILON_GAUSSIAN,
            ):
                logger.info(
                    f"    fill_sigma_candles: {format_number(col_weighting['fill_sigma_candles'])}"
                )
                logger.info(
                    f"    fill_sigma_min_candles: {format_number(col_weighting['fill_sigma_min_candles'])}"
                )
                logger.info(f"    fill_bandwidth: {col_weighting['fill_bandwidth']}")
                logger.info(
                    f"    fill_bandwidth_neighbors: {col_weighting['fill_bandwidth_neighbors']}"
                )
                logger.info(
                    f"    fill_bandwidth_alpha: {format_number(col_weighting['fill_bandwidth_alpha'])}"
                )
            logger.info(f"    support_policy: {col_weighting['support_policy']}")
            logger.info(
                f"    min_pivot_equivalent_count: {col_weighting['min_pivot_equivalent_count']}"
            )
            logger.info(
                f"    min_positive_label_weight_fraction: {format_number(col_weighting['min_positive_label_weight_fraction'])}"
            )
            logger.info(
                f"    min_effective_sample_size: {format_number(col_weighting['min_effective_sample_size'])}"
            )

            col_smoothing = get_label_column_config(
                label_col, label_smoothing["default"], label_smoothing["columns"]
            )
            logger.info("  Smoothing:")
            logger.info(f"    method: {col_smoothing['method']}")
            logger.info(f"    window_candles: {col_smoothing['window_candles']}")
            logger.info(f"    beta: {format_number(col_smoothing['beta'])}")
            logger.info(f"    polyorder: {col_smoothing['polyorder']}")
            logger.info(f"    mode: {col_smoothing['mode']}")
            logger.info(f"    sigma: {format_number(col_smoothing['sigma'])}")

            method = col_smoothing["method"]
            if col_weighting["strategy"] != QuickAdapterV3._WEIGHT_NONE and (
                method == QuickAdapterV3._SMOOTHING_SMM
                or (method == QuickAdapterV3._SMOOTHING_SAVGOL and col_smoothing["polyorder"] >= 2)
            ):
                logger.warning(
                    f"  Label [{label_col}]: smoothing method {method!r} can "
                    f"collapse sparse weight signals (smm zeroes them when "
                    f"fewer than half the window rows are nonzero; savgol "
                    f"with polyorder>=2 adds negative lobes that are clipped "
                    f"to zero), which may trip the all-rows-dropped guard in "
                    f"compose_sample_weights once a non-'none' "
                    f"label_weighting strategy is configured. Prefer a "
                    f"non-negative linear kernel (gaussian, kaiser, "
                    f"kaiser_bessel_derived, triang, sma, gaussian_filter1d)."
                )

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
        logger.info(f"  trade_natr_method: {self.trade_natr_method}")
        logger.info(
            "  final_take_profit_retracement_fraction: "
            f"{format_number(self.final_take_profit_retracement_fraction)}"
        )

        logger.info("Custom Stoploss:")
        logger.info(
            f"  natr_multiplier_fraction: {format_number(QuickAdapterV3._CUSTOM_STOPLOSS_NATR_MULTIPLIER_FRACTION)}"
        )

        logger.info("Partial Take-Profit Stages:")
        for stage, (
            natr_multiplier_fraction,
            stake_percent,
            color,
        ) in QuickAdapterV3.partial_exit_stages.items():
            logger.info(
                f"  stage {stage}: natr_multiplier_fraction={format_number(natr_multiplier_fraction)}, stake_percent={format_number(stake_percent)}, color={color}"
            )

        logger.info(
            f"Final Exit: natr_multiplier_fraction={format_number(QuickAdapterV3._FINAL_EXIT_STAGE_PARAMS[0])}, stake_percent={format_number(QuickAdapterV3._FINAL_EXIT_STAGE_PARAMS[1])}, color={QuickAdapterV3._FINAL_EXIT_STAGE_PARAMS[2]}"
        )

        logger.info("Protections:")
        if self.protections:
            for protection in self.protections:
                method = protection.get("method", "Unknown")
                protection_params = {k: v for k, v in protection.items() if k != "method"}
                logger.info(f"  {method}: {format_dict(protection_params, style='dict')}")
        else:
            logger.info("  No protections enabled")

        logger.info("=" * 60)

    @staticmethod
    def _df_signature(df: DataFrame) -> DfSignature:
        """Candle-cache key ``(row_count, last_date)``; assumes existing rows
        stay immutable (holds under ``process_only_new_candles = True``).
        """
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
        dataframe["%-top_log_return-period"] = top_log_return(
            dataframe, period=period, logger=logger
        )
        dataframe["%-bottom_log_return-period"] = bottom_log_return(
            dataframe, period=period, logger=logger
        )
        dataframe["%-prp-period"] = price_retracement_percent(
            dataframe, period=period, logger=logger
        )
        dataframe["%-cti-period"] = pta.cti(closes, length=period)
        dataframe["%-chop-period"] = pta.chop(
            highs,
            lows,
            closes,
            length=period,
        )
        dataframe["%-linearreg_angle-period"] = ta.LINEARREG_ANGLE(dataframe, timeperiod=period)
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

        close_values = closes.to_numpy(dtype=float)
        invalid_close_count = int(
            np.count_nonzero(~np.isfinite(close_values) | (close_values <= 0.0))
        )
        if invalid_close_count:
            logger.debug(
                "feature_engineering_expand_basic: %d close values are non-finite or non-positive; close log return is NaN at those positions",
                invalid_close_count,
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            dataframe["%-close_log_return"] = Series(
                np.where(
                    np.isfinite(close_values) & (close_values > 0.0),
                    np.log(close_values),
                    np.nan,
                ),
                index=dataframe.index,
            ).diff()
        dataframe["%-raw_volume"] = volumes
        dataframe["%-obv"] = ta.OBV(dataframe)
        label_period_candles = self.get_label_period_candles(str(metadata.get("pair")))
        dataframe["%-atr_label_period_candles"] = ta.ATR(dataframe, timeperiod=label_period_candles)
        dataframe["%-natr_label_period_candles"] = ta.NATR(
            dataframe, timeperiod=label_period_candles
        )
        dataframe["%-ewo"] = ewo(
            dataframe=dataframe,
            pricemode="close",
            mamode="ema",
            zero_lag=True,
            normalize=True,
            logger=logger,
        )
        dataframe["%-diff_to_psar"] = closes - ta.SAR(dataframe, acceleration=0.02, maximum=0.2)
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
        dataframe["%-kc_width"] = safe_divide(
            dataframe["kc_upperband"] - dataframe["kc_lowerband"],
            dataframe["kc_middleband"],
            context="feature_engineering_expand_basic:kc_width",
            logger=logger,
        )
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
        dataframe["%-bb_width"] = safe_divide(
            dataframe["bb_upperband"] - dataframe["bb_lowerband"],
            dataframe["bb_middleband"],
            context="feature_engineering_expand_basic:bb_width",
            logger=logger,
        )
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
        dataframe["%-dist_to_zlema_50"] = get_distance(closes, dataframe["zlema_50"])
        dataframe["%-dist_to_zlema_12"] = get_distance(closes, dataframe["zlema_12"])
        dataframe["%-dist_to_zlema_26"] = get_distance(closes, dataframe["zlema_26"])
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
        dataframe["%-vwap_width"] = safe_divide(
            dataframe["vwap_upperband"] - dataframe["vwap_lowerband"],
            dataframe["vwap_middleband"],
            context="feature_engineering_expand_basic:vwap_width",
            logger=logger,
        )
        dataframe["%-dist_to_vwap_upperband"] = get_distance(closes, dataframe["vwap_upperband"])
        dataframe["%-dist_to_vwap_middleband"] = get_distance(closes, dataframe["vwap_middleband"])
        dataframe["%-dist_to_vwap_lowerband"] = get_distance(closes, dataframe["vwap_lowerband"])
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
        dates = ensure_datetime_series(dataframe.get("date"))

        dataframe["%-day_of_week"] = (dates.dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dates.dt.hour + 1) / 25
        return dataframe

    def get_label_period_candles(
        self,
        pair: str,
        dataframe: DataFrame | None = None,
        candle_idx: int = -1,
    ) -> int:
        if dataframe is not None:
            period_series = dataframe.get("label_period_candles")
            if period_series is not None and not period_series.empty:
                period = period_series.iloc[candle_idx]
                if is_finite_number(period) and int(period) > 0:
                    return int(period)
        period = self._label_params.get(pair, {}).get("label_period_candles")
        if is_finite_number(period) and int(period) > 0:
            return int(period)
        return int(
            self.freqai_info.get("feature_parameters", {}).get(
                "label_period_candles",
                self._label_defaults[0],
            )
        )

    def set_label_period_candles(self, pair: str, label_period_candles: Any) -> None:
        if is_finite_number(label_period_candles) and int(label_period_candles) > 0:
            label_period_candles = int(label_period_candles)
            if self._label_params[pair].get("label_period_candles") != label_period_candles:
                self._label_params[pair]["label_period_candles"] = label_period_candles
                self._invalidate_pair_caches(pair)

    def get_label_horizon_candles(self, pair: str) -> int:
        period = self.get_label_period_candles(pair)
        label_params = self._label_params.get(pair, {})
        feature_parameters = self.freqai_info.get("feature_parameters", {})
        return get_label_horizon_candles(
            {**feature_parameters, **label_params, "label_period_candles": period},
            logger,
        )

    def get_label_natr_multiplier(
        self,
        pair: str,
        dataframe: DataFrame | None = None,
        candle_idx: int = -1,
    ) -> float:
        if dataframe is not None:
            multiplier_series = dataframe.get("label_natr_multiplier")
            if multiplier_series is not None and not multiplier_series.empty:
                multiplier = multiplier_series.iloc[candle_idx]
                if is_finite_number(multiplier) and float(multiplier) > 0.0:
                    return float(multiplier)
        multiplier = self._label_params.get(pair, {}).get("label_natr_multiplier")
        if is_finite_number(multiplier) and float(multiplier) > 0.0:
            return float(multiplier)
        return float(
            self.freqai_info.get("feature_parameters", {}).get(
                "label_natr_multiplier", self._label_defaults[1]
            )
        )

    def set_label_natr_multiplier(self, pair: str, label_natr_multiplier: Any) -> None:
        if is_finite_number(label_natr_multiplier) and float(label_natr_multiplier) > 0.0:
            label_natr_multiplier = float(label_natr_multiplier)
            if self._label_params[pair].get("label_natr_multiplier") != label_natr_multiplier:
                self._label_params[pair]["label_natr_multiplier"] = label_natr_multiplier
                self._invalidate_pair_caches(pair)

    def get_label_natr_multiplier_fraction(
        self,
        pair: str,
        fraction: float,
        dataframe: DataFrame | None = None,
        candle_idx: int = -1,
    ) -> float:
        if not isinstance(fraction, float) or not (0.0 <= fraction <= 1.0):
            raise ValueError(
                f"Invalid fraction value {fraction!r}: must be a float in range [0, 1]"
            )
        return self.get_label_natr_multiplier(pair, dataframe, candle_idx) * fraction

    def get_label_params(self, pair: str, label_col: str) -> dict[str, Any]:
        if label_col == EXTREMA_COLUMN:
            return {
                "natr_period": self.get_label_period_candles(pair),
                "natr_multiplier": self.get_label_natr_multiplier(pair),
                "label_horizon_candles": self.get_label_horizon_candles(pair),
            }
        return {}

    @staticmethod
    @lru_cache(maxsize=_CACHE_MAXSIZE_LARGE)
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
                f"Invalid pattern value {pattern!r}: failed to format with {e!r}"
            ) from e

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        pair = str(metadata.get("pair"))
        series_duration = datetime.timedelta(minutes=len(dataframe) * self.timeframe_minutes)

        label_weighting = self.label_weighting
        label_smoothing = self.label_smoothing
        series_length = len(dataframe)
        causal_mode = get_causal_mode(self.freqai_info.get("feature_parameters", {}), logger)
        finite_gaussian_support = causal_mode

        for label_col in LABEL_COLUMNS:
            label_params = self.get_label_params(pair, label_col)
            label_data = generate_label_data(dataframe, label_col, label_params, logger)

            if len(label_data.indices) == 0:
                logger.warning(
                    f"[{pair}] No {label_col!r} labels | series_duration: {QuickAdapterV3._td_format(series_duration)} | params: {format_dict(label_params, style='params')}"
                )
            else:
                logger.info(
                    f"[{pair}] {len(label_data.indices)} {label_col!r} labels | series_duration: {QuickAdapterV3._td_format(series_duration)} | params: {format_dict(label_params, style='params')}"
                )

            col_weighting_config = get_label_column_config(
                label_col, label_weighting["default"], label_weighting["columns"]
            )

            # Absent column routes downstream to base-weights-only fallback.
            is_weighting_active = (
                col_weighting_config["strategy"] != QuickAdapterV3._WEIGHT_NONE
                and len(label_data.indices) > 0
            )

            dataframe[label_col] = label_data.series

            if label_data.known_at_lookahead is not None:
                dataframe[label_known_at_lookahead_column_name(label_col)] = (
                    label_data.known_at_lookahead
                )

            label_weight_col = label_weight_column_name(label_col)
            if is_weighting_active:
                dataframe[label_weight_col] = compute_label_weights(
                    n_values=len(label_data.series),
                    indices=label_data.indices,
                    metrics=label_data.metrics,
                    weighting_config=col_weighting_config,
                    finite_gaussian_support=finite_gaussian_support,
                    logger=logger,
                    known_at_lookahead=(label_data.known_at_lookahead if causal_mode else None),
                )
                if label_data.known_at_lookahead is not None:
                    if causal_mode:
                        imputation_masks = compute_label_weight_imputation_dependency_mask(
                            len(label_data.indices),
                            label_data.metrics,
                            col_weighting_config,
                        )
                        imputation_dependency_mask = imputation_masks.dependency_mask
                        imputation_leading_stable_mask = imputation_masks.leading_stable_mask
                        imputation_stable_release_index = imputation_masks.stable_release_index
                    else:
                        imputation_dependency_mask = None
                        imputation_leading_stable_mask = None
                        imputation_stable_release_index = -1
                    dataframe[label_weight_known_at_lookahead_column_name(label_col)] = (
                        compute_label_weight_known_at_lookahead(
                            known_at_lookahead=label_data.known_at_lookahead,
                            indices=label_data.indices,
                            fill_radius=weight_fill_radius(col_weighting_config),
                            weighting_config=col_weighting_config,
                            imputation_dependency_mask=imputation_dependency_mask,
                            imputation_leading_stable_mask=imputation_leading_stable_mask,
                            imputation_stable_release_index=imputation_stable_release_index,
                        )
                    )

            if label_col == EXTREMA_COLUMN:
                dataframe[EXTREMA_DIRECTION_COLUMN] = dataframe[label_col]
                if is_weighting_active:
                    dataframe[EXTREMA_WEIGHT_COLUMN] = dataframe[label_weight_col]

            col_smoothing_config = get_label_column_config(
                label_col, label_smoothing["default"], label_smoothing["columns"]
            )

            dataframe[label_col] = smooth(dataframe[label_col], **col_smoothing_config)
            if is_weighting_active:
                smoothed_label_weights = smooth(dataframe[label_weight_col], **col_smoothing_config)
                dataframe[label_weight_col] = smoothed_label_weights.where(
                    np.isfinite(smoothed_label_weights) & smoothed_label_weights.gt(0),
                    0.0,
                )

            # Zero-phase smoothing reads future candles within the kernel
            # half-width; extend the per-row lookahead so causal split guards
            # account for the smoothing lookahead.
            kernel_half_width = get_smoothing_kernel_half_width(
                col_smoothing_config, series_length=series_length
            )
            for lookahead_column in (
                label_known_at_lookahead_column_name(label_col),
                label_weight_known_at_lookahead_column_name(label_col),
            ):
                if lookahead_column in dataframe.columns:
                    dataframe[lookahead_column] = compose_label_lookahead(
                        dataframe[lookahead_column], kernel_half_width
                    )

            if label_col == EXTREMA_COLUMN:
                dataframe[EXTREMA_DIRECTION_SMOOTHED_COLUMN] = dataframe[label_col]
                if is_weighting_active:
                    dataframe[EXTREMA_WEIGHT_SMOOTHED_COLUMN] = dataframe[label_weight_col]

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        di_values = dataframe.get("DI_values")
        di_cutoff = dataframe.get("DI_cutoff")
        if di_values is not None and di_cutoff is not None:
            dataframe["DI_catch"] = np.where(di_values > di_cutoff, 0, 1)
        else:
            dataframe["DI_catch"] = 1

        pair = str(metadata.get("pair"))

        label_period_candles_series = dataframe.get("label_period_candles")
        label_natr_multiplier_series = dataframe.get("label_natr_multiplier")
        if self.is_trade_runmode:
            if label_period_candles_series is not None:
                self.set_label_period_candles(pair, label_period_candles_series.iloc[-1])
            if label_natr_multiplier_series is not None:
                self.set_label_natr_multiplier(pair, label_natr_multiplier_series.iloc[-1])

        if label_period_candles_series is None:
            dataframe["natr_label_period_candles"] = ta.NATR(
                dataframe, timeperiod=self.get_label_period_candles(pair)
            )
        else:
            # Per-candle HPO ``label_period_candles``: NATR is computed once per
            # distinct period, then scattered back to its matching rows (mixing
            # per-row periods within one column is intentional).
            dataframe["natr_label_period_candles"] = np.nan
            fallback_period = self.get_label_period_candles(pair)
            numeric_periods = to_numeric(label_period_candles_series, errors="coerce")
            valid_periods = np.isfinite(numeric_periods) & (numeric_periods >= 1)
            periods = numeric_periods.where(valid_periods, fallback_period).astype(int)
            for period in periods.unique():
                period_rows = periods == period
                period_natr = ta.NATR(dataframe, timeperiod=int(period))
                dataframe.loc[period_rows, "natr_label_period_candles"] = period_natr.loc[
                    period_rows
                ]

        dataframe["minima_threshold"] = dataframe.get(f"{EXTREMA_COLUMN}_minima_threshold", np.nan)
        dataframe["maxima_threshold"] = dataframe.get(f"{EXTREMA_COLUMN}_maxima_threshold", np.nan)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        enter_long_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get("DI_catch") == 1,
            dataframe.get(EXTREMA_COLUMN) < dataframe.get("minima_threshold"),
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_long_conditions),
            ["enter_long", "enter_tag"],
        ] = (1, QuickAdapterV3._TRADE_LONG)

        enter_short_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get("DI_catch") == 1,
            dataframe.get(EXTREMA_COLUMN) > dataframe.get("maxima_threshold"),
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_short_conditions),
            ["enter_short", "enter_tag"],
        ] = (1, QuickAdapterV3._TRADE_SHORT)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        return dataframe

    def get_trade_entry_date(self, trade: Trade) -> datetime.datetime:
        return timeframe_to_prev_date(self.config.get("timeframe"), trade.open_date_utc)

    def get_trade_duration_candles(self, df: DataFrame, trade: Trade) -> int | None:
        entry_date = self.get_trade_entry_date(trade)
        dates = df.get("date")
        if dates is None or dates.empty:
            return None
        current_date = dates.iloc[-1]
        if isna(current_date):
            return None
        return int(((current_date - entry_date).total_seconds() / 60.0) / self.timeframe_minutes)

    def get_trade_annotation_line_start_date(
        self, dataframe: DataFrame, trade: Trade, offset_candles: int | None = None
    ) -> datetime.datetime:
        if offset_candles is None:
            offset_candles = QuickAdapterV3._ANNOTATION_LINE_OFFSET_CANDLES

        trade_duration_candles = self.get_trade_duration_candles(dataframe, trade)

        offset_candles_remaining = max(
            0,
            offset_candles - (trade_duration_candles if trade_duration_candles is not None else 0),
        )

        offset_timedelta = datetime.timedelta(
            minutes=offset_candles_remaining * self.timeframe_minutes
        )

        return trade.open_date_utc - offset_timedelta

    @staticmethod
    @lru_cache(maxsize=_CACHE_MAXSIZE_LARGE)
    def is_trade_duration_valid(trade_duration: float | None) -> bool:
        return isinstance(trade_duration, (int, float)) and not (
            isna(trade_duration) or trade_duration <= 0
        )

    def _trade_natr_window(
        self, df: DataFrame, trade: Trade
    ) -> tuple[Any, float, float | None] | None:
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
            current_natr = None
        else:
            current_natr = trade_label_natr.iloc[-1]
            if isna(current_natr) or current_natr < 0:
                return None
        return trade_label_natr, entry_natr, current_natr

    def get_trade_weighted_average_natr(self, df: DataFrame, trade: Trade) -> float | None:
        window = self._trade_natr_window(df, trade)
        if window is None:
            return None
        trade_label_natr, entry_natr, current_natr = window
        if current_natr is None:
            return entry_natr
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
                + (max_weight - min_weight) * (abs(quantile - 0.5) * 2.0) ** weighting_exponent
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
            logger=logger,
        )

    def get_trade_quantile_interpolation_natr(self, df: DataFrame, trade: Trade) -> float | None:
        window = self._trade_natr_window(df, trade)
        if window is None:
            return None
        trade_label_natr, entry_natr, current_natr = window
        if current_natr is None:
            return entry_natr
        trade_volatility_quantile = calculate_quantile(trade_label_natr.to_numpy(), entry_natr)
        if isna(trade_volatility_quantile):
            trade_volatility_quantile = 0.5
        return np.interp(
            trade_volatility_quantile,
            [0.0, 1.0],
            [current_natr, entry_natr],
        )

    def get_trade_moving_average_natr(
        self, df: DataFrame, pair: str, trade_duration_candles: int
    ) -> float | None:
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
                trade_kama_natr_values = trade_kama_natr_values[np.isfinite(trade_kama_natr_values)]
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
    ) -> float | None:
        trade_natr_methods: dict[str, Callable[[], float | None]] = {
            # 0 - "moving_average"
            TRADE_NATR_METHODS[0]: lambda: self.get_trade_moving_average_natr(
                df, trade.pair, trade_duration_candles
            ),
            # 1 - "quantile_interpolation"
            TRADE_NATR_METHODS[1]: lambda: self.get_trade_quantile_interpolation_natr(df, trade),
            # 2 - "weighted_average"
            TRADE_NATR_METHODS[2]: lambda: self.get_trade_weighted_average_natr(df, trade),
        }
        trade_natr_method_fn = trade_natr_methods.get(self.trade_natr_method)
        if trade_natr_method_fn is None:
            raise ValueError(
                enum_error_message(
                    "trade_natr_method",
                    self.trade_natr_method,
                    TRADE_NATR_METHODS,
                )
            )
        return trade_natr_method_fn()

    @staticmethod
    def get_trade_exit_stage(trade: Trade) -> int:
        n_filled_take_profit_exits = sum(
            1
            for order in trade.select_filled_orders(trade.exit_side)
            if (order.ft_order_tag or "").startswith(QuickAdapterV3._TAKE_PROFIT_ORDER_TAG_PREFIX)
        )
        return min(n_filled_take_profit_exits, QuickAdapterV3._FINAL_EXIT_STAGE)

    @staticmethod
    @lru_cache(maxsize=_CACHE_MAXSIZE_LARGE)
    def get_stoploss_factor(trade_duration_candles: int) -> float:
        return 2.75 / (1.2675 + math.atan(0.25 * trade_duration_candles))

    def get_stoploss_distance(
        self,
        df: DataFrame,
        trade: Trade,
        current_rate: float,
        natr_multiplier_fraction: float,
    ) -> float | None:
        if not (0.0 <= natr_multiplier_fraction <= 1.0):
            raise ValueError(
                f"Invalid natr_multiplier_fraction value {natr_multiplier_fraction!r}: must be in range [0, 1]"
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
            * self.get_label_natr_multiplier_fraction(trade.pair, natr_multiplier_fraction, df)
            * QuickAdapterV3.get_stoploss_factor(
                trade_duration_candles + round(trade.nr_of_successful_exits**1.5)
            )
        )

    @staticmethod
    @lru_cache(maxsize=_CACHE_MAXSIZE_LARGE)
    def get_take_profit_factor(trade_duration_candles: int) -> float:
        return math.log10(9.75 + 0.25 * trade_duration_candles)

    def get_take_profit_distance(
        self, df: DataFrame, trade: Trade, natr_multiplier_fraction: float
    ) -> float | None:
        if not (0.0 <= natr_multiplier_fraction <= 1.0):
            raise ValueError(
                f"Invalid natr_multiplier_fraction value {natr_multiplier_fraction!r}: must be in range [0, 1]"
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
            * self.get_label_natr_multiplier_fraction(trade.pair, natr_multiplier_fraction, df)
            * QuickAdapterV3.get_take_profit_factor(trade_duration_candles)
        )

    def throttle_callback(
        self,
        pair: str,
        current_time: datetime.datetime,
        callback: Callable[[], None],
    ) -> None:
        if not callable(callback):
            raise ValueError(f"Invalid callback value {callback!r}: must be callable")
        timestamp = int(current_time.timestamp())
        candle_duration_secs = max(1, int(self._candle_duration_secs))
        candle_start_secs = (timestamp // candle_duration_secs) * candle_duration_secs
        key = hashlib.sha256(f"{pair}\x00{get_callable_sha256(callback)}".encode()).hexdigest()
        if candle_start_secs != self.last_candle_start_secs.get(key):
            self.last_candle_start_secs[key] = candle_start_secs
            try:
                callback()
            except Exception:
                logger.exception(f"[{pair}] Callback execution failed")

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
    ) -> float | None:
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.config.get("timeframe"))
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
    def can_take_profit(trade: Trade, current_rate: float, take_profit_price: float) -> bool:
        return (trade.is_short and current_rate <= take_profit_price) or (
            not trade.is_short and current_rate >= take_profit_price
        )

    def get_take_profit_target(
        self, df: DataFrame, trade: Trade, exit_stage: int
    ) -> tuple[float, float] | None:
        natr_multiplier_fraction = (
            QuickAdapterV3.partial_exit_stages[exit_stage][0]
            if exit_stage in QuickAdapterV3.partial_exit_stages
            else QuickAdapterV3._FINAL_EXIT_STAGE_PARAMS[0]
        )
        take_profit_distance = self.get_take_profit_distance(df, trade, natr_multiplier_fraction)
        if not is_finite_number(take_profit_distance) or take_profit_distance <= 0:
            return None

        take_profit_price = trade.open_rate + (
            -take_profit_distance if trade.is_short else take_profit_distance
        )
        if take_profit_price == trade.open_rate:
            take_profit_price = math.nextafter(trade.open_rate, 0.0 if trade.is_short else math.inf)
        if not np.isfinite(take_profit_price) or take_profit_price <= 0:
            return None
        return float(take_profit_price), float(take_profit_distance)

    def safe_append_trade_take_profit_price(
        self, trade: Trade, take_profit_price: float, exit_stage: int
    ) -> None:
        history = trade.get_custom_data("history", {})
        if not isinstance(history, dict):
            history = {}
        price_history = history.get("take_profit_price", [])
        if not isinstance(price_history, list):
            price_history = []
        history = {"take_profit_price": price_history}
        previous_take_profit_entry = price_history[-1] if price_history else None
        previous_exit_stage = None
        previous_take_profit_price = None
        if (
            isinstance(previous_take_profit_entry, (tuple, list))
            and len(previous_take_profit_entry) == 2
        ):
            candidate_exit_stage, candidate_take_profit_price = previous_take_profit_entry
            if isinstance(candidate_take_profit_price, bool):
                candidate_take_profit_price = None
            else:
                try:
                    candidate_take_profit_price = float(candidate_take_profit_price)
                except (OverflowError, TypeError, ValueError):
                    candidate_take_profit_price = None
            if (
                isinstance(candidate_exit_stage, int)
                and not isinstance(candidate_exit_stage, bool)
                and candidate_take_profit_price is not None
                and np.isfinite(candidate_take_profit_price)
                and candidate_take_profit_price > 0
            ):
                previous_exit_stage = candidate_exit_stage
                previous_take_profit_price = candidate_take_profit_price
        elif isinstance(previous_take_profit_entry, float) and np.isfinite(
            previous_take_profit_entry
        ):
            previous_exit_stage = -1
            previous_take_profit_price = previous_take_profit_entry
        if (
            previous_take_profit_price is not None
            and (previous_exit_stage is None or previous_exit_stage == exit_stage)
            and np.isclose(previous_take_profit_price, take_profit_price)
        ):
            return

        price_history.append((exit_stage, take_profit_price))
        if len(price_history) > self._max_take_profit_history_size:
            history["take_profit_price"] = price_history[-self._max_take_profit_history_size :]
        trade.set_custom_data("history", history)

    @staticmethod
    def _as_utc_candle_date(value: Any) -> datetime.datetime | None:
        if isinstance(value, str):
            try:
                value = datetime.datetime.fromisoformat(value)
            except (TypeError, ValueError):
                return None
        if not isinstance(value, datetime.datetime):
            return None
        try:
            if value.tzinfo is None or value.utcoffset() is None:
                return None
            return value.astimezone(datetime.UTC)
        except (OverflowError, ValueError):
            return None

    @staticmethod
    def _is_candle_date_aligned(candle_date: datetime.datetime | None, timeframe: str) -> bool:
        normalized_candle_date = QuickAdapterV3._as_utc_candle_date(candle_date)
        if normalized_candle_date is None or not isinstance(timeframe, str) or not timeframe:
            return False
        try:
            return (
                timeframe_to_prev_date(timeframe, normalized_candle_date) == normalized_candle_date
            )
        except (OverflowError, TypeError, ValueError):
            return False

    @staticmethod
    def _normalize_final_take_profit_retracement_distance(
        *,
        best_rate: float,
        retracement_distance: float,
        trade_direction: TradeDirection,
    ) -> float | None:
        if (
            not np.isfinite(best_rate)
            or best_rate <= 0
            or not np.isfinite(retracement_distance)
            or retracement_distance < 0
            or trade_direction not in QuickAdapterV3._TRADE_DIRECTIONS_SET
        ):
            return None

        boundary = best_rate + (
            retracement_distance
            if trade_direction == QuickAdapterV3._TRADE_SHORT
            else -retracement_distance
        )
        if boundary == best_rate:
            boundary = math.nextafter(
                best_rate,
                math.inf if trade_direction == QuickAdapterV3._TRADE_SHORT else 0.0,
            )
            retracement_distance = abs(boundary - best_rate)
        if not np.isfinite(boundary) or (
            boundary <= best_rate
            if trade_direction == QuickAdapterV3._TRADE_SHORT
            else not 0 < boundary < best_rate
        ):
            return None
        return float(retracement_distance)

    @staticmethod
    def _build_final_take_profit_state(
        *,
        exit_stage: int,
        trade_direction: TradeDirection,
        current_rate: float,
        take_profit_distance: float,
        retracement_fraction: float,
        candle_date: datetime.datetime | None,
        timeframe: str,
    ) -> _FinalTakeProfitState | None:
        normalized_candle_date = QuickAdapterV3._as_utc_candle_date(candle_date)
        if (
            type(exit_stage) is not int
            or exit_stage < 0
            or not isinstance(trade_direction, str)
            or trade_direction not in QuickAdapterV3._TRADE_DIRECTIONS_SET
            or not is_finite_number(current_rate)
            or current_rate <= 0
            or not is_finite_number(take_profit_distance)
            or take_profit_distance <= 0
            or not is_finite_number(retracement_fraction)
            or not 0 < retracement_fraction <= 1
            or not QuickAdapterV3._is_candle_date_aligned(normalized_candle_date, timeframe)
            or not isinstance(timeframe, str)
            or not timeframe
        ):
            return None
        retracement_distance = take_profit_distance * retracement_fraction
        if not np.isfinite(retracement_distance) or retracement_distance < 0:
            return None
        retracement_distance = QuickAdapterV3._normalize_final_take_profit_retracement_distance(
            best_rate=float(current_rate),
            retracement_distance=float(retracement_distance),
            trade_direction=trade_direction,
        )
        if retracement_distance is None:
            return None
        candle_date_isoformat = normalized_candle_date.isoformat()
        state: _FinalTakeProfitState = {
            "version": QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_VERSION,
            "exit_stage": exit_stage,
            "trade_direction": trade_direction,
            "best_rate": float(current_rate),
            "retracement_distance": float(retracement_distance),
            "boundary_candle_date": candle_date_isoformat,
            "last_candle_date": candle_date_isoformat,
            "trigger_candle_date": None,
            "timeframe": timeframe,
        }
        return state if QuickAdapterV3._is_valid_final_take_profit_boundary(state) else None

    @staticmethod
    def _normalize_final_take_profit_state(
        state: Any,
        *,
        exit_stage: int,
        trade_direction: TradeDirection,
        open_rate: float,
        timeframe: str,
        minimum_candle_date: datetime.datetime,
        current_candle_date: datetime.datetime,
    ) -> tuple[_FinalTakeProfitState | None, bool]:
        if state is None:
            return None, False
        minimum_candle_date_utc = QuickAdapterV3._as_utc_candle_date(minimum_candle_date)
        current_candle_date_utc = QuickAdapterV3._as_utc_candle_date(current_candle_date)
        if (
            minimum_candle_date_utc is None
            or current_candle_date_utc is None
            or minimum_candle_date_utc > current_candle_date_utc
            or not QuickAdapterV3._is_candle_date_aligned(minimum_candle_date_utc, timeframe)
            or not QuickAdapterV3._is_candle_date_aligned(current_candle_date_utc, timeframe)
            or not isinstance(state, dict)
            or type(state.get("version")) is not int
            or state.get("version")
            not in QuickAdapterV3._FINAL_TAKE_PROFIT_SUPPORTED_STATE_VERSIONS
            or (
                state.get("version") == QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_VERSION
                and "trigger_candle_date" not in state
            )
            or type(state.get("exit_stage")) is not int
            or state.get("exit_stage") != exit_stage
            or not isinstance(state.get("trade_direction"), str)
            or state.get("trade_direction") != trade_direction
            or state.get("trade_direction") not in QuickAdapterV3._TRADE_DIRECTIONS_SET
            or not is_finite_number(open_rate)
            or open_rate <= 0
            or not is_finite_number(state.get("best_rate"))
            or state.get("best_rate") <= 0
            or not is_finite_number(state.get("retracement_distance"))
            or state.get("retracement_distance") <= 0
            or state.get("timeframe") != timeframe
        ):
            return None, True
        state_version = state["version"]
        last_candle_date = QuickAdapterV3._as_utc_candle_date(state.get("last_candle_date"))
        boundary_candle_date = (
            last_candle_date
            if state_version == 1
            else QuickAdapterV3._as_utc_candle_date(state.get("boundary_candle_date"))
        )
        raw_trigger_candle_date = (
            state.get("trigger_candle_date")
            if state_version == QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_VERSION
            else None
        )
        trigger_candle_date = (
            QuickAdapterV3._as_utc_candle_date(raw_trigger_candle_date)
            if raw_trigger_candle_date is not None
            else None
        )
        if (
            last_candle_date is None
            or boundary_candle_date is None
            or not QuickAdapterV3._is_candle_date_aligned(boundary_candle_date, timeframe)
            or not QuickAdapterV3._is_candle_date_aligned(last_candle_date, timeframe)
            or boundary_candle_date < minimum_candle_date_utc
            or boundary_candle_date > last_candle_date
            or last_candle_date > current_candle_date_utc
            or (raw_trigger_candle_date is not None and trigger_candle_date is None)
            or (
                trigger_candle_date is not None
                and (
                    not QuickAdapterV3._is_candle_date_aligned(trigger_candle_date, timeframe)
                    or trigger_candle_date <= boundary_candle_date
                    or trigger_candle_date != last_candle_date
                )
            )
        ):
            return None, True

        try:
            best_rate = float(state["best_rate"])
            retracement_distance = float(state["retracement_distance"])
        except (OverflowError, TypeError, ValueError):
            return None, True
        if (
            not np.isfinite(best_rate)
            or best_rate <= 0
            or (
                best_rate >= open_rate
                if trade_direction == QuickAdapterV3._TRADE_SHORT
                else best_rate <= open_rate
            )
            or not np.isfinite(retracement_distance)
            or retracement_distance <= 0
        ):
            return None, True
        retracement_distance = QuickAdapterV3._normalize_final_take_profit_retracement_distance(
            best_rate=best_rate,
            retracement_distance=retracement_distance,
            trade_direction=trade_direction,
        )
        if retracement_distance is None:
            return None, True

        normalized_state: _FinalTakeProfitState = {
            "version": QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_VERSION,
            "exit_stage": exit_stage,
            "trade_direction": trade_direction,
            "best_rate": best_rate,
            "retracement_distance": retracement_distance,
            "boundary_candle_date": boundary_candle_date.isoformat(),
            "last_candle_date": last_candle_date.isoformat(),
            "trigger_candle_date": (
                trigger_candle_date.isoformat() if trigger_candle_date is not None else None
            ),
            "timeframe": timeframe,
        }
        if not QuickAdapterV3._is_valid_final_take_profit_boundary(normalized_state):
            return None, True
        return normalized_state, normalized_state != state

    @staticmethod
    def _final_take_profit_boundary(state: _FinalTakeProfitState) -> float:
        return state["best_rate"] + (
            state["retracement_distance"]
            if state["trade_direction"] == QuickAdapterV3._TRADE_SHORT
            else -state["retracement_distance"]
        )

    @staticmethod
    def _is_valid_final_take_profit_boundary(
        state: _FinalTakeProfitState,
    ) -> bool:
        best_rate = state["best_rate"]
        boundary = QuickAdapterV3._final_take_profit_boundary(state)
        if not np.isfinite(best_rate) or not np.isfinite(boundary):
            return False
        return (
            boundary > best_rate
            if state["trade_direction"] == QuickAdapterV3._TRADE_SHORT
            else 0 < boundary < best_rate
        )

    @staticmethod
    def _advance_final_take_profit_state(
        state: _FinalTakeProfitState,
        *,
        current_rate: float,
        candle_date: datetime.datetime | None,
    ) -> tuple[float, bool, bool]:
        boundary = QuickAdapterV3._final_take_profit_boundary(state)
        current_candle_date = QuickAdapterV3._as_utc_candle_date(candle_date)
        previous_candle_date = QuickAdapterV3._as_utc_candle_date(state["last_candle_date"])
        if (
            not is_finite_number(current_rate)
            or current_rate <= 0
            or current_candle_date is None
            or previous_candle_date is None
            or current_candle_date < previous_candle_date
        ):
            return boundary, False, False
        if state["trigger_candle_date"] is not None:
            return boundary, True, False
        if current_candle_date == previous_candle_date:
            return boundary, False, False

        previous_best_rate = state["best_rate"]
        candidate_best_rate = (
            min(previous_best_rate, current_rate)
            if state["trade_direction"] == QuickAdapterV3._TRADE_SHORT
            else max(previous_best_rate, current_rate)
        )
        if candidate_best_rate != previous_best_rate:
            candidate_retracement_distance = (
                QuickAdapterV3._normalize_final_take_profit_retracement_distance(
                    best_rate=candidate_best_rate,
                    retracement_distance=state["retracement_distance"],
                    trade_direction=state["trade_direction"],
                )
            )
            if candidate_retracement_distance is not None:
                state["best_rate"] = candidate_best_rate
                state["retracement_distance"] = candidate_retracement_distance
                state["boundary_candle_date"] = current_candle_date.isoformat()
        state["last_candle_date"] = current_candle_date.isoformat()

        boundary = QuickAdapterV3._final_take_profit_boundary(state)
        should_exit = (
            current_rate >= boundary
            if state["trade_direction"] == QuickAdapterV3._TRADE_SHORT
            else current_rate <= boundary
        )
        if should_exit:
            state["trigger_candle_date"] = current_candle_date.isoformat()
        return boundary, should_exit, True

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | tuple[float | None, str | None] | None:
        pair = trade.pair
        if trade.has_open_orders:
            return None

        trade_exit_stage = QuickAdapterV3.get_trade_exit_stage(trade)
        if trade_exit_stage not in QuickAdapterV3.partial_exit_stages:
            return None

        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.config.get("timeframe"))
        if df.empty:
            return None

        trade_take_profit_target = self.get_take_profit_target(df, trade, trade_exit_stage)
        if trade_take_profit_target is None:
            return None
        trade_take_profit_price, _ = trade_take_profit_target

        self.safe_append_trade_take_profit_price(trade, trade_take_profit_price, trade_exit_stage)

        trade_partial_exit = QuickAdapterV3.can_take_profit(
            trade, current_exit_rate, trade_take_profit_price
        )
        if not trade_partial_exit:
            self.throttle_callback(
                pair=pair,
                current_time=current_time,
                callback=lambda: logger.info(
                    f"[{pair}] {trade.trade_direction} partial exit stage {trade_exit_stage} | "
                    f"Take-profit target: {format_number(trade_take_profit_price)}, rate: {format_number(current_exit_rate)}"
                ),
            )
        if trade_partial_exit:
            trade_stake_percent = QuickAdapterV3.partial_exit_stages[trade_exit_stage][1]
            trade_partial_stake_amount = trade_stake_percent * trade.stake_amount
            if min_stake is not None and min_stake > 0:
                current_position_value = trade.amount * current_exit_rate
                # Live/dry-run passes ``min_entry_stake``, while freqtrade's
                # backtesting path already passes the adjusted minimum it guards.
                min_remaining_position_value = min_stake
                if self.is_trade_runmode:
                    # For both the cost- and amount-driven minimum, ``min_exit_stake``
                    # <= ``min_stake`` * max(exit/entry, 1/(1-|sl|)).
                    min_remaining_position_value *= max(
                        current_exit_rate / current_entry_rate,
                        1.0 / (1.0 - abs(self.stoploss)),
                    )
                min_remaining_position_value *= 1.0 + QuickAdapterV3._PARTIAL_EXIT_MIN_STAKE_MARGIN
                if current_position_value <= min_remaining_position_value:
                    return None
                remaining_position_value = current_position_value * (1 - trade_stake_percent)
                if remaining_position_value < min_remaining_position_value:
                    initial_trade_partial_stake_amount = trade_partial_stake_amount
                    trade_partial_stake_amount = trade.stake_amount * (
                        1 - min_remaining_position_value / current_position_value
                    )
                    logger.info(
                        f"[{pair}] {trade.trade_direction} partial exit stage "
                        f"{trade_exit_stage} | stake "
                        f"{format_number(initial_trade_partial_stake_amount)} -> "
                        f"{format_number(trade_partial_stake_amount)} to preserve "
                        f"min_remaining_position_value {format_number(min_remaining_position_value)}"
                    )
            return (
                -trade_partial_stake_amount,
                (
                    f"{QuickAdapterV3._TAKE_PROFIT_ORDER_TAG_PREFIX}"
                    f"{trade.trade_direction}_{trade_exit_stage}"
                ),
            )

        return None

    @staticmethod
    def weighted_close(series: Series, weight: float = 2.0) -> float:
        return float(series.get("high") + series.get("low") + weight * series.get("close")) / (
            2.0 + weight
        )

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

    def _invalidate_pair_caches(self, pair: str, df_signature: DfSignature | None = None) -> None:
        if df_signature is None or self._cached_df_signature.get(pair) != df_signature:
            self._candle_deviation_cache = {
                k: v for k, v in self._candle_deviation_cache.items() if k[0] != pair
            }
            self._candle_threshold_cache = {
                k: v for k, v in self._candle_threshold_cache.items() if k[0] != pair
            }
            if df_signature is None:
                self._cached_df_signature.pop(pair, None)
            else:
                self._cached_df_signature[pair] = df_signature

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
        self._invalidate_pair_caches(pair, df_signature)
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

        candle_idx = QuickAdapterV3._normalize_candle_idx(len(label_natr_series), candle_idx)

        label_natr_values = label_natr_series.iloc[: candle_idx + 1].to_numpy()
        if label_natr_values.size == 0:
            return np.nan
        candle_label_natr_value = label_natr_values[-1]
        if isna(candle_label_natr_value) or candle_label_natr_value < 0:
            return np.nan
        label_period_candles = self.get_label_period_candles(pair, df, candle_idx)
        candle_label_natr_value_quantile = calculate_quantile(
            label_natr_values[-label_period_candles:], candle_label_natr_value
        )
        if isna(candle_label_natr_value_quantile):
            return np.nan

        if interpolation_direction == QuickAdapterV3._INTERPOLATION_DIRECT:
            natr_multiplier_fraction = (
                min_natr_multiplier_fraction
                + (max_natr_multiplier_fraction - min_natr_multiplier_fraction)
                * candle_label_natr_value_quantile**quantile_exponent
            )
        elif interpolation_direction == QuickAdapterV3._INTERPOLATION_INVERSE:
            natr_multiplier_fraction = (
                max_natr_multiplier_fraction
                - (max_natr_multiplier_fraction - min_natr_multiplier_fraction)
                * candle_label_natr_value_quantile**quantile_exponent
            )
        else:
            raise ValueError(
                enum_error_message(
                    "interpolation_direction",
                    interpolation_direction,
                    QuickAdapterV3._INTERPOLATION_DIRECTIONS,
                )
            )
        candle_deviation = (
            candle_label_natr_value / 100.0
        ) * self.get_label_natr_multiplier_fraction(pair, natr_multiplier_fraction, df, candle_idx)
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
        self._invalidate_pair_caches(pair, df_signature)
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
            interpolation_direction=QuickAdapterV3._INTERPOLATION_DIRECTIONS[0],  # "direct"
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

        if side == QuickAdapterV3._TRADE_LONG:
            base_price = (
                QuickAdapterV3.weighted_close(candle) if is_candle_bearish else candle_close
            )
            candle_threshold = base_price * (1 + current_deviation)
        elif side == QuickAdapterV3._TRADE_SHORT:
            base_price = (
                QuickAdapterV3.weighted_close(candle) if is_candle_bullish else candle_close
            )
            candle_threshold = base_price * (1 - current_deviation)
        else:
            raise ValueError(enum_error_message("side", side, QuickAdapterV3._TRADE_DIRECTIONS))
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
        """Confirm a directional reversal using a volatility-adaptive threshold.

        Computes a deviation-based threshold on the latest candle (-1); ``rate``
        must strictly break it (long: ``rate > threshold``; short: ``rate <
        threshold``). When ``lookback_period_candles > 0``, requires that for
        each ``k = 1..lookback_period_candles`` the close at ``-k`` strictly
        broke the threshold recomputed at ``-(k+1)`` with the natr-multiplier
        bounds geometrically decayed by ``decay_fraction ** k`` clamped to
        ``[0, 1]``. A non-finite intermediate close or threshold aborts the
        chain: entries fail closed, while exits retain the valid current-candle
        result to allow exposure reduction without guaranteeing a profitable
        exit. Returns False on empty dataframe, invalid side/order, non-finite
        rate, negative lookback, ``decay_fraction`` outside ``(0, 1]``, or
        invalid min/max ordering.
        """
        if df.empty:
            return False
        if side not in QuickAdapterV3._TRADE_DIRECTIONS_SET:
            return False
        if order not in QuickAdapterV3._ORDER_TYPES_SET:
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
        lookback_period_candles = min(lookback_period_candles, max_lookback_period_candles)
        if not isinstance(decay_fraction, (int, float)):
            logger.debug(f"[{pair}] Denied {trade_direction} {order}: invalid decay_fraction type")
            return False
        if not (0.0 < decay_fraction <= 1.0):
            logger.debug(
                f"[{pair}] Denied {trade_direction} {order}: invalid decay_fraction {format_number(decay_fraction)}, must be in (0, 1]"
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
            (side == QuickAdapterV3._TRADE_LONG and rate > current_threshold)
            or (side == QuickAdapterV3._TRADE_SHORT and rate < current_threshold)
        )
        if order == QuickAdapterV3._ORDER_EXIT:
            if side == QuickAdapterV3._TRADE_LONG:
                trade_direction = QuickAdapterV3._TRADE_SHORT
            if side == QuickAdapterV3._TRADE_SHORT:
                trade_direction = QuickAdapterV3._TRADE_LONG
        if not current_ok:
            logger.debug(
                f"[{pair}] Denied {trade_direction} {order}: rate {format_number(rate)} did not break threshold {format_number(current_threshold)}"
            )
            return False

        if lookback_period_candles == 0:
            return current_ok

        unmeasurable_history_ok = order == QuickAdapterV3._ORDER_EXIT and current_ok
        for k in range(1, lookback_period_candles + 1):
            close_k = df.iloc[-k].get("close")
            if not isinstance(close_k, (int, float)) or not np.isfinite(close_k):
                return unmeasurable_history_ok

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
            if not isinstance(threshold_k, (int, float)) or not np.isfinite(threshold_k):
                return unmeasurable_history_ok

            if (side == QuickAdapterV3._TRADE_LONG and not (close_k > threshold_k)) or (
                side == QuickAdapterV3._TRADE_SHORT and not (close_k < threshold_k)
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
    @lru_cache(maxsize=_CACHE_MAXSIZE_LARGE)
    def is_isoformat(string: str) -> bool:
        if not isinstance(string, str):
            return False
        try:
            datetime.datetime.fromisoformat(string)
        except (ValueError, TypeError):
            return False
        return True

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime.datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.config.get("timeframe"))
        if df.empty:
            return None

        last_candle = df.iloc[-1]
        last_candle_date = QuickAdapterV3._as_utc_candle_date(last_candle.get("date"))
        has_valid_candle_date = QuickAdapterV3._is_candle_date_aligned(
            last_candle_date, self.timeframe
        )
        if last_candle.get("do_predict") == 2:
            return "model_expired"
        if last_candle.get("DI_catch") == 0:
            last_outlier_date_isoformat = trade.get_custom_data("last_outlier_date")
            last_outlier_date = (
                datetime.datetime.fromisoformat(last_outlier_date_isoformat)
                if QuickAdapterV3.is_isoformat(last_outlier_date_isoformat)
                else None
            )
            if has_valid_candle_date and last_outlier_date != last_candle_date:
                n_outliers = trade.get_custom_data("n_outliers", 0)
                n_outliers += 1
                logger.warning(
                    f"[{pair}] Detected new predictions outlier ({n_outliers=}) on trade {trade.id}"
                )
                trade.set_custom_data("n_outliers", n_outliers)
                trade.set_custom_data("last_outlier_date", last_candle_date.isoformat())

        if (
            trade.trade_direction == QuickAdapterV3._TRADE_SHORT
            and last_candle.get("do_predict") == 1
            and last_candle.get("DI_catch") == 1
            and last_candle.get(EXTREMA_COLUMN) < last_candle.get("minima_threshold")
            and self.reversal_confirmed(
                df,
                pair,
                QuickAdapterV3._TRADE_LONG,
                QuickAdapterV3._ORDER_EXIT,
                current_rate,
                self.reversal_confirmation["lookback_period_candles"],
                self.reversal_confirmation["decay_fraction"],
                self.reversal_confirmation["min_natr_multiplier_fraction"],
                self.reversal_confirmation["max_natr_multiplier_fraction"],
            )
        ):
            return "minima_detected_short"
        if (
            trade.trade_direction == QuickAdapterV3._TRADE_LONG
            and last_candle.get("do_predict") == 1
            and last_candle.get("DI_catch") == 1
            and last_candle.get(EXTREMA_COLUMN) > last_candle.get("maxima_threshold")
            and self.reversal_confirmed(
                df,
                pair,
                QuickAdapterV3._TRADE_SHORT,
                QuickAdapterV3._ORDER_EXIT,
                current_rate,
                self.reversal_confirmation["lookback_period_candles"],
                self.reversal_confirmation["decay_fraction"],
                self.reversal_confirmation["min_natr_multiplier_fraction"],
                self.reversal_confirmation["max_natr_multiplier_fraction"],
            )
        ):
            return "maxima_detected_long"

        if trade.has_open_orders:
            return None

        trade_exit_stage = QuickAdapterV3.get_trade_exit_stage(trade)
        if trade_exit_stage in QuickAdapterV3.partial_exit_stages:
            return None

        if not has_valid_candle_date:
            return None

        raw_final_take_profit_state = trade.get_custom_data(
            QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_KEY
        )
        final_take_profit_state, state_normalized = (
            QuickAdapterV3._normalize_final_take_profit_state(
                raw_final_take_profit_state,
                exit_stage=trade_exit_stage,
                trade_direction=trade.trade_direction,
                open_rate=trade.open_rate,
                timeframe=self.timeframe,
                minimum_candle_date=self.get_trade_entry_date(trade),
                current_candle_date=last_candle_date,
            )
        )
        if raw_final_take_profit_state is not None and final_take_profit_state is None:
            trade.set_custom_data(QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_KEY, None)
            self.throttle_callback(
                pair=pair,
                current_time=current_time,
                callback=lambda: logger.warning(
                    f"[{pair}] Ignoring invalid final take-profit state for trade {trade.id}; "
                    "the final exit will re-arm after its target is reached"
                ),
            )

        if final_take_profit_state is not None:
            boundary, trade_exit, state_changed = QuickAdapterV3._advance_final_take_profit_state(
                final_take_profit_state,
                current_rate=current_rate,
                candle_date=last_candle_date,
            )
            if state_normalized or state_changed:
                trade.set_custom_data(
                    QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_KEY,
                    final_take_profit_state,
                )
            if state_changed:
                self.throttle_callback(
                    pair=pair,
                    current_time=current_time,
                    callback=lambda: logger.info(
                        f"[{pair}] {trade.trade_direction} final exit | "
                        "Take-profit trail: "
                        f"best={format_number(final_take_profit_state['best_rate'])}, "
                        f"boundary={format_number(boundary)}, rate={format_number(current_rate)}"
                    ),
                )
            if trade_exit:
                return (
                    f"{QuickAdapterV3._TAKE_PROFIT_ORDER_TAG_PREFIX}{trade.trade_direction}_final"
                )
            return None

        trade_take_profit_target = self.get_take_profit_target(df, trade, trade_exit_stage)
        if trade_take_profit_target is None:
            return None
        trade_take_profit_price, trade_take_profit_distance = trade_take_profit_target

        self.safe_append_trade_take_profit_price(trade, trade_take_profit_price, trade_exit_stage)
        if not QuickAdapterV3.can_take_profit(trade, current_rate, trade_take_profit_price):
            self.throttle_callback(
                pair=pair,
                current_time=current_time,
                callback=lambda: logger.info(
                    f"[{pair}] {trade.trade_direction} final exit | "
                    f"Take-profit target: {format_number(trade_take_profit_price)}, rate: {format_number(current_rate)}"
                ),
            )
            return None

        state = QuickAdapterV3._build_final_take_profit_state(
            exit_stage=trade_exit_stage,
            trade_direction=trade.trade_direction,
            current_rate=current_rate,
            take_profit_distance=trade_take_profit_distance,
            retracement_fraction=self.final_take_profit_retracement_fraction,
            candle_date=(last_candle_date if has_valid_candle_date else None),
            timeframe=self.timeframe,
        )
        if state is None:
            self.throttle_callback(
                pair=pair,
                current_time=current_time,
                callback=lambda: logger.warning(
                    f"[{pair}] {trade.trade_direction} final exit | "
                    "Take-profit target reached but the trailing state is unmeasurable; "
                    "exit not armed"
                ),
            )
            return None

        trade.set_custom_data(QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_KEY, state)
        logger.info(
            f"[{pair}] {trade.trade_direction} final exit | "
            f"Take-profit armed at rate={format_number(current_rate)}, "
            f"retracement_distance={format_number(state['retracement_distance'])}"
        )
        return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime.datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        if side not in QuickAdapterV3._TRADE_DIRECTIONS_SET:
            return False
        if side == QuickAdapterV3._TRADE_SHORT and not self.can_short:
            logger.info(
                f"[{pair}] Denied short {QuickAdapterV3._ORDER_ENTRY}: shorting not allowed"
            )
            return False
        max_open_trades = self.config.get("max_open_trades", 0)
        if (
            not QuickAdapterV3._is_unlimited_max_open_trades(max_open_trades)
            and Trade.get_open_trade_count() >= max_open_trades
        ):
            return False
        max_open_trades_per_side = self.max_open_trades_per_side
        if max_open_trades_per_side >= 0:
            open_trades = Trade.get_open_trades()
            trades_per_side = sum(1 for trade in open_trades if trade.trade_direction == side)
            if trades_per_side >= max_open_trades_per_side:
                return False

        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.config.get("timeframe"))
        if df.empty:
            logger.info(f"[{pair}] Denied {side} {QuickAdapterV3._ORDER_ENTRY}: dataframe is empty")
            return False
        return bool(
            self.reversal_confirmed(
                df,
                pair,
                side,
                QuickAdapterV3._ORDER_ENTRY,
                rate,
                self.reversal_confirmation["lookback_period_candles"],
                self.reversal_confirmation["decay_fraction"],
                self.reversal_confirmation["min_natr_multiplier_fraction"],
                self.reversal_confirmation["max_natr_multiplier_fraction"],
            )
        )

    def is_short_allowed(self) -> bool:
        trading_mode = self.config.get("trading_mode")
        if trading_mode in {
            QuickAdapterV3._TRADING_MODE_MARGIN,
            QuickAdapterV3._TRADING_MODE_FUTURES,
        }:  # margin, futures
            return True
        elif trading_mode == QuickAdapterV3._TRADING_MODE_SPOT:
            return False
        else:
            raise ValueError(
                enum_error_message("trading_mode", trading_mode, QuickAdapterV3._TRADING_MODES)
            )

    @cached_property
    def _configured_leverage(self) -> float | None:
        leverage = self.config.get("leverage")
        if leverage is None:
            return None
        if not is_finite_number(leverage):
            logger.warning(
                f"Invalid leverage value {leverage!r}: must be a finite number, "
                "using proposed_leverage"
            )
            return None
        leverage = float(leverage)
        if leverage < 1.0:
            logger.warning(f"Invalid leverage value {leverage}: must be >= 1.0, clamping to 1.0")
        return leverage

    def leverage(
        self,
        pair: str,
        current_time: datetime.datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs: Any,
    ) -> float:
        configured_leverage = self._configured_leverage
        if configured_leverage is None:
            configured_leverage = proposed_leverage
        return float(max(1.0, min(configured_leverage, max_leverage)))

    def plot_annotations(
        self,
        pair: str,
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        dataframe: DataFrame,
        **kwargs: Any,
    ) -> list[AnnotationType]:
        annotations: list[AnnotationType] = []

        open_trades = Trade.get_trades_proxy(pair=pair, is_open=True)

        annotation_candle_date = (
            QuickAdapterV3._as_utc_candle_date(dataframe.iloc[-1].get("date"))
            if not dataframe.empty
            else None
        )

        for trade in open_trades:
            if trade.open_date_utc > end_date:
                continue

            trade_annotation_line_start_date = self.get_trade_annotation_line_start_date(
                dataframe, trade
            )

            trade_exit_stage = QuickAdapterV3.get_trade_exit_stage(trade)

            for take_profit_stage in QuickAdapterV3.partial_exit_stages:
                if take_profit_stage < trade_exit_stage:
                    continue

                partial_take_profit_target = self.get_take_profit_target(
                    dataframe, trade, take_profit_stage
                )
                if partial_take_profit_target is None:
                    continue
                partial_take_profit_price, _ = partial_take_profit_target
                take_profit_line_annotation: AnnotationType = {
                    "type": "line",
                    "start": max(trade_annotation_line_start_date, start_date),
                    "end": end_date,
                    "y_start": partial_take_profit_price,
                    "y_end": partial_take_profit_price,
                    "color": QuickAdapterV3.partial_exit_stages[take_profit_stage][2],
                    "line_style": "solid",
                    "width": 1,
                    "label": f"Partial Take-Profit Stage {take_profit_stage}",
                    "z_level": 10 + take_profit_stage,
                }
                annotations.append(take_profit_line_annotation)

            final_exit_stage = QuickAdapterV3._FINAL_EXIT_STAGE
            raw_final_take_profit_state = trade.get_custom_data(
                QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_KEY
            )
            final_take_profit_state = None
            if annotation_candle_date is not None:
                final_take_profit_state, _ = QuickAdapterV3._normalize_final_take_profit_state(
                    raw_final_take_profit_state,
                    exit_stage=final_exit_stage,
                    trade_direction=trade.trade_direction,
                    open_rate=trade.open_rate,
                    timeframe=self.timeframe,
                    minimum_candle_date=self.get_trade_entry_date(trade),
                    current_candle_date=annotation_candle_date,
                )

            if final_take_profit_state is not None:
                boundary_candle_date = QuickAdapterV3._as_utc_candle_date(
                    final_take_profit_state["boundary_candle_date"]
                )
                if boundary_candle_date is not None:
                    trail_start = max(boundary_candle_date, start_date)
                    if trail_start <= end_date:
                        final_take_profit_price = QuickAdapterV3._final_take_profit_boundary(
                            final_take_profit_state
                        )
                        annotations.append(
                            {
                                "type": "line",
                                "start": trail_start,
                                "end": end_date,
                                "y_start": final_take_profit_price,
                                "y_end": final_take_profit_price,
                                "color": QuickAdapterV3._FINAL_EXIT_STAGE_PARAMS[2],
                                "line_style": "solid",
                                "width": 1,
                                "label": "Final Take-Profit Trail (current)",
                                "z_level": 10 + final_exit_stage,
                            }
                        )
                continue

            final_take_profit_target = self.get_take_profit_target(
                dataframe, trade, final_exit_stage
            )
            if final_take_profit_target is not None:
                final_take_profit_price, _ = final_take_profit_target
                annotations.append(
                    {
                        "type": "line",
                        "start": max(trade_annotation_line_start_date, start_date),
                        "end": end_date,
                        "y_start": final_take_profit_price,
                        "y_end": final_take_profit_price,
                        "color": QuickAdapterV3._FINAL_EXIT_STAGE_PARAMS[2],
                        "line_style": "solid",
                        "width": 1,
                        "label": "Final Take-Profit Arming Target",
                        "z_level": 10 + final_exit_stage,
                    }
                )

        return annotations

    def optuna_load_best_params(
        self, pair: str, namespace: OptunaNamespace
    ) -> dict[str, Any] | None:
        # Strategy consumes only output tunables (``label_period_candles``,
        # ``label_horizon_candles``, ``label_natr_multiplier``);
        # selection-metadata drift on cached label ``best_params`` is
        # tolerable here. The regressor's ``optuna_load_best_params``
        # passes ``expected_selection_metadata`` and rejects drift before
        # re-running HPO selection.
        return optuna_load_best_params(
            self.models_full_path, pair, namespace, logger, pairs=self.pairs
        )
