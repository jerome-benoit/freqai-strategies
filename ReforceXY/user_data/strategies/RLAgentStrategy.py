import datetime
import logging
import math
from collections.abc import Mapping
from functools import reduce
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

# import talib.abstract as ta
from freqtrade.enums import RunMode
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
from pandas import DataFrame

TradingMode = Literal["margin", "futures", "spot"]
TradeDirection = Literal["long", "short"]

logger = logging.getLogger(__name__)

ACTION_COLUMN: Final = "&-action"


_EPOCH_MS_MIN = 1_262_304_000_000  # 2010-01-01T00:00:00Z
_EPOCH_MS_MAX = 2_051_222_400_000  # 2035-01-01T00:00:00Z


def _ensure_datetime_series(series: pd.Series | None) -> pd.Series:
    """Ensure a date series is datetime64[ms, UTC], following freqtrade's data handler pattern."""
    if series is None:
        raise ValueError(
            "Expected a date Series but received None. "
            "The 'date' column is missing from the dataframe."
        )
    if pd.api.types.is_integer_dtype(series):
        sample = series.dropna()
        if sample.empty:
            return pd.to_datetime(series, unit="ms", utc=True).dt.as_unit("ms")
        probe = int(sample.iat[0])
        if not (_EPOCH_MS_MIN <= probe <= _EPOCH_MS_MAX):
            raise ValueError(
                f"Integer date column value {probe} is outside the expected epoch-ms "
                f"range [{_EPOCH_MS_MIN}, {_EPOCH_MS_MAX}]. "
                "Data is likely corrupted or uses a different unit."
            )
        return pd.to_datetime(series, unit="ms", utc=True).dt.as_unit("ms")
    return series.dt.as_unit("ms")


class RLAgentStrategy(IStrategy):
    """
    RLAgentStrategy
    """

    INTERFACE_VERSION = 3

    _TRADING_MODES: Final[tuple[TradingMode, ...]] = ("margin", "futures", "spot")
    _TRADE_DIRECTIONS: Final[tuple[TradeDirection, ...]] = ("long", "short")
    _ACTION_ENTER_LONG: Final[int] = 1
    _ACTION_EXIT_LONG: Final[int] = 2
    _ACTION_ENTER_SHORT: Final[int] = 3
    _ACTION_EXIT_SHORT: Final[int] = 4
    _EXECUTION_PROFILES: Final[tuple[str, ...]] = ("research", "live")
    _DEFAULT_EXECUTION_PROFILE: Final[str] = "research"
    _RESEARCH_STOPLOSS_SENTINEL: Final[float] = -0.99

    @property
    def can_short(self) -> bool:
        return self.is_short_allowed()

    def _execution_contract_error(self) -> str | None:
        execution_config = self.config.get("reforcexy_execution", {})
        if not isinstance(execution_config, Mapping):
            return "Config: 'reforcexy_execution' must be an object."

        profile = execution_config.get("profile", RLAgentStrategy._DEFAULT_EXECUTION_PROFILE)
        if profile not in RLAgentStrategy._EXECUTION_PROFILES:
            return (
                f"Config: invalid ReforceXY execution profile {profile!r}. "
                f"Expected one of: {list(RLAgentStrategy._EXECUTION_PROFILES)}."
            )

        if getattr(self, "use_exit_signal", None) is not True:
            return "Config: ReforceXY RL actions require Freqtrade use_exit_signal=true."
        if getattr(self, "exit_profit_only", None) is not False:
            return "Config: ReforceXY RL actions require Freqtrade exit_profit_only=false."

        runmode = self.config.get("runmode")
        if runmode == RunMode.LIVE and profile != RLAgentStrategy._EXECUTION_PROFILES[1]:
            return (
                "Config: ReforceXY refuses live trading under the 'research' "
                "execution profile. Set profile='live' only after preregistering "
                "and dry-running the live risk controls."
            )

        if profile == RLAgentStrategy._EXECUTION_PROFILES[1]:
            try:
                stoploss = float(self.stoploss)
            except (TypeError, ValueError):
                return "Config: the ReforceXY live stoploss must be a finite number."
            if (
                not math.isfinite(stoploss)
                or stoploss <= RLAgentStrategy._RESEARCH_STOPLOSS_SENTINEL
            ):
                return (
                    "Config: the ReforceXY 'live' execution profile requires a "
                    "finite Freqtrade stoploss above the research sentinel -0.99."
                )

        return None

    def bot_start(self, **kwargs: Any) -> None:
        contract_error = self._execution_contract_error()
        if contract_error is not None:
            raise ValueError(contract_error)

        runmode = self.config.get("runmode")
        logger.info(
            "ReforceXY execution contract: profile=%s runmode=%s stoploss=%s protections=%d",
            self.config.get("reforcexy_execution", {}).get(
                "profile", RLAgentStrategy._DEFAULT_EXECUTION_PROFILE
            ),
            getattr(runmode, "value", runmode),
            self.stoploss,
            len(self.protections),
        )

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
        **kwargs: Any,
    ) -> bool:
        try:
            contract_error = self._execution_contract_error()
        except Exception as error:
            logger.critical(
                "ReforceXY entry blocked: pair=%s side=%s contract_check_error=%r",
                pair,
                side,
                error,
            )
            return False
        if contract_error is not None:
            logger.critical(
                "ReforceXY entry blocked: pair=%s side=%s reason=%s",
                pair,
                side,
                contract_error,
            )
            return False
        return True

    @property
    def protections(self) -> list[dict[str, Any]]:
        custom_protections = self.config.get("custom_protections", [])
        if not isinstance(custom_protections, list) or not all(
            isinstance(protection, dict) and isinstance(protection.get("method"), str)
            for protection in custom_protections
        ):
            raise ValueError(
                "Config: 'custom_protections' must be a list of protection objects, "
                "each with a 'method' string."
            )
        return custom_protections

    # def feature_engineering_expand_all(
    #     self, dataframe: DataFrame, period: int, metadata: dict[str, Any], **kwargs
    # ) -> DataFrame:
    #     dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)

    #     return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        dataframe["%-close_log_return"] = np.log(dataframe.get("close")).diff()
        dataframe["%-raw_volume"] = dataframe.get("volume")

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        dates = _ensure_datetime_series(dataframe.get("date"))
        dataframe["%-day_of_week"] = (dates.dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dates.dt.hour + 1) / 25

        dataframe["%-raw_close"] = dataframe.get("close")
        dataframe["%-raw_open"] = dataframe.get("open")
        dataframe["%-raw_high"] = dataframe.get("high")
        dataframe["%-raw_low"] = dataframe.get("low")

        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        dataframe[ACTION_COLUMN] = 0

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        enter_long_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get(ACTION_COLUMN) == RLAgentStrategy._ACTION_ENTER_LONG,  # 1,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_long_conditions),
            ["enter_long", "enter_tag"],
        ] = (1, RLAgentStrategy._TRADE_DIRECTIONS[0])  # "long"

        enter_short_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get(ACTION_COLUMN) == RLAgentStrategy._ACTION_ENTER_SHORT,  # 3,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, enter_short_conditions),
            ["enter_short", "enter_tag"],
        ] = (1, RLAgentStrategy._TRADE_DIRECTIONS[1])  # "short"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, Any]) -> DataFrame:
        exit_long_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get(ACTION_COLUMN) == RLAgentStrategy._ACTION_EXIT_LONG,  # 2,
        ]
        dataframe.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get(ACTION_COLUMN) == RLAgentStrategy._ACTION_EXIT_SHORT,  # 4,
        ]
        dataframe.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        last_candle = dataframe.iloc[-1]
        if last_candle.get("do_predict") == 2:
            trades = Trade.get_trades_proxy(pair=metadata.get("pair"), is_open=True)
            for trade in trades:
                last_index = dataframe.index[-1]
                if trade.is_short:
                    dataframe.at[last_index, "exit_short"] = 1
                else:
                    dataframe.at[last_index, "exit_long"] = 1
                dataframe.at[last_index, "exit_tag"] = "freqai_model_expired"

        return dataframe

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

    def is_short_allowed(self) -> bool:
        trading_mode = self.config.get("trading_mode")
        # "margin", "futures"
        if trading_mode in {
            RLAgentStrategy._TRADING_MODES[0],
            RLAgentStrategy._TRADING_MODES[1],
        }:
            return True
        # "spot"
        elif trading_mode == RLAgentStrategy._TRADING_MODES[2]:
            return False
        else:
            raise ValueError(
                f"Config: invalid trading_mode '{trading_mode}'. "
                f"Expected one of: {list(RLAgentStrategy._TRADING_MODES)}"
            )
