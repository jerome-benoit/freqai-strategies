import datetime
import logging
from functools import reduce
from typing import Any, Final, Literal, Optional

# import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy
from pandas import DataFrame

TradingMode = Literal["margin", "futures", "spot"]
TradeDirection = Literal["long", "short"]

logger = logging.getLogger(__name__)

ACTION_COLUMN: Final = "&-action"


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

    @property
    def can_short(self) -> bool:
        return self.is_short_allowed()

    # def feature_engineering_expand_all(
    #     self, dataframe: DataFrame, period: int, metadata: dict[str, Any], **kwargs
    # ) -> DataFrame:
    #     dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)

    #     return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        dataframe["%-close_pct_change"] = dataframe.get("close").pct_change()
        dataframe["%-raw_volume"] = dataframe.get("volume")

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict[str, Any], **kwargs
    ) -> DataFrame:
        dates = dataframe.get("date")
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

    def populate_indicators(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
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

    def populate_exit_trend(
        self, dataframe: DataFrame, metadata: dict[str, Any]
    ) -> DataFrame:
        exit_long_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get(ACTION_COLUMN) == RLAgentStrategy._ACTION_EXIT_LONG,  # 2,
        ]
        dataframe.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        exit_short_conditions = [
            dataframe.get("do_predict") == 1,
            dataframe.get(ACTION_COLUMN) == RLAgentStrategy._ACTION_EXIT_SHORT,  # 4,
        ]
        dataframe.loc[
            reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"
        ] = 1

        last_candle = dataframe.iloc[-1]
        if last_candle.get("do_predict") == 2:
            trades = Trade.get_trades_proxy(pair=metadata.get("pair"), is_open=True)
            for trade in trades:
                last_index = dataframe.index[-1]
                if trade.is_short:
                    dataframe.at[last_index, "exit_short"] = 1
                else:
                    dataframe.at[last_index, "exit_long"] = 1

        return dataframe

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
