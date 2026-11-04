"""Native archived replay must not train against a substituted current strategy."""

import builtins
import re
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import SignalDirection, TradingMode
from freqtrade.resolvers import StrategyResolver
from qa_support import QaTestCase, model_config, temporary_directory

from ReforceXY.user_data.freqaimodels.ReforceXY import Actions, ReforceXY
from scripts.archive_replay_contract import HAS_ARCHIVED_STRATEGY_REPLAY, archived_strategy

STRATEGY_SOURCE = Path(__file__).resolve().parents[1] / "user_data/strategies/RLAgentStrategy.py"


@unittest.skipUnless(HAS_ARCHIVED_STRATEGY_REPLAY, "Native archived replay is unavailable")
class ArchiveReplayTest(QaTestCase):
    def test_archived_strategy_remains_usable_without_cache_helper(self):
        with temporary_directory() as directory:
            config = model_config(
                directory, strategy="RLAgentStrategy", trading_mode=TradingMode.SPOT
            )
            original_import = builtins.__import__

            def without_cache_helper(name, *args, **kwargs):
                if name == "BacktestIdentity":
                    raise ImportError("The standalone cache helper is not archived")
                return original_import(name, *args, **kwargs)

            with (
                archived_strategy(config, STRATEGY_SOURCE) as (replay, source),
                mock.patch("builtins.__import__", side_effect=without_cache_helper),
            ):
                strategy = StrategyResolver.load_strategy(replay)
                self.assertEqual(Path(strategy.__file__).resolve(), source.resolve())
                frame = pd.DataFrame(
                    {
                        "date": pd.date_range(
                            end=pd.Timestamp.now(tz="UTC"), periods=3, freq="5min"
                        ),
                        "close": 100.0,
                        "volume": 1.0,
                        "do_predict": 1,
                        "&-action": [0, 0, Actions.Long_enter.value],
                    }
                )
                signals = strategy.populate_entry_trend(frame, {"pair": "BTC/USDT"})
                self.assertEqual(
                    strategy.get_entry_signal("BTC/USDT", "5m", signals),
                    (SignalDirection.LONG, "long"),
                )

    def test_current_strategy_fallback_cannot_start_training(self):
        with temporary_directory() as directory:
            current = directory / "strategies"
            current.mkdir()
            (current / STRATEGY_SOURCE.name).write_bytes(STRATEGY_SOURCE.read_bytes())
            broken = directory / STRATEGY_SOURCE.name
            broken.write_text("import missing_archived_dependency\n" + STRATEGY_SOURCE.read_text())
            config = model_config(
                directory, strategy="RLAgentStrategy", trading_mode=TradingMode.SPOT
            )
            with archived_strategy(config, broken) as (replay, source):
                strategy = StrategyResolver.load_strategy(replay)
                self.assertEqual(
                    Path(strategy.__file__).resolve(), (current / STRATEGY_SOURCE.name).resolve()
                )
                strategy.dp = DataProvider(replay, exchange=None)
                model = ReforceXY(config=replay)
                with self.assertRaisesRegex(ValueError, re.escape(source.name)):
                    model.start(
                        pd.DataFrame(
                            {
                                "date": pd.date_range(
                                    "2026-01-01", periods=8, freq="5min", tz="UTC"
                                ),
                                "open": 100.0,
                                "high": 101.0,
                                "low": 99.0,
                                "close": 100.0,
                                "volume": 1.0,
                            }
                        ),
                        {"pair": "BTC/USDT"},
                        strategy,
                    )
