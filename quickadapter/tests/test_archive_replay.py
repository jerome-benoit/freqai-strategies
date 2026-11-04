"""Reject native archive replay rather than borrow deployed QuickAdapter helpers."""

import re
import unittest
from pathlib import Path

from freqtrade.enums import TradingMode
from freqtrade.resolvers import StrategyResolver
from qa_support import QaTestCase, model_config, temporary_directory

from scripts.archive_replay_contract import HAS_ARCHIVED_STRATEGY_REPLAY, archived_strategy

STRATEGY_SOURCE = Path(__file__).resolve().parents[1] / "user_data/strategies/QuickAdapterV3.py"
MODEL_DIRECTORY = STRATEGY_SOURCE.parent.parent / "freqaimodels"


@unittest.skipUnless(HAS_ARCHIVED_STRATEGY_REPLAY, "Native archived replay is unavailable")
class ArchiveReplayTest(QaTestCase):
    def test_archived_source_with_cached_helpers_cannot_start_the_model(self):
        with temporary_directory() as directory:
            config = model_config(
                directory,
                strategy="QuickAdapterV3",
                trading_mode=TradingMode.SPOT,
                freqaimodel="QuickAdapterRegressorV3",
                freqaimodel_path=str(MODEL_DIRECTORY),
                freqai={"optuna_hyperopt": {"enabled": False}},
            )
            with archived_strategy(config, STRATEGY_SOURCE) as (replay, source):
                strategy = StrategyResolver.load_strategy(replay)
                self.assertEqual(Path(strategy.__file__).resolve(), source.resolve())
                with self.assertRaisesRegex(ValueError, re.escape(source.name)):
                    strategy.ft_bot_start()
                self.assertFalse((directory / "models").exists())

    def test_current_strategy_fallback_cannot_start_the_model(self):
        with temporary_directory() as directory:
            current = directory / "strategies"
            current.mkdir()
            (current / STRATEGY_SOURCE.name).write_bytes(STRATEGY_SOURCE.read_bytes())
            broken = directory / STRATEGY_SOURCE.name
            broken.write_text("import missing_archived_dependency\n" + STRATEGY_SOURCE.read_text())
            config = model_config(
                directory,
                strategy="QuickAdapterV3",
                trading_mode=TradingMode.SPOT,
                freqaimodel="QuickAdapterRegressorV3",
                freqaimodel_path=str(MODEL_DIRECTORY),
                freqai={"optuna_hyperopt": {"enabled": False}},
            )
            with archived_strategy(config, broken) as (replay, source):
                strategy = StrategyResolver.load_strategy(replay)
                self.assertEqual(
                    Path(strategy.__file__).resolve(), (current / STRATEGY_SOURCE.name).resolve()
                )
                with self.assertRaisesRegex(ValueError, re.escape(source.name)):
                    strategy.ft_bot_start()
                self.assertFalse((directory / "models").exists())
