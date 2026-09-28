"""Runtime contracts for FreqAI's native QuickAdapter prediction history."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from freqtrade.enums import RunMode
from freqtrade.exchange import timeframe_to_seconds
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

from quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 import QuickAdapterRegressorV3

PAIR = "BTC/USDT"
MARKER = "_freqai_strategies_produced"


def model_config(path: str) -> dict:
    return {
        "user_data_dir": Path(path),
        "timeframe": "5m",
        "stake_amount": "unlimited",
        "runmode": RunMode.DRY_RUN,
        "exchange": {"pair_whitelist": [PAIR]},
        "pairlists": [{"method": "StaticPairList"}],
        "freqai": {
            "enabled": True,
            "identifier": "quickadapter-runtime-regression",
            "continual_learning": True,
            "train_period_days": 1,
            "backtest_period_days": 1,
            "conv_width": 1,
            "fit_live_predictions_candles": 2,
            "feature_parameters": {
                "include_timeframes": ["5m"],
                "include_corr_pairlist": [],
                "label_period_candles": 1,
                "shuffle_after_split": False,
            },
            "data_split_parameters": {"test_size": 0, "shuffle": False},
            "model_training_parameters": {"n_estimators": 2, "n_jobs": 1},
            "label_prediction": {"method": "none"},
        },
    }


class PredictionHistoryTest(unittest.TestCase):
    def test_native_append_restart_and_causal_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            config = model_config(temp)
            model = QuickAdapterRegressorV3(config=config)
            model.live = True
            dates = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
            strat_df = pd.DataFrame(
                {"date": dates, "high": [101.0] * 4, "low": [99.0] * 4, "close": [100.0] * 4}
            )
            dk = SimpleNamespace(
                data={
                    "extra_returns_per_train": {
                        "label_period_candles": 1,
                        "label_natr_multiplier": 1.0,
                        "holdout_rmse": np.inf,
                    }
                },
                label_list=["&s-extrema"],
                unique_class_list=[],
                full_df=strat_df,
            )
            bootstrap = pd.DataFrame({"&s-extrema": [21.0, 22.0, 23.0, 24.0]})
            model.set_initial_historic_predictions(bootstrap, dk, PAIR, strat_df)
            model.fit_live_predictions(dk, PAIR)
            model.dd.set_initial_return_values(PAIR, bootstrap, strat_df)
            self.assertEqual(model.dd.historic_predictions[PAIR][MARKER].tolist(), [False] * 4)
            self.assertEqual(dk.data["labels_mean"]["&s-extrema"], 0.0)
            model.dk = SimpleNamespace(check_if_model_expired=lambda _: False)

            def produce(dataframe: pd.DataFrame, value: float, status: int) -> None:
                with mock.patch.object(
                    model,
                    "predict",
                    return_value=(pd.DataFrame({"&s-extrema": [value]}), np.array([status])),
                ):
                    model.build_strategy_return_arrays(dataframe, dk, PAIR, 0)

            produce(strat_df, 1.0, 0)
            extended = pd.concat(
                [
                    strat_df,
                    pd.DataFrame(
                        {
                            "date": [dates[-1] + pd.Timedelta(minutes=5)],
                            "high": [101.0],
                            "low": [99.0],
                            "close": [100.0],
                        }
                    ),
                ],
                ignore_index=True,
            )
            dk.full_df = extended
            produce(extended, 3.0, 1)
            model.fit_live_predictions(dk, PAIR)
            self.assertEqual(dk.data["labels_mean"]["&s-extrema"], 2.0)
            self.assertEqual(dk.data["labels_std"]["&s-extrema"], 1.0)
            self.assertEqual(len(model.dd.historic_predictions[PAIR]), len(extended))

            future = pd.concat(
                [
                    extended,
                    pd.DataFrame(
                        {
                            "date": [dates[-1] + pd.Timedelta(minutes=10)],
                            "high": [101.0],
                            "low": [99.0],
                            "close": [100.0],
                        }
                    ),
                ],
                ignore_index=True,
            )
            dk.full_df = future
            produce(future, 999.0, 1)
            self.assertNotIn(MARKER, dk.return_dataframe)
            dk.full_df = extended
            model.fit_live_predictions(dk, PAIR)
            self.assertEqual(dk.data["labels_mean"]["&s-extrema"], 2.0)
            model.dd.save_historic_predictions_to_disk()

            restored = QuickAdapterRegressorV3(config=config)
            restored.live = True
            self.assertTrue(restored.dd.load_historic_predictions_from_disk())
            restored.fit_live_predictions(dk, PAIR)
            self.assertEqual(dk.data["labels_mean"]["&s-extrema"], 2.0)
            self.assertEqual(dk.data["labels_std"]["&s-extrema"], 1.0)
            self.assertEqual(len(restored.dd.historic_predictions[PAIR]), len(future))
            self.assertEqual(
                restored.dd.historic_predictions[PAIR][MARKER].tolist(), [False] * 3 + [True] * 3
            )

    def test_backtest_does_not_use_future_model_but_resumes_from_earlier_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            config = model_config(temp)
            pair = PAIR

            def frame(day: str, offset: float) -> pd.DataFrame:
                values = np.arange(48)
                return pd.DataFrame(
                    {
                        "date": pd.date_range(day, periods=48, freq="5min", tz="UTC"),
                        "%-feature": np.sin(values / 4) + offset,
                        "&s-extrema": np.cos(values / 5),
                    }
                )

            def kitchen(settings: dict, data: pd.DataFrame, *, live: bool):
                dk = FreqaiDataKitchen(settings, live=live, pair=pair)
                timestamp = int(
                    (
                        data["date"].iloc[-1] + pd.Timedelta(seconds=timeframe_to_seconds("5m"))
                    ).timestamp()
                )
                dk.set_paths(pair, timestamp)
                dk.set_new_model_names(pair, timestamp)
                dk.data_path.mkdir(parents=True, exist_ok=True)
                dk.label_list = ["&s-extrema"]
                dk.training_features_list = ["%-feature"]
                return dk, timestamp

            source = QuickAdapterRegressorV3(config=config)
            source.live = True
            future = frame("2026-02-01", 1000.0)
            future_dk, future_ts = kitchen(config, future, live=True)
            deployed = source.train(future, pair, future_dk)
            source.dd.get_pair_dict_info(pair)
            source.dd.pair_dict[pair]["trained_timestamp"] = future_ts
            source.dd.save_data(deployed, pair, future_dk)

            backtest_config = dict(config)
            backtest_config["runmode"] = RunMode.BACKTEST
            backtest_config["timerange"] = "20260101-20260105"
            backtest_config["config_files"] = [
                "/workspace/quickadapter/user_data/config-template.json"
            ]
            backtest = QuickAdapterRegressorV3(config=backtest_config)
            backtest.live = False
            for day, offset, save_model in (
                ("2026-01-01", 0.0, False),
                ("2026-01-02", 10.0, True),
                ("2026-01-03", 20.0, False),
            ):
                training = frame(day, offset)
                dk, timestamp = kitchen(backtest_config, training, live=False)
                self.assertLess(timestamp, future_ts)
                trained = backtest.train(training, pair, dk)
                transformed = dk.data_dictionary["train_features"]["%-feature"]
                if offset < 20.0:
                    self.assertAlmostEqual(float(transformed.min()), -1.0)
                    self.assertAlmostEqual(float(transformed.max()), 1.0)
                else:
                    self.assertGreater(float(transformed.min()), 2.0)
                if save_model:
                    backtest.dd.pair_dict[pair]["trained_timestamp"] = future_ts
                    backtest.dd.save_data(trained, pair, dk)
                    backtest = QuickAdapterRegressorV3(config=backtest_config)
                    backtest.live = False
                    backtest.dd.get_pair_dict_info(pair)
                    self.assertEqual(backtest.dd.pair_dict[pair]["trained_timestamp"], future_ts)
                else:
                    backtest.dd.save_metadata(dk)


if __name__ == "__main__":
    unittest.main()
