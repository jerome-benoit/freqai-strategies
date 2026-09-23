"""Regression contracts for learning eligibility, replay and temporal observations."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from freqtrade.enums import RunMode
from freqtrade.exceptions import DependencyException
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from optuna import TrialPruned, create_study

from ReforceXY.tests.test_review_contracts import RecordingPolicy, model_config
from ReforceXY.user_data.freqaimodels.ReforceXY import ReforceXY


class TrainingObservationsTest(unittest.TestCase):
    def test_dqn_eligibility_restart_and_chronology(self):
        for algorithm in ("DQN", "QRDQN"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as temp:
                config = model_config(temp)
                info = config["freqai"]
                info["continual_learning"] = True
                info["feature_parameters"]["shuffle_after_split"] = True
                info["rl_config"]["model_type"] = algorithm
                info["model_training_parameters"] = {
                    "learning_starts": 0,
                    "buffer_size": 128,
                    "batch_size": 8,
                    "train_freq": 4,
                    "gradient_steps": 1,
                    "device": "cpu",
                    "policy_kwargs": {"net_arch": [8]},
                }
                model = ReforceXY(config=config)
                model.live = True
                model.can_short = False
                self.addCleanup(model.close_envs)
                frame = pd.DataFrame(
                    {
                        "date": pd.date_range("2026-01-01", periods=64, freq="5min", tz="UTC"),
                        "%-feature": np.sin(np.arange(64)),
                        "&-action": np.zeros(64),
                    }
                )
                for column in ("open", "high", "low", "close"):
                    frame[f"%-raw_{column}"] = 100 + np.arange(64) * 0.1
                dk = FreqaiDataKitchen(config, live=True, pair="BTC/USDT")
                dk.data_path = Path(temp) / "initial"
                dk.model_filename = "cb_btc_initial"
                dk.label_list = ["&-action"]
                dk.training_features_list = [c for c in frame if c.startswith("%")]
                trained = model.train(frame, dk.pair, dk)
                np.testing.assert_allclose(
                    dk.data_dictionary["train_prices"]["open"], frame["%-raw_open"][:48]
                )
                model.dd.pair_dict[dk.pair] = {}
                model.dd.save_data(trained, dk.pair, dk)
                replay = trained.replay_buffer.observations.copy()
                size = trained.replay_buffer.size()
                self.assertGreater(size, 0)
                model.dd.model_dictionary[dk.pair] = trained
                model.dd.meta_data_dictionary.clear()
                direct_clone, _ = model._resolve_deployment_state(dk, dk.pair)
                self.assertEqual(direct_clone.replay_buffer.size(), size)
                np.testing.assert_array_equal(direct_clone.replay_buffer.observations, replay)
                direct_clone.replay_buffer.observations.flat[0] += 42
                np.testing.assert_array_equal(trained.replay_buffer.observations, replay)
                model.dd.model_dictionary.clear()
                restored = model.dd.load_data(dk.pair, dk)
                self.assertEqual(restored.replay_buffer.size(), 0)
                clone, _ = model._resolve_deployment_state(dk, dk.pair)
                self.assertEqual(clone.replay_buffer.size(), size)
                np.testing.assert_array_equal(clone.replay_buffer.observations, replay)
                clone.replay_buffer.observations.flat[0] += 42
                np.testing.assert_array_equal(trained.replay_buffer.observations, replay)
                self.assertEqual(restored.replay_buffer.size(), 0)
                self.assertIs(model.dd.load_data(dk.pair, dk), restored)
                model.dd.model_dictionary.clear()
                (dk.data_path / dk.data["reforcexy_replay"]).unlink()
                inference_model = model.dd.load_data(dk.pair, dk)
                self.assertIsNotNone(inference_model)
                model.continual_learning = False
                self.assertIsNone(model._resolve_deployment_state(dk, dk.pair))
                model.continual_learning = True
                with self.assertRaises(DependencyException):
                    model._resolve_deployment_state(dk, dk.pair)
                params = model.get_model_params()
                trial = create_study(direction="maximize").ask()
                for starts in (64, 50000):
                    with (
                        mock.patch.object(
                            model,
                            "get_optuna_params",
                            return_value={
                                **params,
                                "learning_starts": starts,
                                "buffer_size": 100000,
                            },
                        ),
                        self.assertRaises(TrialPruned),
                    ):
                        model.objective(
                            trial,
                            dk,
                            64,
                            dk.data_dictionary["train_prices"],
                            dk.data_dictionary["test_prices"],
                        )
                with mock.patch.object(model, "get_optuna_params", return_value=params):
                    score = model.objective(
                        trial,
                        dk,
                        64,
                        dk.data_dictionary["train_prices"],
                        dk.data_dictionary["test_prices"],
                    )
                self.assertTrue(np.isfinite(score))

    def test_backtest_rejects_future_archive_but_continues_from_earlier_saved_window(self):
        with tempfile.TemporaryDirectory() as temp:
            config = model_config(temp)
            info = config["freqai"]
            info["continual_learning"] = True
            info["rl_config"]["model_type"] = "DQN"
            info["model_training_parameters"] = {
                "learning_starts": 0,
                "buffer_size": 128,
                "batch_size": 8,
                "train_freq": 4,
                "gradient_steps": 1,
                "device": "cpu",
                "policy_kwargs": {"net_arch": [8]},
            }
            pair = "BTC/USDT"

            def frame(day, feature_offset):
                values = np.arange(64)
                data = pd.DataFrame(
                    {
                        "date": pd.date_range(day, periods=64, freq="5min", tz="UTC"),
                        "%-feature": np.sin(values) + feature_offset,
                        "&-action": np.zeros(64),
                    }
                )
                for column in ("open", "high", "low", "close"):
                    data[f"%-raw_{column}"] = 100.0 + values * 0.1
                return data

            def kitchen(settings, data, *, live):
                dk = FreqaiDataKitchen(settings, live=live, pair=pair)
                timestamp = int((data["date"].iloc[-1] + pd.Timedelta(minutes=5)).timestamp())
                dk.set_paths(pair, timestamp)
                dk.set_new_model_names(pair, timestamp)
                dk.data_path.mkdir(parents=True, exist_ok=True)
                dk.label_list = ["&-action"]
                dk.training_features_list = [column for column in data if column.startswith("%")]
                return dk, timestamp

            source = ReforceXY(config=config)
            source.live = True
            source.can_short = False
            self.addCleanup(source.close_envs)
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
                "/workspace/ReforceXY/user_data/config-template.json"
            ]
            backtest = ReforceXY(config=backtest_config)
            backtest.live = False
            backtest.can_short = False
            self.addCleanup(backtest.close_envs)
            for day, offset, save_model in (
                ("2026-01-01", 0.0, False),
                ("2026-01-02", 10.0, True),
                ("2026-01-03", 20.0, False),
            ):
                training_frame = frame(day, offset)
                dk, timestamp = kitchen(backtest_config, training_frame, live=False)
                self.assertLess(timestamp, future_ts)
                self.assertFalse(backtest.model_exists(dk))
                trained = backtest.train(training_frame, pair, dk)
                transformed = dk.data_dictionary["train_features"]["%-feature"]
                if offset < 20.0:
                    self.assertAlmostEqual(transformed.min(), -1.0)
                    self.assertAlmostEqual(transformed.max(), 1.0)
                else:
                    self.assertGreater(transformed.min(), 2.0)
                backtest.dd.pair_dict[pair]["trained_timestamp"] = timestamp
                if save_model:
                    backtest.dd.save_data(trained, pair, dk)
                else:
                    backtest.dd.save_metadata(dk)

    def test_provenance_does_not_change_other_drawers(self):
        with tempfile.TemporaryDirectory() as temp:
            config = model_config(temp)
            model = ReforceXY(config=config)
            self.addCleanup(model.close_envs)
            drawer = FreqaiDataDrawer(Path(temp), config)
            pair = "BTC/USDT"
            date = pd.Timestamp("2026-01-01", tz="UTC")
            drawer.historic_predictions[pair] = pd.DataFrame(
                {
                    "date_pred": [date],
                    "&-action": [1.0],
                    "do_predict": [1],
                    "close_price": [100.0],
                }
            )
            candles = pd.DataFrame({"date": [date]})
            drawer.set_initial_return_values(pair, pd.DataFrame({"&-action": [99.0]}), candles)
            self.assertNotIn("_freqai_strategies_produced", drawer.historic_predictions[pair])
            self.assertNotIn(
                "_freqai_strategies_produced",
                drawer.attach_return_values_to_return_dataframe(pair, candles),
            )

    def test_native_bootstrap_append_and_restart_provenance(self):
        pair = "BTC/USDT"
        marker = "_freqai_strategies_produced"
        with tempfile.TemporaryDirectory() as temp:
            config = model_config(temp)
            config["freqai"]["fit_live_predictions_candles"] = 3
            model = ReforceXY(config=config)
            model.live = True
            self.addCleanup(model.close_envs)
            dates = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
            strat_df = pd.DataFrame(
                {"date": dates, "high": [101.0] * 4, "low": [99.0] * 4, "close": [100.0] * 4}
            )
            dk = SimpleNamespace(
                data={"extra_returns_per_train": {}}, label_list=["&-action"], unique_class_list=[]
            )
            bootstrap = pd.DataFrame({"&-action": [21.0, 22.0, 23.0, 24.0]})
            model.set_initial_historic_predictions(bootstrap, dk, pair, strat_df)
            model.dd.set_initial_return_values(pair, bootstrap, strat_df)
            self.assertEqual(model.dd.historic_predictions[pair][marker].tolist(), [False] * 4)
            model.fit_live_predictions(dk, pair)
            self.assertEqual(dk.data["labels_mean"]["&-action"], 0.0)
            returned = model.dd.attach_return_values_to_return_dataframe(pair, strat_df)
            self.assertNotIn(marker, returned)

            # Native same-candle append overwrites a bootstrap candle. A rejected
            # prediction is nevertheless produced and must enter the statistics.
            model.dd.append_model_predictions(
                pair, pd.DataFrame({"&-action": [3.0]}), np.array([0]), dk, strat_df
            )
            model.fit_live_predictions(dk, pair)
            self.assertEqual(dk.data["labels_mean"]["&-action"], 3.0)
            self.assertNotIn(
                marker, model.dd.attach_return_values_to_return_dataframe(pair, strat_df)
            )

            future = pd.date_range(dates[-1] + pd.Timedelta(minutes=5), periods=3, freq="5min")
            resumed = pd.concat(
                [
                    strat_df,
                    pd.DataFrame(
                        {
                            "date": future,
                            "high": [101.0] * 3,
                            "low": [99.0] * 3,
                            "close": [100.0] * 3,
                        }
                    ),
                ],
                ignore_index=True,
            )
            model.dd.append_model_predictions(
                pair, pd.DataFrame({"&-action": [9.0]}), np.array([1]), dk, resumed
            )
            history = model.dd.historic_predictions[pair]
            self.assertEqual(history[marker].tolist(), [False] * 3 + [True, False, False, True])
            model.fit_live_predictions(dk, pair)
            self.assertEqual(dk.data["labels_mean"]["&-action"], 6.0)
            self.assertNotIn(
                marker, model.dd.attach_return_values_to_return_dataframe(pair, resumed)
            )
            model.dd.append_model_predictions(
                pair, pd.DataFrame({"&-action": [99.0]}), np.array([2]), dk, resumed
            )
            model.fit_live_predictions(dk, pair)
            self.assertEqual(dk.data["labels_mean"]["&-action"], 3.0)
            model.dd.append_model_predictions(
                pair, pd.DataFrame({"&-action": [9.0]}), np.array([1]), dk, resumed
            )
            self.assertEqual(len(model.dd.historic_predictions[pair]), len(resumed))
            model.fit_live_predictions(dk, pair)
            self.assertEqual(dk.data["labels_mean"]["&-action"], 6.0)

            model.dd.save_historic_predictions_to_disk()
            restored = ReforceXY(config=config)
            restored.live = True
            self.addCleanup(restored.close_envs)
            self.assertTrue(restored.dd.load_historic_predictions_from_disk())
            self.assertEqual(
                restored.dd.historic_predictions[pair][marker].tolist(), history[marker].tolist()
            )
            restored.fit_live_predictions(dk, pair)
            self.assertEqual(dk.data["labels_mean"]["&-action"], 6.0)
            self.assertNotIn(
                marker, restored.dd.attach_return_values_to_return_dataframe(pair, resumed)
            )

    def test_partially_marked_history_restores_provable_observations(self):
        pair = "BTC/USDT"
        marker = "_freqai_strategies_produced"
        with tempfile.TemporaryDirectory() as temp:
            config = model_config(temp)
            config["freqai"]["fit_live_predictions_candles"] = 4
            source = ReforceXY(config=config)
            self.addCleanup(source.close_envs)
            dates = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
            source.dd.historic_predictions[pair] = pd.DataFrame(
                {
                    "date_pred": dates,
                    "&-action": [7.0, 9.0, 99.0, 1000.0],
                    "&-action_mean": [0.0] * 4,
                    "&-action_std": [0.0] * 4,
                    "close_price": [100.0] * 4,
                    "high_price": [101.0] * 4,
                    "low_price": [99.0] * 4,
                    "do_predict": [1, 0, 2, 1],
                    marker: [np.nan, 1.0, np.nan, 0.0],
                }
            )
            source.dd.save_historic_predictions_to_disk()

            restarted = ReforceXY(config=config)
            restarted.live = True
            self.addCleanup(restarted.close_envs)
            dk = SimpleNamespace(
                data={"extra_returns_per_train": {}},
                label_list=["&-action"],
                unique_class_list=[],
                return_dataframe=pd.DataFrame(),
            )
            restarted.predict = lambda frame, kitchen, **kwargs: (
                pd.DataFrame({"&-action": np.ones(len(frame))}),
                np.ones(len(frame), dtype=int),
            )
            restarted.fit_live_predictions(dk, pair)
            self.assertEqual(dk.data["labels_mean"]["&-action"], 8.0)

            candles = pd.DataFrame(
                {"date": dates, "high": [101.0] * 4, "low": [99.0] * 4, "close": [100.0] * 4}
            )
            restarted.build_strategy_return_arrays(candles, dk, pair, 0)
            self.assertEqual(
                restarted.dd.historic_predictions[pair][marker].tolist(), [True, True, False, False]
            )

            next_candle = pd.DataFrame(
                {
                    "date": [dates[-1] + pd.Timedelta(minutes=5)],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.0],
                }
            )
            restarted.dk = SimpleNamespace(check_if_model_expired=lambda timestamp: False)
            restarted.build_strategy_return_arrays(
                pd.concat([candles, next_candle], ignore_index=True), dk, pair, 0
            )
            self.assertEqual(dk.return_dataframe["&-action_mean"].iloc[-1], 8.0)
            self.assertEqual(dk.return_dataframe["&-action_std"].iloc[-1], 1.0)
            self.assertNotIn(marker, dk.return_dataframe)

    def test_duplicate_date_keeps_provable_observation_after_restart(self):
        pair = "BTC/USDT"
        marker = "_freqai_strategies_produced"
        date = pd.Timestamp("2026-01-01", tz="UTC")
        for statuses, markers in (([1, 0], None), ([0, 0], [True, False])):
            with self.subTest(markers=markers), tempfile.TemporaryDirectory() as temp:
                config = model_config(temp)
                config["freqai"]["fit_live_predictions_candles"] = 3
                source = ReforceXY(config=config)
                self.addCleanup(source.close_envs)
                history = pd.DataFrame(
                    {
                        "date_pred": [date, date],
                        "&-action": [7.0, 99.0],
                        "close_price": [100.0, 100.0],
                        "do_predict": statuses,
                    }
                )
                if markers is not None:
                    history[marker] = markers
                source.dd.historic_predictions[pair] = history
                source.dd.save_historic_predictions_to_disk()

                restarted = ReforceXY(config=config)
                restarted.live = True
                self.addCleanup(restarted.close_envs)
                restored = restarted.dd.historic_predictions[pair]
                self.assertEqual(restored["&-action"].tolist(), [7.0])
                self.assertEqual(restored[marker].tolist(), [True])
                dk = SimpleNamespace(data={}, label_list=["&-action"], unique_class_list=[])
                restarted.fit_live_predictions(dk, pair)
                self.assertEqual(dk.data["labels_mean"]["&-action"], 7.0)

    def test_legacy_zero_status_is_ambiguous_but_expired_status_is_excluded(self):
        pair = "BTC/USDT"
        history = pd.DataFrame(
            {
                "date_pred": pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC"),
                "&-action": [99.0, 7.0, 88.0, 77.0],
                "close_price": [100.0] * 4,
                "do_predict": [0, 1, 2, np.nan],
            }
        )
        model = ReforceXY.__new__(ReforceXY)
        model.live = True
        model.freqai_info = {"fit_live_predictions_candles": 4}
        model.dd = SimpleNamespace(historic_predictions={pair: history})
        dk = SimpleNamespace(data={}, label_list=["&-action"], unique_class_list=[])
        model.fit_live_predictions(dk, pair)
        self.assertEqual(dk.data["labels_mean"]["&-action"], 7.0)

    def test_live_action_statistics_resume_persisted_observations(self):
        pair = "BTC/USDT"
        dates = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        history = pd.DataFrame(
            {
                "date_pred": dates,
                "&-action": [1.0, 2.0, 3.0, 99.0],
                "do_predict": [1, 1, 1, 2],
                "close_price": [100.0, 101.0, 102.0, 103.0],
            }
        )
        model = ReforceXY.__new__(ReforceXY)
        model.live = True
        model.freqai_info = {"fit_live_predictions_candles": 4}
        model.dd = SimpleNamespace(
            historic_predictions={pair: history},
            model_return_values={pair: history.tail(1)},
        )
        dk = SimpleNamespace(data={}, label_list=["&-action"], unique_class_list=[])

        model.fit_live_predictions(dk, pair)

        self.assertEqual(dk.data["labels_mean"]["&-action"], 2.0)
        self.assertAlmostEqual(dk.data["labels_std"]["&-action"], np.std([1.0, 2.0, 3.0]))

    def test_frame_validity_and_gap_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            model = ReforceXY(config=model_config(temp))
            self.addCleanup(model.close_envs)
            model.live = True
            model.frame_stacking = 2
            model.get_state_info = lambda pair: (0.5, 0.0, 0)
            policy = RecordingPolicy()
            dk = SimpleNamespace(pair="BTC/USDT", label_list=["&-action"], data_dictionary={})

            def predict(value, minute, valid=1):
                dk.do_predict = np.array([valid])
                dk.data_dictionary["prediction_dates"] = pd.DataFrame(
                    {
                        "date": pd.DatetimeIndex(
                            [pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=minute)]
                        )
                    }
                )
                model.rl_model_predict(pd.DataFrame({"f": [value]}), dk, policy)
                return dk.do_predict[0]

            self.assertEqual(predict(10, 0, 0), 0)
            self.assertEqual(predict(20, 5), 0)
            self.assertEqual(predict(30, 10), 1)
            self.assertEqual(predict(40, 30), 1)
            np.testing.assert_array_equal(policy.observations[-1], [[[0, 40]]])
            predict(50, 30)
            np.testing.assert_array_equal(policy.observations[-1], [[[0, 50]]])
            model.CONV_WIDTH = 3
            model.frame_stacking = 0
            dk.do_predict = np.ones(5, dtype=int)
            dk.data_dictionary["prediction_dates"] = pd.DataFrame(
                {
                    "date": pd.DatetimeIndex(
                        pd.to_datetime(
                            [
                                "2026-01-01 00:00Z",
                                "2026-01-01 00:05Z",
                                "2026-01-01 00:20Z",
                                "2026-01-01 00:25Z",
                                "2026-01-01 00:30Z",
                            ]
                        )
                    )
                }
            )
            model.rl_model_predict(pd.DataFrame({"f": np.arange(5)}), dk, policy)
            np.testing.assert_array_equal(dk.do_predict, [0, 0, 0, 0, 1])
