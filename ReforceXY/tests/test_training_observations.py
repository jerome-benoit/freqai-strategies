"""Regression contracts for learning eligibility, replay and temporal observations."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
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
                model.dd.model_dictionary.clear()
                model.dd.meta_data_dictionary.clear()
                restored = model.dd.load_data(dk.pair, dk)
                self.assertEqual(restored.replay_buffer.size(), size)
                np.testing.assert_array_equal(restored.replay_buffer.observations, replay)
                clone, _ = model._resolve_deployment_state(dk, dk.pair)
                self.assertEqual(clone.replay_buffer.size(), size)
                self.assertIsNot(clone.replay_buffer, restored.replay_buffer)
                clone.replay_buffer.observations.flat[0] += 123.0
                self.assertNotEqual(
                    clone.replay_buffer.observations.flat[0],
                    restored.replay_buffer.observations.flat[0],
                )
                np.testing.assert_array_equal(restored.replay_buffer.observations, replay)
                self.assertIs(model.dd.load_data(dk.pair, dk), restored)
                model.dd.model_dictionary.clear()
                (dk.data_path / dk.data["reforcexy_replay"]).unlink()
                for _ in range(2):
                    with self.assertRaises(FileNotFoundError):
                        model.dd.load_data(dk.pair, dk)
                    self.assertNotIn(dk.pair, model.dd.model_dictionary)
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
