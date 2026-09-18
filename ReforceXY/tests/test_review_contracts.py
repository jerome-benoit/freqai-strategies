"""Regressions for live observations, HPO options and historic prediction alignment."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from freqtrade.enums import RunMode
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from sb3_contrib import MaskablePPO

from ReforceXY.user_data.freqaimodels.ReforceXY import (
    MyRLEnv,
    ReforceXY,
    convert_optuna_params_to_model_params,
    deepmerge,
)


class RecordingPolicy:
    def __init__(self):
        self.observations = []
        self.masks = []

    def predict(self, observation, **kwargs):
        self.observations.append(observation.copy())
        self.masks.append(kwargs.get("action_masks"))
        return np.array([0]), None


def model_config(path):
    return {
        "user_data_dir": Path(path),
        "timeframe": "5m",
        "stake_amount": "unlimited",
        "runmode": RunMode.DRY_RUN,
        "exchange": {"pair_whitelist": ["BTC/USDT"]},
        "freqai": {
            "enabled": True,
            "identifier": "contract-test",
            "train_period_days": 1,
            "backtest_period_days": 1,
            "conv_width": 1,
            "activate_tensorboard": False,
            "feature_parameters": {
                "include_timeframes": ["5m"],
                "include_corr_pairlist": [],
                "label_period_candles": 1,
                "principal_component_analysis": False,
                "noise_standard_deviation": 0,
                "buffer_train_data_candles": 0,
                "shuffle_after_split": False,
            },
            "data_split_parameters": {"test_size": 0.25, "shuffle": False},
            "model_training_parameters": {
                "n_steps": 8,
                "batch_size": 8,
                "n_epochs": 1,
                "device": "cpu",
                "policy_kwargs": {"net_arch": [8]},
            },
            "rl_config": {
                "model_type": "MaskablePPO",
                "policy_type": "MlpPolicy",
                "cpu_count": 1,
                "drop_ohlc_from_features": False,
                "model_reward_parameters": {"rr": 2.0, "profit_aim": 0.03},
                "train_cycles": 1,
                "n_envs": 1,
                "n_eval_envs": 1,
                "n_eval_steps": 16,
                "n_eval_episodes": 1,
                "check_envs": False,
                "add_state_info": False,
            },
        },
    }


class ReviewContractsTest(unittest.TestCase):
    def model(self, *, hold=False, hpo=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        config = model_config(temp.name)
        config["freqai"]["rl_config"]["model_reward_parameters"]["hold_potential_enabled"] = hold
        config["freqai"]["continual_learning"] = hpo
        config["freqai"]["rl_config_optuna"] = {"enabled": hpo}
        model = ReforceXY(config=config)
        model.live = True
        model.can_short = True
        model.get_state_info = lambda pair: (1.0, 0.05, 12)
        self.addCleanup(model.close_envs)
        return model

    def test_training_preserves_raw_prices_and_returns_best_checkpoint(self):
        for drop in (False, True):
            with self.subTest(drop_ohlc_from_features=drop), tempfile.TemporaryDirectory() as temp:
                config = model_config(temp)
                config["freqai"]["rl_config"]["drop_ohlc_from_features"] = drop
                model = ReforceXY(config=config)
                model.live = True
                model.can_short = False
                model.get_state_info = lambda pair: (0.5, 0.0, 0)
                self.addCleanup(model.close_envs)
                dk = FreqaiDataKitchen(config, live=True, pair="BTC/USDT")
                dk.data_path = Path(temp) / "fit"
                dk.data_path.mkdir()
                dk.model_filename = "cb_btc_123"
                dk.label_list = ["&-action"]
                frame = pd.DataFrame(
                    {
                        "date": pd.date_range("2026-01-01", periods=64, freq="5min", tz="UTC"),
                        "%-feature": np.sin(np.arange(64)),
                        "&-action": np.zeros(64),
                    }
                )
                for column in ("open", "high", "low", "close"):
                    frame[f"%-raw_{column}"] = 100 + np.arange(64) * 0.1
                dk.training_features_list = [column for column in frame if column.startswith("%")]
                trained = model.train(frame, dk.pair, dk)
                checkpoint = MaskablePPO.load(dk.data_path / "best_model.zip")
                for name, value in trained.policy.state_dict().items():
                    np.testing.assert_array_equal(
                        value.cpu().numpy(), checkpoint.policy.state_dict()[name].cpu().numpy()
                    )
                prediction = model.rl_model_predict(
                    dk.data_dictionary["train_features"].tail(1), dk, trained
                )
                self.assertIn(int(prediction.iloc[0, 0]), (0, 1))
                train_env, eval_env = model._get_train_and_eval_environments(
                    dk, model_params={"gamma": 0.91}
                )
                try:
                    np.testing.assert_allclose(
                        train_env.get_attr("prices")[0]["open"], frame["%-raw_open"].iloc[:48]
                    )
                    self.assertEqual(train_env.get_attr("_potential_gamma"), [0.91])
                finally:
                    train_env.close()
                    eval_env.close()

    def test_live_stacking_preserves_frames_and_resets_for_new_model(self):
        model = self.model()
        model.frame_stacking = 2
        policy = RecordingPolicy()
        dk = SimpleNamespace(pair="BTC/USDT", label_list=["&-action"])
        model.rl_model_predict(pd.DataFrame({"f": [10.0]}), dk, policy)
        model.rl_model_predict(pd.DataFrame({"f": [20.0]}), dk, policy)
        np.testing.assert_array_equal(policy.observations[-1], [[[10.0, 20.0]]])
        replacement = RecordingPolicy()
        model.rl_model_predict(pd.DataFrame({"f": [30.0]}), dk, replacement)
        np.testing.assert_array_equal(replacement.observations[-1], [[[0.0, 30.0]]])
        model.rl_model_predict(pd.DataFrame({"f": [40.0, 50.0]}), dk, replacement)
        np.testing.assert_array_equal(replacement.observations[-2], [[[0.0, 40.0]]])
        np.testing.assert_array_equal(replacement.observations[-1], [[[40.0, 50.0]]])
        dk.pair = "ETH/USDT"
        model.rl_model_predict(pd.DataFrame({"f": [60.0]}), dk, replacement)
        np.testing.assert_array_equal(replacement.observations[-1], [[[0.0, 60.0]]])

    def test_live_mask_uses_open_position_without_state_features(self):
        model = self.model()
        policy = RecordingPolicy()
        model.rl_model_predict(
            pd.DataFrame({"f": [1.0]}),
            SimpleNamespace(pair="BTC/USDT", label_list=["&-action"]),
            policy,
        )
        np.testing.assert_array_equal(policy.masks[-1], [True, False, True, False, False])

    def test_hold_potential_has_matching_training_and_inference_shapes(self):
        model = self.model(hold=True)
        features = pd.DataFrame({"f": np.arange(16, dtype=float)})
        prices = pd.DataFrame({"open": np.full(16, 100.0)})
        env = MyRLEnv(df=features, prices=prices, **model.pack_env_dict("BTC/USDT"))
        self.addCleanup(env.close)
        policy = MaskablePPO("MlpPolicy", env, n_steps=8, batch_size=8, device="cpu")
        prediction = model.rl_model_predict(
            features.tail(1), SimpleNamespace(pair="BTC/USDT", label_list=["&-action"]), policy
        )
        self.assertIn(int(prediction.iloc[0, 0]), (0, 2))
        model.live = False
        with self.assertRaises(ValueError):
            model.pack_env_dict("BTC/USDT")

    def test_hpo_does_not_reuse_incompatible_continual_model(self):
        model = self.model(hpo=True)
        old = object()
        model.dd.model_dictionary["BTC/USDT"] = old
        self.assertIsNone(model.get_init_model("BTC/USDT"))
        info = model.pack_env_dict("BTC/USDT", {"gamma": 0.999})
        self.assertEqual(
            info["config"]["freqai"]["rl_config"]["model_reward_parameters"]["potential_gamma"],
            0.999,
        )
        self.assertNotIn("potential_gamma", model.reward_params)

    def test_null_target_kl_overrides_user_value(self):
        params = {
            "learning_rate": 0.0003,
            "clip_range": 0.2,
            "n_steps": 8,
            "batch_size": 8,
            "gamma": 0.95,
            "ent_coef": 0.0,
            "n_epochs": 1,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "target_kl": None,
        }
        effective = deepmerge(
            {"target_kl": 0.03}, convert_optuna_params_to_model_params("PPO", params)
        )
        self.assertIsNone(effective["target_kl"])
        del params["target_kl"]
        effective = deepmerge(
            {"target_kl": 0.03}, convert_optuna_params_to_model_params("PPO", params)
        )
        self.assertEqual(effective["target_kl"], 0.03)

    def test_historic_hole_does_not_shift_actions(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        drawer = object.__new__(FreqaiDataDrawer)
        drawer.historic_predictions = {
            "BTC/USDT": pd.DataFrame(
                {"date_pred": dates[[0, 2]], "close_price": [100.0, 102.0], "&-action": [1, 2]}
            )
        }
        drawer.model_return_values = {}
        frame = pd.DataFrame(
            {
                "date": dates,
                "close": [100.0, 101.0, 102.0],
                "high": [100.0, 101.0, 102.0],
                "low": [100.0, 101.0, 102.0],
            }
        )
        drawer.set_initial_return_values("BTC/USDT", pd.DataFrame({"&-action": [0, 0, 0]}), frame)
        result = drawer.attach_return_values_to_return_dataframe("BTC/USDT", frame)
        self.assertEqual(result["&-action"].iloc[0], 1)
        self.assertTrue(pd.isna(result["&-action"].iloc[1]))
        self.assertEqual(result["&-action"].iloc[2], 2)
