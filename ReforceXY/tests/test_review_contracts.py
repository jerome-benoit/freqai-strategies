"""Regressions for live observations, HPO options and historic prediction alignment."""

import tempfile
import unittest
from datetime import datetime as dt, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
                from sb3_contrib.common.maskable.evaluation import evaluate_policy

                evaluated_updates = []

                def evaluate_final(policy, environment, updates=evaluated_updates, **kwargs):
                    updates.append(policy._n_updates)
                    return evaluate_policy(policy, environment, **kwargs)

                with mock.patch(
                    "ReforceXY.user_data.freqaimodels.ReforceXY.evaluate_policy",
                    side_effect=evaluate_final,
                ):
                    trained = model.train(frame, dk.pair, dk)
                self.assertEqual(len(evaluated_updates), 1)
                self.assertGreater(evaluated_updates[0], 0)
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
        dates = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
        drawer = object.__new__(FreqaiDataDrawer)
        drawer.historic_predictions = {
            "BTC/USDT": pd.DataFrame(
                {"date_pred": dates[[0, 3]], "close_price": [100.0, 103.0], "&-action": [1, 2]}
            )
        }
        drawer.model_return_values = {"BTC/USDT": drawer.historic_predictions["BTC/USDT"].copy()}
        frame = pd.DataFrame(
            {
                "date": dates,
                "close": [100.0, 101.0, 102.0, 103.0],
                "high": [100.0, 101.0, 102.0, 103.0],
                "low": [100.0, 101.0, 102.0, 103.0],
            }
        )
        result = drawer.attach_return_values_to_return_dataframe("BTC/USDT", frame)
        self.assertEqual(result["&-action"].iloc[0], 1)
        self.assertTrue(pd.isna(result["&-action"].iloc[1]))
        self.assertTrue(pd.isna(result["&-action"].iloc[2]))
        self.assertEqual(result["&-action"].iloc[3], 2)
        pd.testing.assert_series_equal(result["date"], frame["date"])

    def test_historic_initialization_does_not_shift_actions(self):
        dates = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
        drawer = object.__new__(FreqaiDataDrawer)
        drawer.historic_predictions = {
            "BTC/USDT": pd.DataFrame(
                {"date_pred": dates[[0, 2]], "close_price": [100.0, 102.0], "&-action": [1, 2]}
            )
        }
        drawer.model_return_values = {}
        frame = pd.DataFrame({"date": dates, "close": [100.0, 101.0, 102.0]})
        drawer.set_initial_return_values("BTC/USDT", pd.DataFrame({"&-action": [0, 0, 0]}), frame)
        result = drawer.attach_return_values_to_return_dataframe("BTC/USDT", frame)
        self.assertEqual(result["&-action"].iloc[0], 1)
        self.assertTrue(pd.isna(result["&-action"].iloc[1]))
        self.assertEqual(result["&-action"].iloc[2], 2)

    def test_rejected_prediction_does_not_advance_virtual_position(self):
        model = self.model()
        model.live = False
        model.CONV_WIDTH = 2

        class EntryPolicy(RecordingPolicy):
            def predict(self, observation, **kwargs):
                super().predict(observation, **kwargs)
                return np.array([1 if kwargs["action_masks"][1] else 0]), None

        policy = EntryPolicy()
        dk = SimpleNamespace(
            pair="BTC/USDT", label_list=["&-action"], do_predict=np.array([1, 0, 1, 1])
        )
        prediction = model.rl_model_predict(
            pd.DataFrame({"f": [10.0, 20.0, 30.0, 40.0]}, index=[11, 22, 33, 44]), dk, policy
        )
        np.testing.assert_array_equal(prediction["&-action"].iloc[1:], [1, 1, 0])
        self.assertTrue(policy.masks[1][1])
        self.assertFalse(policy.masks[2][1])
        np.testing.assert_array_equal(policy.observations[1], [[[20.0], [30.0]]])

    def test_execution_and_potential_match_returned_observation(self):
        model = self.model(hold=True)
        model.CONV_WIDTH = 2
        features = pd.DataFrame({"f": np.arange(7, dtype=float) + 10})
        prices = pd.DataFrame({"open": [100.0, 100.0, 100.0, 110.0, 99.0, 105.0, 106.0]})
        env = MyRLEnv(df=features, prices=prices, **model.pack_env_dict("BTC/USDT"))
        env.fee = 0.0
        self.addCleanup(env.close)
        observation, _ = env.reset()
        np.testing.assert_array_equal(observation[:, 0], [10.0, 11.0])
        observation, _, done, _, _ = env.step(1)
        self.assertFalse(done)
        self.assertEqual(env.trade_history[-1]["tick"], 2)
        self.assertEqual(env.trade_history[-1]["price"], 100.0)
        np.testing.assert_array_equal(observation[:, 0], [11.0, 12.0])
        self.assertAlmostEqual(float(observation[-1, 1]), 0.1)
        self.assertEqual(float(observation[-1, 3]), 1.0)
        expected_potential = env._compute_hold_potential(
            env._position,
            env.get_unrealized_profit(),
            env._pnl_target,
            env.get_trade_duration() / max(1, env.max_trade_duration_candles),
            env._hold_potential_ratio * float(model.reward_params.get("base_factor", 100)),
        )
        self.assertAlmostEqual(env._last_next_potential, expected_potential)
        self.assertAlmostEqual(env._last_reward_shaping, env._potential_gamma * expected_potential)
        env.step(2)
        self.assertEqual(env.trade_history[-1]["tick"], 3)
        self.assertEqual(env.trade_history[-1]["price"], 110.0)
        self.assertAlmostEqual(env._last_reward_shaping, -expected_potential)
        env.step(1)
        _, _, done, _, _ = env.step(0)
        self.assertTrue(done)
        self.assertEqual(env._current_tick, 6)
        self.assertEqual(env._last_next_potential, 0.0)

    def test_state_info_normalizes_leveraged_profit_ratio(self):
        trade = SimpleNamespace(
            pair="BTC/USDT",
            is_short=False,
            leverage=2.0,
            open_date_utc=dt.now(timezone.utc),
            calc_profit_ratio=lambda rate: 0.04,
        )

        exchange = SimpleNamespace(get_rate=lambda *args, **kwargs: 102.0)
        other_pair = SimpleNamespace(pair="ETH/USDT", is_short=False, leverage=1.0)

        model = self.model()
        # The shared fixture stubs get_state_info; the class override is the contract here.
        del model.get_state_info
        model.data_provider = SimpleNamespace(_exchange=exchange)
        with mock.patch(
            "ReforceXY.user_data.freqaimodels.ReforceXY.Trade.get_trades_proxy",
            return_value=[trade, other_pair],
        ) as trades:
            side, profit, duration = model.get_state_info("BTC/USDT")
        trades.assert_called_once_with(is_open=True)
        self.assertEqual(side, 1.0)
        # calc_profit_ratio includes leverage (2x): 4% / leverage 2 -> normalized 2%.
        self.assertAlmostEqual(profit, 0.02, places=10)
        self.assertEqual(duration, 0)

    def test_negative_efficiency_coefficient_is_clamped(self):
        features = pd.DataFrame({"f": np.zeros(6)})
        prices = pd.DataFrame({"open": [100.0, 100.0, 100.0, 90.0, 98.0, 100.0]})
        env = MyRLEnv(
            df=features,
            prices=prices,
            df_raw=features.copy(),
            window_size=1,
            reward_kwargs={"rr": 2.0, "profit_aim": 0.03},
            fee=0.0,
            can_short=False,
            config={
                "stake_amount": "unlimited",
                "freqai": {
                    "rl_config": {
                        "add_state_info": False,
                        "max_training_drawdown_pct": 0.99,
                        "model_reward_parameters": {
                            "efficiency_weight": 2.0,
                            "efficiency_center": 0.0,
                        },
                    }
                },
            },
            live=True,
        )
        self.addCleanup(env.close)
        env.reset()
        env.step(1)
        env.step(0)
        env.step(0)
        # weight=2, center=0 with a partially recovered loss: raw coefficient -0.6.
        self.assertAlmostEqual(
            env._compute_efficiency_coefficient(
                -0.02, {"efficiency_weight": 2.0, "efficiency_center": 0.0}
            ),
            0.0,
        )
        exit_info = env.step(2)[-1]
        self.assertAlmostEqual(exit_info["reward_exit"], 0.0)
