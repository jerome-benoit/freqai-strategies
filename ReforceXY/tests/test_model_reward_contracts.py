"""Reward costs, numerical boundaries and the native model-selection objective."""

import math
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
from qa_support import PAIR, QaTestCase, model_config
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from ReforceXY.user_data.freqaimodels.ReforceXY import (
    Actions,
    MyRLEnv,
    Positions,
    ReforceXY,
)


class _TradeOrWaitPolicy:
    """Choose an observable long round trip, or remain in cash."""

    def __init__(self, trade):
        self.trade = trade

    def predict(self, observation, state=None, **kwargs):
        positions = observation[:, -1, -2]
        actions = (
            np.where(
                positions == Positions.Neutral.value,
                Actions.Long_enter.value,
                Actions.Long_exit.value,
            )
            if self.trade
            else np.full(len(observation), Actions.Neutral.value)
        )
        return actions, state


class RewardContractsTest(QaTestCase):
    def env(self, *, opens=None, **parameters):
        prices = pd.DataFrame({"open": [100.0] * 16 if opens is None else opens})
        env = MyRLEnv(
            df=prices.copy(),
            prices=prices,
            df_raw=prices.copy(),
            window_size=1,
            reward_kwargs={"rr": 2.0, "profit_aim": 0.03},
            fee=0.0,
            can_short=True,
            config={
                "stake_amount": "unlimited",
                "freqai": {
                    "rl_config": {
                        "add_state_info": True,
                        "max_training_drawdown_pct": 0.99,
                        "model_reward_parameters": {
                            "max_trade_duration_candles": 1,
                            "max_idle_duration_candles": 1,
                            "hold_potential_enabled": False,
                            **parameters,
                        },
                    }
                },
            },
            live=True,
        )
        self.addCleanup(env.close)
        env.reset(seed=42)
        return env

    def test_invalid_noop_keeps_time_cost_for_every_position(self):
        for entry, invalid in (
            (None, Actions.Long_exit),
            (Actions.Long_enter, Actions.Long_enter),
            (Actions.Short_enter, Actions.Short_enter),
        ):
            for penalty in (0.0, -2.0):
                with self.subTest(entry=entry, penalty=penalty):
                    valid = self.env(invalid_action=penalty)
                    invalid_env = self.env(invalid_action=penalty)
                    for env in (valid, invalid_env):
                        if entry is not None:
                            env.step(entry.value)
                        for _ in range(3):
                            env.step(Actions.Neutral.value)
                    obs, reward, _, _, info = valid.step(Actions.Neutral.value)
                    bad_obs, bad_reward, _, _, bad_info = invalid_env.step(invalid.value)
                    np.testing.assert_array_equal(bad_obs, obs)
                    self.assertLess(reward, 0.0)
                    self.assertAlmostEqual(bad_reward, reward + penalty)
                    self.assertEqual(bad_info["reward_idle"], info["reward_idle"])
                    self.assertEqual(bad_info["reward_hold"], info["reward_hold"])
                    self.assertEqual(invalid_env._total_profit, valid._total_profit)
                    self.assertEqual(len(invalid_env.trade_history), len(valid.trade_history))

    def test_valid_entry_and_exit_do_not_pay_noop_costs(self):
        env = self.env()
        for _ in range(3):
            env.step(Actions.Neutral.value)
        _, reward, _, _, info = env.step(Actions.Long_enter.value)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["reward_idle"], 0.0)
        for _ in range(3):
            env.step(Actions.Neutral.value)
        _, reward, _, _, info = env.step(Actions.Long_exit.value)
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["reward_hold"], 0.0)

    def test_null_idle_scale_derives_trade_scale_but_zero_remains_explicit(self):
        for idle_scale, denominator in ((None, 8), (0, 1)):
            with self.subTest(idle_scale=idle_scale):
                env = self.env(max_trade_duration_candles=2, max_idle_duration_candles=idle_scale)
                for _ in range(3):
                    env.step(Actions.Neutral.value)
                _, reward, _, _, _ = env.step(Actions.Neutral.value)
                self.assertAlmostEqual(reward, -1.5 * (3 / denominator) ** 1.025)

    def test_power_exit_preserves_python_and_numpy_tau(self):
        for tau in (0.25, np.float32(0.25), np.longdouble(0.25), np.int64(1)):
            with self.subTest(tau_type=type(tau)):
                env = self.env(
                    opens=[100.0, 100.0, 105.0, 105.0],
                    exit_attenuation_mode="power",
                    exit_plateau=False,
                    exit_power_tau=tau,
                )
                env.step(Actions.Long_enter.value)
                _, reward, terminated, _, _ = env.step(Actions.Long_exit.value)
                self.assertTrue(terminated)
                # At duration ratio one, K(1)=tau; the fee-free exit has base reward 7.5.
                self.assertAlmostEqual(reward, 7.5 * float(tau))

    def test_power_underflow_does_not_switch_to_a_linear_kernel(self):
        env = self.env()
        params = {"exit_attenuation_mode": "power", "exit_plateau": False, "exit_power_tau": 1e-6}
        ratio = 1e16
        expected = math.exp(math.log(1e-6) / math.log(2) * math.log1p(ratio))
        actual = env._compute_time_attenuation_coefficient(ratio, params)
        self.assertGreater(actual, 0.0)
        # Subnormal rounding needs relative tolerance; an absolute epsilon would hide the defect.
        self.assertTrue(math.isclose(actual, expected, rel_tol=1e-4, abs_tol=0.0))
        self.assertEqual(env._compute_time_attenuation_coefficient(1e18, params), 0.0)

    def test_native_evaluation_preserves_base_ranking_without_losing_training_shaping(self):
        for ratio in (0.4, 1e16):
            for stacking in (0, 2):
                with (
                    self.subTest(ratio=ratio, stacking=stacking),
                    tempfile.TemporaryDirectory() as temp,
                ):
                    config = model_config(
                        temp,
                        freqai={
                            "model_training_parameters": {"gamma": 0.8},
                            "rl_config": {
                                "add_state_info": True,
                                "frame_stacking": stacking,
                                "model_reward_parameters": {
                                    "profit_aim": 0.025,
                                    "hold_potential_enabled": True,
                                    "hold_potential_ratio": ratio,
                                    "max_trade_duration_candles": 1,
                                    "hold_potential_transform_pnl": "clip",
                                    "hold_potential_transform_duration": "clip",
                                    "idle_penalty_ratio": 0.0,
                                },
                            },
                        },
                    )
                    model = ReforceXY(config=config)
                    model.live = True
                    self.addCleanup(model.close_envs)
                    prices = pd.DataFrame({"open": [100.0, 100.0, 105.0, 105.0]})
                    # Use the real factory and VecMonitor, not a replacement evaluation implementation.
                    model.set_train_and_eval_environments(
                        {"train_features": prices.copy(), "test_features": prices.copy()},
                        prices,
                        prices,
                        SimpleNamespace(pair=PAIR),
                    )
                    model.train_env.set_attr("fee", 0.0)
                    model.eval_env.set_attr("fee", 0.0)
                    trade_score, _ = evaluate_policy(
                        _TradeOrWaitPolicy(True), model.eval_env, n_eval_episodes=1, warn=False
                    )
                    cash_score, _ = evaluate_policy(
                        _TradeOrWaitPolicy(False), model.eval_env, n_eval_episodes=1, warn=False
                    )
                    self.assertAlmostEqual(trade_score, 7.5)
                    self.assertEqual(cash_score, 0.0)
                    model.train_env.reset()
                    _, rewards, _, _ = model.train_env.step([Actions.Long_enter.value])
                    np.testing.assert_allclose(rewards, [0.8 * 100 * ratio], rtol=1e-6)
                    model.close_envs()

    def test_evaluation_retains_entry_and_terminal_exit_additives(self):
        with tempfile.TemporaryDirectory() as temp:
            fee = 0.001
            config = model_config(
                temp,
                freqai={
                    "rl_config": {
                        "add_state_info": True,
                        "model_reward_parameters": {
                            "hold_potential_enabled": True,
                            "exit_potential_mode": "non_canonical",
                            "entry_additive_enabled": True,
                            "exit_additive_enabled": True,
                            "max_trade_duration_candles": 1,
                        },
                    }
                },
            )
            model = ReforceXY(config=config)
            model.live = True
            self.addCleanup(model.close_envs)
            prices = pd.DataFrame({"open": [100.0] * 3})
            model.set_train_and_eval_environments(
                {"train_features": prices.copy(), "test_features": prices.copy()},
                prices,
                prices,
                SimpleNamespace(pair=PAIR),
            )
            model.eval_env.set_attr("fee", fee)
            score, _ = evaluate_policy(
                _TradeOrWaitPolicy(True), model.eval_env, n_eval_episodes=1, warn=False
            )
            pnl = 1 / (1 + fee) ** 2 - 1
            entry_additive = 6.25 * 0.5 * math.tanh(pnl / 0.06)
            terminal_additive = 6.25 * 0.5 * (math.tanh(pnl / 0.06) - math.tanh(1.0))
            self.assertAlmostEqual(score, 100 * pnl + entry_additive + terminal_additive, places=6)

    def test_monitored_scores_retain_small_returns_between_large_exits_after_reset(self):
        for multiprocessing in (False, True):
            with (
                self.subTest(multiprocessing=multiprocessing),
                tempfile.TemporaryDirectory() as temp,
            ):
                base = 2**25
                small_mark = 150.0 * (1.0 + 1.0 / base)
                config = model_config(
                    temp,
                    freqai={
                        "rl_config": {
                            "n_eval_envs": 2 if multiprocessing else 1,
                            "eval_multiprocessing": multiprocessing,
                            "model_reward_parameters": {
                                "base_factor": base,
                                "win_reward_factor": 0.0,
                                "efficiency_weight": 0.0,
                                "idle_penalty_ratio": 0.0,
                            },
                        }
                    },
                )
                model = ReforceXY(config=config)
                model.live = True
                self.addCleanup(model.close_envs)
                prices = pd.DataFrame(
                    {
                        "open": [
                            100.0,
                            100.0,
                            150.0,
                            150.0,
                            small_mark,
                            small_mark,
                            1.5 * small_mark,
                        ]
                    }
                )
                model.set_train_and_eval_environments(
                    {"train_features": prices.copy(), "test_features": prices.copy()},
                    prices,
                    prices,
                    SimpleNamespace(pair=PAIR),
                )
                model.eval_env.set_attr("fee", 0.0)
                for _ in range(2):
                    model.eval_env.reset()
                    for action in (
                        Actions.Long_enter,
                        Actions.Long_exit,
                        Actions.Long_enter,
                        Actions.Long_exit,
                        Actions.Short_enter,
                    ):
                        _, _, done, infos = model.eval_env.step(
                            [action.value] * model.eval_env.num_envs
                        )
                    self.assertTrue(done.all())
                    # Three exit rewards are 2**24, 1 and -2**24; equity remains positive.
                    for info in infos:
                        self.assertEqual(info["episode"]["r"], 1.0)
                        self.assertGreater(info["total_profit"], 0.0)
                model.close_envs()
