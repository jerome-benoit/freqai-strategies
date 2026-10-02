"""PBRS transitions, boundary identities and discount resolution in the real RL runtime."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from qa_support import QaTestCase, model_config
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from ReforceXY.user_data.freqaimodels.ReforceXY import Actions, MyRLEnv, Positions, ReforceXY


class PbrsTransitionsTest(QaTestCase):
    def env(self, *, short=False, fee=0.0, **parameters):
        prices = pd.DataFrame({"open": [100.0, 100.0] + [90.0 if short else 110.0] * 10})
        env = MyRLEnv(
            df=prices.copy(),
            prices=prices,
            df_raw=prices.copy(),
            window_size=1,
            reward_kwargs={"rr": 2.0, "profit_aim": 0.03},
            fee=fee,
            can_short=True,
            config={
                "stake_amount": "unlimited",
                "freqai": {
                    "rl_config": {
                        "add_state_info": True,
                        "max_training_drawdown_pct": 0.99,
                        "model_reward_parameters": {
                            "potential_gamma": 0.8,
                            "hold_potential_enabled": True,
                            "entry_additive_enabled": False,
                            "exit_additive_enabled": False,
                            **parameters,
                        },
                    }
                },
            },
            live=True,
        )
        self.addCleanup(env.close)
        env.reset()
        return env

    def test_retained_potential_shapes_every_nonterminal_neutral_step(self):
        for mode in ("retain_previous", "progressive_release", "spike_cancel"):
            with self.subTest(mode=mode):
                env = self.env(exit_potential_mode=mode, exit_potential_decay=0.25)
                for action in (Actions.Long_enter, Actions.Neutral, Actions.Long_exit):
                    _, _, done, truncated, _ = env.step(action.value)
                    self.assertFalse(done or truncated)
                potential = env._last_potential
                self.assertNotEqual(potential, 0.0)
                for _ in range(2):
                    _, _, done, truncated, _ = env.step(Actions.Neutral.value)
                    self.assertFalse(done or truncated)
                    self.assertEqual(env._last_prev_potential, potential)
                    self.assertEqual(env._last_next_potential, potential)
                    self.assertAlmostEqual(env._last_reward_shaping, -0.2 * potential, places=12)

    def test_complete_episodes_telescope_and_change_the_actual_reward_by_the_shaping(self):
        for mode in ReforceXY._EXIT_POTENTIAL_MODES:
            for short in (False, True):
                with self.subTest(mode=mode, short=short):
                    shaped = self.env(short=short, exit_potential_mode=mode)
                    base = self.env(
                        short=short, exit_potential_mode=mode, hold_potential_enabled=False
                    )
                    actions = [
                        Actions.Short_enter if short else Actions.Long_enter,
                        Actions.Neutral,
                        Actions.Short_exit if short else Actions.Long_exit,
                    ] + [Actions.Neutral] * 8
                    previous = 0.0
                    discounted = 0.0
                    for tick, action in enumerate(actions):
                        _, reward, done, truncated, _ = shaped.step(action.value)
                        _, base_reward, base_done, base_truncated, _ = base.step(action.value)
                        self.assertEqual((done, truncated), (base_done, base_truncated))
                        self.assertAlmostEqual(shaped._last_prev_potential, previous, places=12)
                        expected = 0.8 * shaped._last_next_potential - previous
                        self.assertAlmostEqual(shaped._last_reward_shaping, expected, places=12)
                        self.assertAlmostEqual(reward - base_reward, expected, places=12)
                        discounted += 0.8**tick * shaped._last_reward_shaping
                        previous = shaped._last_next_potential
                        if done or truncated:
                            break
                    self.assertTrue(done or truncated)
                    self.assertEqual(previous, 0.0)
                    self.assertAlmostEqual(discounted, 0.0, places=12)
                    history = shaped.get_env_history()
                    np.testing.assert_allclose(
                        history["reward_shaping"],
                        0.8 * history["next_potential"] - history["prev_potential"],
                        atol=1e-12,
                    )

    def test_zero_exit_modes_without_additives_have_identical_observations_and_rewards(self):
        canonical = self.env(exit_potential_mode="canonical")
        optional = self.env(exit_potential_mode="non_canonical")
        for action in [Actions.Long_enter, Actions.Neutral, Actions.Long_exit] + [
            Actions.Neutral
        ] * 8:
            observation, reward, done, truncated, info = canonical.step(action.value)
            other_observation, other_reward, other_done, other_truncated, other_info = (
                optional.step(action.value)
            )
            np.testing.assert_array_equal(observation, other_observation)
            self.assertEqual(reward, other_reward)
            self.assertEqual((done, truncated), (other_done, other_truncated))
            self.assertTrue(info["pbrs_invariant"])
            self.assertTrue(other_info["pbrs_invariant"])
            if done or truncated:
                break

    def test_canonical_disables_requested_additives_but_noncanonical_applies_them_at_fills(self):
        for mode in ("canonical", "non_canonical"):
            with self.subTest(mode=mode):
                env = self.env(
                    fee=0.0015,
                    exit_potential_mode=mode,
                    entry_additive_enabled=True,
                    exit_additive_enabled=True,
                )
                base = self.env(fee=0.0015, exit_potential_mode=mode)
                for action in (
                    Actions.Long_enter,
                    Actions.Neutral,
                    Actions.Long_exit,
                    Actions.Neutral,
                ):
                    _, reward, _, _, info = env.step(action.value)
                    _, base_reward, _, _, _ = base.step(action.value)
                    additive = env._last_entry_additive + env._last_exit_additive
                    self.assertAlmostEqual(reward - base_reward, additive, places=12)
                    if mode == "canonical" or action == Actions.Neutral:
                        self.assertEqual(additive, 0.0)
                    else:
                        self.assertNotEqual(additive, 0.0)
                    self.assertEqual(info["pbrs_invariant"], mode == "canonical")

    def test_tiny_terminal_potentials_reconcile_reward_history_and_discounted_sum(self):
        for mode, explicit_exit in (
            ("canonical", False),
            ("retain_previous", False),
            ("retain_previous", True),
        ):
            with self.subTest(mode=mode, explicit_exit=explicit_exit):
                shaped = self.env(fee=0.0015, hold_potential_ratio=1e-10, exit_potential_mode=mode)
                base = self.env(fee=0.0015, hold_potential_enabled=False, exit_potential_mode=mode)
                shaping = []
                for tick in range(20):
                    action = Actions.Long_enter if tick == 0 else Actions.Neutral
                    if explicit_exit and tick == 2:
                        action = Actions.Long_exit
                    _, reward, done, truncated, _ = shaped.step(action.value)
                    _, base_reward, base_done, base_truncated, _ = base.step(action.value)
                    self.assertEqual((done, truncated), (base_done, base_truncated))
                    expected = 0.8 * shaped._last_next_potential - shaped._last_prev_potential
                    self.assertEqual(shaped._last_reward_shaping, expected)
                    self.assertAlmostEqual(reward - base_reward, expected, places=14)
                    shaping.append(expected)
                    if done or truncated:
                        break
                self.assertTrue(done or truncated)
                self.assertGreater(abs(shaped._last_prev_potential), 0.0)
                self.assertLess(abs(shaped._last_prev_potential), 1e-8)
                self.assertEqual(shaped._last_next_potential, 0.0)
                self.assertEqual(shaped._last_reward_shaping, -shaped._last_prev_potential)
                self.assertAlmostEqual(
                    math.fsum(0.8**t * f for t, f in enumerate(shaping)), 0.0, places=20
                )
                self.assertAlmostEqual(shaped._total_reward_shaping, math.fsum(shaping), places=20)
                self.assertEqual(len(shaped.trade_history), 2)
                history = shaped.get_env_history()
                self.assertEqual(history.iloc[-1]["reward_shaping"], -shaped._last_prev_potential)
                self.assertEqual(history.iloc[-1]["next_potential"], 0.0)

    def test_standalone_environment_rejects_invalid_potential_discounts(self):
        for gamma in (None, math.nan, math.inf, -math.inf, -0.1, 1.1, True, "0.8"):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                self.env(potential_gamma=gamma)

    def test_reset_discards_the_previous_episode_potential_and_additive_totals(self):
        env = self.env(
            exit_potential_mode="retain_previous", entry_additive_enabled=True, fee=0.0015
        )
        env.step(Actions.Long_enter.value)
        self.assertNotEqual(env._last_potential, 0.0)
        env.reset()
        _, _, _, _, _ = env.step(Actions.Neutral.value)
        self.assertEqual(env._last_prev_potential, 0.0)
        self.assertEqual(env._last_reward_shaping, 0.0)
        self.assertEqual(env._total_entry_additive, 0.0)
        self.assertEqual(env._total_exit_additive, 0.0)

    def test_bounded_transforms_preserve_signed_signal_and_extreme_stability(self):
        env = self.env()
        transforms = {
            "tanh": math.tanh,
            "softsign": lambda x: x / (1 + abs(x)),
            "arctan": lambda x: 2 * math.atan(x) / math.pi,
            "sigmoid": lambda x: math.tanh(x / 2),
            "softsign_sqrt": lambda x: x / math.hypot(1, x),
            "clip": lambda x: min(1, max(-1, x)),
        }
        for name, oracle in transforms.items():
            for x in (-2.0, -0.5, 0.0, 0.5, 2.0):
                with self.subTest(transform=name, x=x):
                    self.assertAlmostEqual(env._potential_transform(name, x), oracle(x), places=12)
        for x in (-1000.0, 1000.0):
            self.assertEqual(env._potential_transform("sigmoid", x), math.copysign(1, x))
        for x in (-1e200, 1e200):
            self.assertEqual(env._potential_transform("softsign_sqrt", x), math.copysign(1, x))
        self.assertAlmostEqual(env._potential_transform("unknown", 0.5), math.tanh(0.5))

    def test_hold_signal_amplifies_loss_duration_only_and_clamps_negative_duration(self):
        env = self.env(hold_potential_gain=1.0)
        for position in (Positions.Long, Positions.Short):
            for pnl in (-0.06, 0.0, 0.06):
                for duration in (-0.5, 0.5):
                    with self.subTest(position=position, pnl=pnl, duration=duration):
                        multiplier = 2.0 if pnl < 0 else 1.0
                        expected = 0.2 * (
                            math.tanh(pnl / 0.06)
                            + np.sign(pnl) * multiplier * math.tanh(max(0.0, duration))
                        )
                        self.assertAlmostEqual(
                            env._compute_hold_potential(position, pnl, 0.06, duration, 0.4),
                            expected,
                            places=12,
                        )
        self.assertEqual(env._compute_hold_potential(Positions.Neutral, 0.06, 0.06, 1, 0.4), 0.0)
        self.assertEqual(env._compute_hold_potential(Positions.Long, 0.06, 0.0, 1, 0.4), 0.0)
        self.assertEqual(env._compute_hold_potential(Positions.Long, math.nan, 0.06, 1, 0.4), 0.0)
        for ratio in (0.0, math.nan):
            env.rr = ratio
            self.assertAlmostEqual(
                env._compute_hold_potential(Positions.Long, -0.06, 0.06, 0.5, 0.4),
                -0.2 * (math.tanh(1) + math.tanh(0.5)),
                places=12,
            )

    def test_exit_decay_clamps_to_its_probability_bounds(self):
        for decay, expected in (
            (-0.5, 0.4),
            (0.0, 0.4),
            (0.25, 0.3),
            (1.0, 0.0),
            (2.0, 0.0),
            (math.nan, 0.4),
        ):
            with self.subTest(decay=decay):
                env = self.env(
                    exit_potential_mode="progressive_release", exit_potential_decay=decay
                )
                self.assertAlmostEqual(env._compute_exit_potential(0.4, 0.8), expected)
        env = self.env(exit_potential_mode="spike_cancel")
        self.assertEqual(env._compute_exit_potential(0.4, 0.0), 0.4)
        self.assertAlmostEqual(env._compute_exit_potential(0.4, 0.8), 0.5)


class DiscountResolutionTest(QaTestCase):
    def model(self, **parameters):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        model = ReforceXY(
            config=model_config(temp.name, freqai={"model_training_parameters": parameters})
        )
        model.live = True
        self.addCleanup(model.close_envs)
        return model

    def test_effective_null_gamma_is_rejected_before_environment_construction(self):
        model = self.model(gamma=None)
        with self.assertRaises(ValueError):
            model.pack_env_dict("BTC/USDT")
        model = self.model(gamma=0.83)
        with self.assertRaises(ValueError):
            model.pack_env_dict("BTC/USDT", {"gamma": None})

    def test_effective_discount_requires_a_finite_real_scalar_in_unit_interval(self):
        model = self.model(gamma=0.83)
        for gamma in (
            math.nan,
            math.inf,
            -math.inf,
            -0.1,
            1.1,
            True,
            np.bool_(False),
            "0.8",
            0.8 + 0j,
            [0.8],
            np.array(0.8),
            np.nextafter(np.longdouble(1), np.longdouble(math.inf)),
        ):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                model.pack_env_dict("BTC/USDT", {"gamma": gamma})
        for gamma in (0, 1, np.int64(1), np.float32(0.79), np.float64(0.83)):
            with self.subTest(gamma=gamma):
                kwargs = model.pack_env_dict("BTC/USDT", {"gamma": gamma})
                resolved = kwargs["config"]["freqai"]["rl_config"]["model_reward_parameters"][
                    "potential_gamma"
                ]
                self.assertIs(type(resolved), float)
                self.assertEqual(resolved, float(gamma))
                prices = pd.DataFrame({"open": [100.0] * 16})
                env = MyRLEnv(df=prices.copy(), prices=prices, **kwargs)
                self.addCleanup(env.close)
                learner = MaskablePPO(
                    "MlpPolicy",
                    env,
                    gamma=resolved,
                    n_steps=8,
                    batch_size=8,
                    n_epochs=1,
                    device="cpu",
                )
                learner.learn(8)
                self.assertEqual(learner.gamma, env._potential_gamma)
                self.assertGreater(learner._n_updates, 0)

    def test_invalid_replacement_discount_does_not_close_running_environments(self):
        model = self.model(gamma=0.83)
        features = pd.DataFrame({"f": np.arange(16, dtype=float)})
        prices = pd.DataFrame({"open": [100.0] * 16})
        data = {"train_features": features.copy(), "test_features": features.copy()}
        kitchen = SimpleNamespace(pair="BTC/USDT")
        model.set_train_and_eval_environments(data, prices, prices, kitchen)
        running_train, running_eval = model.train_env, model.eval_env
        first = running_train.reset()
        for gamma in (None, math.nan, math.inf, -math.inf, -0.1, 1.1, True, "0.8"):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                model.set_train_and_eval_environments(
                    data, prices, prices, kitchen, {"gamma": gamma}
                )
            self.assertIs(model.train_env, running_train)
            self.assertIs(model.eval_env, running_eval)
        self.assertIs(model.train_env, running_train)
        self.assertIs(model.eval_env, running_eval)
        second, _, done, _ = running_train.step([Actions.Neutral.value])
        self.assertFalse(done[0])
        np.testing.assert_array_equal(second[:, :, 0], first[:, :, 0] + 1)

    def test_valid_override_supersedes_null_base_and_configured_potential_discount(self):
        model = self.model(gamma=None)
        model.rl_config["model_reward_parameters"]["potential_gamma"] = 0.41
        kwargs = model.pack_env_dict("BTC/USDT", {"gamma": 0.83})
        prices = pd.DataFrame({"open": [100.0] * 16})
        env = MyRLEnv(df=prices.copy(), prices=prices, **kwargs)
        self.addCleanup(env.close)
        learner = MaskablePPO(
            "MlpPolicy", env, gamma=0.83, n_steps=8, batch_size=8, n_epochs=1, device="cpu"
        )
        learner.learn(8)
        self.assertEqual(learner.gamma, env._potential_gamma)
        self.assertGreater(learner._n_updates, 0)

    def test_persisted_numpy_learner_discount_overrides_finite_and_null_base(self):
        with tempfile.TemporaryDirectory() as temp:
            gamma = np.float32(0.79)
            prices = pd.DataFrame({"open": [100.0] * 16})
            for base in (0.95, None):
                with self.subTest(base=base):
                    owner = self.model(gamma=base)
                    kwargs = owner.pack_env_dict("BTC/USDT", {"gamma": gamma})
                    env = MyRLEnv(df=prices.copy(), prices=prices, **kwargs)
                    self.addCleanup(env.close)
                    learner = MaskablePPO(
                        "MlpPolicy",
                        env,
                        gamma=gamma,
                        n_steps=8,
                        batch_size=8,
                        n_epochs=1,
                        device="cpu",
                    )
                    learner.learn(8)
                    archive = Path(temp) / f"learner-{base}"
                    learner.save(archive)
                    loaded = MaskablePPO.load(archive)
                    self.assertIsInstance(loaded.gamma, np.floating)
                    previous_updates = loaded._n_updates
                    observed = []

                    def evaluate(policy, environment, observed=observed, **parameters):
                        observed.append(
                            (
                                policy.gamma,
                                policy.get_env().get_attr("_potential_gamma"),
                                environment.get_attr("_potential_gamma"),
                            )
                        )
                        return evaluate_policy(policy, environment, **parameters)

                    data = {"train_features": prices.copy(), "test_features": prices.copy()}
                    kitchen = SimpleNamespace(pair="BTC/USDT", data_path=Path(temp) / f"fit-{base}")
                    kitchen.data_path.mkdir()
                    with mock.patch(
                        "ReforceXY.user_data.freqaimodels.ReforceXY.evaluate_policy",
                        side_effect=evaluate,
                    ):
                        continued = owner.fit(
                            data,
                            kitchen,
                            prices_train=prices,
                            prices_test=prices,
                            deployment_state=(loaded, None),
                        )
                    self.assertEqual(observed, [(gamma, [float(gamma)], [float(gamma)])])
                    self.assertEqual(float(continued.gamma), float(gamma))
                    self.assertGreater(loaded._n_updates, previous_updates)


if __name__ == "__main__":
    unittest.main()
