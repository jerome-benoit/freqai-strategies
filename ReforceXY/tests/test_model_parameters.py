"""Configuration and sampled parameters exercised by their real SB3 consumers."""

import tempfile
import unittest

import pandas as pd
import torch as th
from optuna.trial import FixedTrial
from qa_support import PAIR, QaTestCase, model_config

from ReforceXY.user_data.freqaimodels.ReforceXY import (
    MyRLEnv,
    ReforceXY,
    convert_optuna_params_to_model_params,
)


class ModelParametersTest(QaTestCase):
    def model(self, model_type="MaskablePPO", *, parameters=None, **rl_options):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = model_config(temporary.name)
        config["freqai"]["rl_config"].update(
            {
                "model_type": model_type,
                "policy_type": "MlpLstmPolicy" if model_type == "RecurrentPPO" else "MlpPolicy",
                **rl_options,
            }
        )
        if parameters is not None:
            config["freqai"]["model_training_parameters"] = parameters
        model = ReforceXY(config=config)
        model.live = True
        self.addCleanup(model.close_envs)
        return model

    def learner(self, model, parameters):
        prices = pd.DataFrame({"open": [100.0] * 16})
        env = MyRLEnv(df=prices.copy(), prices=prices, **model.pack_env_dict(PAIR, parameters))
        self.addCleanup(env.close)
        parameters = {**parameters, "device": "cpu"}
        learner = model.MODELCLASS(model.policy_type, env, **parameters)
        self.assertEqual(learner.gamma, env._potential_gamma)
        return learner

    def test_ppo_architecture_resolution_and_schedules_reach_the_actual_policy(self):
        for architecture, pi_width, vf_width in (
            ([8], 8, 8),
            ("small", 128, 128),
            ("unknown", 128, 128),
            ({"pi": [8], "vf": [16]}, 8, 16),
            ({"pi": None, "vf": [16]}, 128, 16),
            (17, 128, 128),
        ):
            with self.subTest(architecture=architecture):
                model = self.model(
                    parameters={
                        "n_steps": 8,
                        "batch_size": 8,
                        "n_epochs": 1,
                        "learning_rate": 0.002,
                        "clip_range": 0.4,
                        "policy_kwargs": {
                            "net_arch": architecture,
                            "activation_fn": "tanh",
                            "optimizer_class": "rmsprop",
                        },
                    },
                    lr_schedule=True,
                    cr_schedule=True,
                )
                learner = self.learner(model, model.get_model_params())
                self.assertEqual(learner.policy.mlp_extractor.policy_net[0].out_features, pi_width)
                self.assertEqual(learner.policy.mlp_extractor.value_net[0].out_features, vf_width)
                self.assertIsInstance(learner.policy.mlp_extractor.policy_net[1], th.nn.Tanh)
                self.assertIsInstance(learner.policy.optimizer, th.optim.RMSprop)
                self.assertAlmostEqual(learner.lr_schedule(0.5), 0.001)
                self.assertAlmostEqual(learner.clip_range(0.5), 0.2)

    def test_dqn_architecture_and_explicit_gradient_override_survive_normalization(self):
        for architecture, explicit_steps, width, expected_steps in (
            ("small", None, 128, 3),
            ("unknown", 7, 128, 7),
            ({"pi": [8]}, None, 128, 3),
            ([8], None, 8, 3),
        ):
            with self.subTest(architecture=architecture, explicit_steps=explicit_steps):
                model = self.model(
                    "DQN",
                    parameters={
                        "train_freq": 8,
                        "subsample_steps": 3,
                        "gradient_steps": explicit_steps,
                        "learning_starts": 0,
                        "buffer_size": 64,
                        "batch_size": 8,
                        "policy_kwargs": {"net_arch": architecture},
                    },
                )
                learner = self.learner(model, model.get_model_params())
                self.assertEqual(learner.policy.q_net.q_net[0].out_features, width)
                self.assertEqual(learner.gradient_steps, expected_steps)
                self.assertEqual(learner.train_freq.frequency, 8)

    def test_samplers_build_all_algorithm_families_and_keep_optional_policy_controls(self):
        ppo = {
            "n_steps": 512,
            "batch_size": 64,
            "gamma": 0.95,
            "learning_rate": 0.001,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "n_epochs": 1,
            "gae_lambda": 0.9,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "lr_schedule": "linear",
            "cr_schedule": "linear",
            "target_kl": None,
            "ortho_init": False,
            "net_arch": "small",
            "activation_fn": "elu",
            "optimizer_class": "rmsprop",
            "n_lstm_layers": 1,
            "lstm_hidden_size": 64,
            "enable_critic_lstm": True,
        }
        for family, shared in (
            ("PPO", False),
            ("MaskablePPO", False),
            ("RecurrentPPO", False),
            ("RecurrentPPO", True),
        ):
            with self.subTest(family=family, shared=shared):
                policy_kwargs = {"net_arch": [8]}
                if family == "RecurrentPPO":
                    policy_kwargs["shared_lstm"] = shared
                model = self.model(family, parameters={"policy_kwargs": policy_kwargs})
                sampled = model.get_optuna_params(FixedTrial(ppo))
                sampled["policy_kwargs"].update(policy_kwargs if shared else {})
                learner = self.learner(model, sampled)
                self.assertAlmostEqual(learner.lr_schedule(0.5), 0.0005)
                self.assertAlmostEqual(learner.clip_range(0.5), 0.1)
                self.assertIsInstance(learner.policy.optimizer, th.optim.RMSprop)
                if family == "RecurrentPPO":
                    self.assertEqual(learner.policy.lstm_actor.hidden_size, 64)
                    self.assertEqual(learner.policy.lstm_critic is None, shared)

    def test_value_based_exploration_boundaries_change_the_consumed_epsilon_schedule(self):
        for family, initial, final, fraction in (
            ("DQN", 0.9, 0.2, 0.2),
            ("DQN", 0.8, 0.1, 0.15),
            ("QRDQN", 0.6, 0.2, 0.05),
        ):
            with self.subTest(family=family, initial=initial):
                raw = {
                    "train_freq": 8,
                    "subsample_steps": 2,
                    "gamma": 0.95,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "lr_schedule": "linear",
                    "buffer_size": 10000,
                    "exploration_initial_eps": initial,
                    "exploration_final_eps": final,
                    "exploration_fraction": fraction,
                    "target_update_interval": 1000,
                    "learning_starts": 500,
                    "net_arch": "small",
                    "activation_fn": "elu",
                    "optimizer_class": "rmsprop",
                    "n_quantiles": 10,
                }
                model = self.model(family, parameters={"policy_kwargs": {"net_arch": [8]}})
                learner = self.learner(model, model.get_optuna_params(FixedTrial(raw)))
                self.assertAlmostEqual(
                    learner.exploration_schedule(1 - fraction / 2), (initial + final) / 2
                )
                self.assertEqual(learner.gradient_steps, 4)
                if family == "QRDQN":
                    self.assertEqual(learner.policy.quantile_net.n_quantiles, 10)

    def test_incomplete_hyperparameters_fail_before_a_partial_learner_is_returned(self):
        for family, raw in (
            ("PPO", {}),
            ("PPO", {"learning_rate": 0.001}),
            ("DQN", {"learning_rate": 0.001}),
            ("unknown", {"learning_rate": 0.001}),
        ):
            with self.subTest(family=family), self.assertRaises(ValueError):
                convert_optuna_params_to_model_params(family, raw)


if __name__ == "__main__":
    unittest.main()
