"""Study reuse, purge, warm-start and fallback with real local Optuna storage."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import optuna
import pandas as pd
from optuna.distributions import FloatDistribution
from optuna.exceptions import DuplicatedStudyError
from optuna.storages import RDBStorage
from optuna.trial import TrialState
from qa_support import PAIR, QaTestCase, model_config

from ReforceXY.user_data.freqaimodels.ReforceXY import ReforceXY

PPO_PARAMS = {
    "learning_rate": 0.001,
    "clip_range": 0.2,
    "n_steps": 512,
    "batch_size": 64,
    "gamma": 0.83,
    "ent_coef": 0.01,
    "n_epochs": 1,
    "gae_lambda": 0.9,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
}


class OptunaLifecycleTest(QaTestCase):
    def model(self, backend="file", **options):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = model_config(
            temporary.name,
            freqai={
                "model_training_parameters": {"gamma": 0.41, "learning_rate": 0.03},
                "rl_config_optuna": {
                    "enabled": True,
                    "storage": backend,
                    "sampler": "tpe",
                    "seed": 42,
                    "n_trials": 1,
                    "n_startup_trials": 2,
                    **options,
                },
            },
        )
        model = ReforceXY(config=config)
        self.addCleanup(model.close_envs)
        return model

    def storage(self, model, pair=PAIR):
        storage = model.create_storage(pair)
        if isinstance(storage, RDBStorage):
            self.addCleanup(storage.engine.dispose)
        return storage

    def reopened(self, model, pair=PAIR):
        return optuna.load_study(
            study_name=f"{model.freqai_info['identifier']}-{pair}",
            storage=self.storage(model, pair),
        )

    def objective(self, trial, *_):
        # Isolate only the expensive training boundary. Suggestions, study state,
        # winner selection and all on-disk persistence remain real Optuna code.
        gamma = trial.suggest_float("gamma", 0.5, 1.0)
        for name, value in PPO_PARAMS.items():
            if name != "gamma":
                trial.suggest_categorical(name, [value])
        return gamma

    def optimize(self, model, *, pair=PAIR, objective=None):
        with mock.patch.object(model, "objective", side_effect=objective or self.objective):
            return model.optimize(SimpleNamespace(pair=pair), 512, pd.DataFrame(), pd.DataFrame())

    def test_reuse_keeps_previous_winner_and_sampled_values_override_user_configuration(self):
        for backend in ("sqlite", "file"):
            with self.subTest(backend=backend):
                model = self.model(backend)
                first = self.optimize(model)
                first_gamma = first["gamma"]
                model.rl_config_optuna["seed"] = 43
                second = self.optimize(model)
                study = self.reopened(model)
                self.assertEqual([trial.number for trial in study.trials], [0, 1])
                self.assertLess(study.trials[1].value, study.trials[0].value)
                self.assertEqual(second["gamma"], first_gamma)
                self.assertEqual(second["learning_rate"](0.5), 0.001)
                self.assertEqual(second["device"], "cpu")
                self.assertEqual(second["policy_kwargs"]["net_arch"], {"pi": [8], "vf": [8]})
                self.assertEqual(
                    study.user_attrs["objective_identity"], ReforceXY._OPTUNA_OBJECTIVE_IDENTITY
                )
                self.assertEqual(model.load_best_trial_params(PAIR)["gamma"], first_gamma)
                self.assertEqual(model._load_optuna_retrain_counters(PAIR), {"BTC_USDT": 2})

    def test_continuous_mode_resets_trials_instead_of_reusing_a_previous_winner(self):
        model = self.model(continuous=True)
        first = self.optimize(model)
        model.rl_config_optuna["seed"] = 43
        second = self.optimize(model)
        study = self.reopened(model)
        self.assertEqual([trial.number for trial in study.trials], [0])
        self.assertNotEqual(second["gamma"], first["gamma"])
        self.assertEqual(study.best_value, second["gamma"])

    def test_periodic_purge_warm_starts_even_when_warm_start_is_disabled(self):
        model = self.model(purge_period=2, warm_start=False)
        first = self.optimize(model)
        self.optimize(model, pair="ETH/USDT")
        model.rl_config_optuna["seed"] = 43
        second = self.optimize(model)
        study = self.reopened(model)
        self.assertEqual([trial.number for trial in study.trials], [0])
        self.assertEqual(second["gamma"], first["gamma"])
        self.assertEqual(study.trials[0].params["gamma"], first["gamma"])
        self.optimize(model)
        self.assertEqual([trial.number for trial in self.reopened(model).trials], [0, 1])
        self.assertEqual(len(self.reopened(model, "ETH/USDT").trials), 1)
        self.assertEqual(model._load_optuna_retrain_counters(PAIR), {"BTC_USDT": 3, "ETH_USDT": 1})

    def test_warm_start_uses_saved_coordinates_and_missing_coordinates_do_not_block_trials(self):
        for saved in (False, True):
            with self.subTest(saved=saved):
                model = self.model(warm_start=True)
                if saved:
                    model.save_best_trial_params({**PPO_PARAMS, "gamma": 0.91}, PAIR)
                result = self.optimize(model)
                study = self.reopened(model)
                self.assertEqual(study.trials[0].state, TrialState.COMPLETE)
                if saved:
                    self.assertEqual(result["gamma"], 0.91)
                    self.assertEqual(study.best_value, 0.91)
                else:
                    self.assertNotEqual(result["gamma"], 0.91)

    def test_incompatible_or_missing_study_identity_discards_old_trials(self):
        for identity in (None, "terminal-liquidation-risk-normalized-trained-policy-v3"):
            with self.subTest(identity=identity):
                model = self.model()
                study = optuna.create_study(
                    study_name=f"{model.freqai_info['identifier']}-{PAIR}",
                    storage=self.storage(model),
                    direction="maximize",
                )
                if identity is not None:
                    study.set_user_attr("objective_identity", identity)
                study.add_trial(
                    optuna.create_trial(
                        value=100.0,
                        params={"gamma": 0.99},
                        distributions={"gamma": FloatDistribution(0.5, 1.0)},
                    )
                )
                result = self.optimize(model)
                reopened = self.reopened(model)
                self.assertEqual(len(reopened.trials), 1)
                self.assertLess(reopened.best_value, 1.0)
                self.assertEqual(
                    reopened.user_attrs["objective_identity"], ReforceXY._OPTUNA_OBJECTIVE_IDENTITY
                )
                self.assertNotEqual(result["gamma"], 0.99)

    def test_failed_identity_reset_fails_closed_without_reusing_incompatible_trials(self):
        model = self.model()
        storage = self.storage(model)
        study = optuna.create_study(
            study_name=f"{model.freqai_info['identifier']}-{PAIR}",
            storage=storage,
        )
        study.set_user_attr("objective_identity", "legacy-objective")
        with (
            mock.patch.object(
                type(storage), "delete_study", side_effect=OSError("read-only storage")
            ),
            self.assertRaises(DuplicatedStudyError),
        ):
            self.optimize(model)
        self.assertEqual(self.reopened(model).user_attrs["objective_identity"], "legacy-objective")

    def test_pruned_trials_fall_back_to_saved_parameters_or_return_no_candidate(self):
        def pruned(trial, *_):
            trial.report(0.0, 1)
            raise optuna.TrialPruned()

        model = self.model()
        self.assertIsNone(self.optimize(model, objective=pruned))
        model.save_best_trial_params(PPO_PARAMS, PAIR)
        result = self.optimize(model, objective=pruned)
        self.assertEqual(result["gamma"], 0.83)
        self.assertEqual(
            [trial.state for trial in self.reopened(model).trials], [TrialState.PRUNED] * 2
        )
        self.assertEqual(model.load_best_trial_params(PAIR), PPO_PARAMS)

    def test_failed_trial_uses_saved_fallback_even_when_study_has_a_previous_winner(self):
        def failed(*_):
            raise RuntimeError("training failure")

        model = self.model()
        first = self.optimize(model)
        model.save_best_trial_params({**PPO_PARAMS, "gamma": 0.93}, PAIR)
        result = self.optimize(model, objective=failed)
        study = self.reopened(model)
        self.assertEqual(
            [trial.state for trial in study.trials], [TrialState.COMPLETE, TrialState.FAIL]
        )
        self.assertEqual(study.best_params["gamma"], first["gamma"])
        self.assertEqual(result["gamma"], 0.93)
        self.assertEqual(model.load_best_trial_params(PAIR)["gamma"], 0.93)

    def test_keyboard_interrupt_reuses_only_completed_candidates(self):
        def interrupted(*_):
            raise KeyboardInterrupt()

        model = self.model()
        self.assertIsNone(self.optimize(model, objective=interrupted))
        first = self.optimize(model)
        result = self.optimize(model, objective=interrupted)
        self.assertEqual(result["gamma"], first["gamma"])
        self.assertEqual(
            [trial.state for trial in self.reopened(model).trials],
            [TrialState.FAIL, TrialState.COMPLETE, TrialState.FAIL],
        )

    def test_shared_lstm_disables_a_separate_critic_in_the_selected_parameters(self):
        def recurrent(trial, *_):
            value = self.objective(trial)
            trial.suggest_categorical("lstm_hidden_size", [8])
            trial.suggest_categorical("n_lstm_layers", [1])
            trial.suggest_categorical("enable_critic_lstm", [True])
            return value

        model = self.model()
        model.model_type = "RecurrentPPO"
        model.model_training_parameters["policy_kwargs"]["shared_lstm"] = True
        model._model_params_cache = None
        result = self.optimize(model, objective=recurrent)
        self.assertFalse(result["policy_kwargs"]["enable_critic_lstm"])
        self.assertFalse(model.load_best_trial_params(PAIR)["enable_critic_lstm"])

    def test_value_based_lifecycle_derives_gradient_steps_from_sampled_subsampling(self):
        def value_based(trial, *_):
            gamma = trial.suggest_float("gamma", 0.5, 1.0)
            for name, value in {
                "learning_rate": 0.001,
                "batch_size": 8,
                "buffer_size": 64,
                "train_freq": 8,
                "subsample_steps": 3,
                "exploration_fraction": 0.2,
                "exploration_initial_eps": 0.8,
                "exploration_final_eps": 0.1,
                "target_update_interval": 8,
                "learning_starts": 0,
            }.items():
                trial.suggest_categorical(name, [value])
            return gamma

        model = self.model()
        model.model_type = "DQN"
        model.model_training_parameters = {"device": "cpu", "policy_kwargs": {"net_arch": [8]}}
        model._model_params_cache = None
        result = self.optimize(model, objective=value_based)
        self.assertEqual(result["gradient_steps"], 3)
        self.assertNotIn("subsample_steps", result)
        self.assertEqual(self.reopened(model).trials[0].state, TrialState.COMPLETE)


if __name__ == "__main__":
    unittest.main()
