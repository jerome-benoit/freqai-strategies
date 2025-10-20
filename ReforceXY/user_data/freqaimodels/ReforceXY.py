import copy
import gc
import json
import logging
import math
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import torch as th
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment
from freqtrade.freqai.RL.BaseReinforcementLearningModel import (
    BaseReinforcementLearningModel,
)
from freqtrade.freqai.tensorboard.TensorboardCallback import TensorboardCallback
from freqtrade.strategy import timeframe_to_minutes
from gymnasium.spaces import Box
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from optuna import Trial, TrialPruned, create_study, delete_study
from optuna.exceptions import ExperimentalWarning
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna.storages import (
    BaseStorage,
    JournalStorage,
    RDBStorage,
    RetryFailedTrialCallback,
)
from optuna.storages.journal import JournalFileBackend
from optuna.study import Study, StudyDirection
from pandas import DataFrame, merge
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import is_masking_supported
from stable_baselines3.common.callbacks import (
    BaseCallback,
    ProgressBarCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import Figure, HParam
from stable_baselines3.common.type_aliases import TrainFreq
from stable_baselines3.common.utils import ConstantSchedule, set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecMonitor,
)

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ExperimentalWarning)
logger = logging.getLogger(__name__)


class ReforceXY(BaseReinforcementLearningModel):
    """
    Custom Freqtrade Freqai reinforcement learning prediction model.
    Model specific config:
    {
        "freqaimodel": "ReforceXY",
        "strategy": "RLAgentStrategy",
        ...
        "freqai": {
            ...
            "rl_config": {
                ...
                "n_envs": 1,                        // Number of DummyVecEnv or SubProcVecEnv training environments
                "n_eval_envs": 1,                   // Number of DummyVecEnv or SubProcVecEnv evaluation environments
                "multiprocessing": false,           // Use SubprocVecEnv if n_envs>1 (otherwise DummyVecEnv)
                "eval_multiprocessing": false,      // Use SubprocVecEnv if n_eval_envs>1 (otherwise DummyVecEnv)
                "frame_stacking": 0,                // Number of VecFrameStack stacks (set > 1 to use)
                "inference_masking": true,          // Enable action masking during inference
                "lr_schedule": false,               // Enable learning rate linear schedule
                "cr_schedule": false,               // Enable clip range linear schedule
                "n_eval_steps": 10_000,             // Number of environment steps between evaluations
                "n_eval_episodes": 5,               // Number of episodes per evaluation
                "max_no_improvement_evals": 0,      // Maximum consecutive evaluations without a new best model
                "min_evals": 0,                     // Number of evaluations before start to count evaluations without improvements
                "check_envs": true,                 // Check that an environment follows Gym API
                "tensorboard_throttle": 1,          // Number of training calls between tensorboard logs
                "plot_new_best": false,             // Enable tensorboard rollout plot upon finding a new best model
                "plot_window": 2000,                // Environment history window used for tensorboard rollout plot
            },
            "rl_config_optuna": {
                "enabled": false,                   // Enable hyperopt
                "n_trials": 100,                    // Number of trials
                "n_startup_trials": 15,             // Number of initial random trials for TPESampler
                "timeout_hours": 0,                 // Maximum time in hours for hyperopt (0 = no timeout)
                "continuous": false,                // If true, perform continuous optimization
                "warm_start": false,                // If true, enqueue previous best params if exists
                "seed": 42,                         // RNG seed
                "storage": "sqlite",                // Optuna storage backend (sqlite|file)
            }
        }
    }
    Requirements:
        - pip install optuna

    Optional:
        - pip install optuna-dashboard
    """

    _LOG_2 = math.log(2.0)
    DEFAULT_IDLE_DURATION_MULTIPLIER: int = 4
    _action_masks_cache: Dict[Tuple[bool, float], NDArray[np.bool_]] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pairs: List[str] = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.pairs:
            raise ValueError(
                "FreqAI model requires StaticPairList method defined in pairlists configuration and pair_whitelist defined in exchange section configuration"
            )
        self.action_masking: bool = self.model_type == "MaskablePPO"
        self.rl_config.setdefault("action_masking", self.action_masking)
        self.inference_masking: bool = self.rl_config.get("inference_masking", True)
        self.recurrent: bool = self.model_type == "RecurrentPPO"
        self.lr_schedule: bool = self.rl_config.get("lr_schedule", False)
        self.cr_schedule: bool = self.rl_config.get("cr_schedule", False)
        self.n_envs: int = self.rl_config.get("n_envs", 1)
        self.n_eval_envs: int = self.rl_config.get("n_eval_envs", 1)
        self.multiprocessing: bool = self.rl_config.get("multiprocessing", False)
        self.eval_multiprocessing: bool = self.rl_config.get(
            "eval_multiprocessing", False
        )
        self.frame_stacking: int = self.rl_config.get("frame_stacking", 0)
        self.n_eval_steps: int = self.rl_config.get("n_eval_steps", 10_000)
        self.n_eval_episodes: int = self.rl_config.get("n_eval_episodes", 5)
        self.max_no_improvement_evals: int = self.rl_config.get(
            "max_no_improvement_evals", 0
        )
        self.min_evals: int = self.rl_config.get("min_evals", 0)
        self.rl_config.setdefault("tensorboard_throttle", 1)
        self.plot_new_best: bool = self.rl_config.get("plot_new_best", False)
        self.check_envs: bool = self.rl_config.get("check_envs", True)
        self.progressbar_callback: Optional[ProgressBarCallback] = None
        # Optuna hyperopt
        self.rl_config_optuna: Dict[str, Any] = self.freqai_info.get(
            "rl_config_optuna", {}
        )
        self.hyperopt: bool = (
            self.freqai_info.get("enabled", False)
            and self.rl_config_optuna.get("enabled", False)
            and self.data_split_parameters.get("test_size", 0.1) > 0
        )
        self.optuna_timeout_hours: float = self.rl_config_optuna.get("timeout_hours", 0)
        self.optuna_n_trials: int = self.rl_config_optuna.get("n_trials", 100)
        self.optuna_n_startup_trials: int = self.rl_config_optuna.get(
            "n_startup_trials", 15
        )
        self.optuna_eval_callback: Optional[MaskableTrialEvalCallback] = None
        self._model_params_cache: Optional[Dict[str, Any]] = None
        self.unset_unsupported()

    @staticmethod
    def _normalize_position(position: Any) -> Positions:
        if isinstance(position, Positions):
            return position
        try:
            position = float(position)
            if position == float(Positions.Long.value):
                return Positions.Long
            if position == float(Positions.Short.value):
                return Positions.Short
            return Positions.Neutral
        except Exception:
            return Positions.Neutral

    @staticmethod
    def get_action_masks(
        can_short: bool,
        position: Positions,
    ) -> NDArray[np.bool_]:
        position = ReforceXY._normalize_position(position)

        cache_key = (
            can_short,
            float(position.value),
        )
        if cache_key in ReforceXY._action_masks_cache:
            return ReforceXY._action_masks_cache[cache_key]

        action_masks = np.zeros(len(Actions), dtype=np.bool_)

        action_masks[Actions.Neutral.value] = True
        if position == Positions.Neutral:
            action_masks[Actions.Long_enter.value] = True
            if can_short:
                action_masks[Actions.Short_enter.value] = True
        elif position == Positions.Long:
            action_masks[Actions.Long_exit.value] = True
        elif position == Positions.Short:
            action_masks[Actions.Short_exit.value] = True

        ReforceXY._action_masks_cache[cache_key] = action_masks
        return ReforceXY._action_masks_cache[cache_key]

    def unset_unsupported(self) -> None:
        """
        If user has activated any features that may conflict, this
        function will set them to proper values and warn them
        """
        if not isinstance(self.n_envs, int) or self.n_envs < 1:
            logger.warning("Invalid n_envs=%s. Forcing n_envs=1", self.n_envs)
            self.n_envs = 1
        if not isinstance(self.n_eval_envs, int) or self.n_eval_envs < 1:
            logger.warning(
                "Invalid n_eval_envs=%s. Forcing n_eval_envs=1", self.n_eval_envs
            )
            self.n_eval_envs = 1
        if self.multiprocessing and self.n_envs <= 1:
            logger.warning(
                "User tried to use multiprocessing with n_envs=%s. Deactivating multiprocessing",
                self.n_envs,
            )
            self.multiprocessing = False
        if self.eval_multiprocessing and self.n_eval_envs <= 1:
            logger.warning(
                "User tried to use eval_multiprocessing with n_eval_envs=%s. Deactivating eval_multiprocessing",
                self.n_eval_envs,
            )
            self.eval_multiprocessing = False
        if self.multiprocessing and self.plot_new_best:
            logger.warning(
                "User tried to use plot_new_best with multiprocessing=%s. Deactivating plot_new_best",
                self.multiprocessing,
            )
            self.plot_new_best = False
        if not isinstance(self.frame_stacking, int) or self.frame_stacking < 0:
            logger.warning(
                "Invalid frame_stacking=%s. Forcing frame_stacking=0",
                self.frame_stacking,
            )
            self.frame_stacking = 0
        if self.frame_stacking == 1:
            logger.warning(
                "Setting frame_stacking=%s is equivalent to no stacking. Forcing frame_stacking=0",
                self.frame_stacking,
            )
            self.frame_stacking = 0
        if not isinstance(self.n_eval_steps, int) or self.n_eval_steps <= 0:
            logger.warning(
                "Invalid n_eval_steps=%s. Forcing n_eval_steps=10_000",
                self.n_eval_steps,
            )
            self.n_eval_steps = 10_000
        if not isinstance(self.n_eval_episodes, int) or self.n_eval_episodes <= 0:
            logger.warning(
                "Invalid n_eval_episodes=%s. Forcing n_eval_episodes=5",
                self.n_eval_episodes,
            )
            self.n_eval_episodes = 5
        add_state_info = self.rl_config.get("add_state_info", False)
        if not add_state_info:
            logger.warning(
                "Setting add_state_info=%s will lead to desynchronized trade states during inference after restart",
                add_state_info,
            )
        tensorboard_throttle = self.rl_config.get("tensorboard_throttle", 1)
        if not isinstance(tensorboard_throttle, int) or tensorboard_throttle < 1:
            logger.warning(
                "Invalid tensorboard_throttle=%s. Forcing tensorboard_throttle=1",
                tensorboard_throttle,
            )
            self.rl_config["tensorboard_throttle"] = 1
        if self.continual_learning and bool(self.frame_stacking):
            logger.warning(
                "User tried to use continual_learning with frame_stacking=%s. "
                "Deactivating continual_learning",
                self.frame_stacking,
            )
            self.continual_learning = False

    def pack_env_dict(
        self, pair: str, model_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        env_info = super().pack_env_dict(pair)

        config = env_info.setdefault("config", {})
        freqai_cfg = config.setdefault("freqai", {})
        rl_cfg = freqai_cfg.setdefault("rl_config", {})
        model_reward_parameters = rl_cfg.setdefault("model_reward_parameters", {})

        gamma: Optional[float] = None
        best_trial_params: Optional[Dict[str, Any]] = None
        if self.hyperopt:
            best_trial_params = self.load_best_trial_params(pair)

        if model_params and isinstance(model_params.get("gamma"), (int, float)):
            gamma = float(model_params.get("gamma"))
        elif best_trial_params and isinstance(
            best_trial_params.get("gamma"), (int, float)
        ):
            gamma = float(best_trial_params.get("gamma"))
        elif hasattr(self.model, "gamma") and isinstance(
            self.model.gamma, (int, float)
        ):
            gamma = float(self.model.gamma)
        else:
            model_params_gamma = self.get_model_params().get("gamma")
            if isinstance(model_params_gamma, (int, float)):
                gamma = float(model_params_gamma)

        if gamma is not None:
            model_reward_parameters["potential_gamma"] = gamma
        else:
            logger.warning(
                f"{pair}: No valid PBRS discount gamma resolved for environment"
            )

        return env_info

    def set_train_and_eval_environments(
        self,
        data_dictionary: Dict[str, DataFrame],
        prices_train: DataFrame,
        prices_test: DataFrame,
        dk: FreqaiDataKitchen,
    ) -> None:
        """
        Set training and evaluation environments
        """
        if self.train_env is not None or self.eval_env is not None:
            logger.info("Closing environments")
            self.close_envs()

        train_df = data_dictionary.get("train_features")
        test_df = data_dictionary.get("test_features")
        env_dict = self.pack_env_dict(dk.pair)
        seed = self.get_model_params().get("seed", 42)

        if self.check_envs:
            logger.info("Checking environments")
            _train_env_check = MyRLEnv(
                df=train_df,
                prices=prices_train,
                id="train_env_check",
                seed=seed,
                **env_dict,
            )
            try:
                check_env(_train_env_check)
            finally:
                _train_env_check.close()
            _eval_env_check = MyRLEnv(
                df=test_df,
                prices=prices_test,
                id="eval_env_check",
                seed=seed + 10_000,
                **env_dict,
            )
            try:
                check_env(_eval_env_check)
            finally:
                _eval_env_check.close()

        logger.info(
            "Populating %s train and %s eval environments",
            self.n_envs,
            self.n_eval_envs,
        )
        self.train_env, self.eval_env = self._get_train_and_eval_environments(
            dk,
            train_df=train_df,
            test_df=test_df,
            prices_train=prices_train,
            prices_test=prices_test,
            seed=seed,
            env_info=env_dict,
        )

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters
        """
        if self._model_params_cache is not None:
            return copy.deepcopy(self._model_params_cache)

        model_params: Dict[str, Any] = copy.deepcopy(self.model_training_parameters)

        model_params.setdefault("seed", 42)
        model_params.setdefault("gamma", 0.95)

        if not self.hyperopt and self.lr_schedule:
            lr = model_params.get("learning_rate", 0.0003)
            if isinstance(lr, (int, float)):
                lr = float(lr)
                model_params["learning_rate"] = get_schedule("linear", lr)
                logger.info(
                    "Learning rate linear schedule enabled, initial value: %s", lr
                )

        if not self.hyperopt and "PPO" in self.model_type and self.cr_schedule:
            cr = model_params.get("clip_range", 0.2)
            if isinstance(cr, (int, float)):
                cr = float(cr)
                model_params["clip_range"] = get_schedule("linear", cr)
                logger.info("Clip range linear schedule enabled, initial value: %s", cr)

        if "DQN" in self.model_type:
            if model_params.get("gradient_steps") is None:
                model_params["gradient_steps"] = compute_gradient_steps(
                    model_params.get("train_freq"), model_params.get("subsample_steps")
                )
            if "subsample_steps" in model_params:
                model_params.pop("subsample_steps", None)

        if not model_params.get("policy_kwargs"):
            model_params["policy_kwargs"] = {}

        default_net_arch: List[int] = [128, 128]
        net_arch: Union[
            List[int],
            Dict[str, List[int]],
            Literal["small", "medium", "large", "extra_large"],
        ] = model_params.get("policy_kwargs", {}).get("net_arch", default_net_arch)

        if "PPO" in self.model_type:
            if isinstance(net_arch, str):
                model_params["policy_kwargs"]["net_arch"] = get_net_arch(
                    self.model_type, net_arch
                )
            elif isinstance(net_arch, list):
                model_params["policy_kwargs"]["net_arch"] = {
                    "pi": net_arch,
                    "vf": net_arch,
                }
            elif isinstance(net_arch, dict):
                pi = (
                    net_arch.get("pi")
                    if isinstance(net_arch.get("pi"), list)
                    else default_net_arch
                )
                vf = (
                    net_arch.get("vf")
                    if isinstance(net_arch.get("vf"), list)
                    else default_net_arch
                )
                model_params["policy_kwargs"]["net_arch"] = {"pi": pi, "vf": vf}
            else:
                logger.warning(
                    "Unexpected net_arch type=%s, using default", type(net_arch)
                )
                model_params["policy_kwargs"]["net_arch"] = {
                    "pi": default_net_arch,
                    "vf": default_net_arch,
                }
        else:
            if isinstance(net_arch, str):
                model_params["policy_kwargs"]["net_arch"] = get_net_arch(
                    self.model_type, net_arch
                )
            elif isinstance(net_arch, list):
                model_params["policy_kwargs"]["net_arch"] = net_arch
            else:
                logger.warning(
                    "Unexpected net_arch type=%s, using default", type(net_arch)
                )
                model_params["policy_kwargs"]["net_arch"] = default_net_arch

        model_params["policy_kwargs"]["activation_fn"] = get_activation_fn(
            model_params.get("policy_kwargs", {}).get("activation_fn", "relu")
        )
        model_params["policy_kwargs"]["optimizer_class"] = get_optimizer_class(
            model_params.get("policy_kwargs", {}).get("optimizer_class", "adamw")
        )

        self._model_params_cache = model_params
        return copy.deepcopy(self._model_params_cache)

    def get_eval_freq(
        self,
        total_timesteps: int,
        hyperopt: bool = False,
        hyperopt_reduction_factor: float = 4.0,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Calculate evaluation frequency (number of steps between evaluations).

        For PPO:
        - Use n_steps from model_params if available
        - Otherwise, select the largest value from PPO_N_STEPS that is <= total_timesteps

        For DQN:
        - Use n_eval_steps divided by n_envs (rounded up)

        For hyperopt:
        - Reduce eval_freq by hyperopt_reduction_factor to speed up trials

        Args:
            total_timesteps: Total training timesteps
            hyperopt: If True, reduce eval_freq for hyperopt
            hyperopt_reduction_factor: Reduction factor for hyperopt (default: 4.0)
            model_params: Model parameters (to get n_steps for PPO)

        Returns:
            int: Evaluation frequency (capped at total_timesteps)
        """
        if total_timesteps <= 0:
            return 1
        if "PPO" in self.model_type:
            eval_freq: Optional[int] = None
            if model_params:
                n_steps = model_params.get("n_steps")
                if isinstance(n_steps, int) and n_steps > 0:
                    eval_freq = max(1, n_steps)
            if eval_freq is None:
                eval_freq = next(
                    (
                        step
                        for step in sorted(PPO_N_STEPS, reverse=True)
                        if step <= total_timesteps
                    ),
                    PPO_N_STEPS[0],
                )
        else:
            eval_freq = max(1, (self.n_eval_steps + self.n_envs - 1) // self.n_envs)

        if hyperopt and hyperopt_reduction_factor > 1.0:
            eval_freq = max(1, int(round(eval_freq / hyperopt_reduction_factor)))

        return min(eval_freq, total_timesteps)

    def get_callbacks(
        self,
        eval_env: BaseEnvironment,
        eval_freq: int,
        data_path: str,
        trial: Optional[Trial] = None,
    ) -> List[BaseCallback]:
        """
        Get the model specific callbacks
        """
        callbacks: List[BaseCallback] = []
        no_improvement_callback = None
        rollout_plot_callback = None
        verbose = self.get_model_params().get("verbose", 0)

        if self.max_no_improvement_evals:
            no_improvement_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=self.max_no_improvement_evals,
                min_evals=self.min_evals,
                verbose=verbose,
            )

        if self.activate_tensorboard:
            info_callback = InfoMetricsCallback(
                actions=Actions,
                throttle=self.rl_config.get("tensorboard_throttle", 1),
                verbose=verbose,
            )
            callbacks.append(info_callback)
            if self.plot_new_best:
                rollout_plot_callback = RolloutPlotCallback(verbose=verbose)

        if self.rl_config.get("progress_bar", False):
            self.progressbar_callback = ProgressBarCallback()
            callbacks.append(self.progressbar_callback)

        use_masking = self.action_masking and is_masking_supported(eval_env)
        if not trial:
            self.eval_callback = MaskableEvalCallback(
                eval_env,
                n_eval_episodes=self.n_eval_episodes,
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                best_model_save_path=data_path,
                use_masking=use_masking,
                callback_on_new_best=rollout_plot_callback,
                callback_after_eval=no_improvement_callback,
                verbose=verbose,
            )
            callbacks.append(self.eval_callback)
        else:
            trial_data_path = f"{data_path}/hyperopt/trial-{trial.number}"
            self.optuna_eval_callback = MaskableTrialEvalCallback(
                eval_env,
                trial,
                n_eval_episodes=self.n_eval_episodes,
                eval_freq=eval_freq,
                deterministic=True,
                render=False,
                best_model_save_path=trial_data_path,
                use_masking=use_masking,
                verbose=verbose,
            )
            callbacks.append(self.optuna_eval_callback)
        return callbacks

    def fit(
        self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        Model fitting method
        :param data_dictionary: dict = common data dictionary containing all train/test features/labels/weights.
        :param dk: FreqaiDatakitchen = data kitchen for current pair.
        :return:
        model Any = trained model to be used for inference in dry/live/backtesting
        """
        train_df = data_dictionary.get("train_features")
        train_timesteps = len(train_df)
        if train_timesteps <= 0:
            raise ValueError("train_features dataframe has zero length")
        test_df = data_dictionary.get("test_features")
        eval_timesteps = len(test_df)
        train_cycles = max(1, int(self.rl_config.get("train_cycles", 25)))
        total_timesteps = (
            (train_timesteps * train_cycles + self.n_envs - 1) // self.n_envs
        ) * self.n_envs
        train_days = steps_to_days(train_timesteps, self.config.get("timeframe"))
        eval_days = steps_to_days(eval_timesteps, self.config.get("timeframe"))
        total_days = steps_to_days(total_timesteps, self.config.get("timeframe"))

        logger.info("Model: %s", self.model_type)
        logger.info(
            "Train: %s steps (%s days), %s cycles, %s env(s) -> total %s steps (%s days)",
            train_timesteps,
            train_days,
            train_cycles,
            self.n_envs,
            total_timesteps,
            total_days,
        )
        logger.info(
            "Eval: %s steps (%s days), %s episodes, %s env(s)",
            eval_timesteps,
            eval_days,
            self.n_eval_episodes,
            self.n_eval_envs,
        )
        logger.info("Multiprocessing: %s", self.multiprocessing)
        logger.info("Eval multiprocessing: %s", self.eval_multiprocessing)
        logger.info("Frame stacking: %s", self.frame_stacking)
        logger.info("Action masking: %s", self.action_masking)
        logger.info("Recurrent: %s", self.recurrent)
        logger.info("Hyperopt: %s", self.hyperopt)

        start_time = time.time()
        if self.hyperopt:
            best_params = self.optimize(dk, total_timesteps)
            if best_params is None:
                logger.error(
                    "Hyperopt failed. Using default configured model params instead"
                )
                best_params = self.get_model_params()
            model_params = best_params
        else:
            model_params = self.get_model_params()
        logger.info("%s params: %s", self.model_type, model_params)

        if "PPO" in self.model_type:
            n_steps = model_params.get("n_steps", 0)
            min_timesteps = 2 * n_steps * self.n_envs
            if total_timesteps <= min_timesteps:
                logger.warning(
                    "total_timesteps=%s is less than or equal to 2*n_steps*n_envs=%s. This may lead to suboptimal training results for model %s",
                    total_timesteps,
                    min_timesteps,
                    self.model_type,
                )
            if n_steps > 0:
                rollout = n_steps * self.n_envs
                aligned_total_timesteps = (
                    (total_timesteps + rollout - 1) // rollout
                ) * rollout
                if aligned_total_timesteps != total_timesteps:
                    total_timesteps = aligned_total_timesteps
                    logger.info(
                        "Train: aligned total %s steps (%s days) for model %s",
                        total_timesteps,
                        steps_to_days(total_timesteps, self.config.get("timeframe")),
                        self.model_type,
                    )

        if self.activate_tensorboard:
            tensorboard_log_path = Path(
                self.full_path / "tensorboard" / Path(dk.data_path).name
            )
        else:
            tensorboard_log_path = None

        # Rebuild train and eval environments before training to sync model parameters
        prices_train, prices_test = self.build_ohlc_price_dataframes(
            dk.data_dictionary, dk.pair, dk
        )
        self.set_train_and_eval_environments(
            dk.data_dictionary, prices_train, prices_test, dk
        )

        model = self.get_init_model(dk.pair)
        if model is not None:
            logger.info(
                "Continual training activated: starting training from previously trained model state"
            )
            model.set_env(self.train_env)
        else:
            model = self.MODELCLASS(
                self.policy_type,
                self.train_env,
                tensorboard_log=tensorboard_log_path,
                **model_params,
            )

        eval_freq = self.get_eval_freq(total_timesteps, model_params=model_params)
        callbacks = self.get_callbacks(self.eval_env, eval_freq, str(dk.data_path))
        try:
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
        except KeyboardInterrupt:
            pass
        finally:
            if self.progressbar_callback:
                self.progressbar_callback.on_training_end()
            self.close_envs()
            if hasattr(model, "env") and model.env is not None:
                model.env.close()
        time_spent = time.time() - start_time
        self.dd.update_metric_tracker("fit_time", time_spent, dk.pair)

        model_filename = dk.model_filename if dk.model_filename else "best"
        model_filepath = Path(dk.data_path / f"{model_filename}_model.zip")
        if model_filepath.is_file():
            logger.info("Found best model at %s", model_filepath)
            try:
                best_model = self.MODELCLASS.load(
                    dk.data_path / f"{model_filename}_model"
                )
                return best_model
            except Exception as e:
                logger.error(f"Error at loading best model: {repr(e)}", exc_info=True)

        logger.info(
            "Could not find best model at %s, using final model instead", model_filepath
        )

        return model

    def rl_model_predict(
        self, dataframe: DataFrame, dk: FreqaiDataKitchen, model: Any
    ) -> DataFrame:
        """
        A helper function to make predictions in the Reinforcement learning module.
        :param dataframe: DataFrame = the dataframe of features to make the predictions on
        :param dk: FreqaiDatakitchen = data kitchen for the current pair
        :param model: Any = the trained model used to inference the features.
        """
        add_state_info: bool = self.rl_config.get("add_state_info", False)
        virtual_position: Positions = Positions.Neutral
        virtual_trade_duration: int = 0
        if add_state_info and self.live:
            position, _, trade_duration = self.get_state_info(dk.pair)
            virtual_position = ReforceXY._normalize_position(position)
            virtual_trade_duration = trade_duration
        np_dataframe: NDArray[np.float32] = dataframe.to_numpy(
            dtype=np.float32, copy=False
        )
        n = np_dataframe.shape[0]
        window_size: int = self.CONV_WIDTH
        frame_stacking: int = self.frame_stacking
        frame_stacking_enabled: bool = bool(frame_stacking) and frame_stacking > 1
        inference_masking: bool = self.action_masking and self.inference_masking

        if window_size <= 0 or n < window_size:
            return DataFrame(
                {label: [np.nan] * n for label in dk.label_list}, index=dataframe.index
            )

        def _update_virtual_position(action: int, position: Positions) -> Positions:
            if action == Actions.Long_enter.value and position == Positions.Neutral:
                return Positions.Long
            if action == Actions.Short_enter.value and position == Positions.Neutral:
                return Positions.Short
            if action == Actions.Long_exit.value and position == Positions.Long:
                return Positions.Neutral
            if action == Actions.Short_exit.value and position == Positions.Short:
                return Positions.Neutral
            return position

        def _update_virtual_trade_duration(
            virtual_position: Positions,
            previous_virtual_position: Positions,
            current_virtual_trade_duration: int,
        ) -> int:
            if virtual_position != Positions.Neutral:
                if previous_virtual_position == Positions.Neutral:
                    return 1
                else:
                    return current_virtual_trade_duration + 1
            return 0

        frame_buffer = deque(maxlen=frame_stacking if frame_stacking_enabled else None)
        zero_frame: Optional[NDArray[np.float32]] = None
        lstm_states: Optional[Tuple[NDArray[np.float32], NDArray[np.float32]]] = None
        episode_start = np.array([True], dtype=bool)

        def _predict(start_idx: int) -> int:
            nonlocal zero_frame, lstm_states, episode_start
            end_idx: int = start_idx + window_size
            np_observation = np_dataframe[start_idx:end_idx, :]
            action_masks_param: Dict[str, Any] = {}

            if add_state_info:
                if self.live:
                    position, pnl, trade_duration = self.get_state_info(dk.pair)
                    position = ReforceXY._normalize_position(position)
                    state_block = np.tile(
                        np.array(
                            [float(pnl), float(position.value), float(trade_duration)],
                            dtype=np.float32,
                        ),
                        (np_observation.shape[0], 1),
                    )
                    action_mask_position = position
                else:
                    state_block = np.tile(
                        np.array(
                            [
                                0.0,
                                float(virtual_position.value),
                                float(virtual_trade_duration),
                            ],
                            dtype=np.float32,
                        ),
                        (np_observation.shape[0], 1),
                    )
                    action_mask_position = virtual_position
                np_observation = np.concatenate([np_observation, state_block], axis=1)
            else:
                action_mask_position = virtual_position

            if frame_stacking_enabled:
                frame_buffer.append(np_observation)
                if len(frame_buffer) < frame_stacking:
                    pad_count = frame_stacking - len(frame_buffer)
                    if zero_frame is None:
                        zero_frame = np.zeros_like(np_observation, dtype=np.float32)
                    fb_padded = [zero_frame] * pad_count + list(frame_buffer)
                else:
                    fb_padded = list(frame_buffer)
                stacked_observations = np.concatenate(fb_padded, axis=1)
                observations = stacked_observations.reshape(
                    1, stacked_observations.shape[0], stacked_observations.shape[1]
                )
            else:
                observations = np_observation.reshape(
                    1, np_observation.shape[0], np_observation.shape[1]
                )

            if inference_masking:
                action_masks_param["action_masks"] = ReforceXY.get_action_masks(
                    self.can_short, action_mask_position
                )

            if self.recurrent:
                action, lstm_states = model.predict(
                    observations,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                    **action_masks_param,
                )
                episode_start[:] = False
            else:
                action, _ = model.predict(
                    observations, deterministic=True, **action_masks_param
                )

            return int(action)

        predicted_actions: List[int] = []
        for start_idx in range(0, n - window_size + 1):
            action = _predict(start_idx)
            predicted_actions.append(action)
            previous_virtual_position = virtual_position
            virtual_position = _update_virtual_position(action, virtual_position)
            virtual_trade_duration = _update_virtual_trade_duration(
                virtual_position,
                previous_virtual_position,
                virtual_trade_duration,
            )

        pad_count = max(0, n - len(predicted_actions))
        actions_list = ([np.nan] * pad_count) + predicted_actions
        actions_df = DataFrame({"action": actions_list}, index=dataframe.index)

        return DataFrame({label: actions_df["action"] for label in dk.label_list})

    @staticmethod
    def study_delete(study_name: str, storage: BaseStorage) -> None:
        try:
            delete_study(study_name=study_name, storage=storage)
        except Exception:
            pass

    def get_storage(self, pair: str) -> BaseStorage:
        """
        Get the storage for Optuna
        """
        storage_dir = self.full_path
        storage_filename = f"optuna-{pair.split('/')[0]}"
        storage_backend = self.rl_config_optuna.get("storage", "sqlite")
        if storage_backend == "sqlite":
            storage = RDBStorage(
                url=f"sqlite:///{storage_dir}/{storage_filename}.sqlite",
                heartbeat_interval=60,
                failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
            )
        elif storage_backend == "file":
            storage = JournalStorage(
                JournalFileBackend(f"{storage_dir}/{storage_filename}.log")
            )
        else:
            raise ValueError(
                f"Unsupported storage backend: {storage_backend}. Supported backends are: 'sqlite' and 'file'"
            )
        return storage

    @staticmethod
    def study_has_best_trial(study: Optional[Study]) -> bool:
        if study is None:
            return False
        try:
            _ = study.best_trial
            return True
        except (ValueError, KeyError):
            return False

    def optimize(
        self, dk: FreqaiDataKitchen, total_timesteps: int
    ) -> Optional[Dict[str, Any]]:
        """
        Runs hyperparameter optimization using Optuna and returns the best hyperparameters found merged with the user defined parameters
        """
        identifier = self.freqai_info.get("identifier", "no_id_provided")
        study_name = f"{identifier}-{dk.pair}"
        storage = self.get_storage(dk.pair)
        continuous = self.rl_config_optuna.get("continuous", False)
        if continuous:
            ReforceXY.study_delete(study_name, storage)
        if "PPO" in self.model_type:
            resource_eval_freq = min(PPO_N_STEPS)
        else:
            resource_eval_freq = self.get_eval_freq(total_timesteps, hyperopt=True)
        reduction_factor = 3
        max_resource = max(
            reduction_factor * 2, (total_timesteps // self.n_envs) // resource_eval_freq
        )
        min_resource = max(1, max_resource // reduction_factor)
        study: Study = create_study(
            study_name=study_name,
            sampler=TPESampler(
                n_startup_trials=self.optuna_n_startup_trials,
                multivariate=True,
                group=True,
                seed=self.rl_config_optuna.get("seed", 42),
            ),
            pruner=HyperbandPruner(
                min_resource=min_resource,
                max_resource=max_resource,
                reduction_factor=reduction_factor,
            ),
            direction=StudyDirection.MAXIMIZE,
            storage=storage,
            load_if_exists=not continuous,
        )
        if self.rl_config_optuna.get("warm_start", False):
            best_trial_params = self.load_best_trial_params(dk.pair)
            if best_trial_params:
                study.enqueue_trial(best_trial_params)
        hyperopt_failed = False
        start_time = time.time()
        try:
            study.optimize(
                lambda trial: self.objective(trial, dk, total_timesteps),
                n_trials=self.optuna_n_trials,
                timeout=(
                    hours_to_seconds(self.optuna_timeout_hours)
                    if self.optuna_timeout_hours
                    else None
                ),
                gc_after_trial=True,
                show_progress_bar=self.rl_config.get("progress_bar", False),
                # SB3 is not fully thread safe
                n_jobs=1,
            )
        except KeyboardInterrupt:
            pass
        except Exception as e:
            time_spent = time.time() - start_time
            logger.error(
                f"Hyperopt {study_name} failed ({time_spent:.2f} secs): {repr(e)}",
                exc_info=True,
            )
            hyperopt_failed = True
        time_spent = time.time() - start_time
        study_has_best_trial = ReforceXY.study_has_best_trial(study)
        if not study_has_best_trial:
            logger.error(
                f"Hyperopt {study_name} failed ({time_spent:.2f} secs): no study best trial found"
            )
            hyperopt_failed = True

        if hyperopt_failed:
            best_trial_params = self.load_best_trial_params(dk.pair)
            if best_trial_params is None:
                logger.error(
                    f"Hyperopt {study_name} failed ({time_spent:.2f} secs): no previously saved best trial params found"
                )
                return None
        else:
            best_trial_params = study.best_trial.params

        logger.info(
            "------------ Hyperopt %s results (%.2f secs) ------------",
            study_name,
            time_spent,
        )
        if study_has_best_trial:
            logger.info(
                "Best trial: %s. Score: %s",
                study.best_trial.number,
                study.best_trial.value,
            )
        logger.info("Best trial params: %s", best_trial_params)
        logger.info("-------------------------------------------------------")

        self.save_best_trial_params(best_trial_params, dk.pair)

        return deepmerge(
            self.get_model_params(),
            convert_optuna_params_to_model_params(self.model_type, best_trial_params),
        )

    def save_best_trial_params(
        self, best_trial_params: Dict[str, Any], pair: str
    ) -> None:
        """
        Save the best trial hyperparameters found during hyperparameter optimization
        """
        best_trial_params_filename = f"hyperopt-best-params-{pair.split('/')[0]}"
        best_trial_params_path = Path(
            self.full_path / f"{best_trial_params_filename}.json"
        )
        logger.info(
            "%s: saving best params to %s JSON file", pair, best_trial_params_path
        )
        try:
            with best_trial_params_path.open("w", encoding="utf-8") as write_file:
                json.dump(best_trial_params, write_file, indent=4)
        except Exception as e:
            logger.error(
                f"Error saving best trial params to {best_trial_params_path}: {repr(e)}",
                exc_info=True,
            )
            raise

    def load_best_trial_params(self, pair: str) -> Optional[Dict[str, Any]]:
        """
        Load the best trial hyperparameters found and saved during hyperparameter optimization
        """
        best_trial_params_filename = f"hyperopt-best-params-{pair.split('/')[0]}"
        best_trial_params_path = Path(
            self.full_path / f"{best_trial_params_filename}.json"
        )
        if best_trial_params_path.is_file():
            logger.info(
                "%s: loading best params from %s JSON file",
                pair,
                best_trial_params_path,
            )
            with best_trial_params_path.open("r", encoding="utf-8") as read_file:
                best_trial_params = json.load(read_file)
            return best_trial_params
        return None

    def _get_train_and_eval_environments(
        self,
        dk: FreqaiDataKitchen,
        train_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        prices_train: Optional[DataFrame] = None,
        prices_test: Optional[DataFrame] = None,
        seed: Optional[int] = None,
        env_info: Optional[Dict[str, Any]] = None,
        trial: Optional[Trial] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[VecEnv, VecEnv]:
        if (
            train_df is None
            or test_df is None
            or prices_train is None
            or prices_test is None
        ):
            train_df = dk.data_dictionary["train_features"]
            test_df = dk.data_dictionary["test_features"]
            prices_train, prices_test = self.build_ohlc_price_dataframes(
                dk.data_dictionary, dk.pair, dk
            )
        seed = self.get_model_params().get("seed", 42) if seed is None else seed
        if trial is not None:
            seed += trial.number
        set_random_seed(seed)
        env_info: Dict[str, Any] = (
            self.pack_env_dict(dk.pair, model_params) if env_info is None else env_info
        )
        env_prefix = f"trial_{trial.number}_" if trial is not None else ""

        train_fns = [
            make_env(
                MyRLEnv,
                f"{env_prefix}train_env{i}",
                i,
                seed,
                train_df,
                prices_train,
                env_info=env_info,
            )
            for i in range(self.n_envs)
        ]
        eval_fns = [
            make_env(
                MyRLEnv,
                f"{env_prefix}eval_env{i}",
                i,
                seed + 10_000,
                test_df,
                prices_test,
                env_info=env_info,
            )
            for i in range(self.n_eval_envs)
        ]

        if self.multiprocessing and self.n_envs > 1:
            train_env = SubprocVecEnv(train_fns, start_method="spawn")
        else:
            train_env = DummyVecEnv(train_fns)
        if self.eval_multiprocessing and self.n_eval_envs > 1:
            eval_env = SubprocVecEnv(eval_fns, start_method="spawn")
        else:
            eval_env = DummyVecEnv(eval_fns)

        if bool(self.frame_stacking) and self.frame_stacking > 1:
            train_env = VecFrameStack(train_env, n_stack=self.frame_stacking)
            eval_env = VecFrameStack(eval_env, n_stack=self.frame_stacking)

        train_env = VecMonitor(train_env)
        eval_env = VecMonitor(eval_env)

        return train_env, eval_env

    def get_optuna_params(self, trial: Trial) -> Dict[str, Any]:
        if "RecurrentPPO" in self.model_type:
            return sample_params_recurrentppo(trial)
        elif "PPO" in self.model_type:
            return sample_params_ppo(trial)
        elif "QRDQN" in self.model_type:
            return sample_params_qrdqn(trial)
        elif "DQN" in self.model_type:
            return sample_params_dqn(trial)
        else:
            raise NotImplementedError(f"{self.model_type} not supported for hyperopt")

    def objective(
        self, trial: Trial, dk: FreqaiDataKitchen, total_timesteps: int
    ) -> float:
        """
        Objective function for Optuna trials hyperparameter optimization
        """
        logger.info("------------ Hyperopt trial %d ------------", trial.number)

        params = self.get_optuna_params(trial)

        if "PPO" in self.model_type:
            n_steps = params.get("n_steps")
            if n_steps * self.n_envs > total_timesteps:
                raise TrialPruned(
                    f"{n_steps=} * n_envs={self.n_envs}={n_steps * self.n_envs} is greater than {total_timesteps=}"
                )
            batch_size = params.get("batch_size")
            if (n_steps * self.n_envs) % batch_size != 0:
                raise TrialPruned(
                    f"{n_steps=} * {self.n_envs=} = {n_steps * self.n_envs} is not divisible by {batch_size=}"
                )

        if "DQN" in self.model_type:
            gradient_steps = params.get("gradient_steps")
            if isinstance(gradient_steps, int) and gradient_steps <= 0:
                raise TrialPruned(f"{gradient_steps=} is negative or zero")
            batch_size = params.get("batch_size")
            buffer_size = params.get("buffer_size")
            if (batch_size * gradient_steps) > buffer_size:
                raise TrialPruned(
                    f"{batch_size=} * {gradient_steps=}={batch_size * gradient_steps} is greater than {buffer_size=}"
                )
            learning_starts = params.get("learning_starts")
            if learning_starts > buffer_size:
                raise TrialPruned(f"{learning_starts=} is greater than {buffer_size=}")

        # Ensure that the sampled parameters take precedence
        params = deepmerge(self.get_model_params(), params)
        params["seed"] = params.get("seed", 42) + trial.number
        logger.info("Trial %s params: %s", trial.number, params)

        if "PPO" in self.model_type:
            n_steps = params.get("n_steps", 0)
            if n_steps > 0:
                rollout = n_steps * self.n_envs
                aligned_total_timesteps = (
                    (total_timesteps + rollout - 1) // rollout
                ) * rollout
                if aligned_total_timesteps != total_timesteps:
                    total_timesteps = aligned_total_timesteps

        nan_encountered = False

        if self.activate_tensorboard:
            tensorboard_log_path = Path(
                self.full_path
                / "tensorboard"
                / Path(dk.data_path).name
                / "hyperopt"
                / f"trial-{trial.number}"
            )
        else:
            tensorboard_log_path = None

        train_env, eval_env = self._get_train_and_eval_environments(
            dk, trial=trial, model_params=params
        )

        model = self.MODELCLASS(
            self.policy_type,
            train_env,
            tensorboard_log=tensorboard_log_path,
            **params,
        )

        eval_freq = self.get_eval_freq(
            total_timesteps, hyperopt=True, model_params=params
        )
        callbacks = self.get_callbacks(eval_env, eval_freq, str(dk.data_path), trial)
        try:
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
        except AssertionError:
            logger.warning("Optuna encountered NaN (AssertionError)")
            nan_encountered = True
        except ValueError as e:
            if any(x in str(e).lower() for x in ("nan", "inf")):
                logger.warning("Optuna encountered NaN/Inf (ValueError): %r", e)
                nan_encountered = True
            else:
                raise
        except FloatingPointError as e:
            logger.warning("Optuna encountered NaN/Inf (FloatingPointError): %r", e)
            nan_encountered = True
        except RuntimeError as e:
            if any(x in str(e).lower() for x in ("nan", "inf")):
                logger.warning("Optuna encountered NaN/Inf (RuntimeError): %r", e)
                nan_encountered = True
            else:
                raise
        finally:
            if self.progressbar_callback:
                self.progressbar_callback.on_training_end()
            train_env.close()
            eval_env.close()
            if hasattr(model, "env") and model.env is not None:
                model.env.close()
            del model, train_env, eval_env

        if nan_encountered:
            raise TrialPruned("NaN encountered during training")

        if self.optuna_eval_callback.is_pruned:
            raise TrialPruned()

        return self.optuna_eval_callback.best_mean_reward

    def close_envs(self) -> None:
        """
        Closes the training and evaluation environments if they are open
        """
        if self.train_env:
            try:
                self.train_env.close()
            finally:
                self.train_env = None
        if self.eval_env:
            try:
                self.eval_env.close()
            finally:
                self.eval_env = None


def make_env(
    MyRLEnv: Type[BaseEnvironment],
    env_id: str,
    rank: int,
    seed: int,
    df: DataFrame,
    price: DataFrame,
    env_info: Dict[str, Any],
) -> Callable[[], BaseEnvironment]:
    """
    Utility function for multiprocessed env.

    :param MyRLEnv: (Type[BaseEnvironment]) environment class to instantiate
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param df: (DataFrame) feature dataframe for the environment
    :param price: (DataFrame) aligned price dataframe
    :param env_info: (dict) all required arguments to instantiate the environment
    :return:
    (Callable[[], BaseEnvironment]) closure that when called instantiates and returns the environment
    """

    def _init() -> BaseEnvironment:
        return MyRLEnv(df=df, prices=price, id=env_id, seed=seed + rank, **env_info)

    return _init


MyRLEnv: Type[BaseEnvironment]


class MyRLEnv(Base5ActionRLEnv):
    """
    Env
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_observation_space()
        self.action_masking: bool = self.rl_config.get("action_masking", False)

        # === INTERNAL STATE ===
        self._last_closed_position: Optional[Positions] = None
        self._last_closed_trade_tick: int = 0
        self._max_unrealized_profit: float = -np.inf
        self._min_unrealized_profit: float = np.inf
        self._last_potential: float = 0.0
        # === PBRS INSTRUMENTATION ===
        self._last_prev_potential: float = 0.0
        self._last_next_potential: float = 0.0
        self._last_reward_shaping: float = 0.0
        self._total_reward_shaping: float = 0.0
        self._last_invalid_penalty: float = 0.0
        self._last_idle_penalty: float = 0.0
        self._last_hold_penalty: float = 0.0
        self._last_exit_reward: float = 0.0
        self._last_entry_additive: float = 0.0
        self._total_entry_additive: float = 0.0
        self._last_exit_additive: float = 0.0
        self._total_exit_additive: float = 0.0
        model_reward_parameters = self.rl_config.get("model_reward_parameters", {})
        self.max_trade_duration_candles: int = int(
            model_reward_parameters.get(
                "max_trade_duration_candles",
                128,
            )
        )
        self.max_idle_duration_candles: int = int(
            model_reward_parameters.get(
                "max_idle_duration_candles",
                ReforceXY.DEFAULT_IDLE_DURATION_MULTIPLIER
                * self.max_trade_duration_candles,
            )
        )
        # === PBRS COMMON PARAMETERS ===
        potential_gamma = model_reward_parameters.get("potential_gamma")
        if potential_gamma is None:
            logger.warning("potential_gamma not specified; defaulting to 0.95")
            self._potential_gamma = 0.95
        else:
            self._potential_gamma = float(potential_gamma)
        # Validate potential_gamma range (0 <= gamma <= 1)
        if not (0.0 <= self._potential_gamma <= 1.0):
            original_gamma = self._potential_gamma
            self._potential_gamma = min(1.0, max(0.0, self._potential_gamma))
            logger.warning(
                "potential_gamma=%s is outside [0,1]; clamped to %s",
                original_gamma,
                self._potential_gamma,
            )
        # === EXIT POTENTIAL MODE ===
        # exit_potential_mode options:
        #   'canonical'           -> Φ(s')=0 (preserves invariance, disables additives)
        #   'non_canonical'       -> Φ(s')=0 (allows additives, breaks invariance)
        #   'progressive_release' -> Φ(s')=Φ(s)*(1-decay_factor)
        #   'spike_cancel'        -> Φ(s')=Φ(s)/γ (Δ ≈ 0, cancels shaping)
        #   'retain_previous'     -> Φ(s')=Φ(s)
        self._exit_potential_mode = str(
            model_reward_parameters.get("exit_potential_mode", "canonical")
        )
        _allowed_exit_modes = {
            "canonical",
            "non_canonical",
            "progressive_release",
            "spike_cancel",
            "retain_previous",
        }
        if self._exit_potential_mode not in _allowed_exit_modes:
            logger.warning(
                "Unknown exit_potential_mode '%s'; defaulting to 'canonical'",
                self._exit_potential_mode,
            )
            self._exit_potential_mode = "canonical"
        self._exit_potential_decay: float = float(
            model_reward_parameters.get("exit_potential_decay", 0.5)
        )
        # === ENTRY ADDITIVE (non-PBRS additive term) ===
        self._entry_additive_enabled: bool = bool(
            model_reward_parameters.get("entry_additive_enabled", False)
        )
        self._entry_additive_scale: float = float(
            model_reward_parameters.get("entry_additive_scale", 1.0)
        )
        self._entry_additive_gain: float = float(
            model_reward_parameters.get("entry_additive_gain", 1.0)
        )
        self._entry_additive_transform_pnl: str = str(
            model_reward_parameters.get("entry_additive_transform_pnl", "tanh")
        )
        self._entry_additive_transform_duration: str = str(
            model_reward_parameters.get("entry_additive_transform_duration", "tanh")
        )
        # === HOLD POTENTIAL (PBRS function Φ) ===
        self._hold_potential_enabled: bool = bool(
            model_reward_parameters.get("hold_potential_enabled", True)
        )
        self._hold_potential_scale: float = float(
            model_reward_parameters.get("hold_potential_scale", 1.0)
        )
        self._hold_potential_gain: float = float(
            model_reward_parameters.get("hold_potential_gain", 1.0)
        )
        self._hold_potential_transform_pnl: str = str(
            model_reward_parameters.get("hold_potential_transform_pnl", "tanh")
        )
        self._hold_potential_transform_duration: str = str(
            model_reward_parameters.get("hold_potential_transform_duration", "tanh")
        )
        # === EXIT ADDITIVE (non-PBRS additive term) ===
        self._exit_additive_enabled: bool = bool(
            model_reward_parameters.get("exit_additive_enabled", False)
        )
        self._exit_additive_scale: float = float(
            model_reward_parameters.get("exit_additive_scale", 1.0)
        )
        self._exit_additive_gain: float = float(
            model_reward_parameters.get("exit_additive_gain", 1.0)
        )
        self._exit_additive_transform_pnl: str = str(
            model_reward_parameters.get("exit_additive_transform_pnl", "tanh")
        )
        self._exit_additive_transform_duration: str = str(
            model_reward_parameters.get("exit_additive_transform_duration", "tanh")
        )
        # === PBRS INVARIANCE CHECKS ===
        if self._exit_potential_mode == "canonical":
            if self._entry_additive_enabled or self._exit_additive_enabled:
                logger.info(
                    "PBRS canonical mode: additive rewards disabled with Φ(terminal)=0. PBRS invariance is preserved. "
                    "To use additive rewards, set exit_potential_mode='non_canonical'."
                )
                self._entry_additive_enabled = False
                self._exit_additive_enabled = False
        elif self._exit_potential_mode == "non_canonical":
            if self._entry_additive_enabled or self._exit_additive_enabled:
                logger.info(
                    "PBRS non-canonical mode: additive rewards enabled with Φ(terminal)=0. PBRS invariance is intentionally broken."
                )

        if MyRLEnv.is_unsupported_pbrs_config(
            self._hold_potential_enabled, getattr(self, "add_state_info", False)
        ):
            logger.warning(
                "PBRS: hold_potential_enabled=True and add_state_info=False is unsupported. Automatically enabling add_state_info=True."
            )
            self.add_state_info = True

        # === PNL TARGET VALIDATION ===
        pnl_target = self.profit_aim * self.rr
        if MyRLEnv._is_invalid_pnl_target(pnl_target):
            raise ValueError(
                f"Invalid pnl_target={pnl_target:.12g} computed from profit_aim={self.profit_aim:.12g} and rr={self.rr:.12g}"
            )
        self._pnl_target = pnl_target

    def _get_next_position(self, action: int) -> Positions:
        if action == Actions.Long_enter.value and self._position == Positions.Neutral:
            return Positions.Long
        if (
            action == Actions.Short_enter.value
            and self._position == Positions.Neutral
            and self.can_short
        ):
            return Positions.Short
        if action == Actions.Long_exit.value and self._position == Positions.Long:
            return Positions.Neutral
        if action == Actions.Short_exit.value and self._position == Positions.Short:
            return Positions.Neutral
        return self._position

    def _get_next_transition_state(
        self,
        action: int,
        trade_duration: float,
        pnl: float,
    ) -> Tuple[Positions, int, float]:
        """Compute next transition state tuple."""
        next_position = self._get_next_position(action)
        # Entry
        if self._position == Positions.Neutral and next_position in (
            Positions.Long,
            Positions.Short,
        ):
            return next_position, 0, pnl
        # Exit
        if (
            self._position in (Positions.Long, Positions.Short)
            and next_position == Positions.Neutral
        ):
            return next_position, 0, 0.0
        # Hold
        if self._position in (Positions.Long, Positions.Short) and next_position in (
            Positions.Long,
            Positions.Short,
        ):
            return next_position, int(trade_duration) + 1, pnl
        # Neutral self-loop
        return next_position, 0, 0.0

    @staticmethod
    def _is_invalid_pnl_target(pnl_target: float) -> bool:
        """Return True when pnl_target is non-finite, <= 0, or effectively zero within tolerance."""
        return (
            (not np.isfinite(pnl_target))
            or (pnl_target <= 0.0)
            or np.isclose(pnl_target, 0.0)
        )

    def _compute_pnl_duration_signal(
        self,
        *,
        enabled: bool,
        require_position: bool,
        position: Positions,
        pnl: float,
        pnl_target: float,
        duration_ratio: float,
        scale: float,
        gain: float,
        transform_pnl: str,
        transform_duration: str,
    ) -> float:
        """Generic bounded bi-component signal combining PnL and duration.

        Shared logic for:
        - Hold potential Φ(s)
        - Entry additive
        - Exit additive

        Parameters
        ----------
        enabled : bool
            Whether this signal is active
        require_position : bool
            If True, only compute when position in (Long, Short)
        position : Positions
            Current position
        pnl : float
            PnL used for normalization
        pnl_target : float
            Target PnL normalizer (>0)
        duration_ratio : float
            Raw duration ratio
        scale : float
            Output scaling factor
        gain : float
            Gain multiplier before transform
        transform_pnl : str
            Transform name for PnL component
        transform_duration : str
            Transform name for duration component

        Returns
        -------
        float
            Bounded signal in [-scale, scale]
        """
        if not enabled:
            return 0.0
        if require_position and position not in (Positions.Long, Positions.Short):
            return 0.0

        duration_ratio = 0.0 if duration_ratio < 0.0 else duration_ratio
        if duration_ratio > 1.0:
            duration_ratio = 1.0

        try:
            pnl_ratio = pnl / pnl_target
        except Exception:
            return 0.0

        pnl_term = self._potential_transform(transform_pnl, gain * pnl_ratio)
        dur_term = self._potential_transform(transform_duration, gain * duration_ratio)
        value = scale * 0.5 * (pnl_term + dur_term)
        return float(value) if np.isfinite(value) else 0.0

    def _compute_hold_potential(
        self,
        position: Positions,
        duration_ratio: float,
        pnl: float,
        pnl_target: float,
    ) -> float:
        """Compute PBRS potential Φ(s) for position holding states.

        See ``_apply_potential_shaping`` for complete PBRS documentation.
        """
        return self._compute_pnl_duration_signal(
            enabled=self._hold_potential_enabled,
            require_position=True,
            position=position,
            pnl=pnl,
            pnl_target=pnl_target,
            duration_ratio=duration_ratio,
            scale=self._hold_potential_scale,
            gain=self._hold_potential_gain,
            transform_pnl=self._hold_potential_transform_pnl,
            transform_duration=self._hold_potential_transform_duration,
        )

    def _compute_exit_additive(
        self,
        pnl: float,
        pnl_target: float,
        duration_ratio: float,
    ) -> float:
        """Compute exit additive reward for position exit transitions.

        See ``_apply_potential_shaping`` for complete PBRS documentation.
        """
        return self._compute_pnl_duration_signal(
            enabled=self._exit_additive_enabled,
            require_position=False,
            position=Positions.Neutral,
            pnl=pnl,
            pnl_target=pnl_target,
            duration_ratio=duration_ratio,
            scale=self._exit_additive_scale,
            gain=self._exit_additive_gain,
            transform_pnl=self._exit_additive_transform_pnl,
            transform_duration=self._exit_additive_transform_duration,
        )

    def _compute_entry_additive(
        self,
        pnl: float,
        pnl_target: float,
        duration_ratio: float,
    ) -> float:
        """Compute entry additive reward for position entry transitions.

        See ``_apply_potential_shaping`` for complete PBRS documentation.
        """
        return self._compute_pnl_duration_signal(
            enabled=self._entry_additive_enabled,
            require_position=False,
            position=Positions.Neutral,
            pnl=pnl,
            pnl_target=pnl_target,
            duration_ratio=duration_ratio,
            scale=self._entry_additive_scale,
            gain=self._entry_additive_gain,
            transform_pnl=self._entry_additive_transform_pnl,
            transform_duration=self._entry_additive_transform_duration,
        )

    def _potential_transform(self, name: str, x: float) -> float:
        """Apply bounded transform function for potential and additive computations.

        Provides numerical stability by mapping unbounded inputs to bounded outputs
        using various smooth activation functions. Used in both PBRS potentials
        and additive reward calculations.

        Parameters
        ----------
        name : str
            Transform function name: 'tanh', 'softsign', 'arctan', 'sigmoid',
            'asinh', or 'clip'
        x : float
            Input value to transform

        Returns
        -------
        float
            Bounded output in [-1, 1]
        """
        if name == "tanh":
            return math.tanh(x)

        if name == "softsign":
            ax = abs(x)
            return x / (1.0 + ax)

        if name == "arctan":
            return (2.0 / math.pi) * math.atan(x)

        if name == "sigmoid":
            try:
                if x >= 0:
                    exp_neg_x = math.exp(-x)
                    sigma_x = 1.0 / (1.0 + exp_neg_x)
                else:
                    exp_x = math.exp(x)
                    sigma_x = exp_x / (exp_x + 1.0)
                return 2.0 * sigma_x - 1.0
            except OverflowError:
                return 1.0 if x > 0 else -1.0

        if name == "asinh":
            return x / math.hypot(1.0, x)

        if name == "clip":
            return max(-1.0, min(1.0, x))

        logger.warning("Unknown potential transform '%s'; falling back to tanh", name)
        return math.tanh(x)

    def _compute_exit_potential(self, prev_potential: float, gamma: float) -> float:
        """Compute next potential Φ(s') for exit transitions based on exit potential mode.

        See ``_apply_potential_shaping`` for complete PBRS documentation.
        """
        mode = self._exit_potential_mode
        if mode == "canonical" or mode == "non_canonical":
            return 0.0
        if mode == "progressive_release":
            decay = self._exit_potential_decay
            if not np.isfinite(decay) or decay < 0.0:
                decay = 0.0
            if decay > 1.0:
                decay = 1.0
            next_potential = prev_potential * (1.0 - decay)
        elif mode == "spike_cancel":
            if gamma <= 0.0 or not np.isfinite(gamma):
                next_potential = prev_potential
            else:
                next_potential = prev_potential / gamma
        elif mode == "retain_previous":
            next_potential = prev_potential
        else:
            next_potential = 0.0
        if not np.isfinite(next_potential):
            next_potential = 0.0
        return next_potential

    def is_pbrs_invariant_mode(self) -> bool:
        """Return True if current configuration preserves PBRS policy invariance.

        PBRS policy invariance (Ng et al. 1999) requires:
        1. Canonical exit mode: Φ(terminal) = 0
        2. No path-dependent additives: entry_additive = exit_additive = 0

        When True, the shaped policy π'(s) is guaranteed to be equivalent to
        the policy π(s) learned with base rewards only.

        Returns
        -------
        bool
            True if configuration preserves theoretical PBRS invariance
        """
        return self._exit_potential_mode == "canonical" and not (
            self._entry_additive_enabled or self._exit_additive_enabled
        )

    @staticmethod
    def is_unsupported_pbrs_config(
        hold_potential_enabled: bool, add_state_info: bool
    ) -> bool:
        """Return True if PBRS potential relies on hidden state.

        Case: hold_potential enabled while auxiliary state info (pnl, trade_duration) is excluded
        from the observation space (add_state_info=False). In that situation, Φ(s) uses hidden
        variables and PBRS becomes informative, voiding the strict policy invariance guarantee.
        """
        return hold_potential_enabled and not add_state_info

    def _apply_potential_shaping(
        self,
        base_reward: float,
        action: int,
        trade_duration: float,
        max_trade_duration: float,
        pnl: float,
        pnl_target: float,
    ) -> float:
        """Apply potential-based reward shaping (PBRS) (Ng et al. 1999).

        Canonical formula:  R'(s,a,s') = R_base(s,a,s') + γ Φ(s') − Φ(s)

        Notation
        --------
        R_base: base reward; Φ(s)/Φ(s'): potentials (prev/next); γ: shaping discount;
        Δ(s,s') = γΦ(s') − Φ(s); R' = R_base + Δ + optional additives; pnl_ratio = pnl/pnl_target;
        duration_ratio = trade_duration / max_trade_duration (clamped to [0,1]).

        PBRS Theory & Compliance
        ------------------------
        - Ng et al. 1999 (potential-based shaping invariance)
        - Wiewiora et al. 2003 (Φ(terminal)=0 handling)
        - Invariance holds only in canonical mode with additives disabled.

        Architecture & Transitions
        --------------------------
        Three mutually exclusive transition types:

        1. **Entry** (Neutral → Long/Short):
           - Initialize potential Φ for next step: Φ(s') = hold_potential(next_state)
           - PBRS shaping reward: γΦ(s') - Φ(s) where Φ(s)=0 (neutral has no potential)
           - Optional entry additive (non-PBRS additive term, breaks invariance if used)

        2. **Hold** (Long/Short → Long/Short):
           - Standard PBRS: γΦ(s') - Φ(s) where both potentials computed from hold_potential()
           - Φ(s') accounts for updated PnL and trade duration progression

        3. **Exit** (Long/Short → Neutral):
           - **Canonical mode**: Φ(terminal)=0, Δ(s,s') = -Φ(s)
           - **Heuristic modes**: Φ(s') computed by _compute_exit_potential(), Δ(s,s') = γΦ(s')-Φ(s)
           - Optional exit additive (non-PBRS additive term for trade quality summary)

        Potential Function Φ(s)
        -----------------------
        Φ(s) = scale * 0.5 * [T_pnl(g * pnl_ratio) + T_dur(g * duration_ratio)]
        Transforms (bounded in [-1,1]): tanh, softsign, arctan, sigmoid (≈ tanh(0.5x)), asinh, clip.
        Parameters: gain g (sharpens/softens), scale.

        Exit Potential Modes
        --------------------
        **canonical** (PBRS-compliant):
        - Φ(s')=0 for all exit transitions
        - Maintains theoretical invariance guarantees
        - Shaping reward: γ·0-Φ(s) = -Φ(s)
        - Entry/exit additives automatically disabled to preserve invariance

        **non_canonical**:
        - Φ(s')=0 for all exit transitions
        - Entry/exit additives are allowed

        **progressive_release** (heuristic):
        - Φ(s')=Φ(s)*(1-decay_factor), gradual decay
        - Shaping reward: γΦ(s')-Φ(s) = γΦ(s)*(1-d)-Φ(s)

        **spike_cancel** (heuristic):
        - Φ(s')=Φ(s)/γ (γ>0 finite)
        - Shaping reward: γΦ(s')-Φ(s) = γ*(Φ(s)/γ)-Φ(s) = 0

        **retain_previous** (heuristic):
        - Φ(s')=Φ(s), full retention
        - Shaping reward: (γ-1)Φ(s)

        Additive Components & Path Dependence
        ------------------------------------
        **Entry/Exit Additive Terms**: Non-PBRS additive rewards that break invariance
        - Entry additive: Applied at entry transitions, computed via _compute_entry_additive()
        - Exit additive: Applied at exit transitions, computed via _compute_exit_additive()
        - Neither additive persists in stored potential (maintains neutrality)

        **Path Dependence**: Only canonical preserves invariance; others introduce path dependence.

        Invariance & Validation
        -----------------------
        **Theoretical Guarantee**: Canonical + no additives ⇒ Σ_t γ^t Δ_t = 0 (Φ(start)=Φ(end)=0).

        **Deviations from Theory**:
        - Heuristic exit modes violate invariance
        - Entry/exit additives break policy invariance
        - Non-canonical modes may cause path-dependent learning

        **Robustness**:
        - Bounded transforms prevent potential explosion
        - Finite value validation with fallback to 0
        - Terminal state enforcement: Φ(s)=0 when terminated=True
        - All transform functions are strictly bounded in [-1, 1], ensuring numerical stability
        - Bounds: |Φ(s)| ≤ scale ; |Δ(s,s')| ≤ (1+γ)*scale

        Parameters
        ----------
        base_reward : float
            Original reward before shaping
        action : int
            Action taken leading to transition
        trade_duration : float
            Current trade duration in candles
        max_trade_duration : float
            Maximum allowed trade duration
        pnl : float
            Current position PnL
        pnl_target : float
            Target PnL for normalization

        Returns
        -------
        float
            Shaped reward R'(s,a,s') = R_base + Δ(s,s') + optional_additives

        Notes
        -----
        - Canonical mode recommended for invariance
        - Monitor discounted Σ γ^t Δ_t (≈0 per episode canonical)
        - Heuristic exit modes may affect convergence
        - Transform validation delegated to analysis tools
        - Φ reset at exits (canonical) enables telescoping cancellation
        """
        if not self._hold_potential_enabled and not (
            self._entry_additive_enabled or self._exit_additive_enabled
        ):
            return base_reward
        prev_potential = self._last_potential
        next_position, next_trade_duration, next_pnl = self._get_next_transition_state(
            action=action, trade_duration=trade_duration, pnl=pnl
        )
        if max_trade_duration <= 0:
            next_duration_ratio = 0.0
        else:
            next_duration_ratio = next_trade_duration / max_trade_duration

        is_entry = self._position == Positions.Neutral and next_position in (
            Positions.Long,
            Positions.Short,
        )
        is_exit = (
            self._position in (Positions.Long, Positions.Short)
            and next_position == Positions.Neutral
        )
        is_hold = self._position in (
            Positions.Long,
            Positions.Short,
        ) and next_position in (Positions.Long, Positions.Short)

        gamma = self._potential_gamma
        if is_entry:
            if self._hold_potential_enabled:
                potential = self._compute_hold_potential(
                    next_position, next_duration_ratio, next_pnl, pnl_target
                )
                reward_shaping = gamma * potential - prev_potential
                self._last_potential = potential
            else:
                reward_shaping = 0.0
                self._last_potential = 0.0
            self._last_exit_additive = 0.0
            self._last_entry_additive = 0.0
            entry_additive = 0.0
            if self._entry_additive_enabled and not self.is_pbrs_invariant_mode():
                entry_additive = self._compute_entry_additive(
                    pnl=next_pnl,
                    pnl_target=pnl_target,
                    duration_ratio=next_duration_ratio,
                )
                self._last_entry_additive = float(entry_additive)
                self._total_entry_additive += float(entry_additive)
            self._last_reward_shaping = float(reward_shaping)
            self._total_reward_shaping += float(reward_shaping)
            self._last_prev_potential = float(prev_potential)
            self._last_next_potential = float(self._last_potential)
            return base_reward + reward_shaping + entry_additive
        elif is_hold:
            if self._hold_potential_enabled:
                potential = self._compute_hold_potential(
                    next_position, next_duration_ratio, next_pnl, pnl_target
                )
                reward_shaping = gamma * potential - prev_potential
                self._last_potential = potential
            else:
                reward_shaping = 0.0
                self._last_potential = 0.0
            self._last_entry_additive = 0.0
            self._last_exit_additive = 0.0
            self._last_reward_shaping = float(reward_shaping)
            self._total_reward_shaping += float(reward_shaping)
            self._last_prev_potential = float(prev_potential)
            self._last_next_potential = float(self._last_potential)
            return base_reward + reward_shaping
        elif is_exit:
            if (
                self._exit_potential_mode == "canonical"
                or self._exit_potential_mode == "non_canonical"
            ):
                next_potential = 0.0
                exit_reward_shaping = -prev_potential
            else:
                next_potential = self._compute_exit_potential(prev_potential, gamma)
                exit_reward_shaping = gamma * next_potential - prev_potential
            self._last_entry_additive = 0.0
            self._last_exit_additive = 0.0
            exit_additive = 0.0
            if self._exit_additive_enabled and not self.is_pbrs_invariant_mode():
                duration_ratio = trade_duration / max(max_trade_duration, 1)
                exit_additive = self._compute_exit_additive(
                    pnl, pnl_target, duration_ratio
                )
                self._last_exit_additive = float(exit_additive)
                self._total_exit_additive += float(exit_additive)
            self._last_potential = next_potential
            self._last_reward_shaping = float(exit_reward_shaping)
            self._total_reward_shaping += float(exit_reward_shaping)
            self._last_prev_potential = float(prev_potential)
            self._last_next_potential = float(self._last_potential)
            return base_reward + exit_reward_shaping + exit_additive
        else:
            # Neutral self-loop
            self._last_prev_potential = float(prev_potential)
            self._last_next_potential = float(self._last_potential)
            self._last_entry_additive = 0.0
            self._last_exit_additive = 0.0
            return base_reward

    def _set_observation_space(self) -> None:
        """
        Set the observation space
        """
        signal_features = self.signal_features.shape[1]
        if self.add_state_info:
            # STATE_INFO
            self.state_features = ["pnl", "position", "trade_duration"]
            self.total_features = signal_features + len(self.state_features)
        else:
            self.state_features = []
            self.total_features = signal_features

        self.shape = (self.window_size, self.total_features)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32
        )

    def _is_valid(self, action: int) -> bool:
        return ReforceXY.get_action_masks(self.can_short, self._position)[action]

    def reset_env(
        self,
        df: DataFrame,
        prices: DataFrame,
        window_size: int,
        reward_kwargs: Dict[str, Any],
        starting_point=True,
    ) -> None:
        """
        Resets the environment when the agent fails
        """
        super().reset_env(df, prices, window_size, reward_kwargs, starting_point)
        self._set_observation_space()

    def reset(self, seed=None, **kwargs) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        """
        Reset is called at the beginning of every episode
        """
        observation, history = super().reset(seed, **kwargs)
        self._last_closed_position: Optional[Positions] = None
        self._last_closed_trade_tick: int = 0
        self._max_unrealized_profit = -np.inf
        self._min_unrealized_profit = np.inf
        self._last_potential = 0.0
        self._last_prev_potential = 0.0
        self._last_next_potential = 0.0
        self._last_reward_shaping = 0.0
        self._total_reward_shaping = 0.0
        self._last_entry_additive = 0.0
        self._total_entry_additive = 0.0
        self._last_exit_additive = 0.0
        self._total_exit_additive = 0.0
        self._last_invalid_penalty = 0.0
        self._last_idle_penalty = 0.0
        self._last_hold_penalty = 0.0
        self._last_exit_reward = 0.0
        return observation, history

    def _get_exit_factor(
        self,
        factor: float,
        pnl: float,
        duration_ratio: float,
    ) -> float:
        """
        Compute the reward factor at trade exit
        """
        if not (
            np.isfinite(factor) and np.isfinite(pnl) and np.isfinite(duration_ratio)
        ):
            return 0.0
        if duration_ratio < 0.0:
            duration_ratio = 0.0

        model_reward_parameters = self.rl_config.get("model_reward_parameters", {})
        exit_attenuation_mode = str(
            model_reward_parameters.get("exit_attenuation_mode", "linear")
        )
        exit_plateau = bool(model_reward_parameters.get("exit_plateau", True))
        exit_plateau_grace = float(
            model_reward_parameters.get("exit_plateau_grace", 1.0)
        )
        if exit_plateau_grace < 0.0:
            exit_plateau_grace = 0.0
        exit_linear_slope = float(model_reward_parameters.get("exit_linear_slope", 1.0))
        if exit_linear_slope < 0.0:
            exit_linear_slope = 0.0

        def _legacy(f: float, dr: float, p: Mapping) -> float:
            return f * (1.5 if dr <= 1.0 else 0.5)

        def _sqrt(f: float, dr: float, p: Mapping) -> float:
            return f / math.sqrt(1.0 + dr)

        def _linear(f: float, dr: float, p: Mapping) -> float:
            slope = float(p.get("exit_linear_slope", 1.0))
            if slope < 0.0:
                slope = 1.0
            return f / (1.0 + slope * dr)

        def _power(f: float, dr: float, p: Mapping) -> float:
            tau = p.get("exit_power_tau")
            if isinstance(tau, (int, float)):
                tau = float(tau)
                if 0.0 < tau <= 1.0:
                    alpha = -math.log(tau) / ReforceXY._LOG_2
                else:
                    alpha = 1.0
            else:
                alpha = 1.0
            return f / math.pow(1.0 + dr, alpha)

        def _half_life(f: float, dr: float, p: Mapping) -> float:
            hl = float(p.get("exit_half_life", 0.5))
            if hl <= 0.0:
                hl = 0.0
            return f * math.pow(2.0, -dr / hl)

        strategies: Dict[str, Callable[[float, float, Mapping], float]] = {
            "legacy": _legacy,
            "sqrt": _sqrt,
            "linear": _linear,
            "power": _power,
            "half_life": _half_life,
        }

        if exit_plateau:
            if duration_ratio <= exit_plateau_grace:
                effective_dr = 0.0
            else:
                effective_dr = duration_ratio - exit_plateau_grace
        else:
            effective_dr = duration_ratio

        strategy_fn = strategies.get(exit_attenuation_mode, None)
        if strategy_fn is None:
            logger.debug(
                "Unknown exit_attenuation_mode '%s'; defaulting to linear",
                exit_attenuation_mode,
            )
            strategy_fn = _linear

        try:
            factor = strategy_fn(factor, effective_dr, model_reward_parameters)
        except Exception as e:
            logger.warning(
                "exit_attenuation_mode '%s' failed (%r); fallback linear (effective_dr=%.5f)",
                exit_attenuation_mode,
                e,
                effective_dr,
            )
            factor = _linear(factor, effective_dr, model_reward_parameters)

        factor *= self._get_pnl_factor(pnl, self._pnl_target)

        check_invariants = model_reward_parameters.get("check_invariants", True)
        check_invariants = (
            check_invariants if isinstance(check_invariants, bool) else True
        )
        if check_invariants:
            if not np.isfinite(factor):
                logger.debug(
                    "_get_exit_factor produced non-finite factor; resetting to 0.0"
                )
                return 0.0
            if factor < 0.0 and pnl >= 0.0:
                logger.debug(
                    "_get_exit_factor negative with positive pnl (factor=%.5f, pnl=%.5f); clamping to 0.0",
                    factor,
                    pnl,
                )
                factor = 0.0
            exit_factor_threshold = float(
                model_reward_parameters.get("exit_factor_threshold", 10_000.0)
            )
            if exit_factor_threshold > 0 and abs(factor) > exit_factor_threshold:
                logger.warning(
                    "_get_exit_factor |factor|=%.2f exceeds threshold %.2f",
                    factor,
                    exit_factor_threshold,
                )

        return factor

    def _get_pnl_factor(self, pnl: float, pnl_target: float) -> float:
        if not np.isfinite(pnl):
            return 0.0

        model_reward_parameters = self.rl_config.get("model_reward_parameters", {})

        pnl_target_factor = 1.0
        if pnl_target > 0.0:
            pnl_factor_beta = float(model_reward_parameters.get("pnl_factor_beta", 0.5))
            pnl_ratio = pnl / pnl_target
            if abs(pnl_ratio) > 1.0:
                base_pnl_target_factor = math.tanh(
                    pnl_factor_beta * (abs(pnl_ratio) - 1.0)
                )
                win_reward_factor = float(
                    model_reward_parameters.get("win_reward_factor", 2.0)
                )
                if pnl_ratio > 1.0:
                    pnl_target_factor = 1.0 + win_reward_factor * base_pnl_target_factor
                elif pnl_ratio < -(1.0 / self.rr):
                    loss_penalty_factor = win_reward_factor * self.rr
                    pnl_target_factor = (
                        1.0 + loss_penalty_factor * base_pnl_target_factor
                    )

        efficiency_factor = 1.0
        efficiency_weight = float(model_reward_parameters.get("efficiency_weight", 1.0))
        efficiency_center = float(model_reward_parameters.get("efficiency_center", 0.5))
        if efficiency_weight != 0.0 and not np.isclose(pnl, 0.0):
            max_pnl = max(self.get_max_unrealized_profit(), pnl)
            min_pnl = min(self.get_min_unrealized_profit(), pnl)
            range_pnl = max_pnl - min_pnl
            if np.isfinite(range_pnl) and not np.isclose(range_pnl, 0.0):
                efficiency_ratio = (pnl - min_pnl) / range_pnl
                if pnl > 0.0:
                    efficiency_factor = 1.0 + efficiency_weight * (
                        efficiency_ratio - efficiency_center
                    )
                elif pnl < 0.0:
                    efficiency_factor = 1.0 + efficiency_weight * (
                        efficiency_center - efficiency_ratio
                    )

        return max(0.0, pnl_target_factor * efficiency_factor)

    def calculate_reward(self, action: int) -> float:
        """Compute per-step reward and apply potential-based reward shaping (PBRS).

        Reward Pipeline:
            1. Invalid action penalty
            2. Idle penalty
            3. Hold overtime penalty
            4. Exit reward
            5. Default fallback (0.0 if no specific reward)
            6. PBRS application: R'(s,a,s') = R_base + Δ(s,s') + optional_additives

        The final shaped reward is what the RL agent receives for learning.
        In canonical PBRS mode, the learned policy is theoretically equivalent
        to training on base rewards only (policy invariance).

        Parameters
        ----------
        action : int
            Action index taken by the agent

        Returns
        -------
        float
            Shaped reward R'(s,a,s') = R_base + Δ(s,s') + optional_additives
        """
        model_reward_parameters = self.rl_config.get("model_reward_parameters", {})
        base_reward: Optional[float] = None

        self._last_invalid_penalty = 0.0
        self._last_idle_penalty = 0.0
        self._last_hold_penalty = 0.0
        self._last_exit_reward = 0.0

        # 1. Invalid action
        if not self.action_masking and not self._is_valid(action):
            self.tensorboard_log("invalid", category="actions")
            base_reward = float(model_reward_parameters.get("invalid_action", -2.0))
            self._last_invalid_penalty = float(base_reward)

        max_trade_duration = max(1, self.max_trade_duration_candles)
        trade_duration = self.get_trade_duration()
        duration_ratio = trade_duration / max_trade_duration
        base_factor = float(model_reward_parameters.get("base_factor", 100.0))
        idle_factor = base_factor * self._pnl_target / 4.0
        hold_factor = idle_factor

        # 2. Idle penalty
        if (
            base_reward is None
            and action == Actions.Neutral.value
            and self._position == Positions.Neutral
        ):
            max_idle_duration = max(1, self.max_idle_duration_candles)
            idle_penalty_scale = float(
                model_reward_parameters.get("idle_penalty_scale", 0.5)
            )
            idle_penalty_power = float(
                model_reward_parameters.get("idle_penalty_power", 1.025)
            )
            idle_duration = self.get_idle_duration()
            idle_duration_ratio = idle_duration / max(1, max_idle_duration)
            base_reward = (
                -idle_factor
                * idle_penalty_scale
                * idle_duration_ratio**idle_penalty_power
            )
            self._last_idle_penalty = float(base_reward)

        # 3. Hold overtime penalty
        if (
            base_reward is None
            and self._position in (Positions.Short, Positions.Long)
            and action == Actions.Neutral.value
        ):
            hold_penalty_scale = float(
                model_reward_parameters.get("hold_penalty_scale", 0.25)
            )
            hold_penalty_power = float(
                model_reward_parameters.get("hold_penalty_power", 1.025)
            )
            if duration_ratio < 1.0:
                base_reward = 0.0
            else:
                base_reward = (
                    -hold_factor
                    * hold_penalty_scale
                    * (duration_ratio - 1.0) ** hold_penalty_power
                )
                self._last_hold_penalty = float(base_reward)

        # 4. Exit rewards
        pnl = self.get_unrealized_profit()
        if (
            base_reward is None
            and action == Actions.Long_exit.value
            and self._position == Positions.Long
        ):
            base_reward = pnl * self._get_exit_factor(base_factor, pnl, duration_ratio)
            self._last_exit_reward = float(base_reward)
        if (
            base_reward is None
            and action == Actions.Short_exit.value
            and self._position == Positions.Short
        ):
            base_reward = pnl * self._get_exit_factor(base_factor, pnl, duration_ratio)
            self._last_exit_reward = float(base_reward)

        # 5. Default
        if base_reward is None:
            base_reward = 0.0

        # 6. Potential-based reward shaping
        return self._apply_potential_shaping(
            base_reward=base_reward,
            action=action,
            trade_duration=trade_duration,
            max_trade_duration=max_trade_duration,
            pnl=pnl,
            pnl_target=self._pnl_target,
        )

    def _get_observation(self) -> NDArray[np.float32]:
        start_idx = max(self._start_tick, self._current_tick - self.window_size)
        end_idx = min(self._current_tick, len(self.signal_features))
        features_window = self.signal_features.iloc[start_idx:end_idx]
        features_window_array = features_window.to_numpy(dtype=np.float32, copy=False)
        if features_window_array.shape[0] < self.window_size:
            pad_size = self.window_size - features_window_array.shape[0]
            pad_array = np.zeros(
                (pad_size, features_window_array.shape[1]), dtype=np.float32
            )
            features_window_array = np.concatenate(
                [pad_array, features_window_array], axis=0
            )
        if self.add_state_info:
            observations = np.concatenate(
                [
                    features_window_array,
                    np.tile(
                        np.array(
                            [
                                float(self.get_unrealized_profit()),
                                float(self._position.value),
                                float(self.get_trade_duration()),
                            ],
                            dtype=np.float32,
                        ),
                        (self.window_size, 1),
                    ),
                ],
                axis=1,
            )
        else:
            observations = features_window_array

        return np.ascontiguousarray(observations)

    def _get_position(self, action: int) -> Positions:
        return {
            Actions.Long_enter.value: Positions.Long,
            Actions.Short_enter.value: Positions.Short,
        }[action]

    def _enter_trade(self, action: int) -> None:
        self._position = self._get_position(action)
        self._last_trade_tick = self._current_tick
        self._max_unrealized_profit = -np.inf
        self._min_unrealized_profit = np.inf

    def _exit_trade(self) -> None:
        self._update_total_profit()
        self._last_closed_position = self._position
        self._position = Positions.Neutral
        self._last_trade_tick = None
        self._last_closed_trade_tick = self._current_tick
        self._max_unrealized_profit = -np.inf
        self._min_unrealized_profit = np.inf

    def execute_trade(self, action: int) -> Optional[str]:
        """
        Execute trade based on the given action
        """

        if not self.is_tradesignal(action):
            return None

        # Enter trade based on action
        if action in (Actions.Long_enter.value, Actions.Short_enter.value):
            self._enter_trade(action)
            return f"{self._position.name}_enter"

        # Exit trade based on action
        if action in (Actions.Long_exit.value, Actions.Short_exit.value):
            self._exit_trade()
            return f"{self._last_closed_position.name}_exit"

        return None

    def step(
        self, action: int
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment based on the provided action
        """
        self._current_tick += 1
        self._update_unrealized_total_profit()
        pre_pnl = self.get_unrealized_profit()
        self._update_portfolio_log_returns()
        reward = self.calculate_reward(action)
        self.total_reward += reward
        trade_type = self.execute_trade(action)
        if trade_type is not None:
            self.append_trade_history(trade_type, self.current_price(), pre_pnl)
        self._position_history.append(self._position)
        pnl = self.get_unrealized_profit()
        self._update_max_unrealized_profit(pnl)
        self._update_min_unrealized_profit(pnl)
        delta_pnl = pnl - pre_pnl
        max_idle_duration = max(1, self.max_idle_duration_candles)
        idle_duration = self.get_idle_duration()
        trade_duration = self.get_trade_duration()
        info = {
            "tick": self._current_tick,
            "position": float(self._position.value),
            "action": action,
            "pre_pnl": round(pre_pnl, 5),
            "pnl": round(pnl, 5),
            "delta_pnl": round(delta_pnl, 5),
            "max_pnl": round(self.get_max_unrealized_profit(), 5),
            "min_pnl": round(self.get_min_unrealized_profit(), 5),
            "most_recent_return": round(self.get_most_recent_return(), 5),
            "most_recent_profit": round(self.get_most_recent_profit(), 5),
            "total_profit": round(self._total_profit, 5),
            "prev_potential": round(self._last_prev_potential, 5),
            "next_potential": round(self._last_next_potential, 5),
            "reward_entry_additive": round(self._last_entry_additive, 5),
            "reward_exit_additive": round(self._last_exit_additive, 5),
            "reward_shaping": round(self._last_reward_shaping, 5),
            "total_reward_shaping": round(self._total_reward_shaping, 5),
            "reward_invalid": round(self._last_invalid_penalty, 5),
            "reward_idle": round(self._last_idle_penalty, 5),
            "reward_hold": round(self._last_hold_penalty, 5),
            "reward_exit": round(self._last_exit_reward, 5),
            "reward": round(reward, 5),
            "total_reward": round(self.total_reward, 5),
            "pbrs_invariant": self.is_pbrs_invariant_mode(),
            "idle_duration": idle_duration,
            "idle_ratio": (idle_duration / max_idle_duration),
            "trade_duration": trade_duration,
            "duration_ratio": (
                trade_duration / max(1, self.max_trade_duration_candles)
            ),
            "trade_count": int(len(self.trade_history) // 2),
        }
        self._update_history(info)
        terminated = self.is_terminated()
        if terminated:
            # Enforce Φ(terminal)=0 for PBRS invariance (Wiewiora et al. 2003)
            self._last_potential = 0.0
            eps = 1e-6
            if self.is_pbrs_invariant_mode() and abs(self._total_reward_shaping) > eps:
                logger.warning(
                    "PBRS mode %s invariance deviation: |sum Δ|=%.6f > eps=%.6f",
                    self._exit_potential_mode,
                    self._total_reward_shaping,
                    eps,
                )
        return (
            self._get_observation(),
            reward,
            terminated,
            self.is_truncated(),
            info,
        )

    def append_trade_history(
        self, trade_type: str, price: float, profit: float
    ) -> None:
        self.trade_history.append(
            {
                "tick": self._current_tick,
                "type": trade_type.lower(),
                "price": price,
                "profit": profit,
            }
        )

    def is_terminated(self) -> bool:
        return (
            self._current_tick == self._end_tick
            or self._total_profit < self.max_drawdown
            or self._total_unrealized_profit < self.max_drawdown
        )

    def is_truncated(self) -> bool:
        return False

    def is_tradesignal(self, action: int) -> bool:
        """
        Determine if the action is a valid entry or exit
        """
        position = self._position

        action_rules = {
            Actions.Long_enter.value: (Positions.Neutral, False),
            Actions.Short_enter.value: (Positions.Neutral, True),
            Actions.Long_exit.value: (Positions.Long, False),
            Actions.Short_exit.value: (Positions.Short, True),
        }

        if action not in action_rules:
            return False

        required_position, requires_short = action_rules[action]
        return position == required_position and (not requires_short or self.can_short)

    def action_masks(self) -> NDArray[np.bool_]:
        return ReforceXY.get_action_masks(self.can_short, self._position)

    def get_feature_value(
        self,
        name: str,
        period: int = 0,
        shift: int = 0,
        pair: str = "",
        timeframe: str = "",
        raw: bool = True,
    ) -> float:
        """
        Get feature value
        """
        feature_parts = [name]
        if period:
            feature_parts.append(f"_{period}")
        if shift:
            feature_parts.append(f"_shift-{shift}")
        if pair:
            feature_parts.append(f"_{pair.replace(':', '')}")
        if timeframe:
            feature_parts.append(f"_{timeframe}")
        feature_col = "".join(feature_parts)

        if not raw:
            return self.signal_features[feature_col].iloc[self._current_tick]
        return self.raw_features[feature_col].iloc[self._current_tick]

    def get_idle_duration(self) -> int:
        if self._position != Positions.Neutral:
            return 0
        if not self._last_closed_trade_tick:
            return self._current_tick - self._start_tick
        return self._current_tick - self._last_closed_trade_tick

    def get_max_unrealized_profit(self) -> float:
        """
        Get the maximum unrealized profit if the agent is in a trade
        """
        if self._last_trade_tick is None:
            return 0.0
        if self._position == Positions.Neutral:
            return 0.0
        if not np.isfinite(self._max_unrealized_profit):
            return self.get_unrealized_profit()
        return self._max_unrealized_profit

    def _update_max_unrealized_profit(self, pnl: float) -> None:
        if self._position in (Positions.Long, Positions.Short):
            if pnl > self._max_unrealized_profit:
                self._max_unrealized_profit = pnl

    def get_min_unrealized_profit(self) -> float:
        """
        Get the minimum unrealized profit if the agent is in a trade
        """
        if self._last_trade_tick is None:
            return 0.0
        if self._position == Positions.Neutral:
            return 0.0
        if not np.isfinite(self._min_unrealized_profit):
            return self.get_unrealized_profit()
        return self._min_unrealized_profit

    def _update_min_unrealized_profit(self, pnl: float) -> None:
        if self._position in (Positions.Long, Positions.Short):
            if pnl < self._min_unrealized_profit:
                self._min_unrealized_profit = pnl

    def get_most_recent_return(self) -> float:
        """
        Calculate the tick to tick return if the agent is in a trade.
        Return is generated from rising prices in Long and falling prices in Short positions.
        The actions Sell/Buy or Hold during a Long position trigger the sell/buy-fee.
        """
        if self._last_trade_tick is None:
            return 0.0
        if self._position == Positions.Neutral:
            return 0.0
        elif self._position == Positions.Long:
            current_price = self.current_price()
            previous_price = self.previous_price()
            previous_tick = self.previous_tick()
            if (
                self._position_history[previous_tick] == Positions.Short
                or self._position_history[previous_tick] == Positions.Neutral
            ):
                previous_price = self.add_entry_fee(previous_price)
            return np.log(current_price) - np.log(previous_price)
        elif self._position == Positions.Short:
            current_price = self.current_price()
            previous_price = self.previous_price()
            previous_tick = self.previous_tick()
            if (
                self._position_history[previous_tick] == Positions.Long
                or self._position_history[previous_tick] == Positions.Neutral
            ):
                previous_price = self.add_exit_fee(previous_price)
            return np.log(previous_price) - np.log(current_price)
        return 0.0

    def _update_portfolio_log_returns(self):
        self.portfolio_log_returns[self._current_tick] = self.get_most_recent_return()

    def get_most_recent_profit(self) -> float:
        """
        Calculate the tick to tick unrealized profit if the agent is in a trade
        """
        if self._last_trade_tick is None:
            return 0.0
        if self._position == Positions.Neutral:
            return 0.0
        elif self._position == Positions.Long:
            current_price = self.add_exit_fee(self.current_price())
            previous_price = self.add_entry_fee(self.previous_price())
            return (current_price - previous_price) / previous_price
        elif self._position == Positions.Short:
            current_price = self.add_entry_fee(self.current_price())
            previous_price = self.add_exit_fee(self.previous_price())
            return (previous_price - current_price) / previous_price
        return 0.0

    def previous_tick(self) -> int:
        return max(self._current_tick - 1, self._start_tick)

    def previous_price(self) -> float:
        return self.prices.iloc[self.previous_tick()].get("open")

    def get_env_history(self) -> DataFrame:
        """
        Get environment data aligned on ticks, including optional trade events
        """
        if not self.history:
            logger.warning("history is empty")
            return DataFrame()

        _history_df = DataFrame(self.history)
        if "tick" not in _history_df.columns:
            logger.warning("'tick' column is missing from history")
            return DataFrame()

        _rollout_history = _history_df.copy()
        if self.trade_history:
            _trade_history_df = DataFrame(self.trade_history)
            if "tick" in _trade_history_df.columns:
                _rollout_history = merge(
                    _rollout_history, _trade_history_df, on="tick", how="left"
                )

        try:
            history = merge(
                _rollout_history,
                self.prices,
                left_on="tick",
                right_index=True,
                how="left",
            )
        except Exception as e:
            logger.error(
                f"Failed to merge history with prices: {repr(e)}",
                exc_info=True,
            )
            return DataFrame()
        return history

    def get_env_plot(self) -> plt.Figure:
        """
        Plot trades and environment data
        """
        with plt.style.context("dark_background"):
            fig, axs = plt.subplots(
                nrows=5,
                ncols=1,
                figsize=(14, 8),
                height_ratios=[5, 1, 1, 1, 1],
                sharex=True,
            )

            def transform_y_offset(ax, offset):
                return mtransforms.offset_copy(ax.transData, fig=fig, y=offset)

            def plot_markers(ax, xs, ys, marker, color, size, offset):
                ax.plot(
                    xs,
                    ys,
                    marker=marker,
                    color=color,
                    markersize=size,
                    fillstyle="full",
                    transform=transform_y_offset(ax, offset),
                    linestyle="none",
                    zorder=3,
                )

            history = self.get_env_history()
            if len(history) == 0:
                return fig

            plot_window = self.rl_config.get("plot_window", 2000)
            if plot_window > 0 and len(history) > plot_window:
                history = history.iloc[-plot_window:]

            ticks = history.get("tick")
            history_open = history.get("open")
            if (
                ticks is None
                or len(ticks) == 0
                or history_open is None
                or len(history_open) == 0
            ):
                return fig

            axs[0].plot(ticks, history_open, linewidth=1, color="orchid", zorder=1)

            history_type = history.get("type")
            history_price = history.get("price")
            if history_type is not None and history_price is not None:
                trade_markers_config = [
                    ("long_enter", "^", "forestgreen", 5, -0.1, "Long enter"),
                    ("short_enter", "v", "firebrick", 5, 0.1, "Short enter"),
                    ("long_exit", ".", "cornflowerblue", 4, 0.1, "Long exit"),
                    ("short_exit", ".", "thistle", 4, -0.1, "Short exit"),
                ]

                legend_scale_factor = 1.5
                markers_legend = []

                for (
                    type_name,
                    marker,
                    color,
                    size,
                    offset,
                    label,
                ) in trade_markers_config:
                    mask = history_type == type_name
                    if mask.any():
                        xs = ticks[mask]
                        ys = history.loc[mask, "price"]

                        plot_markers(axs[0], xs, ys, marker, color, size, offset)

                    markers_legend.append(
                        Line2D(
                            [0],
                            [0],
                            marker=marker,
                            color="w",
                            markerfacecolor=color,
                            markersize=size * legend_scale_factor,
                            linestyle="none",
                            label=label,
                        )
                    )
                axs[0].legend(handles=markers_legend, loc="upper right", fontsize=8)

            axs[1].set_ylabel("pnl")
            pnl_series = history.get("pnl")
            if pnl_series is not None and len(pnl_series) > 0:
                axs[1].plot(
                    ticks,
                    pnl_series,
                    linewidth=1,
                    color="gray",
                    label="pnl",
                )
            pre_pnl_series = history.get("pre_pnl")
            if pre_pnl_series is not None and len(pre_pnl_series) > 0:
                axs[1].plot(
                    ticks,
                    pre_pnl_series,
                    linewidth=1,
                    color="deepskyblue",
                    label="pre_pnl",
                )
            if (pnl_series is not None and len(pnl_series) > 0) or (
                pre_pnl_series is not None and len(pre_pnl_series) > 0
            ):
                axs[1].legend(loc="upper left", fontsize=8)
            axs[1].axhline(y=0, label="0", alpha=0.33, color="gray")

            axs[2].set_ylabel("reward")
            reward_series = history.get("reward")
            if reward_series is not None and len(reward_series) > 0:
                axs[2].plot(ticks, reward_series, linewidth=1, color="gray")
            axs[2].axhline(y=0, label="0", alpha=0.33)

            axs[3].set_ylabel("total_profit")
            total_profit_series = history.get("total_profit")
            if total_profit_series is not None and len(total_profit_series) > 0:
                axs[3].plot(ticks, total_profit_series, linewidth=1, color="gray")
            axs[3].axhline(y=1, label="1", alpha=0.33)

            axs[4].set_ylabel("total_reward")
            total_reward_series = history.get("total_reward")
            if total_reward_series is not None and len(total_reward_series) > 0:
                axs[4].plot(ticks, total_reward_series, linewidth=1, color="gray")
            axs[4].axhline(y=0, label="0", alpha=0.33)
            axs[4].set_xlabel("tick")

        _borders = ["top", "right", "bottom", "left"]
        for _ax in axs:
            for _border in _borders:
                _ax.spines[_border].set_color("#5b5e4b")

        fig.suptitle(
            f"Total Reward: {self.total_reward:.2f} ~ "
            + f"Total Profit: {self._total_profit:.2f} ~ "
            + f"Trades: {int(len(self.trade_history) // 2)}",
        )
        fig.tight_layout()
        return fig

    def close(self) -> None:
        super().close()
        plt.close()
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()


class InfoMetricsCallback(TensorboardCallback):
    """
    Tensorboard callback
    """

    def __init__(self, *args, throttle: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.throttle = 1 if throttle < 1 else throttle

    def _safe_logger_record(
        self, key: str, value: Any, exclude: Optional[Tuple[str, ...]] = None
    ) -> None:
        try:
            self.logger.record(key, value, exclude=exclude)
        except Exception as e:
            logger.warning("logger.record failed at %r: %r", key, e)
            if exclude is None:
                exclude = ("tensorboard",)
            else:
                exclude_set = set(exclude)
                exclude_set.add("tensorboard")
                exclude_set.discard("stdout")
                exclude = tuple(exclude_set)
            try:
                self.logger.record(key, value, exclude=exclude)
            except Exception as e:
                logger.error("logger.record retry on stdout failed at %r: %r", key, e)
                pass

    @staticmethod
    def _build_train_freq(
        train_freq: Optional[Union[TrainFreq, int, Tuple[int, ...], List[int]]],
    ) -> Optional[int]:
        train_freq_val: Optional[int] = None
        if isinstance(train_freq, TrainFreq) and hasattr(train_freq, "frequency"):
            if isinstance(train_freq.frequency, int):
                train_freq_val = train_freq.frequency
        elif isinstance(train_freq, (tuple, list)) and train_freq:
            if isinstance(train_freq[0], int):
                train_freq_val = train_freq[0]
        elif isinstance(train_freq, int):
            train_freq_val = train_freq

        return train_freq_val

    def _on_training_start(self) -> None:
        lr = getattr(self.model, "learning_rate", None)
        lr_schedule, lr_iv, lr_fv = get_schedule_type(lr)
        n_stack = 1
        env = getattr(self, "training_env", None)
        while env is not None:
            if hasattr(env, "n_stack"):
                try:
                    n_stack = int(getattr(env, "n_stack"))
                except Exception:
                    pass
                break
            env = getattr(env, "venv", None)
        hparam_dict: Dict[str, Any] = {
            "algorithm": self.model.__class__.__name__,
            "n_envs": int(self.model.n_envs),
            "n_stack": n_stack,
            "lr_schedule": lr_schedule,
            "learning_rate_iv": lr_iv,
            "learning_rate_fv": lr_fv,
            "gamma": float(self.model.gamma),
            "batch_size": int(self.model.batch_size),
        }
        try:
            n_updates = getattr(self.model, "n_updates", None)
            if n_updates is None:
                n_updates = getattr(self.model, "_n_updates", None)
            if isinstance(n_updates, (int, float)) and np.isfinite(n_updates):
                hparam_dict.update({"n_updates": int(n_updates)})
        except Exception:
            pass
        if "PPO" in self.model.__class__.__name__:
            cr = getattr(self.model, "clip_range", None)
            cr_schedule, cr_iv, cr_fv = get_schedule_type(cr)
            hparam_dict.update(
                {
                    "cr_schedule": cr_schedule,
                    "clip_range_iv": cr_iv,
                    "clip_range_fv": cr_fv,
                    "gae_lambda": float(self.model.gae_lambda),
                    "n_steps": int(self.model.n_steps),
                    "n_epochs": int(self.model.n_epochs),
                    "ent_coef": float(self.model.ent_coef),
                    "vf_coef": float(self.model.vf_coef),
                    "max_grad_norm": float(self.model.max_grad_norm),
                }
            )
            if getattr(self.model, "target_kl", None) is not None:
                hparam_dict["target_kl"] = float(self.model.target_kl)
            if "RecurrentPPO" in self.model.__class__.__name__:
                policy = getattr(self.model, "policy", None)
                if policy is not None:
                    lstm_actor = getattr(policy, "lstm_actor", None)
                    if lstm_actor is not None:
                        hparam_dict.update(
                            {
                                "lstm_hidden_size": int(lstm_actor.hidden_size),
                                "n_lstm_layers": int(lstm_actor.num_layers),
                            }
                        )
        if "DQN" in self.model.__class__.__name__:
            hparam_dict.update(
                {
                    "buffer_size": int(self.model.buffer_size),
                    "gradient_steps": int(self.model.gradient_steps),
                    "learning_starts": int(self.model.learning_starts),
                    "target_update_interval": int(self.model.target_update_interval),
                    "exploration_initial_eps": float(
                        self.model.exploration_initial_eps
                    ),
                    "exploration_final_eps": float(self.model.exploration_final_eps),
                    "exploration_fraction": float(self.model.exploration_fraction),
                    "exploration_rate": float(self.model.exploration_rate),
                }
            )
            train_freq = InfoMetricsCallback._build_train_freq(
                getattr(self.model, "train_freq", None)
            )
            if train_freq is not None:
                hparam_dict.update({"train_freq": train_freq})
            if "QRDQN" in self.model.__class__.__name__:
                hparam_dict.update({"n_quantiles": int(self.model.n_quantiles)})
        metric_dict: dict[str, float | int] = {
            "eval/mean_reward": 0.0,
            "eval/mean_reward_std": 0.0,
            "rollout/ep_rew_mean": 0.0,
            "rollout/ep_len_mean": 0.0,
            "train/n_updates": 0,
            "train/progress_done": 0.0,
            "train/progress_remaining": 0.0,
            "train/learning_rate": 0.0,
            "info/total_reward": 0.0,
            "info/total_profit": 1.0,
            "info/trade_count": 0,
            "info/trade_duration": 0,
        }
        if "PPO" in self.model.__class__.__name__:
            metric_dict.update(
                {
                    "train/approx_kl": 0.0,
                    "train/entropy_loss": 0.0,
                    "train/policy_gradient_loss": 0.0,
                    "train/clip_fraction": 0.0,
                    "train/clip_range": 0.0,
                    "train/value_loss": 0.0,
                    "train/explained_variance": 0.0,
                }
            )
        if "DQN" in self.model.__class__.__name__:
            metric_dict.update(
                {
                    "train/loss": 0.0,
                    "train/exploration_rate": 0.0,
                }
            )
        self._safe_logger_record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        if self.throttle > 1 and (self.num_timesteps % self.throttle) != 0:
            return True

        logger_exclude = ("stdout", "log", "json", "csv")

        def _is_number(x: Any) -> bool:
            return isinstance(
                x, (int, float, np.integer, np.floating)
            ) and not isinstance(x, bool)

        def _is_finite_number(x: Any) -> bool:
            if not _is_number(x):
                return False
            try:
                return np.isfinite(float(x))
            except Exception:
                return False

        infos_list: List[Dict[str, Any]] | None = self.locals.get("infos")
        aggregated_info: Dict[str, Any] = {}

        if isinstance(infos_list, list) and infos_list:
            numeric_acc: Dict[str, List[float]] = defaultdict(list)
            non_numeric_counts: Dict[str, Dict[Any, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            filtered_values: int = 0

            for info_dict in infos_list:
                if not isinstance(info_dict, dict):
                    continue
                for k, v in info_dict.items():
                    if k in {"episode", "terminal_observation", "TimeLimit.truncated"}:
                        continue
                    if _is_finite_number(v):
                        numeric_acc[k].append(float(v))
                    elif _is_number(v):
                        filtered_values += 1
                    else:
                        non_numeric_counts[k][v] += 1

            for k, values in numeric_acc.items():
                if not values:
                    continue
                aggregated_info[k] = np.mean(values)
                if len(values) > 1:
                    try:
                        aggregated_info[f"{k}_std"] = np.std(values, ddof=1)
                    except Exception:
                        pass

            for key in ("reward", "pnl"):
                values = numeric_acc.get(key)
                if values:
                    try:
                        np_values = np.asarray(values, dtype=float)
                        aggregated_info[f"{key}_min"] = float(np.min(np_values))
                        aggregated_info[f"{key}_max"] = float(np.max(np_values))
                        percentiles = np.percentile(np_values, [25, 50, 75, 90])
                        aggregated_info[f"{key}_p25"] = float(percentiles[0])
                        aggregated_info[f"{key}_p50"] = float(percentiles[1])
                        aggregated_info[f"{key}_p75"] = float(percentiles[2])
                        aggregated_info[f"{key}_p90"] = float(percentiles[3])
                        med = float(percentiles[1])
                        mad = float(np.median(np.abs(np_values - med)))
                        aggregated_info[f"{key}_mad"] = mad
                    except Exception:
                        pass

            for k, counts in non_numeric_counts.items():
                if not counts:
                    continue
                if len(counts) == 1:
                    try:
                        aggregated_info[f"{k}_mode"] = next(iter(counts.keys()))
                    except Exception:
                        pass
                else:
                    aggregated_info[f"{k}_mode"] = "mixed"

            self._safe_logger_record(
                "info/n_envs", int(len(infos_list)), exclude=logger_exclude
            )

            if filtered_values > 0:
                self._safe_logger_record(
                    "info/filtered_values", int(filtered_values), exclude=logger_exclude
                )

        if self.training_env is None:
            return True

        try:
            tensorboard_metrics_list = self.training_env.get_attr("tensorboard_metrics")
        except Exception:
            tensorboard_metrics_list = []

        aggregated_tensorboard_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        aggregated_tensorboard_metric_counts: Dict[str, Dict[str, int]] = defaultdict(
            dict
        )
        for env_metrics in tensorboard_metrics_list or []:
            if not isinstance(env_metrics, dict):
                continue
            for category, metrics in env_metrics.items():
                if not isinstance(metrics, dict):
                    continue
                cat_dict = aggregated_tensorboard_metrics.setdefault(category, {})
                cnt_dict = aggregated_tensorboard_metric_counts.setdefault(category, {})
                for metric, value in metrics.items():
                    if _is_finite_number(value):
                        v = float(value)
                        try:
                            base = float(cat_dict.get(metric, 0.0))
                        except (ValueError, TypeError):
                            base = 0.0
                        cat_dict[metric] = base + v
                        cnt_dict[metric] = cnt_dict.get(metric, 0) + 1
                    else:
                        if cnt_dict.get(metric, 0) == 0:
                            cat_dict[metric] = value

        for metric, value in aggregated_info.items():
            self._safe_logger_record(f"info/{metric}", value, exclude=logger_exclude)

        if isinstance(infos_list, list) and infos_list:
            cat_keys = ("action", "position")
            cat_counts: Dict[str, Dict[Any, int]] = {
                k: defaultdict(int) for k in cat_keys
            }
            cat_totals: Dict[str, int] = {k: 0 for k in cat_keys}
            for info_dict in infos_list:
                if not isinstance(info_dict, dict):
                    continue
                for k in cat_keys:
                    if k in info_dict:
                        v = info_dict.get(k)
                        cat_counts[k][v] += 1
                        cat_totals[k] += 1

            for k, counts in cat_counts.items():
                cat_total = max(1, int(cat_totals.get(k, 0)))
                for name, cnt in counts.items():
                    name = str(name)
                    self._safe_logger_record(
                        f"info/{k}/{name}_count", int(cnt), exclude=logger_exclude
                    )
                    self._safe_logger_record(
                        f"info/{k}/{name}_ratio",
                        float(cnt) / float(cat_total),
                        exclude=logger_exclude,
                    )

        for category, metrics in aggregated_tensorboard_metrics.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    self._safe_logger_record(
                        f"{category}/{metric}_sum", value, exclude=logger_exclude
                    )
                    count = aggregated_tensorboard_metric_counts.get(category, {}).get(
                        metric
                    )
                    if (
                        _is_finite_number(value)
                        and isinstance(count, int)
                        and count > 0
                    ):
                        self._safe_logger_record(
                            f"{category}/{metric}_mean",
                            float(value) / float(count),
                            exclude=logger_exclude,
                        )

        try:
            total_timesteps = getattr(self.model, "_total_timesteps", None)
            if total_timesteps is not None and not np.isclose(total_timesteps, 0.0):
                progress_done = float(self.num_timesteps) / float(total_timesteps)
                progress_done = np.clip(progress_done, 0.0, 1.0)
            else:
                progress_done = 0.0
            progress_remaining = 1.0 - progress_done
            self._safe_logger_record(
                "train/progress_done", progress_done, exclude=logger_exclude
            )
            self._safe_logger_record(
                "train/progress_remaining", progress_remaining, exclude=logger_exclude
            )
        except Exception:
            progress_remaining = 1.0

        try:
            n_updates = getattr(self.model, "n_updates", None)
            if n_updates is None:
                n_updates = getattr(self.model, "_n_updates", None)
            if _is_finite_number(n_updates):
                self._safe_logger_record(
                    "train/n_updates", float(n_updates), exclude=logger_exclude
                )
        except Exception:
            pass

        def _eval_schedule(schedule: Any) -> float | None:
            schedule_type, _, _ = get_schedule_type(schedule)
            try:
                if schedule_type == "linear":
                    return float(schedule(progress_remaining))
                if schedule_type == "constant":
                    if callable(schedule):
                        return float(schedule(0.0))
                    if isinstance(schedule, (int, float)):
                        return float(schedule)
                return None
            except Exception:
                return None

        try:
            lr = getattr(self.model, "learning_rate", None)
            lr = _eval_schedule(lr)
            if _is_finite_number(lr):
                self._safe_logger_record(
                    "train/learning_rate", float(lr), exclude=logger_exclude
                )
        except Exception:
            pass

        if "PPO" in self.model.__class__.__name__:
            try:
                cr = getattr(self.model, "clip_range", None)
                cr = _eval_schedule(cr)
                if _is_finite_number(cr):
                    self._safe_logger_record(
                        "train/clip_range", float(cr), exclude=logger_exclude
                    )
            except Exception:
                pass

        if "DQN" in self.model.__class__.__name__:
            try:
                er = getattr(self.model, "exploration_rate", None)
                if _is_finite_number(er):
                    self._safe_logger_record(
                        "train/exploration_rate", float(er), exclude=logger_exclude
                    )
            except Exception:
                pass

        return True


class RolloutPlotCallback(BaseCallback):
    """
    Tensorboard plot callback
    """

    def record_env(self) -> bool:
        figures = self.training_env.env_method("get_env_plot")
        for i, fig in enumerate(figures):
            figure = Figure(fig, close=True)
            try:
                self.logger.record(
                    f"best/train_env{i}",
                    figure,
                    exclude=("stdout", "log", "json", "csv"),
                )
            except Exception as e:
                logger.error("logger.record failed at %r: %r", f"best/train_env{i}", e)
                pass
        return True

    def _on_step(self) -> bool:
        return self.record_env()


class MaskableTrialEvalCallback(MaskableEvalCallback):
    """
    Optuna maskable trial eval callback
    """

    def __init__(
        self,
        eval_env: BaseEnvironment,
        trial: Trial,
        n_eval_episodes: int = 10,
        eval_freq: int = 2048,
        deterministic: bool = True,
        render: bool = False,
        use_masking: bool = True,
        best_model_save_path: Optional[str] = None,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            render=render,
            best_model_save_path=best_model_save_path,
            use_masking=use_masking,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            verbose=verbose,
            **kwargs,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.is_pruned:
            return False
        _super_on_step = super()._on_step()
        if not _super_on_step:
            return False
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            try:
                last_mean_reward = float(getattr(self, "last_mean_reward", np.nan))
            except Exception as e:
                logger.warning(
                    "Optuna: invalid last_mean_reward at eval %s: %r", self.eval_idx, e
                )
                self.is_pruned = True
                return False
            if not np.isfinite(last_mean_reward):
                logger.warning(
                    "Optuna: non-finite last_mean_reward at eval %s", self.eval_idx
                )
                self.is_pruned = True
                return False
            try:
                self.trial.report(last_mean_reward, self.eval_idx)
            except Exception as e:
                logger.warning(
                    "Optuna: trial.report failed at eval %s: %r", self.eval_idx, e
                )
                self.is_pruned = True
                return False
            try:
                best_mean_reward = float(getattr(self, "best_mean_reward", np.nan))
            except Exception as e:
                logger.warning(
                    "Optuna: invalid best_mean_reward at eval %s: %r",
                    self.eval_idx,
                    e,
                )
            try:
                logger_exclude = ("stdout", "log", "json", "csv")
                self.logger.record(
                    "eval/idx",
                    int(self.eval_idx),
                    exclude=logger_exclude,
                )
                self.logger.record(
                    "eval/last_mean_reward",
                    last_mean_reward,
                    exclude=logger_exclude,
                )
                if np.isfinite(best_mean_reward):
                    self.logger.record(
                        "eval/best_mean_reward",
                        best_mean_reward,
                        exclude=logger_exclude,
                    )
                else:
                    logger.warning(
                        "Optuna: non-finite best_mean_reward at eval %s", self.eval_idx
                    )
            except Exception as e:
                logger.error(
                    "Optuna: logger.record failed at eval %s: %r", self.eval_idx, e
                )
            try:
                if self.trial.should_prune():
                    logger.info(
                        "Optuna: trial pruned at eval %s (score=%.5f)",
                        self.eval_idx,
                        last_mean_reward,
                    )
                    self.is_pruned = True
                    return False
            except Exception as e:
                logger.warning(
                    "Optuna: trial.should_prune failed at eval %s: %r", self.eval_idx, e
                )
                self.is_pruned = True
                return False

        return True


class SimpleLinearSchedule:
    """
    Linear schedule (from initial value to zero),
    simpler than sb3 LinearSchedule.

    :param initial_value: (float or str) The initial value for the schedule
    """

    def __init__(self, initial_value: Union[float, str]) -> None:
        # Force conversion to float
        self.initial_value = float(initial_value)

    def __call__(self, progress_remaining: float) -> float:
        return progress_remaining * self.initial_value

    def __repr__(self) -> str:
        return f"SimpleLinearSchedule(initial_value={self.initial_value})"


def deepmerge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dicts without mutating inputs"""
    dst_copy = copy.deepcopy(dst)
    for k, v in src.items():
        if (
            k in dst_copy
            and isinstance(dst_copy[k], Mapping)
            and isinstance(v, Mapping)
        ):
            dst_copy[k] = deepmerge(dst_copy[k], v)
        else:
            dst_copy[k] = v
    return dst_copy


def _compute_gradient_steps(tf: int, ss: int) -> int:
    if tf > 0 and ss > 0:
        return min(tf, max(math.ceil(tf / ss), 1))
    return -1


def compute_gradient_steps(train_freq: Any, subsample_steps: Any) -> int:
    tf: Optional[int] = None
    if isinstance(train_freq, TrainFreq):
        tf = train_freq.frequency if isinstance(train_freq.frequency, int) else None
    if isinstance(train_freq, (tuple, list)) and train_freq:
        tf = train_freq[0] if isinstance(train_freq[0], int) else None
    elif isinstance(train_freq, int):
        tf = train_freq

    ss: Optional[int] = subsample_steps if isinstance(subsample_steps, int) else None

    if isinstance(tf, int) and isinstance(ss, int):
        return _compute_gradient_steps(tf, ss)
    return -1


def hours_to_seconds(hours: float) -> float:
    """
    Converts hours to seconds
    """
    seconds = hours * 3600
    return seconds


def steps_to_days(steps: int, timeframe: str) -> float:
    """
    Calculate the number of days based on the given number of steps
    """
    total_minutes = steps * timeframe_to_minutes(timeframe)
    days = total_minutes / (24 * 60)
    return round(days, 1)


def get_schedule_type(
    schedule: Any,
) -> Tuple[Literal["constant", "linear", "unknown"], float, float]:
    if isinstance(schedule, (int, float)):
        try:
            schedule = float(schedule)
            return "constant", schedule, schedule
        except Exception:
            return "constant", np.nan, np.nan
    elif isinstance(schedule, ConstantSchedule):
        try:
            return "constant", schedule(1.0), schedule(0.0)
        except Exception:
            return "constant", np.nan, np.nan
    elif isinstance(schedule, SimpleLinearSchedule):
        try:
            return "linear", schedule(1.0), schedule(0.0)
        except Exception:
            return "linear", np.nan, np.nan

    return "unknown", np.nan, np.nan


def get_schedule(
    schedule_type: Literal["linear", "constant"],
    initial_value: float,
) -> Callable[[float], float]:
    if schedule_type == "linear":
        return SimpleLinearSchedule(initial_value)
    elif schedule_type == "constant":
        return ConstantSchedule(initial_value)
    else:
        return ConstantSchedule(initial_value)


def get_net_arch(
    model_type: str, net_arch_type: Literal["small", "medium", "large", "extra_large"]
) -> Union[List[int], Dict[str, List[int]]]:
    """
    Get network architecture
    """
    if "PPO" in model_type:
        return {
            "small": {"pi": [128, 128], "vf": [128, 128]},
            "medium": {"pi": [256, 256], "vf": [256, 256]},
            "large": {"pi": [512, 512], "vf": [512, 512]},
            "extra_large": {"pi": [1024, 1024], "vf": [1024, 1024]},
        }.get(net_arch_type, {"pi": [128, 128], "vf": [128, 128]})
    return {
        "small": [128, 128],
        "medium": [256, 256],
        "large": [512, 512],
        "extra_large": [1024, 1024],
    }.get(net_arch_type, [128, 128])


def get_activation_fn(
    activation_fn_name: Literal["tanh", "relu", "elu", "leaky_relu"],
) -> Type[th.nn.Module]:
    """
    Get activation function
    """
    return {
        "tanh": th.nn.Tanh,
        "relu": th.nn.ReLU,
        "elu": th.nn.ELU,
        "leaky_relu": th.nn.LeakyReLU,
    }.get(activation_fn_name, th.nn.ReLU)


def get_optimizer_class(
    optimizer_class_name: Literal["adam", "adamw", "rmsprop"],
) -> Type[th.optim.Optimizer]:
    """
    Get optimizer class
    """
    return {
        "adam": th.optim.Adam,
        "adamw": th.optim.AdamW,
        "rmsprop": th.optim.RMSprop,
    }.get(optimizer_class_name, th.optim.Adam)


def convert_optuna_params_to_model_params(
    model_type: str, optuna_params: Dict[str, Any]
) -> Dict[str, Any]:
    model_params: Dict[str, Any] = {}
    policy_kwargs: Dict[str, Any] = {}

    lr = optuna_params.get("learning_rate")
    if lr is None:
        raise ValueError(f"missing 'learning_rate' in optuna params for {model_type}")
    lr = get_schedule(optuna_params.get("lr_schedule", "constant"), float(lr))

    if "PPO" in model_type:
        required_ppo_params = [
            "clip_range",
            "n_steps",
            "batch_size",
            "gamma",
            "ent_coef",
            "n_epochs",
            "gae_lambda",
            "max_grad_norm",
            "vf_coef",
        ]
        for param in required_ppo_params:
            if optuna_params.get(param) is None:
                raise ValueError(f"missing '{param}' in optuna params for {model_type}")
        cr = optuna_params.get("clip_range")
        cr = get_schedule(optuna_params.get("cr_schedule", "constant"), float(cr))

        model_params.update(
            {
                "n_steps": int(optuna_params.get("n_steps")),
                "batch_size": int(optuna_params.get("batch_size")),
                "gamma": float(optuna_params.get("gamma")),
                "learning_rate": lr,
                "ent_coef": float(optuna_params.get("ent_coef")),
                "clip_range": cr,
                "n_epochs": int(optuna_params.get("n_epochs")),
                "gae_lambda": float(optuna_params.get("gae_lambda")),
                "max_grad_norm": float(optuna_params.get("max_grad_norm")),
                "vf_coef": float(optuna_params.get("vf_coef")),
            }
        )
        if optuna_params.get("target_kl") is not None:
            model_params["target_kl"] = float(optuna_params.get("target_kl"))
        if "RecurrentPPO" in model_type:
            policy_kwargs["lstm_hidden_size"] = int(
                optuna_params.get("lstm_hidden_size")
            )
            policy_kwargs["n_lstm_layers"] = int(optuna_params.get("n_lstm_layers"))
    elif "DQN" in model_type:
        required_dqn_params = [
            "gamma",
            "batch_size",
            "buffer_size",
            "train_freq",
            "exploration_fraction",
            "exploration_initial_eps",
            "exploration_final_eps",
            "target_update_interval",
            "learning_starts",
            "subsample_steps",
        ]
        for param in required_dqn_params:
            if optuna_params.get(param) is None:
                raise ValueError(f"missing '{param}' in optuna params for {model_type}")
        train_freq = optuna_params.get("train_freq")
        subsample_steps = optuna_params.get("subsample_steps")
        gradient_steps = compute_gradient_steps(train_freq, subsample_steps)

        model_params.update(
            {
                "gamma": float(optuna_params.get("gamma")),
                "batch_size": int(optuna_params.get("batch_size")),
                "learning_rate": lr,
                "buffer_size": int(optuna_params.get("buffer_size")),
                "train_freq": train_freq,
                "gradient_steps": gradient_steps,
                "exploration_fraction": float(
                    optuna_params.get("exploration_fraction")
                ),
                "exploration_initial_eps": float(
                    optuna_params.get("exploration_initial_eps")
                ),
                "exploration_final_eps": float(
                    optuna_params.get("exploration_final_eps")
                ),
                "target_update_interval": int(
                    optuna_params.get("target_update_interval")
                ),
                "learning_starts": int(optuna_params.get("learning_starts")),
            }
        )
        if "QRDQN" in model_type and optuna_params.get("n_quantiles") is not None:
            policy_kwargs["n_quantiles"] = int(optuna_params["n_quantiles"])
    else:
        raise ValueError(f"Model {model_type} not supported")

    if optuna_params.get("net_arch"):
        policy_kwargs["net_arch"] = get_net_arch(
            model_type, str(optuna_params["net_arch"])
        )
    if optuna_params.get("activation_fn"):
        policy_kwargs["activation_fn"] = get_activation_fn(
            str(optuna_params["activation_fn"])
        )
    if optuna_params.get("optimizer_class"):
        policy_kwargs["optimizer_class"] = get_optimizer_class(
            str(optuna_params["optimizer_class"])
        )
    if optuna_params.get("ortho_init") is not None:
        policy_kwargs["ortho_init"] = bool(optuna_params["ortho_init"])

    model_params["policy_kwargs"] = policy_kwargs
    return model_params


PPO_N_STEPS: Tuple[int, ...] = (512, 1024, 2048, 4096)


def get_common_ppo_optuna_params(trial: Trial) -> Dict[str, Any]:
    return {
        "n_steps": trial.suggest_categorical("n_steps", list(PPO_N_STEPS)),
        "batch_size": trial.suggest_categorical(
            "batch_size", [64, 128, 256, 512, 1024]
        ),
        "gamma": trial.suggest_categorical(
            "gamma", [0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999, 0.9999]
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.0005, 0.03, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4, step=0.05),
        "n_epochs": trial.suggest_int("n_epochs", 1, 5),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99, step=0.01),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0, step=0.05),
        "vf_coef": trial.suggest_float("vf_coef", 0.0, 1.0, step=0.05),
        "lr_schedule": trial.suggest_categorical("lr_schedule", ["linear", "constant"]),
        "cr_schedule": trial.suggest_categorical("cr_schedule", ["linear", "constant"]),
        "target_kl": trial.suggest_categorical(
            "target_kl", [None, 0.01, 0.015, 0.02, 0.03, 0.04]
        ),
        "ortho_init": trial.suggest_categorical("ortho_init", [True, False]),
        "net_arch": trial.suggest_categorical(
            "net_arch", ["small", "medium", "large", "extra_large"]
        ),
        "activation_fn": trial.suggest_categorical(
            "activation_fn", ["tanh", "relu", "elu", "leaky_relu"]
        ),
        "optimizer_class": trial.suggest_categorical(
            "optimizer_class", ["adamw", "rmsprop"]
        ),
    }


def sample_params_ppo(trial: Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams
    """
    return convert_optuna_params_to_model_params(
        "PPO", get_common_ppo_optuna_params(trial)
    )


def sample_params_recurrentppo(trial: Trial) -> Dict[str, Any]:
    """
    Sampler for RecurrentPPO hyperparams
    """
    ppo_optuna_params = get_common_ppo_optuna_params(trial)
    ppo_optuna_params.update(
        {
            "lstm_hidden_size": trial.suggest_categorical(
                "lstm_hidden_size", [64, 128, 256, 512]
            ),
            "n_lstm_layers": trial.suggest_int("n_lstm_layers", 1, 2),
        }
    )
    return convert_optuna_params_to_model_params("RecurrentPPO", ppo_optuna_params)


def get_common_dqn_optuna_params(trial: Trial) -> Dict[str, Any]:
    exploration_final_eps = trial.suggest_float(
        "exploration_final_eps", 0.01, 0.2, step=0.01
    )
    exploration_initial_eps = trial.suggest_float(
        "exploration_initial_eps", exploration_final_eps, 1.0
    )
    if exploration_initial_eps >= 0.9:
        min_fraction = 0.2
    elif (exploration_initial_eps - exploration_final_eps) > 0.5:
        min_fraction = 0.15
    else:
        min_fraction = 0.05
    return {
        "train_freq": trial.suggest_categorical(
            "train_freq", [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        ),
        "subsample_steps": trial.suggest_categorical("subsample_steps", [2, 4, 8, 16]),
        "gamma": trial.suggest_categorical(
            "gamma", [0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999, 0.9999]
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", [64, 128, 256, 512, 1024]
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "lr_schedule": trial.suggest_categorical("lr_schedule", ["linear", "constant"]),
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [int(1e4), int(5e4), int(1e5), int(2e5)]
        ),
        "exploration_initial_eps": exploration_initial_eps,
        "exploration_final_eps": exploration_final_eps,
        "exploration_fraction": trial.suggest_float(
            "exploration_fraction", min_fraction, 0.9, step=0.02
        ),
        "target_update_interval": trial.suggest_categorical(
            "target_update_interval", [1000, 2000, 5000, 7500, 10000]
        ),
        "learning_starts": trial.suggest_categorical(
            "learning_starts", [500, 1000, 2000, 3000, 4000, 5000, 8000, 10000]
        ),
        "net_arch": trial.suggest_categorical(
            "net_arch", ["small", "medium", "large", "extra_large"]
        ),
        "activation_fn": trial.suggest_categorical(
            "activation_fn", ["tanh", "relu", "elu", "leaky_relu"]
        ),
        "optimizer_class": trial.suggest_categorical(
            "optimizer_class", ["adamw", "rmsprop"]
        ),
    }


def sample_params_dqn(trial: Trial) -> Dict[str, Any]:
    """
    Sampler for DQN hyperparams
    """
    return convert_optuna_params_to_model_params(
        "DQN", get_common_dqn_optuna_params(trial)
    )


def sample_params_qrdqn(trial: Trial) -> Dict[str, Any]:
    """
    Sampler for QRDQN hyperparams
    """
    dqn_optuna_params = get_common_dqn_optuna_params(trial)
    dqn_optuna_params.update({"n_quantiles": trial.suggest_int("n_quantiles", 10, 160)})
    return convert_optuna_params_to_model_params("QRDQN", dqn_optuna_params)
