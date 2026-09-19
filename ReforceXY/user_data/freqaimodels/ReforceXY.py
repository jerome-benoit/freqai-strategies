import copy
import fcntl
import gc
import json
import logging
import math
import os
import stat
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from functools import wraps
from inspect import iscoroutinefunction, unwrap
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Final,
    Literal,
    NamedTuple,
    assert_never,
    cast,
)
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import optunahub
import pandas as pd
import rapidjson
import torch as th
from datasieve.pipeline import Pipeline
from freqtrade.exceptions import DependencyException
from freqtrade.freqai.data_drawer import (
    FEATURE_PIPELINE,
    METADATA,
    METADATA_NUMBER_MODE,
    FreqaiDataDrawer,
)
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment
from freqtrade.freqai.RL.BaseReinforcementLearningModel import (
    BaseReinforcementLearningModel,
)
from freqtrade.freqai.tensorboard.TensorboardCallback import TensorboardCallback
from freqtrade.persistence import Trade
from freqtrade.strategy import timeframe_to_minutes
from gymnasium.spaces import Box
from joblib.externals import cloudpickle
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from optuna import Trial, TrialPruned, create_study, delete_study
from optuna.exceptions import ExperimentalWarning
from optuna.pruners import BasePruner, HyperbandPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.storages import (
    BaseStorage,
    JournalStorage,
    RDBStorage,
    RetryFailedTrialCallback,
)
from optuna.storages.journal import JournalFileBackend
from optuna.study import Study, StudyDirection
from optuna.trial import TrialState
from pandas import DataFrame, merge
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
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


def _update_eval_best_reward(callback: Any, mean_reward: float, model: Any) -> None:
    """Update plain-fit final-policy bookkeeping and save an improved checkpoint."""
    if not np.isfinite(mean_reward):
        return
    callback.last_mean_reward = mean_reward
    if not hasattr(callback, "best_mean_reward") or mean_reward > callback.best_mean_reward:
        callback.best_mean_reward = mean_reward
        if callback.best_model_save_path is not None and model is not None:
            model.save(Path(callback.best_model_save_path) / "best_model.zip")


_DATE_PRED_DEDUP_SENTINEL = "_freqai_strategies_date_pred_repair_patched"


def _recorded_prediction_mask(frame: pd.DataFrame) -> NDArray[np.bool_]:
    """Identify recorded rows from metadata, never from prediction magnitudes."""
    recorded = np.zeros(len(frame), dtype=bool)
    if "close_price" in frame:
        close = pd.to_numeric(frame["close_price"], errors="coerce")
        recorded |= (close.gt(0) & close.lt(np.inf)).fillna(False).to_numpy(dtype=bool)
    if "do_predict" in frame:
        status = pd.to_numeric(frame["do_predict"], errors="coerce")
        recorded |= (
            (status.gt(-np.inf) & status.lt(np.inf) & status.ne(0))
            .fillna(False)
            .to_numpy(dtype=bool)
        )
    return recorded


def _produced_prediction_mask(frame: pd.DataFrame) -> NDArray[np.bool_]:
    """Exclude expired-model placeholders from recorded model observations."""
    produced = _recorded_prediction_mask(frame)
    if "do_predict" in frame:
        status = pd.to_numeric(frame["do_predict"], errors="coerce")
        produced &= status.ne(2).fillna(True).to_numpy(dtype=bool)
    return produced


def _dedupe_historic_predictions_on_date_pred(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize dates and retain the latest recorded row per candle, in date order.

    Freqtrade fills downtime rows with zeros/NaNs, without a candle close or a
    prediction status. Those rows must not replace recorded predictions, including
    zero predictions and rejected predictions (do_predict == 0 with a candle close).
    Indistinguishable rows use last-write-wins; label magnitudes never rank rows.
    Invalid dates cannot match a candle and are discarded.
    """
    date_pred = pd.to_datetime(frame["date_pred"], utc=True, errors="coerce", format="mixed")
    valid = date_pred.notna()
    if valid.all() and date_pred.is_monotonic_increasing and date_pred.is_unique:
        if date_pred.dtype == frame["date_pred"].dtype:
            return frame
        result = frame.copy()
        result["date_pred"] = date_pred
        return result

    recorded = _recorded_prediction_mask(frame)
    # Rank only metadata, without copying or coercing all prediction columns.
    order = pd.DataFrame(
        {"date_pred": date_pred.array, "recorded": recorded, "position": np.arange(len(frame))}
    )
    kept = (
        order.loc[valid.to_numpy()]
        .sort_values(["date_pred", "recorded", "position"])
        .drop_duplicates("date_pred", keep="last")
    )
    result = frame.iloc[kept.index].copy()
    result["date_pred"] = date_pred.iloc[kept.index].array
    return result.reset_index(drop=True)


def _align_historic_predictions(history: pd.DataFrame, dataframe: pd.DataFrame) -> pd.DataFrame:
    """Align unique, normalized history to requested candle dates and exact index."""
    dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce", format="mixed")
    indexed = history.set_index("date_pred", drop=False)
    aligned = indexed.reindex(pd.DatetimeIndex(dates))
    missing = ~pd.DatetimeIndex(dates).isin(indexed.index)
    aligned.index = dataframe.index
    aligned["date_pred"] = dates.array
    if "do_predict" in aligned:
        aligned["do_predict"] = aligned["do_predict"].where(~missing, 0)
    else:
        aligned["do_predict"] = 0
    return aligned


def _install_date_pred_dedup_patch() -> None:
    """Normalize persisted history and duplicate predictions before upstream writes.

    Normalize before upstream positional writes and before disk repair can discard
    a recorded duplicate. Already-clean upstream results are preserved. Recheck
    these synchronous method contracts on Freqtrade upgrades.
    """
    names = (
        "set_initial_return_values",
        "append_model_predictions",
        "attach_return_values_to_return_dataframe",
    )
    # Freqtrade 2026.7 has no per-frame disk-repair hook.
    original_repair = getattr(FreqaiDataDrawer, "_repair_historic_predictions", None)
    if original_repair is not None:
        names += ("_repair_historic_predictions",)
    originals = tuple(getattr(FreqaiDataDrawer, name) for name in names)
    pending = []
    for original in originals:
        current = unwrap(
            original, stop=lambda method: bool(getattr(method, _DATE_PRED_DEDUP_SENTINEL, False))
        )
        pending.append(not getattr(current, _DATE_PRED_DEDUP_SENTINEL, False))
        if iscoroutinefunction(original) or iscoroutinefunction(current):
            raise RuntimeError("Repair [global]: requires synchronous drawer methods")
    if not any(pending):
        return
    original_set_initial, original_append, original_attach = originals[:3]

    @wraps(original_set_initial)
    def set_initial_return_values(
        self, pair: str, pred_df: pd.DataFrame, dataframe: pd.DataFrame
    ) -> None:
        self.historic_predictions[pair] = _dedupe_historic_predictions_on_date_pred(
            self.historic_predictions[pair]
        )
        original_set_initial(
            self, pair, pred_df.reset_index(drop=True), dataframe.reset_index(drop=True)
        )
        repaired = _dedupe_historic_predictions_on_date_pred(self.historic_predictions[pair])
        self.historic_predictions[pair] = repaired
        self.model_return_values[pair] = _align_historic_predictions(repaired, dataframe)

    @wraps(original_append)
    def append_model_predictions(
        self,
        pair: str,
        predictions: pd.DataFrame,
        do_preds: NDArray[np.int_],
        dk: FreqaiDataKitchen,
        strat_df: pd.DataFrame,
    ) -> None:
        self.historic_predictions[pair] = _dedupe_historic_predictions_on_date_pred(
            self.historic_predictions[pair]
        )
        if self.historic_predictions[pair].empty and not strat_df.empty:
            # Append requires an initialized row; let upstream construct it.
            original_set_initial(
                self,
                pair,
                pd.DataFrame(index=pd.RangeIndex(1)),
                strat_df.tail(1).reset_index(drop=True),
            )
        original_append(self, pair, predictions, do_preds, dk, strat_df)
        repaired = _dedupe_historic_predictions_on_date_pred(self.historic_predictions[pair])
        self.historic_predictions[pair] = repaired
        self.model_return_values[pair] = _align_historic_predictions(repaired, strat_df)

    @wraps(original_attach)
    def attach_return_values_to_return_dataframe(
        self, pair: str, dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        repaired = _dedupe_historic_predictions_on_date_pred(self.historic_predictions[pair])
        self.historic_predictions[pair] = repaired
        self.model_return_values[pair] = _align_historic_predictions(repaired, dataframe)
        return original_attach(self, pair, dataframe)

    replacements = (
        set_initial_return_values,
        append_model_predictions,
        attach_return_values_to_return_dataframe,
    )
    if original_repair is not None:

        @wraps(original_repair)
        def repair_historic_predictions(self, pair: str, pair_df: pd.DataFrame) -> pd.DataFrame:
            if "date_pred" in pair_df:
                pair_df = _dedupe_historic_predictions_on_date_pred(pair_df)
            return original_repair(self, pair, pair_df)

        replacements += (repair_historic_predictions,)
    for name, replacement, needed in zip(names, replacements, pending, strict=True):
        if needed:
            setattr(replacement, _DATE_PRED_DEDUP_SENTINEL, True)
            setattr(FreqaiDataDrawer, name, replacement)


_install_date_pred_dedup_patch()

ModelType = Literal["PPO", "RecurrentPPO", "MaskablePPO", "DQN", "QRDQN"]
ScheduleTypeKnown = Literal["linear", "constant"]
ScheduleType = ScheduleTypeKnown | Literal["unknown"]
ExitPotentialMode = Literal[
    "canonical",
    "non_canonical",
    "progressive_release",
    "spike_cancel",
    "retain_previous",
]
TransformFunction = Literal["tanh", "softsign", "arctan", "sigmoid", "softsign_sqrt", "clip"]
ExitAttenuationMode = Literal["sqrt", "linear", "power", "half_life"]
ActivationFunction = Literal["relu", "tanh", "elu", "leaky_relu"]
OptimizerClassOptuna = Literal["adamw", "rmsprop"]
OptimizerClass = OptimizerClassOptuna | Literal["adam"]
NetArchSize = Literal["small", "medium", "large", "extra_large"]
StorageBackend = Literal["sqlite", "file"]
SamplerType = Literal["tpe", "auto"]
_StorageFactory = Callable[[Path], JournalStorage]


class _Samplers(NamedTuple):
    tpe: Literal["tpe"] = "tpe"
    auto: Literal["auto"] = "auto"


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
            "fit_live_predictions_candles": 0,      // Optional non-negative integer; omitted or 0 disables action statistics
            // Live/dry-run: latest N produced observations per pair after session startup.
            // Restart resets warmup; numeric mean/std are zero until N observations exist.
            // Downtime and expired status 2 do not count; neutral/exit actions and recorded
            // rejected predictions do. Backtests use the previous N rows, not the current row.
            // Numeric objects are coerced; non-finite samples are ignored. Empty finite
            // samples yield zeros; constants have zero spread; nonnumeric objects are skipped.
            // Statistics do not gate RL actions. Population standard deviation is used.
            "model_training_parameters": {
                "device": "auto",                   // PyTorch device (auto|cpu|cuda|cuda:0)
                "gpu_memory_fraction": null,        // GPU VRAM fraction limit per process (0.0, 1.0], null disables
            },
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
                "sampler": "tpe",                   // Optuna sampler (tpe|auto)
                "storage": "sqlite",                // Optuna storage backend (sqlite|file)
                "purge_period": 0,                  // Purge Optuna study every X retrains (0 disables)
                "seed": 42,                         // RNG seed
            }
        }
    }

    Continual learning keeps the deployed policy's fitted feature coordinates frozen,
    including feature selection and scaling. Reset trained models or use a new
    freqai.identifier to change those coordinates.
    First training and continual_learning=false use fresh pipelines and policies.
    Hyperopt always uses fresh current-window pipelines and cold policies; final
    continuation uses the original raw split in the deployed coordinates. Selected
    constructor/network parameters do not rebuild an existing policy's architecture;
    a resumed policy keeps its own discount gamma for environment reward shaping.
    Raw OHLC prices are preserved separately for every environment. Noise augments
    training features only, never evaluation or prediction features.
    Prediction uses the best checkpoint selected by the current fit's evaluation,
    falling back to the final policy if no usable current checkpoint was selected.

    Requirements:
        - pip install optuna optunahub -r https://hub.optuna.org/samplers/auto_sampler/requirements.txt

    Optional:
        - pip install optuna-dashboard
    """

    _LOG_2: Final[float] = math.log(2.0)

    _DEPLOYMENT_COORDINATE_MARKER_KEY: Final[str] = "reforcexy_deployment_coordinates"
    _DEPLOYMENT_COORDINATE_GENERATION: Final[str] = "chronological-frozen-pipelines-v2"

    DEFAULT_BASE_FACTOR: Final[float] = 100.0

    DEFAULT_MAX_TRADE_DURATION_CANDLES: Final[int] = 128
    DEFAULT_IDLE_DURATION_MULTIPLIER: Final[int] = 4

    DEFAULT_EXIT_POTENTIAL_DECAY: Final[float] = 0.5
    DEFAULT_ENTRY_ADDITIVE_ENABLED: Final[bool] = False
    DEFAULT_ENTRY_ADDITIVE_RATIO: Final[float] = 0.0625
    DEFAULT_ENTRY_ADDITIVE_GAIN: Final[float] = 1.0
    DEFAULT_HOLD_POTENTIAL_ENABLED: Final[bool] = False
    DEFAULT_HOLD_POTENTIAL_RATIO: Final[float] = 0.001
    DEFAULT_HOLD_POTENTIAL_GAIN: Final[float] = 1.0
    DEFAULT_EXIT_ADDITIVE_ENABLED: Final[bool] = False
    DEFAULT_EXIT_ADDITIVE_RATIO: Final[float] = 0.0625
    DEFAULT_EXIT_ADDITIVE_GAIN: Final[float] = 1.0

    DEFAULT_EXIT_PLATEAU: Final[bool] = True
    DEFAULT_EXIT_PLATEAU_GRACE: Final[float] = 1.0
    DEFAULT_EXIT_LINEAR_SLOPE: Final[float] = 1.0
    DEFAULT_EXIT_HALF_LIFE: Final[float] = 0.5

    DEFAULT_PNL_AMPLIFICATION_SENSITIVITY: Final[float] = 2.0
    DEFAULT_WIN_REWARD_FACTOR: Final[float] = 2.0
    DEFAULT_EFFICIENCY_WEIGHT: Final[float] = 1.0
    DEFAULT_EFFICIENCY_CENTER: Final[float] = 0.5

    DEFAULT_INVALID_ACTION: Final[float] = -2.0
    DEFAULT_IDLE_PENALTY_RATIO: Final[float] = 1.0
    DEFAULT_IDLE_PENALTY_POWER: Final[float] = 1.025
    DEFAULT_HOLD_PENALTY_RATIO: Final[float] = 1.0
    DEFAULT_HOLD_PENALTY_POWER: Final[float] = 1.025

    DEFAULT_CHECK_INVARIANTS: Final[bool] = True
    DEFAULT_EXIT_FACTOR_THRESHOLD: Final[float] = 1_000.0

    DEFAULT_EFFICIENCY_MIN_RANGE_EPSILON: Final[float] = 1e-6
    DEFAULT_EFFICIENCY_MIN_RANGE_FRACTION: Final[float] = 0.01

    _MODEL_TYPES: Final[tuple[ModelType, ...]] = (
        "PPO",
        "RecurrentPPO",
        "MaskablePPO",
        "DQN",
        "QRDQN",
    )
    _MODEL_TYPES_SET: Final[frozenset[ModelType]] = frozenset(_MODEL_TYPES)
    _SCHEDULE_TYPES_KNOWN: Final[tuple[ScheduleTypeKnown, ...]] = ("linear", "constant")
    _SCHEDULE_TYPES: Final[tuple[ScheduleType, ...]] = (
        *_SCHEDULE_TYPES_KNOWN,
        "unknown",
    )
    _EXIT_POTENTIAL_MODES: Final[tuple[ExitPotentialMode, ...]] = (
        "canonical",
        "non_canonical",
        "progressive_release",
        "spike_cancel",
        "retain_previous",
    )
    _EXIT_POTENTIAL_MODES_SET: Final[frozenset[ExitPotentialMode]] = frozenset(
        _EXIT_POTENTIAL_MODES
    )
    _TRANSFORM_FUNCTIONS: Final[tuple[TransformFunction, ...]] = (
        "tanh",
        "softsign",
        "arctan",
        "sigmoid",
        "softsign_sqrt",
        "clip",
    )
    _TRANSFORM_FUNCTIONS_SET: Final[frozenset[TransformFunction]] = frozenset(_TRANSFORM_FUNCTIONS)
    _EXIT_ATTENUATION_MODES: Final[tuple[ExitAttenuationMode, ...]] = (
        "sqrt",
        "linear",
        "power",
        "half_life",
    )
    _ACTIVATION_FUNCTIONS: Final[tuple[ActivationFunction, ...]] = (
        "relu",
        "tanh",
        "elu",
        "leaky_relu",
    )
    _OPTIMIZER_CLASSES_OPTUNA: Final[tuple[OptimizerClassOptuna, ...]] = (
        "adamw",
        "rmsprop",
    )
    _OPTIMIZER_CLASSES: Final[tuple[OptimizerClass, ...]] = (
        *_OPTIMIZER_CLASSES_OPTUNA,
        "adam",
    )
    _NET_ARCH_SIZES: Final[tuple[NetArchSize, ...]] = (
        "small",
        "medium",
        "large",
        "extra_large",
    )
    _STORAGE_BACKENDS: Final[tuple[StorageBackend, ...]] = ("sqlite", "file")
    _SAMPLERS: Final[_Samplers] = _Samplers()
    _JOURNAL_TAIL_PROBE_BYTES: Final[int] = 64 * 1024
    _JOURNAL_OP_CODE_KEY: Final[str] = "op_code"
    _JOURNAL_OPERATION_CODES: Final[frozenset[int]] = frozenset(range(10))
    _JOURNAL_RECOVERABLE_ERRORS: Final[type[Exception] | tuple[type[Exception], ...]] = (
        KeyError,
        AssertionError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    )
    _QUARANTINE_TAG: Final[str] = "corrupt"
    _QUARANTINE_TIE_BREAK_LIMIT: Final[int] = 99
    _BEST_PARAMS_LOCK_FILENAME: Final[str] = ".hyperopt-best-params.lock"
    # Bump on objective changes that make persisted trials or best params incompatible.
    _OPTUNA_OBJECTIVE_IDENTITY: Final[str] = (
        "terminal-liquidation-risk-normalized-trained-policy-v3"
    )
    _PPO_N_STEPS: Final[tuple[int, ...]] = (512, 1024, 2048, 4096)
    _PPO_N_STEPS_MIN: Final[int] = min(_PPO_N_STEPS)
    _PPO_N_STEPS_MAX: Final[int] = max(_PPO_N_STEPS)
    _DQN_TRAIN_FREQS: Final[tuple[int, ...]] = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    _HYPEROPT_EVAL_FREQ_REDUCTION_FACTOR: Final[float] = 4.0

    _action_masks_cache: ClassVar[dict[tuple[bool, float], NDArray[np.bool_]]] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fit_live_predictions_candles: Any = self.freqai_info.get("fit_live_predictions_candles", 0)
        # Freqtrade's schema validates the integer type, not the range; a
        # negative window produces NaN backtest label statistics.
        if (
            isinstance(fit_live_predictions_candles, bool)
            or not isinstance(fit_live_predictions_candles, int)
            or fit_live_predictions_candles < 0
        ):
            raise ValueError(
                f"Config [global]: fit_live_predictions_candles="
                f"{fit_live_predictions_candles!r} invalid; "
                "must be a non-negative integer (0 disables label statistics)"
            )
        self.freqai_info["fit_live_predictions_candles"] = fit_live_predictions_candles

        self.pairs: list[str] = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.pairs:
            raise ValueError(
                "Config [global]: missing 'pair_whitelist' in exchange section "
                "or StaticPairList method not defined in pairlists configuration"
            )
        self.action_masking: bool = self.model_type == ReforceXY._MODEL_TYPES[2]  # "MaskablePPO"
        self.rl_config.setdefault("action_masking", self.action_masking)
        self.inference_masking: bool = self.rl_config.get("inference_masking", True)
        self.recurrent: bool = self.model_type == ReforceXY._MODEL_TYPES[1]  # "RecurrentPPO"
        self.lr_schedule: bool = self.rl_config.get("lr_schedule", False)
        self.cr_schedule: bool = self.rl_config.get("cr_schedule", False)
        self.n_envs: int = self.rl_config.get("n_envs", 1)
        self.n_eval_envs: int = self.rl_config.get("n_eval_envs", 1)
        self.multiprocessing: bool = self.rl_config.get("multiprocessing", False)
        self.eval_multiprocessing: bool = self.rl_config.get("eval_multiprocessing", False)
        self.frame_stacking: int = self.rl_config.get("frame_stacking", 0)
        self.n_eval_steps: int = self.rl_config.get("n_eval_steps", 10_000)
        self.n_eval_episodes: int = self.rl_config.get("n_eval_episodes", 5)
        self.max_no_improvement_evals: int = self.rl_config.get("max_no_improvement_evals", 0)
        self.min_evals: int = self.rl_config.get("min_evals", 0)
        self.rl_config.setdefault("tensorboard_throttle", 1)
        self.plot_new_best: bool = self.rl_config.get("plot_new_best", False)
        self.check_envs: bool = self.rl_config.get("check_envs", True)
        self.progressbar_callback: ProgressBarCallback | None = None
        # Optuna hyperopt
        self.rl_config_optuna: dict[str, Any] = self.freqai_info.get("rl_config_optuna", {})
        self.hyperopt: bool = (
            self.freqai_info.get("enabled", False)
            and self.rl_config_optuna.get("enabled", False)
            and self.data_split_parameters.get("test_size", 0.1) > 0
        )
        self.optuna_timeout_hours: float = self.rl_config_optuna.get("timeout_hours", 0)
        self.optuna_n_trials: int = self.rl_config_optuna.get("n_trials", 100)
        self.optuna_n_startup_trials: int = self.rl_config_optuna.get("n_startup_trials", 15)
        self.optuna_purge_period: int = int(self.rl_config_optuna.get("purge_period", 0))
        self.optuna_eval_callback: MaskableTrialEvalCallback | None = None
        self._model_params_cache: dict[str, Any] | None = None
        self._frame_buffers: dict[str, tuple[Any, deque[NDArray[np.float32]]]] = {}
        self._lstm_states_cache: dict[
            str,
            tuple[
                int,
                tuple[NDArray[np.float32], NDArray[np.float32]] | None,
                NDArray[np.bool_],
            ],
        ] = {}
        self.unset_unsupported()
        self._configure_gpu_memory()
        self._install_replay_persistence()

    def _install_replay_persistence(self) -> None:
        save_data = self.dd.save_data
        load_data = self.dd.load_data

        @wraps(save_data)
        def save_with_replay(model, coin, dk):
            dk.data = dk.data.copy()
            dk.data.pop("reforcexy_replay", None)
            if hasattr(model, "save_replay_buffer"):
                dk.data_path.mkdir(parents=True, exist_ok=True)
                filename = f"{dk.model_filename}_replay_{uuid4().hex}.pkl"
                destination = dk.data_path / filename
                temporary = destination.with_suffix(".tmp")
                try:
                    model.save_replay_buffer(temporary)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
                dk.data["reforcexy_replay"] = filename
            return save_data(model, coin, dk)

        @wraps(load_data)
        def load_with_replay(coin, dk):
            cached = self.dd.model_dictionary.get(coin) if dk.live else None
            model = load_data(coin, dk)
            if model is not None and model is not cached:
                try:
                    self._restore_replay(model, dk.data, dk.data_path, coin)
                except Exception:
                    if self.dd.model_dictionary.get(coin) is model:
                        self.dd.model_dictionary.pop(coin, None)
                    raise
            return model

        self.dd.save_data = save_with_replay
        self.dd.load_data = load_with_replay

    @staticmethod
    def _restore_replay(model: Any, metadata: dict[str, Any], directory: Path, pair: str) -> None:
        if not hasattr(model, "load_replay_buffer"):
            return
        filename = metadata.get("reforcexy_replay")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise DependencyException(
                f"Training [{pair}]: DQN/QRDQN replay metadata is missing or incompatible; "
                "reset trained models or use a new freqai.identifier"
            )
        model.load_replay_buffer(directory / filename)
        replay = model.replay_buffer
        if (
            replay.observation_space != model.observation_space
            or replay.action_space != model.action_space
            or replay.n_envs != model.n_envs
        ):
            raise DependencyException(
                f"Training [{pair}]: DQN/QRDQN replay buffer is incompatible; reset trained "
                "models or use a new freqai.identifier"
            )

    def _configure_gpu_memory(self) -> None:
        """
        Configure GPU memory fraction limit from model_training_parameters.
        Called after config validation, before any CUDA operations.
        """
        gpu_memory_fraction: float | None = self.model_training_parameters.get(
            "gpu_memory_fraction"
        )
        if gpu_memory_fraction is None:
            return
        if not th.cuda.is_available():
            logger.warning(
                "Config [global]: gpu_memory_fraction=%.2f ignored; CUDA not available",
                gpu_memory_fraction,
            )
            return
        if not 0.0 < gpu_memory_fraction <= 1.0:
            logger.warning(
                "Config [global]: gpu_memory_fraction=%.2f invalid; must be in (0.0, 1.0]",
                gpu_memory_fraction,
            )
            return
        th.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=0)
        logger.info(
            "Config [global]: gpu_memory_fraction=%.2f applied",
            gpu_memory_fraction,
        )

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
            logger.debug(
                "ActionMask [global]: cache hit for can_short=%s position=%s",
                can_short,
                position.name,
            )
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
        logger.debug(
            "ActionMask [global]: cache miss for can_short=%s position=%s; computed=%s",
            can_short,
            position.name,
            action_masks,
        )
        return ReforceXY._action_masks_cache[cache_key]

    def unset_unsupported(self) -> None:
        """
        If user has activated any features that may conflict, this
        function will set them to proper values and warn them
        """
        if not isinstance(self.n_envs, int) or self.n_envs < 1:
            logger.warning("Config [global]: n_envs=%r invalid; defaulting to 1", self.n_envs)
            self.n_envs = 1
        if not isinstance(self.n_eval_envs, int) or self.n_eval_envs < 1:
            logger.warning(
                "Config [global]: n_eval_envs=%r invalid; defaulting to 1",
                self.n_eval_envs,
            )
            self.n_eval_envs = 1
        if self.multiprocessing and self.n_envs <= 1:
            logger.warning(
                "Config [global]: multiprocessing=True requires n_envs=%d>1; defaulting to False",
                self.n_envs,
            )
            self.multiprocessing = False
        if self.eval_multiprocessing and self.n_eval_envs <= 1:
            logger.warning(
                "Config [global]: eval_multiprocessing=True requires n_eval_envs=%d>1; defaulting to False",
                self.n_eval_envs,
            )
            self.eval_multiprocessing = False
        if self.multiprocessing and self.plot_new_best:
            logger.warning(
                "Config [global]: plot_new_best=True incompatible with multiprocessing=True; defaulting to False"
            )
            self.plot_new_best = False
        if not isinstance(self.frame_stacking, int) or self.frame_stacking < 0:
            logger.warning(
                "Config [global]: frame_stacking=%r invalid; defaulting to 0",
                self.frame_stacking,
            )
            self.frame_stacking = 0
        if self.frame_stacking == 1:
            logger.warning(
                "Config [global]: frame_stacking=1 equivalent to no stacking; defaulting to 0"
            )
            self.frame_stacking = 0
        if not isinstance(self.n_eval_steps, int) or self.n_eval_steps <= 0:
            logger.warning(
                "Config [global]: n_eval_steps=%r invalid; defaulting to 10000",
                self.n_eval_steps,
            )
            self.n_eval_steps = 10_000
        if not isinstance(self.n_eval_episodes, int) or self.n_eval_episodes <= 0:
            logger.warning(
                "Config [global]: n_eval_episodes=%r invalid; defaulting to 5",
                self.n_eval_episodes,
            )
            self.n_eval_episodes = 5
        if not isinstance(self.optuna_purge_period, int) or self.optuna_purge_period < 0:
            logger.warning(
                "Config [global]: purge_period=%r invalid; defaulting to 0",
                self.optuna_purge_period,
            )
            self.optuna_purge_period = 0
        if self.rl_config_optuna.get("continuous", False) and self.optuna_purge_period > 0:
            logger.warning(
                "Config [global]: purge_period has no effect when continuous=True; defaulting to 0"
            )
            self.optuna_purge_period = 0
        hold_potential_enabled = self.rl_config.get("model_reward_parameters", {}).get(
            "hold_potential_enabled", ReforceXY.DEFAULT_HOLD_POTENTIAL_ENABLED
        )
        if MyRLEnv.is_unsupported_pbrs_config(
            hold_potential_enabled, self.rl_config.get("add_state_info", False)
        ):
            logger.warning(
                "Config [global]: hold_potential_enabled=True requires add_state_info=True; "
                "enabling add_state_info"
            )
            self.rl_config["add_state_info"] = True
        tensorboard_throttle = self.rl_config.get("tensorboard_throttle", 1)
        if not isinstance(tensorboard_throttle, int) or tensorboard_throttle < 1:
            logger.warning(
                "Config [global]: tensorboard_throttle=%r invalid; defaulting to 1",
                tensorboard_throttle,
            )
            self.rl_config["tensorboard_throttle"] = 1
        if self.continual_learning and bool(self.frame_stacking):
            logger.warning(
                "Config [global]: continual_learning=True incompatible with frame_stacking=%d; defaulting to False",
                self.frame_stacking,
            )
            self.continual_learning = False
        if self.recurrent and bool(self.frame_stacking):
            logger.warning(
                "Config [global]: RecurrentPPO with frame_stacking=%d is redundant; "
                "LSTM already captures temporal dependencies. Consider setting frame_stacking=0",
                self.frame_stacking,
            )

    def unset_outlier_removal(self) -> None:
        super().unset_outlier_removal()
        # RL transitions, durations and rewards require chronological candles;
        # a post-split shuffle would silently corrupt the learned trajectory.
        if self.ft_params.get("shuffle_after_split", False):
            self.ft_params.update({"shuffle_after_split": False})
            logger.warning(
                "Config [global]: shuffle_after_split=True reorders RL transitions; "
                "setting shuffle_after_split to False"
            )

    def pack_env_dict(
        self, pair: str, model_params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not self.live and self.rl_config.get("model_reward_parameters", {}).get(
            "hold_potential_enabled", ReforceXY.DEFAULT_HOLD_POTENTIAL_ENABLED
        ):
            raise ValueError(
                f"Config [{pair}]: backtesting does not support hold_potential_enabled=True "
                "because add_state_info is unavailable"
            )
        env_info = super().pack_env_dict(pair)
        # Each environment owns its effective parameters; do not mutate global config.
        config = copy.deepcopy(env_info["config"])
        env_info["config"] = config
        model_reward_parameters = config["freqai"]["rl_config"].setdefault(
            "model_reward_parameters", {}
        )
        effective_params = self.get_model_params() if model_params is None else model_params
        gamma = effective_params.get("gamma")

        if gamma is not None:
            model_reward_parameters["potential_gamma"] = gamma
        else:
            logger.warning("Env [%s]: no valid discount gamma resolved for environment", pair)

        return env_info

    def set_train_and_eval_environments(
        self,
        data_dictionary: dict[str, DataFrame],
        prices_train: DataFrame,
        prices_test: DataFrame,
        dk: FreqaiDataKitchen,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Set training and evaluation environments
        """
        data_dictionary["train_prices"] = prices_train
        data_dictionary["test_prices"] = prices_test
        if self.train_env is not None or self.eval_env is not None:
            logger.info("Env [%s]: closing environments", dk.pair)
            self.close_envs()

        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]
        env_dict = self.pack_env_dict(dk.pair, model_params)
        effective_params = self.get_model_params() if model_params is None else model_params
        seed = effective_params.get("seed", 42)

        if self.check_envs:
            logger.info("Env [%s]: checking environments", dk.pair)
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
            if self.data_split_parameters.get("test_size", 0.1) > 0:
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
            "Env [%s]: populating %s train and %s eval environments",
            dk.pair,
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

    def get_model_params(self) -> dict[str, Any]:
        """
        Get model parameters
        """
        if self._model_params_cache is not None:
            return copy.deepcopy(self._model_params_cache)

        model_params: dict[str, Any] = copy.deepcopy(self.model_training_parameters)

        model_params.setdefault("seed", 42)
        model_params.setdefault("gamma", 0.95)

        if not self.hyperopt and self.lr_schedule:
            lr = model_params.get("learning_rate", 0.0003)
            if isinstance(lr, (int, float)):
                lr = float(lr)
                model_params["learning_rate"] = get_schedule(
                    cast("ScheduleTypeKnown", ReforceXY._SCHEDULE_TYPES[0]), lr
                )
                logger.info(
                    "Config [global]: learning rate linear schedule enabled, initial=%.6f",
                    lr,
                )

        # "PPO"
        if not self.hyperopt and ReforceXY._MODEL_TYPES[0] in self.model_type and self.cr_schedule:
            cr = model_params.get("clip_range", 0.2)
            if isinstance(cr, (int, float)):
                cr = float(cr)
                model_params["clip_range"] = get_schedule(
                    cast("ScheduleTypeKnown", ReforceXY._SCHEDULE_TYPES[0]), cr
                )
                logger.info(
                    "Config [global]: clip range linear schedule enabled, initial=%.2f",
                    cr,
                )

        # "DQN"
        if ReforceXY._MODEL_TYPES[3] in self.model_type:
            if model_params.get("gradient_steps") is None:
                model_params["gradient_steps"] = compute_gradient_steps(
                    model_params.get("train_freq"), model_params.get("subsample_steps")
                )
            if "subsample_steps" in model_params:
                model_params.pop("subsample_steps", None)

        if not model_params.get("policy_kwargs"):
            model_params["policy_kwargs"] = {}

        default_net_arch: list[int] = [128, 128]
        net_arch: list[int] | dict[str, list[int]] | NetArchSize = model_params.get(
            "policy_kwargs", {}
        ).get("net_arch", default_net_arch)

        # "PPO"
        if ReforceXY._MODEL_TYPES[0] in self.model_type:
            if isinstance(net_arch, str):
                if net_arch in ReforceXY._NET_ARCH_SIZES:
                    model_params["policy_kwargs"]["net_arch"] = get_net_arch(
                        self.model_type,
                        cast("NetArchSize", net_arch),
                    )
                else:
                    logger.warning(
                        "Config [global]: net_arch=%r invalid; defaulting to %r. Valid: %s",
                        net_arch,
                        {"pi": default_net_arch, "vf": default_net_arch},
                        ", ".join(ReforceXY._NET_ARCH_SIZES),
                    )
                    model_params["policy_kwargs"]["net_arch"] = {
                        "pi": default_net_arch,
                        "vf": default_net_arch,
                    }
            elif isinstance(net_arch, list):
                model_params["policy_kwargs"]["net_arch"] = {
                    "pi": net_arch,
                    "vf": net_arch,
                }
            elif isinstance(net_arch, dict):
                pi = (
                    net_arch.get("pi") if isinstance(net_arch.get("pi"), list) else default_net_arch
                )
                vf = (
                    net_arch.get("vf") if isinstance(net_arch.get("vf"), list) else default_net_arch
                )
                model_params["policy_kwargs"]["net_arch"] = {"pi": pi, "vf": vf}
            else:
                logger.warning(
                    "Config [global]: net_arch type=%s invalid; defaulting to %r. Valid: %s",
                    type(net_arch).__name__,
                    {"pi": default_net_arch, "vf": default_net_arch},
                    ", ".join(ReforceXY._NET_ARCH_SIZES),
                )
                model_params["policy_kwargs"]["net_arch"] = {
                    "pi": default_net_arch,
                    "vf": default_net_arch,
                }
        else:
            if isinstance(net_arch, str):
                if net_arch in ReforceXY._NET_ARCH_SIZES:
                    model_params["policy_kwargs"]["net_arch"] = get_net_arch(
                        self.model_type,
                        cast("NetArchSize", net_arch),
                    )
                else:
                    logger.warning(
                        "Config [global]: net_arch=%r invalid; defaulting to %r. Valid: %s",
                        net_arch,
                        default_net_arch,
                        ", ".join(ReforceXY._NET_ARCH_SIZES),
                    )
                    model_params["policy_kwargs"]["net_arch"] = default_net_arch
            elif isinstance(net_arch, list):
                model_params["policy_kwargs"]["net_arch"] = net_arch
            else:
                logger.warning(
                    "Config [global]: net_arch type=%s invalid; defaulting to %r. Valid: %s",
                    type(net_arch).__name__,
                    default_net_arch,
                    ", ".join(ReforceXY._NET_ARCH_SIZES),
                )
                model_params["policy_kwargs"]["net_arch"] = default_net_arch

        model_params["policy_kwargs"]["activation_fn"] = get_activation_fn(
            model_params.get("policy_kwargs", {}).get(
                "activation_fn", ReforceXY._ACTIVATION_FUNCTIONS[0]
            )  # "relu"
        )
        model_params["policy_kwargs"]["optimizer_class"] = get_optimizer_class(
            model_params.get("policy_kwargs", {}).get(
                "optimizer_class", ReforceXY._OPTIMIZER_CLASSES[0]
            )  # "adamw"
        )

        model_params.pop("gpu_memory_fraction", None)

        self._model_params_cache = model_params
        return copy.deepcopy(self._model_params_cache)

    def get_eval_freq(
        self,
        total_timesteps: int,
        hyperopt: bool = False,
        hyperopt_reduction_factor: float = _HYPEROPT_EVAL_FREQ_REDUCTION_FACTOR,
        model_params: dict[str, Any] | None = None,
    ) -> int:
        """Calculate evaluation frequency.

        Stable-Baselines3 triggers evaluation every `eval_freq` callback calls (`n_calls`). With
        `n_envs > 1`, one callback call corresponds to `n_envs` timesteps, so this method works in
        `n_calls` units.

        - PPO: prefer `n_steps`.
        - DQN: use `ceil(n_eval_steps / n_envs)`.
        - Hyperopt: optionally reduce interval.

        Returns:
            Evaluation frequency in callback calls (`n_calls`).
        """
        if total_timesteps <= 0:
            return 1

        n_envs = self.n_envs
        max_n_calls = max(1, total_timesteps // n_envs)

        # "PPO"
        if ReforceXY._MODEL_TYPES[0] in self.model_type:
            eval_freq: int | None = None
            if model_params:
                n_steps = model_params.get("n_steps")
                if isinstance(n_steps, int) and n_steps > 0:
                    eval_freq = max(1, min(n_steps, max_n_calls))
            if eval_freq is None:
                eval_freq = next(
                    (
                        step
                        for step in sorted(ReforceXY._PPO_N_STEPS, reverse=True)
                        if step <= max_n_calls
                    ),
                    max_n_calls,
                )
        else:
            eval_freq = max(1, (self.n_eval_steps + n_envs - 1) // n_envs)

        if hyperopt and hyperopt_reduction_factor > 1.0:
            eval_freq = max(1, round(eval_freq / hyperopt_reduction_factor))

        return min(eval_freq, max_n_calls)

    def get_callbacks(
        self,
        eval_env: VecEnv | None,
        eval_freq: int,
        data_path: str,
        trial: Trial | None = None,
    ) -> list[BaseCallback]:
        """
        Get the model specific callbacks
        """
        callbacks: list[BaseCallback] = []
        self.eval_callback = None
        self.optuna_eval_callback = None
        self.progressbar_callback = None
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

        if eval_env is None:
            return callbacks

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

    def _resolve_deployment_state(
        self, dk: FreqaiDataKitchen, pair: str
    ) -> tuple[Any, Pipeline] | None:
        """Restore independent training copies of the deployed policy and feature pipeline."""
        if not self.continual_learning:
            return None
        model = self.dd.model_dictionary.get(pair)
        cached_model = model
        previous = self.dd.pair_dict.get(pair, {})
        if model is None and not previous.get("model_filename"):
            return None
        try:
            cached = self.dd.meta_data_dictionary.get(pair, {})
            metadata = cached.get(METADATA)
            feature_pipeline = cached.get(FEATURE_PIPELINE)
            if model is None or metadata is None or feature_pipeline is None:
                # The current kitchen points to the new window, not deployed files.
                previous_dk = copy.copy(dk)
                previous_dk.data_path = Path(previous["data_path"])
                previous_dk.model_filename = previous["model_filename"]
                prefix = previous_dk.data_path / previous_dk.model_filename
                if metadata is None:
                    with Path(f"{prefix}_{METADATA}.json").open("r") as fp:
                        metadata = rapidjson.load(fp, number_mode=METADATA_NUMBER_MODE)
                if feature_pipeline is None:
                    with Path(f"{prefix}_{FEATURE_PIPELINE}.pkl").open("rb") as fp:
                        feature_pipeline = cloudpickle.load(fp)
                if model is None:
                    model = self.MODELCLASS.load(Path(f"{prefix}_model"))
                    self._restore_replay(model, metadata, previous_dk.data_path, pair)
            if (
                metadata.get(self._DEPLOYMENT_COORDINATE_MARKER_KEY)
                != self._DEPLOYMENT_COORDINATE_GENERATION
            ):
                raise ValueError(
                    "persisted deployment coordinate generation is missing or incompatible"
                )
            if not isinstance(model, self.MODELCLASS):
                raise ValueError("persisted policy type differs from the configured model")
            for name, expected, actual in (
                ("features", dk.training_features_list, metadata["training_features_list"]),
                ("labels", dk.label_list, metadata["label_list"]),
                (
                    "feature pipeline inputs",
                    dk.training_features_list,
                    feature_pipeline.features_in,
                ),
            ):
                if list(expected) != list(actual):
                    raise ValueError(f"persisted {name} differ in names or order")
            if cached_model is not None:
                # SB3 archives preserve policy/optimizer state but exclude envs and loggers.
                with BytesIO() as archive:
                    model.save(archive)
                    archive.seek(0)
                    model = self.MODELCLASS.load(archive, device=cached_model.device)
                # Off-policy experience is excluded from SB3 model archives.
                if getattr(cached_model, "replay_buffer", None) is not None:
                    model.replay_buffer = copy.deepcopy(cached_model.replay_buffer)
            state = model, copy.deepcopy(feature_pipeline)
        except Exception as exc:
            raise DependencyException(
                f"Training [{pair}]: cannot continue with matching persisted pipelines: {exc}. "
                "Reset trained models or use a new freqai.identifier."
            ) from exc
        logger.info(
            "Training [%s]: continuing deployment in the persisted feature coordinate system; "
            "reset trained models to change pipeline configuration",
            pair,
        )
        return state

    def _apply_training_pipeline(
        self,
        data_dictionary: dict[str, Any],
        dk: FreqaiDataKitchen,
        deployment_state: tuple[Any, Pipeline] | None = None,
    ) -> dict[str, Any]:
        """Preserve chronological raw splits; never transform RL action labels."""
        dd = data_dictionary.copy()
        dk.feature_pipeline = (
            self.define_data_pipeline(threads=dk.thread_count)
            if deployment_state is None
            else deployment_state[1]
        )
        for split in ("train", "test"):
            features = data_dictionary[f"{split}_features"]
            labels = data_dictionary[f"{split}_labels"]
            weights = data_dictionary[f"{split}_weights"]
            if split == "test" and features.empty:
                continue
            transform = (
                dk.feature_pipeline.fit_transform
                if split == "train" and deployment_state is None
                else dk.feature_pipeline.transform
            )
            transformed, transformed_labels, transformed_weights = transform(
                features.copy(), labels.copy(), weights.copy()
            )
            transformed = cast("pd.DataFrame", transformed)
            transformed_labels = cast("pd.DataFrame", transformed_labels)
            transformed_weights = cast("NDArray[np.float64]", np.asarray(transformed_weights))
            if len(transformed) != len(features) or len(transformed_weights) != len(features):
                raise DependencyException(
                    f"Training [{dk.pair}]: RL preprocessing must preserve chronological rows "
                    "and their alignment with raw prices."
                )
            # DataSieve reconstructs frames with RangeIndex while retaining row order.
            transformed.index = features.index
            transformed_labels.index = labels.index
            if split == "train" and deployment_state is not None:
                # Noise.transform is a no-op; only augmentation may be fitted.
                for name, transformer in dk.feature_pipeline.steps:
                    if name == "noise":
                        noisy, _, _, _ = transformer.fit_transform(transformed.to_numpy(copy=True))
                        transformed = pd.DataFrame(
                            noisy, columns=transformed.columns, index=transformed.index
                        )
            dd[f"{split}_features"] = transformed
            dd[f"{split}_labels"] = transformed_labels
            dd[f"{split}_weights"] = transformed_weights
        dk.data_dictionary = dd
        return dd

    def train(self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """Train cold candidates and continue only in persisted deployment coordinates."""
        # Drawer metadata can alias this kitchen; never invalidate the deployed marker.
        dk.data = dk.data.copy()
        dk.data.pop(self._DEPLOYMENT_COORDINATE_MARKER_KEY, None)
        logger.info("Training [%s]: starting", pair)
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df, dk.training_features_list, dk.label_list, training_filter=True
        )
        raw_data = dk.make_train_test_datasets(features_filtered, labels_filtered)
        if (
            not self.hyperopt
            and self.data_split_parameters.get("test_size") == 0
            and len(raw_data["train_features"]) > 0
        ):
            # A zero-size split keeps the schema but must yield no eval environment.
            raw_data["test_features"] = raw_data["train_features"].iloc[:0].copy()
            raw_data["test_labels"] = raw_data["train_labels"].iloc[:0].copy()
            raw_data["test_weights"] = np.asarray(raw_data["train_weights"])[:0].copy()
            raw_data["train_dates"] = raw_data["train_dates"]
            raw_data["test_dates"] = raw_data["train_dates"].iloc[:0]
        self.df_raw = copy.deepcopy(raw_data["train_features"])
        dk.fit_labels()
        # Capture prices once, before normalization and optional raw-OHLC removal.
        prices_train, prices_test = self.build_ohlc_price_dataframes(raw_data, pair, dk)
        deployment_state = self._resolve_deployment_state(dk, pair)
        dd = self._apply_training_pipeline(
            raw_data, dk, deployment_state=None if self.hyperopt else deployment_state
        )
        logger.info(
            "Training [%s]: model on %d features and %d data points",
            pair,
            len(dd["train_features"].columns),
            len(dd["train_features"]),
        )
        model = self.fit(
            dd,
            dk,
            prices_train=prices_train,
            prices_test=prices_test,
            deployment_state=deployment_state,
            raw_data_dictionary=raw_data if self.hyperopt and deployment_state else None,
            **kwargs,
        )
        if model is not None:
            dk.data[self._DEPLOYMENT_COORDINATE_MARKER_KEY] = self._DEPLOYMENT_COORDINATE_GENERATION
        logger.info("Training [%s]: completed", pair)
        return model

    def fit(
        self,
        data_dictionary: dict[str, Any],
        dk: FreqaiDataKitchen,
        *,
        prices_train: DataFrame,
        prices_test: DataFrame,
        deployment_state: tuple[Any, Pipeline] | None = None,
        raw_data_dictionary: dict[str, Any] | None = None,
        **kwargs,
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
            raise ValueError(f"Training [{dk.pair}]: train_features dataframe has zero length")
        test_df = data_dictionary.get("test_features")
        eval_timesteps = len(test_df)
        train_cycles = max(1, int(self.rl_config.get("train_cycles", 25)))
        total_timesteps = ReforceXY._ceil_to_multiple(train_timesteps * train_cycles, self.n_envs)
        train_days = steps_to_days(train_timesteps, self.config.get("timeframe"))
        eval_days = steps_to_days(eval_timesteps, self.config.get("timeframe"))
        total_days = steps_to_days(total_timesteps, self.config.get("timeframe"))

        logger.info("Model [%s]: type=%s", dk.pair, self.model_type)
        logger.info(
            "Training [%s]: %s steps (%s days), %s cycles, %s env(s) -> total %s steps (%s days)",
            dk.pair,
            train_timesteps,
            train_days,
            train_cycles,
            self.n_envs,
            total_timesteps,
            total_days,
        )
        logger.info(
            "Training [%s]: eval %s steps (%s days), %s episodes, %s env(s)",
            dk.pair,
            eval_timesteps,
            eval_days,
            self.n_eval_episodes,
            self.n_eval_envs,
        )
        logger.info(
            "Config [%s]: multiprocessing=%s, eval_multiprocessing=%s, "
            "frame_stacking=%s, action_masking=%s, recurrent=%s, hyperopt=%s",
            dk.pair,
            self.multiprocessing,
            self.eval_multiprocessing,
            self.frame_stacking,
            self.action_masking,
            self.recurrent,
            self.hyperopt,
        )

        start_time = time.time()
        if self.hyperopt:
            best_params = self.optimize(dk, total_timesteps, prices_train, prices_test)
            if best_params is None:
                logger.error(
                    "Hyperopt [%s]: optimization failed, using default model params",
                    dk.pair,
                )
                best_params = self.get_model_params()
            model_params = best_params
        else:
            model_params = self.get_model_params()
        if raw_data_dictionary is not None:
            data_dictionary = self._apply_training_pipeline(
                raw_data_dictionary, dk, deployment_state=deployment_state
            )
        logger.info("Model [%s]: %s params: %s", dk.pair, self.model_type, model_params)

        if self.activate_tensorboard:
            tensorboard_log_path = Path(self.full_path / "tensorboard" / Path(dk.data_path).name)
        else:
            tensorboard_log_path = None

        model = deployment_state[0] if deployment_state is not None else None
        effective_params = dict(model_params)
        if model is not None:
            learner_gamma = getattr(model, "gamma", None)
            if isinstance(learner_gamma, (int, float)) and np.isfinite(learner_gamma):
                effective_params["gamma"] = float(learner_gamma)
        # Preserve raw prices and the resumed learner's discount in its environments.
        try:
            self.set_train_and_eval_environments(
                data_dictionary,
                prices_train,
                prices_test,
                dk,
                model_params=effective_params,
            )
            if model is not None:
                logger.info("Training [%s]: continuing from deployed model state", dk.pair)
                model.tb_logger = getattr(self, "tb_logger", None)
                model.set_env(self.train_env)
            else:
                model = self.MODELCLASS(
                    self.policy_type,
                    self.train_env,
                    tensorboard_log=tensorboard_log_path,
                    **model_params,
                )
            total_timesteps = self._align_model_budget(model, total_timesteps)

            eval_freq = self.get_eval_freq(total_timesteps, model_params=model_params)
            callbacks = self.get_callbacks(self.eval_env, eval_freq, str(dk.data_path))
            logger.debug(
                "Training [%s]: starting model.learn with total_timesteps=%d, eval_freq=%d",
                dk.pair,
                total_timesteps,
                eval_freq,
            )
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
            logger.debug("Training [%s]: model.learn completed", dk.pair)
            if self.eval_env is not None and self.eval_callback is not None:
                # Evaluate final weights before teardown; the periodic callback may not run
                # within a single-rollout budget.
                use_masking = self.eval_callback.use_masking
                final_mean_reward, _ = evaluate_policy(
                    model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    warn=False,
                    use_masking=use_masking,
                )
                _update_eval_best_reward(self.eval_callback, float(final_mean_reward), model)
        except KeyboardInterrupt:
            pass
        finally:
            if self.progressbar_callback:
                self.progressbar_callback.on_training_end()
            self.close_envs()
            if model is not None and hasattr(model, "env") and model.env is not None:
                model.env.close()
        time_spent = time.time() - start_time
        self.dd.update_metric_tracker("fit_time", time_spent, dk.pair)

        model_filepath = Path(dk.data_path) / "best_model.zip"
        current_best = np.isfinite(getattr(self.eval_callback, "best_mean_reward", np.nan))
        if current_best and model_filepath.is_file():
            logger.info("Model [%s]: found best model at %s", dk.pair, model_filepath)
            try:
                best_model = self.MODELCLASS.load(model_filepath)
                # SB3 archives exclude replay buffers; preserve experience for continuation.
                if (
                    model is not None
                    and hasattr(model, "replay_buffer")
                    and getattr(best_model, "replay_buffer", None) is not None
                ):
                    best_model.replay_buffer = model.replay_buffer
                return best_model
            except Exception as e:
                logger.error(
                    "Model [%s]: failed to load best model: %r",
                    dk.pair,
                    e,
                    exc_info=True,
                )

        logger.warning(
            "Model [%s]: no usable current best checkpoint at %s, using final model",
            dk.pair,
            model_filepath,
        )

        return model

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:
        """Compute optional action statistics from prior prediction observations."""
        fit_live_predictions_candles = self.freqai_info.get("fit_live_predictions_candles", 0)
        if not fit_live_predictions_candles:
            return

        warmed_up = True
        history = self.dd.historic_predictions[pair]
        if self.live:
            history = _dedupe_historic_predictions_on_date_pred(history)
            if not hasattr(self, "_prediction_session_cutoffs"):
                self._prediction_session_cutoffs: dict[str, pd.Timestamp] = {}
            if pair not in self._prediction_session_cutoffs:
                initial_dates = pd.to_datetime(
                    self.dd.model_return_values[pair]["date_pred"],
                    utc=True,
                    errors="coerce",
                    format="mixed",
                )
                self._prediction_session_cutoffs[pair] = initial_dates.max()
            cutoff = self._prediction_session_cutoffs[pair]
            eligible = (
                _produced_prediction_mask(history) & history["date_pred"].gt(cutoff).to_numpy()
            )
            history = history.loc[eligible]
            remaining = fit_live_predictions_candles - len(history)
            warmed_up = remaining <= 0
            if not warmed_up:
                logger.warning(
                    "Predict [%s]: fit live predictions not warmed up; "
                    "%d more produced observations required for warmup completion",
                    pair,
                    remaining,
                )
        pred_df = history.tail(fit_live_predictions_candles).reset_index(drop=True)

        dk.data["labels_mean"], dk.data["labels_std"] = {}, {}
        for label_col in dk.label_list + dk.unique_class_list:
            raw_label = pred_df.get(label_col)
            if raw_label is None:
                continue
            # Downtime filling can leave numeric predictions in object-typed columns.
            pred_label = pd.to_numeric(raw_label, errors="coerce")
            if raw_label.dtype == object and pred_label.isna().all():
                continue
            if not warmed_up:
                f = [0.0, 0.0]
            else:
                values = pred_label.to_numpy(dtype=float, na_value=np.nan)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    f = (0.0, 0.0)
                else:
                    sample_mean = float(np.mean(values))
                    sample_std = float(np.std(values, ddof=0))
                    f = (
                        sample_mean if np.isfinite(sample_mean) else 0.0,
                        sample_std if np.isfinite(sample_std) else 0.0,
                    )
            dk.data["labels_mean"][label_col], dk.data["labels_std"][label_col] = (
                f[0],
                f[1],
            )

    def get_state_info(self, pair: str) -> tuple[float, float, int]:
        """Read the live market side, unlevered profit ratio and candle duration.

        ``Trade.calc_profit_ratio`` includes effective leverage. Dividing by a
        finite positive leverage matches the environment's unlevered PnL proxy.
        """
        market_side = 0.5
        current_profit = 0.0
        trade_duration = 0
        for trade in Trade.get_trades_proxy(is_open=True):
            if trade.pair != pair:
                continue
            market_side = 0 if trade.is_short else 1
            now = datetime.now(timezone.utc).timestamp()
            trade_duration = int((now - trade.open_date_utc.timestamp()) / self.base_tf_seconds)
            if self.data_provider is None or self.data_provider._exchange is None:
                logger.warning(
                    "StateInfo [%s]: data provider or exchange unavailable; using profit=0",
                    pair,
                )
                return market_side, 0.0, trade_duration
            current_rate = self.data_provider._exchange.get_rate(
                pair, refresh=False, side="exit", is_short=trade.is_short
            )
            profit_ratio = trade.calc_profit_ratio(current_rate)
            leverage = float(trade.leverage)
            if np.isfinite(leverage) and leverage > 0.0:
                current_profit = profit_ratio / leverage
            else:
                current_profit = profit_ratio
        return market_side, current_profit, int(trade_duration)

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, NDArray[np.int_]]:
        """Feed source dates to inference; parent predict performs the filtering."""
        dk.data_dictionary["prediction_dates"] = unfiltered_df[["date"]]
        try:
            return super().predict(unfiltered_df, dk, **kwargs)
        finally:
            dk.data_dictionary.pop("prediction_dates", None)

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
        if self.live and (add_state_info or (self.action_masking and self.inference_masking)):
            position, _, trade_duration = self.get_state_info(dk.pair)
            virtual_position = ReforceXY._normalize_position(position)
            virtual_trade_duration = trade_duration
        np_dataframe: NDArray[np.float32] = dataframe.to_numpy(dtype=np.float32, copy=False)
        n = np_dataframe.shape[0]
        window_size: int = self.CONV_WIDTH
        frame_stacking: int = self.frame_stacking
        frame_stacking_enabled: bool = bool(frame_stacking) and frame_stacking > 1
        inference_masking: bool = self.action_masking and self.inference_masking
        source_valid = np.asarray(getattr(dk, "do_predict", np.ones(n))) == 1
        source_valid = source_valid & np.isfinite(np_dataframe).all(axis=1)
        dk.do_predict = np.zeros(n, dtype=np.int_)
        dates = getattr(dk, "data_dictionary", {}).get("prediction_dates")
        dates = (
            pd.DatetimeIndex(pd.to_datetime(dates["date"], utc=True)).as_unit("ns")
            if dates is not None
            else None
        )
        if dates is not None and len(dates) != n:
            raise ValueError(
                f"Predict [{dk.pair}]: prediction dates must align with feature rows: "
                f"expected {n}, got {len(dates)}"
            )
        step = pd.Timedelta(seconds=self.base_tf_seconds)
        if not hasattr(self, "_observation_cache"):
            self._observation_cache = {}
        cached = self._observation_cache.get(dk.pair)
        adjacent = (
            dates is not None
            and len(dates) == n
            and n == window_size
            and cached is not None
            and cached[0] is model
            and dates[-1] - cached[1] == step
        )
        frame_validity: deque[bool] = deque(
            cached[2] if adjacent and cached is not None else (), maxlen=max(1, frame_stacking)
        )
        if not adjacent:
            self._frame_buffers.pop(dk.pair, None)
            self._lstm_states_cache.pop(dk.pair, None)

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

        frame_buffer: deque[NDArray[np.float32]] = deque(
            maxlen=frame_stacking if frame_stacking_enabled else None
        )
        if self.live and frame_stacking_enabled:
            cached_frames = self._frame_buffers.get(dk.pair)
            if cached_frames is not None and cached_frames[0] is model and n == window_size:
                frame_buffer = cached_frames[1]
        zero_frame: NDArray[np.float32] | None = None
        model_id = id(model)
        lstm_states_cache_valid = (
            self.live
            and self.recurrent
            and n == window_size
            and dk.pair in self._lstm_states_cache
            and self._lstm_states_cache[dk.pair][0] == model_id
        )
        if lstm_states_cache_valid:
            _, lstm_states, episode_start = self._lstm_states_cache[dk.pair]
            logger.debug(
                "Predict [%s]: using cached LSTM states (model_id=%d, episode_start=%s)",
                dk.pair,
                model_id,
                episode_start,
            )
        else:
            logger.debug(
                "Predict [%s]: initializing new LSTM states (cache_valid=%s, live=%s, recurrent=%s)",
                dk.pair,
                lstm_states_cache_valid,
                self.live,
                self.recurrent,
            )
            lstm_states: tuple[NDArray[np.float32], NDArray[np.float32]] | None = None
            episode_start = np.array([True], dtype=bool)

        def _predict(start_idx: int) -> int:
            nonlocal zero_frame, lstm_states, episode_start
            end_idx: int = start_idx + window_size
            np_observation = np_dataframe[start_idx:end_idx, :]
            action_masks_param: dict[str, Any] = {}

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
                # Own only the retained window, not the full prediction dataframe.
                frame_buffer.append(np_observation.copy())
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
            if not np.isfinite(observations).all():
                dk.do_predict[end_idx - 1] = 0
                lstm_states = None
                episode_start = np.array([True], dtype=bool)
                return Actions.Neutral.value
            if self.recurrent and dk.do_predict[end_idx - 1] != 1:
                lstm_states = None
                episode_start = np.array([True], dtype=bool)
                return Actions.Neutral.value

            if inference_masking:
                action_masks_param["action_masks"] = ReforceXY.get_action_masks(
                    self.can_short, action_mask_position
                )

            if self.recurrent:
                logger.debug(
                    "Predict [%s]: model.predict with LSTM (observations.shape=%s, episode_start=%s)",
                    dk.pair,
                    observations.shape,
                    episode_start,
                )
                action, lstm_states = model.predict(
                    observations,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                    **action_masks_param,
                )
                episode_start[:] = False
                action = int(action.item())
                logger.debug(
                    "Predict [%s]: predicted action=%s (%d)",
                    dk.pair,
                    Actions(action).name,
                    action,
                )
            else:
                logger.debug(
                    "Predict [%s]: model.predict (observations.shape=%s)",
                    dk.pair,
                    observations.shape,
                )
                action, _ = model.predict(observations, deterministic=True, **action_masks_param)
                action = int(action.item())
                logger.debug(
                    "Predict [%s]: predicted action=%s (%d)",
                    dk.pair,
                    Actions(action).name,
                    action,
                )

            return action

        predicted_actions: list[int] = []
        for start_idx in range(0, n - window_size + 1):
            end = start_idx + window_size
            contiguous = dates is None or (
                not dates[start_idx:end].hasnans
                and (
                    np.diff(dates[start_idx:end].to_numpy(dtype="datetime64[ns]"))
                    == step.to_timedelta64()
                ).all()
            )
            if dates is not None and start_idx > 0 and dates[end - 1] - dates[end - 2] != step:
                frame_buffer.clear()
                frame_validity.clear()
                lstm_states = None
                episode_start = np.array([True], dtype=bool)
            valid = bool(source_valid[start_idx:end].all() and contiguous)
            frame_validity.append(valid)
            dk.do_predict[end - 1] = int(
                valid and (not frame_stacking_enabled or all(frame_validity))
            )
            action = _predict(start_idx)
            predicted_actions.append(action)
            if not self.live:
                previous_virtual_position = virtual_position
                prediction_index = start_idx + window_size - 1
                do_predict = getattr(dk, "do_predict", None)
                if do_predict is None or do_predict[prediction_index] == 1:
                    virtual_position = _update_virtual_position(action, virtual_position)
                virtual_trade_duration = _update_virtual_trade_duration(
                    virtual_position,
                    previous_virtual_position,
                    virtual_trade_duration,
                )

        pad_count = max(0, n - len(predicted_actions))
        actions_list = ([np.nan] * pad_count) + predicted_actions
        actions_df = DataFrame({"action": actions_list}, index=dataframe.index)

        if self.live and self.recurrent:
            self._lstm_states_cache[dk.pair] = (model_id, lstm_states, episode_start)
        if self.live and frame_stacking_enabled:
            self._frame_buffers[dk.pair] = (model, frame_buffer)
        if self.live and dates is not None and len(dates) == n:
            self._observation_cache[dk.pair] = (model, dates[-1], tuple(frame_validity))

        return DataFrame(dict.fromkeys(dk.label_list, actions_df["action"]))

    @staticmethod
    def delete_study(study_name: str, storage: BaseStorage) -> None:
        try:
            delete_study(study_name=study_name, storage=storage)
        except Exception as e:
            logger.warning(
                "Hyperopt [%s]: failed to delete study: %r",
                study_name,
                e,
                exc_info=True,
            )

    @staticmethod
    def _sanitize_pair(pair: str) -> str:
        """Normalize a trading pair into a safe key."""
        sanitized = pair.replace("/", "_").replace(":", "_")
        return "".join(ch for ch in sanitized if ch.isalnum() or ch in ("_", "-", "."))

    def _optuna_retrain_counters_path(self) -> Path:
        return Path(self.full_path / "optuna-retrain-counters.json")

    def _load_optuna_retrain_counters(self, pair: str) -> dict[str, int]:
        counters_path = self._optuna_retrain_counters_path()
        if not counters_path.is_file():
            return {}
        try:
            with counters_path.open("r", encoding="utf-8") as read_file:
                data: dict[str, int] = json.load(read_file)
            if isinstance(data, dict):
                result: dict[str, int] = {}
                for key, value in data.items():
                    if isinstance(key, str) and isinstance(value, int):
                        result[key] = value
                return result
        except Exception as e:
            logger.warning(
                "Hyperopt [%s]: failed to load retrain counters from %s: %r",
                pair,
                counters_path,
                e,
                exc_info=True,
            )
        return {}

    def _save_optuna_retrain_counters(self, counters: dict[str, int], pair: str) -> None:
        counters_path = self._optuna_retrain_counters_path()
        try:
            with counters_path.open("w", encoding="utf-8") as write_file:
                json.dump(counters, write_file, indent=4, sort_keys=True)
        except Exception as e:
            logger.warning(
                "Hyperopt [%s]: failed to save retrain counters to %s: %r",
                pair,
                counters_path,
                e,
                exc_info=True,
            )

    def _increment_optuna_retrain_counter(self, pair: str) -> int:
        sanitized_pair = ReforceXY._sanitize_pair(pair)
        counters = self._load_optuna_retrain_counters(pair)
        pair_count = int(counters.get(sanitized_pair, 0)) + 1
        counters[sanitized_pair] = pair_count
        self._save_optuna_retrain_counters(counters, pair)
        return pair_count

    @staticmethod
    def _journal_has_corrupt_tail(journal_path: Path) -> bool:
        try:
            if not journal_path.exists():
                return False
            size = journal_path.stat().st_size
            if size == 0:
                return False
            with journal_path.open("rb") as journal_file:
                journal_file.seek(max(0, size - ReforceXY._JOURNAL_TAIL_PROBE_BYTES))
                tail = journal_file.read()
                if not tail.endswith(b"\n"):
                    return True
                last_newline = tail.rfind(b"\n", 0, len(tail) - 1)
                while last_newline == -1 and len(tail) < size:
                    read_end = size - len(tail)
                    read_start = max(0, read_end - ReforceXY._JOURNAL_TAIL_PROBE_BYTES)
                    journal_file.seek(read_start)
                    tail = journal_file.read(read_end - read_start) + tail
                    last_newline = tail.rfind(b"\n", 0, len(tail) - 1)
        except OSError:
            return False

        if last_newline == -1:
            last_line = tail[:-1]
            fail_open_missing_op_code = size > ReforceXY._JOURNAL_TAIL_PROBE_BYTES
        else:
            last_line = tail[last_newline + 1 : -1]
            fail_open_missing_op_code = False
        if not last_line:
            return True
        try:
            record = json.loads(last_line)
        except ValueError:
            return True
        if not isinstance(record, dict):
            return True
        op_code = record.get(ReforceXY._JOURNAL_OP_CODE_KEY)
        if op_code is None and fail_open_missing_op_code:
            return False
        return type(op_code) is not int or op_code not in ReforceXY._JOURNAL_OPERATION_CODES

    @staticmethod
    def _create_recovered_journal_storage(
        journal_path: Path,
        storage_factory: _StorageFactory | None = None,
    ) -> JournalStorage:
        build_storage = storage_factory or ReforceXY._create_journal_storage
        if ReforceXY._journal_has_corrupt_tail(journal_path):
            ReforceXY._quarantine_journal(
                journal_path,
                ValueError("trailing journal record is truncated or malformed JSON"),
            )
            journal_path.touch()
        try:
            return build_storage(journal_path)
        except ReforceXY._JOURNAL_RECOVERABLE_ERRORS as exc:
            quarantined_path = ReforceXY._quarantine_journal(journal_path, exc)
            if quarantined_path is None:
                raise
            journal_path.touch()
            return build_storage(journal_path)

    @staticmethod
    def _create_journal_storage(journal_path: Path) -> JournalStorage:
        return JournalStorage(JournalFileBackend(str(journal_path)))

    @staticmethod
    def _quarantine_journal(journal_path: Path, cause: Exception) -> Path | None:
        if not journal_path.exists():
            return None
        quarantine_path = ReforceXY._quarantine_path(journal_path, datetime.now(timezone.utc))
        journal_path.rename(quarantine_path)
        logger.warning(
            "Journal [%s]: corrupt (%r); quarantined to %s; resuming with fresh journal",
            journal_path.name,
            cause,
            quarantine_path.name,
        )
        return quarantine_path

    @staticmethod
    def _quarantine_path(path: Path, now: datetime) -> Path:
        """Quarantine target path for a corrupt Optuna artefact.

        The tag and timestamp are appended after the complete filename
        (extension included) so live-artefact globs never match quarantined
        files. Collisions are bounded by ``_QUARANTINE_TIE_BREAK_LIMIT``;
        exhausted candidates raise instead of reusing a quarantine file.
        """
        stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
        base_name = f"{path.name}.{ReforceXY._QUARANTINE_TAG}-{stamp}"
        for index in range(ReforceXY._QUARANTINE_TIE_BREAK_LIMIT + 1):
            suffix = "" if index == 0 else f"-{index}"
            candidate = path.with_name(f"{base_name}{suffix}")
            if not candidate.exists():
                return candidate
        raise FileExistsError(path)

    def create_storage(self, pair: str) -> BaseStorage:
        """
        Get the storage for Optuna
        """
        storage_dir = self.full_path
        storage_filename = f"optuna-{pair.split('/')[0]}"
        storage_backend: StorageBackend = self.rl_config_optuna.get(
            "storage", ReforceXY._STORAGE_BACKENDS[0]
        )  # "sqlite"
        # "sqlite"
        if storage_backend == ReforceXY._STORAGE_BACKENDS[0]:
            storage = RDBStorage(
                url=f"sqlite:///{storage_dir}/{storage_filename}.sqlite",
                heartbeat_interval=60,
                failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
            )
        # "file"
        elif storage_backend == ReforceXY._STORAGE_BACKENDS[1]:
            storage = ReforceXY._create_recovered_journal_storage(
                storage_dir / f"{storage_filename}.log"
            )
        else:
            raise ValueError(
                f"Hyperopt [{pair}]: unsupported storage backend '{storage_backend}'. "
                f"Valid: {', '.join(ReforceXY._STORAGE_BACKENDS)}"
            )
        return storage

    @staticmethod
    def study_has_best_trial(study: Study | None) -> bool:
        if study is None:
            return False
        try:
            _ = study.best_trial
            return True
        except (ValueError, KeyError):
            return False

    def create_sampler(self) -> BaseSampler:
        sampler_value = self.rl_config_optuna.get("sampler", ReforceXY._SAMPLERS.tpe)
        if sampler_value not in ReforceXY._SAMPLERS:
            raise ValueError(
                f"Hyperopt [global]: unsupported sampler '{sampler_value}'. "
                f"Valid: {', '.join(ReforceXY._SAMPLERS)}"
            )
        sampler = cast("SamplerType", sampler_value)
        seed = self.rl_config_optuna.get("seed", 42)
        match sampler:
            case ReforceXY._SAMPLERS.tpe:
                logger.info(
                    "Hyperopt [global]: using TPESampler (n_startup_trials=%d, multivariate=True, group=True, seed=%d)",
                    self.optuna_n_startup_trials,
                    seed,
                )
                return TPESampler(
                    n_startup_trials=self.optuna_n_startup_trials,
                    multivariate=True,
                    group=True,
                    seed=seed,
                )
            case ReforceXY._SAMPLERS.auto:
                logger.info(
                    "Hyperopt [global]: using AutoSampler (seed=%d)",
                    seed,
                )
                return optunahub.load_module("samplers/auto_sampler").AutoSampler(seed=seed)
            case _:
                assert_never(sampler)

    @staticmethod
    def create_pruner(min_resource: int, max_resource: int, reduction_factor: int) -> BasePruner:
        logger.info(
            "Hyperopt [global]: using HyperbandPruner (min_resource=%d, max_resource=%d, reduction_factor=%d)",
            min_resource,
            max_resource,
            reduction_factor,
        )
        return HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
        )

    @staticmethod
    def _ceil_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    @staticmethod
    def _align_model_budget(model: Any, total_timesteps: int) -> int:
        """Use the constructed algorithm's collection window, including defaults."""
        if hasattr(model, "train_freq"):
            frequency = model.train_freq
            if frequency.unit.value != "step":
                return total_timesteps
            steps = frequency.frequency
        else:
            steps = model.n_steps
        return ReforceXY._ceil_to_multiple(total_timesteps, steps * model.n_envs)

    @staticmethod
    def _ppo_resources(total_timesteps: int, n_envs: int, reduction_factor: int) -> tuple[int, int]:
        min_n_steps = ReforceXY._PPO_N_STEPS_MIN
        max_n_steps = ReforceXY._PPO_N_STEPS_MAX
        min_resource = max(
            2 * reduction_factor,
            round(min_n_steps / ReforceXY._HYPEROPT_EVAL_FREQ_REDUCTION_FACTOR) * n_envs,
        )
        rollout = max_n_steps * n_envs
        return (
            min_resource,
            max(min_resource, ReforceXY._ceil_to_multiple(total_timesteps, rollout)),
        )

    def optimize(
        self,
        dk: FreqaiDataKitchen,
        total_timesteps: int,
        prices_train: DataFrame,
        prices_test: DataFrame,
    ) -> dict[str, Any] | None:
        """
        Runs hyperparameter optimization using Optuna and returns the best hyperparameters found merged with the user defined parameters
        Only studies and best params with the current objective identity are reused.
        """
        identifier = self.freqai_info.get("identifier", "no_id_provided")
        study_name = f"{identifier}-{dk.pair}"
        storage = self.create_storage(dk.pair)
        continuous = self.rl_config_optuna.get("continuous", False)

        pair_purge_count = self._increment_optuna_retrain_counter(dk.pair)
        pair_purge_triggered = (
            self.optuna_purge_period > 0 and pair_purge_count % self.optuna_purge_period == 0
        )

        if continuous or pair_purge_triggered:
            ReforceXY.delete_study(study_name, storage)
            if continuous and not pair_purge_triggered:
                logger.info(
                    "Hyperopt [%s]: study deleted (continuous mode)",
                    study_name,
                )

        if pair_purge_triggered:
            logger.info(
                "Hyperopt [%s]: study purged on retrain %s (purge_period=%s)",
                study_name,
                pair_purge_count,
                self.optuna_purge_period,
            )

        reduction_factor = 3
        n_envs = self.n_envs
        if ReforceXY._MODEL_TYPES[0] in self.model_type:  # "PPO"
            min_resource, max_resource = ReforceXY._ppo_resources(
                total_timesteps, n_envs, reduction_factor
            )
        else:  # "DQN"/"QRDQN": gradient windows of train_freq * n_envs
            window = max(ReforceXY._DQN_TRAIN_FREQS)
            min_resource = max(
                2 * reduction_factor,
                self.get_eval_freq(total_timesteps, hyperopt=True) * n_envs,
            )
            max_resource = max(
                min_resource,
                ReforceXY._ceil_to_multiple(total_timesteps, window * n_envs),
            )

        direction = StudyDirection.MAXIMIZE
        load_if_exists = not continuous and not pair_purge_triggered
        if load_if_exists:
            try:
                existing_study_id = storage.get_study_id_from_name(study_name)
            except KeyError:
                pass
            else:
                existing_identity = storage.get_study_user_attrs(existing_study_id).get(
                    "objective_identity"
                )
                if existing_identity != ReforceXY._OPTUNA_OBJECTIVE_IDENTITY:
                    logger.warning(
                        "Hyperopt [%s]: objective identity %r incompatible with %r; resetting study",
                        study_name,
                        existing_identity,
                        ReforceXY._OPTUNA_OBJECTIVE_IDENTITY,
                    )
                    ReforceXY.delete_study(study_name, storage)
                    # Fail closed if deletion failed rather than reload incompatible trials.
                    load_if_exists = False
        study: Study = create_study(
            study_name=study_name,
            sampler=self.create_sampler(),
            pruner=ReforceXY.create_pruner(min_resource, max_resource, reduction_factor),
            direction=direction,
            storage=storage,
            load_if_exists=load_if_exists,
        )
        if study.user_attrs.get("objective_identity") != ReforceXY._OPTUNA_OBJECTIVE_IDENTITY:
            study.set_user_attr("objective_identity", ReforceXY._OPTUNA_OBJECTIVE_IDENTITY)
        logger.info(
            "Hyperopt [%s]: study created (direction=%s, n_trials=%s, timeout=%s, continuous=%s, load_if_exists=%s)",
            study_name,
            direction.name,
            self.optuna_n_trials,
            f"{self.optuna_timeout_hours}h" if self.optuna_timeout_hours else "None",
            continuous,
            load_if_exists,
        )
        if (
            self.rl_config_optuna.get("warm_start", False) and not pair_purge_triggered
        ) or pair_purge_triggered:
            best_trial_params = self.load_best_trial_params(dk.pair)
            if best_trial_params:
                study.enqueue_trial(best_trial_params)
                logger.info(
                    "Hyperopt [%s]: warm start enqueued previous best params",
                    study_name,
                )
            else:
                logger.info(
                    "Hyperopt [%s]: warm start found no previous best params",
                    study_name,
                )
        hyperopt_failed = False
        start_time = time.time()
        try:
            study.optimize(
                lambda trial: self.objective(trial, dk, total_timesteps, prices_train, prices_test),
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
            time_spent = time.time() - start_time
            logger.info(
                "Hyperopt [%s]: interrupted by user after %.2f secs",
                study_name,
                time_spent,
            )
        except Exception as e:
            time_spent = time.time() - start_time
            logger.error(
                "Hyperopt [%s]: optimization failed after %.2f secs: %r",
                study_name,
                time_spent,
                e,
                exc_info=True,
            )
            hyperopt_failed = True
        time_spent = time.time() - start_time
        n_completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
        n_failed = len([t for t in study.trials if t.state == TrialState.FAIL])
        logger.info(
            "Hyperopt [%s]: %s completed, %s pruned, %s failed trials",
            study_name,
            n_completed,
            n_pruned,
            n_failed,
        )
        study_has_best_trial = ReforceXY.study_has_best_trial(study)
        if not study_has_best_trial:
            logger.error(
                "Hyperopt [%s]: no best trial found after %.2f secs",
                study_name,
                time_spent,
            )
            hyperopt_failed = True

        if hyperopt_failed:
            best_trial_params = self.load_best_trial_params(dk.pair)
            if best_trial_params is None:
                logger.error(
                    "Hyperopt [%s]: no previously saved best params found",
                    study_name,
                )
                return None
        else:
            best_trial_params = study.best_trial.params

        logger.info(
            "Hyperopt [%s]: completed in %.2f secs",
            study_name,
            time_spent,
        )
        if study_has_best_trial:
            logger.info(
                "Hyperopt [%s]: best trial #%d with score %s",
                study_name,
                study.best_trial.number,
                study.best_trial.value,
            )
        if self.model_type == ReforceXY._MODEL_TYPES[1] and self.get_model_params().get(
            "policy_kwargs", {}
        ).get("shared_lstm", False):
            best_trial_params = {**best_trial_params, "enable_critic_lstm": False}
        logger.info("Hyperopt [%s]: best params: %s", study_name, best_trial_params)

        self.save_best_trial_params(best_trial_params, dk.pair)

        return deepmerge(
            self.get_model_params(),
            convert_optuna_params_to_model_params(self.model_type, best_trial_params),
        )

    def _best_trial_params_path(self, pair: str) -> Path:
        return self.full_path / f"hyperopt-best-params-{ReforceXY._sanitize_pair(pair)}.json"

    @staticmethod
    @contextmanager
    def _locked_best_trial_params(
        best_trial_params_path: Path, *, exclusive: bool
    ) -> Iterator[None]:
        lock_path = best_trial_params_path.parent / ReforceXY._BEST_PARAMS_LOCK_FILENAME
        # O_NONBLOCK so a pre-existing FIFO (unlike a symlink, not caught by
        # O_NOFOLLOW) cannot hang this open before the S_ISREG guard rejects it.
        # A shared reader omits O_CREAT: a read-only mount cannot create the lock,
        # and os.replace atomicity keeps a lock-free read consistent.
        open_flags = (
            ((os.O_RDWR | os.O_CREAT) if exclusive else os.O_RDONLY)
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
            | os.O_NONBLOCK
        )
        try:
            lock_fd = os.open(lock_path, open_flags, 0o666)
        except FileNotFoundError:
            if exclusive:
                raise
            yield
            return
        try:
            if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                raise OSError(f"Hyperopt [global]: lock {lock_path} must be a regular file")
            fcntl.flock(lock_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            os.close(lock_fd)

    @staticmethod
    def _reject_best_trial_params_symlink(best_trial_params_path: Path) -> None:
        if best_trial_params_path.is_symlink():
            raise OSError(
                f"Hyperopt [global]: best params path {best_trial_params_path} must not be a symlink"
            )

    @staticmethod
    def _quarantine_corrupt_best_trial_params(
        best_trial_params_path: Path, pair: str, cause: Exception
    ) -> Path | None:
        if not best_trial_params_path.exists():
            return None
        quarantine_path = ReforceXY._quarantine_path(
            best_trial_params_path, datetime.now(timezone.utc)
        )
        try:
            best_trial_params_path.rename(quarantine_path)
        except OSError as quarantine_error:
            logger.error(
                "Hyperopt [%s]: best params %s quarantine failed: %r",
                pair,
                best_trial_params_path.name,
                quarantine_error,
                exc_info=True,
            )
            raise
        logger.warning(
            "Hyperopt [%s]: best params %s corrupt (%r); quarantined to %s; "
            "no persisted best params recovered",
            pair,
            best_trial_params_path.name,
            cause,
            quarantine_path.name,
        )
        return quarantine_path

    def save_best_trial_params(self, best_trial_params: dict[str, Any], pair: str) -> None:
        """
        Save the best trial hyperparameters with the current objective identity.
        """
        best_trial_params_path = self._best_trial_params_path(pair)
        logger.info("Hyperopt [%s]: saving best params to %s", pair, best_trial_params_path)
        temporary_path: Path | None = None
        try:
            with self._locked_best_trial_params(best_trial_params_path, exclusive=True):
                self._reject_best_trial_params_symlink(best_trial_params_path)
                try:
                    existing_metadata = best_trial_params_path.stat()
                except FileNotFoundError:
                    existing_metadata = None
                temporary_path = best_trial_params_path.with_name(
                    f".{best_trial_params_path.name}.{uuid4().hex}.tmp"
                )
                with os.fdopen(
                    os.open(
                        temporary_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o666,
                    ),
                    mode="w",
                    encoding="utf-8",
                ) as write_file:
                    if existing_metadata is not None:
                        temporary_metadata = os.fstat(write_file.fileno())
                        if (
                            temporary_metadata.st_uid != existing_metadata.st_uid
                            or temporary_metadata.st_gid != existing_metadata.st_gid
                        ):
                            # Best-effort: a non-root process on a cross-uid bind
                            # mount lacks CAP_CHOWN; the previous in-place write
                            # never chowned, so a failure must not abort the save.
                            try:
                                os.fchown(
                                    write_file.fileno(),
                                    existing_metadata.st_uid
                                    if temporary_metadata.st_uid != existing_metadata.st_uid
                                    else -1,
                                    existing_metadata.st_gid
                                    if temporary_metadata.st_gid != existing_metadata.st_gid
                                    else -1,
                                )
                            except PermissionError as chown_error:
                                logger.debug(
                                    "Hyperopt [%s]: best params ownership preservation skipped: %r",
                                    pair,
                                    chown_error,
                                )
                        os.fchmod(
                            write_file.fileno(),
                            stat.S_IMODE(existing_metadata.st_mode),
                        )
                    json.dump(
                        {
                            "objective_identity": ReforceXY._OPTUNA_OBJECTIVE_IDENTITY,
                            "params": best_trial_params,
                        },
                        write_file,
                        indent=4,
                    )
                    write_file.flush()
                    os.fsync(write_file.fileno())
                temporary_path.replace(best_trial_params_path)
                temporary_path = None
        except BaseException as error:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    logger.error(
                        "Hyperopt [%s]: best params temporary file %s cleanup failed: %r",
                        pair,
                        temporary_path.name,
                        cleanup_error,
                        exc_info=True,
                    )
            if isinstance(error, Exception):
                logger.error(
                    "Hyperopt [%s]: failed to save best params to %s: %r",
                    pair,
                    best_trial_params_path,
                    error,
                    exc_info=True,
                )
            raise

    def load_best_trial_params(self, pair: str) -> dict[str, Any] | None:
        """
        Load saved best trial hyperparameters only for the current objective identity.
        """
        best_trial_params_path = self._best_trial_params_path(pair)
        if not best_trial_params_path.parent.is_dir():
            return None
        malformed = False
        with self._locked_best_trial_params(best_trial_params_path, exclusive=False):
            self._reject_best_trial_params_symlink(best_trial_params_path)
            if not best_trial_params_path.is_file():
                return None
            logger.info(
                "Hyperopt [%s]: loading best params from %s",
                pair,
                best_trial_params_path,
            )
            try:
                with best_trial_params_path.open("r", encoding="utf-8") as read_file:
                    best_trial_params = json.load(read_file)
            except (json.JSONDecodeError, UnicodeDecodeError):
                malformed = True
        if malformed:
            with self._locked_best_trial_params(best_trial_params_path, exclusive=True):
                self._reject_best_trial_params_symlink(best_trial_params_path)
                if not best_trial_params_path.is_file():
                    return None
                try:
                    with best_trial_params_path.open("r", encoding="utf-8") as read_file:
                        best_trial_params = json.load(read_file)
                except (json.JSONDecodeError, UnicodeDecodeError) as decode_error:
                    quarantined = self._quarantine_corrupt_best_trial_params(
                        best_trial_params_path, pair, decode_error
                    )
                    if quarantined is None:
                        raise
                    return None
        if (
            not isinstance(best_trial_params, dict)
            or best_trial_params.get("objective_identity") != ReforceXY._OPTUNA_OBJECTIVE_IDENTITY
            or not isinstance(best_trial_params.get("params"), dict)
        ):
            logger.warning(
                "Hyperopt [%s]: ignoring best params with missing or incompatible objective identity "
                "(expected %r) or invalid params payload",
                pair,
                ReforceXY._OPTUNA_OBJECTIVE_IDENTITY,
            )
            return None
        return best_trial_params["params"]

    def _get_train_and_eval_environments(
        self,
        dk: FreqaiDataKitchen,
        train_df: DataFrame,
        test_df: DataFrame,
        prices_train: DataFrame,
        prices_test: DataFrame,
        seed: int | None = None,
        env_info: dict[str, Any] | None = None,
        trial: Trial | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> tuple[VecEnv, VecEnv | None]:
        seed: int = self.get_model_params().get("seed", 42) if seed is None else seed
        if trial is not None:
            seed += trial.number
        set_random_seed(seed)
        env_info: dict[str, Any] = (
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
            for i in range(
                self.n_eval_envs if self.data_split_parameters.get("test_size", 0.1) > 0 else 0
            )
        ]

        train_env = eval_env = None
        try:
            if self.multiprocessing and self.n_envs > 1:
                train_env = SubprocVecEnv(train_fns, start_method="spawn")
            else:
                train_env = _make_dummy_vec_env(train_fns)
            eval_env = None
            if eval_fns:
                if self.eval_multiprocessing and self.n_eval_envs > 1:
                    eval_env = SubprocVecEnv(eval_fns, start_method="spawn")
                else:
                    eval_env = _make_dummy_vec_env(eval_fns)

            if bool(self.frame_stacking) and self.frame_stacking > 1:
                train_env = VecFrameStack(train_env, n_stack=self.frame_stacking)
                if eval_env is not None:
                    eval_env = VecFrameStack(eval_env, n_stack=self.frame_stacking)

            train_env = VecMonitor(train_env)
            if eval_env is not None:
                eval_env = VecMonitor(eval_env)

            return train_env, eval_env
        except BaseException:
            if train_env is not None:
                train_env.close()
            if eval_env is not None:
                eval_env.close()
            raise

    def get_optuna_params(self, trial: Trial) -> dict[str, Any]:
        # "RecurrentPPO"
        if ReforceXY._MODEL_TYPES[1] in self.model_type:
            return sample_params_recurrentppo(
                trial,
                shared_lstm=self.get_model_params()
                .get("policy_kwargs", {})
                .get("shared_lstm", False),
            )
        # "PPO"
        elif ReforceXY._MODEL_TYPES[0] in self.model_type:
            return sample_params_ppo(trial)
        # "QRDQN"
        elif ReforceXY._MODEL_TYPES[4] in self.model_type:
            return sample_params_qrdqn(trial)
        # "DQN"
        elif ReforceXY._MODEL_TYPES[3] in self.model_type:
            return sample_params_dqn(trial)
        else:
            raise NotImplementedError(
                f"Hyperopt [{trial.study.study_name}]: model type '{self.model_type}' not supported"
            )

    def objective(
        self,
        trial: Trial,
        dk: FreqaiDataKitchen,
        total_timesteps: int,
        prices_train: DataFrame,
        prices_test: DataFrame,
    ) -> float:
        """
        Objective function for Optuna trials hyperparameter optimization
        """
        study_name = trial.study.study_name
        logger.info("Hyperopt [%s]: starting trial #%d", study_name, trial.number)

        params = self.get_optuna_params(trial)

        # "PPO"
        if ReforceXY._MODEL_TYPES[0] in self.model_type:
            n_steps = params.get("n_steps")
            if n_steps * self.n_envs > total_timesteps:
                raise TrialPruned(
                    f"Hyperopt [{study_name}]: n_steps={n_steps} * n_envs={self.n_envs}={n_steps * self.n_envs} is greater than total_timesteps={total_timesteps}"
                )
            batch_size = params.get("batch_size")
            if (n_steps * self.n_envs) % batch_size != 0:
                raise TrialPruned(
                    f"Hyperopt [{study_name}]: n_steps={n_steps} * n_envs={self.n_envs} = {n_steps * self.n_envs} is not divisible by batch_size={batch_size}"
                )

        # "DQN"
        if ReforceXY._MODEL_TYPES[3] in self.model_type:
            gradient_steps = params.get("gradient_steps")
            if isinstance(gradient_steps, int) and gradient_steps <= 0:
                raise TrialPruned(
                    f"Hyperopt [{study_name}]: gradient_steps={gradient_steps} is negative or zero"
                )
            batch_size = params.get("batch_size")
            buffer_size = params.get("buffer_size")
            if (batch_size * gradient_steps) > buffer_size:
                raise TrialPruned(
                    f"Hyperopt [{study_name}]: batch_size={batch_size} * gradient_steps={gradient_steps}={batch_size * gradient_steps} is greater than buffer_size={buffer_size}"
                )
            learning_starts = params.get("learning_starts")
            if learning_starts > buffer_size:
                raise TrialPruned(
                    f"Hyperopt [{study_name}]: learning_starts={learning_starts} is greater than buffer_size={buffer_size}"
                )

        # Ensure that the sampled parameters take precedence
        params = deepmerge(self.get_model_params(), params)
        params["seed"] = params.get("seed", 42) + trial.number
        logger.info("Hyperopt [%s]: trial #%d params: %s", study_name, trial.number, params)

        nan_encountered = False

        # shared_lstm=True forbids a separate critic LSTM; sampling
        # enable_critic_lstm=True would crash RecurrentPPO at construction
        # before any pruning logic can run, so the merged trial parameters
        # never carry the incompatible pair.
        if (
            ReforceXY._MODEL_TYPES[1] in self.model_type  # "RecurrentPPO"
            and bool((params.get("policy_kwargs") or {}).get("shared_lstm", False))
        ):
            params.setdefault("policy_kwargs", {})["enable_critic_lstm"] = False

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

        train_env = eval_env = model = None
        try:
            train_env, eval_env = self._get_train_and_eval_environments(
                dk,
                train_df=dk.data_dictionary["train_features"],
                test_df=dk.data_dictionary["test_features"],
                prices_train=prices_train,
                prices_test=prices_test,
                trial=trial,
                model_params=params,
            )

            model = self.MODELCLASS(
                self.policy_type,
                train_env,
                tensorboard_log=tensorboard_log_path,
                **params,
            )
            total_timesteps = self._align_model_budget(model, total_timesteps)
            off_policy = hasattr(model, "replay_buffer")
            if off_policy and model.learning_starts >= total_timesteps:
                raise TrialPruned(
                    f"Hyperopt [{study_name}]: learning_starts={model.learning_starts} "
                    f"leaves no update within total_timesteps={total_timesteps}"
                )
            initial_updates = getattr(model, "_n_updates", 0)

            eval_freq = self.get_eval_freq(total_timesteps, hyperopt=True, model_params=params)
            callbacks = self.get_callbacks(eval_env, eval_freq, str(dk.data_path), trial)
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
            if off_policy and model._n_updates <= initial_updates:
                raise TrialPruned(f"Hyperopt [{study_name}]: trial completed without learning")
            if not self.optuna_eval_callback.is_pruned and eval_env is not None:
                # Evaluate final weights before teardown; the periodic callback may not run
                # within a single-rollout budget.
                use_masking = self.optuna_eval_callback.use_masking
                final_mean_reward, _ = evaluate_policy(
                    model,
                    eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    warn=False,
                    use_masking=use_masking,
                )
                self.optuna_eval_callback.update_best_reward(float(final_mean_reward), model)
        except AssertionError as e:
            logger.warning(
                "Hyperopt [%s]: trial #%d encountered NaN (AssertionError): %r",
                study_name,
                trial.number,
                e,
                exc_info=True,
            )
            nan_encountered = True
        except ValueError as e:
            if any(x in str(e).lower() for x in ("nan", "inf")):
                logger.warning(
                    "Hyperopt [%s]: trial #%d encountered NaN/Inf (ValueError): %r",
                    study_name,
                    trial.number,
                    e,
                    exc_info=True,
                )
                nan_encountered = True
            else:
                raise
        except FloatingPointError as e:
            logger.warning(
                "Hyperopt [%s]: trial #%d encountered NaN/Inf (FloatingPointError): %r",
                study_name,
                trial.number,
                e,
                exc_info=True,
            )
            nan_encountered = True
        except RuntimeError as e:
            if any(x in str(e).lower() for x in ("nan", "inf")):
                logger.warning(
                    "Hyperopt [%s]: trial #%d encountered NaN/Inf (RuntimeError): %r",
                    study_name,
                    trial.number,
                    e,
                    exc_info=True,
                )
                nan_encountered = True
            else:
                raise
        finally:
            if self.progressbar_callback:
                self.progressbar_callback.on_training_end()
            if train_env is not None:
                train_env.close()
            if eval_env is not None:
                eval_env.close()

        if nan_encountered:
            raise TrialPruned(f"Hyperopt [{study_name}]: NaN encountered during training")

        if self.optuna_eval_callback.is_pruned:
            raise TrialPruned(f"Hyperopt [{study_name}]: pruned by eval callback")

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


def _make_dummy_vec_env(env_fns: list[Callable[[], BaseEnvironment]]) -> DummyVecEnv:
    """Release already-created environments if a later factory or wrapper fails."""
    environments = []
    try:
        for factory in env_fns:
            environments.append(factory())
        return DummyVecEnv([lambda env=env: env for env in environments])
    except BaseException:
        for environment in environments:
            environment.close()
        raise


def make_env(
    MyRLEnv: type[BaseEnvironment],
    env_id: str,
    rank: int,
    seed: int,
    df: DataFrame,
    price: DataFrame,
    env_info: dict[str, Any],
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


MyRLEnv: type[BaseEnvironment]


class MyRLEnv(Base5ActionRLEnv):
    """Env."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_observation_space()
        self.action_masking: bool = self.rl_config.get("action_masking", False)

        # === INTERNAL STATE ===
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
        model_reward_parameters: Mapping[str, Any] = self.rl_config.get(
            "model_reward_parameters", {}
        )
        self.max_trade_duration_candles: int = int(
            model_reward_parameters.get(
                "max_trade_duration_candles",
                ReforceXY.DEFAULT_MAX_TRADE_DURATION_CANDLES,
            )
        )
        self.max_idle_duration_candles: int = int(
            model_reward_parameters.get(
                "max_idle_duration_candles",
                ReforceXY.DEFAULT_IDLE_DURATION_MULTIPLIER * self.max_trade_duration_candles,
            )
        )
        # === PBRS COMMON PARAMETERS ===
        self._potential_gamma = float(model_reward_parameters.get("potential_gamma", 0.95))
        if np.isclose(self._potential_gamma, 0.0):
            logger.warning(
                "PBRS [%s]: potential_gamma=0 detected; PBRS delta will be -Φ(s) "
                "instead of γΦ(s')-Φ(s). This may cause unexpected reward behavior.",
                self.id,
            )

        # === EXIT POTENTIAL MODE ===
        # exit_potential_mode options:
        #   'canonical'           -> Φ(s')=0 (preserves invariance, disables additives)
        #   'non_canonical'       -> Φ(s')=0 (allows additives, breaks invariance)
        #   'progressive_release' -> Φ(s')=Φ(s)*(1-decay_factor)
        #   'spike_cancel'        -> Φ(s')=Φ(s)/γ (Δ ≈ 0, cancels shaping)
        #   'retain_previous'     -> Φ(s')=Φ(s)
        self._exit_potential_mode = str(
            model_reward_parameters.get(
                "exit_potential_mode", ReforceXY._EXIT_POTENTIAL_MODES[0]
            )  # "canonical"
        )
        if self._exit_potential_mode not in ReforceXY._EXIT_POTENTIAL_MODES_SET:
            logger.warning(
                "PBRS [%s]: exit_potential_mode=%r invalid; defaulting to %r. Valid: %s",
                self.id,
                self._exit_potential_mode,
                ReforceXY._EXIT_POTENTIAL_MODES[0],
                ", ".join(ReforceXY._EXIT_POTENTIAL_MODES),
            )
            self._exit_potential_mode = ReforceXY._EXIT_POTENTIAL_MODES[0]  # "canonical"
        self._exit_potential_decay: float = float(
            model_reward_parameters.get(
                "exit_potential_decay", ReforceXY.DEFAULT_EXIT_POTENTIAL_DECAY
            )
        )
        # === ENTRY ADDITIVE (non-PBRS additive term) ===
        self._entry_additive_enabled: bool = bool(
            model_reward_parameters.get(
                "entry_additive_enabled", ReforceXY.DEFAULT_ENTRY_ADDITIVE_ENABLED
            )
        )
        self._entry_additive_ratio: float = float(
            model_reward_parameters.get(
                "entry_additive_ratio", ReforceXY.DEFAULT_ENTRY_ADDITIVE_RATIO
            )
        )
        self._entry_additive_gain: float = float(
            model_reward_parameters.get(
                "entry_additive_gain", ReforceXY.DEFAULT_ENTRY_ADDITIVE_GAIN
            )
        )
        self._entry_additive_transform_pnl: TransformFunction = cast(
            "TransformFunction",
            model_reward_parameters.get(
                "entry_additive_transform_pnl", ReforceXY._TRANSFORM_FUNCTIONS[0]
            ),  # "tanh"
        )
        self._entry_additive_transform_duration: TransformFunction = cast(
            "TransformFunction",
            model_reward_parameters.get(
                "entry_additive_transform_duration", ReforceXY._TRANSFORM_FUNCTIONS[0]
            ),  # "tanh"
        )
        # === HOLD POTENTIAL (PBRS function Φ) ===
        self._hold_potential_enabled: bool = bool(
            model_reward_parameters.get(
                "hold_potential_enabled", ReforceXY.DEFAULT_HOLD_POTENTIAL_ENABLED
            )
        )
        self._hold_potential_ratio: float = float(
            model_reward_parameters.get(
                "hold_potential_ratio", ReforceXY.DEFAULT_HOLD_POTENTIAL_RATIO
            )
        )
        self._hold_potential_gain: float = float(
            model_reward_parameters.get(
                "hold_potential_gain", ReforceXY.DEFAULT_HOLD_POTENTIAL_GAIN
            )
        )
        self._hold_potential_transform_pnl: TransformFunction = cast(
            "TransformFunction",
            model_reward_parameters.get(
                "hold_potential_transform_pnl", ReforceXY._TRANSFORM_FUNCTIONS[0]
            ),  # "tanh"
        )
        self._hold_potential_transform_duration: TransformFunction = cast(
            "TransformFunction",
            model_reward_parameters.get(
                "hold_potential_transform_duration", ReforceXY._TRANSFORM_FUNCTIONS[0]
            ),  # "tanh"
        )
        # === EXIT ADDITIVE (non-PBRS additive term) ===
        self._exit_additive_enabled: bool = bool(
            model_reward_parameters.get(
                "exit_additive_enabled", ReforceXY.DEFAULT_EXIT_ADDITIVE_ENABLED
            )
        )
        self._exit_additive_ratio: float = float(
            model_reward_parameters.get(
                "exit_additive_ratio", ReforceXY.DEFAULT_EXIT_ADDITIVE_RATIO
            )
        )
        self._exit_additive_gain: float = float(
            model_reward_parameters.get("exit_additive_gain", ReforceXY.DEFAULT_EXIT_ADDITIVE_GAIN)
        )
        self._exit_additive_transform_pnl: TransformFunction = cast(
            "TransformFunction",
            model_reward_parameters.get(
                "exit_additive_transform_pnl", ReforceXY._TRANSFORM_FUNCTIONS[0]
            ),  # "tanh"
        )
        self._exit_additive_transform_duration: TransformFunction = cast(
            "TransformFunction",
            model_reward_parameters.get(
                "exit_additive_transform_duration", ReforceXY._TRANSFORM_FUNCTIONS[0]
            ),  # "tanh"
        )
        # === PBRS INVARIANCE CHECKS ===
        # "canonical"
        if self._exit_potential_mode == ReforceXY._EXIT_POTENTIAL_MODES[0]:
            if self._entry_additive_enabled or self._exit_additive_enabled:
                logger.warning(
                    "PBRS [%s]: canonical mode, additive disabled (use exit_potential_mode=%s to enable)",
                    self.id,
                    ReforceXY._EXIT_POTENTIAL_MODES[1],
                )
                self._entry_additive_enabled = False
                self._exit_additive_enabled = False
        # "non_canonical"
        elif self._exit_potential_mode == ReforceXY._EXIT_POTENTIAL_MODES[1] and (
            self._entry_additive_enabled or self._exit_additive_enabled
        ):
            logger.warning("PBRS [%s]: non-canonical mode, additive enabled", self.id)

        if MyRLEnv.is_unsupported_pbrs_config(
            self._hold_potential_enabled, getattr(self, "add_state_info", False)
        ):
            raise ValueError(
                f"PBRS [{self.id}]: hold_potential_enabled=True requires add_state_info=True "
                "before environment construction"
            )

        # === PNL TARGET ===
        self._pnl_target = float(self.profit_aim * self.rr)
        if self._pnl_target <= 0:
            logger.warning(
                "PBRS [%s]: pnl_target=%.6f must be > 0 (profit_aim=%.4f, rr=%.4f); "
                "defaulting to 0.01",
                self.id,
                self._pnl_target,
                self.profit_aim,
                self.rr,
            )
            self._pnl_target = 0.01

    @staticmethod
    def _loss_duration_multiplier(pnl_ratio: float, risk_reward_ratio: float) -> float:
        if not np.isfinite(pnl_ratio) or pnl_ratio >= 0.0:
            return 1.0
        if not np.isfinite(risk_reward_ratio) or risk_reward_ratio <= 0.0:
            return 1.0
        return float(risk_reward_ratio)

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
        transform_pnl: TransformFunction,
        transform_duration: TransformFunction,
        risk_reward_ratio: float | None = None,
    ) -> float:
        """Generic bounded bi-component signal combining PnL and duration."""
        if not enabled:
            return 0.0
        if require_position and position not in (Positions.Long, Positions.Short):
            return 0.0

        duration_ratio = max(0.0, duration_ratio)

        try:
            pnl_ratio = pnl / pnl_target
        except Exception:
            return 0.0

        duration_multiplier = 1.0
        if risk_reward_ratio is not None:
            duration_multiplier = MyRLEnv._loss_duration_multiplier(
                pnl_ratio,
                risk_reward_ratio,
            )
        if not np.isfinite(duration_multiplier) or duration_multiplier < 0.0:
            duration_multiplier = 1.0

        pnl_term = self._potential_transform(transform_pnl, gain * pnl_ratio)
        dur_term = self._potential_transform(transform_duration, gain * duration_ratio)
        pnl_sign = np.sign(pnl_ratio) if not np.isclose(pnl_ratio, 0.0) else 0.0
        value = scale * 0.5 * (pnl_term + pnl_sign * duration_multiplier * dur_term)
        return float(value) if np.isfinite(value) else 0.0

    def _compute_hold_potential(
        self,
        position: Positions,
        pnl: float,
        pnl_target: float,
        duration_ratio: float,
        scale: float,
    ) -> float:
        """Compute PBRS potential Φ(s) for position holding states.

        See ``_compute_pbrs_components`` for PBRS documentation.
        """
        return self._compute_pnl_duration_signal(
            enabled=self._hold_potential_enabled,
            require_position=True,
            position=position,
            pnl=pnl,
            pnl_target=pnl_target,
            duration_ratio=duration_ratio,
            scale=scale,
            gain=self._hold_potential_gain,
            transform_pnl=self._hold_potential_transform_pnl,
            transform_duration=self._hold_potential_transform_duration,
            risk_reward_ratio=self.rr,
        )

    def _compute_exit_additive(
        self,
        pnl: float,
        pnl_target: float,
        duration_ratio: float,
        scale: float,
    ) -> float:
        """Compute exit additive reward for position exit transitions.

        See ``_compute_pbrs_components`` for PBRS documentation.
        """
        return self._compute_pnl_duration_signal(
            enabled=self._exit_additive_enabled,
            require_position=False,
            position=Positions.Neutral,
            pnl=pnl,
            pnl_target=pnl_target,
            duration_ratio=duration_ratio,
            scale=scale,
            gain=self._exit_additive_gain,
            transform_pnl=self._exit_additive_transform_pnl,
            transform_duration=self._exit_additive_transform_duration,
        )

    def _compute_entry_additive(
        self,
        pnl: float,
        pnl_target: float,
        duration_ratio: float,
        scale: float,
    ) -> float:
        """Compute entry additive reward for position entry transitions.

        See ``_compute_pbrs_components`` for PBRS documentation.
        """
        return self._compute_pnl_duration_signal(
            enabled=self._entry_additive_enabled,
            require_position=False,
            position=Positions.Neutral,
            pnl=pnl,
            pnl_target=pnl_target,
            duration_ratio=duration_ratio,
            scale=scale,
            gain=self._entry_additive_gain,
            transform_pnl=self._entry_additive_transform_pnl,
            transform_duration=self._entry_additive_transform_duration,
        )

    def _potential_transform(self, name: TransformFunction, x: float) -> float:
        """Apply bounded transform function for potential and additive computations.

        Provides numerical stability by mapping unbounded inputs to bounded outputs
        using various smooth activation functions. Used in both PBRS potentials
        and additive reward calculations.

        Parameters
        ----------
        name : str
            Transform function name: one of ReforceXY._TRANSFORM_FUNCTIONS
        x : float
            Input value to transform

        Returns
        -------
        float
            Bounded output in [-1, 1]
        """
        if name == ReforceXY._TRANSFORM_FUNCTIONS[0]:  # "tanh"
            return math.tanh(x)

        if name == ReforceXY._TRANSFORM_FUNCTIONS[1]:  # "softsign"
            ax = abs(x)
            return x / (1.0 + ax)

        if name == ReforceXY._TRANSFORM_FUNCTIONS[2]:  # "arctan"
            return (2.0 / math.pi) * math.atan(x)

        if name == ReforceXY._TRANSFORM_FUNCTIONS[3]:  # "sigmoid"
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

        if name == ReforceXY._TRANSFORM_FUNCTIONS[4]:  # "softsign_sqrt"
            return x / math.hypot(1.0, x)

        if name == ReforceXY._TRANSFORM_FUNCTIONS[5]:  # "clip"
            return min(max(-1.0, x), 1.0)

        logger.warning(
            "PBRS [%s]: potential_transform=%r invalid; defaulting to %r. Valid: %s",
            self.id,
            name,
            ReforceXY._TRANSFORM_FUNCTIONS[0],
            ", ".join(ReforceXY._TRANSFORM_FUNCTIONS),
        )
        return math.tanh(x)

    def _compute_exit_potential(self, prev_potential: float, gamma: float) -> float:
        """Compute next potential Φ(s') for exit transitions based on exit potential mode.

        See ``_compute_pbrs_components`` for PBRS documentation.
        """
        mode = self._exit_potential_mode
        # "canonical" or "non_canonical"
        if mode == ReforceXY._EXIT_POTENTIAL_MODES[0] or mode == ReforceXY._EXIT_POTENTIAL_MODES[1]:
            return 0.0
        # "progressive_release"
        if mode == ReforceXY._EXIT_POTENTIAL_MODES[2]:
            decay = self._exit_potential_decay
            if not np.isfinite(decay) or decay < 0.0:
                decay = 0.0
            if decay > 1.0:
                decay = 1.0
            next_potential = prev_potential * (1.0 - decay)
        # "spike_cancel"
        elif mode == ReforceXY._EXIT_POTENTIAL_MODES[3]:
            if gamma <= 0.0 or not np.isfinite(gamma):
                next_potential = prev_potential
            else:
                next_potential = prev_potential / gamma
        # "retain_previous"
        elif mode == ReforceXY._EXIT_POTENTIAL_MODES[4]:
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
        return self._exit_potential_mode == ReforceXY._EXIT_POTENTIAL_MODES[0] and not (
            self._entry_additive_enabled or self._exit_additive_enabled
        )  # "canonical"

    @staticmethod
    def is_unsupported_pbrs_config(hold_potential_enabled: bool, add_state_info: bool) -> bool:
        """Return True if PBRS potential relies on hidden state.

        Case: hold_potential enabled while auxiliary state info (pnl, trade_duration) is excluded
        from the observation space (add_state_info=False). In that situation, Φ(s) uses hidden
        variables and PBRS becomes informative, voiding the strict policy invariance guarantee.
        """
        return hold_potential_enabled and not add_state_info

    def _compute_pbrs_components(
        self,
        *,
        previous_position: Positions,
        next_position: Positions,
        next_trade_duration: float,
        next_pnl: float,
        entry_pnl: float,
        trade_duration: float,
        max_trade_duration: float,
        current_pnl: float,
        pnl_target: float,
        hold_potential_scale: float,
        entry_additive_scale: float,
        exit_additive_scale: float,
    ) -> tuple[float, float, float]:
        """Compute PBRS and optional additive components for one transition.

        ``previous_position`` and ``next_position`` identify entry, hold, exit,
        or neutral self-loop transitions. ``next_pnl`` and
        ``next_trade_duration`` describe the returned observation s', while
        ``current_pnl`` and ``trade_duration`` describe s at the fill.
        ``entry_pnl`` is the fee-aware PnL provisioned on an entry fill.

        The shaping component is Δ = γ·Φ(s') - Φ(s). In canonical mode,
        ``step()`` enforces a zero terminal potential. Therefore an observed
        trajectory satisfies Σ γ^t·Δ_t = -Φ(s_0) + γ^T·Φ(s_T), which is zero
        only when both boundary potentials are zero. Entry and exit additives
        are returned separately and are not PBRS terms.

        Returns
        -------
        tuple[float, float, float]
            ``(reward_shaping, entry_additive, exit_additive)``.
        """

        prev_potential = float(self._last_potential)
        if not self._hold_potential_enabled and not (
            self._entry_additive_enabled or self._exit_additive_enabled
        ):
            logger.debug("PBRS [%s]: all PBRS features disabled, returning zeros", self.id)
            self._last_prev_potential = float(prev_potential)
            self._last_next_potential = float(prev_potential)
            self._last_entry_additive = 0.0
            self._last_exit_additive = 0.0
            self._last_reward_shaping = 0.0
            return 0.0, 0.0, 0.0

        if max_trade_duration <= 0:
            next_duration_ratio = 0.0
        else:
            next_duration_ratio = next_trade_duration / max_trade_duration

        is_entry = previous_position == Positions.Neutral and next_position in (
            Positions.Long,
            Positions.Short,
        )
        is_exit = (
            previous_position in (Positions.Long, Positions.Short)
            and next_position == Positions.Neutral
        )
        is_hold = previous_position in (
            Positions.Long,
            Positions.Short,
        ) and next_position in (Positions.Long, Positions.Short)

        gamma = self._potential_gamma

        reward_shaping = 0.0
        entry_additive = 0.0
        exit_additive = 0.0
        next_potential = prev_potential

        if is_entry or is_hold:
            if self._hold_potential_enabled:
                next_potential = self._compute_hold_potential(
                    next_position,
                    next_pnl,
                    pnl_target,
                    next_duration_ratio,
                    hold_potential_scale,
                )
                reward_shaping = gamma * next_potential - prev_potential
            else:
                next_potential = 0.0
                reward_shaping = 0.0

            if is_entry and self._entry_additive_enabled and not self.is_pbrs_invariant_mode():
                entry_additive = self._compute_entry_additive(
                    entry_pnl,
                    pnl_target,
                    0.0,
                    entry_additive_scale,
                )
                self._total_entry_additive += float(entry_additive)

        elif is_exit:
            if (
                self._exit_potential_mode == ReforceXY._EXIT_POTENTIAL_MODES[0]  # "canonical"
            ) or (
                self._exit_potential_mode == ReforceXY._EXIT_POTENTIAL_MODES[1]  # "non_canonical"
            ):
                next_potential = 0.0
                reward_shaping = -prev_potential
            else:
                next_potential = self._compute_exit_potential(prev_potential, gamma)
                reward_shaping = gamma * next_potential - prev_potential

            if self._exit_additive_enabled and not self.is_pbrs_invariant_mode():
                duration_ratio = trade_duration / max(1, max_trade_duration)
                exit_additive = self._compute_exit_additive(
                    current_pnl, pnl_target, duration_ratio, exit_additive_scale
                )
                self._total_exit_additive += float(exit_additive)

        else:
            # Neutral self-loop
            next_potential = prev_potential
            reward_shaping = 0.0

        self._last_potential = float(next_potential)
        self._last_prev_potential = float(prev_potential)
        self._last_next_potential = float(self._last_potential)
        self._last_entry_additive = float(entry_additive)
        self._last_exit_additive = float(exit_additive)
        self._last_reward_shaping = float(reward_shaping)
        self._total_reward_shaping += float(reward_shaping)

        return float(reward_shaping), float(entry_additive), float(exit_additive)

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
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    def _is_valid(self, action: int) -> bool:
        return ReforceXY.get_action_masks(self.can_short, self._position)[action]

    def reset_env(
        self,
        df: DataFrame,
        prices: DataFrame,
        window_size: int,
        reward_kwargs: dict[str, Any],
        starting_point=True,
    ) -> None:
        """
        Resets the environment when the agent fails
        """
        super().reset_env(df, prices, window_size, reward_kwargs, starting_point)
        self._set_observation_space()

    def reset(self, seed=None, **kwargs) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """
        Reset is called at the beginning of every episode
        """
        observation, history = super().reset(seed, **kwargs)
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

    def _compute_time_attenuation_coefficient(
        self,
        duration_ratio: float,
        model_reward_parameters: Mapping[str, Any],
    ) -> float:
        """
        Calculate time-based attenuation coefficient using configurable strategy
        (sqrt/linear/power/half_life). Optionally apply plateau grace period.
        """
        if duration_ratio < 0.0:
            duration_ratio = 0.0

        exit_attenuation_mode = str(
            model_reward_parameters.get(
                "exit_attenuation_mode", ReforceXY._EXIT_ATTENUATION_MODES[1]
            )  # "linear"
        )
        exit_plateau = bool(
            model_reward_parameters.get("exit_plateau", ReforceXY.DEFAULT_EXIT_PLATEAU)
        )
        exit_plateau_grace = float(
            model_reward_parameters.get("exit_plateau_grace", ReforceXY.DEFAULT_EXIT_PLATEAU_GRACE)
        )
        if exit_plateau_grace < 0.0:
            logger.warning(
                "PBRS [%s]: exit_plateau_grace=%.2f invalid; defaulting to 0.0",
                self.id,
                exit_plateau_grace,
            )
            exit_plateau_grace = 0.0

        def _sqrt(dr: float, p: Mapping[str, Any]) -> float:
            return 1.0 / math.sqrt(1.0 + dr)

        def _linear(dr: float, p: Mapping[str, Any]) -> float:
            slope = float(p.get("exit_linear_slope", ReforceXY.DEFAULT_EXIT_LINEAR_SLOPE))
            if slope < 0.0:
                logger.warning(
                    "PBRS [%s]: exit_linear_slope=%.2f invalid; defaulting to 1.0",
                    self.id,
                    slope,
                )
                slope = 1.0
            return 1.0 / (1.0 + slope * dr)

        def _power(dr: float, p: Mapping[str, Any]) -> float:
            tau = p.get("exit_power_tau")
            if isinstance(tau, (int, float)):
                tau = float(tau)
                alpha = -math.log(tau) / ReforceXY._LOG_2 if 0.0 < tau <= 1.0 else 1.0
            else:
                alpha = 1.0
            return 1.0 / math.pow(1.0 + dr, alpha)

        def _half_life(dr: float, p: Mapping[str, Any]) -> float:
            hl = float(p.get("exit_half_life", ReforceXY.DEFAULT_EXIT_HALF_LIFE))
            if np.isclose(hl, 0.0) or hl < 0.0:
                return 1.0
            return math.pow(2.0, -dr / hl)

        strategies: dict[str, Callable[[float, Mapping[str, Any]], float]] = {
            ReforceXY._EXIT_ATTENUATION_MODES[0]: _sqrt,
            ReforceXY._EXIT_ATTENUATION_MODES[1]: _linear,
            ReforceXY._EXIT_ATTENUATION_MODES[2]: _power,
            ReforceXY._EXIT_ATTENUATION_MODES[3]: _half_life,
        }

        if exit_plateau:
            if duration_ratio <= exit_plateau_grace:
                effective_dr = 0.0
            else:
                effective_dr = duration_ratio - exit_plateau_grace
        else:
            effective_dr = duration_ratio

        strategy_fn = strategies.get(exit_attenuation_mode)
        if strategy_fn is None:
            logger.warning(
                "PBRS [%s]: exit_attenuation_mode=%r invalid; defaulting to %r. Valid: %s",
                self.id,
                exit_attenuation_mode,
                ReforceXY._EXIT_ATTENUATION_MODES[1],  # "linear"
                ", ".join(ReforceXY._EXIT_ATTENUATION_MODES),
            )
            strategy_fn = _linear

        try:
            time_attenuation_coefficient = strategy_fn(effective_dr, model_reward_parameters)
        except Exception as e:
            logger.warning(
                "PBRS [%s]: exit_attenuation_mode=%r failed (%r); defaulting to %r (effective_dr=%.5f)",
                self.id,
                exit_attenuation_mode,
                e,
                ReforceXY._EXIT_ATTENUATION_MODES[1],  # "linear"
                effective_dr,
                exc_info=True,
            )
            time_attenuation_coefficient = _linear(effective_dr, model_reward_parameters)

        return time_attenuation_coefficient

    def _get_exit_factor(
        self,
        base_factor: float,
        pnl: float,
        duration_ratio: float,
        model_reward_parameters: Mapping[str, Any],
    ) -> float:
        """
        Compute exit factor: base_factor · time_attenuation_coefficient · pnl_target_coefficient · efficiency_coefficient.
        """
        if not (np.isfinite(base_factor) and np.isfinite(pnl) and np.isfinite(duration_ratio)):
            return 0.0

        time_attenuation_coefficient = self._compute_time_attenuation_coefficient(
            duration_ratio,
            model_reward_parameters,
        )
        pnl_target_coefficient = self._compute_pnl_target_coefficient(
            pnl, self._pnl_target, model_reward_parameters
        )
        efficiency_coefficient = self._compute_efficiency_coefficient(pnl, model_reward_parameters)

        exit_factor = (
            base_factor
            * time_attenuation_coefficient
            * pnl_target_coefficient
            * efficiency_coefficient
        )

        check_invariants = model_reward_parameters.get(
            "check_invariants", ReforceXY.DEFAULT_CHECK_INVARIANTS
        )
        check_invariants = check_invariants if isinstance(check_invariants, bool) else True
        if check_invariants:
            if not np.isfinite(exit_factor):
                logger.warning(
                    "PBRS [%s]: exit_factor=%.5f non-finite; defaulting to 0.0",
                    self.id,
                    exit_factor,
                )
                return 0.0
            if efficiency_coefficient < 0.0:
                logger.warning(
                    "PBRS [%s]: efficiency_coefficient=%.5f negative",
                    self.id,
                    efficiency_coefficient,
                )
            if exit_factor < 0.0 and pnl >= 0.0:
                logger.warning(
                    "PBRS [%s]: exit_factor=%.5f negative with pnl=%.5f positive; defaulting to 0.0",
                    self.id,
                    exit_factor,
                    pnl,
                )
                exit_factor = 0.0
            exit_factor_threshold = float(
                model_reward_parameters.get(
                    "exit_factor_threshold", ReforceXY.DEFAULT_EXIT_FACTOR_THRESHOLD
                )
            )
            if exit_factor_threshold > 0 and abs(exit_factor) > exit_factor_threshold:
                logger.warning(
                    "PBRS [%s]: |exit_factor|=%.5f exceeds exit_factor_threshold=%.5f",
                    self.id,
                    abs(exit_factor),
                    exit_factor_threshold,
                )

        return exit_factor

    def _compute_pnl_target_coefficient(
        self, pnl: float, pnl_target: float, model_reward_parameters: Mapping[str, Any]
    ) -> float:
        """
        Compute PnL target coefficient (typically 0.5-2.0) using tanh on PnL/target ratio.
        """
        pnl_target_coefficient = 1.0

        if pnl_target > 0.0:
            pnl_amplification_sensitivity = float(
                model_reward_parameters.get(
                    "pnl_amplification_sensitivity",
                    ReforceXY.DEFAULT_PNL_AMPLIFICATION_SENSITIVITY,
                )
            )
            pnl_ratio = pnl / pnl_target
            win_reward_factor = float(
                model_reward_parameters.get(
                    "win_reward_factor", ReforceXY.DEFAULT_WIN_REWARD_FACTOR
                )
            )

            if pnl_ratio > 1.0:
                gain_coefficient = math.tanh(pnl_amplification_sensitivity * (pnl_ratio - 1.0))
                pnl_target_coefficient = 1.0 + win_reward_factor * gain_coefficient
            else:
                loss_threshold = pnl_target / self.rr
                if pnl < -loss_threshold:
                    loss_ratio = (-pnl) / loss_threshold
                    loss_coefficient = math.tanh(pnl_amplification_sensitivity * (loss_ratio - 1.0))
                    loss_penalty_factor = win_reward_factor * self.rr
                    pnl_target_coefficient = 1.0 + loss_penalty_factor * loss_coefficient

        return pnl_target_coefficient

    def _compute_efficiency_coefficient(
        self, pnl: float, model_reward_parameters: Mapping[str, Any]
    ) -> float:
        """
        Compute exit efficiency coefficient (typically 0.5-1.5) based on exit timing quality.

        The coefficient is clamped to be nonnegative so an aggressive weight or
        center can never flip the sign of a losing exit's penalty.
        """
        efficiency_weight = float(
            model_reward_parameters.get("efficiency_weight", ReforceXY.DEFAULT_EFFICIENCY_WEIGHT)
        )
        efficiency_center = float(
            model_reward_parameters.get("efficiency_center", ReforceXY.DEFAULT_EFFICIENCY_CENTER)
        )

        efficiency_coefficient = 1.0
        if efficiency_weight != 0.0 and not np.isclose(pnl, 0.0):
            max_pnl = max(self.get_max_unrealized_profit(), pnl)
            min_pnl = min(self.get_min_unrealized_profit(), pnl)
            range_pnl = max_pnl - min_pnl
            # Guard against division explosion when max_pnl ≈ min_pnl
            min_meaningful_range = max(
                ReforceXY.DEFAULT_EFFICIENCY_MIN_RANGE_EPSILON,
                ReforceXY.DEFAULT_EFFICIENCY_MIN_RANGE_FRACTION * self._pnl_target,
            )
            if np.isfinite(range_pnl) and range_pnl >= min_meaningful_range:
                efficiency_ratio = (pnl - min_pnl) / range_pnl
                if pnl > 0.0:
                    efficiency_coefficient = 1.0 + efficiency_weight * (
                        efficiency_ratio - efficiency_center
                    )
                elif pnl < 0.0:
                    efficiency_coefficient = 1.0 + efficiency_weight * (
                        efficiency_center - efficiency_ratio
                    )
        if efficiency_coefficient < 0.0:
            logger.warning(
                "PBRS [%s]: efficiency_coefficient=%.5f < 0; clamping to 0",
                self.id,
                efficiency_coefficient,
            )
            efficiency_coefficient = 0.0

        return efficiency_coefficient

    def calculate_reward(self, action: int) -> float:
        """Compute base reward at the current action's execution price.

        Shaping is applied by step after execution and advancement, when the
        actual next observation's position, duration and PnL are available.
        """
        model_reward_parameters = self.rl_config.get("model_reward_parameters", {})
        base_reward: float | None = None

        self._last_invalid_penalty = 0.0
        self._last_idle_penalty = 0.0
        self._last_hold_penalty = 0.0
        self._last_exit_reward = 0.0

        # 1. Invalid action
        if not self.action_masking and not self._is_valid(action):
            self.tensorboard_log("invalid", category="actions")
            base_reward = float(
                model_reward_parameters.get("invalid_action", ReforceXY.DEFAULT_INVALID_ACTION)
            )
            self._last_invalid_penalty = float(base_reward)

        max_trade_duration = max(1, self.max_trade_duration_candles)
        trade_duration = self.get_trade_duration()
        duration_ratio = trade_duration / max_trade_duration
        base_factor = float(
            model_reward_parameters.get("base_factor", ReforceXY.DEFAULT_BASE_FACTOR)
        )
        idle_factor = base_factor * (self.profit_aim / self.rr)
        hold_factor = idle_factor

        # 2. Idle penalty
        if (
            base_reward is None
            and action == Actions.Neutral.value
            and self._position == Positions.Neutral
        ):
            max_idle_duration = max(1, self.max_idle_duration_candles)
            idle_penalty_ratio = float(
                model_reward_parameters.get(
                    "idle_penalty_ratio", ReforceXY.DEFAULT_IDLE_PENALTY_RATIO
                )
            )
            idle_penalty_power = float(
                model_reward_parameters.get(
                    "idle_penalty_power", ReforceXY.DEFAULT_IDLE_PENALTY_POWER
                )
            )
            idle_duration = self.get_idle_duration()
            idle_duration_ratio = idle_duration / max(1, max_idle_duration)
            base_reward = (
                -idle_factor * idle_penalty_ratio * idle_duration_ratio**idle_penalty_power
            )
            self._last_idle_penalty = float(base_reward)

        # 3. Hold overtime penalty
        if (
            base_reward is None
            and self._position in (Positions.Short, Positions.Long)
            and action == Actions.Neutral.value
        ):
            hold_penalty_ratio = float(
                model_reward_parameters.get(
                    "hold_penalty_ratio", ReforceXY.DEFAULT_HOLD_PENALTY_RATIO
                )
            )
            hold_penalty_power = float(
                model_reward_parameters.get(
                    "hold_penalty_power", ReforceXY.DEFAULT_HOLD_PENALTY_POWER
                )
            )
            if duration_ratio < 1.0:
                base_reward = 0.0
            else:
                base_reward = (
                    -hold_factor * hold_penalty_ratio * (duration_ratio - 1.0) ** hold_penalty_power
                )
                self._last_hold_penalty = float(base_reward)

        # 4. Exit rewards
        pnl: float = self.get_unrealized_profit()
        if (
            base_reward is None
            and action == Actions.Long_exit.value
            and self._position == Positions.Long
        ):
            base_reward = pnl * self._get_exit_factor(
                base_factor, pnl, duration_ratio, model_reward_parameters
            )
            self._last_exit_reward = float(base_reward)
        if (
            base_reward is None
            and action == Actions.Short_exit.value
            and self._position == Positions.Short
        ):
            base_reward = pnl * self._get_exit_factor(
                base_factor, pnl, duration_ratio, model_reward_parameters
            )
            self._last_exit_reward = float(base_reward)

        # 5. Default
        if base_reward is None:
            base_reward = 0.0

        return base_reward

    def _get_observation(self) -> NDArray[np.float32]:
        start_idx = max(0, self._current_tick - self.window_size)
        end_idx = min(self._current_tick, len(self.signal_features))
        features_window = self.signal_features.iloc[start_idx:end_idx]
        features_window_array = features_window.to_numpy(dtype=np.float32, copy=False)
        if features_window_array.shape[0] < self.window_size:
            pad_size = self.window_size - features_window_array.shape[0]
            pad_array = np.zeros((pad_size, features_window_array.shape[1]), dtype=np.float32)
            features_window_array = np.concatenate([pad_array, features_window_array], axis=0)
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
        self._position = Positions.Neutral
        self._last_trade_tick = None
        self._last_closed_trade_tick = self._current_tick
        self._max_unrealized_profit = -np.inf
        self._min_unrealized_profit = np.inf

    def execute_trade(self, action: int) -> str | None:
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
            closed_position = self._position
            self._exit_trade()
            return f"{closed_position.name}_exit"

        return None

    def _apply_terminal_pbrs_correction(self, reward: float) -> float:
        if not (
            self._hold_potential_enabled
            or self._entry_additive_enabled
            or self._exit_additive_enabled
        ):
            self._last_potential = 0.0
            self._last_next_potential = 0.0
            self._last_reward_shaping = 0.0
            return reward

        prev_potential = self._last_prev_potential
        computed_reward_shaping = self._last_reward_shaping
        terminal_reward_shaping = -prev_potential
        reward_shaping_delta = terminal_reward_shaping - computed_reward_shaping
        last_next_potential = self._last_next_potential

        if np.isclose(computed_reward_shaping, terminal_reward_shaping) and np.isclose(
            last_next_potential, 0.0
        ):
            self._last_potential = 0.0
            self._last_next_potential = 0.0
            return reward

        self._last_potential = 0.0
        self._last_next_potential = 0.0
        self._last_reward_shaping = terminal_reward_shaping
        self._total_reward_shaping += reward_shaping_delta
        return reward + reward_shaping_delta

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment based on the provided action
        """
        previous_equity = self._get_portfolio_equity(self.get_unrealized_profit())
        previous_position = self._position
        pre_trade_duration = self.get_trade_duration()
        pre_pnl = self.get_unrealized_profit()
        execution_tick = self._current_tick
        exit_pnl = None
        reward = self.calculate_reward(action)
        trade_type = self.execute_trade(action)
        entry_pnl = self.get_unrealized_profit() if previous_position == Positions.Neutral else 0.0
        if trade_type is not None:
            self.append_trade_history(
                trade_type, self.current_price(), pre_pnl, execution_tick=execution_tick
            )
            if self._position == Positions.Neutral:
                exit_pnl = pre_pnl
        elif action != Actions.Neutral.value:
            logger.warning(
                "Env [%s]: invalid action=%s (%d) in position=%s at tick=%d",
                self.id,
                Actions(action).name,
                action,
                self._position.name,
                self._current_tick,
            )
        self._current_tick += 1
        self._update_unrealized_total_profit()
        base_factor = float(
            self.rl_config.get("model_reward_parameters", {}).get(
                "base_factor", ReforceXY.DEFAULT_BASE_FACTOR
            )
        )
        shaping, entry_additive, exit_additive = self._compute_pbrs_components(
            previous_position=previous_position,
            next_position=self._position,
            next_trade_duration=self.get_trade_duration(),
            next_pnl=self.get_unrealized_profit(),
            entry_pnl=entry_pnl,
            trade_duration=pre_trade_duration,
            max_trade_duration=max(1, self.max_trade_duration_candles),
            current_pnl=pre_pnl,
            pnl_target=self._pnl_target,
            hold_potential_scale=self._hold_potential_ratio * base_factor,
            entry_additive_scale=self._entry_additive_ratio * base_factor,
            exit_additive_scale=self._exit_additive_ratio * base_factor,
        )
        reward += shaping + entry_additive + exit_additive
        terminated = self.is_terminated()
        terminal_liquidation = terminated and self._position in (Positions.Long, Positions.Short)
        if terminal_liquidation:
            terminal_pnl = self.get_unrealized_profit()
            terminal_duration_ratio = self.get_trade_duration() / max(
                1, self.max_trade_duration_candles
            )
            self._update_max_unrealized_profit(terminal_pnl)
            self._update_min_unrealized_profit(terminal_pnl)
            liquidation_reward = terminal_pnl * self._get_exit_factor(
                base_factor,
                terminal_pnl,
                terminal_duration_ratio,
                self.rl_config.get("model_reward_parameters", {}),
            )
            reward += liquidation_reward
            self._last_exit_reward += liquidation_reward
            if self._exit_additive_enabled and not self.is_pbrs_invariant_mode():
                liquidation_additive = self._compute_exit_additive(
                    terminal_pnl,
                    self._pnl_target,
                    terminal_duration_ratio,
                    self._exit_additive_ratio * base_factor,
                )
                reward += liquidation_additive
                self._last_exit_additive += liquidation_additive
                self._total_exit_additive += liquidation_additive
            exit_pnl = terminal_pnl
            closed_position = self._position
            self._exit_trade()
            self.append_trade_history(
                f"{closed_position.name}_exit",
                self.current_price(),
                terminal_pnl,
                execution_tick=execution_tick,
            )
        if terminated:
            reward = self._apply_terminal_pbrs_correction(reward)
            self._last_potential = 0.0
        self._position_history.append(self._position)
        self.total_reward += reward
        pnl = self.get_unrealized_profit()
        self._update_portfolio_log_returns(previous_equity, self._get_portfolio_equity(pnl))
        self._update_max_unrealized_profit(pnl)
        self._update_min_unrealized_profit(pnl)
        delta_pnl = pnl - pre_pnl
        max_idle_duration = max(1, self.max_idle_duration_candles)
        idle_duration = self.get_idle_duration()
        trade_duration = self.get_trade_duration()
        info = {
            "tick": self._current_tick,
            "execution_tick": execution_tick,
            "terminal_liquidation": bool(terminal_liquidation),
            "exit_pnl": exit_pnl,
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
            "duration_ratio": (trade_duration / max(1, self.max_trade_duration_candles)),
            "trade_count": len(self.trade_history) // 2,
        }
        self._update_history(info)
        return (
            self._get_observation(),
            reward,
            terminated,
            self.is_truncated(),
            info,
        )

    def append_trade_history(
        self, trade_type: str, price: float, profit: float, *, execution_tick: int
    ) -> None:
        self.trade_history.append(
            {
                "tick": execution_tick,
                "type": trade_type.lower(),
                "price": price,
                "profit": profit,
            }
        )

    def is_terminated(self) -> bool:
        return bool(
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
        if (
            self._position in (Positions.Long, Positions.Short)
            and pnl > self._max_unrealized_profit
        ):
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
        if (
            self._position in (Positions.Long, Positions.Short)
            and pnl < self._min_unrealized_profit
        ):
            self._min_unrealized_profit = pnl

    def _get_portfolio_equity(self, pnl: float) -> float:
        """Mark equity to liquidation using the upstream fee and staking conventions."""
        if self.compound_trades:
            return self._total_profit * (1.0 + pnl)
        return self._total_profit + pnl

    def get_most_recent_return(self) -> float:
        """Return the stored log equity change for the last completed transition."""
        return float(self.portfolio_log_returns[self._current_tick])

    def _update_portfolio_log_returns(self, previous_equity: float, current_equity: float) -> None:
        """Record log(E_after / E_before), provisioning round-trip fees at entry.

        Equity includes unrealized liquidation PnL while a position is open and
        realized capital after exit. This accounts for the final price move
        without charging liquidation fees twice. Non-positive or non-finite
        equity has no finite log return and is reported as NaN, not zero.
        """
        if (
            not np.isfinite(previous_equity)
            or not np.isfinite(current_equity)
            or previous_equity <= 0.0
            or current_equity <= 0.0
        ):
            value = np.nan
            logger.warning(
                "Env [%s]: undefined portfolio log return at tick=%d: equity %s -> %s",
                self.id,
                self._current_tick,
                previous_equity,
                current_equity,
            )
        else:
            value = math.log(current_equity) - math.log(previous_equity)
        self.portfolio_log_returns[self._current_tick] = value

    def get_most_recent_profit(self) -> float:
        """Return the simple equity change corresponding to the stored log return."""
        return float(np.expm1(self.get_most_recent_return()))

    def get_env_history(self) -> DataFrame:
        """Return one metrics row per transition, joined only with price data.

        Trade events remain normalized in ``trade_history``. It contains one
        row per economic event, and ordered events may share an execution tick.
        """
        if not self.history:
            logger.warning("Env [%s]: history is empty", self.id)
            return DataFrame()

        _history_df = DataFrame(self.history)
        if "tick" not in _history_df.columns:
            logger.warning("Env [%s]: 'tick' column missing from history", self.id)
            return DataFrame()

        try:
            history = merge(
                _history_df,
                self.prices,
                left_on="tick",
                right_index=True,
                how="left",
            )
        except Exception as e:
            logger.warning(
                "Env [%s]: failed to merge history with prices: %r",
                self.id,
                e,
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
            if ticks is None or len(ticks) == 0 or history_open is None or len(history_open) == 0:
                return fig

            axs[0].plot(ticks, history_open, linewidth=1, color="orchid", zorder=1)

            trades = DataFrame(self.trade_history)
            if not trades.empty:
                trades = trades[
                    trades["tick"].between(history["execution_tick"].min(), ticks.max())
                ]
            history_type = trades.get("type")
            history_price = trades.get("price")
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
                        xs = trades.loc[mask, "tick"]
                        ys = trades.loc[mask, "price"]

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
            + f"Trades: {len(self.trade_history) // 2}",
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
        self, key: str, value: Any, exclude: tuple[str, ...] | None = None
    ) -> None:
        try:
            self.logger.record(key, value, exclude=exclude)
        except Exception as e:
            logger.warning(
                "Tensorboard [global]: logger.record failed at %r: %r",
                key,
                e,
                exc_info=True,
            )
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
                logger.error(
                    "Tensorboard [global]: logger.record retry failed at %r: %r",
                    key,
                    e,
                    exc_info=True,
                )
                pass

    @staticmethod
    def _build_train_freq(
        train_freq: TrainFreq | int | tuple[int, ...] | list[int] | None,
    ) -> int | None:
        train_freq_val: int | None = None
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
                with suppress(Exception):
                    n_stack = int(env.n_stack)
                break
            env = getattr(env, "venv", None)
        hparam_dict: dict[str, Any] = {
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
        if ReforceXY._MODEL_TYPES[0] in self.model.__class__.__name__:  # "PPO"
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
            if ReforceXY._MODEL_TYPES[1] in self.model.__class__.__name__:  # "RecurrentPPO"
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
        if ReforceXY._MODEL_TYPES[3] in self.model.__class__.__name__:  # "DQN"
            hparam_dict.update(
                {
                    "buffer_size": int(self.model.buffer_size),
                    "gradient_steps": int(self.model.gradient_steps),
                    "learning_starts": int(self.model.learning_starts),
                    "target_update_interval": int(self.model.target_update_interval),
                    "exploration_initial_eps": float(self.model.exploration_initial_eps),
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
            if ReforceXY._MODEL_TYPES[4] in self.model.__class__.__name__:  # "QRDQN"
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
        if ReforceXY._MODEL_TYPES[0] in self.model.__class__.__name__:  # "PPO"
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
        if ReforceXY._MODEL_TYPES[3] in self.model.__class__.__name__:  # "DQN"
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
            return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)

        def _is_finite_number(x: Any) -> bool:
            if not _is_number(x):
                return False
            try:
                return np.isfinite(float(x))
            except Exception:
                return False

        infos_list: list[dict[str, Any]] | None = self.locals.get("infos")
        aggregated_info: dict[str, Any] = {}

        if isinstance(infos_list, list) and infos_list:
            numeric_acc: dict[str, list[float]] = defaultdict(list)
            non_numeric_counts: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
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
                    with suppress(Exception):
                        aggregated_info[f"{k}_std"] = np.std(values, ddof=1)

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
                    with suppress(Exception):
                        aggregated_info[f"{k}_mode"] = next(iter(counts.keys()))
                else:
                    aggregated_info[f"{k}_mode"] = "mixed"

            self._safe_logger_record("info/n_envs", len(infos_list), exclude=logger_exclude)

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

        aggregated_tensorboard_metrics: dict[str, dict[str, Any]] = defaultdict(dict)
        aggregated_tensorboard_metric_counts: dict[str, dict[str, int]] = defaultdict(dict)
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
            cat_counts: dict[str, dict[Any, int]] = {k: defaultdict(int) for k in cat_keys}
            cat_totals: dict[str, int] = dict.fromkeys(cat_keys, 0)
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
                    count = aggregated_tensorboard_metric_counts.get(category, {}).get(metric)
                    if _is_finite_number(value) and isinstance(count, int) and count > 0:
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
            self._safe_logger_record("train/progress_done", progress_done, exclude=logger_exclude)
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
                if schedule_type == ReforceXY._SCHEDULE_TYPES[0]:  # "linear"
                    return float(schedule(progress_remaining))
                if schedule_type == ReforceXY._SCHEDULE_TYPES[1]:  # "constant"
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
                self._safe_logger_record("train/learning_rate", float(lr), exclude=logger_exclude)
        except Exception:
            pass

        if ReforceXY._MODEL_TYPES[0] in self.model.__class__.__name__:  # "PPO"
            try:
                cr = getattr(self.model, "clip_range", None)
                cr = _eval_schedule(cr)
                if _is_finite_number(cr):
                    self._safe_logger_record("train/clip_range", float(cr), exclude=logger_exclude)
            except Exception:
                pass

        if ReforceXY._MODEL_TYPES[3] in self.model.__class__.__name__:  # "DQN"
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
                logger.warning(
                    "Tensorboard [global]: logger.record failed at best/train_env%d: %r",
                    i,
                    e,
                    exc_info=True,
                )
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
        best_model_save_path: str | None = None,
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
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
                    "Hyperopt [%s]: trial #%d invalid last_mean_reward (eval_idx=%s, timesteps=%s): %r",
                    self.trial.study.study_name,
                    self.trial.number,
                    self.eval_idx,
                    self.num_timesteps,
                    e,
                    exc_info=True,
                )
                self.is_pruned = True
                return False

            if not np.isfinite(last_mean_reward):
                logger.warning(
                    "Hyperopt [%s]: trial #%d non-finite last_mean_reward (eval_idx=%s, timesteps=%s)",
                    self.trial.study.study_name,
                    self.trial.number,
                    self.eval_idx,
                    self.num_timesteps,
                )
                self.is_pruned = True
                return False

            try:
                self.trial.report(last_mean_reward, self.num_timesteps)
            except Exception as e:
                logger.warning(
                    "Hyperopt [%s]: trial #%d trial.report failed (eval_idx=%s, timesteps=%s): %r",
                    self.trial.study.study_name,
                    self.trial.number,
                    self.eval_idx,
                    self.num_timesteps,
                    e,
                    exc_info=True,
                )
                self.is_pruned = True
                return False

            try:
                best_mean_reward = float(getattr(self, "best_mean_reward", np.nan))
            except Exception as e:
                logger.warning(
                    "Hyperopt [%s]: trial #%d invalid best_mean_reward (eval_idx=%s, timesteps=%s): %r",
                    self.trial.study.study_name,
                    self.trial.number,
                    self.eval_idx,
                    self.num_timesteps,
                    e,
                    exc_info=True,
                )
                best_mean_reward = np.nan

            try:
                logger_exclude = ("stdout", "log", "json", "csv")
                self.logger.record("eval/idx", self.eval_idx, exclude=logger_exclude)
                self.logger.record("eval/num_timesteps", self.num_timesteps, exclude=logger_exclude)
                self.logger.record(
                    "eval/last_mean_reward", last_mean_reward, exclude=logger_exclude
                )
                if np.isfinite(best_mean_reward):
                    self.logger.record(
                        "eval/best_mean_reward",
                        best_mean_reward,
                        exclude=logger_exclude,
                    )
                else:
                    logger.warning(
                        "Hyperopt [%s]: trial #%d non-finite best_mean_reward (eval_idx=%s, timesteps=%s)",
                        self.trial.study.study_name,
                        self.trial.number,
                        self.eval_idx,
                        self.num_timesteps,
                    )
            except Exception as e:
                logger.error(
                    "Hyperopt [%s]: trial #%d logger.record failed (eval_idx=%s, timesteps=%s): %r",
                    self.trial.study.study_name,
                    self.trial.number,
                    self.eval_idx,
                    self.num_timesteps,
                    e,
                    exc_info=True,
                )

            try:
                if self.trial.should_prune():
                    logger.info(
                        "Hyperopt [%s]: trial #%d pruned (eval_idx=%s, timesteps=%s, score=%.5f)",
                        self.trial.study.study_name,
                        self.trial.number,
                        self.eval_idx,
                        self.num_timesteps,
                        last_mean_reward,
                    )
                    self.is_pruned = True
                    return False
            except Exception as e:
                logger.warning(
                    "Hyperopt [%s]: trial #%d should_prune failed (eval_idx=%s, timesteps=%s): %r",
                    self.trial.study.study_name,
                    self.trial.number,
                    self.eval_idx,
                    self.num_timesteps,
                    e,
                    exc_info=True,
                )
                self.is_pruned = True
                return False
        return True

    def update_best_reward(self, mean_reward: float, model: Any) -> None:
        """Update final-policy bookkeeping without reporting another trial step."""
        _update_eval_best_reward(self, mean_reward, model)


class SimpleLinearSchedule:
    """
    Linear schedule from the initial value to zero. Unlike SB3's LinearSchedule,
    the progress factor is clamped to [0, 1] so overshooting the aligned training
    budget never produces a negative learning rate or clip range.

    :param initial_value: (float or str) The initial value for the schedule
    """

    def __init__(self, initial_value: float | str) -> None:
        # Force conversion to float
        self.initial_value = float(initial_value)

    def __call__(self, progress_remaining: float) -> float:
        # SB3 can request negative progress when overshooting the aligned
        # budget (e.g. DQN finishing its train_freq); never emit a negative
        # learning rate or clip range.
        return min(1.0, max(0.0, float(progress_remaining))) * self.initial_value

    def __repr__(self) -> str:
        return f"SimpleLinearSchedule(initial_value={self.initial_value})"


def deepmerge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dicts without mutating inputs"""
    dst_copy = copy.deepcopy(dst)
    for k, v in src.items():
        if k in dst_copy and isinstance(dst_copy[k], Mapping) and isinstance(v, Mapping):
            dst_copy[k] = deepmerge(dst_copy[k], v)
        else:
            dst_copy[k] = v
    return dst_copy


def _compute_gradient_steps(tf: int, ss: int) -> int:
    if tf > 0 and ss > 0:
        return min(tf, max(1, math.ceil(tf / ss)))
    return -1


def compute_gradient_steps(train_freq: Any, subsample_steps: Any) -> int:
    tf: int | None = None
    if isinstance(train_freq, TrainFreq):
        tf = train_freq.frequency if isinstance(train_freq.frequency, int) else None
    if isinstance(train_freq, (tuple, list)) and train_freq:
        tf = train_freq[0] if isinstance(train_freq[0], int) else None
    elif isinstance(train_freq, int):
        tf = train_freq

    ss: int | None = subsample_steps if isinstance(subsample_steps, int) else None

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
) -> tuple[ScheduleType, float, float]:
    if isinstance(schedule, (int, float)):
        try:
            schedule = float(schedule)
            return ReforceXY._SCHEDULE_TYPES[1], schedule, schedule  # "constant"
        except Exception:
            return ReforceXY._SCHEDULE_TYPES[1], np.nan, np.nan  # "constant"
    elif isinstance(schedule, ConstantSchedule):
        try:
            return (
                ReforceXY._SCHEDULE_TYPES[1],
                schedule(1.0),
                schedule(0.0),
            )  # "constant"
        except Exception:
            return ReforceXY._SCHEDULE_TYPES[1], np.nan, np.nan  # "constant"
    elif isinstance(schedule, SimpleLinearSchedule):
        try:
            return (
                ReforceXY._SCHEDULE_TYPES[0],
                schedule(1.0),
                schedule(0.0),
            )  # "linear"
        except Exception:
            return ReforceXY._SCHEDULE_TYPES[0], np.nan, np.nan  # "linear"

    return ReforceXY._SCHEDULE_TYPES[2], np.nan, np.nan  # "unknown"


def get_schedule(
    schedule_type: ScheduleTypeKnown,
    initial_value: float,
) -> Callable[[float], float]:
    if schedule_type == ReforceXY._SCHEDULE_TYPES[0]:  # "linear"
        return SimpleLinearSchedule(initial_value)
    elif schedule_type == ReforceXY._SCHEDULE_TYPES[1]:  # "constant"
        return ConstantSchedule(initial_value)
    else:
        return ConstantSchedule(initial_value)


def get_net_arch(model_type: str, net_arch_type: NetArchSize) -> list[int] | dict[str, list[int]]:
    """
    Get network architecture
    """
    if ReforceXY._MODEL_TYPES[0] in model_type:  # "PPO"
        return {
            ReforceXY._NET_ARCH_SIZES[0]: {
                "pi": [128, 128],
                "vf": [128, 128],
            },  # "small"
            ReforceXY._NET_ARCH_SIZES[1]: {
                "pi": [256, 256],
                "vf": [256, 256],
            },  # "medium"
            ReforceXY._NET_ARCH_SIZES[2]: {
                "pi": [512, 512],
                "vf": [512, 512],
            },  # "large"
            ReforceXY._NET_ARCH_SIZES[3]: {
                "pi": [1024, 1024],
                "vf": [1024, 1024],
            },  # "extra_large"
        }.get(net_arch_type, {"pi": [128, 128], "vf": [128, 128]})
    return {
        ReforceXY._NET_ARCH_SIZES[0]: [128, 128],  # "small"
        ReforceXY._NET_ARCH_SIZES[1]: [256, 256],  # "medium"
        ReforceXY._NET_ARCH_SIZES[2]: [512, 512],  # "large"
        ReforceXY._NET_ARCH_SIZES[3]: [1024, 1024],  # "extra_large"
    }.get(net_arch_type, [128, 128])


def get_activation_fn(
    activation_fn_name: ActivationFunction,
) -> type[th.nn.Module]:
    """
    Get activation function
    """
    return {
        ReforceXY._ACTIVATION_FUNCTIONS[0]: th.nn.ReLU,  # "relu"
        ReforceXY._ACTIVATION_FUNCTIONS[1]: th.nn.Tanh,  # "tanh"
        ReforceXY._ACTIVATION_FUNCTIONS[2]: th.nn.ELU,  # "elu"
        ReforceXY._ACTIVATION_FUNCTIONS[3]: th.nn.LeakyReLU,  # "leaky_relu"
    }.get(activation_fn_name, th.nn.ReLU)


def get_optimizer_class(
    optimizer_class_name: OptimizerClass,
) -> type[th.optim.Optimizer]:
    """
    Get optimizer class
    """
    return {
        ReforceXY._OPTIMIZER_CLASSES[0]: th.optim.AdamW,  # "adamw"
        ReforceXY._OPTIMIZER_CLASSES[1]: th.optim.RMSprop,  # "rmsprop"
        ReforceXY._OPTIMIZER_CLASSES[2]: th.optim.Adam,  # "adam"
    }.get(optimizer_class_name, th.optim.Adam)


def convert_optuna_params_to_model_params(
    model_type: str, optuna_params: dict[str, Any]
) -> dict[str, Any]:
    model_params: dict[str, Any] = {}
    policy_kwargs: dict[str, Any] = {}

    lr = optuna_params.get("learning_rate")
    if lr is None:
        raise ValueError(f"Hyperopt [{model_type}]: missing 'learning_rate' in params")
    lr = get_schedule(
        optuna_params.get("lr_schedule", ReforceXY._SCHEDULE_TYPES[1]), float(lr)
    )  # default: "constant"

    if ReforceXY._MODEL_TYPES[0] in model_type:  # "PPO"
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
                raise ValueError(f"Hyperopt [{model_type}]: missing '{param}' in params")
        cr = optuna_params.get("clip_range")
        cr = get_schedule(
            optuna_params.get("cr_schedule", ReforceXY._SCHEDULE_TYPES[1]),
            float(cr),
        )  # default: "constant"

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
        if "target_kl" in optuna_params:
            target_kl = optuna_params["target_kl"]
            model_params["target_kl"] = None if target_kl is None else float(target_kl)
        if ReforceXY._MODEL_TYPES[1] in model_type:  # "RecurrentPPO"
            policy_kwargs["lstm_hidden_size"] = int(optuna_params.get("lstm_hidden_size"))
            policy_kwargs["n_lstm_layers"] = int(optuna_params.get("n_lstm_layers"))
            if optuna_params.get("enable_critic_lstm") is not None:
                policy_kwargs["enable_critic_lstm"] = bool(optuna_params.get("enable_critic_lstm"))
    elif ReforceXY._MODEL_TYPES[3] in model_type:  # "DQN"
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
                raise ValueError(f"Hyperopt [{model_type}]: missing '{param}' in params")
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
                "exploration_fraction": float(optuna_params.get("exploration_fraction")),
                "exploration_initial_eps": float(optuna_params.get("exploration_initial_eps")),
                "exploration_final_eps": float(optuna_params.get("exploration_final_eps")),
                "target_update_interval": int(optuna_params.get("target_update_interval")),
                "learning_starts": int(optuna_params.get("learning_starts")),
            }
        )
        if (
            ReforceXY._MODEL_TYPES[4] in model_type and optuna_params.get("n_quantiles") is not None
        ):  # "QRDQN"
            policy_kwargs["n_quantiles"] = int(optuna_params["n_quantiles"])
    else:
        raise ValueError(f"Hyperopt [global]: model type '{model_type}' not supported")

    if optuna_params.get("net_arch"):
        net_arch_value = str(optuna_params["net_arch"])
        if net_arch_value in ReforceXY._NET_ARCH_SIZES:
            policy_kwargs["net_arch"] = get_net_arch(
                model_type,
                cast("NetArchSize", net_arch_value),
            )
    if optuna_params.get("activation_fn"):
        activation_fn_value = str(optuna_params["activation_fn"])
        if activation_fn_value in ReforceXY._ACTIVATION_FUNCTIONS:
            policy_kwargs["activation_fn"] = get_activation_fn(
                cast("ActivationFunction", activation_fn_value)
            )
    if optuna_params.get("optimizer_class"):
        optimizer_value = str(optuna_params["optimizer_class"])
        if optimizer_value in ReforceXY._OPTIMIZER_CLASSES:
            policy_kwargs["optimizer_class"] = get_optimizer_class(
                cast("OptimizerClass", optimizer_value)
            )
    if optuna_params.get("ortho_init") is not None:
        policy_kwargs["ortho_init"] = bool(optuna_params["ortho_init"])

    model_params["policy_kwargs"] = policy_kwargs
    return model_params


def get_common_ppo_optuna_params(trial: Trial) -> dict[str, Any]:
    return {
        "n_steps": trial.suggest_categorical("n_steps", list(ReforceXY._PPO_N_STEPS)),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024]),
        "gamma": trial.suggest_categorical(
            "gamma", [0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999, 0.9999]
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.03, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4, step=0.05),
        "n_epochs": trial.suggest_int("n_epochs", 1, 10),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0, step=0.01),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0, step=0.1),
        "vf_coef": trial.suggest_float("vf_coef", 0.0, 1.0, step=0.05),
        "lr_schedule": trial.suggest_categorical(
            "lr_schedule", list(ReforceXY._SCHEDULE_TYPES_KNOWN)
        ),
        "cr_schedule": trial.suggest_categorical(
            "cr_schedule", list(ReforceXY._SCHEDULE_TYPES_KNOWN)
        ),
        "target_kl": trial.suggest_categorical(
            "target_kl", [None, 0.003, 0.01, 0.015, 0.02, 0.03, 0.04, 0.1]
        ),
        "ortho_init": trial.suggest_categorical("ortho_init", [True, False]),
        "net_arch": trial.suggest_categorical("net_arch", list(ReforceXY._NET_ARCH_SIZES)),
        "activation_fn": trial.suggest_categorical(
            "activation_fn", list(ReforceXY._ACTIVATION_FUNCTIONS)
        ),
        "optimizer_class": trial.suggest_categorical(
            "optimizer_class", list(ReforceXY._OPTIMIZER_CLASSES_OPTUNA)
        ),
    }


def sample_params_ppo(trial: Trial) -> dict[str, Any]:
    """
    Sampler for PPO hyperparams
    """
    return convert_optuna_params_to_model_params(
        ReforceXY._MODEL_TYPES[0], get_common_ppo_optuna_params(trial)
    )


def sample_params_recurrentppo(trial: Trial, *, shared_lstm: bool = False) -> dict[str, Any]:
    """
    Sampler for RecurrentPPO hyperparams
    """
    ppo_optuna_params = get_common_ppo_optuna_params(trial)
    ppo_optuna_params.update(
        {
            "n_lstm_layers": trial.suggest_int("n_lstm_layers", 1, 2),
            "lstm_hidden_size": trial.suggest_categorical("lstm_hidden_size", [64, 128, 256, 512]),
            "enable_critic_lstm": False
            if shared_lstm
            else trial.suggest_categorical("enable_critic_lstm", [True, False]),
        }
    )
    return convert_optuna_params_to_model_params(ReforceXY._MODEL_TYPES[1], ppo_optuna_params)


def get_common_dqn_optuna_params(trial: Trial) -> dict[str, Any]:
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.2, step=0.01)
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
        "train_freq": trial.suggest_categorical("train_freq", ReforceXY._DQN_TRAIN_FREQS),
        "subsample_steps": trial.suggest_categorical("subsample_steps", [2, 4, 8, 16]),
        "gamma": trial.suggest_categorical(
            "gamma", [0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999, 0.9999]
        ),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "lr_schedule": trial.suggest_categorical(
            "lr_schedule", list(ReforceXY._SCHEDULE_TYPES_KNOWN)
        ),
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [int(1e4), int(5e4), int(1e5), int(5e5), int(1e6)]
        ),
        "exploration_initial_eps": exploration_initial_eps,
        "exploration_final_eps": exploration_final_eps,
        "exploration_fraction": trial.suggest_float(
            "exploration_fraction", min_fraction, 0.9, step=0.02
        ),
        "target_update_interval": trial.suggest_categorical(
            "target_update_interval", [1, 1000, 2000, 5000, 7500, 10000]
        ),
        "learning_starts": trial.suggest_categorical(
            "learning_starts", [500, 1000, 2000, 5000, 10000, 25000, 50000]
        ),
        "net_arch": trial.suggest_categorical("net_arch", list(ReforceXY._NET_ARCH_SIZES)),
        "activation_fn": trial.suggest_categorical(
            "activation_fn", list(ReforceXY._ACTIVATION_FUNCTIONS)
        ),
        "optimizer_class": trial.suggest_categorical(
            "optimizer_class", list(ReforceXY._OPTIMIZER_CLASSES_OPTUNA)
        ),
    }


def sample_params_dqn(trial: Trial) -> dict[str, Any]:
    """
    Sampler for DQN hyperparams
    """
    return convert_optuna_params_to_model_params(
        ReforceXY._MODEL_TYPES[3], get_common_dqn_optuna_params(trial)
    )


def sample_params_qrdqn(trial: Trial) -> dict[str, Any]:
    """
    Sampler for QRDQN hyperparams
    """
    dqn_optuna_params = get_common_dqn_optuna_params(trial)
    dqn_optuna_params.update({"n_quantiles": trial.suggest_int("n_quantiles", 10, 250)})
    return convert_optuna_params_to_model_params(ReforceXY._MODEL_TYPES[4], dqn_optuna_params)
