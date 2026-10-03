"""Shared builders and process-state resets for the ReforceXY test suite."""

import copy
import hashlib
import importlib
import os
import random
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from freqtrade.enums import RunMode

PAIR = "BTC/USDT"
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT / "ReforceXY"
COVERAGERC = PROJECT_ROOT / ".coveragerc"
TESTS_ROOT = PROJECT_ROOT / "tests"
MEASURED_TREE = PROJECT_ROOT / "user_data"
PRODUCTION_MODULE = "ReforceXY.user_data.freqaimodels.ReforceXY"
STRATEGY_MODULE = "ReforceXY.user_data.strategies.RLAgentStrategy"

# Never handed out: every call deep-copies it, so a test that mutates the returned
# config cannot leak into the next one.
_BASE_CONFIG: dict[str, Any] = {
    "timeframe": "5m",
    "stake_amount": "unlimited",
    "exchange": {"pair_whitelist": [PAIR]},
    "freqai": {
        "enabled": True,
        "identifier": "reforcexy-runtime-regression",
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
        # Use local TPE by default in QA. Explicit overrides are preserved; `auto`
        # loads an OptunaHub sampler and may require network access.
        "rl_config_optuna": {"enabled": False, "sampler": "tpe"},
    },
}


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def model_config(tmp_path: Path | str, **overrides: Any) -> dict[str, Any]:
    """Return a private Freqtrade config rooted at `tmp_path`, deep-merged with `overrides`.

    `runmode` is applied only when the caller supplied none, so a test can ask for
    RunMode.BACKTEST and get it. QuickAdapter assigns it after the merge and is therefore
    not overridable; copying that order here would silently hand DRY_RUN to the backtest
    regressions.
    """
    config = _merge(_BASE_CONFIG, overrides)
    config["user_data_dir"] = Path(config.get("user_data_dir", tmp_path))
    config.setdefault("runmode", RunMode.DRY_RUN)
    return config


@contextmanager
def temporary_directory() -> Iterator[Path]:
    """Yield a temporary directory that is removed on exit. The only call shape."""
    with TemporaryDirectory() as path:
        yield Path(path)


class RecordingPolicy:
    """A stand-in for an SB3 policy that records what it was asked to predict."""

    def __init__(self):
        self.observations = []
        self.masks = []

    def predict(self, observation, **kwargs):
        self.observations.append(observation.copy())
        self.masks.append(kwargs.get("action_masks"))
        return np.array([0]), None


# --- test-order independence ------------------------------------------------------
#
# `unittest.TestLoader.sortTestMethodsUsing` is a staticmethod ON THE LOADER; a
# TestCase has no such hook. Assigning a plain function would bind it as a method and
# raise TypeError at discovery, so the descriptor is explicit. The comparator is a pure
# function of (seed, name): a fresh random sign per comparison is not a transitive
# order, and one shared RNG would make each class's order depend on how many classes
# were loaded before it. With no seed it falls through to name order, which reproduces
# alphabetical discovery exactly, so the patch is unconditional and every invocation is
# covered.
_SEED = os.getenv("FREQAI_QA_SHUFFLE_SEED", "")


def reseed(value: str) -> str:
    """Set the loader's shuffle seed and return its previous value.

    Exists so the suite contract can exercise the comparator the loader really
    holds, rather than a copy of it. Call it only while nothing else is
    discovering tests, and restore the returned value afterwards.
    """
    global _SEED
    previous = _SEED
    _SEED = value
    return previous


def _shuffle_cmp(a: str, b: str) -> int:
    """Order test method names by a seeded hash, or by name when no seed is set."""
    if not _SEED:
        return (a > b) - (a < b)
    ka = int.from_bytes(hashlib.blake2b(f"{_SEED}:{a}".encode(), digest_size=8).digest())
    kb = int.from_bytes(hashlib.blake2b(f"{_SEED}:{b}".encode(), digest_size=8).digest())
    return (ka > kb) - (ka < kb)


unittest.TestLoader.sortTestMethodsUsing = staticmethod(_shuffle_cmp)


# --- process globals --------------------------------------------------------------
#
# Captured once, after the module's import-time registrations, so a test that mutates one
# is undone without destroying the real ones.
_PRODUCTION = importlib.import_module(PRODUCTION_MODULE)
_ACTION_MASKS_CACHE_AT_IMPORT = dict(_PRODUCTION.ReforceXY._action_masks_cache)
_RANDOM_STATE_AT_IMPORT = random.getstate()
_NUMPY_STATE_AT_IMPORT = np.random.get_state()
_TORCH_STATE_AT_IMPORT = None
try:  # torch is present in the RL QA image; a CPU-only box may still lack it
    import torch

    _TORCH_STATE_AT_IMPORT = torch.random.get_rng_state()
except ImportError:  # pragma: no cover - the RL image always provides torch
    torch = None  # type: ignore[assignment]


class QaTestCase(unittest.TestCase):
    """Base case restoring the four process globals, in setUp and again in addCleanup.

    `_action_masks_cache` is a pure memoisation: it is written from a value derived
    deterministically from the `(can_short, position)` key and read only through
    `ReforceXY._action_masks_cache`, so clearing it cannot change any returned mask.
    The RNG states matter because `_get_train_and_eval_environments` calls the
    process-wide `set_random_seed`, so without a restore any stochastic assertion would
    depend on test order.
    """

    def setUp(self):
        self._restore_process_globals()
        self.addCleanup(self._restore_process_globals)

    @staticmethod
    def _restore_process_globals() -> None:
        _PRODUCTION.ReforceXY._action_masks_cache.clear()
        _PRODUCTION.ReforceXY._action_masks_cache.update(_ACTION_MASKS_CACHE_AT_IMPORT)
        random.setstate(_RANDOM_STATE_AT_IMPORT)
        np.random.set_state(_NUMPY_STATE_AT_IMPORT)
        if torch is not None:
            torch.random.set_rng_state(_TORCH_STATE_AT_IMPORT)
