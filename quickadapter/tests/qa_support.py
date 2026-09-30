"""Shared builders and process-state resets for the QuickAdapter test suite."""

import copy
import importlib
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import Utils
from freqtrade.enums import RunMode

PAIR = "BTC/USDT"
REGRESSOR_MODULE = "quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_TEMPLATE = REPO_ROOT / "quickadapter" / "user_data" / "config-template.json"
COVERAGERC = REPO_ROOT / "quickadapter" / ".coveragerc"

# Never handed out: model_config deep-copies it, because both production __init__s call
# migrate_config, which mutates the config in place via _set_path/_delete_path.
_BASE_CONFIG: dict[str, Any] = {
    "timeframe": "5m",
    "stake_amount": "unlimited",
    "exchange": {"pair_whitelist": [PAIR]},
    "pairlists": [{"method": "StaticPairList"}],
    # No api_server: QuickAdapterV3.__init__ installs the RPC monkeypatch when it is enabled.
    "freqai": {
        "enabled": True,
        "identifier": "quickadapter-runtime-regression",
        "continual_learning": True,
        "train_period_days": 1,
        "backtest_period_days": 1,
        "conv_width": 1,
        "fit_live_predictions_candles": 2,
        "feature_parameters": {
            "include_timeframes": ["5m"],
            "include_corr_pairlist": [],
            "label_period_candles": 1,
            "shuffle_after_split": False,
        },
        "data_split_parameters": {"test_size": 0, "shuffle": False},
        "model_training_parameters": {"n_estimators": 2, "n_jobs": 1},
        "label_prediction": {"method": "none"},
        # optuna_create_sampler resolves the "auto" sampler for both namespaces, and "auto"
        # is an outbound fetch from hub.optuna.org. Pin both so no test can reach it.
        "optuna_hyperopt": {"enabled": True, "sampler": "tpe", "label_sampler": "tpe"},
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
    """Return a private Freqtrade config rooted at `tmp_path`, deep-merged with `overrides`."""
    config = _merge(_BASE_CONFIG, overrides)
    config["user_data_dir"] = tmp_path
    config["runmode"] = RunMode.DRY_RUN
    return config


@contextmanager
def temporary_directory() -> Iterator[Path]:
    """Yield a temporary directory that is removed on exit. The only call shape."""
    with TemporaryDirectory() as path:
        yield Path(path)


class QaTestCase(unittest.TestCase):
    """Base case clearing the three process globals, in setUp and again in addCleanup."""

    def setUp(self):
        self._restore_process_globals()
        self.addCleanup(self._restore_process_globals)

    @staticmethod
    def _restore_process_globals() -> None:
        Utils._WARNED_CONFIG_DEPRECATIONS.clear()
        Utils._LABEL_GENERATORS.clear()
        importlib.import_module(REGRESSOR_MODULE)._KNOWN_AT_NONE_LOGGED.clear()
