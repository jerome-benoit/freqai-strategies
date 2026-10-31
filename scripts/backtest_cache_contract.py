"""Exercise native startup and result-cache contracts for standalone strategies."""

import importlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from importlib.metadata import version as distribution_version
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import freqtrade
from BacktestIdentity import implementation_fingerprint
from freqtrade.configuration import TimeRange
from freqtrade.enums import RunMode
from freqtrade.optimize.backtest_caching import get_strategy_run_id
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.optimize.optimize_reports.bt_storage import store_backtest_results
from freqtrade.resolvers import StrategyResolver
from freqtrade.util import dt_now


def assert_native_implementation_cache(
    test: Any,
    directory: Path,
    config_factory: Any,
    strategy_name: str,
    project: Path,
    model_import: str,
) -> None:
    """A real native archive must hit unchanged inputs and miss changed code/SDK."""
    implementations = directory / "implementations"
    implementations.mkdir()
    helper_name = f"_cache_helper_{directory.name.replace('-', '_')}"
    helper = implementations / f"{helper_name}.py"
    helper.write_text("VALUE = 1\ndef coefficient():\n    return VALUE\n")
    model_path = implementations / "cache_model.py"
    model_source = (
        f"{model_import}\n"
        f"from {helper_name} import coefficient\n"
        "class CacheModel(CacheBase):\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        raise AssertionError('Identity must not construct a model')\n"
        "    def cache_coefficient(self):\n"
        "        return coefficient()\n"
    )
    model_path.write_text(model_source)
    config = config_factory(directory)
    config.update(
        runmode=RunMode.BACKTEST,
        strategy=strategy_name,
        freqaimodel="CacheModel",
        freqaimodel_path=str(implementations),
        backtest_cache="day",
    )
    output = directory / "backtest_results"
    output.mkdir()
    test.addCleanup(sys.modules.pop, helper_name, None)
    # Native exec_module does not register cache_model. A colliding module must
    # not replace the selected class's actual globals or imported helper source.
    collision = ModuleType("cache_model")
    collision.__file__ = str(implementations / "unrelated.py")

    def load():
        return StrategyResolver._load_strategy(
            strategy_name, config, extra_dir=str(project / "user_data/strategies")
        )

    def cached(strategy):
        backtest = object.__new__(Backtesting)
        backtest.config = strategy.config
        backtest.strategylist = [strategy]
        backtest.results = {}
        backtest.timerange = TimeRange.parse_timerange("20260101-20260102")
        backtest.load_prior_backtest()
        return backtest.results.get("strategy", {})

    def store(strategy):
        metadata = {
            "run_id": get_strategy_run_id(strategy),
            "backtest_start_time": int(dt_now().timestamp()),
            "backtest_start_ts": 1761955200,
            "backtest_end_ts": 1762041600,
            "timeframe": "5m",
        }
        store_backtest_results(
            {"exportdirectory": output, "original_config": config},
            {
                "metadata": {strategy_name: metadata},
                "strategy": {strategy_name: {"profit_total": 0.125}},
                "strategy_comparison": [{"key": strategy_name}],
            },
            dt_now().strftime("%Y-%m-%d_%H-%M-%S"),
            strategy_files={strategy_name: strategy.__file__},
        )

    with mock.patch.dict(sys.modules, {"cache_model": collision}):
        store(load())
        test.assertEqual(cached(load())[strategy_name]["profit_total"], 0.125)
        nested = config["freqai"]
        snapshot = deepcopy(nested)
        config["api_server"] = {"enabled": False, "username": "unrelated-cache-setting"}
        config["telegram"] = {"enabled": False}
        test.assertEqual(cached(load())[strategy_name]["profit_total"], 0.125)
        test.assertIs(config["freqai"], nested)
        test.assertEqual(nested, snapshot)
        with mock.patch.object(freqtrade, "__version__", freqtrade.__version__ + "+cache-change"):
            test.assertNotIn(strategy_name, cached(load()))
        test.assertEqual(cached(load())[strategy_name]["profit_total"], 0.125)

        for dependency in ("torch", "ft-pandas-ta"):
            with test.subTest(dependency=dependency):

                def upgraded_dependency(name, selected_dependency=dependency):
                    return (
                        "999.0+cache-test"
                        if name == selected_dependency
                        else distribution_version(name)
                    )

                with mock.patch("BacktestIdentity.version", side_effect=upgraded_dependency):
                    test.assertNotIn(strategy_name, cached(load()))
                test.assertEqual(cached(load())[strategy_name]["profit_total"], 0.125)
        helper.write_text("VALUE = -1\ndef coefficient():\n    return VALUE\n")
        importlib.invalidate_caches()
        test.assertNotIn(strategy_name, cached(load()))
        helper.write_text("VALUE = 1\ndef coefficient():\n    return VALUE\n")
        model_path.write_text(model_source + "\nMODEL_REVISION = 2\n")
        test.assertNotIn(strategy_name, cached(load()))
        model_path.write_text(model_source)
        test.assertEqual(cached(load())[strategy_name]["profit_total"], 0.125)
        base_name = helper_name + "_base"
        test.addCleanup(sys.modules.pop, base_name, None)
        (implementations / f"{base_name}.py").write_text(
            f"{model_import}\n"
            "class CacheParent(CacheBase):\n"
            "    def __init__(self, *args, **kwargs):\n"
            "        raise AssertionError('Identity must not construct a model')\n"
            "    def cache_coefficient(self):\n"
            "        return self.cache_scale\n"
        )
        model_path.write_text(
            f"from {base_name} import CacheParent\n"
            f"import {helper_name} as settings\n"
            "class CacheModel(CacheParent):\n"
            "    cache_scale = settings.VALUE\n"
        )
        sys.modules.pop("cache_model")
        store(load())
        test.assertEqual(cached(load())[strategy_name]["profit_total"], 0.125)
        helper.write_text("VALUE = 0.5\ndef coefficient():\n    return VALUE\n")
        test.assertNotIn(strategy_name, cached(load()))
        trainer_name = helper_name + "_trainer"
        test.addCleanup(sys.modules.pop, trainer_name, None)
        (implementations / f"{trainer_name}.py").write_text(
            "def forbidden_init(self, *args, **kwargs):\n"
            "    raise AssertionError('Identity must not construct a model')\n"
        )
        helper.write_text("VALUE = 1\ndef coefficient():\n    return VALUE\n")
        model_path.write_text(
            f"from {base_name} import CacheParent\n"
            f"from {trainer_name} import forbidden_init\n"
            f"import {helper_name} as settings\n"
            "class CacheModel(CacheParent):\n"
            "    __init__ = forbidden_init\n"
            "    cache_scale = settings.VALUE\n"
        )
        store(load())
        test.assertEqual(cached(load())[strategy_name]["profit_total"], 0.125)
        helper.write_text("VALUE = 0.5\ndef coefficient():\n    return VALUE\n")
        test.assertNotIn(strategy_name, cached(load()))
        helper.write_text(
            "from functools import lru_cache\n"
            "VALUE = 1\n@lru_cache(None)\ndef coefficient():\n    return VALUE\n"
        )
        sys.modules.pop(helper_name)
        model_path.write_text(model_source)
        sys.modules["cache_model"] = collision
        store(load())
        test.assertEqual(cached(load())[strategy_name]["profit_total"], 0.125)
        helper.write_text(
            "from functools import lru_cache\n"
            "VALUE = 0.5\n@lru_cache(None)\ndef coefficient():\n    return VALUE\n"
        )
        test.assertNotIn(strategy_name, cached(load()))
        for imports, ancestor in (
            ("from sklearn.base import BaseEstimator", "BaseEstimator"),
            ("from typing import Generic, TypeVar\nT = TypeVar('T')", "Generic[T]"),
        ):
            with test.subTest(external_ancestor=ancestor):
                model_path.write_text(
                    f"{model_import}\n{imports}\n"
                    f"class CacheModel(CacheBase, {ancestor}):\n"
                    "    def __init__(self, *args, **kwargs):\n"
                    "        raise AssertionError('Identity must not construct a model')\n"
                )
                store(load())
                test.assertEqual(cached(load())[strategy_name]["profit_total"], 0.125)
        with test.subTest(helper_import="dynamic", leaf_methods=False):
            helper.write_text("VALUE = 1\n")
            sys.modules.pop(helper_name, None)
            model_path.write_text(
                "import importlib\n"
                f"from {base_name} import CacheParent\n"
                f"settings = importlib.import_module({helper_name!r})\n"
                "class CacheModel(CacheParent):\n"
                "    cache_scale = settings.VALUE\n"
            )
            store(load())
            test.assertEqual(cached(load())[strategy_name]["profit_total"], 0.125)
            helper.write_text("VALUE = 2\n")
            sys.modules.pop(helper_name, None)
            importlib.invalidate_caches()
            test.assertNotIn(strategy_name, cached(load()))
        with test.subTest(source_boundary="conservative"):
            store(load())
            extra = implementations / "extra.py"
            extra.write_text("unused Python need not parse!\n")
            test.assertNotIn(strategy_name, cached(load()))
            store(load())
            extra.write_text("UNUSED = 42\n")
            test.assertNotIn(strategy_name, cached(load()))
            store(load())
            renamed = implementations / "renamed.py"
            extra.rename(renamed)
            test.assertNotIn(strategy_name, cached(load()))
            renamed.unlink()
            marker = implementations / "marker.py"
            marker.write_text("UNUSED_MARKER = True\n")
            extra.write_text("UNUSED = 42\n")
            store(load())
            extra.unlink()
            test.assertNotIn(strategy_name, cached(load()))
            marker.unlink()
            store(load())
            (implementations / "model.joblib").write_bytes(b"generated model")
            (implementations / "metadata.json").write_text('{"generated": true}\n')
            test.assertEqual(cached(load())[strategy_name]["profit_total"], 0.125)


def assert_source_tree_provenance(test: Any, directory: Path) -> None:
    """Source location, aliases and resolver snapshots must not corrupt cache identity."""
    source = "class Selected:\n    pass\n"
    implementations = []
    for name in ("first", "relocated"):
        root = directory / name
        root.mkdir()
        path = root / "selected.py"
        path.write_text(source)
        implementations.append(
            type(
                "Selected",
                (),
                {"__module__": "_cache_provenance", "__file__": str(path), "__source__": source},
            )
        )
    selected, relocated = implementations
    fingerprint = implementation_fingerprint({"model": selected})
    test.assertEqual(fingerprint, implementation_fingerprint({"model": relocated}))
    root = Path(selected.__file__).parent
    path = root / "selected.py"
    path.write_text("class Selected:\n    changed_on_disk = True\n")
    test.assertEqual(fingerprint, implementation_fingerprint({"model": selected}))
    path.unlink()
    test.assertEqual(fingerprint, implementation_fingerprint({"model": selected}))
    path.write_text(source)
    changed = type(
        "Selected",
        (),
        {
            "__module__": "_cache_provenance",
            "__file__": str(path),
            "__source__": source.replace("pass", "changed_snapshot = True"),
        },
    )
    ordered = implementation_fingerprint({"model": selected, "strategy": changed})
    test.assertEqual(ordered, implementation_fingerprint({"strategy": changed, "model": selected}))
    test.assertNotEqual(
        ordered, implementation_fingerprint({"model": changed, "strategy": selected})
    )
    packages = [root / name for name in ("one", "two")]
    for package, value in zip(packages, (1, 2), strict=True):
        package.mkdir()
        (package / "settings.py").write_text(f"VALUE = {value}\n")
    before_swap = implementation_fingerprint({"model": selected})
    for package, value in zip(packages, (2, 1), strict=True):
        (package / "settings.py").write_text(f"VALUE = {value}\n")
    test.assertNotEqual(before_swap, implementation_fingerprint({"model": selected}))
    external = directory / "external"
    external.mkdir()
    helper = external / "helper.py"
    helper.write_text("VALUE = 1\n")
    (root / "linked.py").symlink_to(helper)
    (root / "cycle").symlink_to(root, target_is_directory=True)
    (root / "outside").symlink_to(external, target_is_directory=True)
    before_link_change = implementation_fingerprint({"model": selected})
    helper.write_text("VALUE = 2\n")
    after_link_change = implementation_fingerprint({"model": selected})
    test.assertNotEqual(before_link_change, after_link_change)
    (external / "unrelated.py").write_text("VALUE = 3\n")
    test.assertEqual(after_link_change, implementation_fingerprint({"model": selected}))


def assert_native_startup_predictions(
    test: Any, directory: Path, config_factory: Any, project: Path
) -> None:
    """Train and predict in a fresh interpreter without QA's strategy PYTHONPATH."""
    config = config_factory(directory)
    info = config["freqai"]
    if project.name == "quickadapter":
        info["optuna_hyperopt"]["enabled"] = False
        info["model_training_parameters"]["random_state"] = 42
    else:
        info["model_training_parameters"]["seed"] = 42
    config_path = directory / "startup.json"
    program = """
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from freqtrade.configuration import Configuration
from freqtrade.configuration.load_config import load_config_file
from freqtrade.enums import RunMode
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.resolvers import StrategyResolver
from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver
project = Path(sys.argv[1])
source = project / 'user_data'
options_path = Path(sys.argv[2])
options = json.loads(options_path.read_text())
mode = sys.argv[3]
config = load_config_file(source / 'config-template.json')
options['exchange'] = config['exchange'] | options['exchange']
config.update(options)
config['user_data_dir'] = str(Path(config['user_data_dir']) / mode)
Path(config['user_data_dir']).mkdir(parents=True, exist_ok=True)
quick = project.name == 'quickadapter'
config.update(strategy='QuickAdapterV3' if quick else 'RLAgentStrategy',
              freqaimodel='QuickAdapterRegressorV3' if quick else 'ReforceXY',
              strategy_path=str(source / 'strategies'),
              freqaimodel_path=str(source / 'freqaimodels'))
options_path.write_text(json.dumps(config, default=str))
config = Configuration({'config': [str(options_path)]}, RunMode(mode)).get_config()
strategy = StrategyResolver.load_strategy(config)
owner = FreqaiModelResolver.load_freqaimodel(config)
owner.live = True
owner.can_short = False
index = np.arange(64)
label = '&s-extrema' if quick else '&-action'
frame = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=64, freq='5min', tz='UTC'),
                      '%-feature': np.sin(index / 4),
                      label: np.cos(index / 5) if quick else np.zeros(64)})
if not quick:
    for column in ('open', 'high', 'low', 'close'):
        frame['%-raw_' + column] = 100 + index * 0.1
kitchen = FreqaiDataKitchen(config, live=True, pair='BTC/USDT')
kitchen.set_paths(kitchen.pair, 1767225600)
kitchen.set_new_model_names(kitchen.pair, 1767225600)
kitchen.data_path.mkdir(parents=True, exist_ok=True)
kitchen.training_features_list = [name for name in frame if name.startswith('%')]
kitchen.label_list = [label]
try:
    owner.model = owner.train(frame, kitchen.pair, kitchen)
    owner.live = False
    predictions, status = owner.predict(frame, kitchen)
    assert np.isfinite(predictions[label]).all()
    assert np.all(status == 1)
    if not quick:
        assert predictions[label].isin([0, 1, 2]).all()
    print('NATIVE_PREDICTIONS=' + json.dumps({
        'rows': len(predictions), 'valid_rows': int(np.sum(status == 1)),
        'runmode': str(config['runmode']),
        'strategy_directory_on_path': str(source / 'strategies') in sys.path}))
finally:
    if not quick:
        owner.close_envs()
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    for mode in ("dry_run", "backtest", "hyperopt"):
        with test.subTest(runmode=mode):
            config_path.write_text(json.dumps(config, default=str))
            result = subprocess.run(
                [sys.executable, "-P", "-c", program, str(project), str(config_path), mode],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            test.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            line = next(
                row for row in result.stdout.splitlines() if row.startswith("NATIVE_PREDICTIONS=")
            )
            outcome = json.loads(line.removeprefix("NATIVE_PREDICTIONS="))
            test.assertEqual(outcome["rows"], 64)
            test.assertEqual(outcome["valid_rows"], 64)
            test.assertEqual(outcome["runmode"], mode)
            test.assertFalse(outcome["strategy_directory_on_path"])
