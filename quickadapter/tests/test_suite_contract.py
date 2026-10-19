"""Guards the import identity and coverage denominator of the measured source.; requires the Freqtrade QA image."""

import importlib
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from freqtrade.enums import RunMode
from qa_support import REPO_ROOT, QaTestCase, model_config, temporary_directory
from QuickAdapterV3 import QuickAdapterV3

QUICKADAPTER = REPO_ROOT / "quickadapter"
MEASURED_SOURCE = QUICKADAPTER / "user_data"
_SOURCES = tuple(sorted(MEASURED_SOURCE.rglob("*.py")))
# The measured tree is `quickadapter/user_data` WHOLESALE, so the contract walks it recursively:
# a flat module such as `user_data/helpers.py` enters the denominator and must not be invisible here.
# Strategies modules are imported by bare name, which is what production does; anything else is
# reachable only as a dotted path under `quickadapter.user_data`. Deriving the names from the tree
# is what keeps the contract true when a source file is added.
PRODUCTION_NAMES = tuple(
    src.stem
    if src.parent.name == "strategies"
    else src.relative_to(QUICKADAPTER.parent).with_suffix("").as_posix().replace("/", ".")
    for src in _SOURCES
)
TESTS_ROOT = QUICKADAPTER / "tests"

# Imported here, at module scope, so the parity contract holds on its own: unittest discovery
# imports every test module before running any test, so a lazy import inside a method would make
# the parity assertion unconditionally true.
for _name in PRODUCTION_NAMES:
    importlib.import_module(_name)


def _source_files_under(root: Path) -> list[tuple[str, Path]]:
    """Return module registrations by resolved file, independently of import spelling."""
    found = []
    for name, module in list(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if raw:
            path = Path(raw).resolve()
            if path.suffix == ".py" and path.is_relative_to(root.resolve()):
                found.append((name, path))
    return sorted(found)


class SuiteContractTest(QaTestCase):
    def test_a_every_source_file_has_a_loaded_module(self):
        registered = _source_files_under(MEASURED_SOURCE)
        for src, expected in zip(_SOURCES, PRODUCTION_NAMES, strict=True):
            with self.subTest(module=src.name):
                self.assertEqual(
                    [expected], [name for name, path in registered if path == src.resolve()]
                )

    def test_no_package_marker_under_quickadapter(self):
        markers = sorted(
            path.relative_to(QUICKADAPTER).as_posix() for path in QUICKADAPTER.rglob("__init__.py")
        )
        self.assertEqual(
            [],
            markers,
            "a package marker makes the start directory a package, so the test modules lose "
            "top-level status and `from qa_support import ...` fails under -t discovery",
        )

    def test_no_module_is_registered_under_a_second_name(self):
        self.test_a_every_source_file_has_a_loaded_module()

    def test_every_test_module_is_registered_under_its_bare_stem(self):
        for name, path in _source_files_under(TESTS_ROOT):
            with self.subTest(module=name):
                self.assertEqual(path.stem, name)


class SuiteContractRegressionTest(QaTestCase):
    def test_actual_guard_rejects_aliases_by_file_not_prefix(self):
        for stem, alias in (
            ("QuickAdapterRegressorV3", "QuickAdapterRegressorV3"),
            ("QuickAdapterRegressorV3", "arbitrary_production_alias"),
            ("QuickAdapterV3", "arbitrary_strategy_alias"),
        ):
            with self.subTest(alias=alias):
                source = next(src for src in _SOURCES if src.stem == stem)
                module = ModuleType(alias)
                module.__file__ = str(source)
                result = unittest.TestResult()
                with mock.patch.dict(sys.modules, {alias: module}):
                    SuiteContractTest("test_a_every_source_file_has_a_loaded_module").run(result)
                self.assertEqual(result.errors, [])
                self.assertEqual(len(result.failures), 1)

    def test_actual_guard_rejects_dotted_test_registration(self):
        module = ModuleType("quickadapter.tests.shadow")
        module.__file__ = __file__
        result = unittest.TestResult()
        with mock.patch.dict(sys.modules, {module.__name__: module}):
            SuiteContractTest("test_every_test_module_is_registered_under_its_bare_stem").run(
                result
            )
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.failures), 1)


class ConfigBuilderTest(QaTestCase):
    def test_defaults_and_explicit_overrides_are_private_and_preserved(self):
        with temporary_directory() as directory:
            default = model_config(str(directory))
            self.assertEqual(default["runmode"], RunMode.DRY_RUN)
            self.assertEqual(default["user_data_dir"], directory)
            overrides = {"feature_parameters": {"label_period_candles": 7}}
            config = model_config(
                directory,
                runmode=RunMode.BACKTEST,
                user_data_dir=str(directory / "alternate"),
                trading_mode="futures",
                freqai=overrides,
            )
            self.assertEqual(config["runmode"], RunMode.BACKTEST)
            self.assertEqual(config["user_data_dir"], directory / "alternate")
            self.assertEqual(config["trading_mode"], "futures")
            self.assertEqual(config["freqai"]["feature_parameters"]["include_timeframes"], ["5m"])
            self.assertEqual(config["freqai"]["feature_parameters"]["label_period_candles"], 7)
            strategy = QuickAdapterV3(config)
            self.assertFalse(strategy.is_trade_runmode)
            config["freqai"]["feature_parameters"]["label_period_candles"] = 99
            config["exchange"]["pair_whitelist"].append("ETH/USDT")
            self.assertEqual(overrides["feature_parameters"]["label_period_candles"], 7)
            fresh = model_config(directory)
            self.assertEqual(fresh["freqai"]["feature_parameters"]["label_period_candles"], 1)
            self.assertEqual(len(fresh["exchange"]["pair_whitelist"]), 1)
