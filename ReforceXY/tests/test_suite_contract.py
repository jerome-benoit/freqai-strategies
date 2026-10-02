"""Guards the import identity and coverage denominator of the measured source.; requires the Freqtrade QA image."""

import importlib
import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from qa_support import (
    _ACTION_MASKS_CACHE_AT_IMPORT,
    _NUMPY_STATE_AT_IMPORT,
    _PRODUCTION,
    _RANDOM_STATE_AT_IMPORT,
    _TORCH_STATE_AT_IMPORT,
    MEASURED_TREE,
    PRODUCTION_MODULE,
    STRATEGY_MODULE,
    TESTS_ROOT,
    QaTestCase,
    _shuffle_cmp,
)

PRODUCTION_NAMES = (PRODUCTION_MODULE, STRATEGY_MODULE)
# `ReforceXY/user_data` holds no `__init__.py`, so `ReforceXY`, `ReforceXY.user_data` and
# `ReforceXY.user_data.freqaimodels` are PEP 420 namespace packages carrying no `__file__`
# at all; `Path(None)` would raise instead of asserting, so every walk below filters on the
# truthiness of `__file__` first.
_SOURCES = tuple(sorted(MEASURED_TREE.rglob("*.py")))

# This class derives from unittest.TestCase, NOT from QaTestCase, and so runs no restore.
# That is deliberate: a check of the restore itself must observe what the restore left
# behind, not the state setUp already put back. It is one of the two exemptions from the
# inheritance rule below, together with test_coverage_floor.py.
_EXEMPT_MODULES = {"test_suite_contract", "test_coverage_floor"}

# Imported here, at module scope, so the parity contract holds on its own: unittest discovery
# imports every test module before running any test, so a lazy import inside a method would
# make the assertion unconditionally true. Dotted paths only — CI puts `user_data/strategies`
# on PYTHONPATH, so a bare-stem name would register a second module object for the very file
# this contract pins.
for _name in PRODUCTION_NAMES:
    importlib.import_module(_name)


def _source_files_under(root: Path) -> list[tuple[str, Path]]:
    """Registered modules whose `__file__` is a `.py` under `root`, as (name, resolved path)."""
    found = []
    for name, module in list(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if not raw:
            continue
        path = Path(raw).resolve()
        if path.suffix == ".py" and path.is_relative_to(root):
            found.append((name, path))
    return sorted(found)


class SuiteContractTest(unittest.TestCase):
    def test_each_measured_source_file_is_registered_under_exactly_one_dotted_name(self):
        registered = [
            path
            for name, path in _source_files_under(MEASURED_TREE.parent)
            if name.startswith("ReforceXY.user_data")
        ]
        for src in _SOURCES:
            with self.subTest(module=src.name):
                matches = [path for path in registered if path == src.resolve()]
                self.assertEqual(
                    1,
                    len(matches),
                    "a measured file under exactly one dotted name; a second registration is a "
                    "shadow copy carrying its own module state",
                )

    def test_every_test_module_is_registered_under_its_bare_stem(self):
        # The half that catches a shadow copy of the test tree itself. A measured-tree walk
        # cannot observe `ReforceXY/tests`, so a dotted `from ReforceXY.tests.qa_support
        # import ...` would register a second module object for a file discovery already
        # registered under its stem — and every restore in qa_support would then clean a
        # module the code under test never touches.
        for name, path in _source_files_under(TESTS_ROOT):
            with self.subTest(module=name):
                self.assertEqual(
                    path.stem,
                    name,
                    "a test module registered under anything but its bare stem has a second "
                    "copy, so patching one leaves the other untouched",
                )

    def test_no_package_marker_under_the_test_directory(self):
        markers = sorted(
            path.relative_to(TESTS_ROOT).as_posix() for path in TESTS_ROOT.rglob("__init__.py")
        )
        self.assertEqual(
            [],
            markers,
            "a package marker makes the start directory a package, so the test modules lose "
            "top-level status and `from qa_support import ...` fails under discovery",
        )

    def test_only_the_meta_modules_bypass_the_shared_base_case(self):
        # `cls.__module__ == module.__name__` is the attribution rule that separates a class
        # defined here from one merely imported here: without it, `from unittest import
        # TestCase` or a re-exported base would be reported as a bypass.
        offenders = []
        for name, _ in _source_files_under(TESTS_ROOT):
            if name in _EXEMPT_MODULES:
                continue
            for attr, cls in vars(sys.modules[name]).items():
                if not isinstance(cls, type) or cls.__module__ != name:
                    continue
                if cls is unittest.TestCase or not issubclass(cls, unittest.TestCase):
                    continue
                if not issubclass(cls, QaTestCase):
                    offenders.append(f"{name}.{attr}")
        self.assertEqual(
            [],
            offenders,
            "every TestCase outside the two meta-modules must derive from QaTestCase, or the "
            "process globals it restores leak between tests",
        )

    def test_the_shuffle_is_installed_on_the_loader(self):
        # `sortTestMethodsUsing` is a staticmethod on the LOADER; a TestCase has no such
        # hook, and a plain function would bind as a method and raise TypeError at discovery.
        self.assertIs(_shuffle_cmp, unittest.TestLoader.sortTestMethodsUsing)

    def test_the_restore_puts_the_process_globals_back(self):
        cache = _PRODUCTION.ReforceXY._action_masks_cache
        # Dirty every global the way a test that trains a model would.
        cache[("dirty", 0.5)] = None
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)

        self.addCleanup(QaTestCase._restore_process_globals)
        QaTestCase._restore_process_globals()

        self.assertEqual(
            dict(_ACTION_MASKS_CACHE_AT_IMPORT),
            dict(cache),
            "_action_masks_cache was not restored",
        )
        self.assertEqual(_RANDOM_STATE_AT_IMPORT, random.getstate())
        self.assertEqual(
            _NUMPY_STATE_AT_IMPORT[1].tolist(),
            np.random.get_state()[1].tolist(),
        )
        if _TORCH_STATE_AT_IMPORT is not None:
            self.assertTrue(torch.equal(torch.random.get_rng_state(), _TORCH_STATE_AT_IMPORT))


if __name__ == "__main__":
    unittest.main()
