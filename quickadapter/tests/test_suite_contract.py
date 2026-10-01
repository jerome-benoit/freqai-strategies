"""Guards the import identity and coverage denominator of the measured source."""

import importlib
import sys
from pathlib import Path

from qa_support import REPO_ROOT, QaTestCase

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
# A dotted strategies import is banned, so neither branch below may ever build one.
_FORBIDDEN_PREFIXES = ("quickadapter.user_data.strategies.", "quickadapter.tests.")

# Imported here, at module scope, so the parity contract holds on its own: unittest discovery
# imports every test module before running any test, so a lazy import inside a method would make
# the parity assertion unconditionally true.
for _name in PRODUCTION_NAMES:
    importlib.import_module(_name)


class SuiteContractTest(QaTestCase):
    def test_a_every_source_file_has_a_loaded_module(self):
        loaded = {
            Path(module.__file__).resolve()
            for module in list(sys.modules.values())
            if getattr(module, "__file__", None)
        }
        for src in _SOURCES:
            with self.subTest(module=src.name):
                self.assertIn(src.resolve(), loaded)

    def test_b_every_source_module_imports(self):
        for name in PRODUCTION_NAMES:
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))

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
        found = sorted(name for name in sys.modules if name.startswith(_FORBIDDEN_PREFIXES))
        self.assertEqual(
            [],
            found,
            "a shadow copy carries its own _LABEL_GENERATORS and _WARNED_CONFIG_DEPRECATIONS, so "
            "every reset in qa_support would clean a module the code under test never touches",
        )
