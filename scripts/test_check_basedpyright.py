"""Unit tests for the snapshot comparison logic in check_basedpyright."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_basedpyright import (
    SCHEMA_VERSION,
    QualityCheckError,
    _comparison_key,
    _validate_snapshot,
)

SOURCE = "scripts/check_basedpyright.py"


def _snapshot(*, version, message="boom"):
    return {
        "schemaVersion": SCHEMA_VERSION,
        "basedpyrightVersion": version,
        "filesAnalyzed": 1,
        "sourceFiles": [SOURCE],
        "diagnostics": [{"file": SOURCE, "severity": "error", "message": message}],
    }


class ComparisonKeyTest(unittest.TestCase):
    def test_version_drift_only_matches(self):
        stored = _validate_snapshot(_snapshot(version="1.40.0"))
        current = _validate_snapshot(_snapshot(version="1.40.1"))
        self.assertEqual(_comparison_key(stored), _comparison_key(current))

    def test_diagnostic_change_mismatches(self):
        stored = _validate_snapshot(_snapshot(version="1.40.1"))
        current = _validate_snapshot(_snapshot(version="1.40.1", message="bam"))
        self.assertNotEqual(_comparison_key(stored), _comparison_key(current))

    def test_empty_version_rejected(self):
        with self.assertRaises(QualityCheckError):
            _validate_snapshot(_snapshot(version=""))

    def test_missing_key_rejected(self):
        snapshot = _snapshot(version="1.40.1")
        del snapshot["diagnostics"]
        with self.assertRaises(QualityCheckError):
            _validate_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
