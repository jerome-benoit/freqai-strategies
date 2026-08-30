from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import check_basedpyright as checker


class ExactSnapshotTests(unittest.TestCase):
    def _diagnostic(self, **overrides: object) -> dict[str, object]:
        diagnostic: dict[str, object] = {
            "file": str(checker.REPO_ROOT / "README.md"),
            "severity": "error",
            "rule": "reportAssignmentType",
            "message": "Example diagnostic",
            "range": {
                "start": {"line": 10, "character": 2},
                "end": {"line": 10, "character": 8},
            },
        }
        diagnostic.update(overrides)
        return diagnostic

    def _child(self, diagnostics: list[object] | None = None) -> dict[str, object]:
        return {
            "version": "1.39.10",
            "generalDiagnostics": diagnostics if diagnostics is not None else [self._diagnostic()],
        }

    def test_snapshot_records_exact_child_version_and_location(self) -> None:
        snapshot = checker._snapshot_from_child(self._child())
        self.assertEqual(snapshot["schemaVersion"], 1)
        self.assertEqual(snapshot["basedpyrightVersion"], "1.39.10")
        diagnostic = snapshot["diagnostics"][0]
        self.assertEqual(diagnostic["file"], "README.md")
        self.assertEqual(diagnostic["startLine"], 10)

    def test_duplicate_diagnostics_are_preserved(self) -> None:
        diagnostic = self._diagnostic()
        snapshot = checker._snapshot_from_child(self._child([diagnostic, diagnostic.copy()]))
        self.assertEqual(len(snapshot["diagnostics"]), 2)

    def test_relocated_diagnostic_changes_canonical_snapshot(self) -> None:
        original = checker._snapshot_from_child(self._child())
        moved = self._diagnostic(
            range={"start": {"line": 12, "character": 2}, "end": {"line": 12, "character": 8}}
        )
        relocated = checker._snapshot_from_child(self._child([moved]))
        self.assertNotEqual(checker._canonical_bytes(original), checker._canonical_bytes(relocated))

    def test_strict_json_rejects_nonstandard_constant(self) -> None:
        with self.assertRaises(checker.QualityCheckError):
            checker._strict_json_loads('{"ignored": NaN}')

    def test_strict_json_rejects_duplicate_key(self) -> None:
        with self.assertRaises(checker.QualityCheckError):
            checker._strict_json_loads('{"version": "a", "version": "b"}')

    def test_child_rejects_unicode_surrogate(self) -> None:
        with self.assertRaises(checker.QualityCheckError):
            checker._snapshot_from_child(self._child([self._diagnostic(message="\ud800")]))

    def test_child_rejects_unknown_severity(self) -> None:
        with self.assertRaises(checker.QualityCheckError):
            checker._snapshot_from_child(self._child([self._diagnostic(severity="notice")]))

    def test_child_rejects_empty_message(self) -> None:
        with self.assertRaises(checker.QualityCheckError):
            checker._snapshot_from_child(self._child([self._diagnostic(message="")]))

    def test_child_rejects_inverted_range(self) -> None:
        inverted = {"start": {"line": 11, "character": 0}, "end": {"line": 10, "character": 9}}
        with self.assertRaises(checker.QualityCheckError):
            checker._snapshot_from_child(self._child([self._diagnostic(range=inverted)]))

    def test_snapshot_rejects_unknown_keys(self) -> None:
        snapshot = checker._snapshot_from_child(self._child())
        snapshot["unexpected"] = True
        with self.assertRaises(checker.QualityCheckError):
            checker._validate_snapshot(snapshot)

    def test_snapshot_rejects_missing_file(self) -> None:
        snapshot = checker._snapshot_from_child(self._child())
        snapshot["diagnostics"][0]["file"] = "missing.py"
        with self.assertRaises(checker.QualityCheckError):
            checker._validate_snapshot(snapshot)

    def test_snapshot_rejects_empty_message(self) -> None:
        snapshot = checker._snapshot_from_child(self._child())
        snapshot["diagnostics"][0]["message"] = ""
        with self.assertRaises(checker.QualityCheckError):
            checker._validate_snapshot(snapshot)

    def test_snapshot_rejects_inverted_range(self) -> None:
        snapshot = checker._snapshot_from_child(self._child())
        snapshot["diagnostics"][0]["endLine"] = 9
        with self.assertRaises(checker.QualityCheckError):
            checker._validate_snapshot(snapshot)

    def test_missing_snapshot_is_fail_closed(self) -> None:
        with self.assertRaises(checker.QualityCheckError):
            checker._read_snapshot(checker.REPO_ROOT / "missing-diagnostics.json")

    def test_environment_rejects_direct_host_invocation(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            self.assertRaises(checker.QualityCheckError),
        ):
            checker._validate_environment(checker.PROJECTS["quickadapter"])

    def test_environment_rejects_wrong_qa_project(self) -> None:
        with (
            mock.patch.dict(os.environ, {checker.QA_PROJECT_ENV: "reforcexy"}, clear=True),
            self.assertRaises(checker.QualityCheckError),
        ):
            checker._validate_environment(checker.PROJECTS["quickadapter"])

    def test_checker_accepts_diagnostic_exit_code(self) -> None:
        completed = subprocess.CompletedProcess(
            [], 1, stdout=json.dumps(self._child()).encode(), stderr=b""
        )
        with mock.patch.object(checker.subprocess, "run", return_value=completed):
            parsed = checker._run_basedpyright(
                checker.REPO_ROOT / "pyrightconfig.json", Path("/usr/local/bin/python")
            )
        self.assertIsInstance(parsed, dict)

    def test_checker_rejects_stderr(self) -> None:
        completed = subprocess.CompletedProcess(
            [], 1, stdout=json.dumps(self._child()).encode(), stderr=b"failure"
        )
        with (
            mock.patch.object(checker.subprocess, "run", return_value=completed),
            self.assertRaises(checker.QualityCheckError),
        ):
            checker._run_basedpyright(
                checker.REPO_ROOT / "pyrightconfig.json", Path("/usr/local/bin/python")
            )

    def test_checker_rejects_timeout(self) -> None:
        with (
            mock.patch.object(
                checker.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 1)
            ),
            self.assertRaises(checker.QualityCheckError),
        ):
            checker._run_basedpyright(
                checker.REPO_ROOT / "pyrightconfig.json", Path("/usr/local/bin/python")
            )

    def test_checker_rejects_non_diagnostic_exit_code(self) -> None:
        completed = subprocess.CompletedProcess([], 2, stdout=b"{}", stderr=b"")
        with (
            mock.patch.object(checker.subprocess, "run", return_value=completed),
            self.assertRaises(checker.QualityCheckError),
        ):
            checker._run_basedpyright(
                checker.REPO_ROOT / "pyrightconfig.json", Path("/usr/local/bin/python")
            )

    def test_checker_rejects_invalid_utf8(self) -> None:
        completed = subprocess.CompletedProcess([], 1, stdout=b"\xff", stderr=b"")
        with (
            mock.patch.object(checker.subprocess, "run", return_value=completed),
            self.assertRaises(checker.QualityCheckError),
        ):
            checker._run_basedpyright(
                checker.REPO_ROOT / "pyrightconfig.json", Path("/usr/local/bin/python")
            )

    def test_atomic_write_replaces_complete_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            baseline.write_bytes(b"old")
            checker._atomic_write(baseline, b"new\n")
            self.assertEqual(baseline.read_bytes(), b"new\n")

    def test_child_failure_cannot_start_write(self) -> None:
        with (
            mock.patch.object(
                checker, "_validate_environment", return_value=Path("/usr/local/bin/python")
            ),
            mock.patch.object(
                checker, "_run_basedpyright", side_effect=checker.QualityCheckError("invalid")
            ),
            mock.patch.object(checker, "_atomic_write") as atomic_write,
            self.assertRaises(checker.QualityCheckError),
        ):
            checker._check_project("quickadapter", write=True)
        atomic_write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
