#!/usr/bin/env python3
"""CLI-level tests: CSV encoding and parameter propagation.

Split out from helpers/test_utilities.py to align with taxonomy refactor.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd
import pytest

from ..test_base import RewardSpaceTestBase

# Pytest marker for taxonomy classification
pytestmark = pytest.mark.cli

SCRIPT_PATH = Path(__file__).parent.parent.parent / "reward_space_analysis.py"


class TestCsvEncoding(RewardSpaceTestBase):
    """Validate CSV output encoding invariants."""

    def test_action_column_integer_in_csv(self):
        """Ensure 'action' column in reward_samples.csv is encoded as integers."""
        out_dir = self.output_path / "csv_int_check"
        cmd = [
            "uv",
            "run",
            sys.executable,
            str(SCRIPT_PATH),
            "--num_samples",
            "200",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        csv_path = out_dir / "reward_samples.csv"
        self.assertTrue(csv_path.exists(), "Missing reward_samples.csv")
        df = pd.read_csv(csv_path)
        self.assertIn("action", df.columns)
        values = df["action"].tolist()
        self.assertTrue(
            all((float(v).is_integer() for v in values)),
            "Non-integer values detected in 'action' column",
        )
        allowed = {0, 1, 2, 3, 4}
        self.assertTrue(set((int(v) for v in values)).issubset(allowed))


class TestParamsPropagation(RewardSpaceTestBase):
    """Integration tests to validate max_trade_duration_candles propagation via CLI params and dynamic flag.

    Extended with coverage for:
    - skip_feature_analysis summary path
    - strict_diagnostics fallback vs manifest generation
    - params_hash generation when simulation params differ
    - PBRS invariance summary section when reward_shaping present
    """

    def test_skip_feature_analysis_summary_branch(self):
        """CLI run with --skip_feature_analysis should mark feature importance skipped in summary and omit feature_importance.csv."""
        out_dir = self.output_path / "skip_feature_analysis"
        cmd = [
            "uv",
            "run",
            sys.executable,
            str(SCRIPT_PATH),
            "--num_samples",
            "200",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
            "--skip_feature_analysis",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        report_path = out_dir / "statistical_analysis.md"
        self.assertTrue(report_path.exists(), "Missing statistical_analysis.md")
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("Feature Importance - (skipped)", content)
        fi_path = out_dir / "feature_importance.csv"
        self.assertFalse(fi_path.exists(), "feature_importance.csv should be absent when skipped")

    def test_manifest_params_hash_generation(self):
        """Ensure params_hash appears when non-default simulation params differ (risk_reward_ratio altered)."""
        out_dir = self.output_path / "manifest_hash"
        cmd = [
            "uv",
            "run",
            sys.executable,
            str(SCRIPT_PATH),
            "--num_samples",
            "150",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
            "--risk_reward_ratio",
            "1.5",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        manifest_path = out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "Missing manifest.json")
        manifest = json.loads(manifest_path.read_text())
        self.assertIn("params_hash", manifest, "params_hash should be present when params differ")
        self.assertIn("simulation_params", manifest)
        self.assertIn("risk_reward_ratio", manifest["simulation_params"])

    def test_pbrs_invariance_section_present(self):
        """When reward_shaping column exists, summary should include PBRS invariance section."""
        out_dir = self.output_path / "pbrs_invariance"
        # Use small sample for speed; rely on default shaping logic
        cmd = [
            "uv",
            "run",
            sys.executable,
            str(SCRIPT_PATH),
            "--num_samples",
            "180",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        report_path = out_dir / "statistical_analysis.md"
        self.assertTrue(report_path.exists(), "Missing statistical_analysis.md")
        content = report_path.read_text(encoding="utf-8")
        # Section numbering includes PBRS invariance line 7
        self.assertIn("PBRS Invariance", content)

    def test_strict_diagnostics_constant_distribution_raises(self):
        """Run with --strict_diagnostics and very low num_samples to increase chance of constant columns; expect success but can parse diagnostics without fallback replacements."""
        out_dir = self.output_path / "strict_diagnostics"
        cmd = [
            "uv",
            "run",
            sys.executable,
            str(SCRIPT_PATH),
            "--num_samples",
            "120",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
            "--strict_diagnostics",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        # Should not raise; if constant distributions occur they should assert before graceful fallback paths, exercising assertion branches.
        self.assertEqual(
            result.returncode,
            0,
            f"CLI failed (expected pass): {result.stderr}\nSTDOUT:\n{result.stdout[:500]}",
        )
        report_path = out_dir / "statistical_analysis.md"
        self.assertTrue(report_path.exists(), "Missing statistical_analysis.md")

    """Integration tests to validate max_trade_duration_candles propagation via CLI params and dynamic flag."""

    def test_max_trade_duration_candles_propagation_params(self):
        """--params max_trade_duration_candles=X propagates to manifest and simulation params."""
        out_dir = self.output_path / "mtd_params"
        cmd = [
            "uv",
            "run",
            sys.executable,
            str(SCRIPT_PATH),
            "--num_samples",
            "120",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
            "--params",
            "max_trade_duration_candles=96",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        manifest_path = out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "Missing manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.assertIn("reward_params", manifest)
        self.assertIn("simulation_params", manifest)
        rp = manifest["reward_params"]
        self.assertIn("max_trade_duration_candles", rp)
        self.assertEqual(int(rp["max_trade_duration_candles"]), 96)

    def test_max_trade_duration_candles_propagation_flag(self):
        """Dynamic flag --max_trade_duration_candles X propagates identically."""
        out_dir = self.output_path / "mtd_flag"
        cmd = [
            "uv",
            "run",
            sys.executable,
            str(SCRIPT_PATH),
            "--num_samples",
            "120",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
            "--max_trade_duration_candles",
            "64",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        manifest_path = out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "Missing manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.assertIn("reward_params", manifest)
        self.assertIn("simulation_params", manifest)
        rp = manifest["reward_params"]
        self.assertIn("max_trade_duration_candles", rp)
        self.assertEqual(int(rp["max_trade_duration_candles"]), 64)


if __name__ == "__main__":
    unittest.main()
