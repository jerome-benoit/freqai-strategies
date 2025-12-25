#!/usr/bin/env python3
"""CLI-level tests: CSV encoding and parameter propagation."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd
import pytest

from reward_space_analysis import Actions

from ..constants import SCENARIOS, SEEDS, TOLERANCE
from ..test_base import RewardSpaceTestBase

# Pytest marker for taxonomy classification
pytestmark = pytest.mark.cli

SCRIPT_PATH = Path(__file__).parent.parent.parent / "reward_space_analysis.py"


def _run_cli(*, out_dir: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [
        "uv",
        "run",
        sys.executable,
        str(SCRIPT_PATH),
        "--out_dir",
        str(out_dir),
        *args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)


def _assert_cli_success(
    testcase: unittest.TestCase, result: subprocess.CompletedProcess[str]
) -> None:
    testcase.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")


class TestCsvEncoding(RewardSpaceTestBase):
    """Validate CSV output encoding invariants."""

    def test_action_column_integer_in_csv(self):
        """Ensure 'action' column in reward_samples.csv is encoded as integers."""
        out_dir = self.output_path / "csv_int_check"
        result = _run_cli(
            out_dir=out_dir,
            args=[
                "--num_samples",
                str(SCENARIOS.CLI_NUM_SAMPLES_STANDARD),
                "--seed",
                str(SEEDS.BASE),
            ],
        )
        _assert_cli_success(self, result)
        csv_path = out_dir / "reward_samples.csv"
        self.assertTrue(csv_path.exists(), "Missing reward_samples.csv")
        df = pd.read_csv(csv_path)
        self.assertIn("action", df.columns)
        values = df["action"].tolist()
        self.assertTrue(
            all((float(v).is_integer() for v in values)),
            "Non-integer values detected in 'action' column",
        )
        allowed = {int(action.value) for action in Actions}
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
        result = _run_cli(
            out_dir=out_dir,
            args=[
                "--num_samples",
                str(SCENARIOS.CLI_NUM_SAMPLES_STANDARD),
                "--seed",
                str(SEEDS.BASE),
                "--skip_feature_analysis",
            ],
        )
        _assert_cli_success(self, result)
        report_path = out_dir / "statistical_analysis.md"
        self.assertTrue(report_path.exists(), "Missing statistical_analysis.md")
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("Feature Importance - (skipped)", content)
        fi_path = out_dir / "feature_importance.csv"
        self.assertFalse(fi_path.exists(), "feature_importance.csv should be absent when skipped")

    def test_manifest_params_hash_generation(self):
        """Ensure params_hash appears when non-default simulation params differ (risk_reward_ratio altered)."""
        out_dir = self.output_path / "manifest_hash"
        result = _run_cli(
            out_dir=out_dir,
            args=[
                "--num_samples",
                str(SCENARIOS.CLI_NUM_SAMPLES_HASH),
                "--seed",
                str(SEEDS.BASE),
                "--risk_reward_ratio",
                str(SCENARIOS.CLI_RISK_REWARD_RATIO_NON_DEFAULT),
            ],
        )
        _assert_cli_success(self, result)
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
        result = _run_cli(
            out_dir=out_dir,
            args=[
                "--num_samples",
                str(SCENARIOS.CLI_NUM_SAMPLES_REPORT),
                "--seed",
                str(SEEDS.BASE),
            ],
        )
        _assert_cli_success(self, result)
        report_path = out_dir / "statistical_analysis.md"
        self.assertTrue(report_path.exists(), "Missing statistical_analysis.md")
        content = report_path.read_text(encoding="utf-8")
        # Section numbering includes PBRS invariance line 7
        self.assertIn("PBRS Invariance", content)

    def test_strict_diagnostics_constant_distribution_succeeds(self):
        """Run with --strict_diagnostics and low num_samples; expect success, exercising assertion branches before graceful fallback paths."""
        out_dir = self.output_path / "strict_diagnostics"
        result = _run_cli(
            out_dir=out_dir,
            args=[
                "--num_samples",
                str(SCENARIOS.CLI_NUM_SAMPLES_FAST),
                "--seed",
                str(SEEDS.BASE),
                "--strict_diagnostics",
            ],
        )
        # Should not raise; if constant distributions occur they should assert before graceful fallback paths, exercising assertion branches.
        self.assertEqual(
            result.returncode,
            0,
            f"CLI failed (expected pass): {result.stderr}\nSTDOUT:\n{result.stdout[:500]}",
        )
        report_path = out_dir / "statistical_analysis.md"
        self.assertTrue(report_path.exists(), "Missing statistical_analysis.md")

    def test_max_trade_duration_candles_propagation_params(self):
        """--params max_trade_duration_candles=X propagates to manifest and simulation params."""
        out_dir = self.output_path / "mtd_params"
        result = _run_cli(
            out_dir=out_dir,
            args=[
                "--num_samples",
                str(SCENARIOS.CLI_NUM_SAMPLES_FAST),
                "--seed",
                str(SEEDS.BASE),
                "--params",
                f"max_trade_duration_candles={SCENARIOS.CLI_MAX_TRADE_DURATION_PARAMS}",
            ],
        )
        _assert_cli_success(self, result)
        manifest_path = out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "Missing manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.assertIn("reward_params", manifest)
        self.assertIn("simulation_params", manifest)
        rp = manifest["reward_params"]
        self.assertIn("max_trade_duration_candles", rp)
        self.assertEqual(
            int(rp["max_trade_duration_candles"]), SCENARIOS.CLI_MAX_TRADE_DURATION_PARAMS
        )

    def test_max_trade_duration_candles_propagation_flag(self):
        """Dynamic flag --max_trade_duration_candles X propagates identically."""
        out_dir = self.output_path / "mtd_flag"
        result = _run_cli(
            out_dir=out_dir,
            args=[
                "--num_samples",
                str(SCENARIOS.CLI_NUM_SAMPLES_FAST),
                "--seed",
                str(SEEDS.BASE),
                "--max_trade_duration_candles",
                str(SCENARIOS.CLI_MAX_TRADE_DURATION_FLAG),
            ],
        )
        _assert_cli_success(self, result)
        manifest_path = out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "Missing manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        self.assertIn("reward_params", manifest)
        self.assertIn("simulation_params", manifest)
        rp = manifest["reward_params"]
        self.assertIn("max_trade_duration_candles", rp)
        self.assertEqual(
            int(rp["max_trade_duration_candles"]), SCENARIOS.CLI_MAX_TRADE_DURATION_FLAG
        )

    # Owns invariant: cli-pbrs-csv-columns-121
    def test_csv_contains_pbrs_columns_when_shaping_present(self):
        """Verify reward_samples.csv includes PBRS columns when shaping is enabled.

        Verifies:
        - reward_base, reward_pbrs_delta, reward_invariance_correction columns exist
        - All values are finite (no NaN/inf)
        - Column values align mathematically
        """
        out_dir = self.output_path / "pbrs_csv_columns"
        result = _run_cli(
            out_dir=out_dir,
            args=[
                "--num_samples",
                str(SCENARIOS.CLI_NUM_SAMPLES_HASH),
                "--seed",
                str(SEEDS.BASE),
                # Enable PBRS shaping explicitly
                "--params",
                "exit_potential_mode=canonical",
            ],
        )
        _assert_cli_success(self, result)

        csv_path = out_dir / "reward_samples.csv"
        self.assertTrue(csv_path.exists(), "Missing reward_samples.csv")

        df = pd.read_csv(csv_path)

        # Verify PBRS columns exist
        required_cols = ["reward_base", "reward_pbrs_delta", "reward_invariance_correction"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column: {col}")

        # Verify all values are finite
        for col in required_cols:
            self.assertFalse(df[col].isna().any(), f"Column {col} contains NaN values")
            for i, value in enumerate(df[col].to_numpy()):
                self.assertFinite(float(value), name=f"{col}[{i}]")

        # Verify mathematical alignment (CSV-level invariants)
        # By construction in `calculate_reward()`: reward_shaping = pbrs_delta + invariance_correction
        shaping_residual = (
            df["reward_shaping"] - (df["reward_pbrs_delta"] + df["reward_invariance_correction"])
        ).abs()
        self.assertLessEqual(
            float(shaping_residual.max()),
            TOLERANCE.GENERIC_EQ,
            "Expected reward_shaping == reward_pbrs_delta + reward_invariance_correction",
        )

        # Total reward should decompose into base + shaping + additives
        reward_residual = (
            df["reward"]
            - (
                df["reward_base"]
                + df["reward_shaping"]
                + df["reward_entry_additive"]
                + df["reward_exit_additive"]
            )
        ).abs()
        self.assertLessEqual(
            float(reward_residual.max()),
            TOLERANCE.GENERIC_EQ,
            "Expected reward == reward_base + reward_shaping + additives",
        )


if __name__ == "__main__":
    unittest.main()
