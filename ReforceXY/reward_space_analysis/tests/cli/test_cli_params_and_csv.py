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
            all(float(v).is_integer() for v in values),
            "Non-integer values detected in 'action' column",
        )
        allowed = {int(action.value) for action in Actions}
        self.assertTrue({int(v) for v in values}.issubset(allowed))


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
        """Strict diagnostics accepts constant distributions without fabricated statistics."""
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
        # Constant distributions remain valid in strict mode.
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
        with manifest_path.open() as f:
            manifest = json.load(f)
        self.assertIn("reward_params", manifest)
        self.assertIn("simulation_params", manifest)
        rp = manifest["reward_params"]
        self.assertIn("max_trade_duration_candles", rp)
        self.assertEqual(
            int(rp["max_trade_duration_candles"]), SCENARIOS.CLI_MAX_TRADE_DURATION_PARAMS
        )

    def test_missing_real_episodes_fails_before_artifacts(self):
        """An explicitly requested but missing episodes file fails the run with no artifacts."""
        out_dir = self.output_path / "missing_real"
        missing = self.output_path / "no_such_episodes.pkl"
        result = _run_cli(
            out_dir=out_dir,
            args=["--num_samples", "50", "--real_episodes", str(missing)],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(missing), result.stderr + result.stdout)
        self.assertFalse(out_dir.exists())

    def test_invalid_real_episodes_pickle_fails_before_artifacts(self):
        """A corrupt episodes pickle fails the run with a diagnosed path and no artifacts."""
        out_dir = self.output_path / "invalid_real"
        invalid = self.output_path / "corrupt.pkl"
        invalid.write_bytes(b"not a pickle")
        result = _run_cli(
            out_dir=out_dir,
            args=["--num_samples", "50", "--real_episodes", str(invalid)],
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(invalid), result.stderr + result.stdout)
        self.assertFalse(out_dir.exists())

    def test_valid_real_episodes_produce_real_metrics(self):
        """A valid episodes pickle loads before simulation and enables real metrics."""
        out_dir = self.output_path / "valid_real"
        import pickle

        episodes = [
            {
                "transitions": [
                    {
                        "pnl": 0.01 * (1 if index % 2 else -1),
                        "trade_duration": 2 + index % 4,
                        "idle_duration": index % 5,
                        "position": 1.0 if index % 2 else 0.5,
                        "action": 0,
                        "reward": 0.5 - 0.05 * index,
                    }
                    for index in range(20)
                ]
            }
        ]
        episodes_path = self.output_path / "episodes.pkl"
        with episodes_path.open("wb") as fh:
            pickle.dump(episodes, fh)
        result = _run_cli(
            out_dir=out_dir,
            args=["--num_samples", "50", "--real_episodes", str(episodes_path)],
        )
        _assert_cli_success(self, result)
        report = (out_dir / "statistical_analysis.md").read_text(encoding="utf-8")
        self.assertNotIn("Not performed (no real episodes provided)", report)

    def test_params_override_flags_and_manifest_reflects_effective(self):
        """--params beats explicit flags; manifest effective values follow resolution."""
        out_dir = self.output_path / "params_beat_flags"
        result = _run_cli(
            out_dir=out_dir,
            args=[
                "--num_samples",
                str(SCENARIOS.CLI_NUM_SAMPLES_FAST),
                "--profit_aim",
                "0.05",
                "--base_factor",
                "150.0",
                "--params",
                "profit_aim=0.02",
                "rr=1.5",
            ],
        )
        _assert_cli_success(self, result)
        with (out_dir / "manifest.json").open() as f:
            manifest = json.load(f)
        effective = manifest["effective"]
        self.assertEqual(effective["profit_aim"], 0.02)
        self.assertEqual(effective["risk_reward_ratio"], 1.5)
        self.assertEqual(effective["base_factor"], 150.0)
        self.assertAlmostEqual(manifest["pnl_target"], 0.03)

    def test_simulation_only_params_rejected_before_artifacts(self):
        """Simulation-only keys fail the run before any artifact is written."""
        for key, value in (("num_samples", "1"), ("unrealized_pnl", "true")):
            out_dir = self.output_path / f"rejected_{key}"
            result = _run_cli(out_dir=out_dir, args=["--params", f"{key}={value}"])
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out_dir.exists())

    def test_unrealized_pnl_flag_changes_simulated_trajectory(self):
        common_args = [
            "--num_samples",
            str(SCENARIOS.CLI_NUM_SAMPLES_STANDARD),
            "--seed",
            str(SEEDS.BASE),
            "--skip_feature_analysis",
            "--skip_partial_dependence",
        ]
        default_dir = self.output_path / "unrealized_default"
        enabled_dir = self.output_path / "unrealized_enabled"
        default_result = _run_cli(out_dir=default_dir, args=common_args)
        enabled_result = _run_cli(out_dir=enabled_dir, args=[*common_args, "--unrealized_pnl"])
        _assert_cli_success(self, default_result)
        _assert_cli_success(self, enabled_result)
        default_pnl = pd.read_csv(default_dir / "reward_samples.csv")["pnl"]
        enabled_pnl = pd.read_csv(enabled_dir / "reward_samples.csv")["pnl"]
        self.assertFalse(default_pnl.equals(enabled_pnl))

    def test_inferential_options_rejected_for_dependent_trajectory(self):
        for option, value in (
            ("--bootstrap_resamples", "200"),
            ("--pvalue_adjust", "benjamini_hochberg"),
        ):
            out_dir = self.output_path / option.removeprefix("--")
            result = _run_cli(out_dir=out_dir, args=[option, value])
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out_dir.exists())

    def test_unknown_params_rejected_before_artifacts(self):
        """Unknown keys fail the run before any artifact is written."""
        out_dir = self.output_path / "rejected_unknown"
        result = _run_cli(out_dir=out_dir, args=["--params", "win_reward_factr=2.0"])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(out_dir.exists())

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
        with manifest_path.open() as f:
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
