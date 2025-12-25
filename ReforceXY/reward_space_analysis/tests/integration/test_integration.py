#!/usr/bin/env python3
"""Integration tests for CLI interface and reproducibility."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import pytest

from ..constants import SCENARIOS, SEEDS
from ..test_base import RewardSpaceTestBase

pytestmark = pytest.mark.integration

SCRIPT_PATH = Path(__file__).parent.parent.parent / "reward_space_analysis.py"
CWD = Path(__file__).parent.parent


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
    return subprocess.run(cmd, capture_output=True, text=True, cwd=CWD)


def _assert_cli_success(
    testcase: unittest.TestCase, result: subprocess.CompletedProcess[str]
) -> None:
    testcase.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")


class TestIntegration(RewardSpaceTestBase):
    """CLI + file output integration tests."""

    def test_cli_execution_produces_expected_files(self):
        """CLI produces expected files."""
        result = _run_cli(
            out_dir=self.output_path,
            args=[
                "--num_samples",
                str(SCENARIOS.SAMPLE_SIZE_SMALL),
                "--seed",
                str(SEEDS.BASE),
            ],
        )
        _assert_cli_success(self, result)

        expected_files = [
            "reward_samples.csv",
            "feature_importance.csv",
            "statistical_analysis.md",
            "manifest.json",
            "partial_dependence_trade_duration.csv",
            "partial_dependence_idle_duration.csv",
            "partial_dependence_pnl.csv",
        ]
        for filename in expected_files:
            file_path = self.output_path / filename
            self.assertTrue(file_path.exists(), f"Missing expected file: {filename}")

    def test_manifest_structure_and_reproducibility(self):
        """Manifest structure + reproducibility."""
        result1 = _run_cli(
            out_dir=self.output_path / "run1",
            args=[
                "--num_samples",
                str(SCENARIOS.SAMPLE_SIZE_SMALL),
                "--seed",
                str(SEEDS.BASE),
            ],
        )
        result2 = _run_cli(
            out_dir=self.output_path / "run2",
            args=[
                "--num_samples",
                str(SCENARIOS.SAMPLE_SIZE_SMALL),
                "--seed",
                str(SEEDS.BASE),
            ],
        )
        _assert_cli_success(self, result1)
        _assert_cli_success(self, result2)

        for run_dir in ["run1", "run2"]:
            with open(self.output_path / run_dir / "manifest.json", "r") as f:
                manifest = json.load(f)
            required_keys = {
                "generated_at",
                "num_samples",
                "seed",
                "params_hash",
                "reward_params",
                "simulation_params",
            }
            self.assertTrue(
                required_keys.issubset(manifest.keys()),
                f"Missing keys: {required_keys - set(manifest.keys())}",
            )
            self.assertIsInstance(manifest["reward_params"], dict)
            self.assertIsInstance(manifest["simulation_params"], dict)
            self.assertNotIn("top_features", manifest)
            self.assertNotIn("reward_param_overrides", manifest)
            self.assertNotIn("params", manifest)
            self.assertEqual(manifest["num_samples"], SCENARIOS.SAMPLE_SIZE_SMALL)
            self.assertEqual(manifest["seed"], SEEDS.BASE)

        with open(self.output_path / "run1" / "manifest.json", "r") as f:
            manifest1 = json.load(f)
        with open(self.output_path / "run2" / "manifest.json", "r") as f:
            manifest2 = json.load(f)
        self.assertEqual(
            manifest1["params_hash"],
            manifest2["params_hash"],
            "Same seed should produce same parameters hash",
        )


if __name__ == "__main__":
    unittest.main()
