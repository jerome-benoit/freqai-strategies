#!/usr/bin/env python3
"""Integration tests for CLI interface and reproducibility."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import pytest

from ..test_base import RewardSpaceTestBase
from ..constants import SCENARIOS, SEEDS


pytestmark = pytest.mark.integration


class TestIntegration(RewardSpaceTestBase):
    """CLI + file output integration tests."""

    def test_cli_execution_produces_expected_files(self):
        """CLI produces expected files."""
        cmd = [
            "uv",
            "run",
            sys.executable,
            str(Path(__file__).parent.parent.parent / "reward_space_analysis.py"),
            "--num_samples",
            str(SCENARIOS.SAMPLE_SIZE_SMALL),
            "--seed",
            str(SEEDS.BASE),
            "--out_dir",
            str(self.output_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
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
        cmd1 = [
            "uv",
            "run",
            sys.executable,
            str(Path(__file__).parent.parent.parent / "reward_space_analysis.py"),
            "--num_samples",
            str(SCENARIOS.SAMPLE_SIZE_SMALL),
            "--seed",
            str(SEEDS.BASE),
            "--out_dir",
            str(self.output_path / "run1"),
        ]
        cmd2 = [
            "uv",
            "run",
            sys.executable,
            str(Path(__file__).parent.parent.parent / "reward_space_analysis.py"),
            "--num_samples",
            str(SCENARIOS.SAMPLE_SIZE_SMALL),
            "--seed",
            str(SEEDS.BASE),
            "--out_dir",
            str(self.output_path / "run2"),
        ]
        result1 = subprocess.run(
            cmd1, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        result2 = subprocess.run(
            cmd2, capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        self.assertEqual(result1.returncode, 0)
        self.assertEqual(result2.returncode, 0)
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
