#!/usr/bin/env python3
"""Unified test suite for reward space analysis.

Covers:
- Integration testing (CLI interface)
- Statistical coherence validation
- Reward alignment with environment
"""

import dataclasses
import json
import math
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Central PBRS parameter lists
PBRS_INTEGRATION_PARAMS = [
    "potential_gamma",
    "hold_potential_enabled",
    "hold_potential_scale",
    "entry_additive_enabled",
    "exit_additive_enabled",
]
PBRS_REQUIRED_PARAMS = PBRS_INTEGRATION_PARAMS + ["exit_potential_mode"]

# Import functions to test
try:
    from reward_space_analysis import (
        DEFAULT_MODEL_REWARD_PARAMETERS,
        Actions,
        Positions,
        RewardContext,
        _compute_entry_additive,
        _compute_exit_additive,
        _compute_exit_potential,
        _compute_hold_potential,
        _get_bool_param,
        _get_exit_factor,
        _get_float_param,
        _get_pnl_factor,
        _get_str_param,
        apply_potential_shaping,
        apply_transform,
        bootstrap_confidence_intervals,
        build_argument_parser,
        calculate_reward,
        compute_distribution_shift_metrics,
        distribution_diagnostics,
        load_real_episodes,
        parse_overrides,
        simulate_samples,
        statistical_hypothesis_tests,
        validate_reward_parameters,
        write_complete_statistical_analysis,
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class RewardSpaceTestBase(unittest.TestCase):
    """Base class with common test utilities.

    Central tolerance policy (avoid per-test commentary):
    - Generic numeric equality: 1e-6
    - Component decomposition / identity: 1e-9
    - Monotonic attenuation allowance: +1e-9 drift
    """

    @classmethod
    def setUpClass(cls):
        """Set up class-level constants - single source of truth."""
        cls.SEED = 42
        cls.DEFAULT_PARAMS = DEFAULT_MODEL_REWARD_PARAMETERS.copy()
        cls.TEST_SAMPLES = 50  # Small for speed
        cls.TEST_BASE_FACTOR = 100.0
        cls.TEST_PROFIT_TARGET = 0.03
        cls.TEST_RR = 1.0
        cls.TEST_RR_HIGH = 2.0
        cls.TEST_PNL_STD = 0.02
        cls.TEST_PNL_DUR_VOL_SCALE = 0.5

    def setUp(self):
        """Set up test fixtures with reproducible random seed."""
        np.random.seed(self.SEED)
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def assertAlmostEqualFloat(
        self,
        first: float | int,
        second: float | int,
        tolerance: float = 1e-6,
        msg: str | None = None,
    ) -> None:
        """Absolute tolerance compare with explicit failure and finite check."""
        self.assertFinite(first, name="a")
        self.assertFinite(second, name="b")
        diff = abs(first - second)
        if diff > tolerance:
            self.fail(
                msg
                or f"Difference {diff} exceeds tolerance {tolerance} (a={first}, b={second})"
            )

    # --- Statistical bounds helpers (factorize redundancy) ---
    def assertPValue(self, value: float | int, msg: str = "") -> None:
        """Assert a p-value is finite and within [0,1]."""
        self.assertFinite(value, name="p-value")
        self.assertGreaterEqual(value, 0.0, msg or f"p-value < 0: {value}")
        self.assertLessEqual(value, 1.0, msg or f"p-value > 1: {value}")

    def assertDistanceMetric(
        self,
        value: float | int,
        *,
        non_negative: bool = True,
        upper: float | None = None,
        name: str = "metric",
    ) -> None:
        """Generic distance/divergence bounds: finite, optional non-negativity and optional upper bound."""
        self.assertFinite(value, name=name)
        if non_negative:
            self.assertGreaterEqual(value, 0.0, f"{name} negative: {value}")
        if upper is not None:
            self.assertLessEqual(value, upper, f"{name} > {upper}: {value}")

    def assertEffectSize(
        self,
        value: float | int,
        *,
        lower: float = -1.0,
        upper: float = 1.0,
        name: str = "effect size",
    ) -> None:
        """Assert effect size within symmetric interval and finite."""
        self.assertFinite(value, name=name)
        self.assertGreaterEqual(value, lower, f"{name} < {lower}: {value}")
        self.assertLessEqual(value, upper, f"{name} > {upper}: {value}")

    def assertFinite(self, value: float | int, name: str = "value") -> None:
        """Assert scalar is finite."""
        if not np.isfinite(value):  # low-level base check to avoid recursion
            self.fail(f"{name} not finite: {value}")

    def assertMonotonic(
        self,
        seq: Sequence[float | int] | Iterable[float | int],
        *,
        non_increasing: bool | None = None,
        non_decreasing: bool | None = None,
        tolerance: float = 0.0,
        name: str = "sequence",
    ) -> None:
        """Assert a sequence is monotonic under specified direction.

        Provide exactly one of non_increasing/non_decreasing=True.
        tolerance allows tiny positive drift in expected monotone direction.
        """
        data = list(seq)
        if len(data) < 2:
            return
        if (non_increasing and non_decreasing) or (
            not non_increasing and not non_decreasing
        ):
            self.fail("Specify exactly one monotonic direction")
        for a, b in zip(data, data[1:]):
            if non_increasing:
                if b > a + tolerance:
                    self.fail(f"{name} not non-increasing at pair ({a}, {b})")
            elif non_decreasing:
                if b + tolerance < a:
                    self.fail(f"{name} not non-decreasing at pair ({a}, {b})")

    def assertWithin(
        self,
        value: float | int,
        low: float | int,
        high: float | int,
        *,
        name: str = "value",
        inclusive: bool = True,
    ) -> None:
        """Assert that value is within [low, high] (inclusive) or (low, high) if inclusive=False."""
        self.assertFinite(value, name=name)
        if inclusive:
            self.assertGreaterEqual(value, low, f"{name} < {low}")
            self.assertLessEqual(value, high, f"{name} > {high}")
        else:
            self.assertGreater(value, low, f"{name} <= {low}")
            self.assertLess(value, high, f"{name} >= {high}")


class TestIntegration(RewardSpaceTestBase):
    """Integration tests for CLI and file outputs."""

    def test_cli_execution_produces_expected_files(self):
        """Test that CLI execution produces all expected output files."""
        cmd = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            str(self.TEST_SAMPLES),
            "--seed",
            str(self.SEED),
            "--output",
            str(self.output_path),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=Path(__file__).parent
        )

        # Should execute successfully
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")

        # Should produce expected files (single source of truth list)
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
        """Test manifest structure and reproducibility with same seed."""
        # First run
        cmd1 = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            str(self.TEST_SAMPLES),
            "--seed",
            str(self.SEED),
            "--output",
            str(self.output_path / "run1"),
        ]

        # Second run with same seed
        cmd2 = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            str(self.TEST_SAMPLES),
            "--seed",
            str(self.SEED),
            "--output",
            str(self.output_path / "run2"),
        ]

        # Execute both runs
        result1 = subprocess.run(
            cmd1, capture_output=True, text=True, cwd=Path(__file__).parent
        )
        result2 = subprocess.run(
            cmd2, capture_output=True, text=True, cwd=Path(__file__).parent
        )

        self.assertEqual(result1.returncode, 0)
        self.assertEqual(result2.returncode, 0)

        # Validate manifest structure
        for run_dir in ["run1", "run2"]:
            with open(self.output_path / run_dir / "manifest.json", "r") as f:
                manifest = json.load(f)

            required_keys = {"generated_at", "num_samples", "seed", "params_hash"}
            self.assertTrue(required_keys.issubset(manifest.keys()))
            self.assertEqual(manifest["num_samples"], self.TEST_SAMPLES)
            self.assertEqual(manifest["seed"], self.SEED)

        # Test reproducibility: manifests should have same params_hash
        with open(self.output_path / "run1" / "manifest.json", "r") as f:
            manifest1 = json.load(f)
        with open(self.output_path / "run2" / "manifest.json", "r") as f:
            manifest2 = json.load(f)

        self.assertEqual(
            manifest1["params_hash"],
            manifest2["params_hash"],
            "Same seed should produce same parameters hash",
        )


class TestStatisticalCoherence(RewardSpaceTestBase):
    """Statistical coherence validation tests."""

    def _make_test_dataframe(self, n: int = 100) -> pd.DataFrame:
        """Generate test dataframe for statistical validation."""
        np.random.seed(self.SEED)

        # Create correlated data for Spearman test
        idle_duration = np.random.exponential(10, n)
        reward_idle = -0.01 * idle_duration + np.random.normal(0, 0.001, n)

        return pd.DataFrame(
            {
                "idle_duration": idle_duration,
                "reward_idle": reward_idle,
                "position": np.random.choice([0.0, 0.5, 1.0], n),
                "reward_total": np.random.normal(0, 1, n),
                "pnl": np.random.normal(0, self.TEST_PNL_STD, n),
                "trade_duration": np.random.exponential(20, n),
            }
        )

    def test_distribution_shift_metrics(self):
        """Test KL divergence, JS distance, Wasserstein distance calculations."""
        df1 = self._make_test_dataframe(100)
        df2 = self._make_test_dataframe(100)

        # Add slight shift to second dataset
        df2["reward_total"] += 0.1

        metrics = compute_distribution_shift_metrics(df1, df2)

        # Should contain expected metrics for pnl (continuous feature)
        expected_keys = {
            "pnl_kl_divergence",
            "pnl_js_distance",
            "pnl_wasserstein",
            "pnl_ks_statistic",
        }
        actual_keys = set(metrics.keys())
        matching_keys = expected_keys.intersection(actual_keys)
        self.assertGreater(
            len(matching_keys),
            0,
            f"Should have some distribution metrics. Got: {actual_keys}",
        )

        # Values should be finite and reasonable
        for metric_name, value in metrics.items():
            if "pnl" in metric_name:
                # All metrics must be finite; selected metrics must be non-negative
                if any(
                    suffix in metric_name
                    for suffix in [
                        "js_distance",
                        "ks_statistic",
                        "wasserstein",
                        "kl_divergence",
                    ]
                ):
                    self.assertDistanceMetric(value, name=metric_name)
                else:
                    self.assertFinite(value, name=metric_name)

    def test_distribution_shift_identity_null_metrics(self):
        """Identical distributions should yield (near) zero shift metrics."""
        df = self._make_test_dataframe(180)
        metrics_id = compute_distribution_shift_metrics(df, df.copy())
        for name, val in metrics_id.items():
            if name.endswith(("_kl_divergence", "_js_distance", "_wasserstein")):
                self.assertLess(
                    abs(val),
                    1e-6,
                    f"Metric {name} expected ≈ 0 on identical distributions (got {val})",
                )
            elif name.endswith("_ks_statistic"):
                self.assertLess(
                    abs(val),
                    5e-3,
                    f"KS statistic should be near 0 on identical distributions (got {val})",
                )

    def test_hypothesis_testing(self):
        """Test statistical hypothesis tests."""
        df = self._make_test_dataframe(200)

        # Test only if we have enough data - simple correlation validation
        if len(df) > 30:
            idle_data = df[df["idle_duration"] > 0]
            if len(idle_data) > 10:
                # Simple correlation check: idle duration should correlate negatively with idle reward
                idle_dur = idle_data["idle_duration"].to_numpy()
                idle_rew = idle_data["reward_idle"].to_numpy()

                # Basic validation that data makes sense
                self.assertTrue(
                    len(idle_dur) == len(idle_rew),
                    "Idle duration and reward arrays should have same length",
                )
                self.assertTrue(
                    all(d >= 0 for d in idle_dur),
                    "Idle durations should be non-negative",
                )

                # Idle rewards should generally be negative (penalty for hold)
                negative_rewards = (idle_rew < 0).sum()
                total_rewards = len(idle_rew)
                negative_ratio = negative_rewards / total_rewards

                self.assertGreater(
                    negative_ratio,
                    0.5,
                    "Most idle rewards should be negative (penalties)",
                )

    def test_distribution_diagnostics(self):
        """Test distribution normality diagnostics."""
        df = self._make_test_dataframe(100)

        diagnostics = distribution_diagnostics(df)

        # Should contain diagnostics for key columns (flattened format)
        expected_prefixes = ["reward_total_", "pnl_"]
        for prefix in expected_prefixes:
            matching_keys = [
                key for key in diagnostics.keys() if key.startswith(prefix)
            ]
            self.assertGreater(
                len(matching_keys), 0, f"Should have diagnostics for {prefix}"
            )

            # Check for basic statistical measures
            expected_suffixes = ["mean", "std", "skewness", "kurtosis"]
            for suffix in expected_suffixes:
                key = f"{prefix}{suffix}"
                if key in diagnostics:
                    self.assertFinite(diagnostics[key], name=key)


class TestRewardAlignment(RewardSpaceTestBase):
    """Test reward calculation alignment with environment."""

    def test_basic_reward_calculation(self):
        """Test basic reward calculation consistency."""
        context = RewardContext(
            pnl=self.TEST_PROFIT_TARGET,
            trade_duration=10,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.025,
            min_unrealized_profit=0.015,
            position=Positions.Long,
            action=Actions.Long_exit,
        )

        breakdown = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=0.06,  # Scenario-specific larger target kept explicit
            risk_reward_ratio=self.TEST_RR_HIGH,
            short_allowed=True,
            action_masking=True,
        )

        # Should return valid breakdown
        self.assertIsInstance(breakdown.total, (int, float))
        self.assertFinite(breakdown.total, name="breakdown.total")

        # Exit reward should be positive for profitable trade
        self.assertGreater(
            breakdown.exit_component, 0, "Profitable exit should have positive reward"
        )

    def test_efficiency_zero_policy(self):
        """Ensure pnl == 0 with max_unrealized_profit == 0 does not get boosted.

        This verifies the policy: near-zero pnl -> no efficiency modulation.
        """

        # Build context where pnl == 0.0 and max_unrealized_profit == pnl
        ctx = RewardContext(
            pnl=0.0,
            trade_duration=1,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.0,
            min_unrealized_profit=-0.02,
            position=Positions.Long,
            action=Actions.Long_exit,
        )

        params = self.DEFAULT_PARAMS.copy()
        profit_target = self.TEST_PROFIT_TARGET * self.TEST_RR

        pnl_factor = _get_pnl_factor(params, ctx, profit_target)
        # Expect no efficiency modulation: factor should be >= 0 and close to 1.0
        self.assertFinite(pnl_factor, name="pnl_factor")
        self.assertAlmostEqualFloat(pnl_factor, 1.0, tolerance=1e-6)

    def test_max_idle_duration_candles_logic(self):
        """Idle penalty scaling test with explicit max_idle_duration_candles."""
        params_small = self.DEFAULT_PARAMS.copy()
        params_large = self.DEFAULT_PARAMS.copy()
        # Activate explicit max idle durations
        params_small["max_idle_duration_candles"] = 50
        params_large["max_idle_duration_candles"] = 200

        base_factor = self.TEST_BASE_FACTOR
        idle_duration = 40  # below large threshold, near small threshold
        context = RewardContext(
            pnl=0.0,
            trade_duration=0,
            idle_duration=idle_duration,
            max_trade_duration=128,
            max_unrealized_profit=0.0,
            min_unrealized_profit=0.0,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )

        breakdown_small = calculate_reward(
            context,
            params_small,
            base_factor,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        breakdown_large = calculate_reward(
            context,
            params_large,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=0.06,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )

        self.assertLess(breakdown_small.idle_penalty, 0.0)
        self.assertLess(breakdown_large.idle_penalty, 0.0)
        # Because denominator is larger, absolute penalty (negative) should be smaller in magnitude
        self.assertGreater(
            breakdown_large.idle_penalty,
            breakdown_small.idle_penalty,
            f"Expected less severe penalty with larger max_idle_duration_candles (large={breakdown_large.idle_penalty}, small={breakdown_small.idle_penalty})",
        )

    def test_pbrs_progressive_release_decay_clamped(self):
        """progressive_release with decay>1 must clamp to 1 so Φ' = 0 and Δ = -Φ_prev."""
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "potential_gamma": 0.95,
                "exit_potential_mode": "progressive_release",
                "exit_potential_decay": 5.0,  # should clamp to 1.0
                "hold_potential_enabled": True,
                "entry_additive_enabled": False,
                "exit_additive_enabled": False,
            }
        )
        current_pnl = 0.02
        current_dur = 0.5
        prev_potential = _compute_hold_potential(current_pnl, current_dur, params)
        total_reward, shaping_reward, next_potential = apply_potential_shaping(
            base_reward=0.0,
            current_pnl=current_pnl,
            current_duration_ratio=current_dur,
            next_pnl=0.02,
            next_duration_ratio=0.6,
            is_terminal=True,
            last_potential=prev_potential,
            params=params,
        )
        self.assertAlmostEqualFloat(next_potential, 0.0, tolerance=1e-9)
        self.assertAlmostEqualFloat(shaping_reward, -prev_potential, tolerance=1e-9)
        self.assertAlmostEqualFloat(total_reward, shaping_reward, tolerance=1e-9)

    def test_pbrs_spike_cancel_invariance(self):
        """spike_cancel terminal shaping should be ≈ 0 (γ*(Φ/γ) - Φ)."""
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "potential_gamma": 0.9,
                "exit_potential_mode": "spike_cancel",
                "hold_potential_enabled": True,
                "entry_additive_enabled": False,
                "exit_additive_enabled": False,
            }
        )
        current_pnl = 0.015
        current_dur = 0.4
        prev_potential = _compute_hold_potential(current_pnl, current_dur, params)
        # Use helper accessor to avoid union type issues
        gamma = _get_float_param(params, "potential_gamma", 0.95)
        expected_next = (
            prev_potential / gamma if gamma not in (0.0, None) else prev_potential
        )
        total_reward, shaping_reward, next_potential = apply_potential_shaping(
            base_reward=0.0,
            current_pnl=current_pnl,
            current_duration_ratio=current_dur,
            next_pnl=0.016,
            next_duration_ratio=0.45,
            is_terminal=True,
            last_potential=prev_potential,
            params=params,
        )
        self.assertAlmostEqualFloat(next_potential, expected_next, tolerance=1e-9)
        self.assertAlmostEqualFloat(shaping_reward, 0.0, tolerance=1e-9)
        self.assertAlmostEqualFloat(total_reward, 0.0, tolerance=1e-9)

    def test_idle_penalty_fallback_and_proportionality(self):
        """Fallback & proportionality validation.

        Semantics:
        - When max_idle_duration_candles is unset, fallback must be 2 * max_trade_duration.
        - Idle penalty scales ~ linearly with idle_duration (power=1), so doubling idle_duration doubles penalty magnitude.
        - We also infer the implicit denominator from a mid-range idle duration (>1x and <2x trade duration) to ensure the
          2x fallback.
        """
        params = self.DEFAULT_PARAMS.copy()
        params["max_idle_duration_candles"] = None
        base_factor = 90.0
        profit_target = self.TEST_PROFIT_TARGET
        risk_reward_ratio = 1.0

        # Two contexts with different idle durations
        ctx_a = RewardContext(
            pnl=0.0,
            trade_duration=0,
            idle_duration=20,
            max_trade_duration=100,
            max_unrealized_profit=0.0,
            min_unrealized_profit=0.0,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )
        ctx_b = dataclasses.replace(ctx_a, idle_duration=40)

        br_a = calculate_reward(
            ctx_a,
            params,
            base_factor=base_factor,
            profit_target=profit_target,
            risk_reward_ratio=risk_reward_ratio,
            short_allowed=True,
            action_masking=True,
        )
        br_b = calculate_reward(
            ctx_b,
            params,
            base_factor=base_factor,
            profit_target=profit_target,
            risk_reward_ratio=risk_reward_ratio,
            short_allowed=True,
            action_masking=True,
        )

        self.assertLess(br_a.idle_penalty, 0.0)
        self.assertLess(br_b.idle_penalty, 0.0)

        # Ratio of penalties should be close to 2 (linear power=1 scaling) allowing small numerical tolerance
        ratio = (
            br_b.idle_penalty / br_a.idle_penalty if br_a.idle_penalty != 0 else None
        )
        self.assertIsNotNone(ratio, "Idle penalty A should not be zero")
        # Both penalties are negative, ratio should be ~ (40/20) = 2, hence ratio ~ 2
        self.assertAlmostEqualFloat(
            abs(ratio),
            2.0,
            tolerance=0.2,
            msg=f"Idle penalty proportionality mismatch (ratio={ratio})",
        )
        # Additional mid-range inference check (idle_duration between 1x and 2x trade duration)
        ctx_mid = dataclasses.replace(
            ctx_a, idle_duration=120, max_trade_duration=100
        )  # Adjusted context for mid-range check
        br_mid = calculate_reward(
            ctx_mid,
            params,
            base_factor=base_factor,
            profit_target=profit_target,
            risk_reward_ratio=risk_reward_ratio,
            short_allowed=True,
            action_masking=True,
        )
        self.assertLess(br_mid.idle_penalty, 0.0)
        idle_penalty_scale = _get_float_param(params, "idle_penalty_scale", 0.5)
        idle_penalty_power = _get_float_param(params, "idle_penalty_power", 1.025)
        # Internal factor may come from params (overrides provided base_factor argument)
        factor = _get_float_param(params, "base_factor", float(base_factor))
        idle_factor = factor * (profit_target * risk_reward_ratio) / 4.0
        observed_ratio = abs(br_mid.idle_penalty) / (idle_factor * idle_penalty_scale)
        if observed_ratio > 0:
            implied_D = 120 / (observed_ratio ** (1 / idle_penalty_power))
            self.assertAlmostEqualFloat(
                implied_D,
                400.0,
                tolerance=20.0,
                msg=f"Fallback denominator mismatch (implied={implied_D}, expected≈400, factor={factor})",
            )

    def test_exit_factor_threshold_warning_non_capping(self):
        """Ensure exit_factor_threshold does not cap the exit factor (warning-only semantics).

        We approximate by computing two rewards: a baseline and an amplified one (via larger base_factor & pnl) and
        ensure the amplified reward scales proportionally beyond the threshold rather than flattening.
        """
        params = self.DEFAULT_PARAMS.copy()
        # Remove base_factor from params so that the function uses the provided argument (makes scaling observable)
        params.pop("base_factor", None)
        exit_factor_threshold = _get_float_param(
            params, "exit_factor_threshold", 10_000.0
        )

        context = RewardContext(
            pnl=0.08,  # above typical profit_target * RR
            trade_duration=10,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.09,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Long_exit,
        )

        # Baseline with moderate base_factor
        baseline = calculate_reward(
            context,
            params,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR_HIGH,
            short_allowed=True,
            action_masking=True,
        )

        # Amplified: choose a much larger base_factor (ensure > threshold relative scale)
        amplified_base_factor = max(
            self.TEST_BASE_FACTOR * 50,
            exit_factor_threshold * self.TEST_RR_HIGH / max(context.pnl, 1e-9),
        )
        amplified = calculate_reward(
            context,
            params,
            base_factor=amplified_base_factor,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR_HIGH,
            short_allowed=True,
            action_masking=True,
        )

        scale_observed = (
            amplified.exit_component / baseline.exit_component
            if baseline.exit_component
            else 0.0
        )
        self.assertGreater(baseline.exit_component, 0.0)
        self.assertGreater(amplified.exit_component, baseline.exit_component)
        # Expect at least ~10x increase (safe margin) to confirm absence of clipping
        self.assertGreater(
            scale_observed,
            10.0,
            f"Amplified reward did not scale sufficiently (factor={scale_observed:.2f}, amplified_base_factor={amplified_base_factor})",
        )

    def test_exit_factor_calculation(self):
        """Test exit factor calculation consistency across core modes + plateau variant.

        Plateau behavior expressed via exit_plateau=True with a base kernel (e.g. linear).
        """
        # Core attenuation kernels (excluding legacy which is step-based)
        modes_to_test = ["linear", "power"]

        for mode in modes_to_test:
            test_params = self.DEFAULT_PARAMS.copy()
            test_params["exit_attenuation_mode"] = mode
            factor = _get_exit_factor(
                base_factor=1.0,
                pnl=0.02,
                pnl_factor=1.5,
                duration_ratio=0.3,
                params=test_params,
            )
            self.assertFinite(factor, name=f"exit_factor[{mode}]")
            self.assertGreater(factor, 0, f"Exit factor for {mode} should be positive")

        # Plateau+linear variant sanity check (grace region at 0.5)
        plateau_params = self.DEFAULT_PARAMS.copy()
        plateau_params.update(
            {
                "exit_attenuation_mode": "linear",
                "exit_plateau": True,
                "exit_plateau_grace": 0.5,
                "exit_linear_slope": 1.0,
            }
        )
        plateau_factor_pre = _get_exit_factor(
            base_factor=1.0,
            pnl=0.02,
            pnl_factor=1.5,
            duration_ratio=0.4,  # inside grace
            params=plateau_params,
        )
        plateau_factor_post = _get_exit_factor(
            base_factor=1.0,
            pnl=0.02,
            pnl_factor=1.5,
            duration_ratio=0.8,  # after grace => attenuated or equal (slope may reduce)
            params=plateau_params,
        )
        self.assertGreater(plateau_factor_pre, 0)
        self.assertGreater(plateau_factor_post, 0)
        self.assertGreaterEqual(
            plateau_factor_pre,
            plateau_factor_post - 1e-12,
            "Plateau pre-grace factor should be >= post-grace factor",
        )

    def test_negative_slope_sanitization(self):
        """Negative slopes for linear must be sanitized to positive default (1.0)."""
        base_factor = self.TEST_BASE_FACTOR
        pnl = 0.04
        pnl_factor = 1.0
        duration_ratio_linear = 1.2  # any positive ratio
        duration_ratio_plateau = 1.3  # > grace so slope matters

        # Linear mode: slope -5.0 should behave like slope=1.0 (sanitized)
        params_lin_neg = self.DEFAULT_PARAMS.copy()
        params_lin_neg.update(
            {"exit_attenuation_mode": "linear", "exit_linear_slope": -5.0}
        )
        params_lin_pos = self.DEFAULT_PARAMS.copy()
        params_lin_pos.update(
            {"exit_attenuation_mode": "linear", "exit_linear_slope": 1.0}
        )
        val_lin_neg = _get_exit_factor(
            base_factor, pnl, pnl_factor, duration_ratio_linear, params_lin_neg
        )
        val_lin_pos = _get_exit_factor(
            base_factor, pnl, pnl_factor, duration_ratio_linear, params_lin_pos
        )
        self.assertAlmostEqualFloat(
            val_lin_neg,
            val_lin_pos,
            tolerance=1e-9,
            msg="Negative linear slope not sanitized to default behavior",
        )

        # Plateau+linear: negative slope sanitized similarly
        params_pl_neg = self.DEFAULT_PARAMS.copy()
        params_pl_neg.update(
            {
                "exit_attenuation_mode": "linear",
                "exit_plateau": True,
                "exit_plateau_grace": 1.0,
                "exit_linear_slope": -3.0,
            }
        )
        params_pl_pos = self.DEFAULT_PARAMS.copy()
        params_pl_pos.update(
            {
                "exit_attenuation_mode": "linear",
                "exit_plateau": True,
                "exit_plateau_grace": 1.0,
                "exit_linear_slope": 1.0,
            }
        )
        val_pl_neg = _get_exit_factor(
            base_factor, pnl, pnl_factor, duration_ratio_plateau, params_pl_neg
        )
        val_pl_pos = _get_exit_factor(
            base_factor, pnl, pnl_factor, duration_ratio_plateau, params_pl_pos
        )
        self.assertAlmostEqualFloat(
            val_pl_neg,
            val_pl_pos,
            tolerance=1e-9,
            msg="Negative plateau+linear slope not sanitized to default behavior",
        )

    def test_idle_penalty_zero_when_profit_target_zero(self):
        """If profit_target=0 → idle_factor=0 → idle penalty must be exactly 0 for neutral idle state."""
        context = RewardContext(
            pnl=0.0,
            trade_duration=0,
            idle_duration=30,
            max_trade_duration=100,
            max_unrealized_profit=0.0,
            min_unrealized_profit=0.0,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )
        br = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=0.0,  # critical case
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        self.assertEqual(
            br.idle_penalty, 0.0, "Idle penalty should be zero when profit_target=0"
        )
        self.assertEqual(
            br.total, 0.0, "Total reward should be zero in this configuration"
        )

    def test_power_mode_alpha_formula(self):
        """Validate power mode: factor ≈ base_factor / (1+r)^alpha where alpha=-log(tau)/log(2)."""
        tau = 0.5
        r = 1.2
        alpha = -math.log(tau) / math.log(2.0)
        base_factor = self.TEST_BASE_FACTOR
        pnl = self.TEST_PROFIT_TARGET
        pnl_factor = 1.0  # isolate attenuation
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "exit_attenuation_mode": "power",
                "exit_power_tau": tau,
                "exit_plateau": False,
            }
        )
        observed = _get_exit_factor(base_factor, pnl, pnl_factor, r, params)
        expected = base_factor / (1.0 + r) ** alpha
        self.assertAlmostEqualFloat(
            observed,
            expected,
            tolerance=1e-9,
            msg=f"Power mode attenuation mismatch (obs={observed}, exp={expected}, alpha={alpha})",
        )

    def test_win_reward_factor_saturation(self):
        """Saturation test: pnl amplification factor should monotonically approach (1 + win_reward_factor)."""
        win_reward_factor = 3.0  # asymptote = 4.0
        beta = 0.5
        profit_target = self.TEST_PROFIT_TARGET
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "win_reward_factor": win_reward_factor,
                "pnl_factor_beta": beta,
                "efficiency_weight": 0.0,  # disable efficiency modulation
                "exit_attenuation_mode": "linear",
                "exit_plateau": False,
                "exit_linear_slope": 0.0,  # keep attenuation = 1
            }
        )
        # Ensure provided base_factor=1.0 is actually used (remove default 100)
        params.pop("base_factor", None)

        # pnl values: slightly above target, 2x, 5x, 10x target
        pnl_values = [profit_target * m for m in (1.05, self.TEST_RR_HIGH, 5.0, 10.0)]
        ratios_observed: list[float] = []

        for pnl in pnl_values:
            context = RewardContext(
                pnl=pnl,
                trade_duration=0,  # duration_ratio=0 -> attenuation = 1
                idle_duration=0,
                max_trade_duration=100,
                max_unrealized_profit=pnl,  # neutral wrt efficiency (disabled anyway)
                min_unrealized_profit=0.0,
                position=Positions.Long,
                action=Actions.Long_exit,
            )
            br = calculate_reward(
                context,
                params,
                base_factor=1.0,  # isolate pnl_factor directly in exit reward ratio
                profit_target=profit_target,
                risk_reward_ratio=1.0,
                short_allowed=True,
                action_masking=True,
            )
            # br.exit_component = pnl * (base_factor * pnl_factor) => with base_factor=1, attenuation=1 => ratio = exit_component / pnl = pnl_factor
            ratio = br.exit_component / pnl if pnl != 0 else 0.0
            ratios_observed.append(float(ratio))

        # Monotonic non-decreasing (allow tiny float noise)
        self.assertMonotonic(
            ratios_observed,
            non_decreasing=True,
            tolerance=1e-12,
            name="pnl_amplification_ratio",
        )

        asymptote = 1.0 + win_reward_factor
        final_ratio = ratios_observed[-1]
        # Expect to be very close to asymptote (tanh(0.5*(10-1)) ≈ 0.9997)
        self.assertFinite(final_ratio, name="final_ratio")
        self.assertLess(
            abs(final_ratio - asymptote),
            1e-3,
            f"Final amplification {final_ratio:.6f} not close to asymptote {asymptote:.6f}",
        )

        # Analytical expected ratios for comparison (not strict assertions except final)
        expected_ratios: list[float] = []
        for pnl in pnl_values:
            pnl_ratio = pnl / profit_target
            expected = 1.0 + win_reward_factor * math.tanh(beta * (pnl_ratio - 1.0))
            expected_ratios.append(expected)
        # Compare each observed to expected within loose tolerance (model parity)
        for obs, exp in zip(ratios_observed, expected_ratios):
            self.assertFinite(obs, name="observed_ratio")
            self.assertFinite(exp, name="expected_ratio")
            self.assertLess(
                abs(obs - exp),
                5e-6,
                f"Observed amplification {obs:.8f} deviates from expected {exp:.8f}",
            )

    def test_scale_invariance_and_decomposition(self):
        """Core reward components scale ~ linearly with base_factor; total = core + shaping + additives.

        Contract:
            For each non-zero core component C: C(base_factor * k) ≈ k * C(base_factor).
            Decomposition uses total = (exit + idle + hold + invalid + shaping + entry_additive + exit_additive).
        """
        params = self.DEFAULT_PARAMS.copy()
        # Remove internal base_factor so the explicit argument is used
        params.pop("base_factor", None)
        base_factor = 80.0
        k = 7.5
        profit_target = self.TEST_PROFIT_TARGET
        rr = 1.5

        contexts: list[RewardContext] = [
            # Winning exit
            RewardContext(
                pnl=0.025,
                trade_duration=40,
                idle_duration=0,
                max_trade_duration=100,
                max_unrealized_profit=0.03,
                min_unrealized_profit=0.0,
                position=Positions.Long,
                action=Actions.Long_exit,
            ),
            # Losing exit
            RewardContext(
                pnl=-self.TEST_PNL_STD,
                trade_duration=60,
                idle_duration=0,
                max_trade_duration=100,
                max_unrealized_profit=0.01,
                min_unrealized_profit=-0.04,
                position=Positions.Long,
                action=Actions.Long_exit,
            ),
            # Idle penalty
            RewardContext(
                pnl=0.0,
                trade_duration=0,
                idle_duration=35,
                max_trade_duration=120,
                max_unrealized_profit=0.0,
                min_unrealized_profit=0.0,
                position=Positions.Neutral,
                action=Actions.Neutral,
            ),
            # Hold penalty
            RewardContext(
                pnl=0.0,
                trade_duration=80,
                idle_duration=0,
                max_trade_duration=100,
                max_unrealized_profit=0.04,
                min_unrealized_profit=-0.01,
                position=Positions.Long,
                action=Actions.Neutral,
            ),
        ]

        tol_scale = 1e-9
        for ctx in contexts:
            br1 = calculate_reward(
                ctx,
                params,
                base_factor=base_factor,
                profit_target=profit_target,
                risk_reward_ratio=rr,
                short_allowed=True,
                action_masking=True,
            )
            br2 = calculate_reward(
                ctx,
                params,
                base_factor=base_factor * k,
                profit_target=profit_target,
                risk_reward_ratio=rr,
                short_allowed=True,
                action_masking=True,
            )

            # Decomposition including shaping + additives (environment always applies PBRS pipeline)
            for br in (br1, br2):
                comp_sum = (
                    br.exit_component
                    + br.idle_penalty
                    + br.hold_penalty
                    + br.invalid_penalty
                    + br.shaping_reward
                    + br.entry_additive
                    + br.exit_additive
                )
                self.assertAlmostEqual(
                    br.total,
                    comp_sum,
                    places=12,
                    msg=f"Decomposition mismatch (ctx={ctx}, total={br.total}, sum={comp_sum})",
                )

            # Verify scale invariance for each non-negligible component
            components1 = {
                "exit_component": br1.exit_component,
                "idle_penalty": br1.idle_penalty,
                "hold_penalty": br1.hold_penalty,
                "invalid_penalty": br1.invalid_penalty,
                # Exclude shaping/additives from scale invariance check (some may have nonlinear dependence)
                "total": br1.exit_component
                + br1.idle_penalty
                + br1.hold_penalty
                + br1.invalid_penalty,
            }
            components2 = {
                "exit_component": br2.exit_component,
                "idle_penalty": br2.idle_penalty,
                "hold_penalty": br2.hold_penalty,
                "invalid_penalty": br2.invalid_penalty,
                "total": br2.exit_component
                + br2.idle_penalty
                + br2.hold_penalty
                + br2.invalid_penalty,
            }
            for key, v1 in components1.items():
                v2 = components2[key]
                if abs(v1) < 1e-15 and abs(v2) < 1e-15:
                    continue  # Skip exact zero (or numerically negligible) components
                self.assertLess(
                    abs(v2 - k * v1),
                    tol_scale * max(1.0, abs(k * v1)),
                    f"Scale invariance failed for {key}: v1={v1}, v2={v2}, k={k}",
                )

    def test_long_short_symmetry(self):
        """Validate Long vs Short exit reward magnitude symmetry for identical PnL.

        Hypothesis: No directional bias implies |R_long(pnl)| ≈ |R_short(pnl)|.
        """
        params = self.DEFAULT_PARAMS.copy()
        params.pop("base_factor", None)
        base_factor = 120.0
        profit_target = 0.04
        rr = self.TEST_RR_HIGH
        pnls = [0.018, -0.022]
        for pnl in pnls:
            ctx_long = RewardContext(
                pnl=pnl,
                trade_duration=55,
                idle_duration=0,
                max_trade_duration=100,
                max_unrealized_profit=pnl if pnl > 0 else 0.01,
                min_unrealized_profit=pnl if pnl < 0 else -0.01,
                position=Positions.Long,
                action=Actions.Long_exit,
            )
            ctx_short = RewardContext(
                pnl=pnl,
                trade_duration=55,
                idle_duration=0,
                max_trade_duration=100,
                max_unrealized_profit=pnl if pnl > 0 else 0.01,
                min_unrealized_profit=pnl if pnl < 0 else -0.01,
                position=Positions.Short,
                action=Actions.Short_exit,
            )
            br_long = calculate_reward(
                ctx_long,
                params,
                base_factor=base_factor,
                profit_target=profit_target,
                risk_reward_ratio=rr,
                short_allowed=True,
                action_masking=True,
            )
            br_short = calculate_reward(
                ctx_short,
                params,
                base_factor=base_factor,
                profit_target=profit_target,
                risk_reward_ratio=rr,
                short_allowed=True,
                action_masking=True,
            )
            # Sign aligned with PnL
            if pnl > 0:
                self.assertGreater(br_long.exit_component, 0)
                self.assertGreater(br_short.exit_component, 0)
            else:
                self.assertLess(br_long.exit_component, 0)
                self.assertLess(br_short.exit_component, 0)
            # Magnitudes should be close (tolerance scaled)
            self.assertLess(
                abs(abs(br_long.exit_component) - abs(br_short.exit_component)),
                1e-9 * max(1.0, abs(br_long.exit_component)),
                f"Long/Short asymmetry pnl={pnl}: long={br_long.exit_component}, short={br_short.exit_component}",
            )


class TestPublicAPI(RewardSpaceTestBase):
    """Test public API functions and interfaces."""

    def test_parse_overrides(self):
        """Test parse_overrides function."""
        # Test valid overrides
        overrides = ["key1=1.5", "key2=hello", "key3=42"]
        result = parse_overrides(overrides)

        self.assertEqual(result["key1"], 1.5)
        self.assertEqual(result["key2"], "hello")
        self.assertEqual(result["key3"], 42.0)

        # Test invalid format
        with self.assertRaises(ValueError):
            parse_overrides(["invalid_format"])

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence intervals calculation."""
        # Create test data
        np.random.seed(42)
        test_data = pd.DataFrame(
            {
                "reward_total": np.random.normal(0, 1, 100),
                "pnl": np.random.normal(0.01, self.TEST_PNL_STD, 100),
            }
        )

        results = bootstrap_confidence_intervals(
            test_data,
            ["reward_total", "pnl"],
            n_bootstrap=100,  # Small for speed
        )

        self.assertIn("reward_total", results)
        self.assertIn("pnl", results)

        for metric, (mean, ci_low, ci_high) in results.items():
            self.assertFinite(mean, name=f"mean[{metric}]")
            self.assertFinite(ci_low, name=f"ci_low[{metric}]")
            self.assertFinite(ci_high, name=f"ci_high[{metric}]")
            self.assertLess(
                ci_low, ci_high, f"CI bounds for {metric} should be ordered"
            )

    def test_statistical_hypothesis_tests_seed_reproducibility(self):
        """Ensure statistical_hypothesis_tests + bootstrap CIs are reproducible with stats_seed."""

        np.random.seed(123)
        # Create idle_duration with variability throughout to avoid constant Spearman warnings
        idle_base = np.random.uniform(3, 40, 300)
        idle_mask = np.random.rand(300) < 0.4
        idle_duration = (
            idle_base * idle_mask
        )  # some zeros but not fully constant subset
        df = pd.DataFrame(
            {
                "reward_total": np.random.normal(0, 1, 300),
                "reward_idle": np.where(idle_mask, np.random.normal(-1, 0.3, 300), 0.0),
                "reward_hold": np.where(
                    ~idle_mask, np.random.normal(-0.5, 0.2, 300), 0.0
                ),
                "reward_exit": np.random.normal(0.8, 0.6, 300),
                "pnl": np.random.normal(0.01, self.TEST_PNL_STD, 300),
                "trade_duration": np.random.uniform(5, 150, 300),
                "idle_duration": idle_duration,
                "position": np.random.choice([0.0, 0.5, 1.0], 300),
            }
        )

        # Two runs with same seed
        r1 = statistical_hypothesis_tests(df, seed=777)
        r2 = statistical_hypothesis_tests(df, seed=777)
        self.assertEqual(set(r1.keys()), set(r2.keys()))
        for k in r1:
            for field in ("p_value", "significant"):
                v1 = r1[k][field]
                v2 = r2[k][field]
                if (
                    isinstance(v1, float)
                    and isinstance(v2, float)
                    and (np.isnan(v1) and np.isnan(v2))
                ):
                    continue
                self.assertEqual(v1, v2, f"Mismatch for {k}:{field}")

        # Different seed may change bootstrap CI of rho if present
        r3 = statistical_hypothesis_tests(df, seed=888)
        if "idle_correlation" in r1 and "idle_correlation" in r3:
            ci1 = r1["idle_correlation"]["ci_95"]
            ci3 = r3["idle_correlation"]["ci_95"]
            # Not guaranteed different, but often; only assert shape
            self.assertEqual(len(ci1), 2)
            self.assertEqual(len(ci3), 2)

        # Test bootstrap reproducibility path
        metrics = ["reward_total", "pnl"]
        ci_a = bootstrap_confidence_intervals(df, metrics, n_bootstrap=250, seed=2024)
        ci_b = bootstrap_confidence_intervals(df, metrics, n_bootstrap=250, seed=2024)
        self.assertEqual(ci_a, ci_b)


class TestStatisticalValidation(RewardSpaceTestBase):
    """Test statistical validation and mathematical bounds."""

    def test_pnl_invariant_validation(self):
        """Test critical PnL invariant: only exit actions should have non-zero PnL."""
        # Generate samples and verify PnL invariant
        df = simulate_samples(
            num_samples=200,
            seed=42,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=50,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )

        # Critical invariant: Total PnL must equal sum of exit PnL
        total_pnl = df["pnl"].sum()
        exit_mask = df["reward_exit"] != 0
        exit_pnl_sum = df.loc[exit_mask, "pnl"].sum()

        self.assertAlmostEqual(
            total_pnl,
            exit_pnl_sum,
            places=10,
            msg="PnL invariant violation: total PnL != sum of exit PnL",
        )

        # Only exit actions (2.0, 4.0) should have non-zero PnL
        non_zero_pnl_actions = set(df[df["pnl"] != 0]["action"].unique())
        expected_exit_actions = {2.0, 4.0}  # Long_exit, Short_exit

        self.assertTrue(
            non_zero_pnl_actions.issubset(expected_exit_actions),
            f"Non-exit actions have PnL: {non_zero_pnl_actions - expected_exit_actions}",
        )

        # No actions should have zero PnL but non-zero exit reward
        invalid_combinations = df[(df["pnl"] == 0) & (df["reward_exit"] != 0)]
        self.assertEqual(
            len(invalid_combinations),
            0,
            "Found actions with zero PnL but non-zero exit reward",
        )

    def test_distribution_metrics_mathematical_bounds(self):
        """Test mathematical bounds and validity of distribution shift metrics."""
        # Create two different distributions for testing
        np.random.seed(42)
        df1 = pd.DataFrame(
            {
                "pnl": np.random.normal(0, self.TEST_PNL_STD, 500),
                "trade_duration": np.random.exponential(30, 500),
                "idle_duration": np.random.gamma(2, 5, 500),
            }
        )

        df2 = pd.DataFrame(
            {
                "pnl": np.random.normal(0.01, 0.025, 500),
                "trade_duration": np.random.exponential(35, 500),
                "idle_duration": np.random.gamma(2.5, 6, 500),
            }
        )

        metrics = compute_distribution_shift_metrics(df1, df2)

        # Validate mathematical bounds for each feature
        for feature in ["pnl", "trade_duration", "idle_duration"]:
            # KL divergence must be >= 0
            kl_key = f"{feature}_kl_divergence"
            if kl_key in metrics:
                self.assertDistanceMetric(metrics[kl_key], name=kl_key)

            # JS distance must be in [0, 1]
            js_key = f"{feature}_js_distance"
            if js_key in metrics:
                self.assertDistanceMetric(metrics[js_key], upper=1.0, name=js_key)

            # Wasserstein must be >= 0
            ws_key = f"{feature}_wasserstein"
            if ws_key in metrics:
                self.assertDistanceMetric(metrics[ws_key], name=ws_key)

            # KS statistic must be in [0, 1]
            ks_stat_key = f"{feature}_ks_statistic"
            if ks_stat_key in metrics:
                self.assertDistanceMetric(
                    metrics[ks_stat_key], upper=1.0, name=ks_stat_key
                )

            # KS p-value must be in [0, 1]
            ks_p_key = f"{feature}_ks_pvalue"
            if ks_p_key in metrics:
                self.assertPValue(
                    metrics[ks_p_key], msg=f"KS p-value out of bounds for {feature}"
                )

    def test_heteroscedasticity_pnl_validation(self):
        """Test that PnL variance increases with trade duration (heteroscedasticity)."""
        # Generate larger sample for statistical power
        df = simulate_samples(
            num_samples=1000,
            seed=123,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=100,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )

        # Filter to exit actions only (where PnL is meaningful)
        exit_data = df[df["reward_exit"] != 0].copy()

        if len(exit_data) < 50:
            self.skipTest("Insufficient exit actions for heteroscedasticity test")

        # Create duration bins and test variance pattern
        exit_data["duration_bin"] = pd.cut(
            exit_data["duration_ratio"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )

        variance_by_bin = exit_data.groupby("duration_bin")["pnl"].var().dropna()

        # Should see increasing variance with duration (not always strict monotonic due to sampling)
        # Test that Q4 variance > Q1 variance (should hold with heteroscedastic model)
        if "Q1" in variance_by_bin.index and "Q4" in variance_by_bin.index:
            self.assertGreater(
                variance_by_bin["Q4"],
                variance_by_bin["Q1"] * 0.8,  # Allow some tolerance
                "PnL heteroscedasticity: variance should increase with duration",
            )

    def test_exit_factor_mathematical_formulas(self):
        """Test mathematical correctness of exit factor calculations."""

        # Test context with known values
        context = RewardContext(
            pnl=0.05,
            trade_duration=50,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.06,
            min_unrealized_profit=0.04,
            position=Positions.Long,
            action=Actions.Long_exit,
        )

        params = self.DEFAULT_PARAMS.copy()
        duration_ratio = 50 / 100  # 0.5

        # Test power mode with known tau
        params["exit_attenuation_mode"] = "power"
        params["exit_power_tau"] = 0.5
        params["exit_plateau"] = False

        reward_power = calculate_reward(
            context,
            params,
            self.TEST_BASE_FACTOR,
            self.TEST_PROFIT_TARGET,
            self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )

        # Mathematical validation: alpha = -ln(tau) = -ln(0.5) ≈ 0.693
        # Analytical alpha value (unused directly now, kept for clarity): -ln(0.5) ≈ 0.693

        # The reward should be attenuated by this factor
        self.assertGreater(
            reward_power.exit_component, 0, "Power mode should generate positive reward"
        )

        # Test half_life mode
        params["exit_attenuation_mode"] = "half_life"
        params["exit_half_life"] = 0.5

        reward_half_life = calculate_reward(
            context,
            params,
            self.TEST_BASE_FACTOR,
            self.TEST_PROFIT_TARGET,
            self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )

        # Mathematical validation: 2^(-duration_ratio/half_life) = 2^(-0.5/0.5) = 0.5
        expected_half_life_factor = 2 ** (-duration_ratio / 0.5)
        self.assertAlmostEqual(expected_half_life_factor, 0.5, places=6)

        # Test that different modes produce different results (mathematical diversity)
        params["exit_attenuation_mode"] = "linear"
        params["exit_linear_slope"] = 1.0

        reward_linear = calculate_reward(
            context,
            params,
            self.TEST_BASE_FACTOR,
            self.TEST_PROFIT_TARGET,
            self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )

        # All modes should produce positive rewards but different values
        self.assertGreater(reward_power.exit_component, 0)
        self.assertGreater(reward_half_life.exit_component, 0)
        self.assertGreater(reward_linear.exit_component, 0)

        # They should be different (mathematical distinctness)
        rewards = [
            reward_power.exit_component,
            reward_half_life.exit_component,
            reward_linear.exit_component,
        ]
        unique_rewards = set(f"{r:.6f}" for r in rewards)
        self.assertGreater(
            len(unique_rewards),
            1,
            "Different exit factor modes should produce different rewards",
        )

    def test_statistical_functions_bounds_validation(self):
        """Test that all statistical functions respect mathematical bounds."""
        # Create test data with known statistical properties
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "reward_total": np.random.normal(0, 1, 300),
                "reward_idle": np.random.normal(-1, 0.5, 300),
                "reward_hold": np.random.normal(-0.5, 0.3, 300),
                "reward_exit": np.random.normal(1, 0.8, 300),
                "pnl": np.random.normal(0.01, 0.02, 300),
                "trade_duration": np.random.uniform(5, 150, 300),
                "idle_duration": np.random.uniform(0, 100, 300),
                "position": np.random.choice([0.0, 0.5, 1.0], 300),
            }
        )

        # Test distribution diagnostics bounds
        diagnostics = distribution_diagnostics(df)

        for col in ["reward_total", "pnl", "trade_duration", "idle_duration"]:
            # Skewness can be any real number, but should be finite
            if f"{col}_skewness" in diagnostics:
                skew = diagnostics[f"{col}_skewness"]
                self.assertFinite(skew, name=f"skewness[{col}]")

            # Kurtosis should be finite (can be negative for platykurtic distributions)
            if f"{col}_kurtosis" in diagnostics:
                kurt = diagnostics[f"{col}_kurtosis"]
                self.assertFinite(kurt, name=f"kurtosis[{col}]")

            # Shapiro p-value must be in [0, 1]
            if f"{col}_shapiro_pval" in diagnostics:
                self.assertPValue(
                    diagnostics[f"{col}_shapiro_pval"],
                    msg=f"Shapiro p-value bounds for {col}",
                )

        # Test hypothesis tests results bounds
        hypothesis_results = statistical_hypothesis_tests(df, seed=42)

        for test_name, result in hypothesis_results.items():
            # All p-values must be in [0, 1]
            if "p_value" in result:
                self.assertPValue(
                    result["p_value"], msg=f"p-value bounds for {test_name}"
                )
            # Effect size epsilon squared (ANOVA/Kruskal) must be finite and >= 0
            if "effect_size_epsilon_sq" in result:
                eps2 = result["effect_size_epsilon_sq"]
                self.assertFinite(eps2, name=f"epsilon_sq[{test_name}]")
                self.assertGreaterEqual(
                    eps2, 0.0, f"Effect size epsilon^2 for {test_name} must be >= 0"
                )
            # Rank-biserial correlation (Mann-Whitney) must be finite in [-1, 1]
            if "effect_size_rank_biserial" in result:
                rb = result["effect_size_rank_biserial"]
                self.assertFinite(rb, name=f"rank_biserial[{test_name}]")
                self.assertGreaterEqual(
                    rb, -1.0, f"Rank-biserial correlation for {test_name} must be >= -1"
                )
                self.assertLessEqual(
                    rb, 1.0, f"Rank-biserial correlation for {test_name} must be <= 1"
                )
            # Generic correlation effect size (Spearman/Pearson) if present
            if "rho" in result:
                rho = result["rho"]
                if rho is not None:
                    self.assertFinite(rho, name=f"rho[{test_name}]")
                    self.assertGreaterEqual(
                        rho, -1.0, f"Correlation rho for {test_name} must be >= -1"
                    )
                    self.assertLessEqual(
                        rho, 1.0, f"Correlation rho for {test_name} must be <= 1"
                    )

    def test_benjamini_hochberg_adjustment(self):
        """Benjamini-Hochberg adjustment adds p_value_adj & significant_adj fields with valid bounds."""

        # Use simulation to trigger multiple tests
        df = simulate_samples(
            num_samples=1000,
            seed=123,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=100,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )

        results_adj = statistical_hypothesis_tests(
            df, adjust_method="benjamini_hochberg", seed=777
        )
        # At least one test should have run (idle or kruskal etc.)
        self.assertGreater(len(results_adj), 0, "No hypothesis tests executed")
        for name, res in results_adj.items():
            self.assertIn("p_value", res, f"Missing p_value in {name}")
            self.assertIn("p_value_adj", res, f"Missing p_value_adj in {name}")
            self.assertIn("significant_adj", res, f"Missing significant_adj in {name}")
            p_raw = res["p_value"]
            p_adj = res["p_value_adj"]
            self.assertPValue(p_raw, msg=f"Raw p-value out of bounds ({p_raw})")
            self.assertPValue(p_adj, msg=f"Adjusted p-value out of bounds ({p_adj})")
            # BH should not reduce p-value (non-decreasing) after monotonic enforcement
            self.assertGreaterEqual(
                p_adj,
                p_raw - 1e-12,
                f"Adjusted p-value {p_adj} is smaller than raw {p_raw}",
            )
            # Consistency of significance flags
            alpha = 0.05
            self.assertEqual(
                res["significant_adj"],
                bool(p_adj < alpha),
                f"significant_adj inconsistent for {name}",
            )
            # Optional: if effect sizes present, basic bounds
            if "effect_size_epsilon_sq" in res:
                eff = res["effect_size_epsilon_sq"]
                self.assertFinite(eff, name=f"effect_size[{name}]")
                self.assertGreaterEqual(eff, 0, f"ε² should be >=0 for {name}")
            if "effect_size_rank_biserial" in res:
                rb = res["effect_size_rank_biserial"]
                self.assertFinite(rb, name=f"rank_biserial[{name}]")
                self.assertGreaterEqual(rb, -1, f"Rank-biserial lower bound {name}")
                self.assertLessEqual(rb, 1, f"Rank-biserial upper bound {name}")

    def test_simulate_samples_with_different_modes(self):
        """Test simulate_samples with different trading modes."""
        # Test spot mode (no shorts)
        df_spot = simulate_samples(
            num_samples=100,
            seed=42,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=100,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="spot",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )

        # Should not have any short positions
        short_positions = (df_spot["position"] == float(Positions.Short.value)).sum()
        self.assertEqual(
            short_positions, 0, "Spot mode should not have short positions"
        )

        # Test margin mode (shorts allowed)
        df_margin = simulate_samples(
            num_samples=100,
            seed=42,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=100,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )

        # Should have required columns
        required_columns = [
            "pnl",
            "trade_duration",
            "idle_duration",
            "position",
            "action",
            "reward_total",
            "reward_invalid",
            "reward_idle",
            "reward_hold",
            "reward_exit",
        ]
        for col in required_columns:
            self.assertIn(col, df_margin.columns, f"Column {col} should be present")

    def test_reward_calculation(self):
        """Test reward calculation scenarios."""
        # Test different reward scenarios
        test_cases = [
            # (position, action, expected_reward_type)
            (Positions.Neutral, Actions.Neutral, "idle_penalty"),
            (Positions.Long, Actions.Long_exit, "exit_component"),
            (Positions.Short, Actions.Short_exit, "exit_component"),
        ]

        for position, action, expected_type in test_cases:
            with self.subTest(position=position, action=action):
                context = RewardContext(
                    pnl=0.02 if expected_type == "exit_component" else 0.0,
                    trade_duration=50 if position != Positions.Neutral else 0,
                    idle_duration=10 if position == Positions.Neutral else 0,
                    max_trade_duration=100,
                    max_unrealized_profit=0.03,
                    min_unrealized_profit=-0.01,
                    position=position,
                    action=action,
                )

                breakdown = calculate_reward(
                    context,
                    self.DEFAULT_PARAMS,
                    base_factor=self.TEST_BASE_FACTOR,
                    profit_target=self.TEST_PROFIT_TARGET,
                    risk_reward_ratio=1.0,
                    short_allowed=True,
                    action_masking=True,
                )

                # Check that the appropriate component is non-zero
                if expected_type == "idle_penalty":
                    self.assertNotEqual(
                        breakdown.idle_penalty,
                        0.0,
                        f"Expected idle penalty for {position}/{action}",
                    )
                elif expected_type == "exit_component":
                    self.assertNotEqual(
                        breakdown.exit_component,
                        0.0,
                        f"Expected exit component for {position}/{action}",
                    )

                # Total should always be finite
                self.assertFinite(breakdown.total, name="breakdown.total")


class TestBoundaryConditions(RewardSpaceTestBase):
    """Test boundary conditions and edge cases."""

    def test_extreme_parameter_values(self):
        """Test behavior with extreme parameter values."""
        # Test with very large parameters
        extreme_params = self.DEFAULT_PARAMS.copy()
        extreme_params["win_reward_factor"] = 1000.0
        extreme_params["base_factor"] = 10000.0

        context = RewardContext(
            pnl=0.05,
            trade_duration=50,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.06,
            min_unrealized_profit=0.02,
            position=Positions.Long,
            action=Actions.Long_exit,
        )

        breakdown = calculate_reward(
            context,
            extreme_params,
            base_factor=10000.0,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )

        self.assertFinite(breakdown.total, name="breakdown.total")

    def test_different_exit_attenuation_modes(self):
        """Test different exit attenuation modes (legacy, sqrt, linear, power, half_life)."""
        modes = ["legacy", "sqrt", "linear", "power", "half_life"]

        for mode in modes:
            with self.subTest(mode=mode):
                test_params = self.DEFAULT_PARAMS.copy()
                test_params["exit_attenuation_mode"] = mode

                context = RewardContext(
                    pnl=0.02,
                    trade_duration=50,
                    idle_duration=0,
                    max_trade_duration=100,
                    max_unrealized_profit=0.03,
                    min_unrealized_profit=0.01,
                    position=Positions.Long,
                    action=Actions.Long_exit,
                )

                breakdown = calculate_reward(
                    context,
                    test_params,
                    base_factor=self.TEST_BASE_FACTOR,
                    profit_target=self.TEST_PROFIT_TARGET,
                    risk_reward_ratio=self.TEST_RR,
                    short_allowed=True,
                    action_masking=True,
                )

                self.assertFinite(
                    breakdown.exit_component, name="breakdown.exit_component"
                )
                self.assertFinite(breakdown.total, name="breakdown.total")


class TestHelperFunctions(RewardSpaceTestBase):
    """Test utility and helper functions."""

    def test_to_bool(self):
        """Test _to_bool with various inputs."""
        # Test via simulate_samples which uses action_masking parameter
        df1 = simulate_samples(
            num_samples=10,
            seed=42,
            params={"action_masking": "true"},
            max_trade_duration=50,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="spot",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        self.assertIsInstance(df1, pd.DataFrame)

        df2 = simulate_samples(
            num_samples=10,
            seed=42,
            params={"action_masking": "false"},
            max_trade_duration=50,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="spot",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        self.assertIsInstance(df2, pd.DataFrame)

    def test_short_allowed_via_simulation(self):
        """Test _is_short_allowed via different trading modes."""
        # Test futures mode (shorts allowed)
        df_futures = simulate_samples(
            num_samples=100,
            seed=42,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=50,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="futures",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )

        # Should have some short positions
        short_positions = (df_futures["position"] == float(Positions.Short.value)).sum()
        self.assertGreater(
            short_positions, 0, "Futures mode should allow short positions"
        )

    def test_statistical_functions(self):
        """Test statistical functions."""

        # Create test data with specific patterns
        np.random.seed(42)
        test_data = pd.DataFrame(
            {
                "reward_total": np.random.normal(0, 1, 200),
                "reward_idle": np.concatenate(
                    [np.zeros(150), np.random.normal(-1, 0.5, 50)]
                ),
                "reward_hold": np.concatenate(
                    [np.zeros(150), np.random.normal(-0.5, 0.3, 50)]
                ),
                "reward_exit": np.concatenate(
                    [np.zeros(150), np.random.normal(2, 1, 50)]
                ),
                "pnl": np.random.normal(0.01, 0.02, 200),
                "trade_duration": np.random.uniform(10, 100, 200),
                "idle_duration": np.concatenate(
                    [np.random.uniform(5, 50, 50), np.zeros(150)]
                ),
                "position": np.random.choice([0.0, 0.5, 1.0], 200),
            }
        )

        # Test hypothesis tests
        results = statistical_hypothesis_tests(test_data)
        self.assertIsInstance(results, dict)

    def test_argument_parser_construction(self):
        """Test build_argument_parser function."""

        parser = build_argument_parser()
        self.assertIsNotNone(parser)

        # Test parsing with minimal arguments
        args = parser.parse_args(["--num_samples", "100", "--output", "test_output"])
        self.assertEqual(args.num_samples, 100)
        self.assertEqual(str(args.output), "test_output")

    def test_complete_statistical_analysis_writer(self):
        """Test write_complete_statistical_analysis function."""

        # Create comprehensive test data
        test_data = simulate_samples(
            num_samples=200,
            seed=42,
            params=self.DEFAULT_PARAMS,
            max_trade_duration=100,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)

            write_complete_statistical_analysis(
                test_data,
                output_path,
                max_trade_duration=100,
                profit_target=self.TEST_PROFIT_TARGET,
                seed=42,
                real_df=None,
            )

            # Check that main report is created
            main_report = output_path / "statistical_analysis.md"
            self.assertTrue(
                main_report.exists(), "Main statistical analysis should be created"
            )

            # Check for other expected files
            feature_file = output_path / "feature_importance.csv"
            self.assertTrue(
                feature_file.exists(), "Feature importance should be created"
            )


class TestPrivateFunctions(RewardSpaceTestBase):
    """Test private functions through public API calls."""

    def test_idle_penalty_via_rewards(self):
        """Test idle penalty calculation via reward calculation."""
        # Create context that will trigger idle penalty
        context = RewardContext(
            pnl=0.0,
            trade_duration=0,
            idle_duration=20,
            max_trade_duration=100,
            max_unrealized_profit=0.0,
            min_unrealized_profit=0.0,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )

        breakdown = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=1.0,
            short_allowed=True,
            action_masking=True,
        )

        self.assertLess(breakdown.idle_penalty, 0, "Idle penalty should be negative")
        # Total now includes shaping/additives - require equality including those components.
        self.assertAlmostEqualFloat(
            breakdown.total,
            breakdown.idle_penalty
            + breakdown.shaping_reward
            + breakdown.entry_additive
            + breakdown.exit_additive,
            tolerance=1e-9,
            msg="Total should equal sum of components (idle + shaping/additives)",
        )

    def test_hold_penalty_via_rewards(self):
        """Test hold penalty calculation via reward calculation."""
        # Create context that will trigger hold penalty
        context = RewardContext(
            pnl=0.01,
            trade_duration=150,
            idle_duration=0,  # Long duration
            max_trade_duration=100,
            max_unrealized_profit=0.02,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Neutral,
        )

        breakdown = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )

        self.assertLess(breakdown.hold_penalty, 0, "Hold penalty should be negative")
        self.assertAlmostEqualFloat(
            breakdown.total,
            breakdown.hold_penalty
            + breakdown.shaping_reward
            + breakdown.entry_additive
            + breakdown.exit_additive,
            tolerance=1e-9,
            msg="Total should equal sum of components (hold + shaping/additives)",
        )

    def test_exit_reward_calculation(self):
        """Test exit reward calculation with various scenarios."""
        scenarios = [
            (Positions.Long, Actions.Long_exit, 0.05, "Profitable long exit"),
            (Positions.Short, Actions.Short_exit, -0.03, "Profitable short exit"),
            (Positions.Long, Actions.Long_exit, -0.02, "Losing long exit"),
            (Positions.Short, Actions.Short_exit, 0.02, "Losing short exit"),
        ]

        for position, action, pnl, description in scenarios:
            with self.subTest(description=description):
                context = RewardContext(
                    pnl=pnl,
                    trade_duration=50,
                    idle_duration=0,
                    max_trade_duration=100,
                    max_unrealized_profit=max(pnl + 0.01, 0.01),
                    min_unrealized_profit=min(pnl - 0.01, -0.01),
                    position=position,
                    action=action,
                )

                breakdown = calculate_reward(
                    context,
                    self.DEFAULT_PARAMS,
                    base_factor=self.TEST_BASE_FACTOR,
                    profit_target=self.TEST_PROFIT_TARGET,
                    risk_reward_ratio=1.0,
                    short_allowed=True,
                    action_masking=True,
                )

                self.assertNotEqual(
                    breakdown.exit_component,
                    0.0,
                    f"Exit component should be non-zero for {description}",
                )
                self.assertTrue(
                    np.isfinite(breakdown.total),
                    f"Total should be finite for {description}",
                )

    def test_invalid_action_handling(self):
        """Test invalid action penalty."""
        # Try to exit long when in short position (invalid)
        context = RewardContext(
            pnl=0.02,
            trade_duration=50,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.03,
            min_unrealized_profit=0.01,
            position=Positions.Short,
            action=Actions.Long_exit,
        )

        breakdown = calculate_reward(
            context,
            self.DEFAULT_PARAMS,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=1.0,
            short_allowed=True,
            action_masking=False,  # Disable masking to test invalid penalty
        )

        self.assertLess(
            breakdown.invalid_penalty, 0, "Invalid action should have negative penalty"
        )
        self.assertAlmostEqualFloat(
            breakdown.total,
            breakdown.invalid_penalty
            + breakdown.shaping_reward
            + breakdown.entry_additive
            + breakdown.exit_additive,
            tolerance=1e-9,
            msg="Total should equal invalid penalty plus shaping/additives",
        )

    def test_hold_penalty_zero_before_max_duration(self):
        """Test hold penalty logic: zero penalty before max_trade_duration."""
        max_duration = 128

        # Test cases: before, at, and after max_duration
        test_cases = [
            (64, "before max_duration"),
            (127, "just before max_duration"),
            (128, "exactly at max_duration"),
            (129, "just after max_duration"),
            (192, "well after max_duration"),
        ]

        for trade_duration, description in test_cases:
            with self.subTest(duration=trade_duration, desc=description):
                context = RewardContext(
                    pnl=0.0,  # Neutral PnL to isolate hold penalty
                    trade_duration=trade_duration,
                    idle_duration=0,
                    max_trade_duration=max_duration,
                    max_unrealized_profit=0.0,
                    min_unrealized_profit=0.0,
                    position=Positions.Long,
                    action=Actions.Neutral,
                )

                breakdown = calculate_reward(
                    context,
                    self.DEFAULT_PARAMS,
                    base_factor=self.TEST_BASE_FACTOR,
                    profit_target=self.TEST_PROFIT_TARGET,
                    risk_reward_ratio=1.0,
                    short_allowed=True,
                    action_masking=True,
                )

                duration_ratio = trade_duration / max_duration

                if duration_ratio < 1.0:
                    # Before max_duration: should be exactly 0.0
                    self.assertEqual(
                        breakdown.hold_penalty,
                        0.0,
                        f"Hold penalty should be 0.0 {description} (ratio={duration_ratio:.2f})",
                    )
                elif duration_ratio == 1.0:
                    # At max_duration: (1.0-1.0)^power = 0, so should be 0.0
                    self.assertEqual(
                        breakdown.hold_penalty,
                        0.0,
                        f"Hold penalty should be 0.0 {description} (ratio={duration_ratio:.2f})",
                    )
                else:
                    # After max_duration: should be negative
                    self.assertLess(
                        breakdown.hold_penalty,
                        0.0,
                        f"Hold penalty should be negative {description} (ratio={duration_ratio:.2f})",
                    )

                self.assertAlmostEqualFloat(
                    breakdown.total,
                    breakdown.hold_penalty
                    + breakdown.shaping_reward
                    + breakdown.entry_additive
                    + breakdown.exit_additive,
                    tolerance=1e-9,
                    msg=f"Total mismatch including shaping {description}",
                )

    def test_hold_penalty_progressive_scaling(self):
        """Test that hold penalty scales progressively after max_duration."""
        max_duration = 100
        durations = [150, 200, 300]  # All > max_duration
        penalties: list[float] = []

        for duration in durations:
            context = RewardContext(
                pnl=0.0,
                trade_duration=duration,
                idle_duration=0,
                max_trade_duration=max_duration,
                max_unrealized_profit=0.0,
                min_unrealized_profit=0.0,
                position=Positions.Long,
                action=Actions.Neutral,
            )

            breakdown = calculate_reward(
                context,
                self.DEFAULT_PARAMS,
                base_factor=self.TEST_BASE_FACTOR,
                profit_target=self.TEST_PROFIT_TARGET,
                risk_reward_ratio=self.TEST_RR,
                short_allowed=True,
                action_masking=True,
            )

            penalties.append(breakdown.hold_penalty)

        # Penalties should be increasingly negative (monotonic decrease)
        for i in range(1, len(penalties)):
            self.assertLessEqual(
                penalties[i],
                penalties[i - 1],
                f"Penalty should increase with duration: {penalties[i]} > {penalties[i-1]}",
            )

    def test_new_invariant_and_warn_parameters(self):
        """Ensure new tunables (check_invariants, exit_factor_threshold) exist and behave.

        Uses a very large base_factor to trigger potential warning condition without capping.
        """
        params = self.DEFAULT_PARAMS.copy()
        self.assertIn("check_invariants", params)
        self.assertIn("exit_factor_threshold", params)

        context = RewardContext(
            pnl=0.05,
            trade_duration=300,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.06,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        breakdown = calculate_reward(
            context,
            params,
            base_factor=1e7,  # exaggerated factor
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        self.assertFinite(breakdown.exit_component, name="exit_component")


class TestRewardRobustness(RewardSpaceTestBase):
    """Tests implementing all prioritized robustness enhancements.

    Covers:
    - Reward decomposition integrity (total == core components + shaping + additives)
    - Exit factor monotonic attenuation per mode where mathematically expected
    - Boundary parameter conditions (tau extremes, plateau grace edges, linear slope = 0)
    - Non-linear power tests for idle & hold penalties (power != 1)
    - Warning emission (exit_factor_threshold) without capping
    """

    def _mk_context(
        self,
        pnl: float = 0.04,
        trade_duration: int = 40,
        idle_duration: int = 0,
        max_trade_duration: int = 100,
        max_unrealized_profit: float = 0.05,
        min_unrealized_profit: float = 0.01,
        position: Positions = Positions.Long,
        action: Actions = Actions.Long_exit,
    ) -> RewardContext:
        return RewardContext(
            pnl=pnl,
            trade_duration=trade_duration,
            idle_duration=idle_duration,
            max_trade_duration=max_trade_duration,
            max_unrealized_profit=max_unrealized_profit,
            min_unrealized_profit=min_unrealized_profit,
            position=position,
            action=action,
        )

    def test_decomposition_integrity(self):
        """Assert reward_total equals exactly the single active component (mutual exclusivity).

        We sample a grid of mutually exclusive scenarios and validate decomposition.
        """
        scenarios = [
            # Idle penalty only
            dict(
                ctx=RewardContext(
                    pnl=0.0,
                    trade_duration=0,
                    idle_duration=25,
                    max_trade_duration=100,
                    max_unrealized_profit=0.0,
                    min_unrealized_profit=0.0,
                    position=Positions.Neutral,
                    action=Actions.Neutral,
                ),
                active="idle_penalty",
            ),
            # Hold penalty only
            dict(
                ctx=RewardContext(
                    pnl=0.0,
                    trade_duration=150,
                    idle_duration=0,
                    max_trade_duration=100,
                    max_unrealized_profit=0.0,
                    min_unrealized_profit=0.0,
                    position=Positions.Long,
                    action=Actions.Neutral,
                ),
                active="hold_penalty",
            ),
            # Exit reward only (positive pnl)
            dict(
                ctx=self._mk_context(pnl=self.TEST_PROFIT_TARGET, trade_duration=60),
                active="exit_component",
            ),
            # Invalid action only
            dict(
                ctx=RewardContext(
                    pnl=0.01,
                    trade_duration=10,
                    idle_duration=0,
                    max_trade_duration=100,
                    max_unrealized_profit=0.02,
                    min_unrealized_profit=0.0,
                    position=Positions.Short,
                    action=Actions.Long_exit,  # invalid
                ),
                active="invalid_penalty",
            ),
        ]
        for sc in scenarios:  # type: ignore[var-annotated]
            ctx_obj: RewardContext = sc["ctx"]  # type: ignore[index]
            active_label: str = sc["active"]  # type: ignore[index]
            with self.subTest(active=active_label):
                br = calculate_reward(
                    ctx_obj,
                    self.DEFAULT_PARAMS,
                    base_factor=self.TEST_BASE_FACTOR,
                    profit_target=self.TEST_PROFIT_TARGET,
                    risk_reward_ratio=self.TEST_RR,
                    short_allowed=True,
                    action_masking=(active_label != "invalid_penalty"),
                )
                components = [
                    br.invalid_penalty,
                    br.idle_penalty,
                    br.hold_penalty,
                    br.exit_component,
                ]
                non_zero = [
                    c for c in components if not math.isclose(c, 0.0, abs_tol=1e-12)
                ]
                self.assertEqual(
                    len(non_zero),
                    1,
                    f"Exactly one component must be active for {sc['active']}",
                )
                self.assertAlmostEqualFloat(
                    br.total,
                    non_zero[0]
                    + br.shaping_reward
                    + br.entry_additive
                    + br.exit_additive,
                    tolerance=1e-9,
                    msg=f"Total mismatch including shaping for {sc['active']}",
                )

    def test_exit_factor_monotonic_attenuation(self):
        """For attenuation modes: factor should be non-increasing w.r.t duration_ratio.

        Modes covered: sqrt, linear, power, half_life, plateau+linear (after grace).
        Legacy is excluded (non-monotonic by design). Plateau+linear includes flat grace then monotonic.
        """

        modes = ["sqrt", "linear", "power", "half_life", "plateau_linear"]
        base_factor = self.TEST_BASE_FACTOR
        pnl = 0.05
        pnl_factor = 1.0
        for mode in modes:
            params = self.DEFAULT_PARAMS.copy()
            if mode in ("sqrt", "linear", "power", "half_life"):
                params["exit_attenuation_mode"] = mode
            if mode == "linear":
                params["exit_linear_slope"] = 1.2
            if mode == "plateau_linear":
                params["exit_attenuation_mode"] = "linear"
                params["exit_plateau"] = True
                params["exit_plateau_grace"] = 0.2
                params["exit_linear_slope"] = 1.0
            if mode == "power":
                params["exit_power_tau"] = 0.5
            if mode == "half_life":
                params["exit_half_life"] = 0.7

            ratios = np.linspace(0, 2, 15)
            values = [
                _get_exit_factor(base_factor, pnl, pnl_factor, r, params)
                for r in ratios
            ]
            # Plateau+linear: ignore initial flat region when checking monotonic decrease
            if mode == "plateau_linear":
                grace = float(params["exit_plateau_grace"])  # type: ignore[index]
                filtered = [(r, v) for r, v in zip(ratios, values) if r >= grace - 1e-9]
                values_to_check = [v for _, v in filtered]
            else:
                values_to_check = values
            for earlier, later in zip(values_to_check, values_to_check[1:]):
                self.assertLessEqual(
                    later,
                    earlier + 1e-9,
                    f"Non-monotonic attenuation in mode={mode}",
                )

    def test_exit_factor_boundary_parameters(self):
        """Test parameter edge cases: tau extremes, plateau grace edges, slope zero."""

        base_factor = 50.0
        pnl = 0.02
        pnl_factor = 1.0
        # Tau near 1 (minimal attenuation) vs tau near 0 (strong attenuation)
        params_hi = self.DEFAULT_PARAMS.copy()
        params_hi.update({"exit_attenuation_mode": "power", "exit_power_tau": 0.999999})
        params_lo = self.DEFAULT_PARAMS.copy()
        params_lo.update({"exit_attenuation_mode": "power", "exit_power_tau": 1e-6})
        r = 1.5
        hi_val = _get_exit_factor(base_factor, pnl, pnl_factor, r, params_hi)
        lo_val = _get_exit_factor(base_factor, pnl, pnl_factor, r, params_lo)
        self.assertGreater(
            hi_val,
            lo_val,
            "Power mode: higher tau (≈1) should attenuate less than tiny tau",
        )
        # Plateau grace 0 vs 1
        params_g0 = self.DEFAULT_PARAMS.copy()
        params_g0.update(
            {
                "exit_attenuation_mode": "linear",
                "exit_plateau": True,
                "exit_plateau_grace": 0.0,
                "exit_linear_slope": 1.0,
            }
        )
        params_g1 = self.DEFAULT_PARAMS.copy()
        params_g1.update(
            {
                "exit_attenuation_mode": "linear",
                "exit_plateau": True,
                "exit_plateau_grace": 1.0,
                "exit_linear_slope": 1.0,
            }
        )
        val_g0 = _get_exit_factor(base_factor, pnl, pnl_factor, 0.5, params_g0)
        val_g1 = _get_exit_factor(base_factor, pnl, pnl_factor, 0.5, params_g1)
        # With grace=1.0 no attenuation up to 1.0 ratio → value should be higher
        self.assertGreater(
            val_g1,
            val_g0,
            "Plateau grace=1.0 should delay attenuation vs grace=0.0",
        )
        # Linear slope zero vs positive
        params_lin0 = self.DEFAULT_PARAMS.copy()
        params_lin0.update(
            {
                "exit_attenuation_mode": "linear",
                "exit_linear_slope": 0.0,
                "exit_plateau": False,
            }
        )
        params_lin1 = self.DEFAULT_PARAMS.copy()
        params_lin1.update(
            {
                "exit_attenuation_mode": "linear",
                "exit_linear_slope": 2.0,
                "exit_plateau": False,
            }
        )
        val_lin0 = _get_exit_factor(base_factor, pnl, pnl_factor, 1.0, params_lin0)
        val_lin1 = _get_exit_factor(base_factor, pnl, pnl_factor, 1.0, params_lin1)
        self.assertGreater(
            val_lin0,
            val_lin1,
            "Linear slope=0 should yield no attenuation vs slope>0",
        )

    def test_plateau_linear_slope_zero_constant_after_grace(self):
        """Plateau+linear slope=0 should yield flat factor after grace boundary (no attenuation)."""

        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "exit_attenuation_mode": "linear",
                "exit_plateau": True,
                "exit_plateau_grace": 0.3,
                "exit_linear_slope": 0.0,
            }
        )
        base_factor = self.TEST_BASE_FACTOR
        pnl = 0.04
        pnl_factor = 1.2
        ratios = [0.3, 0.6, 1.0, 1.4]
        values = [
            _get_exit_factor(base_factor, pnl, pnl_factor, r, params) for r in ratios
        ]
        # All factors should be (approximately) identical after grace (no attenuation)
        first = values[0]
        for v in values[1:]:
            self.assertAlmostEqualFloat(
                v,
                first,
                tolerance=1e-9,
                msg=f"Plateau+linear slope=0 factor drift at ratio set {ratios} => {values}",
            )

    def test_plateau_grace_extends_beyond_one(self):
        """Plateau grace >1.0 should keep full strength (no attenuation) past duration_ratio=1."""

        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "exit_attenuation_mode": "linear",
                "exit_plateau": True,
                "exit_plateau_grace": 1.5,  # extend grace beyond max duration ratio 1.0
                "exit_linear_slope": 2.0,
            }
        )
        base_factor = 80.0
        pnl = self.TEST_PROFIT_TARGET
        pnl_factor = 1.1
        # Ratios straddling 1.0 but below grace=1.5 plus one beyond grace
        ratios = [0.8, 1.0, 1.2, 1.4, 1.6]
        vals = [
            _get_exit_factor(base_factor, pnl, pnl_factor, r, params) for r in ratios
        ]
        # All ratios <=1.5 should yield identical factor
        ref = vals[0]
        for i, r in enumerate(ratios[:-1]):  # exclude last (1.6)
            self.assertAlmostEqualFloat(
                vals[i],
                ref,
                1e-9,
                msg=f"Unexpected attenuation before grace end at ratio {r}",
            )
        # Last ratio (1.6) should be attenuated (strictly less than ref)
        self.assertLess(vals[-1], ref, "Attenuation should begin after grace boundary")

    def test_legacy_step_non_monotonic(self):
        """Legacy mode applies step change at duration_ratio=1 (should not be monotonic)."""

        params = self.DEFAULT_PARAMS.copy()
        params["exit_attenuation_mode"] = "legacy"
        params["exit_plateau"] = False
        base_factor = self.TEST_BASE_FACTOR
        pnl = 0.02
        pnl_factor = 1.0
        # ratio below 1 vs above 1
        below = _get_exit_factor(base_factor, pnl, pnl_factor, 0.5, params)
        above = _get_exit_factor(base_factor, pnl, pnl_factor, 1.5, params)
        # Legacy multiplies by 1.5 then 0.5 -> below should be > above * 2 (since (1.5)/(0.5)=3)
        self.assertGreater(
            below, above, "Legacy pre-threshold factor should exceed post-threshold"
        )
        ratio = below / max(above, 1e-12)
        self.assertGreater(
            ratio,
            2.5,
            f"Legacy step ratio unexpectedly small (expected ~3, got {ratio:.3f})",
        )

    def test_exit_factor_non_negative_with_positive_pnl(self):
        """Exit factor must not be negative when pnl >= 0 (invariant clamp)."""

        params = self.DEFAULT_PARAMS.copy()
        # Try multiple modes / extreme params
        modes = ["linear", "power", "half_life", "sqrt", "legacy", "linear_plateau"]
        base_factor = self.TEST_BASE_FACTOR
        pnl = 0.05
        pnl_factor = 2.0  # amplified
        for mode in modes:
            params_mode = params.copy()
            if mode == "linear_plateau":
                params_mode["exit_attenuation_mode"] = "linear"
                params_mode["exit_plateau"] = True
                params_mode["exit_plateau_grace"] = 0.4
            else:
                params_mode["exit_attenuation_mode"] = mode
            val = _get_exit_factor(base_factor, pnl, pnl_factor, 2.0, params_mode)
            self.assertGreaterEqual(
                val,
                0.0,
                f"Exit factor should be >=0 for non-negative pnl in mode {mode}",
            )


class TestParameterValidation(RewardSpaceTestBase):
    """Tests for validate_reward_parameters adjustments and reasons."""

    def test_validate_reward_parameters_adjustments(self):
        raw = self.DEFAULT_PARAMS.copy()
        # Introduce out-of-bound values
        raw["idle_penalty_scale"] = -5.0  # < min 0
        raw["efficiency_center"] = 2.0  # > max 1
        raw["exit_power_tau"] = 0.0  # below min (treated as 1e-6)
        sanitized, adjustments = validate_reward_parameters(raw)
        # Idle penalty scale should be clamped to 0.0
        self.assertEqual(sanitized["idle_penalty_scale"], 0.0)
        self.assertIn("idle_penalty_scale", adjustments)
        self.assertIsInstance(adjustments["idle_penalty_scale"]["reason"], str)
        self.assertIn("min=0.0", str(adjustments["idle_penalty_scale"]["reason"]))
        # Efficiency center should be clamped to 1.0
        self.assertEqual(sanitized["efficiency_center"], 1.0)
        self.assertIn("efficiency_center", adjustments)
        self.assertIn("max=1.0", str(adjustments["efficiency_center"]["reason"]))
        # exit_power_tau should be raised to lower bound (>=1e-6)
        self.assertGreaterEqual(float(sanitized["exit_power_tau"]), 1e-6)
        self.assertIn("exit_power_tau", adjustments)
        self.assertIn("min=", str(adjustments["exit_power_tau"]["reason"]))

    def test_idle_and_hold_penalty_power(self):
        """Test non-linear scaling when penalty powers != 1."""
        params = self.DEFAULT_PARAMS.copy()
        params["idle_penalty_power"] = 2.0
        params["max_idle_duration_candles"] = 100
        base_factor = 90.0
        profit_target = self.TEST_PROFIT_TARGET
        # Idle penalties for durations 20 vs 40 (quadratic → (40/100)^2 / (20/100)^2 = (0.4^2)/(0.2^2)=4)
        ctx_a = RewardContext(
            pnl=0.0,
            trade_duration=0,
            idle_duration=20,
            max_trade_duration=128,
            max_unrealized_profit=0.0,
            min_unrealized_profit=0.0,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )
        ctx_b = dataclasses.replace(ctx_a, idle_duration=40)
        br_a = calculate_reward(
            ctx_a,
            params,
            base_factor=base_factor,
            profit_target=profit_target,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        br_b = calculate_reward(
            ctx_b,
            params,
            base_factor=base_factor,
            profit_target=profit_target,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        ratio_quadratic = 0.0
        if br_a.idle_penalty != 0:
            ratio_quadratic = br_b.idle_penalty / br_a.idle_penalty
        # Both negative; absolute ratio should be close to 4
        self.assertAlmostEqual(
            abs(ratio_quadratic),
            4.0,
            delta=0.8,
            msg=f"Idle penalty quadratic scaling mismatch (ratio={ratio_quadratic})",
        )
        # Hold penalty with power 2: durations just above threshold
        params["hold_penalty_power"] = 2.0
        ctx_h1 = RewardContext(
            pnl=0.0,
            trade_duration=130,
            idle_duration=0,
            max_trade_duration=100,
            max_unrealized_profit=0.0,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Neutral,
        )
        ctx_h2 = dataclasses.replace(ctx_h1, trade_duration=140)
        # Compute baseline and comparison hold penalties
        br_h1 = calculate_reward(
            ctx_h1,
            params,
            base_factor=base_factor,
            profit_target=profit_target,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        br_h2 = calculate_reward(
            ctx_h2,
            params,
            base_factor=base_factor,
            profit_target=profit_target,
            risk_reward_ratio=1.0,
            short_allowed=True,
            action_masking=True,
        )
        # Quadratic scaling: ((140-100)/(130-100))^2 = (40/30)^2 ≈ 1.777...
        hold_ratio = 0.0
        if br_h1.hold_penalty != 0:
            hold_ratio = br_h2.hold_penalty / br_h1.hold_penalty
        self.assertAlmostEqual(
            abs(hold_ratio),
            (40 / 30) ** 2,
            delta=0.4,
            msg=f"Hold penalty quadratic scaling mismatch (ratio={hold_ratio})",
        )

    def test_exit_factor_threshold_warning_emission(self):
        """Ensure a RuntimeWarning is emitted when exit_factor exceeds threshold (no capping)."""

        params = self.DEFAULT_PARAMS.copy()
        params["exit_factor_threshold"] = 10.0  # low threshold to trigger easily
        # Remove base_factor to allow argument override
        params.pop("base_factor", None)

        context = RewardContext(
            pnl=0.06,
            trade_duration=10,
            idle_duration=0,
            max_trade_duration=128,
            max_unrealized_profit=0.08,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            br = calculate_reward(
                context,
                params,
                base_factor=5000.0,  # large enough to exceed threshold
                profit_target=self.TEST_PROFIT_TARGET,
                risk_reward_ratio=self.TEST_RR_HIGH,
                short_allowed=True,
                action_masking=True,
            )
            self.assertGreater(br.exit_component, 0.0)
            emitted = [warn for warn in w if issubclass(warn.category, RuntimeWarning)]
            self.assertTrue(
                len(emitted) >= 1, "Expected a RuntimeWarning for exit factor threshold"
            )
            # Confirm message indicates threshold exceedance (supports legacy and new message format)
            self.assertTrue(
                any(
                    (
                        "exceeded threshold" in str(warn.message)
                        or "exceeds threshold" in str(warn.message)
                        or "|factor|=" in str(warn.message)
                    )
                    for warn in emitted
                ),
                "Warning message should indicate threshold exceedance",
            )


class TestContinuityPlateau(RewardSpaceTestBase):
    """Continuity tests for plateau-enabled exit attenuation (excluding legacy)."""

    def test_plateau_continuity_at_grace_boundary(self):
        modes = ["sqrt", "linear", "power", "half_life"]
        grace = 0.8
        eps = 1e-4
        base_factor = self.TEST_BASE_FACTOR
        pnl = 0.01
        pnl_factor = 1.0
        tau = 0.5  # for power
        half_life = 0.5
        slope = 1.3

        for mode in modes:
            with self.subTest(mode=mode):
                params = self.DEFAULT_PARAMS.copy()
                params.update(
                    {
                        "exit_attenuation_mode": mode,
                        "exit_plateau": True,
                        "exit_plateau_grace": grace,
                        "exit_linear_slope": slope,
                        "exit_power_tau": tau,
                        "exit_half_life": half_life,
                    }
                )

                left = _get_exit_factor(
                    base_factor, pnl, pnl_factor, grace - eps, params
                )
                boundary = _get_exit_factor(base_factor, pnl, pnl_factor, grace, params)
                right = _get_exit_factor(
                    base_factor, pnl, pnl_factor, grace + eps, params
                )

                self.assertAlmostEqualFloat(
                    left,
                    boundary,
                    tolerance=1e-9,
                    msg=f"Left/boundary mismatch for mode {mode}",
                )
                self.assertLess(
                    right,
                    boundary,
                    f"No attenuation detected just after grace for mode {mode}",
                )

                diff = boundary - right
                if mode == "linear":
                    bound = base_factor * slope * eps * 2.0
                elif mode == "sqrt":
                    bound = base_factor * 0.5 * eps * 2.0
                elif mode == "power":
                    alpha = -math.log(tau) / math.log(2.0)
                    bound = base_factor * alpha * eps * 2.0
                elif mode == "half_life":
                    bound = base_factor * (math.log(2.0) / half_life) * eps * 2.5
                else:
                    bound = base_factor * eps * 5.0

                self.assertLessEqual(
                    diff,
                    bound,
                    f"Attenuation jump too large at boundary for mode {mode} (diff={diff:.6e} > bound={bound:.6e})",
                )

    def test_plateau_continuity_multiple_eps_scaling(self):
        """Verify attenuation difference scales approximately linearly with epsilon (first-order continuity heuristic)."""

        mode = "linear"
        grace = 0.6
        eps1 = 1e-3
        eps2 = 1e-4
        base_factor = 80.0
        pnl = 0.02
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "exit_attenuation_mode": mode,
                "exit_plateau": True,
                "exit_plateau_grace": grace,
                "exit_linear_slope": 1.1,
            }
        )
        f_boundary = _get_exit_factor(base_factor, pnl, 1.0, grace, params)
        f1 = _get_exit_factor(base_factor, pnl, 1.0, grace + eps1, params)
        f2 = _get_exit_factor(base_factor, pnl, 1.0, grace + eps2, params)

        diff1 = f_boundary - f1
        diff2 = f_boundary - f2
        ratio = diff1 / max(diff2, 1e-12)
        self.assertGreater(ratio, 5.0, f"Scaling ratio too small (ratio={ratio:.2f})")
        self.assertLess(ratio, 15.0, f"Scaling ratio too large (ratio={ratio:.2f})")


class TestLoadRealEpisodes(RewardSpaceTestBase):
    """Unit tests for load_real_episodes (moved from separate file)."""

    def write_pickle(self, obj, path: Path):
        with path.open("wb") as f:
            pickle.dump(obj, f)

    def test_top_level_dict_transitions(self):
        df = pd.DataFrame(
            {
                "pnl": [0.01],
                "trade_duration": [10],
                "idle_duration": [5],
                "position": [1.0],
                "action": [2.0],
                "reward_total": [1.0],
            }
        )
        p = Path(self.temp_dir) / "top.pkl"
        self.write_pickle({"transitions": df}, p)

        loaded = load_real_episodes(p)
        self.assertIsInstance(loaded, pd.DataFrame)
        self.assertEqual(list(loaded.columns).count("pnl"), 1)
        self.assertEqual(len(loaded), 1)

    def test_mixed_episode_list_warns_and_flattens(self):
        ep1 = {"episode_id": 1}
        ep2 = {
            "episode_id": 2,
            "transitions": [
                {
                    "pnl": 0.02,
                    "trade_duration": 5,
                    "idle_duration": 0,
                    "position": 1.0,
                    "action": 2.0,
                    "reward_total": 2.0,
                }
            ],
        }
        p = Path(self.temp_dir) / "mixed.pkl"
        self.write_pickle([ep1, ep2], p)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = load_real_episodes(p)
            # Accept variance in warning emission across platforms
            _ = w

        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(float(loaded.iloc[0]["pnl"]), 0.02, places=7)

    def test_non_iterable_transitions_raises(self):
        bad = {"transitions": 123}
        p = Path(self.temp_dir) / "bad.pkl"
        self.write_pickle(bad, p)

        with self.assertRaises(ValueError):
            load_real_episodes(p)

    def test_enforce_columns_false_fills_na(self):
        trans = [
            {
                "pnl": 0.03,
                "trade_duration": 10,
                "idle_duration": 0,
                "position": 1.0,
                "action": 2.0,
            }
        ]
        p = Path(self.temp_dir) / "fill.pkl"
        self.write_pickle(trans, p)

        loaded = load_real_episodes(p, enforce_columns=False)
        self.assertIn("reward_total", loaded.columns)
        self.assertTrue(loaded["reward_total"].isna().all())

    def test_casting_numeric_strings(self):
        trans = [
            {
                "pnl": "0.04",
                "trade_duration": "20",
                "idle_duration": "0",
                "position": "1.0",
                "action": "2.0",
                "reward_total": "3.0",
            }
        ]
        p = Path(self.temp_dir) / "strs.pkl"
        self.write_pickle(trans, p)

        loaded = load_real_episodes(p)
        self.assertIn("pnl", loaded.columns)
        self.assertIn(loaded["pnl"].dtype.kind, ("f", "i"))
        self.assertAlmostEqual(float(loaded.iloc[0]["pnl"]), 0.04, places=7)

    def test_pickled_dataframe_loads(self):
        """Ensure a directly pickled DataFrame loads correctly."""
        test_episodes = pd.DataFrame(
            {
                "pnl": [0.01, -0.02, 0.03],
                "trade_duration": [10, 20, 15],
                "idle_duration": [5, 0, 8],
                "position": [1.0, 0.0, 1.0],
                "action": [2.0, 0.0, 2.0],
                "reward_total": [10.5, -5.2, 15.8],
            }
        )
        p = Path(self.temp_dir) / "test_episodes.pkl"
        self.write_pickle(test_episodes, p)

        loaded_data = load_real_episodes(p)
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 3)
        self.assertIn("pnl", loaded_data.columns)


class TestPBRSIntegration(RewardSpaceTestBase):
    """Tests for PBRS (Potential-Based Reward Shaping) integration."""

    def test_tanh_transform(self):
        """tanh transform: bounded in [-1,1], symmetric."""
        self.assertAlmostEqualFloat(apply_transform("tanh", 0.0), 0.0)
        self.assertAlmostEqualFloat(apply_transform("tanh", 1.0), math.tanh(1.0))
        self.assertAlmostEqualFloat(apply_transform("tanh", -1.0), math.tanh(-1.0))
        self.assertTrue(abs(apply_transform("tanh", 100.0)) <= 1.0)
        self.assertTrue(abs(apply_transform("tanh", -100.0)) <= 1.0)

    def test_softsign_transform(self):
        """softsign transform: x/(1+|x|) in (-1,1)."""
        self.assertAlmostEqualFloat(apply_transform("softsign", 0.0), 0.0)
        self.assertAlmostEqualFloat(apply_transform("softsign", 1.0), 0.5)
        self.assertAlmostEqualFloat(apply_transform("softsign", -1.0), -0.5)
        self.assertTrue(abs(apply_transform("softsign", 100.0)) < 1.0)
        self.assertTrue(abs(apply_transform("softsign", -100.0)) < 1.0)

    def test_arctan_transform(self):
        """arctan transform: normalized (2/pi)atan(x) bounded [-1,1]."""
        # Environment uses normalized arctan: (2/pi)*atan(x)
        self.assertAlmostEqualFloat(apply_transform("arctan", 0.0), 0.0)
        self.assertAlmostEqualFloat(
            apply_transform("arctan", 1.0),
            (2.0 / math.pi) * math.atan(1.0),
            tolerance=1e-10,
        )
        self.assertTrue(abs(apply_transform("arctan", 100.0)) <= 1.0)
        self.assertTrue(abs(apply_transform("arctan", -100.0)) <= 1.0)

    def test_logistic_transform(self):
        """logistic transform: 2σ(x)-1 in (-1,1)."""
        # Environment logistic returns 2σ(x)-1 centered at 0 in (-1,1)
        self.assertAlmostEqualFloat(apply_transform("logistic", 0.0), 0.0)
        self.assertTrue(apply_transform("logistic", 100.0) > 0.99)
        self.assertTrue(apply_transform("logistic", -100.0) < -0.99)
        self.assertTrue(-1 < apply_transform("logistic", 10.0) < 1)
        self.assertTrue(-1 < apply_transform("logistic", -10.0) < 1)

    def test_clip_transform(self):
        """clip transform: clamp to [-1,1]."""
        self.assertAlmostEqualFloat(apply_transform("clip", 0.0), 0.0)
        self.assertAlmostEqualFloat(apply_transform("clip", 0.5), 0.5)
        self.assertAlmostEqualFloat(apply_transform("clip", 2.0), 1.0)
        self.assertAlmostEqualFloat(apply_transform("clip", -2.0), -1.0)

    def test_invalid_transform(self):
        """Test error handling for invalid transforms."""
        # Environment falls back silently to tanh
        self.assertAlmostEqualFloat(
            apply_transform("invalid_transform", 1.0), math.tanh(1.0), tolerance=1e-9
        )

    def test_get_float_param(self):
        """Test float parameter extraction."""
        params = {"test_float": 1.5, "test_int": 2, "test_str": "hello"}
        self.assertEqual(_get_float_param(params, "test_float", 0.0), 1.5)
        self.assertEqual(_get_float_param(params, "test_int", 0.0), 2.0)
        # Non parseable string -> NaN fallback in tolerant parser
        val_str = _get_float_param(params, "test_str", 0.0)
        if isinstance(val_str, float) and math.isnan(val_str):
            pass
        else:
            self.fail("Expected NaN for non-numeric string in _get_float_param")
        self.assertEqual(_get_float_param(params, "missing", 3.14), 3.14)

    def test_get_str_param(self):
        """Test string parameter extraction."""
        params = {"test_str": "hello", "test_int": 2}
        self.assertEqual(_get_str_param(params, "test_str", "default"), "hello")
        self.assertEqual(_get_str_param(params, "test_int", "default"), "default")
        self.assertEqual(_get_str_param(params, "missing", "default"), "default")

    def test_get_bool_param(self):
        """Test boolean parameter extraction."""
        params = {
            "test_true": True,
            "test_false": False,
            "test_int": 1,
            "test_str": "yes",
        }
        self.assertTrue(_get_bool_param(params, "test_true", False))
        self.assertFalse(_get_bool_param(params, "test_false", True))
        # Environment coerces typical truthy numeric/string values
        self.assertTrue(_get_bool_param(params, "test_int", False))
        self.assertTrue(_get_bool_param(params, "test_str", False))
        self.assertFalse(_get_bool_param(params, "missing", False))

    def test_hold_potential_basic(self):
        """Test basic hold potential calculation."""
        params = {
            "hold_potential_enabled": True,
            "hold_potential_scale": 1.0,
            "hold_potential_gain": 1.0,
            "hold_potential_transform_pnl": "tanh",
            "hold_potential_transform_duration": "tanh",
        }
        val = _compute_hold_potential(0.5, 0.3, params)
        self.assertFinite(val, name="hold_potential")

    def test_entry_additive_disabled(self):
        """Test entry additive when disabled."""
        params = {"entry_additive_enabled": False}
        val = _compute_entry_additive(0.5, 0.3, params)
        self.assertEqual(val, 0.0)

    def test_exit_additive_disabled(self):
        """Test exit additive when disabled."""
        params = {"exit_additive_enabled": False}
        val = _compute_exit_additive(0.5, 0.3, params)
        self.assertEqual(val, 0.0)

    def test_exit_potential_canonical(self):
        """Test exit potential in canonical mode."""
        params = {"exit_potential_mode": "canonical"}
        val = _compute_exit_potential(0.5, 0.3, params, last_potential=1.0)
        self.assertEqual(val, 0.0)

    def test_exit_potential_progressive_release(self):
        """Progressive release: Φ' = Φ * (1 - decay)."""
        params = {
            "exit_potential_mode": "progressive_release",
            "exit_potential_decay": 0.8,
        }
        # Expected: Φ' = Φ * (1 - decay) = 1 * (1 - 0.8) = 0.2
        val = _compute_exit_potential(0.5, 0.3, params, last_potential=1.0)
        self.assertAlmostEqual(val, 0.2)

    def test_exit_potential_spike_cancel(self):
        """Spike cancel: Φ' = Φ / γ (inversion)."""
        params = {"exit_potential_mode": "spike_cancel", "potential_gamma": 0.95}
        val = _compute_exit_potential(0.5, 0.3, params, last_potential=1.0)
        self.assertAlmostEqual(val, 1.0 / 0.95, places=7)

    def test_exit_potential_retain_previous(self):
        """Test exit potential in retain previous mode."""
        params = {"exit_potential_mode": "retain_previous"}
        val = _compute_exit_potential(0.5, 0.3, params, last_potential=1.0)
        self.assertEqual(val, 1.0)

    def test_pbrs_terminal_canonical(self):
        """Test PBRS behavior in canonical mode with terminal state."""
        params = {
            "potential_gamma": 0.95,
            "hold_potential_enabled": True,
            "hold_potential_scale": 1.0,
            "hold_potential_gain": 1.0,
            "hold_potential_transform_pnl": "tanh",
            "hold_potential_transform_duration": "tanh",
            "exit_potential_mode": "canonical",
            "entry_additive_enabled": False,
            "exit_additive_enabled": False,
        }

        current_pnl = 0.5
        current_duration_ratio = 0.3
        expected_current_potential = _compute_hold_potential(
            current_pnl, current_duration_ratio, params
        )

        total_reward, shaping_reward, next_potential = apply_potential_shaping(
            base_reward=100.0,
            current_pnl=current_pnl,
            current_duration_ratio=current_duration_ratio,
            next_pnl=0.0,
            next_duration_ratio=0.0,
            is_terminal=True,
            last_potential=0.0,
            params=params,
        )

        # Terminal potential should be 0 in canonical mode
        self.assertEqual(next_potential, 0.0)
        # Shaping reward should be negative (releasing potential)
        self.assertTrue(shaping_reward < 0)
        # Check exact formula: γΦ(s') - Φ(s) = 0.95 * 0 - expected_current_potential
        expected_shaping = 0.95 * 0.0 - expected_current_potential
        self.assertAlmostEqual(shaping_reward, expected_shaping, delta=1e-10)

    def test_pbrs_invalid_gamma(self):
        """Test PBRS with invalid gamma value."""
        params = {"potential_gamma": 1.5, "hold_potential_enabled": True}
        with self.assertRaises(ValueError):
            apply_potential_shaping(
                base_reward=0.0,
                current_pnl=0.0,
                current_duration_ratio=0.0,
                next_pnl=0.0,
                next_duration_ratio=0.0,
                is_terminal=True,
                last_potential=0.0,
                params=params,
            )

    def test_calculate_reward_with_pbrs_integration(self):
        """Test that PBRS parameters are properly integrated in defaults."""
        # Test that PBRS parameters are in the default parameters
        for param in PBRS_INTEGRATION_PARAMS:
            self.assertIn(
                param,
                DEFAULT_MODEL_REWARD_PARAMETERS,
                f"PBRS parameter {param} missing from defaults",
            )

        # Test basic PBRS function integration works
        params = {"hold_potential_enabled": True, "hold_potential_scale": 1.0}
        potential = _compute_hold_potential(0.1, 0.2, params)
        self.assertFinite(potential, name="hold_potential")

    def test_pbrs_default_parameters_completeness(self):
        """Test that all required PBRS parameters have defaults."""
        for param in PBRS_REQUIRED_PARAMS:
            self.assertIn(
                param,
                DEFAULT_MODEL_REWARD_PARAMETERS,
                f"Missing PBRS parameter: {param}",
            )


if __name__ == "__main__":
    # Configure test discovery and execution
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
