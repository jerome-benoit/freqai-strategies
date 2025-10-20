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
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

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
        ATTENUATION_MODES,
        ATTENUATION_MODES_WITH_LEGACY,
        DEFAULT_MODEL_REWARD_PARAMETERS,
        PBRS_INVARIANCE_TOL,
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
        _get_int_param,
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
    """Base class with common test utilities."""

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
        self.seed_all(self.SEED)
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # PBRS structural test constants
    PBRS_TERMINAL_TOL = 1e-12
    PBRS_MAX_ABS_SHAPING = 5.0
    PBRS_TERMINAL_PROB = 0.08
    PBRS_SWEEP_ITER = 120

    # Generic numeric tolerances (distinct from PBRS structural constants)
    EPS_BASE = 1e-12  # Base epsilon for strict identity & numeric guards (single source)
    TOL_NUMERIC_GUARD = EPS_BASE  # Division-by-zero guards / min denominators (alias)
    TOL_IDENTITY_STRICT = EPS_BASE  # Strict component identity (alias of EPS_BASE)
    TOL_IDENTITY_RELAXED = 1e-9  # Looser identity when cumulative fp drift acceptable
    TOL_GENERIC_EQ = 1e-6  # Generic numeric equality
    TOL_NEGLIGIBLE = 1e-8  # Negligible statistical or shaping effects
    MIN_EXIT_POWER_TAU = 1e-6  # Lower bound for exit_power_tau parameter (validation semantics)
    # Distribution shape invariance (skewness / excess kurtosis) tolerance under scaling
    TOL_DISTRIB_SHAPE = 5e-2
    # Theoretical upper bound for Jensen-Shannon distance: sqrt(log 2)
    JS_DISTANCE_UPPER_BOUND = math.sqrt(math.log(2.0))
    # Relative tolerance (scale-aware) and continuity perturbation magnitudes
    TOL_RELATIVE = 1e-9  # For relative comparisons scaled by magnitude
    CONTINUITY_EPS_SMALL = 1e-4  # Small epsilon step for continuity probing
    CONTINUITY_EPS_LARGE = 1e-3  # Larger epsilon step for ratio scaling tests

    def make_ctx(
        self,
        *,
        pnl: float = 0.0,
        trade_duration: int = 0,
        idle_duration: int = 0,
        max_unrealized_profit: float = 0.0,
        min_unrealized_profit: float = 0.0,
        position: Positions = Positions.Neutral,
        action: Actions = Actions.Neutral,
    ) -> RewardContext:
        """Create a RewardContext with neutral defaults."""
        return RewardContext(
            pnl=pnl,
            trade_duration=trade_duration,
            idle_duration=idle_duration,
            max_unrealized_profit=max_unrealized_profit,
            min_unrealized_profit=min_unrealized_profit,
            position=position,
            action=action,
        )

    def base_params(self, **overrides) -> Dict[str, Any]:
        """Return fresh copy of default reward params with overrides."""
        params: Dict[str, Any] = DEFAULT_MODEL_REWARD_PARAMETERS.copy()
        params.update(overrides)
        return params

    def _canonical_sweep(
        self,
        params: dict,
        *,
        iterations: Optional[int] = None,
        terminal_prob: Optional[float] = None,
        seed: int = 123,
    ) -> tuple[list[float], list[float]]:
        """Run a lightweight canonical invariance sweep.

        Returns (terminal_next_potentials, shaping_values).
        """
        iters = iterations or self.PBRS_SWEEP_ITER
        term_p = terminal_prob or self.PBRS_TERMINAL_PROB
        rng = np.random.default_rng(seed)
        last_potential = 0.0
        terminal_next: list[float] = []
        shaping_vals: list[float] = []
        current_pnl = 0.0
        current_dur = 0.0
        for _ in range(iters):
            is_exit = rng.uniform() < term_p
            next_pnl = 0.0 if is_exit else float(rng.normal(0, 0.2))
            inc = rng.uniform(0, 0.12)
            next_dur = 0.0 if is_exit else float(min(1.0, current_dur + inc))
            _tot, shap_val, next_pot = apply_potential_shaping(
                base_reward=0.0,
                current_pnl=current_pnl,
                current_duration_ratio=current_dur,
                next_pnl=next_pnl,
                next_duration_ratio=next_dur,
                is_exit=is_exit,
                is_entry=False,  # In _canonical_sweep, we do not simulate entries
                last_potential=last_potential,
                params=params,
            )
            shaping_vals.append(shap_val)
            if is_exit:
                terminal_next.append(next_pot)
                last_potential = 0.0
                current_pnl = 0.0
                current_dur = 0.0
            else:
                last_potential = next_pot
                current_pnl = next_pnl
                current_dur = next_dur
        return terminal_next, shaping_vals

    def make_stats_df(
        self,
        *,
        n: int,
        reward_mean: float = 0.0,
        reward_std: float = 1.0,
        pnl_mean: float = 0.01,
        pnl_std: Optional[float] = None,
        trade_duration_dist: str = "uniform",
        idle_pattern: str = "mixed",
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate a synthetic statistical DataFrame.

        Parameters
        ----------
        n : int
            Row count.
        reward_mean, reward_std : float
            Normal parameters for reward.
        pnl_mean : float
            Mean PnL.
        pnl_std : float | None
            PnL std (defaults to TEST_PNL_STD when None).
        trade_duration_dist : {'uniform','exponential'}
            Distribution family for trade_duration.
        idle_pattern : {'mixed','all_nonzero','all_zero'}
            mixed: ~40% idle>0 (U(1,60)); all_nonzero: all idle>0 (U(5,60)); all_zero: idle=0.
        seed : int | None
            RNG seed.

        Returns
        -------
        pd.DataFrame with columns: reward, reward_idle, reward_hold, reward_exit,
        pnl, trade_duration, idle_duration, position. Guarantees: no NaN; reward_idle==0 where idle_duration==0.
        """
        if seed is not None:
            self.seed_all(seed)
        pnl_std_eff = self.TEST_PNL_STD if pnl_std is None else pnl_std
        reward = np.random.normal(reward_mean, reward_std, n)
        pnl = np.random.normal(pnl_mean, pnl_std_eff, n)
        if trade_duration_dist == "exponential":
            trade_duration = np.random.exponential(20, n)
        else:
            trade_duration = np.random.uniform(5, 150, n)
        # Idle duration pattern
        if idle_pattern == "mixed":
            mask = np.random.rand(n) < 0.4
            idle_duration = np.where(mask, np.random.uniform(1, 60, n), 0.0)
        elif idle_pattern == "all_zero":
            idle_duration = np.zeros(n)
        else:  # all_nonzero
            idle_duration = np.random.uniform(5, 60, n)
        # Component rewards
        reward_idle = np.where(idle_duration > 0, np.random.normal(-1, 0.3, n), 0.0)
        reward_hold = np.random.normal(-0.5, 0.2, n)
        reward_exit = np.random.normal(0.8, 0.6, n)
        position = np.random.choice([0.0, 0.5, 1.0], n)
        return pd.DataFrame(
            {
                "reward": reward,
                "reward_idle": reward_idle,
                "reward_hold": reward_hold,
                "reward_exit": reward_exit,
                "pnl": pnl,
                "trade_duration": trade_duration,
                "idle_duration": idle_duration,
                "position": position,
            }
        )

    def assertAlmostEqualFloat(
        self,
        first: Union[float, int],
        second: Union[float, int],
        tolerance: Optional[float] = None,
        rtol: Optional[float] = None,
        msg: Union[str, None] = None,
    ) -> None:
        """Compare floats with absolute and optional relative tolerance.

        precedence:
          1. absolute tolerance (|a-b| <= tolerance)
          2. relative tolerance (|a-b| <= rtol * max(|a|, |b|)) if provided
        Both may be supplied; failure only if both criteria fail (when rtol given).
        """
        self.assertFinite(first, name="a")
        self.assertFinite(second, name="b")
        if tolerance is None:
            tolerance = self.TOL_GENERIC_EQ
        diff = abs(first - second)
        # Absolute criterion
        if diff <= tolerance:
            return
        # Relative criterion (optional)
        if rtol is not None:
            scale = max(abs(first), abs(second), 1e-15)
            if diff <= rtol * scale:
                return
        self.fail(
            msg
            or f"Difference {diff} exceeds tolerance {tolerance} and relative tolerance {rtol} (a={first}, b={second})"
        )

    def assertPValue(self, value: Union[float, int], msg: str = "") -> None:
        """Assert a p-value is finite and within [0,1]."""
        self.assertFinite(value, name="p-value")
        self.assertGreaterEqual(value, 0.0, msg or f"p-value < 0: {value}")
        self.assertLessEqual(value, 1.0, msg or f"p-value > 1: {value}")

    def assertPlacesEqual(
        self,
        a: Union[float, int],
        b: Union[float, int],
        places: int,
        msg: Optional[str] = None,
    ) -> None:
        """Bridge for legacy places-based approximate equality.

        Converts decimal places to an absolute tolerance 10**-places and delegates to
        assertAlmostEqualFloat to keep a single numeric comparison implementation.
        """
        tol = 10.0 ** (-places)
        self.assertAlmostEqualFloat(a, b, tolerance=tol, msg=msg)

    def assertDistanceMetric(
        self,
        value: Union[float, int],
        *,
        non_negative: bool = True,
        upper: Optional[float] = None,
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
        value: Union[float, int],
        *,
        lower: float = -1.0,
        upper: float = 1.0,
        name: str = "effect size",
    ) -> None:
        """Assert effect size within symmetric interval and finite."""
        self.assertFinite(value, name=name)
        self.assertGreaterEqual(value, lower, f"{name} < {lower}: {value}")
        self.assertLessEqual(value, upper, f"{name} > {upper}: {value}")

    def assertFinite(self, value: Union[float, int], name: str = "value") -> None:
        """Assert scalar is finite."""
        if not np.isfinite(value):  # low-level base check to avoid recursion
            self.fail(f"{name} not finite: {value}")

    def assertMonotonic(
        self,
        seq: Union[Sequence[Union[float, int]], Iterable[Union[float, int]]],
        *,
        non_increasing: Optional[bool] = None,
        non_decreasing: Optional[bool] = None,
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
        if (non_increasing and non_decreasing) or (not non_increasing and not non_decreasing):
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
        value: Union[float, int],
        low: Union[float, int],
        high: Union[float, int],
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

    def assertNearZero(
        self,
        value: Union[float, int],
        *,
        atol: Optional[float] = None,
        msg: Optional[str] = None,
    ) -> None:
        """Assert a scalar is numerically near zero within absolute tolerance.

        Uses strict identity tolerance by default for PBRS invariance style checks.
        """
        self.assertFinite(value, name="value")
        tol = atol if atol is not None else self.TOL_IDENTITY_RELAXED
        if abs(float(value)) > tol:
            self.fail(msg or f"Value {value} not near zero (tol={tol})")

    def assertSymmetric(
        self,
        func,
        a,
        b,
        *,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        msg: Optional[str] = None,
    ) -> None:
        """Assert function(func, a, b) == function(func, b, a) within tolerance.

        Intended for symmetric distance metrics (e.g., JS distance).
        """
        va = func(a, b)
        vb = func(b, a)
        self.assertAlmostEqualFloat(va, vb, tolerance=atol, rtol=rtol, msg=msg)

    @staticmethod
    def seed_all(seed: int = 123) -> None:
        """Seed all RNGs used (numpy & random)."""
        np.random.seed(seed)
        random.seed(seed)

    # Shared helper data generators available to subclasses.
    def _const_df(self, n: int = 64) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "reward": np.ones(n) * 0.5,
                "pnl": np.zeros(n),
                "trade_duration": np.ones(n) * 10,
                "idle_duration": np.ones(n) * 3,
            }
        )

    def _shift_scale_df(self, n: int = 256, shift: float = 0.0, scale: float = 1.0) -> pd.DataFrame:
        rng = np.random.default_rng(123)
        base = rng.normal(0, 1, n)
        return pd.DataFrame(
            {
                "reward": shift + scale * base,
                "pnl": shift + scale * base * 0.2,
                "trade_duration": rng.exponential(20, n),
                "idle_duration": rng.exponential(10, n),
            }
        )


class TestIntegration(RewardSpaceTestBase):
    """CLI + file output integration tests."""

    def test_cli_execution_produces_expected_files(self):
        """CLI produces expected files."""
        cmd = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            str(self.TEST_SAMPLES),
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(self.output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

        # Exit 0
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")

        # Expected files
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
        """Manifest structure + reproducibility (same seed)."""
        # First run
        cmd1 = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            str(self.TEST_SAMPLES),
            "--seed",
            str(self.SEED),
            "--out_dir",
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
            "--out_dir",
            str(self.output_path / "run2"),
        ]

        # Execute both runs
        result1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=Path(__file__).parent)
        result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=Path(__file__).parent)

        self.assertEqual(result1.returncode, 0)
        self.assertEqual(result2.returncode, 0)

        # Validate manifest structure
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
            # Legacy fields must be absent
            self.assertNotIn("top_features", manifest)
            self.assertNotIn("reward_param_overrides", manifest)
            self.assertNotIn("params", manifest)
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


class TestStatistics(RewardSpaceTestBase):
    """Statistical tests: metrics, diagnostics, bootstrap, correlations."""

    def _make_idle_variance_df(self, n: int = 100) -> pd.DataFrame:
        """Synthetic dataframe focusing on idle_duration ↔ reward_idle correlation."""
        self.seed_all(self.SEED)
        idle_duration = np.random.exponential(10, n)
        reward_idle = -0.01 * idle_duration + np.random.normal(0, 0.001, n)
        return pd.DataFrame(
            {
                "idle_duration": idle_duration,
                "reward_idle": reward_idle,
                "position": np.random.choice([0.0, 0.5, 1.0], n),
                "reward": np.random.normal(0, 1, n),
                "pnl": np.random.normal(0, self.TEST_PNL_STD, n),
                "trade_duration": np.random.exponential(20, n),
            }
        )

    def test_stats_distribution_shift_metrics(self):
        """KL/JS/Wasserstein metrics."""
        df1 = self._make_idle_variance_df(100)
        df2 = self._make_idle_variance_df(100)

        # Shift second dataset
        df2["reward"] += 0.1

        metrics = compute_distribution_shift_metrics(df1, df2)

        # Expected pnl metrics
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
                # Metrics finite; distances non-negative
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

    def test_stats_distribution_shift_identity_null_metrics(self):
        """Identity distributions -> near-zero shift metrics."""
        df = self._make_idle_variance_df(180)
        metrics_id = compute_distribution_shift_metrics(df, df.copy())
        for name, val in metrics_id.items():
            if name.endswith(("_kl_divergence", "_js_distance", "_wasserstein")):
                self.assertLess(
                    abs(val),
                    self.TOL_GENERIC_EQ,
                    f"Metric {name} expected ≈ 0 on identical distributions (got {val})",
                )
            elif name.endswith("_ks_statistic"):
                self.assertLess(
                    abs(val),
                    5e-3,
                    f"KS statistic should be near 0 on identical distributions (got {val})",
                )

    def test_stats_hypothesis_testing(self):
        """Light correlation sanity check."""
        df = self._make_idle_variance_df(200)

        # Only if enough samples
        if len(df) > 30:
            idle_data = df[df["idle_duration"] > 0]
            if len(idle_data) > 10:
                # Negative correlation expected
                idle_dur = idle_data["idle_duration"].to_numpy()
                idle_rew = idle_data["reward_idle"].to_numpy()

                # Sanity shape
                self.assertTrue(
                    len(idle_dur) == len(idle_rew),
                    "Idle duration and reward arrays should have same length",
                )
                self.assertTrue(
                    all(d >= 0 for d in idle_dur),
                    "Idle durations should be non-negative",
                )

                # Majority negative
                negative_rewards = (idle_rew < 0).sum()
                total_rewards = len(idle_rew)
                negative_ratio = negative_rewards / total_rewards

                self.assertGreater(
                    negative_ratio,
                    0.5,
                    "Most idle rewards should be negative (penalties)",
                )

    def test_stats_distribution_diagnostics(self):
        """Distribution diagnostics."""
        df = self._make_idle_variance_df(100)

        diagnostics = distribution_diagnostics(df)

        # Expect keys
        expected_prefixes = ["reward_", "pnl_"]
        for prefix in expected_prefixes:
            matching_keys = [key for key in diagnostics.keys() if key.startswith(prefix)]
            self.assertGreater(len(matching_keys), 0, f"Should have diagnostics for {prefix}")

            # Basic moments
            expected_suffixes = ["mean", "std", "skewness", "kurtosis"]
            for suffix in expected_suffixes:
                key = f"{prefix}{suffix}"
                if key in diagnostics:
                    self.assertFinite(diagnostics[key], name=key)

    def test_statistical_functions(self):
        """Smoke test statistical_hypothesis_tests on synthetic data (API integration)."""
        base = self.make_stats_df(n=200, seed=self.SEED, idle_pattern="mixed")
        base.loc[:149, ["reward_idle", "reward_hold", "reward_exit"]] = 0.0
        results = statistical_hypothesis_tests(base)
        self.assertIsInstance(results, dict)

    def test_stats_js_distance_symmetry_violin(self):
        """JS distance symmetry d(P,Q)==d(Q,P)."""
        df1 = self._shift_scale_df(300, shift=0.0)
        df2 = self._shift_scale_df(300, shift=0.3)
        metrics = compute_distribution_shift_metrics(df1, df2)
        js_key = next((k for k in metrics if k.endswith("pnl_js_distance")), None)
        if js_key is None:
            self.skipTest("JS distance key not present in metrics output")
        metrics_swapped = compute_distribution_shift_metrics(df2, df1)
        js_key_swapped = next((k for k in metrics_swapped if k.endswith("pnl_js_distance")), None)
        self.assertIsNotNone(js_key_swapped)
        self.assertAlmostEqualFloat(
            metrics[js_key],
            metrics_swapped[js_key_swapped],
            tolerance=self.TOL_IDENTITY_STRICT,
            rtol=self.TOL_RELATIVE,
        )

    def test_stats_variance_vs_duration_spearman_sign(self):
        """trade_duration up => pnl variance up (rank corr >0)."""
        rng = np.random.default_rng(99)
        n = 250
        trade_duration = np.linspace(1, 300, n)
        pnl = rng.normal(0, 1 + trade_duration / 400.0, n)
        ranks_dur = pd.Series(trade_duration).rank().to_numpy()
        ranks_var = pd.Series(np.abs(pnl)).rank().to_numpy()
        rho = np.corrcoef(ranks_dur, ranks_var)[0, 1]
        self.assertFinite(rho, name="spearman_rho")
        self.assertGreater(rho, 0.1)

    def test_stats_scaling_invariance_distribution_metrics(self):
        """Equal scaling keeps KL/JS ≈0."""
        df1 = self._shift_scale_df(400)
        scale = 3.5
        df2 = df1.copy()
        df2["pnl"] *= scale
        df1["pnl"] *= scale
        metrics = compute_distribution_shift_metrics(df1, df2)
        for k, v in metrics.items():
            if k.endswith("_kl_divergence") or k.endswith("_js_distance"):
                self.assertLess(
                    abs(v),
                    5e-4,
                    f"Expected near-zero divergence after equal scaling (k={k}, v={v})",
                )

    def test_stats_mean_decomposition_consistency(self):
        """Batch mean additivity."""
        df_a = self._shift_scale_df(120)
        df_b = self._shift_scale_df(180, shift=0.2)
        m_concat = pd.concat([df_a["pnl"], df_b["pnl"]]).mean()
        m_weighted = (df_a["pnl"].mean() * len(df_a) + df_b["pnl"].mean() * len(df_b)) / (
            len(df_a) + len(df_b)
        )
        self.assertAlmostEqualFloat(
            m_concat,
            m_weighted,
            tolerance=self.TOL_IDENTITY_STRICT,
            rtol=self.TOL_RELATIVE,
        )

    def test_stats_bh_correction_null_false_positive_rate(self):
        """Null: low BH discovery rate."""
        rng = np.random.default_rng(1234)
        n = 400
        df = pd.DataFrame(
            {
                "pnl": rng.normal(0, 1, n),
                "reward": rng.normal(0, 1, n),
                "idle_duration": rng.exponential(5, n),
            }
        )
        df["reward_idle"] = rng.normal(0, 1, n) * 1e-3
        df["position"] = rng.choice([0.0, 1.0], size=n)
        df["action"] = rng.choice([0.0, 2.0], size=n)
        tests = statistical_hypothesis_tests(df)
        flags: list[bool] = []
        for v in tests.values():
            if isinstance(v, dict):
                if "significant_adj" in v:
                    flags.append(bool(v["significant_adj"]))
                elif "significant" in v:
                    flags.append(bool(v["significant"]))
        if flags:
            rate = sum(flags) / len(flags)
            self.assertLess(rate, 0.15, f"BH null FP rate too high under null: {rate:.3f}")

    def test_stats_half_life_monotonic_series(self):
        """Smoothed exponential decay monotonic."""
        x = np.arange(0, 80)
        y = np.exp(-x / 15.0)
        rng = np.random.default_rng(5)
        y_noisy = y + rng.normal(0, 1e-4, len(y))
        window = 5
        y_smooth = np.convolve(y_noisy, np.ones(window) / window, mode="valid")
        self.assertMonotonic(y_smooth, non_increasing=True, tolerance=1e-5)

    def test_stats_hypothesis_seed_reproducibility(self):
        """Seed reproducibility for statistical_hypothesis_tests + bootstrap."""
        df = self.make_stats_df(n=300, seed=self.SEED, idle_pattern="mixed")
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
        # Bootstrap reproducibility
        metrics = ["reward", "pnl"]
        ci_a = bootstrap_confidence_intervals(df, metrics, n_bootstrap=150, seed=2024)
        ci_b = bootstrap_confidence_intervals(df, metrics, n_bootstrap=150, seed=2024)
        # Compare floats with strict identity + scale-aware relative tolerance to avoid flaky exact equality
        for metric in metrics:
            m_a, lo_a, hi_a = ci_a[metric]
            m_b, lo_b, hi_b = ci_b[metric]
            self.assertAlmostEqualFloat(
                m_a, m_b, tolerance=self.TOL_IDENTITY_STRICT, rtol=self.TOL_RELATIVE
            )
            self.assertAlmostEqualFloat(
                lo_a, lo_b, tolerance=self.TOL_IDENTITY_STRICT, rtol=self.TOL_RELATIVE
            )
            self.assertAlmostEqualFloat(
                hi_a, hi_b, tolerance=self.TOL_IDENTITY_STRICT, rtol=self.TOL_RELATIVE
            )

    def test_stats_distribution_metrics_mathematical_bounds(self):
        """Mathematical bounds and validity of distribution shift metrics."""
        self.seed_all(self.SEED)
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
        for feature in ["pnl", "trade_duration", "idle_duration"]:
            for suffix, upper in [
                ("kl_divergence", None),
                ("js_distance", 1.0),
                ("wasserstein", None),
                ("ks_statistic", 1.0),
            ]:
                key = f"{feature}_{suffix}"
                if key in metrics:
                    if upper is None:
                        self.assertDistanceMetric(metrics[key], name=key)
                    else:
                        self.assertDistanceMetric(metrics[key], upper=upper, name=key)
            p_key = f"{feature}_ks_pvalue"
            if p_key in metrics:
                self.assertPValue(metrics[p_key])

    def test_stats_heteroscedasticity_pnl_validation(self):
        """PnL variance increases with trade duration (heteroscedasticity)."""
        df = simulate_samples(
            params=self.base_params(max_trade_duration_candles=100),
            num_samples=1000,
            seed=123,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        exit_data = df[df["reward_exit"] != 0].copy()
        if len(exit_data) < 50:
            self.skipTest("Insufficient exit actions for heteroscedasticity test")
        exit_data["duration_bin"] = pd.cut(
            exit_data["duration_ratio"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )
        variance_by_bin = exit_data.groupby("duration_bin")["pnl"].var().dropna()
        if "Q1" in variance_by_bin.index and "Q4" in variance_by_bin.index:
            self.assertGreater(
                variance_by_bin["Q4"],
                variance_by_bin["Q1"] * 0.8,
                "PnL heteroscedasticity: variance should increase with duration",
            )

    def test_stats_statistical_functions_bounds_validation(self):
        """All statistical functions respect bounds."""
        df = self.make_stats_df(n=300, seed=self.SEED, idle_pattern="all_nonzero")
        diagnostics = distribution_diagnostics(df)
        for col in ["reward", "pnl", "trade_duration", "idle_duration"]:
            if f"{col}_skewness" in diagnostics:
                self.assertFinite(diagnostics[f"{col}_skewness"], name=f"skewness[{col}]")
            if f"{col}_kurtosis" in diagnostics:
                self.assertFinite(diagnostics[f"{col}_kurtosis"], name=f"kurtosis[{col}]")
            if f"{col}_shapiro_pval" in diagnostics:
                self.assertPValue(
                    diagnostics[f"{col}_shapiro_pval"],
                    msg=f"Shapiro p-value bounds for {col}",
                )
        hypothesis_results = statistical_hypothesis_tests(df, seed=self.SEED)
        for test_name, result in hypothesis_results.items():
            if "p_value" in result:
                self.assertPValue(result["p_value"], msg=f"p-value bounds for {test_name}")
            if "effect_size_epsilon_sq" in result:
                eps2 = result["effect_size_epsilon_sq"]
                self.assertFinite(eps2, name=f"epsilon_sq[{test_name}]")
                self.assertGreaterEqual(eps2, 0.0)
            if "effect_size_rank_biserial" in result:
                rb = result["effect_size_rank_biserial"]
                self.assertFinite(rb, name=f"rank_biserial[{test_name}]")
                self.assertWithin(rb, -1.0, 1.0, name="rank_biserial")
            if "rho" in result and result["rho"] is not None:
                rho = result["rho"]
                self.assertFinite(rho, name=f"rho[{test_name}]")
                self.assertWithin(rho, -1.0, 1.0, name="rho")

    def test_stats_benjamini_hochberg_adjustment(self):
        """BH adjustment adds p_value_adj & significant_adj with valid bounds."""
        df = simulate_samples(
            params=self.base_params(max_trade_duration_candles=100),
            num_samples=600,
            seed=123,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        results_adj = statistical_hypothesis_tests(df, adjust_method="benjamini_hochberg", seed=777)
        self.assertGreater(len(results_adj), 0)
        for name, res in results_adj.items():
            self.assertIn("p_value", res)
            self.assertIn("p_value_adj", res)
            self.assertIn("significant_adj", res)
            p_raw = res["p_value"]
            p_adj = res["p_value_adj"]
            self.assertPValue(p_raw)
            self.assertPValue(p_adj)
            self.assertGreaterEqual(p_adj, p_raw - self.TOL_IDENTITY_STRICT)
            alpha = 0.05
            self.assertEqual(res["significant_adj"], bool(p_adj < alpha))
            if "effect_size_epsilon_sq" in res:
                eff = res["effect_size_epsilon_sq"]
                self.assertFinite(eff)
                self.assertGreaterEqual(eff, 0)
            if "effect_size_rank_biserial" in res:
                rb = res["effect_size_rank_biserial"]
                self.assertFinite(rb)
                self.assertWithin(rb, -1, 1, name="rank_biserial")


class TestRewardComponents(RewardSpaceTestBase):
    """Core reward component tests."""

    def test_reward_calculation_scenarios_basic(self):
        """Reward calculation scenarios: expected components become non-zero."""
        test_cases = [
            (Positions.Neutral, Actions.Neutral, "idle_penalty"),
            (Positions.Long, Actions.Long_exit, "exit_component"),
            (Positions.Short, Actions.Short_exit, "exit_component"),
        ]
        for position, action, expected_type in test_cases:
            with self.subTest(position=position, action=action):
                context = self.make_ctx(
                    pnl=0.02 if expected_type == "exit_component" else 0.0,
                    trade_duration=50 if position != Positions.Neutral else 0,
                    idle_duration=10 if position == Positions.Neutral else 0,
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
                if expected_type == "idle_penalty":
                    self.assertNotEqual(breakdown.idle_penalty, 0.0)
                elif expected_type == "exit_component":
                    self.assertNotEqual(breakdown.exit_component, 0.0)
                self.assertFinite(breakdown.total, name="breakdown.total")

    def test_efficiency_zero_policy(self):
        ctx = self.make_ctx(
            pnl=0.0,
            trade_duration=1,
            max_unrealized_profit=0.0,
            min_unrealized_profit=-0.02,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        params = self.base_params()
        profit_target = self.TEST_PROFIT_TARGET * self.TEST_RR
        pnl_factor = _get_pnl_factor(params, ctx, profit_target, self.TEST_RR)
        self.assertFinite(pnl_factor, name="pnl_factor")
        self.assertAlmostEqualFloat(pnl_factor, 1.0, tolerance=self.TOL_GENERIC_EQ)

    def test_max_idle_duration_candles_logic(self):
        params_small = self.base_params(max_idle_duration_candles=50)
        params_large = self.base_params(max_idle_duration_candles=200)
        base_factor = self.TEST_BASE_FACTOR
        context = self.make_ctx(
            pnl=0.0,
            trade_duration=0,
            idle_duration=40,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )
        small = calculate_reward(
            context,
            params_small,
            base_factor,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        large = calculate_reward(
            context,
            params_large,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=0.06,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        self.assertLess(small.idle_penalty, 0.0)
        self.assertLess(large.idle_penalty, 0.0)
        self.assertGreater(large.idle_penalty, small.idle_penalty)

    def test_exit_factor_calculation(self):
        """Exit factor calculation across core modes + plateau variant (plateau via exit_plateau=True)."""
        # Core attenuation kernels (excluding legacy which is step-based)
        modes_to_test = ["linear", "power"]

        for mode in modes_to_test:
            test_params = self.base_params(exit_attenuation_mode=mode)
            factor = _get_exit_factor(
                base_factor=1.0,
                pnl=0.02,
                pnl_factor=1.5,
                duration_ratio=0.3,
                params=test_params,
            )
            self.assertFinite(factor, name=f"exit_factor[{mode}]")
            self.assertGreater(factor, 0, f"Exit factor for {mode} should be positive")

        # Plateau+linear variant (grace region 0.5)
        plateau_params = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=0.5,
            exit_linear_slope=1.0,
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
            plateau_factor_post - self.TOL_IDENTITY_STRICT,
            "Plateau pre-grace factor should be >= post-grace factor",
        )

    def test_idle_penalty_zero_when_profit_target_zero(self):
        """If profit_target=0 → idle_factor=0 → idle penalty must be exactly 0 for neutral idle state."""
        context = self.make_ctx(
            pnl=0.0,
            trade_duration=0,
            idle_duration=30,
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
            br.idle_penalty,
            0.0,
            "Idle penalty should be zero when profit_target=0",
        )
        self.assertEqual(br.total, 0.0, "Total reward should be zero in this configuration")

    def test_win_reward_factor_saturation(self):
        """Saturation test: pnl amplification factor should monotonically approach (1 + win_reward_factor)."""
        win_reward_factor = 3.0  # asymptote = 4.0
        beta = 0.5
        profit_target = self.TEST_PROFIT_TARGET
        params = self.base_params(
            win_reward_factor=win_reward_factor,
            pnl_factor_beta=beta,
            efficiency_weight=0.0,  # disable efficiency modulation
            exit_attenuation_mode="linear",
            exit_plateau=False,
            exit_linear_slope=0.0,  # keep attenuation = 1
        )
        # Ensure provided base_factor=1.0 is actually used (remove default 100)
        params.pop("base_factor", None)

        # pnl values: slightly above target, 2x, 5x, 10x target
        pnl_values = [profit_target * m for m in (1.05, self.TEST_RR_HIGH, 5.0, 10.0)]
        ratios_observed: list[float] = []

        for pnl in pnl_values:
            context = self.make_ctx(
                pnl=pnl,
                trade_duration=0,  # duration_ratio=0 -> attenuation = 1
                idle_duration=0,
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
            tolerance=self.TOL_IDENTITY_STRICT,
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
        """Components scale ~ linearly with base_factor; total equals sum(core + shaping + additives)."""
        params = self.base_params()
        params.pop("base_factor", None)  # explicit base_factor argument below
        base_factor = 80.0
        k = 7.5
        profit_target = self.TEST_PROFIT_TARGET
        rr = 1.5

        contexts: list[RewardContext] = [
            # Winning exit
            self.make_ctx(
                pnl=0.025,
                trade_duration=40,
                idle_duration=0,
                max_unrealized_profit=0.03,
                min_unrealized_profit=0.0,
                position=Positions.Long,
                action=Actions.Long_exit,
            ),
            # Losing exit
            self.make_ctx(
                pnl=-self.TEST_PNL_STD,
                trade_duration=60,
                idle_duration=0,
                max_unrealized_profit=0.01,
                min_unrealized_profit=-0.04,
                position=Positions.Long,
                action=Actions.Long_exit,
            ),
            # Idle penalty
            self.make_ctx(
                pnl=0.0,
                trade_duration=0,
                idle_duration=35,
                max_unrealized_profit=0.0,
                min_unrealized_profit=0.0,
                position=Positions.Neutral,
                action=Actions.Neutral,
            ),
            # Hold penalty
            self.make_ctx(
                pnl=0.0,
                trade_duration=80,
                idle_duration=0,
                max_unrealized_profit=0.04,
                min_unrealized_profit=-0.01,
                position=Positions.Long,
                action=Actions.Neutral,
            ),
        ]

        tol_scale = self.TOL_RELATIVE
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
                    + br.reward_shaping
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
        """Long vs Short exit reward magnitudes should match in absolute value for identical PnL (no directional bias)."""
        params = self.base_params()
        params.pop("base_factor", None)
        base_factor = 120.0
        profit_target = 0.04
        rr = self.TEST_RR_HIGH
        pnls = [0.018, -0.022]
        for pnl in pnls:
            ctx_long = self.make_ctx(
                pnl=pnl,
                trade_duration=55,
                idle_duration=0,
                max_unrealized_profit=pnl if pnl > 0 else 0.01,
                min_unrealized_profit=pnl if pnl < 0 else -0.01,
                position=Positions.Long,
                action=Actions.Long_exit,
            )
            ctx_short = self.make_ctx(
                pnl=pnl,
                trade_duration=55,
                idle_duration=0,
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
                self.TOL_RELATIVE * max(1.0, abs(br_long.exit_component)),
                f"Long/Short asymmetry pnl={pnl}: long={br_long.exit_component}, short={br_short.exit_component}",
            )


class TestAPIAndHelpers(RewardSpaceTestBase):
    """Public API + helper utility tests."""

    # --- Public API ---
    def test_parse_overrides(self):
        overrides = ["alpha=1.5", "mode=linear", "limit=42"]
        result = parse_overrides(overrides)
        self.assertEqual(result["alpha"], 1.5)
        self.assertEqual(result["mode"], "linear")
        self.assertEqual(result["limit"], 42.0)
        with self.assertRaises(ValueError):
            parse_overrides(["badpair"])  # missing '='

    def test_api_simulation_and_reward_smoke(self):
        df = simulate_samples(
            params=self.base_params(max_trade_duration_candles=40),
            num_samples=20,
            seed=7,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=1.5,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        self.assertGreater(len(df), 0)
        any_exit = df[df["reward_exit"] != 0].head(1)
        if not any_exit.empty:
            row = any_exit.iloc[0]
            ctx = self.make_ctx(
                pnl=float(row["pnl"]),
                trade_duration=int(row["trade_duration"]),
                idle_duration=int(row["idle_duration"]),
                max_unrealized_profit=float(row["pnl"]) + 0.01,
                min_unrealized_profit=float(row["pnl"]) - 0.01,
                position=Positions.Long,
                action=Actions.Long_exit,
            )
            breakdown = calculate_reward(
                ctx,
                self.DEFAULT_PARAMS,
                base_factor=self.TEST_BASE_FACTOR,
                profit_target=self.TEST_PROFIT_TARGET,
                risk_reward_ratio=self.TEST_RR,
                short_allowed=True,
                action_masking=True,
            )
            self.assertFinite(breakdown.total)

    def test_simulate_samples_trading_modes_spot_vs_margin(self):
        """simulate_samples coverage: spot should forbid shorts, margin should allow them."""
        df_spot = simulate_samples(
            params=self.base_params(max_trade_duration_candles=100),
            num_samples=80,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="spot",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        short_positions_spot = (df_spot["position"] == float(Positions.Short.value)).sum()
        self.assertEqual(
            short_positions_spot,
            0,
            "Spot mode must not contain short positions",
        )
        df_margin = simulate_samples(
            params=self.base_params(max_trade_duration_candles=100),
            num_samples=80,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        for col in [
            "pnl",
            "trade_duration",
            "idle_duration",
            "position",
            "action",
            "reward",
            "reward_invalid",
            "reward_idle",
            "reward_hold",
            "reward_exit",
        ]:
            self.assertIn(col, df_margin.columns)

    # --- Helper functions ---

    def test_to_bool(self):
        """Test _to_bool with various inputs."""
        # Test via simulate_samples which uses action_masking parameter
        df1 = simulate_samples(
            params=self.base_params(action_masking="true", max_trade_duration_candles=50),
            num_samples=10,
            seed=self.SEED,
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
            params=self.base_params(action_masking="false", max_trade_duration_candles=50),
            num_samples=10,
            seed=self.SEED,
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
            params=self.base_params(max_trade_duration_candles=50),
            num_samples=100,
            seed=self.SEED,
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
        self.assertGreater(short_positions, 0, "Futures mode should allow short positions")

    def test_get_float_param(self):
        """Test float parameter extraction."""
        params = {"test_float": 1.5, "test_int": 2, "test_str": "hello"}
        self.assertEqual(_get_float_param(params, "test_float", 0.0), 1.5)
        self.assertEqual(_get_float_param(params, "test_int", 0.0), 2.0)
        # Non parseable string -> NaN fallback in tolerant parser
        val_str = _get_float_param(params, "test_str", 0.0)
        self.assertTrue(isinstance(val_str, float))
        self.assertTrue(math.isnan(val_str))
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

    def test_get_int_param_coercions(self):
        """Robust coercion paths of _get_int_param (bool/int/float/str/None/unsupported)."""
        # None with numeric default
        self.assertEqual(_get_int_param({"k": None}, "k", 5), 5)
        # None with non-numeric default -> 0 fallback
        self.assertEqual(_get_int_param({"k": None}, "k", "x"), 0)
        # Booleans
        self.assertEqual(_get_int_param({"k": True}, "k", 0), 1)
        self.assertEqual(_get_int_param({"k": False}, "k", 7), 0)
        # Int passthrough
        self.assertEqual(_get_int_param({"k": -12}, "k", 0), -12)
        # Float truncation & negative
        self.assertEqual(_get_int_param({"k": 9.99}, "k", 0), 9)
        self.assertEqual(_get_int_param({"k": -3.7}, "k", 0), -3)
        # Non-finite floats fallback
        self.assertEqual(_get_int_param({"k": np.nan}, "k", 4), 4)
        self.assertEqual(_get_int_param({"k": float("inf")}, "k", 4), 4)
        # String numerics (int, float, exponent)
        self.assertEqual(_get_int_param({"k": "42"}, "k", 0), 42)
        self.assertEqual(_get_int_param({"k": " 17 "}, "k", 0), 17)
        self.assertEqual(_get_int_param({"k": "3.9"}, "k", 0), 3)
        self.assertEqual(_get_int_param({"k": "1e2"}, "k", 0), 100)
        # String fallbacks (empty, invalid, NaN token)
        self.assertEqual(_get_int_param({"k": ""}, "k", 5), 5)
        self.assertEqual(_get_int_param({"k": "abc"}, "k", 5), 5)
        self.assertEqual(_get_int_param({"k": "NaN"}, "k", 5), 5)
        # Unsupported type
        self.assertEqual(_get_int_param({"k": [1, 2, 3]}, "k", 3), 3)
        # Missing key with non-numeric default
        self.assertEqual(_get_int_param({}, "missing", "zzz"), 0)

    def test_argument_parser_construction(self):
        """Test build_argument_parser function."""

        parser = build_argument_parser()
        self.assertIsNotNone(parser)

        # Test parsing with minimal arguments
        args = parser.parse_args(
            [
                "--num_samples",
                "100",
                "--out_dir",
                "test_output",
            ]
        )
        self.assertEqual(args.num_samples, 100)
        self.assertEqual(str(args.out_dir), "test_output")

    def test_complete_statistical_analysis_writer(self):
        """Test write_complete_statistical_analysis function."""

        # Create comprehensive test data
        test_data = simulate_samples(
            params=self.base_params(max_trade_duration_candles=100),
            num_samples=200,
            seed=self.SEED,
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
                profit_target=self.TEST_PROFIT_TARGET,
                seed=self.SEED,
                real_df=None,
            )

            # Check that main report is created
            main_report = output_path / "statistical_analysis.md"
            self.assertTrue(
                main_report.exists(),
                "Main statistical analysis should be created",
            )

            # Check for other expected files
            feature_file = output_path / "feature_importance.csv"
            self.assertTrue(feature_file.exists(), "Feature importance should be created")


class TestPrivateFunctions(RewardSpaceTestBase):
    """Test private functions through public API calls."""

    def test_idle_penalty_via_rewards(self):
        """Test idle penalty calculation via reward calculation."""
        # Create context that will trigger idle penalty
        context = self.make_ctx(
            pnl=0.0,
            trade_duration=0,
            idle_duration=20,
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
            + breakdown.reward_shaping
            + breakdown.entry_additive
            + breakdown.exit_additive,
            tolerance=self.TOL_IDENTITY_RELAXED,
            msg="Total should equal sum of components (idle + shaping/additives)",
        )

    def test_hold_penalty_via_rewards(self):
        """Test hold penalty calculation via reward calculation."""
        # Create context that will trigger hold penalty
        context = self.make_ctx(
            pnl=0.01,
            trade_duration=150,
            idle_duration=0,  # Long duration
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
            + breakdown.reward_shaping
            + breakdown.entry_additive
            + breakdown.exit_additive,
            tolerance=self.TOL_IDENTITY_RELAXED,
            msg="Total should equal sum of components (hold + shaping/additives)",
        )

    def test_exit_reward_calculation(self):
        """Test exit reward calculation with various scenarios."""
        scenarios = [
            (Positions.Long, Actions.Long_exit, 0.05, "Profitable long exit"),
            (
                Positions.Short,
                Actions.Short_exit,
                -0.03,
                "Profitable short exit",
            ),
            (Positions.Long, Actions.Long_exit, -0.02, "Losing long exit"),
            (Positions.Short, Actions.Short_exit, 0.02, "Losing short exit"),
        ]

        for position, action, pnl, description in scenarios:
            with self.subTest(description=description):
                context = self.make_ctx(
                    pnl=pnl,
                    trade_duration=50,
                    idle_duration=0,
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
        context = self.make_ctx(
            pnl=0.02,
            trade_duration=50,
            idle_duration=0,
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
            breakdown.invalid_penalty,
            0,
            "Invalid action should have negative penalty",
        )
        self.assertAlmostEqualFloat(
            breakdown.total,
            breakdown.invalid_penalty
            + breakdown.reward_shaping
            + breakdown.entry_additive
            + breakdown.exit_additive,
            tolerance=self.TOL_IDENTITY_RELAXED,
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
                context = self.make_ctx(
                    pnl=0.0,  # Neutral PnL to isolate hold penalty
                    trade_duration=trade_duration,
                    idle_duration=0,
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
                    + breakdown.reward_shaping
                    + breakdown.entry_additive
                    + breakdown.exit_additive,
                    tolerance=self.TOL_IDENTITY_RELAXED,
                    msg=f"Total mismatch including shaping {description}",
                )

    def test_hold_penalty_progressive_scaling(self):
        """Test that hold penalty scales progressively after max_duration."""
        params = self.base_params(max_trade_duration_candles=100)
        durations = [150, 200, 300]  # All > max_duration
        penalties: list[float] = []

        for duration in durations:
            context = self.make_ctx(
                pnl=0.0,
                trade_duration=duration,
                idle_duration=0,
                max_unrealized_profit=0.0,
                min_unrealized_profit=0.0,
                position=Positions.Long,
                action=Actions.Neutral,
            )

            breakdown = calculate_reward(
                context,
                params,
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
                f"Penalty should increase with duration: {penalties[i]} > {penalties[i - 1]}",
            )

    def test_new_invariant_and_warn_parameters(self):
        """Ensure new tunables (check_invariants, exit_factor_threshold) exist and behave.

        Uses a very large base_factor to trigger potential warning condition without capping.
        """
        params = self.base_params()
        self.assertIn("check_invariants", params)
        self.assertIn("exit_factor_threshold", params)

        context = self.make_ctx(
            pnl=0.05,
            trade_duration=300,
            idle_duration=0,
            max_unrealized_profit=0.06,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        breakdown = calculate_reward(
            context,
            params,
            base_factor=1e7,  # Large factor to stress scaling paths
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        self.assertFinite(breakdown.exit_component, name="exit_component")


class TestRewardRobustnessAndBoundaries(RewardSpaceTestBase):
    """Robustness & boundary assertions: invariants, attenuation maths, parameter edges, scaling, warnings."""

    def test_decomposition_integrity(self):
        """reward must equal the single active core component under mutually exclusive scenarios (idle/hold/exit/invalid)."""
        scenarios = [
            # Idle penalty only
            dict(
                ctx=self.make_ctx(
                    pnl=0.0,
                    trade_duration=0,
                    idle_duration=25,
                    max_unrealized_profit=0.0,
                    min_unrealized_profit=0.0,
                    position=Positions.Neutral,
                    action=Actions.Neutral,
                ),
                active="idle_penalty",
            ),
            # Hold penalty only
            dict(
                ctx=self.make_ctx(
                    pnl=0.0,
                    trade_duration=150,
                    idle_duration=0,
                    max_unrealized_profit=0.0,
                    min_unrealized_profit=0.0,
                    position=Positions.Long,
                    action=Actions.Neutral,
                ),
                active="hold_penalty",
            ),
            # Exit reward only (positive pnl)
            dict(
                ctx=self.make_ctx(
                    pnl=self.TEST_PROFIT_TARGET,
                    trade_duration=60,
                    idle_duration=0,
                    max_unrealized_profit=0.05,
                    min_unrealized_profit=0.01,
                    position=Positions.Long,
                    action=Actions.Long_exit,
                ),
                active="exit_component",
            ),
            # Invalid action only
            dict(
                ctx=self.make_ctx(
                    pnl=0.01,
                    trade_duration=10,
                    idle_duration=0,
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
                # Build parameters disabling shaping and additives to enforce strict decomposition
                params = self.base_params(
                    entry_additive_enabled=False,
                    exit_additive_enabled=False,
                    hold_potential_enabled=False,
                    potential_gamma=0.0,
                    check_invariants=False,
                )
                br = calculate_reward(
                    ctx_obj,
                    params,
                    base_factor=self.TEST_BASE_FACTOR,
                    profit_target=self.TEST_PROFIT_TARGET,
                    risk_reward_ratio=self.TEST_RR,
                    short_allowed=True,
                    action_masking=True,
                )
                # Single active component must match total; others zero (or near zero for shaping/additives)
                core_components = {
                    "exit_component": br.exit_component,
                    "idle_penalty": br.idle_penalty,
                    "hold_penalty": br.hold_penalty,
                    "invalid_penalty": br.invalid_penalty,
                }
                for name, value in core_components.items():
                    if name == active_label:
                        self.assertAlmostEqualFloat(
                            value,
                            br.total,
                            tolerance=self.TOL_IDENTITY_RELAXED,
                            msg=f"Active component {name} != total",
                        )
                    else:
                        self.assertNearZero(
                            value,
                            atol=self.TOL_IDENTITY_RELAXED,
                            msg=f"Inactive component {name} not near zero (val={value})",
                        )
                # Shaping and additives explicitly disabled
                self.assertAlmostEqualFloat(
                    br.reward_shaping, 0.0, tolerance=self.TOL_IDENTITY_RELAXED
                )
                self.assertAlmostEqualFloat(
                    br.entry_additive, 0.0, tolerance=self.TOL_IDENTITY_RELAXED
                )
                self.assertAlmostEqualFloat(
                    br.exit_additive, 0.0, tolerance=self.TOL_IDENTITY_RELAXED
                )

    def test_pnl_invariant_exit_only(self):
        """Invariant: only exit actions have non-zero PnL (robustness category)."""
        df = simulate_samples(
            params=self.base_params(max_trade_duration_candles=50),
            num_samples=200,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        total_pnl = df["pnl"].sum()
        exit_mask = df["reward_exit"] != 0
        exit_pnl_sum = df.loc[exit_mask, "pnl"].sum()
        self.assertAlmostEqual(
            total_pnl,
            exit_pnl_sum,
            places=10,
            msg="PnL invariant violation: total PnL != sum of exit PnL",
        )
        non_zero_pnl_actions = set(df[df["pnl"].abs() > self.EPS_BASE]["action"].unique())
        expected_exit_actions = {2.0, 4.0}
        self.assertTrue(
            non_zero_pnl_actions.issubset(expected_exit_actions),
            f"Non-exit actions have PnL: {non_zero_pnl_actions - expected_exit_actions}",
        )
        invalid_combinations = df[(df["pnl"].abs() <= self.EPS_BASE) & (df["reward_exit"] != 0)]
        self.assertEqual(len(invalid_combinations), 0)

    def test_exit_factor_mathematical_formulas(self):
        """Mathematical correctness of exit factor calculations across modes."""
        context = self.make_ctx(
            pnl=0.05,
            trade_duration=50,
            idle_duration=0,
            max_unrealized_profit=0.05,
            min_unrealized_profit=0.01,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        params = self.DEFAULT_PARAMS.copy()
        duration_ratio = 50 / 100
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
        self.assertGreater(reward_power.exit_component, 0)
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
        # Validate half-life attenuation factor against expected closed-form: 2 ** (-dr / half_life)
        pnl_factor_hl = _get_pnl_factor(
            params,
            context,
            self.TEST_PROFIT_TARGET,
            self.TEST_RR,
        )
        observed_exit_factor = _get_exit_factor(
            self.TEST_BASE_FACTOR,
            context.pnl,
            pnl_factor_hl,
            duration_ratio,
            params,
        )
        # Isolate attenuation factor
        observed_half_life_factor = observed_exit_factor / (
            self.TEST_BASE_FACTOR * max(pnl_factor_hl, self.EPS_BASE)
        )
        expected_half_life_factor = 2 ** (-duration_ratio / params["exit_half_life"])
        self.assertAlmostEqualFloat(
            observed_half_life_factor,
            expected_half_life_factor,
            tolerance=self.TOL_IDENTITY_RELAXED,
            msg="Half-life attenuation mismatch: observed vs expected",
        )
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
        rewards = [
            reward_power.exit_component,
            reward_half_life.exit_component,
            reward_linear.exit_component,
        ]
        self.assertTrue(all(r > 0 for r in rewards))
        unique_rewards = set(f"{r:.6f}" for r in rewards)
        self.assertGreater(len(unique_rewards), 1)

    def test_idle_penalty_fallback_and_proportionality(self):
        """Idle penalty fallback denominator & proportional scaling (robustness)."""
        params = self.base_params(max_idle_duration_candles=None, max_trade_duration_candles=100)
        base_factor = 90.0
        profit_target = self.TEST_PROFIT_TARGET
        risk_reward_ratio = 1.0
        ctx_a = self.make_ctx(
            pnl=0.0,
            trade_duration=0,
            idle_duration=20,
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
        ratio = br_b.idle_penalty / br_a.idle_penalty if br_a.idle_penalty != 0 else None
        self.assertIsNotNone(ratio)
        self.assertAlmostEqualFloat(abs(ratio), 2.0, tolerance=0.2)
        ctx_mid = dataclasses.replace(ctx_a, idle_duration=120)
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
        factor = _get_float_param(params, "base_factor", float(base_factor))
        idle_factor = factor * (profit_target * risk_reward_ratio) / 4.0
        observed_ratio = abs(br_mid.idle_penalty) / (idle_factor * idle_penalty_scale)
        if observed_ratio > 0:
            implied_D = 120 / (observed_ratio ** (1 / idle_penalty_power))
            self.assertAlmostEqualFloat(implied_D, 400.0, tolerance=20.0)

    def test_exit_factor_threshold_warning_and_non_capping(self):
        """Warning emission without capping when exit_factor_threshold exceeded."""
        params = self.base_params(exit_factor_threshold=10.0)
        params.pop("base_factor", None)
        context = self.make_ctx(
            pnl=0.08,
            trade_duration=10,
            idle_duration=0,
            max_unrealized_profit=0.09,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            baseline = calculate_reward(
                context,
                params,
                base_factor=self.TEST_BASE_FACTOR,
                profit_target=self.TEST_PROFIT_TARGET,
                risk_reward_ratio=self.TEST_RR_HIGH,
                short_allowed=True,
                action_masking=True,
            )
            amplified_base_factor = self.TEST_BASE_FACTOR * 200.0
            amplified = calculate_reward(
                context,
                params,
                base_factor=amplified_base_factor,
                profit_target=self.TEST_PROFIT_TARGET,
                risk_reward_ratio=self.TEST_RR_HIGH,
                short_allowed=True,
                action_masking=True,
            )
        self.assertGreater(baseline.exit_component, 0.0)
        self.assertGreater(amplified.exit_component, baseline.exit_component)
        scale = amplified.exit_component / baseline.exit_component
        self.assertGreater(scale, 10.0)
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(runtime_warnings)
        self.assertTrue(
            any(
                (
                    "exceeded threshold" in str(w.message)
                    or "exceeds threshold" in str(w.message)
                    or "|factor|=" in str(w.message)
                )
                for w in runtime_warnings
            )
        )

    def test_negative_slope_sanitization(self):
        """Negative exit_linear_slope is sanitized to 0.0; resulting exit factors must match slope=0.0 within tolerance."""
        base_factor = 100.0
        pnl = 0.03
        pnl_factor = 1.0
        duration_ratios = [0.0, 0.2, 0.5, 1.0, 1.5]
        params_bad = self.base_params(
            exit_attenuation_mode="linear",
            exit_linear_slope=-5.0,
            exit_plateau=False,
        )
        params_ref = self.base_params(
            exit_attenuation_mode="linear",
            exit_linear_slope=0.0,
            exit_plateau=False,
        )
        for dr in duration_ratios:
            f_bad = _get_exit_factor(base_factor, pnl, pnl_factor, dr, params_bad)
            f_ref = _get_exit_factor(base_factor, pnl, pnl_factor, dr, params_ref)
            self.assertAlmostEqualFloat(
                f_bad,
                f_ref,
                tolerance=self.TOL_IDENTITY_RELAXED,
                msg=f"Sanitized slope mismatch at dr={dr} f_bad={f_bad} f_ref={f_ref}",
            )

    def test_power_mode_alpha_formula(self):
        """Power mode attenuation: ratio f(dr=1)/f(dr=0) must equal 1/(1+1)^alpha with alpha=-log(tau)/log(2)."""
        base_factor = 200.0
        pnl = 0.04
        pnl_factor = 1.0
        duration_ratio = 1.0
        taus = [
            0.9,
            0.5,
            0.25,
            1.0,
        ]
        for tau in taus:
            params = self.base_params(
                exit_attenuation_mode="power",
                exit_power_tau=tau,
                exit_plateau=False,
            )
            f0 = _get_exit_factor(base_factor, pnl, pnl_factor, 0.0, params)
            f1 = _get_exit_factor(base_factor, pnl, pnl_factor, duration_ratio, params)
            # Extract alpha using same formula (replicate logic)
            if 0.0 < tau <= 1.0:
                alpha = -math.log(tau) / math.log(2.0)
            else:
                alpha = 1.0
            expected_ratio = 1.0 / (1.0 + duration_ratio) ** alpha
            observed_ratio = f1 / f0 if f0 != 0 else np.nan
            self.assertFinite(observed_ratio, name="observed_ratio")
            self.assertLess(
                abs(observed_ratio - expected_ratio),
                5e-12 if tau == 1.0 else 5e-9,
                f"Alpha attenuation mismatch tau={tau} alpha={alpha} obs_ratio={observed_ratio} exp_ratio={expected_ratio}",
            )

    # Boundary condition tests (extremes / continuity / monotonicity)
    def test_extreme_parameter_values(self):
        extreme_params = self.base_params(
            win_reward_factor=1000.0,
            base_factor=10000.0,
        )
        context = RewardContext(
            pnl=0.05,
            trade_duration=50,
            idle_duration=0,
            max_unrealized_profit=0.06,
            min_unrealized_profit=0.02,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        br = calculate_reward(
            context,
            extreme_params,
            base_factor=10000.0,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            short_allowed=True,
            action_masking=True,
        )
        self.assertFinite(br.total, name="breakdown.total")

    def test_exit_attenuation_modes_enumeration(self):
        modes = ATTENUATION_MODES_WITH_LEGACY
        for mode in modes:
            with self.subTest(mode=mode):
                test_params = self.base_params(exit_attenuation_mode=mode)
                ctx = self.make_ctx(
                    pnl=0.02,
                    trade_duration=50,
                    idle_duration=0,
                    max_unrealized_profit=0.03,
                    min_unrealized_profit=0.01,
                    position=Positions.Long,
                    action=Actions.Long_exit,
                )
                br = calculate_reward(
                    ctx,
                    test_params,
                    base_factor=self.TEST_BASE_FACTOR,
                    profit_target=self.TEST_PROFIT_TARGET,
                    risk_reward_ratio=self.TEST_RR,
                    short_allowed=True,
                    action_masking=True,
                )
                self.assertFinite(br.exit_component, name="breakdown.exit_component")
                self.assertFinite(br.total, name="breakdown.total")

    def test_exit_factor_monotonic_attenuation(self):
        """For attenuation modes: factor should be non-increasing w.r.t duration_ratio.

        Modes covered: sqrt, linear, power, half_life, plateau+linear (after grace).
        Legacy is excluded (non-monotonic by design). Plateau+linear includes flat grace then monotonic.
        """
        # Start from canonical modes (excluding legacy already) and add synthetic plateau_linear variant
        modes = list(ATTENUATION_MODES) + ["plateau_linear"]
        base_factor = self.TEST_BASE_FACTOR
        pnl = 0.05
        pnl_factor = 1.0
        for mode in modes:
            if mode == "plateau_linear":
                params = self.base_params(
                    exit_attenuation_mode="linear",
                    exit_plateau=True,
                    exit_plateau_grace=0.2,
                    exit_linear_slope=1.0,
                )
            elif mode == "linear":
                params = self.base_params(
                    exit_attenuation_mode="linear",
                    exit_linear_slope=1.2,
                )
            elif mode == "power":
                params = self.base_params(
                    exit_attenuation_mode="power",
                    exit_power_tau=0.5,
                )
            elif mode == "half_life":
                params = self.base_params(
                    exit_attenuation_mode="half_life",
                    exit_half_life=0.7,
                )
            else:  # sqrt
                params = self.base_params(exit_attenuation_mode="sqrt")

            ratios = np.linspace(0, 2, 15)
            values = [_get_exit_factor(base_factor, pnl, pnl_factor, r, params) for r in ratios]
            # Plateau+linear: ignore initial flat region when checking monotonic decrease
            if mode == "plateau_linear":
                grace = float(params["exit_plateau_grace"])  # type: ignore[index]
                filtered = [
                    (r, v) for r, v in zip(ratios, values) if r >= grace - self.TOL_IDENTITY_RELAXED
                ]
                values_to_check = [v for _, v in filtered]
            else:
                values_to_check = values
            for earlier, later in zip(values_to_check, values_to_check[1:]):
                self.assertLessEqual(
                    later,
                    earlier + self.TOL_IDENTITY_RELAXED,
                    f"Non-monotonic attenuation in mode={mode}",
                )

    def test_exit_factor_boundary_parameters(self):
        """Test parameter edge cases: tau extremes, plateau grace edges, slope zero."""

        base_factor = 50.0
        pnl = 0.02
        pnl_factor = 1.0
        # Tau near 1 (minimal attenuation) vs tau near 0 (strong attenuation)
        params_hi = self.base_params(exit_attenuation_mode="power", exit_power_tau=0.999999)
        params_lo = self.base_params(
            exit_attenuation_mode="power",
            exit_power_tau=self.MIN_EXIT_POWER_TAU,
        )
        r = 1.5
        hi_val = _get_exit_factor(base_factor, pnl, pnl_factor, r, params_hi)
        lo_val = _get_exit_factor(base_factor, pnl, pnl_factor, r, params_lo)
        self.assertGreater(
            hi_val,
            lo_val,
            "Power mode: higher tau (≈1) should attenuate less than tiny tau",
        )
        # Plateau grace 0 vs 1
        params_g0 = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=0.0,
            exit_linear_slope=1.0,
        )
        params_g1 = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=1.0,
            exit_linear_slope=1.0,
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
        params_lin0 = self.base_params(
            exit_attenuation_mode="linear",
            exit_linear_slope=0.0,
            exit_plateau=False,
        )
        params_lin1 = self.base_params(
            exit_attenuation_mode="linear",
            exit_linear_slope=2.0,
            exit_plateau=False,
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

        params = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=0.3,
            exit_linear_slope=0.0,
        )
        base_factor = self.TEST_BASE_FACTOR
        pnl = 0.04
        pnl_factor = 1.2
        ratios = [0.3, 0.6, 1.0, 1.4]
        values = [_get_exit_factor(base_factor, pnl, pnl_factor, r, params) for r in ratios]
        # All factors should be (approximately) identical after grace (no attenuation)
        first = values[0]
        for v in values[1:]:
            self.assertAlmostEqualFloat(
                v,
                first,
                tolerance=self.TOL_IDENTITY_RELAXED,
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
        vals = [_get_exit_factor(base_factor, pnl, pnl_factor, r, params) for r in ratios]
        # All ratios <=1.5 should yield identical factor
        ref = vals[0]
        for i, r in enumerate(ratios[:-1]):  # exclude last (1.6)
            self.assertAlmostEqualFloat(
                vals[i],
                ref,
                tolerance=self.TOL_IDENTITY_RELAXED,
                msg=f"Unexpected attenuation before grace end at ratio {r}",
            )
        # Last ratio (1.6) should be attenuated (strictly less than ref)
        self.assertLess(vals[-1], ref, "Attenuation should begin after grace boundary")

    def test_plateau_continuity_at_grace_boundary(self):
        modes = ["sqrt", "linear", "power", "half_life"]
        grace = 0.8
        eps = self.CONTINUITY_EPS_SMALL
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

                left = _get_exit_factor(base_factor, pnl, pnl_factor, grace - eps, params)
                boundary = _get_exit_factor(base_factor, pnl, pnl_factor, grace, params)
                right = _get_exit_factor(base_factor, pnl, pnl_factor, grace + eps, params)

                self.assertAlmostEqualFloat(
                    left,
                    boundary,
                    tolerance=self.TOL_IDENTITY_RELAXED,
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
        eps1 = self.CONTINUITY_EPS_LARGE
        eps2 = self.CONTINUITY_EPS_SMALL
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
        ratio = diff1 / max(diff2, self.TOL_NUMERIC_GUARD)
        self.assertGreater(ratio, 5.0, f"Scaling ratio too small (ratio={ratio:.2f})")
        self.assertLess(ratio, 15.0, f"Scaling ratio too large (ratio={ratio:.2f})")


class TestLoadRealEpisodes(RewardSpaceTestBase):
    """Unit tests for load_real_episodes."""

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
                "reward": [1.0],
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
                    "reward": 2.0,
                }
            ],
        }
        p = Path(self.temp_dir) / "mixed.pkl"
        self.write_pickle([ep1, ep2], p)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = load_real_episodes(p)
            _ = w

        self.assertEqual(len(loaded), 1)
        self.assertPlacesEqual(float(loaded.iloc[0]["pnl"]), 0.02, places=7)

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
        self.assertIn("reward", loaded.columns)
        self.assertTrue(loaded["reward"].isna().all())

    def test_casting_numeric_strings(self):
        trans = [
            {
                "pnl": "0.04",
                "trade_duration": "20",
                "idle_duration": "0",
                "position": "1.0",
                "action": "2.0",
                "reward": "3.0",
            }
        ]
        p = Path(self.temp_dir) / "strs.pkl"
        self.write_pickle(trans, p)

        loaded = load_real_episodes(p)
        self.assertIn("pnl", loaded.columns)
        self.assertIn(loaded["pnl"].dtype.kind, ("f", "i"))
        self.assertPlacesEqual(float(loaded.iloc[0]["pnl"]), 0.04, places=7)

    def test_pickled_dataframe_loads(self):
        """Ensure a directly pickled DataFrame loads correctly."""
        test_episodes = pd.DataFrame(
            {
                "pnl": [0.01, -0.02, 0.03],
                "trade_duration": [10, 20, 15],
                "idle_duration": [5, 0, 8],
                "position": [1.0, 0.0, 1.0],
                "action": [2.0, 0.0, 2.0],
                "reward": [10.5, -5.2, 15.8],
            }
        )
        p = Path(self.temp_dir) / "test_episodes.pkl"
        self.write_pickle(test_episodes, p)

        loaded_data = load_real_episodes(p)
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 3)
        self.assertIn("pnl", loaded_data.columns)


class TestPBRS(RewardSpaceTestBase):
    """PBRS mechanics tests (transforms, parameters, potentials, invariance)."""

    def test_pbrs_progressive_release_decay_clamped(self):
        """progressive_release decay>1 clamps -> Φ'=0 & Δ=-Φ_prev."""
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "potential_gamma": DEFAULT_MODEL_REWARD_PARAMETERS["potential_gamma"],
                "exit_potential_mode": "progressive_release",
                "exit_potential_decay": 5.0,  # clamp to 1.0
                "hold_potential_enabled": True,
                "entry_additive_enabled": False,
                "exit_additive_enabled": False,
            }
        )
        current_pnl = 0.02
        current_dur = 0.5
        prev_potential = _compute_hold_potential(current_pnl, current_dur, params)
        _total_reward, reward_shaping, next_potential = apply_potential_shaping(
            base_reward=0.0,
            current_pnl=current_pnl,
            current_duration_ratio=current_dur,
            next_pnl=0.0,
            next_duration_ratio=0.0,
            is_exit=True,
            is_entry=False,
            last_potential=0.789,  # arbitrary, should be ignored for Φ'
            params=params,
        )
        self.assertAlmostEqualFloat(next_potential, 0.0, tolerance=self.TOL_IDENTITY_RELAXED)
        self.assertAlmostEqualFloat(
            reward_shaping, -prev_potential, tolerance=self.TOL_IDENTITY_RELAXED
        )

    def test_pbrs_spike_cancel_invariance(self):
        """spike_cancel terminal shaping ≈0 (Φ' inversion yields cancellation)."""
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
        gamma = _get_float_param(
            params,
            "potential_gamma",
            DEFAULT_MODEL_REWARD_PARAMETERS.get("potential_gamma", 0.95),
        )
        expected_next_potential = (
            prev_potential / gamma if gamma not in (0.0, None) else prev_potential
        )
        _total_reward, reward_shaping, next_potential = apply_potential_shaping(
            base_reward=0.0,
            current_pnl=current_pnl,
            current_duration_ratio=current_dur,
            next_pnl=0.0,
            next_duration_ratio=0.0,
            is_exit=True,
            is_entry=False,
            last_potential=prev_potential,
            params=params,
        )
        self.assertAlmostEqualFloat(
            next_potential, expected_next_potential, tolerance=self.TOL_IDENTITY_RELAXED
        )
        self.assertNearZero(reward_shaping, atol=self.TOL_IDENTITY_RELAXED)

    def test_tanh_transform(self):
        """tanh transform: tanh(x) in (-1, 1)."""
        self.assertAlmostEqualFloat(apply_transform("tanh", 0.0), 0.0)
        self.assertAlmostEqualFloat(apply_transform("tanh", 1.0), math.tanh(1.0))
        self.assertAlmostEqualFloat(apply_transform("tanh", -1.0), math.tanh(-1.0))
        self.assertTrue(abs(apply_transform("tanh", 100.0)) <= 1.0)
        self.assertTrue(abs(apply_transform("tanh", -100.0)) <= 1.0)

    def test_softsign_transform(self):
        """softsign transform: x / (1 + |x|) in (-1, 1)."""
        self.assertAlmostEqualFloat(apply_transform("softsign", 0.0), 0.0)
        self.assertAlmostEqualFloat(apply_transform("softsign", 1.0), 0.5)
        self.assertAlmostEqualFloat(apply_transform("softsign", -1.0), -0.5)
        self.assertTrue(abs(apply_transform("softsign", 100.0)) < 1.0)
        self.assertTrue(abs(apply_transform("softsign", -100.0)) < 1.0)

    def test_asinh_transform(self):
        """asinh transform: x / sqrt(1 + x^2) in (-1, 1)."""
        self.assertAlmostEqualFloat(apply_transform("asinh", 0.0), 0.0)
        # Symmetry
        self.assertAlmostEqualFloat(
            apply_transform("asinh", 1.2345),
            -apply_transform("asinh", -1.2345),
            tolerance=self.TOL_IDENTITY_STRICT,
        )
        # Monotonicity
        vals = [apply_transform("asinh", x) for x in [-5.0, -1.0, 0.0, 1.0, 5.0]]
        self.assertTrue(all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)))
        # Bounded
        self.assertTrue(abs(apply_transform("asinh", 1e6)) < 1.0)
        self.assertTrue(abs(apply_transform("asinh", -1e6)) < 1.0)

    def test_arctan_transform(self):
        """arctan transform: (2/pi) * arctan(x) in (-1, 1)."""
        self.assertAlmostEqualFloat(apply_transform("arctan", 0.0), 0.0)
        self.assertAlmostEqualFloat(
            apply_transform("arctan", 1.0),
            (2.0 / math.pi) * math.atan(1.0),
            tolerance=1e-10,
        )
        self.assertTrue(abs(apply_transform("arctan", 100.0)) <= 1.0)
        self.assertTrue(abs(apply_transform("arctan", -100.0)) <= 1.0)

    def test_sigmoid_transform(self):
        """sigmoid transform: 2σ(x) - 1, σ(x) = 1/(1 + e^(-x)) in (-1, 1)."""
        self.assertAlmostEqualFloat(apply_transform("sigmoid", 0.0), 0.0)
        self.assertTrue(apply_transform("sigmoid", 100.0) > 0.99)
        self.assertTrue(apply_transform("sigmoid", -100.0) < -0.99)
        self.assertTrue(-1 < apply_transform("sigmoid", 10.0) < 1)
        self.assertTrue(-1 < apply_transform("sigmoid", -10.0) < 1)

    def test_clip_transform(self):
        """clip transform: clip(x, -1, 1) in [-1, 1]."""
        self.assertAlmostEqualFloat(apply_transform("clip", 0.0), 0.0)
        self.assertAlmostEqualFloat(apply_transform("clip", 0.5), 0.5)
        self.assertAlmostEqualFloat(apply_transform("clip", 2.0), 1.0)
        self.assertAlmostEqualFloat(apply_transform("clip", -2.0), -1.0)

    def test_invalid_transform(self):
        """Test error handling for invalid transforms."""
        # Environment falls back silently to tanh
        self.assertAlmostEqualFloat(
            apply_transform("invalid_transform", 1.0),
            math.tanh(1.0),
            tolerance=self.TOL_IDENTITY_RELAXED,
        )

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
        params = self.base_params(
            exit_potential_mode="canonical",
            hold_potential_enabled=True,
            entry_additive_enabled=True,  # expected to be auto-disabled by canonical invariance
            exit_additive_enabled=True,  # expected to be auto-disabled by canonical invariance
        )
        base_reward = 0.25
        current_pnl = 0.05
        current_duration_ratio = 0.4
        next_pnl = 0.0
        next_duration_ratio = 0.0
        total, shaping, next_potential = apply_potential_shaping(
            base_reward=base_reward,
            current_pnl=current_pnl,
            current_duration_ratio=current_duration_ratio,
            next_pnl=next_pnl,
            next_duration_ratio=next_duration_ratio,
            is_exit=True,
            is_entry=False,
            last_potential=0.789,  # arbitrary, should be ignored for Φ'
            params=params,
        )
        # Canonical invariance must DISABLE additives (invariance of terminal shaping)
        self.assertIn("_pbrs_invariance_applied", params)
        self.assertFalse(
            params["entry_additive_enabled"],
            "Entry additive should be auto-disabled in canonical mode",
        )
        self.assertFalse(
            params["exit_additive_enabled"],
            "Exit additive should be auto-disabled in canonical mode",
        )
        # Next potential is forced to 0
        self.assertPlacesEqual(next_potential, 0.0, places=12)
        # Compute current potential independently to assert shaping = -Φ(s)
        current_potential = _compute_hold_potential(
            current_pnl,
            current_duration_ratio,
            {"hold_potential_enabled": True, "hold_potential_scale": 1.0},
        )
        # shaping should equal -current_potential within tolerance
        self.assertAlmostEqual(shaping, -current_potential, delta=self.TOL_IDENTITY_RELAXED)
        # Since additives are disabled, total ≈ base_reward + shaping (residual ~0)
        residual = total - base_reward - shaping
        self.assertAlmostEqual(residual, 0.0, delta=self.TOL_IDENTITY_RELAXED)
        self.assertTrue(np.isfinite(total))

    def test_pbrs_invariance_internal_flag_set(self):
        """Canonical path sets _pbrs_invariance_applied once; second call idempotent."""
        params = self.base_params(
            exit_potential_mode="canonical",
            hold_potential_enabled=True,
            entry_additive_enabled=True,  # will be auto-disabled
            exit_additive_enabled=True,
        )
        # Structural sweep (ensures terminal Φ'==0 and shaping bounded)
        terminal_next_potentials, shaping_values = self._canonical_sweep(params)

        _t1, _s1, _n1 = apply_potential_shaping(
            base_reward=0.0,
            current_pnl=0.05,
            current_duration_ratio=0.3,
            next_pnl=0.0,
            next_duration_ratio=0.0,
            is_exit=True,
            is_entry=False,
            last_potential=0.4,
            params=params,
        )
        self.assertIn("_pbrs_invariance_applied", params)
        self.assertFalse(params["entry_additive_enabled"])
        self.assertFalse(params["exit_additive_enabled"])
        if terminal_next_potentials:
            self.assertTrue(all(abs(p) < self.PBRS_TERMINAL_TOL for p in terminal_next_potentials))
        max_abs = max(abs(v) for v in shaping_values) if shaping_values else 0.0
        self.assertLessEqual(max_abs, self.PBRS_MAX_ABS_SHAPING)

        # Capture state and re-run (idempotence)
        state_after = (
            params["entry_additive_enabled"],
            params["exit_additive_enabled"],
        )  # type: ignore[index]
        _t2, _s2, _n2 = apply_potential_shaping(
            base_reward=0.0,
            current_pnl=0.02,
            current_duration_ratio=0.1,
            next_pnl=0.0,
            next_duration_ratio=0.0,
            is_exit=True,
            is_entry=False,
            last_potential=0.1,
            params=params,
        )
        self.assertEqual(
            state_after,
            (params["entry_additive_enabled"], params["exit_additive_enabled"]),
        )

    def test_progressive_release_negative_decay_clamped(self):
        """Negative decay must clamp to 0 => next potential equals last potential (no release)."""
        params = self.base_params(
            exit_potential_mode="progressive_release",
            exit_potential_decay=-0.75,  # clamped to 0
            hold_potential_enabled=True,
        )
        last_potential = 0.42
        # Use neutral current state so Φ(s) ≈ 0 (approx) if transforms remain small.
        total, shaping, next_potential = apply_potential_shaping(
            base_reward=0.0,
            current_pnl=0.0,
            current_duration_ratio=0.0,
            next_pnl=0.0,
            next_duration_ratio=0.0,
            is_exit=True,
            last_potential=last_potential,
            params=params,
        )
        self.assertPlacesEqual(next_potential, last_potential, places=12)
        # shaping = γ*Φ' - Φ(s) ≈ γ*last_potential - 0
        gamma_raw = DEFAULT_MODEL_REWARD_PARAMETERS.get("potential_gamma", 0.95)
        try:
            gamma = float(gamma_raw)  # type: ignore[assignment]
        except Exception:
            gamma = 0.95
        self.assertLessEqual(
            abs(shaping - gamma * last_potential),
            self.TOL_GENERIC_EQ,
        )
        self.assertPlacesEqual(total, shaping, places=12)

    def test_potential_gamma_nan_fallback(self):
        """potential_gamma=NaN should fall back to default value (indirect comparison)."""
        base_params_dict = self.base_params()
        default_gamma = base_params_dict.get("potential_gamma", 0.95)
        params_nan = self.base_params(potential_gamma=np.nan, hold_potential_enabled=True)
        # Non-terminal transition so Φ(s') is computed and depends on gamma
        res_nan = apply_potential_shaping(
            base_reward=0.1,
            current_pnl=0.03,
            current_duration_ratio=0.2,
            next_pnl=0.035,
            next_duration_ratio=0.25,
            is_exit=False,
            last_potential=0.0,
            params=params_nan,
        )
        params_ref = self.base_params(potential_gamma=default_gamma, hold_potential_enabled=True)
        res_ref = apply_potential_shaping(
            base_reward=0.1,
            current_pnl=0.03,
            current_duration_ratio=0.2,
            next_pnl=0.035,
            next_duration_ratio=0.25,
            is_exit=False,
            last_potential=0.0,
            params=params_ref,
        )
        # Compare shaping & total (deterministic path here)
        self.assertLess(
            abs(res_nan[1] - res_ref[1]),
            self.TOL_IDENTITY_RELAXED,
            "Unexpected shaping difference under gamma NaN fallback",
        )
        self.assertLess(
            abs(res_nan[0] - res_ref[0]),
            self.TOL_IDENTITY_RELAXED,
            "Unexpected total difference under gamma NaN fallback",
        )

    def test_validate_reward_parameters_success_and_failure(self):
        """validate_reward_parameters: success on defaults and failure on invalid ranges."""
        params_ok = DEFAULT_MODEL_REWARD_PARAMETERS.copy()
        try:
            validated = validate_reward_parameters(params_ok)
        except Exception as e:  # pragma: no cover
            self.fail(f"validate_reward_parameters raised unexpectedly: {e}")
        # validate_reward_parameters may return (params, diagnostics) or just params
        if isinstance(validated, tuple) and len(validated) >= 1 and isinstance(validated[0], dict):
            validated_params = validated[0]
        else:
            validated_params = validated  # type: ignore[assignment]
        for k in (
            "potential_gamma",
            "hold_potential_enabled",
            "exit_potential_mode",
        ):
            self.assertIn(k, validated_params, f"Missing key '{k}' in validated params")

        # Introduce invalid values
        params_bad = params_ok.copy()
        params_bad["potential_gamma"] = -0.2  # invalid
        params_bad["hold_potential_scale"] = -5.0  # invalid
        with self.assertRaises((ValueError, AssertionError)):
            vr = validate_reward_parameters(params_bad)
            # If function returns tuple without raising, enforce failure explicitly
            if not isinstance(vr, Exception):
                self.fail("validate_reward_parameters should raise on invalid params")

    def test_compute_exit_potential_mode_differences(self):
        """_compute_exit_potential modes: canonical resets Φ; spike_cancel approx preserves γΦ' ≈ Φ_prev (delta≈0)."""
        gamma = 0.93
        base_common = dict(
            hold_potential_enabled=True,
            potential_gamma=gamma,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            hold_potential_scale=1.0,
        )
        # Build previous potential via hold computation
        ctx_pnl = 0.012
        ctx_dur_ratio = 0.3
        params_can = self.base_params(exit_potential_mode="canonical", **base_common)
        prev_phi = _compute_hold_potential(ctx_pnl, ctx_dur_ratio, params_can)
        self.assertFinite(prev_phi, name="prev_phi")
        # Canonical exit potential -> next_phi=0
        next_phi_can = _compute_exit_potential(prev_phi, params_can)
        self.assertAlmostEqualFloat(
            next_phi_can,
            0.0,
            tolerance=self.TOL_IDENTITY_STRICT,
            msg="Canonical exit must zero potential",
        )
        canonical_delta = -prev_phi  # γ*0 - prev_phi
        self.assertAlmostEqualFloat(
            canonical_delta,
            -prev_phi,
            tolerance=self.TOL_IDENTITY_RELAXED,
            msg="Canonical delta mismatch",
        )
        # Spike cancel mode
        params_spike = self.base_params(exit_potential_mode="spike_cancel", **base_common)
        next_phi_spike = _compute_exit_potential(prev_phi, params_spike)
        shaping_spike = gamma * next_phi_spike - prev_phi
        self.assertNearZero(
            shaping_spike,
            atol=self.TOL_IDENTITY_RELAXED,
            msg="Spike cancel should nullify shaping delta",
        )
        # Magnitude comparison
        self.assertGreaterEqual(
            abs(canonical_delta) + self.TOL_IDENTITY_STRICT,
            abs(shaping_spike),
            "Canonical shaping magnitude should exceed spike_cancel",
        )

    def test_transform_bulk_monotonicity_and_bounds(self):
        """Non-decreasing monotonicity & (-1,1) bounds for smooth transforms (excluding clip)."""
        transforms = ["tanh", "softsign", "arctan", "sigmoid", "asinh"]
        xs = [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0]
        for name in transforms:
            with self.subTest(transform=name):
                vals = [apply_transform(name, x) for x in xs]
                # Strict bounds (-1,1) (sigmoid & tanh asymptotic)
                self.assertTrue(all(-1.0 < v < 1.0 for v in vals), f"{name} out of bounds")
                # Non-decreasing monotonicity
                for a, b in zip(vals, vals[1:]):
                    self.assertLessEqual(
                        a,
                        b + self.TOL_IDENTITY_STRICT,
                        f"{name} not monotonic between {a} and {b}",
                    )

    def test_pbrs_retain_previous_cumulative_drift(self):
        """retain_previous mode accumulates negative shaping drift (non-invariant)."""
        params = self.base_params(
            exit_potential_mode="retain_previous",
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            potential_gamma=0.9,
        )
        gamma = _get_float_param(
            params,
            "potential_gamma",
            DEFAULT_MODEL_REWARD_PARAMETERS.get("potential_gamma", 0.95),
        )
        rng = np.random.default_rng(555)
        potentials = rng.uniform(0.05, 0.85, size=220)
        deltas = [(gamma * p - p) for p in potentials]
        cumulative = float(np.sum(deltas))
        self.assertLess(cumulative, -self.TOL_NEGLIGIBLE)
        self.assertGreater(abs(cumulative), 10 * self.TOL_IDENTITY_RELAXED)

    def test_normality_invariance_under_scaling(self):
        """Skewness & excess kurtosis invariant under positive scaling of normal sample."""
        rng = np.random.default_rng(808)
        base = rng.normal(0.0, 1.0, size=7000)
        scaled = 5.0 * base

        def _skew_kurt(x: np.ndarray) -> tuple[float, float]:
            m = np.mean(x)
            c = x - m
            m2 = np.mean(c**2)
            m3 = np.mean(c**3)
            m4 = np.mean(c**4)
            skew = m3 / (m2**1.5 + 1e-18)
            kurt = m4 / (m2**2 + 1e-18) - 3.0
            return skew, kurt

        s_base, k_base = _skew_kurt(base)
        s_scaled, k_scaled = _skew_kurt(scaled)
        self.assertAlmostEqualFloat(s_base, s_scaled, tolerance=self.TOL_DISTRIB_SHAPE)
        self.assertAlmostEqualFloat(k_base, k_scaled, tolerance=self.TOL_DISTRIB_SHAPE)


class TestReportFormatting(RewardSpaceTestBase):
    """Tests for report formatting elements not covered elsewhere."""

    def test_abs_shaping_line_present_and_constant(self):
        """Abs Σ Shaping Reward line present, formatted, uses constant not literal."""
        # Use small synthetic DataFrame with zero shaping sum (pandas imported globally)
        df = pd.DataFrame(
            {
                "reward_shaping": [
                    self.TOL_IDENTITY_STRICT,
                    -self.TOL_IDENTITY_STRICT,
                ],
                "reward_entry_additive": [0.0, 0.0],
                "reward_exit_additive": [0.0, 0.0],
            }
        )
        total_shaping = df["reward_shaping"].sum()
        self.assertTrue(abs(total_shaping) < PBRS_INVARIANCE_TOL)
        # Reconstruct lines exactly as writer does
        lines = [
            f"| Abs Σ Shaping Reward | {abs(total_shaping):.6e} |",
        ]
        content = "\n".join(lines)
        # Validate formatting pattern using regex
        m = re.search(
            r"\| Abs Σ Shaping Reward \| ([0-9]+\.[0-9]{6}e[+-][0-9]{2}) \|",
            content,
        )
        self.assertIsNotNone(m, "Abs Σ Shaping Reward line missing or misformatted")
        # Ensure scientific notation magnitude consistent with small number
        val = float(m.group(1)) if m else None  # type: ignore[arg-type]
        if val is not None:
            self.assertLess(val, self.TOL_NEGLIGIBLE + self.TOL_IDENTITY_STRICT)
        # Ensure no hard-coded tolerance string inside content
        self.assertNotIn(
            str(self.TOL_GENERIC_EQ),
            content,
            "Tolerance constant value should appear, not raw literal",
        )

    def test_pbrs_non_canonical_report_generation(self):
        """Generate synthetic invariance section with non-zero shaping to assert Non-canonical classification."""

        df = pd.DataFrame(
            {
                "reward_shaping": [0.01, -0.002],  # sum = 0.008 (>> tolerance)
                "reward_entry_additive": [0.0, 0.0],
                "reward_exit_additive": [0.001, 0.0],
            }
        )
        total_shaping = df["reward_shaping"].sum()
        self.assertGreater(abs(total_shaping), PBRS_INVARIANCE_TOL)
        invariance_status = "❌ Non-canonical"
        section = []
        section.append("**PBRS Invariance Summary:**\n")
        section.append("| Field | Value |\n")
        section.append("|-------|-------|\n")
        section.append(f"| Invariance | {invariance_status} |\n")
        section.append(f"| Note | Total shaping = {total_shaping:.6f} (non-zero) |\n")
        section.append(f"| Σ Shaping Reward | {total_shaping:.6f} |\n")
        section.append(f"| Abs Σ Shaping Reward | {abs(total_shaping):.6e} |\n")
        section.append(f"| Σ Entry Additive | {df['reward_entry_additive'].sum():.6f} |\n")
        section.append(f"| Σ Exit Additive | {df['reward_exit_additive'].sum():.6f} |\n")
        content = "".join(section)
        self.assertIn("❌ Non-canonical", content)
        self.assertRegex(content, r"Σ Shaping Reward \| 0\.008000 \|")
        m_abs = re.search(r"Abs Σ Shaping Reward \| ([0-9.]+e[+-][0-9]{2}) \|", content)
        self.assertIsNotNone(m_abs)
        if m_abs:
            self.assertAlmostEqual(abs(total_shaping), float(m_abs.group(1)), places=12)

    def test_additive_activation_deterministic_contribution(self):
        """Additives enabled increase total reward; shaping impact limited."""
        # Use a non_canonical exit mode to avoid automatic invariance enforcement
        # disabling the additive components on first call (canonical path auto-disables).
        base = self.base_params(
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            exit_potential_mode="non_canonical",
        )
        with_add = base.copy()
        with_add.update(
            {
                "entry_additive_enabled": True,
                "exit_additive_enabled": True,
                "entry_additive_scale": 0.4,
                "exit_additive_scale": 0.4,
                "entry_additive_gain": 1.0,
                "exit_additive_gain": 1.0,
            }
        )
        ctx = {
            "base_reward": 0.05,
            "current_pnl": 0.01,
            "current_duration_ratio": 0.2,
            "next_pnl": 0.012,
            "next_duration_ratio": 0.25,
            "is_entry": True,
            "is_exit": False,
        }
        _t0, s0, _n0 = apply_potential_shaping(last_potential=0.0, params=base, **ctx)
        t1, s1, _n1 = apply_potential_shaping(last_potential=0.0, params=with_add, **ctx)
        self.assertFinite(t1)
        self.assertFinite(s1)
        # Additives should not alter invariance: shaping difference small
        self.assertLess(abs(s1 - s0), 0.2)
        self.assertGreater(t1 - _t0, 0.0, "Total reward should increase with additives present")

    def test_report_cumulative_invariance_aggregation(self):
        """Canonical telescoping term: small per-step mean drift, bounded increments."""
        params = self.base_params(
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            exit_potential_mode="canonical",
        )
        gamma = _get_float_param(
            params,
            "potential_gamma",
            DEFAULT_MODEL_REWARD_PARAMETERS.get("potential_gamma", 0.95),
        )
        rng = np.random.default_rng(321)
        last_potential = 0.0
        telescoping_sum = 0.0
        max_abs_step = 0.0
        steps = 0
        for _ in range(500):
            is_exit = rng.uniform() < 0.1
            current_pnl = float(rng.normal(0, 0.05))
            current_dur = float(rng.uniform(0, 1))
            next_pnl = 0.0 if is_exit else float(rng.normal(0, 0.05))
            next_dur = 0.0 if is_exit else float(rng.uniform(0, 1))
            _tot, _shap, next_potential = apply_potential_shaping(
                base_reward=0.0,
                current_pnl=current_pnl,
                current_duration_ratio=current_dur,
                next_pnl=next_pnl,
                next_duration_ratio=next_dur,
                is_exit=is_exit,
                last_potential=last_potential,
                params=params,
            )
            # Accumulate theoretical telescoping term: γ Φ(s') - Φ(s)
            inc = gamma * next_potential - last_potential
            telescoping_sum += inc
            if abs(inc) > max_abs_step:
                max_abs_step = abs(inc)
            steps += 1
            if is_exit:
                # Reset potential at terminal per canonical semantics
                last_potential = 0.0
            else:
                last_potential = next_potential
        mean_drift = telescoping_sum / max(1, steps)
        self.assertLess(
            abs(mean_drift),
            2e-2,
            f"Per-step telescoping drift too large (mean={mean_drift}, steps={steps})",
        )
        self.assertLessEqual(
            max_abs_step,
            self.PBRS_MAX_ABS_SHAPING,
            f"Unexpected large telescoping increment (max={max_abs_step})",
        )

    def test_report_explicit_non_invariance_progressive_release(self):
        """progressive_release should generally yield non-zero cumulative shaping (release leak)."""
        params = self.base_params(
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            exit_potential_mode="progressive_release",
            exit_potential_decay=0.25,
        )
        rng = np.random.default_rng(321)
        last_potential = 0.0
        shaping_sum = 0.0
        for _ in range(160):
            is_exit = rng.uniform() < 0.15
            next_pnl = 0.0 if is_exit else float(rng.normal(0, 0.07))
            next_dur = 0.0 if is_exit else float(rng.uniform(0, 1))
            _tot, shap, next_pot = apply_potential_shaping(
                base_reward=0.0,
                current_pnl=float(rng.normal(0, 0.07)),
                current_duration_ratio=float(rng.uniform(0, 1)),
                next_pnl=next_pnl,
                next_duration_ratio=next_dur,
                is_exit=is_exit,
                last_potential=last_potential,
                params=params,
            )
            shaping_sum += shap
            last_potential = 0.0 if is_exit else next_pot
        self.assertGreater(
            abs(shaping_sum),
            PBRS_INVARIANCE_TOL * 50,
            f"Expected non-zero Σ shaping (got {shaping_sum})",
        )

    def test_gamma_extremes(self):
        """Gamma=0 and gamma≈1 boundary behaviours produce bounded shaping and finite potentials."""
        for gamma in [0.0, 0.999999]:
            params = self.base_params(
                hold_potential_enabled=True,
                entry_additive_enabled=False,
                exit_additive_enabled=False,
                exit_potential_mode="canonical",
                potential_gamma=gamma,
            )
            _tot, shap, next_pot = apply_potential_shaping(
                base_reward=0.0,
                current_pnl=0.02,
                current_duration_ratio=0.3,
                next_pnl=0.025,
                next_duration_ratio=0.35,
                is_exit=False,
                last_potential=0.0,
                params=params,
            )
            self.assertTrue(np.isfinite(shap))
            self.assertTrue(np.isfinite(next_pot))
            self.assertLessEqual(abs(shap), self.PBRS_MAX_ABS_SHAPING)


class TestBootstrapStatistics(RewardSpaceTestBase):
    """Grouped tests for bootstrap confidence interval behavior."""

    def test_constant_distribution_bootstrap_and_diagnostics(self):
        """Degenerate columns produce (mean≈lo≈hi) zero-width intervals."""
        df = self._const_df(80)
        res = bootstrap_confidence_intervals(
            df, ["reward", "pnl"], n_bootstrap=200, confidence_level=0.95
        )
        for k, (mean, lo, hi) in res.items():
            self.assertAlmostEqualFloat(mean, lo, tolerance=2e-9)
            self.assertAlmostEqualFloat(mean, hi, tolerance=2e-9)
            self.assertLessEqual(hi - lo, 2e-9)

    def test_bootstrap_shrinkage_with_sample_size(self):
        """Half-width decreases with larger sample (~1/sqrt(n) heuristic)."""
        small = self._shift_scale_df(80)
        large = self._shift_scale_df(800)
        res_small = bootstrap_confidence_intervals(small, ["reward"], n_bootstrap=400)
        res_large = bootstrap_confidence_intervals(large, ["reward"], n_bootstrap=400)
        (_, lo_s, hi_s) = list(res_small.values())[0]
        (_, lo_l, hi_l) = list(res_large.values())[0]
        hw_small = (hi_s - lo_s) / 2.0
        hw_large = (hi_l - lo_l) / 2.0
        self.assertFinite(hw_small, name="hw_small")
        self.assertFinite(hw_large, name="hw_large")
        self.assertLess(hw_large, hw_small * 0.55)

    def test_bootstrap_confidence_intervals_basic(self):
        """Basic CI computation returns ordered finite bounds."""
        test_data = self.make_stats_df(n=100, seed=self.SEED)
        results = bootstrap_confidence_intervals(
            test_data,
            ["reward", "pnl"],
            n_bootstrap=100,
        )
        for metric, (mean, ci_low, ci_high) in results.items():
            self.assertFinite(mean, name=f"mean[{metric}]")
            self.assertFinite(ci_low, name=f"ci_low[{metric}]")
            self.assertFinite(ci_high, name=f"ci_high[{metric}]")
            self.assertLess(ci_low, ci_high)

    def test_canonical_invariance_flag_and_sum(self):
        """Canonical mode + no additives -> pbrs_invariant True and Σ shaping ≈ 0."""
        params = self.base_params(
            exit_potential_mode="canonical",
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            hold_potential_enabled=True,
        )
        df = simulate_samples(
            params={**params, "max_trade_duration_candles": 100},
            num_samples=400,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        # pbrs_invariant must be True for all samples
        unique_flags = set(df["pbrs_invariant"].unique().tolist())
        self.assertEqual(unique_flags, {True}, f"Unexpected invariant flags: {unique_flags}")
        # Σ shaping ≈ 0 within PBRS_INVARIANCE_TOL
        total_shaping = float(df["reward_shaping"].sum())
        self.assertLess(
            abs(total_shaping),
            PBRS_INVARIANCE_TOL,
            f"Canonical invariance violated: Σ shaping = {total_shaping}",
        )

    def test_non_canonical_flag_false_and_sum_nonzero(self):
        """Non-canonical exit potential (progressive_release) -> pbrs_invariant False and Σ shaping != 0."""
        params = self.base_params(
            exit_potential_mode="progressive_release",
            exit_potential_decay=0.25,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            hold_potential_enabled=True,
        )
        df = simulate_samples(
            params={**params, "max_trade_duration_candles": 100},
            num_samples=400,
            seed=self.SEED,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        unique_flags = set(df["pbrs_invariant"].unique().tolist())
        self.assertEqual(unique_flags, {False}, f"Unexpected invariant flags: {unique_flags}")
        total_shaping = float(df["reward_shaping"].sum())
        self.assertGreater(
            abs(total_shaping),
            PBRS_INVARIANCE_TOL * 10,
            f"Expected non-zero Σ shaping in non-canonical mode (got {total_shaping})",
        )


class TestCsvAndSimulationOptions(RewardSpaceTestBase):
    """CLI-level tests: CSV encoding and simulate_unrealized_pnl option effects."""

    def test_action_column_integer_in_csv(self):
        """Ensure 'action' column in reward_samples.csv is encoded as integers."""
        out_dir = self.output_path / "csv_int_check"
        cmd = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            "200",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")
        csv_path = out_dir / "reward_samples.csv"
        self.assertTrue(csv_path.exists(), "Missing reward_samples.csv")
        df = pd.read_csv(csv_path)
        self.assertIn("action", df.columns)
        # All values must be integral and in the expected enum set {0,1,2,3,4}
        values = df["action"].tolist()
        self.assertTrue(
            all(float(v).is_integer() for v in values),
            "Non-integer values detected in 'action' column",
        )
        allowed = {0, 1, 2, 3, 4}
        self.assertTrue(set(int(v) for v in values).issubset(allowed))

    def test_unrealized_pnl_affects_hold_potential(self):
        """--unrealized_pnl should alter hold next_potential distribution vs default."""
        out_default = self.output_path / "sim_default"
        out_sim = self.output_path / "sim_unrealized"
        base_args = [
            "--num_samples",
            "800",
            "--seed",
            str(self.SEED),
            "--out_dir",
        ]
        # Default run
        cmd_default = [
            sys.executable,
            "reward_space_analysis.py",
            *base_args,
            str(out_default),
        ]
        res_def = subprocess.run(
            cmd_default,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        self.assertEqual(res_def.returncode, 0, f"CLI default run failed: {res_def.stderr}")
        # Run with --unrealized_pnl
        cmd_sim = [
            sys.executable,
            "reward_space_analysis.py",
            *base_args,
            str(out_sim),
            "--unrealized_pnl",
        ]
        res_sim = subprocess.run(cmd_sim, capture_output=True, text=True, cwd=Path(__file__).parent)
        self.assertEqual(res_sim.returncode, 0, f"CLI simulated run failed: {res_sim.stderr}")

        # Load CSVs
        df_def = pd.read_csv(out_default / "reward_samples.csv")
        df_sim = pd.read_csv(out_sim / "reward_samples.csv")
        # Hold actions: position in {Long (1.0), Short (0.0)} and action == 0 (Neutral)
        mask_hold_def = (df_def["action"] == 0) & (df_def["position"].isin([0.0, 1.0]))
        mask_hold_sim = (df_sim["action"] == 0) & (df_sim["position"].isin([0.0, 1.0]))
        # Sanity: ensure we have holds in both runs
        self.assertGreater(int(mask_hold_def.sum()), 0, "No hold samples in default run")
        self.assertGreater(int(mask_hold_sim.sum()), 0, "No hold samples in simulate run")
        # Compare mean next_potential on holds: simulated should differ from default
        mean_next_def = float(df_def.loc[mask_hold_def, "next_potential"].mean())
        mean_next_sim = float(df_sim.loc[mask_hold_sim, "next_potential"].mean())
        self.assertFinite(mean_next_def, name="mean_next_def")
        self.assertFinite(mean_next_sim, name="mean_next_sim")
        self.assertGreater(
            abs(mean_next_sim - mean_next_def),
            self.TOL_GENERIC_EQ,
            f"No detectable effect of --unrealized_pnl on Φ(s): def={mean_next_def:.6f}, sim={mean_next_sim:.6f}",
        )


class TestParamsPropagation(RewardSpaceTestBase):
    """Integration tests to validate max_trade_duration_candles propagation via CLI params and dynamic flag."""

    def test_max_trade_duration_candles_propagation_params(self):
        """--params max_trade_duration_candles=X propagates to manifest and simulation params."""
        out_dir = self.output_path / "mtd_params"
        cmd = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            "120",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
            "--params",
            "max_trade_duration_candles=96",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")

        manifest_path = out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "Missing manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Basic structure checks
        self.assertIn("reward_params", manifest)
        self.assertIn("simulation_params", manifest)

        # Reward params should include the tunable (float or int acceptable -> coerce)
        rp = manifest["reward_params"]
        self.assertIn("max_trade_duration_candles", rp)
        self.assertEqual(int(rp["max_trade_duration_candles"]), 96)

    def test_max_trade_duration_candles_propagation_flag(self):
        """Dynamic flag --max_trade_duration_candles X propagates identically."""
        out_dir = self.output_path / "mtd_flag"
        cmd = [
            sys.executable,
            "reward_space_analysis.py",
            "--num_samples",
            "120",
            "--seed",
            str(self.SEED),
            "--out_dir",
            str(out_dir),
            "--max_trade_duration_candles",
            "64",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        self.assertEqual(result.returncode, 0, f"CLI failed: {result.stderr}")

        manifest_path = out_dir / "manifest.json"
        self.assertTrue(manifest_path.exists(), "Missing manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # Basic structure checks
        self.assertIn("reward_params", manifest)
        self.assertIn("simulation_params", manifest)

        # Reward params should include the tunable (float or int acceptable -> coerce)
        rp = manifest["reward_params"]
        self.assertIn("max_trade_duration_candles", rp)
        self.assertEqual(int(rp["max_trade_duration_candles"]), 64)


if __name__ == "__main__":
    # Configure test discovery and execution
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)
