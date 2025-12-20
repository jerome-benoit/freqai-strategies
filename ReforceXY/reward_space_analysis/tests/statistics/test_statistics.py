#!/usr/bin/env python3
"""Statistical tests, distribution metrics, and bootstrap validation."""

import unittest

import numpy as np
import pandas as pd
import pytest

from reward_space_analysis import (
    RewardDiagnosticsWarning,
    _binned_stats,
    _compute_relationship_stats,
    bootstrap_confidence_intervals,
    compute_distribution_shift_metrics,
    distribution_diagnostics,
    simulate_samples,
    statistical_hypothesis_tests,
)

from ..constants import (
    PARAMS,
    SCENARIOS,
    SEEDS,
    STAT_TOL,
    STATISTICAL,
    TOLERANCE,
)
from ..helpers import assert_diagnostic_warning
from ..test_base import RewardSpaceTestBase

pytestmark = pytest.mark.statistics


class TestStatistics(RewardSpaceTestBase):
    """Statistical tests: metrics, diagnostics, bootstrap, correlations."""

    def test_statistics_feature_analysis_skip_partial_dependence(self):
        """Invariant 107: skip_partial_dependence=True yields empty partial_deps."""
        try:
            from reward_space_analysis import _perform_feature_analysis  # type: ignore
        except ImportError:
            self.skipTest("sklearn not available; skipping feature analysis invariance test")
        # Use existing helper to get synthetic stats df (small for speed)
        df = self.make_stats_df(n=120, seed=SEEDS.BASE, idle_pattern="mixed")
        importance_df, analysis_stats, partial_deps, model = _perform_feature_analysis(
            df, seed=SEEDS.BASE, skip_partial_dependence=True, rf_n_jobs=1, perm_n_jobs=1
        )
        self.assertIsInstance(importance_df, pd.DataFrame)
        self.assertIsInstance(analysis_stats, dict)
        self.assertEqual(
            partial_deps, {}, "partial_deps must be empty when skip_partial_dependence=True"
        )

    def test_statistics_binned_stats_invalid_bins_raises(self):
        """Invariant 110: _binned_stats must raise ValueError for <2 bin edges."""

        df = self.make_stats_df(n=50, seed=SEEDS.BASE)
        with self.assertRaises(ValueError):
            _binned_stats(df, "idle_duration", "reward_idle", [0.0])  # single edge invalid
        # Control: valid case should not raise and produce frame
        result = _binned_stats(df, "idle_duration", "reward_idle", [0.0, 10.0, 20.0])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreaterEqual(len(result), 1)

    def test_statistics_correlation_dropped_constant_columns(self):
        """Invariant 111: constant columns are listed in correlation_dropped and excluded."""

        df = self.make_stats_df(n=90, seed=SEEDS.BASE)
        # Force some columns constant
        df.loc[:, "reward_hold"] = 0.0
        df.loc[:, "idle_duration"] = 5.0
        stats_rel = _compute_relationship_stats(df)
        dropped = stats_rel["correlation_dropped"]
        self.assertIn("reward_hold", dropped)
        self.assertIn("idle_duration", dropped)
        corr = stats_rel["correlation"]
        self.assertIsInstance(corr, pd.DataFrame)
        self.assertNotIn("reward_hold", corr.columns)
        self.assertNotIn("idle_duration", corr.columns)

    def test_statistics_distribution_shift_metrics_degenerate_zero(self):
        """Invariant 112: degenerate distributions yield zero shift metrics and KS p=1.0."""
        # Build two identical constant distributions (length >=10)
        n = 40
        df_const = pd.DataFrame(
            {
                "pnl": np.zeros(n),
                "trade_duration": np.ones(n) * 7.0,
                "idle_duration": np.ones(n) * 3.0,
            }
        )
        metrics = compute_distribution_shift_metrics(df_const, df_const.copy())
        # Each feature should have zero metrics and ks_pvalue=1.0
        for feature in ["pnl", "trade_duration", "idle_duration"]:
            for suffix in ["kl_divergence", "js_distance", "wasserstein", "ks_statistic"]:
                key = f"{feature}_{suffix}"
                if key in metrics:
                    self.assertPlacesEqual(
                        float(metrics[key]),
                        0.0,
                        places=TOLERANCE.DECIMAL_PLACES_STRICT,
                        msg=f"Expected 0 for {key}",
                    )
            p_key = f"{feature}_ks_pvalue"
            if p_key in metrics:
                self.assertPlacesEqual(
                    float(metrics[p_key]),
                    1.0,
                    places=TOLERANCE.DECIMAL_PLACES_STRICT,
                    msg=f"Expected 1.0 for {p_key}",
                )

    def test_statistics_distribution_shift_metrics(self):
        """KL/JS/Wasserstein metrics."""
        df1 = self._make_idle_variance_df(100)
        df2 = self._make_idle_variance_df(100)
        df2["reward"] += 0.1
        metrics = compute_distribution_shift_metrics(df1, df2)
        expected_keys = {
            "pnl_kl_divergence",
            "pnl_js_distance",
            "pnl_wasserstein",
            "pnl_ks_statistic",
        }
        actual_keys = set(metrics.keys())
        matching_keys = expected_keys.intersection(actual_keys)
        self.assertGreater(
            len(matching_keys), 0, f"Should have some distribution metrics. Got: {actual_keys}"
        )
        for metric_name, value in metrics.items():
            if "pnl" in metric_name:
                if any(
                    (
                        suffix in metric_name
                        for suffix in [
                            "js_distance",
                            "ks_statistic",
                            "wasserstein",
                            "kl_divergence",
                        ]
                    )
                ):
                    self.assertDistanceMetric(value, name=metric_name)
                else:
                    self.assertFinite(value, name=metric_name)

    def test_statistics_distribution_shift_identity_null_metrics(self):
        """Identity distributions -> near-zero shift metrics."""
        df = self._make_idle_variance_df(180)
        metrics_id = compute_distribution_shift_metrics(df, df.copy())
        for name, val in metrics_id.items():
            if name.endswith(("_kl_divergence", "_js_distance", "_wasserstein")):
                self.assertLess(
                    abs(val),
                    TOLERANCE.GENERIC_EQ,
                    f"Metric {name} expected ≈ 0 on identical distributions (got {val})",
                )
            elif name.endswith("_ks_statistic"):
                self.assertLess(
                    abs(val),
                    STAT_TOL.KS_STATISTIC_IDENTITY,
                    f"KS statistic should be near 0 on identical distributions (got {val})",
                )

    def test_statistics_hypothesis_testing(self):
        """Light correlation sanity check."""
        df = self._make_idle_variance_df(200)
        if len(df) > 30:
            idle_data = df[df["idle_duration"] > 0]
            if len(idle_data) > 10:
                idle_dur = idle_data["idle_duration"].to_numpy()
                idle_rew = idle_data["reward_idle"].to_numpy()
                self.assertTrue(
                    len(idle_dur) == len(idle_rew),
                    "Idle duration and reward arrays should have same length",
                )
                self.assertTrue(
                    all((d >= 0 for d in idle_dur)), "Idle durations should be non-negative"
                )
                negative_rewards = (idle_rew < 0).sum()
                total_rewards = len(idle_rew)
                negative_ratio = negative_rewards / total_rewards
                self.assertGreater(
                    negative_ratio, 0.5, "Most idle rewards should be negative (penalties)"
                )

    def test_statistics_distribution_constant_fallback_diagnostics(self):
        """Invariant 115: constant distribution triggers fallback diagnostics (zero moments, qq_r2=1.0)."""
        # Build constant reward/pnl columns to force degenerate stats
        n = 60
        df_const = pd.DataFrame(
            {
                "reward": np.zeros(n),
                "reward_idle": np.zeros(n),
                "reward_hold": np.zeros(n),
                "pnl": np.zeros(n),
                "pnl_raw": np.zeros(n),
            }
        )
        diagnostics = distribution_diagnostics(df_const)
        # Mean and std for constant arrays
        for key in ["reward_mean", "reward_std", "pnl_mean", "pnl_std"]:
            if key in diagnostics:
                self.assertAlmostEqualFloat(
                    float(diagnostics[key]), 0.0, tolerance=TOLERANCE.IDENTITY_RELAXED
                )
        # Skewness & kurtosis fallback to INTERNAL_GUARDS['distribution_constant_fallback_moment'] (0.0)
        for key in ["reward_skewness", "reward_kurtosis", "pnl_skewness", "pnl_kurtosis"]:
            if key in diagnostics:
                self.assertAlmostEqualFloat(
                    float(diagnostics[key]), 0.0, tolerance=TOLERANCE.IDENTITY_RELAXED
                )
        # Q-Q plot r2 fallback value
        qq_key = next((k for k in diagnostics if k.endswith("_qq_r2")), None)
        if qq_key is not None:
            self.assertAlmostEqualFloat(
                float(diagnostics[qq_key]), 1.0, tolerance=TOLERANCE.IDENTITY_RELAXED
            )
        # All diagnostic values finite
        for k, v in diagnostics.items():
            self.assertFinite(v, name=k)

    def test_stats_distribution_diagnostics(self):
        """Distribution diagnostics."""
        df = self._make_idle_variance_df(100)
        diagnostics = distribution_diagnostics(df)
        expected_prefixes = ["reward_", "pnl_"]
        for prefix in expected_prefixes:
            matching_keys = [key for key in diagnostics.keys() if key.startswith(prefix)]
            self.assertGreater(len(matching_keys), 0, f"Should have diagnostics for {prefix}")
            expected_suffixes = ["mean", "std", "skewness", "kurtosis"]
            for suffix in expected_suffixes:
                key = f"{prefix}{suffix}"
                if key in diagnostics:
                    self.assertFinite(diagnostics[key], name=key)

    def test_statistical_hypothesis_tests_api_integration(self):
        """Test statistical_hypothesis_tests API integration with synthetic data."""
        base = self.make_stats_df(n=200, seed=SEEDS.BASE, idle_pattern="mixed")
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
            tolerance=TOLERANCE.IDENTITY_STRICT,
            rtol=TOLERANCE.RELATIVE,
        )

    def test_stats_variance_vs_duration_spearman_sign(self):
        """trade_duration up => pnl variance up (rank corr >0)."""
        rng = np.random.default_rng(99)
        n = 250
        trade_duration = np.linspace(1, SCENARIOS.DURATION_LONG, n)
        pnl = rng.normal(0, 1 + trade_duration / 400.0, n)
        ranks_dur = pd.Series(trade_duration).rank().to_numpy()
        ranks_var = pd.Series(np.abs(pnl)).rank().to_numpy()
        rho = np.corrcoef(ranks_dur, ranks_var)[0, 1]
        self.assertFinite(rho, name="spearman_rho")
        self.assertGreater(rho, STAT_TOL.CORRELATION_SIGNIFICANCE)

    def test_stats_scaling_invariance_distribution_metrics(self):
        """Equal scaling keeps KL/JS ≈0."""
        df1 = self._shift_scale_df(SCENARIOS.SAMPLE_SIZE_MEDIUM)
        scale = 3.5
        df2 = df1.copy()
        df2["pnl"] *= scale
        df1["pnl"] *= scale
        metrics = compute_distribution_shift_metrics(df1, df2)
        for k, v in metrics.items():
            if k.endswith("_kl_divergence") or k.endswith("_js_distance"):
                self.assertLess(
                    abs(v),
                    STAT_TOL.DISTRIBUTION_SHIFT,
                    f"Expected near-zero divergence after equal scaling (k={k}, v={v})",
                )

    # Non-owning smoke; ownership: robustness/test_robustness.py:35 (robustness-decomposition-integrity-101)
    @pytest.mark.smoke
    def test_stats_mean_decomposition_consistency(self):
        """Batch mean additivity."""
        df_a = self._shift_scale_df(120)
        df_b = self._shift_scale_df(180, shift=0.2)
        m_concat = pd.concat([df_a["pnl"], df_b["pnl"]]).mean()
        m_weighted = (df_a["pnl"].mean() * len(df_a) + df_b["pnl"].mean() * len(df_b)) / (
            len(df_a) + len(df_b)
        )
        self.assertAlmostEqualFloat(
            m_concat, m_weighted, tolerance=TOLERANCE.IDENTITY_STRICT, rtol=TOLERANCE.RELATIVE
        )

    def test_stats_bh_correction_null_false_positive_rate(self):
        """Null: low BH discovery rate."""

        rng = np.random.default_rng(1234)
        n = SCENARIOS.SAMPLE_SIZE_MEDIUM
        df = pd.DataFrame(
            {
                "pnl": rng.normal(0, 1, n),
                "reward": rng.normal(0, 1, n),
                "idle_duration": rng.exponential(5, n),
            }
        )
        df["reward_idle"] = rng.normal(0, 1, n) * 0.001
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
            self.assertLess(
                rate,
                STATISTICAL.BH_FP_RATE_THRESHOLD,
                f"BH null FP rate too high under null: {rate:.3f}",
            )

    def test_stats_half_life_monotonic_series(self):
        """Smoothed exponential decay monotonic."""
        x = np.arange(0, 80)
        y = np.exp(-x / 15.0)
        rng = np.random.default_rng(5)
        y_noisy = y + rng.normal(0, 0.0001, len(y))
        window = 5
        y_smooth = np.convolve(y_noisy, np.ones(window) / window, mode="valid")
        self.assertMonotonic(y_smooth, non_increasing=True, tolerance=1e-05)

    def test_stats_hypothesis_seed_reproducibility(self):
        """Seed reproducibility for statistical_hypothesis_tests + bootstrap."""
        df = self.make_stats_df(n=300, seed=SEEDS.BASE, idle_pattern="mixed")
        r1 = statistical_hypothesis_tests(df, seed=SEEDS.REPRODUCIBILITY)
        r2 = statistical_hypothesis_tests(df, seed=SEEDS.REPRODUCIBILITY)
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
        metrics = ["reward", "pnl"]
        ci_a = bootstrap_confidence_intervals(
            df, metrics, n_bootstrap=STATISTICAL.BOOTSTRAP_DEFAULT_ITERATIONS, seed=SEEDS.BOOTSTRAP
        )
        ci_b = bootstrap_confidence_intervals(
            df, metrics, n_bootstrap=STATISTICAL.BOOTSTRAP_DEFAULT_ITERATIONS, seed=SEEDS.BOOTSTRAP
        )
        for metric in metrics:
            m_a, lo_a, hi_a = ci_a[metric]
            m_b, lo_b, hi_b = ci_b[metric]
            self.assertAlmostEqualFloat(
                m_a, m_b, tolerance=TOLERANCE.IDENTITY_STRICT, rtol=TOLERANCE.RELATIVE
            )
            self.assertAlmostEqualFloat(
                lo_a, lo_b, tolerance=TOLERANCE.IDENTITY_STRICT, rtol=TOLERANCE.RELATIVE
            )
            self.assertAlmostEqualFloat(
                hi_a, hi_b, tolerance=TOLERANCE.IDENTITY_STRICT, rtol=TOLERANCE.RELATIVE
            )

    def test_stats_distribution_metrics_mathematical_bounds(self):
        """Mathematical bounds and validity of distribution shift metrics."""
        self.seed_all(SEEDS.BASE)
        df1 = pd.DataFrame(
            {
                "pnl": np.random.normal(0, PARAMS.PNL_STD, 500),
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
            num_samples=SCENARIOS.SAMPLE_SIZE_LARGE + 200,
            seed=SEEDS.HETEROSCEDASTICITY,
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=PARAMS.PNL_STD,
            pnl_duration_vol_scale=PARAMS.PNL_DUR_VOL_SCALE,
        )
        exit_data = df[df["reward_exit"] != 0].copy()
        if len(exit_data) < SCENARIOS.SAMPLE_SIZE_TINY:
            self.skipTest("Insufficient exit actions for heteroscedasticity test")
        exit_data["duration_bin"] = pd.cut(
            exit_data["duration_ratio"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )
        variance_by_bin = exit_data.groupby("duration_bin")["pnl"].var().dropna()
        if "Q1" in variance_by_bin.index and "Q4" in variance_by_bin.index:
            self.assertGreater(
                variance_by_bin["Q4"],
                variance_by_bin["Q1"] * STAT_TOL.VARIANCE_RATIO_THRESHOLD,
                "PnL heteroscedasticity: variance should increase with duration",
            )

    def test_stats_statistical_functions_bounds_validation(self):
        """All statistical functions respect bounds."""
        df = self.make_stats_df(n=300, seed=SEEDS.BASE, idle_pattern="all_nonzero")
        diagnostics = distribution_diagnostics(df)
        for col in ["reward", "pnl", "trade_duration", "idle_duration"]:
            if f"{col}_skewness" in diagnostics:
                self.assertFinite(diagnostics[f"{col}_skewness"], name=f"skewness[{col}]")
            if f"{col}_kurtosis" in diagnostics:
                self.assertFinite(diagnostics[f"{col}_kurtosis"], name=f"kurtosis[{col}]")
            if f"{col}_shapiro_pval" in diagnostics:
                self.assertPValue(
                    diagnostics[f"{col}_shapiro_pval"], msg=f"Shapiro p-value bounds for {col}"
                )
        hypothesis_results = statistical_hypothesis_tests(df, seed=SEEDS.BASE)
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
            num_samples=SCENARIOS.SAMPLE_SIZE_LARGE - 200,
            seed=SEEDS.HETEROSCEDASTICITY,
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=PARAMS.PNL_STD,
            pnl_duration_vol_scale=PARAMS.PNL_DUR_VOL_SCALE,
        )
        results_adj = statistical_hypothesis_tests(
            df, adjust_method="benjamini_hochberg", seed=SEEDS.REPRODUCIBILITY
        )
        self.assertGreater(len(results_adj), 0)
        for name, res in results_adj.items():
            self.assertIn("p_value", res)
            self.assertIn("p_value_adj", res)
            self.assertIn("significant_adj", res)
            p_raw = res["p_value"]
            p_adj = res["p_value_adj"]
            self.assertPValue(p_raw)
            self.assertPValue(p_adj)
            self.assertGreaterEqual(p_adj, p_raw - TOLERANCE.IDENTITY_STRICT)
            alpha = 0.05
            self.assertEqual(res["significant_adj"], bool(p_adj < alpha))
            if "effect_size_epsilon_sq" in res:
                eff = res["effect_size_epsilon_sq"]
                self.assertFinite(eff)
                self.assertGreaterEqual(eff, 0)

    def test_bootstrap_confidence_intervals_bounds_ordering(self):
        """Test bootstrap confidence intervals return ordered finite bounds."""
        test_data = self.make_stats_df(n=100, seed=SEEDS.BASE)
        results = bootstrap_confidence_intervals(test_data, ["reward", "pnl"], n_bootstrap=100)
        for metric, (mean, ci_low, ci_high) in results.items():
            self.assertFinite(mean, name=f"mean[{metric}]")
            self.assertFinite(ci_low, name=f"ci_low[{metric}]")
            self.assertFinite(ci_high, name=f"ci_high[{metric}]")
            self.assertLess(ci_low, ci_high)

    def test_stats_bootstrap_shrinkage_with_sample_size(self):
        """Bootstrap CI half-width decreases with larger sample (~1/sqrt(n) heuristic)."""

        small = self._shift_scale_df(SCENARIOS.SAMPLE_SIZE_SMALL - 20)
        large = self._shift_scale_df(SCENARIOS.SAMPLE_SIZE_LARGE)
        res_small = bootstrap_confidence_intervals(small, ["reward"], n_bootstrap=400)
        res_large = bootstrap_confidence_intervals(large, ["reward"], n_bootstrap=400)
        _, lo_s, hi_s = list(res_small.values())[0]
        _, lo_l, hi_l = list(res_large.values())[0]
        hw_small = (hi_s - lo_s) / 2.0
        hw_large = (hi_l - lo_l) / 2.0
        self.assertFinite(hw_small, name="hw_small")
        self.assertFinite(hw_large, name="hw_large")
        self.assertLess(hw_large, hw_small * 0.55)

    # Owns invariant: statistics-constant-dist-widened-ci-113a
    def test_stats_bootstrap_constant_distribution_widening(self):
        """Invariant 113 (non-strict): constant distribution CI widened with warning (positive epsilon width)."""

        df = self._const_df(80)
        with assert_diagnostic_warning(
            ["degenerate", "bootstrap", "CI"],
            warning_category=RewardDiagnosticsWarning,
            strict_mode=False,
        ):
            res = bootstrap_confidence_intervals(
                df,
                ["reward", "pnl"],
                n_bootstrap=200,
                confidence_level=0.95,
                strict_diagnostics=False,
            )
        for _metric, (mean, lo, hi) in res.items():
            self.assertLess(
                lo,
                hi,
                "Degenerate CI should be widened (lo < hi) under non-strict diagnostics",
            )
            width = hi - lo
            self.assertGreater(width, 0.0)
            self.assertLessEqual(
                width, STAT_TOL.CI_WIDTH_EPSILON, "Width should be small epsilon range"
            )
            # Mean should be centered (approx) within widened bounds
            self.assertGreaterEqual(mean, lo)
            self.assertLessEqual(mean, hi)

    # Owns invariant: statistics-constant-dist-strict-omit-113b
    def test_stats_bootstrap_constant_distribution_strict_diagnostics(self):
        """Invariant 113 (strict): constant distribution metrics are omitted (no widened CI returned)."""
        df = self._const_df(60)
        res = bootstrap_confidence_intervals(
            df, ["reward", "pnl"], n_bootstrap=150, confidence_level=0.95, strict_diagnostics=True
        )
        # Strict mode should omit constant metrics entirely
        self.assertTrue(
            all(m not in res for m in ["reward", "pnl"]),
            f"Strict diagnostics should omit constant metrics; got keys: {list(res.keys())}",
        )


if __name__ == "__main__":
    unittest.main()
