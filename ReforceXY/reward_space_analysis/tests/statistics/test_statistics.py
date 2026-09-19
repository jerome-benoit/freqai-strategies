#!/usr/bin/env python3
"""Statistical tests, distribution metrics, and bootstrap validation."""

import unittest

import numpy as np
import pandas as pd
import pytest

import reward_space_analysis
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
from ..test_base import RewardSpaceTestBase

_perform_feature_analysis = getattr(reward_space_analysis, "_perform_feature_analysis", None)

pytestmark = pytest.mark.statistics


class TestStatistics(RewardSpaceTestBase):
    """Statistical tests: metrics, diagnostics, bootstrap, correlations."""

    def test_statistics_feature_analysis_skip_partial_dependence(self):
        """Invariant 107: skip_partial_dependence=True yields empty partial_deps."""
        if _perform_feature_analysis is None:
            self.skipTest("Feature analysis helper unavailable")
        # Use existing helper to get synthetic stats df (small for speed)
        df = self.make_stats_df(n=120, seed=SEEDS.BASE, idle_pattern="mixed")
        try:
            importance_df, analysis_stats, partial_deps, _model = _perform_feature_analysis(
                df, seed=SEEDS.BASE, skip_partial_dependence=True, rf_n_jobs=1, perm_n_jobs=1
            )
        except ImportError:
            self.skipTest("scikit-learn not available; skipping feature analysis invariance test")
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
        metrics = compute_distribution_shift_metrics(
            df_const, df_const.copy(), independent_observations=True
        )
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
        metrics = compute_distribution_shift_metrics(df1, df2, independent_observations=True)
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

    def test_statistics_distribution_shift_identity_null_metrics(self):
        """Identity distributions -> near-zero shift metrics."""
        df = self._make_idle_variance_df(180)
        metrics_id = compute_distribution_shift_metrics(
            df, df.copy(), independent_observations=True
        )
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
                idle_dur = idle_data["idle_duration"].to_numpy(dtype=float)
                idle_rew = idle_data["reward_idle"].to_numpy(dtype=float)
                self.assertTrue(
                    len(idle_dur) == len(idle_rew),
                    "Idle duration and reward arrays should have same length",
                )
                self.assertTrue(
                    all(d >= 0 for d in idle_dur), "Idle durations should be non-negative"
                )
                negative_rewards = (idle_rew < 0).sum()
                total_rewards = len(idle_rew)
                negative_ratio = negative_rewards / total_rewards
                self.assertGreater(
                    negative_ratio, 0.5, "Most idle rewards should be negative (penalties)"
                )

    def test_statistics_distribution_constant_diagnostics(self):
        """Invariant 115: constant distributions keep exact mean/std and mark undefined diagnostics."""
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
        for strict in (False, True):
            with self.subTest(strict_diagnostics=strict):
                diagnostics = distribution_diagnostics(df_const, strict_diagnostics=strict)
                for col in ["reward", "pnl"]:
                    self.assertEqual(diagnostics[f"{col}_mean"], 0.0)
                    self.assertEqual(diagnostics[f"{col}_std"], 0.0)
                    self.assertIsNone(diagnostics[f"{col}_skewness"])
                    self.assertIsNone(diagnostics[f"{col}_kurtosis"])
                    self.assertTrue(diagnostics[f"{col}_constant"])
                    for suffix in ("shapiro_stat", "shapiro_pval", "anderson_stat", "qq_r_squared"):
                        self.assertNotIn(f"{col}_{suffix}", diagnostics)

    def test_stats_distribution_diagnostics(self):
        """Distribution diagnostics."""
        df = self._make_idle_variance_df(100)
        diagnostics = distribution_diagnostics(df)
        expected_prefixes = ["reward_", "pnl_"]
        for prefix in expected_prefixes:
            matching_keys = [key for key in diagnostics if key.startswith(prefix)]
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
        results = statistical_hypothesis_tests(base, independent_observations=True)
        self.assertIsInstance(results, dict)

    def test_stats_js_distance_symmetry_violin(self):
        """JS distance symmetry d(P,Q)==d(Q,P)."""
        df1 = self._shift_scale_df(300, shift=0.0)
        df2 = self._shift_scale_df(300, shift=0.3)
        metrics = compute_distribution_shift_metrics(df1, df2, independent_observations=True)
        self.assertIn("pnl_js_distance", metrics)

        metrics_swapped = compute_distribution_shift_metrics(
            df2, df1, independent_observations=True
        )
        self.assertIn("pnl_js_distance", metrics_swapped)

        self.assertAlmostEqualFloat(
            float(metrics["pnl_js_distance"]),
            float(metrics_swapped["pnl_js_distance"]),
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
        metrics = compute_distribution_shift_metrics(df1, df2, independent_observations=True)
        for k, v in metrics.items():
            if k.endswith("_kl_divergence") or k.endswith("_js_distance"):
                self.assertLess(
                    abs(v),
                    STAT_TOL.DISTRIBUTION_SHIFT,
                    f"Expected near-zero divergence after equal scaling (k={k}, v={v})",
                )

    # Non-owning smoke; ownership: robustness/test_robustness.py:43 (robustness-decomposition-integrity-101)
    @pytest.mark.smoke
    def test_stats_mean_decomposition_consistency(self):
        """Batch mean additivity."""
        df_a = self._shift_scale_df(120)
        df_b = self._shift_scale_df(180, shift=0.2)
        m_concat = float(pd.concat([df_a["pnl"], df_b["pnl"]]).mean())
        m_weighted = float(
            (df_a["pnl"].mean() * len(df_a) + df_b["pnl"].mean() * len(df_b))
            / (len(df_a) + len(df_b))
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
        tests = statistical_hypothesis_tests(df, independent_observations=True)
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
        r1 = statistical_hypothesis_tests(
            df, seed=SEEDS.REPRODUCIBILITY, independent_observations=True
        )
        r2 = statistical_hypothesis_tests(
            df, seed=SEEDS.REPRODUCIBILITY, independent_observations=True
        )
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
            df,
            metrics,
            n_bootstrap=STATISTICAL.BOOTSTRAP_DEFAULT_ITERATIONS,
            seed=SEEDS.BOOTSTRAP,
            independent_observations=True,
        )
        ci_b = bootstrap_confidence_intervals(
            df,
            metrics,
            n_bootstrap=STATISTICAL.BOOTSTRAP_DEFAULT_ITERATIONS,
            seed=SEEDS.BOOTSTRAP,
            independent_observations=True,
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
        metrics = compute_distribution_shift_metrics(df1, df2, independent_observations=True)
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
            params=self.base_params(
                max_trade_duration_candles=PARAMS.MAX_TRADE_DURATION_HETEROSCEDASTICITY
            ),
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
        # Use the action code rather than `reward_exit != 0`.
        # `reward_exit` can be zero for break-even exits, but the exit action still
        # contributes to the heteroscedasticity structure.
        exit_action_codes = (
            float(reward_space_analysis.Actions.Long_exit.value),
            float(reward_space_analysis.Actions.Short_exit.value),
        )
        exit_data = df[df["action"].isin(exit_action_codes)].copy()
        self.assertGreaterEqual(
            len(exit_data),
            SCENARIOS.SAMPLE_SIZE_TINY,
            f"Insufficient exit actions for heteroscedasticity test (n={len(exit_data)})",
        )
        exit_data["duration_bin"] = pd.cut(
            exit_data["duration_ratio"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"]
        )
        variance_by_bin = exit_data.groupby("duration_bin", observed=False)["pnl"].var().dropna()
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
        hypothesis_results = statistical_hypothesis_tests(
            df, seed=SEEDS.BASE, independent_observations=True
        )
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

    def test_bh_excludes_undefined_tests_from_finite_family(self):
        """Undefined constant-input correlation cannot contaminate valid adjusted p-values."""
        rng = np.random.default_rng(SEEDS.BASE)
        n = 120
        df = pd.DataFrame(
            {
                "reward_idle": np.full(n, -1.0),
                "idle_duration": np.ones(n),
                "position": np.repeat([0.0, 1.0], n // 2),
                "pnl": np.tile([-1.0, 1.0], n // 2),
                "reward": rng.normal(size=n),
            }
        )
        results = statistical_hypothesis_tests(
            df, independent_observations=True, adjust_method="benjamini_hochberg"
        )
        undefined = results["idle_correlation"]
        self.assertFalse(undefined["applicable"])
        self.assertIsNone(undefined["significant"])
        self.assertIsNone(undefined["significant_adj"])
        self.assertTrue(np.isnan(undefined["p_value_adj"]))
        valid = sorted(
            [results["position_reward_difference"], results["pnl_sign_reward_difference"]],
            key=lambda result: result["p_value"],
        )
        self.assertAlmostEqual(
            valid[0]["p_value_adj"], min(2 * valid[0]["p_value"], valid[1]["p_value"], 1.0)
        )
        self.assertAlmostEqual(valid[1]["p_value_adj"], valid[1]["p_value"])
        for result in valid:
            self.assertTrue(result["applicable"])
            self.assertTrue(np.isfinite(result["p_value_adj"]))
            self.assertEqual(result["significant_adj"], result["p_value_adj"] < 0.05)

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
            df,
            adjust_method="benjamini_hochberg",
            seed=SEEDS.REPRODUCIBILITY,
            independent_observations=True,
        )
        self.assertGreater(len(results_adj), 0)
        for _name, res in results_adj.items():
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
        test_data = self.make_stats_df(n=SCENARIOS.SAMPLE_SIZE_SMALL, seed=SEEDS.BASE)
        results = bootstrap_confidence_intervals(
            test_data,
            ["reward", "pnl"],
            n_bootstrap=STATISTICAL.BOOTSTRAP_DEFAULT_ITERATIONS,
            independent_observations=True,
        )
        for metric, (mean, ci_low, ci_high) in results.items():
            self.assertFinite(mean, name=f"mean[{metric}]")
            self.assertFinite(ci_low, name=f"ci_low[{metric}]")
            self.assertFinite(ci_high, name=f"ci_high[{metric}]")
            self.assertLess(ci_low, ci_high)

    def test_stats_bootstrap_shrinkage_with_sample_size(self):
        """Bootstrap CI half-width decreases with larger sample (~1/sqrt(n) heuristic)."""

        small = self._shift_scale_df(SCENARIOS.SAMPLE_SIZE_SMALL - 20)
        large = self._shift_scale_df(SCENARIOS.SAMPLE_SIZE_LARGE)
        res_small = bootstrap_confidence_intervals(
            small, ["reward"], n_bootstrap=400, independent_observations=True
        )
        res_large = bootstrap_confidence_intervals(
            large, ["reward"], n_bootstrap=400, independent_observations=True
        )
        _, lo_s, hi_s = next(iter(res_small.values()))
        _, lo_l, hi_l = next(iter(res_large.values()))
        hw_small = (hi_s - lo_s) / 2.0
        hw_large = (hi_l - lo_l) / 2.0
        self.assertFinite(hw_small, name="hw_small")
        self.assertFinite(hw_large, name="hw_large")
        self.assertLess(hw_large, hw_small * 0.55)

    def test_stats_bootstrap_constant_distribution_exact_bounds(self):
        """Constants retain their exact degenerate interval in both diagnostic modes."""
        df = pd.DataFrame({"reward": np.full(40, 2.5)})
        for strict in (False, True):
            with self.subTest(strict_diagnostics=strict):
                res = bootstrap_confidence_intervals(
                    df,
                    ["reward"],
                    n_bootstrap=SCENARIOS.BOOTSTRAP_EXTENDED_ITERATIONS,
                    strict_diagnostics=strict,
                    independent_observations=True,
                )
                self.assertEqual(res, {"reward": (2.5, 2.5, 2.5)})

    def test_stats_bootstrap_percentiles_need_not_contain_mean(self):
        """A single resample yields its own mean, not a widened interval around the estimate."""
        df = pd.DataFrame({"reward": np.arange(10, dtype=float)})
        for strict in (False, True):
            with self.subTest(strict_diagnostics=strict):
                with self.assertWarns(RewardDiagnosticsWarning):
                    res = bootstrap_confidence_intervals(
                        df,
                        ["reward"],
                        n_bootstrap=1,
                        seed=SEEDS.BASE,
                        strict_diagnostics=strict,
                        independent_observations=True,
                    )
                self.assertEqual(res["reward"], (4.5, 3.7, 3.7))

    def test_inference_helpers_require_independent_observations(self):
        """Inferential helpers reject dependent trajectory observations."""
        df = pd.DataFrame({"reward": np.arange(10, dtype=float)})
        with self.assertRaisesRegex(ValueError, "independent_observations=True"):
            statistical_hypothesis_tests(df, independent_observations=False)
        with self.assertRaisesRegex(ValueError, "independent_observations=True"):
            bootstrap_confidence_intervals(
                df,
                ["reward"],
                n_bootstrap=SCENARIOS.BOOTSTRAP_EXTENDED_ITERATIONS,
                independent_observations=False,
            )

    def test_stats_bootstrap_rejects_invalid_bounds(self):
        """Non-finite or reversed bounds are errors, never repaired by the validator."""
        for bounds in (
            (0.0, 1.0, -1.0),
            (np.nan, 0.0, 1.0),
            (0.0, -np.inf, 1.0),
            (0.0, 0.0, np.inf),
        ):
            for strict in (False, True):
                with (
                    self.subTest(bounds=bounds, strict_diagnostics=strict),
                    self.assertRaises(AssertionError),
                ):
                    reward_space_analysis._validate_bootstrap_results(
                        {"reward": bounds}, strict_diagnostics=strict
                    )

    def test_stats_diagnostics_rejects_fabricated_constant_fallbacks(self):
        """A constant marker cannot turn invalid statistics into synthetic moments or R²."""
        for key in ("reward_skewness", "reward_anderson_stat", "reward_qq_r_squared"):
            for strict in (False, True):
                with (
                    self.subTest(key=key, strict_diagnostics=strict),
                    self.assertRaises(AssertionError),
                ):
                    reward_space_analysis._validate_distribution_diagnostics(
                        {"reward_constant": True, "reward_std": 0.0, key: np.nan},
                        strict_diagnostics=strict,
                    )


if __name__ == "__main__":
    unittest.main()
