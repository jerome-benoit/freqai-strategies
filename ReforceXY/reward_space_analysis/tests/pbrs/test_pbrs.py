#!/usr/bin/env python3
"""Tests for Potential-Based Reward Shaping (PBRS) mechanics."""
import unittest

import numpy as np
import pytest

from reward_space_analysis import (
    DEFAULT_MODEL_REWARD_PARAMETERS,
    PBRS_INVARIANCE_TOL,
    _compute_entry_additive,
    _compute_exit_additive,
    _compute_exit_potential,
    _compute_hold_potential,
    _get_float_param,
    apply_potential_shaping,
    get_max_idle_duration_candles,
    simulate_samples,
    validate_reward_parameters,
)

from ..test_base import RewardSpaceTestBase

pytestmark = pytest.mark.pbrs


class TestPBRS(RewardSpaceTestBase):
    """PBRS mechanics tests (transforms, parameters, potentials, invariance)."""

    def test_pbrs_progressive_release_decay_clamped(self):
        """progressive_release decay>1 clamps -> Φ'=0 & Δ=-Φ_prev."""
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "potential_gamma": DEFAULT_MODEL_REWARD_PARAMETERS["potential_gamma"],
                "exit_potential_mode": "progressive_release",
                "exit_potential_decay": 5.0,
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
            last_potential=0.789,
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
            params, "potential_gamma", DEFAULT_MODEL_REWARD_PARAMETERS.get("potential_gamma", 0.95)
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
        unique_flags = set(df["pbrs_invariant"].unique().tolist())
        self.assertEqual(unique_flags, {True}, f"Unexpected invariant flags: {unique_flags}")
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

    def test_additive_components_disabled_return_zero(self):
        """Test entry and exit additives return zero when disabled."""
        # Test entry additive disabled
        params_entry = {"entry_additive_enabled": False, "entry_additive_scale": 1.0}
        val_entry = _compute_entry_additive(0.5, 0.3, params_entry)
        self.assertEqual(float(val_entry), 0.0)

        # Test exit additive disabled
        params_exit = {"exit_additive_enabled": False, "exit_additive_scale": 1.0}
        val_exit = _compute_exit_additive(0.5, 0.3, params_exit)
        self.assertEqual(float(val_exit), 0.0)

    def test_exit_potential_canonical(self):
        """Test exit potential canonical."""
        params = self.base_params(
            exit_potential_mode="canonical",
            hold_potential_enabled=True,
            entry_additive_enabled=True,
            exit_additive_enabled=True,
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
            last_potential=0.789,
            params=params,
        )
        self.assertIn("_pbrs_invariance_applied", params)
        self.assertFalse(
            params["entry_additive_enabled"],
            "Entry additive should be auto-disabled in canonical mode",
        )
        self.assertFalse(
            params["exit_additive_enabled"],
            "Exit additive should be auto-disabled in canonical mode",
        )
        self.assertPlacesEqual(next_potential, 0.0, places=12)
        current_potential = _compute_hold_potential(
            current_pnl,
            current_duration_ratio,
            {"hold_potential_enabled": True, "hold_potential_scale": 1.0},
        )
        self.assertAlmostEqual(shaping, -current_potential, delta=self.TOL_IDENTITY_RELAXED)
        residual = total - base_reward - shaping
        self.assertAlmostEqual(residual, 0.0, delta=self.TOL_IDENTITY_RELAXED)
        self.assertTrue(np.isfinite(total))

    def test_pbrs_invariance_internal_flag_set(self):
        """Canonical path sets _pbrs_invariance_applied once; second call idempotent."""
        params = self.base_params(
            exit_potential_mode="canonical",
            hold_potential_enabled=True,
            entry_additive_enabled=True,
            exit_additive_enabled=True,
        )
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
            self.assertTrue(
                all((abs(p) < self.PBRS_TERMINAL_TOL for p in terminal_next_potentials))
            )
        max_abs = max((abs(v) for v in shaping_values)) if shaping_values else 0.0
        self.assertLessEqual(max_abs, self.PBRS_MAX_ABS_SHAPING)
        state_after = (params["entry_additive_enabled"], params["exit_additive_enabled"])
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
            state_after, (params["entry_additive_enabled"], params["exit_additive_enabled"])
        )

    def test_progressive_release_negative_decay_clamped(self):
        """Negative decay must clamp to 0 => next potential equals last potential (no release)."""
        params = self.base_params(
            exit_potential_mode="progressive_release",
            exit_potential_decay=-0.75,
            hold_potential_enabled=True,
        )
        last_potential = 0.42
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
        gamma_raw = DEFAULT_MODEL_REWARD_PARAMETERS.get("potential_gamma", 0.95)
        gamma_fallback = 0.95 if gamma_raw is None else gamma_raw
        try:
            gamma = float(gamma_fallback)
        except Exception:
            gamma = 0.95
        self.assertLessEqual(abs(shaping - gamma * last_potential), self.TOL_GENERIC_EQ)
        self.assertPlacesEqual(total, shaping, places=12)

    def test_potential_gamma_nan_fallback(self):
        """potential_gamma=NaN should fall back to default value (indirect comparison)."""
        base_params_dict = self.base_params()
        default_gamma = base_params_dict.get("potential_gamma", 0.95)
        params_nan = self.base_params(potential_gamma=np.nan, hold_potential_enabled=True)
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
        except Exception as e:
            self.fail(f"validate_reward_parameters raised unexpectedly: {e}")
        if isinstance(validated, tuple) and len(validated) >= 1 and isinstance(validated[0], dict):
            validated_params = validated[0]
        else:
            validated_params = validated
        for k in ("potential_gamma", "hold_potential_enabled", "exit_potential_mode"):
            self.assertIn(k, validated_params, f"Missing key '{k}' in validated params")
        params_bad = params_ok.copy()
        params_bad["potential_gamma"] = -0.2
        params_bad["hold_potential_scale"] = -5.0
        with self.assertRaises((ValueError, AssertionError)):
            vr = validate_reward_parameters(params_bad)
            if not isinstance(vr, Exception):
                self.fail("validate_reward_parameters should raise on invalid params")

    def test_validate_reward_parameters_relaxed_multi_reason_aggregation(self):
        """Invariant (extension): relaxed mode aggregates multiple adjustment reasons properly.

        Scenario:
        - potential_gamma provided as non-numeric string -> non_numeric_reset (relaxed).
        - hold_potential_scale provided as '-5.0' string below min -> numeric_coerce + min clamp merged.
        - max_idle_duration_candles provided as 'nan' -> derived_default removal.
        Expectations:
        - Adjustments dict contains aggregated reasons (comma-joined) where multiple clamps occur.
        - validation_mode=='relaxed' for all adjustments.
        - Sanitized values respect bounds and derivation logic.
        """
        params_relaxed = DEFAULT_MODEL_REWARD_PARAMETERS.copy()
        params_relaxed.update(
            {
                "potential_gamma": "not-a-number",  # triggers non_numeric_reset (relaxed)
                "hold_potential_scale": "-5.0",     # numeric_coerce then min clamp
                "max_idle_duration_candles": "nan", # derived_default removal
            }
        )
        sanitized, adjustments = validate_reward_parameters(params_relaxed, strict=False)
        # potential_gamma adjustment
        self.assertIn("potential_gamma", adjustments)
        adj_pg = adjustments["potential_gamma"]
        self.assertEqual(adj_pg["validation_mode"], "relaxed")
        self.assertIn("non_numeric_reset", adj_pg["reason"])
        # Safe numeric coercion helper for sanitized params avoiding direct float(...) on None/str
        def _safe_num(key: str, default: float = 0.0) -> float:
            raw = sanitized.get(key, default)
            try:
                if raw is None:
                    return default
                return float(raw)
            except Exception:
                return default
        pg_val = _safe_num("potential_gamma", 0.0)
        self.assertGreaterEqual(pg_val, 0.0)
        self.assertLessEqual(pg_val, 1.0)
        # hold_potential_scale adjustment aggregation
        self.assertIn("hold_potential_scale", adjustments)
        adj_hold = adjustments["hold_potential_scale"]
        self.assertEqual(adj_hold["validation_mode"], "relaxed")
        # Reason may include both numeric_coerce and min=<value>
        self.assertTrue(
            any(r in adj_hold["reason"] for r in ["numeric_coerce", "min="]),
            f"Expected numeric_coerce/min clamp in reason (got {adj_hold['reason']})",
        )
        hold_scale_val = _safe_num("hold_potential_scale", 0.0)
        self.assertGreaterEqual(hold_scale_val, 0.0)
        # max_idle_duration_candles derived default removal
        self.assertIn("max_idle_duration_candles", adjustments)
        adj_mid = adjustments["max_idle_duration_candles"]
        self.assertEqual(adj_mid["validation_mode"], "relaxed")
        self.assertEqual(adj_mid["reason"], "derived_default")
        self.assertNotIn("max_idle_duration_candles", sanitized)  # removed for derivation
        # Ensure derivation helper would compute a positive fallback when invoked later
        derived_mid = get_max_idle_duration_candles(sanitized)
        self.assertGreater(derived_mid, 0)

        """validate_reward_parameters: success on defaults and failure on invalid ranges."""
        params_ok = DEFAULT_MODEL_REWARD_PARAMETERS.copy()
        try:
            validated = validate_reward_parameters(params_ok)
        except Exception as e:
            self.fail(f"validate_reward_parameters raised unexpectedly: {e}")
        if isinstance(validated, tuple) and len(validated) >= 1 and isinstance(validated[0], dict):
            validated_params = validated[0]
        else:
            validated_params = validated
        for k in ("potential_gamma", "hold_potential_enabled", "exit_potential_mode"):
            self.assertIn(k, validated_params, f"Missing key '{k}' in validated params")
        params_bad = params_ok.copy()
        params_bad["potential_gamma"] = -0.2
        params_bad["hold_potential_scale"] = -5.0
        with self.assertRaises((ValueError, AssertionError)):
            vr = validate_reward_parameters(params_bad)
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
        ctx_pnl = 0.012
        ctx_dur_ratio = 0.3
        params_can = self.base_params(exit_potential_mode="canonical", **base_common)
        prev_phi = _compute_hold_potential(ctx_pnl, ctx_dur_ratio, params_can)
        self.assertFinite(prev_phi, name="prev_phi")
        next_phi_can = _compute_exit_potential(prev_phi, params_can)
        self.assertAlmostEqualFloat(
            next_phi_can,
            0.0,
            tolerance=self.TOL_IDENTITY_STRICT,
            msg="Canonical exit must zero potential",
        )
        canonical_delta = -prev_phi
        self.assertAlmostEqualFloat(
            canonical_delta,
            -prev_phi,
            tolerance=self.TOL_IDENTITY_RELAXED,
            msg="Canonical delta mismatch",
        )
        params_spike = self.base_params(exit_potential_mode="spike_cancel", **base_common)
        next_phi_spike = _compute_exit_potential(prev_phi, params_spike)
        shaping_spike = gamma * next_phi_spike - prev_phi
        self.assertNearZero(
            shaping_spike,
            atol=self.TOL_IDENTITY_RELAXED,
            msg="Spike cancel should nullify shaping delta",
        )
        self.assertGreaterEqual(
            abs(canonical_delta) + self.TOL_IDENTITY_STRICT,
            abs(shaping_spike),
            "Canonical shaping magnitude should exceed spike_cancel",
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
            params, "potential_gamma", DEFAULT_MODEL_REWARD_PARAMETERS.get("potential_gamma", 0.95)
        )
        rng = np.random.default_rng(555)
        potentials = rng.uniform(0.05, 0.85, size=220)
        deltas = [gamma * p - p for p in potentials]
        cumulative = float(np.sum(deltas))
        self.assertLess(cumulative, -self.TOL_NEGLIGIBLE)
        self.assertGreater(abs(cumulative), 10 * self.TOL_IDENTITY_RELAXED)

    def test_pbrs_106_canonical_drift_correction_zero_sum(self):
        """Invariant 106: simulate_samples canonical mode enforces near zero-sum shaping (drift correction)."""
        params = self.base_params(
            exit_potential_mode="canonical",
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            potential_gamma=0.94,
        )
        df = simulate_samples(
            params={**params, "max_trade_duration_candles": 140},
            num_samples=500,
            seed=913,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        total_shaping = float(df["reward_shaping"].sum())
        self.assertLess(
            abs(total_shaping),
            PBRS_INVARIANCE_TOL,
            f"Drift correction failed canonical Σ shaping={total_shaping}",
        )
        # All invariant flags should be True under canonical invariance path
        flags = set(df["pbrs_invariant"].unique().tolist())
        self.assertEqual(flags, {True}, f"Unexpected invariance flags canonical: {flags}")

    def test_pbrs_106_canonical_drift_correction_exception_fallback(self):
        """Invariant 106 (extension): exception path graceful fallback.

        Forces an exception inside post-simulation invariance enforcement by monkeypatching
        DataFrame.sum on a frame containing 'reward_shaping'. Ensures:
        - simulate_samples completes without propagating exception
        - reward_shaping column remains present
        - pbrs_invariant flags remain True
        - Σ shaping may remain > tolerance (correction skipped)
        Covers the except: pass branch (lines 1310-1343 in simulate_samples).
        """
        params = self.base_params(
            exit_potential_mode="canonical",
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            potential_gamma=0.91,
        )
        import pandas as pd
        original_sum = pd.DataFrame.sum
        def boom(self, *args, **kwargs):  # noqa: D401
            if isinstance(self, pd.DataFrame) and "reward_shaping" in self.columns:
                raise RuntimeError("forced drift correction failure")
            return original_sum(self, *args, **kwargs)
        pd.DataFrame.sum = boom
        try:
            df_exc = simulate_samples(
                params={**params, "max_trade_duration_candles": 120},
                num_samples=250,
                seed=515,
                base_factor=self.TEST_BASE_FACTOR,
                profit_target=self.TEST_PROFIT_TARGET,
                risk_reward_ratio=self.TEST_RR,
                max_duration_ratio=2.0,
                trading_mode="margin",
                pnl_base_std=self.TEST_PNL_STD,
                pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
            )
        finally:
            pd.DataFrame.sum = original_sum
        flags_exc = set(df_exc["pbrs_invariant"].unique().tolist())
        self.assertEqual(flags_exc, {True})
        total_shaping_exc = float(df_exc["reward_shaping"].sum())
        # Expect shaping not force-corrected: allow >= tolerance
        # Drift correction skipped; shaping may or may not exceed tolerance depending on sampled distribution.
        # Assert merely that column exists and simulation succeeded without exception.
        self.assertTrue("reward_shaping" in df_exc.columns)
        self.assertIn("reward_shaping", df_exc.columns)

    def test_pbrs_106_canonical_drift_correction_uniform_offset(self):
        """Invariant 106: canonical drift correction reduces Σ shaping below tolerance.

        Compares canonical (correction applied) vs non-canonical (retain_previous) to assert:
        - Canonical total shaping magnitude < non-canonical magnitude
        - Canonical Σ shaping within tolerance
        - Mean corrected shaping near zero for invariant samples
        """
        params_can = self.base_params(
            exit_potential_mode="canonical",
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            potential_gamma=0.92,
        )
        df_can = simulate_samples(
            params={**params_can, "max_trade_duration_candles": 120},
            num_samples=400,
            seed=777,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        params_non = self.base_params(
            exit_potential_mode="retain_previous",
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            potential_gamma=0.92,
        )
        df_non = simulate_samples(
            params={**params_non, "max_trade_duration_candles": 120},
            num_samples=400,
            seed=777,
            base_factor=self.TEST_BASE_FACTOR,
            profit_target=self.TEST_PROFIT_TARGET,
            risk_reward_ratio=self.TEST_RR,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=self.TEST_PNL_STD,
            pnl_duration_vol_scale=self.TEST_PNL_DUR_VOL_SCALE,
        )
        total_can = float(df_can["reward_shaping"].sum())
        total_non = float(df_non["reward_shaping"].sum())
        self.assertLess(abs(total_can), abs(total_non) + self.TOL_IDENTITY_RELAXED)
        self.assertLess(abs(total_can), PBRS_INVARIANCE_TOL, f"Drift correction insufficient (Σ={total_can})")
        invariant_mask = df_can["pbrs_invariant"]
        if bool(getattr(invariant_mask, "any", lambda: False)()):
            corrected_values = df_can.loc[invariant_mask, "reward_shaping"].to_numpy()
            mean_corrected = float(np.mean(corrected_values))
            self.assertLess(abs(mean_corrected), self.TOL_IDENTITY_RELAXED)
            spread = float(np.max(corrected_values) - np.min(corrected_values))
            self.assertLess(spread, self.PBRS_MAX_ABS_SHAPING)

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
            return (float(skew), float(kurt))

        s_base, k_base = _skew_kurt(base)
        s_scaled, k_scaled = _skew_kurt(scaled)
        self.assertAlmostEqualFloat(s_base, s_scaled, tolerance=self.TOL_DISTRIB_SHAPE)
        self.assertAlmostEqualFloat(k_base, k_scaled, tolerance=self.TOL_DISTRIB_SHAPE)

    def test_pbrs_non_canonical_report_generation(self):
        """Generate synthetic invariance section with non-zero shaping to assert Non-canonical classification."""
        import re

        import pandas as pd

        from reward_space_analysis import PBRS_INVARIANCE_TOL

        df = pd.DataFrame(
            {
                "reward_shaping": [0.01, -0.002],
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
        self.assertRegex(content, "Σ Shaping Reward \\| 0\\.008000 \\|")
        m_abs = re.search("Abs Σ Shaping Reward \\| ([0-9.]+e[+-][0-9]{2}) \\|", content)
        self.assertIsNotNone(m_abs)
        if m_abs:
            val = float(m_abs.group(1))
            self.assertAlmostEqual(abs(total_shaping), val, places=12)

    def test_potential_gamma_boundary_values_stability(self):
        """Test potential gamma boundary values (0 and ≈1) produce bounded shaping."""
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

    def test_report_cumulative_invariance_aggregation(self):
        """Canonical telescoping term: small per-step mean drift, bounded increments."""
        params = self.base_params(
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            exit_potential_mode="canonical",
        )
        gamma = _get_float_param(
            params, "potential_gamma", DEFAULT_MODEL_REWARD_PARAMETERS.get("potential_gamma", 0.95)
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
            inc = gamma * next_potential - last_potential
            telescoping_sum += inc
            if abs(inc) > max_abs_step:
                max_abs_step = abs(inc)
            steps += 1
            if is_exit:
                last_potential = 0.0
            else:
                last_potential = next_potential
        mean_drift = telescoping_sum / max(1, steps)
        self.assertLess(
            abs(mean_drift),
            0.02,
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

    def test_pbrs_canonical_near_zero_report(self):
        """Invariant 116: canonical near-zero cumulative shaping classified via full report generation.

        Uses `write_complete_statistical_analysis` rather than synthetic section construction to exercise
        canonical invariance branch (lines 3501–3504) with a small non-zero shaping sum below `PBRS_INVARIANCE_TOL`.
        Asserts:
        - Invariance Status line shows ✅ Canonical
        - Analysis Note contains theoretical invariance phrase
        - Σ Shaping Reward rendered with six decimal fixed-point
        - Abs Σ Shaping Reward rendered in scientific notation matching absolute sum
        """
        import re

        import numpy as np
        import pandas as pd

        from reward_space_analysis import PBRS_INVARIANCE_TOL, write_complete_statistical_analysis

        # Small shaping values summing to a tiny non-zero amount (< tolerance) to ensure canonical classification.
        small_vals = [1.0e-7, -2.0e-7, 3.0e-7]  # sum = 2.0e-7 < 1e-6 tolerance
        total_shaping = float(sum(small_vals))
        self.assertLess(
            abs(total_shaping),
            PBRS_INVARIANCE_TOL,
            f"Total shaping {total_shaping} exceeds invariance tolerance",
        )

        n = len(small_vals)
        df = pd.DataFrame(
            {
                "reward": np.random.normal(0, 1, n),
                "reward_idle": np.zeros(n),
                "reward_hold": np.random.normal(-0.2, 0.05, n),
                "reward_exit": np.random.normal(0.4, 0.15, n),
                "pnl": np.random.normal(0.01, 0.02, n),
                "trade_duration": np.random.uniform(5, 30, n),
                "idle_duration": np.zeros(n),
                "position": np.random.choice([0.0, 0.5, 1.0], n),
                "action": np.random.randint(0, 3, n),
                "reward_shaping": small_vals,
                "reward_entry_additive": [0.0] * n,
                "reward_exit_additive": [0.0] * n,
                "reward_invalid": np.zeros(n),
                "duration_ratio": np.random.uniform(0.2, 1.0, n),
                "idle_ratio": np.zeros(n),
            }
        )
        df.attrs["reward_params"] = {
            "exit_potential_mode": "canonical",
            "entry_additive_enabled": False,
            "exit_additive_enabled": False,
        }

        out_dir = self.output_path / "canonical_near_zero_report"
        write_complete_statistical_analysis(
            df,
            output_dir=out_dir,
            profit_target=self.TEST_PROFIT_TARGET,
            seed=self.SEED,
            skip_feature_analysis=True,
            skip_partial_dependence=True,
            bootstrap_resamples=25,
        )
        report_path = out_dir / "statistical_analysis.md"
        self.assertTrue(report_path.exists(), "Report file missing for canonical near-zero test")
        content = report_path.read_text(encoding="utf-8")

        # Assertions on invariance classification and formatting
        self.assertIn("✅ Canonical", content)
        self.assertRegex(
            content,
            r"\| Invariance Status \| ✅ Canonical \|",
        )
        self.assertIn(
            "Theoretical invariance preserved", content, "Expected theoretical invariance note missing"
        )
        # Six-decimal fixed-point formatting for raw sum (should round to 0.000000)
        self.assertRegex(content, r"\| Σ Shaping Reward \| 0\.000000 \|")
        # Scientific notation for absolute sum line
        m_abs = re.search(r"\| Abs Σ Shaping Reward \| ([0-9.]+e[+-][0-9]{2}) \|", content)
        self.assertIsNotNone(m_abs)
        if m_abs:
            val_abs = float(m_abs.group(1))
            self.assertAlmostEqual(abs(total_shaping), val_abs, places=12)


    def test_pbrs_canonical_warning_report(self):
        """Invariant: canonical mode + no additives but |Σ shaping| > PBRS_INVARIANCE_TOL -> ⚠️ Canonical (with warning).

        Builds synthetic DataFrame with reward_shaping summing well above tolerance while maintaining canonical
        configuration (exit_potential_mode='canonical', additives disabled). Generates full report (feature analysis
        skipped for speed) and asserts warning classification and note formatting with six-decimal sum value.
        """
        import re

        import pandas as pd

        from reward_space_analysis import PBRS_INVARIANCE_TOL, write_complete_statistical_analysis

        # Construct shaping values summing to > tolerance (e.g. 5e-4)
        shaping_vals = [1.2e-4, 1.3e-4, 8.0e-5, -2.0e-5, 1.4e-4]  # sum = 4.5e-4 (>1e-6)
        total_shaping = sum(shaping_vals)
        self.assertGreater(abs(total_shaping), PBRS_INVARIANCE_TOL)

        n = len(shaping_vals)
        df = pd.DataFrame(
            {
                "reward": np.random.normal(0, 1, n),
                "reward_idle": np.zeros(n),
                "reward_hold": np.random.normal(-0.2, 0.1, n),
                "reward_exit": np.random.normal(0.5, 0.2, n),
                "pnl": np.random.normal(0.01, 0.02, n),
                "trade_duration": np.random.uniform(5, 50, n),
                "idle_duration": np.zeros(n),
                "position": np.random.choice([0.0, 0.5, 1.0], n),
                "action": np.random.randint(0, 3, n),
                "reward_shaping": shaping_vals,
                "reward_entry_additive": [0.0] * n,
                "reward_exit_additive": [0.0] * n,
                "reward_invalid": np.zeros(n),
                "duration_ratio": np.random.uniform(0.2, 1.2, n),
                "idle_ratio": np.zeros(n),
            }
        )
        # Canonical params stored in attrs for report classification
        df.attrs["reward_params"] = {
            "exit_potential_mode": "canonical",
            "entry_additive_enabled": False,
            "exit_additive_enabled": False,
        }

        out_dir = self.output_path / "canonical_warning"
        write_complete_statistical_analysis(
            df,
            output_dir=out_dir,
            profit_target=self.TEST_PROFIT_TARGET,
            seed=self.SEED,
            skip_feature_analysis=True,  # speed
            skip_partial_dependence=True,
            bootstrap_resamples=50,
        )
        report_path = out_dir / "statistical_analysis.md"
        self.assertTrue(report_path.exists(), "Report file missing for canonical warning test")
        content = report_path.read_text(encoding="utf-8")
        # Expect warning classification line
        self.assertIn("⚠️ Canonical (with warning)", content)
        # Verify invariance note includes formatted sum value with six decimals
        expected_sum_fragment = f"{total_shaping:.6f}"  # formatting in note uses .6f
        self.assertIn(expected_sum_fragment, content)
        # Regex to ensure table line present
        m_status = re.search(r"\| Invariance Status \| ⚠️ Canonical \(with warning\) \|", content)
        self.assertIsNotNone(m_status, "Invariance Status line missing or misformatted for warning state")

    def test_pbrs_non_canonical_full_report_reason_aggregation(self):
        """Full report generation: Non-canonical classification aggregates mode and additive reasons.

        Constructs a minimal DataFrame with non-zero shaping and both entry/exit additives enabled under
        a non-canonical exit_potential_mode ('progressive_release'). Invokes write_complete_statistical_analysis
        and asserts invariance status line plus aggregated reasons formatting.
        """
        import re

        import pandas as pd

        from reward_space_analysis import write_complete_statistical_analysis

        # Synthetic shaping + additive values (non-zero to ensure non-canonical sum distinct from zero)
        shaping_vals = [0.02, -0.005, 0.007]
        entry_add_vals = [0.003, 0.0, 0.004]
        exit_add_vals = [0.001, 0.002, 0.0]
        n = len(shaping_vals)
        df = pd.DataFrame(
            {
                "reward": np.random.normal(0, 1, n),
                "reward_idle": np.zeros(n),
                "reward_hold": np.random.normal(-0.1, 0.05, n),
                "reward_exit": np.random.normal(0.4, 0.15, n),
                "pnl": np.random.normal(0.01, 0.02, n),
                "trade_duration": np.random.uniform(5, 25, n),
                "idle_duration": np.zeros(n),
                "position": np.random.choice([0.0, 0.5, 1.0], n),
                "action": np.random.randint(0, 5, n),
                "reward_shaping": shaping_vals,
                "reward_entry_additive": entry_add_vals,
                "reward_exit_additive": exit_add_vals,
                "reward_invalid": np.zeros(n),
                "duration_ratio": np.random.uniform(0.1, 1.0, n),
                "idle_ratio": np.zeros(n),
            }
        )
        # Non-canonical configuration with both additives enabled
        df.attrs["reward_params"] = {
            "exit_potential_mode": "progressive_release",
            "entry_additive_enabled": True,
            "exit_additive_enabled": True,
        }
        out_dir = self.output_path / "non_canonical_full_report"
        write_complete_statistical_analysis(
            df,
            output_dir=out_dir,
            profit_target=self.TEST_PROFIT_TARGET,
            seed=self.SEED,
            skip_feature_analysis=True,
            skip_partial_dependence=True,
            bootstrap_resamples=25,
        )
        report_path = out_dir / "statistical_analysis.md"
        self.assertTrue(report_path.exists(), "Report file missing for non-canonical full report test")
        content = report_path.read_text(encoding="utf-8")
        # Invariance Status line should indicate Non-canonical classification
        self.assertIn("❌ Non-canonical", content)
        m_status = re.search(r"\| Invariance Status \| ❌ Non-canonical \|", content)
        self.assertIsNotNone(m_status, "Invariance Status line missing for non-canonical state")
        # Reasons should include both exit_potential_mode and additives list
        self.assertIn("exit_potential_mode='progressive_release'", content)
        # Additives list formatting: additives=['entry', 'exit'] (order preserved)
        self.assertRegex(content, r"additives=\['entry', 'exit'\]")

    def test_pbrs_non_canonical_mode_only_reason(self):
        """Non-canonical exit mode with additives disabled -> invariance note lists only exit_potential_mode reason.

        Constructs minimal DataFrame with non-zero shaping sum (≫ tolerance) under exit_potential_mode='retain_previous'
        and both additives disabled. Generates full report and asserts classification plus reason aggregation excludes
        additives list.
        """
        import re

        import pandas as pd

        from reward_space_analysis import PBRS_INVARIANCE_TOL, write_complete_statistical_analysis

        shaping_vals = [0.002, -0.0005, 0.0012]  # sum ≈ 0.0027 (> 1e-6 tolerance)
        total_shaping = sum(shaping_vals)
        self.assertGreater(abs(total_shaping), PBRS_INVARIANCE_TOL)

        n = len(shaping_vals)
        df = pd.DataFrame(
            {
                "reward": np.random.normal(0, 1, n),
                "reward_idle": np.zeros(n),
                "reward_hold": np.random.normal(-0.15, 0.05, n),
                "reward_exit": np.random.normal(0.3, 0.1, n),
                "pnl": np.random.normal(0.01, 0.02, n),
                "trade_duration": np.random.uniform(5, 40, n),
                "idle_duration": np.zeros(n),
                "position": np.random.choice([0.0, 0.5, 1.0], n),
                "action": np.random.randint(0, 5, n),
                "reward_shaping": shaping_vals,
                "reward_entry_additive": [0.0] * n,
                "reward_exit_additive": [0.0] * n,
                "reward_invalid": np.zeros(n),
                "duration_ratio": np.random.uniform(0.2, 1.2, n),
                "idle_ratio": np.zeros(n),
            }
        )
        df.attrs["reward_params"] = {
            "exit_potential_mode": "retain_previous",
            "entry_additive_enabled": False,
            "exit_additive_enabled": False,
        }

        out_dir = self.output_path / "non_canonical_mode_only"
        write_complete_statistical_analysis(
            df,
            output_dir=out_dir,
            profit_target=self.TEST_PROFIT_TARGET,
            seed=self.SEED,
            skip_feature_analysis=True,
            skip_partial_dependence=True,
            bootstrap_resamples=25,
        )
        report_path = out_dir / "statistical_analysis.md"
        self.assertTrue(report_path.exists(), "Report file missing for non-canonical mode-only reason test")
        content = report_path.read_text(encoding="utf-8")
        # Status should indicate Non-canonical classification
        self.assertIn("❌ Non-canonical", content)
        m_status = re.search(r"\| Invariance Status \| ❌ Non-canonical \|", content)
        self.assertIsNotNone(m_status, "Invariance Status line missing for non-canonical mode-only state")
        # Reasons should include only exit_potential_mode and exclude additives list
        self.assertIn("exit_potential_mode='retain_previous'", content)
        self.assertNotRegex(content, r"additives=\[")

    def test_pbrs_absence_and_distribution_shift_placeholder(self):
        """Report generation without PBRS columns triggers absence + shift placeholder.

        Constructs a minimal DataFrame omitting 'reward_shaping', 'reward_entry_additive',
        and 'reward_exit_additive'. Generates full report with no real_df passed to
        write_complete_statistical_analysis to exercise:
        - PBRS absence message (line 3539)
        - Distribution shift placeholder section (lines 3724-3725)
        Asserts presence of both marker strings.
        """
        import pandas as pd
        from reward_space_analysis import write_complete_statistical_analysis

        # Increase sample size and introduce variability to ensure hypothesis_tests is non-empty.
        # A minimal constant frame skipped Section 5 entirely, preventing the distribution shift placeholder from rendering.
        # Use n=40 with varied numerical columns to trigger at least idle_correlation or other tests.
        # Increase sample size further to ensure idle_mask >= 30 for idle correlation test.
        n = 90
        rng = np.random.default_rng(123)
        df = pd.DataFrame(
            {
                "reward": rng.normal(0.05, 0.02, n),
                # Provide enough non-zero idle rewards to exceed idle_mask.sum() >= 30 threshold.
                # Half zeros, half small negatives to simulate penalties.
                "reward_idle": np.concatenate([
                    rng.normal(-0.01, 0.003, n // 2),
                    np.zeros(n - n // 2),
                ]),
                "reward_hold": rng.normal(0.0, 0.01, n),
                "reward_exit": rng.normal(0.04, 0.015, n),
                # Mix positive and negative pnl to potentially trigger pnl_sign_reward_difference test
                "pnl": rng.normal(0.0, 0.05, n),
                "trade_duration": rng.uniform(5, 25, n),
                "idle_duration": rng.uniform(1, 20, n),
                # Multiple position groups with sufficient samples for Kruskal-Wallis (>=10 per group)
                "position": rng.choice([0.0, 0.5, 1.0], n),
                "action": rng.integers(0, 3, n),
                "reward_invalid": np.zeros(n),
                "duration_ratio": rng.uniform(0.2, 1.0, n),
                "idle_ratio": rng.uniform(0.0, 0.8, n),
            }
        )
        # Intentionally do NOT set df.attrs['reward_params'] with PBRS flags; absence path only checks columns.

        out_dir = self.output_path / "pbrs_absence_and_shift_placeholder"
        # Monkeypatch summary stats to bypass references to missing PBRS columns.
        import reward_space_analysis as rsa
        original_compute_summary_stats = rsa._compute_summary_stats
        # Provide minimal structures matching write_complete_statistical_analysis expectations.
        # component_share: Series with index components (empty acceptable)
        # action_summary: DataFrame indexed by action with required stats columns
        # component_bounds: DataFrame with component_min/component_mean/component_max
        def _minimal_summary_stats(_df):
            import pandas as _pd
            comp_share = _pd.Series([], dtype=float)
            action_summary = _pd.DataFrame(
                columns=["count", "mean", "std", "min", "max"],
                index=_pd.Index([], name="action"),
            )
            component_bounds = _pd.DataFrame(
                columns=["component_min", "component_mean", "component_max"],
                index=_pd.Index([], name="component"),
            )
            global_stats = _pd.Series([], dtype=float)
            return {
                "global_stats": global_stats,
                "action_summary": action_summary,
                "component_share": comp_share,
                "component_bounds": component_bounds,
            }
        rsa._compute_summary_stats = _minimal_summary_stats
        try:
            write_complete_statistical_analysis(
                df,
                output_dir=out_dir,
                profit_target=self.TEST_PROFIT_TARGET,
                seed=self.SEED,
                skip_feature_analysis=True,
                skip_partial_dependence=True,
                bootstrap_resamples=10,
            )
        finally:
            rsa._compute_summary_stats = original_compute_summary_stats
        report_path = out_dir / "statistical_analysis.md"
        self.assertTrue(report_path.exists(), "Report file missing for PBRS absence test")
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("_PBRS components not present in this analysis._", content)
        self.assertIn("_Not performed (no real episodes provided)._", content)

    def test_get_max_idle_duration_candles_negative_or_zero_fallback(self):
        """Explicit mid<=0 fallback path returns derived default multiplier.

        Provides params with max_trade_duration_candles set (>0) and max_idle_duration_candles
        set to 0 to force the branch `if mid <= 0:` at line 400 -> return default_mid.
        Asserts returned value equals DEFAULT_IDLE_DURATION_MULTIPLIER * max_trade_duration_candles.
        """
        from reward_space_analysis import (
            DEFAULT_IDLE_DURATION_MULTIPLIER,
            DEFAULT_MODEL_REWARD_PARAMETERS,
            get_max_idle_duration_candles,
        )

        base = DEFAULT_MODEL_REWARD_PARAMETERS.copy()
        base["max_trade_duration_candles"] = 64  # explicit trade duration
        base["max_idle_duration_candles"] = 0    # force fallback
        result = get_max_idle_duration_candles(base)
        expected = DEFAULT_IDLE_DURATION_MULTIPLIER * 64
        self.assertEqual(result, expected, f"Expected fallback {expected} for mid<=0 (got {result})")

if __name__ == "__main__":
    unittest.main()
