#!/usr/bin/env python3
"""Robustness tests and boundary condition validation."""

import math
import unittest

import numpy as np
import pytest

from reward_space_analysis import (
    ATTENUATION_MODES,
    ATTENUATION_MODES_WITH_LEGACY,
    Actions,
    Positions,
    RewardContext,
    _get_exit_factor,
    simulate_samples,
)

from ..constants import (
    CONTINUITY,
    EXIT_FACTOR,
    PARAMS,
    SEEDS,
    TOLERANCE,
)
from ..helpers import (
    assert_diagnostic_warning,
    assert_exit_factor_attenuation_modes,
    assert_exit_mode_mathematical_validation,
    assert_single_active_component_with_additives,
    calculate_reward_with_defaults,
    capture_warnings,
)
from ..test_base import RewardSpaceTestBase

pytestmark = pytest.mark.robustness


class TestRewardRobustnessAndBoundaries(RewardSpaceTestBase):
    """Robustness invariants, attenuation maths, parameter edges, scaling, warnings."""

    # Owns invariant: robustness-decomposition-integrity-101 (robustness category)
    def test_decomposition_integrity(self):
        """reward must equal the single active core component under mutually exclusive scenarios (idle/hold/exit/invalid)."""
        scenarios = [
            {
                "ctx": self.make_ctx(
                    pnl=0.0,
                    trade_duration=0,
                    idle_duration=25,
                    max_unrealized_profit=0.0,
                    min_unrealized_profit=0.0,
                    position=Positions.Neutral,
                    action=Actions.Neutral,
                ),
                "active": "idle_penalty",
            },
            {
                "ctx": self.make_ctx(
                    pnl=0.0,
                    trade_duration=150,
                    idle_duration=0,
                    max_unrealized_profit=0.0,
                    min_unrealized_profit=0.0,
                    position=Positions.Long,
                    action=Actions.Neutral,
                ),
                "active": "hold_penalty",
            },
            {
                "ctx": self.make_ctx(
                    pnl=PARAMS.PROFIT_AIM,
                    trade_duration=60,
                    idle_duration=0,
                    max_unrealized_profit=0.05,
                    min_unrealized_profit=0.01,
                    position=Positions.Long,
                    action=Actions.Long_exit,
                ),
                "active": "exit_component",
            },
            {
                "ctx": self.make_ctx(
                    pnl=0.01,
                    trade_duration=10,
                    idle_duration=0,
                    max_unrealized_profit=0.02,
                    min_unrealized_profit=0.0,
                    position=Positions.Short,
                    action=Actions.Long_exit,
                ),
                "active": "invalid_penalty",
            },
        ]
        for sc in scenarios:
            ctx_obj = sc["ctx"]
            active_label = sc["active"]
            assert isinstance(ctx_obj, RewardContext), (
                f"Expected RewardContext, got {type(ctx_obj)}"
            )
            assert isinstance(active_label, str), f"Expected str, got {type(active_label)}"
            with self.subTest(active=active_label):
                params = self.base_params(
                    entry_additive_enabled=False,
                    exit_additive_enabled=False,
                    hold_potential_enabled=False,
                    potential_gamma=0.0,
                    check_invariants=False,
                )
                br = calculate_reward_with_defaults(ctx_obj, params)
                # Relaxed tolerance: Accumulated floating-point errors across multiple
                # reward component calculations (entry, hold, exit additives, and penalties)
                assert_single_active_component_with_additives(
                    self,
                    br,
                    active_label,
                    TOLERANCE.IDENTITY_RELAXED,
                    inactive_core=[
                        "exit_component",
                        "idle_penalty",
                        "hold_penalty",
                        "invalid_penalty",
                    ],
                )

    # Owns invariant: robustness-exit-pnl-only-117 (robustness category)
    def test_pnl_invariant_exit_only(self):
        """Invariant: PnL only non-zero while in position.

        The simulator uses coherent trajectories, so PnL is a state variable during
        holds and entries; however Neutral samples must have pnl == 0.
        """
        df = simulate_samples(
            params=self.base_params(max_trade_duration_candles=50),
            num_samples=200,
            seed=SEEDS.BASE,
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
            max_duration_ratio=2.0,
            trading_mode="margin",
            pnl_base_std=PARAMS.PNL_STD,
            pnl_duration_vol_scale=PARAMS.PNL_DUR_VOL_SCALE,
        )
        neutral_mask = df["position"] == float(Positions.Neutral.value)
        non_zero_neutral_pnl = df.loc[neutral_mask, "pnl"].abs().max()
        self.assertLessEqual(
            float(non_zero_neutral_pnl),
            np.finfo(float).eps,
            msg="PnL invariant violation: neutral states must have pnl == 0",
        )

    def test_exit_factor_comprehensive(self):
        """Comprehensive exit factor test: mathematical correctness and monotonic attenuation."""
        # Part 1: Mathematical formulas validation
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

        # Relaxed tolerance: Exit factor calculations involve multiple steps
        # (normalization, kernel application, potential transforms)
        assert_exit_mode_mathematical_validation(
            self,
            context,
            params,
            PARAMS.BASE_FACTOR,
            PARAMS.PROFIT_AIM,
            PARAMS.RISK_REWARD_RATIO,
            TOLERANCE.IDENTITY_RELAXED,
        )

        # Part 2: Monotonic attenuation validation
        modes = [*list(ATTENUATION_MODES), "plateau_linear"]
        test_pnl = 0.05
        test_context = self.make_ctx(
            pnl=test_pnl,
            trade_duration=50,
            max_unrealized_profit=0.06,
            min_unrealized_profit=0.0,
        )
        # Relaxed tolerance: Testing across multiple attenuation modes with different
        # numerical characteristics (exponential, polynomial, rational functions)
        assert_exit_factor_attenuation_modes(
            self,
            base_factor=PARAMS.BASE_FACTOR,
            pnl=test_pnl,
            pnl_target=PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO,
            context=test_context,
            attenuation_modes=modes,
            base_params_fn=self.base_params,
            tolerance_relaxed=TOLERANCE.IDENTITY_RELAXED,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
        )

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
        with capture_warnings() as caught:
            baseline = calculate_reward_with_defaults(
                context, params, risk_reward_ratio=PARAMS.RISK_REWARD_RATIO_HIGH
            )
            amplified_base_factor = PARAMS.BASE_FACTOR * 200.0
            amplified = calculate_reward_with_defaults(
                context,
                params,
                base_factor=amplified_base_factor,
                risk_reward_ratio=PARAMS.RISK_REWARD_RATIO_HIGH,
            )
        self.assertGreater(baseline.exit_component, 0.0)
        self.assertGreater(amplified.exit_component, baseline.exit_component)
        scale = amplified.exit_component / baseline.exit_component
        self.assertGreater(scale, 10.0)
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertTrue(runtime_warnings)
        self.assertTrue(
            any(
                (">" in str(w.message) and "threshold" in str(w.message))
                or "|exit_factor|=" in str(w.message)
                for w in runtime_warnings
            )
        )

    def test_negative_slope_sanitization(self):
        """Negative exit_linear_slope is sanitized to 1.0; resulting exit factors must match slope=1.0 within tolerance."""
        base_factor = 100.0
        pnl = 0.03
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO
        test_context = self.make_ctx(
            pnl=pnl, trade_duration=50, max_unrealized_profit=0.04, min_unrealized_profit=0.0
        )
        duration_ratios = [0.0, 0.2, 0.5, 1.0, 1.5]
        params_bad = self.base_params(
            exit_attenuation_mode="linear", exit_linear_slope=-5.0, exit_plateau=False
        )
        params_ref = self.base_params(
            exit_attenuation_mode="linear", exit_linear_slope=1.0, exit_plateau=False
        )
        for dr in duration_ratios:
            f_bad = _get_exit_factor(
                base_factor, pnl, pnl_target, dr, test_context, params_bad, PARAMS.RISK_REWARD_RATIO
            )
            f_ref = _get_exit_factor(
                base_factor, pnl, pnl_target, dr, test_context, params_ref, PARAMS.RISK_REWARD_RATIO
            )
            # Relaxed tolerance: Comparing exit factors computed with different slope values
            # after sanitization; minor numerical differences expected
            self.assertAlmostEqualFloat(
                f_bad,
                f_ref,
                tolerance=TOLERANCE.IDENTITY_RELAXED,
                msg=f"Sanitized slope mismatch at dr={dr} f_bad={f_bad} f_ref={f_ref}",
            )

    def test_power_mode_alpha_formula(self):
        """Power mode attenuation: ratio f(dr=1)/f(dr=0) must equal 1/(1+1)^alpha with alpha=-log(tau)/log(2)."""
        base_factor = 200.0
        pnl = 0.04
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO
        test_context = self.make_ctx(
            pnl=pnl, trade_duration=50, max_unrealized_profit=0.05, min_unrealized_profit=0.0
        )
        duration_ratio = 1.0
        taus = [0.9, 0.5, 0.25, 1.0]
        for tau in taus:
            params = self.base_params(
                exit_attenuation_mode="power", exit_power_tau=tau, exit_plateau=False
            )
            f0 = _get_exit_factor(
                base_factor, pnl, pnl_target, 0.0, test_context, params, PARAMS.RISK_REWARD_RATIO
            )
            f1 = _get_exit_factor(
                base_factor,
                pnl,
                pnl_target,
                duration_ratio,
                test_context,
                params,
                PARAMS.RISK_REWARD_RATIO,
            )
            alpha = -math.log(tau) / math.log(2.0) if 0.0 < tau <= 1.0 else 1.0
            expected_ratio = 1.0 / (1.0 + duration_ratio) ** alpha
            observed_ratio = f1 / f0 if f0 != 0 else np.nan
            self.assertFinite(observed_ratio, name="observed_ratio")
            self.assertLess(
                abs(observed_ratio - expected_ratio),
                5e-12 if tau == 1.0 else 5e-09,
                f"Alpha attenuation mismatch tau={tau} alpha={alpha} obs_ratio={observed_ratio} exp_ratio={expected_ratio}",
            )

    def test_reward_calculation_extreme_parameters_stability(self):
        """Reward calculation remains numerically stable with extreme parameter values.

        Tests numerical stability and finite output when using extreme parameter
        values (win_reward_factor=1000, base_factor=10000) that might cause
        overflow or NaN propagation in poorly designed implementations.

        **Setup:**
        - Extreme parameters: win_reward_factor=1000.0, base_factor=10000.0
        - Context: Long exit with pnl=0.05, duration=50, profit extrema=[0.02, 0.06]
        - Configuration: short_allowed=True, action_masking=True

        **Assertions:**
        - Total reward is finite (not NaN, not Inf)

        **Tolerance rationale:**
        - Uses assertFinite which checks for non-NaN, non-Inf values only
        """
        extreme_params = self.base_params(win_reward_factor=1000.0, base_factor=10000.0)
        context = self.make_ctx(
            pnl=0.05,
            trade_duration=50,
            idle_duration=0,
            max_unrealized_profit=0.06,
            min_unrealized_profit=0.02,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        br = calculate_reward_with_defaults(context, extreme_params, base_factor=10000.0)
        self.assertFinite(br.total, name="breakdown.total")

    def test_exit_attenuation_modes_enumeration(self):
        """All exit attenuation modes produce finite rewards without errors.

        Smoke test ensuring each exit attenuation mode (including legacy modes)
        executes successfully and produces finite reward components. This validates
        that mode enumeration is complete and all modes are correctly implemented.

        **Setup:**
        - Modes tested: All values in ATTENUATION_MODES_WITH_LEGACY
        - Context: Long exit with pnl=0.02, duration=50, profit extrema=[0.01, 0.03]
        - Uses subTest for mode-specific failure isolation

        **Assertions:**
        - Exit component is finite for each mode
        - Total reward is finite for each mode

        **Tolerance rationale:**
        - Uses assertFinite which checks for non-NaN, non-Inf values only
        """
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
                br = calculate_reward_with_defaults(ctx, test_params)
                self.assertFinite(br.exit_component, name="breakdown.exit_component")
                self.assertFinite(br.total, name="breakdown.total")

    def test_exit_factor_boundary_parameters(self):
        """Test parameter edge cases: tau extrema, plateau grace edges, slope zero."""
        base_factor = 50.0
        pnl = 0.02
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO
        test_context = self.make_ctx(
            pnl=pnl, trade_duration=50, max_unrealized_profit=0.03, min_unrealized_profit=0.0
        )
        params_hi = self.base_params(exit_attenuation_mode="power", exit_power_tau=0.999999)
        params_lo = self.base_params(
            exit_attenuation_mode="power", exit_power_tau=EXIT_FACTOR.MIN_POWER_TAU
        )
        r = 1.5
        hi_val = _get_exit_factor(
            base_factor, pnl, pnl_target, r, test_context, params_hi, PARAMS.RISK_REWARD_RATIO
        )
        lo_val = _get_exit_factor(
            base_factor, pnl, pnl_target, r, test_context, params_lo, PARAMS.RISK_REWARD_RATIO
        )
        self.assertGreater(
            hi_val, lo_val, "Power mode: higher tau (≈1) should attenuate less than tiny tau"
        )
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
        val_g0 = _get_exit_factor(
            base_factor, pnl, pnl_target, 0.5, test_context, params_g0, PARAMS.RISK_REWARD_RATIO
        )
        val_g1 = _get_exit_factor(
            base_factor, pnl, pnl_target, 0.5, test_context, params_g1, PARAMS.RISK_REWARD_RATIO
        )
        self.assertGreater(
            val_g1, val_g0, "Plateau grace=1.0 should delay attenuation vs grace=0.0"
        )
        params_lin0 = self.base_params(
            exit_attenuation_mode="linear", exit_linear_slope=0.0, exit_plateau=False
        )
        params_lin1 = self.base_params(
            exit_attenuation_mode="linear", exit_linear_slope=2.0, exit_plateau=False
        )
        val_lin0 = _get_exit_factor(
            base_factor, pnl, pnl_target, 1.0, test_context, params_lin0, PARAMS.RISK_REWARD_RATIO
        )
        val_lin1 = _get_exit_factor(
            base_factor, pnl, pnl_target, 1.0, test_context, params_lin1, PARAMS.RISK_REWARD_RATIO
        )
        self.assertGreater(
            val_lin0, val_lin1, "Linear slope=0 should yield no attenuation vs slope>0"
        )

    def test_plateau_linear_slope_zero_constant_after_grace(self):
        """Plateau+linear slope=0 should yield flat factor after grace boundary (no attenuation)."""
        params = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=0.3,
            exit_linear_slope=0.0,
        )
        base_factor = PARAMS.BASE_FACTOR
        pnl = 0.04
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO
        test_context = self.make_ctx(
            pnl=pnl, trade_duration=50, max_unrealized_profit=0.05, min_unrealized_profit=0.0
        )
        ratios = [0.3, 0.6, 1.0, 1.4]
        values = [
            _get_exit_factor(
                base_factor, pnl, pnl_target, r, test_context, params, PARAMS.RISK_REWARD_RATIO
            )
            for r in ratios
        ]
        first = values[0]
        for v in values[1:]:
            # Relaxed tolerance: Exit factor should remain constant across all duration
            # ratios when slope=0; accumulated errors from multiple calculations
            self.assertAlmostEqualFloat(
                v,
                first,
                tolerance=TOLERANCE.IDENTITY_RELAXED,
                msg=f"Plateau+linear slope=0 factor drift at ratio set {ratios} => {values}",
            )

    def test_plateau_grace_extends_beyond_one(self):
        """Plateau grace >1.0 should keep full strength (no attenuation) past duration_ratio=1."""
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "exit_attenuation_mode": "linear",
                "exit_plateau": True,
                "exit_plateau_grace": 1.5,
                "exit_linear_slope": 2.0,
            }
        )
        base_factor = 80.0
        profit_aim = PARAMS.PROFIT_AIM
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO
        test_context = self.make_ctx(
            pnl=profit_aim, trade_duration=50, max_unrealized_profit=0.04, min_unrealized_profit=0.0
        )
        ratios = [0.8, 1.0, 1.2, 1.4, 1.6]
        vals = [
            _get_exit_factor(
                base_factor,
                profit_aim,
                pnl_target,
                r,
                test_context,
                params,
                PARAMS.RISK_REWARD_RATIO,
            )
            for r in ratios
        ]
        ref = vals[0]
        for i, r in enumerate(ratios[:-1]):
            # Relaxed tolerance: All values before grace boundary should match;
            # minor differences from repeated exit factor computations expected
            self.assertAlmostEqualFloat(
                vals[i],
                ref,
                tolerance=TOLERANCE.IDENTITY_RELAXED,
                msg=f"Unexpected attenuation before grace end at ratio {r}",
            )
        self.assertLess(vals[-1], ref, "Attenuation should begin after grace boundary")

    def test_plateau_continuity_at_grace_boundary(self):
        """Test plateau continuity at grace boundary."""
        modes = list(ATTENUATION_MODES)
        grace = 0.8
        eps = CONTINUITY.EPS_SMALL
        base_factor = PARAMS.BASE_FACTOR
        pnl = 0.01
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO
        test_context = self.make_ctx(
            pnl=pnl, trade_duration=50, max_unrealized_profit=0.02, min_unrealized_profit=0.0
        )
        tau = 0.5
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
                    base_factor,
                    pnl,
                    pnl_target,
                    grace - eps,
                    test_context,
                    params,
                    PARAMS.RISK_REWARD_RATIO,
                )
                boundary = _get_exit_factor(
                    base_factor,
                    pnl,
                    pnl_target,
                    grace,
                    test_context,
                    params,
                    PARAMS.RISK_REWARD_RATIO,
                )
                right = _get_exit_factor(
                    base_factor,
                    pnl,
                    pnl_target,
                    grace + eps,
                    test_context,
                    params,
                    PARAMS.RISK_REWARD_RATIO,
                )
                # Relaxed tolerance: Continuity check at plateau grace boundary;
                # left and boundary values should be nearly identical
                self.assertAlmostEqualFloat(
                    left,
                    boundary,
                    tolerance=TOLERANCE.IDENTITY_RELAXED,
                    msg=f"Left/boundary mismatch for mode {mode}",
                )
                self.assertLess(
                    right, boundary, f"No attenuation detected just after grace for mode {mode}"
                )
                diff = boundary - right
                if mode == "linear":
                    bound = base_factor * slope * eps * CONTINUITY.BOUND_MULTIPLIER_LINEAR
                elif mode == "sqrt":
                    bound = base_factor * 0.5 * eps * CONTINUITY.BOUND_MULTIPLIER_SQRT
                elif mode == "power":
                    alpha = -math.log(tau) / math.log(2.0)
                    bound = base_factor * alpha * eps * CONTINUITY.BOUND_MULTIPLIER_POWER
                elif mode == "half_life":
                    bound = (
                        base_factor
                        * (math.log(2.0) / half_life)
                        * eps
                        * CONTINUITY.BOUND_MULTIPLIER_HALF_LIFE
                    )
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
        eps1 = CONTINUITY.EPS_LARGE
        eps2 = CONTINUITY.EPS_SMALL
        base_factor = 80.0
        pnl = 0.02
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO_HIGH
        test_context = self.make_ctx(
            pnl=pnl, trade_duration=50, max_unrealized_profit=0.03, min_unrealized_profit=0.0
        )
        params = self.DEFAULT_PARAMS.copy()
        params.update(
            {
                "exit_attenuation_mode": mode,
                "exit_plateau": True,
                "exit_plateau_grace": grace,
                "exit_linear_slope": 1.1,
            }
        )
        f_boundary = _get_exit_factor(
            base_factor, pnl, pnl_target, grace, test_context, params, PARAMS.RISK_REWARD_RATIO_HIGH
        )
        f1 = _get_exit_factor(
            base_factor,
            pnl,
            pnl_target,
            grace + eps1,
            test_context,
            params,
            PARAMS.RISK_REWARD_RATIO_HIGH,
        )
        f2 = _get_exit_factor(
            base_factor,
            pnl,
            pnl_target,
            grace + eps2,
            test_context,
            params,
            PARAMS.RISK_REWARD_RATIO_HIGH,
        )
        diff1 = f_boundary - f1
        diff2 = f_boundary - f2
        # NUMERIC_GUARD: Prevent division by zero when computing scaling ratio
        ratio = diff1 / max(diff2, TOLERANCE.NUMERIC_GUARD)
        self.assertGreater(
            ratio,
            EXIT_FACTOR.SCALING_RATIO_MIN,
            f"Scaling ratio too small (ratio={ratio:.2f})",
        )
        self.assertLess(
            ratio,
            EXIT_FACTOR.SCALING_RATIO_MAX,
            f"Scaling ratio too large (ratio={ratio:.2f})",
        )

    # === Robustness invariants 102–105 ===  # noqa: RUF003
    # Owns invariant: robustness-exit-mode-fallback-102
    def test_robustness_102_unknown_exit_mode_fallback_linear(self):
        """Invariant 102: Unknown exit_attenuation_mode gracefully warns and falls back to linear kernel."""
        params = self.base_params(
            exit_attenuation_mode="nonexistent_kernel_xyz", exit_plateau=False
        )
        base_factor = 75.0
        pnl = 0.05
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO
        test_context = self.make_ctx(
            pnl=pnl, trade_duration=50, max_unrealized_profit=0.06, min_unrealized_profit=0.0
        )
        duration_ratio = 0.8
        with assert_diagnostic_warning(["Unknown exit_attenuation_mode"]):
            f_unknown = _get_exit_factor(
                base_factor,
                pnl,
                pnl_target,
                duration_ratio,
                test_context,
                params,
                PARAMS.RISK_REWARD_RATIO_HIGH,
            )
        linear_params = self.base_params(exit_attenuation_mode="linear", exit_plateau=False)
        f_linear = _get_exit_factor(
            base_factor,
            pnl,
            pnl_target,
            duration_ratio,
            test_context,
            linear_params,
            PARAMS.RISK_REWARD_RATIO_HIGH,
        )
        # Relaxed tolerance: Unknown exit mode should fall back to linear mode;
        # verifying identical behavior between fallback and explicit linear
        self.assertAlmostEqualFloat(
            f_unknown,
            f_linear,
            tolerance=TOLERANCE.IDENTITY_RELAXED,
            msg=f"Fallback linear mismatch unknown={f_unknown} linear={f_linear}",
        )

    # Owns invariant: robustness-negative-grace-clamp-103
    def test_robustness_103_negative_plateau_grace_clamped(self):
        """Invariant 103: Negative exit_plateau_grace emits warning and clamps to 0.0 (no plateau extension)."""
        params = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=-2.0,
            exit_linear_slope=1.2,
        )
        base_factor = PARAMS.BASE_FACTOR
        pnl = 0.03
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO_HIGH
        test_context = self.make_ctx(
            pnl=pnl, trade_duration=50, max_unrealized_profit=0.04, min_unrealized_profit=0.0
        )
        duration_ratio = 0.5
        with assert_diagnostic_warning(["exit_plateau_grace=", "< 0"]):
            f_neg = _get_exit_factor(
                base_factor,
                pnl,
                pnl_target,
                duration_ratio,
                test_context,
                params,
                PARAMS.RISK_REWARD_RATIO_HIGH,
            )
        # Reference with grace=0.0 (since negative should clamp)
        ref_params = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=0.0,
            exit_linear_slope=1.2,
        )
        f_ref = _get_exit_factor(
            base_factor,
            pnl,
            pnl_target,
            duration_ratio,
            test_context,
            ref_params,
            PARAMS.RISK_REWARD_RATIO_HIGH,
        )
        # Relaxed tolerance: Negative grace parameter should clamp to 0.0;
        # verifying clamped behavior matches explicit grace=0.0 configuration
        self.assertAlmostEqualFloat(
            f_neg,
            f_ref,
            tolerance=TOLERANCE.IDENTITY_RELAXED,
            msg=f"Negative grace clamp mismatch f_neg={f_neg} f_ref={f_ref}",
        )

    # Owns invariant: robustness-invalid-power-tau-104
    def test_robustness_104_invalid_power_tau_fallback_alpha_one(self):
        """Invariant 104: Invalid exit_power_tau (<=0 or >1 or NaN) warns and falls back alpha=1.0."""
        invalid_taus = [0.0, -0.5, 2.0, float("nan")]
        base_factor = 120.0
        pnl = 0.04
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO
        test_context = self.make_ctx(
            pnl=pnl, trade_duration=50, max_unrealized_profit=0.05, min_unrealized_profit=0.0
        )
        duration_ratio = 1.0
        # Explicit alpha=1 expected ratio: f(dr)/f(0)=1/(1+dr)^1 with plateau disabled to observe attenuation.
        expected_ratio_alpha1 = 1.0 / (1.0 + duration_ratio)
        for tau in invalid_taus:
            params = self.base_params(
                exit_attenuation_mode="power", exit_power_tau=tau, exit_plateau=False
            )
            with assert_diagnostic_warning(["exit_power_tau"]):
                f0 = _get_exit_factor(
                    base_factor,
                    pnl,
                    pnl_target,
                    0.0,
                    test_context,
                    params,
                    PARAMS.RISK_REWARD_RATIO,
                )
                f1 = _get_exit_factor(
                    base_factor,
                    pnl,
                    pnl_target,
                    duration_ratio,
                    test_context,
                    params,
                    PARAMS.RISK_REWARD_RATIO,
                )
            # NUMERIC_GUARD: Prevent division by zero when computing power mode ratio
            ratio = f1 / max(f0, TOLERANCE.NUMERIC_GUARD)
            self.assertAlmostEqual(
                ratio,
                expected_ratio_alpha1,
                places=TOLERANCE.DECIMAL_PLACES_STANDARD,
                msg=f"Alpha=1 fallback ratio mismatch tau={tau} ratio={ratio} expected={expected_ratio_alpha1}",
            )

    # Owns invariant: robustness-near-zero-half-life-105
    def test_robustness_105_half_life_near_zero_fallback(self):
        """Invariant 105: Near-zero exit_half_life yields no attenuation (factor≈base).

        This invariant is specifically about the *time attenuation kernel*:
        `exit_attenuation_mode="half_life"` should return a time coefficient of 1.0 when
        `exit_half_life` is close to zero.

        To isolate the time coefficient, we choose inputs that keep the other
        multiplicative coefficients at 1.0 (pnl_target and efficiency).
        """

        base_factor = 60.0
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO_HIGH
        pnl = 0.5 * pnl_target
        test_context = self.make_ctx(
            pnl=pnl,
            trade_duration=50,
            max_unrealized_profit=pnl,
            min_unrealized_profit=0.0,
        )
        duration_ratio = 0.7

        near_zero_values = [1e-15, 1e-12, 5e-14]
        for hl in near_zero_values:
            params = self.base_params(
                exit_attenuation_mode="half_life",
                exit_half_life=hl,
                efficiency_weight=0.0,
                win_reward_factor=0.0,
            )
            with assert_diagnostic_warning(["exit_half_life=", "<= 0"]):
                f0 = _get_exit_factor(
                    base_factor,
                    pnl,
                    pnl_target,
                    0.0,
                    test_context,
                    params,
                    PARAMS.RISK_REWARD_RATIO_HIGH,
                )
                fdr = _get_exit_factor(
                    base_factor,
                    pnl,
                    pnl_target,
                    duration_ratio,
                    test_context,
                    params,
                    PARAMS.RISK_REWARD_RATIO_HIGH,
                )

            self.assertFinite(fdr, name="fdr")
            self.assertAlmostEqualFloat(
                fdr,
                base_factor,
                tolerance=TOLERANCE.IDENTITY_STRICT,
                msg=f"Expected no time attenuation for near-zero half-life hl={hl} (fdr={fdr})",
            )
            self.assertAlmostEqualFloat(
                f0,
                base_factor,
                tolerance=TOLERANCE.IDENTITY_STRICT,
                msg=f"Expected factor==base at dr=0 for hl={hl} (f0={f0})",
            )
            self.assertAlmostEqualFloat(
                fdr,
                f0,
                tolerance=TOLERANCE.IDENTITY_STRICT,
                msg=f"Expected dr-insensitive factor under half-life near zero hl={hl} (f0={f0}, fdr={fdr})",
            )


if __name__ == "__main__":
    unittest.main()
