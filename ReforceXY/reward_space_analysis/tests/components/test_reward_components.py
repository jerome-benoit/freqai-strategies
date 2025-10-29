#!/usr/bin/env python3
"""Tests for reward calculation components and algorithms."""

import math
import unittest

import pytest

from reward_space_analysis import (
    Actions,
    Positions,
    _compute_hold_potential,
    _get_exit_factor,
    _get_float_param,
    _get_pnl_factor,
    calculate_reward,
)

from ..helpers import (
    assert_component_sum_integrity,
    assert_hold_penalty_threshold_behavior,
    assert_progressive_scaling_behavior,
    assert_reward_calculation_scenarios,
    make_idle_penalty_test_contexts,
)
from ..test_base import RewardSpaceTestBase

pytestmark = pytest.mark.components  # selective execution marker


class TestRewardComponents(RewardSpaceTestBase):
    def test_hold_potential_computation_finite(self):
        """Test hold potential computation returns finite values."""
        params = {
            "hold_potential_enabled": True,
            "hold_potential_scale": 1.0,
            "hold_potential_gain": 1.0,
            "hold_potential_transform_pnl": "tanh",
            "hold_potential_transform_duration": "tanh",
        }
        val = _compute_hold_potential(0.5, 0.3, params)
        self.assertFinite(val, name="hold_potential")

    def test_hold_penalty_comprehensive(self):
        """Comprehensive hold penalty test: calculation, thresholds, and progressive scaling."""
        # Test 1: Basic hold penalty calculation via reward calculation (trade_duration > max_duration)
        context = self.make_ctx(
            pnl=0.01,
            trade_duration=150,  # > default max_duration (128)
            idle_duration=0,
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
        assert_component_sum_integrity(
            self,
            breakdown,
            self.TOL_IDENTITY_RELAXED,
            exclude_components=["idle_penalty", "exit_component", "invalid_penalty"],
            component_description="hold + shaping/additives",
        )

        # Test 2: Zero penalty before max_duration threshold using centralized helper
        max_duration = 128
        threshold_test_cases = [
            (64, "before max_duration"),
            (127, "just before max_duration"),
            (128, "exactly at max_duration"),
            (129, "just after max_duration"),
        ]

        def context_factory(trade_duration):
            return self.make_ctx(
                pnl=0.0,
                trade_duration=trade_duration,
                idle_duration=0,
                position=Positions.Long,
                action=Actions.Neutral,
            )

        assert_hold_penalty_threshold_behavior(
            self,
            threshold_test_cases,
            max_duration,
            context_factory,
            self.DEFAULT_PARAMS,
            self.TEST_BASE_FACTOR,
            self.TEST_PROFIT_TARGET,
            1.0,
            self.TOL_IDENTITY_RELAXED,
        )

        # Test 3: Progressive scaling after max_duration using centralized helper
        params = self.base_params(max_trade_duration_candles=100)
        durations = [150, 200, 300]
        penalties = []
        for duration in durations:
            context = self.make_ctx(
                pnl=0.0,
                trade_duration=duration,
                idle_duration=0,
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

        assert_progressive_scaling_behavior(self, penalties, durations, "Hold penalty")

    def test_idle_penalty_via_rewards(self):
        """Test idle penalty calculation via reward calculation."""
        context = self.make_ctx(
            pnl=0.0,
            trade_duration=0,
            idle_duration=20,
            max_unrealized_profit=0.0,
            min_unrealized_profit=0.0,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )

        def validate_idle_penalty(test_case, breakdown, description, tolerance_relaxed):
            test_case.assertLess(breakdown.idle_penalty, 0, "Idle penalty should be negative")
            assert_component_sum_integrity(
                test_case,
                breakdown,
                tolerance_relaxed,
                exclude_components=["hold_penalty", "exit_component", "invalid_penalty"],
                component_description="idle + shaping/additives",
            )

        scenarios = [(context, self.DEFAULT_PARAMS, "idle_penalty_basic")]
        assert_reward_calculation_scenarios(
            self,
            scenarios,
            self.TEST_BASE_FACTOR,
            self.TEST_PROFIT_TARGET,
            1.0,
            validate_idle_penalty,
            self.TOL_IDENTITY_RELAXED,
        )



    def test_efficiency_zero_policy(self):
        """Test efficiency zero policy."""
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
        """Test max idle duration candles logic."""
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

    # Non-owning smoke; ownership: robustness/test_robustness.py:35 (robustness-decomposition-integrity-101)
    def test_exit_factor_calculation(self):
        """Exit factor calculation across core modes + plateau variant (plateau via exit_plateau=True)."""
        modes_to_test = ["linear", "power"]
        for mode in modes_to_test:
            test_params = self.base_params(exit_attenuation_mode=mode)
            factor = _get_exit_factor(
                base_factor=1.0, pnl=0.02, pnl_factor=1.5, duration_ratio=0.3, params=test_params
            )
            self.assertFinite(factor, name=f"exit_factor[{mode}]")
            self.assertGreater(factor, 0, f"Exit factor for {mode} should be positive")
        plateau_params = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=0.5,
            exit_linear_slope=1.0,
        )
        plateau_factor_pre = _get_exit_factor(
            base_factor=1.0, pnl=0.02, pnl_factor=1.5, duration_ratio=0.4, params=plateau_params
        )
        plateau_factor_post = _get_exit_factor(
            base_factor=1.0, pnl=0.02, pnl_factor=1.5, duration_ratio=0.8, params=plateau_params
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

        def validate_zero_penalty(test_case, breakdown, description, tolerance_relaxed):
            test_case.assertEqual(
                breakdown.idle_penalty, 0.0, "Idle penalty should be zero when profit_target=0"
            )
            test_case.assertEqual(
                breakdown.total, 0.0, "Total reward should be zero in this configuration"
            )

        scenarios = [(context, self.DEFAULT_PARAMS, "profit_target_zero")]
        assert_reward_calculation_scenarios(
            self,
            scenarios,
            self.TEST_BASE_FACTOR,
            0.0,  # profit_target=0
            self.TEST_RR,
            validate_zero_penalty,
            self.TOL_IDENTITY_RELAXED,
        )

    def test_win_reward_factor_saturation(self):
        """Saturation test: pnl amplification factor should monotonically approach (1 + win_reward_factor)."""
        win_reward_factor = 3.0
        beta = 0.5
        profit_target = self.TEST_PROFIT_TARGET
        params = self.base_params(
            win_reward_factor=win_reward_factor,
            pnl_factor_beta=beta,
            efficiency_weight=0.0,
            exit_attenuation_mode="linear",
            exit_plateau=False,
            exit_linear_slope=0.0,
        )
        params.pop("base_factor", None)
        pnl_values = [profit_target * m for m in (1.05, self.TEST_RR_HIGH, 5.0, 10.0)]
        ratios_observed: list[float] = []
        for pnl in pnl_values:
            context = self.make_ctx(
                pnl=pnl,
                trade_duration=0,
                idle_duration=0,
                max_unrealized_profit=pnl,
                min_unrealized_profit=0.0,
                position=Positions.Long,
                action=Actions.Long_exit,
            )
            br = calculate_reward(
                context,
                params,
                base_factor=1.0,
                profit_target=profit_target,
                risk_reward_ratio=1.0,
                short_allowed=True,
                action_masking=True,
            )
            ratio = br.exit_component / pnl if pnl != 0 else 0.0
            ratios_observed.append(float(ratio))
        self.assertMonotonic(
            ratios_observed,
            non_decreasing=True,
            tolerance=self.TOL_IDENTITY_STRICT,
            name="pnl_amplification_ratio",
        )
        asymptote = 1.0 + win_reward_factor
        final_ratio = ratios_observed[-1]
        self.assertFinite(final_ratio, name="final_ratio")
        self.assertLess(
            abs(final_ratio - asymptote),
            0.001,
            f"Final amplification {final_ratio:.6f} not close to asymptote {asymptote:.6f}",
        )
        expected_ratios: list[float] = []
        for pnl in pnl_values:
            pnl_ratio = pnl / profit_target
            expected = 1.0 + win_reward_factor * math.tanh(beta * (pnl_ratio - 1.0))
            expected_ratios.append(expected)
        for obs, exp in zip(ratios_observed, expected_ratios):
            self.assertFinite(obs, name="observed_ratio")
            self.assertFinite(exp, name="expected_ratio")
            self.assertLess(
                abs(obs - exp),
                5e-06,
                f"Observed amplification {obs:.8f} deviates from expected {exp:.8f}",
            )

    def test_idle_penalty_fallback_and_proportionality(self):
        """Idle penalty fallback denominator & proportional scaling."""
        params = self.base_params(max_idle_duration_candles=None, max_trade_duration_candles=100)
        base_factor = 90.0
        profit_target = self.TEST_PROFIT_TARGET
        risk_reward_ratio = 1.0

        # Generate test contexts using helper
        base_context_kwargs = {
            "pnl": 0.0,
            "trade_duration": 0,
            "position": Positions.Neutral,
            "action": Actions.Neutral,
        }
        idle_scenarios = [20, 40, 120]
        contexts_and_descriptions = make_idle_penalty_test_contexts(
            self.make_ctx, idle_scenarios, base_context_kwargs
        )

        # Calculate all rewards
        results = []
        for context, description in contexts_and_descriptions:
            breakdown = calculate_reward(
                context,
                params,
                base_factor=base_factor,
                profit_target=profit_target,
                risk_reward_ratio=risk_reward_ratio,
                short_allowed=True,
                action_masking=True,
            )
            results.append((breakdown, context.idle_duration, description))

        # Validate proportional scaling
        br_a, br_b, br_mid = [r[0] for r in results]
        self.assertLess(br_a.idle_penalty, 0.0)
        self.assertLess(br_b.idle_penalty, 0.0)
        self.assertLess(br_mid.idle_penalty, 0.0)

        # Check 2:1 ratio between 40 and 20 idle duration
        ratio = br_b.idle_penalty / br_a.idle_penalty if br_a.idle_penalty != 0 else None
        self.assertIsNotNone(ratio)
        if ratio is not None:
            self.assertAlmostEqualFloat(abs(ratio), 2.0, tolerance=0.2)

        # Mathematical validation for mid-duration case
        idle_penalty_scale = _get_float_param(params, "idle_penalty_scale", 0.5)
        idle_penalty_power = _get_float_param(params, "idle_penalty_power", 1.025)
        factor = _get_float_param(params, "base_factor", float(base_factor))
        idle_factor = factor * (profit_target * risk_reward_ratio) / 4.0
        observed_ratio = abs(br_mid.idle_penalty) / (idle_factor * idle_penalty_scale)
        if observed_ratio > 0:
            implied_D = 120 / observed_ratio ** (1 / idle_penalty_power)
            self.assertAlmostEqualFloat(implied_D, 400.0, tolerance=20.0)


if __name__ == "__main__":
    unittest.main()
