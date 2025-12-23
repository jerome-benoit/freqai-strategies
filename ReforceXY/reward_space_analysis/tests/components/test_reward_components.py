#!/usr/bin/env python3
"""Tests for reward calculation components and algorithms."""

import math
import unittest

import pytest

from reward_space_analysis import (
    Actions,
    Positions,
    _compute_efficiency_coefficient,
    _compute_hold_potential,
    _compute_pnl_target_coefficient,
    _get_exit_factor,
    _get_float_param,
    calculate_reward,
)

from ..constants import PARAMS, SCENARIOS, TOLERANCE
from ..helpers import (
    RewardScenarioConfig,
    ThresholdTestConfig,
    ValidationConfig,
    assert_component_sum_integrity,
    assert_exit_factor_plateau_behavior,
    assert_hold_penalty_threshold_behavior,
    assert_progressive_scaling_behavior,
    assert_reward_calculation_scenarios,
    make_idle_penalty_test_contexts,
)
from ..test_base import RewardSpaceTestBase

pytestmark = pytest.mark.components


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
        val = _compute_hold_potential(
            0.5, PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO, 0.3, params
        )
        self.assertFinite(val, name="hold_potential")

    def test_hold_penalty_basic_calculation(self):
        """Test hold penalty calculation when trade_duration exceeds max_duration.

        Verifies:
        - trade_duration > max_duration → hold_penalty < 0
        - Total reward equals sum of active components
        """
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
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
            short_allowed=True,
            action_masking=True,
        )
        self.assertLess(breakdown.hold_penalty, 0, "Hold penalty should be negative")
        config = ValidationConfig(
            tolerance_strict=TOLERANCE.IDENTITY_STRICT,
            tolerance_relaxed=TOLERANCE.IDENTITY_RELAXED,
            exclude_components=["idle_penalty", "exit_component", "invalid_penalty"],
            component_description="hold + shaping/additives",
        )
        assert_component_sum_integrity(self, breakdown, config)

    def test_hold_penalty_threshold_behavior(self):
        """Test hold penalty activation at max_duration threshold.

        Verifies:
        - duration < max_duration → hold_penalty = 0
        - duration >= max_duration → hold_penalty <= 0
        """
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

        config = ThresholdTestConfig(
            max_duration=max_duration,
            test_cases=threshold_test_cases,
            tolerance=TOLERANCE.IDENTITY_RELAXED,
        )
        assert_hold_penalty_threshold_behavior(
            self,
            context_factory,
            self.DEFAULT_PARAMS,
            PARAMS.BASE_FACTOR,
            PARAMS.PROFIT_AIM,
            1.0,
            config,
        )

    def test_hold_penalty_progressive_scaling(self):
        """Test hold penalty scales progressively with increasing duration.

        Verifies:
        - For d1 < d2 < d3: penalty(d1) >= penalty(d2) >= penalty(d3)
        - Progressive scaling beyond max_duration threshold
        """

        params = self.base_params(max_trade_duration_candles=100)
        durations = list(SCENARIOS.DURATION_SCENARIOS)
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
                base_factor=PARAMS.BASE_FACTOR,
                profit_aim=PARAMS.PROFIT_AIM,
                risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
                short_allowed=True,
                action_masking=True,
            )
            penalties.append(breakdown.hold_penalty)

        assert_progressive_scaling_behavior(self, penalties, durations, "Hold penalty")

    def test_idle_penalty_calculation(self):
        """Test idle penalty calculation for neutral idle state.

        Verifies:
        - idle_duration > 0 → idle_penalty < 0
        - Component sum integrity maintained
        """
        context = self.make_ctx(
            pnl=0.0,
            trade_duration=0,
            idle_duration=20,
            max_unrealized_profit=0.0,
            min_unrealized_profit=0.0,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )

        def validate_idle_penalty(test_case, breakdown, description, tolerance):
            test_case.assertLess(breakdown.idle_penalty, 0, "Idle penalty should be negative")
            config = ValidationConfig(
                tolerance_strict=TOLERANCE.IDENTITY_STRICT,
                tolerance_relaxed=tolerance,
                exclude_components=["hold_penalty", "exit_component", "invalid_penalty"],
                component_description="idle + shaping/additives",
            )
            assert_component_sum_integrity(test_case, breakdown, config)

        scenarios = [(context, self.DEFAULT_PARAMS, "idle_penalty_basic")]
        config = RewardScenarioConfig(
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=1.0,
            tolerance_relaxed=TOLERANCE.IDENTITY_RELAXED,
        )
        assert_reward_calculation_scenarios(
            self,
            scenarios,
            config,
            validate_idle_penalty,
        )

    def test_efficiency_zero_policy(self):
        """Test efficiency zero policy produces expected PnL coefficient.

        Verifies:
        - efficiency_weight = 0 -> pnl_coefficient ~= 1.0
        - Coefficient is finite and positive
        """
        ctx = self.make_ctx(
            pnl=0.0,
            trade_duration=1,
            max_unrealized_profit=0.0,
            min_unrealized_profit=-0.02,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        params = self.base_params(efficiency_weight=0.0)
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO
        pnl_target_coefficient = _compute_pnl_target_coefficient(
            params, ctx.pnl, pnl_target, PARAMS.RISK_REWARD_RATIO
        )
        efficiency_coefficient = _compute_efficiency_coefficient(params, ctx, ctx.pnl)
        pnl_coefficient = pnl_target_coefficient * efficiency_coefficient
        self.assertFinite(pnl_coefficient, name="pnl_coefficient")
        self.assertAlmostEqualFloat(pnl_coefficient, 1.0, tolerance=TOLERANCE.GENERIC_EQ)

    def test_exit_reward_never_positive_for_loss_due_to_efficiency(self):
        """Exit reward should not become positive for a loss trade.

        This guards against a configuration where the efficiency coefficient becomes
        negative (e.g., extreme efficiency_weight/efficiency_center), which would
        otherwise flip the sign of pnl * exit_factor.
        """
        params = self.base_params(
            efficiency_weight=2.0,
            efficiency_center=0.0,
            exit_attenuation_mode="linear",
            exit_plateau=False,
            exit_linear_slope=0.0,
            hold_potential_enabled=False,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
        )
        params.pop("base_factor", None)

        context = self.make_ctx(
            pnl=-0.01,
            trade_duration=10,
            idle_duration=0,
            max_unrealized_profit=0.0,
            min_unrealized_profit=-0.05,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        breakdown = calculate_reward(
            context,
            params,
            base_factor=1.0,
            profit_aim=0.03,
            risk_reward_ratio=1.0,
            short_allowed=True,
            action_masking=True,
        )
        self.assertLessEqual(
            breakdown.exit_component,
            0.0,
            "Exit component must not be positive when pnl < 0",
        )

    def test_max_idle_duration_candles_logic(self):
        """Test max idle duration candles parameter affects penalty magnitude.

        Verifies:
        - penalty(max=50) < penalty(max=200) < 0
        - Smaller max → larger penalty magnitude
        """
        params_small = self.base_params(max_idle_duration_candles=50)
        params_large = self.base_params(max_idle_duration_candles=200)
        base_factor = PARAMS.BASE_FACTOR
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
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
            short_allowed=True,
            action_masking=True,
        )
        large = calculate_reward(
            context,
            params_large,
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
            short_allowed=True,
            action_masking=True,
        )
        self.assertLess(small.idle_penalty, 0.0)
        self.assertLess(large.idle_penalty, 0.0)
        self.assertGreater(large.idle_penalty, small.idle_penalty)

    @pytest.mark.smoke
    def test_exit_factor_calculation(self):
        """Exit factor calculation smoke test across attenuation modes.

        Non-owning smoke test; ownership: robustness/test_robustness.py:35

        Verifies:
        - Exit factors are finite and positive (linear, power modes)
        - Plateau mode attenuates after grace period
        """
        modes_to_test = ["linear", "power"]
        pnl = 0.02
        pnl_target = 0.045  # 0.03 * 1.5 coefficient
        context = self.make_ctx(
            pnl=pnl,
            trade_duration=50,
            idle_duration=0,
            max_unrealized_profit=0.045,
            min_unrealized_profit=0.0,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )

        for mode in modes_to_test:
            test_params = self.base_params(exit_attenuation_mode=mode)
            factor = _get_exit_factor(
                base_factor=1.0,
                pnl=pnl,
                pnl_target=pnl_target,
                duration_ratio=0.3,
                context=context,
                params=test_params,
                risk_reward_ratio=PARAMS.RISK_REWARD_RATIO_HIGH,
            )
            self.assertFinite(factor, name=f"exit_factor[{mode}]")
            self.assertGreater(factor, 0, f"Exit factor for {mode} should be positive")
        plateau_params = self.base_params(
            exit_attenuation_mode="linear",
            exit_plateau=True,
            exit_plateau_grace=0.5,
            exit_linear_slope=1.0,
        )
        assert_exit_factor_plateau_behavior(
            self,
            _get_exit_factor,
            base_factor=1.0,
            pnl=pnl,
            pnl_target=pnl_target,
            context=context,
            plateau_params=plateau_params,
            grace=0.5,
            tolerance_strict=TOLERANCE.IDENTITY_STRICT,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO_HIGH,
        )

    def test_idle_penalty_zero_when_pnl_target_zero(self):
        """Test idle penalty is zero when pnl_target is zero.

        Verifies:
        - pnl_target = 0 → idle_penalty = 0
        - Total reward is zero in this configuration
        """
        context = self.make_ctx(
            pnl=0.0,
            trade_duration=0,
            idle_duration=30,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )

        def validate_zero_penalty(test_case, breakdown, description, tolerance_relaxed):
            test_case.assertEqual(
                breakdown.idle_penalty, 0.0, "Idle penalty should be zero when profit_aim=0"
            )
            test_case.assertEqual(
                breakdown.total, 0.0, "Total reward should be zero in this configuration"
            )

        scenarios = [(context, self.DEFAULT_PARAMS, "pnl_target_zero")]
        config = RewardScenarioConfig(
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=0.0,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
            tolerance_relaxed=TOLERANCE.IDENTITY_RELAXED,
        )
        assert_reward_calculation_scenarios(
            self,
            scenarios,
            config,
            validate_zero_penalty,
        )

    def test_win_reward_factor_saturation(self):
        """Test PnL amplification factor saturates at asymptotic limit.

        Verifies:
        - Amplification ratio increases monotonically with PnL
        - Saturation approaches (1 + win_reward_factor)
        - Observed matches theoretical saturation behavior
        """
        win_reward_factor = 3.0
        beta = 0.5
        profit_aim = PARAMS.PROFIT_AIM
        params = self.base_params(
            win_reward_factor=win_reward_factor,
            pnl_factor_beta=beta,
            efficiency_weight=0.0,
            exit_attenuation_mode="linear",
            exit_plateau=False,
            exit_linear_slope=0.0,
        )
        params.pop("base_factor", None)
        pnl_values = [profit_aim * m for m in (1.05, PARAMS.RISK_REWARD_RATIO_HIGH, 5.0, 10.0)]
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
                profit_aim=profit_aim,
                risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
                short_allowed=True,
                action_masking=True,
            )
            ratio = br.exit_component / pnl if pnl != 0 else 0.0
            ratios_observed.append(float(ratio))
        self.assertMonotonic(
            ratios_observed,
            non_decreasing=True,
            tolerance=TOLERANCE.IDENTITY_STRICT,
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
            pnl_ratio = pnl / profit_aim
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
        """Test idle penalty fallback and proportional scaling behavior.

        Verifies:
        - max_idle_duration = None → use max_trade_duration as fallback
        - penalty(duration=40) ≈ 2 × penalty(duration=20)
        - Proportional scaling with idle duration
        """
        params = self.base_params(max_idle_duration_candles=None, max_trade_duration_candles=100)
        base_factor = PARAMS.BASE_FACTOR
        profit_aim = PARAMS.PROFIT_AIM
        risk_reward_ratio = 1.0

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

        results = []
        for context, description in contexts_and_descriptions:
            breakdown = calculate_reward(
                context,
                params,
                base_factor=base_factor,
                profit_aim=profit_aim,
                risk_reward_ratio=risk_reward_ratio,
                short_allowed=True,
                action_masking=True,
            )
            results.append((breakdown, context.idle_duration, description))

        br_a, br_b, br_mid = [r[0] for r in results]
        self.assertLess(br_a.idle_penalty, 0.0)
        self.assertLess(br_b.idle_penalty, 0.0)
        self.assertLess(br_mid.idle_penalty, 0.0)

        ratio = br_b.idle_penalty / br_a.idle_penalty if br_a.idle_penalty != 0 else None
        self.assertIsNotNone(ratio)
        if ratio is not None:
            self.assertAlmostEqualFloat(abs(ratio), 2.0, tolerance=0.2)

        idle_penalty_scale = _get_float_param(params, "idle_penalty_scale", 0.5)
        idle_penalty_power = _get_float_param(params, "idle_penalty_power", 1.025)
        base_factor = _get_float_param(params, "base_factor", float(base_factor))
        risk_reward_ratio = _get_float_param(params, "risk_reward_ratio", float(risk_reward_ratio))
        idle_factor = base_factor * (profit_aim / risk_reward_ratio) / 4.0
        observed_ratio = abs(br_mid.idle_penalty) / (idle_factor * idle_penalty_scale)
        if observed_ratio > 0:
            implied_max_idle_duration_candles = 120 / observed_ratio ** (1 / idle_penalty_power)
            self.assertAlmostEqualFloat(implied_max_idle_duration_candles, 400.0, tolerance=20.0)

    # Owns invariant: components-pbrs-breakdown-fields-119
    def test_pbrs_breakdown_fields_finite_and_aligned(self):
        """Test PBRS breakdown fields are finite and mathematically aligned.

        Verifies:
        - base_reward, pbrs_delta, invariance_correction are finite
        - reward_shaping = pbrs_delta + invariance_correction (within tolerance)
        - In canonical mode with no additives: invariance_correction ≈ 0
        """
        # Test with canonical PBRS (invariance_correction should be ~0)
        canonical_params = self.base_params(
            exit_potential_mode="canonical",
            entry_additive_enabled=False,
            exit_additive_enabled=False,
        )
        context = self.make_ctx(
            pnl=0.02,
            trade_duration=50,
            idle_duration=0,
            max_unrealized_profit=0.03,
            min_unrealized_profit=0.01,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        breakdown = calculate_reward(
            context,
            canonical_params,
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
            short_allowed=True,
            action_masking=True,
        )

        # Verify all PBRS fields are finite
        self.assertFinite(breakdown.base_reward, name="base_reward")
        self.assertFinite(breakdown.pbrs_delta, name="pbrs_delta")
        self.assertFinite(breakdown.invariance_correction, name="invariance_correction")

        # Verify mathematical alignment: reward_shaping = pbrs_delta + invariance_correction
        expected_shaping = breakdown.pbrs_delta + breakdown.invariance_correction
        self.assertAlmostEqualFloat(
            breakdown.reward_shaping,
            expected_shaping,
            tolerance=TOLERANCE.IDENTITY_STRICT,
            msg="reward_shaping should equal pbrs_delta + invariance_correction",
        )

        # In canonical mode with no additives, invariance_correction should be ~0
        self.assertAlmostEqualFloat(
            breakdown.invariance_correction,
            0.0,
            tolerance=TOLERANCE.IDENTITY_STRICT,
            msg="invariance_correction should be ~0 in canonical mode",
        )


if __name__ == "__main__":
    unittest.main()
