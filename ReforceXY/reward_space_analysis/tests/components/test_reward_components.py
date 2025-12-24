#!/usr/bin/env python3
"""Tests for reward calculation components and algorithms."""

import math
import unittest

import pytest

from reward_space_analysis import (
    DEFAULT_IDLE_DURATION_MULTIPLIER,
    Actions,
    Positions,
    _compute_efficiency_coefficient,
    _compute_hold_potential,
    _compute_pnl_target_coefficient,
    _get_exit_factor,
    _get_float_param,
    calculate_reward,
    get_max_idle_duration_candles,
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
            "hold_potential_ratio": 1.0,
            "hold_potential_gain": 1.0,
            "hold_potential_transform_pnl": "tanh",
            "hold_potential_transform_duration": "tanh",
        }
        val = _compute_hold_potential(
            0.5,
            PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO,
            0.3,
            PARAMS.RISK_REWARD_RATIO,
            params,
            PARAMS.BASE_FACTOR,
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

    def test_pnl_target_coefficient_zero_pnl(self):
        """PnL target coefficient returns neutral value for zero PnL.

        Validates that zero realized profit/loss produces coefficient = 1.0,
        ensuring no amplification or attenuation of base exit factor.

        **Setup:**
        - PnL: 0.0 (breakeven)
        - pnl_target: profit_aim × risk_reward_ratio
        - Parameters: default base_params

        **Assertions:**
        - Coefficient is finite
        - Coefficient equals 1.0 within TOLERANCE.GENERIC_EQ
        """
        params = self.base_params()
        pnl_target = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO

        coefficient = _compute_pnl_target_coefficient(
            params, pnl=0.0, pnl_target=pnl_target, risk_reward_ratio=PARAMS.RISK_REWARD_RATIO
        )

        self.assertFinite(coefficient, name="pnl_target_coefficient")
        self.assertAlmostEqualFloat(coefficient, 1.0, tolerance=TOLERANCE.GENERIC_EQ)

    def test_pnl_target_coefficient_exceeds_target(self):
        """PnL target coefficient rewards exits that exceed profit target.

        Validates amplification behavior when realized PnL exceeds the target,
        incentivizing the agent to achieve higher profits than baseline.

        **Setup:**
        - PnL: 150% of pnl_target (exceeds target by 50%)
        - pnl_target: 0.045 (profit_aim=0.03 × risk_reward_ratio=1.5)
        - Parameters: win_reward_factor=2.0, pnl_factor_beta=0.5

        **Assertions:**
        - Coefficient is finite
        - Coefficient > 1.0 (rewards exceeding target)
        """
        params = self.base_params(win_reward_factor=2.0, pnl_factor_beta=0.5)
        profit_aim = 0.03
        risk_reward_ratio = 1.5
        pnl_target = profit_aim * risk_reward_ratio
        pnl = pnl_target * 1.5  # 50% above target

        coefficient = _compute_pnl_target_coefficient(
            params, pnl=pnl, pnl_target=pnl_target, risk_reward_ratio=risk_reward_ratio
        )

        self.assertFinite(coefficient, name="pnl_target_coefficient")
        self.assertGreater(
            coefficient, 1.0, "PnL exceeding target should reward with coefficient > 1.0"
        )

    def test_pnl_target_coefficient_below_loss_threshold(self):
        """PnL target coefficient amplifies penalty for excessive losses.

        Validates that losses exceeding risk-adjusted threshold produce
        coefficient > 1.0 to amplify negative reward signal. Penalty applies
        when BOTH conditions met: abs(pnl_ratio) > 1.0 AND pnl_ratio < -(1/rr).

        **Setup:**
        - PnL: -0.06 (exceeds pnl_target magnitude)
        - pnl_target: 0.045 (profit_aim=0.03 × risk_reward_ratio=1.5)
        - Penalty threshold: pnl < -pnl_target = -0.045
        - Parameters: win_reward_factor=2.0, pnl_factor_beta=0.5

        **Assertions:**
        - Coefficient is finite
        - Coefficient > 1.0 (amplifies loss penalty)
        """
        params = self.base_params(win_reward_factor=2.0, pnl_factor_beta=0.5)
        profit_aim = 0.03
        risk_reward_ratio = 1.5
        pnl_target = profit_aim * risk_reward_ratio  # 0.045
        # Need abs(pnl / pnl_target) > 1.0 AND pnl / pnl_target < -1/1.5
        # So pnl < -0.045 (exceeds pnl_target in magnitude)
        pnl = -0.06  # Much more negative than pnl_target

        coefficient = _compute_pnl_target_coefficient(
            params, pnl=pnl, pnl_target=pnl_target, risk_reward_ratio=risk_reward_ratio
        )

        self.assertFinite(coefficient, name="pnl_target_coefficient")
        self.assertGreater(
            coefficient, 1.0, "Excessive loss should amplify penalty with coefficient > 1.0"
        )

    def test_efficiency_coefficient_zero_weight(self):
        """Efficiency coefficient returns neutral value when efficiency disabled.

        Validates that efficiency_weight=0 disables exit timing efficiency
        adjustments, returning coefficient = 1.0 regardless of exit position
        relative to unrealized PnL extremes.

        **Setup:**
        - efficiency_weight: 0.0 (disabled)
        - PnL: 0.02 (between min=-0.01 and max=0.03)
        - Trade context: Long position with unrealized range

        **Assertions:**
        - Coefficient is finite
        - Coefficient equals 1.0 within TOLERANCE.GENERIC_EQ
        """
        params = self.base_params(efficiency_weight=0.0)
        ctx = self.make_ctx(
            pnl=0.02,
            trade_duration=10,
            max_unrealized_profit=0.03,
            min_unrealized_profit=-0.01,
            position=Positions.Long,
            action=Actions.Long_exit,
        )

        coefficient = _compute_efficiency_coefficient(params, ctx, ctx.pnl)

        self.assertFinite(coefficient, name="efficiency_coefficient")
        self.assertAlmostEqualFloat(coefficient, 1.0, tolerance=TOLERANCE.GENERIC_EQ)

    def test_efficiency_coefficient_optimal_profit_exit(self):
        """Efficiency coefficient rewards exits near peak unrealized profit.

        Validates that exiting close to maximum unrealized profit produces
        coefficient > 1.0, incentivizing optimal exit timing for profitable trades.

        **Setup:**
        - PnL: 0.029 (very close to max_unrealized_profit=0.03)
        - Efficiency ratio: (0.029 - 0.0) / (0.03 - 0.0) ≈ 0.967 (high)
        - efficiency_weight: 1.0, efficiency_center: 0.5
        - Trade context: Long position exiting near peak

        **Assertions:**
        - Coefficient is finite
        - Coefficient > 1.0 (rewards optimal timing)
        """
        params = self.base_params(efficiency_weight=1.0, efficiency_center=0.5)
        ctx = self.make_ctx(
            pnl=0.029,  # Close to max
            trade_duration=10,
            max_unrealized_profit=0.03,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Long_exit,
        )

        coefficient = _compute_efficiency_coefficient(params, ctx, ctx.pnl)

        self.assertFinite(coefficient, name="efficiency_coefficient")
        self.assertGreater(
            coefficient, 1.0, "Exit near max profit should reward with coefficient > 1.0"
        )

    def test_efficiency_coefficient_poor_profit_exit(self):
        """Efficiency coefficient penalizes exits far from peak unrealized profit.

        Validates that exiting far below maximum unrealized profit produces
        coefficient < 1.0, penalizing poor exit timing that leaves profit on the table.

        **Setup:**
        - PnL: 0.005 (far from max_unrealized_profit=0.03)
        - Efficiency ratio: (0.005 - 0.0) / (0.03 - 0.0) ≈ 0.167 (low)
        - efficiency_weight: 1.0, efficiency_center: 0.5
        - Trade context: Long position exiting prematurely

        **Assertions:**
        - Coefficient is finite
        - Coefficient < 1.0 (penalizes suboptimal timing)
        """
        params = self.base_params(efficiency_weight=1.0, efficiency_center=0.5)
        ctx = self.make_ctx(
            pnl=0.005,  # Far from max
            trade_duration=10,
            max_unrealized_profit=0.03,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Long_exit,
        )

        coefficient = _compute_efficiency_coefficient(params, ctx, ctx.pnl)

        self.assertFinite(coefficient, name="efficiency_coefficient")
        self.assertLess(
            coefficient, 1.0, "Exit far from max profit should penalize with coefficient < 1.0"
        )

    def test_efficiency_coefficient_optimal_loss_exit(self):
        """Efficiency coefficient rewards loss exits near minimum unrealized loss.

        Validates that exiting close to minimum unrealized loss produces
        coefficient > 1.0, rewarding quick loss-cutting behavior for losing trades.

        **Setup:**
        - PnL: -0.005 (very close to min_unrealized_profit=-0.006)
        - Efficiency ratio: (-0.005 - (-0.006)) / (0.0 - (-0.006)) ≈ 0.167 (low)
        - For losses: coefficient = 1 + weight × (center - ratio) → rewards low ratio
        - efficiency_weight: 1.0, efficiency_center: 0.5
        - Trade context: Long position cutting losses quickly

        **Assertions:**
        - Coefficient is finite
        - Coefficient > 1.0 (rewards optimal loss exit)
        """
        params = self.base_params(efficiency_weight=1.0, efficiency_center=0.5)
        ctx = self.make_ctx(
            pnl=-0.005,  # Close to min loss
            trade_duration=10,
            max_unrealized_profit=0.0,
            min_unrealized_profit=-0.006,
            position=Positions.Long,
            action=Actions.Long_exit,
        )

        coefficient = _compute_efficiency_coefficient(params, ctx, ctx.pnl)

        self.assertFinite(coefficient, name="efficiency_coefficient")
        self.assertGreater(
            coefficient, 1.0, "Exit near min loss should reward with coefficient > 1.0"
        )

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
        base_factor = PARAMS.BASE_FACTOR
        profit_aim = PARAMS.PROFIT_AIM
        risk_reward_ratio = 1.0
        max_trade_duration_candles = 100
        params = self.base_params(
            max_idle_duration_candles=None,
            max_trade_duration_candles=max_trade_duration_candles,
            base_factor=base_factor,
        )
        expected_max_idle_duration_candles = int(
            DEFAULT_IDLE_DURATION_MULTIPLIER * max_trade_duration_candles
        )
        self.assertEqual(
            get_max_idle_duration_candles(params),
            expected_max_idle_duration_candles,
            "Expected fallback max_idle_duration from max_trade_duration",
        )

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

        idle_penalty_scale = _get_float_param(params, "idle_penalty_scale", 1.0)
        idle_penalty_power = _get_float_param(params, "idle_penalty_power", 1.025)
        idle_factor = base_factor * (profit_aim / risk_reward_ratio)
        observed_ratio = abs(br_mid.idle_penalty) / (idle_factor * idle_penalty_scale)
        if observed_ratio > 0:
            implied_max_idle_duration_candles = 120 / observed_ratio ** (1 / idle_penalty_power)
            tolerance = 0.05 * expected_max_idle_duration_candles
            self.assertAlmostEqualFloat(
                implied_max_idle_duration_candles,
                float(expected_max_idle_duration_candles),
                tolerance=tolerance,
            )

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

    def test_rr_alias_matches_risk_reward_ratio(self):
        """`rr` param alias matches `risk_reward_ratio` runtime naming."""
        context = self.make_ctx(
            pnl=0.02,
            trade_duration=40,
            idle_duration=0,
            max_unrealized_profit=0.03,
            min_unrealized_profit=0.01,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        rr_value = 1.75

        # Canonical spelling
        params_ratio = self.base_params(
            exit_potential_mode="canonical",
            risk_reward_ratio=rr_value,
        )
        params_ratio.pop("rr", None)

        # Runtime spelling
        params_rr = self.base_params(
            exit_potential_mode="canonical",
            rr=rr_value,
        )
        params_rr.pop("risk_reward_ratio", None)

        br_ratio = calculate_reward(
            context,
            params_ratio,
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=1.0,
            short_allowed=True,
            action_masking=True,
        )
        br_rr = calculate_reward(
            context,
            params_rr,
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=1.0,
            short_allowed=True,
            action_masking=True,
        )

        self.assertAlmostEqualFloat(
            br_rr.total,
            br_ratio.total,
            tolerance=TOLERANCE.IDENTITY_STRICT,
            msg="Total reward should match when using rr alias",
        )
        self.assertAlmostEqualFloat(
            br_rr.exit_component,
            br_ratio.exit_component,
            tolerance=TOLERANCE.IDENTITY_STRICT,
            msg="Exit component should match when using rr alias",
        )


if __name__ == "__main__":
    unittest.main()
