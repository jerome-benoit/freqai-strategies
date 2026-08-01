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
    get_max_idle_duration_candles,
)

from ..constants import EFFICIENCY, PARAMS, SCENARIOS, TOLERANCE
from ..helpers import (
    RewardScenarioConfig,
    assert_exit_factor_plateau_behavior,
    assert_progressive_scaling_behavior,
    assert_reward_calculation_scenarios,
    calculate_reward_with_defaults,
)
from ..test_base import RewardSpaceTestBase

_DEFAULT_PNL_TARGET = PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO

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

    def test_hold_economic_log_delta_calculation(self):
        """A hold recognizes only the log change in liquidation value."""
        context = self.make_ctx(
            pnl=0.01,
            trade_duration=150,
            idle_duration=0,
            max_unrealized_profit=0.02,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Neutral,
            previous_liquidation_value=0.995,
        )
        breakdown = calculate_reward_with_defaults(context, self.DEFAULT_PARAMS)
        expected = float(self.DEFAULT_PARAMS["base_factor"]) * math.log(1.01 / 0.995)
        self.assertAlmostEqualFloat(
            breakdown.economic_component, expected, tolerance=TOLERANCE.IDENTITY_STRICT
        )
        self.assertEqual(breakdown.hold_penalty, 0.0)
        self.assertAlmostEqualFloat(breakdown.total, expected, tolerance=TOLERANCE.IDENTITY_STRICT)

    def test_hold_reward_is_independent_of_duration_threshold(self):
        """Duration thresholds do not alter the economic mark-to-liquidation reward."""
        rewards = []
        for trade_duration in (64, 127, 128, 129):
            with self.subTest(trade_duration=trade_duration):
                context = self.make_ctx(
                    pnl=0.0,
                    trade_duration=trade_duration,
                    idle_duration=0,
                    position=Positions.Long,
                    action=Actions.Neutral,
                )
                breakdown = calculate_reward_with_defaults(context, self.DEFAULT_PARAMS)
                rewards.append(breakdown.economic_component)
                self.assertEqual(breakdown.hold_penalty, 0.0)
        self.assertTrue(all(value == 0.0 for value in rewards))

    def test_hold_penalty_progressive_scaling(self):
        """Test hold penalty scales progressively with increasing duration.

        Verifies:
        - For d1 < d2 < d3: penalty(d1) >= penalty(d2) >= penalty(d3)
        - Progressive scaling beyond max_duration threshold
        """

        params = self.base_params(max_trade_duration_candles=PARAMS.TRADE_DURATION_MEDIUM)
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
            breakdown = calculate_reward_with_defaults(context, params)
            penalties.append(breakdown.hold_penalty)

        assert_progressive_scaling_behavior(self, penalties, durations, "Hold penalty")

    def test_neutral_self_loop_has_zero_economic_reward(self):
        """A neutral self-loop remains at liquidation value one, regardless of idling."""
        context = self.make_ctx(idle_duration=20)
        breakdown = calculate_reward_with_defaults(context, self.DEFAULT_PARAMS)
        self.assertEqual(breakdown.previous_liquidation_value, 1.0)
        self.assertEqual(breakdown.reward_liquidation_value, 1.0)
        self.assertEqual(breakdown.next_liquidation_value, 1.0)
        self.assertEqual(breakdown.economic_component, 0.0)
        self.assertEqual(breakdown.idle_penalty, 0.0)

    def test_pnl_target_coefficient_zero_pnl(self):
        """PnL target coefficient returns neutral value for zero PnL.

        Validates that zero realized profit/loss produces coefficient = 1.0,
        ensuring no amplification or attenuation of base exit factor.

        **Setup:**
        - PnL: 0.0 (breakeven)
        - pnl_target: profit_aim * risk_reward_ratio
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
        - pnl_target: 0.045 (profit_aim=0.03 * risk_reward_ratio=1.5)
        - Parameters: win_reward_factor=2.0, pnl_amplification_sensitivity=0.5

        **Assertions:**
        - Coefficient is finite
        - Coefficient > 1.0 (rewards exceeding target)
        """
        params = self.base_params(win_reward_factor=2.0, pnl_amplification_sensitivity=0.5)
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
        - pnl_target: 0.045 (profit_aim=0.03 * risk_reward_ratio=1.5)
        - Penalty threshold: pnl < -pnl_target = -0.045
        - Parameters: win_reward_factor=2.0, pnl_amplification_sensitivity=0.5

        **Assertions:**
        - Coefficient is finite
        - Coefficient > 1.0 (amplifies loss penalty)
        """
        params = self.base_params(win_reward_factor=2.0, pnl_amplification_sensitivity=0.5)
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

        coefficient = _compute_efficiency_coefficient(
            params, ctx, ctx.current_pnl, _DEFAULT_PNL_TARGET
        )

        self.assertFinite(coefficient, name="efficiency_coefficient")
        self.assertAlmostEqualFloat(coefficient, 1.0, tolerance=TOLERANCE.GENERIC_EQ)

    def test_efficiency_coefficient_profits_monotonic_with_exact_bounds(self):
        """Verify efficiency coefficient monotonicity for profitable trades.

        **Setup:**
        - efficiency_weight: EFFICIENCY.WEIGHT_DEFAULT (1.0)
        - efficiency_center: EFFICIENCY.CENTER_DEFAULT (0.5)
        - PnL range: EFFICIENCY.PNL_RANGE_PROFIT (6 test points)
        - Unrealized range: [0.0, EFFICIENCY.MAX_UNREALIZED_PROFIT]

        **Assertions:**
        - Strict monotonicity (non-decreasing) as exit quality improves
        - Exact coefficient values at bounds match formula: 1 + weight*(ratio - center)
        - Poor exit: coefficient < 1.0, Optimal exit: coefficient > 1.0
        """
        params = self.base_params(
            efficiency_weight=EFFICIENCY.WEIGHT_DEFAULT,
            efficiency_center=EFFICIENCY.CENTER_DEFAULT,
        )
        max_unrealized_profit = EFFICIENCY.MAX_UNREALIZED_PROFIT
        min_unrealized_profit = 0.0

        pnl_values = list(EFFICIENCY.PNL_RANGE_PROFIT)
        coefficients = []

        for pnl in pnl_values:
            ctx = self.make_ctx(
                pnl=pnl,
                trade_duration=EFFICIENCY.TRADE_DURATION_DEFAULT,
                max_unrealized_profit=max_unrealized_profit,
                min_unrealized_profit=min_unrealized_profit,
                position=Positions.Long,
                action=Actions.Long_exit,
            )
            coefficient = _compute_efficiency_coefficient(
                params, ctx, ctx.current_pnl, _DEFAULT_PNL_TARGET
            )
            self.assertFinite(coefficient, name=f"efficiency_coefficient[pnl={pnl}]")
            coefficients.append(coefficient)

        # Verify strict monotonicity
        self.assertMonotonic(
            coefficients,
            non_decreasing=True,
            tolerance=TOLERANCE.IDENTITY_STRICT,
            name="efficiency_coefficient_for_profits",
        )

        # Verify exact values at bounds using the formula
        # coefficient = 1.0 + weight · (ratio - center)
        # ratio = (pnl - min_pnl) / range_pnl
        range_pnl = max_unrealized_profit - min_unrealized_profit

        # Poor exit bound (first element)
        pnl_poor = pnl_values[0]
        expected_ratio_poor = (pnl_poor - min_unrealized_profit) / range_pnl
        expected_coef_poor = 1.0 + EFFICIENCY.WEIGHT_DEFAULT * (
            expected_ratio_poor - EFFICIENCY.CENTER_DEFAULT
        )
        self.assertAlmostEqualFloat(
            coefficients[0],
            expected_coef_poor,
            tolerance=TOLERANCE.GENERIC_EQ,
            msg=f"Poor exit coefficient {coefficients[0]:.4f} != expected {expected_coef_poor:.4f}",
        )
        self.assertLess(coefficients[0], 1.0, "Poor profit exit should have coefficient < 1.0")

        # Optimal exit bound (last element)
        pnl_optimal = pnl_values[-1]
        expected_ratio_optimal = (pnl_optimal - min_unrealized_profit) / range_pnl
        expected_coef_optimal = 1.0 + EFFICIENCY.WEIGHT_DEFAULT * (
            expected_ratio_optimal - EFFICIENCY.CENTER_DEFAULT
        )
        self.assertAlmostEqualFloat(
            coefficients[-1],
            expected_coef_optimal,
            tolerance=TOLERANCE.GENERIC_EQ,
            msg=f"Optimal exit coefficient {coefficients[-1]:.4f} != expected {expected_coef_optimal:.4f}",
        )
        self.assertGreater(
            coefficients[-1], 1.0, "Optimal profit exit should have coefficient > 1.0"
        )

    def test_efficiency_coefficient_losses_monotonic_with_exact_bounds(self):
        """Verify efficiency coefficient behavior for losing trades.

        **Setup:**
        - efficiency_weight: EFFICIENCY.WEIGHT_DEFAULT (1.0)
        - efficiency_center: EFFICIENCY.CENTER_DEFAULT (0.5)
        - PnL range: EFFICIENCY.PNL_RANGE_LOSS (7 test points, worst to best)
        - Unrealized range: [EFFICIENCY.MIN_UNREALIZED_PROFIT, 0.0]

        **Assertions:**
        - Coefficient DECREASES as exit quality improves (inverted formula)
        - Exact values at bounds match: 1 + weight*(center - ratio)
        - Reward (pnl * coef) is less negative for better exits
        """
        params = self.base_params(
            efficiency_weight=EFFICIENCY.WEIGHT_DEFAULT,
            efficiency_center=EFFICIENCY.CENTER_DEFAULT,
        )
        max_unrealized_profit = 0.0
        min_unrealized_profit = EFFICIENCY.MIN_UNREALIZED_PROFIT

        pnl_values = list(EFFICIENCY.PNL_RANGE_LOSS)
        coefficients = []
        rewards = []

        for pnl in pnl_values:
            ctx = self.make_ctx(
                pnl=pnl,
                trade_duration=EFFICIENCY.TRADE_DURATION_DEFAULT,
                max_unrealized_profit=max_unrealized_profit,
                min_unrealized_profit=min_unrealized_profit,
                position=Positions.Long,
                action=Actions.Long_exit,
            )
            coefficient = _compute_efficiency_coefficient(
                params, ctx, ctx.current_pnl, _DEFAULT_PNL_TARGET
            )
            self.assertFinite(coefficient, name=f"efficiency_coefficient[pnl={pnl}]")
            coefficients.append(coefficient)
            # Simplified reward calculation (ignoring other factors for this test)
            rewards.append(pnl * coefficient)

        # Verify coefficient DECREASES as exit quality improves (monotonically decreasing)
        self.assertMonotonic(
            coefficients,
            non_increasing=True,  # Decreasing for losses!
            tolerance=TOLERANCE.IDENTITY_STRICT,
            name="efficiency_coefficient_for_losses",
        )

        # Verify reward INCREASES (less negative) as exit quality improves
        self.assertMonotonic(
            rewards,
            non_decreasing=True,
            tolerance=TOLERANCE.IDENTITY_STRICT,
            name="exit_reward_for_losses",
        )

        # Verify exact values at bounds using the INVERTED formula for losses
        # coefficient = 1.0 + weight · (center - ratio)
        range_pnl = max_unrealized_profit - min_unrealized_profit

        # Worst exit bound (first element: largest loss)
        pnl_worst = pnl_values[0]
        ratio_worst = (pnl_worst - min_unrealized_profit) / range_pnl
        expected_coef_worst = 1.0 + EFFICIENCY.WEIGHT_DEFAULT * (
            EFFICIENCY.CENTER_DEFAULT - ratio_worst
        )
        self.assertAlmostEqualFloat(
            coefficients[0],
            expected_coef_worst,
            tolerance=TOLERANCE.GENERIC_EQ,
            msg=f"Worst loss coefficient {coefficients[0]:.4f} != expected {expected_coef_worst:.4f}",
        )
        self.assertGreater(
            coefficients[0],
            1.0,
            "Worst loss exit should have coefficient > 1.0 (amplifies penalty)",
        )

        # Optimal exit bound (last element: minimal loss)
        pnl_optimal = pnl_values[-1]
        ratio_optimal = (pnl_optimal - min_unrealized_profit) / range_pnl
        expected_coef_optimal = 1.0 + EFFICIENCY.WEIGHT_DEFAULT * (
            EFFICIENCY.CENTER_DEFAULT - ratio_optimal
        )
        self.assertAlmostEqualFloat(
            coefficients[-1],
            expected_coef_optimal,
            tolerance=TOLERANCE.GENERIC_EQ,
            msg=f"Minimal loss coefficient {coefficients[-1]:.4f} != expected {expected_coef_optimal:.4f}",
        )
        self.assertLess(
            coefficients[-1],
            1.0,
            "Minimal loss exit should have coefficient < 1.0 (attenuates penalty)",
        )

        # Verify the final reward semantics: better exit = less negative reward
        self.assertLess(rewards[0], rewards[-1], "Worst exit should have more negative reward")

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
        breakdown = calculate_reward_with_defaults(
            context, params, base_factor=1.0, profit_aim=0.03
        )
        self.assertLessEqual(
            breakdown.exit_component,
            0.0,
            "Exit component must not be positive when pnl < 0",
        )

    def test_max_idle_duration_does_not_change_economic_reward(self):
        """Sampling limits do not leak into the neutral economic reward."""
        params_small = self.base_params(max_idle_duration_candles=50)
        params_large = self.base_params(max_idle_duration_candles=200)
        context = self.make_ctx(
            pnl=0.0,
            trade_duration=0,
            idle_duration=40,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )
        small = calculate_reward_with_defaults(context, params_small)
        large = calculate_reward_with_defaults(context, params_large)
        self.assertEqual(small.idle_penalty, 0.0)
        self.assertEqual(large.idle_penalty, 0.0)
        self.assertEqual(small.economic_component, 0.0)
        self.assertEqual(large.economic_component, 0.0)

    @pytest.mark.smoke
    def test_exit_factor_calculation(self):
        """Exit factor calculation smoke test across attenuation modes.

        Non-owning smoke test; ownership: robustness/test_robustness.py:43

        Verifies:
        - Exit factors are finite and positive (linear, power modes)
        - Plateau mode attenuates after grace period
        """
        modes_to_test = ["linear", "power"]
        pnl = PARAMS.PNL_SMALL

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

    def test_base_factor_scales_log_return_linearly(self):
        """The sole economic scale is ``base_factor``."""
        context = self.make_ctx(
            pnl=0.08,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        params = self.base_params(win_reward_factor=1000.0)
        params.pop("base_factor", None)
        unit = calculate_reward_with_defaults(context, params, base_factor=1.0)
        for scale in (1.0, PARAMS.RISK_REWARD_RATIO, PARAMS.BASE_FACTOR):
            with self.subTest(scale=scale):
                scaled = calculate_reward_with_defaults(context, params, base_factor=scale)
                self.assertAlmostEqualFloat(
                    scaled.economic_component,
                    scale * unit.economic_component,
                    tolerance=TOLERANCE.IDENTITY_STRICT,
                )

    def test_idle_duration_fallback_is_sampling_only(self):
        """The derived idle limit remains available but does not shape reward."""
        max_trade_duration_candles = PARAMS.TRADE_DURATION_MEDIUM

        params = self.base_params(
            max_idle_duration_candles=None,
            max_trade_duration_candles=max_trade_duration_candles,
        )
        expected_max_idle_duration_candles = int(
            DEFAULT_IDLE_DURATION_MULTIPLIER * max_trade_duration_candles
        )
        self.assertEqual(
            get_max_idle_duration_candles(params),
            expected_max_idle_duration_candles,
            "Expected fallback max_idle_duration from max_trade_duration",
        )

        for idle_duration in (20, 40, 120):
            with self.subTest(idle_duration=idle_duration):
                breakdown = calculate_reward_with_defaults(
                    self.make_ctx(idle_duration=idle_duration), params
                )
                self.assertEqual(breakdown.idle_penalty, 0.0)
                self.assertEqual(breakdown.total, 0.0)

    # Owns invariant: components-economic-pbrs-breakdown-119
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
            pnl=PARAMS.PNL_SMALL,
            trade_duration=PARAMS.TRADE_DURATION_SHORT,
            idle_duration=0,
            max_unrealized_profit=0.03,
            min_unrealized_profit=0.01,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        breakdown = calculate_reward_with_defaults(context, canonical_params)

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

        br_ratio = calculate_reward_with_defaults(
            context, params_ratio, risk_reward_ratio=PARAMS.RISK_REWARD_RATIO
        )
        br_rr = calculate_reward_with_defaults(
            context, params_rr, risk_reward_ratio=PARAMS.RISK_REWARD_RATIO
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
