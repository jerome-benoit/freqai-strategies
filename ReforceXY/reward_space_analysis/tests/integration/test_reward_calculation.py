"""Integration reward calculation smoke tests after deduplication.

This module retains only high-level integration smoke verifying component activation
and basic long/short symmetry. Detailed invariant assertions live in the
components/ and robustness/ suites per TEST_COVERAGE_MAP taxonomy.
"""

from reward_space_analysis import (
    Actions,
    Positions,
    calculate_reward,
)

from ..test_base import RewardSpaceTestBase


class TestRewardCalculation(RewardSpaceTestBase):
    """High-level integration smoke tests for reward calculation."""

    def test_reward_component_activation_smoke(self):  # Smoke-only after refactor
        """Smoke: each primary component activates in a representative scenario.

        Detailed progressive / boundary / proportional invariants are NOT asserted here.
        We only check sign / non-zero activation plus total decomposition identity.
        """
        scenarios = [
            (
                "hold_penalty_active",
                dict(
                    pnl=0.0,
                    trade_duration=160,  # > default threshold
                    idle_duration=0,
                    max_unrealized_profit=0.02,
                    min_unrealized_profit=-0.01,
                    position=Positions.Long,
                    action=Actions.Neutral,
                ),
                "hold_penalty",
            ),
            (
                "idle_penalty_active",
                dict(
                    pnl=0.0,
                    trade_duration=0,
                    idle_duration=25,
                    max_unrealized_profit=0.0,
                    min_unrealized_profit=0.0,
                    position=Positions.Neutral,
                    action=Actions.Neutral,
                ),
                "idle_penalty",
            ),
            (
                "profitable_exit_long",
                dict(
                    pnl=0.04,
                    trade_duration=40,
                    idle_duration=0,
                    max_unrealized_profit=0.05,
                    min_unrealized_profit=0.0,
                    position=Positions.Long,
                    action=Actions.Long_exit,
                ),
                "exit_component",
            ),
            (
                "invalid_action_penalty",
                dict(
                    pnl=0.01,
                    trade_duration=10,
                    idle_duration=0,
                    max_unrealized_profit=0.02,
                    min_unrealized_profit=0.0,
                    position=Positions.Short,
                    action=Actions.Long_exit,  # invalid pairing
                ),
                "invalid_penalty",
            ),
        ]

        for name, ctx_kwargs, expected_component in scenarios:
            with self.subTest(scenario=name):
                ctx = self.make_ctx(**ctx_kwargs)
                breakdown = calculate_reward(
                    ctx,
                    self.DEFAULT_PARAMS,
                    base_factor=self.TEST_BASE_FACTOR,
                    profit_target=self.TEST_PROFIT_TARGET,
                    risk_reward_ratio=self.TEST_RR,
                    short_allowed=True,
                    action_masking=expected_component != "invalid_penalty",
                )

                value = getattr(breakdown, expected_component)
                # Sign / activation expectations
                if expected_component in {"hold_penalty", "idle_penalty", "invalid_penalty"}:
                    self.assertLess(value, 0.0, f"{expected_component} should be negative: {name}")
                elif expected_component == "exit_component":
                    self.assertGreater(value, 0.0, f"exit_component should be positive: {name}")

                # Decomposition identity (relaxed tolerance)
                comp_sum = (
                    breakdown.exit_component
                    + breakdown.idle_penalty
                    + breakdown.hold_penalty
                    + breakdown.invalid_penalty
                    + breakdown.reward_shaping
                    + breakdown.entry_additive
                    + breakdown.exit_additive
                )
                self.assertAlmostEqualFloat(
                    breakdown.total,
                    comp_sum,
                    tolerance=self.TOL_IDENTITY_RELAXED,
                    msg=f"Total != sum components in {name}"
                )

    def test_long_short_symmetry_smoke(self):
        """Smoke: exit component sign & approximate magnitude symmetry for long vs short.

        Strict magnitude precision is tested in robustness suite; here we assert coarse symmetry.
        """
        params = self.base_params()
        params.pop("base_factor", None)
        base_factor = 100.0
        profit_target = 0.04
        rr = self.TEST_RR

        for pnl, label in [(0.02, "profit"), (-0.02, "loss")]:
            with self.subTest(pnl=pnl, label=label):
                ctx_long = self.make_ctx(
                    pnl=pnl,
                    trade_duration=50,
                    idle_duration=0,
                    max_unrealized_profit=abs(pnl) + 0.005,
                    min_unrealized_profit=0.0 if pnl > 0 else pnl,
                    position=Positions.Long,
                    action=Actions.Long_exit,
                )
                ctx_short = self.make_ctx(
                    pnl=pnl,
                    trade_duration=50,
                    idle_duration=0,
                    max_unrealized_profit=abs(pnl) + 0.005 if pnl > 0 else 0.01,
                    min_unrealized_profit=0.0 if pnl > 0 else pnl,
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

                if pnl > 0:
                    self.assertGreater(br_long.exit_component, 0.0)
                    self.assertGreater(br_short.exit_component, 0.0)
                else:
                    self.assertLess(br_long.exit_component, 0.0)
                    self.assertLess(br_short.exit_component, 0.0)

                # Coarse symmetry: relative diff below relaxed tolerance
                rel_diff = abs(abs(br_long.exit_component) - abs(br_short.exit_component)) / max(
                    1e-12, abs(br_long.exit_component)
                )
                self.assertLess(rel_diff, 0.25, f"Excessive asymmetry ({rel_diff:.3f}) for {label}")
