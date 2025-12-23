#!/usr/bin/env python3
"""Additive deterministic contribution tests moved from helpers/test_utilities.py.

Owns invariant: report-additives-deterministic-092 (components category)
"""

import unittest

import pytest

from reward_space_analysis import compute_pbrs_components

from ..constants import PARAMS
from ..test_base import RewardSpaceTestBase

pytestmark = pytest.mark.components


class TestAdditivesDeterministicContribution(RewardSpaceTestBase):
    """Additives enabled increase total reward; shaping impact limited."""

    def test_additive_activation_deterministic_contribution(self):
        """Enabling additives increases total reward while limiting shaping impact.

        **Invariant:** report-additives-deterministic-092

        Validates that when entry/exit additives are enabled, the total reward
        increases deterministically, but the shaping component remains bounded.
        This ensures additives provide meaningful reward contribution without
        destabilizing PBRS shaping dynamics.

        **Setup:**
        - Base configuration: hold_potential enabled, additives disabled
        - Test configuration: entry_additive and exit_additive enabled
        - Additive parameters: scale=0.4, gain=1.0 for both entry/exit
        - Context: base_reward=0.05, pnl=0.01, duration_ratio=0.2

        **Assertions:**
        - Total reward with additives > total reward without additives
        - Shaping difference remains bounded: |s1 - s0| < 0.2
        - Both total and shaping rewards are finite

        **Tolerance rationale:**
        - Custom bound 0.2 for shaping delta: Additives should not cause
          large shifts in shaping component, which maintains PBRS properties
        """
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
        base_reward = 0.05
        ctx = {
            "current_pnl": 0.01,
            "pnl_target": PARAMS.PROFIT_AIM * PARAMS.RISK_REWARD_RATIO,
            "current_duration_ratio": 0.2,
            "next_pnl": 0.012,
            "next_duration_ratio": 0.25,
            "risk_reward_ratio": PARAMS.RISK_REWARD_RATIO,
            "is_entry": True,
            "is_exit": False,
        }
        s0, _n0, _pbrs0, _entry0, _exit0 = compute_pbrs_components(
            prev_potential=0.0, params=base, **ctx
        )
        t0 = base_reward + s0 + _entry0 + _exit0
        s1, _n1, _pbrs1, _entry1, _exit1 = compute_pbrs_components(
            prev_potential=0.0, params=with_add, **ctx
        )
        t1 = base_reward + s1 + _entry1 + _exit1
        self.assertFinite(t1)
        self.assertFinite(s1)
        self.assertLess(abs(s1 - s0), 0.2)
        self.assertGreater(t1 - t0, 0.0, "Total reward should increase with additives present")


if __name__ == "__main__":
    unittest.main()
