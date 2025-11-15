#!/usr/bin/env python3
"""Additive deterministic contribution tests moved from helpers/test_utilities.py.

Owns invariant: report-additives-deterministic-092 (components category)
"""

import unittest

import pytest

from reward_space_analysis import apply_potential_shaping

from ..test_base import RewardSpaceTestBase

pytestmark = pytest.mark.components


class TestAdditivesDeterministicContribution(RewardSpaceTestBase):
    """Additives enabled increase total reward; shaping impact limited."""

    def test_additive_activation_deterministic_contribution(self):
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
        ctx = {
            "base_reward": 0.05,
            "current_pnl": 0.01,
            "current_duration_ratio": 0.2,
            "next_pnl": 0.012,
            "next_duration_ratio": 0.25,
            "is_entry": True,
            "is_exit": False,
        }
        _t0, s0, _n0, _pbrs0, _entry0, _exit0 = apply_potential_shaping(
            last_potential=0.0, params=base, **ctx
        )
        t1, s1, _n1, _pbrs1, _entry1, _exit1 = apply_potential_shaping(
            last_potential=0.0, params=with_add, **ctx
        )
        self.assertFinite(t1)
        self.assertFinite(s1)
        self.assertLess(abs(s1 - s0), 0.2)
        self.assertGreater(t1 - _t0, 0.0, "Total reward should increase with additives present")


if __name__ == "__main__":
    unittest.main()
