#!/usr/bin/env python3
"""Additive deterministic contribution tests moved from helpers/test_utilities.py.

Owns invariant: components-canonical-additives-092 (components category)
"""

import unittest

import pytest

from reward_space_analysis import compute_pbrs_components, validate_reward_parameters

from ..constants import PARAMS, TOLERANCE
from ..test_base import RewardSpaceTestBase

pytestmark = pytest.mark.components


class TestAdditivesDeterministicContribution(RewardSpaceTestBase):
    """Canonical PBRS rejects or suppresses non-potential additives."""

    def test_canonical_additives_rejected_or_suppressed(self):
        """Strict validation rejects additives and relaxed validation records suppression."""
        base = self.base_params(
            hold_potential_enabled=True,
            entry_additive_enabled=False,
            exit_additive_enabled=False,
            exit_potential_mode="canonical",
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
        )
        with_add = base.copy()
        with_add.update(
            {
                "entry_additive_enabled": True,
                "exit_additive_enabled": True,
                "entry_additive_ratio": PARAMS.ADDITIVE_RATIO_DEFAULT,
                "exit_additive_ratio": PARAMS.ADDITIVE_RATIO_DEFAULT,
                "entry_additive_gain": PARAMS.ADDITIVE_GAIN_DEFAULT,
                "exit_additive_gain": PARAMS.ADDITIVE_GAIN_DEFAULT,
            }
        )
        with self.assertRaisesRegex(ValueError, "canonical PBRS"):
            validate_reward_parameters(with_add, strict=True)
        sanitized, adjustments = validate_reward_parameters(with_add, strict=False)
        self.assertFalse(sanitized["entry_additive_enabled"])
        self.assertFalse(sanitized["exit_additive_enabled"])
        self.assertEqual(
            adjustments["entry_additive_enabled"]["reason"],
            "canonical_pbrs_suppresses_additives",
        )
        self.assertEqual(
            adjustments["exit_additive_enabled"]["reason"],
            "canonical_pbrs_suppresses_additives",
        )
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
        shaping, _next, pbrs_delta, entry_additive, exit_additive = compute_pbrs_components(
            params=sanitized,
            base_factor=PARAMS.BASE_FACTOR,
            prev_potential=0.0,
            **ctx,
        )
        self.assertAlmostEqualFloat(shaping, pbrs_delta, tolerance=TOLERANCE.IDENTITY_STRICT)
        self.assertEqual(entry_additive, 0.0)
        self.assertEqual(exit_additive, 0.0)


if __name__ == "__main__":
    unittest.main()
