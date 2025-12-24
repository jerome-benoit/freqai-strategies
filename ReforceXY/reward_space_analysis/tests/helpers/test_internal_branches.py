import math

import numpy as np

from reward_space_analysis import (
    Actions,
    Positions,
    RewardParams,
    _get_bool_param,
)

from ..test_base import make_ctx
from . import calculate_reward_with_defaults


def test_get_bool_param_none_and_invalid_literal():
    """Verify _get_bool_param handles None and invalid literals correctly.

    Tests edge case handling in boolean parameter parsing:
    - None values should coerce to False
    - Invalid string literals should trigger fallback to default value

    **Setup:**
    - Test cases: None value, invalid literal "not_a_bool"
    - Default value: True

    **Assertions:**
    - None coerces to False (covers _to_bool None path)
    - Invalid literal returns default (ValueError fallback path)
    """
    params_none: RewardParams = {"check_invariants": None}
    # None should coerce to False (coverage for _to_bool None path)
    assert _get_bool_param(params_none, "check_invariants", True) is False

    params_invalid: RewardParams = {"check_invariants": "not_a_bool"}
    # Invalid literal triggers ValueError in _to_bool; fallback returns default (True)
    assert _get_bool_param(params_invalid, "check_invariants", True) is True


def test_calculate_reward_unrealized_pnl_hold_path():
    """Verify unrealized PnL branch activates during hold action.

    Tests that when hold_potential_enabled and unrealized_pnl are both True,
    the reward calculation uses max/min unrealized profit to compute next_pnl
    via the tanh transformation path.

    **Setup:**
    - Position: Long, Action: Neutral (hold)
    - PnL: 0.01, max_unrealized_profit: 0.02, min_unrealized_profit: -0.01
    - Parameters: hold_potential_enabled=True, unrealized_pnl=True
    - Trade duration: 5 steps

    **Assertions:**
    - Both prev_potential and next_potential are finite
    - At least one potential is non-zero (shaping should activate)
    """
    # Exercise unrealized_pnl branch during hold to cover next_pnl tanh path
    context = make_ctx(
        pnl=0.01,
        trade_duration=5,
        idle_duration=0,
        max_unrealized_profit=0.02,
        min_unrealized_profit=-0.01,
        position=Positions.Long,
        action=Actions.Neutral,
    )
    params = {
        "hold_potential_enabled": True,
        "unrealized_pnl": True,
        "pnl_factor_beta": 0.5,
    }
    breakdown = calculate_reward_with_defaults(
        context,
        params,
        base_factor=100.0,
        profit_aim=0.05,
        risk_reward_ratio=1.0,
        prev_potential=np.nan,
    )
    assert math.isfinite(breakdown.prev_potential)
    assert math.isfinite(breakdown.next_potential)
    # shaping should activate (non-zero or zero after potential difference)
    assert breakdown.prev_potential != 0.0 or breakdown.next_potential != 0.0
