import math

import numpy as np

from reward_space_analysis import (
    Actions,
    Positions,
    RewardParams,
    _get_bool_param,
)

from ..constants import PARAMS
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


def test_calculate_reward_current_pnl_hold_path():
    """Verify current PnL drives liquidation value during a hold action.

    The reward calculation uses current_pnl for next_pnl. The extrema remain
    available to other diagnostics but do not replace current_pnl here.

    **Setup:**
    - Position: Long, Action: Neutral (hold)
    - Current PnL: 0.01
    - Extrema: max_unrealized_profit=0.02, min_unrealized_profit=-0.01
    - Parameters: hold_potential_enabled=True
    - Trade duration: 5 steps

    **Assertions:**
    - Reward and next liquidation values equal 1 + current_pnl
    - Both prev_potential and next_potential are finite
    - At least one potential is non-zero (shaping should activate)
    """
    # Exercise the fee-aware marked-to-liquidation hold path.
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
        "pnl_amplification_sensitivity": 0.5,
    }
    breakdown = calculate_reward_with_defaults(
        context,
        params,
        base_factor=100.0,
        profit_aim=0.05,
        risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
        prev_potential=np.nan,
    )
    assert math.isclose(breakdown.reward_liquidation_value, 1.01)
    assert math.isclose(breakdown.next_liquidation_value, 1.01)
    assert math.isfinite(breakdown.prev_potential)
    assert math.isfinite(breakdown.next_potential)
    # shaping should activate (non-zero or zero after potential difference)
    assert breakdown.prev_potential != 0.0 or breakdown.next_potential != 0.0
