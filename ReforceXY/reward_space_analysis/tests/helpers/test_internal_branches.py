import math

import numpy as np

from reward_space_analysis import (
    Actions,
    Positions,
    RewardContext,
    _get_bool_param,
    _get_float_param,
    calculate_reward,
)


def test_get_bool_param_none_and_invalid_literal():
    params_none = {"check_invariants": None}
    # None should coerce to False (coverage for _to_bool None path)
    assert _get_bool_param(params_none, "check_invariants", True) is False

    params_invalid = {"check_invariants": "not_a_bool"}
    # Invalid literal triggers ValueError in _to_bool; fallback returns default (True)
    assert _get_bool_param(params_invalid, "check_invariants", True) is True


def test_get_float_param_invalid_string_returns_nan():
    params = {"idle_penalty_scale": "abc"}
    val = _get_float_param(params, "idle_penalty_scale", 0.5)
    assert math.isnan(val)


def test_calculate_reward_unrealized_pnl_hold_path():
    # Exercise unrealized_pnl branch during hold to cover next_pnl tanh path
    context = RewardContext(
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
    breakdown = calculate_reward(
        context,
        params,
        base_factor=100.0,
        profit_aim=0.05,
        risk_reward_ratio=1.0,
        short_allowed=True,
        action_masking=True,
        previous_potential=np.nan,
    )
    assert math.isfinite(breakdown.prev_potential)
    assert math.isfinite(breakdown.next_potential)
    # shaping should activate (non-zero or zero after potential difference)
    assert breakdown.prev_potential != 0.0 or breakdown.next_potential != 0.0
