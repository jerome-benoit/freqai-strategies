import math
import unittest

import pytest

from reward_space_analysis import (
    DEFAULT_MODEL_REWARD_PARAMETERS,
    Actions,
    Positions,
    RewardContext,
    RewardDiagnosticsWarning,
    RewardParams,
    _get_exit_factor,
    _hold_penalty,
    validate_reward_parameters,
)

from ..constants import PARAMS
from ..helpers import (
    assert_exit_factor_invariant_suite,
    run_relaxed_validation_adjustment_cases,
    run_strict_validation_failure_cases,
)
from ..test_base import make_ctx

_raw_max_trade_duration = DEFAULT_MODEL_REWARD_PARAMETERS.get("max_trade_duration_candles")
DEFAULT_MAX_TRADE_DURATION_CANDLES = (
    float(_raw_max_trade_duration) if isinstance(_raw_max_trade_duration, (int, float)) else 128.0
)


class _PyTestAdapter(unittest.TestCase):
    """Adapter leveraging unittest.TestCase for assertion + subTest support.

    Subclassing TestCase provides all assertion helpers and the subTest context manager
    required by shared helpers in tests.helpers.
    """

    def runTest(self):
        # Required abstract method; no-op for adapter usage.
        pass


@pytest.mark.robustness
def test_validate_reward_parameters_strict_failure_batch():
    """Batch strict validation failure scenarios using shared helper."""
    adapter = _PyTestAdapter()
    failure_params = [
        {"exit_linear_slope": "not_a_number"},
        {"exit_power_tau": 0.0},
        {"exit_power_tau": 1.5},
        {"exit_half_life": 0.0},
        {"exit_half_life": float("nan")},
    ]
    run_strict_validation_failure_cases(adapter, failure_params, validate_reward_parameters)


@pytest.mark.robustness
def test_validate_reward_parameters_relaxed_adjustment_batch():
    """Batch relaxed validation adjustment scenarios using shared helper."""
    relaxed_cases = [
        ({"exit_linear_slope": "not_a_number", "strict_validation": False}, ["non_numeric_reset"]),
        ({"exit_power_tau": float("inf"), "strict_validation": False}, ["non_numeric_reset"]),
        ({"max_idle_duration_candles": "bad", "strict_validation": False}, ["derived_default"]),
    ]
    run_relaxed_validation_adjustment_cases(
        _PyTestAdapter(), relaxed_cases, validate_reward_parameters
    )


@pytest.mark.robustness
def test_get_exit_factor_negative_plateau_grace_warning():
    """Verify negative exit_plateau_grace triggers warning but returns valid factor.

    **Setup:**
    - Attenuation mode: linear with plateau
    - exit_plateau_grace: -1.0 (invalid, should be non-negative)
    - Duration ratio: 0.5

    **Assertions:**
    - Warning emitted (RewardDiagnosticsWarning)
    - Factor is non-negative despite invalid parameter
    """
    params: RewardParams = {
        "exit_attenuation_mode": "linear",
        "exit_plateau": True,
        "exit_plateau_grace": -1.0,
    }
    pnl = 0.01
    pnl_target = 0.03
    context = make_ctx(
        pnl=pnl,
        trade_duration=50,
        idle_duration=0,
        max_unrealized_profit=0.02,
        min_unrealized_profit=0.0,
        position=Positions.Neutral,
        action=Actions.Neutral,
    )
    with pytest.warns(RewardDiagnosticsWarning):
        factor = _get_exit_factor(
            base_factor=10.0,
            pnl=pnl,
            pnl_target=pnl_target,
            duration_ratio=0.5,
            context=context,
            params=params,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
        )
    assert factor >= 0.0


@pytest.mark.robustness
def test_get_exit_factor_negative_linear_slope_warning():
    """Verify negative exit_linear_slope triggers warning but returns valid factor.

    **Setup:**
    - Attenuation mode: linear
    - exit_linear_slope: -5.0 (invalid, should be non-negative)
    - Duration ratio: 2.0

    **Assertions:**
    - Warning emitted (RewardDiagnosticsWarning)
    - Factor is non-negative despite invalid parameter
    """
    params: RewardParams = {"exit_attenuation_mode": "linear", "exit_linear_slope": -5.0}
    pnl = 0.01
    pnl_target = 0.03
    context = make_ctx(
        pnl=pnl,
        trade_duration=50,
        idle_duration=0,
        max_unrealized_profit=0.02,
        min_unrealized_profit=0.0,
        position=Positions.Neutral,
        action=Actions.Neutral,
    )
    with pytest.warns(RewardDiagnosticsWarning):
        factor = _get_exit_factor(
            base_factor=10.0,
            pnl=pnl,
            pnl_target=pnl_target,
            duration_ratio=2.0,
            context=context,
            params=params,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
        )
    assert factor >= 0.0


@pytest.mark.robustness
def test_get_exit_factor_invalid_power_tau_relaxed():
    """Verify invalid exit_power_tau (0.0) triggers warning in relaxed mode.

    **Setup:**
    - Attenuation mode: power
    - exit_power_tau: 0.0 (invalid, should be positive)
    - strict_validation: False (relaxed mode)
    - Duration ratio: 1.5

    **Assertions:**
    - Warning emitted (RewardDiagnosticsWarning)
    - Factor is positive (fallback to default tau)
    """
    params: RewardParams = {
        "exit_attenuation_mode": "power",
        "exit_power_tau": 0.0,
        "strict_validation": False,
    }
    pnl = 0.02
    pnl_target = 0.03
    context = make_ctx(
        pnl=pnl,
        trade_duration=50,
        idle_duration=0,
        max_unrealized_profit=0.03,
        min_unrealized_profit=0.0,
        position=Positions.Neutral,
        action=Actions.Neutral,
    )
    with pytest.warns(RewardDiagnosticsWarning):
        factor = _get_exit_factor(
            base_factor=5.0,
            pnl=pnl,
            pnl_target=pnl_target,
            duration_ratio=1.5,
            context=context,
            params=params,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
        )
    assert factor > 0.0


@pytest.mark.robustness
def test_get_exit_factor_half_life_near_zero_relaxed():
    """Verify near-zero exit_half_life triggers warning in relaxed mode.

    **Setup:**
    - Attenuation mode: half_life
    - exit_half_life: 1e-12 (near zero, impractical)
    - strict_validation: False (relaxed mode)
    - Duration ratio: 2.0

    **Assertions:**
    - Warning emitted (RewardDiagnosticsWarning)
    - Factor is non-zero (fallback to sensible value)
    """
    params: RewardParams = {
        "exit_attenuation_mode": "half_life",
        "exit_half_life": 1e-12,
        "strict_validation": False,
    }
    pnl = 0.02
    pnl_target = 0.03
    context = make_ctx(
        pnl=pnl,
        trade_duration=50,
        idle_duration=0,
        max_unrealized_profit=0.03,
        min_unrealized_profit=0.0,
        position=Positions.Neutral,
        action=Actions.Neutral,
    )
    with pytest.warns(RewardDiagnosticsWarning):
        factor = _get_exit_factor(
            base_factor=5.0,
            pnl=pnl,
            pnl_target=pnl_target,
            duration_ratio=2.0,
            context=context,
            params=params,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
        )
    assert factor != 0.0


@pytest.mark.robustness
def test_hold_penalty_short_duration_returns_zero():
    """Verify hold penalty is zero when trade_duration is below max threshold.

    **Setup:**
    - Trade duration: 1 candle (short)
    - Max trade duration: DEFAULT_MAX_TRADE_DURATION_CANDLES
    - Position: Long, Action: Neutral (hold)

    **Assertions:**
    - Penalty equals 0.0 (no penalty for short duration holds)
    """
    context = make_ctx(
        pnl=0.0,
        trade_duration=1,  # shorter than default max trade duration
        idle_duration=0,
        max_unrealized_profit=0.0,
        min_unrealized_profit=0.0,
        position=Positions.Long,
        action=Actions.Neutral,
    )
    params: RewardParams = {"max_trade_duration_candles": DEFAULT_MAX_TRADE_DURATION_CANDLES}
    penalty = _hold_penalty(context, hold_factor=1.0, params=params)
    assert penalty == 0.0


@pytest.mark.robustness
def test_exit_factor_invariant_suite_grouped():
    """Grouped exit factor invariant scenarios using shared helper."""

    def make_context(pnl: float) -> RewardContext:
        """Helper to create context for test cases."""
        max_profit = 0.03
        if isinstance(pnl, float) and math.isfinite(pnl):
            max_profit = max(pnl * 1.2, 0.03)
        return make_ctx(
            pnl=pnl,
            trade_duration=50,
            idle_duration=0,
            max_unrealized_profit=max_profit,
            min_unrealized_profit=0.0,
            position=Positions.Neutral,
            action=Actions.Neutral,
        )

    pnl_target = 0.03

    suite = [
        {
            "base_factor": 15.0,
            "pnl": 0.02,
            "pnl_target": pnl_target,
            "context": make_context(0.02),
            "duration_ratio": -5.0,
            "params": {
                "exit_attenuation_mode": "linear",
                "exit_linear_slope": 1.2,
                "exit_plateau": False,
            },
            "expectation": "non_negative",
        },
        {
            "base_factor": 15.0,
            "pnl": 0.02,
            "pnl_target": pnl_target,
            "context": make_context(0.02),
            "duration_ratio": 0.0,
            "params": {
                "exit_attenuation_mode": "linear",
                "exit_linear_slope": 1.2,
                "exit_plateau": False,
            },
            "expectation": "non_negative",
        },
        {
            "base_factor": float("nan"),
            "pnl": 0.01,
            "pnl_target": pnl_target,
            "context": make_context(0.01),
            "duration_ratio": 0.2,
            "params": {"exit_attenuation_mode": "linear", "exit_linear_slope": 0.5},
            "expectation": "safe_zero",
        },
        {
            "base_factor": 10.0,
            "pnl": float("nan"),
            "pnl_target": pnl_target,
            "context": make_context(float("nan")),
            "duration_ratio": 0.2,
            "params": {"exit_attenuation_mode": "linear", "exit_linear_slope": 0.5},
            "expectation": "safe_zero",
        },
        {
            "base_factor": 10.0,
            "pnl": 0.01,
            "pnl_target": pnl_target,
            "context": make_context(0.01),
            "duration_ratio": float("nan"),
            "params": {"exit_attenuation_mode": "linear", "exit_linear_slope": 0.5},
            "expectation": "safe_zero",
        },
        {
            "base_factor": 10.0,
            "pnl": 0.02,
            "pnl_target": float("inf"),
            "context": make_context(0.02),
            "duration_ratio": 0.5,
            "params": {
                "exit_attenuation_mode": "linear",
                "exit_linear_slope": 1.0,
                "check_invariants": True,
            },
            "expectation": "safe_zero",
        },
        {
            "base_factor": 10.0,
            "pnl": -0.02,
            "pnl_target": 0.03,
            "context": make_context(-0.02),
            "duration_ratio": 2.0,
            "params": {
                "exit_attenuation_mode": "legacy",
                "exit_plateau": False,
                "check_invariants": True,
            },
            "expectation": "clamped",
        },
    ]
    assert_exit_factor_invariant_suite(_PyTestAdapter(), suite, _get_exit_factor)
