import unittest

import pytest

from reward_space_analysis import (
    Actions,
    Positions,
    RewardContext,
    RewardDiagnosticsWarning,
    _get_exit_factor,
    _hold_penalty,
    validate_reward_parameters,
)

from ..helpers import (
    assert_exit_factor_invariant_suite,
    run_relaxed_validation_adjustment_cases,
    run_strict_validation_failure_cases,
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
    params = {"exit_attenuation_mode": "linear", "exit_plateau": True, "exit_plateau_grace": -1.0}
    with pytest.warns(RewardDiagnosticsWarning):
        factor = _get_exit_factor(
            base_factor=10.0,
            pnl=0.01,
            pnl_factor=1.0,
            duration_ratio=0.5,
            params=params,
        )
    assert factor >= 0.0


@pytest.mark.robustness
def test_get_exit_factor_negative_linear_slope_warning():
    params = {"exit_attenuation_mode": "linear", "exit_linear_slope": -5.0}
    with pytest.warns(RewardDiagnosticsWarning):
        factor = _get_exit_factor(
            base_factor=10.0,
            pnl=0.01,
            pnl_factor=1.0,
            duration_ratio=2.0,
            params=params,
        )
    assert factor >= 0.0


@pytest.mark.robustness
def test_get_exit_factor_invalid_power_tau_relaxed():
    params = {"exit_attenuation_mode": "power", "exit_power_tau": 0.0, "strict_validation": False}
    with pytest.warns(RewardDiagnosticsWarning):
        factor = _get_exit_factor(
            base_factor=5.0,
            pnl=0.02,
            pnl_factor=1.0,
            duration_ratio=1.5,
            params=params,
        )
    assert factor > 0.0


@pytest.mark.robustness
def test_get_exit_factor_half_life_near_zero_relaxed():
    params = {
        "exit_attenuation_mode": "half_life",
        "exit_half_life": 1e-12,
        "strict_validation": False,
    }
    with pytest.warns(RewardDiagnosticsWarning):
        factor = _get_exit_factor(
            base_factor=5.0,
            pnl=0.02,
            pnl_factor=1.0,
            duration_ratio=2.0,
            params=params,
        )
    assert factor != 0.0


@pytest.mark.robustness
def test_hold_penalty_short_duration_returns_zero():
    context = RewardContext(
        pnl=0.0,
        trade_duration=1,  # shorter than default max trade duration (128)
        idle_duration=0,
        max_unrealized_profit=0.0,
        min_unrealized_profit=0.0,
        position=Positions.Long,
        action=Actions.Neutral,
    )
    params = {"max_trade_duration_candles": 128}
    penalty = _hold_penalty(context, hold_factor=1.0, params=params)
    assert penalty == 0.0


@pytest.mark.robustness
def test_exit_factor_invariant_suite_grouped():
    """Grouped exit factor invariant scenarios using shared helper."""
    suite = [
        {
            "base_factor": 15.0,
            "pnl": 0.02,
            "pnl_factor": 1.0,
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
            "pnl_factor": 1.0,
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
            "pnl_factor": 1.0,
            "duration_ratio": 0.2,
            "params": {"exit_attenuation_mode": "linear", "exit_linear_slope": 0.5},
            "expectation": "safe_zero",
        },
        {
            "base_factor": 10.0,
            "pnl": float("nan"),
            "pnl_factor": 1.0,
            "duration_ratio": 0.2,
            "params": {"exit_attenuation_mode": "linear", "exit_linear_slope": 0.5},
            "expectation": "safe_zero",
        },
        {
            "base_factor": 10.0,
            "pnl": 0.01,
            "pnl_factor": 1.0,
            "duration_ratio": float("nan"),
            "params": {"exit_attenuation_mode": "linear", "exit_linear_slope": 0.5},
            "expectation": "safe_zero",
        },
        {
            "base_factor": 10.0,
            "pnl": 0.02,
            "pnl_factor": float("inf"),
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
            "pnl": 0.015,
            "pnl_factor": -2.5,
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
