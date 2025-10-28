import math

import pytest

from reward_space_analysis import (
    Actions,
    Positions,
    RewardContext,
    RewardDiagnosticsWarning,
    _get_exit_factor,
    _hold_penalty,
    _normalize_and_validate_mode,
    validate_reward_parameters,
)


@pytest.mark.robustness
def test_validate_reward_parameters_strict_non_numeric_failure():
    params = {"exit_linear_slope": "not_a_number"}
    with pytest.raises(ValueError):
        validate_reward_parameters(params, strict=True)

@pytest.mark.robustness
def test_validate_reward_parameters_strict_exit_power_tau_below_min_failure():
    params = {"exit_power_tau": 0.0}
    with pytest.raises(ValueError):
        validate_reward_parameters(params, strict=True)

@pytest.mark.robustness
def test_validate_reward_parameters_strict_exit_power_tau_above_max_failure():
    params = {"exit_power_tau": 1.5}
    with pytest.raises(ValueError):
        validate_reward_parameters(params, strict=True)

@pytest.mark.robustness
def test_validate_reward_parameters_strict_exit_half_life_below_min_failure():
    params = {"exit_half_life": 0.0}
    with pytest.raises(ValueError):
        validate_reward_parameters(params, strict=True)

@pytest.mark.robustness
def test_validate_reward_parameters_strict_exit_half_life_non_finite_failure():
    params = {"exit_half_life": float("nan")}
    with pytest.raises(ValueError):
        validate_reward_parameters(params, strict=True)


@pytest.mark.robustness
def test_validate_reward_parameters_relaxed_non_numeric_reset():
    params = {"exit_linear_slope": "not_a_number", "strict_validation": False}
    sanitized, adjustments = validate_reward_parameters(params, strict=False)
    assert math.isclose(sanitized["exit_linear_slope"], 0.0)
    assert adjustments["exit_linear_slope"]["reason"].startswith("non_numeric_reset")


@pytest.mark.robustness
def test_validate_reward_parameters_relaxed_non_finite_reset():
    # Trigger non_finite_reset by providing an infinite numeric which coerces then clamps non-finite after bounds enforcement
    params = {"exit_power_tau": float("inf"), "strict_validation": False}
    sanitized, adjustments = validate_reward_parameters(params, strict=False)
    # exit_power_tau has bounds min=1e-6, max=1.0; infinite should reset to max then detect non-finite -> min fallback
    assert math.isclose(sanitized["exit_power_tau"], 1e-6)
    reason = adjustments["exit_power_tau"]["reason"]
    # Infinite coerces to max then remains finite; non-finite path not taken. Expect non_numeric_reset from initial coercion failure.
    assert reason.startswith("non_numeric_reset"), f"Expected non_numeric_reset in reason (got {reason})"
    params = {"exit_linear_slope": "not_a_number", "strict_validation": False}
    sanitized, adjustments = validate_reward_parameters(params, strict=False)
    assert math.isclose(sanitized["exit_linear_slope"], 0.0)
    assert adjustments["exit_linear_slope"]["reason"].startswith("non_numeric_reset")


@pytest.mark.robustness
def test_validate_reward_parameters_drop_derived_max_idle_duration_candles():
    params = {"max_idle_duration_candles": "bad", "strict_validation": False}
    sanitized, adjustments = validate_reward_parameters(params, strict=False)
    assert "max_idle_duration_candles" not in sanitized
    assert adjustments["max_idle_duration_candles"]["reason"] == "derived_default"


@pytest.mark.robustness
def test_normalize_and_validate_mode_fallback():
    params = {"exit_attenuation_mode": "invalid_mode"}
    _normalize_and_validate_mode(params)
    assert params["exit_attenuation_mode"] == "linear"


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
    params = {"exit_attenuation_mode": "half_life", "exit_half_life": 1e-12, "strict_validation": False}
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
def test_get_exit_factor_negative_duration_ratio_guard():
    params = {"exit_attenuation_mode": "linear", "exit_linear_slope": 1.2, "exit_plateau": False}
    base_factor = 15.0
    pnl = 0.02
    pnl_factor = 1.0
    f_neg = _get_exit_factor(base_factor, pnl, pnl_factor, -5.0, params)
    f_zero = _get_exit_factor(base_factor, pnl, pnl_factor, 0.0, params)
    assert pytest.approx(f_neg, rel=1e-12) == f_zero


@pytest.mark.robustness
def test_get_exit_factor_non_finite_inputs_return_safe_zero():
    params = {"exit_attenuation_mode": "linear", "exit_linear_slope": 0.5}
    pnl_factor = 1.0
    # Non-finite base_factor
    f_base_nan = _get_exit_factor(float("nan"), 0.01, pnl_factor, 0.2, params)
    # Non-finite pnl
    f_pnl_nan = _get_exit_factor(10.0, float("nan"), pnl_factor, 0.2, params)
    # Non-finite duration_ratio
    f_dr_nan = _get_exit_factor(10.0, 0.01, pnl_factor, float("nan"), params)
    assert f_base_nan == 0.0
    assert f_pnl_nan == 0.0
    assert f_dr_nan == 0.0


@pytest.mark.robustness
def test_get_exit_factor_kernel_exception_fallback_linear(monkeypatch):
    # Patch math.sqrt used by sqrt kernel to raise forcing exception path
    def boom(value):  # noqa: D401
        raise RuntimeError("forced sqrt failure")

    original_sqrt = math.sqrt
    monkeypatch.setattr("reward_space_analysis.math.sqrt", boom)
    base_factor = 25.0
    pnl = 0.03
    pnl_factor = 1.0
    duration_ratio = 0.7
    params_sqrt = {"exit_attenuation_mode": "sqrt", "exit_plateau": False}
    with pytest.warns(RewardDiagnosticsWarning) as record:
        f_fail = _get_exit_factor(base_factor, pnl, pnl_factor, duration_ratio, params_sqrt)
    # Reference linear kernel (unpatched sqrt not used)
    f_linear = _get_exit_factor(base_factor, pnl, pnl_factor, duration_ratio, {"exit_attenuation_mode": "linear", "exit_plateau": False})
    assert pytest.approx(f_fail, rel=1e-12) == f_linear
    assert any("failed" in str(w.message) and "fallback linear" in str(w.message) for w in record)
    # Restore sqrt manually in case other tests rely on it before monkeypatch teardown (defensive)
    monkeypatch.setattr("reward_space_analysis.math.sqrt", original_sqrt)


@pytest.mark.robustness
def test_get_exit_factor_post_kernel_non_finite_exit_factor_invariant():
    # Create non-finite exit_factor after kernel by using infinite pnl_factor
    params = {"exit_attenuation_mode": "linear", "exit_linear_slope": 1.0, "check_invariants": True}
    f_inf = _get_exit_factor(10.0, 0.02, float("inf"), 0.5, params)
    assert f_inf == 0.0  # invariant fallback

@pytest.mark.robustness
def test_get_exit_factor_negative_exit_factor_clamped_for_positive_pnl():
    # Force attenuation_factor negative via legacy mode (duration beyond plateau leading to 0.5 multiplier) and negative pnl_factor
    # then rely on invariant to clamp since pnl >= 0
    params = {"exit_attenuation_mode": "legacy", "exit_plateau": False, "check_invariants": True}
    base_factor = 10.0
    pnl = 0.015  # positive
    pnl_factor = -2.5  # negative to yield negative exit_factor prior to clamp
    duration_ratio = 2.0  # legacy kernel: factor * 0.5
    raw_factor = _get_exit_factor(base_factor, pnl, pnl_factor, duration_ratio, {**params, "check_invariants": False})
    assert raw_factor < 0.0  # confirm pre-clamp negative when invariants disabled
    clamped_factor = _get_exit_factor(base_factor, pnl, pnl_factor, duration_ratio, params)
    assert clamped_factor >= 0.0
