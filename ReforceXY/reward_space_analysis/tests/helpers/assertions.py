"""Shared assertion helpers for reward_space_analysis test suite.

These functions centralize common numeric and behavioral checks to enforce
single invariant ownership and reduce duplication across taxonomy modules.
"""
from typing import Any, Dict, List, Sequence, Tuple


def safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce value to float safely for test parameter handling.

    Rules:
    - None, '' -> default
    - Numeric types pass through
    - String numeric forms ('3', '3.5', 'nan', 'inf') handled; nan/inf return default
    - Non-numeric strings return default
    Avoids direct float(...) exceptions leaking into tests that target relaxed validation behaviors.
    """
    try:
        if value is None or value == "":
            return default
        coerced = float(value)
        if coerced != coerced or coerced in (float("inf"), float("-inf")):
            return default
        return coerced
    except (TypeError, ValueError):
        return default



def assert_monotonic_nonincreasing(test_case, values: Sequence[float], tolerance: float = 0.0, msg: str = "Values should be non-increasing"):
    """Assert that each subsequent value is <= previous (non-increasing)."""
    for i in range(1, len(values)):
        test_case.assertLessEqual(values[i], values[i - 1] + tolerance, msg)


def assert_monotonic_nonnegative(test_case, values: Sequence[float], tolerance: float = 0.0, msg: str = "Values should be non-negative"):
    """Assert all values are >= 0."""
    for v in values:
        test_case.assertGreaterEqual(v + tolerance, 0.0, msg)


def assert_finite(test_case, values: Sequence[float], msg: str = "Values must be finite"):
    """Assert all values are finite numbers."""
    for v in values:
        test_case.assertTrue((v == v) and (v not in (float("inf"), float("-inf"))), msg)


def assert_almost_equal_list(test_case, values: Sequence[float], target: float, delta: float, msg: str = "Values should be near target"):
    """Assert each value is within delta of target."""
    for v in values:
        test_case.assertAlmostEqual(v, target, delta=delta, msg=msg)


def assert_trend(test_case, values: Sequence[float], trend: str, tolerance: float, msg_prefix: str = "Trend validation failed"):
    """Generic trend assertion for increasing/decreasing/constant sequences."""
    if trend not in {"increasing", "decreasing", "constant"}:
        raise ValueError(f"Unsupported trend '{trend}'")
    if trend == "increasing":
        for i in range(1, len(values)):
            test_case.assertGreaterEqual(values[i], values[i - 1] - tolerance, f"{msg_prefix}: expected increasing")
    elif trend == "decreasing":
        for i in range(1, len(values)):
            test_case.assertLessEqual(values[i], values[i - 1] + tolerance, f"{msg_prefix}: expected decreasing")
    else:  # constant
        base = values[0]
        for v in values[1:]:
            test_case.assertAlmostEqual(v, base, delta=tolerance, msg=f"{msg_prefix}: expected constant")


def assert_component_sum_integrity(test_case, breakdown, tolerance_relaxed, exclude_components=None, component_description="components"):
    if exclude_components is None:
        exclude_components = []
    component_sum = 0.0
    if "hold_penalty" not in exclude_components:
        component_sum += breakdown.hold_penalty
    if "idle_penalty" not in exclude_components:
        component_sum += breakdown.idle_penalty
    if "exit_component" not in exclude_components:
        component_sum += breakdown.exit_component
    if "invalid_penalty" not in exclude_components:
        component_sum += breakdown.invalid_penalty
    if "reward_shaping" not in exclude_components:
        component_sum += breakdown.reward_shaping
    if "entry_additive" not in exclude_components:
        component_sum += breakdown.entry_additive
    if "exit_additive" not in exclude_components:
        component_sum += breakdown.exit_additive
    test_case.assertAlmostEqual(breakdown.total, component_sum, delta=tolerance_relaxed, msg=f"Total should equal sum of {component_description}")


def assert_progressive_scaling_behavior(test_case, penalties_list: Sequence[float], durations: Sequence[int], penalty_type: str = "penalty"):
    """Validate penalty progression patterns consistently."""
    for i in range(1, len(penalties_list)):
        test_case.assertLessEqual(
            penalties_list[i],
            penalties_list[i - 1],
            f"{penalty_type} should increase (more negative) with duration: {penalties_list[i]} <= {penalties_list[i - 1]} (duration {durations[i]} vs {durations[i - 1]})",
        )


def assert_single_active_component(test_case, breakdown, active_name: str, tolerance: float, inactive_core: Sequence[str]):
    """Assert only one core component is active; others near zero."""
    for name in inactive_core:
        if name == active_name:
            test_case.assertAlmostEqual(getattr(breakdown, name), breakdown.total, delta=tolerance, msg=f"Active component {name} should equal total")
        else:
            test_case.assertAlmostEqual(getattr(breakdown, name), 0.0, delta=tolerance, msg=f"Inactive component {name} should be near zero")


def assert_reward_calculation_scenarios(
    test_case,
    scenarios: List[Tuple[Any, Dict[str, Any], str]],
    base_factor: float,
    profit_target: float,
    risk_reward_ratio: float,
    validation_fn,
    tolerance_relaxed: float,
):
    from reward_space_analysis import calculate_reward
    for context, params, description in scenarios:
        with test_case.subTest(scenario=description):
            breakdown = calculate_reward(
                context,
                params,
                base_factor=base_factor,
                profit_target=profit_target,
                risk_reward_ratio=risk_reward_ratio,
                short_allowed=True,
                action_masking=True,
            )
            validation_fn(test_case, breakdown, description, tolerance_relaxed)


def assert_parameter_sensitivity_behavior(
    test_case,
    parameter_variations: List[Dict[str, Any]],
    base_context,
    base_params: Dict[str, Any],
    base_factor: float,
    profit_target: float,
    risk_reward_ratio: float,
    component_name: str,
    expected_trend: str,
    tolerance_relaxed: float,
):
    from reward_space_analysis import calculate_reward
    results = []
    for param_variation in parameter_variations:
        params = base_params.copy()
        params.update(param_variation)
        breakdown = calculate_reward(
            base_context,
            params,
            base_factor=base_factor,
            profit_target=profit_target,
            risk_reward_ratio=risk_reward_ratio,
            short_allowed=True,
            action_masking=True,
        )
        component_value = getattr(breakdown, component_name)
        results.append(component_value)
    if expected_trend == "increasing":
        for i in range(1, len(results)):
            test_case.assertGreaterEqual(results[i], results[i - 1] - tolerance_relaxed, f"{component_name} should increase with parameter variations")
    elif expected_trend == "decreasing":
        for i in range(1, len(results)):
            test_case.assertLessEqual(results[i], results[i - 1] + tolerance_relaxed, f"{component_name} should decrease with parameter variations")
    elif expected_trend == "constant":
        baseline = results[0]
        for result in results[1:]:
            test_case.assertAlmostEqual(result, baseline, delta=tolerance_relaxed, msg=f"{component_name} should remain constant with parameter variations")


def make_idle_penalty_test_contexts(context_factory_fn, idle_duration_scenarios: Sequence[int], base_context_kwargs: Dict[str, Any] = None):
    if base_context_kwargs is None:
        base_context_kwargs = {}
    contexts = []
    for idle_duration in idle_duration_scenarios:
        kwargs = base_context_kwargs.copy()
        kwargs["idle_duration"] = idle_duration
        context = context_factory_fn(**kwargs)
        description = f"idle_duration={idle_duration}"
        contexts.append((context, description))
    return contexts


def assert_exit_factor_attenuation_modes(
    test_case,
    base_factor: float,
    pnl: float,
    pnl_factor: float,
    attenuation_modes: Sequence[str],
    base_params_fn,
    tolerance_relaxed: float,
):
    import numpy as np

    from reward_space_analysis import _get_exit_factor
    for mode in attenuation_modes:
        with test_case.subTest(mode=mode):
            if mode == "plateau_linear":
                mode_params = base_params_fn(
                    exit_attenuation_mode="linear",
                    exit_plateau=True,
                    exit_plateau_grace=0.2,
                    exit_linear_slope=1.0,
                )
            elif mode == "linear":
                mode_params = base_params_fn(exit_attenuation_mode="linear", exit_linear_slope=1.2)
            elif mode == "power":
                mode_params = base_params_fn(exit_attenuation_mode="power", exit_power_tau=0.5)
            elif mode == "half_life":
                mode_params = base_params_fn(exit_attenuation_mode="half_life", exit_half_life=0.7)
            else:
                mode_params = base_params_fn(exit_attenuation_mode="sqrt")
            ratios = np.linspace(0, 2, 15)
            values = [_get_exit_factor(base_factor, pnl, pnl_factor, r, mode_params) for r in ratios]
            if mode == "plateau_linear":
                grace = float(mode_params["exit_plateau_grace"])
                filtered = [(r, v) for r, v in zip(ratios, values) if r >= grace - tolerance_relaxed]
                values_to_check = [v for _, v in filtered]
            else:
                values_to_check = values
            for earlier, later in zip(values_to_check, values_to_check[1:]):
                test_case.assertLessEqual(later, earlier + tolerance_relaxed, f"Non-monotonic attenuation in mode={mode}")


def assert_exit_mode_mathematical_validation(
    test_case,
    context,
    params: Dict[str, Any],
    base_factor: float,
    profit_target: float,
    risk_reward_ratio: float,
    tolerance_relaxed: float,
):
    from reward_space_analysis import _get_exit_factor, _get_pnl_factor, calculate_reward
    duration_ratio = context.trade_duration / 100
    params["exit_attenuation_mode"] = "power"
    params["exit_power_tau"] = 0.5
    params["exit_plateau"] = False
    reward_power = calculate_reward(
        context,
        params,
        base_factor=base_factor,
        profit_target=profit_target,
        risk_reward_ratio=risk_reward_ratio,
        short_allowed=True,
        action_masking=True,
    )
    test_case.assertGreater(reward_power.exit_component, 0)
    params["exit_attenuation_mode"] = "half_life"
    params["exit_half_life"] = 0.5
    reward_half_life = calculate_reward(
        context,
        params,
        base_factor=base_factor,
        profit_target=profit_target,
        risk_reward_ratio=risk_reward_ratio,
        short_allowed=True,
        action_masking=True,
    )
    pnl_factor_hl = _get_pnl_factor(params, context, profit_target, risk_reward_ratio)
    observed_exit_factor = _get_exit_factor(base_factor, context.pnl, pnl_factor_hl, duration_ratio, params)
    eps_base = 1e-8
    observed_half_life_factor = observed_exit_factor / (base_factor * max(pnl_factor_hl, eps_base))
    expected_half_life_factor = 2 ** (-duration_ratio / params["exit_half_life"])
    test_case.assertAlmostEqual(observed_half_life_factor, expected_half_life_factor, delta=tolerance_relaxed, msg="Half-life attenuation mismatch: observed vs expected")
    params["exit_attenuation_mode"] = "linear"
    params["exit_linear_slope"] = 1.0
    reward_linear = calculate_reward(
        context,
        params,
        base_factor=base_factor,
        profit_target=profit_target,
        risk_reward_ratio=risk_reward_ratio,
        short_allowed=True,
        action_masking=True,
    )
    rewards = [reward_power.exit_component, reward_half_life.exit_component, reward_linear.exit_component]
    test_case.assertTrue(all((r > 0 for r in rewards)))
    unique_rewards = set((f"{r:.6f}" for r in rewards))
    test_case.assertGreater(len(unique_rewards), 1)


def assert_multi_parameter_sensitivity(
    test_case,
    parameter_test_cases: List[Tuple[float, float, str]],
    context_factory_fn,
    base_params: Dict[str, Any],
    base_factor: float,
    tolerance_relaxed: float,
):
    from reward_space_analysis import calculate_reward
    for profit_target, risk_reward_ratio, description in parameter_test_cases:
        with test_case.subTest(profit_target=profit_target, risk_reward_ratio=risk_reward_ratio, desc=description):
            idle_context = context_factory_fn(context_type="idle")
            breakdown = calculate_reward(
                idle_context,
                base_params,
                base_factor=base_factor,
                profit_target=profit_target,
                risk_reward_ratio=risk_reward_ratio,
                short_allowed=True,
                action_masking=True,
            )
            if profit_target == 0.0:
                test_case.assertEqual(breakdown.idle_penalty, 0.0)
                test_case.assertEqual(breakdown.total, 0.0)
            else:
                test_case.assertLess(breakdown.idle_penalty, 0.0)
            if profit_target > 0:
                exit_context = context_factory_fn(context_type="exit", profit_target=profit_target)
                exit_breakdown = calculate_reward(
                    exit_context,
                    base_params,
                    base_factor=base_factor,
                    profit_target=profit_target,
                    risk_reward_ratio=risk_reward_ratio,
                    short_allowed=True,
                    action_masking=True,
                )
                test_case.assertNotEqual(exit_breakdown.exit_component, 0.0)


def assert_hold_penalty_threshold_behavior(
    test_case,
    duration_test_cases: Sequence[Tuple[int, str]],
    max_duration: int,
    context_factory_fn,
    params: Dict[str, Any],
    base_factor: float,
    profit_target: float,
    risk_reward_ratio: float,
    tolerance_relaxed: float,
):
    from reward_space_analysis import calculate_reward
    for trade_duration, description in duration_test_cases:
        with test_case.subTest(duration=trade_duration, desc=description):
            context = context_factory_fn(trade_duration=trade_duration)
            breakdown = calculate_reward(
                context,
                params,
                base_factor=base_factor,
                profit_target=profit_target,
                risk_reward_ratio=risk_reward_ratio,
                short_allowed=True,
                action_masking=True,
            )
            duration_ratio = trade_duration / max_duration
            if duration_ratio < 1.0:
                test_case.assertEqual(breakdown.hold_penalty, 0.0)
            elif duration_ratio == 1.0:
                test_case.assertLessEqual(breakdown.hold_penalty, 0.0)
            else:
                test_case.assertLess(breakdown.hold_penalty, 0.0)
