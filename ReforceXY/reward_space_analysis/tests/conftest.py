"""Pytest configuration for reward space analysis tests."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from reward_space_analysis import calculate_reward


@pytest.fixture(scope="session")
def temp_output_dir():
    """Temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def setup_rng():
    """Configure RNG for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def base_reward_params():
    """Default reward parameters."""
    from reward_space_analysis import DEFAULT_MODEL_REWARD_PARAMETERS

    return DEFAULT_MODEL_REWARD_PARAMETERS.copy()


def assert_hold_penalty_threshold_behavior(
    test_case,
    duration_test_cases: List[Tuple[int, str]],
    max_duration: int,
    context_factory_fn,
    params: Dict[str, Any],
    base_factor: float,
    profit_target: float,
    risk_reward_ratio: float,
    tolerance_relaxed: float,
):
    from .helpers import assert_hold_penalty_threshold_behavior as _impl

    return _impl(
        test_case,
        duration_test_cases,
        max_duration,
        context_factory_fn,
        params,
        base_factor,
        profit_target,
        risk_reward_ratio,
        tolerance_relaxed,
    )


def assert_component_sum_integrity(
    test_case,
    breakdown,
    tolerance_relaxed: float,
    exclude_components: List[str] = [],
    component_description: str = "components",
):
    from .helpers import assert_component_sum_integrity as _impl

    return _impl(test_case, breakdown, tolerance_relaxed, exclude_components, component_description)


def assert_progressive_scaling_behavior(
    test_case,
    penalties_list: List[float],
    durations: List[int],
    penalty_type: str = "penalty",
):
    """Validate penalty progression patterns consistently.

    Args:
        test_case: TestCase instance for assertions
        penalties_list: List of penalty values in order
        durations: Corresponding duration values
        penalty_type: Type of penalty for assertion messages
    """
    for i in range(1, len(penalties_list)):
        test_case.assertLessEqual(
            penalties_list[i],
            penalties_list[i - 1],
            f"{penalty_type} should increase (more negative) with duration: "
            f"{penalties_list[i]} <= {penalties_list[i - 1]} "
            f"(duration {durations[i]} vs {durations[i - 1]})",
        )


def assert_decomposition_integrity_scenario(
    test_case,
    breakdown,
    active_component_name: str,
    tolerance_relaxed: float,
):
    """Validate component decomposition with single active component.

    Args:
        test_case: TestCase instance for assertions
        breakdown: Reward breakdown result
        active_component_name: Name of the component that should be active
        tolerance_relaxed: Tolerance for float comparisons
    """
    core_components = {
        "exit_component": breakdown.exit_component,
        "idle_penalty": breakdown.idle_penalty,
        "hold_penalty": breakdown.hold_penalty,
        "invalid_penalty": breakdown.invalid_penalty,
    }

    for name, value in core_components.items():
        if name == active_component_name:
            test_case.assertAlmostEqual(
                value,
                breakdown.total,
                delta=tolerance_relaxed,
                msg=f"Active component {name} should equal total",
            )
        else:
            test_case.assertAlmostEqual(
                value,
                0.0,
                delta=tolerance_relaxed,
                msg=f"Inactive component {name} should be near zero (val={value})",
            )

    test_case.assertAlmostEqual(breakdown.reward_shaping, 0.0, delta=tolerance_relaxed)
    test_case.assertAlmostEqual(breakdown.entry_additive, 0.0, delta=tolerance_relaxed)
    test_case.assertAlmostEqual(breakdown.exit_additive, 0.0, delta=tolerance_relaxed)


def assert_reward_calculation_scenarios(
    test_case,
    scenarios: List[Tuple[Any, Dict[str, Any], str]],
    base_factor: float,
    profit_target: float,
    risk_reward_ratio: float,
    validation_fn,
    tolerance_relaxed: float,
):
    """Centralized reward calculation for multiple scenarios.

    Args:
        test_case: TestCase instance for assertions
        scenarios: List of (context, params, description) tuples
        base_factor: Base factor for reward calculation
        profit_target: Profit target
        risk_reward_ratio: Risk reward ratio
        validation_fn: Function to validate each result (test_case, breakdown, description)
        tolerance_relaxed: Tolerance for float comparisons
    """
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
    """Validate parameter sensitivity behavior patterns.

    Args:
        test_case: TestCase instance for assertions
        parameter_variations: List of parameter dicts to test
        base_context: Base context for calculations
        base_params: Base parameters dict
        base_factor: Base factor for reward calculation
        profit_target: Profit target
        risk_reward_ratio: Risk reward ratio
        component_name: Component name to analyze (e.g., "idle_penalty")
        expected_trend: "increasing", "decreasing", or "constant"
        tolerance_relaxed: Tolerance for float comparisons
    """
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
            test_case.assertGreaterEqual(
                results[i],
                results[i - 1] - tolerance_relaxed,
                f"{component_name} should increase with parameter variations",
            )
    elif expected_trend == "decreasing":
        for i in range(1, len(results)):
            test_case.assertLessEqual(
                results[i],
                results[i - 1] + tolerance_relaxed,
                f"{component_name} should decrease with parameter variations",
            )
    elif expected_trend == "constant":
        baseline = results[0]
        for result in results[1:]:
            test_case.assertAlmostEqual(
                result,
                baseline,
                delta=tolerance_relaxed,
                msg=f"{component_name} should remain constant with parameter variations",
            )


def make_idle_penalty_test_contexts(
    context_factory_fn,
    idle_duration_scenarios: List[int],
    base_context_kwargs: Optional[Dict[str, Any]] = None,
):
    """Generate consistent contexts for idle penalty testing scenarios.

    Args:
        context_factory_fn: Function to create test context
        idle_duration_scenarios: List of idle durations to test
        base_context_kwargs: Base context parameters (default: empty dict)

    Returns:
        List of (context, description) tuples
    """
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
    attenuation_modes: List[str],
    base_params_fn,
    tolerance_relaxed: float,
):
    """Centralized exit factor attenuation mode testing and monotonic validation.

    Args:
        test_case: TestCase instance for assertions
        base_factor: Base factor for exit calculations
        pnl: PnL value for calculations
        pnl_factor: PnL factor for calculations
        attenuation_modes: List of attenuation modes to test
        base_params_fn: Function to create base parameters (should accept **kwargs)
        tolerance_relaxed: Tolerance for float comparisons
    """
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
            values = [
                _get_exit_factor(base_factor, pnl, pnl_factor, r, mode_params) for r in ratios
            ]

            if mode == "plateau_linear":
                grace = float(mode_params["exit_plateau_grace"])
                filtered = [
                    (r, v) for r, v in zip(ratios, values) if r >= grace - tolerance_relaxed
                ]
                values_to_check = [v for _, v in filtered]
            else:
                values_to_check = values

            for earlier, later in zip(values_to_check, values_to_check[1:]):
                test_case.assertLessEqual(
                    later,
                    earlier + tolerance_relaxed,
                    f"Non-monotonic attenuation in mode={mode}",
                )


def assert_exit_mode_mathematical_validation(
    test_case,
    context,
    params: Dict[str, Any],
    base_factor: float,
    profit_target: float,
    risk_reward_ratio: float,
    tolerance_relaxed: float,
):
    """Validate mathematical correctness of exit factor attenuation modes.

    Args:
        test_case: TestCase instance for assertions
        context: Test context for calculations
        params: Base parameters dict (will be modified for different modes)
        base_factor: Base factor for reward calculation
        profit_target: Profit target
        risk_reward_ratio: Risk reward ratio
        tolerance_relaxed: Tolerance for float comparisons
    """
    from reward_space_analysis import _get_exit_factor, _get_pnl_factor, calculate_reward

    duration_ratio = context.trade_duration / 100  # Assuming max_duration = 100

    # Test power mode
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

    # Test half_life mode with mathematical validation
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
    observed_exit_factor = _get_exit_factor(
        base_factor, context.pnl, pnl_factor_hl, duration_ratio, params
    )
    eps_base = 1e-8  # Assuming EPS_BASE constant
    observed_half_life_factor = observed_exit_factor / (base_factor * max(pnl_factor_hl, eps_base))
    expected_half_life_factor = 2 ** (-duration_ratio / params["exit_half_life"])
    test_case.assertAlmostEqual(
        observed_half_life_factor,
        expected_half_life_factor,
        delta=tolerance_relaxed,
        msg="Half-life attenuation mismatch: observed vs expected",
    )

    # Test linear mode
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

    # Validate all modes produce positive, distinct results
    rewards = [
        reward_power.exit_component,
        reward_half_life.exit_component,
        reward_linear.exit_component,
    ]
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
    """Centralized multi-parameter sensitivity testing logic.

    Args:
        test_case: TestCase instance for assertions
        parameter_test_cases: List of (profit_target, risk_reward_ratio, description) tuples
        context_factory_fn: Function to create test contexts (should accept context_type kwarg)
        base_params: Base parameters dict
        base_factor: Base factor for reward calculation
        tolerance_relaxed: Tolerance for float comparisons
    """
    from reward_space_analysis import calculate_reward

    for profit_target, risk_reward_ratio, description in parameter_test_cases:
        with test_case.subTest(
            profit_target=profit_target, risk_reward_ratio=risk_reward_ratio, desc=description
        ):
            # Test idle scenario for profit_target impact on idle penalty
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

            # Special case: zero profit target should result in zero idle penalty
            if profit_target == 0.0:
                test_case.assertEqual(
                    breakdown.idle_penalty,
                    0.0,
                    f"Idle penalty should be zero when profit_target=0 in {description}",
                )
                test_case.assertEqual(
                    breakdown.total,
                    0.0,
                    f"Total reward should be zero in this configuration for {description}",
                )
            else:
                # Non-zero profit target with idle duration should produce negative idle penalty
                test_case.assertLess(
                    breakdown.idle_penalty,
                    0.0,
                    f"Idle penalty should be negative for non-zero profit_target in {description}",
                )

            # Test exit scenario sensitivity (if profit_target > 0)
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

                test_case.assertNotEqual(
                    exit_breakdown.exit_component,
                    0.0,
                    f"Exit component should be non-zero for profitable exit in {description}",
                )
