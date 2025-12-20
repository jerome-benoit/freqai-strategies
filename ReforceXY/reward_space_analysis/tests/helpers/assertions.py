"""Shared assertion helpers for reward_space_analysis test suite.

These functions centralize common numeric and behavioral checks to enforce
single invariant ownership and reduce duplication across taxonomy modules.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from reward_space_analysis import (
    RewardContext,
    _compute_efficiency_coefficient,
    _compute_pnl_target_coefficient,
    _get_exit_factor,
    calculate_reward,
)

from ..constants import TOLERANCE
from .configs import RewardScenarioConfig, ThresholdTestConfig, ValidationConfig


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


def assert_monotonic_nonincreasing(
    test_case,
    values: Sequence[float],
    tolerance: float = 0.0,
    msg: str = "Values should be non-increasing",
):
    """Assert that a sequence is monotonically non-increasing.

    Validates that each element in the sequence is less than or equal to the
    previous element, with an optional tolerance for floating-point comparisons.

    Args:
        test_case: Test case instance with assertion methods
        values: Sequence of numeric values to validate
        tolerance: Numerical tolerance for comparisons (default: 0.0)
        msg: Custom error message for assertion failures

    Example:
        assert_monotonic_nonincreasing(self, [5.0, 4.0, 3.0, 3.0, 2.0])
        # Validates: 4.0 <= 5.0, 3.0 <= 4.0, 3.0 <= 3.0, 2.0 <= 3.0
    """
    for i in range(1, len(values)):
        test_case.assertLessEqual(values[i], values[i - 1] + tolerance, msg)


def assert_monotonic_nonnegative(
    test_case,
    values: Sequence[float],
    tolerance: float = 0.0,
    msg: str = "Values should be non-negative",
):
    """Assert that all values in a sequence are non-negative.

    Validates that each element is greater than or equal to zero, with an
    optional tolerance for floating-point comparisons.

    Args:
        test_case: Test case instance with assertion methods
        values: Sequence of numeric values to validate
        tolerance: Numerical tolerance for comparisons (default: 0.0)
        msg: Custom error message for assertion failures

    Example:
        assert_monotonic_nonnegative(self, [0.0, 1.5, 2.3, 0.1])
    """
    for v in values:
        test_case.assertGreaterEqual(v + tolerance, 0.0, msg)


def assert_finite(test_case, values: Sequence[float], msg: str = "Values must be finite"):
    """Assert that all values are finite (not NaN or infinity).

    Validates that no element in the sequence is NaN, positive infinity, or
    negative infinity. Essential for numerical stability checks.

    Args:
        test_case: Test case instance with assertion methods
        values: Sequence of numeric values to validate
        msg: Custom error message for assertion failures

    Example:
        assert_finite(self, [1.0, 2.5, -3.7, 0.0])  # Passes
        assert_finite(self, [1.0, float('nan')])    # Fails
        assert_finite(self, [1.0, float('inf')])    # Fails
    """
    for v in values:
        test_case.assertTrue((v == v) and (v not in (float("inf"), float("-inf"))), msg)


def assert_almost_equal_list(
    test_case,
    values: Sequence[float],
    target: float,
    delta: float,
    msg: str = "Values should be near target",
):
    """Assert that all values in a sequence are approximately equal to a target.

    Validates that each element is within a specified tolerance (delta) of the
    target value. Useful for checking plateau behavior or constant outputs.

    Args:
        test_case: Test case instance with assertion methods
        values: Sequence of numeric values to validate
        target: Target value for comparison
        delta: Maximum allowed deviation from target
        msg: Custom error message for assertion failures

    Example:
        assert_almost_equal_list(self, [1.0, 1.01, 0.99], 1.0, delta=0.02)
    """
    for v in values:
        test_case.assertAlmostEqual(v, target, delta=delta, msg=msg)


def assert_trend(
    test_case,
    values: Sequence[float],
    trend: str,
    tolerance: float,
    msg_prefix: str = "Trend validation failed",
):
    """Assert that a sequence follows a specific trend pattern.

    Generic trend validation supporting increasing, decreasing, or constant
    patterns. More flexible than specialized monotonic assertions.

    Args:
        test_case: Test case instance with assertion methods
        values: Sequence of numeric values to validate
        trend: Expected trend: "increasing", "decreasing", or "constant"
        tolerance: Numerical tolerance for comparisons
        msg_prefix: Prefix for error messages

    Raises:
        ValueError: If trend parameter is not one of the supported values

    Example:
        assert_trend(self, [1.0, 2.0, 3.0], "increasing", 1e-09)
        assert_trend(self, [5.0, 5.0, 5.0], "constant", 1e-09)
    """
    if trend not in {"increasing", "decreasing", "constant"}:
        raise ValueError(f"Unsupported trend '{trend}'")
    if trend == "increasing":
        for i in range(1, len(values)):
            test_case.assertGreaterEqual(
                values[i], values[i - 1] - tolerance, f"{msg_prefix}: expected increasing"
            )
    elif trend == "decreasing":
        for i in range(1, len(values)):
            test_case.assertLessEqual(
                values[i], values[i - 1] + tolerance, f"{msg_prefix}: expected decreasing"
            )
    else:  # constant
        base = values[0]
        for v in values[1:]:
            test_case.assertAlmostEqual(
                v, base, delta=tolerance, msg=f"{msg_prefix}: expected constant"
            )


def assert_component_sum_integrity(
    test_case,
    breakdown,
    config: ValidationConfig,
):
    """Assert that reward component sum matches total within tolerance.

    Validates the mathematical integrity of reward component decomposition by
    ensuring the sum of individual components equals the reported total.
    Uses ValidationConfig to simplify parameter passing.

    Args:
        test_case: Test case instance with assertion methods
        breakdown: Reward breakdown object with component attributes
        config: ValidationConfig with tolerance and exclusion settings

    Components checked (if not excluded):
        - hold_penalty
        - idle_penalty
        - exit_component
        - invalid_penalty
        - reward_shaping
        - entry_additive
        - exit_additive

    Example:
        config = ValidationConfig(
            tolerance_strict=TOLERANCE.IDENTITY_STRICT,
            tolerance_relaxed=TOLERANCE.IDENTITY_RELAXED,
            exclude_components=["reward_shaping"],
            component_description="core components"
        )
        assert_component_sum_integrity(self, breakdown, config)
    """
    exclude_components = config.exclude_components or []
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
    test_case.assertAlmostEqual(
        breakdown.total,
        component_sum,
        delta=config.tolerance_relaxed,
        msg=f"Total should equal sum of {config.component_description}",
    )


def assert_progressive_scaling_behavior(
    test_case,
    penalties_list: Sequence[float],
    durations: Sequence[int],
    penalty_type: str = "penalty",
):
    """Validate that penalties scale progressively with increasing durations.

    Ensures penalties become more severe (more negative) as duration increases,
    which is a key invariant for hold and idle penalty calculations.

    Args:
        test_case: Test case instance with assertion methods
        penalties_list: Sequence of penalty values (typically negative)
        durations: Corresponding sequence of duration values
        penalty_type: Type of penalty for error messages (default: "penalty")

    Example:
        durations = [10, 50, 100, 200]
        penalties = [-5.0, -10.0, -15.0, -20.0]
        assert_progressive_scaling_behavior(self, penalties, durations, "hold_penalty")
    """
    for i in range(1, len(penalties_list)):
        test_case.assertLessEqual(
            penalties_list[i],
            penalties_list[i - 1],
            f"{penalty_type} should increase (more negative) with duration: {penalties_list[i]} <= {penalties_list[i - 1]} (duration {durations[i]} vs {durations[i - 1]})",
        )


def assert_single_active_component(
    test_case, breakdown, active_name: str, tolerance: float, inactive_core: Sequence[str]
):
    """Assert that exactly one reward component is active in a breakdown.

    Validates reward component isolation by ensuring the active component equals
    the total reward while all other components are negligible (near zero).

    Args:
        test_case: Test case instance with assertion methods
        breakdown: Reward breakdown object with component attributes
        active_name: Name of the component expected to be active
        tolerance: Numerical tolerance for near-zero checks
        inactive_core: List of core component names to check

    Example:
        assert_single_active_component(
            self, breakdown, "exit_component", TOLERANCE.IDENTITY_RELAXED,
            ["hold_penalty", "idle_penalty", "invalid_penalty"]
        )
    """
    for name in inactive_core:
        if name == active_name:
            test_case.assertAlmostEqual(
                getattr(breakdown, name),
                breakdown.total,
                delta=tolerance,
                msg=f"Active component {name} should equal total",
            )
        else:
            test_case.assertAlmostEqual(
                getattr(breakdown, name),
                0.0,
                delta=tolerance,
                msg=f"Inactive component {name} should be near zero",
            )


def assert_single_active_component_with_additives(
    test_case,
    breakdown,
    active_name: str,
    tolerance: float,
    inactive_core: Sequence[str],
    enforce_additives_zero: bool = True,
):
    """Assert single active core component with optional additive checks.

    Extended version of assert_single_active_component that additionally validates
    that additive components (reward_shaping, entry_additive, exit_additive) are
    near zero when they should be inactive.

    Args:
        test_case: Test case instance with assertion methods
        breakdown: Reward breakdown object with component attributes
        active_name: Name of the component expected to be active
        tolerance: Numerical tolerance for near-zero checks
        inactive_core: List of core component names to check
        enforce_additives_zero: If True, also check additives are near zero

    Example:
        assert_single_active_component_with_additives(
            self, breakdown, "exit_component", TOLERANCE.IDENTITY_RELAXED,
            ["hold_penalty", "idle_penalty"],
            enforce_additives_zero=True
        )
    """
    # Delegate core component assertions
    assert_single_active_component(test_case, breakdown, active_name, tolerance, inactive_core)
    if enforce_additives_zero:
        for attr in ("reward_shaping", "entry_additive", "exit_additive"):
            test_case.assertAlmostEqual(
                getattr(breakdown, attr),
                0.0,
                delta=tolerance,
                msg=f"{attr} should be near zero when inactive decomposition scenario",
            )


def assert_reward_calculation_scenarios(
    test_case,
    scenarios: List[Tuple[Any, Dict[str, Any], str]],
    config: RewardScenarioConfig,
    validation_fn,
):
    """Execute and validate multiple reward calculation scenarios.

    Runs a batch of reward calculations with different contexts and parameters,
    applying a custom validation function to each result. Uses RewardScenarioConfig
    to simplify parameter passing and improve maintainability.

    Args:
        test_case: Test case instance with assertion methods
        scenarios: List of (context, params, description) tuples defining test cases
        config: RewardScenarioConfig with all calculation parameters
        validation_fn: Callback function (test_case, breakdown, description, tolerance) -> None

    Example:
        config = RewardScenarioConfig(
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
            tolerance_relaxed=TOLERANCE.IDENTITY_RELAXED
        )
        scenarios = [
            (idle_context, {}, "idle scenario"),
            (exit_context, {"exit_additive": 5.0}, "profitable exit"),
        ]
        assert_reward_calculation_scenarios(
            self, scenarios, config, my_validation_fn
        )
    """
    for context, params, description in scenarios:
        with test_case.subTest(scenario=description):
            breakdown = calculate_reward(
                context,
                params,
                base_factor=config.base_factor,
                profit_aim=config.profit_aim,
                risk_reward_ratio=config.risk_reward_ratio,
                short_allowed=config.short_allowed,
                action_masking=config.action_masking,
            )
            validation_fn(test_case, breakdown, description, config.tolerance_relaxed)


def assert_parameter_sensitivity_behavior(
    test_case,
    parameter_variations: List[Dict[str, Any]],
    base_context,
    base_params: Dict[str, Any],
    component_name: str,
    expected_trend: str,
    config: RewardScenarioConfig,
):
    """Validate that a component responds predictably to parameter changes.

    Tests component sensitivity by applying parameter variations and verifying
    the component value follows the expected trend (increasing, decreasing, or constant).
    Uses RewardScenarioConfig to simplify parameter passing.

    Args:
        test_case: Test case instance with assertion methods
        parameter_variations: List of parameter dicts to merge with base_params
        base_context: Context object for reward calculation
        base_params: Base parameter dictionary
        component_name: Name of component to track (e.g., "exit_component")
        expected_trend: Expected trend: "increasing", "decreasing", or "constant"
        config: RewardScenarioConfig with calculation parameters

    Example:
        config = RewardScenarioConfig(
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
            tolerance_relaxed=TOLERANCE.IDENTITY_RELAXED
        )
        variations = [
            {"exit_additive": 0.0},
            {"exit_additive": 5.0},
            {"exit_additive": 10.0},
        ]
        assert_parameter_sensitivity_behavior(
            self, variations, ctx, params, "exit_component", "increasing", config
        )
    """
    from reward_space_analysis import calculate_reward

    results = []
    for param_variation in parameter_variations:
        params = base_params.copy()
        params.update(param_variation)
        breakdown = calculate_reward(
            base_context,
            params,
            base_factor=config.base_factor,
            profit_aim=config.profit_aim,
            risk_reward_ratio=config.risk_reward_ratio,
            short_allowed=config.short_allowed,
            action_masking=config.action_masking,
        )
        component_value = getattr(breakdown, component_name)
        results.append(component_value)
    if expected_trend == "increasing":
        for i in range(1, len(results)):
            test_case.assertGreaterEqual(
                results[i],
                results[i - 1] - config.tolerance_relaxed,
                f"{component_name} should increase with parameter variations",
            )
    elif expected_trend == "decreasing":
        for i in range(1, len(results)):
            test_case.assertLessEqual(
                results[i],
                results[i - 1] + config.tolerance_relaxed,
                f"{component_name} should decrease with parameter variations",
            )
    elif expected_trend == "constant":
        baseline = results[0]
        for result in results[1:]:
            test_case.assertAlmostEqual(
                result,
                baseline,
                delta=config.tolerance_relaxed,
                msg=f"{component_name} should remain constant with parameter variations",
            )


def make_idle_penalty_test_contexts(
    context_factory_fn,
    idle_duration_scenarios: Sequence[int],
    base_context_kwargs: Dict[str, Any] | None = None,
):
    """Generate contexts for idle penalty testing with varying durations.

    Factory function that creates a list of (context, description) tuples for
    idle penalty scenario testing, reducing boilerplate in test setup.

    Args:
        context_factory_fn: Factory function that creates context objects
        idle_duration_scenarios: Sequence of idle duration values to test
        base_context_kwargs: Base kwargs merged with idle_duration for each scenario

    Returns:
        List of (context, description) tuples

    Example:
        contexts = make_idle_penalty_test_contexts(
            make_context, [0, 50, 100, 200],
            base_context_kwargs={"context_type": "idle"}
        )
        for context, desc in contexts:
            breakdown = calculate_reward(context, ...)
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
    pnl_target: float,
    context,
    attenuation_modes: Sequence[str],
    base_params_fn,
    tolerance_relaxed: float,
    risk_reward_ratio: float = 1.0,
):
    """Validate exit factor attenuation across multiple modes.

    Tests that exit factor decreases monotonically (attenuates) over duration
    for various attenuation modes: linear, power, half_life, sqrt, and plateau_linear.

    Args:
        test_case: Test case instance with assertion methods
        base_factor: Base scaling factor
        pnl: Realized profit/loss
        pnl_target: Target profit threshold (pnl_target = profit_aim × risk_reward_ratio)
        context: RewardContext for efficiency coefficient calculation
        attenuation_modes: List of mode names to test
        base_params_fn: Factory function for creating parameter dicts
        tolerance_relaxed: Numerical tolerance for monotonicity checks

    Supported modes:
        - "plateau_linear": Linear attenuation after grace period
        - "linear": Linear attenuation with configurable slope
        - "power": Power-law attenuation with tau parameter
        - "half_life": Exponential decay with half-life parameter
        - "sqrt": Square root attenuation (default fallback)

    Example:
        assert_exit_factor_attenuation_modes(
            self, 90.0, 0.08, 0.03, context,
            ["linear", "power", "half_life"],
            make_params, 1e-09
        )
    """
    import numpy as np

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
                _get_exit_factor(
                    base_factor, pnl, pnl_target, r, context, mode_params, risk_reward_ratio
                )
                for r in ratios
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
                    later, earlier + tolerance_relaxed, f"Non-monotonic attenuation in mode={mode}"
                )


def assert_exit_mode_mathematical_validation(
    test_case,
    context,
    params: Dict[str, Any],
    base_factor: float,
    profit_aim: float,
    risk_reward_ratio: float,
    tolerance_relaxed: float,
):
    """Validate mathematical correctness of exit factor calculation modes.

    Performs deep mathematical validation of exit factor attenuation modes,
    including verification of half-life exponential decay formula and
    ensuring different modes produce distinct results.

    Args:
        test_case: Test case instance with assertion methods
        context: Context object with trade_duration and pnl attributes
        params: Parameter dictionary (will be modified in-place for testing)
        base_factor: Base scaling factor
        profit_aim: Base profit target
        risk_reward_ratio: Risk/reward ratio
        tolerance_relaxed: Numerical tolerance for formula validation

    Tests performed:
        1. Power mode produces positive exit component
        2. Half-life mode matches theoretical exponential decay formula
        3. Linear mode produces positive exit component
        4. Different modes produce distinguishable results

    Example:
        assert_exit_mode_mathematical_validation(
            self, context, params, PARAMS.BASE_FACTOR, PARAMS.PROFIT_AIM,
            PARAMS.RISK_REWARD_RATIO, TOLERANCE.IDENTITY_RELAXED
        )
    """
    duration_ratio = context.trade_duration / 100
    params["exit_attenuation_mode"] = "power"
    params["exit_power_tau"] = 0.5
    params["exit_plateau"] = False
    reward_power = calculate_reward(
        context,
        params,
        base_factor=base_factor,
        profit_aim=profit_aim,
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
        profit_aim=profit_aim,
        risk_reward_ratio=risk_reward_ratio,
        short_allowed=True,
        action_masking=True,
    )
    pnl_target = profit_aim * risk_reward_ratio
    pnl_target_coefficient = _compute_pnl_target_coefficient(
        params, context.pnl, pnl_target, risk_reward_ratio
    )
    efficiency_coefficient = _compute_efficiency_coefficient(params, context, context.pnl)
    pnl_coefficient = pnl_target_coefficient * efficiency_coefficient
    observed_exit_factor = _get_exit_factor(
        base_factor, context.pnl, pnl_target, duration_ratio, context, params, risk_reward_ratio
    )
    observed_half_life_factor = observed_exit_factor / (
        base_factor * max(pnl_coefficient, np.finfo(float).eps)
    )
    expected_half_life_factor = 2 ** (-duration_ratio / params["exit_half_life"])
    test_case.assertAlmostEqual(
        observed_half_life_factor,
        expected_half_life_factor,
        delta=tolerance_relaxed,
        msg="Half-life attenuation mismatch: observed vs expected",
    )
    params["exit_attenuation_mode"] = "linear"
    params["exit_linear_slope"] = 1.0
    reward_linear = calculate_reward(
        context,
        params,
        base_factor=base_factor,
        profit_aim=profit_aim,
        risk_reward_ratio=risk_reward_ratio,
        short_allowed=True,
        action_masking=True,
    )
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
    config: RewardScenarioConfig,
):
    """Validate reward behavior across multiple parameter combinations.

    Tests reward calculation with various profit_aim and risk_reward_ratio
    combinations, ensuring consistent behavior including edge cases like
    zero profit_aim. Uses RewardScenarioConfig to simplify parameter passing.

    Args:
        test_case: Test case instance with assertion methods
        parameter_test_cases: List of (profit_aim, risk_reward_ratio, description) tuples
        context_factory_fn: Factory function for creating context objects
        base_params: Base parameter dictionary
        config: RewardScenarioConfig with base calculation parameters

    Example:
        config = RewardScenarioConfig(
            base_factor=PARAMS.BASE_FACTOR,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
            tolerance_relaxed=TOLERANCE.IDENTITY_RELAXED
        )
        test_cases = [
            (0.0, PARAMS.RISK_REWARD_RATIO, "zero profit target"),
            (PARAMS.PROFIT_AIM, PARAMS.RISK_REWARD_RATIO, "standard parameters"),
            (0.03, 2.0, "high risk/reward ratio"),
        ]
        assert_multi_parameter_sensitivity(
            self, test_cases, make_context, params, config
        )
    """
    for profit_aim, risk_reward_ratio, description in parameter_test_cases:
        with test_case.subTest(
            profit_aim=profit_aim, risk_reward_ratio=risk_reward_ratio, desc=description
        ):
            idle_context = context_factory_fn(context_type="idle")
            breakdown = calculate_reward(
                idle_context,
                base_params,
                base_factor=config.base_factor,
                profit_aim=profit_aim,
                risk_reward_ratio=risk_reward_ratio,
                short_allowed=config.short_allowed,
                action_masking=config.action_masking,
            )
            if profit_aim == 0.0:
                test_case.assertEqual(breakdown.idle_penalty, 0.0)
                test_case.assertEqual(breakdown.total, 0.0)
            else:
                test_case.assertLess(breakdown.idle_penalty, 0.0)
            if profit_aim > 0:
                exit_context = context_factory_fn(context_type="exit", profit_aim=profit_aim)
                exit_breakdown = calculate_reward(
                    exit_context,
                    base_params,
                    base_factor=config.base_factor,
                    profit_aim=profit_aim,
                    risk_reward_ratio=risk_reward_ratio,
                    short_allowed=config.short_allowed,
                    action_masking=config.action_masking,
                )
                test_case.assertNotEqual(exit_breakdown.exit_component, 0.0)


def assert_hold_penalty_threshold_behavior(
    test_case,
    context_factory_fn,
    params: Dict[str, Any],
    base_factor: float,
    profit_aim: float,
    risk_reward_ratio: float,
    config: ThresholdTestConfig,
):
    """Validate hold penalty activation at max_duration threshold.

    Tests that hold penalty is zero before max_duration, then becomes
    negative (penalty) at and after the threshold. Uses ThresholdTestConfig
    to simplify parameter passing.

    Args:
        test_case: Test case instance with assertion methods
        context_factory_fn: Factory function for creating context objects
        params: Parameter dictionary
        base_factor: Base scaling factor
        profit_aim: Base profit target
        risk_reward_ratio: Risk/reward ratio
        config: ThresholdTestConfig with threshold settings

    Example:
        config = ThresholdTestConfig(
            max_duration=100,
            test_cases=[
                (50, "below threshold"),
                (100, "at threshold"),
                (150, "above threshold"),
            ],
            tolerance=TOLERANCE.IDENTITY_RELAXED
        )
        assert_hold_penalty_threshold_behavior(
            self, make_context, params, PARAMS.BASE_FACTOR, PARAMS.PROFIT_AIM,
            PARAMS.RISK_REWARD_RATIO, config
        )
    """
    for trade_duration, description in config.test_cases:
        with test_case.subTest(duration=trade_duration, desc=description):
            context = context_factory_fn(trade_duration=trade_duration)
            breakdown = calculate_reward(
                context,
                params,
                base_factor=base_factor,
                profit_aim=profit_aim,
                risk_reward_ratio=risk_reward_ratio,
                short_allowed=True,
                action_masking=True,
            )
            duration_ratio = trade_duration / config.max_duration
            if duration_ratio < 1.0:
                test_case.assertEqual(breakdown.hold_penalty, 0.0)
            elif duration_ratio == 1.0:
                test_case.assertLessEqual(breakdown.hold_penalty, 0.0)
            else:
                test_case.assertLess(breakdown.hold_penalty, 0.0)


# ---------------- Validation & invariance helper cases ---------------- #


def build_validation_case(
    param_updates: Dict[str, Any],
    strict: bool,
    expect_error: bool = False,
    expected_reason_substrings: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Build a structured validation test case descriptor.

    Creates a standardized test case dictionary for parameter validation testing,
    supporting both strict (raise on error) and relaxed (adjust and warn) modes.

    Args:
        param_updates: Dictionary of parameter updates to apply
        strict: If True, validation should raise on invalid params
        expect_error: If True, expect validation to raise an exception
        expected_reason_substrings: Substrings expected in adjustment reasons (relaxed mode)

    Returns:
        Dictionary with keys: params, strict, expect_error, expected_reason_substrings

    Example:
        case = build_validation_case(
            {"exit_plateau_grace": -0.5},
            strict=False,
            expected_reason_substrings=["clamped", "exit_plateau_grace"]
        )
    """
    return {
        "params": param_updates,
        "strict": strict,
        "expect_error": expect_error,
        "expected_reason_substrings": list(expected_reason_substrings or []),
    }


def execute_validation_batch(test_case, cases: Sequence[Dict[str, Any]], validate_fn):
    """Execute a batch of parameter validation test cases.

    Runs multiple validation scenarios in batch, handling both strict (error-raising)
    and relaxed (adjustment-collecting) modes. Validates that adjustment reasons
    contain expected substrings in relaxed mode.

    Args:
        test_case: Test case instance with assertion methods
        cases: Sequence of validation case dictionaries from build_validation_case
        validate_fn: Validation function to test (typically validate_reward_parameters)

    Example:
        cases = [
            build_validation_case({"exit_power_tau": -1.0}, strict=True, expect_error=True),
            build_validation_case({"exit_power_tau": -1.0}, strict=False,
                                 expected_reason_substrings=["clamped"]),
        ]
        execute_validation_batch(self, cases, validate_reward_parameters)
    """
    for idx, case in enumerate(cases):
        with test_case.subTest(
            case_index=idx, strict=case["strict"], expect_error=case["expect_error"]
        ):
            params = case["params"].copy()
            strict_flag = case["strict"]
            if strict_flag and case["expect_error"]:
                test_case.assertRaises(Exception, validate_fn, params, True)
                continue
            result = validate_fn(params, strict=strict_flag)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
                sanitized, adjustments = result
            else:
                sanitized, adjustments = result, {}
            # relaxed reason substrings
            for substr in case.get("expected_reason_substrings", []):
                # search across all adjustment reasons
                found = any(substr in adj.get("reason", "") for adj in adjustments.values())
                test_case.assertTrue(
                    found, f"Expected substring '{substr}' in some adjustment reason"
                )
            # basic sanity: sanitized returns a dict
            test_case.assertIsInstance(sanitized, dict)


def assert_adjustment_reason_contains(
    test_case, adjustments: Dict[str, Dict[str, Any]], key: str, expected_substrings: Sequence[str]
):
    """Assert adjustment reason contains all expected substrings.

    Validates that all expected substrings appear in the adjustment reason
    message for a specific parameter key, regardless of order.

    Args:
        test_case: Test case instance with assertion methods
        adjustments: Dictionary of adjustment information from validation
        key: Parameter key to check in adjustments dict
        expected_substrings: List of substrings that must appear in reason

    Example:
        adjustments = {
            "exit_plateau_grace": {
                "reason": "clamped to valid range [0.0, 1.0]",
                "validation_mode": "relaxed"
            }
        }
        assert_adjustment_reason_contains(
            self, adjustments, "exit_plateau_grace", ["clamped", "valid range"]
        )
    """
    test_case.assertIn(key, adjustments, f"Adjustment key '{key}' missing")
    reason = adjustments[key].get("reason", "")
    for sub in expected_substrings:
        test_case.assertIn(sub, reason, f"Missing substring '{sub}' in reason for key '{key}'")


def run_strict_validation_failure_cases(
    test_case, failure_params_list: Sequence[Dict[str, Any]], validate_fn
):
    """Batch test strict validation failures.

    Runs multiple parameter dictionaries through validation in strict mode,
    asserting that each raises a ValueError. Reduces boilerplate for testing
    multiple invalid parameter combinations.

    Args:
        test_case: Test case instance with assertion methods
        failure_params_list: List of parameter dicts that should fail validation
        validate_fn: Validation function to test

    Example:
        invalid_params = [
            {"exit_power_tau": -1.0},
            {"exit_plateau_grace": 1.5},
            {"exit_half_life": 0.0},
        ]
        run_strict_validation_failure_cases(
            self, invalid_params, validate_reward_parameters
        )
    """
    for params in failure_params_list:
        with test_case.subTest(params=params):
            test_case.assertRaises(ValueError, validate_fn, params, True)


def run_relaxed_validation_adjustment_cases(
    test_case,
    relaxed_cases: Sequence[Tuple[Dict[str, Any], Sequence[str]]],
    validate_fn,
):
    """Batch test relaxed validation adjustments.

    Runs multiple parameter dictionaries through validation in relaxed mode,
    asserting that adjustment reasons contain expected substrings. Validates
    that the system properly adjusts and reports issues rather than raising.

    Args:
        test_case: Test case instance with assertion methods
        relaxed_cases: List of (params, expected_reason_substrings) tuples
        validate_fn: Validation function to test

    Example:
        relaxed_cases = [
            ({"exit_power_tau": -1.0}, ["clamped", "tau"]),
            ({"exit_plateau_grace": 1.5}, ["clamped", "grace"]),
        ]
        run_relaxed_validation_adjustment_cases(
            self, relaxed_cases, validate_reward_parameters
        )
    """
    for params, substrings in relaxed_cases:
        with test_case.subTest(params=params):
            sanitized, adjustments = validate_fn(params, strict=False)
            test_case.assertIsInstance(sanitized, dict)
            test_case.assertIsInstance(adjustments, dict)
            # aggregate reasons
            all_reasons = ",".join(adj.get("reason", "") for adj in adjustments.values())
            for s in substrings:
                test_case.assertIn(
                    s, all_reasons, f"Expected '{s}' in aggregated adjustment reasons"
                )


def assert_exit_factor_invariant_suite(
    test_case, suite_cases: Sequence[Dict[str, Any]], exit_factor_fn
):
    """Validate exit factor invariants across multiple scenarios.

    Batch validation of exit factor behavior under various conditions,
    checking different invariants like non-negativity, safe zero handling,
    and clamping behavior.

    Args:
        test_case: Test case instance with assertion methods
        suite_cases: List of scenario dicts with keys:
            - base_factor: Base scaling factor
            - pnl: Realized profit/loss
            - pnl_target: Target profit threshold (pnl_target = profit_aim × risk_reward_ratio) for coefficient calculation
            - context: RewardContext for efficiency coefficient
            - duration_ratio: Duration ratio (0-2)
            - params: Parameter dictionary
            - expectation: Expected invariant ("non_negative", "safe_zero", "clamped")
            - tolerance: Optional numerical tolerance
        exit_factor_fn: Exit factor calculation function to test

    Example:
        cases = [
            {
                "base_factor": 90.0, "pnl": 0.08, "pnl_target": 0.03,
                "context": RewardContext(...),
                "duration_ratio": 0.5, "params": {...},
                "expectation": "non_negative", "tolerance": 1e-09
            },
            {
                "base_factor": 90.0, "pnl": 0.0, "pnl_target": 0.03,
                "context": RewardContext(...),
                "duration_ratio": 0.5, "params": {...},
                "expectation": "safe_zero"
            },
        ]
        assert_exit_factor_invariant_suite(self, cases, _get_exit_factor)
    """
    for i, case in enumerate(suite_cases):
        with test_case.subTest(exit_case=i, expectation=case.get("expectation")):
            f_val = exit_factor_fn(
                base_factor=case["base_factor"],
                pnl=case["pnl"],
                pnl_target=case["pnl_target"],
                duration_ratio=case["duration_ratio"],
                context=case["context"],
                params=case["params"],
                risk_reward_ratio=2.0,
            )
            exp = case.get("expectation")
            if exp == "safe_zero":
                test_case.assertEqual(f_val, 0.0)
            elif exp == "non_negative":
                test_case.assertGreaterEqual(f_val, -case.get("tolerance", 0.0))
            elif exp == "clamped":
                test_case.assertGreaterEqual(f_val, 0.0)
            else:
                test_case.fail(f"Unknown expectation '{exp}' in exit factor suite case")


def assert_exit_factor_kernel_fallback(
    test_case,
    exit_factor_fn,
    base_factor: float,
    pnl: float,
    pnl_target: float,
    duration_ratio: float,
    context,
    bad_params: Dict[str, Any],
    reference_params: Dict[str, Any],
    risk_reward_ratio: float,
):
    """Validate exit factor fallback behavior on kernel failure.

    Tests that when an attenuation kernel fails (e.g., invalid parameters),
    the system falls back to linear mode and produces numerically equivalent
    results. Caller must monkeypatch the kernel to trigger failure before calling.

    Args:
        test_case: Test case instance with assertion methods
        exit_factor_fn: Exit factor calculation function (e.g., _get_exit_factor)
        base_factor: Base scaling factor
        pnl: Realized profit/loss
        pnl_target: Target PnL (profit_aim * risk_reward_ratio)
        duration_ratio: Duration ratio
        context: RewardContext instance
        bad_params: Parameters that trigger kernel failure
        reference_params: Reference linear mode parameters for comparison
        risk_reward_ratio: Risk/reward ratio

    Validates:
        1. Fallback produces non-negative result
        2. Fallback result matches linear reference within tight tolerance (1e-12)

    Note:
        Warning emission should be validated separately with warning context managers.

    Example:
        # After monkeypatching kernel to fail:
        test_context = RewardContext(pnl=0.08, ...)
        assert_exit_factor_kernel_fallback(
            self, _get_exit_factor, 90.0, 0.08, 0.03, 0.5, test_context,
            bad_params={"exit_attenuation_mode": "power", "exit_power_tau": -1.0},
            reference_params={"exit_attenuation_mode": "linear"},
            risk_reward_ratio=1.0
        )
    """

    f_bad = exit_factor_fn(
        base_factor, pnl, pnl_target, duration_ratio, context, bad_params, risk_reward_ratio
    )
    f_ref = exit_factor_fn(
        base_factor, pnl, pnl_target, duration_ratio, context, reference_params, risk_reward_ratio
    )
    test_case.assertAlmostEqual(f_bad, f_ref, delta=TOLERANCE.IDENTITY_STRICT)
    test_case.assertGreaterEqual(f_bad, 0.0)


def assert_relaxed_multi_reason_aggregation(
    test_case,
    validate_fn,
    params: Dict[str, Any],
    key_expectations: Dict[str, Sequence[str]],
):
    """Validate relaxed validation produces expected adjustment reasons.

    Tests that relaxed validation properly aggregates and reports multiple
    adjustment reasons for specified parameter keys, ensuring transparency
    in parameter sanitization.

    Args:
        test_case: Test case instance with assertion methods
        validate_fn: Validation function to test
        params: Parameter dictionary to validate
        key_expectations: Mapping of param_key -> expected reason substrings

    Example:
        key_expectations = {
            "exit_power_tau": ["clamped", "minimum"],
            "exit_plateau_grace": ["clamped", "range"],
        }
        assert_relaxed_multi_reason_aggregation(
            self, validate_reward_parameters, params, key_expectations
        )
    """
    sanitized, adjustments = validate_fn(params, strict=False)
    test_case.assertIsInstance(sanitized, dict)
    for k, subs in key_expectations.items():
        test_case.assertIn(k, adjustments, f"Missing adjustment for key '{k}'")
        reason = adjustments[k].get("reason", "")
        for sub in subs:
            test_case.assertIn(sub, reason, f"Expected substring '{sub}' in reason for key '{k}'")
        test_case.assertEqual(adjustments[k].get("validation_mode"), "relaxed")


def assert_pbrs_invariance_report_classification(
    test_case, content: str, expected_status: str, expect_additives: bool
):
    """Validate PBRS invariance report classification and additive reporting.

    Checks that the invariance report correctly classifies PBRS behavior
    and appropriately reports additive component involvement.

    Args:
        test_case: Test case instance with assertion methods
        content: Report content string to validate
        expected_status: Expected classification: "Canonical",
                        "Canonical (with warning)", or "Non-canonical"
        expect_additives: Whether additive components should be mentioned

    Example:
        assert_pbrs_invariance_report_classification(
            self, report_content, "Canonical", expect_additives=False
        )
        assert_pbrs_invariance_report_classification(
            self, report_content, "Non-canonical", expect_additives=True
        )
    """
    test_case.assertIn(
        expected_status, content, f"Expected invariance status '{expected_status}' not found"
    )
    if expect_additives:
        test_case.assertRegex(
            content, r"additives=\['entry', 'exit'\]|additives=\['exit', 'entry'\]"
        )
    else:
        test_case.assertNotRegex(content, r"additives=\[")


def assert_pbrs_canonical_sum_within_tolerance(test_case, total_shaping: float, tolerance: float):
    """Validate cumulative PBRS shaping satisfies canonical bound.

    For canonical PBRS, the cumulative reward shaping across a trajectory
    must be near zero (within tolerance). This is a core PBRS invariant.

    Args:
        test_case: Test case instance with assertion methods
        total_shaping: Total cumulative reward shaping value
        tolerance: Maximum allowed absolute deviation from zero

    Example:
        assert_pbrs_canonical_sum_within_tolerance(self, 5e-10, 1e-09)
    """
    test_case.assertLess(abs(total_shaping), tolerance)


def assert_non_canonical_shaping_exceeds(
    test_case, total_shaping: float, tolerance_multiple: float
):
    """Validate non-canonical PBRS shaping exceeds threshold.

    For non-canonical PBRS (e.g., with additives), the cumulative shaping
    should exceed a scaled tolerance threshold, indicating violation of
    the canonical PBRS invariant.

    Args:
        test_case: Test case instance with assertion methods
        total_shaping: Total cumulative reward shaping value
        tolerance_multiple: Threshold value (typically scaled tolerance)

    Example:
        # Expect shaping to exceed 10x tolerance for non-canonical case
        assert_non_canonical_shaping_exceeds(self, 0.05, 1e-08)
    """
    test_case.assertGreater(abs(total_shaping), tolerance_multiple)


def assert_exit_factor_plateau_behavior(
    test_case,
    exit_factor_fn,
    base_factor: float,
    pnl: float,
    pnl_target: float,
    context: RewardContext,
    plateau_params: dict,
    grace: float,
    tolerance_strict: float,
    risk_reward_ratio: float,
):
    """Assert plateau behavior: factor before grace >= factor after grace (attenuation begins after grace boundary).

    Args:
        test_case: Test case instance with assertion methods
        exit_factor_fn: Exit factor calculation function (_get_exit_factor)
        base_factor: Base factor for exit calculation
        pnl: PnL value
        pnl_target: Target profit threshold (pnl_target = profit_aim × risk_reward_ratio) for coefficient calculation
        context: RewardContext for efficiency coefficient
        plateau_params: Parameters dict with plateau configuration
        grace: Grace period threshold (exit_plateau_grace value)
        tolerance_strict: Tolerance for numerical comparisons
    """
    # Test points: one before grace, one after grace
    duration_ratio_pre = grace - 0.1 if grace >= 0.1 else grace * 0.5
    duration_ratio_post = grace + 0.3

    plateau_factor_pre = exit_factor_fn(
        base_factor=base_factor,
        pnl=pnl,
        pnl_target=pnl_target,
        duration_ratio=duration_ratio_pre,
        context=context,
        params=plateau_params,
        risk_reward_ratio=risk_reward_ratio,
    )
    plateau_factor_post = exit_factor_fn(
        base_factor=base_factor,
        pnl=pnl,
        pnl_target=pnl_target,
        duration_ratio=duration_ratio_post,
        context=context,
        params=plateau_params,
        risk_reward_ratio=risk_reward_ratio,
    )

    # Both factors should be positive
    test_case.assertGreater(plateau_factor_pre, 0, "Pre-grace factor should be positive")
    test_case.assertGreater(plateau_factor_post, 0, "Post-grace factor should be positive")

    # Pre-grace factor should be >= post-grace factor (attenuation begins after grace)
    test_case.assertGreaterEqual(
        plateau_factor_pre,
        plateau_factor_post - tolerance_strict,
        "Plateau pre-grace factor should be >= post-grace factor",
    )
