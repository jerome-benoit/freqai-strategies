"""Helpers package for reward_space_analysis tests.

Exposes shared assertion utilities centralizing invariant logic.
"""

from .assertions import (
    assert_almost_equal_list,
    assert_component_sum_integrity,
    assert_exit_factor_attenuation_modes,
    assert_exit_mode_mathematical_validation,
    assert_finite,
    assert_hold_penalty_threshold_behavior,
    assert_monotonic_nonincreasing,
    assert_monotonic_nonnegative,
    assert_multi_parameter_sensitivity,
    assert_parameter_sensitivity_behavior,
    assert_progressive_scaling_behavior,
    assert_reward_calculation_scenarios,
    assert_single_active_component,
    assert_trend,
    make_idle_penalty_test_contexts,
)

__all__ = [
    "assert_monotonic_nonincreasing",
    "assert_monotonic_nonnegative",
    "assert_finite",
    "assert_almost_equal_list",
    "assert_trend",
    "assert_component_sum_integrity",
    "assert_progressive_scaling_behavior",
    "assert_single_active_component",
    "assert_reward_calculation_scenarios",
    "assert_parameter_sensitivity_behavior",
    "make_idle_penalty_test_contexts",
    "assert_exit_factor_attenuation_modes",
    "assert_exit_mode_mathematical_validation",
    "assert_multi_parameter_sensitivity",
    "assert_hold_penalty_threshold_behavior",
]
