"""Helpers package for reward_space_analysis tests.

Exposes shared assertion utilities, configuration dataclasses, and warning
capture helpers, centralizing test infrastructure and reducing duplication.
"""

from .assertions import (
    assert_adjustment_reason_contains,
    # Core numeric/trend assertions
    assert_almost_equal_list,
    assert_component_sum_integrity,
    assert_exit_factor_attenuation_modes,
    # Exit factor invariance helpers
    assert_exit_factor_invariant_suite,
    assert_exit_factor_kernel_fallback,
    assert_exit_factor_plateau_behavior,
    assert_exit_mode_mathematical_validation,
    assert_finite,
    assert_hold_penalty_threshold_behavior,
    assert_monotonic_nonincreasing,
    assert_monotonic_nonnegative,
    assert_multi_parameter_sensitivity,
    assert_non_canonical_shaping_exceeds,
    assert_parameter_sensitivity_behavior,
    assert_pbrs_canonical_sum_within_tolerance,
    # PBRS invariance/report helpers
    assert_pbrs_invariance_report_classification,
    assert_progressive_scaling_behavior,
    # Relaxed validation aggregation
    assert_relaxed_multi_reason_aggregation,
    assert_reward_calculation_scenarios,
    assert_single_active_component,
    assert_single_active_component_with_additives,
    assert_trend,
    # Validation batch builders/executors
    build_validation_case,
    execute_validation_batch,
    make_idle_penalty_test_contexts,
    run_relaxed_validation_adjustment_cases,
    run_strict_validation_failure_cases,
    safe_float,
)
from .configs import (
    ContextFactory,
    ExitFactorConfig,
    ProgressiveScalingConfig,
    # Configuration dataclasses
    RewardScenarioConfig,
    SimulationConfig,
    StatisticalTestConfig,
    ThresholdTestConfig,
    # Type aliases
    ValidationCallback,
    ValidationConfig,
    WarningCaptureConfig,
)
from .warnings import (
    assert_diagnostic_warning,
    assert_no_warnings,
    # Warning capture utilities
    capture_warnings,
    validate_warning_content,
)

__all__ = [
    # Core numeric/trend assertions
    "assert_monotonic_nonincreasing",
    "assert_monotonic_nonnegative",
    "assert_finite",
    "assert_almost_equal_list",
    "assert_trend",
    "assert_component_sum_integrity",
    "assert_progressive_scaling_behavior",
    "assert_single_active_component",
    "assert_single_active_component_with_additives",
    "assert_reward_calculation_scenarios",
    "assert_parameter_sensitivity_behavior",
    "make_idle_penalty_test_contexts",
    "assert_exit_factor_attenuation_modes",
    "assert_exit_factor_plateau_behavior",
    "assert_exit_mode_mathematical_validation",
    "assert_multi_parameter_sensitivity",
    "assert_hold_penalty_threshold_behavior",
    "safe_float",
    # Validation batch builders/executors
    "build_validation_case",
    "execute_validation_batch",
    "assert_adjustment_reason_contains",
    "run_strict_validation_failure_cases",
    "run_relaxed_validation_adjustment_cases",
    # Exit factor invariance helpers
    "assert_exit_factor_invariant_suite",
    "assert_exit_factor_kernel_fallback",
    # Relaxed validation aggregation
    "assert_relaxed_multi_reason_aggregation",
    # PBRS invariance/report helpers
    "assert_pbrs_invariance_report_classification",
    "assert_pbrs_canonical_sum_within_tolerance",
    "assert_non_canonical_shaping_exceeds",
    # Configuration dataclasses
    "RewardScenarioConfig",
    "ValidationConfig",
    "ThresholdTestConfig",
    "ProgressiveScalingConfig",
    "ExitFactorConfig",
    "StatisticalTestConfig",
    "SimulationConfig",
    "WarningCaptureConfig",
    "ValidationCallback",
    "ContextFactory",
    # Warning capture utilities
    "capture_warnings",
    "assert_diagnostic_warning",
    "assert_no_warnings",
    "validate_warning_content",
]
