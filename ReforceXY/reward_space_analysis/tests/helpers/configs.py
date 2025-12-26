#!/usr/bin/env python3
"""Configuration dataclasses for test helpers.

This module provides strongly-typed configuration objects to simplify
function signatures in test helpers, following the DRY principle and
reducing parameter proliferation.

Usage:
    >>> from tests.helpers.configs import RewardScenarioConfig
    >>> from tests.constants import PARAMS, TOLERANCE

    >>> config = RewardScenarioConfig(
    ...     base_factor=PARAMS.BASE_FACTOR,
    ...     profit_aim=PARAMS.PROFIT_AIM,
    ...     risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
    ...     tolerance_relaxed=TOLERANCE.IDENTITY_RELAXED
    ... )

    >>> assert_reward_calculation_scenarios(
    ...     test_case, scenarios, config, validation_fn
    ... )
"""

from collections.abc import Callable
from dataclasses import dataclass

from ..constants import PARAMS, SEEDS, STATISTICAL, TOLERANCE


@dataclass
class RewardScenarioConfig:
    """Configuration for reward calculation scenario testing.

    Encapsulates all parameters needed for reward calculation validation,
    reducing function signature complexity and improving maintainability.

    Attributes:
        base_factor: Base scaling factor for reward calculations
        profit_aim: Base profit target
        risk_reward_ratio: Risk/reward ratio for position sizing
        tolerance_relaxed: Numerical tolerance for assertions
        short_allowed: Whether short positions are permitted
        action_masking: Whether to apply action masking
    """

    base_factor: float
    profit_aim: float
    risk_reward_ratio: float
    tolerance_relaxed: float
    short_allowed: bool = True
    action_masking: bool = True


@dataclass
class ValidationConfig:
    """Configuration for validation helper functions.

    Parameters controlling validation behavior, including tolerance levels
    and component exclusion policies.

    Attributes:
        tolerance_strict: Strict numerical tolerance (typically 1e-12)
        tolerance_relaxed: Relaxed numerical tolerance (typically 1e-09)
        exclude_components: List of component names to exclude from validation
        component_description: Human-readable description of validated components
    """

    tolerance_strict: float = TOLERANCE.IDENTITY_STRICT
    tolerance_relaxed: float = TOLERANCE.IDENTITY_RELAXED
    exclude_components: list[str] | None = None
    component_description: str = "reward components"


@dataclass
class ThresholdTestConfig:
    """Configuration for threshold behavior testing.

    Parameters for testing threshold-based behavior, such as hold penalty
    activation at max_duration boundaries.

    Attributes:
        max_duration: Maximum duration threshold
        test_cases: List of (duration, description) tuples for testing
        tolerance: Numerical tolerance for assertions
    """

    max_duration: int
    test_cases: list[tuple[int, str]]
    tolerance: float


@dataclass
class ProgressiveScalingConfig:
    """Configuration for progressive scaling validation.

    Parameters for validating that penalties or rewards scale progressively
    (monotonically) with increasing input values.

    Attributes:
        input_values: Sequence of input values to test (e.g., durations)
        expected_direction: "increasing" or "decreasing"
        tolerance: Numerical tolerance for monotonicity checks
        description: Human-readable description of what's being scaled
    """

    input_values: list[float]
    expected_direction: str  # "increasing" or "decreasing"
    tolerance: float
    description: str


@dataclass
class ExitFactorConfig:
    """Configuration for exit factor validation.

    Parameters specific to exit factor calculations, including coefficient
    decomposition, attenuation mode and plateau behavior.

    The exit factor is computed as:
        exit_factor = base_factor * time_attenuation * pnl_target * efficiency

    Attributes:
        base_factor: Base scaling factor
        pnl: Realized profit/loss
        pnl_target_coefficient: PnL target amplification coefficient (typically 0.5-2.0)
        efficiency_coefficient: Exit timing efficiency coefficient (typically 0.5-1.5)
        duration_ratio: Ratio of current to maximum duration
        attenuation_mode: Mode of attenuation ("linear", "power", etc.)
        plateau_enabled: Whether plateau behavior is active
        plateau_grace: Grace period before attenuation begins
        tolerance: Numerical tolerance for assertions
    """

    base_factor: float
    pnl: float
    pnl_target_coefficient: float
    efficiency_coefficient: float
    duration_ratio: float
    attenuation_mode: str
    plateau_enabled: bool = False
    plateau_grace: float = 0.0
    tolerance: float = TOLERANCE.IDENTITY_RELAXED


@dataclass
class StatisticalTestConfig:
    """Configuration for statistical hypothesis testing.

    Parameters for statistical validation, including bootstrap settings
    and hypothesis test configuration.

    Attributes:
        n_bootstrap: Number of bootstrap resamples
        confidence_level: Confidence level for intervals (0-1)
        seed: Random seed for reproducibility
        adjust_method: Multiple testing correction method
        alpha: Significance level
    """

    n_bootstrap: int = STATISTICAL.BOOTSTRAP_DEFAULT_ITERATIONS
    confidence_level: float = 0.95
    seed: int = SEEDS.BASE
    adjust_method: str | None = None
    alpha: float = 0.05


@dataclass
class SimulationConfig:
    """Configuration for reward simulation.

    Parameters controlling simulation behavior for generating synthetic
    test datasets.

    Attributes:
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        max_duration_ratio: Maximum duration ratio for trades
        trading_mode: Trading mode ("margin", "spot", etc.)
        pnl_base_std: Base standard deviation for PnL generation
        pnl_duration_vol_scale: Volatility scaling factor for duration
    """

    num_samples: int
    seed: int
    max_duration_ratio: float = 2.0
    trading_mode: str = "margin"
    pnl_base_std: float = 0.02
    pnl_duration_vol_scale: float = 0.001


@dataclass
class WarningCaptureConfig:
    """Configuration for warning capture helpers.

    Parameters controlling warning capture behavior in tests.

    Attributes:
        warning_category: Expected warning category class
        expected_substrings: List of substrings expected in warning messages
        strict_mode: If True, all expected substrings must be present
    """

    warning_category: type
    expected_substrings: list[str]
    strict_mode: bool = True


# Type aliases for common callback signatures
ValidationCallback = Callable[[object, object, str, float], None]
ContextFactory = Callable[..., object]


# Default config instances for common test scenarios
# These reduce boilerplate by providing pre-configured defaults

DEFAULT_REWARD_CONFIG: RewardScenarioConfig = RewardScenarioConfig(
    base_factor=PARAMS.BASE_FACTOR,
    profit_aim=PARAMS.PROFIT_AIM,
    risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
    tolerance_relaxed=TOLERANCE.IDENTITY_RELAXED,
    short_allowed=True,
    action_masking=True,
)
"""Default RewardScenarioConfig with standard test parameters."""


DEFAULT_SIMULATION_CONFIG: SimulationConfig = SimulationConfig(
    num_samples=200,
    seed=SEEDS.BASE,
    max_duration_ratio=2.0,
    trading_mode="margin",
    pnl_base_std=PARAMS.PNL_STD,
    pnl_duration_vol_scale=PARAMS.PNL_DUR_VOL_SCALE,
)
"""Default SimulationConfig with standard test parameters."""


__all__ = [
    "DEFAULT_REWARD_CONFIG",
    "DEFAULT_SIMULATION_CONFIG",
    "ContextFactory",
    "ExitFactorConfig",
    "ProgressiveScalingConfig",
    "RewardScenarioConfig",
    "SimulationConfig",
    "StatisticalTestConfig",
    "ThresholdTestConfig",
    "ValidationCallback",
    "ValidationConfig",
    "WarningCaptureConfig",
]
