#!/usr/bin/env python3
"""Test constants and configuration values.

This module serves as the single source of truth for all test constants,
following the DRY principle and repository conventions.

All numeric tolerances, seeds, and test parameters are defined here with
clear documentation of their purpose and usage context.
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ToleranceConfig:
    """Numerical tolerance configuration for assertions.

    These tolerances are used throughout the test suite for floating-point
    comparisons, ensuring consistent precision requirements across all tests.

    Attributes:
        IDENTITY_STRICT: Machine-precision tolerance for identity checks (1e-12)
        IDENTITY_RELAXED: Relaxed tolerance for approximate identity (1e-09)
        GENERIC_EQ: Generic equality tolerance for float comparisons (1e-08)
        NUMERIC_GUARD: Minimum threshold to prevent division by zero (1e-18)
        NEGLIGIBLE: Threshold below which values are considered negligible (1e-15)
        RELATIVE: Relative tolerance for ratio/percentage comparisons (1e-06)
        DISTRIB_SHAPE: Tolerance for distribution shape metrics (skew, kurtosis) (0.15)
    """

    IDENTITY_STRICT: float = 1e-12
    IDENTITY_RELAXED: float = 1e-09
    GENERIC_EQ: float = 1e-08
    NUMERIC_GUARD: float = 1e-18
    NEGLIGIBLE: float = 1e-15
    RELATIVE: float = 1e-06
    DISTRIB_SHAPE: float = 0.15


@dataclass(frozen=True)
class ContinuityConfig:
    """Continuity and smoothness testing configuration.

    Epsilon values for testing continuity at boundaries, particularly for
    plateau and attenuation functions.

    Attributes:
        EPS_SMALL: Small epsilon for tight continuity checks (1e-08)
        EPS_LARGE: Larger epsilon for coarser continuity tests (5e-05)
    """

    EPS_SMALL: float = 1e-08
    EPS_LARGE: float = 5e-05


@dataclass(frozen=True)
class ExitFactorConfig:
    """Exit factor scaling and validation configuration.

    Configuration for exit factor behavior validation, including scaling
    ratio bounds and power mode constraints.

    Attributes:
        SCALING_RATIO_MIN: Minimum expected scaling ratio for continuity (1.5)
        SCALING_RATIO_MAX: Maximum expected scaling ratio for continuity (3.5)
        MIN_POWER_TAU: Minimum valid tau value for power mode (1e-15)
    """

    SCALING_RATIO_MIN: float = 1.5
    SCALING_RATIO_MAX: float = 3.5
    MIN_POWER_TAU: float = 1e-15


@dataclass(frozen=True)
class PBRSConfig:
    """Potential-Based Reward Shaping (PBRS) configuration.

    Thresholds and bounds for PBRS invariance validation and testing.

    Attributes:
        TERMINAL_TOL: Terminal potential must be within this tolerance of zero (1e-09)
        MAX_ABS_SHAPING: Maximum absolute shaping value for bounded checks (10.0)
    """

    TERMINAL_TOL: float = 1e-09
    MAX_ABS_SHAPING: float = 10.0


@dataclass(frozen=True)
class StatisticalConfig:
    """Statistical testing configuration.

    Configuration for statistical hypothesis testing, bootstrap methods,
    and distribution comparisons.

    Attributes:
        BH_FP_RATE_THRESHOLD: Benjamini-Hochberg false positive rate threshold (0.15)
        BOOTSTRAP_DEFAULT_ITERATIONS: Default bootstrap resampling count (100)
    """

    BH_FP_RATE_THRESHOLD: float = 0.15
    BOOTSTRAP_DEFAULT_ITERATIONS: int = 100


@dataclass(frozen=True)
class TestSeeds:
    """Random seed values for reproducible testing.

    Each seed serves a specific purpose to ensure test reproducibility while
    maintaining statistical independence across different test scenarios.

    Seed Strategy:
        - BASE: Default seed for general-purpose tests, ensuring stable baseline
        - REPRODUCIBILITY: Used exclusively for reproducibility validation tests
        - BOOTSTRAP: Prime number for bootstrap confidence interval tests to ensure
          independence from other random sequences
        - HETEROSCEDASTICITY: Dedicated seed for variance structure validation tests

    Attributes:
        BASE: Default seed for standard tests (42)
        REPRODUCIBILITY: Seed for reproducibility validation (12345)
        BOOTSTRAP: Seed for bootstrap CI tests (999)
        HETEROSCEDASTICITY: Seed for heteroscedasticity tests (7890)
    """

    BASE: int = 42
    REPRODUCIBILITY: int = 12345
    BOOTSTRAP: int = 999
    HETEROSCEDASTICITY: int = 7890


@dataclass(frozen=True)
class TestParameters:
    """Standard test parameter values.

    Default parameter values used consistently across the test suite for
    reward calculation and simulation.

    Attributes:
        BASE_FACTOR: Default base factor for reward scaling (90.0)
        PROFIT_TARGET: Target profit threshold (0.06)
        RISK_REWARD_RATIO: Standard risk/reward ratio (1.0)
        RISK_REWARD_RATIO_HIGH: High risk/reward ratio for stress tests (2.0)
        PNL_STD: Standard deviation for PnL generation (0.02)
        PNL_DUR_VOL_SCALE: Duration-based volatility scaling factor (0.001)
        EPS_BASE: Base epsilon for near-zero checks (1e-10)
    """

    BASE_FACTOR: float = 90.0
    PROFIT_TARGET: float = 0.06
    RISK_REWARD_RATIO: float = 1.0
    RISK_REWARD_RATIO_HIGH: float = 2.0
    PNL_STD: float = 0.02
    PNL_DUR_VOL_SCALE: float = 0.001
    EPS_BASE: float = 1e-10


# Global singleton instances for easy import
TOLERANCE: Final[ToleranceConfig] = ToleranceConfig()
CONTINUITY: Final[ContinuityConfig] = ContinuityConfig()
EXIT_FACTOR: Final[ExitFactorConfig] = ExitFactorConfig()
PBRS: Final[PBRSConfig] = PBRSConfig()
STATISTICAL: Final[StatisticalConfig] = StatisticalConfig()
SEEDS: Final[TestSeeds] = TestSeeds()
PARAMS: Final[TestParameters] = TestParameters()


__all__ = [
    "ToleranceConfig",
    "ContinuityConfig",
    "ExitFactorConfig",
    "PBRSConfig",
    "StatisticalConfig",
    "TestSeeds",
    "TestParameters",
    "TOLERANCE",
    "CONTINUITY",
    "EXIT_FACTOR",
    "PBRS",
    "STATISTICAL",
    "SEEDS",
    "PARAMS",
]
