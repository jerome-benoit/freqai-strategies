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
        IDENTITY_STRICT: Tolerance for strict identity checks (1e-12)
        IDENTITY_RELAXED: Tolerance for relaxed identity checks (1e-09)
        GENERIC_EQ: General-purpose equality tolerance (1e-08)
        NUMERIC_GUARD: Minimum threshold to prevent division by zero (1e-18)
        NEGLIGIBLE: Threshold below which values are considered negligible (1e-15)
        RELATIVE: Relative tolerance for ratio/percentage comparisons (1e-06)
        INTEGRATION_RELATIVE_COARSE: Coarse relative tolerance for integration smoke checks (0.25)
        DISTRIB_SHAPE: Tolerance for distribution shape metrics (skew, kurtosis) (0.15)
        DECIMAL_PLACES_STRICT: Decimal places for exact formula validation (12)
        DECIMAL_PLACES_STANDARD: Decimal places for general calculations (9)
        DECIMAL_PLACES_RELAXED: Decimal places for accumulated operations (6)
        DECIMAL_PLACES_DATA_LOADING: Decimal places for data loading/casting tests (7)

        # Additional tolerances for specific test scenarios
        ALPHA_ATTENUATION_STRICT: Strict tolerance for alpha attenuation tests (5e-12)
        ALPHA_ATTENUATION_RELAXED: Relaxed tolerance for alpha attenuation with tau != 1.0 (5e-09)
        SHAPING_BOUND_TOLERANCE: Tolerance for bounded shaping checks (0.2)
    """

    IDENTITY_STRICT: float = 1e-12
    IDENTITY_RELAXED: float = 1e-09
    GENERIC_EQ: float = 1e-08
    NUMERIC_GUARD: float = 1e-18
    NEGLIGIBLE: float = 1e-15
    RELATIVE: float = 1e-06
    INTEGRATION_RELATIVE_COARSE: float = 0.25
    DISTRIB_SHAPE: float = 0.15
    DECIMAL_PLACES_STRICT: int = 12
    DECIMAL_PLACES_STANDARD: int = 9
    DECIMAL_PLACES_RELAXED: int = 6
    DECIMAL_PLACES_DATA_LOADING: int = 7

    # Additional tolerances
    ALPHA_ATTENUATION_STRICT: float = 5e-12
    ALPHA_ATTENUATION_RELAXED: float = 5e-09
    SHAPING_BOUND_TOLERANCE: float = 0.2


@dataclass(frozen=True)
class ContinuityConfig:
    """Continuity and smoothness testing configuration.

    Epsilon values for testing continuity at boundaries, particularly for
    plateau and attenuation functions.

    Attributes:
        EPS_SMALL: Small epsilon for tight continuity checks (1e-06)
        EPS_LARGE: Larger epsilon for coarser continuity tests (1e-05)
        BOUND_MULTIPLIER_LINEAR: Linear mode derivative bound multiplier (2.0)
        BOUND_MULTIPLIER_SQRT: Sqrt mode derivative bound multiplier (2.0)
        BOUND_MULTIPLIER_POWER: Power mode derivative bound multiplier (2.0)
        BOUND_MULTIPLIER_HALF_LIFE: Half-life mode derivative bound multiplier (2.5)
    """

    EPS_SMALL: float = 1e-06
    EPS_LARGE: float = 1e-05
    BOUND_MULTIPLIER_LINEAR: float = 2.0
    BOUND_MULTIPLIER_SQRT: float = 2.0
    BOUND_MULTIPLIER_POWER: float = 2.0
    BOUND_MULTIPLIER_HALF_LIFE: float = 2.5


@dataclass(frozen=True)
class ExitFactorConfig:
    """Exit factor scaling and validation configuration.

    Configuration for exit factor behavior validation, including scaling
    ratio bounds and power mode constraints.

    Attributes:
        SCALING_RATIO_MIN: Minimum expected scaling ratio for continuity (5.0)
        SCALING_RATIO_MAX: Maximum expected scaling ratio for continuity (15.0)
        MIN_POWER_TAU: Minimum valid tau value for power mode (1e-15)
    """

    SCALING_RATIO_MIN: float = 5.0
    SCALING_RATIO_MAX: float = 15.0
    MIN_POWER_TAU: float = 1e-15


@dataclass(frozen=True)
class PBRSConfig:
    """Potential-Based Reward Shaping (PBRS) configuration.

    Thresholds and bounds for PBRS invariance validation and testing.

    Attributes:
        TERMINAL_TOL: Terminal potential must be within this tolerance of zero (1e-09)
        MAX_ABS_SHAPING: Maximum absolute shaping value for bounded checks (50.0)
        TERMINAL_PROBABILITY: Default probability of terminal state in sweeps (0.08)
    """

    TERMINAL_TOL: float = 1e-09
    MAX_ABS_SHAPING: float = 50.0
    TERMINAL_PROBABILITY: float = 0.08


@dataclass(frozen=True)
class StatisticalConfig:
    """Statistical testing configuration.

    Configuration for statistical hypothesis testing, bootstrap methods,
    and distribution comparisons.

    Attributes:
        BH_FP_RATE_THRESHOLD: Benjamini-Hochberg false positive rate threshold (0.15)
        BOOTSTRAP_DEFAULT_ITERATIONS: Default bootstrap resampling count (100)
        EXIT_PROBABILITY_THRESHOLD: Probability threshold for exit events (0.15)
    """

    BH_FP_RATE_THRESHOLD: float = 0.15
    BOOTSTRAP_DEFAULT_ITERATIONS: int = 100
    EXIT_PROBABILITY_THRESHOLD: float = 0.15


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
        - SMOKE_TEST: Seed for smoke tests
        - CANONICAL_SWEEP: Seed for canonical PBRS sweep tests

    Attributes:
        BASE: Default seed for standard tests (42)
        REPRODUCIBILITY: Seed for reproducibility validation (12345)
        BOOTSTRAP: Seed for bootstrap CI tests (999)
        HETEROSCEDASTICITY: Seed for heteroscedasticity tests (7890)
        SMOKE_TEST: Seed for smoke tests (7)
        CANONICAL_SWEEP: Seed for canonical sweep tests (123)

        # PBRS-specific seeds
        PBRS_INVARIANCE_1: Seed for PBRS invariance test case 1 (913)
        PBRS_INVARIANCE_2: Seed for PBRS invariance test case 2 (515)
        PBRS_TERMINAL: Seed for PBRS terminal potential tests (777)

        # Feature analysis failure seeds
        FEATURE_EMPTY: Seed for empty feature tests (17)
        FEATURE_PRIME_7: Seed for feature test variant (7)
        FEATURE_PRIME_11: Seed for feature test variant (11)
        FEATURE_PRIME_13: Seed for feature test variant (13)
        FEATURE_PRIME_21: Seed for feature test variant (21)
        FEATURE_PRIME_33: Seed for feature test variant (33)
        FEATURE_PRIME_47: Seed for feature test variant (47)
        FEATURE_SMALL_5: Seed for small feature test (5)
        FEATURE_SMALL_3: Seed for small feature test (3)

        # Report formatting seeds
        REPORT_FORMAT_1: Seed for report formatting test 1 (234)
        REPORT_FORMAT_2: Seed for report formatting test 2 (321)

        # Additional seeds for various test scenarios
        ALTERNATE_1: Alternate seed for robustness tests (555)
        ALTERNATE_2: Alternate seed for variance tests (808)
    """

    BASE: int = 42
    REPRODUCIBILITY: int = 12345
    BOOTSTRAP: int = 999
    HETEROSCEDASTICITY: int = 7890
    SMOKE_TEST: int = 7
    CANONICAL_SWEEP: int = 123

    # PBRS-specific seeds
    PBRS_INVARIANCE_1: int = 913
    PBRS_INVARIANCE_2: int = 515
    PBRS_TERMINAL: int = 777

    # Feature analysis failure seeds
    FEATURE_EMPTY: int = 17
    FEATURE_PRIME_7: int = 7
    FEATURE_PRIME_11: int = 11
    FEATURE_PRIME_13: int = 13
    FEATURE_PRIME_21: int = 21
    FEATURE_PRIME_33: int = 33
    FEATURE_PRIME_47: int = 47
    FEATURE_SMALL_5: int = 5
    FEATURE_SMALL_3: int = 3

    # Report formatting seeds
    REPORT_FORMAT_1: int = 234
    REPORT_FORMAT_2: int = 321

    # Additional seeds
    ALTERNATE_1: int = 555
    ALTERNATE_2: int = 808


@dataclass(frozen=True)
class TestParameters:
    """Standard test parameter values.

    Default parameter values used consistently across the test suite for
    reward calculation and simulation.

    Attributes:
        BASE_FACTOR: Default base factor for reward scaling (90.0)
        PROFIT_AIM: Target profit threshold (0.06)
        RISK_REWARD_RATIO: Standard risk/reward ratio (1.0)
        RISK_REWARD_RATIO_HIGH: High risk/reward ratio for stress tests (2.0)
        PNL_STD: Standard deviation for PnL generation (0.02)
        PNL_DUR_VOL_SCALE: Duration-based volatility scaling factor (0.001)

        # Common test PnL values
        PNL_TINY: Tiny profit/loss value (0.01)
        PNL_SMALL: Small profit/loss value (0.02)
        PNL_SHORT_PROFIT: Short profit/loss value (0.03)
        PNL_MEDIUM: Medium profit/loss value (0.05)
        PNL_LARGE: Large profit/loss value (0.10)

        # Common duration values
        TRADE_DURATION_SHORT: Short trade duration in steps (50)
        TRADE_DURATION_MEDIUM: Medium trade duration in steps (100)
        TRADE_DURATION_LONG: Long trade duration in steps (200)

        # Simulation configuration
        MAX_TRADE_DURATION_HETEROSCEDASTICITY: Max trade duration used for heteroscedasticity tests (10)

        # Common additive parameters
        ADDITIVE_RATIO_DEFAULT: Default additive ratio (0.4)
        ADDITIVE_GAIN_DEFAULT: Default additive gain (1.0)
    """

    BASE_FACTOR: float = 90.0
    PROFIT_AIM: float = 0.06
    RISK_REWARD_RATIO: float = 1.0
    RISK_REWARD_RATIO_HIGH: float = 2.0
    PNL_STD: float = 0.02
    PNL_DUR_VOL_SCALE: float = 0.001

    # Common PnL values
    PNL_TINY: float = 0.01
    PNL_SMALL: float = 0.02
    PNL_SHORT_PROFIT: float = 0.03
    PNL_MEDIUM: float = 0.05
    PNL_LARGE: float = 0.10

    # Common duration values
    TRADE_DURATION_SHORT: int = 50
    TRADE_DURATION_MEDIUM: int = 100
    TRADE_DURATION_LONG: int = 200

    # Simulation configuration
    MAX_TRADE_DURATION_HETEROSCEDASTICITY: int = 10

    # Additive parameters
    ADDITIVE_RATIO_DEFAULT: float = 0.4
    ADDITIVE_GAIN_DEFAULT: float = 1.0


@dataclass(frozen=True)
class TestScenarios:
    """Test scenario parameters and sample sizes.

    Standard values for test scenarios to ensure consistency across the test
    suite and avoid magic numbers in test implementations.

    Attributes:
        DURATION_SHORT: Short duration scenario (150)
        DURATION_MEDIUM: Medium duration scenario (200)
        DURATION_LONG: Long duration scenario (300)
        DURATION_SCENARIOS: Standard duration test sequence
        SAMPLE_SIZE_TINY: Tiny sample size for smoke tests (50)
        SAMPLE_SIZE_SMALL: Small sample size for quick tests (100)
        SAMPLE_SIZE_MEDIUM: Medium sample size for standard tests (400)
        SAMPLE_SIZE_LARGE: Large sample size for statistical power (800)
        SAMPLE_SIZE_CONST_DF: Sample size for constant dataframes (64)
        SAMPLE_SIZE_SHIFT_SCALE: Sample size for shift/scale tests (256)
        PBRS_SIMULATION_STEPS: Number of steps for PBRS simulation tests (500)
        MONTE_CARLO_ITERATIONS: Monte Carlo simulation iterations (160)
        PBRS_SWEEP_ITERATIONS: Number of iterations for PBRS sweep tests (120)
        BOOTSTRAP_MINIMAL_ITERATIONS: Minimal bootstrap iterations for quick tests (25)
        BOOTSTRAP_EXTENDED_ITERATIONS: Extended bootstrap iterations (200)
        SAMPLE_SIZE_REPORT_MINIMAL: Minimal sample size for report smoke tests (10)
        REPORT_DURATION_SCALE_UP: Duration scale applied to synthetic real episodes (1.01)
        REPORT_DURATION_SCALE_DOWN: Duration scale applied to synthetic real episodes (0.99)
        REPORT_PNL_MEAN_SHIFT: PnL mean shift applied to synthetic real episodes (0.001)

        # API smoke parameters
        API_MAX_IDLE_DURATION_CANDLES: Idle duration cap used in _sample_action tests (20)
        API_IDLE_DURATION_HIGH: High idle duration used to trigger hazard (60)
        API_ENTRY_RATE_DRAWS: Draw count for entry-rate estimation (2000)
        API_MAX_TRADE_DURATION_CANDLES: Max trade duration used in API simulation tests (40)
        API_MAX_DURATION_RATIO: Max duration ratio used in API simulation tests (1.5)
        API_PROBABILITY_UPPER_BOUND: Upper bound for exposed sampling probabilities (0.9)
        API_EXTREME_BASE_FACTOR: Extreme base_factor used to trigger warning paths (10000000.0)

        # CLI smoke parameters
        CLI_NUM_SAMPLES_STANDARD: Default CLI sample size for smoke runs (200)
        CLI_NUM_SAMPLES_REPORT: CLI sample size used in PBRS report smoke (180)
        CLI_NUM_SAMPLES_HASH: CLI sample size used for params_hash checks (150)
        CLI_NUM_SAMPLES_FAST: CLI sample size for quick branch coverage (120)
        CLI_RISK_REWARD_RATIO_NON_DEFAULT: Non-default risk/reward ratio for manifest hashing (1.5)
        CLI_MAX_TRADE_DURATION_PARAMS: CLI max_trade_duration_candles for --params propagation (96)
        CLI_MAX_TRADE_DURATION_FLAG: CLI max_trade_duration_candles for dynamic flag propagation (64)
    """

    DURATION_SHORT: int = 150
    DURATION_MEDIUM: int = 200
    DURATION_LONG: int = 300
    DURATION_SCENARIOS: tuple[int, ...] = (150, 200, 300)

    SAMPLE_SIZE_TINY: int = 50
    SAMPLE_SIZE_SMALL: int = 100
    SAMPLE_SIZE_MEDIUM: int = 400
    SAMPLE_SIZE_LARGE: int = 800
    SAMPLE_SIZE_CONST_DF: int = 64
    SAMPLE_SIZE_SHIFT_SCALE: int = 256

    # Specialized test scenario sizes
    PBRS_SIMULATION_STEPS: int = 500
    MONTE_CARLO_ITERATIONS: int = 160
    PBRS_SWEEP_ITERATIONS: int = 120
    BOOTSTRAP_MINIMAL_ITERATIONS: int = 25
    BOOTSTRAP_EXTENDED_ITERATIONS: int = 200
    SAMPLE_SIZE_REPORT_MINIMAL: int = 10
    REPORT_DURATION_SCALE_UP: float = 1.01
    REPORT_DURATION_SCALE_DOWN: float = 0.99
    REPORT_PNL_MEAN_SHIFT: float = 0.001

    # API smoke parameters
    API_MAX_IDLE_DURATION_CANDLES: int = 20
    API_IDLE_DURATION_HIGH: int = 60
    API_ENTRY_RATE_DRAWS: int = 2000
    API_MAX_TRADE_DURATION_CANDLES: int = 40
    API_MAX_DURATION_RATIO: float = 1.5
    API_PROBABILITY_UPPER_BOUND: float = 0.9
    API_EXTREME_BASE_FACTOR: float = 10_000_000.0

    # CLI smoke parameters
    CLI_NUM_SAMPLES_STANDARD: int = 200
    CLI_NUM_SAMPLES_REPORT: int = 180
    CLI_NUM_SAMPLES_HASH: int = 150
    CLI_NUM_SAMPLES_FAST: int = 120
    CLI_RISK_REWARD_RATIO_NON_DEFAULT: float = 1.5
    CLI_MAX_TRADE_DURATION_PARAMS: int = 96
    CLI_MAX_TRADE_DURATION_FLAG: int = 64


@dataclass(frozen=True)
class StatisticalTolerances:
    """Tolerances for statistical metrics and distribution tests.

    These tolerances are used for statistical hypothesis testing, distribution
    comparison metrics, and other statistical validation operations.

    Attributes:
        DISTRIBUTION_SHIFT: Tolerance for distribution shift metrics (5e-4)
        KS_STATISTIC_IDENTITY: KS statistic threshold for identical distributions (5e-3)
        CORRELATION_SIGNIFICANCE: Minimum correlation for significance (0.1)
        VARIANCE_RATIO_THRESHOLD: Minimum variance ratio for heteroscedasticity (0.8)
        CI_WIDTH_EPSILON: Minimum CI width for degenerate distributions (3e-9)
    """

    DISTRIBUTION_SHIFT: float = 5e-4
    KS_STATISTIC_IDENTITY: float = 5e-3
    CORRELATION_SIGNIFICANCE: float = 0.1
    VARIANCE_RATIO_THRESHOLD: float = 0.8
    CI_WIDTH_EPSILON: float = 3e-9


# Global singleton instances for easy import
TOLERANCE: Final[ToleranceConfig] = ToleranceConfig()
CONTINUITY: Final[ContinuityConfig] = ContinuityConfig()
EXIT_FACTOR: Final[ExitFactorConfig] = ExitFactorConfig()
PBRS: Final[PBRSConfig] = PBRSConfig()
STATISTICAL: Final[StatisticalConfig] = StatisticalConfig()
SEEDS: Final[TestSeeds] = TestSeeds()
PARAMS: Final[TestParameters] = TestParameters()
SCENARIOS: Final[TestScenarios] = TestScenarios()
STAT_TOL: Final[StatisticalTolerances] = StatisticalTolerances()


__all__ = [
    "ToleranceConfig",
    "ContinuityConfig",
    "ExitFactorConfig",
    "PBRSConfig",
    "StatisticalConfig",
    "TestSeeds",
    "TestParameters",
    "TestScenarios",
    "StatisticalTolerances",
    "TOLERANCE",
    "CONTINUITY",
    "EXIT_FACTOR",
    "PBRS",
    "STATISTICAL",
    "SEEDS",
    "PARAMS",
    "SCENARIOS",
    "STAT_TOL",
]
