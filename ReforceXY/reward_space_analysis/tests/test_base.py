#!/usr/bin/env python3
"""Base class and utilities for reward space analysis tests."""

import math
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from reward_space_analysis import (
    DEFAULT_MODEL_REWARD_PARAMETERS,
    Actions,
    Positions,
    RewardContext,
    apply_potential_shaping,
)

from .constants import (
    CONTINUITY,
    EXIT_FACTOR,
    PBRS,
    SCENARIOS,
    SEEDS,
    STATISTICAL,
    TOLERANCE,
)

# Global constants
PBRS_INTEGRATION_PARAMS = [
    "potential_gamma",
    "hold_potential_enabled",
    "hold_potential_scale",
    "entry_additive_enabled",
    "exit_additive_enabled",
]
PBRS_REQUIRED_PARAMS = PBRS_INTEGRATION_PARAMS + ["exit_potential_mode"]


class RewardSpaceTestBase(unittest.TestCase):
    """Base class with common test utilities."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level constants."""
        cls.SEED = SEEDS.BASE
        cls.DEFAULT_PARAMS = DEFAULT_MODEL_REWARD_PARAMETERS.copy()
        cls.TEST_SAMPLES = SCENARIOS.SAMPLE_SIZE_TINY
        cls.TEST_BASE_FACTOR = 100.0
        cls.TEST_PROFIT_TARGET = 0.03
        cls.TEST_RR = 1.0
        cls.TEST_RR_HIGH = 2.0
        cls.TEST_PNL_STD = 0.02
        cls.TEST_PNL_DUR_VOL_SCALE = 0.5
        # Seeds for different test contexts
        cls.SEED_SMOKE_TEST = SEEDS.SMOKE_TEST
        cls.SEED_REPRODUCIBILITY = SEEDS.REPRODUCIBILITY
        cls.SEED_BOOTSTRAP = SEEDS.BOOTSTRAP
        cls.SEED_HETEROSCEDASTICITY = SEEDS.HETEROSCEDASTICITY
        # Statistical test thresholds
        cls.BOOTSTRAP_DEFAULT_ITERATIONS = SCENARIOS.BOOTSTRAP_EXTENDED_ITERATIONS
        cls.BH_FP_RATE_THRESHOLD = STATISTICAL.BH_FP_RATE_THRESHOLD
        cls.EXIT_FACTOR_SCALING_RATIO_MIN = EXIT_FACTOR.SCALING_RATIO_MIN
        cls.EXIT_FACTOR_SCALING_RATIO_MAX = EXIT_FACTOR.SCALING_RATIO_MAX

    def setUp(self):
        """Set up test fixtures with reproducible random seed."""
        self.seed_all(self.SEED)
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ===============================================
    # Constants imported from tests.constants module
    # ===============================================

    # Tolerance constants
    TOL_IDENTITY_STRICT = TOLERANCE.IDENTITY_STRICT
    TOL_IDENTITY_RELAXED = TOLERANCE.IDENTITY_RELAXED
    TOL_GENERIC_EQ = TOLERANCE.GENERIC_EQ
    TOL_NUMERIC_GUARD = TOLERANCE.NUMERIC_GUARD
    TOL_NEGLIGIBLE = TOLERANCE.NEGLIGIBLE
    TOL_RELATIVE = TOLERANCE.RELATIVE
    TOL_DISTRIB_SHAPE = TOLERANCE.DISTRIB_SHAPE

    # PBRS constants
    PBRS_TERMINAL_TOL = PBRS.TERMINAL_TOL
    PBRS_MAX_ABS_SHAPING = PBRS.MAX_ABS_SHAPING

    # Continuity constants
    CONTINUITY_EPS_SMALL = CONTINUITY.EPS_SMALL
    CONTINUITY_EPS_LARGE = CONTINUITY.EPS_LARGE

    # Exit factor constants
    MIN_EXIT_POWER_TAU = EXIT_FACTOR.MIN_POWER_TAU

    # Test-specific constants
    PBRS_TERMINAL_PROB = PBRS.TERMINAL_PROBABILITY
    PBRS_SWEEP_ITER = SCENARIOS.PBRS_SWEEP_ITERATIONS
    JS_DISTANCE_UPPER_BOUND = math.sqrt(math.log(2.0))

    def make_ctx(
        self,
        *,
        pnl: float = 0.0,
        trade_duration: int = 0,
        idle_duration: int = 0,
        max_unrealized_profit: float = 0.0,
        min_unrealized_profit: float = 0.0,
        position: Positions = Positions.Neutral,
        action: Actions = Actions.Neutral,
    ) -> RewardContext:
        """Create a RewardContext with neutral defaults."""
        return RewardContext(
            pnl=pnl,
            trade_duration=trade_duration,
            idle_duration=idle_duration,
            max_unrealized_profit=max_unrealized_profit,
            min_unrealized_profit=min_unrealized_profit,
            position=position,
            action=action,
        )

    def base_params(self, **overrides) -> Dict[str, Any]:
        """Return fresh copy of default reward params with overrides."""
        params: Dict[str, Any] = DEFAULT_MODEL_REWARD_PARAMETERS.copy()
        params.update(overrides)
        return params

    def _canonical_sweep(
        self,
        params: dict,
        *,
        iterations: Optional[int] = None,
        terminal_prob: Optional[float] = None,
        seed: int = SEEDS.CANONICAL_SWEEP,
    ) -> tuple[list[float], list[float]]:
        """Run a lightweight canonical invariance sweep.

        Returns (terminal_next_potentials, shaping_values).
        """
        iters = iterations or self.PBRS_SWEEP_ITER
        term_p = terminal_prob or self.PBRS_TERMINAL_PROB
        rng = np.random.default_rng(seed)
        last_potential = 0.0
        terminal_next: list[float] = []
        shaping_vals: list[float] = []
        current_pnl = 0.0
        current_dur = 0.0
        for _ in range(iters):
            is_exit = rng.uniform() < term_p
            next_pnl = 0.0 if is_exit else float(rng.normal(0, 0.2))
            inc = rng.uniform(0, 0.12)
            next_dur = 0.0 if is_exit else float(min(1.0, current_dur + inc))
            _tot, shap_val, next_pot, _pbrs_delta, _entry_additive, _exit_additive = (
                apply_potential_shaping(
                    base_reward=0.0,
                    current_pnl=current_pnl,
                    current_duration_ratio=current_dur,
                    next_pnl=next_pnl,
                    next_duration_ratio=next_dur,
                    is_exit=is_exit,
                    is_entry=False,
                    last_potential=last_potential,
                    params=params,
                )
            )
            shaping_vals.append(shap_val)
            if is_exit:
                terminal_next.append(next_pot)
                last_potential = 0.0
                current_pnl = 0.0
                current_dur = 0.0
            else:
                last_potential = next_pot
                current_pnl = next_pnl
                current_dur = next_dur
        return (terminal_next, shaping_vals)

    def make_stats_df(
        self,
        *,
        n: int,
        reward_mean: float = 0.0,
        reward_std: float = 1.0,
        pnl_mean: float = 0.01,
        pnl_std: Optional[float] = None,
        trade_duration_dist: str = "uniform",
        idle_pattern: str = "mixed",
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate a synthetic statistical DataFrame.

        Parameters
        ----------
        n : int
            Row count.
        reward_mean, reward_std : float
            Normal parameters for reward.
        pnl_mean : float
            Mean PnL.
        pnl_std : float | None
            PnL std (defaults to TEST_PNL_STD when None).
        trade_duration_dist : {'uniform','exponential'}
            Distribution family for trade_duration.
        idle_pattern : {'mixed','all_nonzero','all_zero'}
            mixed: ~40% idle>0 (U(1,60)); all_nonzero: all idle>0 (U(5,60)); all_zero: idle=0.
        seed : int | None
            RNG seed.

        Returns
        -------
        pd.DataFrame with columns: reward, reward_idle, reward_hold, reward_exit,
        pnl, trade_duration, idle_duration, position. Guarantees: no NaN; reward_idle==0 where idle_duration==0.
        """
        if seed is not None:
            self.seed_all(seed)
        pnl_std_eff = self.TEST_PNL_STD if pnl_std is None else pnl_std
        reward = np.random.normal(reward_mean, reward_std, n)
        pnl = np.random.normal(pnl_mean, pnl_std_eff, n)
        if trade_duration_dist == "exponential":
            trade_duration = np.random.exponential(20, n)
        else:
            trade_duration = np.random.uniform(5, 150, n)
        if idle_pattern == "mixed":
            mask = np.random.rand(n) < 0.4
            idle_duration = np.where(mask, np.random.uniform(1, 60, n), 0.0)
        elif idle_pattern == "all_zero":
            idle_duration = np.zeros(n)
        else:
            idle_duration = np.random.uniform(5, 60, n)
        reward_idle = np.where(idle_duration > 0, np.random.normal(-1, 0.3, n), 0.0)
        reward_hold = np.random.normal(-0.5, 0.2, n)
        reward_exit = np.random.normal(0.8, 0.6, n)
        position = np.random.choice([0.0, 0.5, 1.0], n)
        return pd.DataFrame(
            {
                "reward": reward,
                "reward_idle": reward_idle,
                "reward_hold": reward_hold,
                "reward_exit": reward_exit,
                "pnl": pnl,
                "trade_duration": trade_duration,
                "idle_duration": idle_duration,
                "position": position,
            }
        )

    def assertAlmostEqualFloat(
        self,
        first: Union[float, int],
        second: Union[float, int],
        tolerance: Optional[float] = None,
        rtol: Optional[float] = None,
        msg: Union[str, None] = None,
    ) -> None:
        """Compare floats with absolute and optional relative tolerance.

        precedence:
          1. absolute tolerance (|a-b| <= tolerance)
          2. relative tolerance (|a-b| <= rtol * max(|a|, |b|)) if provided
        Both may be supplied; failure only if both criteria fail (when rtol given).
        """
        self.assertFinite(first, name="a")
        self.assertFinite(second, name="b")
        if tolerance is None:
            tolerance = self.TOL_GENERIC_EQ
        diff = abs(first - second)
        if diff <= tolerance:
            return
        if rtol is not None:
            scale = max(abs(first), abs(second), self.TOL_NEGLIGIBLE)
            if diff <= rtol * scale:
                return
        self.fail(
            msg
            or f"Difference {diff} exceeds tolerance {tolerance} and relative tolerance {rtol} (a={first}, b={second})"
        )

    def assertPValue(self, value: Union[float, int], msg: str = "") -> None:
        """Assert a p-value is finite and within [0,1]."""
        self.assertFinite(value, name="p-value")
        self.assertGreaterEqual(value, 0.0, msg or f"p-value < 0: {value}")
        self.assertLessEqual(value, 1.0, msg or f"p-value > 1: {value}")

    def assertPlacesEqual(
        self, a: Union[float, int], b: Union[float, int], places: int, msg: Optional[str] = None
    ) -> None:
        """Bridge for legacy places-based approximate equality.

        Converts decimal places to an absolute tolerance 10**-places and delegates to
        assertAlmostEqualFloat to keep a single numeric comparison implementation.
        """
        tol = 10.0 ** (-places)
        self.assertAlmostEqualFloat(a, b, tolerance=tol, msg=msg)

    def assertDistanceMetric(
        self,
        value: Union[float, int],
        *,
        non_negative: bool = True,
        upper: Optional[float] = None,
        name: str = "metric",
    ) -> None:
        """Generic distance/divergence bounds: finite, optional non-negativity and optional upper bound."""
        self.assertFinite(value, name=name)
        if non_negative:
            self.assertGreaterEqual(value, 0.0, f"{name} negative: {value}")
        if upper is not None:
            self.assertLessEqual(value, upper, f"{name} > {upper}: {value}")

    def assertEffectSize(
        self,
        value: Union[float, int],
        *,
        lower: float = -1.0,
        upper: float = 1.0,
        name: str = "effect size",
    ) -> None:
        """Assert effect size within symmetric interval and finite."""
        self.assertFinite(value, name=name)
        self.assertGreaterEqual(value, lower, f"{name} < {lower}: {value}")
        self.assertLessEqual(value, upper, f"{name} > {upper}: {value}")

    def assertFinite(self, value: Union[float, int], name: str = "value") -> None:
        """Assert scalar is finite."""
        if not np.isfinite(value):
            self.fail(f"{name} not finite: {value}")

    def assertMonotonic(
        self,
        seq: Union[Sequence[Union[float, int]], Iterable[Union[float, int]]],
        *,
        non_increasing: Optional[bool] = None,
        non_decreasing: Optional[bool] = None,
        tolerance: float = 0.0,
        name: str = "sequence",
    ) -> None:
        """Assert a sequence is monotonic under specified direction.

        Provide exactly one of non_increasing/non_decreasing=True.
        tolerance allows tiny positive drift in expected monotone direction.
        """
        data = list(seq)
        if len(data) < 2:
            return
        if non_increasing and non_decreasing or (not non_increasing and (not non_decreasing)):
            self.fail("Specify exactly one monotonic direction")
        for a, b in zip(data, data[1:]):
            if non_increasing:
                if b > a + tolerance:
                    self.fail(f"{name} not non-increasing at pair ({a}, {b})")
            elif non_decreasing:
                if b + tolerance < a:
                    self.fail(f"{name} not non-decreasing at pair ({a}, {b})")

    def assertWithin(
        self,
        value: Union[float, int],
        low: Union[float, int],
        high: Union[float, int],
        *,
        name: str = "value",
        inclusive: bool = True,
    ) -> None:
        """Assert that value is within [low, high] (inclusive) or (low, high) if inclusive=False."""
        self.assertFinite(value, name=name)
        if inclusive:
            self.assertGreaterEqual(value, low, f"{name} < {low}")
            self.assertLessEqual(value, high, f"{name} > {high}")
        else:
            self.assertGreater(value, low, f"{name} <= {low}")
            self.assertLess(value, high, f"{name} >= {high}")

    def assertNearZero(
        self, value: Union[float, int], *, atol: Optional[float] = None, msg: Optional[str] = None
    ) -> None:
        """Assert a scalar is numerically near zero within absolute tolerance.

        Uses strict identity tolerance by default for PBRS invariance style checks.
        """
        self.assertFinite(value, name="value")
        tol = atol if atol is not None else self.TOL_IDENTITY_RELAXED
        if abs(float(value)) > tol:
            self.fail(msg or f"Value {value} not near zero (tol={tol})")

    def assertSymmetric(
        self,
        func,
        a,
        b,
        *,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        msg: Optional[str] = None,
    ) -> None:
        """Assert function(func, a, b) == function(func, b, a) within tolerance.

        Intended for symmetric distance metrics (e.g., JS distance).
        """
        va = func(a, b)
        vb = func(b, a)
        self.assertAlmostEqualFloat(va, vb, tolerance=atol, rtol=rtol, msg=msg)

    @staticmethod
    def seed_all(seed: int = SEEDS.CANONICAL_SWEEP) -> None:
        """Seed all RNGs used (numpy & random)."""
        np.random.seed(seed)
        random.seed(seed)

    def _const_df(self, n: int = SCENARIOS.SAMPLE_SIZE_CONST_DF) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "reward": np.ones(n) * 0.5,
                "pnl": np.zeros(n),
                "trade_duration": np.ones(n) * 10,
                "idle_duration": np.ones(n) * 3,
            }
        )

    def _shift_scale_df(
        self, n: int = SCENARIOS.SAMPLE_SIZE_SHIFT_SCALE, shift: float = 0.0, scale: float = 1.0
    ) -> pd.DataFrame:
        rng = np.random.default_rng(SEEDS.CANONICAL_SWEEP)
        base = rng.normal(0, 1, n)
        return pd.DataFrame(
            {
                "reward": shift + scale * base,
                "pnl": shift + scale * base * 0.2,
                "trade_duration": rng.exponential(20, n),
                "idle_duration": rng.exponential(10, n),
            }
        )

    def _make_idle_variance_df(self, n: int = 100) -> pd.DataFrame:
        """Synthetic dataframe focusing on idle_duration ↔ reward_idle correlation."""
        self.seed_all(self.SEED)
        idle_duration = np.random.exponential(10, n)
        reward_idle = -0.01 * idle_duration + np.random.normal(0, 0.001, n)
        return pd.DataFrame(
            {
                "idle_duration": idle_duration,
                "reward_idle": reward_idle,
                "position": np.random.choice([0.0, 0.5, 1.0], n),
                "reward": np.random.normal(0, 1, n),
                "pnl": np.random.normal(0, self.TEST_PNL_STD, n),
                "trade_duration": np.random.exponential(20, n),
            }
        )
