#!/usr/bin/env python3
"""Synthetic reward space analysis utilities for ReforceXY.

Implements sample generation, reward computation with PBRS, statistical analysis,
and feature importance calculation with reproducible parameter hashing.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import numbers
import pickle
import random
import warnings
from enum import Enum, IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, probplot

if TYPE_CHECKING:
    from collections.abc import Iterable

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import partial_dependence, permutation_importance
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
except Exception:
    RandomForestRegressor = None
    partial_dependence = None
    permutation_importance = None
    r2_score = None
    train_test_split = None


AttenuationMode = Literal["sqrt", "linear", "power", "half_life"]
TransformFunction = Literal["tanh", "softsign", "arctan", "sigmoid", "clip", "asinh"]
ExitPotentialMode = Literal[
    "canonical", "non_canonical", "progressive_release", "spike_cancel", "retain_previous"
]


class Actions(IntEnum):
    Neutral = 0
    Long_enter = 1
    Long_exit = 2
    Short_enter = 3
    Short_exit = 4


class Positions(Enum):
    Short = 0
    Long = 1
    Neutral = 0.5


# Mathematical constants pre-computed for performance
_LOG_2 = math.log(2.0)

DEFAULT_IDLE_DURATION_MULTIPLIER = 4

# Tolerance for PBRS invariance classification.
#
# When `reward_invariance_correction` is available (reward_shaping - reward_pbrs_delta),
# canonical PBRS should satisfy max|correction| < PBRS_INVARIANCE_TOL.
#
# When that diagnostic column is not available (e.g., reporting from partial datasets),
# we fall back to the weaker heuristic |Σ shaping| < PBRS_INVARIANCE_TOL.
PBRS_INVARIANCE_TOL: float = 1e-6
# Default discount factor γ for potential-based reward shaping  # noqa: RUF003
POTENTIAL_GAMMA_DEFAULT: float = 0.95

# Default risk/reward ratio (RR)
RISK_REWARD_RATIO_DEFAULT: float = 2.0

# Supported attenuation modes
ATTENUATION_MODES: tuple[str, ...] = ("sqrt", "linear", "power", "half_life")
ATTENUATION_MODES_WITH_LEGACY: tuple[str, ...] = ("legacy", *ATTENUATION_MODES)

# Internal numeric guards and behavior toggles
INTERNAL_GUARDS: dict[str, float] = {
    "degenerate_ci_epsilon": 1e-9,
    "distribution_constant_fallback_moment": 0.0,
    "distribution_constant_fallback_qq_r2": 1.0,
    "moment_extreme_threshold": 1e4,
    "bootstrap_min_recommended": 200,
    "sim_pnl_conservation_tol": 1e-10,
    "sim_zero_pnl_epsilon": 1e-12,
    "sim_zero_reward_epsilon": 1e-12,
    "sim_extreme_pnl_threshold": 0.2,
    "histogram_epsilon": 1e-10,
    "distribution_identity_epsilon": 1e-12,
}

# PBRS constants
ALLOWED_TRANSFORMS = {
    "tanh",
    "softsign",
    "arctan",
    "sigmoid",
    "asinh",
    "clip",
}
ALLOWED_EXIT_POTENTIAL_MODES = {
    "canonical",
    "non_canonical",
    "progressive_release",
    "spike_cancel",
    "retain_previous",
}

# Supported trading modes
TRADING_MODES: tuple[str, ...] = ("spot", "margin", "futures")

# Supported p-value adjustment methods
ADJUST_METHODS: tuple[str, ...] = ("none", "benjamini_hochberg")
# Alias without underscore for convenience
_ADJUST_METHODS_ALIASES: frozenset[str] = frozenset({"benjaminihochberg"})

# ---------------------------------------------------------------------------
# Log message prefixes for consistent error/warning formatting
# ---------------------------------------------------------------------------
# Usage: f"{LOG_PREFIX_*}: <message>; falling back to <value>"
#
# Prefixes:
#   Param:  - Parameter validation/coercion issues
#   Data:   - Data loading/validation issues
#   Stats:  - Statistical computation issues
#   PBRS:   - Potential-Based Reward Shaping issues
#   Sim:    - Simulation invariant violations
#   CLI:    - Command-line argument issues
#   Config: - Trading configuration issues
# ---------------------------------------------------------------------------

DEFAULT_MODEL_REWARD_PARAMETERS: RewardParams = {
    "invalid_action": -2.0,
    "base_factor": 100.0,
    # Idle penalty defaults
    "idle_penalty_ratio": 1.0,
    "idle_penalty_power": 1.025,
    "max_trade_duration_candles": 128,
    # Fallback: DEFAULT_IDLE_DURATION_MULTIPLIER * max_trade_duration_candles
    "max_idle_duration_candles": None,
    # Hold penalty defaults
    "hold_penalty_ratio": 1.0,
    "hold_penalty_power": 1.025,
    # Exit attenuation defaults
    "exit_attenuation_mode": "linear",
    "exit_plateau": True,
    "exit_plateau_grace": 1.0,
    "exit_linear_slope": 1.0,
    "exit_power_tau": 0.5,
    "exit_half_life": 0.5,
    # Efficiency factor defaults
    "efficiency_weight": 1.0,
    "efficiency_center": 0.5,
    # Profit factor defaults
    "win_reward_factor": 2.0,
    "pnl_factor_beta": 0.5,
    # Invariant / safety defaults
    "check_invariants": True,
    "exit_factor_threshold": 1000.0,
    # === PBRS PARAMETERS ===
    # Potential-based reward shaping core parameters
    # Discount factor γ for potential term (0 ≤ γ ≤ 1)  # noqa: RUF003
    "potential_gamma": POTENTIAL_GAMMA_DEFAULT,
    # Exit potential modes: canonical | non_canonical | progressive_release | spike_cancel | retain_previous
    "exit_potential_mode": "canonical",
    "exit_potential_decay": 0.5,
    # Hold potential (PBRS function Φ)
    "hold_potential_enabled": True,
    "hold_potential_ratio": 0.015625,
    "hold_potential_gain": 1.0,
    "hold_potential_transform_pnl": "tanh",
    "hold_potential_transform_duration": "tanh",
    # Entry additive (non-PBRS additive term)
    "entry_additive_enabled": False,
    "entry_additive_ratio": 0.125,
    "entry_additive_gain": 1.0,
    "entry_additive_transform_pnl": "tanh",
    "entry_additive_transform_duration": "tanh",
    "entry_fee_rate": 0.0,
    "exit_fee_rate": 0.0,
    # Exit additive (non-PBRS additive term)
    "exit_additive_enabled": False,
    "exit_additive_ratio": 0.125,
    "exit_additive_gain": 1.0,
    "exit_additive_transform_pnl": "tanh",
    "exit_additive_transform_duration": "tanh",
}

DEFAULT_MODEL_REWARD_PARAMETERS_HELP: dict[str, str] = {
    "invalid_action": "Penalty for invalid actions",
    "base_factor": "Base reward scale",
    "idle_penalty_power": "Idle penalty exponent",
    "idle_penalty_ratio": "Idle penalty ratio",
    "max_trade_duration_candles": "Trade duration cap (candles)",
    "max_idle_duration_candles": "Idle duration cap (candles)",
    "hold_penalty_ratio": "Hold penalty ratio",
    "hold_penalty_power": "Hold penalty exponent",
    "exit_attenuation_mode": "Exit kernel (legacy|sqrt|linear|power|half_life)",
    "exit_plateau": "Use plateau before attenuation",
    "exit_plateau_grace": "Plateau grace duration ratio",
    "exit_linear_slope": "Linear kernel slope",
    "exit_power_tau": "Tau for power kernel (0,1]",
    "exit_half_life": "Half-life for exp kernel",
    "efficiency_weight": "Efficiency weight",
    "efficiency_center": "Efficiency pivot in [0,1]",
    "win_reward_factor": "Profit overshoot bonus factor",
    "pnl_factor_beta": "PnL amplification sensitivity",
    "check_invariants": "Enable runtime invariant checks",
    "exit_factor_threshold": "Warn if |exit_factor| exceeds",
    # PBRS parameters
    "potential_gamma": "PBRS discount γ (0-1)",  # noqa: RUF001
    "exit_potential_mode": "Exit potential mode (canonical|non_canonical|progressive_release|spike_cancel|retain_previous)",
    "exit_potential_decay": "Decay for progressive_release (0-1)",
    "hold_potential_enabled": "Enable hold potential Φ",
    "hold_potential_ratio": "Hold potential ratio",
    "hold_potential_gain": "Hold potential gain",
    "hold_potential_transform_pnl": "Hold PnL transform",
    "hold_potential_transform_duration": "Hold duration transform",
    "entry_additive_enabled": "Enable entry additive",
    "entry_additive_ratio": "Entry additive ratio",
    "entry_additive_gain": "Entry additive gain",
    "entry_additive_transform_pnl": "Entry PnL transform",
    "entry_additive_transform_duration": "Entry duration transform",
    "entry_fee_rate": "Entry fee rate",
    "exit_fee_rate": "Exit fee rate",
    "exit_additive_enabled": "Enable exit additive",
    "exit_additive_ratio": "Exit additive ratio",
    "exit_additive_gain": "Exit additive gain",
    "exit_additive_transform_pnl": "Exit PnL transform",
    "exit_additive_transform_duration": "Exit duration transform",
}


# ---------------------------------------------------------------------------
# Parameter validation utilities
# ---------------------------------------------------------------------------

_PARAMETER_BOUNDS: dict[str, dict[str, float]] = {
    # key: {min: ..., max: ...}  (bounds are inclusive where it makes sense)
    "invalid_action": {"max": 0.0},  # penalty should be <= 0
    "base_factor": {"min": 0.0},
    "idle_penalty_power": {"min": 0.0},
    "idle_penalty_ratio": {"min": 0.0},
    "max_trade_duration_candles": {"min": 1.0},
    "max_idle_duration_candles": {"min": 0.0},
    "hold_penalty_ratio": {"min": 0.0},
    "hold_penalty_power": {"min": 0.0},
    "exit_linear_slope": {"min": 0.0},
    "exit_plateau_grace": {"min": 0.0},
    "exit_power_tau": {"min": 1e-6, "max": 1.0},  # open (0,1] approximated
    "exit_half_life": {"min": 1e-6},
    "efficiency_weight": {"min": 0.0, "max": 2.0},
    "efficiency_center": {"min": 0.0, "max": 1.0},
    "win_reward_factor": {"min": 0.0},
    "pnl_factor_beta": {"min": 1e-6},
    # PBRS parameter bounds
    "potential_gamma": {"min": 0.0, "max": 1.0},
    "exit_potential_decay": {"min": 0.0, "max": 1.0},
    "hold_potential_ratio": {"min": 0.0},
    "hold_potential_gain": {"min": 0.0},
    "entry_additive_ratio": {"min": 0.0},
    "entry_additive_gain": {"min": 0.0},
    "entry_fee_rate": {"min": 0.0, "max": 0.1},
    "exit_fee_rate": {"min": 0.0, "max": 0.1},
    "exit_additive_ratio": {"min": 0.0},
    "exit_additive_gain": {"min": 0.0},
}

RewardParamValue = float | str | bool | None
RewardParams = dict[str, RewardParamValue]


class RewardDiagnosticsWarning(RuntimeWarning):
    """Warning category for reward space diagnostic graceful fallbacks.

    Enables selective filtering in tests or CLI harness while leaving other
    runtime warnings unaffected.
    """

    pass


def _warn_unknown_mode(
    mode_type: str,
    provided_value: str,
    valid_values: Iterable[str],
    fallback_value: str,
    stacklevel: int = 2,
) -> None:
    """Emit standardized warning for unknown mode values.

    Args:
        mode_type: Type of mode (e.g., "exit_attenuation_mode")
        provided_value: The invalid value that was provided
        valid_values: Iterable of valid values
        fallback_value: The value being used as fallback
        stacklevel: Stack level for warnings.warn
    """
    valid_sorted = sorted(valid_values)
    warnings.warn(
        f"Param: unknown {mode_type} '{provided_value}', "
        f"expected one of: {valid_sorted}; falling back to '{fallback_value}'",
        RewardDiagnosticsWarning,
        stacklevel=stacklevel,
    )


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    # Unsupported type
    raise ValueError(f"Param: unrecognized boolean literal {value!r}")


def _get_bool_param(params: RewardParams, key: str, default: bool | None = None) -> bool:
    """Extract boolean parameter with type safety.

    Args:
        params: Parameter dictionary to extract from.
        key: Parameter key to look up.
        default: Fallback value. If None, looks up from DEFAULT_MODEL_REWARD_PARAMETERS.

    Returns:
        Boolean value with fallback chain: params[key] -> default -> canonical default.
    """
    if default is None:
        default = DEFAULT_MODEL_REWARD_PARAMETERS.get(key)
    value = params.get(key, default)
    try:
        return _to_bool(value)
    except Exception:
        return bool(default) if default is not None else False


def _resolve_additive_enablement(
    exit_potential_mode: str,
    entry_additive_enabled_raw: bool,
    exit_additive_enabled_raw: bool,
) -> tuple[bool, bool, bool]:
    """Resolve effective additive enablement.

    Canonical exit mode suppresses additive terms at runtime, even if they were
    requested/enabled in configuration.

    Returns:
        (entry_additive_effective, exit_additive_effective, additives_suppressed)
    """

    entry_additive_effective = (
        bool(entry_additive_enabled_raw) if exit_potential_mode != "canonical" else False
    )
    exit_additive_effective = (
        bool(exit_additive_enabled_raw) if exit_potential_mode != "canonical" else False
    )
    additives_suppressed = exit_potential_mode == "canonical" and bool(
        entry_additive_enabled_raw or exit_additive_enabled_raw
    )
    return entry_additive_effective, exit_additive_effective, additives_suppressed


def _get_float_param(
    params: RewardParams, key: str, default: RewardParamValue | None = None
) -> float:
    """Extract float parameter with type safety and default fallback.

    Args:
        params: Parameter dictionary to extract from.
        key: Parameter key to look up.
        default: Fallback value. If None, looks up from DEFAULT_MODEL_REWARD_PARAMETERS.

    Returns:
        Float value with fallback chain: params[key] -> default -> canonical default.
    """
    if default is None:
        default = DEFAULT_MODEL_REWARD_PARAMETERS.get(key)
    value = params.get(key, default)
    # None -> NaN
    if value is None:
        return np.nan
    # Bool: treat explicitly (avoid surprising True->1.0 unless intentional)
    if isinstance(value, bool):
        return float(int(value))
    # Numeric
    if isinstance(value, (int, float)):
        try:
            fval = float(value)
        except (ValueError, TypeError):
            return np.nan
        return fval if np.isfinite(fval) else np.nan
    # String parsing
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return np.nan
        try:
            fval = float(stripped)
        except ValueError:
            return np.nan
        return fval if np.isfinite(fval) else np.nan
    # Unsupported type
    return np.nan


def _clamp_float_to_bounds(
    key: str,
    value: float,
    *,
    bounds: dict[str, float] | None = None,
    strict: bool,
) -> tuple[float, list[str]]:
    """Clamp numeric `value` to bounds for `key`.

    Returns:
        (adjusted_value, reason_parts)

    Notes:
        - Uses `_PARAMETER_BOUNDS` when `bounds` is None.
        - In strict mode, raises on out-of-bounds or non-finite.
        - In relaxed mode, clamps and emits reason tags.
    """

    effective_bounds = bounds if bounds is not None else _PARAMETER_BOUNDS.get(key, {})
    adjusted = float(value)
    reason_parts: list[str] = []

    if "min" in effective_bounds and adjusted < float(effective_bounds["min"]):
        if strict:
            raise ValueError(
                f"Param: '{key}'={adjusted} below min {float(effective_bounds['min'])}"
            )
        adjusted = float(effective_bounds["min"])
        reason_parts.append(f"min={float(effective_bounds['min'])}")

    if "max" in effective_bounds and adjusted > float(effective_bounds["max"]):
        if strict:
            raise ValueError(
                f"Param: '{key}'={adjusted} above max {float(effective_bounds['max'])}"
            )
        adjusted = float(effective_bounds["max"])
        reason_parts.append(f"max={float(effective_bounds['max'])}")

    if not np.isfinite(adjusted):
        if strict:
            raise ValueError(f"Param: '{key}' is non-finite: {adjusted}")
        adjusted = float(effective_bounds.get("min", 0.0))
        reason_parts.append("non_finite_reset")

    return adjusted, reason_parts


def _get_int_param(params: RewardParams, key: str, default: RewardParamValue | None = None) -> int:
    """Extract integer parameter with robust coercion.

    Args:
        params: Parameter dictionary to extract from.
        key: Parameter key to look up.
        default: Fallback value. If None, looks up from DEFAULT_MODEL_REWARD_PARAMETERS.

    Behavior:
    - Accept bool/int/float/str numeric representations.
    - Non-finite floats -> fallback to default coerced to int (or 0).
    - Strings: strip then parse float/int; on failure fallback.
    - None -> fallback.
    - Final value is clamped to a signed 64-bit range implicitly by int().
    """
    if default is None:
        default = DEFAULT_MODEL_REWARD_PARAMETERS.get(key)
    value = params.get(key, default)
    if value is None:
        return int(default) if isinstance(default, (int, float)) else 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            return int(default) if isinstance(default, (int, float)) else 0
        try:
            return int(value)
        except (OverflowError, ValueError):
            return int(default) if isinstance(default, (int, float)) else 0
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return int(default) if isinstance(default, (int, float)) else 0
        try:
            if any(ch in stripped for ch in (".", "e", "E")):
                fval = float(stripped)
                if not np.isfinite(fval):
                    return int(default) if isinstance(default, (int, float)) else 0
                return int(fval)
            return int(stripped)
        except (ValueError, OverflowError):
            return int(default) if isinstance(default, (int, float)) else 0
    # Unsupported type
    return int(default) if isinstance(default, (int, float)) else 0


def _get_str_param(params: RewardParams, key: str, default: str | None = None) -> str:
    """Extract string parameter with type safety and default fallback.

    Args:
        params: Parameter dictionary to extract from.
        key: Parameter key to look up.
        default: Fallback value. If None, looks up from DEFAULT_MODEL_REWARD_PARAMETERS.

    Returns:
        String value with fallback chain: params[key] -> default -> canonical default.
    """
    if default is None:
        default_val = DEFAULT_MODEL_REWARD_PARAMETERS.get(key)
        default = str(default_val) if default_val is not None else ""
    value = params.get(key, default)
    if isinstance(value, str):
        return value
    return default


def _compute_duration_ratio(trade_duration: int, max_trade_duration_candles: int) -> float:
    """Compute duration ratio with safe division."""
    return trade_duration / max(1, max_trade_duration_candles)


def _is_short_allowed(trading_mode: str) -> bool:
    mode = trading_mode.lower()
    if mode in TRADING_MODES[1:]:  # "margin", "futures"
        return True
    if mode == TRADING_MODES[0]:  # "spot"
        return False
    raise ValueError(
        f"Config: unsupported trading mode '{mode}'. Expected one of: {list(TRADING_MODES)}"
    )


def _fail_safely(reason: str) -> float:
    """Return 0.0 on numeric failure."""
    _ = reason
    return 0.0


def get_max_idle_duration_candles(
    params: RewardParams,
    *,
    max_trade_duration_candles: int | None = None,
) -> int:
    mtd = (
        int(max_trade_duration_candles)
        if isinstance(max_trade_duration_candles, (int, float))
        else None
    )
    if mtd is None or mtd <= 0:
        mtd = _get_int_param(params, "max_trade_duration_candles")
        if mtd <= 0:
            mtd = int(DEFAULT_MODEL_REWARD_PARAMETERS.get("max_trade_duration_candles", 128))

    default_mid = int(DEFAULT_IDLE_DURATION_MULTIPLIER * int(mtd))
    mid = _get_int_param(params, "max_idle_duration_candles", default_mid)
    if mid <= 0:
        mid = default_mid
    return int(mid)


def validate_reward_parameters(
    params: RewardParams,
    strict: bool = True,
) -> tuple[RewardParams, dict[str, dict[str, Any]]]:
    """Clamp parameters to bounds and coerce booleans and numeric overrides.

    Returns a sanitized copy plus adjustments mapping (param -> original/adjusted/reason).
    Behavior:
    - Boolean-like keys are coerced to bool.
    - Numeric-bounded keys are coerced to float when provided as str/bool/None.
      * In strict mode: raise on non-numeric or out-of-bounds.
      * In relaxed mode: fallback to min bound or 0.0 with adjustment reason.
    - Non-finite numerics fall back to min bound or 0.0 (relaxed) or raise (strict).
    """
    sanitized = dict(params)
    adjustments: dict[str, dict[str, Any]] = {}

    # Boolean parameter coercion
    _bool_keys = [
        "check_invariants",
        "hold_potential_enabled",
        "entry_additive_enabled",
        "exit_additive_enabled",
    ]
    for bkey in _bool_keys:
        if bkey in sanitized:
            original_val = sanitized[bkey]
            coerced_val = _to_bool(original_val)
            if coerced_val is not original_val:
                sanitized[bkey] = coerced_val
            adjustments.setdefault(
                bkey,
                {
                    "original": original_val,
                    "adjusted": coerced_val,
                    "reason": "bool_coerce",
                    "validation_mode": "strict" if strict else "relaxed",
                },
            )

    # Coerce and clamp numeric-bounded parameters
    for key, bounds in _PARAMETER_BOUNDS.items():
        if key not in sanitized:
            continue

        original_val = sanitized[key]
        # Robust coercion to float using helper (handles None/str/bool/non-finite)
        coerced_val = _get_float_param({key: original_val}, key, np.nan)

        # Handle non-numeric or unparsable values
        if not np.isfinite(coerced_val):
            # Treat derived parameters specially: drop to allow downstream derivation
            if key == "max_idle_duration_candles":
                # Remove the key so downstream helpers derive from max_trade_duration_candles
                del sanitized[key]
                adjustments[key] = {
                    "original": original_val,
                    "adjusted": None,
                    "reason": "derived_default",
                    "validation_mode": "strict" if strict else "relaxed",
                }
                continue
            if strict:
                raise ValueError(f"Param: '{key}' is non-numeric or invalid: {original_val!r}")
            adjusted = bounds.get("min", 0.0)
            sanitized[key] = adjusted
            adjustments[key] = {
                "original": original_val,
                "adjusted": adjusted,
                "reason": "non_numeric_reset",
                "validation_mode": "strict" if strict else "relaxed",
            }
            continue

        original_numeric = float(coerced_val)

        # Track type coercion
        if not isinstance(original_val, (int, float)):
            adjustments.setdefault(
                key,
                {
                    "original": original_val,
                    "adjusted": original_numeric,
                    "reason": "numeric_coerce",
                    "validation_mode": "strict" if strict else "relaxed",
                },
            )
            sanitized[key] = original_numeric

        adjusted, reason_parts = _clamp_float_to_bounds(
            key,
            original_numeric,
            bounds=bounds,
            strict=strict,
        )

        if not np.isclose(adjusted, original_numeric):
            sanitized[key] = adjusted
            prev_reason = adjustments.get(key, {}).get("reason")
            reason: list[str] = []
            if prev_reason:
                reason.append(prev_reason)
            reason.extend(reason_parts)
            reason_str = ",".join(reason) if reason else "clamp"
            adjustments[key] = {
                "original": original_val,
                "adjusted": adjusted,
                "reason": reason_str,
                "validation_mode": "strict" if strict else "relaxed",
            }

    return sanitized, adjustments


def add_tunable_cli_args(parser: argparse.ArgumentParser) -> None:
    """Dynamically add CLI options for each tunable in DEFAULT_MODEL_REWARD_PARAMETERS.

    Rules:
    - Use the same underscored names as option flags (e.g., --idle_penalty_ratio).
    - Defaults are None so only user-provided values override params.
    - For exit_attenuation_mode, enforce allowed choices (case-sensitive).
    - Skip keys already managed as top-level options (e.g., base_factor) to avoid duplicates.
    """
    skip_keys = {"base_factor"}
    for key, default in DEFAULT_MODEL_REWARD_PARAMETERS.items():
        if key in skip_keys:
            continue
        help_text = DEFAULT_MODEL_REWARD_PARAMETERS_HELP.get(key, f"Override tunable '{key}'.")
        if key == "exit_attenuation_mode":
            parser.add_argument(
                f"--{key}",
                type=str,
                choices=sorted(ATTENUATION_MODES_WITH_LEGACY),
                default=None,
                help=help_text,
            )
        elif key == "exit_plateau":
            parser.add_argument(
                f"--{key}",
                type=int,
                choices=[0, 1],
                default=None,
                help=help_text,
            )
        elif key == "exit_potential_mode":
            parser.add_argument(
                f"--{key}",
                type=str,
                choices=sorted(ALLOWED_EXIT_POTENTIAL_MODES),
                default=None,
                help=help_text,
            )
        elif key in [
            "hold_potential_transform_pnl",
            "hold_potential_transform_duration",
            "entry_additive_transform_pnl",
            "entry_additive_transform_duration",
            "exit_additive_transform_pnl",
            "exit_additive_transform_duration",
        ]:
            parser.add_argument(
                f"--{key}",
                type=str,
                choices=sorted(ALLOWED_TRANSFORMS),
                default=None,
                help=help_text,
            )
        elif key in [
            "hold_potential_enabled",
            "entry_additive_enabled",
            "exit_additive_enabled",
        ]:
            parser.add_argument(
                f"--{key}",
                type=int,
                choices=[0, 1],
                default=None,
                help=help_text,
            )
        else:
            # Map numerics to float; leave strings as str
            if isinstance(default, (int, float)):
                parser.add_argument(f"--{key}", type=float, default=None, help=help_text)
            else:
                parser.add_argument(f"--{key}", type=str, default=None, help=help_text)


@dataclasses.dataclass
class RewardContext:
    """Context for reward computation.

    Attributes
    ----------
    current_pnl : float
        Unrealized PnL at the current tick (state s').
    """

    current_pnl: float
    trade_duration: int
    idle_duration: int
    max_unrealized_profit: float
    min_unrealized_profit: float
    position: Positions
    action: Actions


@dataclasses.dataclass
class RewardBreakdown:
    total: float = 0.0
    invalid_penalty: float = 0.0
    idle_penalty: float = 0.0
    hold_penalty: float = 0.0
    exit_component: float = 0.0
    # PBRS components
    reward_shaping: float = 0.0
    entry_additive: float = 0.0
    exit_additive: float = 0.0
    prev_potential: float = 0.0
    next_potential: float = 0.0
    # PBRS helpers
    base_reward: float = 0.0
    pbrs_delta: float = 0.0  # Δ(s,a,s') = γ·Φ(s') − Φ(s)  # noqa: RUF003
    invariance_correction: float = 0.0


def _compute_time_attenuation_coefficient(
    duration_ratio: float,
    params: RewardParams,
) -> float:
    """
    Calculate time-based attenuation coefficient using configurable strategy.

    Returns a coefficient (typically in range [0.5, 2.0]) to multiply with base_factor.
    """
    if duration_ratio < 0.0:
        duration_ratio = 0.0

    exit_attenuation_mode = _get_str_param(params, "exit_attenuation_mode")
    exit_plateau = _get_bool_param(params, "exit_plateau")

    exit_plateau_grace = _get_float_param(params, "exit_plateau_grace")
    if exit_plateau_grace < 0.0:
        warnings.warn(
            f"Param: exit_plateau_grace={exit_plateau_grace} < 0; falling back to 0.0",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )
        exit_plateau_grace = 0.0
    exit_linear_slope = _get_float_param(params, "exit_linear_slope")
    if exit_linear_slope < 0.0:
        warnings.warn(
            f"Param: exit_linear_slope={exit_linear_slope} < 0; falling back to 1.0",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )
        exit_linear_slope = 1.0

    def _legacy_kernel(dr: float) -> float:
        return 1.5 if dr <= 1.0 else 0.5

    def _sqrt_kernel(dr: float) -> float:
        return 1.0 / math.sqrt(1.0 + dr)

    def _linear_kernel(dr: float) -> float:
        return 1.0 / (1.0 + exit_linear_slope * dr)

    def _power_kernel(dr: float) -> float:
        raw_tau = params.get("exit_power_tau", None)
        if raw_tau is None:
            alpha = 1.0
        else:
            tau = _get_float_param(params, "exit_power_tau", np.nan)
            if 0.0 < tau <= 1.0:
                alpha = -math.log(tau) / _LOG_2
            else:
                warnings.warn(
                    f"Param: exit_power_tau={tau} outside (0,1]; falling back to 1.0",
                    RewardDiagnosticsWarning,
                    stacklevel=2,
                )
                alpha = 1.0
        return 1.0 / math.pow(1.0 + dr, alpha)

    def _half_life_kernel(dr: float) -> float:
        hl = _get_float_param(params, "exit_half_life")
        if hl <= 0.0 or np.isclose(hl, 0.0):
            warnings.warn(
                f"Param: exit_half_life={hl} <= 0; falling back to 1.0",
                RewardDiagnosticsWarning,
                stacklevel=2,
            )
            return 1.0
        return math.pow(2.0, -dr / hl)

    kernels = {
        "legacy": _legacy_kernel,
        "sqrt": _sqrt_kernel,
        "linear": _linear_kernel,
        "power": _power_kernel,
        "half_life": _half_life_kernel,
    }
    if exit_plateau:
        if duration_ratio <= exit_plateau_grace:
            effective_dr = 0.0
        else:
            effective_dr = duration_ratio - exit_plateau_grace
    else:
        effective_dr = duration_ratio

    kernel = kernels.get(exit_attenuation_mode)
    if kernel is None:
        _warn_unknown_mode(
            "exit_attenuation_mode",
            exit_attenuation_mode,
            ATTENUATION_MODES_WITH_LEGACY,
            "linear",
            stacklevel=2,
        )
        kernel = _linear_kernel

    try:
        time_attenuation_coefficient = kernel(effective_dr)
    except Exception as e:
        warnings.warn(
            f"Param: exit_attenuation_mode='{exit_attenuation_mode}' failed ({e!r}); falling back to 'linear'",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )
        time_attenuation_coefficient = _linear_kernel(effective_dr)

    return time_attenuation_coefficient


def _get_exit_factor(
    base_factor: float,
    pnl: float,
    pnl_target: float,
    duration_ratio: float,
    context: RewardContext,
    params: RewardParams,
    risk_reward_ratio: float,
) -> float:
    """
    Compute exit reward factor by applying multiplicative coefficients to base_factor.

    Formula: exit_factor = base_factor * time_attenuation_coefficient * pnl_target_coefficient * efficiency_coefficient

    Args:
        base_factor: Base reward value before coefficient adjustments
        pnl: Realized profit/loss
        pnl_target: Target profit threshold (pnl_target = profit_aim * risk_reward_ratio)
        duration_ratio: Trade duration relative to target duration
        context: Trade context with unrealized profit/loss extremes
        params: Reward configuration parameters
        risk_reward_ratio: Risk/reward ratio (must match the value used to calculate pnl_target)

    Returns:
        float: Final exit factor (can be negative for losses)
    """
    if (
        not np.isfinite(base_factor)
        or not np.isfinite(pnl)
        or not np.isfinite(pnl_target)
        or not np.isfinite(duration_ratio)
    ):
        return _fail_safely("non_finite_exit_factor_inputs")

    exit_factor = (
        base_factor
        * _compute_time_attenuation_coefficient(duration_ratio, params)
        * _compute_pnl_target_coefficient(
            params,
            pnl,
            pnl_target,
            risk_reward_ratio,
        )
        * _compute_efficiency_coefficient(params, context, pnl)
    )

    if _get_bool_param(
        params,
        "check_invariants",
    ):
        if not np.isfinite(exit_factor):
            return _fail_safely("non_finite_exit_factor_after_kernel")
        if exit_factor < 0.0 and pnl >= 0.0:
            exit_factor = 0.0
        exit_factor_threshold = _get_float_param(params, "exit_factor_threshold")
        if exit_factor_threshold > 0 and np.isfinite(exit_factor_threshold):  # noqa: SIM102
            if abs(exit_factor) > exit_factor_threshold:
                warnings.warn(
                    f"Param: |exit_factor|={abs(exit_factor):.6g} > threshold={exit_factor_threshold:.6g}",
                    RewardDiagnosticsWarning,
                    stacklevel=2,
                )

    return exit_factor


def _compute_pnl_target_coefficient(
    params: RewardParams,
    pnl: float,
    pnl_target: float,
    risk_reward_ratio: float,
) -> float:
    """
    Compute PnL target coefficient based on PnL/target ratio using tanh.

    Returns a coefficient (typically 0.5-2.0) to be multiplied with base_factor.
    The coefficient rewards trades that exceed pnl_target and penalizes losses
    beyond the risk/reward threshold.

    Args:
        params: Reward configuration parameters
        pnl: Realized profit/loss
        pnl_target: Target profit threshold (pnl_target = profit_aim * risk_reward_ratio)
        risk_reward_ratio: Risk/reward ratio for loss penalty calculation

    Returns:
        float: Coefficient ≥ 0.0 (typically 0.5-2.0 range)
    """
    pnl_target_coefficient = 1.0

    if pnl_target > 0.0:
        win_reward_factor = _get_float_param(params, "win_reward_factor")
        pnl_factor_beta = _get_float_param(params, "pnl_factor_beta")
        rr = risk_reward_ratio if risk_reward_ratio > 0 else RISK_REWARD_RATIO_DEFAULT

        pnl_ratio = pnl / pnl_target
        if abs(pnl_ratio) > 1.0:
            base_pnl_target_coefficient = math.tanh(pnl_factor_beta * (abs(pnl_ratio) - 1.0))
            if pnl_ratio > 1.0:
                pnl_target_coefficient = 1.0 + win_reward_factor * base_pnl_target_coefficient
            elif pnl_ratio < -(1.0 / rr):
                loss_penalty_factor = win_reward_factor * rr
                pnl_target_coefficient = 1.0 + loss_penalty_factor * base_pnl_target_coefficient

    return pnl_target_coefficient


def _compute_efficiency_coefficient(
    params: RewardParams,
    context: RewardContext,
    pnl: float,
) -> float:
    """
    Compute exit efficiency coefficient based on PnL position relative to unrealized extremes.

    Returns a coefficient (typically 0.5-1.5) that rewards exits closer to optimal timing.
    For profitable trades, higher coefficient when exiting near max unrealized profit.
    For losing trades, higher coefficient when exiting near min unrealized loss.

    Args:
        params: Reward configuration parameters containing:
            - efficiency_weight: Amplification factor for efficiency adjustment
            - efficiency_center: Target efficiency ratio (0.0-1.0)
        context: Trade context with unrealized profit/loss extremes
        pnl: Realized profit/loss

    Returns:
        float: Coefficient ≥ 0.0 (typically 0.5-1.5 range)
    """
    efficiency_coefficient = 1.0
    efficiency_weight = _get_float_param(params, "efficiency_weight")
    efficiency_center = _get_float_param(params, "efficiency_center")
    if efficiency_weight != 0.0 and not np.isclose(pnl, 0.0):
        max_pnl = max(context.max_unrealized_profit, pnl)
        min_pnl = min(context.min_unrealized_profit, pnl)
        range_pnl = max_pnl - min_pnl
        if np.isfinite(range_pnl) and not np.isclose(range_pnl, 0.0):
            efficiency_ratio = (pnl - min_pnl) / range_pnl
            # For profits (pnl > 0): high ratio = good exit → higher coefficient → amplify gain
            # For losses (pnl < 0): high ratio = good exit → LOWER coefficient → attenuate penalty
            # The sign inversion for losses ensures reward = pnl * coef behaves correctly:
            #   - Good loss exit: small |pnl| * low coef = small penalty
            #   - Bad loss exit: large |pnl| * high coef = large penalty
            if pnl > 0.0:
                efficiency_coefficient = 1.0 + efficiency_weight * (
                    efficiency_ratio - efficiency_center
                )
            elif pnl < 0.0:
                efficiency_coefficient = 1.0 + efficiency_weight * (
                    efficiency_center - efficiency_ratio
                )

    if not np.isfinite(efficiency_coefficient):
        efficiency_coefficient = 0.0

    if efficiency_coefficient < 0.0:
        if _get_bool_param(params, "check_invariants"):
            warnings.warn(
                f"Param: efficiency_coefficient={efficiency_coefficient:.6g} < 0; falling back to 0.0",
                RewardDiagnosticsWarning,
                stacklevel=2,
            )
        efficiency_coefficient = 0.0

    return efficiency_coefficient


def _is_valid_action(
    position: Positions,
    action: Actions,
    *,
    short_allowed: bool,
) -> bool:
    if action == Actions.Neutral:
        return True
    if action == Actions.Long_enter:
        return position == Positions.Neutral
    if action == Actions.Short_enter:
        return short_allowed and position == Positions.Neutral
    if action == Actions.Long_exit:
        return position == Positions.Long
    if action == Actions.Short_exit:
        return position == Positions.Short
    return False


def _get_next_position(
    position: Positions,
    action: Actions,
    *,
    short_allowed: bool,
) -> Positions:
    """Compute the next position given current position and action."""

    if action == Actions.Long_enter and position == Positions.Neutral:
        return Positions.Long
    if action == Actions.Short_enter and position == Positions.Neutral and short_allowed:
        return Positions.Short
    if action == Actions.Long_exit and position == Positions.Long:
        return Positions.Neutral
    if action == Actions.Short_exit and position == Positions.Short:
        return Positions.Neutral
    return position


def _idle_penalty(context: RewardContext, idle_factor: float, params: RewardParams) -> float:
    """Compute idle penalty."""
    idle_penalty_ratio = _get_float_param(params, "idle_penalty_ratio")
    idle_penalty_power = _get_float_param(params, "idle_penalty_power")
    max_idle_duration_candles = get_max_idle_duration_candles(params)
    idle_duration_ratio = context.idle_duration / max(1, max_idle_duration_candles)
    return -idle_factor * idle_penalty_ratio * idle_duration_ratio**idle_penalty_power


def _hold_penalty(context: RewardContext, hold_factor: float, params: RewardParams) -> float:
    """Compute hold penalty."""
    hold_penalty_ratio = _get_float_param(params, "hold_penalty_ratio")
    hold_penalty_power = _get_float_param(params, "hold_penalty_power")
    max_trade_duration_candles = _get_int_param(params, "max_trade_duration_candles")
    duration_ratio = _compute_duration_ratio(context.trade_duration, max_trade_duration_candles)

    if duration_ratio < 1.0:
        return _fail_safely("hold_penalty_duration_ratio_lt_1")

    return -hold_factor * hold_penalty_ratio * (duration_ratio - 1.0) ** hold_penalty_power


def _compute_exit_reward(
    base_factor: float,
    pnl_target: float,
    duration_ratio: float,
    context: RewardContext,
    params: RewardParams,
    risk_reward_ratio: float,
) -> float:
    """Compose the exit reward: pnl * exit_factor.

    Args:
        base_factor: Base reward value before coefficient adjustments
        pnl_target: Target profit threshold (pnl_target = profit_aim * risk_reward_ratio)
        duration_ratio: Trade duration relative to target duration
        context: Trade context with PnL and unrealized profit/loss extremes
        params: Reward configuration parameters
        risk_reward_ratio: Risk/reward ratio (must match the value used to calculate pnl_target)

    Returns:
        float: Exit reward (pnl * exit_factor)
    """
    exit_factor = _get_exit_factor(
        base_factor,
        context.current_pnl,
        pnl_target,
        duration_ratio,
        context,
        params,
        risk_reward_ratio,
    )
    return context.current_pnl * exit_factor


def calculate_reward(
    context: RewardContext,
    params: RewardParams,
    base_factor: float,
    profit_aim: float,
    risk_reward_ratio: float,
    *,
    short_allowed: bool,
    action_masking: bool,
    prev_potential: float = np.nan,
) -> RewardBreakdown:
    breakdown = RewardBreakdown()

    is_valid = _is_valid_action(
        context.position,
        context.action,
        short_allowed=short_allowed,
    )

    base_reward: float | None = None
    if not is_valid and not action_masking:
        breakdown.invalid_penalty = _get_float_param(params, "invalid_action")
        base_reward = breakdown.invalid_penalty

    base_factor = _get_float_param(params, "base_factor", base_factor)

    if "profit_aim" in params:
        profit_aim = _get_float_param(params, "profit_aim", float(profit_aim))

    if "risk_reward_ratio" in params:
        risk_reward_ratio = _get_float_param(params, "risk_reward_ratio", float(risk_reward_ratio))
    elif "rr" in params:
        risk_reward_ratio = _get_float_param(params, "rr", float(risk_reward_ratio))

    pnl_target = float(profit_aim * risk_reward_ratio)

    idle_factor = base_factor * (profit_aim / risk_reward_ratio)
    hold_factor = idle_factor

    max_trade_duration_candles = _get_int_param(params, "max_trade_duration_candles")
    current_duration_ratio = _compute_duration_ratio(
        context.trade_duration, max_trade_duration_candles
    )

    # Base reward calculation
    if base_reward is None:
        if context.action == Actions.Neutral:
            if context.position == Positions.Neutral:
                base_reward = _idle_penalty(context, idle_factor, params)
                breakdown.idle_penalty = base_reward
            elif context.position in (Positions.Long, Positions.Short):
                base_reward = _hold_penalty(context, hold_factor, params)
                breakdown.hold_penalty = base_reward
            else:
                base_reward = 0.0
        else:
            is_exit_action = (
                context.action == Actions.Long_exit and context.position == Positions.Long
            ) or (context.action == Actions.Short_exit and context.position == Positions.Short)
            if is_exit_action:
                base_reward = _compute_exit_reward(
                    base_factor,
                    pnl_target,
                    current_duration_ratio,
                    context,
                    params,
                    risk_reward_ratio,
                )
                breakdown.exit_component = base_reward
            else:
                base_reward = 0.0

    breakdown.base_reward = float(base_reward)

    # === PBRS INTEGRATION ===
    current_pnl = context.current_pnl if context.position != Positions.Neutral else 0.0

    next_position = _get_next_position(
        context.position, context.action, short_allowed=short_allowed
    )
    is_entry = context.position == Positions.Neutral and next_position in (
        Positions.Long,
        Positions.Short,
    )
    is_exit = (
        context.position
        in (
            Positions.Long,
            Positions.Short,
        )
        and next_position == Positions.Neutral
    )
    is_hold = context.position in (
        Positions.Long,
        Positions.Short,
    ) and next_position in (Positions.Long, Positions.Short)
    is_neutral = context.position == Positions.Neutral and next_position == Positions.Neutral

    if is_entry:
        next_duration_ratio = 0.0
        if context.action == Actions.Long_enter:
            next_pnl = _compute_unrealized_pnl_estimate(
                Positions.Long,
                entry_open=1.0,
                current_open=1.0,
                params=params,
            )
        elif context.action == Actions.Short_enter:
            next_pnl = _compute_unrealized_pnl_estimate(
                Positions.Short,
                entry_open=1.0,
                current_open=1.0,
                params=params,
            )
        else:
            next_pnl = current_pnl
    elif is_hold:
        next_duration_ratio = _compute_duration_ratio(
            context.trade_duration, max_trade_duration_candles
        )
        # Optionally simulate unrealized PnL during holds to feed Φ(s)
        if _get_bool_param(params, "unrealized_pnl", False):
            center_unrealized = 0.5 * (
                context.max_unrealized_profit + context.min_unrealized_profit
            )
            beta = _get_float_param(params, "pnl_factor_beta")
            next_pnl = float(center_unrealized * math.tanh(beta * next_duration_ratio))
        else:
            next_pnl = current_pnl
    elif is_exit:
        next_pnl = 0.0
        next_duration_ratio = 0.0
    else:
        next_pnl = current_pnl
        next_duration_ratio = current_duration_ratio

    # Apply PBRS only if enabled and not neutral self-loop
    exit_mode = _get_str_param(params, "exit_potential_mode")

    hold_potential_enabled = _get_bool_param(params, "hold_potential_enabled")
    entry_additive_enabled = (
        False if exit_mode == "canonical" else _get_bool_param(params, "entry_additive_enabled")
    )
    exit_additive_enabled = (
        False if exit_mode == "canonical" else _get_bool_param(params, "exit_additive_enabled")
    )

    pbrs_enabled = bool(hold_potential_enabled or entry_additive_enabled or exit_additive_enabled)

    if pbrs_enabled:
        # Stored potential carried across steps.
        prev_potential = float(prev_potential) if np.isfinite(prev_potential) else 0.0

        if is_neutral:
            # Neutral self-loop keeps stored potential unchanged.
            breakdown.prev_potential = prev_potential
            breakdown.next_potential = prev_potential
            breakdown.total = base_reward
            return breakdown

        reward_shaping, next_potential, pbrs_delta, entry_additive, exit_additive = (
            compute_pbrs_components(
                current_pnl=current_pnl,
                pnl_target=pnl_target,
                current_duration_ratio=current_duration_ratio,
                next_pnl=next_pnl,
                next_duration_ratio=next_duration_ratio,
                is_exit=is_exit,
                is_entry=is_entry,
                prev_potential=prev_potential,
                params=params,
                risk_reward_ratio=risk_reward_ratio,
                base_factor=base_factor,
            )
        )

        breakdown.reward_shaping = reward_shaping
        breakdown.prev_potential = prev_potential
        breakdown.next_potential = next_potential
        breakdown.entry_additive = entry_additive
        breakdown.exit_additive = exit_additive
        breakdown.pbrs_delta = pbrs_delta
        breakdown.invariance_correction = reward_shaping - pbrs_delta
        breakdown.total = base_reward + reward_shaping + entry_additive + exit_additive
        return breakdown

    prev_potential = float(prev_potential) if np.isfinite(prev_potential) else 0.0
    breakdown.prev_potential = prev_potential
    breakdown.next_potential = prev_potential
    breakdown.total = base_reward

    return breakdown


def _duration_hazard_probability(
    *,
    duration_ratio: float,
    base_probability: float,
    overtime_multiplier: float,
    max_probability: float,
) -> float:
    """Compute a bounded hazard probability keyed on a duration ratio.

    Behavior:
    - duration_ratio <= 1 -> returns base_probability
    - duration_ratio > 1 -> increases linearly with overtime

    Notes
    -----
    This is used for:
    - exit probability when holding past max trade duration
    - entry probability when idling past max idle duration
    """

    if not np.isfinite(duration_ratio):
        return float(np.clip(base_probability, 0.0, max_probability))

    overtime = max(0.0, float(duration_ratio - 1.0))
    probability = base_probability * (1.0 + overtime_multiplier * overtime)
    return float(np.clip(probability, 0.0, max_probability))


_SAMPLE_ENTRY_PROBABILITY_MARGIN = 0.4
_SAMPLE_ENTRY_PROBABILITY_SPOT = 0.3
_SAMPLE_DURATION_HAZARD_OVERTIME_MULTIPLIER = 4.0
_SAMPLE_DURATION_HAZARD_MAX_PROBABILITY = 0.9
_SAMPLE_EXIT_PROBABILITY_MIN = 0.002
_SAMPLE_EXIT_PROBABILITY_MAX = 0.2


def _sampling_probabilities(
    position: Positions,
    *,
    short_allowed: bool,
    trade_duration: int,
    max_trade_duration_candles: int,
    idle_duration: int,
    max_idle_duration_candles: int,
) -> tuple[float, float, float]:
    if position == Positions.Neutral:
        base_entry_prob = (
            _SAMPLE_ENTRY_PROBABILITY_MARGIN if short_allowed else _SAMPLE_ENTRY_PROBABILITY_SPOT
        )
        idle_ratio = idle_duration / max(1, int(max_idle_duration_candles))
        entry_prob = _duration_hazard_probability(
            duration_ratio=idle_ratio,
            base_probability=base_entry_prob,
            overtime_multiplier=_SAMPLE_DURATION_HAZARD_OVERTIME_MULTIPLIER,
            max_probability=_SAMPLE_DURATION_HAZARD_MAX_PROBABILITY,
        )
        neutral_prob = max(0.0, 1.0 - entry_prob)
        return float(entry_prob), float("nan"), float(neutral_prob)

    duration_ratio = _compute_duration_ratio(trade_duration, max_trade_duration_candles)

    base_exit_prob = 1.0 / max(1, int(max_trade_duration_candles))
    base_exit_prob = float(
        np.clip(base_exit_prob, _SAMPLE_EXIT_PROBABILITY_MIN, _SAMPLE_EXIT_PROBABILITY_MAX)
    )

    exit_prob = _duration_hazard_probability(
        duration_ratio=duration_ratio,
        base_probability=base_exit_prob,
        overtime_multiplier=_SAMPLE_DURATION_HAZARD_OVERTIME_MULTIPLIER,
        max_probability=_SAMPLE_DURATION_HAZARD_MAX_PROBABILITY,
    )
    return float("nan"), float(exit_prob), float("nan")


def _sample_action(
    position: Positions,
    rng: random.Random,
    *,
    short_allowed: bool,
    trade_duration: int,
    max_trade_duration_candles: int,
    idle_duration: int,
    max_idle_duration_candles: int,
) -> tuple[Actions, float, float, float]:
    entry_prob, exit_prob, neutral_prob = _sampling_probabilities(
        position,
        short_allowed=short_allowed,
        trade_duration=trade_duration,
        max_trade_duration_candles=max_trade_duration_candles,
        idle_duration=idle_duration,
        max_idle_duration_candles=max_idle_duration_candles,
    )

    if position == Positions.Neutral:
        if short_allowed:
            choices = [Actions.Neutral, Actions.Long_enter, Actions.Short_enter]
            weights = [neutral_prob, entry_prob * 0.5, entry_prob * 0.5]
        else:
            choices = [Actions.Neutral, Actions.Long_enter]
            weights = [neutral_prob, entry_prob]
        action = rng.choices(choices, weights=weights, k=1)[0]
        return action, entry_prob, exit_prob, neutral_prob

    if position == Positions.Long:
        choices = [Actions.Neutral, Actions.Long_exit]
    else:  # Positions.Short
        choices = [Actions.Neutral, Actions.Short_exit]

    weights = [1.0 - exit_prob, exit_prob]
    action = rng.choices(choices, weights=weights, k=1)[0]
    return action, entry_prob, exit_prob, neutral_prob


def parse_overrides(overrides: Iterable[str]) -> RewardParams:
    parsed: RewardParams = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"CLI: invalid override format '{override}'. Expected 'key=value'")
        key, value = override.split("=", 1)
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def simulate_samples(
    num_samples: int,
    seed: int,
    params: RewardParams,
    base_factor: float,
    profit_aim: float,
    risk_reward_ratio: float,
    max_duration_ratio: float,
    trading_mode: str,
    pnl_base_std: float,
    pnl_duration_vol_scale: float,
) -> pd.DataFrame:
    """Simulate synthetic samples for reward analysis.

    The synthetic generator produces a *coherent trajectory* (state carried across samples)
    so PJRS/PBRS stored-potential mechanics can be exercised realistically.

    Notes
    -----
    - PnL is a state variable while in position (may be non-zero on holds).
    - Neutral states always have pnl=0.
    - Realized PnL appears on the exit step (position still Long/Short).
    """

    rng = random.Random(seed)
    max_trade_duration_candles = _get_int_param(params, "max_trade_duration_candles")
    short_allowed = _is_short_allowed(trading_mode)
    action_masking = _get_bool_param(params, "action_masking", True)

    # Theoretical PBRS invariance flag
    exit_mode = _get_str_param(params, "exit_potential_mode")
    entry_enabled_raw = _get_bool_param(params, "entry_additive_enabled")
    exit_enabled_raw = _get_bool_param(params, "exit_additive_enabled")

    entry_enabled, exit_enabled, _additives_suppressed = _resolve_additive_enablement(
        exit_mode,
        entry_enabled_raw,
        exit_enabled_raw,
    )
    pbrs_invariant = bool(exit_mode == "canonical" and not (entry_enabled or exit_enabled))

    max_idle_duration_candles = get_max_idle_duration_candles(
        params, max_trade_duration_candles=max_trade_duration_candles
    )
    max_trade_duration_cap = int(max_trade_duration_candles * max_duration_ratio)

    samples: list[dict[str, float]] = []
    prev_potential: float = 0.0

    # Stateful trajectory variables
    position = Positions.Neutral
    trade_duration = 0
    idle_duration = 0
    pnl = 0.0
    max_unrealized_profit = 0.0
    min_unrealized_profit = 0.0

    # Synthetic market state
    current_open = 1.0
    entry_open = current_open

    for _ in range(num_samples):
        # Simulate synthetic open-price movement.
        duration_ratio = (
            _compute_duration_ratio(trade_duration, max_trade_duration_candles)
            if position in (Positions.Long, Positions.Short)
            else 0.0
        )
        open_return_std = pnl_base_std * (1.0 + pnl_duration_vol_scale * duration_ratio)
        step_return = rng.gauss(0.0, open_return_std)

        # Small directional drift so long/short trajectories are not perfectly symmetric
        drift = 0.001 * duration_ratio
        if position == Positions.Long:
            step_return += drift
        elif position == Positions.Short:
            step_return -= drift

        if not np.isfinite(step_return):
            step_return = 0.0
        step_return = float(np.clip(step_return, -0.95, 0.95))

        current_open = float(max(1e-6, current_open * (1.0 + step_return)))

        # Compute fee-aware unrealized PnL from (entry_open, current_open)
        if position in (Positions.Long, Positions.Short):
            pnl = _compute_unrealized_pnl_estimate(
                position,
                entry_open=entry_open,
                current_open=current_open,
                params=params,
            )
            pnl = float(np.clip(pnl, -0.15, 0.15))
            max_unrealized_profit = max(max_unrealized_profit, pnl)
            min_unrealized_profit = min(min_unrealized_profit, pnl)
        else:
            pnl = 0.0
            max_unrealized_profit = 0.0
            min_unrealized_profit = 0.0

        action, sample_entry_prob, sample_exit_prob, sample_neutral_prob = _sample_action(
            position,
            rng,
            short_allowed=short_allowed,
            trade_duration=trade_duration,
            max_trade_duration_candles=max_trade_duration_candles,
            idle_duration=idle_duration,
            max_idle_duration_candles=max_idle_duration_candles,
        )

        context = RewardContext(
            current_pnl=pnl,
            trade_duration=trade_duration,
            idle_duration=idle_duration,
            max_unrealized_profit=max_unrealized_profit,
            min_unrealized_profit=min_unrealized_profit,
            position=position,
            action=action,
        )

        breakdown = calculate_reward(
            context,
            params,
            base_factor,
            profit_aim,
            risk_reward_ratio,
            short_allowed=short_allowed,
            action_masking=action_masking,
            prev_potential=prev_potential,
        )
        prev_potential = breakdown.next_potential

        idle_ratio = context.idle_duration / max(1, max_idle_duration_candles)
        samples.append(
            {
                "pnl": context.current_pnl,
                "trade_duration": context.trade_duration,
                "idle_duration": context.idle_duration,
                "duration_ratio": _compute_duration_ratio(
                    context.trade_duration, max_trade_duration_candles
                ),
                "idle_ratio": idle_ratio,
                "position": float(context.position.value),
                "action": int(context.action.value),
                # Sampling diagnostics
                "sample_entry_prob": sample_entry_prob,
                "sample_exit_prob": sample_exit_prob,
                "sample_neutral_prob": sample_neutral_prob,
                "reward": breakdown.total,
                "reward_invalid": breakdown.invalid_penalty,
                "reward_idle": breakdown.idle_penalty,
                "reward_hold": breakdown.hold_penalty,
                "reward_exit": breakdown.exit_component,
                # PBRS components
                "reward_shaping": breakdown.reward_shaping,
                "reward_entry_additive": breakdown.entry_additive,
                "reward_exit_additive": breakdown.exit_additive,
                "prev_potential": breakdown.prev_potential,
                "next_potential": breakdown.next_potential,
                # PBRS columns
                "reward_base": breakdown.base_reward,
                "reward_pbrs_delta": breakdown.pbrs_delta,
                "reward_invariance_correction": breakdown.invariance_correction,
                "is_invalid": float(breakdown.invalid_penalty != 0.0),
                "pbrs_invariant": bool(pbrs_invariant),
            }
        )

        # Transition state
        if position == Positions.Neutral:
            if action == Actions.Neutral:
                idle_duration = min(idle_duration + 1, max_idle_duration_candles)
            elif action == Actions.Long_enter:
                position = Positions.Long
                trade_duration = 0
                idle_duration = 0
                entry_open = current_open
                pnl = _compute_unrealized_pnl_estimate(
                    Positions.Long,
                    entry_open=entry_open,
                    current_open=current_open,
                    params=params,
                )
                max_unrealized_profit = pnl
                min_unrealized_profit = pnl
            elif action == Actions.Short_enter and short_allowed:
                position = Positions.Short
                trade_duration = 0
                idle_duration = 0
                entry_open = current_open
                pnl = _compute_unrealized_pnl_estimate(
                    Positions.Short,
                    entry_open=entry_open,
                    current_open=current_open,
                    params=params,
                )
                max_unrealized_profit = pnl
                min_unrealized_profit = pnl
        else:
            idle_duration = 0
            if action == Actions.Neutral:
                trade_duration = min(trade_duration + 1, max_trade_duration_cap)
            elif action in (Actions.Long_exit, Actions.Short_exit):
                position = Positions.Neutral
                trade_duration = 0
                idle_duration = 0
                entry_open = current_open

    df = pd.DataFrame(samples)
    df.attrs["reward_params"] = dict(params)

    # Validate critical algorithmic invariants
    _validate_simulation_invariants(df)

    return df


def _validate_simulation_invariants(df: pd.DataFrame) -> None:
    """Fail fast if simulation violates action/state invariants."""

    eps_pnl = float(INTERNAL_GUARDS.get("sim_zero_pnl_epsilon", 1e-12))
    eps_reward = float(INTERNAL_GUARDS.get("sim_zero_reward_epsilon", 1e-12))
    thr_extreme = float(INTERNAL_GUARDS.get("sim_extreme_pnl_threshold", 0.2))

    # INVARIANT 1: Action-position compatibility
    long_exits = df[(df["action"] == 2.0) & (df["position"] != 1.0)]
    short_exits = df[(df["action"] == 4.0) & (df["position"] != 0.0)]
    if len(long_exits) > 0:
        raise AssertionError(f"Sim: {len(long_exits)} Long_exit actions without Long position")
    if len(short_exits) > 0:
        raise AssertionError(f"Sim: {len(short_exits)} Short_exit actions without Short position")

    long_entries = df[(df["action"] == 1.0) & (df["position"] != 0.5)]
    short_entries = df[(df["action"] == 3.0) & (df["position"] != 0.5)]
    if len(long_entries) > 0:
        raise AssertionError(
            f"Sim: {len(long_entries)} Long_enter actions without Neutral position"
        )
    if len(short_entries) > 0:
        raise AssertionError(
            f"Sim: {len(short_entries)} Short_enter actions without Neutral position"
        )

    # INVARIANT 2: Duration logic
    neutral_with_trade = df[(df["position"] == 0.5) & (df["trade_duration"] > 0)]
    if len(neutral_with_trade) > 0:
        raise AssertionError(
            f"Sim: {len(neutral_with_trade)} Neutral positions with non-zero trade_duration"
        )

    inpos_with_idle = df[(df["position"] != 0.5) & (df["idle_duration"] > 0)]
    if len(inpos_with_idle) > 0:
        raise AssertionError(
            f"Sim: {len(inpos_with_idle)} in-position samples with idle_duration > 0"
        )

    # INVARIANT 3: Neutral states have zero PnL (simulation design)
    neutral_with_pnl = df[(df["position"] == 0.5) & (df["pnl"].abs() > eps_pnl)]
    if len(neutral_with_pnl) > 0:
        raise AssertionError(f"Sim: {len(neutral_with_pnl)} Neutral positions with non-zero pnl")

    # INVARIANT 4: Exit rewards only appear on exit actions
    non_exit_with_exit_reward = df[
        (~df["action"].isin([2.0, 4.0])) & (df["reward_exit"].abs() > eps_reward)
    ]
    if len(non_exit_with_exit_reward) > 0:
        raise AssertionError(
            f"Sim: {len(non_exit_with_exit_reward)} non-exit actions have non-zero exit reward"
        )

    # INVARIANT 5: Bounded values
    extreme_pnl = df[(df["pnl"].abs() > thr_extreme)]
    if len(extreme_pnl) > 0:
        max_abs_pnl = float(df["pnl"].abs().max())
        raise AssertionError(
            f"Sim: {len(extreme_pnl)} samples with extreme PnL, max |PnL| = {max_abs_pnl:.6f}"
        )


def _compute_summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics without writing to file."""
    action_summary = df.groupby("action")["reward"].agg(["count", "mean", "std", "min", "max"])
    component_share = df[
        [
            "reward_invalid",
            "reward_idle",
            "reward_hold",
            "reward_exit",
            "reward_shaping",
            "reward_entry_additive",
            "reward_exit_additive",
        ]
    ].apply(lambda col: (col != 0).mean())

    components = [
        "reward_invalid",
        "reward_idle",
        "reward_hold",
        "reward_exit",
        "reward",
    ]
    component_bounds = (
        df[components]
        .agg(["min", "mean", "max"])
        .T.rename(
            columns={
                "min": "component_min",
                "mean": "component_mean",
                "max": "component_max",
            }
        )
        .round(6)
    )

    global_stats = df["reward"].describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

    return {
        "global_stats": global_stats,
        "action_summary": action_summary,
        "component_share": component_share,
        "component_bounds": component_bounds,
    }


def _binned_stats(
    df: pd.DataFrame,
    column: str,
    target: str,
    bins: Iterable[float],
) -> pd.DataFrame:
    """Return count/mean/std/min/max of target grouped by clipped bins of column."""
    bins_arr = np.asarray(list(bins), dtype=float)
    if bins_arr.ndim != 1 or bins_arr.size < 2:
        raise ValueError("Stats: bins must contain at least two edges")
    clipped = df[column].clip(lower=float(bins_arr[0]), upper=float(bins_arr[-1]))
    categories = pd.cut(
        clipped,
        bins=bins_arr,
        include_lowest=True,
        duplicates="drop",
    )
    aggregated = (
        pd.DataFrame({"bin": categories, target: df[target]})
        .dropna(subset=["bin"])
        .groupby("bin", observed=False)[target]
        .agg(["count", "mean", "std", "min", "max"])
    )
    aggregated.index = aggregated.index.astype(str)
    return aggregated


def _compute_relationship_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Return binned stats dict for idle, trade duration and pnl (uniform bins).

    Defensive against missing optional columns (e.g., reward_invalid when synthetic
    test helper omits it). Only present numeric columns are used for correlation.
    Constant columns are dropped and reported.
    """
    reward_params: RewardParams = (
        dict(df.attrs.get("reward_params"))
        if isinstance(df.attrs.get("reward_params"), dict)
        else {}
    )
    max_trade_duration_candles = _get_int_param(reward_params, "max_trade_duration_candles")
    idle_bins = np.linspace(0, max_trade_duration_candles * 3.0, 13)
    trade_bins = np.linspace(0, max_trade_duration_candles * 3.0, 13)
    pnl_min = float(df["pnl"].min())
    pnl_max = float(df["pnl"].max())
    if np.isclose(pnl_min, pnl_max):
        pnl_max = pnl_min + 1e-6
    pnl_bins = np.linspace(pnl_min, pnl_max, 13)

    idle_stats = _binned_stats(df, "idle_duration", "reward_idle", idle_bins)
    hold_stats = _binned_stats(df, "trade_duration", "reward_hold", trade_bins)
    exit_stats = _binned_stats(df, "pnl", "reward_exit", pnl_bins)

    idle_stats = idle_stats.round(6)
    hold_stats = hold_stats.round(6)
    exit_stats = exit_stats.round(6)

    requested_fields = [
        "reward",
        "reward_invalid",
        "reward_idle",
        "reward_hold",
        "reward_exit",
        "pnl",
        "trade_duration",
        "idle_duration",
    ]
    correlation_fields = [c for c in requested_fields if c in df.columns]
    if not correlation_fields:
        correlation = pd.DataFrame()
        constant_cols: list[str] = []
    else:
        numeric_subset = df[correlation_fields]
        constant_cols = [c for c in numeric_subset.columns if numeric_subset[c].nunique() <= 1]
        filtered = numeric_subset.drop(columns=constant_cols) if constant_cols else numeric_subset
        correlation = filtered.corr().round(4)

    return {
        "idle_stats": idle_stats,
        "hold_stats": hold_stats,
        "exit_stats": exit_stats,
        "correlation": correlation,
        "correlation_dropped": constant_cols,
    }


def _compute_representativity_stats(
    df: pd.DataFrame,
    profit_aim: float,
    risk_reward_ratio: float,
) -> dict[str, Any]:
    """Compute representativity statistics for the reward space."""
    pnl_target = float(profit_aim * risk_reward_ratio)
    total = len(df)
    # Map numeric position codes to readable labels to avoid casting Neutral (0.5) to 0
    pos_label_map = {0.0: "Short", 0.5: "Neutral", 1.0: "Long"}
    pos_labeled = df["position"].map(pos_label_map)
    pos_counts = (
        pos_labeled.value_counts().reindex(["Short", "Neutral", "Long"]).fillna(0).astype(int)
    )
    # Actions are encoded as float enum values, casting to int is safe here
    act_counts = df["action"].astype(int).value_counts().sort_index()

    pnl_above_target = float((df["pnl"] > pnl_target).mean())
    pnl_near_target = float(
        ((df["pnl"] >= 0.8 * pnl_target) & (df["pnl"] <= 1.2 * pnl_target)).mean()
    )
    pnl_extreme = float((df["pnl"].abs() >= 0.14).mean())

    duration_overage_share = float((df["duration_ratio"] > 1.0).mean())
    idle_activated = float((df["reward_idle"] != 0).mean())
    hold_activated = float((df["reward_hold"] != 0).mean())
    exit_activated = float((df["reward_exit"] != 0).mean())

    return {
        "total": total,
        "pos_counts": pos_counts,
        "act_counts": act_counts,
        "pnl_above_target": pnl_above_target,
        "pnl_near_target": pnl_near_target,
        "pnl_extreme": pnl_extreme,
        "duration_overage_share": duration_overage_share,
        "idle_activated": idle_activated,
        "hold_activated": hold_activated,
        "exit_activated": exit_activated,
    }


def _perform_feature_analysis(
    df: pd.DataFrame,
    seed: int,
    *,
    skip_partial_dependence: bool = False,
    rf_n_jobs: int = 1,
    perm_n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame], RandomForestRegressor | None]:
    """Compute feature importances using RandomForestRegressor.

    Parameters
    ----------
    df : pd.DataFrame
        Sample data with 'reward' column and feature columns.
    seed : int
        Random seed for reproducibility.
    skip_partial_dependence : bool, default False
        Skip partial dependence computation.
    rf_n_jobs : int, default 1
        Parallel jobs for RandomForestRegressor.
    perm_n_jobs : int, default 1
        Parallel jobs for permutation_importance.

    Returns
    -------
    importance_df : pd.DataFrame
        Feature importances with columns: feature, importance_mean, importance_std.
    analysis_stats : Dict[str, Any]
        Model performance metrics and feature counts.
    partial_deps : Dict[str, pd.DataFrame]
        Partial dependence plots per feature (empty if skipped).
    model : Optional[RandomForestRegressor]
        Fitted model or None on failure.
    """

    def build_feature_analysis_fallback_result(
        *, usable_features, dropped_features, n_train, n_test, top_feature_override=None
    ):
        importance_df = pd.DataFrame(
            {
                "feature": list(usable_features),
                "importance_mean": [np.nan] * len(usable_features),
                "importance_std": [np.nan] * len(usable_features),
            }
        )
        top_feature = top_feature_override
        if top_feature is None:
            top_feature = usable_features[0] if usable_features else None
        analysis_stats = {
            "r2_score": np.nan,
            "n_features": len(usable_features),
            "n_samples_train": int(n_train),
            "n_samples_test": int(n_test),
            "top_feature": top_feature,
            "top_importance": np.nan,
            "dropped_features": list(dropped_features),
            "model_fitted": False,
        }
        return importance_df, analysis_stats, {}, None

    if (
        RandomForestRegressor is None
        or train_test_split is None
        or permutation_importance is None
        or r2_score is None
    ):
        raise ImportError("Stats: feature analysis requires scikit-learn")

    canonical_features = [
        "pnl",
        "trade_duration",
        "idle_duration",
        "duration_ratio",
        "idle_ratio",
        "position",
        "action",
        "is_invalid",
    ]
    available_features = [c for c in canonical_features if c in df.columns]
    # Reward column must exist; if absent produce empty stub outputs
    if "reward" not in df.columns or len(df) == 0 or len(available_features) == 0:
        empty_importance = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
        return (
            empty_importance,
            {
                "r2_score": np.nan,
                "n_features": 0,
                "n_samples_train": 0,
                "n_samples_test": 0,
                "top_feature": None,
                "top_importance": np.nan,
                "dropped_features": [],
                "model_fitted": False,
            },
            {},
            None,
        )

    X = df[available_features].copy()
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    y = df["reward"].copy()

    # Drop wholly NaN or constant columns (provide no signal)
    drop_cols: list[str] = []
    for col in list(X.columns):
        col_series = X[col]
        if col_series.isna().all() or col_series.nunique(dropna=True) <= 1:
            drop_cols.append(col)
    if drop_cols:
        X = X.drop(columns=drop_cols)
    usable_features = list(X.columns)
    if len(usable_features) < 2 or X.isna().any().any():
        return build_feature_analysis_fallback_result(
            usable_features=usable_features,
            dropped_features=drop_cols,
            n_train=0,
            n_test=0,
        )
    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    except Exception:
        return build_feature_analysis_fallback_result(
            usable_features=usable_features,
            dropped_features=drop_cols,
            n_train=0,
            n_test=0,
        )

    model: RandomForestRegressor | None = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=seed,
        n_jobs=rf_n_jobs,
    )

    r2_local: float = np.nan
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2_local = r2_score(y_test, y_pred)
        model_fitted_flag = True
    except Exception:
        # Model failed to fit; drop to stub path
        model = None
        model_fitted_flag = False
        return build_feature_analysis_fallback_result(
            usable_features=usable_features,
            dropped_features=drop_cols,
            n_train=len(X_train),
            n_test=len(X_test),
        )

    # Permutation importance
    try:
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=25,
            random_state=seed,
            n_jobs=perm_n_jobs,
        )
        importance_df = (
            pd.DataFrame(
                {
                    "feature": usable_features,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        importance_df = pd.DataFrame(
            {
                "feature": usable_features,
                "importance_mean": [np.nan] * len(usable_features),
                "importance_std": [np.nan] * len(usable_features),
            }
        )

    # Partial dependence (optional)
    partial_deps: dict[str, pd.DataFrame] = {}
    if model is not None and not skip_partial_dependence:
        for feature in [
            f for f in ["trade_duration", "idle_duration", "pnl"] if f in X_test.columns
        ]:
            try:
                pd_result = partial_dependence(
                    model,
                    X_test,
                    [feature],
                    grid_resolution=50,
                    kind="average",
                )
                value_key = "values" if "values" in pd_result else "grid_values"
                values = pd_result[value_key][0]
                averaged = pd_result["average"][0]
                partial_deps[feature] = pd.DataFrame(
                    {feature: values, "partial_dependence": averaged}
                )
            except Exception:
                continue

    top_feature = (
        importance_df.iloc[0]["feature"]
        if not importance_df.empty and pd.notna(importance_df.iloc[0]["importance_mean"])
        else (usable_features[0] if usable_features else None)
    )
    top_importance = importance_df.iloc[0]["importance_mean"] if not importance_df.empty else np.nan

    analysis_stats = {
        "r2_score": r2_local,
        "n_features": len(usable_features),
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "top_feature": top_feature,
        "top_importance": top_importance,
        "dropped_features": drop_cols,
        "model_fitted": model_fitted_flag,
    }

    return importance_df, analysis_stats, partial_deps, model


def load_real_episodes(path: Path, *, enforce_columns: bool = True) -> pd.DataFrame:
    """Load serialized episodes into normalized DataFrame.

    Parameters
    ----------
    path : Path
        Pickle file path.
    enforce_columns : bool, default True
        Require all expected columns (raise on missing) or fill with NaN.

    Returns
    -------
    pd.DataFrame
        Normalized transitions with required columns.
    """

    try:
        with path.open("rb") as f:
            episodes_data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Data: failed to unpickle '{path}': {e!r}") from e

    # Top-level dict with 'transitions'
    if isinstance(episodes_data, dict) and "transitions" in episodes_data:
        candidate = episodes_data["transitions"]
        if isinstance(candidate, pd.DataFrame):
            df = candidate.copy()
        else:
            try:
                df = pd.DataFrame(list(candidate))
            except TypeError as e:
                raise ValueError(
                    f"Data: 'transitions' in '{path}' is not iterable (type {type(candidate)!r})"
                ) from e
            except Exception as e:
                raise ValueError(
                    f"Data: could not build DataFrame from 'transitions' in '{path}': {e!r}"
                ) from e
    # List of episodes where some entries have 'transitions'
    elif isinstance(episodes_data, list) and any(
        isinstance(e, dict) and "transitions" in e for e in episodes_data
    ):
        all_transitions = []
        skipped = 0
        for episode in episodes_data:
            if isinstance(episode, dict) and "transitions" in episode:
                trans = episode["transitions"]
                if isinstance(trans, pd.DataFrame):
                    all_transitions.extend(trans.to_dict(orient="records"))
                else:
                    try:
                        all_transitions.extend(list(trans))
                    except TypeError as e:
                        raise ValueError(
                            f"Data: episode 'transitions' is not iterable in '{path}' (type {type(trans)!r})"
                        ) from e
            else:
                skipped += 1
        if skipped:
            warnings.warn(
                f"Data: skipped {skipped} episode(s) without 'transitions' when loading '{path}'",
                RewardDiagnosticsWarning,
                stacklevel=2,
            )
        try:
            df = pd.DataFrame(all_transitions)
        except Exception as e:
            raise ValueError(
                f"Data: could not build DataFrame from transitions in '{path}': {e!r}"
            ) from e
    else:
        try:
            if isinstance(episodes_data, pd.DataFrame):
                df = episodes_data.copy()
            else:
                df = pd.DataFrame(episodes_data)
        except Exception as e:
            raise ValueError(
                f"Data: could not convert pickled object from '{path}' to DataFrame: {e!r}"
            ) from e

    # Coerce common numeric fields; warn when values are coerced to NaN
    numeric_expected = {
        "pnl",
        "trade_duration",
        "idle_duration",
        "position",
        "action",
        "reward",
    }

    # Keep optional list stable and explicit
    numeric_optional = {
        "reward_exit",
        "reward_idle",
        "reward_hold",
        "reward_invalid",
        "duration_ratio",
        "idle_ratio",
        "max_unrealized_profit",
        "min_unrealized_profit",
        # Additive / shaping components
        "reward_entry_additive",
        "reward_exit_additive",
        "prev_potential",
        "next_potential",
        "is_invalid",
    }

    for col in list(numeric_expected | numeric_optional):
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            coerced = df[col].isna().sum() - before_na
            if coerced > 0:
                frac = coerced / len(df) if len(df) > 0 else 0.0
                warnings.warn(
                    f"Data: coerced {coerced} non-numeric value(s) ({frac:.1%}) in column '{col}' to NaN when loading '{path}'",
                    RewardDiagnosticsWarning,
                    stacklevel=2,
                )

    # Ensure required columns exist (or fill with NaN if allowed)
    required = {
        "pnl",
        "trade_duration",
        "idle_duration",
        "position",
        "action",
        "reward",
    }
    missing_required = required - set(df.columns)
    if missing_required:
        if enforce_columns:
            raise ValueError(
                f"Data: missing required columns {sorted(missing_required)}. "
                f"Found: {sorted(df.columns)}"
            )
        warnings.warn(
            f"Data: missing columns {sorted(missing_required)}; filled with NaN when loading (enforce_columns=False)",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )
        for col in missing_required:
            df[col] = np.nan

    # Ensure all optional numeric columns exist for schema stability
    for opt_col in numeric_optional:
        if opt_col not in df.columns:
            df[opt_col] = np.nan

    # Drop exact duplicates (rare but can appear after flattening)
    before_dupes = len(df)
    df = df.drop_duplicates()
    if len(df) != before_dupes:
        warnings.warn(
            f"Data: dropped {before_dupes - len(df)} duplicate row(s) when loading '{path}'",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )

    return df


def compute_distribution_shift_metrics(
    synthetic_df: pd.DataFrame,
    real_df: pd.DataFrame,
) -> dict[str, float]:
    """Compute distribution shift metrics between synthetic and real samples.

    Returns KL divergence, JS distance, Wasserstein distance, and KS test
    results for continuous features (pnl, trade_duration, idle_duration).
    """
    metrics = {}
    continuous_features = ["pnl", "trade_duration", "idle_duration"]

    for feature in continuous_features:
        synth_values = synthetic_df[feature].dropna().values
        real_values = real_df[feature].dropna().values

        if len(synth_values) < 10 or len(real_values) < 10:
            continue

        min_val = min(synth_values.min(), real_values.min())
        max_val = max(synth_values.max(), real_values.max())
        # Guard against degenerate distributions (all values identical)
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            continue
        if np.isclose(
            max_val,
            min_val,
            rtol=0,
            atol=float(INTERNAL_GUARDS.get("distribution_identity_epsilon", 1e-12)),
        ):
            # All mass at a single point -> shift metrics are all zero by definition
            metrics[f"{feature}_kl_divergence"] = 0.0
            metrics[f"{feature}_js_distance"] = 0.0
            metrics[f"{feature}_wasserstein"] = 0.0
            metrics[f"{feature}_ks_statistic"] = 0.0
            metrics[f"{feature}_ks_pvalue"] = 1.0
            continue
        bins = np.linspace(min_val, max_val, 50)

        # Use density=False to get counts, then normalize to probabilities
        hist_synth, _ = np.histogram(synth_values, bins=bins, density=False)
        hist_real, _ = np.histogram(real_values, bins=bins, density=False)

        # Add small epsilon to avoid log(0) in KL divergence
        epsilon = float(INTERNAL_GUARDS.get("histogram_epsilon", 1e-10))
        hist_synth = hist_synth + epsilon
        hist_real = hist_real + epsilon
        # Normalize to create probability distributions (sum to 1)
        hist_synth = hist_synth / hist_synth.sum()
        hist_real = hist_real / hist_real.sum()

        # KL(synthetic||real): measures how much synthetic diverges from real
        metrics[f"{feature}_kl_divergence"] = float(entropy(hist_synth, hist_real))
        # JS distance (square root of JS divergence) is symmetric
        metrics[f"{feature}_js_distance"] = float(jensenshannon(hist_synth, hist_real))
        metrics[f"{feature}_wasserstein"] = float(
            stats.wasserstein_distance(synth_values, real_values)
        )

        ks_stat, ks_pval = stats.ks_2samp(synth_values, real_values)
        metrics[f"{feature}_ks_statistic"] = float(ks_stat)
        metrics[f"{feature}_ks_pvalue"] = float(ks_pval)

    # Validate distribution shift metrics bounds
    _validate_distribution_metrics(metrics)

    return metrics


def _validate_distribution_metrics(metrics: dict[str, float]) -> None:
    """Validate mathematical bounds of distribution shift metrics."""
    for key, value in metrics.items():
        if not np.isfinite(value):
            raise AssertionError(f"Stats: distribution metric {key} is not finite: {value}")

        # KL divergence must be >= 0
        if "kl_divergence" in key and value < 0:
            raise AssertionError(f"Stats: KL divergence {key} must be >= 0, got {value:.6f}")

        # JS distance must be in [0, 1]
        if "js_distance" in key and not (0 <= value <= 1):
            raise AssertionError(f"Stats: JS distance {key} must be in [0,1], got {value:.6f}")

        # Wasserstein distance must be >= 0
        if "wasserstein" in key and value < 0:
            raise AssertionError(f"Stats: Wasserstein distance {key} must be >= 0, got {value:.6f}")

        # KS statistic must be in [0, 1]
        if "ks_statistic" in key and not (0 <= value <= 1):
            raise AssertionError(f"Stats: KS statistic {key} must be in [0,1], got {value:.6f}")

        # p-values must be in [0, 1]
        if "pvalue" in key and not (0 <= value <= 1):
            raise AssertionError(f"Stats: p-value {key} must be in [0,1], got {value:.6f}")


def statistical_hypothesis_tests(
    df: pd.DataFrame, *, adjust_method: str = ADJUST_METHODS[0], seed: int = 42
) -> dict[str, Any]:
    """Statistical hypothesis tests (Spearman, Kruskal-Wallis, Mann-Whitney).

    Parameters
    ----------
    df : pd.DataFrame
        Synthetic samples.
    adjust_method : {'none','benjamini_hochberg'}
        Optional p-value multiple test correction.
    seed : int
        Random seed for bootstrap resampling.
    """
    results = {}
    alpha = 0.05

    # Test 1: Idle correlation
    idle_mask = df["reward_idle"] != 0
    if idle_mask.sum() >= 30:
        idle_dur = df.loc[idle_mask, "idle_duration"]
        idle_rew = df.loc[idle_mask, "reward_idle"]

        rho, p_val = stats.spearmanr(idle_dur, idle_rew)

        # Bootstrap CI: resample pairs to preserve bivariate structure
        bootstrap_rhos = []
        n_boot = 1000
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = rng.choice(len(idle_dur), size=len(idle_dur), replace=True)
            boot_rho, _ = stats.spearmanr(idle_dur.iloc[idx], idle_rew.iloc[idx])
            # Handle NaN case if all values identical in bootstrap sample
            if np.isfinite(boot_rho):
                bootstrap_rhos.append(boot_rho)

        if len(bootstrap_rhos) > 0:
            ci_low, ci_high = np.percentile(bootstrap_rhos, [2.5, 97.5])
        else:
            # Fallback if no valid bootstrap samples
            ci_low, ci_high = np.nan, np.nan

        results["idle_correlation"] = {
            "test": "Spearman rank correlation",
            "rho": float(rho),
            "p_value": float(p_val),
            "ci_95": (float(ci_low), float(ci_high)),
            "significant": bool(p_val < alpha),
            "interpretation": "Strong negative correlation expected"
            if rho < -0.7
            else "Check the logic",
            "n_samples": int(idle_mask.sum()),
        }

    # Test 2: Position reward differences
    position_groups = [
        df[df["position"] == pos]["reward"].dropna().values for pos in df["position"].unique()
    ]
    position_groups = [g for g in position_groups if len(g) >= 10]

    if len(position_groups) >= 2:
        h_stat, p_val = stats.kruskal(*position_groups)
        n_total = sum(len(g) for g in position_groups)
        epsilon_sq = h_stat / (n_total - 1) if n_total > 1 else 0.0

        results["position_reward_difference"] = {
            "test": "Kruskal-Wallis H",
            "statistic": float(h_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < alpha),
            "effect_size_epsilon_sq": float(epsilon_sq),
            "interpretation": "Large effect"
            if epsilon_sq > 0.14
            else "Medium effect"
            if epsilon_sq > 0.06
            else "Small effect",
            "n_groups": len(position_groups),
        }

    # Test 3: PnL sign differences
    pnl_positive = df[df["pnl"] > 0]["reward"].dropna()
    pnl_negative = df[df["pnl"] < 0]["reward"].dropna()

    if len(pnl_positive) >= 30 and len(pnl_negative) >= 30:
        u_stat, p_val = stats.mannwhitneyu(pnl_positive, pnl_negative)

        n1, n2 = len(pnl_positive), len(pnl_negative)
        rb = 1.0 - 2.0 * (float(u_stat) / float(n1 * n2)) if n1 > 0 and n2 > 0 else np.nan
        if np.isfinite(rb):
            rb = float(np.clip(rb, -1.0, 1.0))

        results["pnl_sign_reward_difference"] = {
            "test": "Mann-Whitney U (pnl+ vs pnl-)",
            "statistic": float(u_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < alpha),
            "median_pnl_positive": float(pnl_positive.median()),
            "median_pnl_negative": float(pnl_negative.median()),
            "effect_size_rank_biserial": float(rb) if np.isfinite(rb) else np.nan,
        }

    # Optional multiple testing correction (Benjamini-Hochberg)
    _valid_adjust = set(ADJUST_METHODS) | _ADJUST_METHODS_ALIASES
    if adjust_method not in _valid_adjust:
        raise ValueError(
            f"Stats: unsupported adjust_method '{adjust_method}'. "
            f"Expected one of: {list(ADJUST_METHODS)}"
        )
    if adjust_method in _valid_adjust - {ADJUST_METHODS[0]} and results:
        # Collect p-values
        items = list(results.items())
        pvals = np.array([v[1]["p_value"] for v in items])
        m = len(pvals)
        order = np.argsort(pvals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, m + 1)
        adj = pvals * m / ranks
        # enforce monotonicity
        adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
        adj_final = np.empty_like(adj_sorted)
        adj_final[order] = np.clip(adj_sorted, 0, 1)
        # Attach adjusted p-values and recompute significance
        for (name, res), p_adj in zip(items, adj_final, strict=False):
            res["p_value_adj"] = float(p_adj)
            res["significant_adj"] = bool(p_adj < alpha)
            results[name] = res

    # Validate hypothesis test results
    _validate_hypothesis_test_results(results)

    return results


def _validate_hypothesis_test_results(results: dict[str, Any]) -> None:
    """Validate statistical properties of hypothesis test results."""
    for test_name, result in results.items():
        # All p-values must be in [0, 1] or NaN (for cases like constant input)
        if "p_value" in result:
            p_val = result["p_value"]
            if not (np.isnan(p_val) or (0 <= p_val <= 1)):
                raise AssertionError(
                    f"Stats: invalid p-value for {test_name}: {p_val:.6f} not in [0,1] or NaN"
                )

        # Adjusted p-values must also be in [0, 1] or NaN
        if "p_value_adj" in result:
            p_adj = result["p_value_adj"]
            if not (np.isnan(p_adj) or (0 <= p_adj <= 1)):
                raise AssertionError(
                    f"Stats: invalid adjusted p-value for {test_name}: {p_adj:.6f} not in [0,1] or NaN"
                )

        # Effect sizes must be finite and in valid ranges
        if "effect_size_epsilon_sq" in result:
            epsilon_sq = result["effect_size_epsilon_sq"]
            if not np.isfinite(epsilon_sq) or epsilon_sq < 0:
                raise AssertionError(
                    f"Stats: invalid ε² for {test_name}: {epsilon_sq:.6f} (must be finite and >= 0)"
                )

        if "effect_size_rank_biserial" in result:
            rb_corr = result["effect_size_rank_biserial"]
            if not np.isfinite(rb_corr) or not (-1 <= rb_corr <= 1):
                raise AssertionError(
                    f"Stats: invalid rank-biserial correlation for {test_name}: {rb_corr:.6f} "
                    f"(must be finite and in [-1,1])"
                )

        # Correlation coefficients must be in [-1, 1]
        if "rho" in result:
            rho = result["rho"]
            if np.isfinite(rho) and not (-1 <= rho <= 1):
                raise AssertionError(
                    f"Stats: invalid correlation coefficient for {test_name}: {rho:.6f} not in [-1,1]"
                )

        # Confidence intervals must be properly ordered
        if (
            "ci_95" in result
            and isinstance(result["ci_95"], (tuple, list))
            and len(result["ci_95"]) == 2
        ):
            ci_low, ci_high = result["ci_95"]
            if np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low > ci_high:
                raise AssertionError(
                    f"Stats: invalid CI ordering for {test_name}: [{ci_low:.6f}, {ci_high:.6f}]"
                )


def bootstrap_confidence_intervals(
    df: pd.DataFrame,
    metrics: list[str],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
    *,
    strict_diagnostics: bool = False,
) -> dict[str, tuple[float, float, float]]:
    """Compute bootstrap confidence intervals for metric means.

    Returns percentile-based CIs, skipping metrics with <10 samples.
    """
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)

    results = {}

    # Advisory: very low bootstrap counts produce unstable CI widths
    min_rec = int(INTERNAL_GUARDS.get("bootstrap_min_recommended", 200))
    if n_bootstrap < min_rec:
        warnings.warn(
            f"Stats: n_bootstrap={n_bootstrap} < {min_rec}; confidence intervals may be unstable",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )

    # Local RNG to avoid mutating global NumPy RNG state
    rng = np.random.default_rng(seed)

    for metric in metrics:
        if metric not in df.columns:
            continue

        data = df[metric].dropna()
        if len(data) < 10:
            continue

        # Mean point estimate
        point_est = float(data.mean())

        # Constant distribution detection (all values identical / zero variance)
        data_array = data.values  # speed
        if data_array.size == 0:
            continue
        if np.ptp(data_array) == 0:  # zero range -> constant
            if strict_diagnostics:
                # In strict mode, skip constant metrics entirely to avoid degenerate CI raise.
                continue
            # Graceful mode: record degenerate CI; validator will widen.
            results[metric] = (point_est, point_est, point_est)
            continue

        # Bootstrap resampling
        bootstrap_means = []
        n = len(data_array)
        for _ in range(n_bootstrap):
            indices = rng.integers(0, n, size=n)
            bootstrap_means.append(float(data_array[indices].mean()))

        ci_low = float(np.percentile(bootstrap_means, lower_percentile))
        ci_high = float(np.percentile(bootstrap_means, upper_percentile))

        results[metric] = (point_est, ci_low, ci_high)

    # Validate bootstrap confidence intervals
    _validate_bootstrap_results(results, strict_diagnostics=strict_diagnostics)

    return results


def _validate_bootstrap_results(
    results: dict[str, tuple[float, float, float]], *, strict_diagnostics: bool
) -> None:
    """Validate each bootstrap CI: finite bounds, ordered, positive width (adjust or raise)."""
    for metric, (mean, ci_low, ci_high) in results.items():
        # CI bounds must be finite
        if not (np.isfinite(mean) and np.isfinite(ci_low) and np.isfinite(ci_high)):
            raise AssertionError(
                f"Stats: bootstrap CI for {metric}: non-finite values "
                f"(mean={mean}, ci_low={ci_low}, ci_high={ci_high})"
            )

        # CI must be properly ordered
        if not (ci_low <= mean <= ci_high):
            raise AssertionError(
                f"Stats: bootstrap CI for {metric}: ordering violation "
                f"({ci_low:.6f} <= {mean:.6f} <= {ci_high:.6f})"
            )

        # CI width should be positive (non-degenerate)
        width = ci_high - ci_low
        if width <= 0:
            if strict_diagnostics:
                raise AssertionError(
                    f"Stats: bootstrap CI for {metric}: non-positive width {width:.6f}"
                )
            # Graceful mode: expand interval symmetrically
            epsilon = INTERNAL_GUARDS["degenerate_ci_epsilon"] if width == 0 else abs(width) * 1e-06
            center = mean
            # Adjust only if current bounds are identical; otherwise enforce ordering minimally.
            if ci_low == ci_high:
                ci_low = center - epsilon
                ci_high = center + epsilon
            else:
                # Ensure proper ordering if inverted or collapsed negatively.
                lower = min(ci_low, ci_high) - epsilon
                upper = max(ci_low, ci_high) + epsilon
                ci_low, ci_high = lower, upper
            results[metric] = (mean, ci_low, ci_high)
            warnings.warn(
                f"Stats: bootstrap_ci for '{metric}' degenerate (width={width:.6g}); falling back to epsilon={epsilon:.6g}",
                RewardDiagnosticsWarning,
                stacklevel=2,
            )


def distribution_diagnostics(
    df: pd.DataFrame,
    *,
    seed: int | None = None,
    strict_diagnostics: bool = False,
) -> dict[str, Any]:
    """Return mapping col-> diagnostics (tests, moments, entropy, divergences).

    Skips missing columns; selects Shapiro-Wilk when n<=5000 else K2; ignores non-finite intermediates.
    """
    diagnostics = {}
    _ = seed  # placeholder to keep signature for future reproducibility extensions

    for col in ["reward", "pnl", "trade_duration", "idle_duration"]:
        if col not in df.columns:
            continue

        data = df[col].dropna().values
        if len(data) < 10:
            continue

        diagnostics[f"{col}_mean"] = float(np.mean(data))
        diagnostics[f"{col}_std"] = float(np.std(data, ddof=1))
        skew_v = float(stats.skew(data))
        kurt_v = float(stats.kurtosis(data, fisher=True))
        diagnostics[f"{col}_skewness"] = skew_v
        diagnostics[f"{col}_kurtosis"] = kurt_v
        thr = INTERNAL_GUARDS.get("moment_extreme_threshold", 1e4)
        if abs(skew_v) > thr or abs(kurt_v) > thr:
            msg = f"Stats: extreme moment(s) for {col}: skew={skew_v:.3e}, kurtosis={kurt_v:.3e} exceeds threshold {thr}"
            if strict_diagnostics:
                raise AssertionError(msg)
            warnings.warn(msg, RewardDiagnosticsWarning, stacklevel=2)

        if len(data) < 5000:
            sw_stat, sw_pval = stats.shapiro(data)
            diagnostics[f"{col}_shapiro_stat"] = float(sw_stat)
            diagnostics[f"{col}_shapiro_pval"] = float(sw_pval)
            diagnostics[f"{col}_is_normal_shapiro"] = bool(sw_pval > 0.05)

        ad_result = stats.anderson(data, dist="norm")
        diagnostics[f"{col}_anderson_stat"] = float(ad_result.statistic)
        diagnostics[f"{col}_anderson_critical_5pct"] = float(ad_result.critical_values[2])
        diagnostics[f"{col}_is_normal_anderson"] = bool(
            ad_result.statistic < ad_result.critical_values[2]
        )

        (_osm, _osr), (_slope, _intercept, r) = probplot(data, dist="norm", plot=None)
        diagnostics[f"{col}_qq_r_squared"] = float(r**2)

    _validate_distribution_diagnostics(diagnostics, strict_diagnostics=strict_diagnostics)
    return diagnostics


def _validate_distribution_diagnostics(diag: dict[str, Any], *, strict_diagnostics: bool) -> None:
    """Validate mathematical properties of distribution diagnostics.

    Ensures all reported statistics are finite and within theoretical bounds where applicable.
    Invoked automatically inside distribution_diagnostics(); raising AssertionError on violation
    enforces fail-fast semantics consistent with other validation helpers.
    """
    # Pre-compute zero-variance flags to allow graceful handling of undefined higher moments.
    zero_var_columns = set()
    for k, v in diag.items():
        if k.endswith("_std") and (not np.isfinite(v) or v == 0):
            prefix = k[: -len("_std")]
            zero_var_columns.add(prefix)

    for key, value in list(diag.items()):
        if any(suffix in key for suffix in ["_mean", "_std", "_skewness", "_kurtosis"]):  # noqa: SIM102
            if not np.isfinite(value):
                # Graceful degradation for constant distributions: skewness/kurtosis become NaN.
                constant_problem = any(
                    key.startswith(prefix)
                    and (key.endswith("_skewness") or key.endswith("_kurtosis"))
                    for prefix in zero_var_columns
                )
                if constant_problem and not strict_diagnostics:
                    fallback = INTERNAL_GUARDS.get("distribution_constant_fallback_moment", 0.0)
                    diag[key] = fallback
                    warnings.warn(
                        f"Stats: {key} undefined (constant distribution); falling back to {fallback}",
                        RewardDiagnosticsWarning,
                        stacklevel=2,
                    )
                else:
                    raise AssertionError(
                        f"Stats: distribution diagnostic {key} is not finite: {value}"
                    )
        if key.endswith("_shapiro_pval") and not (0 <= value <= 1):
            raise AssertionError(f"Stats: Shapiro p-value {key} must be in [0,1], got {value}")
        if key.endswith("_anderson_stat") or key.endswith("_anderson_critical_5pct"):  # noqa: SIM102
            if not np.isfinite(value):
                prefix = key.rsplit("_", 2)[0]
                if prefix in zero_var_columns and not strict_diagnostics:
                    fallback = INTERNAL_GUARDS.get("distribution_constant_fallback_moment", 0.0)
                    diag[key] = fallback
                    warnings.warn(
                        f"Stats: {key} undefined (constant distribution); falling back to {fallback}",
                        RewardDiagnosticsWarning,
                        stacklevel=2,
                    )
                    continue
                raise AssertionError(f"Stats: Anderson statistic {key} must be finite, got {value}")
        if key.endswith("_qq_r_squared"):  # noqa: SIM102
            if not (isinstance(value, (int, float)) and np.isfinite(value) and 0 <= value <= 1):
                prefix = key[: -len("_qq_r_squared")]
                if prefix in zero_var_columns and not strict_diagnostics:
                    fallback_r2 = INTERNAL_GUARDS.get("distribution_constant_fallback_qq_r2", 1.0)
                    diag[key] = fallback_r2
                    warnings.warn(
                        f"Stats: {key} undefined (constant distribution); falling back to {fallback_r2}",
                        RewardDiagnosticsWarning,
                        stacklevel=2,
                    )
                else:
                    raise AssertionError(f"Stats: Q-Q R^2 {key} must be in [0,1], got {value}")


"""PBRS (Potential-Based Reward Shaping) transforms & helpers."""

# === PBRS TRANSFORM FUNCTIONS ===


def _apply_transform_tanh(value: float) -> float:
    """tanh: tanh(x) in (-1, 1)."""
    return float(math.tanh(value))


def _apply_transform_softsign(value: float) -> float:
    """softsign: x / (1 + |x|) in (-1, 1)."""
    x = value
    return float(x / (1.0 + abs(x)))


def _apply_transform_arctan(value: float) -> float:
    """arctan: (2/pi) * arctan(x) in (-1, 1)."""
    return float((2.0 / math.pi) * math.atan(value))


def _apply_transform_sigmoid(value: float) -> float:
    """sigmoid: 2σ(x) - 1, σ(x) = 1/(1 + e^(-x)) in (-1, 1)."""  # noqa: RUF002
    x = value
    try:
        if x >= 0:
            exp_neg_x = math.exp(-x)
            sigma_x = 1.0 / (1.0 + exp_neg_x)
        else:
            exp_x = math.exp(x)
            sigma_x = exp_x / (exp_x + 1.0)
        return 2.0 * sigma_x - 1.0
    except OverflowError:
        return 1.0 if x > 0 else -1.0


def _apply_transform_asinh(value: float) -> float:
    """asinh: x / sqrt(1 + x^2) in (-1, 1)."""
    return float(value / math.hypot(1.0, value))


def _apply_transform_clip(value: float) -> float:
    """clip: clip(x, -1, 1) in [-1, 1]."""
    return float(np.clip(value, -1.0, 1.0))


def apply_transform(transform_name: TransformFunction | str, value: float, **kwargs: Any) -> float:
    """Apply named transform; unknown names fallback to tanh with warning."""
    transforms = {
        "tanh": _apply_transform_tanh,
        "softsign": _apply_transform_softsign,
        "arctan": _apply_transform_arctan,
        "sigmoid": _apply_transform_sigmoid,
        "asinh": _apply_transform_asinh,
        "clip": _apply_transform_clip,
    }

    if transform_name not in transforms:
        _warn_unknown_mode(
            "potential_transform",
            transform_name,
            transforms.keys(),
            "tanh",
            stacklevel=2,
        )
        return _apply_transform_tanh(value)
    return transforms[transform_name](value, **kwargs)


# === PBRS HELPER FUNCTIONS ===


def _get_potential_gamma(params: RewardParams) -> float:
    """Return validated potential_gamma.

    Process:
    - If NaN -> default POTENTIAL_GAMMA_DEFAULT with warning (missing or unparsable).
    - If outside [0,1] -> clamp + warning including original value.
    - Guarantee returned float ∈ [0,1].
    """
    gamma = _get_float_param(params, "potential_gamma", np.nan)
    if not np.isfinite(gamma):
        warnings.warn(
            f"PBRS: potential_gamma not specified; falling back to {POTENTIAL_GAMMA_DEFAULT}",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )
        return POTENTIAL_GAMMA_DEFAULT

    raw_gamma = float(gamma)
    gamma, reason_parts = _clamp_float_to_bounds("potential_gamma", raw_gamma, strict=False)
    if reason_parts:
        warnings.warn(
            f"PBRS: potential_gamma={raw_gamma} outside [0,1]; falling back to {gamma}",
            RewardDiagnosticsWarning,
            stacklevel=2,
        )
    return float(gamma)


# === PBRS IMPLEMENTATION ===


def _get_fee_rates(params: RewardParams) -> tuple[float, float]:
    """Return clamped `(entry_fee_rate, exit_fee_rate)`.

    Semantics follow Freqtrade's `BaseEnvironment` fee helpers:
    - Entry fee is applied as multiplication: `price * (1 + entry_fee_rate)`.
    - Exit fee is applied as division: `price / (1 + exit_fee_rate)`.

    Notes:
    - Supports two tunables (`entry_fee_rate`, `exit_fee_rate`).
    - Missing/non-finite values fall back to the min bound (usually 0.0).
    - Values are clamped to `_PARAMETER_BOUNDS`.

    This function intentionally clamps (never raises) so callers do not need to
    pre-run `validate_reward_parameters()`.
    """

    raw_entry_fee_rate = _get_float_param(params, "entry_fee_rate")
    raw_exit_fee_rate = _get_float_param(params, "exit_fee_rate")

    entry_fee_rate, _ = _clamp_float_to_bounds(
        "entry_fee_rate",
        float(raw_entry_fee_rate),
        strict=False,
    )
    exit_fee_rate, _ = _clamp_float_to_bounds(
        "exit_fee_rate",
        float(raw_exit_fee_rate),
        strict=False,
    )

    return entry_fee_rate, exit_fee_rate


def _apply_entry_fee(price: float, entry_fee_rate: float) -> float:
    return float(price * (1.0 + entry_fee_rate))


def _apply_exit_fee(price: float, exit_fee_rate: float) -> float:
    denom = 1.0 + exit_fee_rate
    if denom <= 0.0 or not np.isfinite(denom):
        return float(price)
    return float(price / denom)


def _compute_unrealized_pnl_estimate(
    position: Positions,
    *,
    entry_open: float,
    current_open: float,
    params: RewardParams,
) -> float:
    """Estimate unrealized PnL using fee application parity with Freqtrade.

    Long:
        entry_price = entry_open * (1 + entry_fee_rate)
        current_price = current_open / (1 + exit_fee_rate)
        pnl = (current_price - entry_price) / entry_price

    Short:
        entry_price = entry_open / (1 + exit_fee_rate)
        current_price = current_open * (1 + entry_fee_rate)
        pnl = (entry_price - current_price) / entry_price
    """

    if position not in (Positions.Long, Positions.Short):
        return 0.0

    if not np.isfinite(entry_open) or entry_open <= 0.0:
        return 0.0
    if not np.isfinite(current_open) or current_open <= 0.0:
        return 0.0

    entry_fee_rate, exit_fee_rate = _get_fee_rates(params)

    if position == Positions.Long:
        current_price = _apply_exit_fee(current_open, exit_fee_rate)
        entry_price = _apply_entry_fee(entry_open, entry_fee_rate)
        if entry_price == 0.0 or not np.isfinite(entry_price):
            return 0.0
        pnl = (current_price - entry_price) / entry_price
    else:
        current_price = _apply_entry_fee(current_open, entry_fee_rate)
        entry_price = _apply_exit_fee(entry_open, exit_fee_rate)
        if entry_price == 0.0 or not np.isfinite(entry_price):
            return 0.0
        pnl = (entry_price - current_price) / entry_price

    if not np.isfinite(pnl):
        return 0.0
    return float(pnl)


def _loss_duration_multiplier(pnl_ratio: float, risk_reward_ratio: float) -> float:
    """Return duration multiplier for loss-side holds."""

    if not np.isfinite(pnl_ratio) or pnl_ratio >= 0.0:
        return 1.0
    if not np.isfinite(risk_reward_ratio) or risk_reward_ratio <= 0.0:
        return 1.0
    return float(risk_reward_ratio)


def _compute_hold_potential(
    pnl: float,
    pnl_target: float,
    duration_ratio: float,
    risk_reward_ratio: float,
    params: RewardParams,
    base_factor: float,
) -> float:
    """Compute PBRS hold potential Φ(s)."""
    if not _get_bool_param(params, "hold_potential_enabled"):
        return _fail_safely("hold_potential_disabled")

    return _compute_pnl_duration_signal(
        kind="hold_potential",
        pnl=pnl,
        pnl_target=pnl_target,
        duration_ratio=duration_ratio,
        params=params,
        scale_key="hold_potential_ratio",
        gain_key="hold_potential_gain",
        transform_pnl_key="hold_potential_transform_pnl",
        transform_dur_key="hold_potential_transform_duration",
        non_finite_key="non_finite_hold_potential",
        risk_reward_ratio=risk_reward_ratio,
        base_factor=base_factor,
    )


def _compute_entry_additive(
    pnl: float,
    pnl_target: float,
    duration_ratio: float,
    params: RewardParams,
    base_factor: float,
) -> float:
    if not _get_bool_param(params, "entry_additive_enabled"):
        return _fail_safely("entry_additive_disabled")
    return _compute_pnl_duration_signal(
        kind="entry_additive",
        pnl=pnl,
        pnl_target=pnl_target,
        duration_ratio=duration_ratio,
        params=params,
        scale_key="entry_additive_ratio",
        gain_key="entry_additive_gain",
        transform_pnl_key="entry_additive_transform_pnl",
        transform_dur_key="entry_additive_transform_duration",
        non_finite_key="non_finite_entry_additive",
        base_factor=base_factor,
    )


def _compute_exit_additive(
    pnl: float,
    pnl_target: float,
    duration_ratio: float,
    params: RewardParams,
    base_factor: float,
) -> float:
    if not _get_bool_param(params, "exit_additive_enabled"):
        return _fail_safely("exit_additive_disabled")
    return _compute_pnl_duration_signal(
        kind="exit_additive",
        pnl=pnl,
        pnl_target=pnl_target,
        duration_ratio=duration_ratio,
        params=params,
        scale_key="exit_additive_ratio",
        gain_key="exit_additive_gain",
        transform_pnl_key="exit_additive_transform_pnl",
        transform_dur_key="exit_additive_transform_duration",
        non_finite_key="non_finite_exit_additive",
        base_factor=base_factor,
    )


def _compute_exit_potential(prev_potential: float, params: RewardParams) -> float:
    """Exit potential per mode (canonical/non_canonical -> 0; others transform Φ(prev))."""
    mode = _get_str_param(params, "exit_potential_mode")
    if mode == "canonical" or mode == "non_canonical":
        return _fail_safely("canonical_exit_potential")

    if mode == "progressive_release":
        decay = _get_float_param(params, "exit_potential_decay")
        if not np.isfinite(decay) or decay < 0.0:
            warnings.warn(
                f"PBRS: exit_potential_decay={decay} invalid or < 0; falling back to 0.0",
                RewardDiagnosticsWarning,
                stacklevel=2,
            )
            decay = 0.0
        if decay > 1.0:
            warnings.warn(
                f"PBRS: exit_potential_decay={decay} > 1; falling back to 1.0",
                RewardDiagnosticsWarning,
                stacklevel=2,
            )
            decay = 1.0
        next_potential = prev_potential * (1.0 - decay)
    elif mode == "spike_cancel":
        gamma = _get_potential_gamma(params)
        if gamma <= 0.0 or not np.isfinite(gamma):
            next_potential = prev_potential
        else:
            next_potential = prev_potential / gamma
    elif mode == "retain_previous":
        next_potential = prev_potential
    else:
        _warn_unknown_mode(
            "exit_potential_mode",
            mode,
            sorted(ALLOWED_EXIT_POTENTIAL_MODES),
            "canonical (via _fail_safely)",
            stacklevel=2,
        )
        next_potential = _fail_safely("invalid_exit_potential_mode")

    if not np.isfinite(next_potential):
        next_potential = _fail_safely("non_finite_next_exit_potential")
    return float(next_potential)


def compute_pbrs_components(
    current_pnl: float,
    pnl_target: float,
    current_duration_ratio: float,
    next_pnl: float,
    next_duration_ratio: float,
    params: RewardParams,
    *,
    base_factor: float,
    risk_reward_ratio: float,
    prev_potential: float,
    is_exit: bool = False,
    is_entry: bool = False,
) -> tuple[float, float, float, float, float]:
    """Compute potential-based reward shaping (PBRS) components.

    This function computes the PBRS shaping terms.

    Canonical PBRS Formula
    ----------------------
    R'(s,a,s') = R(s,a,s') + Δ(s,a,s')

    where:
        Δ(s,a,s') = gamma * Phi(s') - Phi(s)  (PBRS shaping term)

    Hold Potential Formula
    ----------------------
    Let:
        r_pnl = pnl / pnl_target
        r_dur = max(duration_ratio, 0)
        scale = base_factor · hold_potential_ratio
        g = gain
        T_pnl, T_dur = configured bounded transforms
        m_dur = 1.0 if r_pnl >= 0 else loss_duration_multiplier(r_pnl, risk_reward_ratio)

    Then:
        Φ_hold(s) = scale · 0.5 · [T_pnl(g·r_pnl) + sign(r_pnl)·m_dur·T_dur(g·r_dur)]
    """
    gamma = _get_potential_gamma(params)

    prev_potential = float(prev_potential) if np.isfinite(prev_potential) else 0.0

    exit_mode = _get_str_param(params, "exit_potential_mode")
    canonical_mode = exit_mode == "canonical"

    hold_potential_enabled = _get_bool_param(params, "hold_potential_enabled")

    if is_exit:
        next_potential = _compute_exit_potential(prev_potential, params)
        pbrs_delta = gamma * next_potential - prev_potential
        reward_shaping = pbrs_delta
    else:
        if not hold_potential_enabled:
            next_potential = 0.0
            pbrs_delta = 0.0
            reward_shaping = 0.0
        else:
            next_potential = _compute_hold_potential(
                next_pnl,
                pnl_target,
                next_duration_ratio,
                risk_reward_ratio,
                params,
                base_factor,
            )
            pbrs_delta = gamma * next_potential - prev_potential
            reward_shaping = pbrs_delta

    if canonical_mode:
        entry_additive = 0.0
        exit_additive = 0.0
    else:
        cand_entry_add = _compute_entry_additive(
            next_pnl, pnl_target, next_duration_ratio, params, base_factor
        )
        cand_exit_add = _compute_exit_additive(
            current_pnl, pnl_target, current_duration_ratio, params, base_factor
        )
        entry_additive = cand_entry_add if is_entry else 0.0
        exit_additive = cand_exit_add if is_exit else 0.0

    if not (
        np.isfinite(reward_shaping)
        and np.isfinite(next_potential)
        and np.isfinite(pbrs_delta)
        and np.isfinite(entry_additive)
        and np.isfinite(exit_additive)
    ):
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        float(reward_shaping),
        float(next_potential),
        float(pbrs_delta),
        float(entry_additive),
        float(exit_additive),
    )


def apply_potential_shaping(
    base_reward: float,
    current_pnl: float,
    pnl_target: float,
    current_duration_ratio: float,
    next_pnl: float,
    next_duration_ratio: float,
    params: RewardParams,
    *,
    base_factor: float,
    risk_reward_ratio: float,
    prev_potential: float,
    is_exit: bool = False,
    is_entry: bool = False,
) -> tuple[float, float, float, float, float, float]:
    """Compute shaped reward and PBRS diagnostics.

    .. deprecated::
        This function exists only for backward compatibility with existing tests.
        New code should use :func:`compute_pbrs_components` and compute the total reward manually.

    This is a thin wrapper around `compute_pbrs_components()` that adds PBRS and
    optional additive terms to the provided `base_reward`.

    Returns
    -------
    tuple[float, float, float, float, float, float]
        (reward, reward_shaping, next_potential, pbrs_delta, entry_additive, exit_additive)
    """

    reward_shaping, next_potential, pbrs_delta, entry_additive, exit_additive = (
        compute_pbrs_components(
            current_pnl,
            pnl_target,
            current_duration_ratio,
            next_pnl,
            next_duration_ratio,
            params,
            base_factor=base_factor,
            risk_reward_ratio=risk_reward_ratio,
            prev_potential=prev_potential,
            is_exit=is_exit,
            is_entry=is_entry,
        )
    )

    reward = float(base_reward) + reward_shaping + entry_additive + exit_additive
    if not np.isfinite(reward):
        return float(base_reward), 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        float(reward),
        float(reward_shaping),
        float(next_potential),
        float(pbrs_delta),
        float(entry_additive),
        float(exit_additive),
    )


def _compute_pnl_duration_signal(
    kind: str,
    pnl: float,
    pnl_target: float,
    duration_ratio: float,
    params: RewardParams,
    scale_key: str,
    gain_key: str,
    transform_pnl_key: str,
    transform_dur_key: str,
    non_finite_key: str,
    *,
    base_factor: float,
    risk_reward_ratio: float | None = None,
) -> float:
    """Generic helper for (pnl, duration) bi-component transforms."""
    if not (np.isfinite(pnl) and np.isfinite(pnl_target) and np.isfinite(duration_ratio)):
        return _fail_safely(non_finite_key)
    if pnl_target <= 0.0:
        return _fail_safely(f"{kind}_invalid_pnl_target")

    pnl_ratio = float(pnl / pnl_target)
    duration_ratio = float(max(0.0, duration_ratio))

    ratio = _get_float_param(params, scale_key)
    scale = ratio * base_factor
    gain = _get_float_param(params, gain_key)
    transform_pnl = _get_str_param(params, transform_pnl_key)
    transform_duration = _get_str_param(params, transform_dur_key)

    duration_multiplier = 1.0
    if risk_reward_ratio is not None:
        duration_multiplier = _loss_duration_multiplier(pnl_ratio, risk_reward_ratio)
    if not np.isfinite(duration_multiplier) or duration_multiplier < 0.0:
        duration_multiplier = 1.0

    t_pnl = apply_transform(transform_pnl, gain * pnl_ratio)
    t_dur = apply_transform(transform_duration, gain * duration_ratio)
    value = scale * 0.5 * (t_pnl + np.sign(pnl_ratio) * duration_multiplier * t_dur)
    if not np.isfinite(value):
        return _fail_safely(non_finite_key)
    return float(value)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic stress-test of the ReforceXY reward shaping logic."
    )
    parser.add_argument(
        "--skip_feature_analysis",
        action="store_true",
        help="Skip feature importance and model-based analysis for all scenarios.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20000,
        help="Number of synthetic state/action samples to generate (default: 20000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--skip_partial_dependence",
        action="store_true",
        help="Skip partial dependence computation to speed up analysis.",
    )
    parser.add_argument(
        "--rf_n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for RandomForestRegressor (default: -1 for all CPUs).",
    )
    parser.add_argument(
        "--perm_n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for permutation_importance (default: -1 for all CPUs).",
    )
    parser.add_argument(
        "--stats_seed",
        type=int,
        default=None,
        help="Optional separate seed for statistical analyses (default: same as --seed).",
    )
    parser.add_argument(
        "--base_factor",
        type=float,
        default=100.0,
        help="Base reward scaling factor (default: 100).",
    )
    parser.add_argument(
        "--profit_aim",
        type=float,
        default=0.03,
        help="Target profit threshold (default: 0.03).",
    )
    parser.add_argument(
        "--risk_reward_ratio",
        type=float,
        default=RISK_REWARD_RATIO_DEFAULT,
        help=f"Risk reward ratio multiplier (default: {RISK_REWARD_RATIO_DEFAULT}).",
    )
    parser.add_argument(
        "--max_duration_ratio",
        type=float,
        default=2.5,
        help="Multiple of max duration used when sampling trade/idle durations.",
    )
    parser.add_argument(
        "--pnl_base_std",
        type=float,
        default=0.02,
        help="Base standard deviation for synthetic PnL generation (pre-scaling).",
    )
    parser.add_argument(
        "--pnl_duration_vol_scale",
        type=float,
        default=0.5,
        help="Scaling factor of additional PnL volatility proportional to trade duration ratio.",
    )
    parser.add_argument(
        "--trading_mode",
        type=str.lower,
        choices=list(TRADING_MODES),
        default=TRADING_MODES[0],
        help=("Trading mode to simulate (spot disables shorts). Default: spot."),
    )
    parser.add_argument(
        "--action_masking",
        dest="action_masking",
        action="store_true",
        default=True,
        help="Enable action masking simulation (default: enabled).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("reward_space_outputs"),
        help="Output directory for artifacts (default: reward_space_outputs).",
    )
    parser.add_argument(
        "--params",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override reward parameters, e.g. hold_penalty_ratio=0.5",
    )
    # Dynamically add CLI options for all tunables
    add_tunable_cli_args(parser)
    parser.add_argument(
        "--real_episodes",
        type=Path,
        default=None,
        help="Path to real episodes pickle for distribution shift analysis (optional).",
    )
    parser.add_argument(
        "--pvalue_adjust",
        type=str.lower,
        choices=list(ADJUST_METHODS),
        default=ADJUST_METHODS[0],
        help="Multiple testing correction method for hypothesis tests (default: none).",
    )
    parser.add_argument(
        "--strict_diagnostics",
        action="store_true",
        help=(
            "Enable fail-fast mode for statistical diagnostics: raise on zero-width bootstrap CIs or undefined "
            "skewness/kurtosis/Anderson/Q-Q metrics produced by constant distributions instead of applying graceful replacements."
        ),
    )
    parser.add_argument(
        "--strict_validation",
        dest="strict_validation",
        action="store_true",
        default=True,
        help="Enable strict parameter validation (raise on out-of-bounds or non-finite reward parameters). Default: enabled.",
    )
    parser.add_argument(
        "--bootstrap_resamples",
        type=int,
        default=10000,
        metavar="N",
        help=(
            "Number of bootstrap resamples for confidence intervals (default: 10000). "
            "Lower this (e.g. 200-1000) for faster smoke tests; increase for more stable CI width estimates."
        ),
    )
    parser.add_argument(
        "--unrealized_pnl",
        action="store_true",
        help="Simulate unrealized PnL during holds to feed Φ(s) (optional; default: disabled).",
    )
    return parser


def write_complete_statistical_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    profit_aim: float,
    risk_reward_ratio: float,
    seed: int,
    real_df: pd.DataFrame | None = None,
    *,
    adjust_method: str = ADJUST_METHODS[0],
    stats_seed: int | None = None,
    strict_diagnostics: bool = False,
    bootstrap_resamples: int = 10000,
    skip_partial_dependence: bool = False,
    skip_feature_analysis: bool = False,
    rf_n_jobs: int = -1,
    perm_n_jobs: int = -1,
) -> None:
    """Generate a single comprehensive statistical analysis report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "statistical_analysis.md"

    reward_params: RewardParams = (
        dict(df.attrs.get("reward_params"))
        if isinstance(df.attrs.get("reward_params"), dict)
        else {}
    )
    max_trade_duration_candles = _get_int_param(reward_params, "max_trade_duration_candles")

    # Helpers: consistent Markdown table renderers
    def _fmt_val(v: Any, ndigits: int = 6) -> str:
        try:
            if isinstance(v, numbers.Integral):
                return f"{int(v)}"
            elif isinstance(v, numbers.Real):
                fv = float(v)
                if math.isnan(fv):
                    return "NaN"
                return f"{fv:.{ndigits}f}"
            return str(v)
        except Exception:
            return str(v)

    def _series_to_md(series: pd.Series, value_name: str = "value", ndigits: int = 6) -> str:
        lines = [f"| Metric | {value_name} |", "|--------|-----------|"]
        for k, v in series.items():
            lines.append(f"| {k} | {_fmt_val(v, ndigits)} |")
        return "\n".join(lines) + "\n\n"

    def _df_to_md(df: pd.DataFrame, index_name: str = "index", ndigits: int = 6) -> str:
        if df.empty:
            return "_No data._\n\n"
        # Prepare header
        cols = list(df.columns)
        header = "| " + index_name + " | " + " | ".join(cols) + " |\n"
        sep = "|" + "-" * (len(index_name) + 2)
        for c in cols:
            sep += "|" + "-" * (len(str(c)) + 2)
        sep += "|\n"
        # Rows
        rows: list[str] = []
        for idx, row in df.iterrows():
            vals = [_fmt_val(row[c], ndigits) for c in cols]
            rows.append("| " + str(idx) + " | " + " | ".join(vals) + " |")
        return header + sep + "\n".join(rows) + "\n\n"

    # Compute all statistics
    summary_stats = _compute_summary_stats(df)
    relationship_stats = _compute_relationship_stats(df)
    representativity_stats = _compute_representativity_stats(df, profit_aim, risk_reward_ratio)

    # Model analysis: skip if requested or not enough samples
    importance_df = None
    analysis_stats = None
    partial_deps = {}
    if skip_feature_analysis or len(df) < 4:
        print("CLI: Skipping feature analysis; insufficient samples or flag set")
        # Do NOT create feature_importance.csv when skipped (tests expect absence)
        # Create minimal partial dependence placeholders only if feature analysis was NOT explicitly skipped
        if not skip_feature_analysis and not skip_partial_dependence:
            for feature in ["trade_duration", "idle_duration", "pnl"]:
                (output_dir / f"partial_dependence_{feature}.csv").write_text(
                    f"{feature},partial_dependence\n", encoding="utf-8"
                )
    else:
        try:
            importance_df, analysis_stats, partial_deps, _model = _perform_feature_analysis(
                df,
                seed,
                skip_partial_dependence=skip_partial_dependence,
                rf_n_jobs=rf_n_jobs if isinstance(rf_n_jobs, int) else 1,
                perm_n_jobs=perm_n_jobs if isinstance(perm_n_jobs, int) else 1,
            )
            # Save feature importance CSV
            importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
            # Save partial dependence CSVs
            if not skip_partial_dependence:
                for feature, pd_df in partial_deps.items():
                    pd_df.to_csv(
                        output_dir / f"partial_dependence_{feature}.csv",
                        index=False,
                    )
            else:
                # Create empty files to keep outputs stable
                for feature in ["trade_duration", "idle_duration", "pnl"]:
                    (output_dir / f"partial_dependence_{feature}.csv").write_text(
                        f"{feature},partial_dependence\n", encoding="utf-8"
                    )
        except ImportError:
            print("CLI: Skipping feature analysis; scikit-learn unavailable")
            (output_dir / "feature_importance.csv").write_text(
                "feature,importance_mean,importance_std\n", encoding="utf-8"
            )
            for feature in ["trade_duration", "idle_duration", "pnl"]:
                (output_dir / f"partial_dependence_{feature}.csv").write_text(
                    f"{feature},partial_dependence\n", encoding="utf-8"
                )

    # Enhanced statistics
    test_seed = (
        stats_seed if isinstance(stats_seed, int) else (seed if isinstance(seed, int) else 42)
    )
    hypothesis_tests = statistical_hypothesis_tests(df, adjust_method=adjust_method, seed=test_seed)
    metrics_for_ci = [
        "reward",
        "reward_idle",
        "reward_hold",
        "reward_exit",
        "pnl",
    ]
    # Include PBRS-related metrics when present
    extra_ci_cols = [
        col
        for col in ["reward_shaping", "reward_entry_additive", "reward_exit_additive"]
        if col in df.columns
    ]
    metrics_for_ci.extend(extra_ci_cols)
    bootstrap_ci = bootstrap_confidence_intervals(
        df,
        metrics_for_ci,
        n_bootstrap=int(bootstrap_resamples),
        seed=test_seed,
        strict_diagnostics=strict_diagnostics,
    )
    dist_diagnostics = distribution_diagnostics(
        df, seed=test_seed, strict_diagnostics=strict_diagnostics
    )

    distribution_shift = None
    if real_df is not None:
        distribution_shift = compute_distribution_shift_metrics(df, real_df)

    # Write comprehensive report
    with report_path.open("w", encoding="utf-8") as f:
        # Header
        f.write("# Reward Space Analysis Report\n\n")
        f.write("### Run Configuration\n\n")
        f.write("| Key | Value |\n")
        f.write("|-----|-------|\n")
        f.write(f"| Generated | {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} |\n")
        f.write(f"| Total Samples | {len(df):,} |\n")
        f.write(f"| Random Seed | {seed} |\n")
        # Blank separator to visually group core simulation vs PBRS parameters
        f.write("|  |  |\n")
        # Core PBRS parameters exposed in run configuration if present
        reward_params: RewardParams = (
            dict(df.attrs.get("reward_params"))
            if isinstance(df.attrs.get("reward_params"), dict)
            else {}
        )
        exit_mode = _get_str_param(reward_params, "exit_potential_mode")
        potential_gamma = _get_potential_gamma(reward_params)
        f.write(f"| exit_potential_mode | {exit_mode} |\n")
        f.write(f"| potential_gamma | {potential_gamma} |\n")
        # Additional configuration details
        f.write(f"| max_trade_duration_candles | {max_trade_duration_candles} |\n")
        max_idle_duration_candles = get_max_idle_duration_candles(
            reward_params, max_trade_duration_candles=max_trade_duration_candles
        )
        f.write(f"| max_idle_duration_candles | {max_idle_duration_candles} |\n")
        f.write(f"| strict_diagnostics | {strict_diagnostics} |\n")
        f.write(f"| skip_feature_analysis | {skip_feature_analysis} |\n")
        f.write(f"| skip_partial_dependence | {skip_partial_dependence} |\n")
        f.write(f"| rf_n_jobs | {rf_n_jobs} |\n")
        f.write(f"| perm_n_jobs | {perm_n_jobs} |\n")
        f.write(f"| bootstrap_resamples | {bootstrap_resamples} |\n")
        f.write(f"| pvalue_adjust_method | {adjust_method} |\n")
        # Blank separator before overrides block
        f.write("|  |  |\n")

        overrides_pairs: list[str] = []
        if reward_params:
            for k, default_v in DEFAULT_MODEL_REWARD_PARAMETERS.items():
                if k in ("exit_potential_mode", "potential_gamma"):
                    continue  # already printed explicitly
                try:
                    if k in reward_params and reward_params[k] != default_v:
                        overrides_pairs.append(f"{k}={reward_params[k]}")
                except Exception:
                    continue
        if overrides_pairs:
            f.write(f"| Overrides | {', '.join(sorted(overrides_pairs))} |\n")
        else:
            f.write("| Overrides | (none) |\n")
        f.write("\n")

        f.write("---\n\n")

        # Section 1: Global Statistics
        f.write("## 1. Global Statistics\n\n")

        f.write("### 1.1 Reward Distribution\n\n")
        f.write(_series_to_md(summary_stats["global_stats"], value_name="reward", ndigits=6))

        f.write("### 1.2 Reward Statistics by Action\n\n")
        action_df = summary_stats["action_summary"].copy()
        # Cast index to int for cleaner display (0,1,2,3,4)
        action_df.index = action_df.index.astype(int)
        if action_df.index.name is None:
            action_df.index.name = "action"
        f.write(_df_to_md(action_df, index_name=action_df.index.name, ndigits=6))

        f.write("### 1.3 Component Activation Rates\n\n")
        f.write("Percentage of samples where each reward component is non-zero:\n\n")
        comp_share = summary_stats["component_share"].copy()
        formatted_rows: list[str] = [
            "| Component | Activation Rate |",
            "|-----------|----------------|",
        ]
        # Enforce deterministic logical ordering of components if present
        preferred_order = [
            "reward_invalid",
            "reward_idle",
            "reward_hold",
            "reward_exit",
            "reward_shaping",
            "reward_entry_additive",
            "reward_exit_additive",
        ]
        for comp in preferred_order:
            if comp in comp_share.index:
                val = comp_share.loc[comp]
                formatted_rows.append(f"| {comp} | {val:.1%} |")
        f.write("\n".join(formatted_rows) + "\n\n")

        f.write("### 1.4 Component Value Ranges\n\n")
        bounds_df = summary_stats["component_bounds"].copy()
        if bounds_df.index.name is None:
            bounds_df.index.name = "component"
        f.write(_df_to_md(bounds_df, index_name=bounds_df.index.name, ndigits=6))

        # Section 2: Representativity Analysis
        f.write("---\n\n")
        f.write("## 2. Sample Representativity\n\n")
        f.write("This section evaluates whether the synthetic samples adequately represent ")
        f.write("the full reward space across different market scenarios.\n\n")

        f.write("### 2.1 Position Distribution\n\n")
        f.write(
            _series_to_md(
                representativity_stats["pos_counts"],
                value_name="count",
                ndigits=0,
            )
        )

        f.write("### 2.2 Action Distribution\n\n")
        f.write(
            _series_to_md(
                representativity_stats["act_counts"],
                value_name="count",
                ndigits=0,
            )
        )

        f.write("### 2.3 Critical Regime Coverage\n\n")
        f.write("| Regime | Coverage |\n")
        f.write("|--------|----------|\n")
        f.write(f"| PnL > target | {representativity_stats['pnl_above_target']:.1%} |\n")
        f.write(f"| PnL near target (±20%) | {representativity_stats['pnl_near_target']:.1%} |\n")
        f.write(
            f"| Duration overage (>1.0) | {representativity_stats['duration_overage_share']:.1%} |\n"
        )
        f.write(f"| Extreme PnL (\\|pnl\\|≥0.14) | {representativity_stats['pnl_extreme']:.1%} |\n")
        f.write("\n")

        f.write("### 2.4 Component Activation Rates\n\n")
        f.write("| Component | Activation Rate |\n")
        f.write("|-----------|----------------|\n")
        f.write(f"| Idle penalty | {representativity_stats['idle_activated']:.1%} |\n")
        f.write(f"| Hold penalty | {representativity_stats['hold_activated']:.1%} |\n")
        f.write(f"| Exit reward | {representativity_stats['exit_activated']:.1%} |\n")
        f.write("\n")

        # Section 3: Reward Component Relationships
        f.write("---\n\n")
        f.write("## 3. Reward Component Analysis\n\n")
        f.write("Analysis of how reward components behave under different conditions.\n\n")

        f.write("### 3.1 Idle Penalty vs Duration\n\n")
        if relationship_stats["idle_stats"].empty:
            f.write("_No idle samples present._\n\n")
        else:
            idle_df = relationship_stats["idle_stats"].copy()
            if idle_df.index.name is None:
                idle_df.index.name = "bin"
            f.write(_df_to_md(idle_df, index_name=idle_df.index.name, ndigits=6))

        f.write("### 3.2 Hold Penalty vs Trade Duration\n\n")
        if relationship_stats["hold_stats"].empty:
            f.write("_No hold samples present._\n\n")
        else:
            hold_df = relationship_stats["hold_stats"].copy()
            if hold_df.index.name is None:
                hold_df.index.name = "bin"
            f.write(_df_to_md(hold_df, index_name=hold_df.index.name, ndigits=6))

        f.write("### 3.3 Exit Reward vs PnL\n\n")
        if relationship_stats["exit_stats"].empty:
            f.write("_No exit samples present._\n\n")
        else:
            exit_df = relationship_stats["exit_stats"].copy()
            if exit_df.index.name is None:
                exit_df.index.name = "bin"
            f.write(_df_to_md(exit_df, index_name=exit_df.index.name, ndigits=6))

        f.write("### 3.4 Correlation Matrix\n\n")
        f.write("Pearson correlation coefficients between key metrics:\n\n")
        corr_df = relationship_stats["correlation"].copy()
        if corr_df.index.name is None:
            corr_df.index.name = "feature"
        f.write(_df_to_md(corr_df, index_name=corr_df.index.name, ndigits=4))
        _dropped = relationship_stats.get("correlation_dropped") or []
        if _dropped:
            dropped_strs: list[str] = [str(x) for x in _dropped]
            f.write("\n_Constant features removed: " + ", ".join(dropped_strs) + "._\n\n")

        # Section 3.5: PBRS Analysis
        f.write("### 3.5 PBRS (Potential-Based Reward Shaping) Analysis\n\n")

        # Check if PBRS components are present in the data
        pbrs_components = [
            "reward_shaping",
            "reward_entry_additive",
            "reward_exit_additive",
        ]
        pbrs_present = all(col in df.columns for col in pbrs_components)

        if pbrs_present:
            # PBRS activation rates
            pbrs_activation = {}
            for comp in pbrs_components:
                pbrs_activation[comp.replace("reward_", "")] = (df[comp] != 0).mean()

            f.write("**PBRS Component Activation Rates:**\n\n")
            f.write("| Component | Activation Rate | Description |\n")
            f.write("|-----------|-----------------|-------------|\n")
            f.write(
                f"| Shaping (Φ) | {pbrs_activation['shaping']:.1%} | Potential-based reward shaping |\n"
            )
            f.write(
                f"| Entry Additive | {pbrs_activation['entry_additive']:.1%} | Non-PBRS entry reward |\n"
            )
            f.write(
                f"| Exit Additive | {pbrs_activation['exit_additive']:.1%} | Non-PBRS exit reward |\n"
            )
            f.write("\n")

            # PBRS statistics
            f.write("**PBRS Component Statistics:**\n\n")
            pbrs_stats = df[pbrs_components].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            pbrs_stats_df = pbrs_stats.round(6).T  # Transpose to make it DataFrame-compatible
            pbrs_stats_df.index.name = "component"
            f.write(_df_to_md(pbrs_stats_df, index_name="component", ndigits=6))

            # PBRS metrics
            pbrs_tracing_cols = ["reward_base", "reward_pbrs_delta", "reward_invariance_correction"]
            if all(col in df.columns for col in pbrs_tracing_cols):
                f.write("**PBRS Metrics:**\n\n")
                f.write("Internal decomposition of reward shaping for diagnostic analysis:\n\n")

                # Calculate key metrics
                mean_base = df["reward_base"].mean()
                std_base = df["reward_base"].std()
                mean_pbrs = df["reward_pbrs_delta"].mean()
                std_pbrs = df["reward_pbrs_delta"].std()
                mean_inv_corr = df["reward_invariance_correction"].mean()
                std_inv_corr = df["reward_invariance_correction"].std()
                max_inv_corr = df["reward_invariance_correction"].abs().max()

                # Calculate ratio of |pbrs_delta| / |base_reward| (only where base_reward != 0)
                base_nonzero = df[df["reward_base"].abs() > 1e-10]
                if len(base_nonzero) > 0:
                    pbrs_to_base_ratio = (
                        base_nonzero["reward_pbrs_delta"].abs() / base_nonzero["reward_base"].abs()
                    ).mean()
                else:
                    pbrs_to_base_ratio = 0.0

                f.write("| Metric | Value | Description |\n")
                f.write("|--------|-------|-------------|\n")
                f.write(f"| Mean Base Reward | {mean_base:.6f} | Average reward before PBRS |\n")
                f.write(f"| Std Base Reward | {std_base:.6f} | Variability of base reward |\n")
                f.write(f"| Mean PBRS Delta | {mean_pbrs:.6f} | Average γ·Φ(s')−Φ(s) |\n")  # noqa: RUF001
                f.write(f"| Std PBRS Delta | {std_pbrs:.6f} | Variability of PBRS delta |\n")
                f.write(
                    f"| Mean Invariance Correction | {mean_inv_corr:.6f} | Average reward_shaping − pbrs_delta |\n"  # noqa: RUF001
                )
                f.write(
                    f"| Std Invariance Correction | {std_inv_corr:.6f} | Variability of correction |\n"
                )
                f.write(
                    f"| Max \\|Invariance Correction\\| | {max_inv_corr:.6e} | Peak deviation from pure PBRS |\n"
                )
                f.write(
                    f"| Mean \\|PBRS\\| / \\|Base\\| Ratio | {pbrs_to_base_ratio:.4f} | Shaping magnitude vs base reward |\n"
                )
                f.write("\n")

            # PBRS invariance check
            total_shaping = df["reward_shaping"].sum()
            entry_add_total = df.get("reward_entry_additive", pd.Series([0])).sum()
            exit_add_total = df.get("reward_exit_additive", pd.Series([0])).sum()

            # Get configuration for proper invariance assessment
            reward_params = df.attrs.get("reward_params", {}) if hasattr(df, "attrs") else {}
            exit_potential_mode = _get_str_param(reward_params, "exit_potential_mode", "canonical")
            entry_additive_enabled_raw = _get_bool_param(
                reward_params, "entry_additive_enabled", False
            )
            exit_additive_enabled_raw = _get_bool_param(
                reward_params, "exit_additive_enabled", False
            )

            (
                entry_additive_effective,
                exit_additive_effective,
                additives_suppressed,
            ) = _resolve_additive_enablement(
                exit_potential_mode,
                entry_additive_enabled_raw,
                exit_additive_enabled_raw,
            )

            # True PBRS invariance classification:
            # - Canonical requires canonical mode AND no effective additives.
            # - When `reward_invariance_correction` is present, we use it as the primary
            #   diagnostic (reward_shaping - reward_pbrs_delta).
            # - Otherwise, we fall back to the weaker heuristic |Σ shaping| ≈ 0.
            is_theoretically_invariant = exit_potential_mode == "canonical" and not (
                entry_additive_effective or exit_additive_effective
            )

            has_inv_correction = "reward_invariance_correction" in df.columns
            max_abs_inv_correction: float | None
            if has_inv_correction:
                max_abs_inv_correction = float(df["reward_invariance_correction"].abs().max())
                correction_near_zero = max_abs_inv_correction < PBRS_INVARIANCE_TOL
            else:
                max_abs_inv_correction = None
                correction_near_zero = None
            shaping_near_zero = abs(total_shaping) < PBRS_INVARIANCE_TOL

            suppression_note = ""
            if additives_suppressed:
                suppression_note = (
                    " Additives are suppressed in canonical mode"
                    f" (requested entry_additive_enabled={bool(entry_additive_enabled_raw)},"
                    f" exit_additive_enabled={bool(exit_additive_enabled_raw)})."
                )

            # Prepare invariance summary markdown block
            if is_theoretically_invariant:
                if correction_near_zero is True:
                    invariance_status = "✅ Canonical"
                    invariance_note = (
                        "Theoretical invariance preserved (canonical mode, no additives, max|correction|≈0)."
                        + suppression_note
                    )
                elif correction_near_zero is False:
                    invariance_status = "⚠️ Canonical (with warning)"
                    invariance_note = (
                        "Canonical mode but invariance correction is non-zero"
                        f" (max|correction|={max_abs_inv_correction:.6e})." + suppression_note
                    )
                else:
                    # Fallback: without invariance correction, use Σ shaping as a heuristic.
                    if shaping_near_zero:
                        invariance_status = "✅ Canonical"
                        invariance_note = (
                            "Theoretical invariance preserved (canonical mode, no additives, Σ≈0)."
                            + suppression_note
                        )
                    else:
                        invariance_status = "⚠️ Canonical (with warning)"
                        invariance_note = (
                            "Canonical mode but Σ shaping is non-zero"
                            f" (Σ={total_shaping:.6f}; correction column unavailable)."
                            + suppression_note
                        )
            else:
                invariance_status = "❌ Non-canonical"
                reasons = []
                if exit_potential_mode != "canonical":
                    reasons.append(f"exit_potential_mode='{exit_potential_mode}'")
                if entry_additive_effective or exit_additive_effective:
                    additive_types = []
                    if entry_additive_effective:
                        additive_types.append("entry")
                    if exit_additive_effective:
                        additive_types.append("exit")
                    reasons.append(f"additives={additive_types}")
                invariance_note = f"Modified for flexibility: {', '.join(reasons)}"
            # Summarize PBRS invariance
            f.write("**PBRS Invariance Summary:**\n\n")
            f.write("| Field | Value |\n")
            f.write("|-------|-------|\n")
            f.write(f"| Invariance Status | {invariance_status} |\n")
            f.write(f"| Analysis Note | {invariance_note} |\n")
            f.write(f"| Exit Potential Mode | {exit_potential_mode} |\n")
            f.write(f"| Entry Additive Enabled | {bool(entry_additive_enabled_raw)} |\n")
            f.write(f"| Exit Additive Enabled | {bool(exit_additive_enabled_raw)} |\n")
            f.write(f"| Entry Additive Effective | {bool(entry_additive_effective)} |\n")
            f.write(f"| Exit Additive Effective | {bool(exit_additive_effective)} |\n")
            f.write(f"| Σ Shaping Reward | {total_shaping:.6f} |\n")
            f.write(f"| Abs Σ Shaping Reward | {abs(total_shaping):.6e} |\n")
            f.write(f"| Σ Entry Additive | {entry_add_total:.6f} |\n")
            f.write(f"| Σ Exit Additive | {exit_add_total:.6f} |\n\n")

        else:
            f.write("_PBRS components not present in this analysis._\n\n")

        # Section 4: Feature Importance Analysis
        f.write("---\n\n")
        f.write("## 4. Feature Importance\n\n")
        if skip_feature_analysis or len(df) < 4:
            reason = []
            if skip_feature_analysis:
                reason.append("flag --skip_feature_analysis set")
            if len(df) < 4:
                reason.append("insufficient samples <4")
            f.write("Feature Importance - (skipped)\n\n")
            if skip_partial_dependence:
                f.write(
                    "_Note: --skip_partial_dependence is redundant when feature analysis is skipped._\n\n"
                )
        else:
            if importance_df is None or analysis_stats is None:
                f.write(
                    "_Feature analysis unavailable (scikit-learn not installed); placeholder artifacts generated._\n\n"
                )
            else:
                f.write(
                    "Machine learning analysis to identify which features most influence total reward.\n\n"
                )
                f.write("**Model:** Random Forest Regressor (400 trees)  \n")
                f.write(f"**R² Score:** {analysis_stats['r2_score']:.4f}\n\n")

                f.write("### 4.1 Top 10 Features by Importance\n\n")
                top_imp = importance_df.head(10).copy().reset_index(drop=True)
                # Render as markdown without index column
                header = "| feature | importance_mean | importance_std |\n"
                sep = "|---------|------------------|----------------|\n"
                rows: list[str] = []
                for _, r in top_imp.iterrows():
                    rows.append(
                        f"| {r['feature']} | {_fmt_val(r['importance_mean'], 6)} | {_fmt_val(r['importance_std'], 6)} |"
                    )
                f.write(header + sep + "\n".join(rows) + "\n\n")
                f.write("**Exported Data:**\n")
                f.write("- Full feature importance: `feature_importance.csv`\n")
                if not skip_partial_dependence:
                    f.write("- Partial dependence plots: `partial_dependence_*.csv`\n\n")
                else:
                    f.write(
                        "- Partial dependence plots: (skipped via --skip_partial_dependence)\n\n"
                    )

        # Section 5: Statistical Validation
        if hypothesis_tests:
            f.write("---\n\n")
            f.write("## 5. Statistical Validation\n\n")
            f.write("Rigorous statistical tests to validate reward behavior and relationships.\n\n")

            f.write("### 5.1 Hypothesis Tests\n\n")

            if "idle_correlation" in hypothesis_tests:
                h = hypothesis_tests["idle_correlation"]
                f.write("#### 5.1.1 Idle Duration → Idle Penalty Correlation\n\n")
                f.write(f"**Test Method:** {h['test']}\n\n")
                f.write(f"- Spearman ρ: **{h['rho']:.4f}**\n")  # noqa: RUF001
                f.write(f"- p-value: {h['p_value']:.4g}\n")
                if "p_value_adj" in h:
                    f.write(
                        f"- p-value (adj BH): {h['p_value_adj']:.4g} -> {'✅ Yes' if h['significant_adj'] else '❌ No'} (α=0.05)\n"  # noqa: RUF001
                    )
                f.write(f"- 95% CI: [{h['ci_95'][0]:.4f}, {h['ci_95'][1]:.4f}]\n")
                f.write(f"- CI width: {(h['ci_95'][1] - h['ci_95'][0]):.4f}\n")
                f.write(f"- Sample size: {h['n_samples']:,}\n")
                f.write(f"- Significant (α=0.05): {'✅ Yes' if h['significant'] else '❌ No'}\n")  # noqa: RUF001
                f.write(f"- **Interpretation:** {h['interpretation']}\n\n")

            if "position_reward_difference" in hypothesis_tests:
                h = hypothesis_tests["position_reward_difference"]
                f.write("#### 5.1.2 Position-Based Reward Differences\n\n")
                f.write(f"**Test Method:** {h['test']}\n\n")
                f.write(f"- H-statistic: **{h['statistic']:.4f}**\n")
                f.write(f"- p-value: {h['p_value']:.4g}\n")
                if "p_value_adj" in h:
                    f.write(
                        f"- p-value (adj BH): {h['p_value_adj']:.4g} -> {'✅ Yes' if h['significant_adj'] else '❌ No'} (α=0.05)\n"  # noqa: RUF001
                    )
                f.write(f"- Effect size (ε²): {h['effect_size_epsilon_sq']:.4f}\n")
                f.write(f"- Number of groups: {h['n_groups']}\n")
                f.write(f"- Significant (α=0.05): {'✅ Yes' if h['significant'] else '❌ No'}\n")  # noqa: RUF001
                f.write(f"- **Interpretation:** {h['interpretation']} effect\n\n")

            if "pnl_sign_reward_difference" in hypothesis_tests:
                h = hypothesis_tests["pnl_sign_reward_difference"]
                f.write("#### 5.1.3 Positive vs Negative PnL Comparison\n\n")
                f.write(f"**Test Method:** {h['test']}\n\n")
                f.write(f"- U-statistic: **{h['statistic']:.4f}**\n")
                f.write(f"- p-value: {h['p_value']:.4g}\n")
                if "p_value_adj" in h:
                    f.write(
                        f"- p-value (adj BH): {h['p_value_adj']:.4g} -> {'✅ Yes' if h['significant_adj'] else '❌ No'} (α=0.05)\n"  # noqa: RUF001
                    )
                f.write(f"- Median (PnL+): {h['median_pnl_positive']:.4f}\n")
                f.write(f"- Median (PnL-): {h['median_pnl_negative']:.4f}\n")
                f.write(f"- Significant (α=0.05): {'✅ Yes' if h['significant'] else '❌ No'}\n\n")  # noqa: RUF001

            # Bootstrap CI
            if bootstrap_ci:
                f.write("### 5.2 Confidence Intervals\n\n")
                f.write(
                    "Bootstrap confidence intervals (95%, 10,000 resamples) for key metrics:\n\n"
                )
                f.write("| Metric | Mean | 95% CI Lower | 95% CI Upper | Width |\n")
                f.write("|--------|------|--------------|--------------|-------|\n")
                for metric, (mean, ci_low, ci_high) in bootstrap_ci.items():
                    width = ci_high - ci_low
                    f.write(
                        f"| {metric} | {mean:.6f} | {ci_low:.6f} | {ci_high:.6f} | {width:.6f} |\n"
                    )
                f.write("\n")

            # Distribution diagnostics
            if dist_diagnostics:
                f.write("### 5.3 Distribution Normality Tests\n\n")
                f.write("Statistical tests for normality of key distributions:\n\n")
                for col in ["reward", "pnl", "trade_duration", "idle_duration"]:
                    if f"{col}_mean" in dist_diagnostics:
                        f.write(f"#### {col.replace('_', ' ').title()}\n\n")
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        f.write(f"| Mean | {dist_diagnostics[f'{col}_mean']:.4f} |\n")
                        f.write(f"| Std Dev | {dist_diagnostics[f'{col}_std']:.4f} |\n")
                        f.write(f"| Skewness | {dist_diagnostics[f'{col}_skewness']:.4f} |\n")
                        f.write(f"| Kurtosis | {dist_diagnostics[f'{col}_kurtosis']:.4f} |\n")
                        if f"{col}_shapiro_pval" in dist_diagnostics:
                            is_normal = (
                                "✅ Yes"
                                if dist_diagnostics[f"{col}_is_normal_shapiro"]
                                else "❌ No"
                            )
                            f.write(
                                f"| Normal? (Shapiro-Wilk) | {is_normal} (p={dist_diagnostics[f'{col}_shapiro_pval']:.4e}) |\n"
                            )
                        # Anderson-Darling diagnostics
                        if f"{col}_anderson_stat" in dist_diagnostics:
                            f.write(
                                f"| Anderson-Darling stat | {dist_diagnostics[f'{col}_anderson_stat']:.4f} |\n"
                            )
                            f.write(
                                f"| Anderson 5% critical | {dist_diagnostics[f'{col}_anderson_critical_5pct']:.4f} |\n"
                            )
                            is_normal_anderson = (
                                "✅ Yes"
                                if dist_diagnostics.get(f"{col}_is_normal_anderson", False)
                                else "❌ No"
                            )
                            f.write(f"| Normal? (Anderson-Darling) | {is_normal_anderson} |\n")
                        if f"{col}_qq_r_squared" in dist_diagnostics:
                            f.write(
                                f"| Q-Q Plot R² | {dist_diagnostics[f'{col}_qq_r_squared']:.4f} |\n"
                            )
                        f.write("\n")

            # Distribution shift (if real data provided)
            if distribution_shift:
                f.write("### 5.4 Distribution Shift Analysis\n\n")
                f.write("Comparison between synthetic and real data distributions:\n\n")
                f.write("| Feature | KL Div | JS Dist | Wasserstein | KS Stat | KS p-value |\n")
                f.write("|---------|--------|---------|-------------|---------|------------|\n")

                features = ["pnl", "trade_duration", "idle_duration"]
                for feature in features:
                    kl = distribution_shift.get(f"{feature}_kl_divergence", np.nan)
                    js = distribution_shift.get(f"{feature}_js_distance", np.nan)
                    ws = distribution_shift.get(f"{feature}_wasserstein", np.nan)
                    ks_stat = distribution_shift.get(f"{feature}_ks_statistic", np.nan)
                    ks_p = distribution_shift.get(f"{feature}_ks_pvalue", np.nan)

                    f.write(
                        f"| {feature} | {kl:.4f} | {js:.4f} | {ws:.4f} | {ks_stat:.4f} | {ks_p:.4g} |\n"
                    )
                f.write("\n")
                f.write("**Interpretation Guide:**\n\n")
                f.write("| Metric | Threshold | Meaning |\n")
                f.write("|--------|-----------|--------|\n")
                f.write("| KL Divergence | < 0.3 | ✅ Yes: Good representativeness |\n")
                f.write("| JS Distance | < 0.2 | ✅ Yes: Similar distributions |\n")
                f.write("| KS p-value | > 0.05 | ✅ Yes: No significant difference |\n\n")
            else:
                # Placeholder keeps numbering stable and explicit
                f.write("### 5.4 Distribution Shift Analysis\n\n")
                f.write("_Not performed (no real episodes provided)._\n\n")

        # Footer
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write("This comprehensive report includes:\n\n")
        f.write(
            "1. **Global Statistics** - Overall reward distributions and component activation\n"
        )
        f.write("2. **Sample Representativity** - Coverage of critical market scenarios\n")
        f.write(
            "3. **Component Analysis** - Relationships between rewards and conditions (including PBRS)\n"
        )
        if skip_feature_analysis or len(df) < 4:
            f.write(
                "4. **Feature Importance** - (skipped) Machine learning analysis of key drivers\n"
            )
        else:
            f.write("4. **Feature Importance** - Machine learning analysis of key drivers\n")
        f.write("5. **Statistical Validation** - Hypothesis tests and confidence intervals\n")
        if distribution_shift:
            f.write("6. **Distribution Shift** - Comparison with real trading data\n")
        else:
            f.write("6. **Distribution Shift** - Not performed (no real episodes provided)\n")
        if "reward_shaping" in df.columns:
            _total_shaping = float(df["reward_shaping"].sum())
            if "reward_invariance_correction" in df.columns:
                _max_abs_corr = float(df["reward_invariance_correction"].abs().max())
                _canonical = _max_abs_corr < PBRS_INVARIANCE_TOL
                _pbrs_summary = (
                    "Canonical (max|correction| ≈ 0)"
                    if _canonical
                    else f"Canonical (with warning; max|correction|={_max_abs_corr:.6e})"
                )
            else:
                _canonical = abs(_total_shaping) < PBRS_INVARIANCE_TOL
                _pbrs_summary = (
                    "Canonical (Σ shaping ≈ 0)"
                    if _canonical
                    else f"Canonical (with warning; Σ shaping={_total_shaping:.6f})"
                )

            f.write("7. **PBRS Invariance** - " + _pbrs_summary + "\n")
        f.write("\n")
        f.write("**Generated Files:**\n")
        f.write("- `reward_samples.csv` - Raw synthetic samples\n")
        if not skip_feature_analysis and len(df) >= 4:
            f.write("- `feature_importance.csv` - Complete feature importance rankings\n")
            f.write("- `partial_dependence_*.csv` - Partial dependence data for visualization\n")


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    params = dict(DEFAULT_MODEL_REWARD_PARAMETERS)
    # Merge CLI tunables first (only those explicitly provided)
    _tunable_keys = set(DEFAULT_MODEL_REWARD_PARAMETERS.keys())
    for key, value in vars(args).items():
        if key in _tunable_keys and value is not None:
            params[key] = value
    # Then apply --params KEY=VALUE overrides (highest precedence)
    params.update(parse_overrides(args.params))

    params_validated, adjustments = validate_reward_parameters(
        params, strict=args.strict_validation
    )
    params = params_validated
    if adjustments:
        # Compact adjustments summary (param: original->adjusted [reason])
        adj_lines = [
            f"  - {k}: {v['original']} -> {v['adjusted']} ({v['reason']})"
            for k, v in adjustments.items()
        ]
        print("CLI: Parameter adjustments applied\n" + "\n".join(adj_lines))

    base_factor = _get_float_param(params, "base_factor", float(args.base_factor))
    profit_aim = _get_float_param(params, "profit_aim", float(args.profit_aim))
    risk_reward_ratio = _get_float_param(
        params,
        "risk_reward_ratio",
        _get_float_param(params, "rr", float(args.risk_reward_ratio)),
    )

    cli_action_masking = _to_bool(args.action_masking)
    if "action_masking" in params:
        params["action_masking"] = _to_bool(params["action_masking"])
    else:
        params["action_masking"] = cli_action_masking
    params["unrealized_pnl"] = bool(getattr(args, "unrealized_pnl", False))
    # Propagate strict flag into params for downstream runtime guards
    params["strict_validation"] = bool(getattr(args, "strict_validation", True))

    # Deterministic seeds cascade
    random.seed(args.seed)
    np.random.seed(args.seed)

    df = simulate_samples(
        num_samples=args.num_samples,
        seed=args.seed,
        params=params,
        base_factor=base_factor,
        profit_aim=profit_aim,
        risk_reward_ratio=risk_reward_ratio,
        max_duration_ratio=args.max_duration_ratio,
        trading_mode=args.trading_mode,
        pnl_base_std=args.pnl_base_std,
        pnl_duration_vol_scale=args.pnl_duration_vol_scale,
    )
    # Post-simulation critical NaN validation (non-PBRS structural columns)
    critical_cols = [
        "pnl",
        "trade_duration",
        "idle_duration",
        "position",
        "action",
        "reward",
        "reward_invalid",
        "reward_idle",
        "reward_hold",
        "reward_exit",
    ]
    nan_issues = {
        c: int(df[c].isna().sum()) for c in critical_cols if c in df.columns and df[c].isna().any()
    }
    if nan_issues:
        raise AssertionError(
            "Sim: NaN values detected in critical simulated columns: "
            + ", ".join(f"{k}={v}" for k, v in nan_issues.items())
        )
    # Attach simulation parameters for downstream manifest
    try:
        defaults = {
            a.dest: getattr(a, "default", None) for a in parser._actions if hasattr(a, "dest")
        }
    except Exception:
        defaults = {}
    args_dict = vars(args)

    candidate_keys = [
        "num_samples",
        "seed",
        "out_dir",
        "trading_mode",
        "risk_reward_ratio",
        "profit_aim",
        "max_duration_ratio",
        "pnl_base_std",
        "pnl_duration_vol_scale",
        "rf_n_jobs",
        "perm_n_jobs",
        "skip_feature_analysis",
        "skip_partial_dependence",
        "stats_seed",
        "strict_diagnostics",
        "bootstrap_resamples",
        "pvalue_adjust",
        "real_episodes",
        "unrealized_pnl",
        "action_masking",
    ]

    sim_params: dict[str, Any] = {}
    for k in candidate_keys:
        if k in args_dict:
            v = args_dict[k]
            v_norm = str(v) if isinstance(v, Path) else v
            d = defaults.get(k)
            d_norm = str(d) if isinstance(d, Path) else d
            if d_norm != v_norm:
                sim_params[k] = v_norm

    # Deduplicate any keys that overlap with reward_params (single source of truth)
    for k in list(sim_params.keys()):
        if k in params:
            sim_params.pop(k)

    df.attrs["simulation_params"] = sim_params
    df.attrs["reward_params"] = dict(params)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "reward_samples.csv"
    df.to_csv(csv_path, index=False)
    sample_output_message = f"Samples saved to {csv_path}"

    # Load real episodes if provided
    real_df = None
    if args.real_episodes and args.real_episodes.exists():
        print(f"CLI: Loading real episodes from {args.real_episodes}")
        real_df = load_real_episodes(args.real_episodes)

    # Generate consolidated statistical analysis report (with enhanced tests)
    print("CLI: Generating statistical analysis")

    write_complete_statistical_analysis(
        df,
        args.out_dir,
        profit_aim=profit_aim,
        risk_reward_ratio=risk_reward_ratio,
        seed=args.seed,
        real_df=real_df,
        adjust_method=args.pvalue_adjust,
        stats_seed=(args.stats_seed if getattr(args, "stats_seed", None) is not None else None),
        strict_diagnostics=bool(getattr(args, "strict_diagnostics", False)),
        bootstrap_resamples=getattr(args, "bootstrap_resamples", 10000),
        skip_partial_dependence=bool(getattr(args, "skip_partial_dependence", False)),
        skip_feature_analysis=bool(getattr(args, "skip_feature_analysis", False)),
        rf_n_jobs=int(getattr(args, "rf_n_jobs", -1)),
        perm_n_jobs=int(getattr(args, "perm_n_jobs", -1)),
    )
    print(f"CLI: Statistical analysis saved to {args.out_dir / 'statistical_analysis.md'}")
    # Generate manifest summarizing key metrics
    try:
        manifest_path = args.out_dir / "manifest.json"
        resolved_reward_params: dict[str, Any] = dict(
            params
        )  # already validated/normalized upstream
        manifest: dict[str, Any] = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "num_samples": len(df),
            "seed": int(args.seed),
            "pnl_target": float(profit_aim * risk_reward_ratio),
            "pvalue_adjust_method": args.pvalue_adjust,
            "parameter_adjustments": adjustments,
            "reward_params": resolved_reward_params,
        }
        sim_params_dict = df.attrs.get("simulation_params", {})
        if not isinstance(sim_params_dict, dict):
            sim_params_dict = {}
        sim_params: dict[str, Any] = dict(sim_params_dict)
        if sim_params:
            excluded_for_hash = {"out_dir", "real_episodes"}
            sim_params_for_hash: dict[str, Any] = {
                k: sim_params[k] for k in sim_params if k not in excluded_for_hash
            }
            _hash_source: dict[str, Any] = {
                **{f"sim::{k}": sim_params_for_hash[k] for k in sorted(sim_params_for_hash)},
                **{
                    f"reward::{k}": resolved_reward_params[k]
                    for k in sorted(resolved_reward_params)
                },
            }
            _hash_source_str = json.dumps(_hash_source, sort_keys=True)
            manifest["params_hash"] = hashlib.sha256(_hash_source_str.encode("utf-8")).hexdigest()
            manifest["simulation_params"] = sim_params
        with manifest_path.open("w", encoding="utf-8") as mh:
            json.dump(manifest, mh, indent=2)
        print(f"CLI: Manifest saved to {manifest_path}")
    except Exception as e:
        print(f"CLI: Manifest generation failed; {e}")

    print(f"CLI: Generated {len(df):,} synthetic samples")
    print(f"CLI: {sample_output_message}")
    print(f"CLI: Artifacts saved to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
