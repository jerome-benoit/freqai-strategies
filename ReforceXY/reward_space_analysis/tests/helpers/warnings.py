#!/usr/bin/env python3
"""Warning capture and assertion helpers.

This module provides standardized context managers and utilities for
capturing and validating warnings in tests, reducing boilerplate code
and ensuring consistent warning handling patterns.

Usage:
    >>> from tests.helpers.warnings import assert_diagnostic_warning

    >>> with assert_diagnostic_warning(["exit_factor", "threshold"]) as caught:
    ...     result = calculate_something_that_warns()

    # Assertions are automatic; caught warnings available for inspection
"""

import warnings
from contextlib import contextmanager

import reward_space_analysis

RewardDiagnosticsWarning = getattr(
    reward_space_analysis, "RewardDiagnosticsWarning", RuntimeWarning
)


@contextmanager
def capture_warnings(warning_category: type[Warning] = Warning, always_capture: bool = True):
    """Context manager for capturing warnings during test execution.

    Provides a standardized way to capture warnings with consistent
    configuration across the test suite.

    Args:
        warning_category: Warning category to filter (default: Warning for all)
        always_capture: If True, use simplefilter("always") to capture all warnings

    Yields:
        list: List of captured warning objects

    Example:
        with capture_warnings(RewardDiagnosticsWarning) as caught:
            result = function_that_warns()
        assert len(caught) > 0
    """
    with warnings.catch_warnings(record=True) as caught:
        if always_capture:
            warnings.simplefilter("always", warning_category)
        else:
            warnings.simplefilter("default", warning_category)
        yield caught


@contextmanager
def assert_diagnostic_warning(
    expected_substrings: list[str],
    warning_category: type[Warning] | None = None,
    strict_mode: bool = True,
):
    """Context manager that captures warnings and asserts their presence.

    Automatically validates that expected warning substrings are present
    in captured warning messages. Reduces boilerplate in tests that need
    to validate warning behavior.

    Args:
        expected_substrings: List of substrings expected in warning messages
        warning_category: Warning category to filter (default: use module's default)
        strict_mode: If True, all substrings must be present; if False, at least one

    Yields:
        list: List of captured warning objects for additional inspection

    Raises:
        AssertionError: If expected warnings are not found

    Example:
        with assert_diagnostic_warning(["invalid", "clamped"]) as caught:
            result = function_with_invalid_param()
    """
    category = warning_category if warning_category is not None else RewardDiagnosticsWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category)
        yield caught

    # Filter to only warnings of the expected category
    filtered = [w for w in caught if issubclass(w.category, category)]

    if not filtered:
        raise AssertionError(
            f"Expected {category.__name__} but no warnings of that category were captured. "
            f"Total warnings: {len(caught)}"
        )

    # Check for expected substrings
    all_messages = " ".join(str(w.message) for w in filtered)

    if strict_mode:
        # All substrings must be present
        for substring in expected_substrings:
            if substring not in all_messages:
                raise AssertionError(
                    f"Expected substring '{substring}' not found in warning messages. "
                    f"Captured messages: {all_messages}"
                )
    else:
        # At least one substring must be present
        found = any(substring in all_messages for substring in expected_substrings)
        if not found:
            raise AssertionError(
                f"None of the expected substrings {expected_substrings} found in warnings. "
                f"Captured messages: {all_messages}"
            )


__all__ = [
    "assert_diagnostic_warning",
    "capture_warnings",
]
