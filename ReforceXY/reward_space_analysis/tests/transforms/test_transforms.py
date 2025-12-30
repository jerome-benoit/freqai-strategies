"""Comprehensive transform function tests consolidating duplicate patterns.

This module centralizes transform function tests that appear across multiple test files,
reducing duplication while maintaining full functional coverage for mathematical transforms.
"""

import math
from typing import ClassVar

import pytest

from reward_space_analysis import ALLOWED_TRANSFORMS, apply_transform

from ..constants import TOLERANCE
from ..test_base import RewardSpaceTestBase

pytestmark = pytest.mark.transforms


class TestTransforms(RewardSpaceTestBase):
    """Comprehensive transform function tests with parameterized scenarios."""

    # Transform function test data
    SMOOTH_TRANSFORMS: ClassVar[list[str]] = [t for t in ALLOWED_TRANSFORMS if t != "clip"]
    ALL_TRANSFORMS: ClassVar[list[str]] = list(ALLOWED_TRANSFORMS)

    def test_transform_exact_values(self):
        """Test transform functions produce exact expected values for specific inputs."""
        test_cases = [
            # tanh transform: tanh(x) in (-1, 1)
            ("tanh", [0.0, 1.0, -1.0], [0.0, math.tanh(1.0), math.tanh(-1.0)]),
            # softsign transform: x / (1 + |x|) in (-1, 1)
            ("softsign", [0.0, 1.0, -1.0], [0.0, 0.5, -0.5]),
            # asinh transform: x / sqrt(1 + x^2) in (-1, 1)
            ("asinh", [0.0], [0.0]),  # More complex calculations tested separately
            # arctan transform: (2/π) · arctan(x) in (-1, 1)
            ("arctan", [0.0, 1.0], [0.0, 2.0 / math.pi * math.atan(1.0)]),
            # sigmoid transform: 2σ(x) - 1, σ(x) = 1/(1 + e^(-x)) in (-1, 1)  # noqa: RUF003
            ("sigmoid", [0.0], [0.0]),  # More complex calculations tested separately
            # clip transform: clip(x, -1, 1) in [-1, 1]
            ("clip", [0.0, 0.5, 2.0, -2.0], [0.0, 0.5, 1.0, -1.0]),
        ]

        for transform_name, test_values, expected_values in test_cases:
            for test_val, expected_value in zip(test_values, expected_values, strict=False):
                with self.subTest(
                    transform=transform_name, input=test_val, expected=expected_value
                ):
                    result = apply_transform(transform_name, test_val)
                    self.assertAlmostEqualFloat(
                        result,
                        expected_value,
                        tolerance=TOLERANCE.GENERIC_EQ,
                        msg=f"{transform_name}({test_val}) should equal {expected_value}",
                    )

    def test_transform_bounds_smooth(self):
        """Test that smooth transforms stay within [-1, 1] bounds for extreme values."""
        extreme_values = [-1000000.0, -100.0, -10.0, 10.0, 100.0, 1000000.0]

        for transform_name in self.SMOOTH_TRANSFORMS:
            for extreme_val in extreme_values:
                with self.subTest(transform=transform_name, input=extreme_val):
                    result = apply_transform(transform_name, extreme_val)
                    self.assertTrue(
                        -1.0 <= result <= 1.0,
                        f"{transform_name}({extreme_val}) = {result} should be in [-1, 1]",
                    )

    def test_transform_bounds_clip(self):
        """Test that clip transform stays within [-1, 1] bounds (inclusive)."""
        extreme_values = [-1000.0, -100.0, -2.0, -1.0, 0.0, 1.0, 2.0, 100.0, 1000.0]

        for extreme_val in extreme_values:
            with self.subTest(input=extreme_val):
                result = apply_transform("clip", extreme_val)
                self.assertTrue(
                    -1.0 <= result <= 1.0, f"clip({extreme_val}) = {result} should be in [-1, 1]"
                )

    def test_transform_monotonicity_smooth(self):
        """Test that smooth transforms are monotonically non-decreasing."""
        test_sequence = [-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0]

        for transform_name in self.SMOOTH_TRANSFORMS:
            transform_values = [apply_transform(transform_name, x) for x in test_sequence]

            # Check monotonicity: each value should be <= next value
            for i in range(len(transform_values) - 1):
                with self.subTest(transform=transform_name, index=i):
                    current_val = transform_values[i]
                    next_val = transform_values[i + 1]
                    self.assertLessEqual(
                        current_val,
                        next_val + TOLERANCE.IDENTITY_STRICT,
                        f"{transform_name} not monotonic: values[{i}]={current_val:.6f} > values[{i + 1}]={next_val:.6f}",
                    )

    def test_transform_clip_monotonicity(self):
        """Test that clip transform is monotonically non-decreasing within bounds."""
        test_sequence = [-10.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 10.0]
        transform_values = [apply_transform("clip", x) for x in test_sequence]

        # Expected: [-1.0, -1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0, 1.0]
        for i in range(len(transform_values) - 1):
            with self.subTest(index=i):
                current_val = transform_values[i]
                next_val = transform_values[i + 1]
                self.assertLessEqual(
                    current_val,
                    next_val + TOLERANCE.IDENTITY_STRICT,
                    f"clip not monotonic: values[{i}]={current_val:.6f} > values[{i + 1}]={next_val:.6f}",
                )

    def test_transform_zero_input(self):
        """Test that all transforms return 0.0 for zero input."""
        for transform_name in self.ALL_TRANSFORMS:
            with self.subTest(transform=transform_name):
                result = apply_transform(transform_name, 0.0)
                self.assertAlmostEqualFloat(
                    result,
                    0.0,
                    tolerance=TOLERANCE.IDENTITY_STRICT,
                    msg=f"{transform_name}(0.0) should equal 0.0",
                )

    def test_transform_asinh_symmetry(self):
        """Test asinh transform symmetry: asinh(x) = -asinh(-x)."""
        test_values = [1.2345, 2.0, 5.0, 0.1]

        for test_val in test_values:
            with self.subTest(input=test_val):
                pos_result = apply_transform("asinh", test_val)
                neg_result = apply_transform("asinh", -test_val)
                self.assertAlmostEqualFloat(
                    pos_result,
                    -neg_result,
                    tolerance=TOLERANCE.IDENTITY_STRICT,
                    msg=f"asinh({test_val}) should equal -asinh({-test_val})",
                )

    def test_transform_sigmoid_extreme_behavior(self):
        """Test sigmoid transform behavior at extreme values."""
        # High positive values should approach 1.0
        high_positive = apply_transform("sigmoid", 100.0)
        self.assertTrue(high_positive > 0.99, f"sigmoid(100.0) = {high_positive} should be > 0.99")

        # High negative values should approach -1.0
        high_negative = apply_transform("sigmoid", -100.0)
        self.assertTrue(
            high_negative < -0.99, f"sigmoid(-100.0) = {high_negative} should be < -0.99"
        )

        # Moderate values should be strictly within bounds
        moderate_positive = apply_transform("sigmoid", 10.0)
        moderate_negative = apply_transform("sigmoid", -10.0)

        self.assertTrue(
            -1 < moderate_positive < 1, f"sigmoid(10.0) = {moderate_positive} should be in (-1, 1)"
        )
        self.assertTrue(
            -1 < moderate_negative < 1, f"sigmoid(-10.0) = {moderate_negative} should be in (-1, 1)"
        )

    def test_transform_finite_output(self):
        """Test that all transforms produce finite outputs for reasonable inputs."""
        test_inputs = [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0]

        for transform_name in self.ALL_TRANSFORMS:
            for test_input in test_inputs:
                with self.subTest(transform=transform_name, input=test_input):
                    result = apply_transform(transform_name, test_input)
                    self.assertFinite(result, name=f"{transform_name}({test_input})")

    def test_transform_invalid_fallback(self):
        """Test that invalid transform names fall back to tanh."""
        invalid_result = apply_transform("invalid_transform", 1.0)
        expected_result = math.tanh(1.0)

        self.assertAlmostEqualFloat(
            invalid_result,
            expected_result,
            tolerance=TOLERANCE.IDENTITY_RELAXED,
            msg="Invalid transform should fall back to tanh",
        )

    def test_transform_consistency_comprehensive(self):
        """Test comprehensive consistency across different input ranges for all transforms."""
        transform_descriptions = [
            ("tanh", "Hyperbolic tangent"),
            ("softsign", "Softsign activation"),
            ("asinh", "Inverse hyperbolic sine normalized"),
            ("arctan", "Scaled arctangent"),
            ("sigmoid", "Scaled sigmoid"),
            ("clip", "Hard clipping"),
        ]

        test_ranges = [
            (-100.0, -10.0, 10),  # Large negative range
            (-2.0, -0.1, 10),  # Medium negative range
            (-0.1, 0.1, 10),  # Near-zero range
            (0.1, 2.0, 10),  # Medium positive range
            (10.0, 100.0, 10),  # Large positive range
        ]

        for transform_name, description in transform_descriptions:
            with self.subTest(transform=transform_name, desc=description):
                for start, end, num_points in test_ranges:
                    # Generate test points in this range
                    step = (end - start) / (num_points - 1) if num_points > 1 else 0
                    test_points = [start + i * step for i in range(num_points)]

                    # Apply transform to all points
                    for point in test_points:
                        result = apply_transform(transform_name, point)

                        # Basic validity checks
                        self.assertFinite(result, name=f"{transform_name}({point})")

                        # Bounds checking based on transform type
                        if transform_name in self.SMOOTH_TRANSFORMS:
                            self.assertTrue(
                                -1.0 <= result <= 1.0,
                                f"{transform_name}({point}) = {result} should be in [-1, 1]",
                            )
                        elif transform_name == "clip":
                            self.assertTrue(
                                -1.0 <= result <= 1.0,
                                f"clip({point}) = {result} should be in [-1, 1]",
                            )

    def test_transform_derivative_approximation_smoothness(self):
        """Test smoothness of transforms using finite difference approximation."""
        # Test points around zero where derivatives should be well-behaved
        test_points = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]
        h = 1e-6  # Small step for finite difference

        for transform_name in self.SMOOTH_TRANSFORMS:  # Skip clip as it's not smooth
            with self.subTest(transform=transform_name):
                for x in test_points:
                    # Compute finite difference approximation of derivative
                    f_plus = apply_transform(transform_name, x + h)
                    f_minus = apply_transform(transform_name, x - h)
                    approx_derivative = (f_plus - f_minus) / (2 * h)

                    # Derivative should be finite and non-negative (monotonicity)
                    self.assertFinite(approx_derivative, name=f"d/dx {transform_name}({x})")
                    self.assertGreaterEqual(
                        approx_derivative,
                        -TOLERANCE.IDENTITY_STRICT,  # Allow small numerical errors
                        f"Derivative of {transform_name} at x={x} should be non-negative",
                    )
