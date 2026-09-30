"""Extrema/threshold surfaces, distance/TOPSIS algebra and multi-objective trial selection; requires the Freqtrade QA image."""

import unittest

import numpy as np
import optuna
import pandas as pd
from EnumErrors import enum_error_message
from LabelTransformer import EXTREMA_SELECTION_METHODS, SKIMAGE_THRESHOLD_METHODS
from qa_support import REGRESSOR_MODULE, QaTestCase

from quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 import QuickAdapterRegressorV3

# The module logger is named after the module's __name__, so the shared path constant
# is also the logger name assertLogs/assertNoLogs needs.
LOGGER_NAME = REGRESSOR_MODULE

# §5.6: the atol constant is scoped to constant-column detection on the normalised
# objective matrix, so it is the tolerance for that family and for TOPSIS scores.
NORMALIZED_ATOL = QuickAdapterRegressorV3._NON_CONSTANT_OBJECTIVE_ATOL
# §5.6: metric distances and pairwise distance sums are O(1)-O(k) quantities.
DISTANCE_RTOL = 1e-12

MINIMIZE = optuna.study.StudyDirection.MINIMIZE
MAXIMIZE = optuna.study.StudyDirection.MAXIMIZE

RAISE = "raise"
WARN = "warn"
NONE = "none"

UNIT_WEIGHTS = np.ones(2)
#: A three-trial front whose two objectives sum to a constant per row, so the
#: distance to the all-ones ideal and to the all-zeros anti-ideal add to a fixed total.
FRONT = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
IDEAL = np.ones(2)
ANTI_IDEAL = np.zeros(2)
IDEAL_2D = np.ones((1, 2))
#: Power means 0, 1 and 2 against a reference of 1: below, at, and above the ideal.
BELOW_IDEAL = np.array([[0.0, 0.0]])
AT_IDEAL = np.array([[1.0, 1.0]])
ABOVE_IDEAL = np.array([[2.0, 2.0]])
#: A row whose two objectives are unequal, so its power mean genuinely depends on p.
ASYMMETRIC_ROW = np.array([[0.25, 0.75]])


def front() -> np.ndarray:
    """Return a three-trial, two-objective normalised objective matrix.

    The rows are the all-zeros trial, the all-ones trial and a midpoint trial, so the
    aggregate scores can be read straight off their distance to the all-ones ideal.
    """
    return np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])


def raw_front() -> np.ndarray:
    """Return the un-normalised values `front()` is the MINIMIZE normalisation of."""
    return np.array([[9.0, 9.0], [1.0, 1.0], [5.0, 5.0]])


def unique_rows(n_rows: int) -> np.ndarray:
    """Return a two-column matrix of `n_rows` distinct rows, no RNG involved."""
    return np.arange(n_rows * 2, dtype=float).reshape(n_rows, 2)


def repeated_rows(n_rows: int, n_distinct: int) -> np.ndarray:
    """Return a front of `n_rows` rows cycling through `n_distinct` distinct points."""
    distinct = np.arange(n_distinct * 2, dtype=float).reshape(n_distinct, 2)
    return distinct[np.arange(n_rows) % n_distinct]


def frozen_trial(number: int) -> optuna.trial.FrozenTrial:
    """Build a completed FrozenTrial carrying only its trial number, which is all selection reads."""
    return optuna.trial.FrozenTrial(
        number=number,
        state=optuna.trial.TrialState.COMPLETE,
        value=float(number),
        datetime_start=None,
        datetime_complete=None,
        params={},
        distributions={},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        trial_id=number,
    )


def trials(*numbers: int) -> list[optuna.trial.FrozenTrial]:
    return [frozen_trial(number) for number in numbers]


def threshold_spy() -> tuple[list, object]:
    """Return a call log and a threshold function that records its input and returns a marker."""
    calls: list[np.ndarray] = []

    def spy(values: np.ndarray) -> float:
        calls.append(np.array(values, copy=True))
        return 99.0

    return calls, spy


class RegressorSelectionTest(QaTestCase):
    # ------------------------------------------------------------------ category and defaults

    def test_every_selection_method_maps_to_its_category(self):
        expected = {
            "compromise_programming": "distance",
            "topsis": "distance",
            "kmeans": "cluster",
            "kmeans2": "cluster",
            "kmedoids": "cluster",
            "knn": "density",
            "medoid": "density",
        }
        for method, category in expected.items():
            with self.subTest(method=method):
                self.assertEqual(QuickAdapterRegressorV3._get_selection_category(method), category)
        grouped = {
            method: category
            for category, methods in QuickAdapterRegressorV3._SELECTION_CATEGORIES.items()
            for method in methods
        }
        self.assertEqual(grouped, expected)

    def test_an_unknown_selection_method_has_no_category(self):
        self.assertIsNone(QuickAdapterRegressorV3._get_selection_category("kmedoid"))

    def test_only_minkowski_and_power_mean_carry_a_p_order_default(self):
        # The other fifteen metrics are either SciPy-native or a fixed named power mean,
        # so a bare p-order has nothing to resolve to.
        expected = {"minkowski": 2.0, "power_mean": 1.0}
        for metric in QuickAdapterRegressorV3._DISTANCE_METRICS:
            with self.subTest(metric=metric):
                self.assertEqual(
                    QuickAdapterRegressorV3._get_label_p_order_default(metric), expected.get(metric)
                )

    def test_density_methods_default_to_the_scipy_compatible_metric_of_their_route(self):
        self.assertEqual(
            QuickAdapterRegressorV3._get_label_density_metric_default("knn"),
            QuickAdapterRegressorV3._METRIC_MINKOWSKI,
        )
        self.assertEqual(
            QuickAdapterRegressorV3._get_label_density_metric_default("medoid"),
            QuickAdapterRegressorV3._METRIC_EUCLIDEAN,
        )

    def test_a_density_method_outside_the_known_set_has_no_metric_default(self):
        self.assertIsNone(QuickAdapterRegressorV3._get_label_density_metric_default("kde"))

    def test_only_the_two_aggregations_that_reduce_over_neighbours_carry_a_param_default(self):
        # min and max are order statistics and take no parameter at all.
        self.assertEqual(
            QuickAdapterRegressorV3._get_label_density_aggregation_param_default("power_mean"), 1.0
        )
        self.assertEqual(
            QuickAdapterRegressorV3._get_label_density_aggregation_param_default("quantile"), 0.5
        )
        for aggregation in ("min", "max"):
            with self.subTest(aggregation=aggregation):
                self.assertIsNone(
                    QuickAdapterRegressorV3._get_label_density_aggregation_param_default(
                        aggregation
                    )
                )

    # ------------------------------------------------------------------ objective normalisation

    def test_minimise_maps_the_worst_value_to_zero_and_the_best_to_one(self):
        raw = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._normalize_objective_values(raw, [MINIMIZE, MINIMIZE]),
            np.array([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]]),
            rtol=0.0,
            atol=NORMALIZED_ATOL,
        )

    def test_maximise_is_the_mirror_image_of_minimise(self):
        raw = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]])
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._normalize_objective_values(raw, [MAXIMIZE, MAXIMIZE]),
            np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
            rtol=0.0,
            atol=NORMALIZED_ATOL,
        )

    def test_each_objective_is_scaled_against_its_own_range(self):
        # A shared range would compress the narrow column; per-column scaling does not.
        raw = np.array([[0.0, 100.0], [1.0, 200.0], [2.0, 900.0]])
        result = QuickAdapterRegressorV3._normalize_objective_values(raw, [MINIMIZE, MINIMIZE])
        np.testing.assert_allclose(
            result[:, 0], np.array([1.0, 0.5, 0.0]), rtol=0.0, atol=NORMALIZED_ATOL
        )
        np.testing.assert_allclose(
            result[:, 1], np.array([1.0, 0.875, 0.0]), rtol=0.0, atol=NORMALIZED_ATOL
        )

    def test_each_objective_resolves_its_own_direction(self):
        raw = np.array([[1.0], [3.0], [5.0]])
        result = QuickAdapterRegressorV3._normalize_objective_values(raw, [MAXIMIZE])
        np.testing.assert_allclose(
            result[:, 0], np.array([0.0, 0.5, 1.0]), rtol=0.0, atol=NORMALIZED_ATOL
        )

    def test_a_positive_infinity_is_the_worst_value_when_minimising(self):
        raw = np.array([[1.0], [3.0], [5.0], [np.inf]])
        result = QuickAdapterRegressorV3._normalize_objective_values(raw, [MINIMIZE])
        self.assertAlmostEqual(float(result[-1, 0]), 0.0, places=9)

    def test_a_negative_infinity_is_the_worst_value_when_maximising(self):
        raw = np.array([[1.0], [3.0], [5.0], [-np.inf]])
        result = QuickAdapterRegressorV3._normalize_objective_values(raw, [MAXIMIZE])
        self.assertAlmostEqual(float(result[-1, 0]), 0.0, places=9)

    def test_an_infinite_lands_on_the_end_of_the_unit_interval_its_direction_puts_it_at(self):
        raw = np.array([[1.0], [3.0], [5.0], [np.inf], [-np.inf]])
        for direction, positive_infinity, negative_infinity in (
            (MINIMIZE, 0.0, 1.0),
            (MAXIMIZE, 1.0, 0.0),
        ):
            with self.subTest(direction=direction):
                result = QuickAdapterRegressorV3._normalize_objective_values(raw, [direction])
                self.assertAlmostEqual(float(result[3, 0]), positive_infinity, places=9)
                self.assertAlmostEqual(float(result[4, 0]), negative_infinity, places=9)
                # The finite samples are scaled between the two infinities, so the
                # direction's worst infinity is the worst score of the column and no
                # finite sample escapes the unit interval.
                low, high = sorted((positive_infinity, negative_infinity))
                self.assertTrue(np.all(result[:3, 0] >= low))
                self.assertTrue(np.all(result[:3, 0] <= high))

    def test_a_degenerate_range_with_no_infinity_resolves_to_the_midpoint(self):
        raw = np.array([[2.0], [2.0], [2.0]])
        for direction in (MINIMIZE, MAXIMIZE):
            with self.subTest(direction=direction):
                result = QuickAdapterRegressorV3._normalize_objective_values(raw, [direction])
                np.testing.assert_allclose(
                    result, np.full((3, 1), 0.5), rtol=0.0, atol=NORMALIZED_ATOL
                )

    def test_a_degenerate_range_below_a_positive_infinity_resolves_away_from_it(self):
        # With no spread to scale, the finite samples take the value that keeps the
        # column ordered: better than +inf under MAXIMIZE, worse under MINIMIZE.
        raw = np.array([[1.0], [1.0], [np.inf]])
        self.assertAlmostEqual(
            float(QuickAdapterRegressorV3._normalize_objective_values(raw, [MINIMIZE])[0, 0]),
            1.0,
            places=9,
        )
        self.assertAlmostEqual(
            float(QuickAdapterRegressorV3._normalize_objective_values(raw, [MAXIMIZE])[0, 0]),
            0.0,
            places=9,
        )

    def test_a_degenerate_range_above_a_negative_infinity_resolves_away_from_it(self):
        raw = np.array([[1.0], [1.0], [-np.inf]])
        self.assertAlmostEqual(
            float(QuickAdapterRegressorV3._normalize_objective_values(raw, [MINIMIZE])[0, 0]),
            0.0,
            places=9,
        )
        self.assertAlmostEqual(
            float(QuickAdapterRegressorV3._normalize_objective_values(raw, [MAXIMIZE])[0, 0]),
            1.0,
            places=9,
        )

    def test_a_degenerate_range_between_both_infinities_resolves_to_the_midpoint(self):
        raw = np.array([[1.0], [1.0], [np.inf], [-np.inf]])
        for direction in (MINIMIZE, MAXIMIZE):
            with self.subTest(direction=direction):
                result = QuickAdapterRegressorV3._normalize_objective_values(raw, [direction])
                np.testing.assert_allclose(
                    result[:2, 0], np.array([0.5, 0.5]), rtol=0.0, atol=NORMALIZED_ATOL
                )
                # The infinities keep their own ends: the pair still spans the unit interval.
                self.assertAlmostEqual(float(result[2:].max()), 1.0, places=9)
                self.assertAlmostEqual(float(result[2:].min()), 0.0, places=9)

    def test_a_range_below_ten_eps_is_treated_as_degenerate(self):
        # 10 * finfo.eps, not finfo.eps: the guard exists so float noise on an
        # otherwise-constant column cannot produce a meaningless [0,1] scale. A range
        # of 5 * eps is degenerate at 10 * eps and a real spread at 1 * eps, so both
        # sides of the declared threshold are pinned. The column starts at 0.0 so the
        # stored range is exactly the delta under test.
        eps = np.finfo(float).eps
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._normalize_objective_values(
                np.array([[0.0], [5 * eps]]), [MINIMIZE]
            ),
            np.array([[0.5], [0.5]]),
            rtol=0.0,
            atol=NORMALIZED_ATOL,
        )
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._normalize_objective_values(
                np.array([[0.0], [100 * eps]]), [MINIMIZE]
            ),
            np.array([[1.0], [0.0]]),
            rtol=0.0,
            atol=NORMALIZED_ATOL,
        )

    def test_a_range_too_wide_to_subtract_is_refused_rather_than_scaled_to_nan(self):
        # max - min overflows to inf for a column spanning more than float64 can hold,
        # so the scaled value is inf/inf = nan. Refusing beats ranking on a NaN score.
        for direction in (MINIMIZE, MAXIMIZE):
            with (
                self.subTest(direction=direction),
                self.assertRaisesRegex(
                    ValueError, "must contain only finite values after normalization"
                ),
            ):
                QuickAdapterRegressorV3._normalize_objective_values(
                    np.array([[-1e308], [0.0], [1e308]]), [direction]
                )

    def test_a_column_with_no_finite_sample_keeps_only_its_infinite_endpoints(self):
        raw = np.array([[np.inf], [-np.inf]])
        for direction, expected in ((MINIMIZE, [0.0, 1.0]), (MAXIMIZE, [1.0, 0.0])):
            with self.subTest(direction=direction):
                result = QuickAdapterRegressorV3._normalize_objective_values(raw, [direction])
                np.testing.assert_allclose(
                    result[:, 0], np.array(expected), rtol=0.0, atol=NORMALIZED_ATOL
                )

    def test_a_failed_trial_is_normalised_as_the_worst_value_rather_than_rejected(self):
        # NaN is neither finite nor an infinity, so it keeps the zeros the output was
        # allocated with. A single non-finite objective must not void a whole trial.
        raw = np.array([[np.nan, 1.0], [2.0, 2.0]])
        result = QuickAdapterRegressorV3._normalize_objective_values(raw, [MINIMIZE, MINIMIZE])
        self.assertAlmostEqual(float(result[0, 0]), 0.0, places=9)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_a_front_that_is_not_two_dimensional_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be 2-dimensional"):
            QuickAdapterRegressorV3._normalize_objective_values(np.array([1.0, 2.0]), [MINIMIZE])

    def test_an_empty_front_or_an_empty_objective_axis_is_rejected(self):
        for raw, directions in (
            (np.zeros((0, 2)), [MINIMIZE, MINIMIZE]),
            (np.zeros((2, 0)), []),
        ):
            with (
                self.subTest(shape=raw.shape),
                self.assertRaisesRegex(ValueError, "at least one sample and one objective"),
            ):
                QuickAdapterRegressorV3._normalize_objective_values(raw, directions)

    def test_a_direction_count_that_disagrees_with_the_objective_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must match number of objectives"):
            QuickAdapterRegressorV3._normalize_objective_values(np.ones((2, 2)), [MINIMIZE])

    # ------------------------------------------------------------------ constant-column detection

    def test_the_returned_positions_are_column_indices_in_intp_dtype(self):
        # A one-column non-constant matrix yields exactly one index, and assertEqual on a
        # one-element ndarray passes silently instead of raising, so the list form is required.
        indices = QuickAdapterRegressorV3._non_constant_objective_indices(np.array([[0.0], [1.0]]))
        self.assertEqual(indices.dtype, np.dtype(np.intp))
        self.assertEqual(indices.tolist(), [0])

    def test_only_the_varying_columns_of_a_mixed_front_are_reported(self):
        matrix = np.array(
            [
                [0.0, 0.5, 0.25, 0.75],
                [0.0, 0.5, 0.75, 0.25],
                [0.0, 0.5, 0.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(
            QuickAdapterRegressorV3._non_constant_objective_indices(matrix), np.array([2, 3])
        )

    def test_an_entirely_constant_front_reports_no_columns_at_all(self):
        indices = QuickAdapterRegressorV3._non_constant_objective_indices(np.ones((4, 3)))
        self.assertEqual(indices.dtype, np.dtype(np.intp))
        self.assertEqual(indices.tolist(), [])

    def test_the_tolerance_is_absolute_so_a_ninth_decimal_column_is_constant(self):
        # atol 1e-8 on a [0,1] column: a spread of 1e-9 is inside it, 1e-6 is outside.
        inside = np.array([[0.0], [1e-9]])
        outside = np.array([[0.0], [1e-6]])
        self.assertEqual(
            QuickAdapterRegressorV3._non_constant_objective_indices(inside).tolist(), []
        )
        self.assertEqual(
            QuickAdapterRegressorV3._non_constant_objective_indices(outside).tolist(), [0]
        )

    def test_a_ninth_decimal_spread_stays_reported_even_on_a_large_magnitude_column(self):
        # rtol=0 is the point: numpy's default rtol=1e-5 would call this column constant
        # because the column magnitude is 1e9, and silently drop a live objective.
        column = np.array([[1e9], [1e9 + 1.0]])
        self.assertTrue(np.allclose(column[:, 0], 1e9))
        self.assertEqual(
            QuickAdapterRegressorV3._non_constant_objective_indices(column).tolist(), [0]
        )

    def test_a_mid_magnitude_spread_is_not_absorbed_by_a_relative_tolerance(self):
        # 1e-6 against a 0.5 reference is 2e-6 relatively: inside numpy's default rtol,
        # outside the absolute 1e-8 the source declares.
        column = np.array([[0.5], [0.5 + 1e-6]])
        self.assertTrue(np.allclose(column[:, 0], 0.5))
        self.assertEqual(
            QuickAdapterRegressorV3._non_constant_objective_indices(column).tolist(), [0]
        )

    def test_the_reference_for_each_column_is_its_own_first_row(self):
        # Row 0 differs from itself nowhere; a column is constant exactly when every
        # later row matches row 0, not when the column's rows all differ from each other.
        matrix = np.array([[0.25, 0.75], [0.25, 0.5], [0.25, 0.75]])
        np.testing.assert_array_equal(
            QuickAdapterRegressorV3._non_constant_objective_indices(matrix), np.array([1])
        )

    def test_a_non_finite_entry_is_refused_rather_than_treated_as_a_spread(self):
        for bad in (np.array([1.0, 2.0]), np.array([[1.0], [np.nan]]), np.array([[1.0], [np.inf]])):
            with self.subTest(bad=bad.tolist()), self.assertRaises(ValueError):
                QuickAdapterRegressorV3._non_constant_objective_indices(bad)

    # ------------------------------------------------------------------ distance family

    def test_a_scipy_metric_branch_distances_to_the_ideal(self):
        for metric, expected in (
            ("euclidean", [np.sqrt(2.0), 1.0, np.sqrt(0.5)]),
            ("cityblock", [2.0, 1.0, 1.0]),
            ("chebyshev", [1.0, 1.0, 0.5]),
            ("sqeuclidean", [2.0, 1.0, 0.5]),
        ):
            with self.subTest(metric=metric):
                np.testing.assert_allclose(
                    QuickAdapterRegressorV3._distance_to_reference(
                        FRONT,
                        IDEAL,
                        metric,
                        weights=UNIT_WEIGHTS,
                        p=None,
                        method="topsis",
                        apply_abs=True,
                        cdist_kwargs={},
                    ),
                    np.array(expected),
                    rtol=DISTANCE_RTOL,
                    atol=0.0,
                )

    def test_the_scipy_branch_hands_the_prepared_p_to_cdist(self):
        kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
            "minkowski", p=1.0, reference_matrix=unique_rows(4)
        )
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._distance_to_reference(
                FRONT,
                IDEAL,
                "minkowski",
                weights=UNIT_WEIGHTS,
                p=kwargs.get("p"),
                method="topsis",
                apply_abs=True,
                cdist_kwargs=kwargs,
            ),
            QuickAdapterRegressorV3._distance_to_reference(
                FRONT,
                IDEAL,
                "cityblock",
                weights=UNIT_WEIGHTS,
                p=None,
                method="topsis",
                apply_abs=True,
                cdist_kwargs={},
            ),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    def test_the_power_mean_branch_is_signed_relative_to_the_reference(self):
        # reference power mean minus row power mean: negative means the row sits above
        # the reference, which is what makes a signed argmin meaningful downstream.
        below = QuickAdapterRegressorV3._power_mean_distance(
            BELOW_IDEAL, IDEAL, "power_mean", weights=UNIT_WEIGHTS, p=1.0
        )
        above = QuickAdapterRegressorV3._power_mean_distance(
            ABOVE_IDEAL, IDEAL, "power_mean", weights=UNIT_WEIGHTS, p=1.0
        )
        at = QuickAdapterRegressorV3._power_mean_distance(
            AT_IDEAL, IDEAL, "power_mean", weights=UNIT_WEIGHTS, p=1.0
        )
        self.assertGreater(float(below[0]), 0.0)
        self.assertLess(float(above[0]), 0.0)
        self.assertAlmostEqual(float(at[0]), 0.0, places=12)

    def test_the_weighted_sum_branch_is_signed_in_the_same_direction(self):
        below = QuickAdapterRegressorV3._weighted_sum_distance(
            BELOW_IDEAL, IDEAL, weights=UNIT_WEIGHTS
        )
        above = QuickAdapterRegressorV3._weighted_sum_distance(
            ABOVE_IDEAL, IDEAL, weights=UNIT_WEIGHTS
        )
        self.assertGreater(float(below[0]), 0.0)
        self.assertLess(float(above[0]), 0.0)

    def test_apply_abs_is_the_only_thing_separating_the_signed_and_magnitude_branches(self):
        for metric, p in (("power_mean", 1.0), ("weighted_sum", None)):
            with self.subTest(metric=metric):
                stack = np.vstack([AT_IDEAL, ABOVE_IDEAL])
                signed = QuickAdapterRegressorV3._distance_to_reference(
                    stack,
                    IDEAL,
                    metric,
                    weights=UNIT_WEIGHTS,
                    p=p,
                    method="topsis",
                    apply_abs=False,
                    cdist_kwargs={},
                )
                magnitude = QuickAdapterRegressorV3._distance_to_reference(
                    stack,
                    IDEAL,
                    metric,
                    weights=UNIT_WEIGHTS,
                    p=p,
                    method="topsis",
                    apply_abs=True,
                    cdist_kwargs={},
                )
                self.assertLess(float(signed[1]), 0.0)
                np.testing.assert_allclose(magnitude, np.abs(signed), rtol=DISTANCE_RTOL, atol=0.0)

    def test_a_named_power_mean_metric_ignores_a_supplied_p(self):
        # The five named means are fixed points of the map; only the generic
        # "power_mean" metric reads the caller's p.
        for metric, power in QuickAdapterRegressorV3._POWER_MEAN_MAP.items():
            with self.subTest(metric=metric):
                # The map's value is the order the named mean stands for, and the metric
                # route computes that order itself, so a supplied p must be ignored.
                with_default = QuickAdapterRegressorV3._power_mean_distance(
                    ASYMMETRIC_ROW, IDEAL, metric, weights=UNIT_WEIGHTS
                )
                with_other_p = QuickAdapterRegressorV3._power_mean_distance(
                    ASYMMETRIC_ROW, IDEAL, metric, weights=UNIT_WEIGHTS, p=99.0
                )
                np.testing.assert_allclose(with_other_p, with_default, rtol=DISTANCE_RTOL, atol=0.0)
                # And the order the map records is the one the named metric applies.
                explicit = QuickAdapterRegressorV3._power_mean_distance(
                    ASYMMETRIC_ROW, IDEAL, "power_mean", weights=UNIT_WEIGHTS, p=power
                )
                np.testing.assert_allclose(with_default, explicit, rtol=DISTANCE_RTOL, atol=0.0)
        self.assertEqual(
            QuickAdapterRegressorV3._POWER_MEAN_MAP,
            {
                "harmonic_mean": -1.0,
                "geometric_mean": 0.0,
                "arithmetic_mean": 1.0,
                "quadratic_mean": 2.0,
                "cubic_mean": 3.0,
            },
        )
        # The generic metric does read p, so on an asymmetric row the two orders differ:
        # a row of zeros has power mean zero at every p and would not separate them.
        linear = QuickAdapterRegressorV3._power_mean_distance(
            ASYMMETRIC_ROW, IDEAL, "power_mean", weights=UNIT_WEIGHTS, p=1.0
        )
        quadratic = QuickAdapterRegressorV3._power_mean_distance(
            ASYMMETRIC_ROW, IDEAL, "power_mean", weights=UNIT_WEIGHTS, p=2.0
        )
        np.testing.assert_allclose(linear, np.array([0.5]), rtol=DISTANCE_RTOL, atol=0.0)
        np.testing.assert_allclose(
            quadratic, np.array([1.0 - np.sqrt(0.3125)]), rtol=DISTANCE_RTOL, atol=0.0
        )

    def test_a_zero_weighted_objective_does_not_contribute_to_the_power_mean(self):
        # The observable contract: an objective the user weighted to zero is excluded
        # from the aggregate rather than dragging it toward the reference. Only the
        # second objective counts, so the mean is the second column on its own.
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
        only_second = QuickAdapterRegressorV3._power_mean_distance(
            matrix, IDEAL, "power_mean", weights=np.array([0.0, 1.0]), p=1.0
        )
        only_first = QuickAdapterRegressorV3._power_mean_distance(
            matrix, IDEAL, "power_mean", weights=np.array([1.0, 0.0]), p=1.0
        )
        np.testing.assert_allclose(only_second, np.array([1.0, 0.0]), rtol=DISTANCE_RTOL, atol=0.0)
        np.testing.assert_allclose(only_first, np.array([0.0, 1.0]), rtol=DISTANCE_RTOL, atol=0.0)
        # With two objectives surviving the order still separates them, so a zero
        # weight is not what flattens the mean against the reference.
        three = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        surviving = np.array([0.0, 1.0, 1.0])
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._power_mean_distance(
                three, np.ones(3), "power_mean", weights=surviving, p=1.0
            ),
            np.array([1.0, 0.5]),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._power_mean_distance(
                three, np.ones(3), "power_mean", weights=surviving, p=2.0
            ),
            np.array([1.0, 1.0 - np.sqrt(0.5)]),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    def test_the_power_mean_metric_falls_back_to_p_one_when_p_is_rejected(self):
        for mode in (WARN, NONE):
            with self.subTest(mode=mode):
                rejected = QuickAdapterRegressorV3._power_mean_distance(
                    BELOW_IDEAL, IDEAL, "power_mean", p=np.inf, mode=mode, weights=UNIT_WEIGHTS
                )
                fallback = QuickAdapterRegressorV3._power_mean_distance(
                    BELOW_IDEAL, IDEAL, "power_mean", p=1.0, mode=mode, weights=UNIT_WEIGHTS
                )
                np.testing.assert_allclose(rejected, fallback, rtol=DISTANCE_RTOL, atol=0.0)

    def test_an_unsupported_metric_names_the_calling_method_in_its_error(self):
        for method in ("topsis", "compromise_programming"):
            with (
                self.subTest(method=method),
                self.assertRaisesRegex(
                    ValueError, rf"for {method}: supported values are euclidean, minkowski"
                ),
            ):
                QuickAdapterRegressorV3._distance_to_reference(
                    FRONT,
                    IDEAL,
                    "cosine",
                    weights=UNIT_WEIGHTS,
                    p=None,
                    method=method,
                    apply_abs=True,
                    cdist_kwargs={},
                )

    def test_the_hellinger_branch_weighs_the_root_transformed_sqrt_difference(self):
        matrix = np.array([[0.25, 0.25], [0.5, 0.5]])
        result = QuickAdapterRegressorV3._hellinger_distance(matrix, IDEAL)
        np.testing.assert_allclose(
            result,
            np.array(
                [
                    np.sqrt(2 * (0.5 - 1.0) ** 2) / np.sqrt(2.0),
                    np.sqrt(2 * (np.sqrt(0.5) - 1.0) ** 2) / np.sqrt(2.0),
                ]
            ),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    def test_the_shellinger_branch_inverts_the_variance_of_each_objective(self):
        matrix = np.array([[0.25, 0.25], [0.5, 0.5]])
        plain = QuickAdapterRegressorV3._hellinger_distance(matrix, IDEAL)
        variances = np.var(np.sqrt(matrix), axis=0, ddof=1)
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._hellinger_distance(matrix, IDEAL, standardized=True),
            plain / np.sqrt(variances),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    def test_shellinger_refuses_a_front_with_a_constant_objective(self):
        # The 1/variance weight is undefined there, so the whole metric is refused
        # rather than silently producing an infinite distance.
        matrix = np.array([[0.5, 0.25], [0.5, 0.75]])
        with self.assertRaisesRegex(ValueError, "requires non-zero variance"):
            QuickAdapterRegressorV3._hellinger_distance(matrix, IDEAL, standardized=True)

    def test_explicit_weights_scale_the_hellinger_branch(self):
        matrix = np.array([[0.25, 0.25], [0.5, 0.5]])
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._hellinger_distance(matrix, IDEAL, weights=np.full(2, 2.0)),
            np.sqrt(2.0) * QuickAdapterRegressorV3._hellinger_distance(matrix, IDEAL),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    # ------------------------------------------------------------------ TOPSIS and compromise programming

    def test_topsis_ranks_by_distance_to_the_ideal_so_the_lowest_score_wins(self):
        # The score is d_ideal / (d_ideal + d_anti_ideal): it grows with distance from
        # the ideal, and the selection caller takes the argmin. The all-ones trial sits
        # on the ideal and scores 0; the all-zeros trial sits on the anti-ideal and scores 1.
        for metric, extra in (
            ("euclidean", {}),
            ("cityblock", {}),
            ("chebyshev", {}),
            ("sqeuclidean", {}),
            ("power_mean", {"p": 1.0}),
        ):
            with self.subTest(metric=metric):
                scores = QuickAdapterRegressorV3._topsis_scores(
                    front(), metric, distance_kwargs={}, **extra
                )
                np.testing.assert_allclose(
                    scores, np.array([1.0, 0.0, 0.5]), rtol=0.0, atol=NORMALIZED_ATOL
                )
                self.assertEqual(int(np.argmin(scores)), 1)
                self.assertEqual(int(np.argmax(scores)), 0)

    def test_a_probability_metric_ranks_a_midpoint_trial_closer_to_the_anti_ideal(self):
        # Hellinger compares root-transformed distributions, so a trial at the midpoint
        # of every objective reads nearer the anti-ideal than a raw distance puts it.
        scores = QuickAdapterRegressorV3._topsis_scores(front(), "hellinger", distance_kwargs={})
        np.testing.assert_allclose(
            scores, np.array([1.0, 0.0, 0.2928932188134525]), rtol=0.0, atol=NORMALIZED_ATOL
        )

    def test_topsis_stays_inside_the_unit_interval_for_every_distance_metric(self):
        normalized = QuickAdapterRegressorV3._normalize_objective_values(
            raw_front(), [MINIMIZE, MINIMIZE]
        )
        np.testing.assert_allclose(normalized, front(), rtol=0.0, atol=NORMALIZED_ATOL)
        for metric in ("euclidean", "cityblock", "power_mean", "hellinger"):
            with self.subTest(metric=metric):
                kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
                    metric, p=1.0, reference_matrix=normalized
                )
                scores = QuickAdapterRegressorV3._topsis_scores(
                    normalized, metric, distance_kwargs=kwargs, p=1.0
                )
                self.assertTrue(
                    np.all((scores >= -NORMALIZED_ATOL) & (scores <= 1.0 + NORMALIZED_ATOL))
                )

    def test_a_zero_denominator_scores_one_half_rather_than_dividing(self):
        # A fully-zeroed weight vector makes every trial equidistant from the ideal and
        # the anti-ideal, so the ratio is 0/0. The declared answer is the neutral 0.5.
        scores = QuickAdapterRegressorV3._topsis_scores(
            front(), "weighted_sum", distance_kwargs={}, weights=np.zeros(2)
        )
        np.testing.assert_allclose(scores, np.full(3, 0.5), rtol=0.0, atol=NORMALIZED_ATOL)
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_a_negligible_denominator_is_the_same_degenerate_case_as_an_exact_zero(self):
        # np.isclose's 1e-8 draws the line: weights small enough that every trial's
        # ideal-plus-anti-ideal total falls under it score a flat 0.5, while weights
        # above it rank normally. Both sides are asserted so the branch cannot drift.
        for weight, expected in ((1e-9, [0.5, 0.5, 0.5]), (1e-7, [1.0, 0.0, 0.5])):
            with self.subTest(weight=weight):
                scores = QuickAdapterRegressorV3._topsis_scores(
                    front(), "weighted_sum", distance_kwargs={}, weights=np.full(2, weight)
                )
                np.testing.assert_allclose(
                    scores, np.array(expected), rtol=0.0, atol=NORMALIZED_ATOL
                )

    def test_the_weighted_sum_branch_needs_no_weight_validation_to_reach_that_branch(self):
        # weighted_sum and the probability metrics consume the caller's weight vector
        # verbatim, so an all-zero vector really does arrive at the distance functions.
        neutral = QuickAdapterRegressorV3._topsis_scores(
            front(), "weighted_sum", distance_kwargs={}, weights=np.zeros(2)
        )
        decisive = QuickAdapterRegressorV3._topsis_scores(
            front(), "weighted_sum", distance_kwargs={}, weights=UNIT_WEIGHTS
        )
        np.testing.assert_allclose(
            decisive, np.array([1.0, 0.0, 0.5]), rtol=0.0, atol=NORMALIZED_ATOL
        )
        np.testing.assert_allclose(neutral, np.full(3, 0.5), rtol=0.0, atol=NORMALIZED_ATOL)

    def test_a_lone_trial_is_unrankable_so_topsis_scores_it_one_half(self):
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._topsis_scores(
                np.array([[0.3, 0.4]]), "euclidean", distance_kwargs={}
            ),
            np.array([0.5]),
            rtol=0.0,
            atol=NORMALIZED_ATOL,
        )

    def test_an_empty_front_scores_nothing_for_both_aggregate_methods(self):
        empty = np.zeros((0, 2))
        np.testing.assert_array_equal(
            QuickAdapterRegressorV3._topsis_scores(empty, "euclidean", distance_kwargs={}),
            np.array([]),
        )
        np.testing.assert_array_equal(
            QuickAdapterRegressorV3._compromise_programming_scores(
                empty, "euclidean", distance_kwargs={}
            ),
            np.array([]),
        )

    def test_a_lone_trial_has_no_distance_to_the_ideal_under_compromise_programming(self):
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._compromise_programming_scores(
                np.array([[0.3, 0.4]]), "euclidean", distance_kwargs={}
            ),
            np.array([0.0]),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    def test_compromise_programming_measures_unsigned_distance_to_the_ideal(self):
        # An unsigned distance, so the argmin is the trial nearest the all-ones ideal.
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._compromise_programming_scores(
                FRONT, "euclidean", distance_kwargs={}
            ),
            np.array([np.sqrt(2.0), 1.0, np.sqrt(0.5)]),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )
        self.assertEqual(
            int(
                np.argmin(
                    QuickAdapterRegressorV3._compromise_programming_scores(
                        FRONT, "euclidean", distance_kwargs={}
                    )
                )
            ),
            2,
        )

    def test_compromise_programming_stays_signed_and_therefore_prefers_an_over_ideal_trial(self):
        # The ideal point is all-ones, so a trial exceeding it scores negative and wins
        # the argmin. A magnitude-based score would pick the other trial instead: the
        # sign is load-bearing, not cosmetic.
        stack = np.vstack([AT_IDEAL, ABOVE_IDEAL])
        signed = QuickAdapterRegressorV3._compromise_programming_scores(
            stack, "power_mean", distance_kwargs={}, p=1.0
        )
        self.assertLess(float(signed[1]), 0.0)
        self.assertEqual(int(np.argmin(signed)), 1)
        magnitude = np.abs(
            QuickAdapterRegressorV3._compromise_programming_scores(
                stack, "euclidean", distance_kwargs={}
            )
        )
        self.assertEqual(int(np.argmin(magnitude)), 0)

    def test_the_two_aggregate_methods_can_pick_different_trials_on_one_front(self):
        # Both rank with an argmin, but euclidean measures the Euclidean distance to
        # the ideal while power_mean measures the signed arithmetic mean, so the row
        # nearest in one sense is not the row nearest in the other.
        euclidean = QuickAdapterRegressorV3._compromise_programming_scores(
            FRONT, "euclidean", distance_kwargs={}
        )
        power_mean = QuickAdapterRegressorV3._compromise_programming_scores(
            FRONT, "power_mean", distance_kwargs={}, p=1.0
        )
        np.testing.assert_allclose(
            euclidean, np.array([np.sqrt(2.0), 1.0, np.sqrt(0.5)]), rtol=DISTANCE_RTOL, atol=0.0
        )
        np.testing.assert_allclose(
            power_mean, np.array([1.0, 0.5, 0.5]), rtol=DISTANCE_RTOL, atol=0.0
        )
        self.assertEqual(int(np.argmin(euclidean)), 2)
        self.assertEqual(int(np.argmin(power_mean)), 1)

    def test_topsis_uses_magnitudes_where_compromise_programming_uses_the_sign(self):
        stack = np.vstack([AT_IDEAL, ABOVE_IDEAL])
        ranking = QuickAdapterRegressorV3._topsis_scores(
            stack, "power_mean", distance_kwargs={}, p=1.0
        )
        np.testing.assert_allclose(
            ranking, np.array([0.0, 1.0 / 3.0]), rtol=0.0, atol=NORMALIZED_ATOL
        )

    def test_a_trials_distance_to_the_ideal_matches_the_aggregate_scoring_surviving_route(self):
        # _calculate_trial_distance_to_ideal is what the reported best-trial distance
        # comes from, so it must agree with the per-row aggregate distance.
        for trial_index in range(FRONT.shape[0]):
            with self.subTest(trial_index=trial_index):
                np.testing.assert_allclose(
                    QuickAdapterRegressorV3._calculate_trial_distance_to_ideal(
                        FRONT, trial_index, IDEAL_2D, "euclidean", distance_kwargs={}
                    ),
                    np.sqrt(np.sum((IDEAL - FRONT[trial_index]) ** 2)),
                    rtol=DISTANCE_RTOL,
                    atol=0.0,
                )

    def test_the_reported_distance_honours_the_prepared_p(self):
        kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
            "minkowski", p=1.0, reference_matrix=unique_rows(4)
        )
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._calculate_trial_distance_to_ideal(
                FRONT, 2, IDEAL_2D, "minkowski", distance_kwargs=kwargs
            ),
            QuickAdapterRegressorV3._calculate_trial_distance_to_ideal(
                FRONT, 2, IDEAL_2D, "cityblock", distance_kwargs={}
            ),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    # ------------------------------------------------------------------ pairwise distance sums

    def test_each_row_carries_the_sum_of_its_distances_to_every_other_row(self):
        matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        d01, d02 = 1.0, 2.0
        d12 = float(np.hypot(1.0, 2.0))
        expected = np.array([d01 + d02, d01 + d12, d02 + d12])
        for row_index, value in enumerate(expected):
            with self.subTest(row_index=row_index):
                result = QuickAdapterRegressorV3._pairwise_distance_sums(
                    matrix, "euclidean", distance_kwargs={}
                )
                self.assertAlmostEqual(float(result[row_index]), value, places=10)

    def test_a_squared_metric_sums_squares_not_distances(self):
        matrix = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        result = QuickAdapterRegressorV3._pairwise_distance_sums(
            matrix, "sqeuclidean", distance_kwargs={}
        )
        for row_index, value in enumerate([5.0, 6.0, 9.0]):
            with self.subTest(row_index=row_index):
                self.assertAlmostEqual(float(result[row_index]), value, places=10)

    def test_a_single_trial_and_an_empty_front_short_circuit_the_pairwise_sums(self):
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._pairwise_distance_sums(
                np.array([[1.0, 2.0]]), "euclidean", distance_kwargs={}
            ),
            np.array([0.0]),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            QuickAdapterRegressorV3._pairwise_distance_sums(
                np.zeros((0, 2)), "euclidean", distance_kwargs={}
            ),
            np.array([]),
        )

    def test_a_matrix_the_pairwise_route_cannot_score_is_refused(self):
        for bad, pattern in (
            (np.array([1.0, 2.0]), "must be 2-dimensional"),
            (np.zeros((3, 0)), "at least one feature"),
            (np.array([[1.0], [np.nan]]), "only finite values"),
            (np.array([[1.0], [np.inf]]), "only finite values"),
        ):
            with self.subTest(bad=bad.tolist()), self.assertRaisesRegex(ValueError, pattern):
                QuickAdapterRegressorV3._pairwise_distance_sums(
                    bad, "euclidean", distance_kwargs={}
                )

    # ------------------------------------------------------------------ knn density selection

    def test_every_trial_is_scored_by_its_own_aggregated_neighbour_distance(self):
        # Points at 0, 1 and 4 on a line: the two nearest neighbours of each are at
        # 1&4, 1&3 and 3&4 respectively.
        line = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        expected = {
            "power_mean": [2.5, 2.0, 3.5],
            "quantile": [2.5, 2.0, 3.5],
            "min": [1.0, 1.0, 3.0],
            "max": [4.0, 3.0, 4.0],
        }
        for aggregation, values in expected.items():
            with self.subTest(aggregation=aggregation):
                result = QuickAdapterRegressorV3._knn_based_selection(
                    line,
                    aggregation,
                    distance_kwargs={},
                    distance_metric="euclidean",
                    n_neighbors=2,
                )
                np.testing.assert_allclose(result, np.array(values), rtol=DISTANCE_RTOL, atol=0.0)

    def test_a_single_neighbour_collapses_every_aggregation_to_the_nearest_distance(self):
        line = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        for aggregation in QuickAdapterRegressorV3._DENSITY_AGGREGATIONS:
            with self.subTest(aggregation=aggregation):
                np.testing.assert_allclose(
                    QuickAdapterRegressorV3._knn_based_selection(
                        line,
                        aggregation,
                        distance_kwargs={},
                        distance_metric="euclidean",
                        n_neighbors=1,
                    ),
                    np.array([1.0, 1.0, 3.0]),
                    rtol=DISTANCE_RTOL,
                    atol=0.0,
                )

    def test_the_aggregation_param_moves_the_aggregate_when_it_is_supplied(self):
        line = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        squared = QuickAdapterRegressorV3._knn_based_selection(
            line,
            "power_mean",
            distance_kwargs={},
            distance_metric="euclidean",
            n_neighbors=2,
            aggregation_param=2.0,
        )
        np.testing.assert_allclose(
            squared, np.sqrt(np.array([8.5, 5.0, 12.5])), rtol=DISTANCE_RTOL, atol=0.0
        )
        first_quartile = QuickAdapterRegressorV3._knn_based_selection(
            line,
            "quantile",
            distance_kwargs={},
            distance_metric="euclidean",
            n_neighbors=2,
            aggregation_param=0.25,
        )
        np.testing.assert_allclose(
            first_quartile, np.array([1.75, 1.5, 3.25]), rtol=DISTANCE_RTOL, atol=0.0
        )

    def test_an_absent_aggregation_param_falls_back_to_the_aggregation_default(self):
        line = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        for aggregation, param in (("power_mean", 1.0), ("quantile", 0.5)):
            with self.subTest(aggregation=aggregation):
                implicit = QuickAdapterRegressorV3._knn_based_selection(
                    line,
                    aggregation,
                    distance_kwargs={},
                    distance_metric="euclidean",
                    n_neighbors=2,
                    aggregation_param=None,
                )
                explicit = QuickAdapterRegressorV3._knn_based_selection(
                    line,
                    aggregation,
                    distance_kwargs={},
                    distance_metric="euclidean",
                    n_neighbors=2,
                    aggregation_param=param,
                )
                np.testing.assert_allclose(implicit, explicit, rtol=DISTANCE_RTOL, atol=0.0)

    def test_a_neighbourhood_larger_than_the_front_is_clamped_to_the_front(self):
        line = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._knn_based_selection(
                line, "min", distance_kwargs={}, distance_metric="euclidean", n_neighbors=99
            ),
            QuickAdapterRegressorV3._knn_based_selection(
                line, "min", distance_kwargs={}, distance_metric="euclidean", n_neighbors=2
            ),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    def test_an_empty_neighbourhood_makes_every_trial_unselectable(self):
        line = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        for n_neighbors in (0, -1):
            with self.subTest(n_neighbors=n_neighbors):
                result = QuickAdapterRegressorV3._knn_based_selection(
                    line,
                    "min",
                    distance_kwargs={},
                    distance_metric="euclidean",
                    n_neighbors=n_neighbors,
                )
                self.assertTrue(np.all(np.isinf(result)))

    def test_a_front_smaller_than_two_is_short_circuited(self):
        np.testing.assert_array_equal(
            QuickAdapterRegressorV3._knn_based_selection(
                np.zeros((0, 2)),
                "min",
                distance_kwargs={},
                distance_metric="euclidean",
                n_neighbors=3,
            ),
            np.array([]),
        )
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._knn_based_selection(
                np.ones((1, 2)),
                "min",
                distance_kwargs={},
                distance_metric="euclidean",
                n_neighbors=3,
            ),
            np.array([0.0]),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    def test_the_knn_route_raises_on_an_unknown_aggregation(self):
        line = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        aggregations = QuickAdapterRegressorV3._DENSITY_AGGREGATIONS
        with self.assertRaises(ValueError) as caught:
            QuickAdapterRegressorV3._knn_based_selection(
                line, "median", distance_kwargs={}, distance_metric="euclidean", n_neighbors=1
            )
        self.assertEqual(
            str(caught.exception), enum_error_message("aggregation", "median", aggregations)
        )

    def test_the_knn_route_validates_its_aggregation_param_and_raises_rather_than_warns(self):
        # The knn route hard-codes the raise mode, so a bad aggregation param aborts
        # the selection instead of being quietly downgraded to a different aggregate.
        line = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
        for aggregation, param, constraint in (
            ("quantile", 1.5, r"must be in \[0, 1\]"),
            ("quantile", -0.1, r"must be in \[0, 1\]"),
            ("power_mean", np.inf, "must be finite"),
        ):
            with self.subTest(aggregation=aggregation, param=param):
                with self.assertRaisesRegex(ValueError, constraint) as caught:
                    QuickAdapterRegressorV3._knn_based_selection(
                        line,
                        aggregation,
                        distance_kwargs={},
                        distance_metric="euclidean",
                        n_neighbors=2,
                        aggregation_param=param,
                    )
                self.assertIn("label_density_aggregation_param", str(caught.exception))

    # ------------------------------------------------------------------ trial selection

    def test_the_lowest_trial_number_wins_independently_of_input_order(self):
        for order in ((6, 1, 3), (3, 6, 1), (1, 3, 6)):
            with self.subTest(order=order):
                self.assertEqual(
                    QuickAdapterRegressorV3._select_lowest_number_trial(trials(*order)).number, 1
                )

    def test_the_closest_trial_to_the_ideal_wins(self):
        self.assertEqual(
            QuickAdapterRegressorV3._select_best_trial_by_distance(
                trials(5, 2, 8), np.array([0.5, 0.1, 0.4])
            ).number,
            2,
        )

    def test_a_distance_tie_is_broken_by_the_lowest_trial_number(self):
        self.assertEqual(
            QuickAdapterRegressorV3._select_best_trial_by_distance(
                trials(9, 4, 7), np.array([2.0, 1.0, 1.0])
            ).number,
            4,
        )
        self.assertEqual(
            QuickAdapterRegressorV3._select_best_trial_by_distance(
                trials(5, 2, 8), np.array([1.0, 1.0, 2.0])
            ).number,
            2,
        )

    def test_a_non_finite_distance_removes_its_trial_from_the_contest(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(distance=bad):
                self.assertEqual(
                    QuickAdapterRegressorV3._select_best_trial_by_distance(
                        trials(6, 1, 3), np.array([bad, np.inf, 4.0])
                    ).number,
                    3,
                )

    def test_a_fully_non_finite_front_falls_back_to_the_lowest_trial_number(self):
        for distances in (np.full(3, np.nan), np.full(3, np.inf), np.full(3, -np.inf)):
            with self.subTest(distances=distances.tolist()):
                with self.assertLogs(LOGGER_NAME, "WARNING") as captured:
                    chosen = QuickAdapterRegressorV3._select_best_trial_by_distance(
                        trials(6, 1, 3), distances
                    )
                self.assertEqual(chosen.number, 1)
                self.assertIn("falling back to lowest trial number", captured.output[0])

    def test_a_distance_vector_that_does_not_cover_the_trials_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must match trials length"):
            QuickAdapterRegressorV3._select_best_trial_by_distance(
                trials(1, 2, 3), np.array([1.0, 2.0])
            )

    # ------------------------------------------------------------------ cluster count

    def test_a_front_that_cannot_support_two_clusters_asks_for_one(self):
        for matrix in (np.array([[1.0, 2.0]]), np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])):
            with self.subTest(shape=matrix.shape):
                self.assertEqual(QuickAdapterRegressorV3._get_n_clusters(matrix), 1)

    def test_a_small_front_asks_for_one_cluster_per_distinct_row(self):
        for n_rows in (2, 3):
            with self.subTest(n_rows=n_rows):
                self.assertEqual(
                    QuickAdapterRegressorV3._get_n_clusters(unique_rows(n_rows)), n_rows
                )

    def test_a_larger_front_follows_the_unique_row_growth_rule_within_its_bounds(self):
        expected = {4: 2, 8: 3, 12: 4, 20: 4, 50: 6, 100: 8, 200: 10, 1000: 10}
        for n_rows, n_clusters in expected.items():
            with self.subTest(n_rows=n_rows):
                self.assertEqual(
                    QuickAdapterRegressorV3._get_n_clusters(unique_rows(n_rows)), n_clusters
                )

    def test_the_cluster_count_counts_distinct_rows_not_rows(self):
        # A front of 10 rows drawn from 4 distinct points has 4 distinct points, and the
        # count grows with the distinct-row count rather than the row count. 10 distinct
        # rows would ask for 3, so the two are separable here.
        self.assertEqual(QuickAdapterRegressorV3._get_n_clusters(repeated_rows(10, 4)), 2)
        self.assertEqual(QuickAdapterRegressorV3._get_n_clusters(repeated_rows(10, 10)), 3)
        self.assertEqual(QuickAdapterRegressorV3._get_n_clusters(repeated_rows(50, 4)), 2)
        self.assertEqual(QuickAdapterRegressorV3._get_n_clusters(repeated_rows(50, 50)), 6)

    def test_a_duplicated_front_of_one_point_never_reaches_the_growth_rule(self):
        for n_rows in (2, 5, 50):
            with self.subTest(n_rows=n_rows):
                self.assertEqual(
                    QuickAdapterRegressorV3._get_n_clusters(repeated_rows(n_rows, 1)), 1
                )

    def test_the_cluster_count_never_leaves_the_configured_window(self):
        matrix = unique_rows(4)
        self.assertEqual(QuickAdapterRegressorV3._get_n_clusters(matrix, min_n_clusters=3), 3)
        self.assertEqual(
            QuickAdapterRegressorV3._get_n_clusters(unique_rows(100), max_n_clusters=2), 2
        )
        self.assertEqual(
            QuickAdapterRegressorV3._get_n_clusters(unique_rows(1000), max_n_clusters=4), 4
        )

    # ------------------------------------------------------------------ extrema thresholds

    def test_an_empty_extremum_set_yields_no_candidates(self):
        # size 0 is the one case that may legitimately keep nothing, and keep_fraction
        # does not force a candidate out of a set that does not exist.
        self.assertEqual(QuickAdapterRegressorV3._calculate_n_kept_extrema(0, 1.0), 0)
        self.assertEqual(QuickAdapterRegressorV3._calculate_n_kept_extrema(0, 0.0075), 0)

    def test_a_non_empty_extremum_set_always_keeps_at_least_one(self):
        for keep_fraction in (0.0, 0.0075, 0.5, 1.0):
            with self.subTest(keep_fraction=keep_fraction):
                self.assertGreaterEqual(
                    QuickAdapterRegressorV3._calculate_n_kept_extrema(10, keep_fraction), 1
                )

    def test_the_kept_count_uses_banker_rounding_not_ceiling(self):
        # round() is round-half-to-even: 133 * 0.5 == 66.5 keeps 66, not 67, and a
        # keep_fraction of 0.0075 over 133 extrema still keeps only the floor of one.
        self.assertEqual(QuickAdapterRegressorV3._calculate_n_kept_extrema(133, 0.5), 66)
        self.assertEqual(QuickAdapterRegressorV3._calculate_n_kept_extrema(3, 0.5), 2)
        self.assertEqual(QuickAdapterRegressorV3._calculate_n_kept_extrema(133, 0.0075), 1)
        self.assertEqual(QuickAdapterRegressorV3._calculate_n_kept_extrema(10, 0.5), 5)
        self.assertEqual(QuickAdapterRegressorV3._calculate_n_kept_extrema(10, 1.0), 10)

    def test_partition_separates_the_two_sides_of_the_normalised_range(self):
        # ±10 * eps, not 0: a label that rounds to exactly zero is neither extremum.
        series = pd.Series([0.1, -0.5, 0.9, -0.2, 0.3, 0.05, -0.95, 0.7])
        minima, maxima = QuickAdapterRegressorV3.get_pred_min_max(series, "partition")
        self.assertEqual(minima.tolist(), [-0.5, -0.2, -0.95])
        self.assertEqual(maxima.tolist(), [0.1, 0.9, 0.3, 0.05, 0.7])

    def test_a_label_inside_the_partition_epsilon_is_neither_minimum_nor_maximum(self):
        eps = 10 * np.finfo(float).eps
        series = pd.Series([eps / 2, -eps / 2, 1.0, -1.0])
        minima, maxima = QuickAdapterRegressorV3.get_pred_min_max(series, "partition")
        self.assertEqual(minima.tolist(), [-1.0])
        self.assertEqual(maxima.tolist(), [1.0])

    def test_ranking_keeps_the_most_extreme_peaks_first(self):
        series = pd.Series([0.1, -0.5, 0.9, -0.2, 0.3, 0.05, -0.95, 0.7])
        for selection_method in ("rank_extrema", "rank_peaks"):
            with self.subTest(selection_method=selection_method):
                minima, maxima = QuickAdapterRegressorV3.get_pred_min_max(series, selection_method)
                self.assertEqual(minima.tolist(), [-0.95, -0.5, -0.2])
                self.assertEqual(maxima.iloc[0], 0.9)

    def test_the_keep_fraction_shrinks_the_candidate_pool_to_the_extremes(self):
        wave = pd.Series([0.0, 1.0, 0.0, -1.0, 0.0, 2.0, 0.0, -2.0, 0.0])
        for selection_method in ("rank_extrema", "rank_peaks"):
            with self.subTest(selection_method=selection_method):
                full = QuickAdapterRegressorV3.get_pred_min_max(wave, selection_method, 1.0)
                half = QuickAdapterRegressorV3.get_pred_min_max(wave, selection_method, 0.5)
                self.assertEqual(full[0].tolist(), [-2.0, -1.0])
                self.assertEqual(half[0].tolist(), [-2.0])
                self.assertEqual(full[1].tolist(), [2.0, 1.0])
                self.assertEqual(half[1].tolist(), [2.0])

    def test_a_monotone_series_has_no_interior_extrema_at_all(self):
        rising = pd.Series([1.0, 2.0, 3.0])
        for selection_method in ("rank_extrema", "rank_peaks"):
            with self.subTest(selection_method=selection_method):
                minima, maxima = QuickAdapterRegressorV3.get_pred_min_max(rising, selection_method)
                self.assertTrue(minima.empty)
                self.assertTrue(maxima.empty)

    def test_a_non_finite_label_cannot_become_a_candidate(self):
        series = pd.Series([0.1, np.inf, -np.inf, np.nan, 0.9, -0.5])
        minima, maxima = QuickAdapterRegressorV3.get_pred_min_max(series, "partition")
        self.assertEqual(minima.tolist(), [-0.5])
        self.assertEqual(maxima.tolist(), [0.1, 0.9])

    def test_a_fully_non_finite_series_yields_no_candidates_and_the_sentinel_bounds(self):
        minima, maxima = QuickAdapterRegressorV3.get_pred_min_max(
            pd.Series([np.nan, np.inf, -np.inf]), "partition"
        )
        self.assertTrue(minima.empty)
        self.assertTrue(maxima.empty)
        self.assertEqual(
            QuickAdapterRegressorV3.median_min_max(pd.Series(dtype=float), "partition"), (-2.0, 2.0)
        )

    def test_a_non_numeric_label_is_coerced_to_nan_and_dropped(self):
        minima, maxima = QuickAdapterRegressorV3.get_pred_min_max(
            pd.Series(["0.5", "bad", "-0.5"]), "partition"
        )
        self.assertEqual(minima.tolist(), [-0.5])
        self.assertEqual(maxima.tolist(), [0.5])

    def test_partition_preserves_the_callers_index(self):
        series = pd.Series([0.1, 0.2, 0.3], index=[10, 11, 12])
        self.assertEqual(
            QuickAdapterRegressorV3.get_pred_min_max(series, "partition")[1].index.tolist(),
            [10, 11, 12],
        )

    def test_an_unknown_selection_method_is_named_in_the_canonical_enum_error(self):
        with self.assertRaises(ValueError) as caught:
            QuickAdapterRegressorV3.get_pred_min_max(pd.Series([1.0, 2.0]), "extremes")
        self.assertEqual(
            str(caught.exception),
            enum_error_message("selection_method", "extremes", EXTREMA_SELECTION_METHODS),
        )

    # ------------------------------------------------------------------ sentinel bounds

    def test_the_fallback_bounds_sit_outside_the_normalised_label_range(self):
        low, high = QuickAdapterRegressorV3.RANGE_DEFAULT
        self.assertEqual(QuickAdapterRegressorV3.safe_min_pred(pd.Series(dtype=float)), -2.0)
        self.assertEqual(QuickAdapterRegressorV3.safe_max_pred(pd.Series(dtype=float)), 2.0)
        self.assertLess(-2.0, low)
        self.assertGreater(2.0, high)

    def test_an_empty_or_non_finite_prediction_takes_the_sentinel(self):
        for series in (
            pd.Series(dtype=float),
            pd.Series([np.nan, np.nan]),
            pd.Series([np.nan, np.inf]),
        ):
            with self.subTest(series=series.tolist()):
                self.assertEqual(QuickAdapterRegressorV3.safe_min_pred(series), -2.0)
                self.assertEqual(QuickAdapterRegressorV3.safe_max_pred(series), 2.0)

    def test_a_finite_prediction_never_takes_the_sentinel(self):
        series = pd.Series([-0.5, 0.1, 0.2])
        self.assertAlmostEqual(QuickAdapterRegressorV3.safe_min_pred(series), -0.5, places=12)
        self.assertAlmostEqual(QuickAdapterRegressorV3.safe_max_pred(series), 0.2, places=12)

    def test_a_reducer_that_raises_or_returns_nothing_usable_takes_the_sentinel(self):
        def raising(_series: pd.Series) -> float:
            raise RuntimeError("no rows")

        series = pd.Series([1.0, 2.0])
        for reducer in (
            raising,
            lambda s: s,
            lambda s: np.nan,
            lambda s: np.inf,
            lambda s: "x",
            lambda s: np.array([1.0]),
        ):
            with self.subTest(reducer=reducer):
                self.assertEqual(QuickAdapterRegressorV3._safe_pred(series, reducer, 9.0), 9.0)
        self.assertEqual(QuickAdapterRegressorV3._safe_pred(series, lambda s: 0.5, 9.0), 0.5)

    def test_only_the_non_finite_side_is_replaced_by_the_sentinel(self):
        series = pd.Series([-1.0, 1.0])
        self.assertEqual(
            QuickAdapterRegressorV3._resolve_min_max(np.nan, np.nan, series), (-1.0, 1.0)
        )
        self.assertEqual(
            QuickAdapterRegressorV3._resolve_min_max(0.25, np.nan, series), (0.25, 1.0)
        )
        self.assertEqual(
            QuickAdapterRegressorV3._resolve_min_max(np.nan, 0.75, series), (-1.0, 0.75)
        )
        self.assertEqual(QuickAdapterRegressorV3._resolve_min_max(0.25, 0.75, series), (0.25, 0.75))

    def test_a_one_sided_extremum_set_leaves_the_other_side_on_the_sentinel(self):
        positive = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        negative = pd.Series([-0.1, -0.2, -0.3, -0.4, -0.5])
        # partition keeps only one side of zero, so the absent side has no candidate and
        # falls back to the full-series extremum rather than to the sentinel.
        minimum, maximum = QuickAdapterRegressorV3.median_min_max(positive, "partition")
        self.assertAlmostEqual(float(minimum), 0.1, places=9)
        self.assertAlmostEqual(float(maximum), 0.3, places=9)
        minimum, maximum = QuickAdapterRegressorV3.median_min_max(negative, "partition")
        self.assertAlmostEqual(float(minimum), -0.3, places=9)
        self.assertAlmostEqual(float(maximum), -0.1, places=9)

    # ------------------------------------------------------------------ threshold surfaces

    def test_the_median_surface_medianises_the_candidate_sets(self):
        series = pd.Series([0.1, -0.5, 0.9, -0.2, 0.3, 0.05, -0.95, 0.7])
        minimum, maximum = QuickAdapterRegressorV3.median_min_max(series, "rank_extrema")
        self.assertAlmostEqual(float(minimum), -0.5, places=9)
        self.assertAlmostEqual(float(maximum), 0.8, places=9)

    def test_the_soft_extremum_surface_hardens_itself_as_alpha_grows(self):
        series = pd.Series([0.1, -0.5, 0.9, -0.2, 0.3, 0.05, -0.95, 0.7])
        soft = QuickAdapterRegressorV3.soft_extremum_min_max(series, 12.0, "rank_extrema")
        hard = QuickAdapterRegressorV3.get_pred_min_max(series, "rank_extrema")
        self.assertAlmostEqual(float(soft[0]), -0.9478847957509023, places=9)
        self.assertAlmostEqual(float(soft[1]), 0.8833654607012157, places=9)
        self.assertGreater(float(soft[0]), float(hard[0].min()))
        self.assertLess(float(soft[1]), float(hard[1].max()))

    def test_a_zero_alpha_reduces_the_soft_extremum_to_the_plain_mean(self):
        series = pd.Series([0.1, -0.5, 0.9, -0.2, 0.3, 0.05, -0.95, 0.7])
        minimum, maximum = QuickAdapterRegressorV3.soft_extremum_min_max(
            series, 0.0, "rank_extrema"
        )
        minima, maxima = QuickAdapterRegressorV3.get_pred_min_max(series, "rank_extrema")
        self.assertAlmostEqual(float(minimum), float(minima.mean()), places=9)
        self.assertAlmostEqual(float(maximum), float(maxima.mean()), places=9)

    def test_a_negative_alpha_is_refused_before_any_extremum_is_computed(self):
        with self.assertRaisesRegex(ValueError, "must be >= 0"):
            QuickAdapterRegressorV3.soft_extremum_min_max(
                pd.Series([1.0, 2.0]), -0.1, "rank_extrema"
            )

    def test_the_skimage_surface_delegates_to_the_requested_threshold(self):
        series = pd.Series([0.1, -0.5, 0.9, -0.2, 0.3, 0.05, -0.95, 0.7])
        for method in SKIMAGE_THRESHOLD_METHODS:
            with self.subTest(method=method):
                minimum, maximum = QuickAdapterRegressorV3.skimage_min_max(
                    series, method, "rank_extrema"
                )
                self.assertTrue(-1.0 <= float(minimum) <= float(maximum) <= 1.0)
        np.testing.assert_allclose(
            QuickAdapterRegressorV3.skimage_min_max(series, "otsu", "rank_extrema"),
            (-0.94853515625, 0.8),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    def test_an_unknown_skimage_threshold_is_named_in_the_canonical_enum_error(self):
        with self.assertRaises(ValueError) as caught:
            QuickAdapterRegressorV3.skimage_min_max(
                pd.Series([0.1, 0.2, 0.3]), "entropy", "rank_extrema"
            )
        self.assertEqual(
            str(caught.exception),
            enum_error_message("skimage threshold method", "entropy", SKIMAGE_THRESHOLD_METHODS),
        )

    def test_an_empty_extremum_set_returns_nan_so_the_caller_falls_back(self):
        # The threshold helper signals "cannot be computed" with NaN, and _resolve_min_max
        # turns that into the ±2.0 sentinel, so the reported bounds stay finite.
        self.assertTrue(
            np.isnan(
                QuickAdapterRegressorV3.apply_skimage_threshold(
                    pd.Series(dtype=float), lambda v: 0.0
                )
            )
        )
        for surface in (
            QuickAdapterRegressorV3.median_min_max,
            QuickAdapterRegressorV3.skimage_min_max,
        ):
            with self.subTest(surface=surface.__name__):
                minimum, maximum = (
                    surface(pd.Series(dtype=float), "rank_extrema")
                    if surface is QuickAdapterRegressorV3.median_min_max
                    else surface(pd.Series(dtype=float), "otsu", "rank_extrema")
                )
                self.assertEqual((minimum, maximum), (-2.0, 2.0))

    def test_a_thresholdable_extremum_set_never_returns_nan(self):
        series = pd.Series([0.1, -0.5, 0.9, -0.2, 0.3, 0.05, -0.95, 0.7])
        for method in SKIMAGE_THRESHOLD_METHODS:
            with self.subTest(method=method):
                for surface in (
                    QuickAdapterRegressorV3.median_min_max,
                    QuickAdapterRegressorV3.skimage_min_max,
                ):
                    result = (
                        surface(series, "rank_extrema")
                        if surface is QuickAdapterRegressorV3.median_min_max
                        else surface(series, method, "rank_extrema")
                    )
                    self.assertFalse(np.isnan(result[0]))
                    self.assertFalse(np.isnan(result[1]))

    def test_a_single_element_or_two_distinct_candidate_set_falls_back_to_its_median(self):
        calls, spy = threshold_spy()
        self.assertAlmostEqual(
            QuickAdapterRegressorV3.apply_skimage_threshold(pd.Series([4.0]), spy), 4.0, places=12
        )
        self.assertAlmostEqual(
            QuickAdapterRegressorV3.apply_skimage_threshold(pd.Series([1.0, 4.0]), spy),
            2.5,
            places=12,
        )
        self.assertEqual(calls, [])

    def test_a_candidate_set_with_fewer_than_three_distinct_values_cannot_be_thresholded(self):
        # Otsu and friends need a histogram with interior bins, so a two-valued
        # candidate set is medianised instead of being forced through the function.
        calls, spy = threshold_spy()
        result = QuickAdapterRegressorV3.apply_skimage_threshold(pd.Series([1.0, 4.0]), spy)
        self.assertEqual(calls, [])
        self.assertAlmostEqual(result, 2.5, places=12)

    def test_a_numerically_flat_candidate_set_is_medianised(self):
        calls, spy = threshold_spy()
        result = QuickAdapterRegressorV3.apply_skimage_threshold(pd.Series([1.0, 1.0, 1.0]), spy)
        self.assertEqual(calls, [])
        self.assertAlmostEqual(result, 1.0, places=12)

    def test_a_healthy_candidate_set_is_handed_to_the_threshold_verbatim(self):
        calls, spy = threshold_spy()
        result = QuickAdapterRegressorV3.apply_skimage_threshold(pd.Series([1.0, 2.0, 5.0]), spy)
        self.assertEqual(result, 99.0)
        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(calls[0], np.array([1.0, 2.0, 5.0]), rtol=0.0, atol=0.0)

    def test_a_threshold_that_raises_is_reported_and_medianised(self):
        def exploding(_values: np.ndarray) -> float:
            raise RuntimeError("no interior bins")

        series = pd.Series([1.0, 2.0, 5.0])
        with self.assertLogs(LOGGER_NAME, "WARNING") as captured:
            result = QuickAdapterRegressorV3.apply_skimage_threshold(series, exploding)
        self.assertAlmostEqual(float(result), 2.0, places=12)
        self.assertIn("falling back to median", captured.output[0])

    def test_an_all_nan_candidate_set_has_no_median_either(self):
        calls, spy = threshold_spy()
        self.assertTrue(
            np.isnan(
                QuickAdapterRegressorV3.apply_skimage_threshold(
                    pd.Series([np.nan, np.nan, np.nan]), spy
                )
            )
        )
        self.assertEqual(calls, [])

    def test_an_integer_candidate_series_reaches_the_threshold_unconverted(self):
        # to_numpy() preserves the caller's dtype: an int64 label reaches the
        # threshold function as int64, which is what skimage's histogram wants.
        calls, spy = threshold_spy()
        self.assertEqual(
            QuickAdapterRegressorV3.apply_skimage_threshold(pd.Series([1, 2, 9]), spy), 99.0
        )
        self.assertEqual(calls[0].dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(calls[0], np.array([1, 2, 9]))

    # ------------------------------------------------------------------ scalar validators

    def test_a_non_finite_scalar_is_refused_in_warn_and_raise_modes_and_silently_dropped_in_none(
        self,
    ):
        for value in (np.inf, -np.inf, np.nan):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be finite"):
                QuickAdapterRegressorV3._validate_scalar(value, ctx="label_p_order", mode=RAISE)
                with self.assertLogs(LOGGER_NAME, "WARNING") as captured:
                    warned = QuickAdapterRegressorV3._validate_scalar(
                        value, ctx="label_p_order", mode=WARN
                    )
                self.assertIsNone(warned)
                self.assertIn("must be finite", captured.output[0])
                with self.assertNoLogs(LOGGER_NAME, "WARNING"):
                    self.assertIsNone(
                        QuickAdapterRegressorV3._validate_scalar(
                            value, ctx="label_p_order", mode=NONE
                        )
                    )

    def test_a_predicate_violation_is_reported_with_the_supplied_constraint(self):
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            QuickAdapterRegressorV3._validate_scalar(
                -1.0,
                ctx="label_p_order",
                mode=RAISE,
                predicate=lambda v: v > 0,
                constraint="must be > 0",
            )
        with self.assertLogs(LOGGER_NAME, "WARNING") as captured:
            self.assertIsNone(
                QuickAdapterRegressorV3._validate_scalar(
                    -1.0,
                    ctx="label_p_order",
                    mode=WARN,
                    predicate=lambda v: v > 0,
                    constraint="must be > 0",
                )
            )
        self.assertIn("must be > 0", captured.output[0])

    def test_a_valid_scalar_is_returned_as_a_float_in_every_mode(self):
        for mode in (RAISE, WARN, NONE):
            with self.subTest(mode=mode), self.assertNoLogs(LOGGER_NAME, "WARNING"):
                result = QuickAdapterRegressorV3._validate_scalar(3, ctx="label_p_order", mode=mode)
                self.assertIsInstance(result, float)
                self.assertAlmostEqual(result, 3.0, places=12)

    def test_a_missing_scalar_is_never_validated(self):
        for mode in (RAISE, WARN, NONE):
            with self.subTest(mode=mode), self.assertNoLogs(LOGGER_NAME, "WARNING"):
                self.assertIsNone(
                    QuickAdapterRegressorV3._validate_scalar(None, ctx="x", mode=mode)
                )

    def test_the_minkowski_p_must_be_strictly_positive(self):
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            QuickAdapterRegressorV3._validate_minkowski_p(0.0, ctx="label_p_order")
        for valid in (1e-9, 1.0, 2.0, np.inf - 0.0):
            with self.subTest(valid=valid):
                if not np.isfinite(valid):
                    continue
                self.assertAlmostEqual(
                    QuickAdapterRegressorV3._validate_minkowski_p(valid, ctx="label_p_order"),
                    valid,
                    places=12,
                )
        self.assertIsNone(QuickAdapterRegressorV3._validate_minkowski_p(None, ctx="label_p_order"))

    def test_the_quantile_is_closed_at_both_ends(self):
        for valid in (0.0, 0.5, 1.0):
            with self.subTest(valid=valid):
                self.assertAlmostEqual(
                    QuickAdapterRegressorV3._validate_quantile_q(valid, ctx="label_quantile"),
                    valid,
                    places=12,
                )
        for invalid in (-1e-9, 1.0 + 1e-9):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, r"must be in \[0, 1\]"),
            ):
                QuickAdapterRegressorV3._validate_quantile_q(invalid, ctx="label_quantile")

    def test_the_power_mean_p_is_only_required_to_be_finite(self):
        # p=0 is the geometric mean and p<0 a harmonic-style mean, so unlike the
        # Minkowski order there is no sign constraint here.
        for valid in (-2.0, -1.0, 0.0, 1.0, 99.0):
            with self.subTest(valid=valid):
                self.assertAlmostEqual(
                    QuickAdapterRegressorV3._validate_power_mean_p(valid, ctx="label_p_order"),
                    valid,
                    places=12,
                )
        for invalid in (np.inf, np.nan):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "must be finite"),
            ):
                QuickAdapterRegressorV3._validate_power_mean_p(invalid, ctx="label_p_order")

    def test_every_scalar_validator_keeps_its_own_declared_default_mode(self):
        # The defaults are the user-facing contract: raise for a scalar the user typed,
        # but _validate_metric_weights_support and _validate_label_selection_metric
        # default to warn because they guard a whole config block.
        for validator in (
            QuickAdapterRegressorV3._validate_minkowski_p,
            QuickAdapterRegressorV3._validate_quantile_q,
            QuickAdapterRegressorV3._validate_power_mean_p,
        ):
            with self.subTest(validator=validator.__name__), self.assertRaises(ValueError):
                validator(np.inf, ctx="ctx")
        with self.assertLogs(LOGGER_NAME, "WARNING"):
            self.assertIsNone(
                QuickAdapterRegressorV3._validate_metric_weights_support(
                    "mahalanobis", ctx="label_distance_metric"
                )
            )
        with self.assertLogs(LOGGER_NAME, "WARNING"):
            self.assertEqual(
                QuickAdapterRegressorV3._validate_label_selection_metric(
                    "hellinger",
                    ctx="label_distance_metric",
                    default="euclidean",
                    aggregate_allowed=False,
                ),
                "euclidean",
            )

    def test_the_p_order_default_only_applies_when_the_user_supplied_nothing(self):
        self.assertAlmostEqual(
            QuickAdapterRegressorV3._resolve_p_order("minkowski", None, ctx="label_p_order"),
            2.0,
            places=12,
        )
        self.assertAlmostEqual(
            QuickAdapterRegressorV3._resolve_p_order("power_mean", None, ctx="label_p_order"),
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            QuickAdapterRegressorV3._resolve_p_order("minkowski", 3.0, ctx="label_p_order"),
            3.0,
            places=12,
        )

    def test_a_metric_without_a_p_order_default_yields_none(self):
        for metric in ("euclidean", "cityblock", "hellinger", "weighted_sum"):
            with self.subTest(metric=metric):
                self.assertIsNone(QuickAdapterRegressorV3._resolve_p_order(metric, None, ctx="p"))

    def test_only_the_minkowski_p_order_is_validated_on_the_resolved_value(self):
        # Every other metric's p is either unused or a power mean, and this source
        # deliberately does not constrain it, so the value passes straight through.
        self.assertAlmostEqual(
            QuickAdapterRegressorV3._resolve_p_order("power_mean", -5.0, ctx="label_p_order"),
            -5.0,
            places=12,
        )
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            QuickAdapterRegressorV3._resolve_p_order("minkowski", -1.0, ctx="label_p_order")
        for mode in (WARN, NONE):
            with self.subTest(mode=mode):
                self.assertIsNone(
                    QuickAdapterRegressorV3._resolve_p_order(
                        "minkowski", -1.0, ctx="label_p_order", mode=mode
                    )
                )

    def test_the_metric_weights_support_gate_names_the_metrics_that_reject_weights(self):
        for metric in QuickAdapterRegressorV3._UNSUPPORTED_WEIGHTS_METRICS:
            with self.subTest(metric=metric):
                self.assertEqual(
                    QuickAdapterRegressorV3._validate_metric_weights_support(
                        metric, ctx="label_distance_metric", mode=NONE
                    ),
                    None,
                )
                with self.assertRaisesRegex(ValueError, "does not support custom weights"):
                    QuickAdapterRegressorV3._validate_metric_weights_support(
                        metric, ctx="label_distance_metric", mode=RAISE
                    )
                with self.assertLogs(LOGGER_NAME, "WARNING") as captured:
                    self.assertIsNone(
                        QuickAdapterRegressorV3._validate_metric_weights_support(
                            metric, ctx="label_distance_metric", mode=WARN
                        )
                    )
                self.assertIn("using uniform weights", captured.output[0])

    def test_a_metric_that_accepts_weights_is_returned_unchanged(self):
        for metric in ("euclidean", "minkowski", "power_mean", "weighted_sum", "hellinger"):
            with self.subTest(metric=metric), self.assertNoLogs(LOGGER_NAME, "WARNING"):
                self.assertEqual(
                    QuickAdapterRegressorV3._validate_metric_weights_support(
                        metric, ctx="label_distance_metric", mode=RAISE
                    ),
                    metric,
                )

    def test_a_missing_weight_vector_becomes_a_uniform_one(self):
        for n_objectives in (1, 4, 7):
            with self.subTest(n_objectives=n_objectives):
                result = QuickAdapterRegressorV3._validate_label_weights(
                    None, n_objectives, ctx="label_weights"
                )
                np.testing.assert_allclose(
                    result, np.full(n_objectives, 1.0 / n_objectives), rtol=DISTANCE_RTOL, atol=0.0
                )

    def test_a_weight_vector_is_normalised_to_sum_to_one(self):
        for weights in ([1.0, 3.0], (1.0, 3.0), np.array([1.0, 3.0]), [0.0, 2.0], [5.0, 5.0]):
            with self.subTest(weights=np.asarray(weights).tolist()):
                result = QuickAdapterRegressorV3._validate_label_weights(
                    weights, 2, ctx="label_weights"
                )
                self.assertAlmostEqual(float(result.sum()), 1.0, places=12)
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._validate_label_weights([1.0, 3.0], 2, ctx="label_weights"),
            np.array([0.25, 0.75]),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    def test_normalising_a_weight_vector_never_mutates_the_callers_array(self):
        weights = np.array([1.0, 3.0])
        QuickAdapterRegressorV3._validate_label_weights(weights, 2, ctx="label_weights")
        np.testing.assert_array_equal(weights, np.array([1.0, 3.0]))

    def test_a_weight_vector_is_scaled_by_its_own_maximum_before_summing(self):
        # Dividing by the maximum first is what keeps a vector of huge weights from
        # overflowing the sum into a non-finite normalisation.
        result = QuickAdapterRegressorV3._validate_label_weights(
            [1e308, 1e308], 2, ctx="label_weights"
        )
        np.testing.assert_allclose(result, np.array([0.5, 0.5]), rtol=DISTANCE_RTOL, atol=0.0)
        tiny = QuickAdapterRegressorV3._validate_label_weights(
            [1e-320, 2e-320], 2, ctx="label_weights"
        )
        np.testing.assert_allclose(
            tiny, np.array([1.0 / 3.0, 2.0 / 3.0]), rtol=DISTANCE_RTOL, atol=0.0
        )

    def test_an_unusable_weight_vector_falls_back_to_a_uniform_one_per_mode(self):
        cases = [
            ("label_weights", "str", lambda: "nope", ValueError, "must be a list, tuple, or array"),
            ("label_weights", "ndim", lambda: [[1.0, 2.0]], ValueError, "one-dimensional"),
            ("label_weights", "size", lambda: [1.0, 2.0], ValueError, "must contain 3 weights"),
            ("label_weights", "non-finite", lambda: [1.0, np.inf], ValueError, "non-finite"),
            ("label_weights", "negative", lambda: [1.0, -1.0], ValueError, "negative"),
            ("label_weights", "zero sum", lambda: [0.0, 0.0], ValueError, "sum is zero"),
            ("label_weights", "non-numeric", lambda: ["a", "b"], ValueError, "numeric weights"),
        ]
        for ctx, label, build, error, pattern in cases:
            with self.subTest(case=label):
                n_objectives = 3 if label == "size" else 2
                with self.assertRaisesRegex(error, pattern):
                    QuickAdapterRegressorV3._validate_label_weights(build(), n_objectives, ctx=ctx)
                with self.assertLogs(LOGGER_NAME, "WARNING") as captured:
                    warned = QuickAdapterRegressorV3._validate_label_weights(
                        build(), n_objectives, ctx=ctx, mode=WARN
                    )
                with self.assertNoLogs(LOGGER_NAME, "WARNING"):
                    silent = QuickAdapterRegressorV3._validate_label_weights(
                        build(), n_objectives, ctx=ctx, mode=NONE
                    )
                np.testing.assert_allclose(
                    warned, np.full(n_objectives, 1.0 / n_objectives), rtol=DISTANCE_RTOL, atol=0.0
                )
                np.testing.assert_allclose(
                    silent, np.full(n_objectives, 1.0 / n_objectives), rtol=DISTANCE_RTOL, atol=0.0
                )
                self.assertIn("using uniform weights", captured.output[0])

    def test_a_fully_zero_weight_vector_is_refused_rather_than_normalised_to_nan(self):
        with self.assertRaisesRegex(ValueError, "sum is zero"):
            QuickAdapterRegressorV3._validate_label_weights([0.0, 0.0], 2, ctx="label_weights")

    def test_a_weight_vector_with_a_zero_and_a_positive_entry_normalises(self):
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._validate_label_weights([0.0, 2.0], 2, ctx="label_weights"),
            np.array([0.0, 1.0]),
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    def test_an_enum_value_outside_the_valid_set_raises_the_canonical_message(self):
        valid = {"euclidean", "cityblock"}
        options = ("euclidean", "cityblock")
        for value in ("cosine", 5, None, b"euclidean"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    QuickAdapterRegressorV3._validate_enum_value(
                        value, valid, options, ctx="label_distance_metric"
                    )
                self.assertEqual(
                    str(caught.exception),
                    enum_error_message("label_distance_metric", value, options),
                )

    def test_an_enum_value_outside_the_valid_set_resolves_to_the_declared_default(self):
        valid = {"euclidean", "cityblock"}
        for mode in (WARN, NONE):
            with self.subTest(mode=mode):
                if mode == WARN:
                    with self.assertLogs(LOGGER_NAME, "WARNING") as captured:
                        result = QuickAdapterRegressorV3._validate_enum_value(
                            "cosine",
                            valid,
                            ("euclidean", "cityblock"),
                            ctx="label_distance_metric",
                            mode=mode,
                            default="euclidean",
                        )
                    self.assertIn("using 'euclidean'", captured.output[0])
                else:
                    with self.assertNoLogs(LOGGER_NAME, "WARNING"):
                        result = QuickAdapterRegressorV3._validate_enum_value(
                            "cosine",
                            valid,
                            ("euclidean", "cityblock"),
                            ctx="label_distance_metric",
                            mode=mode,
                            default="euclidean",
                        )
                self.assertEqual(result, "euclidean")

    def test_a_valid_enum_value_passes_through_untouched_and_unlogged(self):
        with self.assertNoLogs(LOGGER_NAME, "WARNING"):
            self.assertEqual(
                QuickAdapterRegressorV3._validate_enum_value(
                    "cityblock", {"euclidean", "cityblock"}, ("euclidean", "cityblock"), ctx="ctx"
                ),
                "cityblock",
            )

    def test_an_aggregate_route_accepts_every_non_probability_metric(self):
        # The aggregate methods reduce over objectives themselves, so the whole non-probability
        # set is admissible, including the named power means and the weighted sum.
        for metric in QuickAdapterRegressorV3._DISTANCE_METRICS:
            if metric in QuickAdapterRegressorV3._PROBABILITY_DISTANCE_METRICS_SET:
                continue
            with self.subTest(metric=metric), self.assertNoLogs(LOGGER_NAME, "WARNING"):
                self.assertEqual(
                    QuickAdapterRegressorV3._validate_label_selection_metric(
                        metric,
                        ctx="label_distance_metric",
                        default="euclidean",
                        aggregate_allowed=True,
                    ),
                    metric,
                )

    def test_an_aggregate_route_rejects_the_probability_metrics_too(self):
        # The three probability metrics are excluded from the aggregate set as well: they
        # compare distributions across objectives, and a minimised objective column is a
        # per-objective score, not a distribution over objectives.
        for metric in QuickAdapterRegressorV3._PROBABILITY_DISTANCE_METRICS_SET:
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                QuickAdapterRegressorV3._validate_label_selection_metric(
                    metric,
                    ctx="label_distance_metric",
                    default="euclidean",
                    aggregate_allowed=True,
                    mode=RAISE,
                )

    def test_a_cluster_or_density_route_rejects_the_aggregate_metrics(self):
        # Those routes hand the metric to SciPy/sklearn, which computes no reduction and
        # therefore cannot evaluate a power mean or a weighted sum at all.
        for metric in ("harmonic_mean", "quadratic_mean", "power_mean", "weighted_sum"):
            with self.subTest(metric=metric):
                with self.assertRaises(ValueError) as caught:
                    QuickAdapterRegressorV3._validate_label_selection_metric(
                        metric,
                        ctx="label_distance_metric",
                        default="euclidean",
                        aggregate_allowed=False,
                        mode=RAISE,
                    )
                self.assertIn("supported values are euclidean, minkowski", str(caught.exception))
                self.assertNotIn(
                    "power_mean", str(caught.exception).split("supported values are ")[1]
                )

    def test_a_cluster_or_density_route_rejects_the_probability_metrics_too(self):
        for metric in QuickAdapterRegressorV3._PROBABILITY_DISTANCE_METRICS_SET:
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                QuickAdapterRegressorV3._validate_label_selection_metric(
                    metric,
                    ctx="label_distance_metric",
                    default="euclidean",
                    aggregate_allowed=False,
                    mode=RAISE,
                )

    def test_a_cluster_or_density_route_warns_towards_its_default_by_default(self):
        with self.assertLogs(LOGGER_NAME, "WARNING") as captured:
            result = QuickAdapterRegressorV3._validate_label_selection_metric(
                "hellinger",
                ctx="label_distance_metric",
                default="cityblock",
                aggregate_allowed=False,
            )
        self.assertEqual(result, "cityblock")
        self.assertIn("using 'cityblock'", captured.output[0])

    # ------------------------------------------------------------------ cdist keyword preparation

    def test_no_weights_and_no_p_means_no_keywords_at_all(self):
        for metric in ("euclidean", "cityblock", "hellinger", "power_mean"):
            with self.subTest(metric=metric):
                self.assertEqual(
                    QuickAdapterRegressorV3._prepare_distance_kwargs(
                        metric, reference_matrix=unique_rows(4)
                    ),
                    {},
                )

    def test_a_supported_metric_receives_the_weight_vector_verbatim(self):
        weights = np.array([0.25, 0.75])
        kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
            "cityblock", weights=weights, reference_matrix=unique_rows(4)
        )
        self.assertEqual(set(kwargs), {"w"})
        np.testing.assert_array_equal(kwargs["w"], weights)

    def test_an_unsupported_metric_drops_the_weight_vector_silently_by_default(self):
        # The default mode is "none": the block is dropped, no warning, and the
        # standardized keyword is still built so the distance is computable.
        reference = unique_rows(4)
        for metric in QuickAdapterRegressorV3._UNSUPPORTED_WEIGHTS_METRICS:
            if metric not in QuickAdapterRegressorV3._STANDARDIZED_DISTANCE_METRICS_SET:
                with self.subTest(metric=metric), self.assertNoLogs(LOGGER_NAME, "WARNING"):
                    kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
                        metric, weights=np.array([0.5, 0.5]), reference_matrix=reference
                    )
                    self.assertNotIn("w", kwargs)
                continue
            with self.subTest(metric=metric), self.assertNoLogs(LOGGER_NAME, "WARNING"):
                kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
                    metric, weights=np.array([0.5, 0.5]), reference_matrix=reference
                )
                self.assertNotIn("w", kwargs)
                self.assertIn("V" if metric == "seuclidean" else "VI", kwargs)

    def test_the_weights_gate_can_be_turned_into_a_hard_failure(self):
        for metric in QuickAdapterRegressorV3._UNSUPPORTED_WEIGHTS_METRICS:
            with (
                self.subTest(metric=metric),
                self.assertRaisesRegex(ValueError, "does not support custom weights"),
            ):
                QuickAdapterRegressorV3._prepare_distance_kwargs(
                    metric,
                    weights=np.array([0.5, 0.5]),
                    mode=RAISE,
                    reference_matrix=unique_rows(4),
                )
                with self.assertLogs(LOGGER_NAME, "WARNING") as captured:
                    kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
                        metric,
                        weights=np.array([0.5, 0.5]),
                        mode=WARN,
                        reference_matrix=unique_rows(4),
                    )
                self.assertNotIn("w", kwargs)
                self.assertIn("using uniform weights", captured.output[0])

    def test_a_minkowski_p_reaches_cdist_only_when_it_validates(self):
        reference = unique_rows(4)
        np.testing.assert_allclose(
            QuickAdapterRegressorV3._prepare_distance_kwargs(
                "minkowski", p=3.0, reference_matrix=reference
            )["p"],
            3.0,
            rtol=0.0,
            atol=0.0,
        )
        self.assertNotIn(
            "p",
            QuickAdapterRegressorV3._prepare_distance_kwargs(
                "minkowski", reference_matrix=reference
            ),
        )
        self.assertNotIn(
            "p",
            QuickAdapterRegressorV3._prepare_distance_kwargs(
                "minkowski", p=0.0, reference_matrix=reference
            ),
        )
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            QuickAdapterRegressorV3._prepare_distance_kwargs(
                "minkowski", p=0.0, mode=RAISE, reference_matrix=reference
            )

    def test_a_non_minkowski_metric_never_receives_a_p(self):
        kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
            "cityblock", p=3.0, reference_matrix=unique_rows(4)
        )
        self.assertNotIn("p", kwargs)

    def test_a_standardized_metric_receives_its_variance_vector(self):
        reference = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
        kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
            "seuclidean", reference_matrix=reference
        )
        self.assertEqual(set(kwargs), {"V"})
        np.testing.assert_allclose(
            kwargs["V"], np.var(reference, axis=0, ddof=1), rtol=DISTANCE_RTOL, atol=0.0
        )

    def test_a_mahalanobis_metric_receives_the_pseudoinverse_of_its_covariance(self):
        reference = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
        kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
            "mahalanobis", reference_matrix=reference
        )
        self.assertEqual(set(kwargs), {"VI"})
        covariance = np.cov(reference, rowvar=False, ddof=1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        # A full-rank front has every eigenvalue well above the floor, so the spectral
        # inverse is the ordinary matrix inverse.
        self.assertTrue(np.all(eigenvalues > eigenvalues[-1] * np.sqrt(np.finfo(float).eps)))
        np.testing.assert_allclose(
            kwargs["VI"], np.linalg.inv(covariance), rtol=DISTANCE_RTOL, atol=0.0
        )

    def test_a_mahalanobis_metric_floors_a_singular_covariance_instead_of_failing(self):
        # Collinear objectives pass the positive-variance guard yet give a rank-deficient
        # covariance, so the plain inverse is singular. The floor keeps the keyword
        # finite and the metric computable rather than raising on a legitimate front.
        collinear = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        covariance = np.cov(collinear, rowvar=False, ddof=1)
        self.assertLess(np.linalg.matrix_rank(covariance), 2)
        kwargs = QuickAdapterRegressorV3._prepare_distance_kwargs(
            "mahalanobis", reference_matrix=collinear
        )
        self.assertTrue(np.all(np.isfinite(kwargs["VI"])))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = eigenvalues[-1] * np.sqrt(np.finfo(float).eps)
        np.testing.assert_allclose(
            kwargs["VI"],
            (eigenvectors / np.maximum(eigenvalues, floor)) @ eigenvectors.T,
            rtol=DISTANCE_RTOL,
            atol=0.0,
        )

    def test_a_standardized_metric_refuses_a_front_that_cannot_support_a_covariance(self):
        constant = np.array([[1.0, 1.0], [1.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "positive active-objective variances"):
            QuickAdapterRegressorV3._prepare_distance_kwargs(
                "seuclidean", reference_matrix=constant
            )
        with self.assertRaisesRegex(ValueError, "at least two rows"):
            QuickAdapterRegressorV3._prepare_distance_kwargs(
                "seuclidean", reference_matrix=np.array([[1.0, 2.0]])
            )
        with self.assertRaisesRegex(ValueError, "at least two rows"):
            QuickAdapterRegressorV3._prepare_distance_kwargs(
                "seuclidean", reference_matrix=np.array([[1.0, 2.0], [np.nan, 3.0], [3.0, 4.0]])
            )


if __name__ == "__main__":
    unittest.main()
