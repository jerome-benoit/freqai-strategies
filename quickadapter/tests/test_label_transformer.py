"""Label scaling pipeline contract: round trip, non-finite handling and fit-time refusal.; requires the Freqtrade QA image."""

import unittest

import numpy as np
import pandas as pd
from LabelTransformer import (
    DEFAULTS_LABEL_PIPELINE,
    NORMALIZATION_TYPES,
    STANDARDIZATION_TYPES,
    LabelTransformer,
    enum_error_message,
)
from qa_support import QaTestCase

LABEL_ARRAY_RTOL = 1e-9
LABEL_ARRAY_ATOL = 1e-9
ROUND_TRIP_RTOL = 1e-6
ROUND_TRIP_ATOL = 1e-9


def label_matrix() -> np.ndarray:
    """Multi-column label matrix: both signs, non-constant in every column, |w| <= 5.

    The magnitude bound is load-bearing: ``sigmoid`` saturates towards the closed
    interval, so a wide column loses the information its inverse needs. Inside
    +/-5 every standardization/normalization pair stays inside the invertible region.
    """
    return np.array(
        [
            [0.0, -1.0, 2.0],
            [1.0, -0.25, -3.0],
            [-2.0, 0.5, 4.0],
            [0.75, 2.0, -0.5],
            [-0.5, -3.0, 1.5],
            [2.5, 1.25, 0.0],
            [-1.5, -1.0, 3.0],
            [0.125, 0.75, -4.0],
            [-0.25, 3.0, 0.25],
            [1.5, -2.5, -1.0],
        ]
    )


def label_series() -> np.ndarray:
    """Single label vector, one element per distinct power of two scale."""
    return np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.25, -0.25, 4.0, -4.0, 0.5, -0.5])


def mixed_finite_matrix() -> np.ndarray:
    """Label matrix carrying NaN, +inf and -inf at known positions in every column."""
    return np.array(
        [
            [1.0, np.nan, 0.5],
            [2.0, 3.0, -0.5],
            [np.inf, 1.5, 0.25],
            [-3.0, -np.inf, 2.0],
            [4.0, 0.75, -1.5],
            [-np.inf, 2.5, 0.125],
        ]
    )


def transformer(standardization: str = "none", normalization: str = "none", **overrides):
    """Build a LabelTransformer from pipeline keys, leaving the rest at canonical defaults."""
    return LabelTransformer(
        label_transformer={
            "default": {
                "standardization": standardization,
                "normalization": normalization,
                **overrides,
            }
        }
    )


class LabelTransformerTest(QaTestCase):
    def test_robust_scaling_centers_on_the_median_and_divides_by_the_iqr(self):
        series = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
        fitted = transformer("robust", "none")
        fitted.fit(series)
        scaled, *_ = fitted.transform(series)
        # Median 2, quartiles 1 and 3: the outlier does not set the scale.
        np.testing.assert_allclose(scaled, [-1.0, -0.5, 0.0, 0.5, 49.0], rtol=1e-12)

    def test_inverse_transform_recovers_the_input_for_every_scaler_pair(self):
        matrix = label_matrix()
        for standardization in STANDARDIZATION_TYPES:
            for normalization in NORMALIZATION_TYPES:
                with self.subTest(standardization=standardization, normalization=normalization):
                    fitted = transformer(standardization, normalization)
                    fitted.fit(matrix)
                    scaled, *_ = fitted.transform(matrix)
                    recovered, *_ = fitted.inverse_transform(scaled)
                    np.testing.assert_allclose(
                        recovered, matrix, rtol=ROUND_TRIP_RTOL, atol=ROUND_TRIP_ATOL
                    )

    def test_a_non_finite_entry_keeps_its_position_and_kind_through_both_directions(self):
        matrix = mixed_finite_matrix()
        finite = np.isfinite(matrix)
        for standardization in STANDARDIZATION_TYPES:
            for normalization in NORMALIZATION_TYPES:
                with self.subTest(standardization=standardization, normalization=normalization):
                    fitted = transformer(standardization, normalization, gamma=0.5)
                    fitted.fit(matrix)
                    scaled, *_ = fitted.transform(matrix)
                    recovered, *_ = fitted.inverse_transform(scaled)
                    for result in (scaled, recovered):
                        np.testing.assert_array_equal(np.isnan(result), np.isnan(matrix))
                        np.testing.assert_array_equal(np.isposinf(result), np.isposinf(matrix))
                        np.testing.assert_array_equal(np.isneginf(result), np.isneginf(matrix))
                        self.assertTrue(np.all(np.isfinite(result[finite])))

    def test_the_finite_entries_are_still_scaled_next_to_a_dropped_one(self):
        matrix = mixed_finite_matrix()
        fitted = transformer("none", "maxabs")
        fitted.fit(matrix)
        scaled, *_ = fitted.transform(matrix)
        finite = np.isfinite(matrix)
        self.assertEqual(float(np.max(np.abs(scaled[finite]))), 1.0)
        self.assertGreater(float(np.max(np.abs(scaled[finite] - matrix[finite]))), 0.0)

    def test_an_all_non_finite_column_is_fitted_on_the_fallback_and_left_untouched(self):
        matrix = np.array([[1.0, np.nan], [2.0, np.inf], [3.0, np.nan], [4.0, -np.inf]])
        fitted = transformer("zscore", "minmax")
        with self.assertLogs("LabelTransformer", level="WARNING") as captured:
            fitted.fit(matrix)
            scaled, *_ = fitted.transform(matrix)
        self.assertTrue(any("No finite values found" in line for line in captured.output))
        np.testing.assert_array_equal(np.isnan(scaled[:, 1]), [True, False, True, False])
        np.testing.assert_array_equal(np.isposinf(scaled[:, 1]), [False, True, False, False])
        np.testing.assert_array_equal(np.isneginf(scaled[:, 1]), [False, False, False, True])
        recovered, *_ = fitted.inverse_transform(scaled)
        np.testing.assert_array_equal(np.isnan(recovered[:, 1]), [True, False, True, False])
        np.testing.assert_array_equal(np.isneginf(recovered[:, 1]), [False, False, False, True])

    def test_transform_before_fit_names_the_forward_verb(self):
        fitted = transformer()
        with self.assertRaises(RuntimeError) as caught:
            fitted.transform(label_matrix())
        self.assertIn("must be fitted before transform", str(caught.exception))

    def test_inverse_transform_before_fit_names_the_inverse_verb(self):
        fitted = transformer()
        with self.assertRaises(RuntimeError) as caught:
            fitted.inverse_transform(label_matrix())
        self.assertIn("must be fitted before inverse_transform", str(caught.exception))

    def test_fit_transform_marks_the_transformer_as_fitted(self):
        matrix = label_matrix()
        fitted = transformer("zscore", "minmax")
        scaled, *_ = fitted.fit_transform(matrix)
        recovered, *_ = fitted.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, matrix, rtol=ROUND_TRIP_RTOL, atol=ROUND_TRIP_ATOL)

    def test_an_unseen_column_name_is_refused(self):
        matrix = label_matrix()
        fitted = transformer()
        fitted.fit(matrix, feature_list=["a", "b", "c"])
        with self.assertRaises(ValueError) as caught:
            fitted.transform(matrix, feature_list=["a", "b", "renamed"])
        self.assertEqual(str(caught.exception), "Column 'renamed' was not present during fitting")

    def test_a_column_count_mismatch_names_both_counts(self):
        fitted = transformer()
        fitted.fit(label_matrix(), feature_list=["a", "b", "c"])
        for n_columns in (1, 5):
            with self.subTest(n_columns=n_columns):
                with self.assertRaises(ValueError) as caught:
                    fitted.transform(np.zeros((4, n_columns)))
                self.assertEqual(
                    str(caught.exception),
                    f"Column count mismatch: fitted on 3 columns, got {n_columns}",
                )

    def test_a_one_dimensional_input_comes_back_one_dimensional(self):
        series = label_series()
        for standardization in STANDARDIZATION_TYPES:
            with self.subTest(standardization=standardization):
                fitted = transformer(standardization, "minmax")
                fitted.fit(series)
                scaled, *_ = fitted.transform(series)
                recovered, *_ = fitted.inverse_transform(scaled)
                self.assertEqual(scaled.ndim, 1)
                self.assertEqual(recovered.ndim, 1)
                np.testing.assert_allclose(
                    recovered, series, rtol=ROUND_TRIP_RTOL, atol=ROUND_TRIP_ATOL
                )

    def test_the_column_vector_form_comes_back_two_dimensional(self):
        series = label_series()
        fitted = transformer("zscore", "minmax")
        fitted.fit(series)
        scaled_1d, *_ = fitted.transform(series)
        scaled_2d, *_ = fitted.transform(series.reshape(-1, 1))
        self.assertEqual(scaled_2d.ndim, 2)
        np.testing.assert_array_equal(scaled_2d.ravel(), scaled_1d)

    def test_the_feature_list_names_the_columns_it_matches(self):
        matrix = label_matrix()
        fitted = LabelTransformer(
            label_transformer={
                "default": {"standardization": "none", "normalization": "none"},
                "columns": {"b": {"standardization": "zscore"}},
            }
        )
        fitted.fit(matrix, feature_list=["a", "b", "c"])
        scaled, *_ = fitted.transform(matrix)
        np.testing.assert_allclose(scaled[:, 0], matrix[:, 0], rtol=0.0, atol=0.0)
        self.assertLess(float(np.max(np.abs(scaled[:, 1]))), np.max(np.abs(matrix[:, 1])))
        np.testing.assert_allclose(scaled[:, 2], matrix[:, 2], rtol=0.0, atol=0.0)

    def test_a_feature_list_of_the_wrong_length_leaves_the_columns_positionally_named(self):
        matrix = label_matrix()
        fitted = LabelTransformer(
            label_transformer={
                "default": {"standardization": "none", "normalization": "none"},
                "columns": {"column_1": {"standardization": "zscore"}},
            }
        )
        fitted.fit(matrix, feature_list=["only_one"])
        scaled, *_ = fitted.transform(matrix)
        np.testing.assert_allclose(scaled[:, 0], matrix[:, 0], rtol=0.0, atol=0.0)
        self.assertLess(float(np.max(np.abs(scaled[:, 1]))), np.max(np.abs(matrix[:, 1])))
        np.testing.assert_allclose(scaled[:, 2], matrix[:, 2], rtol=0.0, atol=0.0)

    def test_constant_columns_scale_to_zero_and_round_trip_exactly(self):
        smallest = np.nextafter(0.0, 1.0)
        for value in (2.5, 1.7e308, -1.7e308, smallest, -smallest):
            for rows in (1, 2, 3):
                for normalization in ("none", "maxabs"):
                    with self.subTest(value=value, rows=rows, normalization=normalization):
                        column = np.full((rows, 1), value)
                        fitted = transformer("mmad", normalization)
                        with np.errstate(over="raise", invalid="raise"):
                            fitted.fit(column)
                            scaled, *_ = fitted.transform(column)
                            recovered, *_ = fitted.inverse_transform(scaled)
                        np.testing.assert_array_equal(scaled, np.zeros_like(column))
                        np.testing.assert_array_equal(recovered, column)

    def test_a_mad_below_the_isclose_tolerance_is_treated_as_collapsed(self):
        rng = np.random.default_rng(3)
        column = 1.0 + rng.normal(scale=1e-10, size=(12, 1))
        fitted = transformer("mmad", "none")
        fitted.fit(column)
        scaled, *_ = fitted.transform(column)
        self.assertTrue(np.all(np.isfinite(scaled)))
        # Dividing by the tiny MAD would produce order-one standardized residuals;
        # the collapsed scale keeps them near their original magnitude.
        self.assertLess(float(np.max(np.abs(scaled))), 1e-6)
        recovered, *_ = fitted.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, column, rtol=ROUND_TRIP_RTOL, atol=ROUND_TRIP_ATOL)

    def test_sigmoid_is_the_identity_when_its_scale_is_zero_or_not_finite(self):
        series = label_series()
        for scale in (0.0, 1e-12, np.inf, np.nan):
            with self.subTest(sigmoid_scale=scale):
                fitted = transformer("none", "sigmoid", sigmoid_scale=scale)
                fitted.fit(series)
                scaled, *_ = fitted.transform(series)
                recovered, *_ = fitted.inverse_transform(scaled)
                np.testing.assert_array_equal(scaled, series)
                np.testing.assert_array_equal(recovered, series)

    def test_the_sigmoid_inverse_clips_to_the_open_interval_and_stays_finite(self):
        series = label_series()
        fitted = transformer("none", "sigmoid")
        fitted.fit(series)
        with self.assertLogs("LabelTransformer", level="WARNING") as captured:
            recovered, *_ = fitted.inverse_transform(np.array([-4.0, -1.0, 0.0, 1.0, 4.0]))
        self.assertTrue(any("Clipped 4 value(s)" in line for line in captured.output))
        self.assertTrue(np.all(np.isfinite(recovered)))
        eps = np.finfo(float).eps
        # logit((clip + 1) / 2) at clip = 1 - eps is the largest magnitude the inverse can emit.
        bound = float(np.log((2.0 - eps) / eps))
        self.assertLessEqual(float(np.max(np.abs(recovered))), bound)

    def test_gamma_preserves_the_sign_and_is_the_identity_at_one(self):
        series = label_series()
        fitted = transformer("none", "none", gamma=0.5)
        fitted.fit(series)
        scaled, *_ = fitted.transform(series)
        np.testing.assert_array_equal(np.sign(scaled), np.sign(series))
        np.testing.assert_allclose(
            scaled,
            np.sign(series) * np.sqrt(np.abs(series)),
            rtol=LABEL_ARRAY_RTOL,
            atol=LABEL_ARRAY_ATOL,
        )
        recovered, *_ = fitted.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, series, rtol=ROUND_TRIP_RTOL, atol=ROUND_TRIP_ATOL)

    def test_gamma_is_the_identity_at_one_and_at_a_degenerate_value(self):
        series = label_series()
        for gamma in (1.0, 0.0, -1.0, np.inf, np.nan):
            with self.subTest(gamma=gamma):
                fitted = transformer("none", "none", gamma=gamma)
                fitted.fit(series)
                scaled, *_ = fitted.transform(series)
                recovered, *_ = fitted.inverse_transform(scaled)
                np.testing.assert_array_equal(scaled, series)
                np.testing.assert_array_equal(recovered, series)

    def test_an_unknown_standardization_name_is_refused_at_fit_with_the_canonical_message(self):
        with self.assertRaises(ValueError) as caught:
            transformer("zigzag").fit(label_matrix())
        self.assertEqual(
            str(caught.exception),
            enum_error_message("standardization", "zigzag", STANDARDIZATION_TYPES),
        )

    def test_an_unknown_normalization_name_is_refused_at_fit_with_the_canonical_message(self):
        with self.assertRaises(ValueError) as caught:
            transformer("none", "softmax").fit(label_matrix())
        self.assertEqual(
            str(caught.exception),
            enum_error_message("normalization", "softmax", NORMALIZATION_TYPES),
        )

    def test_the_canonical_pipeline_defaults_are_not_mutated_by_construction(self):
        before = dict(DEFAULTS_LABEL_PIPELINE)
        transformer("zscore", "minmax", gamma=2.0).fit(label_matrix())
        self.assertEqual(DEFAULTS_LABEL_PIPELINE, before)

    def test_a_flat_config_dict_is_read_as_the_default_block(self):
        # The regressor hands over whatever get_label_pipeline_config produced, which may carry
        # pipeline keys at the top level instead of under "default".
        #
        # A round trip alone is satisfied by ANY invertible pipeline, including one that
        # ignores the flat dict entirely and falls back to `DEFAULTS_LABEL_PIPELINE`
        # (standardization "none", normalization "maxabs"). Two mutations proved it: dropping
        # the config merge and dropping the key filter each left the suite green while the
        # fitted scaler changed. So this asserts what the config bought, and that the
        # unknown key was dropped rather than carried.
        matrix = label_matrix()
        fitted = LabelTransformer(
            label_transformer={"standardization": "zscore", "normalization": "minmax", "ignored": 1}
        )
        fitted.fit(matrix)
        scaled, *_ = fitted.transform(matrix)

        self.assertEqual(fitted._config.default["standardization"], "zscore")
        self.assertEqual(fitted._config.default["normalization"], "minmax")
        self.assertNotIn("ignored", fitted._config.default)
        # minmax pins the extremes, and that is what the `DEFAULTS_LABEL_PIPELINE` fallback
        # (none/maxabs) cannot produce: maxabs maps into (0, 1], so its minimum is strictly
        # positive. The column's MEAN is deliberately not asserted — minmax is affine and runs
        # after zscore, so centring does not survive it.
        column = scaled[:, 0]
        self.assertAlmostEqual(float(column.min()), -1.0, places=12)
        self.assertAlmostEqual(float(column.max()), 1.0, places=12)
        recovered, *_ = fitted.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, matrix, rtol=ROUND_TRIP_RTOL, atol=ROUND_TRIP_ATOL)

    def test_a_labelled_frame_round_trips_through_its_own_column_names(self):
        frame = pd.DataFrame({"&-extrema": [1.0, -2.0, 0.5, 3.0], "other": [0.0, 1.0, -1.0, 2.0]})
        fitted = transformer("zscore", "minmax")
        fitted.fit(frame, feature_list=list(frame.columns))
        scaled, *_ = fitted.transform(frame)
        self.assertEqual(scaled.shape, frame.shape)
        recovered, *_ = fitted.inverse_transform(scaled)
        np.testing.assert_allclose(
            recovered, frame.to_numpy(dtype=float), rtol=ROUND_TRIP_RTOL, atol=ROUND_TRIP_ATOL
        )


if __name__ == "__main__":
    unittest.main()
