"""Tunable validation algebra; requires the Freqtrade QA image."""

import logging
import typing
import unittest

import numpy as np
from qa_support import QaTestCase
from Utils import (
    _BoolValidator,
    _DictValidator,
    _EnumValidator,
    _NumericValidator,
    _ParamSpec,
    _RangeValidator,
    _validate_params,
    require_bool,
    require_numeric,
    validate_range,
)

LOGGER = logging.getLogger("test-validation")
NAME = "sample"


class EnumValidatorTest(QaTestCase):
    def test_membership_is_the_whole_contract(self):
        validator = _EnumValidator(("none", "uniform"))
        self.assertTrue(validator("none"))
        self.assertFalse(validator("zigzag"))
        self.assertFalse(validator(None))

    def test_the_message_lists_the_supported_values(self):
        self.assertEqual(_EnumValidator(("a", "b")).message(NAME), "supported values are a, b")


class NumericValidatorTest(QaTestCase):
    def test_bounds_are_inclusive_by_default(self):
        validator = _NumericValidator(min_value=0.0, max_value=1.0)
        self.assertTrue(validator(0.0))
        self.assertTrue(validator(1.0))
        self.assertFalse(validator(-0.1))
        self.assertFalse(validator(1.1))

    def test_exclusive_bounds_reject_the_endpoint(self):
        validator = _NumericValidator(
            min_value=0.0, max_value=1.0, min_exclusive=True, max_exclusive=True
        )
        self.assertFalse(validator(0.0))
        self.assertFalse(validator(1.0))
        self.assertTrue(validator(0.5))

    def test_booleans_are_refused_even_though_they_are_integers(self):
        validator = _NumericValidator()
        self.assertFalse(validator(True))
        self.assertFalse(validator(False))

    def test_non_finite_values_are_refused(self):
        validator = _NumericValidator()
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=repr(value)):
                self.assertFalse(validator(value))

    def test_require_int_refuses_a_float(self):
        validator = _NumericValidator(require_int=True)
        self.assertTrue(validator(3))
        self.assertFalse(validator(3.0))

    def test_the_message_reflects_the_declared_constraints(self):
        self.assertEqual(
            _NumericValidator(min_value=0.0, max_value=1.0).message(NAME),
            "must be a finite number >= 0.0 <= 1.0",
        )
        self.assertEqual(
            _NumericValidator(require_int=True, min_value=2, min_exclusive=True).message(NAME),
            "must be an integer > 2",
        )


class RangeValidatorTest(QaTestCase):
    def test_a_rising_pair_is_accepted(self):
        validator = _RangeValidator()
        self.assertTrue(validator((1, 2)))
        self.assertTrue(validator([1.5, 2.5]))

    def test_an_equal_pair_is_refused(self):
        self.assertFalse(_RangeValidator()((1, 1)))

    def test_a_falling_pair_is_refused(self):
        self.assertFalse(_RangeValidator()((2, 1)))

    def test_the_pair_must_have_exactly_two_finite_numbers(self):
        validator = _RangeValidator()
        for value in ((1,), (1, 2, 3), (1, np.nan), (1, "2"), 5, None):
            with self.subTest(value=repr(value)):
                self.assertFalse(validator(value))

    def test_outer_bounds_are_enforced(self):
        validator = _RangeValidator(min_bound=1.0, max_bound=10.0)
        self.assertTrue(validator((1.0, 10.0)))
        self.assertFalse(validator((0.5, 10.0)))
        self.assertFalse(validator((1.0, 10.5)))

    def test_the_message_states_the_closed_form(self):
        self.assertEqual(_RangeValidator().message(NAME), "must be (low, high) with low < high")
        self.assertEqual(
            _RangeValidator(min_bound=1.0, max_bound=9.0).message(NAME),
            "must be (low, high) with 1.0 <= low < high <= 9.0",
        )


class DictAndBoolValidatorTest(QaTestCase):
    def test_the_dict_validator_accepts_any_mapping(self):
        validator = _DictValidator()
        self.assertTrue(validator({}))
        self.assertTrue(validator({"a": 1}))
        self.assertFalse(validator([("a", 1)]))
        self.assertFalse(validator(None))

    def test_the_bool_validator_is_strict(self):
        validator = _BoolValidator()
        self.assertTrue(validator(True))
        self.assertFalse(validator(1))
        self.assertFalse(validator("true"))


class ValidateParamsTest(QaTestCase):
    SPECS: typing.ClassVar[dict] = {
        "mode": _ParamSpec(_EnumValidator(("a", "b"))),
        "scale": _ParamSpec(_NumericValidator(min_value=0.0), output_type=float),
        "window": _ParamSpec(_NumericValidator(require_int=True), output_type=int),
        "pair": _ParamSpec(_RangeValidator(), output_type=tuple),
        "enabled": _ParamSpec(_BoolValidator()),
        "options": _ParamSpec(_DictValidator(valid_keys=("x", "y"))),
    }
    DEFAULTS: typing.ClassVar[dict] = {
        "mode": "a",
        "scale": 1.0,
        "window": 3,
        "pair": (0, 1),
        "enabled": False,
        "options": {"x": 1},
    }

    def _validate(self, config):
        return _validate_params(config, LOGGER, "test", self.SPECS, self.DEFAULTS)

    def test_every_specified_key_is_present_in_the_result(self):
        self.assertEqual(set(self._validate({})), set(self.SPECS))

    def test_a_missing_key_takes_its_default(self):
        self.assertEqual(self._validate({})["mode"], "a")

    def test_a_valid_value_is_kept(self):
        self.assertEqual(self._validate({"mode": "b"})["mode"], "b")

    def test_an_invalid_value_falls_back_to_the_default_with_a_warning(self):
        with self.assertLogs(LOGGER, level="WARNING") as captured:
            result = self._validate({"mode": "zzz"})
        self.assertEqual(result["mode"], "a")
        self.assertTrue(any("Invalid test mode" in line for line in captured.output))

    def test_output_type_coercion_is_applied(self):
        result = self._validate({"scale": 2, "window": 5})
        self.assertIsInstance(result["scale"], float)
        self.assertIsInstance(result["window"], int)

    def test_a_list_pair_becomes_a_two_tuple(self):
        result = self._validate({"pair": [3, 4]})
        self.assertEqual(result["pair"], (3, 4))
        self.assertIsInstance(result["pair"], tuple)

    def test_unknown_dictionary_keys_are_dropped_with_a_warning(self):
        with self.assertLogs(LOGGER, level="WARNING") as captured:
            result = self._validate({"options": {"x": 1, "zzz": 2}})
        self.assertEqual(result["options"], {"x": 1})
        self.assertTrue(any("keys ['zzz']" in line for line in captured.output))

    def test_coercion_runs_even_after_a_fallback(self):
        result = self._validate({"scale": -1})
        self.assertIsInstance(result["scale"], float)
        self.assertEqual(result["scale"], 1.0)


class RequireNumericTest(QaTestCase):
    def test_a_builtin_number_is_returned_unchanged(self):
        self.assertEqual(require_numeric(2, NAME, context="ctx"), 2)
        self.assertEqual(require_numeric(2.5, NAME, context="ctx"), 2.5)

    def test_booleans_are_refused(self):
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            require_numeric(True, NAME, context="ctx")

    def test_a_numpy_scalar_is_refused_because_the_type_is_exact(self):
        # The check is `type(value) not in (int, float)`, so a numpy scalar is rejected
        # even though it compares and divides like a number.
        with self.assertRaisesRegex(ValueError, "must be a finite number"):
            require_numeric(np.float64(1.0), NAME, context="ctx")
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            require_numeric(np.int64(1), NAME, context="ctx", require_int=True)

    def test_bounds_are_enforced_with_the_configured_exclusivity(self):
        with self.assertRaisesRegex(ValueError, "must be a finite number >= 1.0"):
            require_numeric(0.5, NAME, context="ctx", minimum=1.0)
        self.assertEqual(require_numeric(1.0, NAME, context="ctx", minimum=1.0), 1.0)
        with self.assertRaisesRegex(ValueError, "must be a finite number > 1.0"):
            require_numeric(1.0, NAME, context="ctx", minimum=1.0, min_exclusive=True)

    def test_require_int_refuses_a_float(self):
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            require_numeric(1.0, NAME, context="ctx", require_int=True)

    def test_the_message_names_the_context_and_the_parameter(self):
        with self.assertRaisesRegex(ValueError, r"Invalid ctx\.sample value"):
            require_numeric("x", NAME, context="ctx")


class RequireBoolTest(QaTestCase):
    def test_a_real_boolean_passes_through(self):
        self.assertIs(require_bool(True, NAME, context="ctx"), True)
        self.assertIs(require_bool(False, NAME, context="ctx"), False)

    def test_an_integer_one_is_refused(self):
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            require_bool(1, NAME, context="ctx")


class ValidateRangeTest(QaTestCase):
    def _range(self, low, high, **kwargs):
        return validate_range(
            low,
            high,
            LOGGER,
            name="window",
            default_min=kwargs.pop("default_min", 2),
            default_max=kwargs.pop("default_max", 10),
            **kwargs,
        )

    def test_a_valid_rising_range_is_returned_unchanged(self):
        self.assertEqual(self._range(3, 8), (3, 8))

    def test_a_falling_range_falls_back_to_the_defaults(self):
        with self.assertLogs(LOGGER, level="WARNING"):
            self.assertEqual(self._range(8, 3), (2, 10))

    def test_an_equal_range_falls_back_unless_equal_is_allowed(self):
        with self.assertLogs(LOGGER, level="WARNING"):
            self.assertEqual(self._range(4, 4), (2, 10))
        self.assertEqual(self._range(4, 4, allow_equal=True), (4, 4))

    def test_a_non_numeric_component_falls_back_on_its_own(self):
        with self.assertLogs(LOGGER, level="WARNING"):
            self.assertEqual(self._range("x", 8), (2, 8))
        with self.assertLogs(LOGGER, level="WARNING"):
            self.assertEqual(self._range(3, None), (3, 10))

    def test_a_negative_component_falls_back_by_default(self):
        with self.assertLogs(LOGGER, level="WARNING"):
            self.assertEqual(self._range(-1, 8), (2, 8))

    def test_a_component_above_the_configured_maximum_falls_back(self):
        with self.assertLogs(LOGGER, level="WARNING"):
            self.assertEqual(self._range(3, 50, max_value=10), (3, 10))

    def test_non_finite_components_fall_back(self):
        with self.assertLogs(LOGGER, level="WARNING"):
            self.assertEqual(self._range(np.nan, 8), (2, 8))

    def test_malformed_defaults_are_a_hard_failure_not_a_fallback(self):
        for kwargs in (
            {"default_min": "x"},
            {"default_min": 10, "default_max": 2},
            {"default_min": 5, "default_max": 5},
            {"default_min": 11, "default_max": 20, "max_value": 10},
        ):
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, "Invalid window"):
                self._range(3, 8, **kwargs)

    def test_equal_defaults_are_allowed_when_equal_is_permitted(self):
        self.assertEqual(
            validate_range(
                5,
                5,
                LOGGER,
                name="window",
                default_min=5,
                default_max=5,
                allow_equal=True,
            ),
            (5, 5),
        )


if __name__ == "__main__":
    unittest.main()
