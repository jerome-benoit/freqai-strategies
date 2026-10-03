"""Canonical enum error message contract; requires the Freqtrade QA image."""

import unittest

from EnumErrors import enum_error_message
from qa_support import QaTestCase


class EnumErrorMessageTest(QaTestCase):
    def test_the_message_carries_the_context_the_value_and_the_options(self):
        self.assertEqual(
            enum_error_message("data_split_parameters.shuffle", True, ("true", "false")),
            "Invalid data_split_parameters.shuffle value True: supported values are true, false",
        )

    def test_the_value_is_rendered_with_its_repr(self):
        # repr, not str: a boolean must read as True rather than a bare 1, and a string
        # must keep its quotes so an empty value is still visible.
        self.assertIn("value True:", enum_error_message("x", True, ("a",)))
        self.assertIn("value '':", enum_error_message("x", "", ("a",)))
        self.assertIn("value None:", enum_error_message("x", None, ("a",)))

    def test_options_are_joined_in_the_given_order(self):
        self.assertTrue(
            enum_error_message("x", 1, ("first", "second")).endswith(
                "supported values are first, second"
            )
        )

    def test_a_single_option_is_not_pluralised_awkwardly(self):
        self.assertTrue(enum_error_message("x", 1, ("only",)).endswith("supported values are only"))

    def test_no_options_yields_an_empty_list(self):
        self.assertTrue(enum_error_message("x", 1, ()).endswith("supported values are "))

    def test_the_same_inputs_always_produce_the_same_message(self):
        first = enum_error_message("ctx", "v", ("a", "b"))
        second = enum_error_message("ctx", "v", ("a", "b"))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
