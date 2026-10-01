"""Log formatting of scalars and structures; requires the Freqtrade QA image."""

import os
import subprocess
import sys
import unittest

import numpy as np
from qa_support import QaTestCase
from Utils import (
    _format_value,
    _FormatContext,
    format_dict,
    format_number,
)


def _ctx(quote_strings: bool = True, sig_digits: int = 5) -> _FormatContext:
    return _FormatContext(quote_strings=quote_strings, sig_digits=sig_digits)


class UtilsFormattingTest(QaTestCase):
    def test_non_finite_scalars_render_as_symbols(self):
        for value, expected in (
            (float("inf"), "+∞"),
            (float("-inf"), "-∞"),
            (float("nan"), "NaN"),
            (np.float64("inf"), "+∞"),
            (np.float32("-inf"), "-∞"),
            (np.float32("nan"), "NaN"),
        ):
            with self.subTest(value=repr(value)):
                self.assertEqual(expected, format_number(value))
        self.assertEqual(
            "{p: +∞, n: -∞, x: NaN}",
            format_dict({"p": float("inf"), "n": float("-inf"), "x": float("nan")}),
        )

    def test_zero_renders_without_a_decimal_point(self):
        for value in (0, 0.0, -0.0, np.float64(0.0), np.int64(0)):
            with self.subTest(value=repr(value)):
                self.assertEqual("0", format_number(value))
        self.assertEqual("{a: 0, b: 0}", format_dict({"a": 0, "b": -0.0}))

    def test_scientific_notation_is_used_outside_the_fixed_range(self):
        # The bounds are inclusive on the outside: |v| >= 1e12 and 0 < |v| <= 1e-6 switch over,
        # so 1e-6 is scientific while the next float above it is fixed.
        scientific = (1e-6, 9.99e-7, 1e-7, 5e-324, 1e12, 1.5e12, 1e20, -1e-6, -1e12, -5e-324)
        fixed = (1.0000001e-6, 1e-5, 0.1, 1.5, 12345.6789, 1e11, 999_999_999_999.0, -1e12 + 1.0)
        for value in scientific:
            with self.subTest(value=repr(value), expected="scientific"):
                self.assertIn("e", format_number(value))
        for value in fixed:
            with self.subTest(value=repr(value), expected="fixed"):
                self.assertNotIn("e", format_number(value))

    def test_fixed_notation_strips_trailing_zeros(self):
        for value, expected in (
            (2.0, "2"),
            (1.0, "1"),
            (100.0, "100"),
            (1.5, "1.5"),
            (0.1, "0.1"),
            (-1.5, "-1.5"),
            (3.14159265, "3.1416"),
            (1.23456789, "1.2346"),
        ):
            with self.subTest(value=repr(value)):
                self.assertEqual(expected, format_number(value))
        # A value whose trailing digits round to zero loses them and its decimal point.
        for value, expected in ((2.50, "2.5"), (1.10, "1.1"), (0.10, "0.1"), (123.0, "123")):
            with self.subTest(value=repr(value)):
                self.assertEqual(expected, format_number(value))

    def test_the_fixed_branch_counts_significant_digits_not_decimal_places(self):
        # The decimal places come from the magnitude, so the same request keeps the digit count
        # as the value grows: five significant digits, whether or not a decimal point appears.
        for value, expected in (
            (1.23456789, "1.2346"),
            (12.3456789, "12.346"),
            (123.456789, "123.46"),
            (1234.56789, "1234.6"),
            (12345.6789, "12346"),
            (-12345.6789, "-12346"),
            (9999.99, "10000"),
        ):
            with self.subTest(value=repr(value)):
                self.assertEqual(expected, format_number(value))
        # Past five digits the precision goes negative and the value rounds to a whole number
        # carrying more digits than were requested.
        for value, expected in ((99999.9, "100000"), (123456.0, "123460"), (1234567.0, "1234600")):
            with self.subTest(value=repr(value)):
                self.assertEqual(expected, format_number(value))

    def test_significant_digits_scale_the_output(self):
        for digits, expected in ((2, "3.1"), (3, "3.14"), (5, "3.1416"), (8, "3.1415927")):
            with self.subTest(digits=digits):
                self.assertEqual(expected, format_number(3.14159265358979, digits))
                # One digit fewer than the request goes to the exponent in the scientific branch.
                self.assertEqual(f"{1e-7:.{digits - 1}e}", format_number(1e-7, digits))

    def test_an_int_wider_than_a_numpy_integer_is_converted_before_formatting(self):
        # format_number hands its argument to numpy, which cannot classify a Python int beyond
        # 64 bits; the int registration converts to float first, so the structure still renders.
        self.assertEqual("{z: 1.1806e+21}", format_dict({"z": 2**70}))

    def test_significant_digits_reach_the_nested_values(self):
        for digits, expected in (
            (2, "{o: {y: 3.1}}"),
            (3, "{o: {y: 3.14}}"),
            (5, "{o: {y: 3.1416}}"),
        ):
            with self.subTest(digits=digits):
                self.assertEqual(
                    expected, format_dict({"o": {"y": 3.14159265358979}}, significant_digits=digits)
                )
        # The request also reaches a sequence element and the scientific branch below it.
        self.assertEqual(
            "{o: [3.14]}", format_dict({"o": [3.14159265358979]}, significant_digits=3)
        )
        self.assertEqual(
            "{o: {y: 1.00e-07}}", format_dict({"o": {"y": 1e-7}}, significant_digits=3)
        )

    def test_a_non_numeric_value_falls_back_to_its_str(self):
        self.assertEqual("abc", format_number("abc"))
        self.assertEqual("1.5", format_number(np.float64(1.5)))
        self.assertEqual("7", format_number(np.int64(7)))

    def test_bools_render_as_words_in_a_structure_and_as_digits_alone(self):
        # bool is an int subclass, so the singledispatch registration is what keeps "True" out of
        # format_number, which would render it as "1".
        ctx = _ctx()
        for value in (True, False, np.bool_(True), np.bool_(False)):
            with self.subTest(value=repr(value)):
                self.assertEqual(str(bool(value)), _format_value(value, ctx, 0))
        self.assertEqual("1", format_number(True))
        self.assertEqual("0", format_number(False))
        self.assertEqual("{a: True, b: False}", format_dict({"a": True, "b": np.bool_(False)}))

    def test_an_empty_mapping_renders_as_the_empty_brace_form_or_an_empty_string(self):
        self.assertEqual("{}", format_dict({}))
        self.assertEqual("", format_dict({}, style="params"))

    def test_the_dict_style_quotes_strings_and_the_params_style_does_not(self):
        payload = {
            "a": 1,
            "b": "txt",
            "c": 1.5,
            "d": True,
            "e": None,
            "f": [1, 2],
            "g": (3,),
            "h": {"k": 2},
        }
        self.assertEqual(
            "{a: 1, b: 'txt', c: 1.5, d: True, e: None, f: [1, 2], g: (3,), h: {k: 2}}",
            format_dict(payload),
        )
        self.assertEqual(
            "a=1, b=txt, c=1.5, d=True, e=None, f=[1, 2], g=(3,), h={k=2}",
            format_dict(payload, style="params"),
        )

    def test_keys_are_never_quoted_or_escaped(self):
        # Only values go through _format_value, in either style and at any nesting depth.
        self.assertEqual(
            "{a b: 1, with'quote: 2, with=eq: 3}",
            format_dict({"a b": 1, "with'quote": 2, "with=eq": 3}),
        )
        self.assertEqual("{a: {b'c: 1}}", format_dict({"a": {"b'c": 1}}))
        self.assertEqual("k={a b=1}", format_dict({"k": {"a b": 1}}, style="params"))

    def test_a_self_referential_mapping_terminates_as_circular(self):
        root = {"name": "root"}
        root["self"] = root
        self.assertEqual(
            "{name: 'root', self: {name: 'root', self: {<circular>}}}", format_dict(root)
        )
        # Wrapped one level deeper the same cycle is found one level earlier in the expansion.
        self.assertEqual("{root: {name: 'root', self: {<circular>}}}", format_dict({"root": root}))

    def test_a_self_referential_sequence_terminates_as_circular(self):
        cyclic = [1]
        cyclic.append(cyclic)
        self.assertEqual("{l: [1, [<circular>]]}", format_dict({"l": cyclic}))

        holder = [1]
        pair = (holder,)
        holder.append(pair)
        self.assertEqual("{l: [1, ([<circular>],)]}", format_dict({"l": holder}))

        mapping = {"a": {}}
        mapping["a"]["back"] = mapping
        self.assertEqual("{a: {back: {a: {<circular>}}}}", format_dict(mapping))

    def test_the_seen_set_tracks_the_current_path_and_not_every_visited_object(self):
        # ctx.seen is added on the way down and discarded on the way back up, so a repeated
        # sibling reference is rendered in full twice while an ancestor reference is not.
        shared = {"k": 1}
        self.assertEqual("{a: {k: 1}, b: {k: 1}}", format_dict({"a": shared, "b": shared}))
        # Identity, not equality, is what seen tracks: two distinct objects with equal contents
        # render exactly as the single repeated object does.
        self.assertEqual(
            format_dict({"a": shared, "b": shared}), format_dict({"a": {"k": 1}, "b": {"k": 1}})
        )
        # A chain of distinct objects terminates on the depth cap, not on identity.
        chain = current = {}
        for index in range(6):
            current["next"] = {"i": index}
            current = current["next"]
        self.assertEqual("{root: {next: {i: 0, next: {...}}}}", format_dict({"root": chain}))

    def test_nesting_is_capped_at_depth_two(self):
        self.assertEqual("{a: {b: {c: {...}}}}", format_dict({"a": {"b": {"c": {"d": 1}}}}))
        self.assertEqual("{a: [[[...]]]}", format_dict({"a": [[["x"]]]}))
        self.assertEqual("{a: (((...),),)}", format_dict({"a": (((1,),),)}))
        # A mapping hits the cap before its emptiness check, a sequence after it, so an empty
        # mapping is elided at depth two where an empty sequence is still spelled out.
        self.assertEqual(
            "{a: {b: [[], {...}, (), set()]}}", format_dict({"a": {"b": [[], {}, (), set()]}})
        )
        for depth in range(2, 5):
            with self.subTest(depth=depth):
                self.assertEqual("{...}", _format_value({"b": 1}, _ctx(), depth))
                self.assertEqual("[...]", _format_value([1], _ctx(), depth))
                self.assertEqual("(...)", _format_value((1,), _ctx(), depth))

    def test_collections_are_capped_at_ten_items_with_an_overflow_count(self):
        payload = {
            "list": list(range(12)),
            "tuple": tuple(range(12)),
            "set": set(range(12)),
            "dict": {f"k{i}": i for i in range(12)},
        }
        for kind, rendered in (
            ("list", "{list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...+2]}"),
            ("tuple", "{tuple: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...+2)}"),
            (
                "dict",
                "{dict: {k0: 0, k1: 1, k2: 2, k3: 3, k4: 4, k5: 5, k6: 6, k7: 7, k8: 8, k9: 9, ...+2}}",
            ),
        ):
            with self.subTest(kind=kind):
                self.assertEqual(rendered, format_dict({kind: payload[kind]}))
        # The count is the overflow, not the total, and a full set of exactly ten items has none.
        self.assertEqual(
            "{s: {0, 1, 10, 2, 3, 4, 5, 6, 7, 8, ...+1}}", format_dict({"s": set(range(11))})
        )
        self.assertEqual("{l: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}", format_dict({"l": list(range(10))}))
        self.assertEqual(
            "{l: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...+1]}", format_dict({"l": list(range(11))})
        )

    def test_the_mapping_passed_to_format_dict_is_rendered_in_full(self):
        # format_dict joins d.items() itself, so the ten-item cap reaches only nested mappings.
        keys = [f"k{i}" for i in range(25)]
        self.assertEqual(25, len(format_dict(dict.fromkeys(keys, 1)).split(", ")))
        capped = format_dict({"k": list(range(15))})
        self.assertEqual("{k: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...+5]}", capped)
        self.assertEqual(10, len(capped.split(", ")) - 1)

    def test_a_set_renders_its_items_in_string_order(self):
        words = {"delta", "alpha", "charlie", "bravo", "echo", "foxtrot", "golf", "hotel", "india"}
        self.assertEqual(
            "{s: {'alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india'}}",
            format_dict({"s": words}),
        )
        # String order, not numeric order: "10" precedes "2".
        self.assertEqual(
            "{s: {10, 2, 33, 4, 55, 6, 77, 8, 99}}",
            format_dict({"s": {10, 2, 33, 4, 55, 6, 77, 8, 99}}),
        )
        self.assertEqual(
            "{s: {(3, 4), 1, 2.5, 'a', 'b'}}", format_dict({"s": {"b", 1, "a", 2.5, (3, 4)}})
        )
        # The cap keeps the first ten in that order, which is what makes the overflow count stable.
        twelve = {f"s{i}" for i in range(12)}
        expected = ["s0", "s1", "s10", "s11", "s2", "s3", "s4", "s5", "s6", "s7"]
        self.assertEqual(
            "{s: {" + ", ".join(f"'{item}'" for item in expected) + ", ...+2}}",
            format_dict({"s": twelve}),
        )
        self.assertEqual(expected, sorted(twelve, key=str)[:10])

    def test_backslashes_and_control_characters_are_escaped(self):
        raw = "a\\b\nc\rd\te"
        quoted = format_dict({"s": raw})
        self.assertEqual(r"{s: 'a\\b\nc\rd\te'}", quoted)
        self.assertEqual(r"s=a\\b\nc\rd\te", format_dict({"s": raw}, style="params"))
        for control in ("\n", "\r", "\t"):
            with self.subTest(control=repr(control)):
                self.assertNotIn(control, format_dict({"s": f"x{control}y"}))
                self.assertEqual(1, format_dict({"s": f"x{control}y"}).count("\\"))

    def test_a_single_quote_is_escaped_only_where_strings_are_quoted(self):
        # The dict style wraps in quotes and so escapes the inner ones; params leaves both bare.
        self.assertEqual(r"{s: 'it\'s a \'test\''}", format_dict({"s": "it's a 'test'"}))
        self.assertEqual(r"{o: {s: 'it\'s'}}", format_dict({"o": {"s": "it's"}}))
        self.assertEqual("s=it's a 'test'", format_dict({"s": "it's a 'test'"}, style="params"))

    def test_long_strings_are_truncated_after_escaping(self):
        self.assertEqual("{s: '" + "x" * 50 + "'}", format_dict({"s": "x" * 50}))
        self.assertEqual("{s: '" + "x" * 50 + "...'}", format_dict({"s": "x" * 51}))
        # Escaping runs first, so fifty backslashes are a hundred characters and are cut.
        self.assertEqual("{s: '" + "\\" * 50 + "...'}", format_dict({"s": "\\" * 50}))
        # Quote escaping runs last, so a truncated string of quotes loses its tail to the cap.
        self.assertEqual("{s: '" + "\\'" * 50 + "...'}", format_dict({"s": "'" * 60}))

    def test_a_single_element_tuple_keeps_its_trailing_comma(self):
        self.assertEqual("{t: (1,), l: [1], s: {1}}", format_dict({"t": (1,), "l": [1], "s": {1}}))
        self.assertEqual("{t: 1}", format_dict({"t": 1}))
        self.assertEqual("{t: (1, 2)}", format_dict({"t": (1, 2)}))

    def test_an_unregistered_type_falls_back_to_its_repr(self):
        class Unregistered:
            def __repr__(self) -> str:
                return "<unregistered>"

        self.assertEqual("{a: <unregistered>}", format_dict({"a": Unregistered()}))
        self.assertEqual(
            "{a: frozenset({1}), b: bytearray(b'x')}",
            format_dict({"a": frozenset({1}), "b": bytearray(b"x")}),
        )

    def test_an_array_renders_as_its_shape(self):
        self.assertEqual(
            "{a: array(3, 4), b: array(2,)}",
            format_dict({"a": np.zeros((3, 4)), "b": np.array([1.0, 2.0])}),
        )

    def test_repeated_renders_of_the_same_structure_are_byte_identical(self):
        payload = {
            "s": {"delta", "alpha", "bravo", "echo", "foxtrot"},
            "i": {10, 2, 33, 4, 55},
            "l": [1.25, None, True, (2,), {"a"}],
            "n": {"x": {"y": 2.5}},
            "a": np.zeros((2, 2)),
        }
        for style in ("dict", "params"):
            with self.subTest(style=style):
                expected = format_dict(payload, style=style)
                self.assertTrue(
                    all(format_dict(payload, style=style) == expected for _ in range(100))
                )
        self.assertEqual("{n: {x: {y: 2.5}}}", format_dict({"n": payload["n"]}))

    def test_the_output_does_not_depend_on_the_interpreter_hash_seed(self):
        # Sets iterate in hash order, so this is the property that makes the log line stable
        # across restarts. A subprocess is the only way to vary PYTHONHASHSEED, which is read
        # once at interpreter start.
        payload = {
            "s": {
                "delta",
                "alpha",
                "charlie",
                "bravo",
                "echo",
                "foxtrot",
                "golf",
                "hotel",
                "india",
            },
            "i": {10, 2, 33, 4, 55, 6, 77, 8, 99},
            "t": {(1, 2), frozenset({3}), "z"},
            "l": ["b", "a", "c"],
        }
        script = f"from Utils import format_dict\nprint(format_dict({payload!r}, style='params'))\n"
        rendered = set()
        for seed in ("0", "1", "12345", "999"):
            with self.subTest(seed=seed):
                proc = subprocess.run(
                    [sys.executable, "-c", script],
                    capture_output=True,
                    text=True,
                    check=True,
                    env=dict(os.environ, PYTHONHASHSEED=seed),
                )
                rendered.add(proc.stdout.strip())
        self.assertEqual(1, len(rendered))


if __name__ == "__main__":
    unittest.main()
