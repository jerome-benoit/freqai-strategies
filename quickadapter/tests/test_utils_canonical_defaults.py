"""Canonical defaults contract between every `*_SPECS` map, its `DEFAULTS_*` map and the per-label overlay; requires the Freqtrade QA image."""

import inspect
import logging
from typing import Any
from unittest import mock

import Utils
from LabelTransformer import (
    DEFAULTS_LABEL_PIPELINE,
    DEFAULTS_LABEL_PREDICTION,
    DEFAULTS_LABEL_SMOOTHING,
    DEFAULTS_LABEL_WEIGHTING,
    SMOOTHING_METHOD_MODES,
    SMOOTHING_METHODS,
    SMOOTHING_MODES,
    get_label_column_config,
)
from qa_support import QaTestCase, ohlcv_frame
from Utils import (
    _ParamSpec,
    _validate_params,
    generate_label_data,
    get_label_kind_config,
    get_label_pipeline_config,
    get_label_prediction_config,
    get_label_smoothing_config,
    get_label_weighting_config,
)

LOGGER = logging.getLogger("test-canonical-defaults")

LABEL_COLUMN = "&s-extrema"
CONFIG_NAME = "label_smoothing"

# The nine (specs, defaults) pairs whose key sets are equal. `_FIT_LIVE_PREDICTIONS_SPECS`
# and `_REVERSAL_CONFIRMATION_SCALAR_SPECS` are deliberately not here: their defaults are
# an inline literal and a documented strict subset, asserted separately below.
EXACT_PAIRS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("_WEIGHTING_SPECS", DEFAULTS_LABEL_WEIGHTING),
    ("_PIPELINE_SPECS", DEFAULTS_LABEL_PIPELINE),
    ("_SMOOTHING_SPECS", DEFAULTS_LABEL_SMOOTHING),
    ("_PREDICTION_SPECS", DEFAULTS_LABEL_PREDICTION),
    ("_EXIT_PRICING_SPECS", Utils.DEFAULTS_EXIT_PRICING),
    ("_CUSTOM_PROTECTIONS_SPECS", Utils.DEFAULTS_CUSTOM_PROTECTIONS),
    ("_COOLDOWN_PROTECTION_SPECS", Utils.DEFAULTS_COOLDOWN_PROTECTION),
    ("_DRAWDOWN_PROTECTION_SPECS", Utils.DEFAULTS_DRAWDOWN_PROTECTION),
    ("_STOPLOSS_PROTECTION_SPECS", Utils.DEFAULTS_STOPLOSS_PROTECTION),
)

LABEL_KINDS: tuple[tuple[str, Any, dict[str, Any]], ...] = (
    ("label_weighting", get_label_weighting_config, DEFAULTS_LABEL_WEIGHTING),
    ("label_pipeline", get_label_pipeline_config, DEFAULTS_LABEL_PIPELINE),
    ("label_smoothing", get_label_smoothing_config, DEFAULTS_LABEL_SMOOTHING),
    ("label_prediction", get_label_prediction_config, DEFAULTS_LABEL_PREDICTION),
)

# One rejected value per kind: each is refused by its own spec, so the accessor must
# substitute the shipped default rather than pass the value through.
INVALID_VALUES: dict[str, tuple[str, Any]] = {
    "label_weighting": ("softmax_temperature", -1),
    "label_pipeline": ("gamma", 0),
    "label_smoothing": ("window_candles", 5.0),
    "label_prediction": ("method", "bogus"),
}

# One value per kind that its spec accepts but its `output_type` rewrites.
COERCED_VALUES: dict[str, tuple[str, Any, Any, type]] = {
    "label_weighting": ("fill_sigma_candles", 25, 25.0, float),
    "label_pipeline": ("robust_quantiles", [0.1, 0.9], (0.1, 0.9), tuple),
    "label_smoothing": ("beta", 1, 1.0, float),
    "label_prediction": ("keep_fraction", 1, 1.0, float),
}


def specs_map(name: str) -> dict[str, _ParamSpec]:
    return getattr(Utils, name)


def discovered_specs() -> list[str]:
    return sorted(
        name
        for name, value in vars(Utils).items()
        if name.endswith("_SPECS")
        and isinstance(value, dict)
        and all(isinstance(spec, _ParamSpec) for spec in value.values())
    )


def smoothing_mode_mode_message(method: str, mode: str, config_name: str = CONFIG_NAME) -> str:
    return (
        f"Invalid {config_name} mode value {mode!r} for method {method!r}: "
        f"supported values are {', '.join(SMOOTHING_METHOD_MODES[method])}"
    )


class UtilsCanonicalDefaultsTest(QaTestCase):
    def test_every_specs_map_in_the_module_is_accounted_for(self):
        expected = sorted(
            [name for name, _ in EXACT_PAIRS]
            + ["_FIT_LIVE_PREDICTIONS_SPECS", "_REVERSAL_CONFIRMATION_SCALAR_SPECS"]
        )
        self.assertEqual(discovered_specs(), expected)

    def test_every_zigzag_entry_point_shares_the_canonical_natr_defaults(self):
        # One tunable pair was written down three times with no canonical source: the
        # params.get fallbacks in _generate_extrema_label and the two _zigzag/zigzag
        # signatures. All three now derive from the constants, so "change the default
        # once" is true; a per-site pin would only have been a fourth copy to drift.
        for function in (Utils._zigzag, Utils.zigzag):
            with self.subTest(function=function.__name__):
                parameters = inspect.signature(function).parameters
                self.assertEqual(parameters["natr_period"].default, Utils.DEFAULT_LABEL_NATR_PERIOD)
                self.assertEqual(
                    parameters["natr_multiplier"].default,
                    Utils.DEFAULT_MIN_LABEL_NATR_MULTIPLIER,
                )

    def test_the_params_fallbacks_agree_with_the_canonical_constants(self):
        # The third copy: the generator's own fallback, reached whenever a params dict
        # omits the keys. Exercised through the public entry point.
        with mock.patch.object(Utils, "_zigzag", wraps=Utils._zigzag) as spied:
            generate_label_data(
                ohlcv_frame([7.0] * 20),
                LABEL_COLUMN,
                params={},
                logger=logging.getLogger("t"),
            )
        used = {call.kwargs["natr_period"] for call in spied.call_args_list}
        multipliers = {call.kwargs["natr_multiplier"] for call in spied.call_args_list}
        self.assertEqual(used, {Utils.DEFAULT_LABEL_NATR_PERIOD})
        self.assertEqual(multipliers, {Utils.DEFAULT_MIN_LABEL_NATR_MULTIPLIER})

    def test_a_pair_of_maps_with_disagreeing_keys_would_silently_lose_a_default(self):
        # A key present in one map and absent from the other is the whole failure this
        # guards: `_validate_params` reads `defaults[param]` for every spec key, so a spec
        # key with no default is a KeyError on every load, and a default key with no spec
        # is a tunable no user can ever set.
        for name, defaults in EXACT_PAIRS:
            with self.subTest(specs=name):
                self.assertEqual(sorted(specs_map(name)), sorted(defaults))

    def test_every_shipped_default_satisfies_its_own_validator(self):
        for name, defaults in EXACT_PAIRS:
            specs = specs_map(name)
            for key, value in defaults.items():
                with self.subTest(specs=name, key=key):
                    self.assertTrue(specs[key].validator(value), f"{name}.{key} = {value!r}")

    def test_validating_the_defaults_rewrites_nothing_and_warns_about_nothing(self):
        # A default that failed its own validator would be replaced by itself on every
        # load — a warning per load, and a shipped constant that documents a value the
        # loader can never accept.
        for name, defaults in EXACT_PAIRS:
            with self.subTest(specs=name):
                with self.assertNoLogs(LOGGER, level="WARNING"):
                    validated = _validate_params({}, LOGGER, name, specs_map(name), defaults)
                self.assertEqual(validated, defaults)

    def test_the_scalars_spec_map_covers_all_but_the_coupled_natr_pair(self):
        # `get_reversal_confirmation_config` validates the (min, max) natr pair through
        # `validate_range`, which `_validate_params` cannot express, so the specs map is
        # a documented strict subset rather than an equal one.
        specs = specs_map("_REVERSAL_CONFIRMATION_SCALAR_SPECS")
        defaults = Utils.DEFAULTS_REVERSAL_CONFIRMATION
        self.assertLess(specs.keys(), defaults.keys())
        self.assertEqual(
            sorted(defaults.keys() - specs.keys()),
            ["max_natr_multiplier_fraction", "min_natr_multiplier_fraction"],
        )
        for key, value in defaults.items():
            if key in specs:
                with self.subTest(key=key):
                    self.assertTrue(specs[key].validator(value))

    def test_the_fit_live_predictions_default_is_declared_where_it_is_consumed(self):
        specs = specs_map("_FIT_LIVE_PREDICTIONS_SPECS")
        inline_defaults = {
            "fit_live_predictions_candles": Utils.DEFAULT_FIT_LIVE_PREDICTIONS_CANDLES
        }
        self.assertEqual(sorted(specs), sorted(inline_defaults))
        with self.assertNoLogs(LOGGER, level="WARNING"):
            self.assertEqual(
                _validate_params({}, LOGGER, "freqai", specs, inline_defaults), inline_defaults
            )

    def test_the_label_kind_registry_wires_each_kind_to_its_own_pair(self):
        self.assertEqual(
            sorted(Utils._LABEL_KIND_REGISTRY),
            sorted(kind for kind, _, _ in LABEL_KINDS),
        )
        for kind, _, defaults in LABEL_KINDS:
            specs, registry_defaults, cross_field = Utils._LABEL_KIND_REGISTRY[kind]
            with self.subTest(kind=kind):
                self.assertIs(registry_defaults, defaults)
                self.assertEqual(sorted(specs), sorted(defaults))
                self.assertIs(cross_field is not None, kind == "label_smoothing")

    def test_each_accessor_returns_the_defaults_for_an_empty_config(self):
        for kind, accessor, defaults in LABEL_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(
                    accessor({}, LOGGER),
                    {"default": defaults, "columns": {}},
                )

    def test_an_absent_config_is_silent_and_a_non_mapping_one_warns(self):
        # `as_config_section` treats `None` as "section not supplied" and only warns for a
        # value that is present and not a mapping; both land on the shipped defaults.
        for kind, accessor, defaults in LABEL_KINDS:
            with self.subTest(kind=kind, config=None), self.assertNoLogs(LOGGER, level="WARNING"):
                self.assertEqual(accessor(None, LOGGER)["default"], defaults)
            for config in ("junk", 7, ["junk"]):
                with (
                    self.subTest(kind=kind, config=config),
                    self.assertLogs(LOGGER, level="WARNING"),
                ):
                    self.assertEqual(accessor(config, LOGGER)["default"], defaults)

    def test_each_accessor_substitutes_the_default_for_a_rejected_value(self):
        for kind, accessor, defaults in LABEL_KINDS:
            key, rejected = INVALID_VALUES[kind]
            with self.subTest(kind=kind, key=key):
                with self.assertLogs(LOGGER, level="WARNING") as captured:
                    resolved = accessor({key: rejected}, LOGGER)["default"]
                self.assertEqual(resolved[key], defaults[key])
                self.assertIn(key, "\n".join(captured.output))

    def test_each_accessor_applies_the_specified_output_type(self):
        for kind, accessor, _ in LABEL_KINDS:
            key, supplied, expected, expected_type = COERCED_VALUES[kind]
            with self.subTest(kind=kind, key=key):
                with self.assertNoLogs(LOGGER, level="WARNING"):
                    resolved = accessor({key: supplied}, LOGGER)["default"][key]
                self.assertIs(type(resolved), expected_type)
                self.assertEqual(resolved, expected)

    def test_an_unknown_key_is_dropped_rather_than_carried_through(self):
        for kind, accessor, defaults in LABEL_KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(
                    accessor({"not_a_tunable": 1}, LOGGER)["default"],
                    defaults,
                )

    def test_a_per_label_config_is_reported_as_default_plus_columns(self):
        resolved = get_label_smoothing_config(
            {"default": {"window_candles": 11}, "columns": {LABEL_COLUMN: {"beta": 2.0}}},
            LOGGER,
        )
        self.assertEqual(sorted(resolved), ["columns", "default"])
        self.assertEqual(resolved["default"]["window_candles"], 11)
        self.assertEqual(resolved["default"]["beta"], DEFAULTS_LABEL_SMOOTHING["beta"])
        self.assertEqual(resolved["columns"], {LABEL_COLUMN: {"beta": 2.0}})
        # Sibling flat keys are not merged into either half.
        with self.assertLogs(LOGGER, level="WARNING"):
            flat = get_label_smoothing_config(
                {"default": {}, "columns": {}, "window_candles": 99},
                LOGGER,
            )
        self.assertEqual(flat["default"], DEFAULTS_LABEL_SMOOTHING)

    def test_the_per_label_format_is_detected_by_the_presence_of_either_half(self):
        for config in ({"default": {}}, {"columns": {}}, {"default": {}, "columns": {}}):
            with self.subTest(config=config):
                self.assertEqual(
                    sorted(get_label_kind_config("label_weighting", config, LOGGER)),
                    ["columns", "default"],
                )

    def test_an_unknown_column_key_is_ignored_and_a_non_mapping_half_is_dropped(self):
        with self.assertLogs(LOGGER, level="WARNING") as captured:
            resolved = get_label_smoothing_config(
                {"columns": {LABEL_COLUMN: "nope", "not-a-label": {"not_a_tunable": 1}}},
                LOGGER,
            )
        self.assertEqual(resolved["columns"], {})
        self.assertIn("not_a_tunable", "\n".join(captured.output))

    def test_an_exact_pattern_beats_every_wildcard(self):
        resolved = get_label_column_config(
            LABEL_COLUMN,
            {"window_candles": 5},
            {
                "*": {"window_candles": 1},
                "&s-*": {"window_candles": 2},
                "&s-ext*": {"window_candles": 3},
                LABEL_COLUMN: {"window_candles": 4},
            },
        )
        self.assertEqual(resolved["window_candles"], 4)

    def test_wildcards_are_applied_in_ascending_specificity(self):
        # Specificity is the count of non-glob characters, and the overlays are folded in
        # from least to most specific, so the most specific wildcard has the last word.
        self.assertEqual(
            get_label_column_config(
                LABEL_COLUMN,
                {"window_candles": 5},
                {
                    "&s-ext*": {"window_candles": 3},
                    "&s-*": {"window_candles": 2},
                    "*": {"window_candles": 1},
                },
            ),
            {"window_candles": 3},
        )
        self.assertEqual(
            get_label_column_config(
                LABEL_COLUMN,
                {"window_candles": 5},
                {"&s-*": {"window_candles": 2}, "*": {"window_candles": 1}},
            ),
            {"window_candles": 2},
        )

    def test_bracket_and_question_globs_are_wildcards_not_exact_patterns(self):
        # `&s-extrem?` and `&s-[ex]xtrema` both match, but neither is exact, so the truly
        # exact pattern must still win.
        self.assertEqual(
            get_label_column_config(
                LABEL_COLUMN,
                {"window_candles": 5},
                {
                    "&s-extrem?": {"window_candles": 1},
                    "&s-[ex]xtrema": {"window_candles": 2},
                    LABEL_COLUMN: {"window_candles": 3},
                },
            ),
            {"window_candles": 3},
        )
        # Against each other they rank by literal characters: 9 for `&s-extrem?`, 7 for
        # `&s-ext*`.
        self.assertEqual(
            get_label_column_config(
                LABEL_COLUMN,
                {"window_candles": 5},
                {"&s-ext*": {"window_candles": 1}, "&s-extrem?": {"window_candles": 2}},
            ),
            {"window_candles": 2},
        )

    def test_patterns_of_equal_specificity_resolve_by_declaration_order(self):
        # `&s-e*` and `&s-*a` each contribute three literal characters, so neither outranks
        # the other: the stable sort keeps declaration order and the later overlay is folded
        # in last, so the two orderings of the same pairs disagree.
        first = {"&s-e*": {"w": 1}, "&s-*a": {"w": 2}}
        second = {"&s-*a": {"w": 2}, "&s-e*": {"w": 1}}
        self.assertEqual(get_label_column_config(LABEL_COLUMN, {"w": 0}, first), {"w": 2})
        self.assertEqual(get_label_column_config(LABEL_COLUMN, {"w": 0}, second), {"w": 1})

    def test_only_matching_patterns_contribute_and_defaults_survive(self):
        self.assertEqual(
            get_label_column_config(
                "&-amplitude",
                {"window_candles": 5, "beta": 8.0},
                {LABEL_COLUMN: {"window_candles": 3}},
            ),
            {"window_candles": 5, "beta": 8.0},
        )
        # A matching pattern overlays only the keys it names.
        self.assertEqual(
            get_label_column_config(
                LABEL_COLUMN,
                {"window_candles": 5, "beta": 8.0},
                {LABEL_COLUMN: {"beta": 2.0}},
            ),
            {"window_candles": 5, "beta": 2.0},
        )

    def test_the_default_config_is_deep_copied_into_the_result(self):
        defaults = {"nested": {"a": [1, 2]}}
        resolved = get_label_column_config(LABEL_COLUMN, defaults, {})
        self.assertIsNot(resolved, defaults)
        self.assertIsNot(resolved["nested"], defaults["nested"])
        resolved["nested"]["a"].append(3)
        resolved["nested"]["a"][0] = 99
        self.assertEqual(defaults, {"nested": {"a": [1, 2]}})

    def test_only_the_smoothing_method_mode_pair_is_cross_field_validated(self):
        unconstrained = [m for m in SMOOTHING_METHODS if m not in SMOOTHING_METHOD_MODES]
        self.assertTrue(unconstrained)
        for method in unconstrained:
            for mode in SMOOTHING_MODES:
                with self.subTest(method=method, mode=mode):
                    self.assertIsNone(
                        Utils._validate_smoothing_method_mode(
                            {"method": method, "mode": mode}, CONFIG_NAME
                        )
                    )

    def test_the_method_mode_table_is_a_non_empty_subset_of_the_declared_modes(self):
        self.assertLessEqual(set(SMOOTHING_METHOD_MODES), set(SMOOTHING_METHODS))
        for method, modes in SMOOTHING_METHOD_MODES.items():
            with self.subTest(method=method):
                self.assertTrue(modes)
                self.assertLessEqual(set(modes), set(SMOOTHING_MODES))

    def test_a_mode_the_method_does_not_support_raises_rather_than_warns(self):
        for method, modes in SMOOTHING_METHOD_MODES.items():
            for mode in SMOOTHING_MODES:
                config = {"method": method, "mode": mode}
                if mode in modes:
                    with self.subTest(method=method, mode=mode):
                        self.assertIsNone(
                            Utils._validate_smoothing_method_mode(config, CONFIG_NAME)
                        )
                else:
                    with self.subTest(method=method, mode=mode):
                        with self.assertRaises(ValueError) as raised:
                            Utils._validate_smoothing_method_mode(config, CONFIG_NAME)
                        self.assertEqual(
                            str(raised.exception), smoothing_mode_mode_message(method, mode)
                        )

    def test_interp_is_valid_for_savgol_and_invalid_for_gaussian_filter1d(self):
        # The only concrete disagreement between the two constrained methods, read off the
        # table rather than assumed.
        self.assertIn("interp", SMOOTHING_METHOD_MODES["savgol"])
        self.assertNotIn("interp", SMOOTHING_METHOD_MODES["gaussian_filter1d"])
        self.assertIsNone(
            Utils._validate_smoothing_method_mode(
                {"method": "savgol", "mode": "interp"}, CONFIG_NAME
            )
        )
        with self.assertRaises(ValueError):
            Utils._validate_smoothing_method_mode(
                {"method": "gaussian_filter1d", "mode": "interp"}, CONFIG_NAME
            )

    def test_the_smoothing_accessor_raises_on_an_inconsistent_default(self):
        with self.assertRaises(ValueError) as raised:
            get_label_smoothing_config({"method": "gaussian_filter1d", "mode": "interp"}, LOGGER)
        self.assertEqual(
            str(raised.exception),
            smoothing_mode_mode_message(
                "gaussian_filter1d",
                "interp",
                f"{CONFIG_NAME} for label {LABEL_COLUMN!r}",
            ),
        )

    def test_the_cross_field_check_reads_the_overlaid_label_config(self):
        # The check runs on `get_label_column_config` applied to each shipped label column,
        # so a per-column override can repair an inconsistent default for that column.
        resolved = get_label_smoothing_config(
            {
                "default": {"method": "gaussian_filter1d", "mode": "interp"},
                "columns": {LABEL_COLUMN: {"mode": "mirror"}},
            },
            LOGGER,
        )
        self.assertEqual(resolved["default"]["method"], "gaussian_filter1d")
        self.assertEqual(resolved["default"]["mode"], "interp")
        self.assertEqual(resolved["columns"], {LABEL_COLUMN: {"mode": "mirror"}})
        # The same inconsistency left standing on the column raises.
        with self.assertRaises(ValueError):
            get_label_smoothing_config(
                {
                    "default": {"method": "gaussian_filter1d", "mode": "interp"},
                    "columns": {LABEL_COLUMN: {"mode": "interp"}},
                },
                LOGGER,
            )
