"""Strategy-facing config resolvers: defaults precedence, warning text, and per-call isolation.; requires the Freqtrade QA image."""

import copy
import logging
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
from qa_support import QaTestCase
from Utils import (
    DEFAULT_FIT_LIVE_PREDICTIONS_CANDLES,
    DEFAULTS_CUSTOM_PROTECTIONS,
    DEFAULTS_EXIT_PRICING,
    DEFAULTS_REVERSAL_CONFIRMATION,
    get_causal_mode,
    get_custom_protections_config,
    get_exit_pricing_config,
    get_fit_live_predictions_candles,
    get_label_horizon_candles,
    get_reversal_confirmation_config,
    normalize_fit_live_predictions_config,
)

CANONICAL_EXIT_PRICING: dict[str, Any] = {
    "trade_natr_method": "moving_average",
    "final_take_profit_retracement_fraction": 0.25,
    "take_profit_stage_fraction_series": "golden_ratio",
}
CANONICAL_PROTECTIONS: dict[str, Any] = {
    "trade_duration_candles": 72,
    "lookback_period_fraction": 0.5,
    "cooldown": {"enabled": True, "stop_duration_candles": 4},
    "drawdown": {"enabled": True, "max_allowed_drawdown": 0.2},
    "stoploss": {"enabled": True},
}
CANONICAL_REVERSAL: dict[str, Any] = {
    "lookback_period_candles": 0,
    "decay_fraction": 0.5,
    "min_natr_multiplier_fraction": 0.0095,
    "max_natr_multiplier_fraction": 0.0125,
}


class _WarningSink(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.NOTSET)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(record.getMessage())


@contextmanager
def recorded_warnings() -> Iterator[tuple[logging.Logger, list[str]]]:
    """Yield a logger and the list its WARNING messages are appended to, call shape: log, sink."""
    sink: list[str] = []
    logger = logging.getLogger("quickadapter.tests.config_resolution")
    handler = _WarningSink(sink)
    previous_propagate = logger.propagate
    previous_level = logger.level
    logger.propagate = False
    # The level is set explicitly AND restored. NOTSET makes the logger inherit the root
    # level, and a record the logger has already filtered out never reaches a private
    # handler — so under a root at ERROR every case asserting on warning text fails on an
    # empty sink. Restoring it on exit matters too: leaving it changed makes later tests
    # depend on ordering.
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield logger, sink
    finally:
        logger.removeHandler(handler)
        logger.propagate = previous_propagate
        logger.setLevel(previous_level)


class UtilsConfigResolutionTest(QaTestCase):
    def test_exit_pricing_absent_section_resolves_to_canonical_defaults(self) -> None:
        for section in (None, {}):
            with self.subTest(section=section):
                with recorded_warnings() as (logger, sink):
                    resolved = get_exit_pricing_config(section, logger)
                self.assertEqual(CANONICAL_EXIT_PRICING, resolved)
                self.assertEqual([], sink)

    def test_exit_pricing_non_mapping_section_warns_with_the_section_name(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_exit_pricing_config(7, logger)
        self.assertEqual(CANONICAL_EXIT_PRICING, resolved)
        self.assertEqual(["Invalid exit_pricing value 7: must be a mapping, using defaults"], sink)

    def test_exit_pricing_unknown_natr_method_falls_back_with_the_enum_message(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_exit_pricing_config({"trade_natr_method": "nope"}, logger)
        self.assertEqual(CANONICAL_EXIT_PRICING, resolved)
        self.assertEqual(
            [
                "Invalid exit_pricing trade_natr_method value 'nope': supported values are "
                "moving_average, quantile_interpolation, weighted_average, "
                "using default 'moving_average'"
            ],
            sink,
        )

    def test_exit_pricing_every_supported_natr_method_is_preserved(self) -> None:
        for method in ("moving_average", "quantile_interpolation", "weighted_average"):
            with self.subTest(method=method):
                with recorded_warnings() as (logger, sink):
                    resolved = get_exit_pricing_config({"trade_natr_method": method}, logger)
                self.assertEqual(method, resolved["trade_natr_method"])
                self.assertEqual([], sink)

    def test_exit_pricing_retracement_fraction_range_is_half_open_at_zero(self) -> None:
        with recorded_warnings() as (logger, sink):
            self.assertEqual(
                1.0,
                get_exit_pricing_config({"final_take_profit_retracement_fraction": 1}, logger)[
                    "final_take_profit_retracement_fraction"
                ],
            )
        self.assertEqual([], sink)

        for rejected in (0, 0.0, -0.5, None, True, "0.25"):
            with self.subTest(rejected=rejected):
                with recorded_warnings() as (logger, sink):
                    resolved = get_exit_pricing_config(
                        {"final_take_profit_retracement_fraction": rejected}, logger
                    )
                self.assertEqual(0.25, resolved["final_take_profit_retracement_fraction"])
                self.assertEqual(
                    [
                        f"Invalid exit_pricing final_take_profit_retracement_fraction value "
                        f"{rejected!r}: must be a finite number > 0 <= 1, using default 0.25"
                    ],
                    sink,
                )

    def test_exit_pricing_returns_a_fresh_dict_each_call(self) -> None:
        with recorded_warnings() as (logger, sink):
            first = get_exit_pricing_config({}, logger)
            second = get_exit_pricing_config({}, logger)
        self.assertEqual([], sink)
        first["trade_natr_method"] = "weighted_average"
        self.assertIsNot(first, second)
        self.assertEqual("moving_average", second["trade_natr_method"])
        self.assertEqual("moving_average", DEFAULTS_EXIT_PRICING["trade_natr_method"])

    def test_custom_protections_absent_section_resolves_to_canonical_defaults(self) -> None:
        for section in (None, {}):
            with self.subTest(section=section):
                with recorded_warnings() as (logger, sink):
                    resolved = get_custom_protections_config(section, logger)
                self.assertEqual(CANONICAL_PROTECTIONS, resolved)
                self.assertEqual([], sink)

    def test_custom_protections_non_mapping_section_warns_with_the_section_name(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_custom_protections_config(7, logger)
        self.assertEqual(CANONICAL_PROTECTIONS, resolved)
        self.assertEqual(
            ["Invalid custom_protections value 7: must be a mapping, using defaults"], sink
        )

    def test_custom_protections_top_level_fields_fall_back_with_their_constraints(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_custom_protections_config(
                {"trade_duration_candles": 0, "lookback_period_fraction": 0}, logger
            )
        self.assertEqual(72, resolved["trade_duration_candles"])
        self.assertEqual(0.5, resolved["lookback_period_fraction"])
        self.assertEqual(
            [
                "Invalid custom_protections trade_duration_candles value 0: "
                "must be an integer >= 1, using default 72",
                "Invalid custom_protections lookback_period_fraction value 0: "
                "must be a finite number > 0 <= 1, using default 0.5",
            ],
            sink,
        )

    def test_custom_protections_top_level_fields_reject_non_integers(self) -> None:
        for rejected in (True, 5.0, "5", None):
            with self.subTest(rejected=rejected):
                with recorded_warnings() as (logger, sink):
                    resolved = get_custom_protections_config(
                        {"trade_duration_candles": rejected}, logger
                    )
                self.assertEqual(72, resolved["trade_duration_candles"])
                self.assertEqual(
                    [
                        f"Invalid custom_protections trade_duration_candles value {rejected!r}: "
                        f"must be an integer >= 1, using default 72"
                    ],
                    sink,
                )

    def test_custom_protections_sub_configs_fall_back_independently(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_custom_protections_config(
                {
                    "cooldown": {"enabled": "yes", "stop_duration_candles": 0},
                    "drawdown": {"enabled": 1, "max_allowed_drawdown": 0.2},
                    "stoploss": {"enabled": None},
                },
                logger,
            )
        self.assertEqual({"enabled": True, "stop_duration_candles": 4}, resolved["cooldown"])
        self.assertEqual({"enabled": True, "max_allowed_drawdown": 0.2}, resolved["drawdown"])
        self.assertEqual({"enabled": True}, resolved["stoploss"])
        self.assertEqual(
            [
                "Invalid custom_protections.cooldown enabled value 'yes': "
                "must be a boolean, using default True",
                "Invalid custom_protections.cooldown stop_duration_candles value 0: "
                "must be an integer >= 1, using default 4",
                "Invalid custom_protections.drawdown enabled value 1: "
                "must be a boolean, using default True",
                "Invalid custom_protections.stoploss enabled value None: "
                "must be a boolean, using default True",
            ],
            sink,
        )

    def test_custom_protections_scalar_sub_sections_fall_back_to_their_own_defaults(self) -> None:
        for sub_name in ("cooldown", "drawdown", "stoploss"):
            with self.subTest(sub_name=sub_name):
                with recorded_warnings() as (logger, sink):
                    resolved = get_custom_protections_config({sub_name: 3}, logger)
                self.assertEqual(CANONICAL_PROTECTIONS[sub_name], resolved[sub_name])
                self.assertEqual(
                    [
                        f"Invalid custom_protections.{sub_name} value 3: "
                        f"must be a mapping, using defaults"
                    ],
                    sink,
                )

    def test_custom_protections_drawdown_range_is_open_at_both_ends(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_custom_protections_config(
                {"drawdown": {"max_allowed_drawdown": 0.999}}, logger
            )
        self.assertEqual(0.999, resolved["drawdown"]["max_allowed_drawdown"])
        self.assertEqual([], sink)

        for rejected in (0, 0.0, 1.0, 1.5, -0.1, True, "0.2"):
            with self.subTest(rejected=rejected):
                with recorded_warnings() as (logger, sink):
                    resolved = get_custom_protections_config(
                        {"drawdown": {"max_allowed_drawdown": rejected}}, logger
                    )
                self.assertEqual(0.2, resolved["drawdown"]["max_allowed_drawdown"])
                self.assertEqual(
                    [
                        f"Invalid custom_protections.drawdown max_allowed_drawdown value "
                        f"{rejected!r}: must be a finite number > 0 < 1, using default 0.2"
                    ],
                    sink,
                )

    def test_custom_protections_sub_configs_can_be_disabled_independently(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_custom_protections_config(
                {
                    "cooldown": {"enabled": False},
                    "drawdown": {"enabled": False},
                    "stoploss": {"enabled": False},
                },
                logger,
            )
        self.assertEqual([], sink)
        self.assertEqual(
            {
                "trade_duration_candles": 72,
                "lookback_period_fraction": 0.5,
                "cooldown": {"enabled": False, "stop_duration_candles": 4},
                "drawdown": {"enabled": False, "max_allowed_drawdown": 0.2},
                "stoploss": {"enabled": False},
            },
            resolved,
        )

    def test_custom_protections_returns_fresh_nested_dicts_each_call(self) -> None:
        with recorded_warnings() as (logger, sink):
            first = get_custom_protections_config({}, logger)
            second = get_custom_protections_config({}, logger)
        self.assertEqual([], sink)
        first["cooldown"]["enabled"] = "leaked"
        first["drawdown"]["max_allowed_drawdown"] = 0.99
        self.assertIsNot(first, second)
        self.assertIsNot(first["cooldown"], second["cooldown"])
        self.assertEqual({"enabled": True, "stop_duration_candles": 4}, second["cooldown"])
        self.assertEqual(0.2, second["drawdown"]["max_allowed_drawdown"])
        self.assertEqual(
            {"trade_duration_candles": 72, "lookback_period_fraction": 0.5},
            DEFAULTS_CUSTOM_PROTECTIONS,
        )

    def test_fit_live_predictions_candles_defaults_when_the_key_is_absent(self) -> None:
        for section in (None, {}, {"enabled": True}):
            with self.subTest(section=section):
                with recorded_warnings() as (logger, sink):
                    resolved = get_fit_live_predictions_candles(section, logger)
                self.assertEqual(DEFAULT_FIT_LIVE_PREDICTIONS_CANDLES, resolved)
                self.assertEqual(100, resolved)
                self.assertEqual([], sink)

    def test_fit_live_predictions_candles_preserves_a_positive_integer(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_fit_live_predictions_candles(
                {"fit_live_predictions_candles": 250}, logger
            )
        self.assertEqual(250, resolved)
        self.assertIsInstance(resolved, int)
        self.assertEqual([], sink)

    def test_fit_live_predictions_candles_rejects_every_non_integer_spelling(self) -> None:
        for rejected in (0, -1, 250.0, np.int64(250), True, "250", None, [250]):
            with self.subTest(rejected=rejected):
                with recorded_warnings() as (logger, sink):
                    resolved = get_fit_live_predictions_candles(
                        {"fit_live_predictions_candles": rejected}, logger
                    )
                self.assertEqual(100, resolved)
                self.assertEqual(
                    [
                        f"Invalid freqai fit_live_predictions_candles value {rejected!r}: "
                        f"must be an integer >= 1, using default 100"
                    ],
                    sink,
                )

    def test_fit_live_predictions_candles_non_mapping_section_warns_once(self) -> None:
        for section in (7, [1]):
            with self.subTest(section=section):
                with recorded_warnings() as (logger, sink):
                    resolved = get_fit_live_predictions_candles(section, logger)
                self.assertEqual(100, resolved)
                self.assertEqual(
                    [f"Invalid freqai value {section!r}: must be a mapping, using defaults"], sink
                )

    def test_normalize_is_a_silent_no_op_without_a_nonempty_freqai_section(self) -> None:
        for config in ({}, {"freqai": {}}, {"freqai": None}, {"freqai": 42}, {"freqai": [1]}):
            with self.subTest(config=config):
                snapshot = copy.deepcopy(config)
                with recorded_warnings() as (logger, sink):
                    returned = normalize_fit_live_predictions_config(config, logger)
                self.assertIsNone(returned)
                self.assertEqual([], sink)
                self.assertEqual(snapshot, config)

    def test_normalize_rewrites_a_bare_int_value_in_place(self) -> None:
        config = {"freqai": {"fit_live_predictions_candles": 250}}
        with recorded_warnings() as (logger, sink):
            returned = normalize_fit_live_predictions_config(config, logger)
        self.assertIsNone(returned)
        self.assertEqual([], sink)
        self.assertEqual(250, config["freqai"]["fit_live_predictions_candles"])
        self.assertIsInstance(config["freqai"]["fit_live_predictions_candles"], int)

    def test_normalize_adds_the_key_when_the_freqai_section_omits_it(self) -> None:
        config = {"freqai": {"enabled": True}}
        with recorded_warnings() as (logger, sink):
            normalize_fit_live_predictions_config(config, logger)
        self.assertEqual({"enabled": True, "fit_live_predictions_candles": 100}, config["freqai"])
        self.assertEqual([], sink)

    def test_normalize_replaces_an_invalid_value_and_warns(self) -> None:
        for rejected in ("250", 0, 250.0, np.int64(250), True):
            with self.subTest(rejected=rejected):
                config = {"freqai": {"fit_live_predictions_candles": rejected}}
                with recorded_warnings() as (logger, sink):
                    normalize_fit_live_predictions_config(config, logger)
                self.assertEqual(100, config["freqai"]["fit_live_predictions_candles"])
                self.assertEqual(
                    [
                        f"Invalid freqai fit_live_predictions_candles value {rejected!r}: "
                        f"must be an integer >= 1, using default 100"
                    ],
                    sink,
                )

    def test_reversal_confirmation_absent_section_resolves_to_canonical_defaults(self) -> None:
        for section in (None, {}):
            with self.subTest(section=section):
                with recorded_warnings() as (logger, sink):
                    resolved = get_reversal_confirmation_config(section, logger)
                self.assertEqual(CANONICAL_REVERSAL, resolved)
                self.assertEqual([], sink)

    def test_reversal_confirmation_non_mapping_section_warns_with_the_section_name(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_reversal_confirmation_config("x", logger)
        self.assertEqual(CANONICAL_REVERSAL, resolved)
        self.assertEqual(
            ["Invalid reversal_confirmation value 'x': must be a mapping, using defaults"], sink
        )

    def test_reversal_confirmation_scalar_fields_fall_back_with_their_constraints(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_reversal_confirmation_config(
                {"lookback_period_candles": -1, "decay_fraction": 0}, logger
            )
        self.assertEqual(0, resolved["lookback_period_candles"])
        self.assertEqual(0.5, resolved["decay_fraction"])
        self.assertEqual(
            [
                "Invalid reversal_confirmation lookback_period_candles value -1: "
                "must be an integer >= 0, using default 0",
                "Invalid reversal_confirmation decay_fraction value 0: "
                "must be a finite number > 0 <= 1, using default 0.5",
            ],
            sink,
        )

    def test_reversal_confirmation_scalar_field_bounds(self) -> None:
        for value in (0, 10):
            with self.subTest(value=value):
                with recorded_warnings() as (logger, sink):
                    resolved = get_reversal_confirmation_config(
                        {"lookback_period_candles": value}, logger
                    )
                self.assertEqual(value, resolved["lookback_period_candles"])
                self.assertEqual([], sink)

        for rejected in (True, 10.0, np.int64(10), "10", None):
            with self.subTest(rejected=rejected):
                with recorded_warnings() as (logger, sink):
                    resolved = get_reversal_confirmation_config(
                        {"lookback_period_candles": rejected}, logger
                    )
                self.assertEqual(0, resolved["lookback_period_candles"])
                self.assertEqual(
                    [
                        f"Invalid reversal_confirmation lookback_period_candles value "
                        f"{rejected!r}: must be an integer >= 0, using default 0"
                    ],
                    sink,
                )

    def test_reversal_confirmation_decay_fraction_range_is_half_open_at_zero(self) -> None:
        for value in (1, 0.25):
            with self.subTest(value=value):
                with recorded_warnings() as (logger, sink):
                    resolved = get_reversal_confirmation_config({"decay_fraction": value}, logger)
                self.assertEqual(float(value), resolved["decay_fraction"])
                self.assertEqual([], sink)

        for rejected in (0, 0.0, -0.1, 1.01, True, None):
            with self.subTest(rejected=rejected):
                with recorded_warnings() as (logger, sink):
                    resolved = get_reversal_confirmation_config(
                        {"decay_fraction": rejected}, logger
                    )
                self.assertEqual(0.5, resolved["decay_fraction"])
                self.assertEqual(
                    [
                        f"Invalid reversal_confirmation decay_fraction value {rejected!r}: "
                        f"must be a finite number > 0 <= 1, using default 0.5"
                    ],
                    sink,
                )

    def test_reversal_confirmation_inverted_pair_falls_back_to_both_defaults(self) -> None:
        for pair in ((0.05, 0.01), (0.01, 0.01), (0.005, 0.0), (0.0095, 0.0)):
            with self.subTest(pair=pair):
                supplied_min, supplied_max = pair
                with recorded_warnings() as (logger, sink):
                    resolved = get_reversal_confirmation_config(
                        {
                            "min_natr_multiplier_fraction": supplied_min,
                            "max_natr_multiplier_fraction": supplied_max,
                        },
                        logger,
                    )
                self.assertEqual(0.0095, resolved["min_natr_multiplier_fraction"])
                self.assertEqual(0.0125, resolved["max_natr_multiplier_fraction"])
                self.assertEqual(
                    [
                        "Invalid natr_multiplier_fraction ordering: must have "
                        "min_natr_multiplier_fraction < max_natr_multiplier_fraction, "
                        f"got min_natr_multiplier_fraction={supplied_min!r}, "
                        f"max_natr_multiplier_fraction={supplied_max!r}, "
                        "using defaults 0.0095, 0.0125",
                        "Invalid natr_multiplier_fraction range "
                        f"(min_natr_multiplier_fraction={supplied_min!r}, "
                        f"max_natr_multiplier_fraction={supplied_max!r}), "
                        "using (0.0095, 0.0125)",
                    ],
                    sink,
                )

    def test_reversal_confirmation_one_invalid_bound_keeps_the_other(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_reversal_confirmation_config(
                {"min_natr_multiplier_fraction": -1, "max_natr_multiplier_fraction": 0.02}, logger
            )
        self.assertEqual(0.0095, resolved["min_natr_multiplier_fraction"])
        self.assertEqual(0.02, resolved["max_natr_multiplier_fraction"])
        self.assertEqual(
            [
                "Invalid min_natr_multiplier_fraction -1: "
                "must be finite non-negative numeric <= 1, using default 0.0095",
                "Invalid natr_multiplier_fraction range "
                "(min_natr_multiplier_fraction=-1, max_natr_multiplier_fraction=0.02), "
                "using (0.0095, 0.02)",
            ],
            sink,
        )

    def test_reversal_confirmation_invalid_upper_bound_keeps_the_lower(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_reversal_confirmation_config(
                {"min_natr_multiplier_fraction": 0.01, "max_natr_multiplier_fraction": "x"}, logger
            )
        self.assertEqual(0.01, resolved["min_natr_multiplier_fraction"])
        self.assertEqual(0.0125, resolved["max_natr_multiplier_fraction"])
        self.assertEqual(
            [
                "Invalid max_natr_multiplier_fraction 'x': "
                "must be finite non-negative numeric <= 1, using default 0.0125",
                "Invalid natr_multiplier_fraction range "
                "(min_natr_multiplier_fraction=0.01, max_natr_multiplier_fraction='x'), "
                "using (0.01, 0.0125)",
            ],
            sink,
        )

    def test_reversal_confirmation_non_finite_and_numpy_bounds_fall_back(self) -> None:
        for rejected in (float("nan"), float("inf"), np.int64(1), True):
            with self.subTest(rejected=rejected):
                with recorded_warnings() as (logger, sink):
                    resolved = get_reversal_confirmation_config(
                        {"min_natr_multiplier_fraction": rejected}, logger
                    )
                self.assertEqual(0.0095, resolved["min_natr_multiplier_fraction"])
                self.assertEqual(0.0125, resolved["max_natr_multiplier_fraction"])
                self.assertEqual(
                    [
                        f"Invalid min_natr_multiplier_fraction {rejected!r}: "
                        f"must be finite non-negative numeric <= 1, using default 0.0095",
                        "Invalid natr_multiplier_fraction range "
                        f"(min_natr_multiplier_fraction={rejected!r}, "
                        "max_natr_multiplier_fraction=0.0125), using (0.0095, 0.0125)",
                    ],
                    sink,
                )

    def test_reversal_confirmation_upper_bound_is_capped_at_one(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_reversal_confirmation_config({"max_natr_multiplier_fraction": 1}, logger)
        self.assertEqual(1.0, resolved["max_natr_multiplier_fraction"])
        self.assertEqual([], sink)

        for rejected in (1.0001, 1.5):
            with self.subTest(rejected=rejected):
                with recorded_warnings() as (logger, sink):
                    resolved = get_reversal_confirmation_config(
                        {"max_natr_multiplier_fraction": rejected}, logger
                    )
                self.assertEqual(0.0125, resolved["max_natr_multiplier_fraction"])
                self.assertEqual(
                    [
                        f"Invalid max_natr_multiplier_fraction {rejected!r}: "
                        f"must be finite non-negative numeric <= 1, using default 0.0125",
                        "Invalid natr_multiplier_fraction range "
                        "(min_natr_multiplier_fraction=0.0095, "
                        f"max_natr_multiplier_fraction={rejected!r}), using (0.0095, 0.0125)",
                    ],
                    sink,
                )

    def test_reversal_confirmation_returns_a_fresh_dict_each_call(self) -> None:
        with recorded_warnings() as (logger, sink):
            first = get_reversal_confirmation_config({}, logger)
            second = get_reversal_confirmation_config({}, logger)
        self.assertEqual([], sink)
        first["decay_fraction"] = 0.99
        self.assertIsNot(first, second)
        self.assertEqual(0.5, second["decay_fraction"])
        self.assertEqual(0.5, DEFAULTS_REVERSAL_CONFIRMATION["decay_fraction"])

    def test_causal_mode_defaults_to_true_when_the_key_is_absent(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_causal_mode({}, logger)
        self.assertIs(True, resolved)
        self.assertEqual([], sink)

    def test_causal_mode_honours_an_explicit_boolean_including_false(self) -> None:
        for supplied, expected in ((True, True), (False, False)):
            with self.subTest(supplied=supplied):
                with recorded_warnings() as (logger, sink):
                    resolved = get_causal_mode({"causal_mode": supplied}, logger)
                self.assertIs(expected, resolved)
                self.assertEqual([], sink)

    def test_causal_mode_falls_back_to_true_for_every_non_boolean(self) -> None:
        for rejected in ("false", "True", 0, 1, 0.0, None, [], {}, np.True_):
            with self.subTest(rejected=rejected):
                with recorded_warnings() as (logger, sink):
                    resolved = get_causal_mode({"causal_mode": rejected}, logger)
                self.assertIs(True, resolved)
                self.assertEqual(
                    [f"Invalid causal_mode value {rejected!r}: must be bool, using True"], sink
                )

    def test_causal_mode_falsy_non_boolean_does_not_disable_the_purge(self) -> None:
        for rejected in ("", 0, 0.0, [], {}, None):
            with self.subTest(rejected=rejected):
                with recorded_warnings() as (logger, sink):
                    self.assertIs(True, get_causal_mode({"causal_mode": rejected}, logger))
                self.assertEqual(1, len(sink))

    def test_label_horizon_defaults_to_one_for_an_empty_config(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_label_horizon_candles({}, logger)
        self.assertEqual(1, resolved)
        self.assertIsInstance(resolved, int)
        self.assertEqual([], sink)

    def test_label_horizon_uses_its_own_value_when_a_positive_int(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_label_horizon_candles(
                {"label_period_candles": 12, "label_horizon_candles": 5}, logger
            )
        self.assertEqual(5, resolved)
        self.assertIsInstance(resolved, int)
        self.assertEqual([], sink)

    def test_label_horizon_falls_back_to_label_period_candles(self) -> None:
        with recorded_warnings() as (logger, sink):
            resolved = get_label_horizon_candles({"label_period_candles": 12}, logger)
        self.assertEqual(12, resolved)
        self.assertEqual([], sink)

    def test_label_period_candles_is_clamped_to_at_least_one(self) -> None:
        for rejected in (0, -3, "12", 12.0, True, None, [], np.int64(0)):
            with self.subTest(rejected=rejected):
                with recorded_warnings() as (logger, sink):
                    resolved = get_label_horizon_candles({"label_period_candles": rejected}, logger)
                self.assertEqual(1, resolved)
                self.assertEqual([], sink)

    def test_label_horizon_invalid_value_warns_and_uses_the_label_period(self) -> None:
        for rejected in (0, -1, 5.0, True, "5", None, []):
            with self.subTest(rejected=rejected):
                with recorded_warnings() as (logger, sink):
                    resolved = get_label_horizon_candles(
                        {"label_period_candles": 12, "label_horizon_candles": rejected}, logger
                    )
                self.assertEqual(12, resolved)
                self.assertEqual(
                    [
                        f"Invalid label_horizon_candles value {rejected!r}: "
                        f"must be int >= 1, using 12"
                    ],
                    sink,
                )

    def test_label_horizon_both_invalid_yields_one_and_names_the_fallback(self) -> None:
        for config in (
            {"label_period_candles": 0, "label_horizon_candles": -1},
            {"label_period_candles": 0, "label_horizon_candles": 0},
        ):
            with self.subTest(config=config):
                with recorded_warnings() as (logger, sink):
                    resolved = get_label_horizon_candles(config, logger)
                self.assertEqual(1, resolved)
                self.assertEqual(
                    [
                        f"Invalid label_horizon_candles value "
                        f"{config['label_horizon_candles']!r}: must be int >= 1, using 1"
                    ],
                    sink,
                )

    def test_label_horizon_accepts_a_numpy_integer_period(self) -> None:
        with recorded_warnings() as (logger, sink):
            self.assertEqual(
                9, get_label_horizon_candles({"label_period_candles": np.int64(9)}, logger)
            )
            self.assertEqual(
                7,
                get_label_horizon_candles(
                    {"label_period_candles": np.int64(9), "label_horizon_candles": np.int64(7)},
                    logger,
                ),
            )
        self.assertEqual([], sink)


if __name__ == "__main__":
    unittest.main()
