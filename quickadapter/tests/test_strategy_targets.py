"""Label target, indicator and entry-signal contracts; requires the Freqtrade QA image."""

import logging
import unittest

import numpy as np
import pandas as pd
from qa_support import PAIR, QaTestCase, model_config, temporary_directory
from QuickAdapterV3 import QuickAdapterV3
from Utils import EXTREMA_COLUMN, generate_label_data, smooth

LONG = QuickAdapterV3._TRADE_LONG
SHORT = QuickAdapterV3._TRADE_SHORT
PREDICT_COLUMN = "do_predict"
CATCH_COLUMN = "DI_catch"
MINIMA_COLUMN = "minima_threshold"
MAXIMA_COLUMN = "maxima_threshold"
LABEL_HORIZON_COLUMN = "s-extrema_known_at_lookahead"
LABEL_WEIGHT_COLUMN = "s-extrema_weight"
WEIGHT_COLUMNS = (
    LABEL_WEIGHT_COLUMN,
    "extrema_weight",
    "extrema_weight_smoothed",
    "s-extrema_weight_known_at_lookahead",
)
logger = logging.getLogger(__name__)


def label_strategy(feature_parameters=None, label_params=None):
    """A strategy carrying only the label state the accessors read."""
    model = object.__new__(QuickAdapterV3)
    model.freqai_info = {"feature_parameters": dict(feature_parameters or {})}
    model._label_params = dict(label_params or {PAIR: {}})
    model._candle_deviation_cache = {}
    model._candle_threshold_cache = {}
    model._cached_df_signature = {}
    return model


def runtime(tmp, **overrides):
    """A strategy from a real config, with the FreqAI frame seam stubbed."""
    config = model_config(tmp, **overrides)
    model = QuickAdapterV3(config)
    model.freqai_info = config["freqai"]
    model.freqai = type("FreqaiKitchen", (), {"start": staticmethod(lambda df, md, s: df)})()
    return model


def candles(rows: int = 6, **columns) -> pd.DataFrame:
    """A strictly rising OHLCV frame, extended with the named columns."""
    index = np.arange(rows)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC"),
            "high": 101.0 + index,
            "low": 99.0 - index,
            "close": 100.0 + index * 0.5,
        }
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


def zigzag(rows: int = 200, amplitude: float = 6.0, period: int = 17) -> pd.DataFrame:
    """A deterministic oscillating OHLCV frame that produces extrema labels."""
    close = 100 + amplitude * np.sin(2 * np.pi * np.arange(rows) / period)
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC"),
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(rows, 10.0),
        }
    )


def entry_frame(predicts, catches, extrema, minima=0.0, maxima=0.0) -> pd.DataFrame:
    rows = len(predicts)
    return pd.DataFrame(
        {
            PREDICT_COLUMN: predicts,
            CATCH_COLUMN: catches,
            EXTREMA_COLUMN: extrema,
            MINIMA_COLUMN: np.full(rows, minima),
            MAXIMA_COLUMN: np.full(rows, maxima),
        }
    )


def fired(result: pd.DataFrame, column: str) -> list[int]:
    """The row positions where an entry column carries 1; a void column fires nowhere."""
    return [i for i, value in enumerate(result[column].astype(object)) if value == 1]


def tags(result: pd.DataFrame) -> dict[int, str]:
    return {
        row: result["enter_tag"].astype(object).iloc[row]
        for column in ("enter_long", "enter_short")
        for row in fired(result, column)
    }


class StrategyTargetsTest(QaTestCase):
    def test_a_long_entry_needs_a_prediction_a_caught_divergence_and_a_below_minimum(self):
        model = object.__new__(QuickAdapterV3)
        for predicts in (1, 0):
            for catches in (1, 0):
                for below in (True, False):
                    with self.subTest(predict=predicts, catch=catches, below=below):
                        frame = entry_frame(
                            [predicts], [catches], [-1.0 if below else 1.0], maxima=2.0
                        )
                        result = model.populate_entry_trend(frame, {"pair": PAIR})
                        expected = [0] if predicts == catches == 1 and below else []
                        self.assertEqual(fired(result, "enter_long"), expected)
                        self.assertEqual(fired(result, "enter_short"), [])

    def test_a_short_entry_needs_a_prediction_a_caught_divergence_and_an_above_maximum(self):
        model = object.__new__(QuickAdapterV3)
        for predicts in (1, 0):
            for catches in (1, 0):
                for above in (True, False):
                    with self.subTest(predict=predicts, catch=catches, above=above):
                        frame = entry_frame(
                            [predicts], [catches], [1.0 if above else -1.0], minima=-2.0
                        )
                        result = model.populate_entry_trend(frame, {"pair": PAIR})
                        expected = [0] if predicts == catches == 1 and above else []
                        self.assertEqual(fired(result, "enter_short"), expected)
                        self.assertEqual(fired(result, "enter_long"), [])

    def test_a_prediction_other_than_one_is_not_an_entry(self):
        model = object.__new__(QuickAdapterV3)
        frame = entry_frame([-1, 2, 1.0], [1, 1, 1], [-1.0, -1.0, -1.0], maxima=2.0)
        result = model.populate_entry_trend(frame, {"pair": PAIR})
        self.assertEqual(fired(result, "enter_long"), [2])

    def test_ordered_thresholds_make_the_two_sides_mutually_exclusive(self):
        model = object.__new__(QuickAdapterV3)
        frame = entry_frame([1] * 3, [1] * 3, [-3.0, 0.0, 3.0], minima=-2.0, maxima=2.0)
        result = model.populate_entry_trend(frame, {"pair": PAIR})
        longs, shorts = fired(result, "enter_long"), fired(result, "enter_short")
        self.assertEqual(longs, [0])
        self.assertEqual(shorts, [2])
        self.assertFalse(set(longs) & set(shorts))

    def test_the_enter_tag_names_the_side_that_fired(self):
        model = object.__new__(QuickAdapterV3)
        frame = entry_frame([1, 1], [1, 1], [-1.0, 1.0])
        result = model.populate_entry_trend(frame, {"pair": PAIR})
        self.assertEqual(tags(result), {0: LONG, 1: SHORT})
        self.assertEqual([tags(result)[row] for row in fired(result, "enter_long")], [LONG])

    def test_inverted_thresholds_fire_both_sides_and_leave_the_short_tag(self):
        model = object.__new__(QuickAdapterV3)
        frame = entry_frame([1], [1], [0.0], minima=1.0, maxima=-1.0)
        result = model.populate_entry_trend(frame, {"pair": PAIR})
        self.assertEqual(fired(result, "enter_long"), [0])
        self.assertEqual(fired(result, "enter_short"), [0])
        self.assertEqual(tags(result), {0: SHORT})

    def test_a_missing_condition_column_denies_both_sides(self):
        columns = {
            PREDICT_COLUMN: [1],
            CATCH_COLUMN: [1],
            EXTREMA_COLUMN: [-1.0],
            MINIMA_COLUMN: [0.0],
            MAXIMA_COLUMN: [0.0],
        }
        for dropped in (PREDICT_COLUMN, CATCH_COLUMN, EXTREMA_COLUMN):
            with self.subTest(dropped=dropped):
                model = object.__new__(QuickAdapterV3)
                frame = pd.DataFrame(
                    {name: values for name, values in columns.items() if name != dropped}
                )
                result = model.populate_entry_trend(frame, {"pair": PAIR})
                self.assertEqual(fired(result, "enter_long"), [])
                self.assertEqual(fired(result, "enter_short"), [])

    def test_a_missing_threshold_column_denies_only_the_side_that_reads_it(self):
        columns = {
            PREDICT_COLUMN: [1, 1],
            CATCH_COLUMN: [1, 1],
            EXTREMA_COLUMN: [-1.0, 1.0],
            MINIMA_COLUMN: [0.0, 0.0],
            MAXIMA_COLUMN: [0.0, 0.0],
        }
        # A dropped threshold leaves dataframe.get at None, whose comparison is False on
        # every row, so the side reading it is denied and the opposite side stands.
        survivors = {
            "minima_threshold": ("enter_short", [1]),
            "maxima_threshold": ("enter_long", [0]),
        }
        for dropped, (survivor, survivor_rows) in survivors.items():
            with self.subTest(dropped=dropped):
                model = object.__new__(QuickAdapterV3)
                frame = pd.DataFrame(
                    {name: values for name, values in columns.items() if name != dropped}
                )
                result = model.populate_entry_trend(frame, {"pair": PAIR})
                denied = "enter_short" if survivor == "enter_long" else "enter_long"
                self.assertEqual(fired(result, survivor), survivor_rows)
                self.assertEqual(fired(result, denied), [])

    def test_a_nan_threshold_never_fires_its_side(self):
        model = object.__new__(QuickAdapterV3)
        frame = entry_frame([1, 1], [1, 1], [-1.0, 1.0], minima=np.nan)
        result = model.populate_entry_trend(frame, {"pair": PAIR})
        self.assertEqual(fired(result, "enter_long"), [])
        self.assertEqual(fired(result, "enter_short"), [1])

    def test_a_catch_is_one_without_both_freqai_divergence_columns(self):
        for columns in ({}, {"DI_values": [4.0] * 6}, {"DI_cutoff": [2.0] * 6}):
            with self.subTest(supplied=sorted(columns)), temporary_directory() as temp:
                model = runtime(temp)
                model.bot_start()
                result = model.populate_indicators(candles(**columns), {"pair": PAIR})
                self.assertEqual(result[CATCH_COLUMN].tolist(), [1] * 6)

    def test_a_divergence_is_caught_unless_it_exceeds_the_cutoff(self):
        with temporary_directory() as temp:
            model = runtime(temp)
            model.bot_start()
            result = model.populate_indicators(
                candles(5, DI_values=[1.0, 2.0, 2.0001, 9.0, np.nan], DI_cutoff=[2.0] * 5),
                {"pair": PAIR},
            )
            self.assertEqual(result[CATCH_COLUMN].tolist(), [1, 1, 0, 0, 1])

    def test_the_extrema_thresholds_are_copied_verbatim(self):
        minima, maxima = [-2.0, -1.5, 0.0], [2.0, 1.5, 0.0]
        with temporary_directory() as temp:
            model = runtime(temp)
            model.bot_start()
            result = model.populate_indicators(
                candles(
                    3,
                    **{
                        f"{EXTREMA_COLUMN}_minima_threshold": minima,
                        f"{EXTREMA_COLUMN}_maxima_threshold": maxima,
                    },
                ),
                {"pair": PAIR},
            )
            self.assertEqual(result[MINIMA_COLUMN].tolist(), minima)
            self.assertEqual(result[MAXIMA_COLUMN].tolist(), maxima)

    def test_an_absent_extrema_threshold_becomes_nan(self):
        with temporary_directory() as temp:
            model = runtime(temp)
            model.bot_start()
            absent = model.populate_indicators(candles(), {"pair": PAIR})
            self.assertTrue(absent[MINIMA_COLUMN].isna().all())
            self.assertTrue(absent[MAXIMA_COLUMN].isna().all())
            only_minima = model.populate_indicators(
                candles(**{f"{EXTREMA_COLUMN}_minima_threshold": [-1.0] * 6}), {"pair": PAIR}
            )
            self.assertEqual(only_minima[MINIMA_COLUMN].tolist(), [-1.0] * 6)
            self.assertTrue(only_minima[MAXIMA_COLUMN].isna().all())
            only_maxima = model.populate_indicators(
                candles(**{f"{EXTREMA_COLUMN}_maxima_threshold": [1.0] * 6}), {"pair": PAIR}
            )
            self.assertEqual(only_maxima[MAXIMA_COLUMN].tolist(), [1.0] * 6)
            self.assertTrue(only_maxima[MINIMA_COLUMN].isna().all())

    def test_a_per_candle_label_period_scatters_natR_over_its_own_rows(self):
        periods = [2, 2, 2, 3, 3, 3]
        with temporary_directory() as temp:
            model = runtime(temp)
            model.bot_start()
            scattered = model.populate_indicators(
                candles(label_period_candles=periods), {"pair": PAIR}
            )
            for period in sorted(set(periods)):
                with self.subTest(period=period):
                    rows = [i for i, value in enumerate(periods) if value == period]
                    alone = model.populate_indicators(
                        candles(label_period_candles=[period] * 6), {"pair": PAIR}
                    )
                    np.testing.assert_allclose(
                        scattered["natr_label_period_candles"].iloc[rows].to_numpy(),
                        alone["natr_label_period_candles"].iloc[rows].to_numpy(),
                        rtol=1e-9,
                        atol=1e-12,
                    )

    def test_the_last_candle_of_a_per_candle_column_reaches_the_pair_state(self):
        with temporary_directory() as temp:
            model = runtime(temp)
            model.bot_start()
            model.populate_indicators(
                candles(label_period_candles=[2, 2, 2, 2, 2, 7], label_natr_multiplier=[5.0] * 6),
                {"pair": PAIR},
            )
            self.assertEqual(model._label_params[PAIR]["label_period_candles"], 7)
            self.assertEqual(model.get_label_period_candles(PAIR), 7)
            self.assertEqual(model._label_params[PAIR]["label_natr_multiplier"], 5.0)
            self.assertEqual(model.get_label_natr_multiplier(PAIR), 5.0)

    def test_bot_start_refuses_a_missing_or_empty_pair_whitelist(self):
        for pair_whitelist in (None, []):
            with (
                self.subTest(pair_whitelist=pair_whitelist),
                temporary_directory() as temp,
            ):
                config = model_config(temp)
                if pair_whitelist is None:
                    config["exchange"].pop("pair_whitelist")
                else:
                    config["exchange"]["pair_whitelist"] = pair_whitelist
                model = QuickAdapterV3(config)
                model.freqai_info = config["freqai"]
                with self.assertRaisesRegex(ValueError, "pair_whitelist"):
                    model.bot_start()

    def test_bot_start_refuses_a_missing_blank_or_non_string_identifier(self):
        for identifier in (None, "", "   ", 7):
            with self.subTest(identifier=repr(identifier)), temporary_directory() as temp:
                config = model_config(temp)
                if identifier is None:
                    config["freqai"].pop("identifier")
                else:
                    config["freqai"]["identifier"] = identifier
                model = QuickAdapterV3(config)
                model.freqai_info = config["freqai"]
                with self.assertRaisesRegex(ValueError, "identifier"):
                    model.bot_start()

    def test_bot_start_refuses_wrap_smoothing_under_causal_mode(self):
        with temporary_directory() as temp:
            config = model_config(
                temp,
                freqai={
                    "feature_parameters": {"causal_mode": True},
                    "label_smoothing": {"default": {"method": "savgol", "mode": "wrap"}},
                },
            )
            model = QuickAdapterV3(config)
            model.freqai_info = config["freqai"]
            with self.assertRaisesRegex(ValueError, "wrap.*incompatible"):
                model.bot_start()

    def test_bot_start_accepts_a_compatible_smoothing_configuration(self):
        for causal_mode, method, mode in (
            (True, "savgol", "mirror"),
            (True, "savgol", "interp"),
            (False, "savgol", "wrap"),
        ):
            with (
                self.subTest(causal_mode=causal_mode, method=method, mode=mode),
                temporary_directory() as temp,
            ):
                config = model_config(
                    temp,
                    freqai={
                        "feature_parameters": {"causal_mode": causal_mode},
                        "label_smoothing": {"default": {"method": method, "mode": mode}},
                    },
                )
                model = QuickAdapterV3(config)
                model.freqai_info = config["freqai"]
                model.bot_start()
                self.assertEqual(model.pairs, [PAIR])

    def test_bot_start_seeds_the_pair_state_from_the_configuration(self):
        with temporary_directory() as temp:
            config = model_config(temp, freqai={"feature_parameters": {"label_period_candles": 7}})
            model = QuickAdapterV3(config)
            model.freqai_info = config["freqai"]
            model.bot_start()
            self.assertEqual(model.pairs, [PAIR])
            self.assertEqual(
                model.models_full_path, temp / "models" / config["freqai"]["identifier"]
            )
            self.assertEqual(model._label_params[PAIR]["label_period_candles"], 7)
            self.assertIsInstance(model._label_params[PAIR]["label_natr_multiplier"], float)
            self.assertEqual(model._candle_duration_secs, 300)
            self.assertEqual(model._max_take_profit_history_size, 144)
            self.assertEqual(model._candle_deviation_cache, {})
            self.assertEqual(model._candle_threshold_cache, {})
            self.assertEqual(model._cached_df_signature, {})

    def test_a_non_finite_or_non_positive_label_period_is_refused(self):
        for value in (np.nan, np.inf, -np.inf, 0, -3, "4", None, True):
            with self.subTest(value=repr(value)):
                model = label_strategy(label_params={PAIR: {"label_period_candles": 5}})
                model.set_label_period_candles(PAIR, value)
                self.assertEqual(model._label_params[PAIR]["label_period_candles"], 5)
                self.assertEqual(model.get_label_period_candles(PAIR), 5)

    def test_a_non_finite_or_non_positive_nat_multiplier_is_refused(self):
        for value in (np.nan, np.inf, 0.0, -2.0, "1.5", None, True):
            with self.subTest(value=repr(value)):
                model = label_strategy(label_params={PAIR: {"label_natr_multiplier": 10.5}})
                model.set_label_natr_multiplier(PAIR, value)
                self.assertEqual(model._label_params[PAIR]["label_natr_multiplier"], 10.5)
                self.assertEqual(model.get_label_natr_multiplier(PAIR), 10.5)

    def test_setting_a_label_parameter_drops_the_pair_caches_and_changes_the_read(self):
        cases = (
            ("set_label_period_candles", "label_period_candles", 3, "get_label_period_candles"),
            (
                "set_label_natr_multiplier",
                "label_natr_multiplier",
                4.5,
                "get_label_natr_multiplier",
            ),
        )
        for setter, key, value, getter in cases:
            with self.subTest(parameter=key):
                initial = 5 if key == "label_period_candles" else 10.5
                model = label_strategy(label_params={PAIR: {key: initial}})
                model._candle_deviation_cache = {
                    (PAIR, (1, None), 0.0, 1.0, -1, "direct", 1.5): 0.25
                }
                model._candle_threshold_cache = {(PAIR, (1, None), LONG, -1, 0.0, 1.0): 110.0}
                model._cached_df_signature = {PAIR: (1, None)}
                getattr(model, setter)(PAIR, value)
                self.assertEqual(model._candle_deviation_cache, {})
                self.assertEqual(model._candle_threshold_cache, {})
                self.assertEqual(model._cached_df_signature, {})
                self.assertEqual(getattr(model, getter)(PAIR), value)

    def test_setting_an_unchanged_label_parameter_keeps_the_pair_caches(self):
        model = label_strategy(label_params={PAIR: {"label_period_candles": 5}})
        model._cached_df_signature = {PAIR: (1, None)}
        model.set_label_period_candles(PAIR, 5)
        self.assertEqual(model._cached_df_signature, {PAIR: (1, None)})

    def test_a_candle_parameter_prefers_its_column_over_the_pair_value(self):
        model = label_strategy(
            label_params={PAIR: {"label_period_candles": 5, "label_natr_multiplier": 10.5}}
        )
        frame = pd.DataFrame(
            {
                "label_period_candles": [2, 4, np.nan, 0],
                "label_natr_multiplier": [1.5, np.nan, 3.0, -1.0],
            }
        )
        self.assertEqual(
            [model.get_label_period_candles(PAIR, frame, i) for i in range(len(frame))],
            [2, 4, 5, 5],
        )
        self.assertEqual(
            [model.get_label_natr_multiplier(PAIR, frame, i) for i in range(len(frame))],
            [1.5, 10.5, 3.0, 10.5],
        )
        self.assertEqual(model.get_label_period_candles(PAIR, frame), 5)
        self.assertEqual(model.get_label_natr_multiplier(PAIR, frame), 10.5)

    def test_an_unknown_pair_falls_back_to_the_canonical_defaults(self):
        model = label_strategy()
        self.assertEqual(model.get_label_period_candles(PAIR), 18)
        self.assertEqual(model.get_label_natr_multiplier(PAIR), 10.5)
        self.assertEqual(model.get_label_horizon_candles(PAIR), 18)

    def test_a_configured_horizon_overrides_the_label_period(self):
        model = label_strategy(
            feature_parameters={"label_horizon_candles": 30},
            label_params={PAIR: {"label_period_candles": 4}},
        )
        self.assertEqual(model.get_label_horizon_candles(PAIR), 30)

    def test_the_label_params_carry_the_extrema_triple_only(self):
        model = label_strategy(
            feature_parameters={"label_horizon_candles": 12},
            label_params={PAIR: {"label_period_candles": 4, "label_natr_multiplier": 2.0}},
        )
        self.assertEqual(
            model.get_label_params(PAIR, EXTREMA_COLUMN),
            {"natr_period": 4, "natr_multiplier": 2.0, "label_horizon_candles": 12},
        )
        self.assertEqual(model.get_label_params(PAIR, "other-label"), {})

    def test_the_nat_multiplier_fraction_validates_its_own_bounds(self):
        model = label_strategy(label_params={PAIR: {"label_natr_multiplier": 10.5}})
        for fraction, expected in ((0.0, 0.0), (0.5, 5.25), (1.0, 10.5)):
            with self.subTest(fraction=fraction):
                self.assertAlmostEqual(
                    model.get_label_natr_multiplier_fraction(PAIR, fraction),
                    expected,
                    places=12,
                )
        for fraction in (0, 1, -0.1, 1.1, "0.5", None):
            with (
                self.subTest(fraction=repr(fraction)),
                self.assertRaisesRegex(ValueError, r"must be a float in range \[0, 1\]"),
            ):
                model.get_label_natr_multiplier_fraction(PAIR, fraction)

    def test_a_label_period_truncates_to_an_integer_and_a_multiplier_widens_to_a_float(self):
        model = label_strategy()
        model.set_label_period_candles(PAIR, 3.9)
        model.set_label_natr_multiplier(PAIR, 3)
        self.assertEqual(model._label_params[PAIR]["label_period_candles"], 3)
        self.assertIsInstance(model._label_params[PAIR]["label_period_candles"], int)
        self.assertEqual(model._label_params[PAIR]["label_natr_multiplier"], 3.0)
        self.assertIsInstance(model._label_params[PAIR]["label_natr_multiplier"], float)

    def test_the_label_column_is_the_smoothed_series_and_the_direction_is_not(self):
        with temporary_directory() as temp:
            model = runtime(temp)
            model.bot_start()
            frame = zigzag()
            result = model.set_freqai_targets(frame.copy(), {"pair": PAIR})
            raw = generate_label_data(
                frame.copy(),
                EXTREMA_COLUMN,
                model.get_label_params(PAIR, EXTREMA_COLUMN),
                logger,
            )
            self.assertTrue((result["extrema_direction"].to_numpy() == raw.series.to_numpy()).all())
            self.assertEqual(
                sorted(result["extrema_direction"].unique().tolist()), [-1.0, 0.0, 1.0]
            )
            pd.testing.assert_series_equal(
                result[EXTREMA_COLUMN],
                smooth(raw.series, **model.label_smoothing["default"]),
                check_names=False,
                check_exact=False,
                rtol=1e-9,
                atol=1e-9,
            )
            pd.testing.assert_series_equal(
                result["extrema_direction_smoothed"],
                result[EXTREMA_COLUMN],
                check_names=False,
            )
            self.assertNotIn(EXTREMA_COLUMN, frame.columns)
            self.assertTrue(result[EXTREMA_COLUMN].notna().any())

    def test_every_labelled_candle_carries_at_least_one_candle_of_lookahead(self):
        with temporary_directory() as temp:
            model = runtime(temp)
            model.bot_start()
            result = model.set_freqai_targets(zigzag(), {"pair": PAIR})
            horizon = result[LABEL_HORIZON_COLUMN].to_numpy()
            labelled = result[EXTREMA_COLUMN].to_numpy() != 0.0
            self.assertTrue(labelled.any())
            self.assertTrue(np.isfinite(horizon).all())
            self.assertTrue((horizon == np.round(horizon)).all())
            self.assertTrue((horizon[labelled] >= 1).all())

    def test_a_flat_price_series_produces_no_labels_and_no_label_weights(self):
        flat = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=60, freq="5min", tz="UTC"),
                "open": np.full(60, 100.0),
                "high": np.full(60, 100.1),
                "low": np.full(60, 99.9),
                "close": np.full(60, 100.0),
                "volume": np.full(60, 10.0),
            }
        )
        with temporary_directory() as temp:
            model = runtime(temp, freqai={"label_weighting": {"default": {"strategy": "uniform"}}})
            model.bot_start()
            result = model.set_freqai_targets(flat, {"pair": PAIR})
            self.assertEqual(result[EXTREMA_COLUMN].fillna(0.0).ne(0.0).sum(), 0)
            for column in WEIGHT_COLUMNS:
                with self.subTest(column=column):
                    self.assertNotIn(column, result.columns)

    def test_a_label_weight_is_positive_on_every_labelled_candle(self):
        with temporary_directory() as temp:
            model = runtime(temp, freqai={"label_weighting": {"default": {"strategy": "uniform"}}})
            model.bot_start()
            result = model.set_freqai_targets(zigzag(), {"pair": PAIR})
            labelled = result[EXTREMA_COLUMN].to_numpy() != 0.0
            weights = result[LABEL_WEIGHT_COLUMN]
            smoothed = result["extrema_weight_smoothed"]
            self.assertTrue((weights[labelled] > 0.0).all())
            self.assertTrue((weights >= 0.0).all())
            self.assertTrue(np.isfinite(smoothed.to_numpy()).all())
            self.assertTrue((smoothed >= 0.0).all())

    def test_no_label_weight_column_appears_under_an_inactive_weighting_strategy(self):
        with temporary_directory() as temp:
            model = runtime(temp)
            model.bot_start()
            self.assertEqual(model.label_weighting["default"]["strategy"], "none")
            result = model.set_freqai_targets(zigzag(), {"pair": PAIR})
            for column in WEIGHT_COLUMNS:
                with self.subTest(column=column):
                    self.assertNotIn(column, result.columns)

    def test_the_label_parameters_drive_the_generated_labels(self):
        with temporary_directory() as temp:
            model = runtime(temp)
            model.bot_start()
            narrow = model.set_freqai_targets(zigzag(), {"pair": PAIR})
            model.set_label_period_candles(PAIR, 24)
            wide = model.set_freqai_targets(zigzag(), {"pair": PAIR})
            self.assertTrue((narrow[EXTREMA_COLUMN] != 0.0).any())
            self.assertEqual(wide[EXTREMA_COLUMN].fillna(0.0).ne(0.0).sum(), 0)


if __name__ == "__main__":
    unittest.main()
