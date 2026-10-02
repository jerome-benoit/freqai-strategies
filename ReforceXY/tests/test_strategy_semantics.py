"""Strategy signal, validation and sizing contracts; requires the Freqtrade QA image."""

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from qa_support import QaTestCase

from ReforceXY.user_data.freqaimodels.ReforceXY import Actions
from ReforceXY.user_data.strategies.RLAgentStrategy import (
    ACTION_COLUMN,
    RLAgentStrategy,
    _ensure_datetime_series,
)


def _raised(column):
    """True where the flag is set. An untouched row is NaN, not 0: the column is created
    by the assignment, so "no signal" arrives as a missing float rather than a zero."""
    return [bool(value) if value == value else False for value in column]


def _frame(actions, do_predict=1, count=None):
    values = list(actions)
    count = count if count is not None else len(values)
    return pd.DataFrame(
        {
            "open": np.linspace(100.0, 110.0, count),
            "high": np.linspace(101.0, 111.0, count),
            "low": np.linspace(99.0, 109.0, count),
            "close": np.linspace(100.5, 110.5, count),
            "volume": np.full(count, 10.0),
            "date": pd.date_range("2024-01-01", periods=count, freq="5min", tz="UTC"),
            ACTION_COLUMN: values,
            "do_predict": [do_predict] * count,
        }
    )


class ActionLiteralTest(QaTestCase):
    def test_the_strategy_literals_are_the_production_action_codes(self):
        # Asserted as a cross-object equality, not as two literals: the strategy's constants
        # are correct only because they equal the enum's values, and two independent
        # literals would both still pass after a freqtrade renumbering.
        self.assertEqual(Actions.Long_enter.value, RLAgentStrategy._ACTION_ENTER_LONG)
        self.assertEqual(Actions.Long_exit.value, RLAgentStrategy._ACTION_EXIT_LONG)
        self.assertEqual(Actions.Short_enter.value, RLAgentStrategy._ACTION_ENTER_SHORT)
        self.assertEqual(Actions.Short_exit.value, RLAgentStrategy._ACTION_EXIT_SHORT)


class DateColumnTest(QaTestCase):
    def test_a_missing_date_column_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            _ensure_datetime_series(None)
        self.assertIn("date", str(caught.exception))

    def test_a_non_null_date_column_keeps_its_values(self):
        series = pd.Series(pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"))
        self.assertEqual(3, len(_ensure_datetime_series(series)))

    def test_an_integer_column_inside_the_epoch_range_is_read_as_milliseconds(self):
        millis = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)
        series = _ensure_datetime_series(pd.Series([millis, millis + 300_000]))
        self.assertEqual(2, len(series))

    def test_an_integer_epoch_in_seconds_is_refused_as_a_corrupted_unit(self):
        millis = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)
        with self.assertRaises(ValueError) as caught:
            _ensure_datetime_series(pd.Series([millis // 1000]))
        self.assertIn("epoch-ms", str(caught.exception))

    def test_an_all_null_integer_column_is_read_without_a_range_probe(self):
        # Nothing to probe means nothing to reject; refusing here would reject a column
        # that has no data to be wrong about.
        series = _ensure_datetime_series(pd.Series([None, None], dtype="Int64"))
        self.assertTrue(series.isna().all())


class TradingModeTest(QaTestCase):
    def _strategy(self, **config):
        strategy = RLAgentStrategy.__new__(RLAgentStrategy)
        strategy.config = config
        return strategy

    def test_the_two_leveraged_modes_allow_shorting(self):
        for mode in ("margin", "futures"):
            with self.subTest(mode=mode):
                self.assertTrue(self._strategy(trading_mode=mode).is_short_allowed())

    def test_spot_forbids_shorting(self):
        self.assertFalse(self._strategy(trading_mode="spot").is_short_allowed())

    def test_an_undeclared_trading_mode_is_refused_rather_than_defaulted(self):
        with self.assertRaises(ValueError) as caught:
            self._strategy(trading_mode="margin-ish").is_short_allowed()
        self.assertIn("trading_mode", str(caught.exception))

    def test_can_short_follows_the_mode(self):
        self.assertTrue(self._strategy(trading_mode="futures").can_short)
        self.assertFalse(self._strategy(trading_mode="spot").can_short)


class ProtectionConfigTest(QaTestCase):
    def _strategy(self, custom_protections):
        strategy = RLAgentStrategy.__new__(RLAgentStrategy)
        strategy.config = {"custom_protections": custom_protections}
        return strategy

    def test_a_well_formed_protection_list_is_passed_through_untouched(self):
        given = [{"method": "CooldownPeriod", "stop_duration": 2}]
        self.assertIs(given, self._strategy(given).protections)

    def test_an_empty_list_is_accepted(self):
        self.assertEqual([], self._strategy([]).protections)

    def test_a_protection_without_a_method_name_is_refused(self):
        strategy = self._strategy([{"stop_duration": 2}])
        with self.assertRaises(ValueError):
            _ = strategy.protections

    def test_a_non_list_protection_value_is_refused(self):
        strategy = self._strategy({"method": "CooldownPeriod"})
        with self.assertRaises(ValueError):
            _ = strategy.protections


class LeverageTest(QaTestCase):
    def _strategy(self, configured):
        strategy = RLAgentStrategy.__new__(RLAgentStrategy)
        strategy.config = {"leverage": configured}
        strategy._leverage_warning_state = None
        return strategy

    def _arguments(self, **overrides):
        arguments = {
            "pair": "BTC/USDT",
            "current_time": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "current_rate": 100.0,
            "proposed_leverage": 2.0,
            "max_leverage": 5.0,
            "entry_tag": None,
            "side": "long",
        }
        arguments.update(overrides)
        return arguments

    def test_an_absent_configuration_yields_the_proposed_leverage(self):
        self.assertEqual(2.0, self._strategy(None).leverage(**self._arguments()))

    def test_a_configured_value_is_clamped_into_the_pair_bounds(self):
        for configured, expected in ((1.5, 1.5), (0.5, 1.0), (9.0, 5.0)):
            with self.subTest(configured=configured):
                self.assertEqual(expected, self._strategy(configured).leverage(**self._arguments()))

    def test_a_non_numeric_configuration_falls_back_to_the_proposed_leverage(self):
        for configured in ("two", None, [1]):
            with self.subTest(configured=configured):
                if configured is None:
                    continue
                self.assertEqual(2.0, self._strategy(configured).leverage(**self._arguments()))

    def test_a_boolean_is_not_read_as_a_number(self):
        # bool is an int subclass, so float(True) succeeds; treating 1.0 as a leverage
        # configuration would silently disable the clamp.
        self.assertEqual(2.0, self._strategy(True).leverage(**self._arguments()))

    def test_a_non_finite_configuration_falls_back_to_the_proposed_leverage(self):
        for configured in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(configured=configured):
                self.assertEqual(2.0, self._strategy(configured).leverage(**self._arguments()))

    def test_the_result_is_always_at_least_one(self):
        self.assertEqual(1.0, self._strategy(0.5).leverage(**self._arguments(max_leverage=0.2)))


class EntrySignalTest(QaTestCase):
    def _strategy(self):
        return RLAgentStrategy.__new__(RLAgentStrategy)

    def test_a_long_entry_action_raises_the_long_flag_and_nothing_else(self):
        frame = _frame([1, 0, 0])
        result = self._strategy().populate_entry_trend(frame, {"pair": "BTC/USDT"})
        self.assertEqual([True, False, False], _raised(result["enter_long"]))
        self.assertEqual([False] * 3, _raised(result["enter_short"]))

    def test_a_short_entry_action_raises_the_short_flag(self):
        result = self._strategy().populate_entry_trend(_frame([3, 0]), {"pair": "BTC/USDT"})
        self.assertEqual([True, False], _raised(result["enter_short"]))

    def test_no_action_raises_no_entry(self):
        for action in (0, 2, 4):
            with self.subTest(action=action):
                result = self._strategy().populate_entry_trend(
                    _frame([action]), {"pair": "BTC/USDT"}
                )
                self.assertFalse(_raised(result["enter_long"])[0])
                self.assertFalse(_raised(result["enter_short"])[0])

    def test_a_row_without_a_prediction_raises_no_entry(self):
        # do_predict != 1 means the prediction is not yet confirmed; acting on it would
        # trade on a value the model has not published.
        result = self._strategy().populate_entry_trend(
            _frame([1, 1], do_predict=0), {"pair": "BTC/USDT"}
        )
        self.assertEqual([False, False], _raised(result["enter_long"]))


class ExitSignalTest(QaTestCase):
    def _strategy(self):
        return RLAgentStrategy.__new__(RLAgentStrategy)

    def test_an_exit_action_raises_only_its_own_side(self):
        result = self._strategy().populate_exit_trend(_frame([2, 0, 4]), {"pair": "BTC/USDT"})
        self.assertEqual([True, False, False], _raised(result["exit_long"]))
        self.assertEqual([False, False, True], _raised(result["exit_short"]))

    def test_an_entry_action_raises_no_exit(self):
        for action in (1, 3):
            with self.subTest(action=action):
                result = self._strategy().populate_exit_trend(
                    _frame([action]), {"pair": "BTC/USDT"}
                )
                self.assertFalse(_raised(result["exit_long"])[0])
                self.assertFalse(_raised(result["exit_short"])[0])

    def test_a_rejected_prediction_closes_the_open_position_of_its_own_side(self):
        # do_predict == 2 is a rejected prediction: the open trade must be closed, and
        # its side decides which flag is set, not the recorded action.
        frame = _frame([1], do_predict=2)
        for is_short, expected in ((False, "exit_long"), (True, "exit_short")):
            with self.subTest(is_short=is_short):
                trade = SimpleNamespace(is_short=is_short)
                with mock.patch(
                    "ReforceXY.user_data.strategies.RLAgentStrategy.Trade.get_trades_proxy",
                    return_value=[trade],
                ):
                    result = self._strategy().populate_exit_trend(
                        frame.copy(), {"pair": "BTC/USDT"}
                    )
                self.assertTrue(_raised(result[expected])[0])
                other = "exit_short" if expected == "exit_long" else "exit_long"
                self.assertFalse(_raised(result[other])[0])

    def test_a_rejected_prediction_with_no_open_trade_raises_no_exit(self):
        frame = _frame([1], do_predict=2)
        with mock.patch(
            "ReforceXY.user_data.strategies.RLAgentStrategy.Trade.get_trades_proxy",
            return_value=[],
        ):
            result = self._strategy().populate_exit_trend(frame, {"pair": "BTC/USDT"})
        self.assertFalse(_raised(result["exit_long"])[0])
        self.assertFalse(_raised(result["exit_short"])[0])


class FeatureEngineeringTest(QaTestCase):
    def _strategy(self):
        return RLAgentStrategy.__new__(RLAgentStrategy)

    def test_the_standard_features_are_bounded_copies_of_the_ohlc(self):
        frame = _frame([0, 0, 0])
        result = self._strategy().feature_engineering_standard(frame, {})
        for source, produced in (("close", "%-raw_close"), ("open", "%-raw_open")):
            with self.subTest(feature=produced):
                self.assertEqual(list(frame[source]), list(result[produced]))

    def test_the_time_features_are_normalised_into_the_unit_interval(self):
        result = self._strategy().feature_engineering_standard(_frame([0] * 48), {})
        for feature in ("%-day_of_week", "%-hour_of_day"):
            with self.subTest(feature=feature):
                self.assertTrue(((result[feature] > 0.0) & (result[feature] <= 1.0)).all())

    def test_the_day_of_week_never_reports_a_zero_weekday(self):
        frame = pd.DataFrame({"date": pd.Series(pd.to_datetime(["2024-01-01"], utc=True))})
        result = self._strategy().feature_engineering_standard(frame, {})
        self.assertGreater(result["%-day_of_week"].iloc[0], 0.0)

    def test_the_expanding_feature_is_the_log_difference_of_the_close(self):
        frame = _frame([0, 0, 0])
        frame["close"] = [100.0, 110.0, 121.0]
        result = self._strategy().feature_engineering_expand_basic(frame, {})
        self.assertTrue(pd.isna(result["%-close_log_return"].iloc[0]))
        self.assertAlmostEqual(np.log(1.1), result["%-close_log_return"].iloc[1], places=12)

    def test_the_target_column_is_the_neutral_action(self):
        result = self._strategy().set_freqai_targets(_frame([1, 3]), {})
        self.assertEqual([0, 0], list(result[ACTION_COLUMN]))


if __name__ == "__main__":
    unittest.main()
