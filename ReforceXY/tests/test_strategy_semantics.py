"""Strategy signal, validation and sizing contracts; requires the Freqtrade QA image."""

import unittest
from datetime import datetime, timezone
from unittest import mock

import numpy as np
import pandas as pd
from freqtrade.persistence import LocalTrade, Trade
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


class DateColumnTest(QaTestCase):
    def test_a_missing_date_column_is_refused(self):
        with self.assertRaises(ValueError):
            _ensure_datetime_series(None)

    def test_datetime_conversion_preserves_instants_and_index(self):
        series = pd.Series(
            pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"),
            index=[7, 11, 19],
        )
        expected = pd.Series(
            [
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T00:05:00Z"),
                pd.Timestamp("2024-01-01T00:10:00Z"),
            ],
            index=[7, 11, 19],
            dtype="datetime64[ms, UTC]",
        )
        pd.testing.assert_series_equal(expected, _ensure_datetime_series(series))

    def test_an_integer_column_inside_the_epoch_range_is_read_as_milliseconds(self):
        millis = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)
        series = _ensure_datetime_series(pd.Series([millis, millis + 300_000]))
        pd.testing.assert_series_equal(
            pd.Series(
                [
                    pd.Timestamp("2024-01-01T00:00:00Z"),
                    pd.Timestamp("2024-01-01T00:05:00Z"),
                ],
                dtype="datetime64[ms, UTC]",
            ),
            series,
        )

    def test_an_integer_epoch_in_seconds_is_refused_as_a_corrupted_unit(self):
        millis = int(pd.Timestamp("2024-01-01", tz="UTC").value // 1_000_000)
        with self.assertRaises(ValueError):
            _ensure_datetime_series(pd.Series([millis // 1000]))

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
        with self.assertRaises(ValueError):
            self._strategy(trading_mode="margin-ish").is_short_allowed()

    def test_can_short_follows_the_mode(self):
        self.assertTrue(self._strategy(trading_mode="futures").can_short)
        self.assertFalse(self._strategy(trading_mode="spot").can_short)


class ProtectionConfigTest(QaTestCase):
    def _strategy(self, custom_protections):
        strategy = RLAgentStrategy.__new__(RLAgentStrategy)
        strategy.config = {"custom_protections": custom_protections}
        return strategy

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

    def test_absent_and_null_configuration_use_the_proposed_leverage(self):
        strategy = self._strategy(None)
        self.assertEqual(2.0, strategy.leverage(**self._arguments()))
        strategy.config.clear()
        self.assertEqual(2.0, strategy.leverage(**self._arguments()))

    def test_a_configured_value_is_clamped_into_the_pair_bounds(self):
        for configured, expected in ((1.5, 1.5), (0.5, 1.0), (-5.0, 1.0), (9.0, 5.0)):
            with self.subTest(configured=configured):
                self.assertEqual(expected, self._strategy(configured).leverage(**self._arguments()))

    def test_a_non_numeric_configuration_falls_back_to_the_proposed_leverage(self):
        for configured in ("two", [1], {"a": 1}):
            with self.subTest(configured=configured):
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

    def test_integer_conversion_overflow_falls_back_instead_of_escaping_the_callback(self):
        self.assertEqual(2.0, self._strategy(10**500).leverage(**self._arguments()))

    def test_warnings_follow_invalid_configuration_transitions(self):
        strategy = self._strategy(float("nan"))
        with self.assertLogs(RLAgentStrategy.__module__, level="WARNING") as logs:
            for _ in range(3):
                self.assertEqual(strategy.leverage(**self._arguments()), 2.0)
            self.assertEqual(len(logs.records), 1)
            strategy.config["leverage"] = float("inf")
            self.assertEqual(strategy.leverage(**self._arguments()), 2.0)
            self.assertEqual(len(logs.records), 2)
            strategy.config["leverage"] = 0.5
            for _ in range(2):
                self.assertEqual(strategy.leverage(**self._arguments()), 1.0)
            self.assertEqual(len(logs.records), 3)
            strategy.config["leverage"] = 2.5
            self.assertEqual(strategy.leverage(**self._arguments()), 2.5)
            strategy.config["leverage"] = float("nan")
            self.assertEqual(strategy.leverage(**self._arguments()), 2.0)
            self.assertEqual(len(logs.records), 4)

    def test_missing_configuration_resets_warning_deduplication(self):
        strategy = self._strategy("invalid")
        with self.assertLogs(RLAgentStrategy.__module__, level="WARNING") as logs:
            self.assertEqual(strategy.leverage(**self._arguments()), 2.0)
            strategy.config.clear()
            self.assertEqual(strategy.leverage(**self._arguments()), 2.0)
            self.assertEqual(len(logs.records), 1)
            strategy.config["leverage"] = "invalid"
            self.assertEqual(strategy.leverage(**self._arguments()), 2.0)
            self.assertEqual(len(logs.records), 2)


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
        # trade on a value the model has not published. Both sides are exercised, because
        # the two conditions are independent and either one can be dropped on its own.
        result = self._strategy().populate_entry_trend(
            _frame(
                [Actions.Long_enter.value, Actions.Short_enter.value],
                do_predict=0,
            ),
            {"pair": "BTC/USDT"},
        )
        self.assertEqual([False, False], _raised(result["enter_long"]))
        self.assertEqual([False, False], _raised(result["enter_short"]))


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

    def test_an_expired_model_signals_exits_only_for_open_positions_of_the_current_pair(self):
        closed = LocalTrade(pair="BTC/USDT", is_open=False, is_short=False)
        long = LocalTrade(pair="BTC/USDT", is_open=True, is_short=False)
        short = LocalTrade(pair="BTC/USDT", is_open=True, is_short=True)
        unrelated_long = LocalTrade(pair="ETH/USDT", is_open=True, is_short=False)
        unrelated_short = LocalTrade(pair="ETH/USDT", is_open=True, is_short=True)
        for open_trades, expected in (
            ([], (False, False)),
            ([short], (False, True)),
            ([long, unrelated_short], (True, False)),
            ([short, unrelated_long], (False, True)),
            ([unrelated_long], (False, False)),
            ([unrelated_short], (False, False)),
        ):
            with self.subTest(expected=expected, trades=open_trades):
                with (
                    mock.patch.object(Trade, "use_db", False),
                    mock.patch.object(LocalTrade, "bt_trades", [closed]),
                    mock.patch.object(LocalTrade, "bt_trades_open", open_trades),
                ):
                    result = self._strategy().populate_exit_trend(
                        _frame([Actions.Neutral.value], do_predict=2), {"pair": "BTC/USDT"}
                    )
                self.assertEqual([expected[0]], _raised(result["exit_long"]))
                self.assertEqual([expected[1]], _raised(result["exit_short"]))

    def test_a_row_without_a_prediction_raises_no_exit(self):
        result = self._strategy().populate_exit_trend(
            _frame(
                [Actions.Long_exit.value, Actions.Short_exit.value],
                do_predict=0,
            ),
            {"pair": "BTC/USDT"},
        )
        self.assertEqual([False, False], _raised(result["exit_long"]))
        self.assertEqual([False, False], _raised(result["exit_short"]))


class FeatureEngineeringTest(QaTestCase):
    def _strategy(self):
        return RLAgentStrategy.__new__(RLAgentStrategy)

    def test_time_features_encode_week_and_day_boundaries_with_the_original_index(self):
        frame = _frame([0, 0, 0])
        frame.index = [7, 13, 29]
        frame["date"] = pd.to_datetime(
            ["2024-01-01T00:00:00Z", "2024-01-07T23:00:00Z", "2024-01-08T00:00:00Z"]
        )
        result = self._strategy().feature_engineering_standard(frame, {})
        self.assertEqual([7, 13, 29], list(result.index))
        np.testing.assert_allclose(result["%-day_of_week"], [1 / 7, 1, 1 / 7], atol=1e-12)
        np.testing.assert_allclose(result["%-hour_of_day"], [1 / 25, 24 / 25, 1 / 25], atol=1e-12)

    def test_log_returns_keep_positive_negative_and_flat_moves_in_index_order(self):
        frame = _frame([0, 0, 0, 0])
        frame.index = [7, 13, 29, 41]
        frame["close"] = [100.0, 110.0, 99.0, 99.0]
        result = self._strategy().feature_engineering_expand_basic(frame, {})
        self.assertEqual([7, 13, 29, 41], list(result.index))
        np.testing.assert_allclose(
            result["%-close_log_return"],
            [np.nan, np.log(1.1), np.log(0.9), 0.0],
            atol=1e-12,
            equal_nan=True,
        )

    def test_the_target_column_is_the_neutral_action(self):
        result = self._strategy().set_freqai_targets(
            _frame([Actions.Long_enter.value, Actions.Short_enter.value]), {}
        )
        self.assertEqual([0, 0], list(result[ACTION_COLUMN]))


if __name__ == "__main__":
    unittest.main()
