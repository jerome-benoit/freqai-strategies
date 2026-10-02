"""Position sizing and throttle contracts; requires the Freqtrade QA image."""

import datetime
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from freqtrade.enums import RunMode
from freqtrade.persistence import Trade
from qa_support import PAIR, QaTestCase, model_config, temporary_directory
from QuickAdapterV3 import QuickAdapterV3


def strategy(*, duration=10, natr=2.0, fraction=0.5, candle_secs=300):
    """A strategy whose sizing collaborators return fixed, stated values."""
    model = object.__new__(QuickAdapterV3)
    model.get_trade_duration_candles = lambda df, trade: duration
    model.get_trade_natr = lambda df, trade, duration_candles: natr
    model.get_label_natr_multiplier_fraction = lambda pair, value, df: value
    model._candle_duration_secs = candle_secs
    model.last_candle_start_secs = {}
    return model


def trade(*, open_rate=100.0, successful_exits=0):
    return type(
        "Trade",
        (),
        {
            "pair": PAIR,
            "open_rate": open_rate,
            "nr_of_successful_exits": successful_exits,
            "open_date_utc": datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC),
        },
    )()


def frame(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame({"close": np.linspace(100.0, 104.0, rows)})


class StoplossFactorTest(QaTestCase):
    def test_the_factor_matches_its_formula(self):
        for duration in (1, 5, 20, 100):
            with self.subTest(duration=duration):
                expected = 2.75 / (1.2675 + math.atan(0.25 * duration))
                self.assertAlmostEqual(
                    QuickAdapterV3.get_stoploss_factor(duration), expected, places=12
                )

    def test_the_factor_decays_with_duration(self):
        factors = [QuickAdapterV3.get_stoploss_factor(d) for d in (1, 10, 50, 200)]
        self.assertEqual(factors, sorted(factors, reverse=True))

    def test_the_factor_stays_positive_at_extreme_durations(self):
        for duration in (1, 10_000):
            with self.subTest(duration=duration):
                self.assertGreater(QuickAdapterV3.get_stoploss_factor(duration), 0.0)


class TakeProfitFactorTest(QaTestCase):
    def test_the_factor_matches_its_formula(self):
        for duration in (1, 5, 20, 100):
            with self.subTest(duration=duration):
                self.assertAlmostEqual(
                    QuickAdapterV3.get_take_profit_factor(duration),
                    math.log10(9.75 + 0.25 * duration),
                    places=12,
                )

    def test_the_factor_grows_with_duration(self):
        factors = [QuickAdapterV3.get_take_profit_factor(d) for d in (1, 10, 50, 200)]
        self.assertEqual(factors, sorted(factors))


class TradeDurationTest(QaTestCase):
    def test_only_a_positive_finite_duration_is_valid(self):
        for duration, expected in ((1, True), (0.5, True), (0, False), (-1, False), (None, False)):
            with self.subTest(duration=duration):
                self.assertIs(QuickAdapterV3.is_trade_duration_valid(duration), expected)

    def test_a_non_finite_duration_is_invalid(self):
        # is_trade_duration_valid rejects NaN and non-positive values. It does not reject
        # +inf, which is unreachable: get_trade_duration_candles returns an int or None.
        for duration in (np.nan, -np.inf):
            with self.subTest(duration=duration):
                self.assertFalse(QuickAdapterV3.is_trade_duration_valid(duration))

    def test_the_duration_is_the_elapsed_candles_since_entry(self):
        # A bare instance, so the real get_trade_duration_candles runs.
        model = object.__new__(QuickAdapterV3)
        model.timeframe_minutes = 5
        dates = pd.date_range("2026-01-01 00:00", periods=4, freq="5min", tz="UTC")
        model.get_trade_entry_date = lambda trade: dates[0]
        self.assertEqual(
            model.get_trade_duration_candles(pd.DataFrame({"date": dates}), trade()), 3
        )

    def test_a_frame_without_dates_yields_no_duration(self):
        model = object.__new__(QuickAdapterV3)
        model.timeframe_minutes = 5
        model.get_trade_entry_date = lambda trade: datetime.datetime(
            2026, 1, 1, tzinfo=datetime.UTC
        )
        self.assertIsNone(model.get_trade_duration_candles(pd.DataFrame({"close": [1.0]}), trade()))


class DistanceTest(QaTestCase):
    def _stoploss(self, model, **kwargs):
        return model.get_stoploss_distance(
            frame(), trade(**kwargs.pop("trade", {})), 100.0, **kwargs
        )

    def _take_profit(self, model, **kwargs):
        return model.get_take_profit_distance(frame(), trade(**kwargs.pop("trade", {})), **kwargs)

    def test_the_take_profit_distance_uses_the_open_rate_not_the_current_rate(self):
        model = strategy(duration=10, natr=2.0)
        expected = 100.0 * (2.0 / 100.0) * 0.5 * QuickAdapterV3.get_take_profit_factor(10)
        self.assertAlmostEqual(
            model.get_take_profit_distance(frame(), trade(open_rate=100.0), 0.5),
            expected,
            places=10,
        )
        # A different open rate moves the target proportionally, which is what fixing the
        # target to the entry volatility assumption means.
        doubled = model.get_take_profit_distance(frame(), trade(open_rate=200.0), 0.5)
        self.assertAlmostEqual(doubled, 2 * expected, places=10)

    def test_a_successful_exit_narrows_the_stoploss(self):
        model = strategy(duration=10, natr=2.0)
        without = model.get_stoploss_distance(frame(), trade(), 100.0, 0.5)
        with_exit = model.get_stoploss_distance(frame(), trade(successful_exits=9), 100.0, 0.5)
        self.assertLess(with_exit, without)

    def test_both_distances_refuse_a_multiplier_fraction_outside_the_unit_interval(self):
        for fraction in (-0.1, 1.1):
            with self.subTest(fraction=fraction):
                model = strategy()
                with self.assertRaisesRegex(ValueError, "must be in range"):
                    model.get_stoploss_distance(frame(), trade(), 100.0, fraction)
                with self.assertRaisesRegex(ValueError, "must be in range"):
                    model.get_take_profit_distance(frame(), trade(), fraction)

    def test_both_distances_accept_the_closed_unit_interval(self):
        # BOTH helpers. This case only called the stoploss one, so the take-profit path's
        # endpoints were untested: relaxing its valid condition to `0.0 < fraction <= 1.0`,
        # or returning None for a zero fraction before the range check, each left the suite
        # green. The upper endpoint is covered by sibling tests in test_strategy_features;
        # the lower one is only observable here.
        model = strategy()
        for fraction, is_zero in ((0.0, True), (1.0, False)):
            with self.subTest(fraction=fraction):
                stoploss = model.get_stoploss_distance(frame(), trade(), 100.0, fraction)
                self.assertEqual(stoploss == 0.0, is_zero)
                self.assertGreaterEqual(stoploss, 0.0)
                take_profit = model.get_take_profit_distance(frame(), trade(), fraction)
                self.assertIsNotNone(take_profit, "a valid fraction must return a distance")
                self.assertGreaterEqual(take_profit, 0.0)
        # A zero fraction is the endpoint that can silently become an error or None.
        self.assertEqual(model.get_take_profit_distance(frame(), trade(), 0.0), 0.0)

    def test_both_distances_refuse_an_invalid_duration(self):
        model = strategy(duration=0)
        self.assertIsNone(model.get_stoploss_distance(frame(), trade(), 100.0, 0.5))
        self.assertIsNone(model.get_take_profit_distance(frame(), trade(), 0.5))

    def test_both_distances_refuse_a_missing_or_negative_natr(self):
        for natr in (np.nan, -1.0):
            with self.subTest(natr=natr):
                model = strategy(natr=natr)
                self.assertIsNone(model.get_stoploss_distance(frame(), trade(), 100.0, 0.5))
                self.assertIsNone(model.get_take_profit_distance(frame(), trade(), 0.5))

    def test_a_zero_natr_collapses_both_distances(self):
        model = strategy(natr=0.0)
        self.assertEqual(model.get_stoploss_distance(frame(), trade(), 100.0, 0.5), 0.0)
        self.assertEqual(model.get_take_profit_distance(frame(), trade(), 0.5), 0.0)


class RuntimeSizingTest(QaTestCase):
    def setUp(self):
        super().setUp()
        directory = self.enterContext(temporary_directory())
        config = model_config(
            directory,
            runmode=RunMode.BACKTEST,
            exit_pricing={"trade_natr_method": "quantile_interpolation"},
        )
        self.model = QuickAdapterV3(config)
        self.model.freqai_info = config["freqai"]
        self.model.bot_start()
        self.now = datetime.datetime(2026, 1, 1, 0, 10, tzinfo=datetime.UTC)
        self.frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC"),
                "open": [100.0] * 3,
                "high": [101.0] * 3,
                "low": [99.0] * 3,
                "close": [100.0] * 3,
                "natr_label_period_candles": [1.0, 4.0, 8.0],
                "label_natr_multiplier": [3.0] * 3,
            }
        )
        self.model.dp = SimpleNamespace(
            get_analyzed_dataframe=lambda **kwargs: (self.frame, None),
            _exchange=SimpleNamespace(
                get_min_pair_stake_amount=lambda *args: 1.0,
                amount_to_contract_precision=lambda pair, amount: amount,
            ),
        )

    def position(self, short=False):
        return Trade(
            pair=PAIR,
            open_rate=100.0,
            open_date=self.now - datetime.timedelta(minutes=5),
            is_short=short,
            stake_amount=100.0,
            amount=1.0,
            leverage=2.0,
            exchange="binance",
            fee_open=0.0,
            fee_close=0.0,
            orders=[],
        )

    def test_stoploss_distance_uses_current_not_entry_price(self):
        # Entry NATR 4 has mean rank 0.25 in [4, 8], giving interpolated NATR 7.
        # One candle has elapsed since entry; the price has moved from 100 to 120.
        expected = 120.0 * 0.07 * 3.0 * 0.5 * 2.75 / (1.2675 + math.atan(0.25))
        self.assertAlmostEqual(
            self.model.get_stoploss_distance(self.frame, self.position(), 120.0, 0.5),
            expected,
            places=11,
        )

    def test_stoploss_callback_converts_distance_relative_to_current_rate(self):
        fraction = QuickAdapterV3._CUSTOM_STOPLOSS_NATR_MULTIPLIER_FRACTION
        expected = 0.07 * 3.0 * fraction * 2.75 / (1.2675 + math.atan(0.25)) * 2.0
        for short in (False, True):
            with self.subTest(short=short):
                self.assertAlmostEqual(
                    self.model.custom_stoploss(
                        PAIR, self.position(short), self.now, 120.0, 0.2, False
                    ),
                    expected,
                    places=11,
                )

    def test_partial_exit_reduces_stake_after_the_real_target_is_crossed(self):
        for short, favorable, unfavorable in ((False, 120.0, 105.0), (True, 80.0, 95.0)):
            with self.subTest(short=short):
                position = self.position(short)
                store = {}
                # Isolate only database persistence, not target or sizing calculations.
                with (
                    mock.patch.object(
                        position,
                        "get_custom_data",
                        side_effect=lambda key, default=None, store=store: store.get(key, default),
                    ),
                    mock.patch.object(
                        position,
                        "set_custom_data",
                        side_effect=lambda key, value, store=store: store.update({key: value}),
                    ),
                ):
                    before = self.model.adjust_trade_position(
                        position,
                        self.now,
                        favorable,
                        0.2,
                        1.0,
                        1000.0,
                        favorable,
                        unfavorable,
                        0.2,
                        0.2,
                    )
                    self.assertIsNone(before)
                    after = self.model.adjust_trade_position(
                        position,
                        self.now,
                        unfavorable,
                        0.2,
                        1.0,
                        1000.0,
                        unfavorable,
                        favorable,
                        0.2,
                        0.2,
                    )
                    stake_reduction = -100.0 * QuickAdapterV3.partial_exit_stages[0][1]
                    direction = "short" if short else "long"
                    self.assertEqual(after, (stake_reduction, f"take_profit_{direction}_0"))


class ThrottleCallbackTest(QaTestCase):
    def test_a_non_callable_is_refused(self):
        model = strategy()
        with self.assertRaisesRegex(ValueError, "must be callable"):
            model.throttle_callback(PAIR, datetime.datetime.now(datetime.UTC), "not callable")

    def test_the_callback_fires_once_per_candle(self):
        model = strategy(candle_secs=300)
        fired = []
        start = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.UTC)
        for minute in (0, 1, 4):
            model.throttle_callback(
                PAIR, start + datetime.timedelta(minutes=minute), lambda: fired.append(1)
            )
        self.assertEqual(len(fired), 1)

    def test_the_callback_fires_again_on_the_next_candle(self):
        model = strategy(candle_secs=300)
        fired = []
        start = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.UTC)
        for minute in (0, 4, 5):
            model.throttle_callback(
                PAIR, start + datetime.timedelta(minutes=minute), lambda: fired.append(1)
            )
        self.assertEqual(len(fired), 2)

    def test_a_failing_callback_does_not_stop_the_next_candle(self):
        model = strategy(candle_secs=300)
        start = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.UTC)

        def boom():
            raise RuntimeError("boom")

        model.throttle_callback(PAIR, start, boom)
        with self.assertLogs("QuickAdapterV3", level="ERROR"):
            model.throttle_callback(PAIR, start + datetime.timedelta(minutes=5), boom)
        self.assertEqual(len(model.last_candle_start_secs), 1)

    def test_the_throttle_key_is_the_bytecode_not_the_callable_identity(self):
        # get_callable_sha256 hashes __code__.co_code, so two functions with the same body
        # are one callback as far as the throttle is concerned, whatever their names.
        model = strategy(candle_secs=300)
        start = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.UTC)
        fired = []

        def first():
            fired.append("first")

        def second():
            fired.append("second")

        self.assertIsNot(first, second)
        model.throttle_callback(PAIR, start, first)
        model.throttle_callback(PAIR, start, second)
        self.assertEqual(fired, ["first"])
        self.assertEqual(len(model.last_candle_start_secs), 1)

    def test_callables_with_different_bodies_get_distinct_keys(self):
        model = strategy(candle_secs=300)
        start = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.UTC)
        fired = []

        def first():
            fired.append(1)
            return None

        def second():
            fired.append(2)
            return None

        model.throttle_callback(PAIR, start, first)
        model.throttle_callback(PAIR, start, second)
        self.assertEqual(fired, [1, 2])
        self.assertEqual(len(model.last_candle_start_secs), 2)

    def test_stale_keys_are_evicted_beyond_ten_candles(self):
        # Two callbacks are required: a single call can only ever produce one key, and the
        # eviction loop compares the triggering key — whose timestamp it just wrote —
        # against the rest of the dict, so with one key there is nothing to evict.
        #
        # They must differ by a SMALL-INT constant, not a string. get_callable_sha256
        # hashes co_code, which carries opcodes and opargs only and never the constant
        # VALUES: in CPython 3.14 a string constant is loaded through a constant-index
        # oparg, so `return "stale"` and `return "fresh"` compile to byte-identical
        # bytecode and share a key, while 3.14 inlines small ints into the oparg and
        # `append(1)` differs from `append(2)`.
        model = strategy(candle_secs=300)
        start = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.UTC)
        fired = []

        def stale():
            fired.append(1)

        def fresh():
            fired.append(2)

        model.throttle_callback(PAIR, start, stale)
        stale_key = next(iter(model.last_candle_start_secs))
        self.assertEqual(fired, [1])

        model.throttle_callback(PAIR, start, fresh)
        self.assertEqual(fired, [1, 2])
        self.assertEqual(len(model.last_candle_start_secs), 2)

        # Eleven candles on: the budget is ten * candle_duration_secs = 50 minutes at
        # 300-second candles, and 55 clears it. The key just written is fresh.
        later = start + datetime.timedelta(minutes=11 * 5)
        model.throttle_callback(PAIR, later, fresh)
        self.assertEqual(fired, [1, 2, 2])
        self.assertEqual(len(model.last_candle_start_secs), 1)
        self.assertNotIn(
            stale_key,
            model.last_candle_start_secs,
            "the stale callback's key survived eviction",
        )

    def test_pairs_are_throttled_independently(self):
        model = strategy(candle_secs=300)
        start = datetime.datetime(2026, 1, 1, 12, 0, tzinfo=datetime.UTC)

        def callback():
            return None

        model.throttle_callback("BTC/USDT", start, callback)
        model.throttle_callback("ETH/USDT", start, callback)
        self.assertEqual(len(model.last_candle_start_secs), 2)


if __name__ == "__main__":
    unittest.main()
