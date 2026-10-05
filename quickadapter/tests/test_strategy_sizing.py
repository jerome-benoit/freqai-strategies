"""Trade callbacks, position sizing and throttle contracts; requires the Freqtrade QA image."""

import copy
import datetime
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from freqtrade.enums import RunMode
from freqtrade.persistence import Order, Trade
from qa_support import PAIR, QaTestCase, model_config, temporary_directory
from QuickAdapterV3 import QuickAdapterV3
from Utils import EXTREMA_COLUMN


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
            exit_pricing={
                "trade_natr_method": "quantile_interpolation",
                "final_take_profit_retracement_fraction": 0.25,
            },
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
                "do_predict": [1] * 3,
                "DI_catch": [1] * 3,
                EXTREMA_COLUMN: [0.0] * 3,
                "minima_threshold": [-0.5] * 3,
                "maxima_threshold": [0.5] * 3,
            }
        )
        self.model.dp = SimpleNamespace(
            get_analyzed_dataframe=lambda **kwargs: (self.frame, None),
            _exchange=SimpleNamespace(
                get_min_pair_stake_amount=lambda *args: 1.0,
                amount_to_contract_precision=lambda pair, amount: amount,
            ),
        )

    def position(self, short=False, exit_stage=0):
        position = Trade(
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
        position.orders = [
            Order(
                ft_order_side=position.exit_side,
                ft_is_open=False,
                status="closed",
                filled=0.1,
                ft_order_tag=f"take_profit_{position.trade_direction}_{stage}",
            )
            for stage in range(exit_stage)
        ]
        return position

    def custom_data(self, position, store=None):
        # Isolate database persistence only; all trade/exit calculations remain native.
        store = {} if store is None else store
        self.enterContext(
            mock.patch.object(
                position,
                "get_custom_data",
                side_effect=lambda key, default=None: store.get(key, default),
            )
        )
        self.enterContext(
            mock.patch.object(
                position,
                "set_custom_data",
                side_effect=lambda key, value: store.update({key: value}),
            )
        )
        return store

    def exit(self, position, rate=100.0):
        return self.model.custom_exit(PAIR, position, self.now, rate, 0.0)

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
        for short in (False, True):
            with self.subTest(short=short):
                position = self.position(short)
                target = self.model.get_take_profit_target(self.frame, position, 0)[0]
                # Priced off the resolved target so the ladder remains a free parameter.
                crossed = target + (-1.0 if short else 1.0)
                below = position.open_rate
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
                        crossed,
                        0.2,
                        1.0,
                        1000.0,
                        crossed,
                        below,
                        0.2,
                        0.2,
                    )
                    self.assertIsNone(before)
                    after = self.model.adjust_trade_position(
                        position,
                        self.now,
                        below,
                        0.2,
                        1.0,
                        1000.0,
                        below,
                        crossed,
                        0.2,
                        0.2,
                    )
                    # Four exits release the position in equal shares, so the
                    # first one closes a quarter of the 100 stake.
                    stake_reduction = -25.0
                    direction = "short" if short else "long"
                    self.assertEqual(after, (stake_reduction, f"take_profit_{direction}_0"))

    def _walk_partial_stages(self, position, stage_count):
        """Cross each partial target in turn and return what every stage released."""
        released = []
        store = {}
        # Isolate only the custom-data store; targets and sizing stay native.
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
            for stage in range(stage_count):
                target = self.model.get_take_profit_target(self.frame, position, stage)[0]
                direction = -1.0 if position.is_short else 1.0
                reduction, _ = self.model.adjust_trade_position(
                    position,
                    self.now,
                    target + direction,
                    0.2,
                    1.0,
                    1000.0,
                    target + direction,
                    target + direction,
                    0.2,
                    0.2,
                )
                self.assertIsNotNone(reduction, f"stage {stage} did not exit at its target")
                released.append(-reduction)
                # Only the open stake and the filled-exit count advance here.
                position.stake_amount += reduction
                position.orders.append(
                    Order(
                        ft_order_side=position.exit_side,
                        ft_is_open=False,
                        status="closed",
                        filled=1.0,
                        ft_order_tag=f"take_profit_{position.trade_direction}_{stage}",
                    )
                )
        return released

    def test_every_exit_releases_an_equal_share_of_the_initial_stake(self):
        # The ladder splits the initial stake into equal absolute amounts, and
        # the final full exit closes the last one.
        initial_stake = 100.0
        stage_count = len(QuickAdapterV3.exit_stages)
        for short in (False, True):
            with self.subTest(short=short):
                position = self.position(short)
                released = self._walk_partial_stages(position, stage_count - 1)
                for released_stake in released:
                    self.assertAlmostEqual(
                        released_stake, initial_stake / stage_count, places=9, msg=str(released)
                    )
                self.assertAlmostEqual(position.stake_amount, initial_stake / stage_count, places=9)

    def test_the_stake_shares_resize_with_the_stage_count(self):
        initial_stake = 100.0
        for stage_count in (2, 3, 5):
            with self.subTest(stage_count=stage_count):
                stages = dict.fromkeys(range(stage_count), "lime")
                with mock.patch.object(QuickAdapterV3, "exit_stages", stages):
                    # A fresh model so its derived ladders follow the new count.
                    model = QuickAdapterV3(self.model.config)
                    model.freqai_info = self.model.config["freqai"]
                    model.bot_start()
                    model.dp = self.model.dp
                    previous_model = self.model
                    self.model = model
                    for short in (False, True):
                        position = self.position(short)
                        released = self._walk_partial_stages(position, stage_count - 1)
                        self.assertEqual(len(released), stage_count - 1)
                        for released_stake in released:
                            self.assertAlmostEqual(
                                released_stake,
                                initial_stake / stage_count,
                                places=9,
                                msg=f"{stage_count} stages: {released}",
                            )
                        self.assertAlmostEqual(
                            position.stake_amount, initial_stake / stage_count, places=9
                        )
                    self.model = previous_model

    def test_the_final_stage_takes_the_remainder_the_partials_left(self):
        stage_count = len(QuickAdapterV3.exit_stages)
        position = self.position()
        released = self._walk_partial_stages(position, stage_count - 1)
        self.assertEqual(QuickAdapterV3.get_trade_exit_stage(position), stage_count - 1)
        # Past the last partial rung the ladder must stop requesting reductions.
        self.assertIsNone(
            self.model.adjust_trade_position(
                position,
                self.now,
                1e9,
                0.2,
                1.0,
                1000.0,
                position.open_rate,
                1e9,
                0.2,
                0.2,
            )
        )
        # The final exit closes whatever remains, one share of the initial stake.
        self.assertAlmostEqual(position.stake_amount, 100.0 / stage_count, places=9)
        self.assertEqual(len(released), stage_count - 1)

    def test_expired_models_exit_before_order_stage_and_date_guards(self):
        self.frame.loc[self.frame.index[-1], "date"] = pd.NaT
        self.frame["do_predict"] = 2
        self.frame["DI_catch"] = 0
        for short in (False, True):
            with self.subTest(short=short):
                position = self.position(short)
                position.orders.append(
                    Order(
                        ft_order_side=position.exit_side, ft_is_open=True, status="open", filled=0.0
                    )
                )
                store = self.custom_data(position, {"n_outliers": 17})
                self.assertEqual(self.exit(position), "model_expired")
                self.assertEqual(store, {"n_outliers": 17})

    def test_outliers_count_once_per_valid_candle(self):
        position = self.position()
        store = self.custom_data(position, {"last_outlier_date": "not-a-date"})
        self.frame["DI_catch"] = 0
        self.assertIsNone(self.exit(position))
        self.assertIsNone(self.exit(position))
        self.assertEqual(store["n_outliers"], 1)
        self.assertEqual(store["last_outlier_date"], self.now.isoformat())
        self.now += datetime.timedelta(minutes=5)
        self.frame.loc[self.frame.index[-1], "date"] = self.now
        self.assertIsNone(self.exit(position))
        self.assertEqual(store["n_outliers"], 2)
        self.assertEqual(store["last_outlier_date"], self.now.isoformat())
        self.frame.loc[self.frame.index[-1], "date"] = pd.NaT
        self.assertIsNone(self.exit(position))
        self.assertEqual(store["n_outliers"], 2)

    def test_reversal_exits_require_strict_scores_and_real_price_confirmation(self):
        self.model.reversal_confirmation.update(
            {
                "lookback_period_candles": 0,
                "decay_fraction": 1.0,
                "min_natr_multiplier_fraction": 0.5,
                "max_natr_multiplier_fraction": 0.5,
            }
        )
        for short, boundary, score, rate, tag in (
            (False, 0.5, 0.6, 80.0, "maxima_detected_long"),
            (True, -0.5, -0.6, 120.0, "minima_detected_short"),
        ):
            with self.subTest(short=short):
                position = self.position(short)
                self.custom_data(position)
                self.frame[EXTREMA_COLUMN] = boundary
                self.assertIsNone(self.exit(position, rate))
                self.frame[EXTREMA_COLUMN] = score
                self.assertIsNone(self.exit(position, 100.0))
                for prediction, inlier in ((0, 1), (1, 0)):
                    self.frame["do_predict"] = prediction
                    self.frame["DI_catch"] = inlier
                    self.assertIsNone(self.exit(position, rate))
                self.frame["do_predict"] = 1
                self.frame["DI_catch"] = 1
                self.assertEqual(self.exit(position, rate), tag)

    def test_final_exit_guards_preserve_unarmed_state(self):
        original = self.frame.copy()
        for guard in ("empty", "open_order", "partial_stage", "invalid_date"):
            with self.subTest(guard=guard):
                self.frame = original.copy()
                position = self.position(exit_stage=0 if guard == "partial_stage" else 3)
                store = self.custom_data(position)
                if guard == "empty":
                    self.frame = self.frame.iloc[:0]
                elif guard == "open_order":
                    position.orders.append(
                        Order(
                            ft_order_side=position.exit_side,
                            ft_is_open=True,
                            status="open",
                            filled=0.0,
                        )
                    )
                elif guard == "invalid_date":
                    self.frame.loc[self.frame.index[-1], "date"] = pd.NaT
                self.assertIsNone(self.exit(position, 150.0))
                self.assertEqual(store, {})

    def test_final_exit_trails_survive_restart_and_confirm_on_a_later_candle(self):
        key = QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_KEY
        for short, target, arm, best, retrace, tag in (
            (False, 121.0, 130.0, 135.0, 110.0, "take_profit_long_final"),
            (True, 79.0, 70.0, 65.0, 90.0, "take_profit_short_final"),
        ):
            with self.subTest(short=short):
                self.now = datetime.datetime(2026, 1, 1, 0, 10, tzinfo=datetime.UTC)
                self.frame.loc[self.frame.index[-1], "date"] = self.now
                position = self.position(short, exit_stage=3)
                store = self.custom_data(position)
                # Entry NATR rank gives 7%; final fraction 1 and one-candle factor 1.
                np.testing.assert_allclose(
                    self.model.get_take_profit_target(self.frame, position, 3),
                    (target, 21.0),
                    rtol=1e-12,
                    atol=0.0,
                )
                self.assertIsNone(self.exit(position, 100.0))
                self.assertNotIn(key, store)
                self.assertEqual(store["history"]["take_profit_price"], [(3, target)])
                self.assertIsNone(self.exit(position, arm))
                self.assertEqual(store[key]["best_rate"], arm)
                self.assertAlmostEqual(store[key]["retracement_distance"], 5.25, places=12)
                self.now += datetime.timedelta(minutes=5)
                self.frame.loc[self.frame.index[-1], "date"] = self.now
                self.assertIsNone(self.exit(position, best))
                self.assertEqual(store[key]["best_rate"], best)
                provider = self.model.dp
                config = copy.deepcopy(self.model.config)
                self.model = QuickAdapterV3(config)
                self.model.freqai_info = config["freqai"]
                self.model.bot_start()
                self.model.dp = provider
                position = self.position(short, exit_stage=3)
                self.custom_data(position, store)
                self.assertIsNone(self.exit(position, retrace))
                self.assertEqual(store[key]["best_rate"], best)
                self.now += datetime.timedelta(minutes=5)
                self.frame.loc[self.frame.index[-1], "date"] = self.now
                self.assertEqual(self.exit(position, retrace), tag)

    def test_invalid_final_exit_state_is_cleared_then_rearmed(self):
        key = QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_KEY
        position = self.position(exit_stage=3)
        store = self.custom_data(position, {key: {"best_rate": "corrupt"}})
        self.assertIsNone(self.exit(position, 100.0))
        self.assertIsNone(store[key])
        self.assertIsNone(self.exit(position, 130.0))
        self.assertEqual(store[key]["best_rate"], 130.0)
        self.assertAlmostEqual(store[key]["retracement_distance"], 5.25, places=12)

    def test_an_unmeasurable_final_trail_does_not_arm_or_exit(self):
        position = self.position(short=True, exit_stage=3)
        store = self.custom_data(position)
        # Target 79 is crossed, but a zero current rate cannot seed a positive trail.
        self.assertIsNone(self.exit(position, 0.0))
        self.assertNotIn(QuickAdapterV3._FINAL_TAKE_PROFIT_STATE_KEY, store)
        self.assertEqual(store["history"]["take_profit_price"], [(3, 79.0)])

    def test_a_missing_final_take_profit_target_leaves_state_unarmed(self):
        position = self.position(exit_stage=3)
        store = self.custom_data(position)
        self.frame["natr_label_period_candles"] = np.nan
        self.assertIsNone(self.exit(position, 150.0))
        self.assertEqual(store, {})


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
