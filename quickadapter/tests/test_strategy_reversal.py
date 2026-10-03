"""Reversal confirmation gate contracts; requires the Freqtrade QA image."""

import unittest

import numpy as np
import pandas as pd
from qa_support import QaTestCase
from QuickAdapterV3 import QuickAdapterV3

LONG = QuickAdapterV3._TRADE_LONG
SHORT = QuickAdapterV3._TRADE_SHORT
ENTRY = QuickAdapterV3._ORDER_ENTRY
EXIT = QuickAdapterV3._ORDER_EXIT
PAIR = "BTC/USDT"
NAN = float("nan")


def strategy(thresholds):
    """A strategy whose threshold collaborator replays a scripted sequence and records its inputs.

    The threshold's own arithmetic is covered alongside the candle caches; here it is a
    collaborator, so each case states the threshold it reasons about and the recorded calls
    let the decay schedule be asserted.
    """
    model = object.__new__(QuickAdapterV3)
    state = {"pending": list(thresholds), "calls": []}

    def threshold(
        df, pair, side, *, min_natr_multiplier_fraction, max_natr_multiplier_fraction, candle_idx
    ):
        pending = state["pending"]
        value = pending.pop(0) if pending else NAN
        state["calls"].append(
            {
                "candle_idx": candle_idx,
                "min": min_natr_multiplier_fraction,
                "max": max_natr_multiplier_fraction,
            }
        )
        return value

    model._calculate_candle_threshold = threshold
    return model, state


def frame(closes):
    return pd.DataFrame({"close": np.array(closes, dtype=float)})


class ReversalConfirmationTest(QaTestCase):
    def _confirm(self, model, df, *, side, order, rate, lookback=0, decay=1.0, lo=0.5, hi=1.0):
        return model.reversal_confirmed(
            df,
            PAIR,
            side,
            order,
            rate,
            lookback_period_candles=lookback,
            decay_fraction=decay,
            min_natr_multiplier_fraction=lo,
            max_natr_multiplier_fraction=hi,
        )

    # --- the current candle alone -------------------------------------------------
    def test_a_long_entry_breaks_strictly_upwards(self):
        for rate, expected in ((101.0, True), (100.0, False), (99.0, False)):
            with self.subTest(rate=rate):
                model, _ = strategy([100.0])
                self.assertIs(
                    self._confirm(model, frame([100.0] * 3), side=LONG, order=ENTRY, rate=rate),
                    expected,
                )

    def test_a_short_entry_breaks_strictly_downwards(self):
        for rate, expected in ((99.0, True), (100.0, False), (101.0, False)):
            with self.subTest(rate=rate):
                model, _ = strategy([100.0])
                self.assertIs(
                    self._confirm(model, frame([100.0] * 3), side=SHORT, order=ENTRY, rate=rate),
                    expected,
                )

    # --- refusals ----------------------------------------------------------------
    def test_an_empty_frame_fails_closed(self):
        model, _ = strategy([100.0])
        self.assertFalse(self._confirm(model, frame([]), side=LONG, order=ENTRY, rate=101.0))

    def test_an_unknown_side_or_order_fails_closed(self):
        for side, order in (("sideways", ENTRY), (LONG, "adjust"), (None, ENTRY)):
            with self.subTest(side=side, order=order):
                model, _ = strategy([100.0])
                self.assertFalse(
                    self._confirm(model, frame([100.0] * 3), side=side, order=order, rate=101.0)
                )

    def test_a_non_finite_or_non_numeric_rate_fails_closed(self):
        for rate in (np.nan, np.inf, -np.inf, "101.0", None):
            with self.subTest(rate=rate):
                model, _ = strategy([100.0])
                self.assertFalse(
                    self._confirm(model, frame([100.0] * 3), side=LONG, order=ENTRY, rate=rate)
                )

    def test_an_inverted_multiplier_range_fails_closed(self):
        for lo, hi in ((1.0, 0.5), (-0.1, 1.0), (np.nan, 1.0), (0.5, np.inf)):
            with self.subTest(lo=lo, hi=hi):
                model, _ = strategy([100.0])
                self.assertFalse(
                    self._confirm(
                        model, frame([100.0] * 3), side=LONG, order=ENTRY, rate=101.0, lo=lo, hi=hi
                    )
                )

    def test_a_decay_fraction_outside_the_half_open_range_fails_closed(self):
        for decay in (0.0, -0.1, 1.0001, "0.5"):
            with self.subTest(decay=decay):
                model, _ = strategy([100.0])
                self.assertFalse(
                    self._confirm(
                        model, frame([100.0] * 3), side=LONG, order=ENTRY, rate=101.0, decay=decay
                    )
                )

    # --- lookback depth ----------------------------------------------------------
    def test_an_entry_lookback_deeper_than_the_frame_fails_closed(self):
        model, _ = strategy([100.0])
        self.assertFalse(
            self._confirm(model, frame([100.0] * 3), side=LONG, order=ENTRY, rate=101.0, lookback=5)
        )

    def test_an_exit_lookback_deeper_than_the_frame_is_clamped_to_the_frame(self):
        # Three rows allow a lookback of two at most, so the gate runs k=1 and k=2 rather
        # than refusing, and the deeper request never reaches an unmeasurable candle.
        model, state = strategy([100.0, 90.0, 90.0, 90.0, 90.0])
        self.assertTrue(
            self._confirm(
                model, frame([100.0, 100.0, 100.0]), side=LONG, order=EXIT, rate=101.0, lookback=5
            )
        )
        self.assertEqual([call["candle_idx"] for call in state["calls"]], [-1, -2, -3])

    # --- the chain ----------------------------------------------------------------
    def test_a_chain_breaking_at_the_first_history_candle_refuses_both_orders(self):
        for order in (ENTRY, EXIT):
            with self.subTest(order=order):
                model, _ = strategy([100.0, 150.0, 150.0])
                self.assertFalse(
                    self._confirm(
                        model,
                        frame([100.0, 100.0, 100.0, 101.0]),
                        side=LONG,
                        order=order,
                        rate=101.0,
                        lookback=2,
                    )
                )

    def test_a_chain_breaking_at_the_second_history_candle_refuses_both_orders(self):
        for order in (ENTRY, EXIT):
            with self.subTest(order=order):
                model, _ = strategy([100.0, 90.0, 150.0, 150.0])
                self.assertFalse(
                    self._confirm(
                        model,
                        frame([100.0, 100.0, 100.0, 101.0]),
                        side=LONG,
                        order=order,
                        rate=101.0,
                        lookback=2,
                    )
                )

    def test_a_chain_breaking_throughout_confirms(self):
        model, _ = strategy([100.0, 90.0, 90.0, 90.0])
        self.assertTrue(
            self._confirm(
                model,
                frame([100.0, 100.0, 100.0, 101.0]),
                side=LONG,
                order=ENTRY,
                rate=101.0,
                lookback=2,
            )
        )

    def test_a_short_chain_must_break_downwards_at_every_step(self):
        # A short breaks downwards, so a LOW threshold is the hard case: close(-1) = 99.0
        # cannot get below 90.0 and the chain breaks at the first history candle.
        model, _ = strategy([100.0, 90.0, 90.0])
        self.assertFalse(
            self._confirm(
                model,
                frame([100.0, 100.0, 100.0, 99.0]),
                side=SHORT,
                order=ENTRY,
                rate=99.0,
                lookback=2,
            )
        )

    # --- unmeasurable history: entry closes, exit degrades open -------------------
    def test_a_non_finite_history_close_fails_an_entry_closed(self):
        model, _ = strategy([100.0, 90.0, 90.0])
        self.assertFalse(
            self._confirm(
                model,
                frame([100.0, 100.0, 100.0, np.nan]),
                side=LONG,
                order=ENTRY,
                rate=101.0,
                lookback=2,
            )
        )

    def test_a_non_finite_history_close_keeps_an_exit_degraded_open(self):
        model, _ = strategy([100.0, 90.0, 90.0])
        self.assertTrue(
            self._confirm(
                model,
                frame([100.0, 100.0, 100.0, np.nan]),
                side=LONG,
                order=EXIT,
                rate=101.0,
                lookback=2,
            )
        )

    def test_a_non_finite_history_threshold_keeps_an_exit_degraded_open(self):
        for order, expected in ((EXIT, True), (ENTRY, False)):
            with self.subTest(order=order):
                model, _ = strategy([100.0, np.nan, 90.0])
                self.assertIs(
                    self._confirm(
                        model,
                        frame([100.0, 100.0, 100.0, 101.0]),
                        side=LONG,
                        order=order,
                        rate=101.0,
                        lookback=2,
                    ),
                    expected,
                )

    def test_degradation_never_manufactures_a_confirmation_of_a_broken_current_candle(self):
        model, _ = strategy([100.0, np.nan])
        self.assertFalse(
            self._confirm(
                model,
                frame([100.0, 100.0, 100.0, 101.0]),
                side=LONG,
                order=EXIT,
                rate=99.0,
                lookback=2,
            )
        )

    # --- the decay schedule -------------------------------------------------------
    def test_the_multiplier_bounds_decay_geometrically_and_clamp_to_one(self):
        model, state = strategy([100.0, 90.0, 90.0])
        self._confirm(
            model,
            frame([100.0, 100.0, 100.0, 101.0]),
            side=LONG,
            order=ENTRY,
            rate=101.0,
            lookback=2,
            decay=0.5,
            lo=0.8,
            hi=1.0,
        )
        self.assertEqual([call["candle_idx"] for call in state["calls"]], [-1, -2, -3])
        self.assertAlmostEqual(state["calls"][0]["min"], 0.8)
        self.assertAlmostEqual(state["calls"][0]["max"], 1.0)
        self.assertAlmostEqual(state["calls"][1]["min"], 0.8 * 0.5)
        self.assertAlmostEqual(state["calls"][2]["min"], 0.8 * 0.25)

    def test_a_decayed_minimum_never_falls_below_zero(self):
        model, state = strategy([100.0, 90.0, 90.0])
        self._confirm(
            model,
            frame([100.0, 100.0, 100.0, 101.0]),
            side=LONG,
            order=ENTRY,
            rate=101.0,
            lookback=2,
            decay=0.1,
            lo=0.001,
            hi=1.0,
        )
        for call in state["calls"][1:]:
            self.assertGreaterEqual(call["min"], 0.0)
            self.assertLessEqual(call["max"], 1.0)

    def test_a_decayed_maximum_never_falls_below_the_decayed_minimum(self):
        model, state = strategy([100.0, 90.0, 90.0])
        self._confirm(
            model,
            frame([100.0, 100.0, 100.0, 101.0]),
            side=LONG,
            order=ENTRY,
            rate=101.0,
            lookback=2,
            decay=0.5,
            lo=1.0,
            hi=1.0,
        )
        # The ORDERING is a consequence of the argument validation above (lo <= hi is refused
        # earlier), so `max >= min` holds for any input and proves nothing about the decay.
        # What is actually observable is the historical MAXIMUM decaying alongside the
        # minimum. Measured with these arguments, max is [0.5, 0.25] over the two follow-up
        # calls; dropping the decay on the max gives [1.0, 1.0] and squaring the factor does
        # too, and both survived the suite green.
        for call in state["calls"]:
            self.assertGreaterEqual(call["max"], call["min"])
        self.assertAlmostEqual(state["calls"][0]["max"], 1.0)
        self.assertAlmostEqual(state["calls"][1]["max"], 0.5)
        self.assertAlmostEqual(state["calls"][2]["max"], 0.25)


if __name__ == "__main__":
    unittest.main()
