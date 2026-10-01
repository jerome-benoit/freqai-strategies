"""Provability ranking and per-candle deduplication of historic FreqAI predictions.; requires the Freqtrade QA image."""

import unittest
from typing import Any

import numpy as np
import pandas as pd
from freqtrade.freqai.data_drawer import FreqaiDataDrawer
from qa_support import PAIR, QaTestCase

import quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 as regressor
from quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 import (
    _align_historic_predictions,
    _dedupe_historic_predictions_on_date_pred,
    _ensure_produced_prediction_column,
    _legacy_produced_prediction_mask,
    _produced_prediction_mask,
    _recorded_prediction_rank,
)

MARKER = regressor._PRODUCED_PREDICTION_COLUMN
DATES = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")

# One expired (`do_predict == 2`) row per marker form. `np.nan` in the marker column is the
# "recorded but unlabelled" form; omitting the key drops the column entirely. The
# close-bearing row is the legacy shape, which carries a positive price instead of a marker.
_EXPIRED_FORMS: dict[str, dict[str, Any]] = {
    "explicit marker": {"close_price": [np.nan], MARKER: [True]},
    "null marker": {"close_price": [np.nan], MARKER: [np.nan]},
    "absent marker column": {"close_price": [np.nan]},
    "close-bearing legacy row": {"close_price": [100.0]},
}


def drawer_stub(history: pd.DataFrame) -> FreqaiDataDrawer:
    """Use native drawer methods without its disk and model initialization."""
    drawer = object.__new__(FreqaiDataDrawer)
    drawer.historic_predictions = {PAIR: history}
    drawer.model_return_values = {}
    return drawer


class RegressorPredictionDedupTest(QaTestCase):
    def test_rank_orders_proven_then_close_then_placeholder(self):
        frame = pd.DataFrame(
            {
                "date_pred": [DATES[0]] * 10,
                "do_predict": [1, 3, 0, 0, 2, 0, 0, 0, 0, 1],
                "close_price": [
                    np.nan,
                    np.nan,
                    100.0,
                    np.nan,
                    100.0,
                    np.nan,
                    0.0,
                    -100.0,
                    np.inf,
                    np.nan,
                ],
                MARKER: [
                    True,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    False,
                    np.nan,
                    np.nan,
                    np.nan,
                    False,
                ],
                "value": [float(i) for i in range(1, 11)],
            }
        )
        # The last row carries a nonzero status yet an explicit False marker: the recorded
        # status must not resurrect it, and the close bound must exclude zero, negatives and inf.
        self.assertEqual(_recorded_prediction_rank(frame).tolist(), [2, 2, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(
            _legacy_produced_prediction_mask(frame).tolist(),
            [True, True, False, False, False, False, False, False, False, True],
        )
        self.assertEqual(
            _produced_prediction_mask(frame).tolist(),
            [True, True, False, False, False, False, False, False, False, False],
        )

    def test_expired_status_is_never_produced_on_any_marker_form(self):
        for form, columns in _EXPIRED_FORMS.items():
            for status, produced in ((2, False), (1, True)):
                with self.subTest(form=form, do_predict=status):
                    frame = pd.DataFrame(
                        {"date_pred": DATES[:1], "do_predict": [status], **columns}
                    )
                    self.assertEqual(_produced_prediction_mask(frame).tolist(), [produced])
                    self.assertEqual(
                        _recorded_prediction_rank(frame).tolist(), [2 if produced else 0]
                    )
                    labelled = _ensure_produced_prediction_column(frame)
                    self.assertEqual(_produced_prediction_mask(labelled).tolist(), [produced])

    def test_ensure_column_keeps_an_explicit_marker_on_an_expired_row(self):
        frame = pd.DataFrame(
            {"date_pred": DATES[:1], "do_predict": [2], "close_price": [np.nan], MARKER: [True]}
        )
        self.assertEqual(_ensure_produced_prediction_column(frame)[MARKER].tolist(), [True])
        self.assertEqual(_produced_prediction_mask(frame).tolist(), [False])

    def test_dedup_applies_the_full_tier_order_to_every_candle(self):
        # Four candles, four distinct tier pairs, deliberately out of date order so the sort is
        # load-bearing. Three candles put the proven or close-bearing row before its rival, so the
        # rank must outrank the original position: a proven row beats downtime (D0), a close beats
        # downtime (D1), an explicit marker beats a close (D2), and a close beats an expired
        # model (D3). Each candle has one unambiguous winner.
        frame = pd.DataFrame(
            {
                "date_pred": [
                    DATES[2],
                    DATES[0],
                    DATES[3],
                    DATES[1],
                    DATES[1],
                    DATES[2],
                    DATES[0],
                    DATES[3],
                ],
                "do_predict": [1, 0, 0, 0, 0, 0, 1, 2],
                "close_price": [np.nan, np.nan, 100.0, 100.0, np.nan, 100.0, np.nan, np.nan],
                MARKER: [True, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }
        )
        self.assertEqual(_recorded_prediction_rank(frame).tolist(), [2, 0, 1, 1, 0, 1, 2, 0])
        result = _dedupe_historic_predictions_on_date_pred(frame, PAIR)
        self.assertEqual(result["date_pred"].tolist(), list(DATES))
        self.assertEqual(result["value"].tolist(), [7.0, 4.0, 1.0, 3.0])

    def test_dedup_keeps_one_row_per_date_in_ascending_order_with_last_write_wins(self):
        frame = pd.DataFrame(
            {
                "date_pred": [
                    DATES[2],
                    DATES[0],
                    DATES[2],
                    DATES[0],
                    DATES[1],
                    DATES[3],
                    DATES[3],
                ],
                "do_predict": [0] * 7,
                "close_price": [np.nan] * 7,
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            }
        )
        result = _dedupe_historic_predictions_on_date_pred(frame, PAIR)
        self.assertEqual(result["date_pred"].tolist(), list(DATES))
        self.assertTrue(result["date_pred"].is_unique)
        self.assertEqual(result["value"].tolist(), [4.0, 5.0, 3.0, 7.0])
        self.assertEqual(result.index.tolist(), [0, 1, 2, 3])

    def test_dedup_drops_unparseable_date_pred_with_a_warning(self):
        frame = pd.DataFrame(
            {
                "date_pred": [*DATES[:3], "not-a-date", None],
                "do_predict": [0] * 5,
                "value": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        with self.assertLogs(regressor.logger, level="WARNING") as captured:
            result = _dedupe_historic_predictions_on_date_pred(frame, PAIR)
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(
            captured.records[0].getMessage(),
            f"FreqAI prediction history [{PAIR}]: discarded invalid date_pred entries (count=2)",
        )
        self.assertEqual(result["date_pred"].tolist(), list(DATES[:3]))
        self.assertEqual(result["value"].tolist(), [1.0, 2.0, 3.0])

    def test_fast_path_returns_the_same_object_when_already_normalized(self):
        frame = pd.DataFrame(
            {
                "date_pred": list(DATES),
                "do_predict": [0, 1, 2, 0],
                "close_price": [np.nan] * 4,
            }
        )
        with self.assertNoLogs(regressor.logger, level="WARNING"):
            result = _dedupe_historic_predictions_on_date_pred(frame, PAIR)
        self.assertIs(result, frame)
        self.assertEqual(result["date_pred"].dtype, frame["date_pred"].dtype)

    def test_fast_path_corrects_a_timezone_naive_date_pred_column(self):
        naive = pd.DataFrame(
            {
                "date_pred": pd.DatetimeIndex(DATES).tz_localize(None),
                "do_predict": [0, 1, 2, 0],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        result = _dedupe_historic_predictions_on_date_pred(naive, PAIR)
        self.assertIsNot(result, naive)
        self.assertEqual(result["date_pred"].dtype, DATES.dtype)
        self.assertEqual(result["date_pred"].tolist(), list(DATES))
        self.assertEqual(naive["date_pred"].dtype, pd.DatetimeIndex(DATES).tz_localize(None).dtype)

    def test_slow_path_normalizes_surviving_dates_after_ranking(self):
        naive_dates = pd.DatetimeIndex(DATES).tz_localize(None)
        frame = pd.DataFrame(
            {
                # A duplicate forces the ranking slow path, where the survivors are rewritten
                # from the parsed UTC series rather than from the raw column.
                "date_pred": [naive_dates[0], naive_dates[0], naive_dates[1]],
                "do_predict": [0, 1, 0],
                "close_price": [np.nan] * 3,
                "value": [1.0, 2.0, 3.0],
            }
        )
        result = _dedupe_historic_predictions_on_date_pred(frame, PAIR)
        self.assertEqual(result["value"].tolist(), [2.0, 3.0])
        self.assertEqual(result["date_pred"].dtype, DATES.dtype)
        self.assertEqual(result["date_pred"].tolist(), [DATES[0], DATES[1]])

    def test_ensure_produced_prediction_column_copies_instead_of_mutating(self):
        without = pd.DataFrame({"date_pred": list(DATES), "do_predict": [1, 0, 2, 0]})
        result = _ensure_produced_prediction_column(without)
        self.assertIsNot(result, without)
        self.assertNotIn(MARKER, without.columns)
        self.assertEqual(result[MARKER].tolist(), [True, False, False, False])

        partial = pd.DataFrame(
            {
                "date_pred": list(DATES),
                "do_predict": [1, 0, 2, 0],
                MARKER: [True, np.nan, True, np.nan],
            }
        )
        filled = _ensure_produced_prediction_column(partial)
        self.assertIsNot(filled, partial)
        self.assertEqual(partial[MARKER].isna().tolist(), [False, True, False, True])
        self.assertEqual(filled[MARKER].tolist(), [True, False, True, False])

        complete = pd.DataFrame(
            {
                "date_pred": list(DATES),
                "do_predict": [1, 0, 2, 0],
                MARKER: [True, False, True, False],
            }
        )
        self.assertIs(_ensure_produced_prediction_column(complete), complete)

    def test_align_rebuilds_model_return_values_against_the_requested_candle_dates(self):
        history = pd.DataFrame(
            {
                "date_pred": [DATES[0], DATES[2]],
                "value": [10.0, 30.0],
                "do_predict": [1, 1],
                MARKER: [True, True],
            }
        )
        strategy = pd.DataFrame(
            {"date": list(DATES), "close": [1.0, 2.0, 3.0, 4.0]}, index=[7, 8, 9, 10]
        )
        aligned = _align_historic_predictions(history, strategy)
        self.assertEqual(aligned.index.tolist(), strategy.index.tolist())
        self.assertEqual(aligned["date_pred"].tolist(), strategy["date"].tolist())
        pd.testing.assert_series_equal(
            aligned["value"],
            pd.Series([10.0, np.nan, 30.0, np.nan], index=strategy.index, name="value"),
        )
        pd.testing.assert_series_equal(
            aligned["do_predict"],
            pd.Series([1.0, 0.0, 1.0, 0.0], index=strategy.index, name="do_predict"),
        )

        drawer = drawer_stub(history)
        merged = FreqaiDataDrawer.attach_return_values_to_return_dataframe(drawer, PAIR, strategy)
        pd.testing.assert_frame_equal(drawer.model_return_values[PAIR], aligned)
        self.assertEqual(merged["date"].tolist(), strategy["date"].tolist())
        pd.testing.assert_series_equal(
            merged["value"],
            pd.Series([10.0, np.nan, 30.0, np.nan], name="value"),
            check_names=False,
        )

    def test_align_zeroes_do_predict_when_the_history_carries_no_status_column(self):
        history = pd.DataFrame({"date_pred": [DATES[1]], "value": [20.0]})
        strategy = pd.DataFrame({"date": list(DATES)})
        aligned = _align_historic_predictions(history, strategy)
        self.assertEqual(aligned["do_predict"].tolist(), [0, 0, 0, 0])
        self.assertEqual(aligned["value"].isna().tolist(), [True, False, True, True])

    def test_align_rejects_a_duplicate_date_pred(self):
        duplicated = pd.DataFrame(
            {"date_pred": [DATES[0], DATES[0]], "value": [1.0, 2.0], "do_predict": [0, 1]}
        )
        strategy = pd.DataFrame({"date": list(DATES)})
        with self.assertRaisesRegex(ValueError, "duplicate labels"):
            _align_historic_predictions(duplicated, strategy)

    def test_patched_attach_deduplicates_before_it_aligns(self):
        duplicated = pd.DataFrame(
            {
                "date_pred": [DATES[0], DATES[0], DATES[2]],
                "value": [10.0, 11.0, 30.0],
                "do_predict": [0, 1, 1],
            }
        )
        strategy = pd.DataFrame({"date": list(DATES)})
        drawer = drawer_stub(duplicated)
        FreqaiDataDrawer.attach_return_values_to_return_dataframe(drawer, PAIR, strategy)
        repaired = drawer.historic_predictions[PAIR]
        self.assertTrue(repaired["date_pred"].is_unique)
        self.assertEqual(repaired["value"].tolist(), [11.0, 30.0])
        self.assertEqual(drawer.model_return_values[PAIR].index.tolist(), strategy.index.tolist())

    def test_patched_attach_raises_key_error_without_a_date_column(self):
        history = pd.DataFrame(
            {"date_pred": [DATES[0]], "value": [10.0], "do_predict": [1]},
        )
        drawer = drawer_stub(history)
        strategy = pd.DataFrame({"date": list(DATES)}).drop(columns=["date"])
        with self.assertRaises(KeyError) as caught:
            FreqaiDataDrawer.attach_return_values_to_return_dataframe(drawer, PAIR, strategy)
        self.assertEqual(caught.exception.args[0], "date")
        self.assertNotIn(PAIR, drawer.model_return_values)


if __name__ == "__main__":
    unittest.main()
