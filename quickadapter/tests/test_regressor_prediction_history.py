"""Calibration warmup arithmetic, DI cutoff fallbacks and throttling; requires the Freqtrade QA image."""

import random
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import scipy as sp
from freqtrade.freqai.freqai_interface import IFreqaiModel
from qa_support import PAIR, QaTestCase
from Utils import _OPTUNA_NAMESPACES, DEFAULTS_LABEL_PIPELINE

from quickadapter.user_data.freqaimodels import QuickAdapterRegressorV3 as regressor_module
from quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 import (
    _PRODUCED_PREDICTION_COLUMN,
    QuickAdapterRegressorV3,
    _log_known_at_none_once,
)

LABEL = "&s-extrema"
CALIBRATION_START_KEY = QuickAdapterRegressorV3._CALIBRATION_START_KEY
DI_CUTOFF_DEFAULT = QuickAdapterRegressorV3._DI_CUTOFF_DEFAULT
# The normalized label range the canonical pipeline declares; the cold-start
# sentinels must sit strictly outside it or they would trigger on every candle.
LABEL_RANGE = DEFAULTS_LABEL_PIPELINE["minmax_range"]
TIMEFRAME = "5m"
CANDLE_FREQ = "5min"


def regressor(**attrs: object) -> QuickAdapterRegressorV3:
    """Build a regressor without running __init__, injecting only what the code under test reads."""
    model = object.__new__(QuickAdapterRegressorV3)
    model.live = True
    model.config = {"timeframe": TIMEFRAME}
    model.pairs = [PAIR]
    model._calibration_current_candles = {}
    model._optuna_label_params = {}
    model._optuna_hp_params = {}
    model._holdout_rmse = {}
    model._session_fitted_pairs = {PAIR}
    model._optuna_hyperopt = False
    model.ft_params = {"label_period_candles": 1}
    model._label_defaults = (1, 5.0)
    model._optuna_label_candles = {PAIR: 0}
    model._optuna_label_candle = {PAIR: 0}
    model._optuna_label_incremented_pairs = []
    for name, value in attrs.items():
        setattr(model, name, value)
    return model


def drawer(extras: dict[str, str] | None = None, shared: dict | None = None) -> SimpleNamespace:
    """A data drawer stub whose pair entries share one extras dict, as Freqtrade's shallow copy does."""
    empty = {
        "model_filename": "",
        "trained_timestamp": 0,
        "data_path": "",
        "extras": {} if shared is None else shared,
    }
    pair_dict: dict[str, dict] = {
        PAIR: {**empty, "extras": {} if extras is None else dict(extras)},
    }

    def get_pair_dict_info(pair: str) -> tuple[str, int]:
        pair_dict.setdefault(pair, {**empty})
        return pair_dict[pair]["model_filename"], pair_dict[pair]["trained_timestamp"]

    return SimpleNamespace(
        pair_dict=pair_dict,
        get_pair_dict_info=get_pair_dict_info,
        save_drawer_to_disk=mock.Mock(),
        historic_predictions={},
        model_return_values={},
        empty_pair_dict=empty,
    )


def history(
    dates: list[str],
    produced: list[bool],
    values: list[float] | None = None,
    di: list[float] | None = None,
    holdout: list[float] | None = None,
) -> pd.DataFrame:
    """Build a persisted prediction history; ``produced`` is the explicit marker column."""
    index = pd.RangeIndex(len(dates))
    frame = pd.DataFrame(
        {
            "date_pred": pd.to_datetime(pd.Series(dates), utc=True),
            _PRODUCED_PREDICTION_COLUMN: pd.Series(produced, dtype=object),
            LABEL: pd.Series([0.0] * len(dates) if values is None else values, dtype="float64"),
        },
        index=index,
    )
    if di is not None:
        frame["DI_values"] = pd.Series(di, dtype="float64")
    if holdout is not None:
        frame["holdout_rmse"] = pd.Series(holdout, dtype="float64")
    return frame


def kitchen() -> SimpleNamespace:
    return SimpleNamespace(
        label_list=[LABEL],
        unique_class_list=[],
        data={"extra_returns_per_train": {}},
        full_df=pd.DataFrame(),
    )


def calibrated(
    candles: int = 2,
    di: list[float] | None = None,
    holdout: list[float] | None = None,
    budget: int | None = None,
) -> tuple[QuickAdapterRegressorV3, SimpleNamespace, list[str]]:
    """A live regressor with `candles` produced observations, ready for fit_live_predictions."""
    dates = pd.date_range("2026-01-01", periods=candles, freq=CANDLE_FREQ, tz="UTC")
    frame = history(
        [d.isoformat() for d in dates],
        [True] * candles,
        values=[0.25, -0.25, 0.5, -0.5][:candles],
        di=di,
        holdout=holdout,
    )
    model = regressor(
        _fit_live_predictions_candles=candles if budget is None else budget,
        label_prediction={
            "default": {
                "method": "thresholding",
                "selection_method": "rank_extrema",
                "threshold_method": "mean",
                "outlier_quantile": 0.999,
                "soft_extremum_alpha": 12.0,
                "keep_fraction": 0.0075,
            },
            "columns": {},
        },
        min_max_pred=lambda *args, **kwargs: (-0.4, 0.6),
    )
    model.dd = drawer()
    model.dd.historic_predictions[PAIR] = frame
    return model, kitchen(), [d.isoformat() for d in dates]


class RegressorPredictionHistoryTest(QaTestCase):
    # ------------------------------------------------------------------ calibration warmup

    def test_bootstrap_only_history_restarts_warmup_cold_at_the_decision_time(self):
        dates = ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:10"]
        frame = history(dates, [False, False, False])
        model = regressor(dd=drawer())
        model._calibration_current_candles[PAIR] = pd.Timestamp("2026-01-01 00:10", tz="UTC")

        eligible = model._calibration_history(model.dd, PAIR, frame, 2)

        self.assertTrue(eligible.empty)
        self.assertEqual(
            model.dd.pair_dict[PAIR]["extras"][CALIBRATION_START_KEY],
            "2026-01-01T00:10:00+00:00",
        )

    def test_boundary_persisted_in_the_future_is_discarded_and_warmup_restarts_cold(self):
        # A strategy candle that predates every saved prediction: nothing was provably
        # produced at or before the decision time, so the warmup must restart there
        # rather than trust a boundary that claims observations from the future.
        dates = ["2026-01-01 00:30", "2026-01-01 00:35", "2026-01-01 00:40"]
        frame = history(dates, [True, True, True])
        model = regressor(
            dd=drawer(extras={CALIBRATION_START_KEY: "2026-01-01T01:00:00+00:00"}),
        )
        decision = pd.Timestamp("2026-01-01 00:10", tz="UTC")
        model._calibration_current_candles[PAIR] = decision

        eligible = model._calibration_history(model.dd, PAIR, frame, 2)

        self.assertTrue(eligible.empty)
        self.assertEqual(
            model.dd.pair_dict[PAIR]["extras"][CALIBRATION_START_KEY],
            "2026-01-01T00:10:00+00:00",
        )

    def test_a_future_boundary_is_rebuilt_from_provable_rows_not_from_the_future(self):
        dates = ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:10"]
        frame = history(dates, [False, True, True])
        model = regressor(
            dd=drawer(extras={CALIBRATION_START_KEY: "2026-01-01T01:00:00+00:00"}),
        )
        model._calibration_current_candles[PAIR] = pd.Timestamp("2026-01-01 00:10", tz="UTC")

        eligible = model._calibration_history(model.dd, PAIR, frame, 2)

        self.assertEqual(
            model.dd.pair_dict[PAIR]["extras"][CALIBRATION_START_KEY],
            "2026-01-01T00:05:00+00:00",
        )
        self.assertEqual(
            eligible["date_pred"].tolist(),
            [
                pd.Timestamp("2026-01-01 00:05", tz="UTC"),
                pd.Timestamp("2026-01-01 00:10", tz="UTC"),
            ],
        )

    def test_only_produced_rows_inside_the_persisted_boundary_are_warmup(self):
        dates = [
            "2026-01-01 00:00",  # produced, before the persisted start
            "2026-01-01 00:05",  # produced, exactly on the persisted start
            "2026-01-01 00:10",  # bootstrap placeholder inside the window
            "2026-01-01 00:15",  # produced
            "2026-01-01 00:20",  # produced, at the decision time
            "2026-01-01 00:25",  # produced, after the decision time
        ]
        frame = history(dates, [True, True, False, True, True, True])
        model = regressor(
            dd=drawer(extras={CALIBRATION_START_KEY: "2026-01-01T00:05:00+00:00"}),
        )
        model._calibration_current_candles[PAIR] = pd.Timestamp("2026-01-01 00:20", tz="UTC")

        eligible = model._calibration_history(model.dd, PAIR, frame, 4)

        self.assertEqual(
            eligible["date_pred"].tolist(),
            [
                pd.Timestamp("2026-01-01 00:05", tz="UTC"),
                pd.Timestamp("2026-01-01 00:15", tz="UTC"),
                pd.Timestamp("2026-01-01 00:20", tz="UTC"),
            ],
        )
        self.assertEqual(model.dd.save_drawer_to_disk.call_count, 0)

    def test_interior_gap_longer_than_the_candle_budget_moves_the_boundary_to_the_gap_start(self):
        dates = [
            "2026-01-01 00:00",
            "2026-01-01 00:05",
            "2026-01-01 00:10",
            "2026-01-01 00:15",
            "2026-01-01 01:00",
            "2026-01-01 01:05",
        ]
        frame = history(dates, [True] * len(dates))
        model = regressor(
            dd=drawer(extras={CALIBRATION_START_KEY: "2026-01-01T00:00:00+00:00"}),
        )
        model._calibration_current_candles[PAIR] = pd.Timestamp("2026-01-01 01:05", tz="UTC")

        eligible = model._calibration_history(model.dd, PAIR, frame, 2)

        self.assertEqual(
            eligible["date_pred"].tolist(),
            [
                pd.Timestamp("2026-01-01 01:00", tz="UTC"),
                pd.Timestamp("2026-01-01 01:05", tz="UTC"),
            ],
        )
        self.assertEqual(
            model.dd.pair_dict[PAIR]["extras"][CALIBRATION_START_KEY],
            "2026-01-01T01:00:00+00:00",
        )

    def test_gap_between_the_boundary_and_the_first_observation_moves_the_boundary_forward(self):
        # No interior gap at all - the only gap is between the persisted boundary and the
        # first surviving row, which is the case a diff()-based search cannot see.
        dates = ["2026-01-01 00:30", "2026-01-01 00:35", "2026-01-01 00:40"]
        frame = history(dates, [True] * len(dates))
        model = regressor(
            dd=drawer(extras={CALIBRATION_START_KEY: "2026-01-01T00:00:00+00:00"}),
        )
        model._calibration_current_candles[PAIR] = pd.Timestamp("2026-01-01 00:40", tz="UTC")

        eligible = model._calibration_history(model.dd, PAIR, frame, 2)

        self.assertEqual(
            eligible["date_pred"].tolist(),
            [
                pd.Timestamp("2026-01-01 00:30", tz="UTC"),
                pd.Timestamp("2026-01-01 00:35", tz="UTC"),
                pd.Timestamp("2026-01-01 00:40", tz="UTC"),
            ],
        )
        self.assertEqual(
            model.dd.pair_dict[PAIR]["extras"][CALIBRATION_START_KEY],
            "2026-01-01T00:30:00+00:00",
        )

    def test_the_boundary_moves_to_the_last_gap_not_the_first(self):
        # Two gaps: resuming after the first would leave a second, older-than-budget hole
        # inside the warmup window, so only the last gap is a valid restart point.
        dates = [
            "2026-01-01 00:00",
            "2026-01-01 00:05",
            "2026-01-01 01:00",
            "2026-01-01 01:05",
            "2026-01-01 02:00",
            "2026-01-01 02:05",
        ]
        frame = history(dates, [True] * len(dates))
        model = regressor(
            dd=drawer(extras={CALIBRATION_START_KEY: "2026-01-01T00:00:00+00:00"}),
        )
        model._calibration_current_candles[PAIR] = pd.Timestamp("2026-01-01 02:05", tz="UTC")

        eligible = model._calibration_history(model.dd, PAIR, frame, 2)

        self.assertEqual(
            eligible["date_pred"].tolist(),
            [
                pd.Timestamp("2026-01-01 02:00", tz="UTC"),
                pd.Timestamp("2026-01-01 02:05", tz="UTC"),
            ],
        )
        self.assertEqual(
            model.dd.pair_dict[PAIR]["extras"][CALIBRATION_START_KEY],
            "2026-01-01T02:00:00+00:00",
        )

    def test_observations_older_than_the_candle_budget_start_a_new_warmup(self):
        dates = ["2026-01-01 00:00", "2026-01-01 00:05"]
        frame = history(dates, [True, True])
        model = regressor(
            dd=drawer(extras={CALIBRATION_START_KEY: "2026-01-01T00:00:00+00:00"}),
        )
        decision = pd.Timestamp("2026-01-01 02:00", tz="UTC")
        model._calibration_current_candles[PAIR] = decision

        eligible = model._calibration_history(model.dd, PAIR, frame, 2)

        self.assertTrue(eligible.empty)
        self.assertEqual(
            model.dd.pair_dict[PAIR]["extras"][CALIBRATION_START_KEY],
            "2026-01-01T02:00:00+00:00",
        )

    def test_gap_exactly_at_the_candle_budget_keeps_the_earlier_boundary(self):
        dates = ["2026-01-01 00:00", "2026-01-01 00:10", "2026-01-01 00:20"]
        frame = history(dates, [True] * len(dates))
        model = regressor(
            dd=drawer(extras={CALIBRATION_START_KEY: "2026-01-01T00:00:00+00:00"}),
        )
        model._calibration_current_candles[PAIR] = pd.Timestamp("2026-01-01 00:20", tz="UTC")

        eligible = model._calibration_history(model.dd, PAIR, frame, 2)

        self.assertEqual(len(eligible), 3)
        self.assertEqual(
            model.dd.pair_dict[PAIR]["extras"][CALIBRATION_START_KEY],
            "2026-01-01T00:00:00+00:00",
        )

    def test_no_decision_time_yields_no_history_and_persists_nothing(self):
        frame = history(["2026-01-01 00:00"], [True])
        model = regressor(dd=drawer())
        model._calibration_current_candles[PAIR] = pd.NaT

        eligible = model._calibration_history(model.dd, PAIR, frame, 2)

        self.assertTrue(eligible.empty)
        self.assertEqual(model.dd.pair_dict[PAIR]["extras"], {})
        self.assertEqual(model.dd.save_drawer_to_disk.call_count, 0)

    def test_decision_time_prefers_the_live_candle_then_full_df_then_history(self):
        frame = history(["2026-01-01 00:00"], [True])
        model = regressor(dd=drawer())
        live = model._calibration_current_candles

        live[PAIR] = pd.Timestamp("2026-01-01 00:20", tz="UTC")
        dk = kitchen()
        dk.full_df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01 01:00"], utc=True)})
        self.assertEqual(
            model._calibration_decision_time(dk, PAIR, frame),
            pd.Timestamp("2026-01-01 00:20", tz="UTC"),
        )

        live[PAIR] = pd.NaT
        self.assertIsNone(model._calibration_decision_time(dk, PAIR, frame))

        live.pop(PAIR)
        self.assertEqual(
            model._calibration_decision_time(dk, PAIR, frame),
            pd.Timestamp("2026-01-01 01:00", tz="UTC"),
        )

        self.assertEqual(
            model._calibration_decision_time(kitchen(), PAIR, frame),
            pd.Timestamp("2026-01-01 00:00", tz="UTC"),
        )
        self.assertIsNone(model._calibration_decision_time(kitchen(), PAIR, frame.iloc[:0]))

    def test_persisting_a_boundary_never_mutates_the_drawer_shared_extras(self):
        # Freqtrade seeds a new pair entry with empty_pair_dict.copy(), a SHALLOW copy, so
        # every fresh pair shares one extras dict. Persisting in place would hand one
        # pair's warmup boundary to every other pair on the next restart.
        shared: dict = {}
        dd = drawer(shared=shared)
        model = regressor(dd=dd)
        fresh = "ETH/USDT"
        dd.get_pair_dict_info(fresh)
        self.assertIs(dd.pair_dict[fresh]["extras"], shared)

        model._persist_calibration_start(fresh, pd.Timestamp("2026-01-01 00:05", tz="UTC"))

        self.assertEqual(shared, {})
        self.assertEqual(
            dd.pair_dict[fresh]["extras"][CALIBRATION_START_KEY], "2026-01-01T00:05:00+00:00"
        )
        self.assertNotIn(CALIBRATION_START_KEY, dd.pair_dict[PAIR]["extras"])
        dd.save_drawer_to_disk.assert_called_once_with()

    def test_one_pairs_boundary_is_never_visible_to_another_pair(self):
        shared: dict = {}
        dd = drawer(shared=shared)
        model = regressor(dd=dd)
        other = "ETH/USDT"
        dd.get_pair_dict_info(other)

        model._persist_calibration_start(PAIR, pd.Timestamp("2026-01-01 00:05", tz="UTC"))

        # The other pair has no boundary of its own, so it migrates from its own
        # history rather than inheriting BTC's - a shared dict would have leaked it.
        self.assertNotIn(CALIBRATION_START_KEY, dd.pair_dict[other]["extras"])
        model._calibration_history(
            dd,
            other,
            history(["2026-01-01 01:00", "2026-01-01 01:05"], [True, True]),
            2,
        )
        self.assertEqual(
            dd.pair_dict[other]["extras"][CALIBRATION_START_KEY], "2026-01-01T01:00:00+00:00"
        )

    def test_persisting_an_unchanged_boundary_does_not_rewrite_the_drawer(self):
        dd = drawer(extras={CALIBRATION_START_KEY: "2026-01-01T00:05:00+00:00"})
        model = regressor(dd=dd)

        model._persist_calibration_start(PAIR, pd.Timestamp("2026-01-01 00:05", tz="UTC"))

        dd.save_drawer_to_disk.assert_not_called()

    def test_migration_resumes_from_the_earliest_provably_produced_row(self):
        dates = ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:10"]
        frame = history(dates, [False, True, True])
        model = regressor(dd=drawer())
        decision = pd.Timestamp("2026-01-01 00:10", tz="UTC")

        start = model._migrate_calibration_start(PAIR, frame, decision)

        self.assertEqual(start, pd.Timestamp("2026-01-01 00:05", tz="UTC"))
        self.assertEqual(
            model.dd.pair_dict[PAIR]["extras"][CALIBRATION_START_KEY],
            "2026-01-01T00:05:00+00:00",
        )

    # ------------------------------------------------- produced marker and private column

    def _model_with_appended_history(
        self, saved_dates: list[str]
    ) -> tuple[QuickAdapterRegressorV3, SimpleNamespace]:
        frame = history(saved_dates, [False] * len(saved_dates), values=[0.1] * len(saved_dates))
        dd = drawer()
        dd.historic_predictions[PAIR] = frame
        dd.model_return_values[PAIR] = frame.tail(1).reset_index(drop=True)
        model = regressor(dd=dd)
        return model, dd

    def _strategy_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01 00:05"], utc=True),
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
            }
        )

    def test_marker_is_set_on_the_current_candle_and_its_return_mirror_only(self):
        model, dd = self._model_with_appended_history(
            ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:10"]
        )
        dk = kitchen()
        dk.return_dataframe = dd.historic_predictions[PAIR].copy()

        def native_overwrite(self, dataframe, dk, pair, trained_timestamp):
            current = pd.Timestamp(dataframe["date"].iloc[-1])
            frame = dd.historic_predictions[pair]
            frame.at[frame.index[-1], "date_pred"] = current
            dd.model_return_values[pair] = frame.tail(1).reset_index(drop=True)
            dk.return_dataframe = frame.tail(1).reset_index(drop=True)

        with mock.patch.object(
            IFreqaiModel,
            "build_strategy_return_arrays",
            autospec=True,
            side_effect=native_overwrite,
        ):
            model.build_strategy_return_arrays(self._strategy_frame(), dk, PAIR, 0)

        self.assertEqual(
            dd.historic_predictions[PAIR][_PRODUCED_PREDICTION_COLUMN].tolist(),
            [False, False, True],
        )
        self.assertEqual(dd.model_return_values[PAIR][_PRODUCED_PREDICTION_COLUMN].tolist(), [True])
        self.assertNotIn(_PRODUCED_PREDICTION_COLUMN, dk.return_dataframe)
        self.assertEqual(model._calibration_current_candles, {})

    def test_no_marker_when_the_saved_candle_predates_the_strategy_frame(self):
        model, dd = self._model_with_appended_history(
            ["2026-01-01 00:50", "2026-01-01 00:55", "2026-01-01 01:00"]
        )
        dk = kitchen()
        dk.return_dataframe = dd.historic_predictions[PAIR].copy()

        def stale_append(self, dataframe, dk, pair, trained_timestamp):
            dk.return_dataframe = dd.historic_predictions[pair].tail(1).reset_index(drop=True)

        with mock.patch.object(
            IFreqaiModel, "build_strategy_return_arrays", autospec=True, side_effect=stale_append
        ):
            model.build_strategy_return_arrays(self._strategy_frame(), dk, PAIR, 0)

        self.assertEqual(
            dd.historic_predictions[PAIR][_PRODUCED_PREDICTION_COLUMN].tolist(), [False] * 3
        )
        self.assertEqual(
            dd.model_return_values[PAIR][_PRODUCED_PREDICTION_COLUMN].tolist(), [False]
        )
        self.assertNotIn(_PRODUCED_PREDICTION_COLUMN, dk.return_dataframe)

    def test_the_current_candle_is_released_even_when_the_native_append_fails(self):
        model, dd = self._model_with_appended_history(
            ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:10"]
        )
        dk = kitchen()
        dk.return_dataframe = dd.historic_predictions[PAIR].copy()

        with (
            mock.patch.object(
                IFreqaiModel,
                "build_strategy_return_arrays",
                autospec=True,
                side_effect=RuntimeError("drawer unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "drawer unavailable"),
        ):
            model.build_strategy_return_arrays(self._strategy_frame(), dk, PAIR, 0)

        self.assertEqual(model._calibration_current_candles, {})
        self.assertEqual(
            dd.historic_predictions[PAIR][_PRODUCED_PREDICTION_COLUMN].tolist(), [False] * 3
        )

    def test_return_dataframe_is_always_stripped_of_the_private_column(self):
        model, _ = self._model_with_appended_history(
            ["2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:10"]
        )
        dk = kitchen()
        dk.return_dataframe = pd.DataFrame({_PRODUCED_PREDICTION_COLUMN: [True], LABEL: [0.0]})

        with mock.patch.object(IFreqaiModel, "build_strategy_return_arrays", autospec=True):
            model.build_strategy_return_arrays(self._strategy_frame(), dk, PAIR, 0)

        self.assertNotIn(_PRODUCED_PREDICTION_COLUMN, dk.return_dataframe)
        self.assertIn(LABEL, dk.return_dataframe)

    def test_bootstrap_history_gains_an_all_false_produced_column(self):
        frame = history(["2026-01-01 00:00", "2026-01-01 00:05"], [False, False])
        dd = drawer()
        dd.historic_predictions[PAIR] = frame
        model = regressor(dd=dd)
        strat_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:05"], utc=True),
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.0],
            }
        )
        bootstrap = pd.DataFrame({LABEL: [0.0, 0.0]})
        dk = kitchen()
        dk.data["extra_returns_per_train"] = {}

        with mock.patch.object(
            IFreqaiModel,
            "set_initial_historic_predictions",
            autospec=True,
            side_effect=lambda self, *args: self.dd.historic_predictions.__setitem__(
                PAIR, bootstrap.copy()
            ),
        ):
            model.set_initial_historic_predictions(bootstrap, dk, PAIR, strat_df)

        self.assertEqual(
            dd.historic_predictions[PAIR][_PRODUCED_PREDICTION_COLUMN].tolist(),
            [False, False],
        )

    # ---------------------------------------------------------------- DI cutoff sentinels

    def test_cold_pair_uses_sentinels_outside_the_normalized_label_range(self):
        # One produced observation against a two-candle budget: warmup is not complete,
        # so no threshold may be derived from an unobserved distribution.
        model, dk, _ = calibrated(candles=1, budget=2)
        extra = dk.data["extra_returns_per_train"]

        model.fit_live_predictions(dk, PAIR)

        self.assertEqual(extra[f"{LABEL}_minima_threshold"], -2.0)
        self.assertEqual(extra[f"{LABEL}_maxima_threshold"], 2.0)
        self.assertEqual(extra["DI_value_param1"], 0.0)
        self.assertEqual(extra["DI_value_param2"], 0.0)
        self.assertEqual(extra["DI_value_param3"], 0.0)
        self.assertEqual(extra["DI_cutoff"], DI_CUTOFF_DEFAULT)
        # Outside [-1, 1] on purpose: a cold pair has no observed distribution, so a
        # sentinel inside the reachable range would manufacture an extremum threshold
        # from nothing and fire on the very first candle.
        self.assertLess(extra[f"{LABEL}_minima_threshold"], LABEL_RANGE[0])
        self.assertGreater(extra[f"{LABEL}_maxima_threshold"], LABEL_RANGE[1])
        self.assertEqual(LABEL_RANGE, (-1.0, 1.0))
        self.assertGreater(DI_CUTOFF_DEFAULT, LABEL_RANGE[1])
        self.assertTrue(np.isfinite(extra["DI_cutoff"]))

    def test_degenerate_di_fit_substitutes_the_declared_default_instead_of_nan(self):
        model, dk, _ = calibrated(candles=3, di=[2.5, 2.5, 2.5])
        extra = dk.data["extra_returns_per_train"]

        model.fit_live_predictions(dk, PAIR)

        self.assertEqual(extra["DI_value_param1"], 0.0)
        self.assertEqual(extra["DI_value_param2"], 0.0)
        self.assertEqual(extra["DI_value_param3"], 0.0)
        self.assertTrue(np.isfinite(extra["DI_cutoff"]))
        self.assertEqual(extra["DI_cutoff"], DI_CUTOFF_DEFAULT)

    def test_a_non_finite_ppf_is_replaced_even_when_the_fit_degrades(self):
        model, dk, _ = calibrated(candles=3, di=[1.0, 2.0, 3.0])
        extra = dk.data["extra_returns_per_train"]

        with mock.patch("scipy.stats.weibull_min.ppf", return_value=float("nan")) as ppf:
            model.fit_live_predictions(dk, PAIR)

        ppf.assert_called_once()
        self.assertEqual(extra["DI_cutoff"], DI_CUTOFF_DEFAULT)
        self.assertFalse(pd.isna(extra["DI_cutoff"]))

    def test_a_healthy_di_sample_keeps_its_fitted_cutoff(self):
        di = [0.5, 1.25, 1.75, 2.5]
        expected_params = sp.stats.weibull_min.fit(di, floc=0)
        expected_cutoff = sp.stats.weibull_min.ppf(0.999, *expected_params)
        model, dk, _ = calibrated(candles=4, di=di)
        extra = dk.data["extra_returns_per_train"]

        model.fit_live_predictions(dk, PAIR)

        np.testing.assert_allclose(
            [extra[f"DI_value_param{i}"] for i in (1, 2, 3)],
            expected_params,
            rtol=1e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(extra["DI_cutoff"], expected_cutoff, rtol=1e-12, atol=0.0)

    def test_missing_and_non_positive_di_values_never_yield_a_nan_cutoff(self):
        for di in ([np.nan] * 4, [0.0, 0.0, 0.0, 0.0], [-1.0, -2.0, -3.0, -4.0]):
            with self.subTest(di_values=di):
                model, dk, _ = calibrated(candles=4, di=di)
                extra = dk.data["extra_returns_per_train"]

                model.fit_live_predictions(dk, PAIR)

                self.assertEqual(extra["DI_cutoff"], DI_CUTOFF_DEFAULT)

    def test_a_fit_without_di_values_uses_the_default_cutoff(self):
        model, dk, _ = calibrated(candles=2)
        extra = dk.data["extra_returns_per_train"]

        model.fit_live_predictions(dk, PAIR)

        self.assertEqual(extra["DI_cutoff"], DI_CUTOFF_DEFAULT)
        self.assertEqual(dk.data["DI_value_mean"], 0.0)
        self.assertEqual(dk.data["DI_value_std"], 0.0)

    # ------------------------------------------------------------ holdout rmse validation

    def test_validate_value_maps_every_non_finite_input_to_none(self):
        for value in (
            float("nan"),
            float("inf"),
            float("-inf"),
            np.nan,
            np.inf,
            None,
            "0.5",
            [1.0],
        ):
            with self.subTest(value=value):
                self.assertIsNone(QuickAdapterRegressorV3.optuna_validate_value(value))

    def test_validate_value_passes_finite_numbers_through_unchanged(self):
        self.assertEqual(QuickAdapterRegressorV3.optuna_validate_value(3), 3)
        self.assertEqual(QuickAdapterRegressorV3.optuna_validate_value(-1.25), -1.25)
        self.assertEqual(QuickAdapterRegressorV3.optuna_validate_value(0.0), 0.0)

    def test_holdout_rmse_is_infinite_rather_than_a_fabricated_score(self):
        # (in-memory score, persisted replay column, pair already fitted this session)
        for holdout, replay, session_fitted in (
            ({PAIR: float("inf")}, None, True),
            ({}, None, False),
            ({PAIR: float("inf")}, [float("nan"), float("nan")], False),
        ):
            with self.subTest(holdout=holdout, replay=replay, session_fitted=session_fitted):
                model, dk, _ = calibrated(candles=2, holdout=replay)
                model._holdout_rmse = dict(holdout)
                if not session_fitted:
                    model._session_fitted_pairs = set()
                extra = dk.data["extra_returns_per_train"]

                model.fit_live_predictions(dk, PAIR)

                self.assertTrue(np.isposinf(extra["holdout_rmse"]))
                self.assertIsNone(
                    QuickAdapterRegressorV3.optuna_validate_value(extra["holdout_rmse"])
                )

    def test_replayed_holdout_is_as_of_the_decision_and_independent_of_calibration(self):
        persisted = history(
            [
                "2026-01-01 00:00",
                "2026-01-01 00:00",
                "2026-01-01 00:05",
                "2026-01-01 00:04",
                "2026-01-01 00:10",
                "2026-01-01 00:15",
            ],
            [True, True, True, False, True, True],
            values=[-0.5, 0.5, 0.0, 5.0, 9.0, 7.0],
            holdout=[0.25, 0.5, np.nan, 888.0, 999.0, 777.0],
        )
        persisted.loc[5, "date_pred"] = pd.NaT
        for order in (list(range(6)), [4, 2, 0, 3, 1, 5]):
            with self.subTest(order=order):
                model, dk, _ = calibrated(candles=2)
                model.dd.historic_predictions[PAIR] = persisted.iloc[order].reset_index(drop=True)
                model.dd.pair_dict[PAIR]["extras"][CALIBRATION_START_KEY] = (
                    "2026-01-01T00:05:00+00:00"
                )
                model._calibration_current_candles[PAIR] = pd.Timestamp(
                    "2026-01-01 00:05", tz="UTC"
                )
                model._holdout_rmse = {PAIR: float("inf")}
                model._session_fitted_pairs = set()
                model.fit_live_predictions(dk, PAIR)
                self.assertEqual(dk.data["extra_returns_per_train"]["holdout_rmse"], 0.5)
                self.assertEqual(dk.data["labels_mean"][LABEL], 0.0)

    def test_unavailable_replayed_holdout_does_not_resurrect_an_older_score(self):
        for latest, expected in ((np.inf, np.inf), (np.nan, 0.5)):
            with self.subTest(latest=latest):
                model, dk, _ = calibrated(candles=2, holdout=[0.5, latest])
                model._holdout_rmse = {PAIR: np.inf}
                model._session_fitted_pairs = set()
                model.fit_live_predictions(dk, PAIR)
                self.assertEqual(dk.data["extra_returns_per_train"]["holdout_rmse"], expected)

    def test_an_unfitted_pair_with_no_replay_column_reports_infinite(self):
        model, dk, _ = calibrated(candles=2)
        model._holdout_rmse = {}
        model._session_fitted_pairs = set()
        extra = dk.data["extra_returns_per_train"]

        model.fit_live_predictions(dk, PAIR)

        self.assertTrue(np.isposinf(extra["holdout_rmse"]))

    # --------------------------------------------------------------- optuna throttling

    def _throttled(self, budget: int, pool: list[int] | None = None) -> QuickAdapterRegressorV3:
        model = regressor(_label_frequency_candles=3)
        model._optuna_label_candle_pool = [3, 4, 5] if pool is None else list(pool)
        model._optuna_label_candle_pool_full_cache = {3: [3, 4, 5]}
        model._optuna_label_shuffle_rng = random.Random(7)
        model._optuna_label_candle = {PAIR: budget}
        return model

    def _tick(self, model: QuickAdapterRegressorV3, candle: int, fired: list[int]) -> None:
        model.optuna_throttle_callback(
            PAIR, _OPTUNA_NAMESPACES.label, lambda candle=candle: fired.append(candle)
        )

    def test_throttle_fires_only_once_the_candle_budget_is_spent(self):
        model = self._throttled(2, [3, 4, 5])
        fired: list[int] = []

        for candle in range(4):
            self._tick(model, candle, fired)

        # Two candles of budget: the first is throttled, the second emits. A fabricated
        # per-candle emission would have fired on all four.
        self.assertEqual(fired, [1])
        self.assertEqual(model._optuna_label_candles[PAIR], 2)

    def test_throttle_rearms_a_fresh_budget_after_emitting(self):
        model = self._throttled(1, [5])
        fired: list[int] = []

        for candle in range(6):
            self._tick(model, candle, fired)

        # Emitted on candle 0, re-armed to 5, so the next emission is candle 5 - the
        # budget is re-spent rather than reset to zero (which would fire every candle).
        self.assertEqual(fired, [0, 5])
        self.assertEqual(model._optuna_label_candles[PAIR], 0)
        self.assertEqual(model._optuna_label_candle[PAIR], 4)

    def test_a_raising_callback_is_logged_and_the_following_candle_still_emits(self):
        model = self._throttled(1, [5])
        fired: list[int] = []

        def callback(candle: int) -> None:
            if candle == 0:
                raise RuntimeError("optuna exploded")
            fired.append(candle)

        with mock.patch.object(regressor_module.logger, "exception") as logged:
            for candle in range(6):
                model.optuna_throttle_callback(
                    PAIR,
                    _OPTUNA_NAMESPACES.label,
                    lambda candle=candle: callback(candle),
                )

        logged.assert_called_once()
        self.assertEqual(fired, [5])
        self.assertEqual(model._optuna_label_candles[PAIR], 0)

    def test_incremented_pairs_reset_once_every_pair_has_been_counted(self):
        model = self._throttled(5)
        model.pairs = [PAIR, "ETH/USDT"]
        model._optuna_label_candle = {PAIR: 5, "ETH/USDT": 5}
        model._optuna_label_candles = {PAIR: 0, "ETH/USDT": 0}

        model.optuna_throttle_callback(PAIR, _OPTUNA_NAMESPACES.label, lambda: None)
        self.assertEqual(model._optuna_label_incremented_pairs, [PAIR])

        model.optuna_throttle_callback("ETH/USDT", _OPTUNA_NAMESPACES.label, lambda: None)
        self.assertEqual(model._optuna_label_incremented_pairs, [])

    def test_throttle_rejects_a_foreign_namespace_and_a_non_callable(self):
        model = self._throttled(5)

        with self.assertRaisesRegex(ValueError, "namespace"):
            model.optuna_throttle_callback(PAIR, _OPTUNA_NAMESPACES.hp, lambda: None)

        with self.assertRaisesRegex(ValueError, "must be callable"):
            model.optuna_throttle_callback(PAIR, _OPTUNA_NAMESPACES.label, "not callable")

        self.assertEqual(model._optuna_label_candles[PAIR], 0)

    # ------------------------------------------------------------ warn-once known-at-none

    def test_known_at_none_warns_once_per_pair_and_context(self):
        with mock.patch.object(regressor_module.logger, "info") as logged:
            _log_known_at_none_once(PAIR, "label")
            _log_known_at_none_once(PAIR, "label")
            _log_known_at_none_once(PAIR, "label_weight")
            _log_known_at_none_once("ETH/USDT", "label")

        self.assertEqual(logged.call_count, 3)
        self.assertEqual(
            sorted(regressor_module._KNOWN_AT_NONE_LOGGED),
            sorted({(PAIR, "label"), (PAIR, "label_weight"), ("ETH/USDT", "label")}),
        )


if __name__ == "__main__":
    unittest.main()
