"""Prediction validity and transport contracts in the native Freqtrade QA image."""

import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import optuna
import pandas as pd
from freqtrade.enums import RunMode
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.plot.plotting import add_indicators
from freqtrade.resolvers.freqaimodel_resolver import FreqaiModelResolver
from freqtrade.resolvers.strategy_resolver import StrategyResolver
from freqtrade.rpc.rpc import RPC
from plotly.subplots import make_subplots
from qa_support import (
    CONFIG_TEMPLATE,
    PAIR,
    QaTestCase,
    _merge,
    model_config,
    ohlcv_frame,
    temporary_directory,
)
from QuickAdapterV3 import QuickAdapterV3

from quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 import (
    _PRODUCED_PREDICTION_COLUMN,
    QuickAdapterRegressorV3,
)

LABEL = "&s-extrema"


def trained_regressor(directory: Path, di_threshold: float = 0):
    config = model_config(
        directory,
        freqai={
            "continual_learning": False,
            "optuna_hyperopt": {"enabled": False},
            "feature_parameters": {"DI_threshold": di_threshold, "causal_mode": True},
            "data_split_parameters": {"test_size": 0.2, "shuffle": False},
            "model_training_parameters": {"n_estimators": 3, "n_jobs": 1, "verbosity": 0},
        },
    )
    position = np.arange(120)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(position), freq="5min", tz="UTC"),
            "%-x": np.sin(position / 5),
            "%-lag": np.cos(position / 7),
            LABEL: 0.7 * np.sin(position / 5) + 0.2 * np.cos(position / 7),
        }
    )
    model = QuickAdapterRegressorV3(config=config)
    model.live = True
    kitchen = FreqaiDataKitchen(config, live=True, pair=PAIR)
    kitchen.label_list = [LABEL]
    kitchen.training_features_list = ["%-x", "%-lag"]
    timestamp = int(frame["date"].iloc[-1].timestamp()) + 300
    kitchen.set_paths(PAIR, timestamp)
    kitchen.set_new_model_names(PAIR, timestamp)
    kitchen.data_path.mkdir(parents=True, exist_ok=True)
    model.model = model.train(frame, PAIR, kitchen)
    return model, kitchen, frame


class PredictionValidityTest(QaTestCase):
    def test_nonfinite_features_cannot_be_admitted_after_native_preprocessing(self):
        for threshold, expected in ((0, [1, 0, 1]), (10, [1, 0, 1])):
            with self.subTest(di_threshold=threshold), temporary_directory() as directory:
                model, kitchen, frame = trained_regressor(directory, threshold)
                for missing in (np.nan, np.inf, -np.inf):
                    with self.subTest(missing=missing):
                        inputs = frame.iloc[30:33].copy()
                        inputs.iloc[1, inputs.columns.get_loc("%-x")] = missing

                        predictions, status = model.predict(inputs, kitchen)

                        self.assertEqual(
                            kitchen.data["filter_drop_index_prediction"].tolist(),
                            [False, True, False],
                        )
                        np.testing.assert_array_equal(status, expected)
                        self.assertTrue(np.isfinite(predictions[LABEL]).all())

    def test_missing_features_and_native_di_rejections_accumulate(self):
        with temporary_directory() as directory:
            model, kitchen, frame = trained_regressor(directory, 10)
            inputs = frame.iloc[30:33].copy()
            inputs.iloc[1, inputs.columns.get_loc("%-x")] = np.nan
            inputs.iloc[1, inputs.columns.get_loc("%-lag")] = 1e6

            _, status = model.predict(inputs, kitchen)

            np.testing.assert_array_equal(status, [1, -1, 1])
            np.testing.assert_array_equal(
                kitchen.__dict__["_quickadapter_prediction_produced"], [True, False, True]
            )

    def test_nonfinite_backend_outputs_cannot_enter_calibration(self):
        for threshold in (0, 10):
            with self.subTest(di_threshold=threshold), temporary_directory() as directory:
                model, kitchen, training = trained_regressor(directory, threshold)
                native_predict = model.model.predict
                encoded_zero, _, _ = kitchen.label_pipeline.transform(pd.DataFrame({LABEL: [0.0]}))
                for value in (0.0, np.nan, np.inf, -np.inf):
                    for missing in (False, True):
                        with self.subTest(output=value, missing_input=missing):
                            inputs = training.iloc[30:33].copy()
                            if missing:
                                inputs.iloc[1, inputs.columns.get_loc("%-x")] = np.nan
                            if threshold:
                                inputs.iloc[1, inputs.columns.get_loc("%-lag")] = 1e6
                            backend_value = encoded_zero[LABEL].iloc[0] if value == 0 else value

                            def damaged(
                                features,
                                *args,
                                replacement=backend_value,
                                native=native_predict,
                                **kwargs,
                            ):
                                values = np.array(native(features, *args, **kwargs), copy=True)
                                values[1] = replacement
                                return values

                            with patch.object(model.model, "predict", side_effect=damaged):
                                predictions, status = model.predict(inputs, kitchen)

                            invalid = missing or not np.isfinite(value)
                            expected_status = 1 - int(bool(threshold)) - int(invalid)
                            np.testing.assert_array_equal(status, [1, expected_status, 1])
                            produced = kitchen.__dict__["_quickadapter_prediction_produced"]
                            np.testing.assert_array_equal(produced, [True, not invalid, True])
                            history = pd.DataFrame(
                                {
                                    "date_pred": inputs["date"].to_numpy(),
                                    LABEL: predictions[LABEL].to_numpy(),
                                    "do_predict": status,
                                    _PRODUCED_PREDICTION_COLUMN: produced,
                                }
                            )
                            model._calibration_current_candles[PAIR] = inputs["date"].iloc[
                                -1
                            ] + pd.Timedelta(minutes=5)
                            eligible = model._calibration_history(kitchen, PAIR, history, 3)
                            expected_dates = inputs["date"].iloc[[0, 2] if invalid else [0, 1, 2]]
                            self.assertEqual(
                                eligible["date_pred"].tolist(), expected_dates.tolist()
                            )
                            public = model._normalise_prediction_returns(history.copy())
                            np.testing.assert_allclose(
                                public["prediction_available"], [1, 0 if invalid else 1, 1]
                            )
                            if invalid:
                                self.assertTrue(pd.isna(public[LABEL].iloc[1]))
                            else:
                                self.assertAlmostEqual(public[LABEL].iloc[1], 0.0)

    def test_loaded_model_metric_survives_missing_history_and_bootstrap(self):
        with temporary_directory() as directory:
            model, kitchen, frame = trained_regressor(directory)
            model.dd.get_pair_dict_info(PAIR)
            for stored, expected in ((0.25, 0.25), (0, 0), (np.inf, np.inf), (-1, np.inf)):
                with self.subTest(stored=stored):
                    kitchen.data["extra_returns_per_train"]["holdout_rmse"] = stored
                    model.dd.save_data(model.model, PAIR, kitchen)
                    model.dd.save_drawer_to_disk()
                    restored = QuickAdapterRegressorV3(config=model.config)
                    restored.live = True
                    loaded_kitchen = FreqaiDataKitchen(model.config, live=True, pair=PAIR)
                    restored.model = restored.dd.load_data(PAIR, loaded_kitchen)
                    inputs = frame.iloc[30:33].copy()
                    predictions, _ = restored.predict(inputs, loaded_kitchen)
                    strategy = ohlcv_frame(np.full(len(inputs), 100.0))
                    strategy["date"] = inputs["date"].to_numpy()
                    restored.set_initial_historic_predictions(
                        predictions, loaded_kitchen, PAIR, strategy
                    )
                    restored.fit_live_predictions(loaded_kitchen, PAIR)

                    self.assertEqual(
                        loaded_kitchen.data["extra_returns_per_train"]["holdout_rmse"], expected
                    )
                    self.assertEqual(
                        restored.dd.historic_predictions[PAIR][
                            _PRODUCED_PREDICTION_COLUMN
                        ].tolist(),
                        [False] * len(inputs),
                    )


class PredictionTransportTest(QaTestCase):
    def test_restart_keeps_unknown_rmse_and_ambiguous_legacy_zero(self):
        with temporary_directory() as directory:
            model = QuickAdapterRegressorV3(
                config=model_config(directory, freqai={"optuna_hyperopt": {"enabled": False}})
            )
            dates = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
            strategy = pd.DataFrame({"date": dates, "close": [100.0] * 3})
            predictions = pd.DataFrame({LABEL: [0.0, 0.0, 0.2]})
            model.dd.historic_predictions[PAIR] = pd.DataFrame(
                {
                    "date_pred": dates,
                    LABEL: predictions[LABEL],
                    "do_predict": [0, 0, 1],
                    "holdout_rmse": [0.0, np.nan, np.nan],
                    _PRODUCED_PREDICTION_COLUMN: [True, np.nan, True],
                }
            )

            model.dd.save_historic_predictions_to_disk()
            restored = QuickAdapterRegressorV3(config=model.config)
            restored.dd.set_initial_return_values(PAIR, predictions, strategy)
            result = restored._normalise_prediction_returns(restored.dd.model_return_values[PAIR])

            np.testing.assert_allclose(result[LABEL], [0.0, 0.0, 0.2])
            np.testing.assert_allclose(result["prediction_available"], [1, np.nan, 1])
            np.testing.assert_allclose(result["holdout_rmse"], [0, np.nan, np.nan])
            np.testing.assert_allclose(
                restored.dd.historic_predictions[PAIR]["holdout_rmse"], [0, np.nan, np.nan]
            )

    def test_restored_public_returns_distinguish_absence_legacy_and_real_zero(self):
        with temporary_directory() as directory:
            model, kitchen, training = trained_regressor(directory)
            inputs = training.iloc[30:34].copy()
            predictions, _ = model.predict(inputs, kitchen)
            strategy = ohlcv_frame(np.full(len(inputs), 100.0))
            strategy["date"] = inputs["date"].to_numpy()
            strategy["%-x"] = inputs["%-x"].to_numpy()
            strategy["%-lag"] = inputs["%-lag"].to_numpy()
            model.dd.get_pair_dict_info(PAIR)
            model.dd.save_data(model.model, PAIR, kitchen)
            model.dd.save_drawer_to_disk()
            model.dd.historic_predictions[PAIR] = pd.DataFrame(
                {
                    "date_pred": strategy["date"].iloc[1:].to_numpy(),
                    LABEL: [0.0, 0.0, predictions[LABEL].iloc[-1]],
                    "do_predict": [0, 0, 1],
                    _PRODUCED_PREDICTION_COLUMN: [np.nan, True, True],
                }
            )
            model.dd.save_historic_predictions_to_disk()
            restored = QuickAdapterRegressorV3(config=model.config)
            restored.live = True
            loaded = FreqaiDataKitchen(model.config, live=True, pair=PAIR)
            restored.model = restored.dd.load_data(PAIR, loaded)

            restored.build_strategy_return_arrays(strategy, loaded, PAIR, 0)

            result = loaded.return_dataframe
            np.testing.assert_allclose(result["prediction_available"], [0, np.nan, 1, 1])
            np.testing.assert_allclose(
                result[LABEL], [np.nan, 0.0, 0.0, predictions[LABEL].iloc[-1]]
            )
            self.assertEqual(result["date"].tolist(), strategy["date"].tolist())
            payload = RPC._convert_dataframe_to_dict(
                "QuickAdapterV3",
                PAIR,
                "5m",
                result,
                strategy["date"].iloc[-1].to_pydatetime(),
                None,
                [],
            )
            availability = payload["columns"].index("prediction_available")
            prediction = payload["columns"].index(LABEL)
            self.assertEqual([row[availability] for row in payload["data"]], [0, None, 1, 1])
            self.assertEqual([row[prediction] for row in payload["data"]][:3], [None, 0.0, 0.0])
            self.assertNotIn(_PRODUCED_PREDICTION_COLUMN, payload["columns"])

    def test_historic_live_model_replay_distinguishes_missing_and_legacy_rows(self):
        for with_marker in (False, True):
            with self.subTest(with_marker=with_marker), temporary_directory() as directory:
                config = model_config(directory, freqai={"optuna_hyperopt": {"enabled": False}})
                config.update(
                    runmode=RunMode.BACKTEST,
                    trading_mode="spot",
                    timerange="20240101-20240102",
                    config_files=[str(CONFIG_TEMPLATE)],
                    freqai_backtest_live_models=True,
                )
                frame = ohlcv_frame([100.0, 101.0, 102.0, 103.0])
                frame["date"] = pd.to_datetime(frame["date"], utc=True)
                history = pd.DataFrame(
                    {
                        "date_pred": frame["date"].iloc[[0, 2, 3]].to_numpy(),
                        LABEL: [0.0, 0.5, 0.0],
                        "do_predict": [1, 1, 0],
                        "holdout_rmse": [0.2] * 3,
                    }
                )
                if with_marker:
                    history[_PRODUCED_PREDICTION_COLUMN] = [True, True, np.nan]
                model = QuickAdapterRegressorV3(config=config)
                model.dd.historic_predictions[PAIR] = history
                model.dd.save_historic_predictions_to_disk()
                restored = QuickAdapterRegressorV3(config=config)
                strategy = QuickAdapterV3(config)
                strategy.freqai = restored
                strategy.freqai_info = config["freqai"]
                strategy.bot_start()
                strategy.dp = SimpleNamespace(runmode=RunMode.BACKTEST)

                result = restored.start(frame, {"pair": PAIR}, strategy)

                np.testing.assert_allclose(result["prediction_available"], [1, 0, 1, np.nan])
                np.testing.assert_allclose(result[LABEL], [0.0, np.nan, 0.5, 0.0])
                np.testing.assert_allclose(result["holdout_rmse"], [0.2, np.nan, 0.2, 0.2])
                np.testing.assert_allclose(result["do_predict"], [1, np.nan, 1, 0])
                pd.testing.assert_frame_equal(restored.dd.historic_predictions[PAIR], history)
                payload = RPC._convert_dataframe_to_dict(
                    "QuickAdapterV3",
                    PAIR,
                    "5m",
                    result,
                    frame["date"].iloc[-1].to_pydatetime(),
                    None,
                    [],
                )
                availability = payload["columns"].index("prediction_available")
                self.assertEqual([row[availability] for row in payload["data"]], [1, 0, 1, None])
                self.assertNotIn(_PRODUCED_PREDICTION_COLUMN, payload["columns"])

    def test_old_history_accepts_new_metrics_without_fabricating_past_scores(self):
        for flag in (True, False, np.bool_(True), np.bool_(False)):
            with self.subTest(flag=flag), temporary_directory() as directory:
                config = model_config(directory, freqai={"optuna_hyperopt": {"enabled": False}})
                model = QuickAdapterRegressorV3(config=config)
                model.live = True
                kitchen = FreqaiDataKitchen(config, live=True, pair=PAIR)
                kitchen.label_list = [LABEL]
                kitchen.data.update(
                    labels_mean={LABEL: 0.0},
                    labels_std={LABEL: 0.1},
                    extra_returns_per_train={
                        "holdout_rmse": 0.25,
                        "label_period_candles": 1,
                        "has_signal": flag,
                    },
                )
                dates = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
                model.dd.historic_predictions[PAIR] = pd.DataFrame(
                    {
                        "date_pred": dates[:2],
                        LABEL: [0.1, 0.0],
                        "do_predict": [1, 0],
                        "close_price": [100.0, 100.0],
                        "high_price": [101.0, 101.0],
                        "low_price": [99.0, 99.0],
                        _PRODUCED_PREDICTION_COLUMN: [True, True],
                    }
                )
                strategy = pd.DataFrame(
                    {"date": dates, "high": [101.0] * 3, "low": [99.0] * 3, "close": [100.0] * 3}
                )

                model.dd.append_model_predictions(
                    PAIR, pd.DataFrame({LABEL: [0.0]}), np.array([0]), kitchen, strategy
                )
                model.dd.save_historic_predictions_to_disk()
                restart_strategy = pd.concat(
                    [
                        strategy,
                        strategy.iloc[-1:].assign(date=dates[-1] + pd.Timedelta(minutes=5)),
                    ],
                    ignore_index=True,
                )
                restored = QuickAdapterRegressorV3(config=config)
                restored.dd.set_initial_return_values(
                    PAIR, pd.DataFrame({LABEL: np.zeros(4)}), restart_strategy
                )
                result = restored._normalise_prediction_returns(
                    restored.dd.model_return_values[PAIR]
                )

                np.testing.assert_allclose(result["holdout_rmse"], [np.nan, np.nan, 0.25, np.nan])
                np.testing.assert_allclose(result["prediction_available"], [1, 1, 1, 0])
                np.testing.assert_allclose(result[LABEL], [0.1, 0.0, 0.0, np.nan])
                self.assertTrue(result["has_signal"].iloc[:2].isna().all())
                self.assertEqual(result["has_signal"].dtype, object)
                self.assertIsInstance(result["has_signal"].iloc[2], (bool, np.bool_))
                self.assertEqual(result["has_signal"].iloc[2], flag)

    def test_native_no_model_return_does_not_expose_zero_predictions(self):
        with temporary_directory() as directory:
            fields = ("DI_cutoff", "DI_value_param1", "DI_value_param2", "DI_value_param3")
            config = model_config(
                directory,
                freqai={
                    "optuna_hyperopt": {"enabled": False},
                    "extra_returns_per_train": dict.fromkeys(fields, 2.0),
                },
            )
            model = QuickAdapterRegressorV3(config=config)
            kitchen = FreqaiDataKitchen(config, live=True, pair=PAIR)
            frame = ohlcv_frame(np.full(3, 100.0))
            frame["%-feature"] = [0.1, 0.2, 0.3]
            frame[LABEL] = [0.5, 0.0, -0.5]

            model.dd.return_null_values_to_strategy(frame, kitchen)

            np.testing.assert_array_equal(kitchen.return_dataframe["do_predict"], [0, 0, 0])
            np.testing.assert_array_equal(
                kitchen.return_dataframe["prediction_available"], [0, 0, 0]
            )
            self.assertTrue(kitchen.return_dataframe[LABEL].isna().all())
            self.assertTrue(kitchen.return_dataframe["holdout_rmse"].isna().all())
            for column in fields:
                self.assertTrue(kitchen.return_dataframe[column].isna().all())
            payload = RPC._convert_dataframe_to_dict(
                "QuickAdapterV3",
                PAIR,
                "5m",
                kitchen.return_dataframe.copy(),
                frame["date"].iloc[-1].to_pydatetime(),
                None,
                [],
            )
            for column in fields:
                position = payload["columns"].index(column)
                self.assertEqual([row[position] for row in payload["data"]], [None] * 3)

    def test_di_diagnostics_keep_real_and_legacy_zeros_but_not_expired_values(self):
        with temporary_directory() as directory:
            model = QuickAdapterRegressorV3(
                config=model_config(directory, freqai={"optuna_hyperopt": {"enabled": False}})
            )
            fields = ("DI_cutoff", "DI_value_param1", "DI_value_param2", "DI_value_param3")
            frame = pd.DataFrame(
                {
                    "do_predict": [0, 1, 2, 0],
                    LABEL: [0.0] * 4,
                    _PRODUCED_PREDICTION_COLUMN: [True, True, True, np.nan],
                    **dict.fromkeys(fields, 0.0),
                }
            )

            result = model._normalise_prediction_returns(frame)

            self.assertEqual(result["do_predict"].tolist(), [0, 1, 2, 0])
            np.testing.assert_allclose(result["prediction_available"], [1, 1, 0, np.nan])
            for column in (LABEL, *fields):
                np.testing.assert_allclose(result[column], [0, 0, np.nan, 0])

    def test_native_plotly_draws_real_zero_and_missing_prediction_gaps(self):
        with temporary_directory() as directory:
            config = model_config(directory, freqai={"optuna_hyperopt": {"enabled": False}})
            model = QuickAdapterRegressorV3(config=config)
            strategy = QuickAdapterV3(config)
            dates = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
            frame = pd.DataFrame(
                {
                    "date": dates,
                    LABEL: [0, 0, 0, 0, 0.5],
                    "holdout_rmse": [0, 0, -1, np.inf, 0.25],
                    "do_predict": [0, 0, 1, 2, 1],
                    _PRODUCED_PREDICTION_COLUMN: [True, False, True, True, True],
                }
            )
            for column in (
                "extrema_direction",
                "extrema_direction_smoothed",
                "extrema_weight",
                "extrema_weight_smoothed",
            ):
                frame[column] = 0.0
            result = model._normalise_prediction_returns(frame)
            figure = make_subplots(rows=len(strategy.plot_config["subplots"]), cols=1)
            for row, indicators in enumerate(strategy.plot_config["subplots"].values(), start=1):
                add_indicators(figure, row, indicators, result)
            traces = {trace.name: trace for trace in figure.data}

            np.testing.assert_allclose(traces[LABEL].y, [0, np.nan, 0, np.nan, 0.5])
            np.testing.assert_allclose(traces["holdout_rmse"].y, [0, np.nan, np.nan, np.nan, 0.25])
            np.testing.assert_array_equal(traces["extrema_direction_smoothed"].y, [0] * 5)
            self.assertEqual(traces[LABEL].type, "scatter")
            self.assertEqual(traces[LABEL].mode, "lines")
            self.assertEqual(traces["holdout_rmse"].mode, "lines")
            self.assertEqual(traces["extrema_direction_smoothed"].mode, "lines")
            self.assertEqual(traces["extrema_weight_smoothed"].mode, "lines")
            self.assertEqual(traces["extrema_direction"].type, "bar")
            self.assertEqual(traces["extrema_weight"].type, "bar")


class BacktestContractsTest(QaTestCase):
    def test_replay_uses_produced_observations_and_current_model_metric(self):
        with temporary_directory() as directory:
            config = model_config(directory, freqai={"optuna_hyperopt": {"enabled": False}})
            model = QuickAdapterRegressorV3(config=config)
            model.live = False
            dates = pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")
            frame = pd.DataFrame(
                {
                    "date": dates,
                    LABEL: [0.8, 0.6, 0, 0.4, 0.2, 0.0],
                    "do_predict": [1, 0, 0, 1, 1, 1],
                    "holdout_rmse": [0.8, 0.6, np.nan, 0.4, 0.2, 0.0],
                    _PRODUCED_PREDICTION_COLUMN: [True, True, False, True, True, True],
                }
            )
            kitchen = SimpleNamespace(
                pair=PAIR,
                label_list=[LABEL],
                unique_class_list=[],
                full_df=frame,
                data={"extra_returns_per_train": {}},
            )

            model.backtesting_fit_live_predictions(kitchen)

            self.assertAlmostEqual(frame.loc[4, f"{LABEL}_mean"], 0.5)
            self.assertAlmostEqual(frame.loc[4, f"{LABEL}_std"], 0.1)
            self.assertEqual(frame["holdout_rmse"].tolist()[-3:], [0.4, 0.2, 0])
            # Current candle must not influence its own calibration.
            self.assertAlmostEqual(frame.loc[5, f"{LABEL}_mean"], 0.3)

    def test_replay_and_live_reset_warmup_after_the_same_long_gap(self):
        with temporary_directory() as directory:
            config = model_config(directory, freqai={"optuna_hyperopt": {"enabled": False}})
            frame = pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        [
                            "2026-01-01 00:00",
                            "2026-01-01 00:05",
                            "2026-01-01 00:25",
                            "2026-01-01 00:30",
                            "2026-01-01 00:35",
                            "2026-01-01 00:40",
                        ],
                        utc=True,
                    ),
                    LABEL: [0.8, 0.6, 0.0, 0.4, 0.2, 0.0],
                    "do_predict": [1, 0, 0, 1, 1, 1],
                    _PRODUCED_PREDICTION_COLUMN: [True, True, False, True, True, True],
                }
            )
            replay = QuickAdapterRegressorV3(config=config)
            replay.live = False
            kitchen = SimpleNamespace(
                pair=PAIR,
                label_list=[LABEL],
                unique_class_list=[],
                full_df=frame.copy(),
                data={"extra_returns_per_train": {}},
            )
            replay.backtesting_fit_live_predictions(kitchen)
            live = QuickAdapterRegressorV3(config=config)
            live.live = True
            for position in range(len(frame)):
                live.dd.historic_predictions[PAIR] = frame.iloc[:position].rename(
                    columns={"date": "date_pred"}
                )
                live._calibration_current_candles[PAIR] = frame["date"].iloc[position]
                live.fit_live_predictions(kitchen, PAIR)
                self.assertAlmostEqual(
                    kitchen.data["labels_mean"][LABEL],
                    kitchen.full_df.loc[position, f"{LABEL}_mean"],
                )
                self.assertAlmostEqual(
                    kitchen.data["labels_std"][LABEL],
                    kitchen.full_df.loc[position, f"{LABEL}_std"],
                )
            np.testing.assert_allclose(kitchen.full_df.loc[2:4, f"{LABEL}_mean"], 0)
            self.assertAlmostEqual(kitchen.full_df.loc[5, f"{LABEL}_mean"], 0.3)

    @staticmethod
    def _run_backtest(
        directory: Path,
        frame: pd.DataFrame,
        weighting: str = "uniform",
        *,
        strategy_name: str = "QuickAdapterV3",
        model_name: str = "QuickAdapterRegressorV3",
        strategy_path: Path | None = None,
        model_path: Path | None = None,
        freqai_overrides: dict | None = None,
        config_overrides: dict | None = None,
    ):
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        config = model_config(
            directory,
            freqai=_merge(
                {
                    "continual_learning": False,
                    "save_backtest_models": True,
                    "activate_tensorboard": False,
                    "fit_live_predictions_candles": 12,
                    "label_weighting": {"strategy": weighting},
                    "optuna_hyperopt": {"enabled": False},
                    "feature_parameters": {
                        "indicator_periods_candles": [4, 8],
                        "include_shifted_candles": 0,
                        "buffer_train_data_candles": 0,
                        "causal_mode": True,
                    },
                    "data_split_parameters": {"test_size": 0.2, "shuffle": False},
                    "model_training_parameters": {
                        "n_estimators": 3,
                        "n_jobs": 1,
                        "verbosity": 0,
                        "random_state": 7,
                    },
                },
                freqai_overrides or {},
            ),
        )
        config.update(
            runmode=RunMode.BACKTEST,
            timerange="20240102-20240103",
            config_files=[str(CONFIG_TEMPLATE)],
            strategy=strategy_name,
            freqaimodel=model_name,
            freqaimodel_path=str(model_path or CONFIG_TEMPLATE.parent / "freqaimodels"),
        )
        config = _merge(config, config_overrides or {})
        model = FreqaiModelResolver.load_freqaimodel(config)
        model.live = False
        strategy = StrategyResolver._load_strategy(
            strategy_name,
            config,
            extra_dir=str(strategy_path or CONFIG_TEMPLATE.parent / "strategies"),
        )
        strategy.freqai = model
        strategy.freqai_info = config["freqai"]
        strategy.bot_start()
        kitchen = FreqaiDataKitchen(config, live=False, pair=PAIR)
        model.dk = kitchen
        model.start_backtesting(frame, {"pair": PAIR}, kitchen, strategy)
        kitchen.return_dataframe = kitchen.remove_features_from_df(kitchen.return_dataframe)
        return kitchen

    def test_replay_returns_effective_parameters_in_fresh_and_cached_backtests(self):
        with temporary_directory() as directory:
            frame = ohlcv_frame(
                100 + 8 * np.sin(np.arange(720) / 6) + 2 * np.sin(np.arange(720) / 23)
            )
            overrides = {
                "feature_parameters": {"label_period_candles": 24, "label_natr_multiplier": 7.0},
                "extra_returns_per_train": {
                    "label_period_candles": 18,
                    "label_natr_multiplier": 10.5,
                },
            }
            for cached in (False, True):
                with self.subTest(cached=cached):
                    kitchen = self._run_backtest(directory, frame, freqai_overrides=overrides)
                    result = kitchen.return_dataframe
                    produced = result.loc[result["prediction_available"].eq(1)]
                    np.testing.assert_array_equal(produced["label_period_candles"], 24)
                    np.testing.assert_array_equal(produced["label_natr_multiplier"], 7.0)
                    self.assertEqual(
                        produced["date"].tolist(),
                        pd.date_range("2024-01-02", periods=288, freq="5min", tz="UTC").tolist(),
                    )

    def test_real_label_hpo_replay_matches_live_parameters_and_keeps_window_rmse(self):
        with temporary_directory() as directory:
            config = model_config(
                directory,
                freqai={
                    "fit_live_predictions_candles": 864,
                    "feature_parameters": {
                        "label_period_candles": 18,
                        "label_natr_multiplier": 10.5,
                    },
                    "optuna_hyperopt": {
                        "enabled": False,
                        "n_trials": 1,
                        "n_jobs": 1,
                        "timeout": 30,
                        "warm_start": False,
                    },
                },
            )
            config.update(timerange="20240102-20240103", config_files=[str(CONFIG_TEMPLATE)])
            model = QuickAdapterRegressorV3(config=config)
            model.live = False
            model._optuna_hyperopt = True
            model._optuna_label_candle[PAIR] = 0
            raw = ohlcv_frame(100 + 4 * np.sin(np.arange(600) / 4))
            raw["date"] = pd.to_datetime(raw["date"], utc=True)
            model.data_provider = SimpleNamespace(get_pair_dataframe=lambda **kwargs: raw)
            chosen = {
                "label_period_candles": model._min_label_period_candles,
                "label_natr_multiplier": model._min_label_natr_multiplier,
            }

            def create_study(**kwargs):
                study = optuna.create_study(
                    directions=kwargs["directions"], sampler=optuna.samplers.RandomSampler(seed=7)
                )
                study.enqueue_trial(chosen)
                return study

            kitchen = FreqaiDataKitchen(config, live=False, pair=PAIR)
            kitchen.label_list = [LABEL]
            kitchen.full_df = raw.iloc[500:504][["date"]].reset_index(drop=True)
            kitchen.full_df[LABEL] = [-0.2, 0.1, 0.4, -0.3]
            kitchen.full_df["do_predict"] = 1
            kitchen.full_df[_PRODUCED_PREDICTION_COLUMN] = True
            kitchen.full_df["holdout_rmse"] = [0.1, 0.2, 0.3, 0.4]
            kitchen.full_df["label_period_candles"] = 18
            kitchen.full_df["label_natr_multiplier"] = 10.5
            model.dd.get_pair_dict_info(PAIR)

            with patch.object(model, "optuna_create_study", side_effect=create_study):
                model.backtesting_fit_live_predictions(kitchen)

            self.assertEqual(model.get_optuna_params(PAIR, "label"), chosen)
            for column, value in chosen.items():
                np.testing.assert_array_equal(kitchen.full_df[column], value)
            np.testing.assert_allclose(kitchen.full_df["holdout_rmse"], [0.1, 0.2, 0.3, 0.4])
            model.live = True
            model._optuna_hyperopt = False
            model.fit_live_predictions(kitchen, PAIR)
            for column, value in chosen.items():
                self.assertEqual(kitchen.data["extra_returns_per_train"][column], value)

    def test_external_target_input_change_isolates_models_and_predictions(self):
        with temporary_directory() as directory:
            implementations = directory / "implementations"
            implementations.mkdir()
            (implementations / "funding_strategy.py").write_text(
                "from QuickAdapterV3 import QuickAdapterV3\n"
                "class FundingStrategy(QuickAdapterV3):\n"
                "    def set_freqai_targets(self, dataframe, metadata, **kwargs):\n"
                "        result = super().set_freqai_targets(dataframe, metadata, **kwargs)\n"
                "        result['&s-extrema'] *= result['funding_rate']\n"
                "        return result\n"
            )
            frame = ohlcv_frame(
                100 + 8 * np.sin(np.arange(720) / 6) + 2 * np.sin(np.arange(720) / 23)
            )
            frame["funding_rate"] = 1.0
            frame["sentiment"] = pd.Series(["neutral"] * len(frame), dtype="string")
            options = {"strategy_name": "FundingStrategy", "strategy_path": implementations}
            first = self._run_backtest(directory / "shared", frame, **options)
            frame["funding_rate"] = -1.0

            changed = self._run_backtest(directory / "shared", frame, **options)
            fresh = self._run_backtest(directory / "fresh", frame, **options)
            cached = self._run_backtest(directory / "shared", frame, **options)

            self.assertNotEqual(first.data_path.parent, changed.data_path.parent)
            self.assertEqual(changed.data_path.parent, cached.data_path.parent)
            pd.testing.assert_frame_equal(changed.return_dataframe, fresh.return_dataframe)
            pd.testing.assert_frame_equal(cached.return_dataframe, fresh.return_dataframe)
            self.assertFalse(
                np.allclose(
                    first.return_dataframe[LABEL], fresh.return_dataframe[LABEL], equal_nan=True
                )
            )

    def test_root_target_configuration_isolates_models_and_predictions(self):
        with temporary_directory() as directory:
            implementations = directory / "implementations"
            implementations.mkdir()
            (implementations / "configured_targets.py").write_text(
                "from QuickAdapterV3 import QuickAdapterV3\n"
                "class ConfiguredTargets(QuickAdapterV3):\n"
                "    def set_freqai_targets(self, dataframe, metadata, **kwargs):\n"
                "        result = super().set_freqai_targets(dataframe, metadata, **kwargs)\n"
                "        result['&s-extrema'] *= self.config['target_scale']\n"
                "        return result\n"
            )
            frame = ohlcv_frame(
                100 + 8 * np.sin(np.arange(720) / 6) + 2 * np.sin(np.arange(720) / 23)
            )
            options = {"strategy_name": "ConfiguredTargets", "strategy_path": implementations}
            first = self._run_backtest(
                directory / "shared", frame, config_overrides={"target_scale": 1.0}, **options
            )
            changed = self._run_backtest(
                directory / "shared", frame, config_overrides={"target_scale": -1.0}, **options
            )
            fresh = self._run_backtest(
                directory / "fresh", frame, config_overrides={"target_scale": -1.0}, **options
            )
            cached = self._run_backtest(
                directory / "shared", frame, config_overrides={"target_scale": -1.0}, **options
            )

            self.assertNotEqual(first.data_path.parent, changed.data_path.parent)
            self.assertEqual(changed.data_path.parent, cached.data_path.parent)
            pd.testing.assert_frame_equal(changed.return_dataframe, fresh.return_dataframe)
            pd.testing.assert_frame_equal(cached.return_dataframe, fresh.return_dataframe)
            self.assertFalse(
                np.allclose(
                    first.return_dataframe[LABEL], fresh.return_dataframe[LABEL], equal_nan=True
                )
            )

    def test_typed_target_inputs_cannot_reuse_lossy_prediction_identities(self):
        with temporary_directory() as directory:
            implementations = directory / "implementations"
            implementations.mkdir()
            (implementations / "sentiment_strategy.py").write_text(
                "import pandas as pd\n"
                "from QuickAdapterV3 import QuickAdapterV3\n"
                "class SentimentStrategy(QuickAdapterV3):\n"
                "    def set_freqai_targets(self, dataframe, metadata, **kwargs):\n"
                "        result = super().set_freqai_targets(dataframe, metadata, **kwargs)\n"
                "        sentiment = result['sentiment']\n"
                "        if isinstance(sentiment.dtype, pd.CategoricalDtype):\n"
                "            direction = 2.0 * sentiment.cat.codes - 1.0\n"
                "        else:\n"
                "            direction = sentiment.map(lambda value: 1.0 if isinstance(value, int) else -1.0)\n"
                "        result['&s-extrema'] *= direction\n"
                "        return result\n"
            )
            options = {"strategy_name": "SentimentStrategy", "strategy_path": implementations}
            for kind in ("object", "category"):
                with self.subTest(kind=kind):
                    frame = ohlcv_frame(
                        100 + 8 * np.sin(np.arange(720) / 6) + 2 * np.sin(np.arange(720) / 23)
                    )
                    values = [1 if index % 2 == 0 else "neutral" for index in range(len(frame))]
                    frame["sentiment"] = pd.Series(values, dtype=object)
                    changed = frame.copy()
                    if kind == "object":
                        changed["sentiment"] = frame["sentiment"].map(str).astype(object)
                    else:
                        frame["sentiment"] = pd.Categorical(
                            values, categories=[1, "neutral"], ordered=True
                        )
                        changed["sentiment"] = frame["sentiment"].cat.reorder_categories(
                            ["neutral", 1]
                        )

                    first = self._run_backtest(directory / kind / "shared", frame, **options)
                    reused = self._run_backtest(directory / kind / "shared", changed, **options)
                    fresh = self._run_backtest(directory / kind / "fresh", changed, **options)
                    repeated = self._run_backtest(directory / kind / "shared", changed, **options)

                    self.assertNotEqual(first.data_path.parent, reused.data_path.parent)
                    pd.testing.assert_frame_equal(reused.return_dataframe, fresh.return_dataframe)
                    self.assertNotEqual(reused.data_path.parent, repeated.data_path.parent)
                    pd.testing.assert_frame_equal(repeated.return_dataframe, fresh.return_dataframe)
                    self.assertFalse(
                        np.allclose(
                            first.return_dataframe[LABEL],
                            fresh.return_dataframe[LABEL],
                            equal_nan=True,
                        )
                    )

    def test_native_orderflow_structures_use_isolated_backtest_namespaces(self):
        with temporary_directory() as directory:
            frame = ohlcv_frame(
                100 + 8 * np.sin(np.arange(720) / 6) + 2 * np.sin(np.arange(720) / 23)
            )
            frame["orderflow"] = [{100.0: {"bid": 1.0, "ask": 2.0}} for _ in range(len(frame))]

            first = self._run_backtest(directory, frame)
            second = self._run_backtest(directory, frame)

            self.assertNotEqual(first.data_path.parent, second.data_path.parent)
            pd.testing.assert_frame_equal(first.return_dataframe, second.return_dataframe)

    def test_native_class_identity_isolates_cached_predictions_and_model_reload(self):
        with temporary_directory() as directory:
            implementations = directory / "implementations"
            implementations.mkdir()
            (implementations / "identity_strategies.py").write_text(
                "from QuickAdapterV3 import QuickAdapterV3\n"
                "class SharedStrategy(QuickAdapterV3):\n"
                "    pass\n"
                "class CacheStrategyA(SharedStrategy):\n"
                "    pass\n"
                "class CacheStrategyB(SharedStrategy):\n"
                "    def set_freqai_targets(self, dataframe, metadata, **kwargs):\n"
                "        result = super().set_freqai_targets(dataframe, metadata, **kwargs)\n"
                "        result['&s-extrema'] = -result['&s-extrema']\n"
                "        return result\n"
            )
            (implementations / "identity_models.py").write_text(
                "from quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 import QuickAdapterRegressorV3\n"
                "class CacheRegressorA(QuickAdapterRegressorV3):\n"
                "    pass\n"
                "class CacheRegressorB(CacheRegressorA):\n"
                "    def predict(self, unfiltered_df, dk, **kwargs):\n"
                "        result, status = super().predict(unfiltered_df, dk, **kwargs)\n"
                "        result['&s-extrema'] = -result['&s-extrema']\n"
                "        return result, status\n"
            )
            frame = ohlcv_frame(100.0 + np.sin(np.arange(720) / 9.0) * 4.0)
            names = {"strategy_path": implementations, "model_path": implementations}
            baseline = self._run_backtest(
                directory / "shared",
                frame,
                strategy_name="CacheStrategyA",
                model_name="CacheRegressorA",
                **names,
            )
            for strategy_name, model_name in (
                ("CacheStrategyB", "CacheRegressorA"),
                ("CacheStrategyA", "CacheRegressorB"),
            ):
                with self.subTest(strategy=strategy_name, model=model_name):
                    options = dict(names, strategy_name=strategy_name, model_name=model_name)
                    cached = self._run_backtest(directory / "shared", frame, **options)
                    fresh = self._run_backtest(
                        directory / model_name / strategy_name, frame, **options
                    )
                    self.assertNotEqual(cached.data_path.parent, baseline.data_path.parent)
                    pd.testing.assert_frame_equal(cached.return_dataframe, fresh.return_dataframe)
                    baseline_values = baseline.return_dataframe[LABEL].dropna().to_numpy()
                    cached_values = cached.return_dataframe[LABEL].dropna().to_numpy()
                    self.assertFalse(np.array_equal(baseline_values, cached_values))
                    if model_name == "CacheRegressorB":
                        np.testing.assert_allclose(cached_values, -baseline_values)

                    path = cached.backtesting_results_path
                    damaged = pd.read_feather(path).drop(columns=[_PRODUCED_PREDICTION_COLUMN])
                    damaged.to_feather(path)
                    resumed = self._run_backtest(directory / "shared", frame, **options)
                    self.assertEqual(resumed.data_path.parent, cached.data_path.parent)
                    pd.testing.assert_frame_equal(resumed.return_dataframe, fresh.return_dataframe)

    def test_inherited_source_change_invalidates_native_strategy_cache(self):
        with temporary_directory() as directory:
            implementations = directory / "implementations"
            implementations.mkdir()
            base_name = f"_qa_cache_base_{directory.name.replace('-', '_')}"
            self.addCleanup(sys.modules.pop, base_name, None)
            base_path = implementations / f"{base_name}.py"
            base_source = (
                "from QuickAdapterV3 import QuickAdapterV3\n"
                "class CacheBaseStrategy(QuickAdapterV3):\n"
                "    target_scale = 1.0\n"
                "    def set_freqai_targets(self, dataframe, metadata, **kwargs):\n"
                "        result = super().set_freqai_targets(dataframe, metadata, **kwargs)\n"
                "        result['&s-extrema'] *= self.target_scale\n"
                "        return result\n"
            )
            base_path.write_text(base_source)
            (implementations / "inherited_strategy.py").write_text(
                f"from {base_name} import CacheBaseStrategy\n"
                "class InheritedCacheStrategy(CacheBaseStrategy):\n"
                "    pass\n"
            )
            frame = ohlcv_frame(100.0 + np.sin(np.arange(720) / 9.0) * 4.0)
            options = {"strategy_name": "InheritedCacheStrategy", "strategy_path": implementations}
            first = self._run_backtest(directory / "shared", frame, **options)

            base_path.write_text(base_source.replace("target_scale = 1.0", "target_scale = -1.0"))
            sys.modules.pop(base_name)
            importlib.invalidate_caches()
            changed = self._run_backtest(directory / "shared", frame, **options)
            fresh = self._run_backtest(directory / "fresh", frame, **options)

            self.assertNotEqual(first.data_path.parent, changed.data_path.parent)
            pd.testing.assert_frame_equal(changed.return_dataframe, fresh.return_dataframe)
            self.assertFalse(
                np.array_equal(
                    first.return_dataframe[LABEL].dropna().to_numpy(),
                    changed.return_dataframe[LABEL].dropna().to_numpy(),
                )
            )

    def test_shifted_cache_is_recomputed_and_changed_inputs_isolate_models(self):
        with temporary_directory() as directory:
            position = np.arange(720)
            frame = ohlcv_frame(100 + 8 * np.sin(position / 6) + 2 * np.sin(position / 23))
            frame["date"] = pd.to_datetime(frame["date"], utc=True)
            original = self._run_backtest(directory, frame)
            result = original.return_dataframe
            available = result["date"].between(
                pd.Timestamp("2024-01-02", tz="UTC"),
                pd.Timestamp("2024-01-03", tz="UTC"),
                inclusive="left",
            )
            np.testing.assert_array_equal(result.loc[available, "prediction_available"], 1)
            self.assertTrue(result.loc[~available, LABEL].isna().all())
            for column in ("extrema_direction_smoothed", "extrema_weight_smoothed"):
                self.assertTrue(np.isfinite(result.loc[available, column]).all())
                self.assertNotIn(column, original.training_features_list)
            for damage in ("dates", LABEL, "do_predict", "extrema_direction_smoothed"):
                with self.subTest(cache_damage=damage):
                    cached = pd.read_feather(original.backtesting_results_path)
                    if damage == "dates":
                        cached["date"] += pd.Timedelta(minutes=5)
                    else:
                        cached = cached.drop(columns=[damage])
                    cached.to_feather(original.backtesting_results_path)

                    restored = self._run_backtest(directory, frame)

                    pd.testing.assert_series_equal(restored.return_dataframe[LABEL], result[LABEL])
                    np.testing.assert_array_equal(
                        restored.return_dataframe.loc[available, "prediction_available"], 1
                    )
                    np.testing.assert_array_equal(
                        restored.return_dataframe.loc[available, "do_predict"], 1
                    )
            changed = frame.copy()
            changed["volume"] = 10 + np.sin(position / 3)
            different = self._run_backtest(directory, changed)
            self.assertNotEqual(original.data_path.parent, different.data_path.parent)
            self.assertTrue(
                (original.data_path / f"{original.model_filename}_model.joblib").is_file()
            )
            self.assertTrue(
                (different.data_path / f"{different.model_filename}_model.joblib").is_file()
            )


if __name__ == "__main__":
    unittest.main()
