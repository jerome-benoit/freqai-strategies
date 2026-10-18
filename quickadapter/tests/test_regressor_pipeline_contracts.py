"""Prediction validity and transport contracts in the native Freqtrade QA image."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from freqtrade.enums import RunMode
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.plot.plotting import add_indicators
from plotly.subplots import make_subplots
from qa_support import (
    CONFIG_TEMPLATE,
    PAIR,
    QaTestCase,
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

    def test_old_history_accepts_new_metrics_without_fabricating_past_scores(self):
        with temporary_directory() as directory:
            config = model_config(directory, freqai={"optuna_hyperopt": {"enabled": False}})
            model = QuickAdapterRegressorV3(config=config)
            model.live = True
            kitchen = FreqaiDataKitchen(config, live=True, pair=PAIR)
            kitchen.label_list = [LABEL]
            kitchen.data.update(
                labels_mean={LABEL: 0.0},
                labels_std={LABEL: 0.1},
                extra_returns_per_train={"holdout_rmse": 0.25, "label_period_candles": 1},
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
            result = model._normalise_prediction_returns(model.dd.historic_predictions[PAIR])

            np.testing.assert_allclose(result["holdout_rmse"], [np.nan, np.nan, 0.25])
            np.testing.assert_allclose(result["prediction_available"], [1, 1, 1])
            np.testing.assert_allclose(result[LABEL], [0.1, 0.0, 0.0])

    def test_native_no_model_return_does_not_expose_zero_predictions(self):
        with temporary_directory() as directory:
            config = model_config(directory, freqai={"optuna_hyperopt": {"enabled": False}})
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
    def _run_backtest(directory: Path, frame: pd.DataFrame, weighting: str = "uniform"):
        config = model_config(
            directory,
            freqai={
                "continual_learning": False,
                "save_backtest_models": True,
                "activate_tensorboard": False,
                "label_weighting": {"strategy": weighting},
                "optuna_hyperopt": {"enabled": False},
                "feature_parameters": {
                    "indicator_periods_candles": [4, 8],
                    "include_shifted_candles": 0,
                    "buffer_train_data_candles": 0,
                    "causal_mode": True,
                },
                "data_split_parameters": {"test_size": 0.2, "shuffle": False},
                "model_training_parameters": {"n_estimators": 3, "n_jobs": 1, "verbosity": 0},
            },
        )
        config.update(
            runmode=RunMode.BACKTEST,
            timerange="20240102-20240103",
            config_files=[str(CONFIG_TEMPLATE)],
        )
        model = QuickAdapterRegressorV3(config=config)
        model.live = False
        strategy = QuickAdapterV3(config)
        strategy.freqai = model
        strategy.freqai_info = config["freqai"]
        strategy.bot_start()
        kitchen = FreqaiDataKitchen(config, live=False, pair=PAIR)
        model.dk = kitchen
        model.start_backtesting(frame, {"pair": PAIR}, kitchen, strategy)
        return kitchen

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
