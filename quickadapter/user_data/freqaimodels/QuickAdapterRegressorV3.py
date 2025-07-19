import copy
from enum import IntEnum
import logging
import json
import math
import random
from statistics import median
import time
import numpy as np
import pandas as pd
import scipy as sp
import optuna
import sklearn
import warnings
import talib.abstract as ta

from functools import cached_property, lru_cache
from typing import Any, Callable, Optional
from pathlib import Path
from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

debug = False

TEST_SIZE = 0.1

EXTREMA_COLUMN = "&s-extrema"
MAXIMA_THRESHOLD_COLUMN = "&s-maxima_threshold"
MINIMA_THRESHOLD_COLUMN = "&s-minima_threshold"

warnings.simplefilter(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class QuickAdapterRegressorV3(BaseRegressionModel):
    """
    The following freqaimodel is released to sponsors of the non-profit FreqAI open-source project.
    If you find the FreqAI project useful, please consider supporting it by becoming a sponsor.
    We use sponsor money to help stimulate new features and to pay for running these public
    experiments, with a an objective of helping the community make smarter choices in their
    ML journey.

    This freqaimodel is experimental (as with all models released to sponsors). Do *not* expect
    returns. The goal is to demonstrate gratitude to people who support the project and to
    help them find a good starting point for their own creativity.

    If you have questions, please direct them to our discord: https://discord.gg/xE4RMg4QYw

    https://github.com/sponsors/robcaulk
    """

    version = "3.7.94"

    @cached_property
    def _optuna_config(self) -> dict[str, Any]:
        optuna_default_config = {
            "enabled": False,
            "n_jobs": min(
                self.freqai_info.get("optuna_hyperopt", {}).get("n_jobs", 1),
                max(int(self.max_system_threads / 4), 1),
            ),
            "storage": "file",
            "continuous": True,
            "warm_start": True,
            "n_startup_trials": 15,
            "n_trials": 36,
            "timeout": 7200,
            "label_candles_step": 4,
            "train_candles_step": 10,
            "expansion_factor": 0.4,
            "seed": 1,
        }
        return {
            **optuna_default_config,
            **self.freqai_info.get("optuna_hyperopt", {}),
        }

    @property
    def _optuna_label_candle_pool_full(self) -> list[int]:
        if not hasattr(self, "pairs") or not self.pairs:
            raise RuntimeError(
                "Failed to initialize optuna label candle pool full: pairs property is not defined or empty"
            )
        n_pairs = len(self.pairs)
        label_frequency_candles = max(
            2, 2 * n_pairs, int(self.ft_params.get("label_frequency_candles", 12))
        )
        cache_key = label_frequency_candles
        if cache_key not in self._optuna_label_candle_pool_full_cache:
            half_label_frequency_candles = int(label_frequency_candles / 2)
            min_offset = -half_label_frequency_candles
            max_offset = half_label_frequency_candles
            self._optuna_label_candle_pool_full_cache[cache_key] = [
                max(1, label_frequency_candles + offset)
                for offset in range(min_offset, max_offset + 1)
            ]
        return copy.deepcopy(self._optuna_label_candle_pool_full_cache[cache_key])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pairs = self.config.get("exchange", {}).get("pair_whitelist")
        if not self.pairs:
            raise ValueError(
                "FreqAI model requires StaticPairList method defined in pairlists configuration and 'pair_whitelist' defined in exchange section configuration"
            )
        if (
            not isinstance(self.freqai_info.get("identifier"), str)
            or self.freqai_info.get("identifier").strip() == ""
        ):
            raise ValueError(
                "FreqAI model requires 'identifier' defined in the freqai section configuration"
            )
        self._optuna_hyperopt: Optional[bool] = (
            self.freqai_info.get("enabled", False)
            and self._optuna_config.get("enabled")
            and self.data_split_parameters.get("test_size", TEST_SIZE) > 0
        )
        self._optuna_hp_value: dict[str, float] = {}
        self._optuna_train_value: dict[str, float] = {}
        self._optuna_label_values: dict[str, list] = {}
        self._optuna_hp_params: dict[str, dict[str, Any]] = {}
        self._optuna_train_params: dict[str, dict[str, Any]] = {}
        self._optuna_label_params: dict[str, dict[str, Any]] = {}
        self._optuna_label_candle_pool_full_cache: dict[int, list[int]] = {}
        self.init_optuna_label_candle_pool()
        self._optuna_label_candle: dict[str, int] = {}
        self._optuna_label_candles: dict[str, int] = {}
        self._optuna_label_incremented_pairs: list[str] = []
        for pair in self.pairs:
            self._optuna_hp_value[pair] = -1
            self._optuna_train_value[pair] = -1
            self._optuna_label_values[pair] = [-1, -1]
            self._optuna_hp_params[pair] = (
                self.optuna_load_best_params(pair, "hp")
                if self.optuna_load_best_params(pair, "hp")
                else {}
            )
            self._optuna_train_params[pair] = (
                self.optuna_load_best_params(pair, "train")
                if self.optuna_load_best_params(pair, "train")
                else {}
            )
            self._optuna_label_params[pair] = (
                self.optuna_load_best_params(pair, "label")
                if self.optuna_load_best_params(pair, "label")
                else {
                    "label_period_candles": self.ft_params.get(
                        "label_period_candles", 50
                    ),
                    "label_natr_ratio": float(
                        self.ft_params.get("label_natr_ratio", 6.0)
                    ),
                }
            )
            self.set_optuna_label_candle(pair)
            self._optuna_label_candles[pair] = 0

        logger.info(
            f"Initialized {self.__class__.__name__} {self.freqai_info.get('regressor', 'xgboost')} regressor model version {self.version}"
        )

    def get_optuna_params(self, pair: str, namespace: str) -> dict[str, Any]:
        if namespace == "hp":
            params = self._optuna_hp_params.get(pair)
        elif namespace == "train":
            params = self._optuna_train_params.get(pair)
        elif namespace == "label":
            params = self._optuna_label_params.get(pair)
        else:
            raise ValueError(f"Invalid namespace: {namespace}")
        return params

    def set_optuna_params(
        self, pair: str, namespace: str, params: dict[str, Any]
    ) -> None:
        if namespace == "hp":
            self._optuna_hp_params[pair] = params
        elif namespace == "train":
            self._optuna_train_params[pair] = params
        elif namespace == "label":
            self._optuna_label_params[pair] = params
        else:
            raise ValueError(f"Invalid namespace: {namespace}")

    def get_optuna_value(self, pair: str, namespace: str) -> float:
        if namespace == "hp":
            value = self._optuna_hp_value.get(pair)
        elif namespace == "train":
            value = self._optuna_train_value.get(pair)
        else:
            raise ValueError(f"Invalid namespace: {namespace}")
        return value

    def set_optuna_value(self, pair: str, namespace: str, value: float) -> None:
        if namespace == "hp":
            self._optuna_hp_value[pair] = value
        elif namespace == "train":
            self._optuna_train_value[pair] = value
        else:
            raise ValueError(f"Invalid namespace: {namespace}")

    def get_optuna_values(self, pair: str, namespace: str) -> list:
        if namespace == "label":
            values = self._optuna_label_values.get(pair)
        else:
            raise ValueError(f"Invalid namespace: {namespace}")
        return values

    def set_optuna_values(self, pair: str, namespace: str, values: list) -> None:
        if namespace == "label":
            self._optuna_label_values[pair] = values
        else:
            raise ValueError(f"Invalid namespace: {namespace}")

    def init_optuna_label_candle_pool(self) -> None:
        optuna_label_candle_pool_full = self._optuna_label_candle_pool_full
        if len(optuna_label_candle_pool_full) == 0:
            raise RuntimeError("Failed to initialize optuna label candle pool full")
        self._optuna_label_candle_pool = optuna_label_candle_pool_full
        random.shuffle(self._optuna_label_candle_pool)
        if len(self._optuna_label_candle_pool) == 0:
            raise RuntimeError("Failed to initialize optuna label candle pool")

    def set_optuna_label_candle(self, pair: str) -> None:
        if len(self._optuna_label_candle_pool) == 0:
            logger.warning(
                "Optuna label candle pool is empty, reinitializing it ("
                f"{self._optuna_label_candle_pool=} ,"
                f"{self._optuna_label_candle_pool_full=} ,"
                f"{self._optuna_label_candle.values()=} ,"
                f"{self._optuna_label_candles.values()=} ,"
                f"{self._optuna_label_incremented_pairs=})"
            )
            self.init_optuna_label_candle_pool()
        optuna_label_candle_pool = copy.deepcopy(self._optuna_label_candle_pool)
        for p in self.pairs:
            if p == pair:
                continue
            optuna_label_candle = self._optuna_label_candle.get(p)
            optuna_label_candles = self._optuna_label_candles.get(p)
            if optuna_label_candle is not None and optuna_label_candles is not None:
                if (
                    self._optuna_label_incremented_pairs
                    and p not in self._optuna_label_incremented_pairs
                ):
                    optuna_label_candles += 1
                remaining_candles = optuna_label_candle - optuna_label_candles
                if remaining_candles in optuna_label_candle_pool:
                    optuna_label_candle_pool.remove(remaining_candles)
        optuna_label_candle = optuna_label_candle_pool.pop()
        self._optuna_label_candle[pair] = optuna_label_candle
        self._optuna_label_candle_pool.remove(optuna_label_candle)
        optuna_label_available_candles = (
            set(self._optuna_label_candle_pool_full)
            - set(self._optuna_label_candle_pool)
            - set(self._optuna_label_candle.values())
        )
        if len(optuna_label_available_candles) > 0:
            self._optuna_label_candle_pool.extend(optuna_label_available_candles)
            random.shuffle(self._optuna_label_candle_pool)

    def fit(
        self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        User sets up the training and test data to fit their desired model here
        :param data_dictionary: the dictionary constructed by DataHandler to hold
                                all the training and test data/labels.
        :param dk: the FreqaiDataKitchen object
        """

        X = data_dictionary.get("train_features")
        y = data_dictionary.get("train_labels")
        train_weights = data_dictionary.get("train_weights")

        X_test = data_dictionary.get("test_features")
        y_test = data_dictionary.get("test_labels")
        test_weights = data_dictionary.get("test_weights")

        model_training_parameters = self.model_training_parameters

        start = time.time()
        if self._optuna_hyperopt:
            self.optuna_optimize(
                pair=dk.pair,
                namespace="hp",
                objective=lambda trial: hp_objective(
                    trial,
                    self.freqai_info.get("regressor", "xgboost"),
                    X,
                    y,
                    train_weights,
                    X_test,
                    y_test,
                    test_weights,
                    self.get_optuna_params(dk.pair, "hp"),
                    model_training_parameters,
                    self._optuna_config.get("expansion_factor"),
                ),
                direction=optuna.study.StudyDirection.MINIMIZE,
            )

            optuna_hp_params = self.get_optuna_params(dk.pair, "hp")
            if optuna_hp_params:
                model_training_parameters = {
                    **model_training_parameters,
                    **optuna_hp_params,
                }

            self.optuna_optimize(
                pair=dk.pair,
                namespace="train",
                objective=lambda trial: train_objective(
                    trial,
                    self.freqai_info.get("regressor", "xgboost"),
                    X,
                    y,
                    train_weights,
                    X_test,
                    y_test,
                    test_weights,
                    self.data_split_parameters.get("test_size", TEST_SIZE),
                    self.freqai_info.get("fit_live_predictions_candles", 100),
                    self._optuna_config.get("train_candles_step"),
                    model_training_parameters,
                ),
                direction=optuna.study.StudyDirection.MINIMIZE,
            )

            optuna_train_params = self.get_optuna_params(dk.pair, "train")
            if optuna_train_params:
                value: float = optuna_train_params.get("value")
                if isinstance(value, float) and np.isfinite(value):
                    train_window = optuna_train_params.get("train_period_candles")
                    if isinstance(train_window, int) and train_window > 0:
                        X = X.iloc[-train_window:]
                        y = y.iloc[-train_window:]
                        train_weights = train_weights[-train_window:]

                    test_window = optuna_train_params.get("test_period_candles")
                    if isinstance(test_window, int) and test_window > 0:
                        X_test = X_test.iloc[-test_window:]
                        y_test = y_test.iloc[-test_window:]
                        test_weights = test_weights[-test_window:]

        eval_set, eval_weights = self.eval_set_and_weights(X_test, y_test, test_weights)

        model = fit_regressor(
            regressor=self.freqai_info.get("regressor", "xgboost"),
            X=X,
            y=y,
            train_weights=train_weights,
            eval_set=eval_set,
            eval_weights=eval_weights,
            model_training_parameters=model_training_parameters,
            init_model=self.get_init_model(dk.pair),
        )
        time_spent = time.time() - start
        self.dd.update_metric_tracker("fit_time", time_spent, dk.pair)

        return model

    def optuna_throttle_callback(
        self,
        pair: str,
        namespace: str,
        callback: Callable[[], None],
    ) -> None:
        if namespace != "label":
            raise ValueError(f"Invalid namespace: {namespace}")
        self._optuna_label_candles[pair] += 1
        if pair not in self._optuna_label_incremented_pairs:
            self._optuna_label_incremented_pairs.append(pair)
        optuna_label_remaining_candles = self._optuna_label_candle.get(
            pair
        ) - self._optuna_label_candles.get(pair)
        if optuna_label_remaining_candles <= 0:
            try:
                callback()
            except Exception as e:
                logger.error(
                    f"Error executing optuna {pair} {namespace} callback: {str(e)}",
                    exc_info=True,
                )
            finally:
                self.set_optuna_label_candle(pair)
                self._optuna_label_candles[pair] = 0
        else:
            logger.info(
                f"Optuna {pair} {namespace} callback throttled, still {optuna_label_remaining_candles} candles to go"
            )
        if len(self._optuna_label_incremented_pairs) >= len(self.pairs):
            self._optuna_label_incremented_pairs = []

    def fit_live_predictions(self, dk: FreqaiDataKitchen, pair: str) -> None:
        warmed_up = True

        fit_live_predictions_candles = self.freqai_info.get(
            "fit_live_predictions_candles", 100
        )

        if self._optuna_hyperopt:
            self.optuna_throttle_callback(
                pair=pair,
                namespace="label",
                callback=lambda: self.optuna_optimize(
                    pair=pair,
                    namespace="label",
                    objective=lambda trial: label_objective(
                        trial,
                        self.data_provider.get_pair_dataframe(
                            pair=pair, timeframe=self.config.get("timeframe")
                        ),
                        fit_live_predictions_candles,
                        self._optuna_config.get("label_candles_step"),
                    ),
                    directions=[
                        optuna.study.StudyDirection.MAXIMIZE,
                        optuna.study.StudyDirection.MAXIMIZE,
                    ],
                ),
            )

        if self.live:
            if not hasattr(self, "exchange_candles"):
                self.exchange_candles = len(self.dd.model_return_values[pair].index)
            candles_diff = len(self.dd.historic_predictions[pair].index) - (
                fit_live_predictions_candles + self.exchange_candles
            )
            if candles_diff < 0:
                logger.warning(
                    f"{pair}: fit live predictions not warmed up yet, still {abs(candles_diff)} candles to go"
                )
                warmed_up = False

        pred_df_full = (
            self.dd.historic_predictions[pair]
            .iloc[-fit_live_predictions_candles:]
            .reset_index(drop=True)
        )

        if not warmed_up:
            dk.data["extra_returns_per_train"][MINIMA_THRESHOLD_COLUMN] = -2
            dk.data["extra_returns_per_train"][MAXIMA_THRESHOLD_COLUMN] = 2
        else:
            min_pred, max_pred = self.min_max_pred(
                pred_df_full,
                fit_live_predictions_candles,
                self.get_optuna_params(pair, "label").get("label_period_candles"),
            )
            dk.data["extra_returns_per_train"][MINIMA_THRESHOLD_COLUMN] = min_pred
            dk.data["extra_returns_per_train"][MAXIMA_THRESHOLD_COLUMN] = max_pred

        dk.data["labels_mean"], dk.data["labels_std"] = {}, {}
        for label in dk.label_list + dk.unique_class_list:
            pred_df_full_label = pred_df_full.get(label)
            if pred_df_full_label is None or pred_df_full_label.dtype == object:
                continue
            if not warmed_up:
                f = [0, 0]
            else:
                f = sp.stats.norm.fit(pred_df_full_label)
            dk.data["labels_mean"][label], dk.data["labels_std"][label] = f[0], f[1]

        di_values = pred_df_full.get("DI_values")

        # fit the DI_threshold
        if not warmed_up:
            f = [0, 0, 0]
            cutoff = 2
        else:
            f = sp.stats.weibull_min.fit(
                pd.to_numeric(di_values, errors="coerce").dropna()
            )
            cutoff = sp.stats.weibull_min.ppf(
                self.freqai_info.get("outlier_threshold", 0.999), *f
            )

        dk.data["DI_value_mean"] = di_values.mean()
        dk.data["DI_value_std"] = di_values.std()
        dk.data["extra_returns_per_train"]["DI_value_param1"] = f[0]
        dk.data["extra_returns_per_train"]["DI_value_param2"] = f[1]
        dk.data["extra_returns_per_train"]["DI_value_param3"] = f[2]
        dk.data["extra_returns_per_train"]["DI_cutoff"] = cutoff

        dk.data["extra_returns_per_train"]["label_period_candles"] = (
            self.get_optuna_params(pair, "label").get("label_period_candles")
        )
        dk.data["extra_returns_per_train"]["label_natr_ratio"] = self.get_optuna_params(
            pair, "label"
        ).get("label_natr_ratio")

        dk.data["extra_returns_per_train"]["hp_rmse"] = self.get_optuna_value(
            pair, "hp"
        )
        dk.data["extra_returns_per_train"]["train_rmse"] = self.get_optuna_value(
            pair, "train"
        )

    def eval_set_and_weights(
        self, X_test: pd.DataFrame, y_test: pd.DataFrame, test_weights: np.ndarray
    ) -> tuple[
        Optional[list[tuple[pd.DataFrame, pd.DataFrame]]], Optional[list[np.ndarray]]
    ]:
        if self.data_split_parameters.get("test_size", TEST_SIZE) == 0:
            eval_set = None
            eval_weights = None
        else:
            eval_set = [(X_test, y_test)]
            eval_weights = [test_weights]

        return eval_set, eval_weights

    def min_max_pred(
        self,
        pred_df: pd.DataFrame,
        fit_live_predictions_candles: int,
        label_period_candles: int,
    ) -> tuple[float, float]:
        temperature = float(
            self.freqai_info.get("prediction_thresholds_temperature", 300.0)
        )
        extrema = pred_df.get(EXTREMA_COLUMN).iloc[
            -(
                max(2, int(fit_live_predictions_candles / label_period_candles))
                * label_period_candles
            ) :
        ]
        min_pred = smoothed_min(extrema, temperature=temperature)
        max_pred = smoothed_max(extrema, temperature=temperature)
        return min_pred, max_pred

    def get_multi_objective_study_best_trial(
        self, namespace: str, study: optuna.study.Study
    ) -> Optional[optuna.trial.FrozenTrial]:
        if namespace != "label":
            raise ValueError(f"Invalid namespace: {namespace}")
        n_objectives = len(study.directions)
        if n_objectives < 2:
            raise ValueError(
                f"Multi-objective study must have at least 2 objectives, but got {n_objectives}"
            )
        if not QuickAdapterRegressorV3.optuna_study_has_best_trials(study):
            return None

        metrics = {
            "braycurtis",
            "canberra",
            "chebyshev",
            "cityblock",
            "correlation",
            "cosine",
            "dice",
            "euclidean",
            "hamming",
            "jaccard",
            "jensenshannon",
            "kulczynski1",
            "mahalanobis",
            "matching",
            "minkowski",
            "rogerstanimoto",
            "russellrao",
            "seuclidean",
            "sokalmichener",
            "sokalsneath",
            "sqeuclidean",
            "yule",
            "hellinger",
            "shellinger",
            "geometric_mean",
            "harmonic_mean",
            "power_mean",
            "weighted_sum",
            "kmeans",
            "kmeans2",
            "knn_d1",
            "knn_d2_mean",
            "knn_d2_median",
            "knn_d2_max",
        }
        label_metric = self.ft_params.get("label_metric", "seuclidean")
        if label_metric not in metrics:
            raise ValueError(
                f"Unsupported label metric: {label_metric}. Supported metrics are {', '.join(metrics)}"
            )

        best_trials = [
            trial
            for trial in study.best_trials
            if (
                trial.values is not None
                and len(trial.values) == n_objectives
                and all(
                    isinstance(value, (int, float)) and not np.isnan(value)
                    for value in trial.values
                )
            )
        ]
        if not best_trials:
            return None

        def calculate_distances(
            normalized_matrix: np.ndarray, metric: str
        ) -> np.ndarray:
            n_objectives = normalized_matrix.shape[1]
            n_samples = normalized_matrix.shape[0]
            label_p_order = float(self.ft_params.get("label_p_order", 2.0))
            np_weights = np.array(
                self.ft_params.get("label_weights", [1.0] * n_objectives)
            )
            if np_weights.size != n_objectives:
                raise ValueError("label_weights length must match number of objectives")
            if np.any(np_weights < 0):
                raise ValueError("label_weights values must be non-negative")
            label_weights_sum = np.sum(np_weights)
            if np.isclose(label_weights_sum, 0):
                raise ValueError("label_weights sum cannot be zero")
            np_weights = np_weights / label_weights_sum
            knn_kwargs = {}
            label_knn_metric = self.ft_params.get("label_knn_metric", "seuclidean")
            if label_knn_metric == "minkowski" and isinstance(label_p_order, float):
                knn_kwargs["p"] = label_p_order

            ideal_point = np.ones(n_objectives)
            ideal_point_2d = ideal_point.reshape(1, -1)

            if metric in {
                "braycurtis",
                "canberra",
                "chebyshev",
                "cityblock",
                "correlation",
                "cosine",
                "dice",
                "euclidean",
                "hamming",
                "jaccard",
                "jensenshannon",
                "kulczynski1",  # deprecated since version 1.15.0
                "mahalanobis",
                "matching",
                "minkowski",
                "rogerstanimoto",
                "russellrao",
                "seuclidean",
                "sokalmichener",  # deprecated since version 1.15.0
                "sokalsneath",
                "sqeuclidean",
                "yule",
            }:
                cdist_kwargs = {"w": np_weights}
                if metric in {
                    "jensenshannon",
                    "mahalanobis",
                    "seuclidean",
                }:
                    del cdist_kwargs["w"]
                if metric == "minkowski" and isinstance(label_p_order, float):
                    cdist_kwargs["p"] = label_p_order
                return sp.spatial.distance.cdist(
                    normalized_matrix,
                    ideal_point_2d,
                    metric=metric,
                    **cdist_kwargs,
                ).flatten()
            elif metric in {"hellinger", "shellinger"}:
                np_sqrt_normalized_matrix = np.sqrt(normalized_matrix)
                if metric == "shellinger":
                    np_weights = 1 / np.var(np_sqrt_normalized_matrix, axis=0, ddof=1)
                return np.sqrt(
                    np.sum(
                        np_weights
                        * (np_sqrt_normalized_matrix - np.sqrt(ideal_point)) ** 2,
                        axis=1,
                    )
                ) / np.sqrt(2.0)
            elif metric in {"geometric_mean", "harmonic_mean", "power_mean"}:
                p = {
                    "geometric_mean": 0.0,
                    "harmonic_mean": -1.0,
                    "power_mean": label_p_order,
                }[metric]
                return sp.stats.pmean(
                    ideal_point, p=p, weights=np_weights
                ) - sp.stats.pmean(normalized_matrix, p=p, weights=np_weights, axis=1)
            elif metric == "weighted_sum":
                return np.sum(np_weights * (ideal_point - normalized_matrix), axis=1)
            elif metric in {"kmeans", "kmeans2"}:
                if n_samples < 2:
                    return np.full(n_samples, np.inf)
                n_clusters = min(max(2, int(np.sqrt(n_samples / 2))), 10, n_samples)
                if metric == "kmeans":
                    kmeans = sklearn.cluster.KMeans(
                        n_clusters=n_clusters, random_state=42, n_init=10
                    )
                    cluster_labels = kmeans.fit_predict(normalized_matrix)
                    cluster_centers = kmeans.cluster_centers_
                elif metric == "kmeans2":
                    cluster_centers, cluster_labels = sp.cluster.vq.kmeans2(
                        normalized_matrix, n_clusters, rng=42, minit="++"
                    )
                label_kmeans_metric = self.ft_params.get(
                    "label_kmeans_metric", "seuclidean"
                )
                cdist_kwargs = {}
                if label_kmeans_metric == "minkowski" and isinstance(
                    label_p_order, float
                ):
                    cdist_kwargs["p"] = label_p_order
                cluster_distances_to_ideal = sp.spatial.distance.cdist(
                    cluster_centers,
                    ideal_point_2d,
                    metric=label_kmeans_metric,
                    **cdist_kwargs,
                ).flatten()
                return cluster_distances_to_ideal[cluster_labels]
            elif metric == "knn_d1":
                if n_samples < 2:
                    return np.full(n_samples, np.inf)
                nbrs = sklearn.neighbors.NearestNeighbors(
                    n_neighbors=2, metric=label_knn_metric, **knn_kwargs
                ).fit(normalized_matrix)
                distances, _ = nbrs.kneighbors(normalized_matrix)
                return distances[:, 1]
            elif metric in {"knn_d2_mean", "knn_d2_median", "knn_d2_max"}:
                if n_samples < 2:
                    return np.full(n_samples, np.inf)
                n_neighbors = (
                    min(
                        int(self.ft_params.get("label_knn_d2_n_neighbors", 4)),
                        n_samples - 1,
                    )
                    + 1
                )
                nbrs = sklearn.neighbors.NearestNeighbors(
                    n_neighbors=n_neighbors, metric=label_knn_metric, **knn_kwargs
                ).fit(normalized_matrix)
                distances, _ = nbrs.kneighbors(normalized_matrix)
                if metric == "knn_d2_mean":
                    return np.mean(distances[:, 1:], axis=1)
                elif metric == "knn_d2_median":
                    return np.median(distances[:, 1:], axis=1)
                elif metric == "knn_d2_max":
                    return np.max(distances[:, 1:], axis=1)
            else:
                raise ValueError(
                    f"Unsupported label metric: {metric}. Supported metrics are {', '.join(metrics)}"
                )

        objective_values_matrix = np.array([trial.values for trial in best_trials])
        normalized_matrix = np.zeros_like(objective_values_matrix, dtype=float)

        for i in range(objective_values_matrix.shape[1]):
            current_column = objective_values_matrix[:, i]
            current_direction = study.directions[i]

            is_neg_inf_mask = np.isneginf(current_column)
            is_pos_inf_mask = np.isposinf(current_column)
            if current_direction == optuna.study.StudyDirection.MAXIMIZE:
                normalized_matrix[is_neg_inf_mask, i] = 0.0
                normalized_matrix[is_pos_inf_mask, i] = 1.0
            else:
                normalized_matrix[is_neg_inf_mask, i] = 1.0
                normalized_matrix[is_pos_inf_mask, i] = 0.0

            is_finite_mask = np.isfinite(current_column)

            if np.any(is_finite_mask):
                finite_col = current_column[is_finite_mask]
                finite_min_val = np.min(finite_col)
                finite_max_val = np.max(finite_col)
                finite_range_val = finite_max_val - finite_min_val

                if np.isclose(finite_range_val, 0):
                    if np.any(is_pos_inf_mask) and np.any(is_neg_inf_mask):
                        normalized_matrix[is_finite_mask, i] = 0.5
                    elif np.any(is_pos_inf_mask):
                        normalized_matrix[is_finite_mask, i] = (
                            0.0
                            if current_direction == optuna.study.StudyDirection.MAXIMIZE
                            else 1.0
                        )
                    elif np.any(is_neg_inf_mask):
                        normalized_matrix[is_finite_mask, i] = (
                            1.0
                            if current_direction == optuna.study.StudyDirection.MAXIMIZE
                            else 0.0
                        )
                    else:
                        normalized_matrix[is_finite_mask, i] = 0.5
                else:
                    if current_direction == optuna.study.StudyDirection.MAXIMIZE:
                        normalized_matrix[is_finite_mask, i] = (
                            finite_col - finite_min_val
                        ) / finite_range_val
                    else:
                        normalized_matrix[is_finite_mask, i] = (
                            finite_max_val - finite_col
                        ) / finite_range_val

        trial_distances = calculate_distances(normalized_matrix, metric=label_metric)

        return best_trials[np.argmin(trial_distances)]

    def optuna_optimize(
        self,
        pair: str,
        namespace: str,
        objective: Callable[[optuna.trial.Trial], float],
        direction: Optional[optuna.study.StudyDirection] = None,
        directions: Optional[list[optuna.study.StudyDirection]] = None,
    ) -> None:
        is_study_single_objective = direction is not None and directions is None
        if not is_study_single_objective and len(directions) < 2:
            raise ValueError(
                "Multi-objective study must have at least 2 directions specified"
            )

        study = self.optuna_create_study(
            pair=pair,
            namespace=namespace,
            direction=direction,
            directions=directions,
        )
        if not study:
            return

        if self._optuna_config.get("warm_start"):
            self.optuna_enqueue_previous_best_params(pair, namespace, study)

        if is_study_single_objective is True:
            objective_type = "single"
        else:
            objective_type = "multi"
        logger.info(
            f"Optuna {pair} {namespace} {objective_type} objective hyperopt started"
        )
        start_time = time.time()
        try:
            study.optimize(
                objective,
                n_trials=self._optuna_config.get("n_trials"),
                n_jobs=self._optuna_config.get("n_jobs"),
                timeout=self._optuna_config.get("timeout"),
                gc_after_trial=True,
            )
        except Exception as e:
            time_spent = time.time() - start_time
            logger.error(
                f"Optuna {pair} {namespace} {objective_type} objective hyperopt failed ({time_spent:.2f} secs): {str(e)}",
                exc_info=True,
            )
            return

        time_spent = time.time() - start_time
        if is_study_single_objective:
            if not QuickAdapterRegressorV3.optuna_study_has_best_trial(study):
                logger.error(
                    f"Optuna {pair} {namespace} {objective_type} objective hyperopt failed ({time_spent:.2f} secs): no study best trial found"
                )
                return
            self.set_optuna_value(pair, namespace, study.best_value)
            self.set_optuna_params(pair, namespace, study.best_params)
            study_results = {
                "value": self.get_optuna_value(pair, namespace),
                **self.get_optuna_params(pair, namespace),
            }
            metric_log_msg = ""
        else:
            best_trial = self.get_multi_objective_study_best_trial("label", study)
            if not best_trial:
                logger.error(
                    f"Optuna {pair} {namespace} {objective_type} objective hyperopt failed ({time_spent:.2f} secs): no study best trial found"
                )
                return
            self.set_optuna_values(pair, namespace, best_trial.values)
            self.set_optuna_params(pair, namespace, best_trial.params)
            study_results = {
                "values": self.get_optuna_values(pair, namespace),
                **self.get_optuna_params(pair, namespace),
            }
            metric_log_msg = (
                f" using {self.ft_params.get('label_metric', 'seuclidean')} metric"
            )
        logger.info(
            f"Optuna {pair} {namespace} {objective_type} objective done{metric_log_msg} ({time_spent:.2f} secs)"
        )
        for key, value in study_results.items():
            logger.info(
                f"Optuna {pair} {namespace} {objective_type} objective hyperopt | {key:>20s} : {value}"
            )
        self.optuna_save_best_params(pair, namespace)

    def optuna_storage(self, pair: str) -> optuna.storages.BaseStorage:
        storage_dir = self.full_path
        storage_filename = f"optuna-{pair.split('/')[0]}"
        storage_backend = self._optuna_config.get("storage")
        if storage_backend == "sqlite":
            storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{storage_dir}/{storage_filename}.sqlite",
                heartbeat_interval=60,
                failed_trial_callback=optuna.storages.RetryFailedTrialCallback(
                    max_retry=3
                ),
            )
        elif storage_backend == "file":
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(
                    f"{storage_dir}/{storage_filename}.log"
                )
            )
        else:
            raise ValueError(
                f"Unsupported optuna storage backend: {storage_backend}. Supported backends are 'sqlite' and 'file'"
            )
        return storage

    def optuna_create_study(
        self,
        pair: str,
        namespace: str,
        direction: Optional[optuna.study.StudyDirection] = None,
        directions: Optional[list[optuna.study.StudyDirection]] = None,
    ) -> Optional[optuna.study.Study]:
        identifier = self.freqai_info.get("identifier")
        study_name = f"{identifier}-{pair}-{namespace}"
        try:
            storage = self.optuna_storage(pair)
        except Exception as e:
            logger.error(
                f"Failed to create optuna storage for study {study_name}: {str(e)}",
                exc_info=True,
            )
            return None

        continuous = self._optuna_config.get("continuous")
        if continuous:
            QuickAdapterRegressorV3.optuna_study_delete(study_name, storage)

        try:
            return optuna.create_study(
                study_name=study_name,
                sampler=optuna.samplers.TPESampler(
                    n_startup_trials=self._optuna_config.get("n_startup_trials"),
                    multivariate=True,
                    group=True,
                    seed=self._optuna_config.get("seed"),
                ),
                pruner=optuna.pruners.HyperbandPruner(min_resource=3),
                direction=direction,
                directions=directions,
                storage=storage,
                load_if_exists=not continuous,
            )
        except Exception as e:
            logger.error(
                f"Failed to create optuna study {study_name}: {str(e)}", exc_info=True
            )
            return None

    def optuna_enqueue_previous_best_params(
        self, pair: str, namespace: str, study: optuna.study.Study
    ) -> None:
        best_params = self.get_optuna_params(pair, namespace)
        if best_params:
            study.enqueue_trial(best_params)
        else:
            best_params = self.optuna_load_best_params(pair, namespace)
            if best_params:
                study.enqueue_trial(best_params)

    def optuna_save_best_params(self, pair: str, namespace: str) -> None:
        best_params_path = Path(
            self.full_path / f"optuna-{namespace}-best-params-{pair.split('/')[0]}.json"
        )
        try:
            with best_params_path.open("w", encoding="utf-8") as write_file:
                json.dump(self.get_optuna_params(pair, namespace), write_file, indent=4)
        except Exception as e:
            logger.error(
                f"Failed to save optuna {namespace} best params for {pair}: {str(e)}",
                exc_info=True,
            )
            raise

    def optuna_load_best_params(
        self, pair: str, namespace: str
    ) -> Optional[dict[str, Any]]:
        best_params_path = Path(
            self.full_path / f"optuna-{namespace}-best-params-{pair.split('/')[0]}.json"
        )
        if best_params_path.is_file():
            with best_params_path.open("r", encoding="utf-8") as read_file:
                return json.load(read_file)
        return None

    @staticmethod
    def optuna_study_delete(
        study_name: str, storage: optuna.storages.BaseStorage
    ) -> None:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except Exception:
            pass

    @staticmethod
    def optuna_study_load(
        study_name: str, storage: optuna.storages.BaseStorage
    ) -> Optional[optuna.study.Study]:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
        except Exception:
            study = None
        return study

    @staticmethod
    def optuna_study_has_best_trial(study: Optional[optuna.study.Study]) -> bool:
        if study is None:
            return False
        try:
            _ = study.best_trial
            return True
        except (ValueError, KeyError):
            return False

    @staticmethod
    def optuna_study_has_best_trials(study: Optional[optuna.study.Study]) -> bool:
        if study is None:
            return False
        try:
            _ = study.best_trials
            return True
        except (ValueError, KeyError):
            return False


regressors = {"xgboost", "lightgbm"}


def get_callbacks(trial: optuna.trial.Trial, regressor: str) -> list[Callable]:
    if regressor == "xgboost":
        callbacks = [
            optuna.integration.XGBoostPruningCallback(trial, "validation_0-rmse")
        ]
    elif regressor == "lightgbm":
        callbacks = [optuna.integration.LightGBMPruningCallback(trial, "rmse")]
    else:
        raise ValueError(
            f"Unsupported regressor model: {regressor} (supported: {', '.join(regressors)})"
        )
    return callbacks


def fit_regressor(
    regressor: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_weights: np.ndarray,
    eval_set: Optional[list[tuple[pd.DataFrame, pd.DataFrame]]],
    eval_weights: Optional[list[np.ndarray]],
    model_training_parameters: dict[str, Any],
    init_model: Any = None,
    callbacks: Optional[list[Callable]] = None,
) -> Any:
    if regressor == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            callbacks=callbacks,
            **model_training_parameters,
        )
        model.fit(
            X=X,
            y=y,
            sample_weight=train_weights,
            eval_set=eval_set,
            sample_weight_eval_set=eval_weights,
            xgb_model=init_model,
        )
    elif regressor == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(objective="regression", **model_training_parameters)
        model.fit(
            X=X,
            y=y,
            sample_weight=train_weights,
            eval_set=eval_set,
            eval_sample_weight=eval_weights,
            eval_metric="rmse",
            init_model=init_model,
            callbacks=callbacks,
        )
    else:
        raise ValueError(
            f"Unsupported regressor model: {regressor} (supported: {', '.join(regressors)})"
        )
    return model


@lru_cache(maxsize=128)
def calculate_min_extrema(
    size: int, fit_live_predictions_candles: int, min_extrema: int = 4
) -> int:
    return int(round(size / fit_live_predictions_candles) * min_extrema)


def train_objective(
    trial: optuna.trial.Trial,
    regressor: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_weights: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    test_weights: np.ndarray,
    test_size: float,
    fit_live_predictions_candles: int,
    candles_step: int,
    model_training_parameters: dict[str, Any],
) -> float:
    test_ok = True
    test_length = len(X_test)
    if debug:
        test_extrema = y_test.get(EXTREMA_COLUMN)
        n_test_minima: int = sp.signal.find_peaks(-test_extrema)[0].size
        n_test_maxima: int = sp.signal.find_peaks(test_extrema)[0].size
        n_test_extrema: int = n_test_minima + n_test_maxima
        min_test_extrema: int = calculate_min_extrema(
            test_length, fit_live_predictions_candles
        )
        logger.info(
            f"{test_length=}, {n_test_minima=}, {n_test_maxima=}, {n_test_extrema=}, {min_test_extrema=}"
        )
    min_test_window: int = fit_live_predictions_candles * 2
    if test_length < min_test_window:
        logger.warning(f"Insufficient test data: {test_length} < {min_test_window}")
        test_ok = False
    max_test_window: int = test_length
    test_window: int = trial.suggest_int(
        "test_period_candles", min_test_window, max_test_window, step=candles_step
    )
    X_test = X_test.iloc[-test_window:]
    y_test = y_test.iloc[-test_window:]
    test_extrema = y_test.get(EXTREMA_COLUMN)
    n_test_minima: int = sp.signal.find_peaks(-test_extrema)[0].size
    n_test_maxima: int = sp.signal.find_peaks(test_extrema)[0].size
    n_test_extrema: int = n_test_minima + n_test_maxima
    min_test_extrema: int = calculate_min_extrema(
        test_window, fit_live_predictions_candles
    )
    if n_test_extrema < min_test_extrema:
        if debug:
            logger.warning(
                f"Insufficient extrema in test data with {test_window=}: {n_test_extrema=} < {min_test_extrema=}"
            )
        test_ok = False
    test_weights = test_weights[-test_window:]

    train_ok = True
    train_length = len(X)
    if debug:
        train_extrema = y.get(EXTREMA_COLUMN)
        n_train_minima: int = sp.signal.find_peaks(-train_extrema)[0].size
        n_train_maxima: int = sp.signal.find_peaks(train_extrema)[0].size
        n_train_extrema: int = n_train_minima + n_train_maxima
        min_train_extrema: int = calculate_min_extrema(
            train_length, fit_live_predictions_candles
        )
        logger.info(
            f"{train_length=}, {n_train_minima=}, {n_train_maxima=}, {n_train_extrema=}, {min_train_extrema=}"
        )
    min_train_window: int = min_test_window * int(round(1 / test_size - 1))
    if train_length < min_train_window:
        logger.warning(f"Insufficient train data: {train_length} < {min_train_window}")
        train_ok = False
    max_train_window: int = train_length
    train_window: int = trial.suggest_int(
        "train_period_candles", min_train_window, max_train_window, step=candles_step
    )
    X = X.iloc[-train_window:]
    y = y.iloc[-train_window:]
    train_extrema = y.get(EXTREMA_COLUMN)
    n_train_minima: int = sp.signal.find_peaks(-train_extrema)[0].size
    n_train_maxima: int = sp.signal.find_peaks(train_extrema)[0].size
    n_train_extrema: int = n_train_minima + n_train_maxima
    min_train_extrema: int = calculate_min_extrema(
        train_window, fit_live_predictions_candles
    )
    if n_train_extrema < min_train_extrema:
        if debug:
            logger.warning(
                f"Insufficient extrema in train data with {train_window=}: {n_train_extrema=} < {min_train_extrema=}"
            )
        train_ok = False
    train_weights = train_weights[-train_window:]

    if not test_ok or not train_ok:
        return np.inf

    model = fit_regressor(
        regressor=regressor,
        X=X,
        y=y,
        train_weights=train_weights,
        eval_set=[(X_test, y_test)],
        eval_weights=[test_weights],
        model_training_parameters=model_training_parameters,
        callbacks=get_callbacks(trial, regressor),
    )
    y_pred = model.predict(X_test)

    return sklearn.metrics.root_mean_squared_error(
        y_test, y_pred, sample_weight=test_weights
    )


def get_optuna_study_model_parameters(
    trial: optuna.trial.Trial,
    regressor: str,
    model_training_best_parameters: dict[str, Any],
    expansion_factor: float,
) -> dict[str, Any]:
    if regressor not in regressors:
        raise ValueError(
            f"Unsupported regressor model: {regressor} (supported: {', '.join(regressors)})"
        )
    default_ranges = {
        "learning_rate": (1e-3, 0.5),
        "min_child_weight": (1e-8, 100.0),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha": (1e-8, 100.0),
        "reg_lambda": (1e-8, 100.0),
        "max_depth": (3, 13),
        "gamma": (1e-8, 10.0),
        "num_leaves": (8, 256),
        "min_split_gain": (1e-8, 10.0),
        "min_child_samples": (10, 100),
    }

    log_scaled_params = {
        "learning_rate",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "gamma",
        "min_split_gain",
    }

    ranges = copy.deepcopy(default_ranges)
    if model_training_best_parameters:
        for param, (default_min, default_max) in default_ranges.items():
            center_value = model_training_best_parameters.get(param)

            if (
                center_value is None
                or not isinstance(center_value, (int, float))
                or not np.isfinite(center_value)
            ):
                continue

            if param in log_scaled_params:
                new_min = center_value / (1 + expansion_factor)
                new_max = center_value * (1 + expansion_factor)
            else:
                margin = (default_max - default_min) * expansion_factor / 2
                new_min = center_value - margin
                new_max = center_value + margin

            param_min = max(default_min, new_min)
            param_max = min(default_max, new_max)

            if param_min < param_max:
                ranges[param] = (param_min, param_max)

    study_model_parameters = {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            ranges["learning_rate"][0],
            ranges["learning_rate"][1],
            log=True,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            ranges["min_child_weight"][0],
            ranges["min_child_weight"][1],
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample", ranges["subsample"][0], ranges["subsample"][1]
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            ranges["colsample_bytree"][0],
            ranges["colsample_bytree"][1],
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", ranges["reg_alpha"][0], ranges["reg_alpha"][1], log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", ranges["reg_lambda"][0], ranges["reg_lambda"][1], log=True
        ),
    }
    if regressor == "xgboost":
        study_model_parameters.update(
            {
                "max_depth": trial.suggest_int(
                    "max_depth",
                    int(ranges["max_depth"][0]),
                    int(ranges["max_depth"][1]),
                ),
                "gamma": trial.suggest_float(
                    "gamma", ranges["gamma"][0], ranges["gamma"][1], log=True
                ),
            }
        )
    elif regressor == "lightgbm":
        study_model_parameters.update(
            {
                "num_leaves": trial.suggest_int(
                    "num_leaves",
                    int(ranges["num_leaves"][0]),
                    int(ranges["num_leaves"][1]),
                ),
                "min_split_gain": trial.suggest_float(
                    "min_split_gain",
                    ranges["min_split_gain"][0],
                    ranges["min_split_gain"][1],
                    log=True,
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples",
                    int(ranges["min_child_samples"][0]),
                    int(ranges["min_child_samples"][1]),
                ),
            }
        )
    return study_model_parameters


def hp_objective(
    trial: optuna.trial.Trial,
    regressor: str,
    X: pd.DataFrame,
    y: pd.DataFrame,
    train_weights: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    test_weights: np.ndarray,
    model_training_best_parameters: dict[str, Any],
    model_training_parameters: dict[str, Any],
    expansion_factor: float,
) -> float:
    study_model_parameters = get_optuna_study_model_parameters(
        trial, regressor, model_training_best_parameters, expansion_factor
    )
    model_training_parameters = {**model_training_parameters, **study_model_parameters}

    model = fit_regressor(
        regressor=regressor,
        X=X,
        y=y,
        train_weights=train_weights,
        eval_set=[(X_test, y_test)],
        eval_weights=[test_weights],
        model_training_parameters=model_training_parameters,
        callbacks=get_callbacks(trial, regressor),
    )
    y_pred = model.predict(X_test)

    return sklearn.metrics.root_mean_squared_error(
        y_test, y_pred, sample_weight=test_weights
    )


def calculate_quantile(values: np.ndarray, value: float) -> float:
    if values.size == 0:
        return np.nan

    first_value = values[0]
    if np.all(np.isclose(values, first_value)):
        return (
            0.5
            if np.isclose(value, first_value)
            else (0.0 if value < first_value else 1.0)
        )

    return np.sum(values <= value) / values.size


class TrendDirection(IntEnum):
    NEUTRAL = 0
    UP = 1
    DOWN = -1


def zigzag(
    df: pd.DataFrame,
    natr_period: int = 14,
    natr_ratio: float = 6.0,
) -> tuple[list[int], list[float], list[TrendDirection], list[float]]:
    n = len(df)
    if df.empty or n < natr_period:
        return [], [], [], []

    natr_values = (ta.NATR(df, timeperiod=natr_period).bfill() / 100.0).to_numpy()

    indices: list[int] = df.index.tolist()
    thresholds: np.ndarray = natr_values * natr_ratio
    closes = df.get("close").to_numpy()
    highs = df.get("high").to_numpy()
    lows = df.get("low").to_numpy()

    state: TrendDirection = TrendDirection.NEUTRAL

    pivots_indices: list[int] = []
    pivots_values: list[float] = []
    pivots_directions: list[TrendDirection] = []
    pivots_thresholds: list[float] = []
    last_pivot_pos: int = -1

    candidate_pivot_pos: int = -1
    candidate_pivot_value: float = np.nan

    volatility_quantile_cache: dict[int, float] = {}

    def calculate_volatility_quantile(pos: int) -> float:
        if pos not in volatility_quantile_cache:
            start = max(0, pos + 1 - natr_period)
            end = min(pos + 1, n)
            if start >= end:
                volatility_quantile_cache[pos] = np.nan
            else:
                volatility_quantile_cache[pos] = calculate_quantile(
                    natr_values[start:end], natr_values[pos]
                )

        return volatility_quantile_cache[pos]

    def calculate_slopes_ok_threshold(
        pos: int,
        min_threshold: float = 0.65,
        max_threshold: float = 0.85,
    ) -> float:
        volatility_quantile = calculate_volatility_quantile(pos)
        if np.isnan(volatility_quantile):
            return median([min_threshold, max_threshold])

        return min_threshold + (max_threshold - min_threshold) * volatility_quantile

    def update_candidate_pivot(pos: int, value: float):
        nonlocal candidate_pivot_pos, candidate_pivot_value
        if 0 <= pos < n:
            candidate_pivot_pos = pos
            candidate_pivot_value = value

    def reset_candidate_pivot():
        nonlocal candidate_pivot_pos, candidate_pivot_value
        candidate_pivot_pos = -1
        candidate_pivot_value = np.nan

    def add_pivot(pos: int, value: float, direction: TrendDirection):
        nonlocal last_pivot_pos
        if pivots_indices and indices[pos] == pivots_indices[-1]:
            return
        pivots_indices.append(indices[pos])
        pivots_values.append(value)
        pivots_directions.append(direction)
        pivots_thresholds.append(thresholds[pos])
        last_pivot_pos = pos
        reset_candidate_pivot()

    slope_ok_cache: dict[tuple[int, int, TrendDirection, float], bool] = {}

    def get_slope_ok(
        pos: int,
        candidate_pivot_pos: int,
        direction: TrendDirection,
        min_slope: float,
    ) -> bool:
        cache_key = (
            pos,
            candidate_pivot_pos,
            direction,
            min_slope,
        )

        if cache_key in slope_ok_cache:
            return slope_ok_cache[cache_key]

        if pos <= candidate_pivot_pos:
            slope_ok_cache[cache_key] = False
            return slope_ok_cache[cache_key]

        log_candidate_pivot_close = np.log(closes[candidate_pivot_pos])
        log_current_close = np.log(closes[pos])

        log_slope_close = (log_current_close - log_candidate_pivot_close) / (
            pos - candidate_pivot_pos
        )

        if direction == TrendDirection.UP:
            slope_ok_cache[cache_key] = log_slope_close > min_slope
        elif direction == TrendDirection.DOWN:
            slope_ok_cache[cache_key] = log_slope_close < -min_slope
        else:
            slope_ok_cache[cache_key] = False

        return slope_ok_cache[cache_key]

    def is_pivot_confirmed(
        pos: int,
        candidate_pivot_pos: int,
        direction: TrendDirection,
        min_slope: float = np.finfo(float).eps,
        alpha: float = 0.05,
    ) -> bool:
        start_pos = min(candidate_pivot_pos + 1, n)
        end_pos = min(pos + 1, n)
        n_slopes = max(0, end_pos - start_pos)

        if n_slopes < 1:
            return False

        slopes_ok: list[bool] = []
        for i in range(start_pos, end_pos):
            slopes_ok.append(
                get_slope_ok(
                    pos=i,
                    candidate_pivot_pos=candidate_pivot_pos,
                    direction=direction,
                    min_slope=min_slope,
                )
            )

        slopes_ok_threshold = calculate_slopes_ok_threshold(candidate_pivot_pos)
        n_slopes_ok = sum(slopes_ok)
        binomtest = sp.stats.binomtest(
            k=n_slopes_ok, n=n_slopes, p=0.5, alternative="greater"
        )

        return (
            binomtest.pvalue <= alpha
            and (n_slopes_ok / n_slopes) >= slopes_ok_threshold
        )

    start_pos = 0
    initial_high_pos = start_pos
    initial_low_pos = start_pos
    initial_high = highs[initial_high_pos]
    initial_low = lows[initial_low_pos]
    for i in range(start_pos + 1, n):
        current_high = highs[i]
        current_low = lows[i]
        if current_high > initial_high:
            initial_high, initial_high_pos = current_high, i
        if current_low < initial_low:
            initial_low, initial_low_pos = current_low, i

        initial_move_from_high = (initial_high - current_low) / initial_high
        initial_move_from_low = (current_high - initial_low) / initial_low
        is_initial_high_move_significant = (
            initial_move_from_high >= thresholds[initial_high_pos]
        )
        is_initial_low_move_significant = (
            initial_move_from_low >= thresholds[initial_low_pos]
        )
        if is_initial_high_move_significant and is_initial_low_move_significant:
            if initial_move_from_high > initial_move_from_low:
                add_pivot(initial_high_pos, initial_high, TrendDirection.UP)
                state = TrendDirection.DOWN
                break
            else:
                add_pivot(initial_low_pos, initial_low, TrendDirection.DOWN)
                state = TrendDirection.UP
                break
        else:
            if is_initial_high_move_significant:
                add_pivot(initial_high_pos, initial_high, TrendDirection.UP)
                state = TrendDirection.DOWN
                break
            elif is_initial_low_move_significant:
                add_pivot(initial_low_pos, initial_low, TrendDirection.DOWN)
                state = TrendDirection.UP
                break
    else:
        return [], [], [], []

    for i in range(last_pivot_pos + 1, n):
        current_high = highs[i]
        current_low = lows[i]

        if state == TrendDirection.UP:
            if np.isnan(candidate_pivot_value) or current_high > candidate_pivot_value:
                update_candidate_pivot(i, current_high)
            if (
                candidate_pivot_value - current_low
            ) / candidate_pivot_value >= thresholds[
                candidate_pivot_pos
            ] and is_pivot_confirmed(i, candidate_pivot_pos, TrendDirection.DOWN):
                add_pivot(candidate_pivot_pos, candidate_pivot_value, TrendDirection.UP)
                state = TrendDirection.DOWN

        elif state == TrendDirection.DOWN:
            if np.isnan(candidate_pivot_value) or current_low < candidate_pivot_value:
                update_candidate_pivot(i, current_low)
            if (
                current_high - candidate_pivot_value
            ) / candidate_pivot_value >= thresholds[
                candidate_pivot_pos
            ] and is_pivot_confirmed(i, candidate_pivot_pos, TrendDirection.UP):
                add_pivot(
                    candidate_pivot_pos, candidate_pivot_value, TrendDirection.DOWN
                )
                state = TrendDirection.UP

    return pivots_indices, pivots_values, pivots_directions, pivots_thresholds


@lru_cache(maxsize=8)
def largest_divisor(integer: int, step: int) -> Optional[int]:
    if not isinstance(integer, int) or integer <= 0:
        raise ValueError("integer must be a positive integer")
    if not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")

    q_start = math.floor(0.5 * step) + 1
    q_end = math.ceil(1.5 * step) - 1

    if q_start > q_end:
        return None

    for q in range(q_start, q_end + 1):
        if integer % q == 0:
            return int(integer / q)

    return None


def label_objective(
    trial: optuna.trial.Trial,
    df: pd.DataFrame,
    fit_live_predictions_candles: int,
    candles_step: int,
) -> tuple[float, int]:
    fit_live_predictions_candles_largest_divisor = largest_divisor(
        fit_live_predictions_candles, candles_step
    )
    if fit_live_predictions_candles_largest_divisor is None:
        raise ValueError(
            f"Could not find a suitable largest divisor for {fit_live_predictions_candles} with step {candles_step}, please change your fit_live_predictions_candles or candles_step parameters"
        )
    min_label_period_candles: int = round_to_nearest_int(
        max(
            fit_live_predictions_candles
            // fit_live_predictions_candles_largest_divisor,
            candles_step,
        ),
        candles_step,
    )
    max_label_period_candles: int = round_to_nearest_int(
        max(fit_live_predictions_candles // 4, min_label_period_candles),
        candles_step,
    )
    label_period_candles = trial.suggest_int(
        "label_period_candles",
        min_label_period_candles,
        max_label_period_candles,
        step=candles_step,
    )
    label_natr_ratio = trial.suggest_float("label_natr_ratio", 2.0, 38.0, step=0.01)

    df = df.iloc[
        -(
            max(2, int(fit_live_predictions_candles / label_period_candles))
            * label_period_candles
        ) :
    ]

    if df.empty:
        return -np.inf, -np.inf

    _, pivots_values, _, pivots_thresholds = zigzag(
        df,
        natr_period=label_period_candles,
        natr_ratio=label_natr_ratio,
    )

    return np.median(pivots_thresholds), len(pivots_values)


def smoothed_max(series: pd.Series, temperature=1.0) -> float:
    data_array = series.to_numpy()
    if data_array.size == 0:
        return np.nan
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if np.isclose(temperature, 0):
        return data_array.max()
    return sp.special.logsumexp(temperature * data_array) / temperature


def smoothed_min(series: pd.Series, temperature=1.0) -> float:
    data_array = series.to_numpy()
    if data_array.size == 0:
        return np.nan
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if np.isclose(temperature, 0):
        return data_array.min()
    return -sp.special.logsumexp(-temperature * data_array) / temperature


def boltzmann_operator(series: pd.Series, alpha: float) -> float:
    """
    Compute the Boltzmann operator of a series with parameter alpha.
    """
    data_array = series.to_numpy()
    if data_array.size == 0:
        return np.nan
    if alpha == 0:
        return np.mean(data_array)
    scaled_data = alpha * data_array
    shifted_exponentials = np.exp(scaled_data - np.max(scaled_data))
    numerator = np.sum(data_array * shifted_exponentials)
    denominator = np.sum(shifted_exponentials)
    return numerator / denominator


def round_to_nearest_int(value: float, step: int) -> int:
    """
    Round a value to the nearest multiple of a given step.
    :param value: The value to round.
    :param step: The step size to round to (must be non-zero).
    :return: The rounded value.
    :raises ValueError: If step is zero.
    """
    if not isinstance(step, int) or step <= 0:
        raise ValueError("step must be a positive integer")
    return int(round(value / step) * step)
