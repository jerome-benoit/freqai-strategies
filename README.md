# FreqAI strategies

## Table of contents

- [QuickAdapter](#quickadapter)
  - [Quick start](#quick-start)
  - [Configuration tunables](#configuration-tunables)
- [ReforceXY](#reforcexy)
  - [Quick start](#quick-start-1)
  - [Supported models](#supported-models)
  - [Configuration tunables](#configuration-tunables-1)
- [Common workflows](#common-workflows)
- [Note](#note)

## QuickAdapter

### Quick start

Change the timezone according to your location in [`docker-compose.yml`](./quickadapter/docker-compose.yml).

From the repository root, configure, build and start the QuickAdapter container:

```shell
cd quickadapter
cp user_data/config-template.json user_data/config.json
```

Adapt the configuration to your needs: edit `user_data/config.json` to set your exchange API keys and tune the `freqai` section.

Then build and start the container:

```shell
docker compose up -d --build
```

### Configuration tunables

| Path                                                 | Default           | Type / Range                                                                                                                     | Description                                                                                                                                                                                                                                                                         |
| ---------------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _Protections_                                        |                   |                                                                                                                                  |                                                                                                                                                                                                                                                                                     |
| estimated_trade_duration_candles                     | 48                | int >= 1                                                                                                                         | Heuristic for StoplossGuard tuning.                                                                                                                                                                                                                                                 |
| _Leverage_                                           |                   |                                                                                                                                  |                                                                                                                                                                                                                                                                                     |
| leverage                                             | proposed_leverage | float [1.0, max_leverage]                                                                                                        | Leverage. Fallback to proposed_leverage for the pair.                                                                                                                                                                                                                               |
| _Exit pricing_                                       |                   |                                                                                                                                  |                                                                                                                                                                                                                                                                                     |
| exit_pricing.trade_price_target                      | `moving_average`  | enum {`moving_average`,`interpolation`,`weighted_interpolation`}                                                                 | Trade NATR computation method.                                                                                                                                                                                                                                                      |
| exit_pricing.thresholds_calibration.decline_quantile | 0.90              | float (0,1)                                                                                                                      | PnL decline quantile threshold.                                                                                                                                                                                                                                                     |
| _Reversal confirmation_                              |                   |                                                                                                                                  |                                                                                                                                                                                                                                                                                     |
| reversal_confirmation.lookback_period                | 0                 | int >= 0                                                                                                                         | Prior confirming candles; 0 = none.                                                                                                                                                                                                                                                 |
| reversal_confirmation.decay_ratio                    | 0.5               | float (0,1]                                                                                                                      | Geometric per-candle relaxation factor.                                                                                                                                                                                                                                             |
| reversal_confirmation.min_natr_ratio_percent         | 0.01              | float [0,1]                                                                                                                      | Lower bound fraction for volatility adjusted reversal threshold.                                                                                                                                                                                                                    |
| reversal_confirmation.max_natr_ratio_percent         | 0.1               | float [0,1]                                                                                                                      | Upper bound fraction (>= lower bound) for volatility adjusted reversal threshold.                                                                                                                                                                                                   |
| _Regressor model_                                    |                   |                                                                                                                                  |                                                                                                                                                                                                                                                                                     |
| freqai.regressor                                     | `xgboost`         | enum {`xgboost`,`lightgbm`}                                                                                                      | Machine learning regressor algorithm.                                                                                                                                                                                                                                               |
| _Extrema smoothing_                                  |                   |                                                                                                                                  |                                                                                                                                                                                                                                                                                     |
| freqai.extrema_smoothing.method                      | `gaussian`        | enum {`gaussian`,`kaiser`,`triang`,`smm`,`sma`}                                                                                  | Extrema smoothing kernel (smm=simple moving median, sma=simple moving average).                                                                                                                                                                                                     |
| freqai.extrema_smoothing.window                      | 5                 | int >= 3                                                                                                                         | Window size for extrema smoothing.                                                                                                                                                                                                                                                  |
| freqai.extrema_smoothing.beta                        | 8.0               | float > 0                                                                                                                        | Kaiser kernel shape parameter.                                                                                                                                                                                                                                                      |
| _Extrema weighting_                                  |                   |                                                                                                                                  |                                                                                                                                                                                                                                                                                     |
| freqai.extrema_weighting.strategy                    | `none`            | enum {`none`,`amplitude`,`amplitude_excess`}                                                                                     | Extrema weighting source: unweighted (`none`), swing amplitude (`amplitude`), or volatility-adjusted swing amplitude (`amplitude_excess`).                                                                                                                                          |
| freqai.extrema_weighting.normalization               | `minmax`          | enum {`minmax`,`zscore`,`l1`,`l2`,`robust`,`softmax`,`tanh`,`rank`,`none`}                                                       | Normalization method for weights.                                                                                                                                                                                                                                                   |
| freqai.extrema_weighting.gamma                       | 1.0               | float (0,10]                                                                                                                     | Contrast exponent applied after normalization (>1 emphasizes extrema, 0<gamma<1 softens).                                                                                                                                                                                           |
| freqai.extrema_weighting.softmax_temperature         | 1.0               | float > 0                                                                                                                        | Temperature parameter for softmax normalization (lower values sharpen distribution, higher values flatten it).                                                                                                                                                                      |
| freqai.extrema_weighting.tanh_scale                  | 1.0               | float > 0                                                                                                                        | Scale parameter for tanh normalization.                                                                                                                                                                                                                                             |
| freqai.extrema_weighting.tanh_gain                   | 1.0               | float > 0                                                                                                                        | Gain parameter for tanh normalization.                                                                                                                                                                                                                                              |
| freqai.extrema_weighting.robust_quantiles            | [0.25, 0.75]      | list[float] where 0 <= q_low < q_high <= 1                                                                                       | Quantile range for robust normalization.                                                                                                                                                                                                                                            |
| freqai.extrema_weighting.rank_method                 | `average`         | enum {`average`,`min`,`max`,`dense`,`ordinal`}                                                                                   | Ranking method for rank normalization.                                                                                                                                                                                                                                              |
| _Feature parameters_                                 |                   |                                                                                                                                  |                                                                                                                                                                                                                                                                                     |
| freqai.feature_parameters.label_period_candles       | min/max midpoint  | int >= 1                                                                                                                         | Zigzag labeling NATR horizon.                                                                                                                                                                                                                                                       |
| freqai.feature_parameters.min_label_period_candles   | 12                | int >= 1                                                                                                                         | Minimum labeling NATR horizon used for reversals labeling HPO.                                                                                                                                                                                                                      |
| freqai.feature_parameters.max_label_period_candles   | 24                | int >= 1                                                                                                                         | Maximum labeling NATR horizon used for reversals labeling HPO.                                                                                                                                                                                                                      |
| freqai.feature_parameters.label_natr_ratio           | min/max midpoint  | float > 0                                                                                                                        | Zigzag labeling NATR ratio.                                                                                                                                                                                                                                                         |
| freqai.feature_parameters.min_label_natr_ratio       | 9.0               | float > 0                                                                                                                        | Minimum labeling NATR ratio used for reversals labeling HPO.                                                                                                                                                                                                                        |
| freqai.feature_parameters.max_label_natr_ratio       | 12.0              | float > 0                                                                                                                        | Maximum labeling NATR ratio used for reversals labeling HPO.                                                                                                                                                                                                                        |
| freqai.feature_parameters.label_frequency_candles    | `auto`            | int >= 2 \| `auto`                                                                                                               | Reversals labeling frequency. `auto` = max(2, 2 \* number of whitelisted pairs).                                                                                                                                                                                                    |
| freqai.feature_parameters.label_metric               | `euclidean`       | string (supported: `euclidean`,`minkowski`,`cityblock`,`chebyshev`,`mahalanobis`,`seuclidean`,`jensenshannon`,`sqeuclidean`,...) | Metric used in distance calculations to ideal point.                                                                                                                                                                                                                                |
| freqai.feature_parameters.label_weights              | [1/3,1/3,1/3]     | list[float]                                                                                                                      | Per-objective weights used in distance calculations to ideal point. First objective is the number of detected reversals. Second objective is the median swing amplitude of Zigzag reversals (reversals quality). Third objective is the median volatility-adjusted swing amplitude. |
| freqai.feature_parameters.label_p_order              | `None`            | float                                                                                                                            | p-order used by Minkowski / power-mean calculations (optional).                                                                                                                                                                                                                     |
| freqai.feature_parameters.label_medoid_metric        | `euclidean`       | string                                                                                                                           | Metric used with `medoid`.                                                                                                                                                                                                                                                          |
| freqai.feature_parameters.label_kmeans_metric        | `euclidean`       | string                                                                                                                           | Metric used for k-means clustering.                                                                                                                                                                                                                                                 |
| freqai.feature_parameters.label_kmeans_selection     | `min`             | enum {`min`,`medoid`}                                                                                                            | Strategy to select trial in the best kmeans cluster.                                                                                                                                                                                                                                |
| freqai.feature_parameters.label_kmedoids_metric      | `euclidean`       | string                                                                                                                           | Metric used for k-medoids clustering.                                                                                                                                                                                                                                               |
| freqai.feature_parameters.label_kmedoids_selection   | `min`             | enum {`min`,`medoid`}                                                                                                            | Strategy to select trial in the best k-medoids cluster.                                                                                                                                                                                                                             |
| freqai.feature_parameters.label_knn_metric           | `minkowski`       | string                                                                                                                           | Distance metric for KNN.                                                                                                                                                                                                                                                            |
| freqai.feature_parameters.label_knn_p_order          | `None`            | float                                                                                                                            | p-order for KNN Minkowski metric distance. (optional)                                                                                                                                                                                                                               |
| freqai.feature_parameters.label_knn_n_neighbors      | 5                 | int >= 1                                                                                                                         | Number of neighbors for KNN.                                                                                                                                                                                                                                                        |
| _Predictions extrema_                                |                   |                                                                                                                                  |                                                                                                                                                                                                                                                                                     |
| freqai.predictions_extrema.selection_method          | `rank`            | enum {`rank`,`values`,`partition`}                                                                                               | Extrema selection method. `values` uses reversal values, `rank` uses ranked extrema values, `partition` uses sign-based partitioning.                                                                                                                                               |
| freqai.predictions_extrema.thresholds_smoothing      | `mean`            | enum {`mean`,`isodata`,`li`,`minimum`,`otsu`,`triangle`,`yen`,`median`,`soft_extremum`}                                          | Thresholding method for prediction thresholds smoothing.                                                                                                                                                                                                                            |
| freqai.predictions_extrema.thresholds_alpha          | 12.0              | float > 0                                                                                                                        | Alpha for `soft_extremum`.                                                                                                                                                                                                                                                          |
| freqai.predictions_extrema.threshold_outlier         | 0.999             | float (0,1)                                                                                                                      | Quantile threshold for predictions outlier filtering.                                                                                                                                                                                                                               |
| _Optuna / HPO_                                       |                   |                                                                                                                                  |                                                                                                                                                                                                                                                                                     |
| freqai.optuna_hyperopt.enabled                       | true              | bool                                                                                                                             | Enables HPO.                                                                                                                                                                                                                                                                        |
| freqai.optuna_hyperopt.sampler                       | `tpe`             | enum {`tpe`,`auto`}                                                                                                              | HPO sampler algorithm. `tpe` uses TPESampler with multivariate and group, `auto` uses AutoSampler.                                                                                                                                                                                  |
| freqai.optuna_hyperopt.storage                       | `file`            | enum {`file`,`sqlite`}                                                                                                           | HPO storage backend.                                                                                                                                                                                                                                                                |
| freqai.optuna_hyperopt.continuous                    | true              | bool                                                                                                                             | Continuous HPO.                                                                                                                                                                                                                                                                     |
| freqai.optuna_hyperopt.warm_start                    | true              | bool                                                                                                                             | Warm start HPO with previous best value(s).                                                                                                                                                                                                                                         |
| freqai.optuna_hyperopt.n_startup_trials              | 15                | int >= 0                                                                                                                         | HPO startup trials.                                                                                                                                                                                                                                                                 |
| freqai.optuna_hyperopt.n_trials                      | 50                | int >= 1                                                                                                                         | Maximum HPO trials.                                                                                                                                                                                                                                                                 |
| freqai.optuna_hyperopt.n_jobs                        | CPU threads / 4   | int >= 1                                                                                                                         | Parallel HPO workers.                                                                                                                                                                                                                                                               |
| freqai.optuna_hyperopt.timeout                       | 7200              | int >= 0                                                                                                                         | HPO wall-clock timeout in seconds.                                                                                                                                                                                                                                                  |
| freqai.optuna_hyperopt.label_candles_step            | 1                 | int >= 1                                                                                                                         | Step for Zigzag NATR horizon search space.                                                                                                                                                                                                                                          |
| freqai.optuna_hyperopt.train_candles_step            | 10                | int >= 1                                                                                                                         | Step for training sets size search space.                                                                                                                                                                                                                                           |
| freqai.optuna_hyperopt.space_reduction               | false             | bool                                                                                                                             | Enable/disable HPO search space reduction based on previous best parameters.                                                                                                                                                                                                        |
| freqai.optuna_hyperopt.expansion_ratio               | 0.4               | float [0,1]                                                                                                                      | HPO search space expansion ratio.                                                                                                                                                                                                                                                   |
| freqai.optuna_hyperopt.min_resource                  | 3                 | int >= 1                                                                                                                         | Minimum resource per Hyperband pruner rung.                                                                                                                                                                                                                                         |
| freqai.optuna_hyperopt.seed                          | 1                 | int >= 0                                                                                                                         | HPO RNG seed.                                                                                                                                                                                                                                                                       |

## ReforceXY

### Quick start

Change the timezone according to your location in [`docker-compose.yml`](./ReforceXY/docker-compose.yml).

From the repository root, configure, build and start the ReforceXY container:

```shell
cd ReforceXY
cp user_data/config-template.json user_data/config.json
```

Adapt the configuration to your needs: edit `user_data/config.json` to set your exchange API keys and tune the `freqai` section.

Then build and start the container:

```shell
docker compose up -d --build
```

### Supported models

PPO, MaskablePPO, RecurrentPPO, DQN, QRDQN

### Configuration tunables

The documented list of model tunables is at the top of the [ReforceXY.py](./ReforceXY/user_data/freqaimodels/ReforceXY.py) file.

The rewarding logic and tunables are documented in the [reward space analysis](./ReforceXY/reward_space_analysis/README.md).

## Common workflows

**List running compose services and the containers they created:**

```shell
docker compose ps
```

**Enter a running service:**

```shell
# use the compose service name (e.g. "freqtrade")
docker compose exec freqtrade /bin/sh
```

**View logs:**

```shell
# service logs (compose maps service -> container(s))
docker compose logs -f freqtrade

# or follow a specific container's logs
docker logs -f freqtrade-quickadapter
```

**Stop and remove the compose stack:**

```shell
docker compose down
```

**Automatically update docker images:**

```shell
cd ReforceXY  # or quickadapter
cp ../scripts/docker-upgrade.sh .
./docker-upgrade.sh
```

The script checks for new Freqtrade image versions on Docker Hub, rebuilds and restarts containers if updates are found, sends Telegram notifications (if configured), and cleans up unused images.

_Configuration and environment variables:_

| Variable            | Default                                  | Description                          |
| ------------------- | ---------------------------------------- | ------------------------------------ |
| FREQTRADE_CONFIG    | `./user_data/config.json`                | Freqtrade configuration file path    |
| LOCAL_DOCKER_IMAGE  | `reforcexy-freqtrade`                    | Local image name                     |
| REMOTE_DOCKER_IMAGE | `freqtradeorg/freqtrade:stable_freqairl` | Freqtrade image to track for updates |

_Cronjob setup (daily check at 3:00 AM):_

```cron
0 3 * * * cd /path/to/freqai-strategies/ReforceXY && ./docker-upgrade.sh >> user_data/logs/docker-upgrade.log 2>&1
```

---

## Note

> Do not expect any support of any kind on the Internet. Nevertheless, PRs implementing documentation, bug fixes, cleanups or sensible features will be discussed and might get merged.
