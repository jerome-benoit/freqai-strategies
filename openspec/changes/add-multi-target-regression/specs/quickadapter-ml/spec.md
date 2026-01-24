## ADDED Requirements

### Requirement: Multi-Target Configuration

The system SHALL support a `prediction_targets` configuration key within the
`freqai` section that specifies which additional prediction targets to enable.

The primary target `extrema` SHALL always be enabled implicitly and cannot be
disabled. The `prediction_targets` list contains only additional targets.

The configuration SHALL accept target identifiers from the set: `amplitude`,
`time_to_pivot`, `efficiency`, `natr`.

The system SHALL validate that all specified targets are from the known set and
raise a configuration error for unknown target identifiers.

#### Scenario: Valid multi-target configuration

- **GIVEN** a FreqAI configuration with `prediction_targets` set to
  `["amplitude", "time_to_pivot"]`
- **WHEN** the model is initialized
- **THEN** the model SHALL enable `extrema` (implicit) plus the two specified targets
- **AND** the effective target list SHALL be `["extrema", "amplitude", "time_to_pivot"]`

#### Scenario: Invalid target identifier

- **GIVEN** a FreqAI configuration with `prediction_targets` containing
  `"unknown_target"`
- **WHEN** the model is initialized
- **THEN** the system SHALL raise a `ValueError` listing valid prediction target identifiers

#### Scenario: Default single-target behavior

- **GIVEN** a FreqAI configuration without a `prediction_targets` key
- **WHEN** the model is initialized
- **THEN** the system SHALL default to single-target mode with only `extrema`
- **AND** maintain full backward compatibility with existing behavior

#### Scenario: Empty prediction_targets

- **GIVEN** a FreqAI configuration with `prediction_targets` set to `[]`
- **WHEN** the model is initialized
- **THEN** the system SHALL use single-target mode with only `extrema`

### Requirement: Multi-Target Label Computation

The system SHALL compute label columns for all enabled targets during the
`set_freqai_targets()` method execution using **position-aware labeling**.

Each target SHALL produce a DataFrame column with the appropriate `&` prefix:
- `extrema` → `&s-extrema` (smoothed zigzag direction, always computed)
- `amplitude` → `&-amplitude` (remaining log move to next pivot)
- `time_to_pivot` → `&-time_to_pivot` (candles until next pivot)
- `efficiency` → `&-efficiency` (cumulative move efficiency so far)
- `natr` → `&-natr` (forward-looking NATR)

**Position-Aware Labeling**: Each candle SHALL have a **unique target value**
reflecting its position within the current move. This ensures gradient boosting
receives strong learning signals (unlike forward-fill which creates identical
targets for different feature vectors).

Label computation strategy SHALL be:
- `amplitude`: **Remaining move** - `log(pivot_price / current_price)` (decreases toward 0)
- `time_to_pivot`: **Countdown** - `pivot_index - current_index` (decreasing values)
- `efficiency`: **Cumulative efficiency** - `abs(net_move) / path_length` up to current candle
- `natr`: **Shifted NATR** - `NATR.shift(-label_period_candles)` (forward-looking volatility, period from HPO or fallback)

#### Scenario: Compute amplitude labels (remaining move)

- **GIVEN** a dataframe with a pivot at index 200 with price 105
- **AND** `amplitude` is in `prediction_targets`
- **AND** prices at indices [196, 197, 198, 199] are [100, 101, 102, 103]
- **WHEN** `set_freqai_targets()` is called
- **THEN** the `&-amplitude` column SHALL contain remaining log move values:
  - Index 196: `log(105/100) ≈ 0.049`
  - Index 197: `log(105/101) ≈ 0.039`
  - Index 198: `log(105/102) ≈ 0.029`
  - Index 199: `log(105/103) ≈ 0.019`
  - Index 200 (pivot): `log(105/105) = 0.0`
- **AND** each candle has a unique target value

#### Scenario: Compute time_to_pivot labels (countdown)

- **GIVEN** a dataframe with zigzag pivots at indices [100, 200, 350]
- **AND** `time_to_pivot` is in `prediction_targets`
- **WHEN** `set_freqai_targets()` is called
- **THEN** the `&-time_to_pivot` column SHALL contain decreasing countdown values:
  - Indices 100-199: countdown from 100 down to 1 (toward pivot at 200)
  - Index 200: value 0 (at pivot)
  - Indices 201-349: countdown from 149 down to 1 (toward pivot at 350)
- **AND** look-ahead is acceptable since labels always use future information

#### Scenario: Compute efficiency labels (cumulative)

- **GIVEN** a dataframe with a move starting at index 100 (price 100) to pivot at 200 (price 110)
- **AND** `efficiency` is in `prediction_targets`
- **AND** price path oscillates with cumulative path length increasing
- **WHEN** `set_freqai_targets()` is called
- **THEN** the `&-efficiency` column SHALL contain cumulative efficiency values:
  - Each candle: `abs(current_price - start_price) / cumulative_path_length`
  - Efficiency typically decreases if price oscillates (path length grows faster than net move)
- **AND** each candle has a unique target value

#### Scenario: Amplitude for bearish move (negative remaining move)

- **GIVEN** a dataframe with a bearish move: prices [110, 108, 106, 104] toward pivot at 100
- **AND** `amplitude` is in `prediction_targets`
- **WHEN** `set_freqai_targets()` is called
- **THEN** the `&-amplitude` column SHALL contain **negative** values:
  - `log(100/110) ≈ -0.095` (price will go DOWN)
  - `log(100/108) ≈ -0.077`
  - `log(100/106) ≈ -0.058`
  - `log(100/104) ≈ -0.039`
- **AND** the sign indicates direction (positive=bullish, negative=bearish)
- **AND** this aligns with `&s-extrema` sign convention

#### Scenario: Candles after last known pivot

- **GIVEN** a dataframe where the last detected pivot is at index 500
- **AND** the dataframe has candles up to index 520
- **WHEN** `set_freqai_targets()` is called
- **THEN** candles at indices 501-520 SHALL have `NaN` for prediction targets
- **AND** these candles are excluded from training (no future pivot known)

#### Scenario: Candles before first known pivot

- **GIVEN** a dataframe where the first detected pivot is at index 50
- **AND** the dataframe has candles starting at index 0
- **WHEN** `set_freqai_targets()` is called
- **THEN** candles at indices 0-49 SHALL have `NaN` for `amplitude` and `time_to_pivot`
- **AND** candles at indices 0-49 SHALL have `NaN` for `efficiency` (no previous pivot to measure from)
- **AND** these candles are excluded from training (no surrounding pivots known)

#### Scenario: No pivots detected in dataframe

- **GIVEN** a dataframe with price data
- **AND** `prediction_targets` is non-empty
- **AND** the zigzag algorithm detects zero pivots (e.g., monotonic price movement)
- **WHEN** `set_freqai_targets()` is called
- **THEN** all prediction target columns SHALL contain `NaN` values
- **AND** `&s-extrema` SHALL also be `NaN` (no pivot directions to label)
- **AND** the system SHALL NOT raise an exception
- **AND** these rows are effectively excluded from training (no valid labels)

#### Scenario: Efficiency at move start

- **GIVEN** a candle at the start of a move (index equals previous pivot)
- **AND** `efficiency` is in `prediction_targets`
- **WHEN** `set_freqai_targets()` is called
- **THEN** `&-efficiency` SHALL be `1.0` (perfect efficiency, no wasted movement yet)
- **AND** this handles the `0/0` edge case when `path_length = 0` and `net_move = 0`
- **AND** implementation SHALL use `np.where(path_length == 0, 1.0, abs(net_move) / path_length)`

#### Scenario: Only compute enabled targets

- **GIVEN** `prediction_targets` is `["amplitude"]`
- **WHEN** `set_freqai_targets()` is called
- **THEN** `&s-extrema` and `&-amplitude` columns SHALL be added
- **AND** `&-time_to_pivot`, `&-efficiency`, `&-natr` SHALL NOT be present

#### Scenario: Compute natr labels

- **GIVEN** a dataframe with NATR indicator calculated
- **AND** `natr` is in `prediction_targets`
- **AND** `label_period_candles` is 18 (from HPO or fallback)
- **WHEN** `set_freqai_targets()` is called
- **THEN** the `&-natr` column SHALL contain forward-shifted NATR values
- **AND** `&-natr[i] = NATR[i + label_period_candles]` for each candle
- **AND** the last `label_period_candles` candles SHALL have `NaN` (no future data available)
- **AND** the NATR period used for computation SHALL also be `label_period_candles`

### Requirement: Multi-Output Regressor Training

The system SHALL train a multi-output regression model when more than one target
is enabled (i.e., when `prediction_targets` is non-empty).

The multi-output strategy SHALL vary by regressor type:
- CatBoost: Use native `loss_function="MultiRMSE"` with `eval_metric="MultiRMSE"`
- XGBoost: Use native `multi_strategy="one_output_per_tree"` parameter
- LightGBM: Use `FreqaiMultiOutputRegressor` wrapper with per-target `fit_params`
- HistGradientBoostingRegressor: Use `FreqaiMultiOutputRegressor` wrapper
- NGBoost: Use `FreqaiMultiOutputRegressor` wrapper with per-target validation data

The system SHALL preserve the single-target training path when `prediction_targets`
is absent or empty.

#### Scenario: CatBoost multi-target training

- **GIVEN** `regressor` is `"catboost"`
- **AND** `prediction_targets` is `["amplitude"]`
- **WHEN** `fit()` is called
- **THEN** the system SHALL use CatBoostRegressor with `loss_function="MultiRMSE"`
- **AND** set `eval_metric="MultiRMSE"` for early stopping
- **AND** train a single model predicting both `extrema` and `amplitude` jointly

#### Scenario: XGBoost multi-target training

- **GIVEN** `regressor` is `"xgboost"`
- **AND** `prediction_targets` is `["amplitude", "efficiency"]`
- **WHEN** `fit()` is called
- **THEN** the system SHALL configure XGBRegressor with `multi_strategy="one_output_per_tree"`
- **AND** train a single model that outputs predictions for all three targets

#### Scenario: LightGBM multi-target training

- **GIVEN** `regressor` is `"lightgbm"`
- **AND** `prediction_targets` is `["amplitude"]`
- **WHEN** `fit()` is called
- **THEN** the system SHALL wrap LGBMRegressor with `FreqaiMultiOutputRegressor`
- **AND** prepare `fit_params` as list of dicts with per-target `eval_set` and `init_model`
- **AND** train one sub-model per target with early stopping on each

#### Scenario: Single target preserves existing behavior

- **GIVEN** `prediction_targets` is absent or `[]`
- **WHEN** `fit()` is called
- **THEN** the system SHALL use the existing single-target training path
- **AND** NOT use MultiOutputRegressor wrapper
- **AND** NOT use MultiRMSE loss function

### Requirement: Multi-Target Prediction Output

The system SHALL return predictions for all enabled targets as separate columns
in the prediction DataFrame.

Each prediction column SHALL use the same naming convention as the corresponding
label column (`&s-extrema`, `&-amplitude`, etc.).

#### Scenario: Multi-target prediction columns

- **GIVEN** a trained model with `prediction_targets` `["amplitude"]`
- **WHEN** `predict()` is called
- **THEN** the returned DataFrame SHALL contain columns `&s-extrema` and
  `&-amplitude`

#### Scenario: Single-target prediction unchanged

- **GIVEN** a model trained without `prediction_targets`
- **WHEN** `predict()` is called
- **THEN** the returned DataFrame SHALL contain only `&s-extrema`
- **AND** the prediction values SHALL be identical to the pre-multi-target
  implementation

### Requirement: Unified MultiRMSE Metric

The system SHALL implement a `compute_multi_rmse()` helper function that computes
the MultiRMSE metric aligned with CatBoost's formula.

The MultiRMSE formula SHALL be:
$$\text{MultiRMSE} = \sqrt{\frac{\sum_{i=1}^{N}\sum_{d=1}^{D}(y_{i,d} - \hat{y}_{i,d})^2 \cdot w_i}{\sum_{i=1}^{N} w_i}}$$

Where N is the number of samples, D is the number of targets, and w_i are
optional sample weights.

#### Scenario: MultiRMSE computation

- **GIVEN** predictions for 2 targets on 100 samples
- **AND** ground truth values for those targets
- **WHEN** `compute_multi_rmse(y_true, y_pred)` is called
- **THEN** the result SHALL match CatBoost's MultiRMSE calculation

### Requirement: Multi-Target Hyperparameter Optimization

The system SHALL use `compute_multi_rmse()` as the HPO objective function when
multiple targets are enabled.

The HPO objective SHALL be computed on all targets equally (no weighting).

This ensures HPO optimizes for overall multi-target performance, aligning with
the model's training objective.

#### Scenario: HPO with multi-target model

- **GIVEN** `prediction_targets` is `["amplitude", "time_to_pivot"]`
- **AND** Optuna HPO is enabled
- **WHEN** the optimization objective is computed
- **THEN** the system SHALL compute `compute_multi_rmse()` on all 3 targets
- **AND** use this single value as the minimization objective

#### Scenario: HPO with single-target model

- **GIVEN** `prediction_targets` is absent or `[]`
- **AND** Optuna HPO is enabled
- **WHEN** the optimization objective is computed
- **THEN** the system SHALL compute standard RMSE on `&s-extrema` only
- **AND** maintain backward compatibility

### Requirement: Early Stopping with Multi-Output

The system SHALL apply early stopping based on the regressor's native capabilities.

| Regressor | Early Stopping Behavior |
|-----------|------------------------|
| CatBoost | Global on MultiRMSE metric |
| XGBoost | Per-target or custom (experimental) |
| LightGBM | Per sub-model independently |
| HistGradientBoosting | Per sub-model independently |
| NGBoost | Per sub-model independently |

#### Scenario: CatBoost early stopping with MultiRMSE

- **GIVEN** `regressor` is `"catboost"`
- **AND** `prediction_targets` is non-empty
- **AND** `early_stopping_rounds` is configured
- **WHEN** training proceeds
- **THEN** early stopping SHALL be based on the MultiRMSE eval metric
- **AND** stop when MultiRMSE on validation set stops improving

#### Scenario: FreqaiMultiOutputRegressor early stopping

- **GIVEN** `regressor` is `"lightgbm"`
- **AND** `prediction_targets` is non-empty
- **AND** `early_stopping_rounds` is configured
- **WHEN** training proceeds
- **THEN** each sub-model SHALL receive its own `eval_set` via `fit_params`
- **AND** each sub-model applies early stopping on its own validation loss
- **AND** early stopping works correctly (unlike sklearn's MultiOutputRegressor)

### Requirement: Target Normalization

The system SHALL apply standardization and normalization to all prediction target
columns using `LabelTransformer`
to ensure MultiRMSE optimization is balanced across targets with different scales.

The `LabelTransformer` supports multiple standardization methods (zscore, robust,
mmad, power_yj) and normalization methods (maxabs, minmax, sigmoid), configured
via `label_config`. All targets are processed uniformly through the same pipeline.

The transformer state SHALL be stored and used to inverse-transform predictions
back to original scale.

#### Scenario: Multi-target normalization

- **GIVEN** `prediction_targets` is `["amplitude", "time_to_pivot"]`
- **WHEN** the label pipeline is applied during training
- **THEN** the system SHALL apply `LabelTransformer` to all target columns uniformly
- **AND** the same standardization/normalization config applies to all targets
- **AND** `&s-extrema`, `&-amplitude`, and `&-time_to_pivot` are all transformed

#### Scenario: Prediction inverse transform

- **GIVEN** a trained model with normalized targets
- **WHEN** `predict()` is called
- **THEN** the system SHALL inverse-transform predictions to original scale
- **AND** return values in the same units as the original labels

### Requirement: HPO Pruning Compatibility

The system SHALL disable Optuna pruning callbacks for wrapper-based regressors
(LightGBM, HistGradientBoosting, NGBoost) when multi-target is enabled.

Optuna pruning callbacks are incompatible with `FreqaiMultiOutputRegressor` because
they expect direct access to the regressor's training internals during iteration.

#### Scenario: Native regressor HPO with pruning

- **GIVEN** `regressor` is `"catboost"` or `"xgboost"`
- **AND** `prediction_targets` is non-empty
- **AND** Optuna HPO is enabled
- **WHEN** HPO trials are run
- **THEN** pruning callbacks SHALL be enabled (native multi-output supported)

#### Scenario: FreqaiMultiOutputRegressor HPO without pruning

- **GIVEN** `regressor` is `"lightgbm"`, `"histgradientboostingregressor"`, or `"ngboost"`
- **AND** `prediction_targets` is non-empty
- **AND** Optuna HPO is enabled
- **WHEN** HPO trials are run
- **THEN** Optuna pruning callbacks SHALL be disabled (incompatible with wrapper)
- **AND** trials SHALL run to completion (no early trial pruning)
- **AND** per-submodel early stopping still works via `fit_params`

### Requirement: Prediction Column Order

The system SHALL store target column names during `fit()` and use them to
reconstruct the prediction DataFrame in `predict()` with correct column alignment.

#### Scenario: Column order consistency

- **GIVEN** a model trained with targets `["&s-extrema", "&-amplitude", "&-time_to_pivot"]`
- **WHEN** `predict()` is called
- **THEN** the returned DataFrame columns SHALL be in the same order as training
- **AND** each column SHALL contain predictions for the corresponding target
