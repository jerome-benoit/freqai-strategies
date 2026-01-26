# Tasks: Multi-Target Regression Support

> **Note**: The `LabelGenerator` architecture foundation was implemented in PR #45.
> Tasks below represent remaining work to achieve multi-target regression.

## 1. Configuration Infrastructure

- [ ] 1.1 Define `PREDICTION_TARGETS` constant mapping target names to column names in `Utils.py`
      Example: `{"amplitude": "&-amplitude", "time_to_pivot": "&-time_to_pivot", ...}`
- [ ] 1.2 Add `get_prediction_targets()` helper to parse `prediction_targets` from freqai config
      Returns list of column names based on config list (e.g., `["amplitude"]` -> `["&-amplitude"]`)
- [ ] 1.3 Add validation: targets must be subset of known prediction targets
      Raise `ValueError` for unknown targets, listing valid options
- [ ] 1.4 Add `compute_multi_rmse()` function in `Utils.py` (aligned with CatBoost formula)
- [ ] 1.5 Define `WRAPPER_REGRESSORS` constant listing regressors that use `FreqaiMultiOutputRegressor`
      Value: `("lightgbm", "histgradientboostingregressor", "ngboost")`
- [ ] 1.6 Add `get_label_columns()` function that dynamically builds label columns tuple
      Combines `EXTREMA_COLUMN` with enabled prediction targets from config

## 2. Label Computation (Strategy)

Position-aware labeling: each candle gets a unique target value based on its position.
See design.md Decision 4 for architecture and Decision 11 for mathematical justification.

**Existing infrastructure** (from PR #45):
- `LabelData`, `LabelGenerator`, `register_label_generator()`, `generate_label_data()`
- `EXTREMA_COLUMN` generator registered
- `set_freqai_targets()` loops over `LABEL_COLUMNS`

**Remaining work**:

- [ ] 2.1 Add `_build_pivot_lookup()` internal helper in `Utils.py`
      Creates O(n) mapping: for each candle index, store (prev_pivot_idx, next_pivot_idx)
      Called once, shared by amplitude/time_to_pivot generators
- [ ] 2.2 Add `_generate_amplitude_label()` label generator in `Utils.py`
      Formula: `log(next_pivot_price / current_price)` - decreases toward 0 at pivot
      Register with `register_label_generator("&-amplitude", _generate_amplitude_label)`
- [ ] 2.3 Add `_generate_time_to_pivot_label()` label generator in `Utils.py`
      Formula: `next_pivot_index - current_index` - countdown to next pivot (decreasing)
      Register with `register_label_generator("&-time_to_pivot", _generate_time_to_pivot_label)`
- [ ] 2.4 Add `_generate_efficiency_label()` label generator in `Utils.py`
      Formula: `abs(close - swing_start_price) / cumulative_path_length`
      Register with `register_label_generator("&-efficiency", _generate_efficiency_label)`
- [ ] 2.5 Add `_generate_natr_label()` label generator in `Utils.py`
      Formula: `NATR(timeperiod=label_period_candles).shift(-label_period_candles)`
      Register with `register_label_generator("&-natr", _generate_natr_label)`
- [ ] 2.6 Update `LABEL_COLUMNS` to be computed dynamically from config
      Replace static tuple with call to `get_label_columns(prediction_targets)`

## 3. Multi-Output Regressor Support

### 3.1 CatBoost (Native MultiRMSE)
- [ ] 3.1.1 Configure `loss_function='MultiRMSE'` when multi-target enabled
- [ ] 3.1.2 Configure `eval_metric='MultiRMSE'` for early stopping alignment
- [ ] 3.1.3 Native early stopping works unchanged

### 3.2 XGBoost (Native Multi-Output - Experimental)
- [ ] 3.2.1 Configure `multi_strategy='one_output_per_tree'` when multi-target enabled
- [ ] 3.2.2 Test native multi-output behavior
- [ ] 3.2.3 Fallback to `FreqaiMultiOutputRegressor` wrapper if native fails

### 3.3 LightGBM (FreqaiMultiOutputRegressor Wrapper)
- [ ] 3.3.1 Wrap with `FreqaiMultiOutputRegressor` when multi-target enabled
- [ ] 3.3.2 Prepare `fit_params` as list of dicts (one per target) with `eval_set`, `eval_sample_weight`, `init_model`
- [ ] 3.3.3 Enable parallel training via `n_jobs` parameter
- [ ] 3.3.4 Document: early stopping runs per-submodel with own eval_set

### 3.4 HistGradientBoosting (FreqaiMultiOutputRegressor Wrapper)
- [ ] 3.4.1 Wrap with `FreqaiMultiOutputRegressor` when multi-target enabled
- [ ] 3.4.2 Prepare `fit_params` as list of dicts (HistGB has limited fit_params support)
- [ ] 3.4.3 Document: early stopping runs per-submodel independently

### 3.5 NGBoost (FreqaiMultiOutputRegressor Wrapper)
- [ ] 3.5.1 Wrap with `FreqaiMultiOutputRegressor` when multi-target enabled
- [ ] 3.5.2 Prepare `fit_params` as list of dicts with `X_val`, `Y_val` for early stopping
- [ ] 3.5.3 Document: early stopping runs per-submodel independently

## 4. Model Training Integration

- [ ] 4.1 Modify `QuickAdapterRegressorV3.fit()` to detect multi-target mode via `prediction_targets`
- [ ] 4.2 Route to appropriate multi-output setup based on regressor type
- [ ] 4.3 Preserve single-target code path when `prediction_targets` empty/absent
- [ ] 4.4 Log enabled targets and multi-output strategy used
- [ ] 4.5 Store `self._target_columns = y.columns.tolist()` during fit for prediction alignment
- [ ] 4.6 Log per-target RMSE + aggregate MultiRMSE during training

## 5. Prediction Integration

- [ ] 5.1 Handle multi-column prediction output in `predict()` method
- [ ] 5.2 Use stored `self._target_columns` to reconstruct DataFrame with correct column names
- [ ] 5.3 Ensure single-target prediction unchanged

## 6. Hyperparameter Optimization

- [ ] 6.1 Modify HPO objective to use `compute_multi_rmse()` when multi-target enabled
- [ ] 6.2 Verify Optuna HPO works with native multi-output (CatBoost, XGBoost)
- [ ] 6.3 Disable Optuna pruning callbacks for wrapper-based regressors (incompatible with FreqaiMultiOutputRegressor)
- [ ] 6.4 Single-objective optimization (no Pareto)
- [ ] 6.5 Document that HPO optimizes combined MultiRMSE, not per-target
- [ ] 6.6 Document: wrapper HPO trials run longer (no early pruning)

## 7. Testing

- [ ] 7.1 Add unit tests for label computation helpers (remaining_amplitude, time_to_pivot countdown, cumulative_efficiency, natr)
- [ ] 7.2 Add unit tests for `compute_multi_rmse()` function
- [ ] 7.3 Add unit tests for multi-output setup per regressor
- [ ] 7.4 Add unit tests for `LabelTransformer` multi-target handling (fit/transform/inverse_transform)
- [ ] 7.5 Add integration test: single-target backward compatibility
- [ ] 7.6 Add integration test: multi-target training with CatBoost (native)
- [ ] 7.7 Add integration test: multi-target training with wrappers
- [ ] 7.8 Add integration test: config validation errors
- [ ] 7.9 Add integration test: prediction column order matches training

## 8. Documentation

- [ ] 8.1 Update config-template.json with `prediction_targets` example
- [ ] 8.2 Document available targets and their semantics
- [ ] 8.3 Document per-regressor multi-output behavior (native vs wrapper)
- [ ] 8.4 Document HPO behavior with multi-target (single-objective MultiRMSE)
- [ ] 8.5 Add example config snippets for common use cases
