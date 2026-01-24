# Tasks: Multi-Target Regression Support

## 1. Configuration Infrastructure

- [ ] 1.1 Define `PREDICTION_TARGETS` constant mapping target names to column names in `Utils.py`
- [ ] 1.2 Add `get_prediction_targets()` helper to parse `prediction_targets` from freqai config
- [ ] 1.3 Add validation: targets must be subset of known prediction targets
- [ ] 1.4 Add `compute_multi_rmse()` function in `Utils.py` (aligned with CatBoost formula)
- [ ] 1.5 Define `WRAPPER_REGRESSORS` constant listing regressors that use `FreqaiMultiOutputRegressor`

## 2. Label Computation (Strategy)

Position-aware labeling: each candle gets a unique target value based on its position.
See design.md Decision 4 for architecture and Decision 11 for mathematical justification.

Architecture: `zigzag()` returns pivot indices/values only. A separate helper computes per-candle targets:

```
zigzag(df) → (pivots_indices, pivots_values_log, ...)
                           ↓
compute_prediction_targets(df, pivots_indices, pivots_values_log, enabled_targets)
                           ↓
               DataFrame with per-candle target columns
```

- [ ] 2.1 Add `compute_prediction_targets()` main function in `Utils.py`
      Signature: `(df, pivots_indices, pivots_values_log, enabled_targets) -> pd.DataFrame`
      Orchestrates calls to individual target helpers based on `enabled_targets` list
- [ ] 2.2 Add `_build_pivot_lookup()` internal helper in `Utils.py`
      Creates O(n) mapping: for each candle index, store (prev_pivot_idx, next_pivot_idx)
      Called once, shared by amplitude/time_to_pivot helpers
- [ ] 2.3 Add `_compute_remaining_amplitude()` internal helper in `Utils.py`
      Formula: `log(next_pivot_price / current_price)` - decreases toward 0 at pivot
      Uses pivot_lookup + OHLCV close prices (no interpolation)
- [ ] 2.4 Add `_compute_time_to_pivot()` internal helper in `Utils.py`
      Formula: `next_pivot_index - current_index` - countdown to next pivot (decreasing)
      Uses pivot_lookup (integer arithmetic only)
- [ ] 2.5 Add `_compute_cumulative_efficiency()` internal helper in `Utils.py`
      Formula: `abs(close - swing_start_price) / cumulative_path_length`
      Tracks incremental path length per candle within each swing
- [ ] 2.6 Add `_compute_natr()` internal helper in `Utils.py`
      Formula: `NATR(timeperiod=label_period_candles).shift(-label_period_candles)`
      Period sourced from HPO (or fallback to `label_period_candles` from config)
      Standalone (does not use pivot_lookup)
- [ ] 2.7 Extend `set_freqai_targets()` to call `compute_prediction_targets()`
      Pass zigzag outputs + enabled targets from config
- [ ] 2.8 Ensure `&s-extrema` always computed (implicit)

## 3. Multi-Output Regressor Support

### 3.1 CatBoost (Native MultiRMSE)
- [ ] 3.1.1 Configure `loss_function='MultiRMSE'` when multi-target enabled
- [ ] 3.1.2 Configure `eval_metric='MultiRMSE'` for early stopping alignment
- [ ] 3.1.3 Native early stopping works unchanged

### 3.2 XGBoost (Native Multi-Output - Experimental)
- [ ] 3.2.1 Configure `multi_strategy='one_output_per_tree'` when multi-target enabled
- [ ] 3.2.2 Test native multi-output behavior
- [ ] 3.2.3 Fallback to `MultiOutputRegressor` wrapper if needed

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
- [ ] 4.6 Rename `ExtremaWeightingTransformer` → `LabelTransformer` to reflect generalized role
- [ ] 4.7 Log per-target RMSE + aggregate MultiRMSE during training

## 5. Prediction Integration

- [ ] 5.1 Handle multi-column prediction output in `predict()` method
- [ ] 5.2 Use stored `self._target_columns` to reconstruct DataFrame with correct column names
- [ ] 5.3 Ensure single-target prediction unchanged

## 6. Hyperparameter Optimization

- [ ] 6.1 Modify HPO objective to use `compute_multi_rmse()` when multi-target enabled
- [ ] 6.2 Verify Optuna HPO works with native multi-output (CatBoost, XGBoost)
- [ ] 6.3 Disable Optuna pruning callbacks for wrapper-based regressors (incompatible with MultiOutputRegressor)
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
