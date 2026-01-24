# Design: Multi-Target Regression Support

## Context

QuickAdapterV3 predicts `&s-extrema` as its primary target. The zigzag labeling
algorithm computes multiple metrics per pivot. This design enables prediction of
multiple targets simultaneously while maintaining backward compatibility and
supporting all 5 regressors.

### Stakeholders

- Strategy developers: Need flexible target configuration
- Model maintainers: Must support 5 different regressor backends
- FreqAI integration: Must respect column conventions and data flow

## Goals / Non-Goals

### Goals

1. Enable prediction of multiple complementary targets from zigzag metrics
2. Provide configuration-driven target selection via `prediction_targets`
3. Support all 5 regressors with their specific multi-output APIs
4. Maintain backward compatibility (default = single-target behavior)
5. No new model files; use `QuickAdapterRegressorV3.py`
6. Leverage `zigzag()` output without re-computation
7. Align HPO objective with model training metric (MultiRMSE)

### Non-Goals

1. Classification targets (out of scope; all targets are regression)
2. Custom user-defined targets (fixed set of known targets)
3. Per-target hyperparameter optimization (shared parameters across targets)
4. Target-specific loss weighting (uniform treatment, all targets equal weight)

## Decisions

### Decision 1: Configuration Structure

**What**: Use a `prediction_targets` section in FreqAI config with a list of
targets to predict. The target `extrema` is always included implicitly.

**Why**: 
- `extrema` is the decision target for the swing strategy (always required)
- Additional prediction targets provide signals for position sizing and risk management

```python
"freqai": {
    "prediction_targets": ["amplitude", "time_to_pivot"],
    # extrema is ALWAYS included implicitly
    # Available prediction targets: amplitude, time_to_pivot, efficiency, natr
}
```

**Default behavior**: When `prediction_targets` is absent or empty, only
`extrema` is predicted (backward compatible single-target mode).

**Alternatives considered**:
- `enabled_targets` including extrema: Confusing, user might try to disable it
- Nested config with weights: Premature optimization, adds complexity

### Decision 2: Multi-Output Strategy per Regressor

**What**: Use native multi-output support where available; wrap with
`FreqaiMultiOutputRegressor` otherwise.

| Regressor | Multi-Output Strategy | Native Loss/Metric |
|-----------|----------------------|-------------------|
| CatBoost | Native: `loss_function="MultiRMSE"` | MultiRMSE |
| XGBoost | Native: `multi_strategy="one_output_per_tree"` | MSE (per target) |
| LightGBM | `FreqaiMultiOutputRegressor` wrapper | RMSE (per sub-model with eval_set) |
| HistGradientBoostingRegressor | `FreqaiMultiOutputRegressor` wrapper | squared_error (per sub-model) |
| NGBoost | `FreqaiMultiOutputRegressor` wrapper | NLL (per sub-model) |

**Why**: Native support provides better performance and feature interaction
modeling. `FreqaiMultiOutputRegressor` (FreqAI's extension of sklearn's
`MultiOutputRegressor`) accepts `fit_params` as a list of dicts, enabling
per-submodel `eval_set`, `init_model`, and early stopping parameters.

**XGBoost Native Multi-Output**:
```python
model = XGBRegressor(
    tree_method="hist",
    multi_strategy="one_output_per_tree",
)
model.fit(X, y)  # y shape: (n_samples, n_targets)
```

**CatBoost Native Multi-Output**:
```python
model = CatBoostRegressor(
    loss_function="MultiRMSE",
    eval_metric="MultiRMSE",
)
model.fit(X, y)  # y shape: (n_samples, n_targets)
```

**Alternatives considered**:
- Always use wrapper: Loses native multi-output benefits for XGBoost/CatBoost
- Always use native: Not all regressors support it

### Decision 3: Unified MultiRMSE Metric

**What**: Implement a `compute_multi_rmse()` helper function that matches
CatBoost's MultiRMSE formula. Use this for HPO objective across ALL regressors.

**Why**: 
- Ensures HPO optimizes the same metric regardless of regressor
- Aligns with CatBoost's native MultiRMSE (de facto standard)
- Enables fair comparison between regressors

**MultiRMSE Formula** (CatBoost standard):
$$\text{MultiRMSE} = \sqrt{\frac{\sum_{i=1}^{N}\sum_{d=1}^{D}(y_{i,d} - \hat{y}_{i,d})^2 \cdot w_i}{\sum_{i=1}^{N} w_i}}$$

Where:
- $N$ = number of samples
- $D$ = number of targets
- $w_i$ = sample weight (optional)

**Implementation**:
```python
def compute_multi_rmse(
    y_true: np.ndarray,      # (n_samples, n_targets)
    y_pred: np.ndarray,      # (n_samples, n_targets)
    sample_weights: Optional[np.ndarray] = None,
) -> float:
    """Compute MultiRMSE aligned with CatBoost's formula."""
    if sample_weights is None:
        sample_weights = np.ones(y_true.shape[0])
    
    squared_errors = (y_true - y_pred) ** 2  # (n_samples, n_targets)
    weighted_sum = np.sum(sample_weights[:, None] * squared_errors)
    
    return np.sqrt(weighted_sum / sample_weights.sum())
```

**Metric Alignment Table**:

| Regressor | Training Loss | Early Stop Metric | HPO Objective |
|-----------|---------------|-------------------|---------------|
| CatBoost | MultiRMSE | MultiRMSE | `compute_multi_rmse()` |
| XGBoost | MSE (sum) | Custom eval or post-hoc | `compute_multi_rmse()` |
| LightGBM | RMSE × N | Per-model RMSE | `compute_multi_rmse()` |
| HistGB | squared_error × N | Per-model loss | `compute_multi_rmse()` |
| NGBoost | NLL × N | Per-model NLL | `compute_multi_rmse()` |

### Decision 4: Label Computation in Strategy

**What**: Compute all enabled target columns in `set_freqai_targets()` using
a **separate helper function** that takes `zigzag()` output plus the OHLCV DataFrame.

**Why**: 
- `zigzag()` detects pivots and computes per-move metrics (amplitude, efficiency at pivot)
- Auxiliary targets require **per-candle** computation (not interpolation)
- Separating concerns: `zigzag()` handles pivot detection, helper handles target expansion
- Each candle gets an **exact calculated value** based on its position and OHLCV data

**Architecture**:
```
zigzag(df) → (pivots_indices, pivots_values_log, ...)
                           ↓
compute_prediction_targets(df, pivots_indices, pivots_values_log, enabled_targets)
                           ↓
               DataFrame with per-candle target columns
```

**Target Column Naming**:
| Target | Column Name | Source |
|--------|-------------|--------|
| extrema | `&s-extrema` | Smoothed zigzag direction (implicit) |
| amplitude | `&-amplitude` | Log remaining move to next pivot |
| time_to_pivot | `&-time_to_pivot` | Candles until next pivot |
| efficiency | `&-efficiency` | Cumulative move efficiency so far |
| natr | `&-natr` | Forward-looking NATR |

**Label computation approach**:
```python
def compute_prediction_targets(
    df: pd.DataFrame,
    pivots_indices: list[int],
    pivots_values_log: list[float],
    enabled_targets: list[str],
) -> pd.DataFrame:
    """
    Compute per-candle target values using zigzag output and OHLCV data.
    
    No interpolation - each candle gets an exact calculated value based on:
    - Its position relative to surrounding pivots
    - Its actual OHLCV values
    """
    targets_df = pd.DataFrame(index=df.index)
    closes_log = np.log(df["close"].to_numpy())
    
    # Build pivot lookup: for each candle, find prev/next pivot
    # ... (implementation details)
    
    if "amplitude" in enabled_targets:
        # For each candle: log(next_pivot_price / current_price)
        targets_df["&-amplitude"] = _compute_remaining_amplitude(
            closes_log, pivots_indices, pivots_values_log
        )
    
    if "time_to_pivot" in enabled_targets:
        # For each candle: next_pivot_index - current_index
        targets_df["&-time_to_pivot"] = _compute_time_to_pivot(
            len(df), pivots_indices
        )
    
    if "efficiency" in enabled_targets:
        # For each candle: cumulative efficiency from prev_pivot to current
        targets_df["&-efficiency"] = _compute_cumulative_efficiency(
            closes_log, pivots_indices
        )
    
    return targets_df

# In set_freqai_targets():
targets = self.freqai_info.get("prediction_targets", [])
if targets:
    targets_df = compute_prediction_targets(
        dataframe, pivots_indices, pivots_values_log, targets
    )
    dataframe = dataframe.join(targets_df)
```

**Key Implementation Note**: The helper functions iterate over ALL candles, not just
pivots. For efficiency, we pre-compute a mapping of `candle_index → (prev_pivot, next_pivot)`
to avoid repeated searches.

### Decision 5: Continuous Position-Aware Labeling

**What**: Use **position-aware labeling** that gives each candle a unique target
value reflecting its position within the current move. See Decision 11 for full
mathematical justification.

| Target | Labeling Strategy | Formula |
|--------|------------------|---------|
| `amplitude` | **Remaining move** to next pivot | `log(pivot_price / current_price)` |
| `time_to_pivot` | **Countdown** to next pivot | `pivot_index - current_index` |
| `efficiency` | **Cumulative efficiency** of move so far | `abs(net_move) / path_length` |
| `natr` | **Shifted rolling NATR** | `NATR.shift(-period)` |

**Why**: Forward-fill labeling is mathematically unsound for regression because
it creates identical targets for different feature vectors. Position-aware
labeling ensures:

1. Each candle has a **unique target** based on its position
2. Targets vary **continuously** (no jumps except at pivots)
3. Gradient boosting receives **strong gradient signals** for learning

**Note**: All strategies require knowledge of future pivot locations, which is
acceptable since labels by definition use future information.

### Decision 6: HPO Single Objective with MultiRMSE

**What**: Use single-objective HPO with `compute_multi_rmse()` as the objective
function, not Pareto front multi-objective.

**Why**:
- Simpler: One best trial, not a Pareto front to select from
- Aligned: Matches CatBoost's training objective
- Consistent: Same metric for all regressors enables fair comparison

**HPO Objective Implementation**:
```python
def hp_objective_multi_target(
    trial, regressor, X, y, train_weights, X_test, y_test, test_weights, ...
) -> float:
    # ... fit model ...
    y_pred = model.predict(X_test)
    
    # Single objective: MultiRMSE across all targets
    return compute_multi_rmse(y_test.values, y_pred, test_weights)
```

**Alternatives considered**:
- Pareto front (multi-objective): More complex, requires selection method
- Primary-only RMSE: Ignores other prediction targets in optimization
- Weighted sum: Adds configuration complexity (deferred to future)

### Decision 7: Early Stopping Behavior

**What**: Early stopping behavior varies by regressor type.

| Regressor | Early Stopping Behavior |
|-----------|------------------------|
| CatBoost | Global on MultiRMSE (native) |
| XGBoost | Per-target or custom callback (experimental) |
| LightGBM | Per sub-model independently |
| HistGB | Per sub-model independently |
| NGBoost | Per sub-model independently |

**Why**: Native multi-output regressors can use global early stopping on the
combined metric. Wrapper-based regressors must stop each sub-model independently
as there's no unified loss during training.

**Mitigation for wrappers**: The final model quality is evaluated via
`compute_multi_rmse()` in HPO, ensuring the best hyperparameters optimize
for all targets even if early stopping is per-model.

## Risks / Trade-offs

### Risk 1: Training Time Increase
**Risk**: Multi-output training may be slower, especially with wrappers.
**Mitigation**: Wrapper fits models sequentially; native multi-output is
preferred. Document expected overhead (~N× for N targets with wrapper).

### Risk 2: Memory Usage
**Risk**: Multiple models in memory for wrapper approach.
**Mitigation**: Each sub-model is smaller than single-target model;
total memory similar to N independent models.

### Risk 3: Target Correlation
**Risk**: Correlated targets may not improve strategy performance.
**Mitigation**: Targets are optional; users can enable/disable based on
backtesting results. Default config has no prediction targets.

### Risk 4: Early Stopping Inconsistency
**Risk**: Wrapper-based regressors can't do global early stopping on MultiRMSE.
**Mitigation**: HPO still optimizes MultiRMSE; per-model early stopping is
acceptable as each sub-model learns its target well.

### Risk 5: XGBoost Multi-Output Experimental
**Risk**: XGBoost multi-output is marked experimental, API may change.
**Mitigation**: Use `one_output_per_tree` strategy which is more stable.
Monitor XGBoost releases for breaking changes.

## Migration Plan

### Phase 1: Backward-Compatible Default
- Default: no `prediction_targets` = single-target `&s-extrema` only
- No migration required for users

### Phase 2: Opt-In Multi-Target
- Users add `prediction_targets` to config as needed
- Models trained without `prediction_targets` continue to work

### Rollback
- Remove `prediction_targets` from config
- No data migration needed

### Decision 8: Target Normalization for MultiRMSE

**What**: Apply standardization and normalization to all target columns using
`LabelTransformer`. This transformer supports multiple standardization methods
(zscore, robust, mmad, power_yj) and normalization methods (maxabs, minmax, sigmoid).

**Rename**: `ExtremaWeightingTransformer` → `LabelTransformer` to reflect its
generalized role in processing all label columns, not just extrema weighting.

**Why**: MultiRMSE sums squared errors across all targets without weighting.
Targets have vastly different scales:

| Target | Typical Range | Squared Error Scale |
|--------|---------------|---------------------|
| `extrema` | [-1, +1] | ~0.01-0.1 |
| `amplitude` | [0.01, 0.5] | ~0.001-0.01 |
| `time_to_pivot` | [1, 500] | ~100-10000 |
| `efficiency` | [0, 1] | ~0.01-0.1 |
| `natr` | [0.5%, 5%] | ~0.0001-0.001 |

Without normalization, `time_to_pivot` would dominate optimization (errors ~10000×
larger than `efficiency`).

**Implementation**: `LabelTransformer` provides comprehensive transformation capabilities:
- **Standardization**: zscore, robust, mmad, power_yj
- **Normalization**: maxabs, minmax, sigmoid
- **Post-processing**: gamma correction

The `define_label_pipeline()` applies the transformer to all label columns
uniformly. Datasieve pipelines process all columns, so multi-target is handled
naturally.

```python
# In define_label_pipeline():
return Pipeline([
    (
        "label_transformer",
        LabelTransformer(label_config=label_config),
    ),
])
```

**Alternatives considered**:
- New MultiTargetStandardScaler: Unnecessary, `LabelTransformer` is sufficient
- Per-target different configs: Adds complexity, uniform treatment is simpler
- No scaling: Mathematically unsound for MultiRMSE

### Decision 9: Wrapper Capabilities with FreqaiMultiOutputRegressor

**What**: Use `FreqaiMultiOutputRegressor` (FreqAI's enhanced wrapper) which
supports per-submodel `fit_params` including `eval_set` and `init_model`.

**FreqaiMultiOutputRegressor capabilities**:

| Feature | Native (CatBoost/XGBoost) | FreqaiMultiOutputRegressor (LightGBM/HistGB/NGBoost) |
|---------|---------------------------|------------------------------------------------------|
| Early stopping | Global on MultiRMSE | Per sub-model on own eval_set |
| eval_set support | Native | **Supported via fit_params list** |
| init_model (warm start) | Native | **Supported via fit_params list** |
| HPO pruning | Supported | **Disabled** (incompatible) |
| Training time | 1 model | N models (parallel if n_jobs set) |

**Why**: `FreqaiMultiOutputRegressor` extends sklearn's `MultiOutputRegressor`
to accept `fit_params` as a list of dicts (one per target). This enables:
- Per-target `eval_set` for early stopping
- Per-target `init_model` for warm starting
- Parallel training via `n_jobs` parameter

**Remaining limitation**: Optuna pruning callbacks still incompatible because
they require direct access to training internals during iteration.

**Mitigation**:
1. Disable Optuna pruning callbacks when using wrapper-based multi-output
2. HPO still optimizes MultiRMSE on final predictions (post-training)
3. Document that wrapper HPO trials run longer (no early pruning)

**Implementation (LightGBM example)**:
```python
# Prepare fit_params as list of dicts (one per target)
fit_params = []
for i in range(n_targets):
    fit_params.append({
        "eval_set": [(X_test, y_test.iloc[:, i])],
        "eval_sample_weight": [test_weights],
        "init_model": init_models[i] if init_models else None,
    })

model = FreqaiMultiOutputRegressor(estimator=lgb, n_jobs=n_targets)
model.fit(X=X, y=y, sample_weight=sample_weight, fit_params=fit_params)
```

### Decision 10: Prediction Column Order Tracking

**What**: Explicitly store target column names during `fit()` and use them
to reconstruct prediction DataFrame in `predict()`.

**Why**: `MultiOutputRegressor.predict()` returns columns in `y.columns` order
from training. If order changes between sessions, predictions misalign.

**Implementation**:
```python
# In fit():
self._target_columns = y.columns.tolist()

# In predict():
predictions = model.predict(X)
return pd.DataFrame(predictions, columns=self._target_columns, index=X.index)
```

### Decision 11: Mathematically Coherent Labeling Strategy

**What**: Use **continuous, position-aware labeling** for all prediction targets
to ensure mathematical coherence with gradient boosting regression.

**Why**: Forward-fill labeling creates **degenerate learning signals**:

```
# Forward-fill problem: same target for different features
Candle:     |  0  |  1  |  2  |  3  | 4=pivot |
Features:   | F0  | F1  | F2  | F3  |   F4    |  ← all different
amplitude:  | 0.05| 0.05| 0.05| 0.05|  0.05   |  ← all same!
```

The model sees different inputs mapping to identical outputs, learning to predict
the **mean** with low confidence. This is mathematically unsound for regression.

**State of the Art Reference**: The Triple Barrier Method (López de Prado, 2018)
labels each sample with its **specific outcome** (return at barrier hit), not a
constant forward-filled value. This ensures each sample has a unique, meaningful target.

**Labeling Strategy per Target**:

| Target | Strategy | Formula | Rationale |
|--------|----------|---------|-----------|
| `amplitude` | **Remaining move** | `log(pivot_price / current_price)` | "How much movement remains?" - useful for position sizing |
| `time_to_pivot` | **Countdown** | `pivot_index - current_index` | Decreasing to 0 at pivot |
| `efficiency` | **Cumulative efficiency** | `abs(net_move) / path_length` up to current candle | Quality of move so far |
| `natr` | **Shifted NATR** | `NATR.shift(-period)` | Forward-looking volatility (period = `label_period_candles` from HPO or fallback) |

**Remaining Move Amplitude Example**:

```
Candle:      |   0   |   1   |   2   |   3   | 4=pivot |
Price:       |  100  |  101  |  102  |  103  |   105   |
Pivot price: |  105  |  105  |  105  |  105  |   105   |
amplitude:   | 0.049 | 0.039 | 0.029 | 0.019 |  0.00   |
             # log(105/100), log(105/101), log(105/102), ...
```

Each candle has a **unique target** that reflects its position in the move.
The model learns: "given these features, how much movement remains?"

**Mathematical Properties**:

1. **Monotonicity**: `amplitude` decreases monotonically toward pivot (no jumps)
2. **Continuity**: Values change smoothly between candles
3. **Boundedness**: `amplitude` ∈ [0, max_amplitude], `time_to_pivot` ∈ [0, max_duration]
4. **Coherence with extrema**: All targets vary continuously, matching smoothed extrema

**Comparison with Forward-Fill**:

| Property | Forward-Fill | Remaining Move |
|----------|--------------|----------------|
| Unique targets per candle | ❌ No | ✅ Yes |
| Gradient signal quality | ❌ Weak (constant) | ✅ Strong (varying) |
| Trading utility | ⚠️ "What was the total move?" | ✅ "How much remains?" |
| Mathematical coherence | ❌ Degenerate | ✅ Sound |

**Implementation Note**: The "remaining move" requires knowing the next pivot
price, which is available during labeling (offline) but not during prediction.
This is acceptable because **all labels use future information by definition**
(including `&s-extrema` which uses future pivot locations for smoothing).

## Open Questions

1. **Logging**: How to log per-target metrics during training?
   **Recommendation**: Log individual RMSE per target + MultiRMSE aggregate.

2. **XGBoost eval_metric**: Should we implement custom XGBoost eval callback
   for MultiRMSE during training?
   **Recommendation**: Start without (rely on HPO), add if needed.
