# Change: Add Multi-Target Regression Support

## Why

Single-target regression (`&s-extrema`) limits the strategy to directional signals only.
The zigzag algorithm computes rich metrics (amplitude, efficiency ratio, duration,
volatility) that can inform trading decisions. Predicting these additional targets
enables:

- **Smarter position sizing**: amplitude prediction informs expected move magnitude
- **Dynamic stoploss/take-profit**: future volatility prediction improves risk management
- **Timing optimization**: time-to-pivot prediction helps entry/exit patience logic
- **Signal quality filtering**: efficiency ratio prediction identifies clean vs choppy moves

## What Changes

- **Configuration**: Add `prediction_targets` list in freqai config section
- **Labeling**: Register additional label generators for each prediction target
- **Training**: Use native multi-output for CatBoost (MultiRMSE) and XGBoost (one_output_per_tree), wrap others with `FreqaiMultiOutputRegressor`
- **Prediction**: Handle multi-column predictions and route to strategy via `&` column convention
- **HPO**: Single-objective optimization using `compute_multi_rmse()` aligned with CatBoost formula
- **Backward compatibility**: Empty `prediction_targets` = single-target mode

### Target Architecture

`extrema` is always included implicitly and cannot be disabled. It drives trading
decisions (entry/exit signals).

`prediction_targets` lists additional targets to predict - optional targets that
enhance position sizing, risk management, and signal quality filtering.

### Available Targets

| Target | Column | Description | Labeling Strategy |
|--------|--------|-------------|-------------------|
| extrema | `&s-extrema` | Smoothed zigzag direction (-1/0/+1) | **Always included (implicit)** |
| amplitude | `&-amplitude` | Remaining log move to next pivot | `log(pivot_price / current_price)` |
| time_to_pivot | `&-time_to_pivot` | Countdown to next pivot | `pivot_index - current_index` |
| efficiency | `&-efficiency` | Cumulative move efficiency so far | `abs(net_move) / path_length` |
| natr | `&-natr` | Forward-looking NATR | `NATR.shift(-label_period_candles)` |

**Position-Aware Labeling**: Each candle has a **unique target value** reflecting its
position within the current move. This ensures gradient boosting receives strong
learning signals (unlike forward-fill which creates degenerate targets).
See design.md Decision 11 for mathematical justification.

### Configuration Example

```json
{
  "freqai": {
    "prediction_targets": ["amplitude", "time_to_pivot"]
  }
}
```

This configuration trains a multi-output model predicting 3 targets:
1. `&s-extrema` (implicit)
2. `&-amplitude`
3. `&-time_to_pivot`

**Single-target mode** (default):
```json
{
  "freqai": {
    "prediction_targets": []
  }
}
```
Or simply omit `prediction_targets` entirely.

### Regressor Support

| Regressor | Multi-Output Support | Loss/Eval |
|-----------|---------------------|-----------|
| CatBoost | Native | `MultiRMSE` loss + eval_metric |
| XGBoost | Native | `multi_strategy="one_output_per_tree"` |
| LightGBM | `FreqaiMultiOutputRegressor` wrapper | Per-submodel with eval_set |
| HistGradientBoosting | `FreqaiMultiOutputRegressor` wrapper | Per-submodel |
| NGBoost | `FreqaiMultiOutputRegressor` wrapper | Per-submodel |

**Note**: `FreqaiMultiOutputRegressor` is a FreqAI extension of sklearn's `MultiOutputRegressor`
that accepts `fit_params` as a list of dicts (one per target), enabling per-submodel `eval_set`,
`init_model`, and early stopping parameters.

### Target Normalization

Prediction targets have different scales (e.g., `time_to_pivot` ~1-500 vs `efficiency` ~0-1).
To ensure balanced MultiRMSE optimization, all targets are normalized via `LabelTransformer`
which supports multiple standardization methods (zscore, robust, mmad, power_yj) and
normalization methods (maxabs, minmax, sigmoid). Predictions are inverse-transformed to
original scale.

## Impact

- Affected specs: `quickadapter-ml` (new capability)
- Affected code:
  - `QuickAdapterV3.py`: `set_freqai_targets()` method (extend dynamic label columns)
  - `QuickAdapterRegressorV3.py`: `fit()`, `predict()` methods
  - `Utils.py`: `fit_regressor()` function, new multi-target wrapper, `compute_multi_rmse()`, new label generators
- **No new model files**: Implementation uses `QuickAdapterRegressorV3.py`
- **All 5 regressors supported**: XGBoost, LightGBM, HistGradientBoosting, NGBoost, CatBoost

## Remaining Work

This proposal extends the existing `LabelGenerator` infrastructure by:

1. Adding `PREDICTION_TARGETS` mapping and config parsing
2. Registering additional generators for each prediction target
3. Making `LABEL_COLUMNS` dynamic based on config
4. Implementing `FreqaiMultiOutputRegressor` wrapper
5. Adding multi-output support per regressor backend in `fit_regressor()`
6. Implementing `compute_multi_rmse()` metric
7. Integrating with HPO via unified MultiRMSE metric
