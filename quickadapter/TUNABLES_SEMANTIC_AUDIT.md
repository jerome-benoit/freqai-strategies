# QuickAdapter Tunables Semantic Audit

This document inventories **user-exposed tunables** in QuickAdapter and tracks naming/units normalization changes.

## Scope

- **In-scope**: tunables that **directly change QuickAdapter's own algorithmic behavior** (branches/formulas/selection logic) inside:
  - `quickadapter/user_data/strategies/QuickAdapterV3.py`
  - `quickadapter/user_data/freqaimodels/QuickAdapterRegressorV3.py`
  - `quickadapter/user_data/strategies/Utils.py`
- **Out-of-scope**: keys that primarily configure **third-party behavior** (Freqtrade/FreqAI core, Freqtrade protections, Optuna), even if they are read/forwarded by our code.

### Out-of-scope examples (not inventoried in the table)

- Core Freqtrade config: `timeframe`, `max_open_trades`, `trading_mode`, `exchange.*`, etc.
- Freqtrade protections configuration (e.g. `custom_protections.*`).
- Optuna hyperopt configuration (e.g. `freqai.optuna_hyperopt.*`).
- Core FreqAI plumbing keys (e.g. `freqai.identifier`, `freqai.fit_live_predictions_candles`).

## Implementation Status

**All naming normalization changes have been implemented** with backward-compatible deprecated aliases.

### Migration mechanism

A centralized utility function `get_config_value_with_deprecated_alias()` in `Utils.py` handles all deprecated alias lookups:
- Checks for the new canonical key first
- Falls back to the deprecated key with a warning log
- Returns the default if neither key is present

This ensures backward compatibility while encouraging migration to the new names.

## Renamed Tunables

| New canonical name | Deprecated alias | Config path | Implementation status |
| --- | --- | --- | --- |
| `trade_price_target_method` | `trade_price_target` | `exit_pricing.*` | Implemented |
| `lookback_period_candles` | `lookback_period` | `reversal_confirmation.*` | Implemented |
| `decay_fraction` | `decay_ratio` | `reversal_confirmation.*` | Implemented |
| `min_natr_ratio_fraction` | `min_natr_ratio_percent` | `reversal_confirmation.*` | Implemented |
| `max_natr_ratio_fraction` | `max_natr_ratio_percent` | `reversal_confirmation.*` | Implemented |
| `window_candles` | `window` | `freqai.extrema_smoothing.*` | Implemented |
| `label_natr_multiplier` | `label_natr_ratio` | `freqai.feature_parameters.*` | Implemented |
| `min_label_natr_multiplier` | `min_label_natr_ratio` | `freqai.feature_parameters.*` | Implemented |
| `max_label_natr_multiplier` | `max_label_natr_ratio` | `freqai.feature_parameters.*` | Implemented |
| `outlier_threshold_fraction` | `threshold_outlier` | `freqai.predictions_extrema.*` | Implemented |
| `threshold_smoothing_method` | `thresholds_smoothing` | `freqai.predictions_extrema.*` | Implemented |
| `soft_extremum_alpha` | `thresholds_alpha` | `freqai.predictions_extrema.*` | Implemented |
| `keep_extrema_fraction` | `extrema_fraction` | `freqai.predictions_extrema.*` | Implemented |
| `space_fraction` | `expansion_ratio` | `freqai.optuna_hyperopt.*` | Implemented |

## Unchanged Tunables (already correct)

| Config path | Semantics | Notes |
| --- | --- | --- |
| `exit_pricing.thresholds_calibration.decline_quantile` | quantile `[0,1]` | Name is semantically correct |
| `freqai.extrema_weighting.strategy` | enum/string | OK |
| `freqai.extrema_weighting.standardization` | enum/string | OK |
| `freqai.extrema_weighting.robust_quantiles` | quantiles | OK |
| `freqai.extrema_weighting.mmad_scaling_factor` | scale factor | OK |
| `freqai.extrema_weighting.normalization` | enum/string | OK |
| `freqai.extrema_weighting.minmax_range` | numeric range | OK |
| `freqai.extrema_weighting.sigmoid_scale` | scale factor | OK |
| `freqai.extrema_weighting.softmax_temperature` | temperature | OK |
| `freqai.extrema_weighting.rank_method` | enum/string | OK |
| `freqai.extrema_weighting.gamma` | exponent | OK |
| `freqai.extrema_weighting.source_weights.<source>` | weights | OK |
| `freqai.extrema_weighting.aggregation` | enum/string | OK |
| `freqai.extrema_weighting.aggregation_normalization` | enum/string | OK |
| `freqai.extrema_smoothing.method` | enum/string | OK |
| `freqai.extrema_smoothing.beta` | Kaiser window beta | OK |
| `freqai.extrema_smoothing.polyorder` | polynomial order | OK |
| `freqai.extrema_smoothing.mode` | enum/string | OK |
| `freqai.extrema_smoothing.sigma` | Gaussian sigma | OK |
| `freqai.predictions_extrema.selection_method` | enum/string | OK |
| `feature_parameters.label_period_candles` | candles | OK |
| `feature_parameters.min_label_period_candles` | candles | OK |
| `feature_parameters.max_label_period_candles` | candles | OK |

## Naming conventions applied

1. **`_candles` suffix**: for parameters representing time periods in candle units
2. **`_fraction` suffix**: for parameters representing values in range `[0,1]` used as fractional multipliers
3. **`_multiplier` suffix**: for scaling factors that multiply another value (e.g., NATR)
4. **`_method` suffix**: for enum/string parameters selecting an algorithm variant
5. **Semantic accuracy**: names now accurately describe what the parameter represents (e.g., `space_fraction` describes the fraction of search space, not an expansion ratio)

## Backward compatibility

All deprecated aliases remain functional and emit a warning when used:

```
WARNING - Deprecated config key reversal_confirmation.lookback_period detected; use reversal_confirmation.lookback_period_candles instead
```

Users can migrate at their convenience by updating their configuration files to use the new canonical names.

## Internal variable renames (code consistency)

In addition to config key renames, the following internal variables and function parameters were also renamed for consistency:

| Old name | New name | Location | Notes |
| --- | --- | --- | --- |
| `threshold_outlier` | `outlier_threshold_fraction` | `QuickAdapterRegressorV3.predictions_extrema` | Local variable |
| `thresholds_alpha` | `soft_extremum_alpha` | `QuickAdapterRegressorV3.predictions_extrema` | Local variable |
| `extrema_fraction` | `keep_extrema_fraction` | `QuickAdapterRegressorV3.predictions_extrema` | Local variable |
| `extrema_fraction` | `keep_extrema_fraction` | `QuickAdapterRegressorV3._get_ranked_peaks()` | Function parameter |
| `extrema_fraction` | `keep_extrema_fraction` | `QuickAdapterRegressorV3._get_ranked_extrema()` | Function parameter |
| `extrema_fraction` | `keep_extrema_fraction` | `QuickAdapterRegressorV3.get_pred_min_max()` | Function parameter |
| `extrema_fraction` | `keep_extrema_fraction` | `QuickAdapterRegressorV3.soft_extremum_min_max()` | Function parameter |
| `extrema_fraction` | `keep_extrema_fraction` | `QuickAdapterRegressorV3.median_min_max()` | Function parameter |
| `extrema_fraction` | `keep_extrema_fraction` | `QuickAdapterRegressorV3.skimage_min_max()` | Function parameter |
| `_reversal_lookback_period` | `_reversal_lookback_period_candles` | `QuickAdapterV3` | Instance variable |
| `natr_ratio` | `natr_multiplier` | `Utils.zigzag()` | Function parameter |
