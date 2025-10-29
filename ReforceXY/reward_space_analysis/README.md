# Reward Space Analysis (ReforceXY)

Deterministic synthetic sampling with diagnostics for reward shaping, penalties, PBRS invariance.

## Key Capabilities

- Scalable synthetic scenario generation (reproducible)
- Reward component decomposition & bounds checks
- PBRS modes: canonical, non_canonical, progressive_release, spike_cancel, retain_previous
- Feature importance & optional partial dependence
- Statistical tests (hypothesis, bootstrap CIs, distribution diagnostics)
- Real vs synthetic shift metrics
- Manifest + parameter hash

## Quick Start

```shell
# Install
cd ReforceXY/reward_space_analysis
uv sync --all-groups

# Run a default analysis
uv run python reward_space_analysis.py --num_samples 20000 --out_dir out

# Run test suite (coverage ≥85% enforced)
uv run pytest
```

Minimal selective test example:

```shell
uv run pytest -m pbrs -q
```

Full test documentation: `tests/README.md`.

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Common Use Cases](#common-use-cases)
- [CLI Parameters](#cli-parameters)
  - [Simulation & Environment](#simulation--environment)
  - [Reward & Shaping](#reward--shaping)
  - [Diagnostics & Validation](#diagnostics--validation)
  - [Overrides](#overrides)
  - [Reward Parameter Cheat Sheet](#reward-parameter-cheat-sheet)
  - [Exit Attenuation Kernels](#exit-attenuation-kernels)
  - [Transform Functions](#transform-functions)
  - [Skipping Feature Analysis](#skipping-feature-analysis)
  - [Reproducibility](#reproducibility)
  - [Overrides vs --params](#overrides-vs--params)
- [Examples](#examples)
- [Outputs](#outputs)
- [Advanced Usage](#advanced-usage)
  - [Parameter Sweeps](#parameter-sweeps)
  - [PBRS Rationale](#pbrs-rationale)
  - [Real Data Comparison](#real-data-comparison)
  - [Batch Analysis](#batch-analysis)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Requirements:

- [Python 3.11+](https://www.python.org/downloads/)
- ≥4GB RAM
- [uv](https://docs.astral.sh/uv/getting-started/installation/) project manager

Setup with uv:

```shell
cd ReforceXY/reward_space_analysis
uv sync --all-groups
```

Run:

```shell
uv run python reward_space_analysis.py --num_samples 20000 --out_dir out
```

## Common Use Cases

### 1. Validate Reward Logic

```shell
uv run python reward_space_analysis.py --num_samples 20000 --out_dir reward_space_outputs
```

See `statistical_analysis.md` (1–3): positive exit averages (long & short), negative invalid penalties, monotonic idle reduction, zero invariance failures.

### 2. Parameter Sensitivity

Single-run example:

```shell
uv run python reward_space_analysis.py \
  --num_samples 30000 \
  --params win_reward_factor=4.0 idle_penalty_scale=1.5 \
  --out_dir sensitivity_test
```

Compare reward distribution & component share deltas across runs.

### 3. Debug Anomalies

```shell
uv run python reward_space_analysis.py \
  --num_samples 50000 \
  --out_dir debug_analysis
```

Focus: feature importance, shaping activation, invariance drift, extremes.

### 4. Real vs Synthetic

```shell
uv run python reward_space_analysis.py \
  --num_samples 100000 \
  --real_episodes path/to/episode_rewards.pkl \
  --out_dir real_vs_synthetic
```

Generates shift metrics for comparison (see Outputs section).

---

## CLI Parameters

### Simulation & Environment

**`--num_samples`** (int, default: 20000) – Synthetic scenarios. More = better stats (slower). Recommended: 10k (quick), 50k (standard), 100k+ (deep). (Simulation-only; not overridable via `--params`).
**`--seed`** (int, default: 42) – Master seed (reuse for identical runs). (Simulation-only).
**`--trading_mode`** (spot|margin|futures, default: spot) – spot: no shorts; margin/futures: shorts enabled. (Simulation-only).
**`--action_masking`** (bool, default: true) – Simulate environment action masking; invalid actions receive penalties only if masking disabled. (Simulation-only; not present in reward params; cannot be set via `--params`).
**`--max_duration_ratio`** (float, default: 2.5) – Upper multiple for sampled trade durations (idle derived). (Simulation-only; not in reward params; cannot be set via `--params`).
**`--pnl_base_std`** (float, default: 0.02) – Base standard deviation for synthetic PnL generation (pre-scaling). (Simulation-only).
**`--pnl_duration_vol_scale`** (float, default: 0.5) – Additional PnL volatility scale proportional to trade duration ratio. (Simulation-only).
**`--real_episodes`** (path, optional) – Episodes pickle for real vs synthetic distribution shift metrics. (Simulation-only; triggers additional outputs when provided).
**`--unrealized_pnl`** (flag, default: false) – Simulate unrealized PnL accrual during holds for shaping potential Φ. (Simulation-only; affects PBRS components).

### Reward & Shaping

**`--base_factor`** (float, default: 100.0) – Base reward scale.
**`--profit_target`** (float, default: 0.03) – Target profit (e.g. 0.03=3%). (May be overridden via `--params` though not stored in `reward_params` object.)
**`--risk_reward_ratio`** (float, default: 1.0) – Adjusts effective profit target (`profit_target * risk_reward_ratio`). (May be overridden via `--params`).
**`--win_reward_factor`** (float, default: 2.0) – Profit overshoot multiplier.
**Duration penalties**: idle / hold scales & powers shape time-cost.
**Exit attenuation**: kernel factors applied to exit duration ratio.
**Efficiency weighting**: scales efficiency contribution.

### Diagnostics & Validation

**`--check_invariants`** (bool, default: true) – Enable runtime invariant checks (diagnostics become advisory if disabled). Toggle rarely; disabling may hide reward drift or invariance violations.
**`--strict_validation`** (flag, default: true) – Enforce parameter bounds and finite checks; raises instead of silent clamp/discard when enabled.
**`--strict_diagnostics`** (flag, default: false) – Fail-fast on degenerate statistical diagnostics (zero-width CIs, undefined distribution metrics) instead of graceful fallbacks.
**`--exit_factor_threshold`** (float, default: 10000.0) – Warn if exit factor exceeds threshold.
**`--pvalue_adjust`** (none|benjamini_hochberg, default: none) – Multiple testing p-value adjustment method.
**`--bootstrap_resamples`** (int, default: 10000) – Bootstrap iterations for confidence intervals; lower for speed (e.g. 500) during smoke tests.
**`--skip_feature_analysis`** / **`--skip_partial_dependence`** – Skip feature importance or PD grids (see Skipping Feature Analysis section); influence runtime only.
**`--rf_n_jobs`** / **`--perm_n_jobs`** (int, default: -1) – Parallel worker counts for RandomForest and permutation importance (-1 = all cores).

### Overrides

**`--out_dir`** (path, default: reward_space_outputs) – Output directory (auto-created). (Simulation-only).
**`--params`** (k=v ...) – Bulk override reward params and selected hybrid scalars (`profit_target`, `risk_reward_ratio`). Conflicts: individual flags vs `--params` ⇒ `--params` wins.

### Reward Parameter Cheat Sheet

#### Core

| Parameter           | Default | Description                 |
| ------------------- | ------- | --------------------------- |
| `base_factor`       | 100.0   | Base reward scale           |
| `invalid_action`    | -2.0    | Penalty for invalid actions |
| `win_reward_factor` | 2.0     | Profit overshoot multiplier |
| `pnl_factor_beta`   | 0.5     | PnL amplification beta      |

#### Duration Penalties

| Parameter                    | Default | Description                |
| ---------------------------- | ------- | -------------------------- |
| `max_trade_duration_candles` | 128     | Trade duration cap         |
| `max_idle_duration_candles`  | None    | Fallback 4× trade duration |
| `idle_penalty_scale`         | 0.5     | Idle penalty scale         |
| `idle_penalty_power`         | 1.025   | Idle penalty exponent      |
| `hold_penalty_scale`         | 0.25    | Hold penalty scale         |
| `hold_penalty_power`         | 1.025   | Hold penalty exponent      |

#### Exit Attenuation

| Parameter               | Default | Description                    |
| ----------------------- | ------- | ------------------------------ |
| `exit_attenuation_mode` | linear  | Kernel mode                    |
| `exit_plateau`          | true    | Flat region before attenuation |
| `exit_plateau_grace`    | 1.0     | Plateau grace ratio            |
| `exit_linear_slope`     | 1.0     | Linear slope                   |
| `exit_power_tau`        | 0.5     | Power kernel tau (0,1]         |
| `exit_half_life`        | 0.5     | Half-life for half_life kernel |

#### Efficiency

| Parameter           | Default | Description                    |
| ------------------- | ------- | ------------------------------ |
| `efficiency_weight` | 1.0     | Efficiency contribution weight |
| `efficiency_center` | 0.5     | Efficiency pivot in [0,1]      |

Formula (unrealized profit normalization):
Let `max_u = max_unrealized_profit`, `min_u = min_unrealized_profit`, `range = max_u - min_u`, `ratio = (pnl - min_u)/range`. Then:

- If `pnl > 0`: `efficiency_factor = 1 + efficiency_weight * (ratio - efficiency_center)`
- If `pnl < 0`: `efficiency_factor = 1 + efficiency_weight * (efficiency_center - ratio)`
- Else: `efficiency_factor = 1`
  Final exit multiplier path: `exit_reward = pnl * exit_factor`, where `exit_factor = kernel(base_factor, duration_ratio_adjusted) * pnl_factor` and `pnl_factor` includes the efficiency_factor above.

#### Validation

| Parameter               | Default | Description                       |
| ----------------------- | ------- | --------------------------------- |
| `check_invariants`      | true    | Invariant enforcement (see above) |
| `exit_factor_threshold` | 10000.0 | Warn on excessive factor          |

#### PBRS (Potential-Based Reward Shaping)

| Parameter                | Default   | Description                        |
| ------------------------ | --------- | ---------------------------------- |
| `potential_gamma`        | 0.95      | Discount γ for shaping potential Φ |
| `exit_potential_mode`    | canonical | Potential release mode             |
| `exit_potential_decay`   | 0.5       | Decay for progressive_release      |
| `hold_potential_enabled` | true      | Enable hold potential Φ            |

PBRS invariance holds when: `exit_potential_mode=canonical` AND `entry_additive_enabled=false` AND `exit_additive_enabled=false`. Under this condition the algorithm enforces zero-sum shaping: if the summed shaping term deviates by more than 1e-6 (`PBRS_INVARIANCE_TOL`), a uniform drift correction subtracts the mean shaping offset across invariant samples.

#### Hold Potential Transforms

| Parameter                           | Default | Description          |
| ----------------------------------- | ------- | -------------------- |
| `hold_potential_scale`              | 1.0     | Hold potential scale |
| `hold_potential_gain`               | 1.0     | Gain multiplier      |
| `hold_potential_transform_pnl`      | tanh    | PnL transform        |
| `hold_potential_transform_duration` | tanh    | Duration transform   |

#### Entry Additive (Optional)

| Parameter                           | Default | Description           |
| ----------------------------------- | ------- | --------------------- |
| `entry_additive_enabled`            | false   | Enable entry additive |
| `entry_additive_scale`              | 1.0     | Scale                 |
| `entry_additive_gain`               | 1.0     | Gain                  |
| `entry_additive_transform_pnl`      | tanh    | PnL transform         |
| `entry_additive_transform_duration` | tanh    | Duration transform    |

#### Exit Additive (Optional)

| Parameter                          | Default | Description          |
| ---------------------------------- | ------- | -------------------- |
| `exit_additive_enabled`            | false   | Enable exit additive |
| `exit_additive_scale`              | 1.0     | Scale                |
| `exit_additive_gain`               | 1.0     | Gain                 |
| `exit_additive_transform_pnl`      | tanh    | PnL transform        |
| `exit_additive_transform_duration` | tanh    | Duration transform   |

### Exit Attenuation Kernels

r = duration ratio and grace = `exit_plateau_grace`.

```
r* = 0            if exit_plateau and r <= grace
r* = r - grace    if exit_plateau and r >  grace
r* = r            if not exit_plateau
```

| Mode      | Multiplier (applied to base*factor * pnl \_ pnl_factor \* efficiency_factor) | Monotonic | Notes                                       |
| --------- | ---------------------------------------------------------------------------- | --------- | ------------------------------------------- |
| legacy    | step: ×1.5 if r\* ≤ 1 else ×0.5                                              | No        | Non-monotonic legacy mode (not recommended) |
| sqrt      | 1 / sqrt(1 + r\*)                                                            | Yes       | Sub-linear decay                            |
| linear    | 1 / (1 + slope _ r_)                                                         | Yes       | Slope = `exit_linear_slope`                 |
| power     | (1 + r\*)^(-alpha)                                                           | Yes       | alpha = -ln(tau)/ln(2); tau=1 ⇒ alpha=0     |
| half_life | 2^(- r\* / hl)                                                               | Yes       | hl = `exit_half_life`; r\*=hl ⇒ factor ×0.5 |

### Transform Functions

| Transform  | Formula            | Range   | Characteristics   | Use Case                      |
| ---------- | ------------------ | ------- | ----------------- | ----------------------------- |
| `tanh`     | tanh(x)            | (-1, 1) | Smooth sigmoid    | Balanced transforms (default) |
| `softsign` | x / (1 + \|x\|)    | (-1, 1) | Linear near 0     | Less aggressive saturation    |
| `arctan`   | (2/π) \* arctan(x) | (-1, 1) | Slower saturation | Wide dynamic range            |
| `sigmoid`  | 2σ(x) - 1          | (-1, 1) | Standard sigmoid  | Generic shaping               |
| `asinh`    | x / sqrt(1 + x^2)  | (-1, 1) | Outlier robust    | Extreme stability             |
| `clip`     | clip(x, -1, 1)     | [-1, 1] | Hard clipping     | Preserve linearity            |

### Skipping Feature Analysis

Flags hierarchy:
| Scenario | `--skip_feature_analysis` | `--skip_partial_dependence` | Feature Importance | Partial Dependence | Report Section 4 |
|----------|---------------------------|-----------------------------|--------------------|-------------------|------------------|
| Default | ✗ | ✗ | Yes | Yes | Full |
| PD skipped | ✗ | ✓ | Yes | No | PD note |
| Feature analysis skipped | ✓ | ✗ | No | No | Marked “(skipped)” |
| Both skipped | ✓ | ✓ | No | No | Marked “(skipped)” |
Auto-skip if `num_samples < 4`.

### Reproducibility

| Component                             | Controlled By                      | Notes                               |
| ------------------------------------- | ---------------------------------- | ----------------------------------- |
| Sample simulation                     | `--seed`                           | Drives action sampling & PnL noise  |
| Statistical tests / bootstrap         | `--stats_seed` (fallback `--seed`) | Isolated RNG                        |
| RandomForest & permutation importance | `--seed`                           | Identical splits and trees          |
| Partial dependence grids              | Deterministic                      | Depends only on fitted model & data |

Patterns:

```shell
uv run python reward_space_analysis.py --num_samples 50000 --seed 123 --stats_seed 9001 --out_dir run_stats1
uv run python reward_space_analysis.py --num_samples 50000 --seed 123 --stats_seed 9002 --out_dir run_stats2
# Fully deterministic
uv run python reward_space_analysis.py --num_samples 50000 --seed 777
```

### Overrides vs --params

Direct flags and `--params` produce identical outcomes; conflicts resolved by bulk `--params` values.

```shell
uv run python reward_space_analysis.py --win_reward_factor 3.0 --idle_penalty_scale 2.0 --num_samples 15000
uv run python reward_space_analysis.py --params win_reward_factor=3.0 idle_penalty_scale=2.0 --num_samples 15000
```

`--params` wins on conflicts.
Simulation-only keys (not allowed in `--params`): `num_samples`, `seed`, `trading_mode`, `action_masking`, `max_duration_ratio`, `out_dir`, `stats_seed`, `pnl_base_std`, `pnl_duration_vol_scale`, `real_episodes`, `unrealized_pnl`, `strict_diagnostics`, `strict_validation`, `bootstrap_resamples`, `skip_feature_analysis`, `skip_partial_dependence`, `rf_n_jobs`, `perm_n_jobs`, `pvalue_adjust`. Hybrid override keys allowed in `--params`: `profit_target`, `risk_reward_ratio`. Reward parameter keys (tunable via either direct flag or `--params`) correspond to those listed under Cheat Sheet, Exit Attenuation, Efficiency, Validation, PBRS, Hold/Entry/Exit additive transforms.

## Examples

```shell
# Quick test with defaults
uv run python reward_space_analysis.py --num_samples 10000
# Full analysis with custom profit target
uv run python reward_space_analysis.py \
  --num_samples 50000 \
  --profit_target 0.05 \
  --trading_mode futures \
  --bootstrap_resamples 5000 \
  --out_dir custom_analysis
# PBRS potential shaping analysis
uv run python reward_space_analysis.py \
  --num_samples 40000 \
  --params hold_potential_enabled=true exit_potential_mode=spike_cancel potential_gamma=0.95 \
  --out_dir pbrs_test
# Real vs synthetic comparison (see Common Use Cases #4)
uv run python reward_space_analysis.py \
  --num_samples 100000 \
  --real_episodes path/to/episode_rewards.pkl \
  --out_dir validation
```

---

## Outputs

### Main Report (`statistical_analysis.md`)

Includes: global stats, representativity, component + PBRS analysis, feature importance/PD, statistical validation (tests, CIs, diagnostics), optional shift metrics, summary.

### Data Exports

| File                       | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| `reward_samples.csv`       | Raw synthetic samples                                |
| `feature_importance.csv`   | Feature importance rankings                          |
| `partial_dependence_*.csv` | Partial dependence data                              |
| `manifest.json`            | Runtime manifest (simulation + reward params + hash) |

### Manifest (`manifest.json`)

| Field                     | Type              | Description                           |
| ------------------------- | ----------------- | ------------------------------------- |
| `generated_at`            | string (ISO 8601) | Generation timestamp (not hashed)     |
| `num_samples`             | int               | Synthetic samples count               |
| `seed`                    | int               | Master random seed                    |
| `profit_target_effective` | float             | Effective profit target after scaling |
| `pvalue_adjust_method`    | string            | Multiple testing correction mode      |
| `parameter_adjustments`   | object            | Bound clamp adjustments (if any)      |
| `reward_params`           | object            | Final reward params                   |
| `simulation_params`       | object            | All simulation inputs                 |
| `params_hash`             | string (sha256)   | Deterministic run hash                |

Two runs match iff `params_hash` identical.

### Distribution Shift Metrics

| Metric            | Definition                            | Notes                         |
| ----------------- | ------------------------------------- | ----------------------------- |
| `*_kl_divergence` | KL(synth‖real) = Σ p_s log(p_s / p_r) | 0 ⇒ identical histograms      |
| `*_js_distance`   | √(0.5 KL(p_s‖m) + 0.5 KL(p_r‖m))      | Symmetric, [0,1]              |
| `*_wasserstein`   | 1D Earth Mover's Distance             | Units of feature              |
| `*_ks_statistic`  | KS two-sample statistic               | [0,1]; higher ⇒ divergence    |
| `*_ks_pvalue`     | KS test p-value                       | High ⇒ cannot reject equality |

Implementation: 50-bin hist; add ε=1e-10; constants ⇒ zero divergence & KS p=1.0.

---

## Advanced Usage

### Parameter Sweeps

Loop multiple values:

```shell
for factor in 1.5 2.0 2.5 3.0; do
  uv run python reward_space_analysis.py \
    --num_samples 20000 \
    --params win_reward_factor=$factor \
    --out_dir analysis_factor_$factor
done
```

Combine with other overrides cautiously; use distinct `out_dir` per configuration.

### PBRS Rationale

Canonical mode seeks near zero-sum shaping (Φ terminal ≈ 0) ensuring invariance: reward differences reflect environment performance, not potential leakage. Non-canonical modes or additives (entry/exit) trade strict invariance for potential extra signal shaping. Progressive release & spike cancel adjust temporal release of Φ. Choose canonical for theory alignment; use non-canonical or additives only when empirical gain outweighs invariance guarantees. Symbol Φ denotes shaping potential. See invariance condition and drift correction mechanics under PBRS section.

### Real Data Comparison

```shell
uv run python reward_space_analysis.py \
  --num_samples 100000 \
  --real_episodes path/to/episode_rewards.pkl \
  --out_dir real_vs_synthetic
```

Shift metrics: lower divergence preferred (except p-value: higher ⇒ cannot reject equality).

### Batch Analysis

(Alternate sweep variant)

```shell
while read target; do
  uv run python reward_space_analysis.py \
    --num_samples 30000 \
    --params profit_target=$target \
    --out_dir pt_${target}
done <<EOF
0.02
0.03
0.05
EOF
```

---

## Testing

Quick validation:

```shell
uv run pytest
```

Selective example:

```shell
uv run pytest -m pbrs -q
```

Coverage threshold enforced: 85% (`--cov-fail-under=85` in `pyproject.toml`). Full coverage, invariants, markers, smoke policy, and maintenance workflow: `tests/README.md`.

---

## Troubleshooting

### No Output Files

Check permissions, disk space, working directory.

### Unexpected Reward Values

Run tests; inspect overrides; confirm trading mode, PBRS settings, clamps.

### Slow Execution

Lower samples; skip PD/feature analysis; reduce resamples; ensure SSD.

### Memory Errors

Reduce samples; ensure 64‑bit Python; batch processing; add RAM/swap.
