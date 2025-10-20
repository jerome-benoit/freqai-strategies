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

## Table of contents

- [Key Capabilities](#key-capabilities)
- [Prerequisites](#prerequisites)
- [Common Use Cases](#common-use-cases)
    - [1. Validate Reward Logic](#1-validate-reward-logic)
    - [2. Parameter Sensitivity](#2-parameter-sensitivity)
    - [3. Debug Anomalies](#3-debug-anomalies)
    - [4. Real vs Synthetic](#4-real-vs-synthetic)
- [CLI Parameters](#cli-parameters)
    - [Required Parameters](#required-parameters)
    - [Core Simulation](#core-simulation)
    - [Reward Configuration](#reward-configuration)
    - [PnL / Volatility](#pnl--volatility)
    - [Trading Environment](#trading-environment)
    - [Output & Overrides](#output--overrides)
    - [Parameter Cheat Sheet](#parameter-cheat-sheet)
    - [Exit Attenuation Kernels](#exit-attenuation-kernels)
    - [Transform Functions](#transform-functions)
    - [Skipping Feature Analysis](#skipping-feature-analysis)
    - [Reproducibility](#reproducibility)
    - [Overrides vs `--params`](#overrides-vs---params)
- [Examples](#examples)
- [Outputs](#outputs)
    - [Main Report](#main-report-statistical_analysismd)
    - [Data Exports](#data-exports)
    - [Manifest](#manifest-manifestjson)
    - [Distribution Shift Metrics](#distribution-shift-metrics)
- [Advanced Usage](#advanced-usage)
    - [Custom Parameter Testing](#custom-parameter-testing)
    - [Real Data Comparison](#real-data-comparison)
    - [Batch Analysis](#batch-analysis)
- [Testing](#testing)
    - [Run Tests](#run-tests)
    - [Coverage](#coverage)
    - [When to Run Tests](#when-to-run-tests)
    - [Focused Test Sets](#focused-test-sets)
- [Troubleshooting](#troubleshooting)
    - [No Output Files](#no-output-files)
    - [Unexpected Reward Values](#unexpected-reward-values)
    - [Slow Execution](#slow-execution)
    - [Memory Errors](#memory-errors)

## Prerequisites

Requirements: Python 3.9+, ≥4GB RAM (CPU only). Recommended venv:

```shell
cd ReforceXY/reward_space_analysis
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn pytest
```

Run:
```shell
python reward_space_analysis.py --num_samples 20000 --out_dir out
```

## Common Use Cases

### 1. Validate Reward Logic

```shell
python reward_space_analysis.py --num_samples 20000 --out_dir reward_space_outputs
```

See `statistical_analysis.md` (1–3): positive exit averages (long & short), negative invalid penalties, monotonic idle reduction, zero invariance failures.

### 2. Parameter Sensitivity

```shell
# Test different win reward factors
python reward_space_analysis.py \
    --num_samples 30000 \
    --params win_reward_factor=2.0 \
    --out_dir conservative_rewards

python reward_space_analysis.py \
    --num_samples 30000 \
    --params win_reward_factor=4.0 \
    --out_dir aggressive_rewards

# Test PBRS potential shaping
python reward_space_analysis.py \
    --num_samples 30000 \
    --params hold_potential_enabled=true potential_gamma=0.9 exit_potential_mode=progressive_release \
    --out_dir pbrs_analysis
```

Compare reward distribution & component share deltas across runs.

### 3. Debug Anomalies

```shell
# Generate detailed analysis
python reward_space_analysis.py \
    --num_samples 50000 \
    --out_dir debug_analysis
```

Focus: feature importance, shaping activation, invariance drift, extremes.

### 4. Real vs Synthetic

```shell
# First, collect real episodes (see Advanced Usage section)
# Then compare:
python reward_space_analysis.py \
    --num_samples 100000 \
    --real_episodes path/to/episode_rewards.pkl \
    --out_dir real_vs_synthetic
```

---

## CLI Parameters

### Required Parameters

None (all have defaults).

### Core Simulation

**`--num_samples`** (int, default: 20000) – Synthetic scenarios. More = better stats (slower). Recommended: 10k (quick), 50k (standard), 100k+ (deep).

**`--seed`** (int, default: 42) – Master seed (reuse for identical runs).

### Reward Configuration

**`--base_factor`** (float, default: 100.0) – Base reward scale (match environment).

**`--profit_target`** (float, default: 0.03) – Target profit (e.g. 0.03=3%) for exit reward.

**`--risk_reward_ratio`** (float, default: 1.0) – Adjusts effective profit target.

**`--max_duration_ratio`** (float, default: 2.5) – Upper multiple for sampled trade/idle durations (higher = more variety).

### PnL / Volatility

Controls synthetic PnL variance (heteroscedastic; grows with duration):

**`--pnl_base_std`** (float, default: 0.02) – Volatility floor.

**`--pnl_duration_vol_scale`** (float, default: 0.5) – Extra volatility × (duration/max_trade_duration). Higher ⇒ stronger.

### Trading Environment

**`--trading_mode`** (spot|margin|futures, default: spot) – spot: no shorts; margin/futures: shorts enabled.

**`--action_masking`** (bool, default: true) – Simulate action masking (match environment).

### Output & Overrides

**`--out_dir`** (path, default: reward_space_outputs) – Output directory (auto-created).

**`--params`** (k=v ...) – Override reward params. Example: `--params win_reward_factor=3.0 idle_penalty_scale=2.0`.

All tunables mirror `DEFAULT_MODEL_REWARD_PARAMETERS`. Flags or `--params` (wins on conflicts).

### Parameter Cheat Sheet

Core frequently tuned parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `win_reward_factor` | 2.0 | Profit overshoot multiplier |
| `pnl_factor_beta` | 0.5 | PnL amplification beta |
| `idle_penalty_scale` | 0.5 | Idle penalty scale |
| `idle_penalty_power` | 1.025 | Idle penalty exponent |
| `max_trade_duration_candles` | 128 | Trade duration cap | 
| `max_idle_duration_candles` | None | Idle duration cap; fallback 4× max trade duration |
| `hold_penalty_scale` | 0.25 | Hold penalty scale |
| `hold_penalty_power` | 1.025 | Hold penalty exponent |
| `exit_attenuation_mode` | linear | Exit attenuation kernel |
| `exit_plateau` | true | Flat region before attenuation starts |
| `exit_plateau_grace` | 1.0 | Plateau duration ratio grace |
| `exit_linear_slope` | 1.0 | Linear kernel slope |
| `exit_power_tau` | 0.5 | Tau controlling power kernel decay (0,1] |
| `exit_half_life` | 0.5 | Half-life for half_life kernel |
| `potential_gamma` | 0.95 | PBRS discount γ |
| `exit_potential_mode` | canonical | Exit potential mode |
| `efficiency_weight` | 1.0 | Efficiency contribution weight |
| `efficiency_center` | 0.5 | Efficiency pivot in [0,1] |

For full list & exact defaults see `reward_space_analysis.py` (`DEFAULT_MODEL_REWARD_PARAMETERS`).

### Exit Attenuation Kernels

Let r be the raw duration ratio and grace = `exit_plateau_grace`.

```
effective_r = 0            if exit_plateau and r <= grace
effective_r = r - grace    if exit_plateau and r >  grace
effective_r = r            if not exit_plateau
```

| Mode | Multiplier (applied to base_factor * pnl * pnl_factor * efficiency_factor) | Monotonic decreasing (Yes/No) | Notes |
|------|---------------------------------------------------------------------|-------------------------------|-------|
| legacy | step: ×1.5 if r* ≤ 1 else ×0.5 | No | Historical discontinuity retained (not smoothed) |
| sqrt | 1 / sqrt(1 + r*) | Yes | Sub-linear decay |
| linear | 1 / (1 + slope * r*) | Yes | slope = `exit_linear_slope` (≥0) |
| power | (1 + r*)^(-alpha) | Yes | alpha = -ln(tau)/ln(2), tau = `exit_power_tau` ∈ (0,1]; tau=1 ⇒ alpha=0 (flat); invalid tau ⇒ alpha=1 (default) |
| half_life | 2^(- r* / hl) | Yes | hl = `exit_half_life`; r* = hl ⇒ factor × 0.5 |

Where r* = `effective_r` above.

### Transform Functions

| Transform | Formula | Range | Characteristics | Use Case |
|-----------|---------|-------|-----------------|----------|
| `tanh` | tanh(x) | (-1, 1) | Smooth sigmoid, symmetric around 0 | Balanced PnL/duration transforms (default) |
| `softsign` | x / (1 + \|x\|) | (-1, 1) | Smoother than tanh, linear near 0 | Less aggressive saturation |
| `arctan` | (2/pi) * arctan(x) | (-1, 1) | Slower saturation than tanh | Wide dynamic range |
| `sigmoid` | 2σ(x) - 1, σ(x) = 1/(1 + e^(-x)) | (-1, 1) | Sigmoid mapped to (-1, 1) | Standard sigmoid activation |
| `asinh` | x / sqrt(1 + x^2) | (-1, 1) | Normalized asinh-like transform | Extreme outlier robustness |
| `clip` | clip(x, -1, 1) | [-1, 1] | Hard clipping at ±1 | Preserve linearity within bounds |

Invariant toggle: disable only for performance experiments (diagnostics become advisory).

### Skipping Feature Analysis

**`--skip_partial_dependence`**: skip PD curves (faster).

**`--skip_feature_analysis`**: skip model, importance, PD.

Hierarchy / precedence of skip flags:

| Scenario | `--skip_feature_analysis` | `--skip_partial_dependence` | Feature Importance | Partial Dependence | Report Section 4 |
|----------|---------------------------|-----------------------------|--------------------|-------------------|------------------|
| Default (no flags) | ✗ | ✗ | Yes | Yes | Full (R², top features, exported data) |
| PD only skipped | ✗ | ✓ | Yes | No | Full (PD line shows skipped note) |
| Feature analysis skipped | ✓ | ✗ | No | No | Marked “(skipped)” with reason(s) |
| Both flags | ✓ | ✓ | No | No | Marked “(skipped)” + note PD redundant |

Auto-skip if `num_samples < 4`.

### Reproducibility

| Component | Controlled By | Notes |
|-----------|---------------|-------|
| Sample simulation | `--seed` | Drives action sampling, PnL noise generation. |
| Statistical tests / bootstrap | `--stats_seed` (fallback `--seed`) | Local RNG; isolation prevents side-effects in user code. |
| RandomForest & permutation importance | `--seed` | Ensures identical splits and tree construction. |
| Partial dependence grids | Deterministic | Depends only on fitted model & data. |

Patterns:
```shell
# Same synthetic data, two different statistical re-analysis runs
python reward_space_analysis.py --num_samples 50000 --seed 123 --stats_seed 9001 --out_dir run_stats1
python reward_space_analysis.py --num_samples 50000 --seed 123 --stats_seed 9002 --out_dir run_stats2

# Fully reproducible end-to-end (all aspects deterministic)
python reward_space_analysis.py --num_samples 50000 --seed 777
```

### Overrides vs `--params`

Reward parameters also have individual flags:

```shell
# Direct flag style
python reward_space_analysis.py --win_reward_factor 3.0 --idle_penalty_scale 2.0 --num_samples 15000

# Equivalent using --params
python reward_space_analysis.py --params win_reward_factor=3.0 idle_penalty_scale=2.0 --num_samples 15000
```

`--params` wins on conflicts.

## Examples

```shell
# Quick test with defaults
python reward_space_analysis.py --num_samples 10000

# Full analysis with custom profit target
python reward_space_analysis.py \
    --num_samples 50000 \
    --profit_target 0.05 \
    --trading_mode futures \
    --out_dir custom_analysis

# Parameter sensitivity testing
python reward_space_analysis.py \
    --num_samples 30000 \
    --params win_reward_factor=3.0 idle_penalty_scale=1.5 \
    --out_dir sensitivity_test

# PBRS potential shaping analysis
python reward_space_analysis.py \
    --num_samples 40000 \
    --params hold_potential_enabled=true exit_potential_mode=spike_cancel potential_gamma=0.95 \
    --out_dir pbrs_test

# Real vs synthetic comparison
python reward_space_analysis.py \
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
| `reward_samples.csv`       | Raw synthetic samples for custom analysis            |
| `feature_importance.csv`   | Feature importance rankings from random forest model |
| `partial_dependence_*.csv` | Partial dependence data for key features             |
| `manifest.json`            | Runtime manifest (simulation + reward params + hash) |

### Manifest (`manifest.json`)

| Field | Type | Description |
|-------|------|-------------|
| `generated_at` | string (ISO 8601) | Timestamp of generation (not part of hash). |
| `num_samples` | int | Number of synthetic samples generated. |
| `seed` | int | Master random seed driving simulation determinism. |
| `profit_target_effective` | float | Profit target after risk/reward scaling. |
| `pvalue_adjust_method` | string | Multiple testing correction mode (`none` or `benjamini_hochberg`). |
| `parameter_adjustments` | object | Map of any automatic bound clamps (empty if none). |
| `reward_params` | object | Full resolved reward parameter set (post-validation). |
| `simulation_params` | object | All simulation inputs (num_samples, seed, volatility knobs, etc.). |
| `params_hash` | string (sha256) | Hash over ALL `simulation_params` (excluding `out_dir`, `real_episodes`) + ALL `reward_params` (lexicographically ordered). |

Two runs match iff `params_hash` identical (defaults included in hash scope).

### Distribution Shift Metrics

| Metric | Definition | Notes |
|--------|------------|-------|
| `*_kl_divergence` | KL(synthetic‖real) = Σ p_synth log(p_synth / p_real) | Asymmetric; 0 iff identical histograms (after binning). |
| `*_js_distance` | d_JS(p_synth, p_real) = √( 0.5 KL(p_synth‖m) + 0.5 KL(p_real‖m) ), m = 0.5 (p_synth + p_real) | Symmetric, bounded [0,1]; square-root of JS divergence; stable vs KL when supports differ. |
| `*_wasserstein` | 1D Earth Mover's Distance | Non-negative; same units as feature. |
| `*_ks_statistic` | KS two-sample statistic | ∈ [0,1]; higher = greater divergence. |
| `*_ks_pvalue` | KS test p-value | ∈ [0,1]; small ⇒ reject equality (at α). |

Implementation: 50-bin hist; add ε=1e-10 before normalizing; constants ⇒ zero divergence, KS p=1.0.

---

## Advanced Usage

### Custom Parameter Testing

Test reward parameter configurations:

```shell
# Test power-based exit attenuation with custom tau
python reward_space_analysis.py \
    --num_samples 25000 \
    --params exit_attenuation_mode=power exit_power_tau=0.5 efficiency_weight=0.8 \
    --out_dir custom_test

# Test aggressive hold penalties
python reward_space_analysis.py \
    --num_samples 25000 \
    --params hold_penalty_scale=0.5 \
    --out_dir aggressive_hold

# Canonical PBRS (strict invariance, additives disabled)
python reward_space_analysis.py \
    --num_samples 25000 \
    --params hold_potential_enabled=true entry_additive_enabled=true exit_additive_enabled=false exit_potential_mode=canonical \
    --out_dir pbrs_canonical

# Non-canonical PBRS (allows additives with Φ(terminal)=0, breaks invariance)
python reward_space_analysis.py \
    --num_samples 25000 \
    --params hold_potential_enabled=true entry_additive_enabled=true exit_additive_enabled=true exit_potential_mode=non_canonical \
    --out_dir pbrs_non_canonical

python reward_space_analysis.py \
    --num_samples 25000 \
    --params hold_potential_transform_pnl=sigmoid hold_potential_gain=2.0 \
    --out_dir pbrs_sigmoid_transforms
```

### Real Data Comparison

Compare with real trading episodes:

```shell
python reward_space_analysis.py \
    --num_samples 100000 \
    --real_episodes path/to/episode_rewards.pkl \
    --out_dir real_vs_synthetic
```

Shift metrics: lower is better (except p-value: higher ⇒ cannot reject equality).

### Batch Analysis

```shell
# Test multiple parameter combinations
for factor in 1.5 2.0 2.5 3.0; do
    python reward_space_analysis.py \
        --num_samples 20000 \
        --params win_reward_factor=$factor \
    --out_dir analysis_factor_$factor
done
```

---

## Testing

### Run Tests

```shell
# activate the venv first
source .venv/bin/activate
pip install pytest packaging
pytest -q
```

### Coverage

```shell
pip install pytest-cov
pytest -q --cov=. --cov-report=term-missing
pytest -q --cov=. --cov-report=html # open htmlcov/index.html
```

### When to Run Tests

- After modifying reward logic
- Before important analyses
- When results seem unexpected
- After updating dependencies or Python version
- When contributing new features (aim for >80% coverage on new code)

### Focused Test Sets

```shell
pytest -q test_reward_space_analysis.py::TestIntegration
pytest -q test_reward_space_analysis.py::TestStatisticalCoherence
pytest -q test_reward_space_analysis.py::TestRewardAlignment

```

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
