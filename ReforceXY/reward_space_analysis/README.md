# Reward Space Analysis (ReforceXY)

This tool is the analytical clone of ReforceXY's pair-local economic reward. It
generates deterministic synthetic trajectories, decomposes every transition,
checks liquidation-value invariants, and produces statistical diagnostics.

It does not validate `MyRLEnv`, MaskablePPO, FreqAI, Freqtrade's order engine,
or live portfolio performance. Those require separate local runtime and dry-run
validation.

The active runtime contract is dry/live only, spot, `stake_amount="unlimited"`,
and `add_state_info=true`. Stateful FreqAI backtesting is not supported by the
current FreqAI API. The synthetic `trading_mode` option remains useful for
analytical stress generation, but it must not be presented as runtime-parity
validation.

## Reward contract

The base reward is the scaled change in a fee-aware liquidation value:

```text
reward_economic = base_factor * log(
    reward_liquidation_value / previous_liquidation_value
)
```

The transition table is:

| Transition | `reward_liquidation_value` | `next_liquidation_value` |
| --- | ---: | ---: |
| Neutral self-loop | `1` | `1` |
| Valid long/short entry | `1 + entry_pnl_net` | same value |
| Hold an open position | `1 + current_pnl_net` | same value |
| Valid exit | `1 + current_pnl_net` | `1` |
| Invalid action while in position | `1 + current_pnl_net` | same value |

An invalid action may additionally receive `invalid_action` when action masking
is disabled. With the normal masked policy it should not occur.

For a completed trade, the economic components telescope:

```text
sum(reward_economic) / base_factor = log(1 + realized_pnl_net)
```

Across trades, this is a synthetic pair-local compounded return. It is not the
global Freqtrade wallet equity.

## Cost accounting

`fee_rate` is the single analytical fee input. Its default, `0.0015`, mirrors
the fallback in Freqtrade `BaseEnvironment` 2026.4:

```text
add_entry_fee(price) = price * (1 + fee_rate)
add_exit_fee(price)  = price / (1 + fee_rate)
```

Consequently the entry liquidation value already contains the complete
simulated round-trip fee. The reward never subtracts another fee at entry or
exit.

Included:

- simulated entry fee;
- simulated exit fee.

Not modelled:

- spread;
- slippage;
- funding;
- latency;
- market impact.

This remains an analytical assumption. Exact parity requires passing the
effective fee used by the actual Freqtrade environment, which may come from the
exchange or an explicit Freqtrade configuration. The manifest records the
assumption and this parity requirement explicitly. Cost stress tests must alter
the unified fee assumption or be performed in the external execution harness;
they must not add another fee term to the reward.

## PBRS

Potential-based reward shaping is off by default:

```text
hold_potential_enabled = false
```

Only canonical PBRS is supported:

```text
F(s, a, s') = gamma * Phi(s') - Phi(s)
```

A true terminal state is economic ruin. Its liquidation value is clamped to
`1e-12`, the potential is released to zero, and the synthetic trajectory stops.
A normal sample-limit or dataset end is a Gymnasium truncation: its potential is
preserved so the value function can bootstrap. Entry and exit additive rewards
are always suppressed. Strict mode rejects non-canonical modes and enabled
additives; relaxed mode normalizes the mode to `canonical` and disables the
additives explicitly.

The report never uses the raw sum `sum(reward_shaping)` to classify invariance.
For `gamma < 1`, canonical PBRS telescopes only after discounting:

```text
sum(gamma^t * F_t) = -Phi_0 + gamma^T * Phi_T
```

The discounted sum, boundary term, and residual are diagnostics. Classification
requires both a canonical/no-additive configuration and a finite term-by-term
`reward_shaping - reward_pbrs_delta` check within tolerance. If that correction
column is absent, the result is `Unverified`, irrespective of any raw sum. On a
true termination `Phi_T` must be zero; on a bootstrapable truncation it is
preserved and the boundary term may be non-zero.

`profit_aim`, `rr`/`risk_reward_ratio`, and duration parameters remain available
only for the optional observable potential and FreqAI constructor compatibility.
They do not alter `reward_economic`.

## Deprecated parameters

The following legacy reward parameters may still be accepted during migration,
but they are diagnostics/no-ops for the economic component:

- idle and hold penalty parameters;
- exit attenuation parameters;
- MFE/MAE efficiency parameters;
- target amplification parameters;
- entry and exit additive parameters;
- asymmetric `entry_fee_rate` and `exit_fee_rate`.

New analyses must use `fee_rate`. Legacy asymmetric fees cannot reproduce
Freqtrade's accounting.

## Installation and quick start

```shell
cd ReforceXY/reward_space_analysis
uv sync --all-groups
uv run python reward_space_analysis.py \
  --num_samples 20000 \
  --params fee_rate=0.001 \
  --out_dir reward_space_outputs
```

For a faster smoke run:

```shell
uv run python reward_space_analysis.py \
  --num_samples 1000 \
  --bootstrap_resamples 200 \
  --skip_feature_analysis \
  --out_dir reward_space_smoke
```

## Relevant parameters

### Reward

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `base_factor` | `100.0` | Positive scale for log returns |
| `fee_rate` | `0.0015` | Analytical BaseEnvironment fallback fee |
| `invalid_action` | `-2.0` | Additional unmasked invalid-action penalty |
| `hold_potential_enabled` | `false` | Enable canonical PBRS |
| `potential_gamma` | `0.95` | PBRS discount factor |
| `exit_potential_mode` | `canonical` | Only supported PBRS mode |

`base_factor` must be strictly positive. A positive scaling does not change the
ordering of unshaped economic returns.

### Simulation

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `--num_samples` | `20000` | Number of trajectory transitions |
| `--seed` | `42` | Market/action RNG seed |
| `--trading_mode` | `spot` | Spot disables short actions |
| `--max_duration_ratio` | `2.5` | Synthetic duration cap multiplier |
| `--pnl_base_std` | `0.02` | Base open-return volatility |
| `--pnl_duration_vol_scale` | `0.5` | Duration-dependent volatility scale |
| `--action_masking` | enabled | Simulate masked action selection |
| `--real_episodes` | unset | Optional real transition pickle |
| `--stats_seed` | `--seed` | Separate statistics seed |

Explicit `--params KEY=VALUE` overrides direct tunable flags.
When masking is disabled, the generator samples invalid actions with a fixed 5%
stress probability so the separate invalid-action penalty remains observable.

## Output schema

### `reward_samples.csv`

The economic contract is observable through:

| Column | Meaning |
| --- | --- |
| `reward` | Economic + invalid penalty + canonical PBRS |
| `reward_economic` | Scaled net log-liquidation return |
| `reward_invalid` | Separate invalid-action penalty |
| `reward_shaping` | Canonical PBRS delta |
| `previous_liquidation_value` | Stored value before the transition |
| `reward_liquidation_value` | Value used to calculate this reward |
| `next_liquidation_value` | Value carried to the next transition |
| `economic_log_return` | `reward_economic / base_factor` |
| `cumulative_pair_log_return` | Sum of economic log returns |
| `synthetic_pair_equity` | Exponential of the cumulative log return |
| `economic_ruin` | Liquidation value reached zero before clamping |
| `terminated` | True only for economic ruin |
| `truncated` | True when the requested sample limit ends normally |
| `drawdown_breached` | Unavailable (`NaN`) in the pair-local simulator |
| `drawdown_breached_available` | Always false for analytical samples |

The legacy `reward_idle`, `reward_hold`, `reward_entry_additive`, and
`reward_exit_additive` columns remain zero-valued schema compatibility fields.
`reward_exit` is a diagnostic view of the economic component on valid exits; it
is not a separate reward term.

### `manifest.json`

The manifest contains:

- `reward_contract`: versioned name, exact formula, and PBRS status;
- `cost_accounting`: unified fee and explicit included/excluded costs;
- `episode_boundaries`: termination, truncation, PBRS, and drawdown semantics;
- `compatibility_only`: legacy target and deprecated parameter inventory;
- final reward and simulation parameters;
- parameter adjustments and deterministic parameter hash.

The generation timestamp is not included in the hash.

### Other artifacts

- `statistical_analysis.md`: reward, component, PBRS, and statistical report;
- `feature_importance.csv`: optional model-based diagnostic;
- `partial_dependence_*.csv`: optional partial-dependence diagnostics.

Feature importance describes the synthetic generator. It is not evidence that a
feature is predictive in live trading.

## Validation invariants

The simulator fails fast when:

- actions and positions are incompatible;
- neutral states contain a trade PnL;
- carried liquidation values are non-finite or non-positive; a newly reached
  economic-ruin value is first clamped to `1e-12` by the explicit terminal path;
- a liquidation value is not carried to the next transition;
- `reward_economic` differs from the log-liquidation formula;
- total reward differs from economic + invalid + PBRS components.
- termination does not correspond exactly to economic ruin;
- a trajectory continues after termination or lacks a final truncation.

The exact runtime comparison against `MyRLEnv` belongs in a local harness outside
the repository. Existing analytical tests can be run with:

```shell
uv run pytest
```

See [tests/README.md](./tests/README.md) for the existing suite. Its success
validates this analytical implementation only, not the real FreqAI environment.

## Reproducibility

Use identical `--seed`, `--stats_seed`, inputs, and reward parameters. Retain the
manifest and input data hash for every comparison. Parameter sweeps must use a
different output directory per cell and the same seed set across candidates.

## Troubleshooting

- Economic ruin: the value is clamped to `1e-12`, terminal PBRS is applied, and
  the trajectory stops. Inspect the price/PnL path before interpreting it.
- Unexpected fee loss at entry: the liquidation mark pre-charges the simulated
  round-trip fee by design.
- Non-canonical PBRS error: remove the legacy mode and use `canonical`.
- Slow run: reduce samples/bootstrap resamples or skip feature analysis.
