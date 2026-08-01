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

The following causal transition clock belongs to `MyRLEnv`
training/evaluation and to the causal analytical simulator:

```text
state s[t-1] : features, position, PnL, and raw candle duration through close[t-1]
action       : selected from s[t-1]
fill         : open[t]
mark         : close[t] while open; open[t] for an exit
state s[t]   : features, position, PnL, and raw candle duration through close[t]
```

The transition table is:

| Transition | Fill/mark used by reward | `next_liquidation_value` |
| --- | --- | ---: |
| Neutral self-loop | `1` | `1` |
| Valid long/short entry | `1 + pnl_net(entry open[t], close[t])` | same value |
| Hold an open position | `1 + pnl_net(entry open, close[t])` | same value |
| Valid exit | `1 + pnl_net(entry open, exit open[t])` | `1` |
| Invalid action while in position | `1 + pnl_net(entry open, close[t])` | same value |

This is not a universal dry/live pricing clock. During dry/live prediction,
inherited Freqtrade state uses the current authoritative exit-side rate. That
rate is asynchronous and is not guaranteed to equal `close[t]`.

An invalid action may additionally receive `invalid_action` when action masking
is disabled. `MaskablePPO` excludes invalid actions during training, evaluation,
and inference. Non-maskable algorithms may select invalid actions and receive
the separate invalid-action penalty; they are never supplied action masks.

For a completed trade, the economic components telescope:

```text
sum(reward_economic) / base_factor = log(1 + realized_pnl_net)
```

Across trades, this is a synthetic pair-local compounded return. It is not the
global Freqtrade wallet equity.

## Cost accounting

`fee_rate` is the single analytical fee input. Its default, `0.0015`, mirrors
the spot, leverage-one `LocalTrade` used by the runtime:

```text
long:
  open_trade_value  = entry_open * (1 + fee_rate)
  close_trade_value = mark * (1 - fee_rate)
  pnl_net            = close_trade_value / open_trade_value - 1

short:
  open_trade_value  = entry_open * (1 - fee_rate)
  close_trade_value = mark * (1 + fee_rate)
  pnl_net            = 1 - close_trade_value / open_trade_value
```

Consequently the entry liquidation value already contains the complete
simulated round-trip fee. The reward never subtracts another fee at entry or
exit. `LocalTrade.calc_profit_ratio()` rounds the resulting PnL ratio to eight
decimal places, which the analytical mirror reproduces.

The causal simulator carries that native fee-aware PnL without an artificial
clip. Synthetic trajectories can therefore contain moves beyond `±15%`; their
PnL fields must remain finite.

`MyRLEnv` creates one unregistered `LocalTrade` at entry, reuses it for every
mark, and discards it at exit. It never calls `add_bt_trade()` and therefore
does not add the object to Freqtrade's backtesting registries or a database. In
the recorded benchmark under the pinned 2026.6 runtime, reusing a trade took
approximately `4.31 µs` per mark over 200,000 calls, versus `10.13 µs` when
recreating it for every mark (`2.35x` slower).

The earlier BaseEnvironment approximation used `p / (1 + f)` instead of the
native close value `p * (1 - f)`. Their absolute price difference is
`p * f² / (1 + f)`; relative to `p * (1 - f)`, the difference is
`f² / (1 - f²)`: approximately `1.000001 ppm` at `f=0.001`,
`2.250005 ppm` at `f=0.0015`, and `100.010001 ppm` at `f=0.01`.
That approximation remains available only in the non-promotable analytical
PBRS compatibility mode.

Included:

- simulated entry fee;
- simulated exit fee.

Not modelled:

- spread;
- slippage;
- funding;
- latency;
- market impact.

This remains an analytical assumption. At runtime, `pack_env_dict(pair)`
resolves the exchange fee for each pair unless an explicit top-level Freqtrade
`fee` configuration is present; configured `0.0` is also authoritative. Exact
analytical parity requires passing that effective fee to `fee_rate`. The
manifest records the assumption and this parity requirement explicitly. Cost
stress tests must alter the unified fee assumption or be performed in the
external execution harness; they must not add another fee term to the reward.

## Reward shaping and promotion

The promotable ReforceXY runtime fails closed unless potential-based reward
shaping and entry/exit additive bonuses are all disabled:

```text
hold_potential_enabled = false
entry_additive_enabled = false
exit_additive_enabled = false
```

The analytical tool retains canonical PBRS as a compatibility-only research
mode:

```text
F(s, a, s') = gamma * Phi(s') - Phi(s)
```

Any analytical run with shaping enabled is explicitly non-promotable and does
not represent the active runtime reward.

A true terminal state is economic ruin. Its liquidation value is clamped to
`1e-12`, and the synthetic trajectory stops. A normal sample-limit or dataset
end is a Gymnasium truncation so the value function can bootstrap. In the
compatibility-only analytical PBRS mode, potential is released on termination
and preserved on truncation. Entry and exit additive rewards are always
suppressed. Strict mode rejects non-canonical modes and enabled additives;
relaxed mode normalizes the mode to `canonical` and disables the additives
explicitly.

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

The legacy `--unrealized_pnl` option and `unrealized_pnl` parameter override
remain accepted only as deprecated compatibility aliases. They emit a warning,
have no effect, and are omitted from operational reward and simulation
metadata. Analytical trajectories are always fee-aware marked-to-liquidation.

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
| `fee_rate` | `0.0015` | Analytical spot `LocalTrade` fee assumption |
| `invalid_action` | `-2.0` | Additional unmasked invalid-action penalty |
| `hold_potential_enabled` | `false` | Enable non-promotable analytical PBRS |
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
| `--pnl_base_std` | `0.02` | Base transition volatility split across gap and intrabar returns |
| `--pnl_duration_vol_scale` | `0.5` | Duration-dependent volatility scale |
| `--action_masking` | enabled | Simulate masked action selection |
| `--real_episodes` | unset | Optional real transition pickle |
| `--stats_seed` | `--seed` | Separate statistics seed |

Explicit `--params KEY=VALUE` overrides direct tunable flags.
When masking is disabled, the generator samples invalid actions with a fixed 5%
stress probability so the separate invalid-action penalty remains observable.
Durations remain raw candle-row counts; they are not normalized. With missing
OHLC candles, this differs from live UTC elapsed duration and is a known
runtime-parity limitation.

Direct compatibility calls that construct `RewardContext` without `next_pnl`
retain the legacy analytical fallback and are non-promotable. The causal
simulator always supplies `next_pnl` and `next_trade_duration`; only that path
duplicates the `MyRLEnv` training/evaluation transition clock. It does not
duplicate Freqtrade's asynchronous dry/live pricing clock or establish
same-close numerical observation parity.

## Output schema

### `reward_samples.csv`

The economic contract is observable through:

| Column | Meaning |
| --- | --- |
| `reward` | Economic + invalid penalty + canonical PBRS |
| `reward_economic` | Scaled net log-liquidation return |
| `reward_invalid` | Separate invalid-action penalty |
| `reward_shaping` | Canonical PBRS delta |
| `pnl` | Fee-aware PnL observable at the previous close |
| `next_pnl` | Fee-aware PnL at the causal fill/mark |
| `previous_close` | Close used by the state that selected the action |
| `fill_open` | Next open where the action is filled |
| `mark_close` | Close used for a position retained after the fill |
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
- `pnl` or `next_pnl` is non-finite;
- neutral states contain a trade PnL;
- carried liquidation values are non-finite or non-positive; a newly reached
  economic-ruin value is first clamped to `1e-12` by the explicit terminal path;
- a liquidation value is not carried to the next transition;
- `reward_economic` differs from the log-liquidation formula;
- total reward differs from economic + invalid + PBRS components.
- termination does not correspond exactly to economic ruin;
- a trajectory continues after termination or lacks a final truncation.

Shapiro-Wilk and Anderson-Darling normality tests are undefined for constant
distributions. Relaxed diagnostics report them as
`N/A (constant distribution)` without invoking SciPy; `--strict_diagnostics`
fails fast instead.

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
