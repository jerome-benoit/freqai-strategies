# ReforceXY

ReforceXY trains a reinforcement-learning policy. RLAgentStrategy translates its
five actions into Freqtrade entry/exit signals; episode reward is not portfolio
profitability.

## Contents

- [Quick start](#quick-start)
- [Actions and safety](#actions-and-safety)
- [Supplied profile and effective configuration](#supplied-profile-and-effective-configuration)
- [Model and policy compatibility](#model-and-policy-compatibility)
- [Training and HPO](#training-and-hpo)
- [Continual learning](#continual-learning)
- [Live inference and backtesting](#live-inference-and-backtesting)
- [Configuration reference](#configuration-reference)
- [Reward and portfolio accounting](#reward-and-portfolio-accounting)

## Quick start

From the repository root:

```shell
cd ReforceXY
cp user_data/config-template.json user_data/config.json
```

Review exchange, pairs, sizing and `freqai` in `user_data/config.json`. Keep
dry-run enabled while checking behavior. Add private exchange credentials only
for operations that require them. Review the timezone in
[docker-compose.yml](docker-compose.yml) and the
[API/security and maintenance guidance](../README.md#start-safely).

The supplied profile enables HPO, eight training environments and state
observations. For a smaller first run, disable `freqai.rl_config_optuna.enabled`,
use `n_envs=1` and `multiprocessing=false`; these are intentional overrides,
not the shipped profile. For native backtests, also follow the state-observation
restrictions below.

```shell
docker compose up -d --build
```

The build follows the moving `stable_freqairl` base. Record resolved digests and
dependency versions for reproducible evaluations.

## Actions and safety

| Predicted action | Value | Strategy decision when `do_predict=1`                    |
| ---------------- | ----- | -------------------------------------------------------- |
| Neutral          | 0     | No new entry or ordinary exit signal.                    |
| Long_enter       | 1     | Long entry.                                              |
| Long_exit        | 2     | Long exit.                                               |
| Short_enter      | 3     | Short entry; shorting is allowed only in margin/futures. |
| Short_exit       | 4     | Short exit.                                              |

`do_predict=0` does not generate these ordinary signals. On the final candle,
`do_predict=2` (expired model) requests exits only for the current pair's open
positions, using each position's side independently of the predicted action.
Other pairs cannot generate its exit signals. Freqtrade capacity, protections,
order handling and exchange execution still determine whether an order is
placed or filled.

Only MaskablePPO supports action masking in this integration.
`inference_masking=false` disables its prediction-time mask, not its training
or evaluation masks. Live masks use the real position even without state
observations. Do not set `rl_config.action_masking` independently: it cannot
add masking support to another algorithm and can make reward invalid-action
handling inconsistent with the selected model.

Reward `freqai.rl_config.model_reward_parameters.max_trade_duration_candles`
and `freqai.rl_config.model_reward_parameters.max_idle_duration_candles`
normalize duration-dependent reward components; they are not live exit timers.
`max_training_drawdown_pct` limits training episodes, not exchange losses.
`RLAgentStrategy` does not turn these training settings into a live stoploss.
Review strategy protections, position size and execution risk separately.
The supplied exchange configuration sets `stoploss=-0.99`; the training
drawdown value 0.02 is not a replacement for reviewing that live risk limit.

## Supplied profile and effective configuration

[config-template.json](user_data/config-template.json) supplies dry-run, spot,
5m, MaskablePPO/MlpPolicy, 25 training cycles, `n_envs=8`, multiprocessing,
`frame_stacking=2`, `add_state_info=true`, holdout 0.333 and HPO enabled for
100 trials with no timeout. Runtime fallbacks differ: one environment, no
multiprocessing or stacking, HPO disabled and optional action statistics disabled.

[RLAgentStrategy](user_data/strategies/RLAgentStrategy.py) adds log returns,
volume, calendar and raw OHLC features, but does not implement a period-dependent
`feature_engineering_expand_all()` callback. The supplied
`freqai.feature_parameters.indicator_periods_candles=[8,16,32]` therefore adds
no period-dependent indicators.

Explicit configuration overrides schema defaults and constructor fallbacks. The
tables distinguish those defaults and requirements from the supplied profile;
inspect normalized startup logs. Set `test_size`
explicitly: the local HPO enablement check falls back to 0.1 when omitted, but
the native split is 25% when both `test_size` and `train_size` are omitted.
Do not treat the HPO gate fallback as an effective data-split default.

## Model and policy compatibility

These five algorithms are supported by the application's HPO path and decision
workflow. Policy names are not automatically changed when `model_type` changes.

| Model        | Policy for the numerical observations used here | Masking                                          |
| ------------ | ----------------------------------------------- | ------------------------------------------------ |
| PPO          | MlpPolicy                                       | No                                               |
| MaskablePPO  | MlpPolicy                                       | Training/evaluation; inference toggle available. |
| RecurrentPPO | MlpLstmPolicy                                   | No                                               |
| DQN          | MlpPolicy                                       | No                                               |
| QRDQN        | MlpPolicy                                       | No                                               |

Changing only the supplied model name to RecurrentPPO leaves MlpPolicy in place
and fails construction. Use MlpLstmPolicy and prefer `frame_stacking=0`:
recurrent memory plus stacked frames is permitted but warns about redundancy.
For LSTM policy kwargs, `shared_lstm=true` requires
`enable_critic_lstm=false`; plain fits forward these settings to the SDK.
Use a new `freqai.identifier` for a different policy/observation architecture.

A backtest-safe **override fragment**, to merge into a complete Freqtrade
configuration (not a standalone exchange config):

```json
{
  "leverage": 1,
  "freqai": {
    "enabled": true,
    "identifier": "ReforceXY-RecurrentPPO",
    "continual_learning": false,
    "data_split_parameters": {
      "test_size": 0.333,
      "shuffle": false
    },
    "model_training_parameters": {
      "device": "cpu",
      "policy_kwargs": {
        "net_arch": "small",
        "shared_lstm": false,
        "enable_critic_lstm": true
      }
    },
    "rl_config": {
      "model_type": "RecurrentPPO",
      "policy_type": "MlpLstmPolicy",
      "train_cycles": 25,
      "n_envs": 1,
      "n_eval_envs": 1,
      "multiprocessing": false,
      "eval_multiprocessing": false,
      "frame_stacking": 0,
      "add_state_info": false,
      "model_reward_parameters": {
        "rr": 2,
        "profit_aim": 0.025,
        "hold_potential_enabled": false
      },
      "check_envs": true,
      "plot_new_best": false
    },
    "rl_config_optuna": {
      "enabled": false
    }
  }
}
```

## Training and HPO

HPO runs only when `freqai.enabled=true`,
`freqai.rl_config_optuna.enabled=true` and the effective configured
`freqai.data_split_parameters.test_size > 0`. Disable it explicitly with
`enabled=false` in `rl_config_optuna`. A zero holdout disables evaluation/HPO
and remains supported for plain training, including with raw OHLC feature
removal.

`train_cycles` requests approximately training rows × cycles per HPO trial
and final fit; environment and algorithm update boundaries round the actual
budget. It is not an exact episode count. Trials are serialized (one Optuna
worker), although each learner can use several environments. Bound `n_trials`
and use a positive `timeout_hours` for a soft HPO wall-clock budget checked
between trials. Optuna does not interrupt a running trial when that budget
expires, so the current trial can exceed it. The subsequent final fit and
evaluation are outside this timeout. The supplied zero timeout disables the HPO
timeout. Evaluation environments/episodes also cost resources; a larger budget
is not evidence of better economics.

The objective maximizes the best mean deterministic evaluation **episode reward**
across periodic and final evaluations. Progress may display the latest mean
reward, which is not the selected best mean reward. The objective is not net
portfolio profit, Sharpe or Sortino. The holdout participates in model/HPO
selection; reserve a separate unseen market period for economic confirmation
under the [evaluation protocol](../docs/evaluation.md).

Optuna trains fresh candidates in current-window feature coordinates. Sampled
parameters override supplied model parameters during trials. Hyperband prunes
unpromising trials; invalid rollout/batch/buffer combinations and DQN-family
warmup budgets without gradient updates are rejected. An explicitly sampled
`target_kl=null` disables KL stopping even if a numeric value was configured.
There is no user-configurable objective or search-range schema in this wrapper.

With continual learning, the final fit resumes the deployed policy in frozen
feature coordinates, including its discount factor; otherwise it trains a fresh
policy. Selected constructor parameters do not rebuild a resumed architecture.
The current evaluation's best usable checkpoint is deployed; absent one, the
final policy is used. Interrupted fits are logged and follow the same checkpoint
selection. Compatible prior best parameters can be reused after an unsuccessful
search; absent those, effective plain model parameters remain the fallback.

An absent configured `gamma` uses the canonical discount. The effective discount
must be a finite Python/NumPy integer or floating scalar in `[0, 1]`, normalized
to a Python float. Null, booleans, strings, complex values and arrays are rejected
before constructing or replacing environments. Validation follows HPO overrides
and the resumed learner's gamma, so a valid higher-priority value can supersede
an invalid base value. A persisted learner with missing or invalid gamma cannot
continue training. Learner and environment use the same effective discount.
Incompatible reward-objective identities reset persisted HPO studies and cause
stale best-parameter payloads to be ignored rather than warm-started.

## Continual learning

Continuation requires `freqai.continual_learning=true` and effective
`freqai.rl_config.frame_stacking=0`. Nonzero stacking disables continuation
with a warning even if it was requested. The supplied stacking value is 2;
change it explicitly before enabling continuation. Restoring a saved policy for
inference is not the same as continuing its training.

Continual learning trains an independent copy of the deployed policy with its
fitted feature pipeline. DQN/QRDQN deployments each persist their replay buffer;
it is loaded only when continual training starts. Missing or incompatible replay
data prevents continuation but does not prevent inference. The deployment
generation covers frozen feature coordinates and PBRS reward semantics. Otherwise
compatible older generations, including `chronological-frozen-pipelines-v2`, support
inference but cannot continue training. Reset trained models or use a new
`freqai.identifier` to migrate. Training disables `shuffle_after_split`. HPO
studies and saved best parameters are reused only when their objective identity
matches.

Backtests continue only from a saved policy whose training cutoff precedes
the current window's end and its last available candle boundary. A later
deployment under the same identifier is not reused for an earlier window,
even when FreqAI has saved only metadata for intervening windows. Live and
dry-run restarts still restore compatible deployed policies.

## Live inference and backtesting

Optional `fit_live_predictions_candles` statistics use the latest persisted real
predictions per pair, excluding FreqAI bootstrap rows. Available observations
are used before a full window accumulates and survive restarts. FreqAI returns
the initial strategy frame before calculating live statistics; restored
statistics appear on the next prediction update. Legacy rows missing provenance,
including rows in partly marked histories, can count when their nonzero,
nonexpired prediction status distinguishes them from bootstrap; zero-status rows
remain excluded because bootstrap and rejected predictions cannot be
distinguished. Explicit false markers remain excluded.
On duplicate candle dates, a provable prediction takes precedence over an
ambiguous close-bearing legacy row during history restoration.
Rows with an invalid `date_pred` are discarded with a per-pair warning and the
discarded-row count; valid duplicates retain the same precedence.
Optional action statistics do not gate RL actions.

With `hold_potential_enabled=true`, ReforceXY enables `add_state_info` before
constructing environments so training and inference use the same observations.
Freqtrade does not support these state observations in backtesting; disable hold
potential and state observations for backtests. Live action masks use the real
open position even when state observations are disabled. Live frame stacks and
recurrent states persist per pair and model only across adjacent candles. Gaps,
repeated candles and model replacement start a new sequence; historical live
position features are not reconstructed. Prediction validity covers every source
row in the observation and every retained frame, not only the final candle.

For native backtests, set both `freqai.rl_config.add_state_info=false` and
`freqai.rl_config.model_reward_parameters.hold_potential_enabled=false`.
Use a fresh identifier after observation-shape changes. Keep chronological
splits; training disables `shuffle_after_split`, and the native RL base
disables row-removing SVM/DBSCAN/DI outlier removal to preserve trajectories.

## Configuration reference

The implementation and runtime validation are in
[ReforceXY.py](user_data/freqaimodels/ReforceXY.py); the supplied profile is
[config-template.json](user_data/config-template.json). Expected types below
are not promises of validation where the table says an option is forwarded or
not locally validated. This guide covers ReforceXY behavior, overrides and SDK
boundaries. For standard framework settings, use the
[Freqtrade configuration reference](https://www.freqtrade.io/en/stable/configuration/)
and the [FreqAI parameter reference](https://www.freqtrade.io/en/stable/freqai-parameter-table/).
Reward formulas and tunable descriptions are in the [reward reference](reward_space_analysis/README.md#reward-tunables-reference); live environment defaults are defined by `ReforceXY.DEFAULT_*` and the native RL base.

Normal Freqtrade startup injects declared JSON-schema defaults before the FreqAI
model is initialized. In the runtime table, **Schema** values apply to omitted
keys after validation; **direct-constructor** values describe initialization
without that injection. Other fallback entries apply when a native or app field
remains omitted. Supplied values take precedence over defaults. These schema
values were verified with Freqtrade 2026.8 and 2026.9; see the
[versioned native schema](https://github.com/freqtrade/freqtrade/blob/2026.8/freqtrade/config_schema/config_schema.py).
Moving base images can change them; record the runtime version.

- [Runtime, observations and evaluation](#runtime-observations-and-evaluation)
- [HPO settings](#hpo-settings)
- [Model constructor and policy parameters](#model-constructor-and-policy-parameters)

### Runtime, observations and evaluation

| Path                                                                | Schema default / constructor fallback / requirement                    | Type / constraints                                                                                                 | Supplied profile                                                            | Behavior                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| freqai.rl_config.model_type                                         | Schema: PPO; direct constructor requires the key                       | String: app-supported PPO, RecurrentPPO, MaskablePPO, DQN, QRDQN                                                   | MaskablePPO                                                                 | Omitting the key selects PPO, not the supplied MaskablePPO profile. Native loader also accepts A2C/TRPO/ARS, but ReforceXY HPO explicitly does not support them; do not present them as supported app HPO choices.                                                                                                                                                             |
| freqai.rl_config.policy_type                                        | Schema: MlpPolicy; direct constructor requires the key                 | Algorithm-compatible SB3 policy name                                                                               | MlpPolicy                                                                   | Use MlpLstmPolicy with RecurrentPPO; MlpPolicy with ordinary PPO/MaskablePPO/DQN/QRDQN. The schema does not adapt the default to model_type. Forwarded to model constructor, not renamed by app.                                                                                                                                                                               |
| freqai.continual_learning                                           | false                                                                  | Boolean                                                                                                            | Omitted                                                                     | Requires frame_stacking=0. Nonzero effective stacking disables continuation. Keeps deployed fitted feature coordinates/pipeline and policy; architecture changes require new identifier/reset; HPO remains cold/current-window.                                                                                                                                                |
| freqai.rl_config.train_cycles                                       | 25                                                                     | Integer intended; int(value), clamped to minimum 1; no app upper bound                                             | 25                                                                          | Requested budget is training rows × cycles, rounded to environment/model update boundaries; shared by each HPO trial and final fit. Not an exact number of completed episodes.                                                                                                                                                                                                 |
| freqai.data_split_parameters.test_size                              | Gate fallback 0.1; actual omitted split 25% if train_size also omitted | 0 disables evaluation/HPO; otherwise sklearn-valid holdout fraction/count with nonempty training and test datasets | 0.333                                                                       | Set explicitly. HPO additionally requires both freqai.enabled and rl_config_optuna.enabled. Native split forced chronological.                                                                                                                                                                                                                                                 |
| freqai.rl_config.n_envs                                             | 1                                                                      | Integer >=1; invalid reset to 1                                                                                    | 8                                                                           | Parallel training environments; multiplication affects rollout budget and resource requirements.                                                                                                                                                                                                                                                                               |
| freqai.rl_config.n_eval_envs                                        | 1                                                                      | Integer >=1; invalid reset to 1                                                                                    | Omitted                                                                     | Evaluation environments exist only when test_size>0.                                                                                                                                                                                                                                                                                                                           |
| freqai.rl_config.multiprocessing                                    | false                                                                  | Boolean                                                                                                            | true                                                                        | SubprocVecEnv only when n_envs>1; otherwise forced false. Disables plot_new_best.                                                                                                                                                                                                                                                                                              |
| freqai.rl_config.eval_multiprocessing                               | false                                                                  | Boolean                                                                                                            | Omitted                                                                     | SubprocVecEnv only when n_eval_envs>1; otherwise forced false.                                                                                                                                                                                                                                                                                                                 |
| freqai.rl_config.frame_stacking                                     | 0                                                                      | Integer >=0; invalid/1 normalize to 0; >1 enables stacks                                                           | 2                                                                           | Nonzero effective stacking disables continual learning. RecurrentPPO warns redundancy but does not forcibly remove stacks.                                                                                                                                                                                                                                                     |
| freqai.rl_config.action_masking                                     | Derived from model_type == MaskablePPO when omitted                    | Boolean; do not override independently                                                                             | Omitted                                                                     | Model-side training/evaluation/inference compatibility is derived from exact MaskablePPO model name; an explicit field only changes environment invalid-action reward handling, not model capability. Keep omitted to avoid inconsistent settings.                                                                                                                             |
| freqai.rl_config.inference_masking                                  | true                                                                   | Boolean                                                                                                            | Omitted                                                                     | Only effective for MaskablePPO; RecurrentPPO does not support maskable inference. Disabling does not disable training/evaluation masking.                                                                                                                                                                                                                                      |
| freqai.rl_config.lr_schedule                                        | false                                                                  | Boolean                                                                                                            | false                                                                       | Plain fit only: turns numeric learning_rate into clamped linear initial-to-zero schedule. HPO samples its own schedule.                                                                                                                                                                                                                                                        |
| freqai.rl_config.cr_schedule                                        | false                                                                  | Boolean                                                                                                            | false                                                                       | Plain fit only, PPO family only: turns numeric clip_range into clamped linear initial-to-zero schedule. HPO samples its own schedule.                                                                                                                                                                                                                                          |
| freqai.rl_config.n_eval_steps                                       | 10000                                                                  | Integer >0; invalid reset to 10000                                                                                 | Omitted                                                                     | Non-PPO evaluation interval in aggregate timesteps, converted to callback calls via ceil(n_eval_steps/n_envs). PPO prefers n_steps/rollout-sized interval instead. HPO reduces interval by factor 4.                                                                                                                                                                           |
| freqai.rl_config.n_eval_episodes                                    | 5                                                                      | Integer >0; invalid reset to 5                                                                                     | Omitted                                                                     | Episodes averaged in periodic and final deterministic policy evaluation.                                                                                                                                                                                                                                                                                                       |
| freqai.rl_config.max_no_improvement_evals                           | 0                                                                      | Integer intended; 0 disables; no local range/type validation                                                       | 0                                                                           | Adds SB3 StopTrainingOnNoModelImprovement for ordinary fit evaluation, not HPO trial callbacks.                                                                                                                                                                                                                                                                                |
| freqai.rl_config.min_evals                                          | 0                                                                      | Integer intended; no local range/type validation                                                                   | 0                                                                           | Wait evaluations before no-improvement stopping is counted; relevant only with max_no_improvement_evals enabled.                                                                                                                                                                                                                                                               |
| freqai.rl_config.check_envs                                         | true                                                                   | Boolean                                                                                                            | true                                                                        | Gym API smoke validation of training env and, with test_size>0, eval env.                                                                                                                                                                                                                                                                                                      |
| freqai.rl_config.tensorboard_throttle                               | 1                                                                      | Integer >=1; invalid reset to 1                                                                                    | Omitted                                                                     | When activate_tensorboard is enabled, InfoMetricsCallback collects metrics when aggregate num_timesteps is divisible by this value (1 collects on every callback call). This is a timestep-modulo filter, not a callback-call interval: with n_envs=8 and throttle=8, every call passes. TensorBoard write/flush frequency is separate.                                        |
| freqai.rl_config.plot_new_best                                      | false                                                                  | Boolean                                                                                                            | false                                                                       | TensorBoard rollout plot on a new best ordinary-fit checkpoint; disabled with training multiprocessing.                                                                                                                                                                                                                                                                        |
| freqai.rl_config.plot_window                                        | 2000                                                                   | Integer intended; positive truncates; <=0 retains full history; no local type validation                           | Omitted                                                                     | Environment history rows retained in rollout plot.                                                                                                                                                                                                                                                                                                                             |
| freqai.rl_config.progress_bar                                       | Schema: true; direct-constructor fallback: false                       | Boolean                                                                                                            | Omitted                                                                     | Enables training progress callback and Optuna progress display; omitted validated configuration enables it.                                                                                                                                                                                                                                                                    |
| freqai.rl_config.add_state_info                                     | false                                                                  | Boolean                                                                                                            | true                                                                        | Adds unlevered unrealized PnL, position, trade duration to observations; unsupported in backtesting. hold_potential_enabled automatically enables it, so hold potential also cannot be backtested.                                                                                                                                                                             |
| freqai.rl_config.cpu_count                                          | 1                                                                      | Positive integer suitable for torch threads; no app validation                                                     | 4                                                                           | Native torch thread count capped by half max_system_threads; separate from environment count.                                                                                                                                                                                                                                                                                  |
| freqai.rl_config.max_training_drawdown_pct                          | Schema: 0.02; direct-constructor fallback: 0.8                         | Numeric intended; no local range validation; do not claim enforced [0,1]                                           | 0.02                                                                        | Native equity floor is 1-value, so an omitted validated value gives 0.98. Training episode risk cutoff, not a live exchange stoploss.                                                                                                                                                                                                                                          |
| freqai.fit_live_predictions_candles                                 | 0                                                                      | Nonnegative integer; bool rejected; invalid raises ValueError                                                      | Omitted                                                                     | 0 disables action mean/population std statistics; live uses persisted produced observations, backtest prior rows; statistics do not gate RL actions.                                                                                                                                                                                                                           |
| freqai.rl_config.model_reward_parameters                            | Required map; no schema or application fallback                        | Dictionary; rr and profit_aim are schema-defaulted members                                                         | rr=2, profit_aim=0.025, max_trade_duration_candles=96, idle_penalty_ratio=0 | Include the map. Schema validation supplies omitted rr/profit_aim members; a direct native constructor requires both. Analyzer defaults do not supply this map. Optional reward settings have their own environment fallbacks. Reward shaping and duration normalization are not live order timeouts. Model gamma overrides potential_gamma when environments are constructed. |
| freqai.rl_config.model_reward_parameters.rr                         | Schema: 1; direct constructor requires the member                      | Numeric reward-to-risk factor                                                                                      | 2                                                                           | The schema injects an omitted member into an existing reward map; direct environment construction without it raises. Native reward target uses rr × profit_aim.                                                                                                                                                                                                                |
| freqai.rl_config.model_reward_parameters.profit_aim                 | Schema: 0.025; direct constructor requires the member                  | Numeric profit target                                                                                              | 0.025                                                                       | The schema injects an omitted member into an existing reward map; direct environment construction without it raises. This reward target does not place a live take-profit order.                                                                                                                                                                                               |
| freqai.rl_config.model_reward_parameters.max_trade_duration_candles | 128                                                                    | Integer-convertible value; no local range validation                                                               | 96                                                                          | Duration scale for reward penalties, not a live timeout. The schema accepts the top-level homonym `freqai.rl_config.max_trade_duration_candles`, but ReforceXY ignores it; only this reward-map member takes effect.                                                                                                                                                           |
| freqai.rl_config.model_reward_parameters.max_idle_duration_candles  | 4 × effective max_trade_duration_candles                               | Integer-convertible value; no local range validation                                                               | Omitted (384 from the supplied trade scale)                                 | Idle duration scale for reward penalties, not a live timer.                                                                                                                                                                                                                                                                                                                    |

### HPO settings

| Path                                     | Runtime fallback / requirement | Type / constraints                                                                                             | Supplied profile | Behavior                                                                                                                                                                                                                                                                        |
| ---------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| freqai.rl_config_optuna.enabled          | false                          | Boolean                                                                                                        | true             | Requires freqai.enabled=true and test_size>0; optimize PPO/RecurrentPPO/MaskablePPO/DQN/QRDQN.                                                                                                                                                                                  |
| freqai.rl_config_optuna.n_trials         | 100                            | Integer intended; forwarded to Optuna, no local range validation                                               | 100              | New trial budget per optimize call; does not mean lifetime cap on reusable study; serialized n_jobs=1.                                                                                                                                                                          |
| freqai.rl_config_optuna.n_startup_trials | 15                             | Integer intended; forwarded to TPE, no local range validation                                                  | 15               | TPE random-startup trials; not passed to AutoSampler.                                                                                                                                                                                                                           |
| freqai.rl_config_optuna.timeout_hours    | 0                              | Numeric hours; no local range validation                                                                       | 0                | 0 disables the HPO timeout; nonzero values are multiplied by 3600 and passed to study.optimize. Positive values set a soft budget checked between trials: a running trial may exceed it, and the subsequent final fit/evaluation are outside it. Does not replace train_cycles. |
| freqai.rl_config_optuna.continuous       | false                          | Boolean                                                                                                        | Omitted          | IMPORTANT: true deletes/recreates the study each optimize call, not endless trial execution. false reuses compatible objective-identity study.                                                                                                                                  |
| freqai.rl_config_optuna.warm_start       | false                          | Boolean                                                                                                        | Omitted          | Enqueues saved compatible best parameters. A purge-triggered run enqueues previous best even if warm_start=false.                                                                                                                                                               |
| freqai.rl_config_optuna.sampler          | tpe                            | tpe \| auto; invalid raises ValueError                                                                         | Omitted          | tpe uses multivariate/group TPE; auto loads OptunaHub auto sampler and its dependencies.                                                                                                                                                                                        |
| freqai.rl_config_optuna.storage          | sqlite                         | sqlite \| file; invalid raises ValueError                                                                      | Omitted          | sqlite RDB or journal file under identifier model directory; filename uses base symbol.                                                                                                                                                                                         |
| freqai.rl_config_optuna.purge_period     | 0                              | int(value) conversion occurs before validator; negative reset to 0; unconvertible raises during initialization | Omitted          | 0 disables periodic purge; positive resets every X pair retrains; ignored/forced 0 when continuous=true.                                                                                                                                                                        |
| freqai.rl_config_optuna.seed             | 42                             | Integer intended; no local validation                                                                          | Omitted          | Sampler seed; distinct from model_training_parameters.seed used for models/envs.                                                                                                                                                                                                |

### Model constructor and policy parameters

| Path                                                           | Runtime fallback / requirement                                                               | Type / constraints                                                                                       | Supplied profile                                   | Behavior                                                                                                                                                                                            |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| freqai.model_training_parameters                               | {} native                                                                                    | Dictionary                                                                                               | device:auto, verbose:1                             | Deep-copied then normalized; remaining kwargs forwarded to chosen SB3 constructor. This is SDK boundary, not an app-owned exhaustive hyperparameter schema.                                         |
| freqai.model_training_parameters.seed                          | 42                                                                                           | Integer intended; SDK validation                                                                         | Omitted                                            | Local seed default; envs derive seeds and HPO increments by trial number.                                                                                                                           |
| freqai.model_training_parameters.gamma                         | 0.95                                                                                         | Finite Python/NumPy integer or floating scalar in [0,1]; null, bool, string, complex and arrays rejected | Omitted                                            | Effective value is validated after HPO/resumed-learner precedence and propagated to environment potential_gamma. Invalid persisted gamma blocks continuation.                                       |
| freqai.model_training_parameters.learning_rate                 | SDK default unless lr_schedule=true; then initial 0.0003 if omitted                          | SDK numeric/schedule; app wraps numeric only                                                             | Omitted                                            | Do not call 0.0003 an unconditional app default.                                                                                                                                                    |
| freqai.model_training_parameters.clip_range                    | SDK default unless PPO-family cr_schedule=true; then initial 0.2 if omitted                  | SDK numeric/schedule; app wraps numeric only                                                             | Omitted                                            | Only PPO family schedule conversion.                                                                                                                                                                |
| freqai.model_training_parameters.gpu_memory_fraction           | null/omitted disables                                                                        | Numeric in (0,1] intended; ignored invalid bounds/no CUDA, not robust type validation                    | Omitted                                            | App-only; applies torch CUDA per-process limit to device 0, removed before SDK constructor.                                                                                                         |
| freqai.model_training_parameters.device                        | No app fallback; SDK auto                                                                    | PyTorch/SB3 device specification                                                                         | auto                                               | Forwarded; doc examples auto/cpu/cuda/cuda:0 are not an app-enforced exhaustive choice list.                                                                                                        |
| freqai.model_training_parameters.verbose                       | SDK constructor default; callback lookup fallback 0                                          | SDK verbosity integer                                                                                    | 1                                                  | Forwarded to constructor, also read for callback verbosity.                                                                                                                                         |
| freqai.model_training_parameters.policy_kwargs.net_arch        | [128,128] per branch for PPO family; [128,128] for others                                    | small\|medium\|large\|extra_large, list of layer widths; PPO family additionally {pi:[...],vf:[...]}     | Omitted                                            | Presets use two layers of 128/256/512/1024; PPO list expanded to both heads, invalid/missing head resets to [128,128]. Other families accept list or presets. Layer element validation remains SDK. |
| freqai.model_training_parameters.policy_kwargs.activation_fn   | relu                                                                                         | relu\|tanh\|elu\|leaky_relu                                                                              | Omitted                                            | Converted to torch class; unknown name silently falls back to ReLU.                                                                                                                                 |
| freqai.model_training_parameters.policy_kwargs.optimizer_class | adamw                                                                                        | adamw\|rmsprop\|adam                                                                                     | Omitted                                            | Converted to torch class; unknown name falls back to Adam (not AdamW).                                                                                                                              |
| freqai.model_training_parameters.subsample_steps               | Absent                                                                                       | Positive integer intended; no local rejection                                                            | Omitted                                            | DQN/QRDQN-only app convenience, removed before constructor; computes gradient_steps=min(train_freq,max(1,ceil(train_freq/subsample_steps))) when both positive integers. Otherwise -1.              |
| freqai.model_training_parameters.gradient_steps                | DQN-family absent/None -> derived value, usually -1 without valid train_freq+subsample_steps | SDK integer                                                                                              | Omitted                                            | Explicit non-None value retained; not an unconditional SDK-default forwarding.                                                                                                                      |
| leverage                                                       | proposed_leverage, not fixed 1                                                               | Finite float-convertible requested value; effective clamped [1,max_leverage]                             | Omitted (the commented example is not an override) | Top-level strategy field, not an RL/model parameter. Invalid/nonfinite values use the proposal; live state PnL is divided by trade leverage to align the training proxy.                            |
| custom_protections                                             | []                                                                                           | List of dictionaries, each with a string method; otherwise ValueError                                    | []                                                 | Top-level strategy field. Protection-specific settings are delegated to Freqtrade protection plugins. An empty list does not configure custom protections.                                          |

Unlisted algorithm-specific constructor kwargs are forwarded to the selected
SDK after application normalization; use the compatible algorithm's documented
parameters, not a common parameter list for all models:
[PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html),
[DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html),
[MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html),
[RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html),
[QRDQN](https://sb3-contrib.readthedocs.io/en/master/modules/qrdqn.html).
Application HPO uses its fixed algorithm-specific search, not every SDK option.

LSTM `shared_lstm`, `enable_critic_lstm`, `lstm_hidden_size` and `n_lstm_layers`
are SDK policy kwargs. HPO forces the critic LSTM off when a shared LSTM is
selected; plain fits forward the configured combination. Use
`model_training_parameters.policy_kwargs.net_arch` for this model:
`rl_config.net_arch` is not an equivalent architecture override.

## Reward and portfolio accounting

The reward logic and tunables are documented in the
[reward space analysis](reward_space_analysis/README.md).

Environment diagnostics `most_recent_return` (log return) and
`most_recent_profit` (simple return) measure changes in liquidation equity,
including unrealized PnL and Freqtrade's staking convention. Actions fill at
`open[t]` while the observation window ends at candle `t-1`; equity marks, PnL
features and trade durations in the returned observation refer to candle `t+1`.
Round-trip fees are provisioned at entry; exits realize at the fill price without
charging fees again. `portfolio_log_returns` stores the same log returns.
Non-positive or non-finite equity produces NaN diagnostics rather than a zero
return. These diagnostics do not change the training reward or realized capital.
Rewards combine the fill-time base components with a potential-based shaping
delta over the returned next observation. Termination liquidates any remaining
position once and sets terminal shaping to `-prev_potential`, even for tiny
nonzero potentials. `get_env_history()` returns one metrics/price row per transition.
Its `execution_tick` is the transition/action/fill
key before the tick increment; its `tick` is the returned post-increment price and
observation row (normally `execution_tick + 1`). Exit-efficiency extrema include
the fee-adjusted PnL at entry and subsequent retained market marks. Ordered
trade events remain separate in `trade_history`: their tick identifies the
candle whose price filled the event. Action fills use `execution_tick`;
terminal liquidations use the returned post-increment tick. `terminal_liquidation`
and `exit_pnl` remain on the transition history row.

Every enabled PBRS transition, including neutral steps with retained potential,
uses `gamma * next_potential - prev_potential`. Both `canonical` and
`non_canonical` have zero exit potential; the latter permits optional additives.
The `pbrs_invariant` diagnostic recognizes either zero-exit mode when both
effective additives are disabled. It is a conservative configuration check, not
a guarantee about fitted policies: theoretical invariance also requires an
observable state potential, the learner's discount, and appropriate boundaries.
