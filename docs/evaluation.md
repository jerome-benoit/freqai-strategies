# Strategy evaluation protocol

[Operator guides](../README.md#documentation) describe how to run each strategy.
This protocol governs evidence for changing configurations or strategy behavior.

Evaluate a proposed change against the current configuration on the same unseen
market history. Judge portfolio performance after costs, not training loss. This
procedure does not establish that the current defaults are optimal.

## What the backtest measures

In Freqtrade 2026.8, the [native backtest][freqai-running] constructs each pair's
rolling predictions before replaying enabled `fit_live_predictions()` updates
([training loop][freqai-source], [replay loop][freqai-replay]). It exercises
rolling model fits, threshold replay and strategy decisions, but a label-HPO
update during replay cannot affect an already-trained model. Testing that live
feedback requires a chronological runner that interleaves training, prediction
and state updates, or a forward dry-run. This repository provides no such runner;
do not present native-backtest results as validation of the complete live loop.

QuickAdapter predicts smoothed Zigzag morphology, not returns. `holdout_rmse`
measures the selection model's weighted error on the original label scale,
on a holdout within the training window, before any deployment refit. An empty
holdout, including one emptied by causal purging, yields `holdout_rmse=inf`
(unavailable). With `method=train_test_split`, `test_size=0` disables internal
validation and final refit, not later rolling predictions; `timeseries_split`
does not accept zero. Use RMSE to diagnose prediction quality, not profitability.

## Design the comparison

1. **Fix the question before inspecting results.** Specify the incumbent,
   candidate change, pair universe, evaluation dates, training/prediction window
   lengths, HPO budget, seeds and costs. Choose a primary economic metric, a
   minimum worthwhile improvement and acceptable risk limits. Record all tried
   configurations, including failures. Reserve a final chronological period for
   confirmation; once used to revise the strategy, it is no longer unseen.
2. **Reproduce the information available at each decision.** Train on earlier
   data and compare both configurations on identical subsequent timestamps.
   Account for listing/delisting dates and missing candles; selecting only
   today's surviving pairs biases historical results. Fit preprocessing and
   select features/model hyperparameters inside each training window, using
   time-ordered inner validation. Keep scoring windows outside model selection.
   Threshold calibration must use only predictions available at that time.
3. **Respect label availability.** Keep `causal_mode` enabled. A historical row
   is not usable for training until all observations needed for its labels and
   weights are known. Add each `known_at_lookahead` candle offset to its row
   position in the unsliced window; use the latest availability across labels
   and weights. Audit it against each split cutoff, rejecting unknown or
   out-of-frame availability. `causal_mode` alone is not proof of this invariant.
   Allow for additional publication/execution delays where relevant. Purging
   removes overlapping label information; an embargo excludes training samples
   immediately _after_ a validation block when a split uses future training data
   ([López de Prado][afml]). Prefer earlier-only training here, not an arbitrary
   universal embargo duration.
4. **Isolate the change and its state.** Start with fixed label/model parameters
   when comparing a component; evaluate tuning separately if it is part of the
   proposed behavior. Dynamic label HPO optimizes morphology in
   `fit_live_predictions()`, not held-out trading returns: judge its choices on
   subsequent economic results using the live-loop evaluation above. With
   validation enabled, QuickAdapter cold-starts regressor trials and the
   selection model; inherited models are reserved for deployment refit. Use separate
   `freqai.identifier` values and model, prediction and Optuna storage for each
   configuration/seed. `--cache none` bypasses backtest-result caching, not FreqAI
   model or prediction reuse.

## Measure economics and uncertainty

- **Model costs and execution.** Hold sizing, protections and execution rules
  constant unless they are the change under test. Set `--fee` explicitly, use
  `--enable-protections` when evaluating protections, and use downloaded detail
  candles with `--timeframe-detail` where feasible. Compare plausible base and
  adverse cost scenarios, including spread, slippage, impact and funding/borrow
  costs where applicable. Freqtrade's [candle assumptions][freqtrade-backtesting]
  do not establish realistic fills or capacity; non-fee execution effects need
  a separate model. A dry-run checks forward behavior, not actual exchange fills.
- **Report portfolio outcomes.** Compare net return, maximum drawdown, exposure,
  turnover and trade count, with results by period and long/short side. State the
  equity convention and sampling interval. Closed-trade balance omits unrealized
  losses: use equity including open positions for portfolio drawdown, or label
  the reported balance-based measure and its limitation. Do not average window
  drawdowns. Report prediction coverage, failed windows, `holdout_rmse` and
  training latency alongside economics. Do not discard failed runs to improve
  averages. Cash/buy-and-hold provide context, not a replacement for the incumbent.
- **Separate market uncertainty from training randomness.** Repeat stochastic
  fits/searches with the same planned seed list for both configurations and
  report the paired differences, not just the best run. Seeds reuse the same
  market history; they are not independent market samples. There is no universal
  sufficient seed count. Record sampler/model seeds and parallelism; a fixed
  seed alone does not guarantee identical HPO or GPU results.
- **Match inference to the data.** For uncertainty in mean performance, compare
  aligned portfolio returns at a stated frequency. A paired block bootstrap can
  preserve temporal dependence by resampling the same time blocks for both
  configurations ([Politis and Romano][stationary-bootstrap]). State the effect,
  interval method, confidence level, block-length choice and sensitivity to it.
  Justify the dependence/stationarity assumptions; neither extra seeds nor more
  bootstrap draws compensate for short history or regime changes. Maximum
  drawdown is path-dependent: an interval for mean return is not its risk bound.
  Report results as inconclusive when the data cannot support the intended claim.
- **Account for strategy selection.** Repeatedly choosing the best backtest
  inflates apparent performance ([Bailey et al.][pbo]). If making significance
  claims across candidates, define the comparison family and use valid
  dependence-aware tests with a multiple-testing correction such as
  [Holm's procedure][holm]; correction cannot repair invalid underlying p-values.
  Report effect sizes and uncertainty, not only significance. Keep drawdown and
  cost sensitivity visible rather than reducing the decision to a single score.

## Confirm and preserve the evidence

Run [lookahead analysis][lookahead-analysis] and [recursive
analysis][recursive-analysis] to investigate leakage and startup sensitivity.
Use adequate history for every informative timeframe and a separate disposable
FreqAI identifier for each analysis, with no existing model directory. **Both
commands delete the selected identifier's model directory during analysis.**
Never use retained or live-run identifiers. Exempt only confirmed
target-construction flags; investigate feature and signal differences. Clean
results cover only the paths exercised, not the absence of all leakage.

Evaluate the frozen candidate on the reserved period, then check forward behavior
in dry-run. Adopt it only if the evidence supports the planned economic and risk
criteria; otherwise retain the incumbent and distinguish rejection from
insufficient evidence. Archive a timestamped run manifest with commits, resolved
image/dependency versions, configuration/data hashes, commands, identifiers,
seeds, HPO histories, split cutoffs, costs and results. The Docker base tag moves;
record the image digest, not just `stable_freqai`.

[afml]: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
[freqai-running]: https://www.freqtrade.io/en/stable/freqai-running/
[freqai-replay]: https://github.com/freqtrade/freqtrade/blob/2026.8/freqtrade/freqai/freqai_interface.py#L895-L935
[freqai-source]: https://github.com/freqtrade/freqtrade/blob/2026.8/freqtrade/freqai/freqai_interface.py#L272-L409
[freqtrade-backtesting]: https://www.freqtrade.io/en/stable/backtesting/
[holm]: https://www.jstor.org/stable/4615733
[lookahead-analysis]: https://www.freqtrade.io/en/stable/lookahead-analysis/
[pbo]: https://doi.org/10.21314/JCF.2016.322
[recursive-analysis]: https://www.freqtrade.io/en/stable/recursive-analysis/
[stationary-bootstrap]: https://doi.org/10.1080/01621459.1994.10476870
