# Project Context

## Purpose

Research, prototype, and refine advanced ML‑driven and RL‑driven trading strategies for the FreqAI / Freqtrade ecosystem. Two strategies:

- QuickAdapter: ML strategy + adaptive execution heuristics for partial exits, volatility‑aware stop/take profit logic.
- ReforceXY: RL strategy and reward space analysis.
  Reward space analysis goals:
  - Provide deterministic synthetic sampling and statistical diagnostics to validate reward components and potential‑based shaping behavior and reason about reward parameterization before RL training.
  - Maintain deterministic runs and rich diagnostics to accelerate iteration and anomaly debugging.

## Tech Stack

- Python 3.11+.
- Freqtrade + FreqAI (strategy framework & ML integration).
- TA libraries: TA-Lib, pandas_ta, custom technical helpers.
- ReforceXY reward space analysis:
  - Project management: uv.
  - Scientific stack: numpy, pandas, scipy, scikit‑learn.
  - Linting: Ruff.
  - Testing: PyTest + pytest-cov.
- Docker + docker-compose (containerized runs / reproducibility).

## Project Conventions

### Code Style

- Base formatting guided by `.editorconfig` (UTF-8, LF, final newline, trimming whitespace, Python indent = 4 spaces, global indent_size=2 for non‑Python where appropriate, max Python line length target 100; Markdown max line length disabled).
- Naming:
  - Functions & methods: `snake_case`.
  - Constants: `UPPER_SNAKE_CASE`.
  - Internal strategy transient labels/features use prefixes: `"%-"` for engineered feature columns; special markers like `"&s-"` / `"&-"` for internal prediction target(s).
  - Private helpers or internal state use leading underscore (`_exit_thresholds_calibration`).
- Avoid one‑letter variable names; prefer descriptive one (e.g. `trade_duration_candles`, `natr_ratio_fraction`).
- Prefer explicit type hints (Python 3.11+ built‑in generics: `list[str]`, `dict[str, float]`).
- Logging: use module logger (`logger = logging.getLogger(__name__)`), info for decision denials, warning for anomalies, error for exceptions.
- No non-English terms in code, docs, comments, logs.

### Architecture Patterns

- Strategy classes subclass `IStrategy` with model classes subclass `IFreqaiModel`; separate standalone strategy under root directory.
- Reward Space Analysis: standalone CLI module (`reward_space_analysis.py`) + tests focusing on deterministic synthetic scenario generation, decomposition, statistical validation, potential‑based reward shaping (PBRS) variants.
- Separation of concerns: reward analytic tooling does not depend on strategy runtime state; consumption is offline pre‑training / validation.

### Reward Space Analysis Testing Strategy

- PyTest test modules in `reward_space_analysis/tests/<focus>`.
- Focus: correctness of reward calculations, statistical invariants, PBRS modes, transforms, robustness, integration end‑to‑end.
- Logging configured for concise INFO output; colored, warnings disabled by default in test runs.
- Coverage goal: ≥85% on new analytical logic; critical reward shaping paths must be exercised (component bounds, invariance checks, exit attenuation kernels, transform functions, distribution metrics).
- Focused test invocation examples (integration, statistical coherence, reward alignment) documented in README.
- Run tests after: modifying reward logic; before major analyses; dependency or Python version changes; unexpected anomalies.

### Git Workflow

- Primary branch: `main`. Feature / experiment branches should be: `feat/<concise-topic>`, `exp/<strategy-or-reward-param>`. Fix branches should be: `fix/<bug>`.
- Commit messages: imperative, follow Conventional Commits. Emphasize WHY over raw WHAT when non‑obvious.
- Avoid large mixed commits; isolate analytical tooling changes from strategy behavior changes.
- Keep manifests and generated outputs out of version control (only code + templates); user data directories contain `.gitkeep` placeholders.

## Domain Context

- Strategies operate on sequential market OHLCV data.
- QuickAdapterV3 features engineered include volatility metrics (NATR/ATR), momentum (MACD, EWO), market structure shift (extrema labeling via zigzag), band widths (BB, KC, VWAP), and price distance measures.
- QuickAdapterV3 integrates dynamic volatility interpolation (weighted/moving average/interpolation modes) to derive adaptive NATR for stoploss/take profit calculations; partial exits based on staged NATR ratio percentages.
- ReforceXY reward shaping emphasizes potential‑based reward shaping (PBRS) invariance: canonical vs non canonical modes, hold/entry/exit additive toggles, duration penalties, exit attenuation kernels (linear/power/half‑life/etc.).

## Important Constraints

- Python version ≥3.11 (target for type hints).
- Trading mode affects short availability (spot disallows shorts); logic must gate short entries accordingly.
- Computations must handle missing/NaN gracefully.
- Regulatory / business: none explicit; treat strategies as experimental research (no performance guarantees) and avoid embedding sensitive credentials.

## External Dependencies

- Freqtrade / FreqAI framework APIs.
- Docker images defined per strategy project (`Dockerfile.quickadapter`, `Dockerfile.reforcexy`) for containerized execution.
