# Tests: Reward Space Analysis

Authoritative documentation for invariant ownership, taxonomy layout, smoke policies, and maintenance workflows. This single README now contains the full coverage mapping (previously in `TEST_COVERAGE_MAP.md`).

## Purpose

The suite enforces:
- Reward component mathematics & transform correctness
- PBRS invariance mechanics (canonical drift correction, near-zero classification)
- Robustness under extreme / invalid parameter settings
- Statistical metrics integrity (bootstrap, constant distributions)
- CLI parameter propagation & report formatting
- Cross-component smoke scenarios

Single ownership per invariant is tracked in the Coverage Mapping section of this README.

## Taxonomy Directories

| Directory | Marker | Scope |
|-----------|--------|-------|
| `components/` | components | Component math & transforms |
| `transforms/` | transforms | Transform function behavior |
| `robustness/` | robustness | Edge cases, stability, progression |
| `api/` | api | Public API helpers & parsing |
| `cli/` | cli | CLI parameter propagation & artifacts |
| `pbrs/` | pbrs | Potential-based shaping invariance & modes |
| `statistics/` | statistics | Statistical metrics, tests, bootstrap |
| `integration/` | integration | Smoke scenarios & report formatting |
| `helpers/` | (none) | Helper utilities (data loading, assertions) |

Markers are declared in `pyproject.toml` and enforced with `--strict-markers`.

## Running Tests

Full suite (coverage ≥85% enforced):
```shell
uv run pytest
```
Selective markers:
```shell
uv run pytest -m pbrs -q
uv run pytest -m robustness -q
uv run pytest -m "components or robustness" -q
uv run pytest -m "not slow" -q
```
Coverage reports:
```shell
uv run pytest --cov=reward_space_analysis --cov-report=term-missing
uv run pytest --cov=reward_space_analysis --cov-report=html && open htmlcov/index.html
```
Slow statistical tests:
```shell
uv run pytest -m "statistics and slow" -q
```

## Coverage Mapping (Invariant Ownership)

Columns:
- ID: Stable identifier (`<category>-<shortname>-NNN`) or numeric-only legacy (statistics block).
- Category: Taxonomy directory marker.
- Description: Concise invariant statement.
- Owning File: Path:line of primary declaration.
- Notes: Clarifications (sub-modes, extensions, non-owning references elsewhere).

| ID | Category | Description | Owning File | Notes |
|----|----------|-------------|-------------|-------|
| report-abs-shaping-line-091 | integration | Abs Σ Shaping Reward line present & formatted | integration/test_report_formatting.py:4 | PBRS report uses line non-owning (format owned here) |
| report-additives-deterministic-092 | components | Additives deterministic report section | components/test_additives.py:4 | Integration/PBRS may reference outcome non-owning |
| robustness-decomposition-integrity-101 | robustness | Single active core component equals total reward under mutually exclusive scenarios | robustness/test_robustness.py:35 | Scenarios: idle, hold, exit, invalid; non-owning refs integration/test_reward_calculation.py |
| robustness-exit-mode-fallback-102 | robustness | Unknown exit_attenuation_mode falls back to linear w/ warning | robustness/test_robustness.py:520 | Owns invariant (numeric tag) |
| robustness-negative-grace-clamp-103 | robustness | Negative exit_plateau_grace clamps to 0.0 w/ warning | robustness/test_robustness.py:549 | Owns invariant (numeric tag) |
| robustness-invalid-power-tau-104 | robustness | Invalid power tau falls back alpha=1.0 w/ warning | robustness/test_robustness.py:585 | Owns invariant (numeric tag) |
| robustness-near-zero-half-life-105 | robustness | Near-zero half life yields no attenuation (factor≈base) | robustness/test_robustness.py:613 | Owns invariant (numeric tag) |
| pbrs-canonical-drift-correction-106 | pbrs | Canonical drift correction enforces near zero-sum shaping | pbrs/test_pbrs.py:448 | Owns invariant: zero-sum, exception fallback (474), comparison path (516) |
| pbrs-canonical-near-zero-report-116 | pbrs | Canonical near-zero cumulative shaping classification | pbrs/test_pbrs.py:747 | Owns invariant (classification canonical near-zero cumulative shaping report) |
| statistics-partial-deps-skip-107 | statistics | skip_partial_dependence => empty PD structures | statistics/test_statistics.py:28 | |
| helpers-duplicate-rows-drop-108 | helpers | Duplicate rows dropped w/ warning counting removals | helpers/test_utilities.py:26 | |
| helpers-missing-cols-fill-109 | helpers | Missing required columns filled with NaN + single warning | helpers/test_utilities.py:50 | |
| statistics-binned-stats-min-edges-110 | statistics | <2 bin edges raises ValueError | statistics/test_statistics.py:45 | |
| statistics-constant-cols-exclusion-111 | statistics | Constant columns excluded & listed | statistics/test_statistics.py:57 | |
| statistics-degenerate-distribution-shift-112 | statistics | Degenerate dist: zero shift metrics & KS p=1.0 | statistics/test_statistics.py:74 | |
| statistics-constant-dist-widened-ci-113a | statistics | Non-strict: widened CI with warning | statistics/test_statistics.py:529 | Shares base distribution context with 113b |
| statistics-constant-dist-strict-omit-113b | statistics | Strict: omit metrics (no widened CI) | statistics/test_statistics.py:561 | Sub-mode of constant distribution invariant |
| statistics-fallback-diagnostics-115 | statistics | Fallback diagnostics constant distribution (qq_r2=1.0 etc.) | statistics/test_statistics.py:190 | |
| robustness-exit-pnl-only-117 | robustness | Only exit actions have non-zero PnL | robustness/test_robustness.py:125 | Newly assigned ID (previously unnumbered) |

### Non-Owning Smoke / Reference Checks

Files that reference invariant outcomes (formatting, aggregation) without owning the invariant must include a leading comment:
```python
# Non-owning smoke; ownership: <owning file>
```
Table tracks approximate line ranges and source ownership:

| File | Lines (approx) | References | Ownership Source |
|------|----------------|-----------|------------------|
| integration/test_reward_calculation.py | 20-25 | Decomposition identity (sum components) | robustness/test_robustness.py:35 |
| components/test_reward_components.py | 214-243 | Exit factor finiteness & plateau behavior | robustness/test_robustness.py:156+ |
| pbrs/test_pbrs.py | 616,624,799 | Abs Σ Shaping Reward line formatting | integration/test_report_formatting.py:87-100 |
| pbrs/test_pbrs.py | 591-630 | Non/Canonical classification formatting | robustness/test_robustness.py:35, robustness/test_robustness.py:125 |
| pbrs/test_pbrs.py | 742-806 | Canonical near-zero cumulative shaping classification | robustness/test_robustness.py:35 |
| pbrs/test_pbrs.py | 807-860 | Canonical warning classification (|Σ shaping| > tolerance) | robustness/test_robustness.py:35 |
| pbrs/test_pbrs.py | 861-915 | Non-canonical full report reason aggregation | robustness/test_robustness.py:35 |
| pbrs/test_pbrs.py | 916-969 | Non-canonical mode-only reason (additives disabled) | robustness/test_robustness.py:35 |
| statistics/test_statistics.py | 292 | Mean decomposition consistency | robustness/test_robustness.py:35 |

### Deprecated / Reserved IDs

| ID | Status | Rationale |
|----|--------|-----------|
| 093 | deprecated | CLI invariance consolidated; no dedicated test yet |
| 094 | deprecated | CLI encoding/data migration removed in refactor |
| 095 | deprecated | Report CLI propagation assertions merged into test_cli_params_and_csv |
| 114 | reserved | Gap retained for potential future statistics invariant |

## Adding New Invariants

1. Assign ID `<category>-<shortname>-NNN` (NNN numeric). Reserve gaps explicitly if needed (see deprecated/reserved table).
2. Add a row in Coverage Mapping BEFORE writing the test.
3. Implement test in correct taxonomy directory; add marker if outside default selection.
4. Optionally declare inline ownership:
   ```python
   # Owns invariant: <id>
   def test_<short_description>(...):
       ...
   ```
5. Run duplication audit and coverage before committing.

## Duplication Audit

Each invariant shortname must appear in exactly one taxonomy directory path:
```shell
cd ReforceXY/reward_space_analysis/tests
grep -R "<shortname>" -n .
```
Expect a single directory path. Examples:
```shell
grep -R "drift_correction" -n .
grep -R "near_zero" -n .
```

## Parity & Migration Rationale

Refactor preserved semantic coverage while moving detailed assertions from integration (report formatting only) into targeted directories (components, robustness). IDs 091–095 clarified ownership; 106 spans multi-path (primary + fallback + comparison) inside a single file.

## When to Run Tests

Run after changes to: reward component logic, PBRS mechanics, CLI parsing/output, statistical routines, dependency or Python version upgrades, or before publishing analysis reliant on invariants.

---

This README is the single authoritative source for test coverage, invariant ownership, smoke policies, and maintenance guidelines.