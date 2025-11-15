# Tests: Reward Space Analysis

Authoritative documentation for invariant ownership, taxonomy layout, smoke policies, maintenance workflows, and full coverage mapping.

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

| Directory      | Marker      | Scope                                       |
| -------------- | ----------- | ------------------------------------------- |
| `components/`  | components  | Component math & transforms                 |
| `robustness/`  | robustness  | Edge cases, stability, progression          |
| `api/`         | api         | Public API helpers & parsing                |
| `cli/`         | cli         | CLI parameter propagation & artifacts       |
| `pbrs/`        | pbrs        | Potential-based shaping invariance & modes  |
| `statistics/`  | statistics  | Statistical metrics, tests, bootstrap       |
| `integration/` | integration | Smoke scenarios & report formatting         |
| `helpers/`     | (none)      | Helper utilities (data loading, assertions) |

Markers are declared in `pyproject.toml` and enforced with `--strict-markers`.

## Test Framework

The test suite uses **pytest as the runner** with **unittest.TestCase as the base class** (via `RewardSpaceTestBase`).

### Hybrid Approach Rationale

This design provides:

- **pytest features**: Rich fixture system, parametrization, markers, and selective execution
- **unittest assertions**: Familiar assertion methods (`assertAlmostEqual`, `assertFinite`, `assertLess`, etc.)
- **Custom assertions**: Project-specific helpers (e.g., `assert_component_sum_integrity`) built on unittest base
- **Backward compatibility**: Gradual migration path from pure unittest

### Base Class

All test classes inherit from `RewardSpaceTestBase` (defined in `test_base.py`):

```python
from ..test_base import RewardSpaceTestBase

class TestMyFeature(RewardSpaceTestBase):
    def test_something(self):
        self.assertFinite(value)  # unittest-style assertion
```

### Markers

Module-level markers are declared via `pytestmark`:

```python
import pytest

pytestmark = pytest.mark.components
```

Individual tests can add additional markers:

```python
@pytest.mark.smoke
def test_quick_check(self):
    ...
```

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

- ID: Stable identifier (`<category>-<shortname>-NNN`).
- Category: Taxonomy directory marker.
- Description: Concise invariant statement.
- Owning File: Path:line of primary declaration (prefer comment line `# Owns invariant:` when present; otherwise docstring line).
- Notes: Clarifications (sub-modes, extensions, non-owning references elsewhere, line clusters for multi-path coverage).

| ID                                           | Category    | Description                                                                         | Owning File                               | Notes                                                                                                                       |
| -------------------------------------------- | ----------- | ----------------------------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| report-abs-shaping-line-091                  | integration | Abs Σ Shaping Reward line present & formatted                                       | integration/test_report_formatting.py:4   | Module docstring; primary test at line 84. PBRS report may render line; formatting owned here (core assertion lines 84–103) |
| report-additives-deterministic-092           | components  | Additives deterministic report section                                              | components/test_additives.py:4            | Integration/PBRS may reference outcome non-owning                                                                           |
| robustness-decomposition-integrity-101       | robustness  | Single active core component equals total reward under mutually exclusive scenarios | robustness/test_robustness.py:36          | Scenarios: idle, hold, exit, invalid; non-owning refs integration/test_reward_calculation.py                                |
| robustness-exit-mode-fallback-102            | robustness  | Unknown exit_attenuation_mode falls back to linear w/ warning                       | robustness/test_robustness.py:525         | Comment line (function at :526)                                                                                             |
| robustness-negative-grace-clamp-103          | robustness  | Negative exit_plateau_grace clamps to 0.0 w/ warning                                | robustness/test_robustness.py:555         |                                                                                                                             |
| robustness-invalid-power-tau-104             | robustness  | Invalid power tau falls back alpha=1.0 w/ warning                                   | robustness/test_robustness.py:592         |                                                                                                                             |
| robustness-near-zero-half-life-105           | robustness  | Near-zero half life yields no attenuation (factor≈base)                             | robustness/test_robustness.py:621         |                                                                                                                             |
| pbrs-canonical-drift-correction-106          | pbrs        | Canonical drift correction enforces near zero-sum shaping                           | pbrs/test_pbrs.py:449                     | Multi-path: extension fallback (475), comparison path (517)                                                                 |
| pbrs-canonical-near-zero-report-116          | pbrs        | Canonical near-zero cumulative shaping classification                               | pbrs/test_pbrs.py:748                     | Full report classification                                                                                                  |
| statistics-partial-deps-skip-107             | statistics  | skip_partial_dependence => empty PD structures                                      | statistics/test_statistics.py:28          | Docstring line                                                                                                              |
| helpers-duplicate-rows-drop-108              | helpers     | Duplicate rows dropped w/ warning counting removals                                 | helpers/test_utilities.py:26              | Docstring line                                                                                                              |
| helpers-missing-cols-fill-109                | helpers     | Missing required columns filled with NaN + single warning                           | helpers/test_utilities.py:50              | Docstring line                                                                                                              |
| statistics-binned-stats-min-edges-110        | statistics  | <2 bin edges raises ValueError                                                      | statistics/test_statistics.py:45          | Docstring line                                                                                                              |
| statistics-constant-cols-exclusion-111       | statistics  | Constant columns excluded & listed                                                  | statistics/test_statistics.py:57          | Docstring line                                                                                                              |
| statistics-degenerate-distribution-shift-112 | statistics  | Degenerate dist: zero shift metrics & KS p=1.0                                      | statistics/test_statistics.py:74          | Docstring line                                                                                                              |
| statistics-constant-dist-widened-ci-113a     | statistics  | Non-strict: widened CI with warning                                                 | statistics/test_statistics.py:533         | Test docstring labels "Invariant 113 (non-strict)"                                                                          |
| statistics-constant-dist-strict-omit-113b    | statistics  | Strict: omit metrics (no widened CI)                                                | statistics/test_statistics.py:565         | Test docstring labels "Invariant 113 (strict)"                                                                              |
| statistics-fallback-diagnostics-115          | statistics  | Fallback diagnostics constant distribution (qq_r2=1.0 etc.)                         | statistics/test_statistics.py:190         | Docstring line                                                                                                              |
| robustness-exit-pnl-only-117                 | robustness  | Only exit actions have non-zero PnL                                                 | robustness/test_robustness.py:126         | Newly assigned ID (previously unnumbered)                                                                                   |
| pbrs-absence-shift-placeholder-118           | pbrs        | Placeholder shift line present (absence displayed)                                  | pbrs/test_pbrs.py:979                     | Ensures placeholder appears when shaping shift absent                                                                       |
| components-pbrs-breakdown-fields-119         | components  | PBRS breakdown fields finite and mathematically aligned                             | components/test_reward_components.py:454  | Tests base_reward, pbrs_delta, invariance_correction fields and their alignment                                             |
| integration-pbrs-metrics-section-120         | integration | PBRS Metrics section present in report with tracing metrics                         | integration/test_report_formatting.py:156 | Verifies PBRS Metrics (Tracing) subsection rendering in statistical_analysis.md                                             |
| cli-pbrs-csv-columns-121                     | cli         | PBRS columns in reward_samples.csv when shaping enabled                             | cli/test_cli_params_and_csv.py:240        | Ensures reward_base, reward_pbrs_delta, reward_invariance_correction columns exist and contain finite values                |

### Non-Owning Smoke / Reference Checks

Files that reference invariant outcomes (formatting, aggregation) without owning the invariant must include a leading comment:

```python
# Non-owning smoke; ownership: <owning file>
```

Table tracks approximate line ranges and source ownership:

| File                                   | Lines (approx) | References                                               | Ownership Source                                                    |
| -------------------------------------- | -------------- | -------------------------------------------------------- | ------------------------------------------------------------------- |
| integration/test_reward_calculation.py | 15-22          | Decomposition identity (sum components)                  | robustness/test_robustness.py:36                                    |
| components/test_reward_components.py   | 212-242        | Exit factor finiteness & plateau behavior                | robustness/test_robustness.py:156+                                  |
| pbrs/test_pbrs.py                      | 591-630        | Canonical vs non-canonical classification formatting     | robustness/test_robustness.py:36, robustness/test_robustness.py:126 |
| pbrs/test_pbrs.py                      | 616,624,799    | Abs Σ Shaping Reward line formatting                     | integration/test_report_formatting.py:84-103                        |
| pbrs/test_pbrs.py                      | 742-806        | Canonical near-zero cumulative shaping classification    | robustness/test_robustness.py:36                                    |
| pbrs/test_pbrs.py                      | 807-860        | Canonical warning classification (Σ shaping > tolerance) | robustness/test_robustness.py:36                                    |
| pbrs/test_pbrs.py                      | 861-915        | Non-canonical full report reason aggregation             | robustness/test_robustness.py:36                                    |
| pbrs/test_pbrs.py                      | 916-969        | Non-canonical mode-only reason (additives disabled)      | robustness/test_robustness.py:36                                    |
| statistics/test_statistics.py          | 292            | Mean decomposition consistency                           | robustness/test_robustness.py:36                                    |

### Deprecated / Reserved IDs

| ID  | Status     | Rationale                                                             |
| --- | ---------- | --------------------------------------------------------------------- |
| 093 | deprecated | CLI invariance consolidated; no dedicated test yet                    |
| 094 | deprecated | CLI encoding/data migration removed in refactor                       |
| 095 | deprecated | Report CLI propagation assertions merged into test_cli_params_and_csv |
| 114 | reserved   | Gap retained for potential future statistics invariant                |

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

## Coverage Parity Notes

Detailed assertions reside in targeted directories (components, robustness) while integration tests focus on report formatting. Ownership IDs (e.g. 091–095, 106) reflect current scope (multi-path when noted).

## When to Run Tests

Run after changes to: reward component logic, PBRS mechanics, CLI parsing/output, statistical routines, dependency or Python version upgrades, or before publishing analysis reliant on invariants.

---

This README is the single authoritative source for test coverage, invariant ownership, smoke policies, and maintenance guidelines.
