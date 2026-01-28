# Tests: Reward Space Analysis

Authoritative documentation for invariant ownership, taxonomy layout, smoke
policies, maintenance workflows, and full coverage mapping.

## Purpose

The suite enforces:

- Reward component mathematics & transform correctness
- PBRS shaping mechanics (canonical exit semantics, near-zero classification)
- Robustness under extreme / invalid parameter settings
- Statistical metrics integrity (bootstrap, constant distributions)
- CLI parameter propagation & report formatting
- Cross-component smoke scenarios

Single ownership per invariant is tracked in the Coverage Mapping section of
this README.

## Taxonomy Directories

| Directory      | Marker      | Scope                                       |
| -------------- | ----------- | ------------------------------------------- |
| `components/`  | components  | Component math                              |
| `transforms/`  | transforms  | Mathematical transform functions            |
| `robustness/`  | robustness  | Edge cases, stability, progression          |
| `api/`         | api         | Public API helpers & parsing                |
| `cli/`         | cli         | CLI parameter propagation & artifacts       |
| `pbrs/`        | pbrs        | Potential-based shaping invariance & modes  |
| `statistics/`  | statistics  | Statistical metrics, tests, bootstrap       |
| `integration/` | integration | Smoke scenarios & report formatting         |
| `helpers/`     | (none)      | Helper utilities (data loading, assertions) |

Markers are declared in `pyproject.toml` and enforced with `--strict-markers`.

## Test Framework

The test suite uses **pytest as the runner** with **unittest.TestCase as the
base class** (via `RewardSpaceTestBase`).

### Hybrid Approach Rationale

This design provides:

- **pytest features**: Rich fixture system, parametrization, markers, and
  selective execution
- **unittest assertions**: Familiar assertion methods (`assertAlmostEqual`,
  `assertFinite`, `assertLess`, etc.)
- **Custom assertions**: Project-specific helpers (e.g.,
  `assert_component_sum_integrity`) built on unittest base
- **Backward compatibility**: Gradual migration path from pure unittest

### Base Class

All test classes inherit from `RewardSpaceTestBase` (defined in `test_base.py`):

```python
from ..test_base import RewardSpaceTestBase

class TestMyFeature(RewardSpaceTestBase):
    def test_something(self):
        self.assertFinite(value)  # unittest-style assertion
```

### Constants & Configuration

All test constants are centralized in `tests/constants.py` using frozen
dataclasses as a single source of truth:

```python
from tests.constants import TOLERANCE, SEEDS, PARAMS, EXIT_FACTOR

# Use directly in tests
assert abs(result - expected) < TOLERANCE.IDENTITY_RELAXED
seed_all(SEEDS.FIXED_UNIT)
```

**Key constant groups:**

- `TOLERANCE.*` - Numerical tolerances (documented in dataclass docstring)
- `SEEDS.*` - Fixed random seeds for reproducibility
- `PARAMS.*` - Standard test parameters (PnL, durations, ratios)
- `EXIT_FACTOR.*` - Exit factor scenarios
- `CONTINUITY.*` - Continuity check parameters
- `STATISTICAL.*` - Statistical test thresholds
- `EFFICIENCY.*` - Efficiency coefficient testing configuration
- `PBRS.*` - Potential-Based Reward Shaping thresholds
- `SCENARIOS.*` - Test scenario parameters and sample sizes
- `STAT_TOL.*` - Tolerances for statistical metrics

**Never use magic numbers** - add new constants to `constants.py` instead.

### Tolerance Selection

Choose appropriate numerical tolerances to prevent flaky tests. All tolerance
constants are defined and documented in `tests/constants.py` with their
rationale.

**Common tolerances:**

- `IDENTITY_STRICT` (1e-12) - Machine-precision checks
- `IDENTITY_RELAXED` (1e-09) - Multi-step operations with accumulated errors
- `GENERIC_EQ` (1e-08) - General floating-point equality (default)

Always document non-default tolerance choices with inline comments explaining
the error accumulation model.

### Test Documentation

All tests should follow the standardized docstring format in
**`.docstring_template.md`**:

- One-line summary (imperative mood)
- Invariant reference (if applicable)
- Extended description (what and why)
- Setup (parameters, scenarios, sample sizes)
- Assertions (what each validates)
- Tolerance rationale (required for non-default tolerances)
- See also (related tests/docs)

**Template provides three complexity levels** (minimal, standard, complex) with
examples for property-based tests, regression tests, and integration tests.

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
- Owning File: Path:line of primary declaration (prefer comment line
  `# Owns invariant:` when present; otherwise docstring line).
- Notes: Clarifications (sub-modes, extensions, non-owning references elsewhere,
  line clusters for multi-path coverage).

| ID                                           | Category    | Description                                                                         | Owning File                               | Notes                                                                                                        |
| -------------------------------------------- | ----------- | ----------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| report-abs-shaping-line-091                  | integration | Abs Σ Shaping Reward line present & formatted                                       | integration/test_report_formatting.py:4   | Module docstring; primary test at line 95. PBRS report may render line; formatting owned here                |
| report-additives-deterministic-092           | components  | Additives deterministic report section                                              | components/test_additives.py:4            | Integration/PBRS may reference outcome non-owning                                                            |
| robustness-decomposition-integrity-101       | robustness  | Single active core component equals total reward under mutually exclusive scenarios | robustness/test_robustness.py:43          | Scenarios: idle, hold, exit, invalid; non-owning refs integration/test_reward_calculation.py                 |
| robustness-exit-mode-fallback-102            | robustness  | Unknown exit_attenuation_mode falls back to linear w/ warning                       | robustness/test_robustness.py:654         | Comment line (function at :655)                                                                              |
| robustness-negative-grace-clamp-103          | robustness  | Negative exit_plateau_grace clamps to 0.0 w/ warning                                | robustness/test_robustness.py:696         |                                                                                                              |
| robustness-invalid-power-tau-104             | robustness  | Invalid power tau falls back alpha=1.0 w/ warning                                   | robustness/test_robustness.py:747         |                                                                                                              |
| robustness-near-zero-half-life-105           | robustness  | Near-zero half life yields no attenuation (factor≈base)                             | robustness/test_robustness.py:792         |                                                                                                              |
| pbrs-canonical-exit-semantic-106             | pbrs        | Canonical exit uses shaping=-prev_potential and next_potential=0.0                  | pbrs/test_pbrs.py:374                     | Uses stored potential across steps; no drift correction applied                                              |
| pbrs-canonical-near-zero-report-116          | pbrs        | Canonical near-zero cumulative shaping classification                               | pbrs/test_pbrs.py:1223                    | Full report classification                                                                                   |
| statistics-partial-deps-skip-107             | statistics  | skip_partial_dependence => empty PD structures                                      | statistics/test_statistics.py:42          | Docstring line                                                                                               |
| helpers-duplicate-rows-drop-108              | helpers     | Duplicate rows dropped w/ warning counting removals                                 | helpers/test_utilities.py:27              | Docstring line                                                                                               |
| helpers-missing-cols-fill-109                | helpers     | Missing required columns filled with NaN + single warning                           | helpers/test_utilities.py:51              | Docstring line                                                                                               |
| statistics-binned-stats-min-edges-110        | statistics  | <2 bin edges raises ValueError                                                      | statistics/test_statistics.py:60          | Docstring line                                                                                               |
| statistics-constant-cols-exclusion-111       | statistics  | Constant columns excluded & listed                                                  | statistics/test_statistics.py:71          | Docstring line                                                                                               |
| statistics-degenerate-distribution-shift-112 | statistics  | Degenerate dist: zero shift metrics & KS p=1.0                                      | statistics/test_statistics.py:87          | Docstring line                                                                                               |
| statistics-constant-dist-widened-ci-113a     | statistics  | Non-strict: widened CI with warning                                                 | statistics/test_statistics.py:551         | Test docstring labels "Invariant 113 (non-strict)"                                                           |
| statistics-constant-dist-strict-omit-113b    | statistics  | Strict: omit metrics (no widened CI)                                                | statistics/test_statistics.py:583         | Test docstring labels "Invariant 113 (strict)"                                                               |
| statistics-fallback-diagnostics-115          | statistics  | Fallback diagnostics constant distribution (qq_r2=1.0 etc.)                         | statistics/test_statistics.py:191         | Docstring line                                                                                               |
| robustness-exit-pnl-only-117                 | robustness  | Only exit actions have non-zero PnL                                                 | robustness/test_robustness.py:127         | Comment line                                                                                                 |
| pbrs-absence-shift-placeholder-118           | pbrs        | Placeholder shift line present (absence displayed)                                  | pbrs/test_pbrs.py:1523                    | Ensures placeholder appears when shaping shift absent                                                        |
| components-pbrs-breakdown-fields-119         | components  | PBRS breakdown fields finite and mathematically aligned                             | components/test_reward_components.py:783  | Tests base_reward, pbrs_delta, invariance_correction fields and their alignment                              |
| integration-pbrs-metrics-section-120         | integration | PBRS Metrics section present in report with tracing metrics                         | integration/test_report_formatting.py:155 | Verifies PBRS Metrics (Tracing) subsection rendering in statistical_analysis.md                              |
| cli-pbrs-csv-columns-121                     | cli         | PBRS columns in reward_samples.csv when shaping enabled                             | cli/test_cli_params_and_csv.py:221        | Ensures reward_base, reward_pbrs_delta, reward_invariance_correction columns exist and contain finite values |

### Non-Owning Smoke / Reference Checks

Files that reference invariant outcomes (formatting, aggregation) without owning
the invariant must include a leading comment:

```python
# Non-owning smoke; ownership: <owning file>
```

Table tracks approximate line ranges and source ownership:

| File                                   | Lines (approx) | References                                               | Ownership Source                                                    |
| -------------------------------------- | -------------- | -------------------------------------------------------- | ------------------------------------------------------------------- |
| integration/test_reward_calculation.py | 44             | Decomposition identity (sum components)                  | robustness/test_robustness.py:43                                    |
| components/test_reward_components.py   | 551            | Exit factor finiteness & plateau behavior                | robustness/test_robustness.py:43+                                   |
| pbrs/test_pbrs.py                      | 1053           | Canonical vs non-canonical classification formatting     | robustness/test_robustness.py:43, robustness/test_robustness.py:127 |
| pbrs/test_pbrs.py                      | 1222,1292,1415 | Abs Σ Shaping Reward line formatting                     | integration/test_report_formatting.py:95                            |
| pbrs/test_pbrs.py                      | 1222           | Canonical near-zero cumulative shaping classification    | robustness/test_robustness.py:43                                    |
| pbrs/test_pbrs.py                      | 1292           | Canonical warning classification (Σ shaping > tolerance) | robustness/test_robustness.py:43                                    |
| pbrs/test_pbrs.py                      | 1415           | Non-canonical full report reason aggregation             | robustness/test_robustness.py:43                                    |
| pbrs/test_pbrs.py                      | 1469           | Non-canonical mode-only reason (additives disabled)      | robustness/test_robustness.py:43                                    |
| statistics/test_statistics.py          | 292            | Mean decomposition consistency                           | robustness/test_robustness.py:43                                    |

### Deprecated / Reserved IDs

| ID  | Status     | Rationale                                                             |
| --- | ---------- | --------------------------------------------------------------------- |
| 093 | deprecated | CLI invariance consolidated; no dedicated test yet                    |
| 094 | deprecated | CLI encoding/data migration removed in refactor                       |
| 095 | deprecated | Report CLI propagation assertions merged into test_cli_params_and_csv |
| 114 | reserved   | Gap retained for potential future statistics invariant                |

## Adding New Invariants

1. Assign ID `<category>-<shortname>-NNN` (NNN numeric). Reserve gaps explicitly
   if needed (see deprecated/reserved table).
2. Add a row in Coverage Mapping BEFORE writing the test.
3. Implement test in correct taxonomy directory; add marker if outside default
   selection.
4. Follow the docstring template in `.docstring_template.md`.
5. Use constants from `tests/constants.py` - never use magic numbers.
6. Document tolerance choices with inline comments explaining error
   accumulation.
7. Optionally declare inline ownership:
   ```python
   # Owns invariant: <id>
   def test_<short_description>(...):
       ...
   ```
8. Run duplication audit and coverage before committing.

## Maintenance Guidelines

### Constant Management

All test constants live in `tests/constants.py`:

- Import constants directly: `from tests.constants import TOLERANCE, SEEDS`
- Never use class attributes for constants (e.g., `self.TEST_*`)
- Add new constants to appropriate dataclass in `constants.py`
- Frozen dataclasses prevent accidental modification

### Tolerance Documentation

When using non-default tolerances (anything other than `GENERIC_EQ`), add an
inline comment explaining the error accumulation:

```python
# IDENTITY_RELAXED: Exit factor involves normalization + kernel + transform
assert abs(exit_factor - expected) < TOLERANCE.IDENTITY_RELAXED
```

### Test Documentation Standards

- Follow `.docstring_template.md` for all new tests
- Include invariant IDs in docstrings when applicable
- Document Setup section with parameter choices and sample sizes
- Explain non-obvious assertions in Assertions section
- Always include tolerance rationale for non-default choices

## Duplication Audit

Each invariant shortname must appear in exactly one taxonomy directory path:

```shell
cd ReforceXY/reward_space_analysis/tests
grep -R "<shortname>" -n .
```

Expect a single directory path. Examples:

```shell
grep -R "near_zero" -n .
grep -R "pbrs_delta" -n .
```

## Coverage Parity Notes

Detailed assertions reside in targeted directories (components, robustness)
while integration tests focus on report formatting. Ownership IDs (e.g.
091–095, 106) reflect current scope (multi-path when noted).

## When to Run Tests

Run after changes to: reward component logic, PBRS mechanics, CLI
parsing/output, statistical routines, dependency or Python version upgrades, or
before publishing analysis reliant on invariants.

## Additional Resources

- **`.docstring_template.md`** - Standardized test documentation template with
  examples for minimal, standard, and complex tests
- **`constants.py`** - Single source of truth for all test constants (frozen
  dataclasses with comprehensive documentation)
- **`helpers/assertions.py`** - 20+ custom assertion functions for invariant
  validation
- **`test_base.py`** - Base class with common utilities (`make_ctx`, `seed_all`,
  etc.)

---

This README is the single authoritative source for test coverage, invariant
ownership, smoke policies, and maintenance guidelines.
