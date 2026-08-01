# Tests: Reward Space Analysis

Authoritative documentation for invariant ownership, taxonomy layout, smoke
policies, maintenance workflows, and full coverage mapping.

## Purpose

The suite enforces the analytical clone's current economic contract:

- pair-local net log liquidation returns with a positive liquidation-value floor;
- unified Freqtrade-compatible entry and exit fee primitives;
- exact `economic + invalid + canonical PBRS` reward decomposition;
- economic ruin as termination and finite sample exhaustion as truncation;
- canonical PBRS termwise correction and discounted telescoping diagnostics;
- CLI, CSV, report, and manifest propagation of the economic contract;
- robustness and statistical diagnostics around that contract.

The raw sum of shaping rewards is descriptive only. For discount
`gamma < 1`, PBRS verification uses the termwise correction and the discounted
identity `sum(gamma^t F_t) = -Phi(s_0) + gamma^T Phi(s_T)`.

Legacy idle, hold, exit-attenuation, efficiency, and additive helpers remain
covered only as compatibility surfaces. They are not active components of the
economic reward.

Single ownership per invariant is tracked in the Coverage Mapping section of
this README.

## Taxonomy Directories

| Directory      | Marker      | Scope                                       |
| -------------- | ----------- | ------------------------------------------- |
| `components/`  | components  | Economic decomposition and compatibility math |
| `transforms/`  | transforms  | Mathematical transform functions            |
| `robustness/`  | robustness  | Numerical stability, ruin, and boundaries   |
| `api/`         | api         | Public API helpers & parsing                |
| `cli/`         | cli         | Economic contract propagation and artifacts |
| `pbrs/`        | pbrs        | Canonical shaping, fees, and telescoping     |
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
| report-raw-shaping-diagnostic-091            | integration | Raw shaping sum is explicitly descriptive and never an invariance classifier        | integration/test_report_formatting.py:4   | Canonical classification is owned by invariant 116                                                           |
| components-canonical-additives-092           | components  | Strict canonical PBRS rejects additives; relaxed validation suppresses them          | components/test_additives.py:4            | Additive fields remain zero-valued compatibility outputs                                                     |
| robustness-economic-decomposition-101       | robustness  | Total reward equals economic return plus invalid penalty and canonical PBRS          | robustness/test_robustness.py:44          | Covers neutral, hold, exit, and invalid transitions                                                          |
| robustness-exit-mode-fallback-102            | robustness  | Unknown exit_attenuation_mode falls back to linear w/ warning                       | robustness/test_robustness.py:654         | Comment line (function at :655)                                                                              |
| robustness-negative-grace-clamp-103          | robustness  | Negative exit_plateau_grace clamps to 0.0 w/ warning                                | robustness/test_robustness.py:696         |                                                                                                              |
| robustness-invalid-power-tau-104             | robustness  | Invalid power tau falls back alpha=1.0 w/ warning                                   | robustness/test_robustness.py:747         |                                                                                                              |
| robustness-near-zero-half-life-105           | robustness  | Near-zero half life yields no attenuation (factor≈base)                             | robustness/test_robustness.py:792         |                                                                                                              |
| pbrs-canonical-exit-semantic-106             | pbrs        | Canonical exit uses shaping=-prev_potential and next_potential=0.0                  | pbrs/test_pbrs.py:374                     | Uses stored potential across steps; no drift correction applied                                              |
| pbrs-canonical-correction-report-116         | pbrs        | Termwise correction and discounted telescoping verify canonical PBRS                 | pbrs/test_pbrs.py:1183                    | A non-zero raw shaping sum is allowed and deliberately exercised                                            |
| statistics-partial-deps-skip-107             | statistics  | skip_partial_dependence => empty PD structures                                      | statistics/test_statistics.py:42          | Docstring line                                                                                               |
| helpers-duplicate-rows-drop-108              | helpers     | Duplicate rows dropped w/ warning counting removals                                 | helpers/test_utilities.py:27              | Docstring line                                                                                               |
| helpers-missing-cols-fill-109                | helpers     | Missing required columns filled with NaN + single warning                           | helpers/test_utilities.py:51              | Docstring line                                                                                               |
| statistics-binned-stats-min-edges-110        | statistics  | <2 bin edges raises ValueError                                                      | statistics/test_statistics.py:60          | Docstring line                                                                                               |
| statistics-constant-cols-exclusion-111       | statistics  | Constant columns excluded & listed                                                  | statistics/test_statistics.py:71          | Docstring line                                                                                               |
| statistics-degenerate-distribution-shift-112 | statistics  | Degenerate dist: zero shift metrics & KS p=1.0                                      | statistics/test_statistics.py:87          | Docstring line                                                                                               |
| statistics-constant-dist-widened-ci-113a     | statistics  | Non-strict: widened CI with warning                                                 | statistics/test_statistics.py:551         | Test docstring labels "Invariant 113 (non-strict)"                                                           |
| statistics-constant-dist-strict-omit-113b    | statistics  | Strict: omit metrics (no widened CI)                                                | statistics/test_statistics.py:583         | Test docstring labels "Invariant 113 (strict)"                                                               |
| statistics-fallback-diagnostics-115          | statistics  | Fallback diagnostics constant distribution (qq_r2=1.0 etc.)                         | statistics/test_statistics.py:191         | Docstring line                                                                                               |
| robustness-episode-boundaries-117            | robustness  | Neutral PnL, mark carry, ruin termination, and finite-horizon truncation are coherent | robustness/test_robustness.py:86          | Final sample is exactly one of terminated or truncated                                                       |
| pbrs-unverified-shift-placeholder-118        | pbrs        | Missing termwise correction is Unverified and shift absence remains explicit        | pbrs/test_pbrs.py:1499                    | Prevents a raw shaping sum from being promoted to an invariance proof                                        |
| components-economic-pbrs-breakdown-119       | components  | Economic base, PBRS delta, and correction fields are finite and aligned              | components/test_reward_components.py:625  | Canonical correction is zero within numerical tolerance                                                     |
| integration-pbrs-metrics-section-120         | integration | PBRS tracing metrics are rendered in the statistical report                          | integration/test_report_formatting.py:163 | Includes base, PBRS delta, correction, and magnitude ratio diagnostics                                       |
| cli-economic-contract-columns-121            | cli         | CSV exposes economic marks, decomposition, PBRS tracing, ruin, and boundary flags     | cli/test_cli_params_and_csv.py:248        | Values are finite, positive where required, and mathematically aligned                                       |
| pbrs-unified-fee-primitives-122               | pbrs        | Long and short unrealized PnL use the unified Freqtrade entry/exit fee formulas       | pbrs/test_pbrs.py:598                     | Covers exact analytical formulas for both directions                                                        |
| robustness-economic-ruin-123                  | robustness  | Economic ruin floors liquidation value, terminates, and releases PBRS potential       | robustness/test_robustness.py:244         | Extreme inputs remain finite                                                                                 |
| cli-economic-manifest-124                     | cli         | Manifest declares reward formula, fee assumptions, boundaries, and compatibility data | cli/test_cli_params_and_csv.py:103        | CLI overrides remain visible in canonical reward parameters                                                 |

### Non-Owning Smoke / Reference Checks

Files that reference invariant outcomes (formatting, aggregation) without owning
the invariant must include a leading comment:

```python
# Non-owning smoke; ownership: <owning file>
```

Table tracks approximate line ranges and source ownership:

| File                                   | Lines (approx) | References                                               | Ownership Source                                                    |
| -------------------------------------- | -------------- | -------------------------------------------------------- | ------------------------------------------------------------------- |
| integration/test_reward_calculation.py | 39             | Economic transition decomposition                       | robustness/test_robustness.py:44                  |
| components/test_reward_components.py   | 492            | Legacy exit-factor helper finiteness                     | compatibility-only helper coverage                |
| pbrs/test_pbrs.py                      | 1010-1490      | Eligible, violated, unverified, and ineligible reporting | pbrs/test_pbrs.py:1183 and :1499                   |
| pbrs/test_pbrs.py                      | 1183           | Raw shaping diagnostic formatting                       | integration/test_report_formatting.py:4            |
| statistics/test_statistics.py          | 292            | Mean economic decomposition consistency                 | robustness/test_robustness.py:44                   |

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
