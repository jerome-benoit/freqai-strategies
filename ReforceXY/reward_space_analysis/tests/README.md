# Tests: Reward Space Analysis

Authoritative documentation for invariant ownership, taxonomy layout, smoke policies, and maintenance workflows.

## Purpose

The suite enforces:
- Reward component mathematics & transform correctness
- PBRS invariance mechanics (canonical drift correction, near-zero classification)
- Robustness under extreme / invalid parameter settings
- Statistical metrics integrity (bootstrap, constant distributions)
- CLI propagation & report formatting
- Cross-component smoke scenarios

Single ownership per invariant: see `TEST_COVERAGE_MAP.md`.

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

## Invariant Ownership Quick Index

Refer to `TEST_COVERAGE_MAP.md` for complete table. Key PBRS and robustness invariants:

| ID | Owning File |
|----|-------------|
| robustness-exit-mode-fallback-102 | robustness/test_robustness.py:520 |
| robustness-negative-grace-clamp-103 | robustness/test_robustness.py:549 |
| robustness-invalid-power-tau-104 | robustness/test_robustness.py:585 |
| robustness-near-zero-half-life-105 | robustness/test_robustness.py:613 |
| pbrs-canonical-drift-correction-106 | pbrs/test_pbrs.py:448 (primary), 474 (extension), 516 (comparison) |
| pbrs-canonical-near-zero-report-116 | pbrs/test_pbrs.py:747 |
| robustness-exit-pnl-only-117 | robustness/test_robustness.py:125 |

## Adding New Invariants

1. Assign ID `<category>-<shortname>-NNN` (NNN numeric). Reserve gaps explicitly if needed (see reserved section).
2. Add row to `TEST_COVERAGE_MAP.md` before writing a test.
3. Implement test in correct taxonomy directory; optional marker if outside auto selection.
4. Declare ownership inline if beneficial:
   ```python
   # Owns invariant: <id>
   def test_<short_description>(...):
       ...
   ```
5. Run duplication audit and coverage before committing.

## Non-Owning Smoke Policy

Smoke tests referencing invariant outcomes or formatting without asserting underlying math must include a leading comment:
```python
# Non-owning smoke; ownership: <owning file> [,<second owning file>]
```
Multiple invariants may be listed if the smoke test renders a combined report section. Line ranges tracked in the "Non-Owning Smoke / Reference Checks" table of `TEST_COVERAGE_MAP.md`.

Do not add IDs to smoke tests—IDs live only in owning tests + coverage map.

## Duplication Audit

Each invariant shortname must appear in exactly one taxonomy directory path. Prior to commit:
```shell
cd ReforceXY/reward_space_analysis/tests
grep -R "<shortname>" -n .
```
Expect a single directory path. Example patterns:
```shell
grep -R "drift_correction" -n .
grep -R "near_zero" -n .
```

## Parity & Migration Rationale

Refactor preserved semantic coverage while moving detailed assertions from integration (report formatting only) into targeted directories (components, robustness). IDs 091–095 clarified ownership; 106 spans multi-path (primary + fallback + comparison) inside a single file.

## Slow Statistical Tests

Tag long bootstrap / distribution tests with `@pytest.mark.slow` for selective runs:
```shell
uv run pytest -m "statistics and slow" -q
```

## When to Run Tests

Run after changes to: reward component logic, PBRS mechanics, CLI parsing/output, statistical routines, dependency or Python version upgrades, or before publishing analysis reliant on invariants.

---

Refer back to `TEST_COVERAGE_MAP.md` for authoritative ownership details.
