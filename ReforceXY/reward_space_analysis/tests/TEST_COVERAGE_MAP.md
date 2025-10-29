# TEST_COVERAGE_MAP

Authoritative single ownership mapping for reward-space invariants and structural checks.

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
| robustness-exit-mode-fallback-102 | robustness | Unknown exit_attenuation_mode falls back to linear w/ warning | robustness/test_robustness.py:520 | Owns invariant (numeric tag) |
| robustness-negative-grace-clamp-103 | robustness | Negative exit_plateau_grace clamps to 0.0 w/ warning | robustness/test_robustness.py:549 | Owns invariant (numeric tag) |
| robustness-invalid-power-tau-104 | robustness | Invalid power tau falls back alpha=1.0 w/ warning | robustness/test_robustness.py:585 | Owns invariant (numeric tag) |
| robustness-near-zero-half-life-105 | robustness | Near-zero half life yields no attenuation (factor≈base) | robustness/test_robustness.py:613 | Owns invariant (numeric tag) |
| pbrs-canonical-drift-correction-106 | pbrs | Canonical drift correction enforces near zero-sum shaping | pbrs/test_pbrs.py:448 | Owns invariant: zero-sum, exception fallback (474), comparison path (516) |
| pbrs-canonical-near-zero-report-116 | pbrs | Canonical near-zero cumulative shaping classification | pbrs/test_pbrs.py:747 | Owns invariant (classification canonical near-zero cumulative shaping report; see tests/README.md) |
| statistics-partial-deps-skip-107 | statistics | skip_partial_dependence => empty PD structures | statistics/test_statistics.py:28 | |
| helpers-duplicate-rows-drop-108 | helpers | Duplicate rows dropped w/ warning counting removals | helpers/test_utilities.py:26 | |
| helpers-missing-cols-fill-109 | helpers | Missing required columns filled with NaN + single warning | helpers/test_utilities.py:50 | |
| statistics-binned-stats-min-edges-110 | statistics | <2 bin edges raises ValueError |_statistics/test_statistics.py:45 | |
| statistics-constant-cols-exclusion-111 | statistics | Constant columns excluded & listed | statistics/test_statistics.py:57 | |
| statistics-degenerate-distribution-shift-112 | statistics | Degenerate dist: zero shift metrics & KS p=1.0 | statistics/test_statistics.py:74 | |
| statistics-constant-dist-widened-ci-113a | statistics | Non-strict: widened CI with warning | statistics/test_statistics.py:529 | Shares base distribution context with 113b |
| statistics-constant-dist-strict-omit-113b | statistics | Strict: omit metrics (no widened CI) | statistics/test_statistics.py:561 | Sub-mode of constant distribution invariant |
| statistics-fallback-diagnostics-115 | statistics | Fallback diagnostics constant distribution (qq_r2=1.0 etc.) | statistics/test_statistics.py:190 | |
| robustness-exit-pnl-only-117 | robustness | Only exit actions have non-zero PnL | robustness/test_robustness.py:125 | Newly assigned ID (previously unnumbered) |

## Non-Owning Smoke / Reference Checks

The following tests reference invariants but do not own them. They must include a leading comment of the form:
`# Non-owning smoke; ownership: <owning file>`

| File | Lines (approx) | References | Ownership Source |
|------|----------------|------------|------------------|
| integration/test_reward_calculation.py | 20-25 | Decomposition identity (sum components) | robustness/test_robustness.py:35 |
| components/test_reward_components.py | 214-243 | Exit factor finiteness & plateau behavior | robustness/test_robustness.py:156+ (comprehensive exit factor) |
| pbrs/test_pbrs.py | 616,624,799 | Abs Σ Shaping Reward line formatting | integration/test_report_formatting.py:87-100 |
| pbrs/test_pbrs.py | 591-630 | Non/Canonical classification formatting (multi invariance statuses) | robustness/test_robustness.py:35, robustness/test_robustness.py:125 |
| pbrs/test_pbrs.py | 742-806 | Canonical near-zero cumulative shaping classification | robustness/test_robustness.py:35 |
| pbrs/test_pbrs.py | 807-860 | Canonical warning classification (|Σ shaping| > tolerance) | robustness/test_robustness.py:35 |
| pbrs/test_pbrs.py | 861-915 | Non-canonical full report reason aggregation (mode + additives) | robustness/test_robustness.py:35 |
| pbrs/test_pbrs.py | 916-969 | Non-canonical mode-only reason (additives disabled) | robustness/test_robustness.py:35 |
| statistics/test_statistics.py | 292 | Mean decomposition consistency | robustness/test_robustness.py:35 |

## Deprecated / Reserved IDs

| ID | Status | Rationale |
|----|--------|-----------|
| 093 | deprecated | CLI invariance consolidated; no dedicated test yet |
| 094 | deprecated | CLI encoding/data migration removed in refactor |
| 095 | deprecated | Report CLI propagation assertions merged into test_cli_params_and_csv |
| 114 | reserved | Gap retained for potential future statistics invariant |

## Guidelines

1. Add new invariant row BEFORE writing test.
2. Declare ownership with either table row + (optional) inline leading comment `Owns invariant:`.
3. Non-owning references must have explicit smoke comment.
4. Run duplication audit: `grep -R "<shortname>" -n .` expecting single taxonomy directory path.

