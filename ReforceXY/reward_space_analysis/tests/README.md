# Tests: Reward Space Analysis

Authoritative documentation for invariant ownership, taxonomy layout, smoke
policies, maintenance workflows, and full coverage mapping.

## Purpose

The suite enforces:

- Reward component mathematics & transform correctness
- PBRS shaping mechanics (canonical exit semantics, near-zero classification)
- Robustness under extreme / invalid parameter settings
- Statistical metrics integrity (bootstrap, constant distributions)
- CLI input validation and artifact lifecycle
- Cross-component smoke scenarios

Single ownership per invariant is tracked in the Coverage Mapping section of
this README.

## Taxonomy Directories

| Directory      | Marker      | Scope                                         |
| -------------- | ----------- | --------------------------------------------- |
| `components/`  | components  | Component math                                |
| `transforms/`  | transforms  | Mathematical transform functions              |
| `robustness/`  | robustness  | Edge cases, stability, progression            |
| `api/`         | api         | Public API helpers & parsing                  |
| `cli/`         | cli         | CLI parameter propagation & artifacts         |
| `pbrs/`        | pbrs        | Potential-based shaping invariance & modes    |
| `statistics/`  | statistics  | Statistical metrics, tests, bootstrap         |
| `integration/` | integration | Cross-component reward behavior and workflows |
| `helpers/`     | (none)      | Helper utilities (data loading, assertions)   |

Markers are declared in `pyproject.toml` and enforced with `--strict-markers`.

## Test Framework

The suite runs under **pytest**. Most class-based tests inherit from
`RewardSpaceTestBase` (a `unittest.TestCase` subclass); others are standalone
pytest functions.

### Hybrid Approach Rationale

This design provides:

- **pytest features**: Rich fixture system, parametrization, markers, and
  selective execution
- **unittest assertions**: Familiar assertion methods (`assertAlmostEqual`,
  `assertFinite`, `assertLess`, etc.)
- **Custom assertions**: Project-specific helpers (e.g.,
  `assert_component_sum_integrity`) built on unittest base

### Base Class

Class-based tests normally inherit from `RewardSpaceTestBase` (in `test_base.py`):

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

The template gives field guidelines and a real canonical-exit docstring example,
including PBRS mode and trajectory limits.

### Markers

Module-level markers are declared via `pytestmark`:

```python
import pytest

pytestmark = pytest.mark.components
```

Individual tests can add additional markers:

```python
@pytest.mark.smoke
def test_quick_check(self): ...
```

## Running Tests

Full suite (coverage ≥85% enforced):

```shell
uv run --locked --extra dev pytest
```

Selective markers:

```shell
uv run --locked --extra dev pytest -m pbrs -q
uv run --locked --extra dev pytest -m robustness -q
uv run --locked --extra dev pytest -m "components or robustness" -q
uv run --locked --extra dev pytest -m "not slow" -q
```

Coverage reports:

```shell
uv run --locked --extra dev pytest --cov=reward_space_analysis --cov-report=term-missing
uv run --locked --extra dev pytest --cov=reward_space_analysis --cov-report=html && open htmlcov/index.html
```

Slow statistical tests:

```shell
uv run --locked --extra dev pytest -m "statistics and slow" -q
```

## Coverage Mapping (Invariant Ownership)

Columns:

- ID: Stable identifier (`<category>-<shortname>-NNN`; optional letter for
  split invariants).
- Category: Taxonomy directory marker.
- Description: Concise invariant statement.
- Owning test: Relative source path and function name (`path.py::test_name`),
  stable across unrelated line insertions.
- Notes: Sub-modes, non-owning references and multi-path coverage.

| ID                                            | Category   | Description                                                                                    | Owning test                                                                                            | Notes                                                                                                                                                                          |
| --------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| report-additives-deterministic-092            | components | Additives deterministic report section                                                         | components/test_additives.py::test_additive_activation_deterministic_contribution                      | Integration/PBRS may reference outcome non-owning                                                                                                                              |
| robustness-decomposition-integrity-101        | robustness | Single active core component equals total reward under mutually exclusive scenarios            | robustness/test_robustness.py::test_decomposition_integrity                                            | Scenarios: idle, hold, exit, invalid; non-owning refs integration/test_reward_calculation.py                                                                                   |
| robustness-exit-mode-fallback-102             | robustness | Unknown exit_attenuation_mode falls back to linear w/ warning                                  | robustness/test_robustness.py::test_robustness_102_unknown_exit_mode_fallback_linear                   |                                                                                                                                                                                |
| robustness-negative-grace-clamp-103           | robustness | Negative exit_plateau_grace clamps to 0.0 w/ warning                                           | robustness/test_robustness.py::test_robustness_103_negative_plateau_grace_clamped                      |                                                                                                                                                                                |
| robustness-invalid-power-tau-104              | robustness | Invalid power tau falls back alpha=1.0 w/ warning                                              | robustness/test_robustness.py::test_robustness_104_invalid_power_tau_fallback_alpha_one                |                                                                                                                                                                                |
| robustness-near-zero-half-life-105            | robustness | Near-zero half life yields no attenuation (factor≈base)                                        | robustness/test_robustness.py::test_robustness_105_half_life_near_zero_fallback                        |                                                                                                                                                                                |
| pbrs-canonical-exit-semantic-106              | pbrs       | Canonical exit uses shaping=-prev_potential and next_potential=0.0                             | pbrs/test_pbrs.py::test_exit_step_shaping_matches_exit_step_rules                                      | Uses stored potential across steps; no drift correction applied                                                                                                                |
| statistics-partial-deps-skip-107              | statistics | skip_partial_dependence => empty PD structures                                                 | statistics/test_statistics.py::test_statistics_feature_analysis_skip_partial_dependence                |                                                                                                                                                                                |
| helpers-transitions-preserve-multiplicity-108 | helpers    | Repeated transitions retain their empirical multiplicity                                       | helpers/test_utilities.py::test_repeated_transitions_preserve_multiplicity                             |                                                                                                                                                                                |
| helpers-missing-cols-fill-109                 | helpers    | Missing required columns filled with NaN + single warning                                      | helpers/test_utilities.py::test_missing_multiple_required_columns_single_warning                       |                                                                                                                                                                                |
| statistics-binned-stats-min-edges-110         | statistics | <2 bin edges raises ValueError                                                                 | statistics/test_statistics.py::test_statistics_binned_stats_invalid_bins_raises                        | Docstring line                                                                                                                                                                 |
| statistics-constant-cols-exclusion-111        | statistics | Constant columns excluded & listed                                                             | statistics/test_statistics.py::test_statistics_correlation_dropped_constant_columns                    | Docstring line                                                                                                                                                                 |
| statistics-degenerate-distribution-shift-112  | statistics | Constants: zero distances; KS p only with declared independent observations                    | statistics/test_statistics.py::test_statistics_distribution_shift_metrics_degenerate_zero              | Docstring line                                                                                                                                                                 |
| statistics-constant-dist-exact-ci-113a        | statistics | Both modes retain exact constant CI bounds                                                     | statistics/test_statistics.py::test_stats_bootstrap_constant_distribution_exact_bounds                 |                                                                                                                                                                                |
| statistics-percentile-outside-mean-113b       | statistics | Percentile bounds need not contain the sample mean                                             | statistics/test_statistics.py::test_stats_bootstrap_percentiles_need_not_contain_mean                  |                                                                                                                                                                                |
| statistics-constant-diagnostics-115           | statistics | Constants have N/A higher moments, normality tests and Q-Q fits in both modes                  | statistics/test_statistics.py::test_statistics_distribution_constant_diagnostics                       |                                                                                                                                                                                |
| pbrs-canonical-near-zero-report-116           | pbrs       | Canonical trajectories with valid evidence are classified as verified                          | pbrs/test_pbrs.py::test_pbrs_canonical_near_zero_report                                                | Requires local identity, continuity, discounted terminal boundary, and zero observed additives; the non-owning boundary test also covers a complete singleton terminal episode |
| robustness-exit-pnl-only-117                  | robustness | Only exit actions have non-zero PnL                                                            | robustness/test_robustness.py::test_pnl_invariant_exit_only                                            |                                                                                                                                                                                |
| pbrs-absence-shift-placeholder-118            | pbrs       | Placeholder shift line present when shaping shift is absent                                    | pbrs/test_pbrs.py::test_pbrs_absence_and_distribution_shift_placeholder                                |                                                                                                                                                                                |
| components-pbrs-breakdown-fields-119          | components | PBRS breakdown fields finite and mathematically aligned                                        | components/test_reward_components.py::test_pbrs_breakdown_fields_finite_and_aligned                    | Tests base_reward, pbrs_delta and invariance_correction alignment                                                                                                              |
| cli-pbrs-csv-columns-121                      | cli        | PBRS columns in reward_samples.csv when shaping enabled                                        | cli/test_cli_params_and_csv.py::test_csv_contains_pbrs_columns_when_shaping_present                    | Verifies finite reward_base, reward_pbrs_delta and reward_invariance_correction values                                                                                         |
| statistics-bh-finite-family-122               | statistics | Undefined tests excluded from finite-only BH family; marked non-applicable                     | statistics/test_statistics.py::test_bh_excludes_undefined_tests_from_finite_family                     |                                                                                                                                                                                |
| statistics-independence-contract-123          | statistics | Inferential helpers require independent_observations=True                                      | statistics/test_statistics.py::test_inference_helpers_require_independent_observations                 | Covers hypothesis tests and bootstrap intervals                                                                                                                                |
| pbrs-discounted-evidence-125                  | pbrs       | Verification requires local identity, continuity and discounted terminal boundary              | pbrs/test_pbrs.py::test_pbrs_canonical_discontinuous_potentials_report                                 | Discontinuous potentials are not verified                                                                                                                                      |
| statistics-proportional-histograms-126        | statistics | Proportional histograms ignore sample count; moved mass yields positive KL/JS                  | statistics/test_statistics.py::test_distribution_shift_proportional_histograms_ignore_sample_count     | KL and JS remain finite and non-negative                                                                                                                                       |
| pbrs-near-bound-clamp-127                     | pbrs       | Relaxed near-bound clamps apply exact endpoints and retain all reasons                         | pbrs/test_pbrs.py::test_validate_reward_parameters_records_near_bound_clamps_exactly                   | Includes numeric-string coercion                                                                                                                                               |
| pbrs-exit-mode-validation-128                 | pbrs       | Exit-potential choices are strict or canonicalized; direct calls fail safe                     | pbrs/test_pbrs.py::test_invalid_exit_mode_warns_at_direct_and_simulation_boundaries                    | Direct calculation and simulation boundaries warn before fallback; PBRS calls suppress additives                                                                               |
| cli-invalid-exit-mode-129                     | cli        | Invalid `--params exit_potential_mode` fails before artifacts                                  | cli/test_cli_params_and_csv.py::test_invalid_exit_potential_mode_params_fails_before_artifacts         | Strict CLI validation                                                                                                                                                          |
| cli-warning-header-recognition-130            | cli        | Warning counts accept only anchored Python warning header formats                              | cli/test_cli_params_and_csv.py::test_warning_header_positive_and_negative_formats                      | Covers POSIX, relative, synthetic and Windows source locations                                                                                                                 |
| pbrs-invalid-mode-provenance-131              | pbrs       | Invalid imported exit-mode metadata cannot certify canonical invariance                        | pbrs/test_pbrs.py::test_report_rejects_invalid_exit_mode_provenance                                    | Preserves the invalid raw value and reports effective additive settings as unknown                                                                                             |
| pbrs-synthetic-fee-floor-132                  | pbrs       | High-fee entry loss remains in long/short synthetic PnL and exit rewards                       | pbrs/test_pbrs.py::test_synthetic_fee_loss_extrema_match_retained_pnl                                  | Direct and transformed trajectories                                                                                                                                            |
| pbrs-synthetic-profitable-mark-133            | pbrs       | Favorable long/short marks remain profitable after high-fee synthetic transformation           | pbrs/test_pbrs.py::test_synthetic_high_fee_winner_retains_profitable_mark                              | Covers immediate gains and recovery after an initial loss on the independent sampled market path                                                                               |
| pbrs-synthetic-fee-boundary-134               | pbrs       | Fee-boundary rounding is accepted while materially extreme PnL is rejected                     | pbrs/test_pbrs.py::test_synthetic_high_fee_short_boundary_rejects_real_excess                          | Non-unit entry price exposes floating-point roundoff                                                                                                                           |
| pbrs-synthetic-latent-price-135               | pbrs       | Clipped candidate PnL does not erase the sampled market path for later holds                   | pbrs/test_pbrs.py::test_unrealized_pnl_retains_sampled_market_path_after_candidate_cap                 | Equal first retained marks from distinct market prices diverge after the same adverse return                                                                                   |
| api-unmasked-invalid-actions-136              | api        | Unmasked invalid actions keep positions and receive penalties                                  | api/test_api_helpers.py::test_unmasked_simulation_samples_invalid_actions_without_changing_position    | Includes spot, futures and zero-penalty invalid actions                                                                                                                        |
| api-terminal-invalid-exit-137                 | api        | Wrong-side terminal exits liquidate the held side                                              | api/test_api_helpers.py::test_invalid_terminal_exit_liquidates_the_held_position                       | Prevents dropped terminal trades                                                                                                                                               |
| cli-stale-generated-artifacts-138             | cli        | Skipped analyses remove only reserved outputs from an earlier run                              | cli/test_cli_params_and_csv.py::test_skipped_analysis_removes_only_stale_generated_artifacts           | Preserves unrelated files                                                                                                                                                      |
| helpers-nonfinite-episodes-139                | helpers    | Infinite real observations become missing without dropping rows                                | helpers/test_utilities.py::test_nonfinite_numeric_episodes_are_marked_missing                          | Retains transition multiplicity                                                                                                                                                |
| statistics-finite-shift-140                   | statistics | Finite observations remain comparable despite an infinite value                                | statistics/test_statistics.py::test_distribution_shift_uses_finite_observations                        | Rejects all-nonfinite features                                                                                                                                                 |
| statistics-rank-direction-141                 | statistics | Rank-biserial effect follows the named first-group advantage                                   | statistics/test_statistics.py::test_pnl_rank_biserial_direction_matches_named_first_group              | Checks both directions                                                                                                                                                         |
| statistics-bootstrap-count-142                | statistics | Zero and negative resample counts fail for variable and constant data                          | statistics/test_statistics.py::test_bootstrap_rejects_nonpositive_resample_count                       | Rejects missing bootstrap                                                                                                                                                      |
| api-unmasked-sample-probabilities-144         | api        | Unmasked sample probabilities match marginal valid-action frequencies                          | api/test_api_helpers.py::test_unmasked_sampling_probabilities_match_action_frequencies                 | Spot/futures entries, long/short exits, and neutral probability                                                                                                                |
| statistics-report-inference-projection-146    | statistics | Constant confidence intervals require declared independence, even with an empty hypothesis map | statistics/test_statistics.py::test_report_projects_constant_intervals_only_with_declared_independence | Numeric projection; True → False → default rewrites the same report; non-owning bounds reference: statistics-constant-dist-exact-ci-113a                                       |
| statistics-report-feature-availability-147    | statistics | Complete importance exports require a fitted model and finite means and standard deviations    | statistics/test_statistics.py::test_report_certifies_feature_importance_only_from_finite_estimates     | Real finite-zero, single-feature, empty-feature and one-test-row states; numerical CSV and availability.                                                                       |

| cli-effective-reward-boundaries-145 | cli | Effective profit-target fallback and explicit zero idle denominator reach artifacts | cli/test_cli_params_and_csv.py::test_zero_target_and_idle_cap_reach_rewards_and_manifest | Metadata and near-target market coverage agree with runtime boundaries |
| pbrs-native-integer-durations-150 | pbrs | Native integer durations and adjustment metadata retain exact values | pbrs/test_pbrs.py::test_native_integer_durations_preserve_exact_values | Python, signed NumPy and unsigned NumPy integers; strict and relaxed |
| pbrs-native-duration-bounds-151 | pbrs | Floating duration bounds are checked before truncation without narrowing | pbrs/test_pbrs.py::test_native_duration_bounds_precede_truncation | Extended-precision cases run when supported by the platform |
| pbrs-native-gamma-reward-148 | pbrs | Finite out-of-range native gamma clamps reach the public reward | pbrs/test_pbrs.py::test_native_finite_gamma_clamps_the_public_reward | Extended-range cases run when supported by the platform |
| pbrs-relaxed-native-gamma-149 | pbrs | Relaxed native gamma clamps reach rewards while strict validation rejects | pbrs/test_pbrs.py::test_relaxed_native_gamma_clamps_before_reward_calculation | Preserves direct, strict and relaxed boundary contracts |

### Non-Owning Smoke / Reference Checks

Tests that check an invariant owned elsewhere identify its owner in a comment
or docstring, for example:

```python
# Non-owning smoke; ownership: <owning file>
```

The following tests also check outcomes owned elsewhere:

| Non-owning test                                                                                        | Reference check                             | Owning test                                                                                              |
| ------------------------------------------------------------------------------------------------------ | ------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| integration/test_reward_calculation.py::test_reward_component_activation_smoke                         | Core component activation and decomposition | robustness/test_robustness.py::test_decomposition_integrity                                              |
| components/test_reward_components.py::test_exit_factor_calculation                                     | Exit factor modes and plateau behavior      | robustness/test_robustness.py::test_exit_factor_comprehensive; test_plateau_continuity_at_grace_boundary |
| pbrs/test_pbrs.py::test_pbrs_canonical_near_zero_report                                                | Canonical classification and decomposition  | robustness/test_robustness.py::test_decomposition_integrity                                              |
| pbrs/test_pbrs.py::test_pbrs_non_canonical_full_report_reason_aggregation                              | Non-canonical report reasons                | robustness/test_robustness.py::test_decomposition_integrity                                              |
| pbrs/test_pbrs.py::test_pbrs_non_canonical_mode_only_reason                                            | Non-canonical exit mode without additives   | robustness/test_robustness.py::test_decomposition_integrity                                              |
| statistics/test_statistics.py::test_stats_mean_decomposition_consistency                               | Mean decomposition consistency              | robustness/test_robustness.py::test_decomposition_integrity                                              |
| statistics/test_statistics.py::test_report_projects_constant_intervals_only_with_declared_independence | Exact point-mass interval bounds            | statistics/test_statistics.py::test_stats_bootstrap_constant_distribution_exact_bounds                   |

### Deprecated / Reserved IDs

| ID  | Status     | Rationale                                                                                                      |
| --- | ---------- | -------------------------------------------------------------------------------------------------------------- |
| 093 | deprecated | CLI invariance consolidated; no dedicated test yet                                                             |
| 094 | deprecated | CLI encoding/data migration removed in refactor                                                                |
| 095 | deprecated | Report CLI propagation assertions merged into test_cli_params_and_csv                                          |
| 114 | reserved   | Gap retained for potential future statistics invariant                                                         |
| 120 | reserved   | No active behavior invariant; prose-only report checks are not ownership.                                      |
| 124 | reserved   | No prose owner; numeric confidence-interval projection is owned by statistics-report-inference-projection-146. |

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

Use the **complete invariant ID** and its Coverage Mapping row, not a shortname
or the number of matching directory paths. From the repository root:

```shell
cd ReforceXY/reward_space_analysis/tests
invariant=pbrs-canonical-exit-semantic-106
grep -R -n -F --include='*.py' -- "$invariant" .
```

Resolve the mapped owner
`pbrs/test_pbrs.py::test_exit_step_shaping_matches_exit_step_rules` and inspect
its actual assertions. Inline IDs/ownership declarations are optional in the
existing suite, so no search hit (grep status 1) does not invalidate that mapping.
Classify each hit as an owning declaration or non-owning reference; documentation
mentions and reserved entries are not additional owners.

Check one registered mapping owner per active ID, the owner's behavior against
the invariant statement, and any competing ownership declarations. Multi-path
coverage and non-owning smoke checks are allowed when identified by the mapping,
its notes or the Non-Owning Smoke / Reference Checks table. A shortname such as
`near_zero` or `pbrs_delta` spans unrelated IDs and cannot prove duplication.
Do not delete a valid reference because it is in another taxonomy directory.

## Coverage Parity Notes

Detailed numerical assertions reside in their targeted directories. Integration
checks exercise cross-component behavior; matching report text is not numerical
correctness. The mapping defines current ownership and allowed non-owning or
multi-path coverage.

Real runtime/reference parity is owned by
`ReforceXY/tests/test_model_pbrs_transitions.py::PbrsTransitionsTest::test_analysis_matches_real_rewards_across_modes_and_parameter_boundaries`.
Native-precision rejection of out-of-range analytical discounts is owned by
`ReforceXY/tests/test_model_pbrs_transitions.py::PbrsTransitionsTest::test_analysis_rejects_gamma_outside_bounds_before_float_rounding`.
Run these tests in the Freqtrade RL QA image through the canonical runtime suite; the
standalone analysis environment deliberately does not require Freqtrade/SB3.
The comparator exercises direct/strict/relaxed parameter paths, real fills and
terminal liquidation, carries reference potential independently, and checks tiny
shaping components more tightly than the report-verification tolerance.
Raw random shaping sums are not discounted invariance oracles. Complete-episode
discounted identities and report continuity checks retain their existing owners.

## When to Run Tests

Run after changes to: reward component logic, PBRS mechanics, CLI
parsing/output, statistical routines, dependency or Python version upgrades, or
before publishing analysis reliant on invariants.

## Additional Resources

- **`.docstring_template.md`** - Standardized test documentation template with
  examples for minimal, standard, and complex tests
- **`constants.py`** - Single source of truth for all test constants (frozen
  dataclasses with comprehensive documentation)
- **`helpers/assertions.py`** - Custom assertion helpers for invariant validation
- **`test_base.py`** - Base class with common utilities (`make_ctx`, `seed_all`,
  etc.)

---

This README is the single authoritative source for test coverage, invariant
ownership, smoke policies, and maintenance guidelines.
