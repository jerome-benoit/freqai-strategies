# Tolerance Selection Guide

This guide explains the rationale behind numerical tolerance choices in the test suite.

## Tolerance Constants

All tolerances are defined in `tests/constants.py` in the `ToleranceConfig` class:

```python
TOLERANCE = ToleranceConfig(
    IDENTITY_STRICT=1e-12,      # Machine-precision identity checks
    IDENTITY_RELAXED=1e-09,     # Approximate identity (accumulated errors)
    GENERIC_EQ=1e-08,           # General floating-point equality
    NUMERIC_GUARD=1e-18,        # Division-by-zero prevention
    NEGLIGIBLE=1e-15,           # Threshold for negligible values
    RELATIVE=1e-06,             # Relative tolerance for ratios
    DISTRIB_SHAPE=0.15,         # Distribution shape metrics tolerance
)
```

## When to Use Each Tolerance

### `TOLERANCE.IDENTITY_STRICT` (1e-12)

**Use for:**
- Direct mathematical identities with minimal computation
- Single-operation comparisons
- Tests verifying exact mathematical properties

**Examples:**
- Checking if a single transform is its own inverse: `transform(transform(x)) ≈ x`
- Verifying symmetry: `f(x) ≈ f(-x)` for symmetric functions
- Direct arithmetic: `a + b - b ≈ a`

**Files:** `test_reward_components.py`, `test_transforms.py`

---

### `TOLERANCE.IDENTITY_RELAXED` (1e-09)

**Use for:**
- Multi-step calculations with accumulated floating-point errors
- PBRS invariance checks (multiple potential calculations + gamma discounting)
- Exit factor computations (normalization + kernel application + transforms)
- Cross-mode comparisons (verifying fallback behavior)
- Reward component decomposition (multiple additions/subtractions)

**Examples:**
- **PBRS terminal state checks:** `next_potential ≈ 0.0`
  - *Rationale:* Accumulated errors from gamma discounting, potential calculations, and reward shaping
- **Exit factor mathematical validation:** Comparing factors across different attenuation modes
  - *Rationale:* Each mode involves normalization, kernel evaluation, and potential transformation
- **Decomposition integrity:** Verifying reward components sum correctly
  - *Rationale:* Multiple component calculations with independent floating-point operations
- **Plateau continuity:** Checking exit factors at grace boundaries
  - *Rationale:* Boundary calculations involve piecewise function evaluation
- **Fallback behavior:** Unknown modes falling back to linear
  - *Rationale:* Comparing independent calculation paths

**Files:** `test_pbrs.py` (15+ usages), `test_robustness.py` (9+ usages), `test_reward_components.py`

---

### `TOLERANCE.GENERIC_EQ` (1e-08)

**Use for:**
- General-purpose floating-point equality when exact identity not expected
- Comparisons involving intermediate calculations
- Default tolerance when relationship between operands is unclear

**Examples:**
- PnL coefficient checks in reward calculations
- General metric comparisons in statistical tests
- Non-critical assertion fallbacks

**Files:** `test_pbrs.py`, `test_reward_components.py`, `test_statistics.py`

---

### `TOLERANCE.NUMERIC_GUARD` (1e-18)

**Use for:**
- **Division-by-zero prevention** in ratio calculations
- Ensuring denominators are never exactly zero
- Protecting against numerical instability

**Examples:**
```python
# Scaling ratio calculation
ratio = diff1 / max(diff2, TOLERANCE.NUMERIC_GUARD)

# Power mode ratio
ratio = f1 / max(f0, TOLERANCE.NUMERIC_GUARD)

# Skewness/kurtosis calculations
skew = m3 / (m2**1.5 + TOLERANCE.NUMERIC_GUARD)
kurt = m4 / (m2**2 + TOLERANCE.NUMERIC_GUARD) - 3.0
```

**Key principle:** Used as **additive guard**, not as assertion tolerance.

**Files:** `test_robustness.py` (2 usages), `test_pbrs.py` (2 usages in distribution tests)

---

### `TOLERANCE.NEGLIGIBLE` (1e-15)

**Use for:**
- Distinguishing "effectively zero" from "non-zero but small"
- Scale determination in relative tolerance calculations
- Threshold checks for significant values

**Examples:**
- Ensuring cumulative drift is meaningfully non-zero: `abs(cumulative) > 10 * TOLERANCE.NEGLIGIBLE`
- Determining scale in relative comparisons: `scale = max(abs(a), abs(b), TOLERANCE.NEGLIGIBLE)`

**Files:** `test_base.py`, `test_pbrs.py`

---

### `TOLERANCE.RELATIVE` (1e-06)

**Use for:**
- Relative error comparisons: `|a - b| / max(|a|, |b|) < rtol`
- Scale-independent assertions
- Percentage-based tolerances

**Usage pattern:**
```python
self.assertAlmostEqualFloat(
    a, b,
    tolerance=TOLERANCE.IDENTITY_STRICT,  # Absolute tolerance
    rtol=TOLERANCE.RELATIVE                # Relative tolerance
)
```

**Files:** `test_statistics.py` (bootstrap CI comparisons)

---

### `TOLERANCE.DISTRIB_SHAPE` (0.15)

**Use for:**
- Statistical distribution shape metrics: skewness, kurtosis
- Stochastic process comparisons
- Bootstrap and Monte Carlo test tolerances

**Rationale:** Shape metrics are inherently noisy and sensitive to sampling.
A tolerance of 0.15 accounts for:
- Sample size limitations
- Stochastic variation
- Estimation bias in higher moments

**Examples:**
- Verifying skewness invariance under reward scaling
- Checking kurtosis preservation in PBRS transformations

**Files:** `test_pbrs.py` (normality invariance tests)

---

## Decision Matrix

| Scenario | Tolerance | Rationale |
|----------|-----------|-----------|
| Single arithmetic operation | `IDENTITY_STRICT` | Minimal error accumulation |
| 2-3 operations | `GENERIC_EQ` | Some error accumulation |
| PBRS checks (5+ operations) | `IDENTITY_RELAXED` | Gamma, potentials, transforms |
| Exit factor cross-mode | `IDENTITY_RELAXED` | Multiple kernel evaluations |
| Division denominator | `NUMERIC_GUARD` | Prevent div-by-zero |
| "Effectively zero" threshold | `NEGLIGIBLE` | Distinguish zero from noise |
| Ratio/percentage | `RELATIVE` | Scale-independent |
| Distribution shape | `DISTRIB_SHAPE` | Statistical variation |

---

## Anti-Patterns to Avoid

### ❌ Using IDENTITY_STRICT for multi-step calculations
```python
# BAD: Accumulated errors will cause flakiness
potential = compute_hold_potential(...)
shaping = apply_shaping(potential, ...)
self.assertAlmostEqual(shaping, expected, delta=TOLERANCE.IDENTITY_STRICT)
```

### ❌ Using IDENTITY_RELAXED for simple operations
```python
# BAD: Unnecessarily loose
result = a + b
self.assertAlmostEqual(result, expected, delta=TOLERANCE.IDENTITY_RELAXED)
```

### ❌ Hardcoding tolerance values
```python
# BAD: Magic number, no justification
self.assertAlmostEqual(value, 0.0, delta=1e-10)

# GOOD: Named constant with context
self.assertAlmostEqual(value, 0.0, delta=TOLERANCE.IDENTITY_RELAXED)
```

### ❌ Using assertion tolerance for division guards
```python
# BAD: Wrong use of NUMERIC_GUARD
self.assertAlmostEqual(value, expected, delta=TOLERANCE.NUMERIC_GUARD)

# GOOD: Guard denominators, not assertions
ratio = numerator / max(denominator, TOLERANCE.NUMERIC_GUARD)
self.assertAlmostEqual(ratio, expected_ratio, delta=TOLERANCE.GENERIC_EQ)
```

---

## Adding New Tolerances

If you need a new tolerance:

1. **Check if existing tolerances are appropriate** - Don't create duplicates
2. **Document the mathematical reason** - Why this specific magnitude?
3. **Add to `ToleranceConfig`** with clear docstring
4. **Update this guide** with usage examples

---

## References

- **Floating-Point Arithmetic:** What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg, 1991)
- **Test Flakiness:** Avoiding flaky tests by choosing appropriate tolerances based on error accumulation models
- **PBRS Theory:** Ng et al., "Policy Invariance Under Reward Shaping" (1999) - establishes theoretical foundations for PBRS tolerances

---

## Revision History

- **2025-12-20:** Initial guide created during Phase 1.2 of test suite improvements
