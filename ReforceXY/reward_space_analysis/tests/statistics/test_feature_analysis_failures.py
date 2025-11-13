#!/usr/bin/env python3
"""Targeted tests for _perform_feature_analysis failure and edge paths.

Covers early stub returns and guarded exception branches to raise coverage:
- Missing reward column
- Empty frame
- Single usable feature (<2 features path)
- NaNs present after preprocessing (>=2 features path)
- Model fitting failure (monkeypatched fit)
- Permutation importance failure (monkeypatched permutation_importance) while partial dependence still computed
- Successful partial dependence computation path (not skipped)
- scikit-learn import fallback (RandomForestRegressor/train_test_split/permutation_importance/r2_score unavailable)
"""

import numpy as np
import pandas as pd
import pytest

from reward_space_analysis import _perform_feature_analysis  # type: ignore

pytestmark = pytest.mark.statistics


def _minimal_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "pnl": rng.normal(0, 1, n),
            "trade_duration": rng.integers(1, 10, n),
            "idle_duration": rng.integers(1, 5, n),
            "position": rng.choice([0.0, 1.0], n),
            "action": rng.integers(0, 3, n),
            "is_invalid": rng.choice([0, 1], n),
            "duration_ratio": rng.random(n),
            "idle_ratio": rng.random(n),
            "reward": rng.normal(0, 1, n),
        }
    )


def test_feature_analysis_missing_reward_column():
    df = _minimal_df().drop(columns=["reward"])  # remove reward
    importance_df, stats, partial_deps, model = _perform_feature_analysis(
        df, seed=7, skip_partial_dependence=True
    )
    assert importance_df.empty
    assert stats["model_fitted"] is False
    assert stats["n_features"] == 0
    assert partial_deps == {}
    assert model is None


def test_feature_analysis_empty_frame():
    df = _minimal_df(0)  # empty
    importance_df, stats, partial_deps, model = _perform_feature_analysis(
        df, seed=7, skip_partial_dependence=True
    )
    assert importance_df.empty
    assert stats["n_features"] == 0
    assert model is None


def test_feature_analysis_single_feature_path():
    df = pd.DataFrame({"pnl": np.random.normal(0, 1, 25), "reward": np.random.normal(0, 1, 25)})
    importance_df, stats, partial_deps, model = _perform_feature_analysis(
        df, seed=11, skip_partial_dependence=True
    )
    assert stats["n_features"] == 1
    # Importance stub path returns NaNs
    assert importance_df["importance_mean"].isna().all()
    assert model is None


def test_feature_analysis_nans_present_path():
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {
            "pnl": rng.normal(0, 1, 40),
            "trade_duration": [1.0, np.nan] * 20,  # introduces NaNs but not wholly NaN column
            "reward": rng.normal(0, 1, 40),
        }
    )
    importance_df, stats, partial_deps, model = _perform_feature_analysis(
        df, seed=13, skip_partial_dependence=True
    )
    # Should hit NaN stub path (model_fitted False)
    assert stats["model_fitted"] is False
    assert importance_df["importance_mean"].isna().all()
    assert model is None


def test_feature_analysis_model_fitting_failure(monkeypatch):
    # Monkeypatch model fit to raise
    from reward_space_analysis import RandomForestRegressor  # type: ignore

    _ = RandomForestRegressor.fit  # preserve reference for clarity (unused)

    def boom(self, *a, **kw):  # noqa: D401
        raise RuntimeError("forced fit failure")

    monkeypatch.setattr(RandomForestRegressor, "fit", boom)
    df = _minimal_df(50)
    importance_df, stats, partial_deps, model = _perform_feature_analysis(
        df, seed=21, skip_partial_dependence=True
    )
    assert stats["model_fitted"] is False
    assert model is None
    assert importance_df["importance_mean"].isna().all()
    # Restore (pytest monkeypatch will revert automatically at teardown)


def test_feature_analysis_permutation_failure_partial_dependence(monkeypatch):
    # Monkeypatch permutation_importance to raise while allowing partial dependence
    def perm_boom(*a, **kw):  # noqa: D401
        raise RuntimeError("forced permutation failure")

    monkeypatch.setattr("reward_space_analysis.permutation_importance", perm_boom)
    df = _minimal_df(60)
    importance_df, stats, partial_deps, model = _perform_feature_analysis(
        df, seed=33, skip_partial_dependence=False
    )
    assert stats["model_fitted"] is True
    # Importance should be NaNs due to failure
    assert importance_df["importance_mean"].isna().all()
    # Partial dependencies should still attempt and produce entries for available features listed in function
    assert len(partial_deps) >= 1  # at least one PD computed
    assert model is not None


def test_feature_analysis_success_partial_dependence():
    df = _minimal_df(70)
    importance_df, stats, partial_deps, model = _perform_feature_analysis(
        df, seed=47, skip_partial_dependence=False
    )
    # Expect at least one non-NaN importance (model fitted path)
    assert importance_df["importance_mean"].notna().any()
    assert stats["model_fitted"] is True
    assert len(partial_deps) >= 1
    assert model is not None


def test_feature_analysis_import_fallback(monkeypatch):
    """Simulate scikit-learn components unavailable to hit ImportError early raise."""
    # Set any one (or all) of the guarded sklearn symbols to None; function should fast-fail.
    monkeypatch.setattr("reward_space_analysis.RandomForestRegressor", None)
    monkeypatch.setattr("reward_space_analysis.train_test_split", None)
    monkeypatch.setattr("reward_space_analysis.permutation_importance", None)
    monkeypatch.setattr("reward_space_analysis.r2_score", None)
    df = _minimal_df(10)
    with pytest.raises(ImportError):
        _perform_feature_analysis(df, seed=5, skip_partial_dependence=True)


def test_module_level_sklearn_import_failure_reload():
    """Force module-level sklearn import failure to execute fallback block (lines 32–42).

    Strategy:
    - Temporarily monkeypatch builtins.__import__ to raise on any 'sklearn' import.
    - Remove 'reward_space_analysis' from sys.modules and re-import to trigger try/except.
    - Assert guarded sklearn symbols are None (fallback assigned) in newly loaded module.
    - Call its _perform_feature_analysis to confirm ImportError path surfaces.
    - Restore original importer and original module to avoid side-effects on other tests.
    """
    import builtins
    import importlib
    import sys

    orig_mod = sys.modules.get("reward_space_analysis")
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: D401
        if name.startswith("sklearn"):
            raise RuntimeError("forced sklearn import failure")
        return orig_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        # Drop existing module to force fresh execution of top-level imports
        if "reward_space_analysis" in sys.modules:
            del sys.modules["reward_space_analysis"]
        import reward_space_analysis as rsa_fallback  # noqa: F401

        # Fallback assigns sklearn symbols to None
        assert getattr(rsa_fallback, "RandomForestRegressor") is None
        assert getattr(rsa_fallback, "train_test_split") is None
        assert getattr(rsa_fallback, "permutation_importance") is None
        assert getattr(rsa_fallback, "r2_score") is None
        # Perform feature analysis should raise ImportError under missing components
        df = _minimal_df(15)
        with pytest.raises(ImportError):
            rsa_fallback._perform_feature_analysis(df, seed=3, skip_partial_dependence=True)  # type: ignore[attr-defined]
    finally:
        # Restore importer
        builtins.__import__ = orig_import
        # Restore original module state if it existed
        if orig_mod is not None:
            sys.modules["reward_space_analysis"] = orig_mod
        else:
            if "reward_space_analysis" in sys.modules:
                del sys.modules["reward_space_analysis"]
            importlib.import_module("reward_space_analysis")
