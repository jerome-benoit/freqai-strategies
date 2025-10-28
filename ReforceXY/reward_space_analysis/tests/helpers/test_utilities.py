#!/usr/bin/env python3
"""Utility tests narrowed to data loading behaviors.

Moved tests:
- Report formatting invariants -> integration/test_report_formatting.py
- Additives deterministic contribution -> components/test_additives.py
- CLI CSV + params propagation -> cli/test_cli_params_and_csv.py
"""
import pickle
import unittest
import warnings
from pathlib import Path

import pandas as pd

from reward_space_analysis import load_real_episodes

from ..test_base import RewardSpaceTestBase


class TestLoadRealEpisodes(RewardSpaceTestBase):
    """Unit tests for load_real_episodes."""

    def test_drop_exact_duplicates_warns(self):
        """Invariant 108: duplicate rows dropped with warning showing count removed."""
        df = pd.DataFrame(
            {
                "pnl": [0.01, 0.01, -0.02],  # first two duplicate
                "trade_duration": [10, 10, 20],
                "idle_duration": [5, 5, 0],
                "position": [1.0, 1.0, 0.0],
                "action": [2.0, 2.0, 0.0],
                "reward": [1.0, 1.0, -0.5],
            }
        )
        p = Path(self.temp_dir) / "dupes.pkl"
        self.write_pickle(df, p)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = load_real_episodes(p)
        self.assertEqual(len(loaded), 2, "Expected duplicate row removal to reduce length")
        msgs = [str(warning.message) for warning in w]
        dup_msgs = [m for m in msgs if "duplicate transition" in m]
        self.assertTrue(any("Removed" in m for m in dup_msgs), f"No duplicate removal warning found in: {msgs}")

    def test_missing_multiple_required_columns_single_warning(self):
        """Invariant 109: enforce_columns=False fills all missing required cols with NaN and single warning."""
        transitions = [
            {"pnl": 0.02, "trade_duration": 12},  # Missing idle_duration, position, action, reward
            {"pnl": -0.01, "trade_duration": 3},
        ]
        p = Path(self.temp_dir) / "missing_multi.pkl"
        self.write_pickle(transitions, p)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = load_real_episodes(p, enforce_columns=False)
        required = {"idle_duration", "position", "action", "reward"}
        for col in required:
            self.assertIn(col, loaded.columns)
            self.assertTrue(loaded[col].isna().all(), f"Column {col} should be all NaN")
        msgs = [str(warning.message) for warning in w]
        miss_msgs = [m for m in msgs if "missing columns" in m]
        self.assertEqual(len(miss_msgs), 1, f"Expected single missing columns warning (got {miss_msgs})")

    def write_pickle(self, obj, path: Path):
        with path.open("wb") as f:
            pickle.dump(obj, f)

    def test_top_level_dict_transitions(self):
        df = pd.DataFrame(
            {
                "pnl": [0.01],
                "trade_duration": [10],
                "idle_duration": [5],
                "position": [1.0],
                "action": [2.0],
                "reward": [1.0],
            }
        )
        p = Path(self.temp_dir) / "top.pkl"
        self.write_pickle({"transitions": df}, p)
        loaded = load_real_episodes(p)
        self.assertIsInstance(loaded, pd.DataFrame)
        self.assertEqual(list(loaded.columns).count("pnl"), 1)
        self.assertEqual(len(loaded), 1)

    def test_mixed_episode_list_warns_and_flattens(self):
        ep1 = {"episode_id": 1}
        ep2 = {
            "episode_id": 2,
            "transitions": [
                {
                    "pnl": 0.02,
                    "trade_duration": 5,
                    "idle_duration": 0,
                    "position": 1.0,
                    "action": 2.0,
                    "reward": 2.0,
                }
            ],
        }
        p = Path(self.temp_dir) / "mixed.pkl"
        self.write_pickle([ep1, ep2], p)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = load_real_episodes(p)
            _ = w
        self.assertEqual(len(loaded), 1)
        self.assertPlacesEqual(float(loaded.iloc[0]["pnl"]), 0.02, places=7)

    def test_non_iterable_transitions_raises(self):
        bad = {"transitions": 123}
        p = Path(self.temp_dir) / "bad.pkl"
        self.write_pickle(bad, p)
        with self.assertRaises(ValueError):
            load_real_episodes(p)

    def test_enforce_columns_false_fills_na(self):
        trans = [
            {"pnl": 0.03, "trade_duration": 10, "idle_duration": 0, "position": 1.0, "action": 2.0}
        ]
        p = Path(self.temp_dir) / "fill.pkl"
        self.write_pickle(trans, p)
        loaded = load_real_episodes(p, enforce_columns=False)
        self.assertIn("reward", loaded.columns)
        self.assertTrue(loaded["reward"].isna().all())

    def test_casting_numeric_strings(self):
        trans = [
            {
                "pnl": "0.04",
                "trade_duration": "20",
                "idle_duration": "0",
                "position": "1.0",
                "action": "2.0",
                "reward": "3.0",
            }
        ]
        p = Path(self.temp_dir) / "strs.pkl"
        self.write_pickle(trans, p)
        loaded = load_real_episodes(p)
        self.assertIn("pnl", loaded.columns)
        self.assertIn(loaded["pnl"].dtype.kind, ("f", "i"))
        self.assertPlacesEqual(float(loaded.iloc[0]["pnl"]), 0.04, places=7)

    def test_pickled_dataframe_loads(self):
        test_episodes = pd.DataFrame(
            {
                "pnl": [0.01, -0.02, 0.03],
                "trade_duration": [10, 20, 15],
                "idle_duration": [5, 0, 8],
                "position": [1.0, 0.0, 1.0],
                "action": [2.0, 0.0, 2.0],
                "reward": [10.5, -5.2, 15.8],
            }
        )
        p = Path(self.temp_dir) / "test_episodes.pkl"
        self.write_pickle(test_episodes, p)
        loaded_data = load_real_episodes(p)
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 3)
        self.assertIn("pnl", loaded_data.columns)


if __name__ == "__main__":
    unittest.main()
