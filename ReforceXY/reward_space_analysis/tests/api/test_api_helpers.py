#!/usr/bin/env python3
"""Tests for public API and helper functions."""

import math
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from reward_space_analysis import (
    Actions,
    Positions,
    RewardParams,
    _get_bool_param,
    _get_float_param,
    _get_int_param,
    _get_str_param,
    _sample_action,
    build_argument_parser,
    parse_overrides,
    write_complete_statistical_analysis,
)

from ..constants import PARAMS, SCENARIOS, SEEDS, TOLERANCE
from ..helpers import calculate_reward_with_defaults, simulate_samples_with_defaults
from ..test_base import RewardSpaceTestBase

pytestmark = pytest.mark.api


class TestAPIAndHelpers(RewardSpaceTestBase):
    """Public API + helper utility tests."""

    def test_sample_action_idle_hazard_increases_entry_rate(self):
        """_sample_action() increases entry probability past idle cap.

        This guards the synthetic simulator against unrealistically long neutral streaks.
        The test is statistical but deterministic via fixed RNG seeds.
        """

        max_idle_duration_candles = SCENARIOS.API_MAX_IDLE_DURATION_CANDLES
        max_trade_duration_candles = PARAMS.TRADE_DURATION_MEDIUM

        def sample_entry_rate(*, idle_duration: int, short_allowed: bool) -> float:
            rng = random.Random(SEEDS.REPRODUCIBILITY)
            draws = SCENARIOS.API_ENTRY_RATE_DRAWS
            entries = 0
            for _ in range(draws):
                action, _, _, _ = _sample_action(
                    Positions.Neutral,
                    rng,
                    short_allowed=short_allowed,
                    trade_duration=0,
                    max_trade_duration_candles=max_trade_duration_candles,
                    idle_duration=idle_duration,
                    max_idle_duration_candles=max_idle_duration_candles,
                )
                if action in (Actions.Long_enter, Actions.Short_enter):
                    entries += 1
            return entries / draws

        idle_duration_low = 0
        idle_duration_high = SCENARIOS.API_IDLE_DURATION_HIGH

        low_idle_rate = sample_entry_rate(idle_duration=idle_duration_low, short_allowed=True)
        high_idle_rate = sample_entry_rate(idle_duration=idle_duration_high, short_allowed=True)

        self.assertGreater(
            high_idle_rate,
            low_idle_rate,
            "Entry rate should increase after exceeding max idle duration",
        )

        low_idle_rate_spot = sample_entry_rate(idle_duration=idle_duration_low, short_allowed=False)
        high_idle_rate_spot = sample_entry_rate(
            idle_duration=idle_duration_high, short_allowed=False
        )
        self.assertGreater(high_idle_rate_spot, low_idle_rate_spot)

    def test_parse_overrides(self):
        """Test parse overrides."""
        overrides = ["alpha=1.5", "mode=linear", "limit=42"]
        result = parse_overrides(overrides)
        self.assertEqual(result["alpha"], 1.5)
        self.assertEqual(result["mode"], "linear")
        self.assertEqual(result["limit"], 42.0)
        with self.assertRaises(ValueError):
            parse_overrides(["badpair"])

    def test_api_simulation_and_reward_smoke(self):
        """Test api simulation and reward smoke."""
        df = simulate_samples_with_defaults(
            self.base_params(max_trade_duration_candles=SCENARIOS.API_MAX_TRADE_DURATION_CANDLES),
            num_samples=SCENARIOS.SAMPLE_SIZE_TINY,
            seed=SEEDS.SMOKE_TEST,
            max_duration_ratio=SCENARIOS.API_MAX_DURATION_RATIO,
        )
        self.assertGreater(len(df), 0)
        any_exit = df[df["reward_exit"] != 0].head(1)
        if not any_exit.empty:
            row = any_exit.iloc[0]
            unrealized_pad = PARAMS.PNL_SMALL / 2
            pnl = float(row["pnl"])
            ctx = self.make_ctx(
                pnl=pnl,
                trade_duration=int(row["trade_duration"]),
                idle_duration=int(row["idle_duration"]),
                max_unrealized_profit=pnl + unrealized_pad,
                min_unrealized_profit=pnl - unrealized_pad,
                position=Positions.Long,
                action=Actions.Long_exit,
            )
            breakdown = calculate_reward_with_defaults(ctx, self.DEFAULT_PARAMS)
            self.assertFinite(breakdown.total)

    def test_simulate_samples_trading_modes_spot_vs_margin(self):
        """simulate_samples coverage: spot should forbid shorts, margin should allow them."""
        df_spot = simulate_samples_with_defaults(
            self.base_params(max_trade_duration_candles=PARAMS.TRADE_DURATION_MEDIUM),
            num_samples=SCENARIOS.SAMPLE_SIZE_SMALL,
            trading_mode="spot",
        )
        short_positions_spot = (df_spot["position"] == float(Positions.Short.value)).sum()
        self.assertEqual(short_positions_spot, 0, "Spot mode must not contain short positions")
        df_margin = simulate_samples_with_defaults(
            self.base_params(max_trade_duration_candles=PARAMS.TRADE_DURATION_MEDIUM),
            num_samples=SCENARIOS.SAMPLE_SIZE_SMALL,
        )
        for col in [
            "pnl",
            "trade_duration",
            "idle_duration",
            "position",
            "action",
            "sample_entry_prob",
            "sample_exit_prob",
            "sample_neutral_prob",
            "reward",
            "reward_invalid",
            "reward_idle",
            "reward_hold",
            "reward_exit",
        ]:
            self.assertIn(col, df_margin.columns)

    def test_simulate_samples_sampling_probabilities_are_bounded(self):
        """simulate_samples() exposes bounded sampling probabilities."""

        df = simulate_samples_with_defaults(
            self.base_params(max_trade_duration_candles=SCENARIOS.API_MAX_TRADE_DURATION_CANDLES),
            seed=SEEDS.SMOKE_TEST,
            max_duration_ratio=SCENARIOS.API_MAX_DURATION_RATIO,
        )

        for col in ["sample_entry_prob", "sample_exit_prob", "sample_neutral_prob"]:
            self.assertIn(col, df.columns)

        values = (
            df[["sample_entry_prob", "sample_exit_prob", "sample_neutral_prob"]].stack().dropna()
        )
        prob_upper_bound = SCENARIOS.API_PROBABILITY_UPPER_BOUND
        self.assertTrue(((values >= 0.0) & (values <= prob_upper_bound)).all())

    def test_simulate_samples_interprets_bool_string_params(self):
        """Test simulate_samples correctly interprets string boolean params like action_masking."""
        df1 = simulate_samples_with_defaults(
            self.base_params(
                action_masking="true", max_trade_duration_candles=PARAMS.TRADE_DURATION_SHORT
            ),
            num_samples=SCENARIOS.SAMPLE_SIZE_REPORT_MINIMAL,
            trading_mode="spot",
        )
        self.assertIsInstance(df1, pd.DataFrame)
        df2 = simulate_samples_with_defaults(
            self.base_params(
                action_masking="false", max_trade_duration_candles=PARAMS.TRADE_DURATION_SHORT
            ),
            num_samples=SCENARIOS.SAMPLE_SIZE_REPORT_MINIMAL,
            trading_mode="spot",
        )
        self.assertIsInstance(df2, pd.DataFrame)

    def test_short_allowed_via_simulation(self):
        """Test _is_short_allowed via different trading modes."""
        df_futures = simulate_samples_with_defaults(
            self.base_params(max_trade_duration_candles=PARAMS.TRADE_DURATION_SHORT),
            num_samples=SCENARIOS.SAMPLE_SIZE_SMALL,
            trading_mode="futures",
        )
        short_positions = (df_futures["position"] == float(Positions.Short.value)).sum()
        self.assertGreater(short_positions, 0, "Futures mode should allow short positions")

    def test_get_float_param(self):
        """Test float parameter extraction."""
        params = {"test_float": 1.5, "test_int": 2, "test_str": "hello"}
        self.assertEqual(_get_float_param(params, "test_float", 0.0), 1.5)
        self.assertEqual(_get_float_param(params, "test_int", 0.0), 2.0)
        val_str = _get_float_param(params, "test_str", 0.0)
        self.assertTrue(isinstance(val_str, float))
        self.assertTrue(math.isnan(val_str))
        self.assertEqual(_get_float_param(params, "missing", 3.14), 3.14)

    def test_get_float_param_edge_cases(self):
        """Robust coercion edge cases for _get_float_param.

        Enumerates:
        - None -> NaN
        - bool True/False -> 1.0/0.0
        - empty string -> NaN
        - invalid string literal -> NaN
        - numeric strings (integer, float, scientific, whitespace) -> parsed float
        - non-finite float (inf, -inf) -> NaN
        - np.nan -> NaN
        - unsupported container type -> NaN
        """
        self.assertTrue(math.isnan(_get_float_param({"k": None}, "k", 0.0)))
        self.assertEqual(_get_float_param({"k": True}, "k", 0.0), 1.0)
        self.assertEqual(_get_float_param({"k": False}, "k", 1.0), 0.0)
        self.assertTrue(math.isnan(_get_float_param({"k": ""}, "k", 0.0)))
        self.assertTrue(math.isnan(_get_float_param({"k": "abc"}, "k", 0.0)))
        self.assertEqual(_get_float_param({"k": "42"}, "k", 0.0), 42.0)
        self.assertAlmostEqual(
            _get_float_param({"k": " 17.5 "}, "k", 0.0),
            17.5,
            places=TOLERANCE.DECIMAL_PLACES_RELAXED,
            msg="Whitespace trimmed numeric string should parse",
        )
        self.assertEqual(_get_float_param({"k": "1e2"}, "k", 0.0), 100.0)
        self.assertTrue(math.isnan(_get_float_param({"k": float("inf")}, "k", 0.0)))
        self.assertTrue(math.isnan(_get_float_param({"k": float("-inf")}, "k", 0.0)))
        self.assertTrue(math.isnan(_get_float_param({"k": np.nan}, "k", 0.0)))
        self.assertTrue(
            math.isnan(_get_float_param(cast(RewardParams, {"k": cast(Any, [1, 2, 3])}), "k", 0.0))
        )

    def test_get_str_param(self):
        """Test string parameter extraction."""
        params = {"test_str": "hello", "test_int": 2}
        self.assertEqual(_get_str_param(params, "test_str", "default"), "hello")
        self.assertEqual(_get_str_param(params, "test_int", "default"), "default")
        self.assertEqual(_get_str_param(params, "missing", "default"), "default")

    def test_get_bool_param(self):
        """Test boolean parameter extraction."""
        params = {"test_true": True, "test_false": False, "test_int": 1, "test_str": "yes"}
        self.assertTrue(_get_bool_param(params, "test_true", False))
        self.assertFalse(_get_bool_param(params, "test_false", True))
        self.assertTrue(_get_bool_param(params, "test_int", False))
        self.assertTrue(_get_bool_param(params, "test_str", False))
        self.assertFalse(_get_bool_param(params, "missing", False))

    def test_get_int_param_coercions(self):
        """Robust coercion paths of _get_int_param (bool/int/float/str/None/unsupported).

        This test intentionally enumerates edge coercion semantics:
        - None returns default (numeric default or 0 if non-numeric fallback provided)
        - bool maps via int(True)=1 / int(False)=0
        - float truncates toward zero (positive and negative)
        - NaN/inf treated as invalid -> default
        - numeric-like strings parsed (including scientific notation, whitespace strip, float truncation)
        - empty/invalid/NaN strings fall back to default
        - unsupported container types fall back to default
        - missing key with non-numeric default coerces to 0
        Ensures downstream reward parameter normalization logic has consistent integer handling regardless of input source.
        """
        self.assertEqual(_get_int_param({"k": None}, "k", 5), 5)
        self.assertEqual(_get_int_param({"k": None}, "k", "x"), 0)
        self.assertEqual(_get_int_param({"k": True}, "k", 0), 1)
        self.assertEqual(_get_int_param({"k": False}, "k", 7), 0)
        self.assertEqual(_get_int_param({"k": -12}, "k", 0), -12)
        self.assertEqual(_get_int_param({"k": 9.99}, "k", 0), 9)
        self.assertEqual(_get_int_param({"k": -3.7}, "k", 0), -3)
        self.assertEqual(_get_int_param({"k": np.nan}, "k", 4), 4)
        self.assertEqual(_get_int_param({"k": float("inf")}, "k", 4), 4)
        self.assertEqual(_get_int_param({"k": "42"}, "k", 0), 42)
        self.assertEqual(_get_int_param({"k": " 17 "}, "k", 0), 17)
        self.assertEqual(_get_int_param({"k": "3.9"}, "k", 0), 3)
        self.assertEqual(_get_int_param({"k": "1e2"}, "k", 0), 100)
        self.assertEqual(_get_int_param({"k": ""}, "k", 5), 5)
        self.assertEqual(_get_int_param({"k": "abc"}, "k", 5), 5)
        self.assertEqual(_get_int_param({"k": "NaN"}, "k", 5), 5)
        self.assertEqual(_get_int_param(cast(RewardParams, {"k": cast(Any, [1, 2, 3])}), "k", 3), 3)
        self.assertEqual(_get_int_param({}, "missing", "zzz"), 0)

    def test_argument_parser_construction(self):
        """Test build_argument_parser function."""
        parser = build_argument_parser()
        self.assertIsNotNone(parser)
        args = parser.parse_args(
            [
                "--num_samples",
                str(SCENARIOS.SAMPLE_SIZE_SMALL),
                "--out_dir",
                "test_output",
            ]
        )
        self.assertEqual(args.num_samples, 100)
        self.assertEqual(str(args.out_dir), "test_output")

    def test_complete_statistical_analysis_writer(self):
        """Test write_complete_statistical_analysis function."""
        test_data = simulate_samples_with_defaults(
            self.base_params(max_trade_duration_candles=100),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir)
            write_complete_statistical_analysis(
                test_data,
                output_path,
                profit_aim=PARAMS.PROFIT_AIM,
                risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
                seed=SEEDS.BASE,
                real_df=None,
            )
            main_report = output_path / "statistical_analysis.md"
            self.assertTrue(main_report.exists(), "Main statistical analysis should be created")
            feature_file = output_path / "feature_importance.csv"
            self.assertTrue(feature_file.exists(), "Feature importance should be created")


class TestPrivateFunctions(RewardSpaceTestBase):
    """Test private functions through public API calls."""

    def test_exit_reward_calculation(self):
        """Test exit reward calculation with various scenarios."""
        scenarios = [
            (Positions.Long, Actions.Long_exit, PARAMS.PNL_MEDIUM, "Profitable long exit"),
            (
                Positions.Short,
                Actions.Short_exit,
                -PARAMS.PNL_SHORT_PROFIT,
                "Profitable short exit",
            ),
            (Positions.Long, Actions.Long_exit, -PARAMS.PNL_SMALL, "Losing long exit"),
            (Positions.Short, Actions.Short_exit, PARAMS.PNL_SMALL, "Losing short exit"),
        ]
        unrealized_pad = PARAMS.PNL_SMALL / 2
        for position, action, pnl, description in scenarios:
            with self.subTest(description=description):
                context = self.make_ctx(
                    pnl=pnl,
                    trade_duration=PARAMS.TRADE_DURATION_SHORT,
                    idle_duration=0,
                    max_unrealized_profit=max(pnl + unrealized_pad, unrealized_pad),
                    min_unrealized_profit=min(pnl - unrealized_pad, -unrealized_pad),
                    position=position,
                    action=action,
                )
                breakdown = calculate_reward_with_defaults(context, self.DEFAULT_PARAMS)
                self.assertNotEqual(
                    breakdown.exit_component,
                    0.0,
                    f"Exit component should be non-zero for {description}",
                )
                self.assertFinite(breakdown.total, name="total")

    def test_invalid_action_handling(self):
        """Test invalid action penalty."""
        context = self.make_ctx(
            pnl=PARAMS.PNL_SMALL,
            trade_duration=PARAMS.TRADE_DURATION_SHORT,
            idle_duration=0,
            max_unrealized_profit=PARAMS.PNL_SHORT_PROFIT,
            min_unrealized_profit=PARAMS.PNL_TINY,
            position=Positions.Short,
            action=Actions.Long_exit,
        )
        breakdown = calculate_reward_with_defaults(
            context, self.DEFAULT_PARAMS, action_masking=False
        )
        self.assertLess(breakdown.invalid_penalty, 0, "Invalid action should have negative penalty")
        self.assertAlmostEqualFloat(
            breakdown.total,
            breakdown.invalid_penalty
            + breakdown.reward_shaping
            + breakdown.entry_additive
            + breakdown.exit_additive,
            tolerance=TOLERANCE.IDENTITY_RELAXED,
            msg="Total should equal invalid penalty plus shaping/additives",
        )

    def test_new_invariant_and_warn_parameters(self):
        """Ensure new tunables (check_invariants, exit_factor_threshold) exist and behave.

        Uses a very large base_factor to trigger potential warning condition without capping.
        """
        params = self.base_params()
        self.assertIn("check_invariants", params)
        self.assertIn("exit_factor_threshold", params)
        context = self.make_ctx(
            pnl=PARAMS.PNL_MEDIUM,
            trade_duration=SCENARIOS.DURATION_LONG,
            idle_duration=0,
            max_unrealized_profit=PARAMS.PROFIT_AIM,
            min_unrealized_profit=0.0,
            position=Positions.Long,
            action=Actions.Long_exit,
        )
        breakdown = calculate_reward_with_defaults(
            context, params, base_factor=SCENARIOS.API_EXTREME_BASE_FACTOR
        )
        self.assertFinite(breakdown.exit_component, name="exit_component")


if __name__ == "__main__":
    unittest.main()
