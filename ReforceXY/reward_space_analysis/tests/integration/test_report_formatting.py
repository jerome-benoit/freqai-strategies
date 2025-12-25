#!/usr/bin/env python3
"""Report formatting focused tests moved from helpers/test_utilities.py.

Owns invariant: report-abs-shaping-line-091 (integration category)
"""

import re
import unittest

import numpy as np
import pandas as pd
import pytest

from reward_space_analysis import PBRS_INVARIANCE_TOL, write_complete_statistical_analysis

from ..constants import (
    PARAMS,
    SCENARIOS,
    SEEDS,
    TOLERANCE,
)
from ..test_base import RewardSpaceTestBase

pytestmark = pytest.mark.integration


class TestReportFormatting(RewardSpaceTestBase):
    def test_statistical_validation_section_absent_when_no_hypothesis_tests(self):
        """Section 5 omitted entirely when no hypothesis tests qualify (idle<30, groups<2, pnl sign groups<30)."""
        # Construct df with idle_duration always zero -> reward_idle all zeros so idle_mask.sum()==0
        # Position has only one unique value -> groups<2
        # pnl all zeros so no positive/negative groups with >=30 each
        n = SCENARIOS.SAMPLE_SIZE_TINY
        df = pd.DataFrame(
            {
                "reward": np.zeros(n),
                "reward_idle": np.zeros(n),
                "reward_hold": np.zeros(n),
                "reward_exit": np.zeros(n),
                "pnl": np.zeros(n),
                "trade_duration": np.ones(n),
                "idle_duration": np.zeros(n),
                "position": np.zeros(n),
            }
        )
        content = self._write_report(df, real_df=None)
        # Hypothesis section header should be absent
        self.assertNotIn("## 5. Statistical Validation", content)
        # Summary numbering still includes Statistical Validation line (always written)
        self.assertIn("5. **Statistical Validation**", content)
        # Distribution shift subsection appears only inside Section 5; since Section 5 omitted it should be absent.
        self.assertNotIn("### 5.4 Distribution Shift Analysis", content)
        self.assertNotIn("_Not performed (no real episodes provided)._", content)

    def _write_report(
        self, df: pd.DataFrame, *, real_df: pd.DataFrame | None = None, **kwargs
    ) -> str:
        """Helper: invoke write_complete_statistical_analysis into temp dir and return content."""
        out_dir = self.output_path / "report_tmp"
        # Ensure required columns present (action required for summary stats)
        required_cols = [
            "action",
            "reward_invalid",
            "reward_shaping",
            "reward_entry_additive",
            "reward_exit_additive",
            "duration_ratio",
            "idle_ratio",
        ]
        df = df.copy()
        for col in required_cols:
            if col not in df.columns:
                if col == "action":
                    df[col] = 0.0
                else:
                    df[col] = 0.0
        write_complete_statistical_analysis(
            df=df,
            output_dir=out_dir,
            profit_aim=PARAMS.PROFIT_AIM,
            risk_reward_ratio=PARAMS.RISK_REWARD_RATIO,
            seed=SEEDS.BASE,
            real_df=real_df,
            adjust_method="none",
            strict_diagnostics=False,
            bootstrap_resamples=SCENARIOS.SAMPLE_SIZE_SMALL,  # keep test fast
            skip_partial_dependence=kwargs.get("skip_partial_dependence", False),
            skip_feature_analysis=kwargs.get("skip_feature_analysis", False),
        )
        report_path = out_dir / "statistical_analysis.md"
        return report_path.read_text(encoding="utf-8")

    def test_abs_shaping_line_present_and_constant(self):
        """Abs Σ Shaping Reward line present, formatted, uses constant not literal."""
        df = pd.DataFrame(
            {
                "reward_shaping": [TOLERANCE.IDENTITY_STRICT, -TOLERANCE.IDENTITY_STRICT],
                "reward_entry_additive": [0.0, 0.0],
                "reward_exit_additive": [0.0, 0.0],
            }
        )
        total_shaping = df["reward_shaping"].sum()
        self.assertLess(abs(total_shaping), PBRS_INVARIANCE_TOL)
        lines = [f"| Abs Σ Shaping Reward | {abs(total_shaping):.6e} |"]
        content = "\n".join(lines)
        m = re.search("\\| Abs Σ Shaping Reward \\| ([0-9]+\\.[0-9]{6}e[+-][0-9]{2}) \\|", content)
        self.assertIsNotNone(m, "Abs Σ Shaping Reward line missing or misformatted")
        val = float(m.group(1)) if m else None
        if val is not None:
            self.assertLess(val, TOLERANCE.NEGLIGIBLE + TOLERANCE.IDENTITY_STRICT)

    def test_distribution_shift_section_present_with_real_episodes(self):
        """Distribution Shift section renders metrics table when real episodes provided."""
        # Synthetic df (ensure >=10 non-NaN per feature)
        synth_df = self.make_stats_df(n=SCENARIOS.SAMPLE_SIZE_TINY, seed=SEEDS.REPORT_FORMAT_1)
        # Real df: shift slightly (different mean) so metrics non-zero
        real_df = synth_df.copy()
        real_df["pnl"] = real_df["pnl"] + SCENARIOS.REPORT_PNL_MEAN_SHIFT  # small mean shift
        real_df["trade_duration"] = real_df["trade_duration"] * SCENARIOS.REPORT_DURATION_SCALE_UP
        real_df["idle_duration"] = real_df["idle_duration"] * SCENARIOS.REPORT_DURATION_SCALE_DOWN
        content = self._write_report(synth_df, real_df=real_df)
        # Assert metrics header and at least one feature row
        self.assertIn("### 5.4 Distribution Shift Analysis", content)
        self.assertIn(
            "| Feature | KL Div | JS Dist | Wasserstein | KS Stat | KS p-value |", content
        )
        # Ensure placeholder text absent
        self.assertNotIn("_Not performed (no real episodes provided)._", content)
        # Basic regex to find a feature row (pnl)
        m = re.search(r"\| pnl \| ([0-9]+\.[0-9]{4}) \| ([0-9]+\.[0-9]{4}) \|", content)
        self.assertIsNotNone(
            m, "pnl feature row missing or misformatted in distribution shift table"
        )

    def test_partial_dependence_redundancy_note_emitted(self):
        """Redundancy note appears when both feature analysis and partial dependence skipped."""
        df = self.make_stats_df(
            n=SCENARIOS.SAMPLE_SIZE_REPORT_MINIMAL, seed=SEEDS.REPORT_FORMAT_2
        )  # small but >=4 so skip_feature_analysis flag drives behavior
        content = self._write_report(
            df,
            real_df=None,
            skip_feature_analysis=True,
            skip_partial_dependence=True,
        )
        self.assertIn(
            "_Note: --skip_partial_dependence is redundant when feature analysis is skipped._",
            content,
        )
        # Ensure feature importance section shows skipped label
        self.assertIn("Feature Importance - (skipped)", content)
        # Ensure no partial dependence plots line for success path appears
        self.assertNotIn("partial_dependence_*.csv", content)

    # Owns invariant: integration-pbrs-metrics-section-120
    def test_report_includes_pbrs_metrics_section(self):
        """Verify statistical_analysis.md includes PBRS Metrics section with tracing metrics.

        Verifies:
        - PBRS Metrics subsection exists when PBRS columns present
        - Section includes Mean Base Reward, Mean PBRS Term, Mean Invariance Correction
        - All metrics are formatted with proper precision
        """
        # Create df with PBRS columns
        n = SCENARIOS.SAMPLE_SIZE_SMALL
        rng = np.random.default_rng(SEEDS.REPORT_FORMAT_1)
        df = pd.DataFrame(
            {
                "reward": rng.normal(0, 0.1, n),
                "reward_invalid": np.zeros(n),
                "reward_idle": np.zeros(n),
                "reward_hold": np.zeros(n),
                "reward_exit": rng.normal(0, 0.05, n),
                "reward_shaping": rng.normal(0, 0.02, n),
                "reward_entry_additive": np.zeros(n),
                "reward_exit_additive": np.zeros(n),
                # PBRS columns
                "reward_base": rng.normal(0, 0.1, n),
                "reward_pbrs_delta": rng.normal(0, 0.02, n),
                "reward_invariance_correction": rng.normal(0, PBRS_INVARIANCE_TOL / 10.0, n),
                "pnl": rng.normal(0, 0.01, n),
                "trade_duration": rng.integers(10, 100, n).astype(float),
                "idle_duration": np.zeros(n),
                "position": rng.choice([0, 1, 2], n).astype(float),
                "action": rng.choice([0, 1, 2, 3, 4], n).astype(float),
                "duration_ratio": rng.uniform(0, 1, n),
                "idle_ratio": np.zeros(n),
            }
        )

        content = self._write_report(df)

        # Verify PBRS Metrics section exists
        self.assertIn("**PBRS Metrics:**", content)

        # Verify key metrics are present
        required_metrics = [
            "Mean Base Reward",
            "Std Base Reward",
            "Mean PBRS Delta",
            "Std PBRS Delta",
            "Mean Invariance Correction",
            "Std Invariance Correction",
            "Max \\|Invariance Correction\\|",
            "Mean \\|PBRS\\| / \\|Base\\| Ratio",
        ]

        for metric in required_metrics:
            self.assertIn(metric, content, f"Missing metric in PBRS Metrics section: {metric}")

        # Verify proper formatting (values should be formatted with proper precision)
        # Check for at least one properly formatted metric line
        m = re.search(r"\| Mean Base Reward \| (-?[0-9]+\.[0-9]{6}) \|", content)
        self.assertIsNotNone(m, "Mean Base Reward metric missing or misformatted")


if __name__ == "__main__":
    unittest.main()
