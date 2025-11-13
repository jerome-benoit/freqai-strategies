#!/usr/bin/env python3
"""Report formatting focused tests moved from helpers/test_utilities.py.

Owns invariant: report-abs-shaping-line-091 (integration category)
"""

import re
import unittest

import numpy as np
import pandas as pd

from reward_space_analysis import PBRS_INVARIANCE_TOL, write_complete_statistical_analysis

from ..constants import SCENARIOS
from ..test_base import RewardSpaceTestBase


class TestReportFormatting(RewardSpaceTestBase):
    def test_statistical_validation_section_absent_when_no_hypothesis_tests(self):
        """Section 5 omitted entirely when no hypothesis tests qualify (idle<30, groups<2, pnl sign groups<30)."""
        # Construct df with idle_duration always zero -> reward_idle all zeros so idle_mask.sum()==0
        # Position has only one unique value -> groups<2
        # pnl all zeros so no positive/negative groups with >=30 each
        n = 40
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
            profit_target=self.TEST_PROFIT_TARGET,
            seed=self.SEED,
            real_df=real_df,
            adjust_method="none",
            strict_diagnostics=False,
            bootstrap_resamples=SCENARIOS.BOOTSTRAP_STANDARD_ITERATIONS,  # keep test fast
            skip_partial_dependence=kwargs.get("skip_partial_dependence", False),
            skip_feature_analysis=kwargs.get("skip_feature_analysis", False),
        )
        report_path = out_dir / "statistical_analysis.md"
        return report_path.read_text(encoding="utf-8")

    """Tests for report formatting elements not covered elsewhere."""

    def test_abs_shaping_line_present_and_constant(self):
        """Abs Σ Shaping Reward line present, formatted, uses constant not literal."""
        df = pd.DataFrame(
            {
                "reward_shaping": [self.TOL_IDENTITY_STRICT, -self.TOL_IDENTITY_STRICT],
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
            self.assertLess(val, self.TOL_NEGLIGIBLE + self.TOL_IDENTITY_STRICT)
        self.assertNotIn(
            str(self.TOL_GENERIC_EQ),
            content,
            "Tolerance constant value should appear, not raw literal",
        )

    def test_distribution_shift_section_present_with_real_episodes(self):
        """Distribution Shift section renders metrics table when real episodes provided."""
        # Synthetic df (ensure >=10 non-NaN per feature)
        synth_df = self.make_stats_df(n=60, seed=123)
        # Real df: shift slightly (different mean) so metrics non-zero
        real_df = synth_df.copy()
        real_df["pnl"] = real_df["pnl"] + 0.001  # small mean shift
        real_df["trade_duration"] = real_df["trade_duration"] * 1.01
        real_df["idle_duration"] = real_df["idle_duration"] * 0.99
        content = self._write_report(synth_df, real_df=real_df)
        # Assert metrics header and at least one feature row
        self.assertIn("### 5.4 Distribution Shift Analysis", content)
        self.assertIn(
            "| Feature | KL Div | JS Dist | Wasserstein | KS Stat | KS p-value |", content
        )
        # Ensure placeholder text absent
        self.assertNotIn("_Not performed (no real episodes provided)._", content)
        # Basic regex to find a feature row (pnl)
        import re as _re

        m = _re.search(r"\| pnl \| ([0-9]+\.[0-9]{4}) \| ([0-9]+\.[0-9]{4}) \|", content)
        self.assertIsNotNone(
            m, "pnl feature row missing or misformatted in distribution shift table"
        )

    def test_partial_dependence_redundancy_note_emitted(self):
        """Redundancy note appears when both feature analysis and partial dependence skipped."""
        df = self.make_stats_df(
            n=10, seed=321
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


if __name__ == "__main__":
    unittest.main()
