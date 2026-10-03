"""Config-template parser regressions; requires the matching Freqtrade QA image."""

from pathlib import Path

from qa_support import QaTestCase

from scripts.config_template_contract import (
    assert_optional_ccxt_rate_limits_load_without_delimiter_edits,
)


class ConfigTemplateTest(QaTestCase):
    def test_optional_ccxt_rate_limits_load_without_delimiter_edits(self):
        template = Path(__file__).resolve().parents[1] / "user_data/config-template.json"
        assert_optional_ccxt_rate_limits_load_without_delimiter_edits(self, template)
