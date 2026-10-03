"""Shared native config-template contract for the strategy test suites."""

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from freqtrade.configuration.load_config import load_config_file


def assert_optional_ccxt_rate_limits_load_without_delimiter_edits(
    test: unittest.TestCase, template: Path
) -> None:
    """Only uncommenting selected rateLimit examples may change the parsed config."""
    source = template.read_text(encoding="utf-8")
    baseline = load_config_file(str(template))
    sections = ("ccxt_config", "ccxt_async_config")
    example = re.compile(r'(?m)^[ \t]*(?P<marker>//)[ \t]*"rateLimit"\s*:\s*')
    decoder = json.JSONDecoder()
    for section in sections:
        test.assertNotIn("rateLimit", baseline["exchange"][section])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config-template.json"
        for enabled in ((), sections[:1], sections[1:], sections):
            with test.subTest(enabled=enabled):
                text = source
                expected = copy.deepcopy(baseline)
                for section in enabled:
                    start = re.search(rf'"{section}"\s*:\s*\{{', text)
                    if start is None:
                        test.fail(f"Missing CCXT section {section}")
                    match = example.search(text, start.end())
                    if match is None:
                        test.fail(f"Missing commented rateLimit example for {section}")
                    value, _ = decoder.raw_decode(text[match.end() :])
                    marker = match.start("marker")
                    text = text[:marker] + text[marker + 2 :]
                    expected["exchange"][section]["rateLimit"] = value
                path.write_text(text, encoding="utf-8")
                test.assertEqual(load_config_file(str(path)), expected)
