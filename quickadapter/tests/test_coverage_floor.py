"""Guards the coverage gate itself; requires the Freqtrade QA image."""

import configparser
import unittest

from qa_support import COVERAGERC, QaTestCase

FLOOR = "fail_under"
PLACES = "precision"
BRANCH = "branch"
NAMESPACE_PACKAGES = "include_namespace_packages"
OMIT = "omit"


class CoverageFloorTest(QaTestCase):
    def setUp(self):
        super().setUp()
        self.text = COVERAGERC.read_text()
        self.parser = configparser.ConfigParser()
        self.parser.read_string(self.text)

    def _raw_floor(self) -> str:
        """Extract fail_under by whole-line partitioning, not by regex.

        A regex such as ^fail_under\\s*=\\s*(\\S+) pulls the leading digits out of
        `fail_under = 10        # measured ...` and passes, so it cannot see the
        malformation that actually breaks the run. Whole-line partitioning yields the
        comment too, and float() then rejects it, which is the behaviour we want.
        """
        for line in self.text.splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == FLOOR:
                return value.strip()
        raise AssertionError(f"{FLOOR} is absent from {COVERAGERC}")

    def test_the_placeholder_is_substituted(self):
        self.assertNotIn("__FLOOR__", self.text, "the coverage floor is still a placeholder")

    def test_the_floor_is_a_percentage_in_range(self):
        value = float(self._raw_floor())
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 100.0)

    def test_the_floor_carries_its_provenance(self):
        # The measured date is what tells a reader whether the number is current, and
        # it is the only thing distinguishing a real floor from a provisional one.
        lines = [line.strip() for line in self.text.splitlines()]
        index = next(i for i, line in enumerate(lines) if line.startswith(f"{FLOOR} ="))
        annotation = " ".join(lines[max(0, index - 2) : index])
        self.assertIn("measured", annotation, f"the line above {FLOOR} must record the measurement")
        self.assertNotIn("PROVISIONAL", annotation)

    def test_branch_tracing_is_enabled(self):
        # Without it, every guard — an early return, a raise — counts as covered the
        # moment its `if` is evaluated, and the gate measures nothing useful.
        self.assertTrue(self.parser.getboolean("run", BRANCH))

    def test_unexecuted_files_stay_in_the_denominator(self):
        self.assertTrue(self.parser.getboolean("report", NAMESPACE_PACKAGES))

    def test_precision_is_set_so_the_floor_is_interpreted_consistently(self):
        self.assertIn(PLACES, self.parser["report"])

    def test_no_production_code_is_omitted(self):
        # An omit entry would shrink the denominator while the floor stayed put, so the
        # gate would report the same number over a smaller measurement.
        self.assertNotIn(OMIT, self.parser["report"])

    def test_the_source_is_the_measured_tree(self):
        self.assertEqual(
            [line.strip() for line in self.parser["run"]["source"].splitlines() if line.strip()],
            ["quickadapter/user_data"],
        )


if __name__ == "__main__":
    unittest.main()
