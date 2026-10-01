"""Guards the coverage gate itself; requires the Freqtrade QA image."""

import configparser
import re
import unittest

import yaml
from qa_support import COVERAGERC, REPO_ROOT, QaTestCase

FLOOR = "fail_under"
PLACES = "precision"
BRANCH = "branch"
NAMESPACE_PACKAGES = "include_namespace_packages"
OMIT = "omit"
# The lowest floor the project will accept. Raising it is free; dropping below this is the
# silent gate disabling that `.coveragerc` and README.md:532 both forbid.
MINIMUM_FLOOR = 67.0
PRAGMA = re.compile(r"pragma\s*:\s*no\s*cover", re.IGNORECASE)
# The other two of coverage.py's three DEFAULT_EXCLUDE patterns.
ELLIPSIS_BODY = re.compile(r"^\s*(((async )?def .*?)?[\])]+(\s*->.*?)?:\s*)?\.\.\.\s*(#|$)")
TYPE_CHECKING = re.compile(r"^\s*if (typing\.)?TYPE_CHECKING:")
INCLUDE = "include"
MEASURED_TREE = REPO_ROOT / "quickadapter" / "user_data"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "quality.yml"


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

    def test_precision_is_fine_enough_for_the_gate_to_compare_what_it_prints(self):
        # coverage's predicate is `round(total, precision) < fail_under`, so precision is not
        # cosmetic: at 0 the measured total is rounded to a whole number and a run up to half a
        # point BELOW the floor passes. Measured with the shipped floor of 67, a true total of
        # 66.6 fails at precision 1 and passes at precision 0, and the same divergence reaches
        # `coverage report`'s exit code through the CLI. The key was present; its value was
        # never read, so `precision = 0` shipped with the suite green.
        precision = self.parser.getint("report", PLACES)
        self.assertGreaterEqual(precision, 1)

    def test_the_floor_is_bound_in_the_section_coverage_reads_it_from(self):
        # coverage binds fail_under as `report:fail_under` ONLY. The same key under
        # [run] is silently inert, which would leave the gate dead while the value, the
        # provenance and the placeholder checks all still passed, because they read the
        # file as text. Executed: moving the line (and its comment) into [run] with the
        # value raised to 99 kept all eight tests green while the CI step exited 0 at
        # 69.2%. Section membership is therefore asserted, not inferred.
        self.assertIn(FLOOR, self.parser["report"])
        self.assertNotIn(FLOOR, self.parser["run"])

    def test_no_production_code_is_excluded_by_any_other_key(self):
        # `omit` was only the first way to shrink the denominator: `exclude_lines`,
        # `exclude_also` and the partial_* family are the same lever under other names, and
        # adding `exclude_also = ^\s*raise\b` was measured to raise the reported total from
        # 69.2 % to 70.1 % with every other floor test still green. Measured against coverage
        # 7.16.2, all of these are `[report]`-only; it REJECTS exclude_lines, exclude_also
        # and partial_branches under `[run]`, so checking them in both sections is free and
        # forward-looking.
        for section, keys in (
            ("run", (OMIT, INCLUDE)),
            (
                "report",
                (
                    OMIT,
                    INCLUDE,
                    "exclude_lines",
                    "exclude_also",
                    "partial_branches",
                    "partial_also",
                    "partial_branches_always",
                ),
            ),
        ):
            for key in keys:
                with self.subTest(section=section, key=key):
                    self.assertNotIn(key, self.parser[section])

    def test_the_floor_is_never_lowered(self):
        # `.coveragerc` and README.md:532 both say "raise only, never lower", but nothing
        # enforced it: `fail_under = 5` kept every test in this class green while the gate
        # stopped constraining anything. The floor is a control the project believes it has,
        # so the floor test now pins its LOWER bound. Raising it stays free, which is exactly
        # the direction the policy allows.
        self.assertGreaterEqual(float(self._raw_floor()), MINIMUM_FLOOR)

    def test_the_provenance_records_a_date_and_a_measurement_that_clears_the_floor(self):
        # The class comment claims the measured date is what tells a reader whether the
        # number is current, but the original assertion only looked for the word "measured":
        # both `# measured whenever someone remembered` and a bare date passed. Two things
        # are actually checkable here, and both are asserted.
        #
        # Recency is deliberately NOT one of them. Refusing a date older than N months would
        # make this suite fail on a repository nobody has touched, which is a time bomb, not
        # a gate. What is enforceable is that the annotation records a real measurement and
        # that the measurement clears the floor it justifies: a floor of 70 annotated with a
        # 69.3% measurement is a gate that cannot pass, and the stale number is the symptom.
        lines = [line.strip() for line in self.text.splitlines()]
        index = next(i for i, line in enumerate(lines) if line.startswith(f"{FLOOR} ="))
        annotation = " ".join(lines[max(0, index - 2) : index])
        with self.subTest(requirement="a well-formed measurement date"):
            self.assertRegex(annotation, r"\d{4}-\d{2}-\d{2}")
        with self.subTest(requirement="a recorded measurement"):
            found = re.search(r"measured[^%]*?([0-9]+(?:\.[0-9]+)?)%", annotation)
            self.assertIsNotNone(found, f"no percentage recorded above {FLOOR}: {annotation!r}")
            self.assertGreaterEqual(
                float(found.group(1)),
                float(self._raw_floor()),
                "the annotated measurement must clear the floor it justifies",
            )

    def test_no_production_source_widens_coverage_s_default_exclusions(self):
        # The source side of the same lever the config-side keys above cover. coverage.py
        # always applies a DEFAULT_EXCLUDE of three patterns, and each removes statements from
        # the denominator without any config change at all: a `# pragma: no cover` comment, a
        # `...`-only body, and an `if TYPE_CHECKING:` block. Measured for the pragma: one on
        # `_normalize_final_take_profit_state` took QuickAdapterV3.py from 58.8 % to 60.2 % and
        # the total from 69.3 % to 69.6 %, with the suite green.
        #
        # The tree is NOT free of the lever today: `if TYPE_CHECKING:` appears in Utils.py and
        # is excluded today. A type-checking block is legitimate — it holds imports that do not
        # run — so it is permitted up to a pinned count rather than banned outright. The other
        # two patterns have no legitimate use here, so they must stay at zero.
        counts = {
            "pragma: no cover": 0,
            "ellipsis body": 0,
            "if TYPE_CHECKING:": 1,
        }
        found = dict.fromkeys(counts, 0)
        for path in sorted(MEASURED_TREE.rglob("*.py")):
            for line in path.read_text().splitlines():
                if PRAGMA.search(line):
                    found["pragma: no cover"] += 1
                elif ELLIPSIS_BODY.match(line):
                    found["ellipsis body"] += 1
                elif TYPE_CHECKING.match(line):
                    found["if TYPE_CHECKING:"] += 1
        self.assertEqual(counts, found, "these shrink the denominator without any config change")

    def test_the_workflow_runs_the_gate_with_this_configuration_and_propagates_it(self):
        # Everything above constrains the CONFIGURATION to be meaningful; these two constrain
        # something to actually read it and to act on the result.
        #
        # The rcfile binding: coverage looks for `.coveragerc` in the directory it runs from
        # and does not search parents, and the QA image has no `.coveragerc`, `setup.cfg`,
        # `tox.ini` or `pyproject.toml` at the workspace root. Deleting the one
        # `--env COVERAGE_RCFILE=` line therefore leaves coverage with NO configuration: branch
        # tracing off, no `source`, `fail_under = 0`, measuring the standard library and
        # exiting 0. Nothing else in this class is sensitive to that line.
        workflow = WORKFLOW.read_text()
        self.assertRegex(
            workflow,
            r"--env COVERAGE_RCFILE=\S*\$\{\{ matrix\.context \}\}/\.coveragerc",
            "the coverage step must be pointed at the context's .coveragerc",
        )
        # The propagation: `exit $status` is the only thing that turns a failing
        # `coverage report` into a failing job. It is scoped to the COVERAGE STEP, matched on
        # its own indent: every `- name:` in the file is indented, so searching for an
        # unindented one silently runs off the end of the file and a decoy token anywhere
        # later in the workflow satisfies the assertion.
        step = re.search(
            r"^      - name: Run runtime regressions with coverage$.*?(?=^      - name:|\n\S)",
            workflow,
            re.DOTALL | re.MULTILINE,
        )
        self.assertIsNotNone(step, "the coverage step is gone from the workflow")
        self.assertIn("python -m coverage report", step.group(0))
        self.assertIn("exit $status", step.group(0), "the step must exit with the report status")

    def test_the_workflow_actually_dispatches_the_coverage_step(self):
        # The previous guards constrain the CONFIGURATION and the body of the coverage step.
        # This constrains whether the step runs AT ALL, which is the same lever one level up:
        # flipping the QuickAdapter matrix entry to `coverage: false`, or the step's own `if:`
        # to a constant, leaves every other test green while CI stops running the gate
        # entirely. A job that skips the coverage step is indistinguishable from a green one.
        #
        # Parsed rather than grepped: a matrix entry that loses its `coverage` key, or an `if:`
        # that stops referencing the matrix, must fail here rather than pass a substring test.
        workflow = yaml.safe_load(WORKFLOW.read_text())
        entries = workflow["jobs"]["strategy-qa"]["strategy"]["matrix"]["include"]
        by_name = {entry["name"]: entry for entry in entries}

        self.assertIn("QuickAdapter", by_name)
        self.assertIs(
            by_name["QuickAdapter"].get("coverage"),
            True,
            "the QuickAdapter matrix entry must request the coverage run",
        )
        steps = workflow["jobs"]["strategy-qa"]["steps"]
        coverage_steps = [
            step for step in steps if step.get("name") == "Run runtime regressions with coverage"
        ]
        self.assertEqual(1, len(coverage_steps), "the coverage step must exist exactly once")
        self.assertEqual(
            "matrix.coverage",
            coverage_steps[0].get("if"),
            "the coverage step must be dispatched by the matrix flag, not a constant",
        )

    def test_the_source_is_the_measured_tree(self):
        self.assertEqual(
            [line.strip() for line in self.parser["run"]["source"].splitlines() if line.strip()],
            ["quickadapter/user_data"],
        )


if __name__ == "__main__":
    unittest.main()
