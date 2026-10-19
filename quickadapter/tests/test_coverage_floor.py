"""Guards the coverage gate itself; requires the Freqtrade QA image."""

import ast
import configparser
import os
import re
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from coverage.config import DEFAULT_EXCLUDE, DEFAULT_PARTIAL, DEFAULT_PARTIAL_ALWAYS, CoverageConfig
from coverage.files import find_python_files
from coverage.misc import join_regex
from coverage.parser import PythonParser
from qa_support import COVERAGERC, REPO_ROOT, QaTestCase, temporary_directory

FLOOR = "fail_under"
PLACES = "precision"
BRANCH = "branch"
NAMESPACE_PACKAGES = "include_namespace_packages"
OMIT = "omit"
# The lowest floor the project will accept. Raising it is free; dropping below this is the
# silent gate disabling that `.coveragerc` and README.md:532 both forbid.
MINIMUM_FLOOR = 68.0
PRAGMA = re.compile(r"#\s*pragma\s*:?\s*no\s*(?:cover|branch)\b", re.IGNORECASE)
INCLUDE = "include"
MEASURED_TREE = REPO_ROOT / "quickadapter" / "user_data"
COVERAGE_RUNNER = REPO_ROOT / "scripts" / "run-coverage.sh"
EXCLUSION_KEYS = (
    "omit",
    "include",
    "patch",
    "exclude_lines",
    "exclude_also",
    "partial_branches",
    "partial_also",
    "partial_branches_always",
)


def _branch_counts(parser: PythonParser) -> dict[int, int]:
    """Count the branch units that coverage actually includes in its denominator."""
    return {line: count for line, count in parser.exit_counts().items() if count > 1}


def _legal_options(section: str) -> set[str]:
    return {
        spec[1].split(":")[1]
        for spec in CoverageConfig.CONFIG_FILE_OPTIONS
        if spec[1].startswith(f"{section}:") and spec[1] != "run:_crash"
    }


class CoverageFloorTest(QaTestCase):
    def setUp(self):
        super().setUp()
        self.text = COVERAGERC.read_text()
        self.parser = configparser.ConfigParser()
        self.parser.read_string(self.text)

    def test_a_config_has_only_supported_sections_and_options(self):
        self.assertEqual(sorted(self.parser.sections()), ["report", "run"])
        for section in ("run", "report"):
            for option in self.parser[section]:
                with self.subTest(section=section, option=option):
                    self.assertIn(option, _legal_options(section))

    def test_no_other_option_declares_a_measurement_root(self):
        for option in ("source_pkgs", "source_dirs", "plugins"):
            with self.subTest(option=option):
                self.assertNotIn(option, self.parser["run"])

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
        for section in ("run", "report"):
            for key in EXCLUSION_KEYS:
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

    def _assert_source_exclusions(self, tree: Path, *, type_checking_blocks: int = 0):
        measured = {Path(path).resolve() for path in find_python_files(str(tree), True)}
        self.assertEqual(measured, {path.resolve() for path in tree.rglob("*.py")})
        allowed_blocks = 0
        for path in sorted(measured):
            text = path.read_text()
            self.assertIsNone(PRAGMA.search(text), "coverage pragmas shrink the denominator")
            baseline = PythonParser(filename=str(path))
            baseline.parse_source()
            guarded = PythonParser(filename=str(path), exclude=join_regex(DEFAULT_EXCLUDE))
            guarded.parse_source()
            allowed_statements = set()
            allowed_branches = set()
            for node in ast.walk(ast.parse(text)):
                if not isinstance(node, ast.If) or not (
                    isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
                ):
                    continue
                self.assertEqual(path, (MEASURED_TREE / "strategies" / "Utils.py").resolve())
                self.assertEqual(len(node.body), 1)
                imported = node.body[0]
                self.assertIsInstance(imported, ast.ImportFrom)
                self.assertEqual(imported.module, "xgboost.callback")
                self.assertEqual(
                    [(alias.name, alias.asname) for alias in imported.names],
                    [("TrainingCallback", "XGBoostTrainingCallback")],
                )
                self.assertEqual(len(node.orelse), 1)
                self.assertEqual(
                    ast.dump(node.orelse[0], include_attributes=False),
                    ast.dump(ast.parse("XGBoostTrainingCallback = object").body[0]),
                )
                allowed_blocks += 1
                allowed_statements.update(range(node.lineno, imported.end_lineno + 1))
                allowed_branches.add(node.lineno)
            with self.subTest(module=path.name):
                self.assertEqual(
                    baseline.statements - allowed_statements,
                    guarded.statements - allowed_statements,
                )
                self.assertEqual(
                    {
                        line: count
                        for line, count in _branch_counts(baseline).items()
                        if line not in allowed_branches
                    },
                    {
                        line: count
                        for line, count in _branch_counts(guarded).items()
                        if line not in allowed_branches
                    },
                )
                self.assertEqual(
                    set(),
                    baseline.lines_matching(join_regex(DEFAULT_PARTIAL + DEFAULT_PARTIAL_ALWAYS)),
                )
        self.assertEqual(allowed_blocks, type_checking_blocks)

    def test_no_production_source_widens_coverage_s_default_exclusions(self):
        self._assert_source_exclusions(MEASURED_TREE, type_checking_blocks=1)

    def _run_coverage(self, floor, *, tests_pass=True, source_text=None):
        with temporary_directory() as directory:
            source = directory / "source"
            tests = directory / "tests"
            source.mkdir()
            tests.mkdir()
            (source / "sample.py").write_text(
                source_text or "def choose(flag):\n    if flag:\n        return 1\n    return 2\n"
            )
            expected = 1 if tests_pass else 2
            (tests / "test_sample.py").write_text(
                "import unittest\nfrom sample import choose\n"
                "class SampleTest(unittest.TestCase):\n"
                "    def test_one_branch(self):\n"
                f"        self.assertEqual(choose(True), {expected})\n"
            )
            rcfile = directory / ".coveragerc"
            rcfile.write_text(
                f"[run]\nbranch = True\nsource = {source}\n[report]\nfail_under = {floor}\n"
            )
            env = {
                **os.environ,
                "PYTHONPATH": str(source),
                "COVERAGE_RCFILE": str(rcfile),
                "COVERAGE_FILE": str(directory / ".coverage"),
            }
            return subprocess.run(
                ["sh", str(COVERAGE_RUNNER), "-s", str(tests), "-v"],
                cwd=directory,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )

    def test_the_runner_propagates_the_actual_coverage_report_status(self):
        # The same passing suite covers only one branch: 100 fails, 50 passes.
        for floor, expected_status in ((100, 2), (50, 0)):
            with self.subTest(floor=floor):
                result = self._run_coverage(floor)
                self.assertEqual(result.returncode, expected_status, result.stdout + result.stderr)

    def test_the_runner_preserves_a_test_failure_even_above_the_coverage_floor(self):
        result = self._run_coverage(50, tests_pass=False)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_source_exemptions_that_bypass_a_real_report_are_refused(self):
        # No branch suppresses missing arcs: combined coverage rises from 4/6 to 5/6
        # despite one missing statement. No cover excludes the entire untaken clause.
        baseline = self._run_coverage(80)
        self.assertEqual(baseline.returncode, 2, baseline.stdout + baseline.stderr)
        for kind in ("branch", "cover"):
            for separator in (":", ""):
                with self.subTest(kind=kind, separator=separator):
                    pragma = f"# pragma{separator} no {kind}"
                    if_comment = f"  {pragma}" if kind == "branch" else ""
                    else_comment = f"  {pragma}" if kind == "cover" else ""
                    source_text = (
                        f"def choose(flag):\n    if flag:{if_comment}\n"
                        f"        return 1\n    else:{else_comment}\n        return 2\n"
                    )
                    with temporary_directory() as source:
                        (source / "sample.py").write_text(source_text)
                        with self.assertRaises(AssertionError):
                            self._assert_source_exclusions(source)
                    result = self._run_coverage(80, source_text=source_text)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_the_source_is_the_measured_tree(self):
        self.assertEqual(
            [line.strip() for line in self.parser["run"]["source"].splitlines() if line.strip()],
            ["quickadapter/user_data"],
        )


class CoverageGuardRegressionTest(QaTestCase):
    def _run_guard(self, text, method):
        with temporary_directory() as directory:
            fixture = directory / ".coveragerc"
            fixture.write_text(text)
            result = unittest.TestResult()
            with mock.patch.dict(globals(), COVERAGERC=fixture):
                CoverageFloorTest(method).run(result)
            self.assertEqual(result.testsRun, 1)
            self.assertEqual(result.errors, [])
            return not result.failures

    def test_floor_boundary_uses_actual_guard(self):
        text = COVERAGERC.read_text()
        for floor, valid in ((67, False), (68, True)):
            with self.subTest(floor=floor):
                changed = re.sub(r"(?m)^fail_under\s*=.*$", f"fail_under = {floor}", text)
                self.assertEqual(self._run_guard(changed, "test_the_floor_is_never_lowered"), valid)

    def test_config_changes_cannot_shrink_or_replace_the_denominator(self):
        baseline = COVERAGERC.read_text()
        for option, method in (
            ("source_dirs = elsewhere", "test_no_other_option_declares_a_measurement_root"),
            ("plugins = plugin", "test_no_other_option_declares_a_measurement_root"),
            ("exclude_also = return", "test_no_production_code_is_excluded_by_any_other_key"),
            ("exclude_line = return", "test_a_config_has_only_supported_sections_and_options"),
        ):
            with self.subTest(option=option):
                changed = baseline.replace("[run]", f"[run]\n{option}")
                self.assertFalse(self._run_guard(changed, method))
        self.assertFalse(
            self._run_guard(
                baseline + "\n[unexpected]\n",
                "test_a_config_has_only_supported_sections_and_options",
            )
        )

    def test_provenance_is_dated_final_and_sufficient(self):
        baseline = COVERAGERC.read_text()
        for annotation, valid in (
            ("", False),
            ("# measured 76.0%", False),
            ("# measured 2026-10-02", False),
            ("# measured 2026-10-02 at 67.9%", False),
            ("# measured 2026-10-02 at 76.0% PROVISIONAL", False),
            ("# measured 2000-01-01 at 76.0%", True),
        ):
            with self.subTest(annotation=annotation):
                text = "\n".join(
                    annotation if line.startswith("# measured") else line
                    for line in baseline.splitlines()
                )
                valid_result = self._run_guard(text, "test_the_floor_carries_its_provenance")
                if valid_result:
                    valid_result = self._run_guard(
                        text,
                        "test_the_provenance_records_a_date_and_a_measurement_that_clears_the_floor",
                    )
                self.assertEqual(valid_result, valid)

    def test_default_source_exclusions_are_refused(self):
        for text in (
            "def omitted():\n    ...\n",
            "def choose(flag):\n    if True:\n        return flag\n    return None\n",
            'def choose(flag):\n    marker = "# pragma: no cover"\n    return flag\n',
        ):
            with self.subTest(source=text), temporary_directory() as directory:
                (directory / "sample.py").write_text(text)
                with self.assertRaises(AssertionError):
                    CoverageFloorTest()._assert_source_exclusions(directory)


if __name__ == "__main__":
    unittest.main()
