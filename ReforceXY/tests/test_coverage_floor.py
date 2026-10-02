"""Guards the coverage gate itself; requires the Freqtrade QA image."""

import configparser
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

from coverage.config import DEFAULT_EXCLUDE, DEFAULT_PARTIAL, DEFAULT_PARTIAL_ALWAYS, CoverageConfig
from coverage.files import find_python_files
from coverage.misc import join_regex
from coverage.parser import PythonParser
from qa_support import COVERAGERC, MEASURED_TREE, REPO_ROOT, QaTestCase, temporary_directory

FLOOR = "fail_under"
PLACES = "precision"
BRANCH = "branch"
NAMESPACE_PACKAGES = "include_namespace_packages"
OMIT = "omit"
INCLUDE = "include"
# This is the floor ReforceXY has actually reached, not the target. The gate is only as
# strong as the measurement behind it, so `MINIMUM_FLOOR` starts at the calibrated value
# and may only rise. Reaching the agreed 70% target is a separate delivery; until then a
# lower value here would be a control the project believes it has and does not.
MINIMUM_FLOOR = 57.0
EXCLUSION_KEYS = (
    OMIT,
    INCLUDE,
    "patch",
    "exclude_lines",
    "exclude_also",
    "partial_branches",
    "partial_also",
    "partial_branches_always",
)
COVERAGERC_SECTIONS = ["report", "run"]
COVERAGERC_SECTION_MESSAGE = (
    "coverage accepts a [coverage:run] section form; the bare parser used by this class sees it "
    "literally, so every other guard would KeyError or read an empty section"
)
COVERAGE_RUNNER = REPO_ROOT / "scripts" / "run-coverage.sh"


def _branch_counts(parser: PythonParser) -> dict[int, int]:
    """Exactly what Analysis._total_branches() sums — coverage/results.py:148-150.

    `arcs()` is built from the AST alone and never consults `excluded`, so an arc-set
    comparison on it is identically green in every scenario. `exit_counts()` is that set
    with excluded endpoints dropped, and the raw dict differs at count-1 entries, so the
    `> 1` filter is both necessary and sufficient.
    """
    return {line: n for line, n in parser.exit_counts().items() if n > 1}


def _legal_options(section: str) -> set[str]:
    """Option names coverage binds in `section`, read from its own option table.

    `spec[1]` is `"section:option"`; `spec[0]` is the ATTRIBUTE name and differs for five
    of the eight exclusion keys (`run_include` vs `include`, `exclude_list` vs
    `exclude_lines`, …). Reading `spec[0]` would falsely red any legal `omit`.
    """
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

    def test_a_the_config_declares_exactly_the_run_and_report_sections(self):
        # Runs first alphabetically among the `test_…` names, because every other method in
        # this class reaches `self.parser` and a `[coverage:run]`-prefixed file makes that
        # access raise before any assertion here could report the real cause.
        self.assertEqual(
            sorted(self.parser.sections()),
            COVERAGERC_SECTIONS,
            COVERAGERC_SECTION_MESSAGE,
        )

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
        # coverage's predicate is `round(total, precision) < fail_under`, so at 0 a run
        # up to half a point BELOW the floor passes, and the same divergence reaches the
        # CLI exit code.
        precision = self.parser.getint("report", PLACES)
        self.assertGreaterEqual(precision, 1)

    def test_the_floor_is_bound_in_the_section_coverage_reads_it_from(self):
        # coverage binds fail_under as `report:fail_under` ONLY. The same key under [run]
        # is silently inert, which would leave the gate dead while the value, the
        # provenance and the placeholder checks all still passed.
        self.assertIn(FLOOR, self.parser["report"])
        self.assertNotIn(FLOOR, self.parser["run"])

    def test_no_production_code_is_excluded_by_any_other_key(self):
        # `omit` was only the first way to shrink the denominator: `exclude_lines`,
        # `exclude_also` and the partial_* family are the same lever under other names.
        # An UNRECOGNISED key only warns (`config.py:349-354`), so a misplaced key is
        # silently inert rather than fatal — the danger is an unnoticed no-op.
        for section in COVERAGERC_SECTIONS:
            for key in EXCLUSION_KEYS:
                with self.subTest(section=section, key=key):
                    self.assertNotIn(key, self.parser[section])

    def test_every_present_option_is_a_legal_name_for_its_section(self):
        # The eight-key absence assertion cannot see a TYPO — `exclude_line` is not
        # `exclude_lines` — and a typo is silently inert. Reading the legal names from
        # coverage's own option table also makes the guard self-update on a rename.
        for section in COVERAGERC_SECTIONS:
            legal = _legal_options(section)
            for key in self.parser[section]:
                with self.subTest(section=section, key=key):
                    self.assertIn(
                        key,
                        legal,
                        "coverage would warn and ignore this option, so the gate would be "
                        "weaker than it reads",
                    )

    def test_the_floor_is_never_lowered(self):
        # `.coveragerc` and README.md:545 both say "raise only, never lower", but nothing
        # enforced it. Raising the floor stays free, which is exactly the direction the
        # policy allows.
        self.assertGreaterEqual(float(self._raw_floor()), MINIMUM_FLOOR)

    def test_the_provenance_records_a_date_and_a_measurement_that_clears_the_floor(self):
        # Recency is deliberately NOT asserted: refusing a date older than N months would
        # make this suite fail on a repository nobody has touched. What is enforceable is
        # that the annotation records a real measurement and that the measurement clears
        # the floor it justifies.
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
        # The SOURCE side of the same lever. coverage applies DEFAULT_EXCLUDE by
        # regex-searching the raw file text, so a pragma inside a string literal removes a
        # real statement, and `while True:` / `if True:` remove branch units — 27% of this
        # denominator — with no config key reachable at all. Parsing twice, once with the
        # defaults and once without, is exact by construction: coverage applies them only
        # when `self.exclude` is truthy, so the difference is the lever and nothing else.
        measured = {Path(p) for p in find_python_files(str(MEASURED_TREE), True)}
        self.assertEqual(
            measured,
            set(MEASURED_TREE.rglob("*.py")),
            "the guard must audit exactly the files coverage measures; find_python_files also "
            "admits .pyw and rejects odd-character names, so a future one turns this red with "
            "no coverage defect",
        )
        for path in sorted(measured):
            baseline = PythonParser(filename=str(path))
            baseline.parse_source()
            guarded = PythonParser(filename=str(path), exclude=join_regex(DEFAULT_EXCLUDE))
            guarded.parse_source()
            with self.subTest(module=path.name):
                self.assertEqual(
                    set(baseline.statements),
                    set(guarded.statements),
                    "coverage's default exclusions remove these statements with no config key",
                )
                self.assertEqual(
                    _branch_counts(baseline),
                    _branch_counts(guarded),
                    "coverage's default exclusions remove these branch units with no config key",
                )
                # Asserted on `baseline`: both parsers read the file identically, so naming
                # `guarded` would imply a dependency that does not exist. lines_matching is
                # not token-aware, so a comment mentioning `while True:` fails this too —
                # stricter is the safe direction, and it is documented rather than discovered.
                self.assertEqual(
                    set(),
                    baseline.lines_matching(join_regex(DEFAULT_PARTIAL + DEFAULT_PARTIAL_ALWAYS)),
                    "a branch-level default exclusion matches here",
                )

    def test_the_config_is_the_one_coverage_resolves(self):
        # coverage looks for `.coveragerc` in the directory it is run from and does not
        # search parents, so this runs from the strategy directory with the variable
        # emptied (`config.py:652-653` tests it for truthiness, so "" is unset).
        result = subprocess.run(
            [sys.executable, "-m", "coverage", "debug", "config"],
            cwd=COVERAGERC.parent,
            env={**os.environ, "COVERAGE_RCFILE": ""},
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        found = re.search(r"^\s*config_file:[ \t]*(.+?)[ \t]*$", result.stdout, re.MULTILINE)
        self.assertIsNotNone(found, result.stdout + result.stderr)
        self.assertEqual(COVERAGERC.resolve(), Path(found.group(1)).resolve())

    def _run_coverage(self, floor, *, tests_pass=True):
        with temporary_directory() as directory:
            source = directory / "source"
            tests = directory / "tests"
            source.mkdir()
            tests.mkdir()
            (source / "sample.py").write_text(
                "def choose(flag):\n    if flag:\n        return 1\n    return 2\n"
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

    def test_the_source_is_the_measured_tree(self):
        self.assertEqual(
            [line.strip() for line in self.parser["run"]["source"].splitlines() if line.strip()],
            ["ReforceXY/user_data"],
        )


if __name__ == "__main__":
    unittest.main()
