# Contributing

Run checks from the repository root. Use the matching Freqtrade QA image for
strategy runtime and type checks; the standalone analysis suite uses its locked
uv environment. Operator guides belong with each strategy; economic claims must
follow the [evaluation protocol](docs/evaluation.md).

## Contents

- [Runtime regressions](#runtime-regressions)
- [Coverage gate](#coverage-gate)
- [Quality checks](#quality-checks)

## Runtime regressions

Run a strategy's suite inside its matching Freqtrade QA image, with the checkout
mounted at `/workspace` and that directory as the working directory. Use the
shared runner for canonical discovery and the coverage gate:

```shell
# Inside the ReforceXY QA image; select quickadapter inside its own QA image.
strategy=ReforceXY
export PYTHONPATH=/workspace/$strategy/user_data/strategies
export COVERAGE_RCFILE=$strategy/.coveragerc COVERAGE_FILE=/tmp/.coverage
sh scripts/run-coverage.sh -s "$strategy/tests" -v
```

Each suite's `test_config_template.py` applies the
[shared native assertion](scripts/config_template_contract.py) to its own shipped
template. It checks the unchanged default and independent/joint activation of
the two CCXT `rateLimit` examples. Only the selected comment markers are removed;
the parsed configuration must otherwise remain unchanged. Example values are
read from the template, not duplicated in the tests.

Both coverage commands need the same explicit configuration and data path; the
runner propagates test and coverage-report failures. QuickAdapter needs the
strategy path for bare-name imports; it is optional for ReforceXY.

For a focused debug run without coverage, use `python -m unittest discover` with
the same suite path and an exact `module.Class.method` pattern, for example
`-k 'test_model_pbrs_transitions.*' -v` for ReforceXY. A pattern matching no tests
fails with exit code 5. This debug run does not establish the coverage gate.

Each strategy QA matrix entry runs its type check and canonical coverage suite.
ReforceXY additionally runs once with `FREQAI_QA_SHUFFLE_SEED=1`. This permutes
methods within test classes, not module or class order. `QaTestCase` restores the
action-mask cache and Python, NumPy and CPU Torch RNG states before and after
each test. Lifecycle probes cover successful cases, body failures and setup or
teardown errors; each test still owns isolation of any other mutable state.
The reward-space analysis suite runs separately with `uv`, without a Freqtrade
image.

Regression tests should cover behavior, numerical boundaries and error conditions.
For private diagnostics, check the exception type and relevant input identity;
do not pin incidental wording, number formatting or arbitrary message lengths.

## Coverage gate

Both strategy suites measure their complete `user_data` trees, including branches
and non-imported namespace files. Their gates are defined in the linked coverage
configurations:

| Suite        | Coverage source                                      | Minimum |
| ------------ | ---------------------------------------------------- | ------- |
| QuickAdapter | [`quickadapter/user_data`](quickadapter/.coveragerc) | 68%     |
| ReforceXY    | [`ReforceXY/user_data`](ReforceXY/.coveragerc)       | 70%     |

Each strategy denominator excludes tests, the analytical package and the other
strategy. The standalone analysis suite has a separate 85% gate; see
[its testing documentation](ReforceXY/reward_space_analysis/tests/README.md).

Each strategy's `.coveragerc` is selected explicitly because coverage.py does
not search parent directories. Set `strategy=quickadapter` in its matching QA
image and use the same canonical runner command above for its own gate.

The shared runner gives both coverage commands the same environment. When running
`coverage run` and `coverage report` separately, export both variables for both
commands: prefixing only the run command does not configure the later report,
which then looks in its default data path and fails with `No data to report.`

Branch coverage distinguishes a guard's alternatives; statement coverage alone
does not prove both paths ran. `include_namespace_packages=true` enables the
discovery of completely unexecuted Python files inside namespace-package
subdirectories (directories without `__init__.py`). It does not mean imported
modules or an explicitly selected source root disappear when the flag is false.
The gate keeps it enabled so unexecuted nested files can contribute to the
denominator.

`fail_under` is an absolute floor, never per-module. To change it, re-measure
with the shipped configuration already in place — a measurement taken without
`branch` and `include_namespace_packages` reports a different denominator and
is not a valid input — then update the value and the `# measured` annotation
above it in the same commit. Raise the floor only; a drop needs the reason in
the pull request. Each strategy's `tests/test_coverage_floor.py` refuses a
placeholder, a missing or undated measurement annotation, a measurement below
the floor it justifies, a floor below the current minimum, a floor that is not a
percentage, a `precision` coarse enough to round the total past the floor, a
disabled branch trace, a disabled namespace walk, a source tree that is not the
measured one, a `fail_under` that has drifted into the inert `[run]` section, the
source-level `no cover` or `no branch` pragmas (with or without a colon),
ellipsis-only bodies, additional `TYPE_CHECKING` blocks beyond the existing
import-only block, and any `omit`, `include`, `exclude_lines`, `exclude_also` or
`partial_*` of production code. The shared `scripts/run-coverage.sh` runner is
exercised with real passing and failing coverage reports and a failing test suite;
the checks assert process exit status, not workflow token spelling.

## Quality checks

Run repository quality checks from the repository root:

Ruff does not need the Freqtrade runtime or project dependencies:

```shell
uvx ruff@latest check .
uvx ruff@latest format --check .
```

BasedPyright must run inside the matching Freqtrade QA image. The repository
wrapper records the sorted repository-relative identities of every configured
Python source, requires that inventory to match BasedPyright's analyzed-file
count, and compares it with every emitted diagnostic field—including an optional
rule and source range—against the project's exact snapshot. Build each QA target
and mount the checkout read-only:

```shell
# QuickAdapter
docker build --pull --target qa --tag freqai-strategies-quickadapter-qa quickadapter
docker run --rm \
  --mount "type=bind,src=$PWD,dst=/workspace,readonly" \
  --entrypoint python \
  freqai-strategies-quickadapter-qa \
  /workspace/scripts/check_basedpyright.py --project quickadapter

# ReforceXY
docker build --pull --target qa --tag freqai-strategies-reforcexy-qa ReforceXY
docker run --rm \
  --mount "type=bind,src=$PWD,dst=/workspace,readonly" \
  --entrypoint python \
  freqai-strategies-reforcexy-qa \
  /workspace/scripts/check_basedpyright.py --project reforcexy
```

The check fails when a configured source identity or diagnostic is added, removed,
moved, or changed, or when the analyzed-file count differs from the source
inventory. Each project's direct `include` entries must be normalized,
non-overlapping relative file or directory paths; glob syntax and symbolic links
are rejected. Snapshot updates are deliberate writable operations in the matching
QA image. For example:

```shell
docker run --rm \
  --mount "type=bind,src=$PWD,dst=/workspace" \
  --entrypoint python \
  freqai-strategies-quickadapter-qa \
  /workspace/scripts/check_basedpyright.py --project quickadapter --write
```

Review the generated `.basedpyright/diagnostics.json` diff. Use the ReforceXY image
and `--project reforcexy` for its snapshot. The writer preserves existing file
permissions and uses mode `0644` when creating a missing snapshot. Snapshot targets
must be regular files; symbolic links and other special files are rejected. The
wrapper rejects direct host and wrong-image execution so Freqtrade imports and
dependency versions remain exact.

The BasedPyright, type-stub and coverage versions are pinned in each project's
`.devcontainer/requirements-dev.txt`.
The Freqtrade base images intentionally follow their rolling `stable_freqai`
and `stable_freqairl` tags, so record the resolved image digests when a
reproducible audit is required.
