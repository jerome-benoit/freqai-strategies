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

Run each suite in its matching Freqtrade QA image, with the repository mounted
at `/workspace` and `/workspace` as the working directory:

```shell
# ReforceXY
python -m unittest discover -s ReforceXY/tests -v

# QuickAdapter
PYTHONPATH=/workspace/quickadapter/user_data/strategies \
  python -m unittest discover -s quickadapter/tests -v
```

Both commands must be run from the repository root, which is what the
container's `--workdir /workspace` provides. To select one concern, pass a
pattern that matches the whole `module.Class.method` name; a bare substring
selects more than you want, and a pattern that matches nothing runs zero tests
and fails with exit code 5:

```shell
PYTHONPATH=quickadapter/user_data/strategies \
  python -m unittest discover -s quickadapter/tests -k 'test_utils_zigzag.*' -v
```

CI runs type checks and runtime regressions in one QA matrix entry per strategy.
The shared runtime step sets each strategy's `PYTHONPATH`. QuickAdapter needs
this for direct `unittest` discovery because its model imports `LabelTransformer`
and `Utils` by bare names; ReforceXY resolves its imports without it, so the
setting is optional there. QuickAdapter's regressions additionally run under
`coverage.py`; the reward-space analysis suite runs separately with `uv`,
without a Freqtrade image.

## Coverage gate

Within the strategy QA matrix, QuickAdapter is the only strategy whose runtime
regressions enforce a coverage gate. The standalone reward-space analysis suite
has its own gate; see [its testing documentation](ReforceXY/reward_space_analysis/tests/README.md).
QuickAdapter's configuration is `quickadapter/.coveragerc`, selected explicitly
because coverage.py looks for `.coveragerc` in the directory it is run from and
does not search parents:

```shell
export PYTHONPATH=quickadapter/user_data/strategies
export COVERAGE_RCFILE=quickadapter/.coveragerc COVERAGE_FILE=/tmp/.coverage

sh scripts/run-coverage.sh -s quickadapter/tests -v
```

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
the pull request. `quickadapter/tests/test_coverage_floor.py` refuses a
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

The BasedPyright and type-stub versions are pinned in each project's
`.devcontainer/requirements-dev.txt`, as is `coverage` in QuickAdapter's.
The Freqtrade base images intentionally follow their rolling `stable_freqai`
and `stable_freqairl` tags, so record the resolved image digests when a
reproducible audit is required.
