#!/bin/sh
# Run unittest discovery and propagate both test and coverage-report failures.
set -eu

python -m coverage run -m unittest discover "$@"
echo "::group::Coverage"
status=0
python -m coverage report || status=$?
echo "::endgroup::"
exit "$status"
