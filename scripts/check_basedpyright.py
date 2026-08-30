#!/usr/bin/env python3
"""Check BasedPyright diagnostics against exact, project-specific snapshots."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION = 4
CHECK_TIMEOUT_SECONDS = 300
QA_PROJECT_ENV = "FREQAI_QUALITY_PROJECT"
DEFAULT_SNAPSHOT_MODE = 0o644
ALLOWED_SEVERITIES = frozenset({"error", "warning", "information"})
SOURCE_SUFFIXES = frozenset({".py", ".pyi"})
INCLUDE_GLOB_CHARACTERS = frozenset("*?[]")
SNAPSHOT_KEYS = frozenset(
    {"schemaVersion", "basedpyrightVersion", "filesAnalyzed", "sourceFiles", "diagnostics"}
)
BASEDPYRIGHT_OUTPUT_KEYS = frozenset({"version", "time", "generalDiagnostics", "summary"})
BASEDPYRIGHT_SUMMARY_KEYS = frozenset(
    {"filesAnalyzed", "errorCount", "warningCount", "informationCount", "timeInSec"}
)
BASEDPYRIGHT_DIAGNOSTIC_REQUIRED_KEYS = frozenset({"file", "severity", "message"})
BASEDPYRIGHT_DIAGNOSTIC_OPTIONAL_KEYS = frozenset({"range", "rule"})
BASEDPYRIGHT_RANGE_KEYS = frozenset({"start", "end"})
BASEDPYRIGHT_POSITION_KEYS = frozenset({"line", "character"})
DIAGNOSTIC_REQUIRED_KEYS = frozenset({"file", "severity", "message"})
DIAGNOSTIC_RANGE_KEYS = frozenset({"startLine", "startCharacter", "endLine", "endCharacter"})
DIAGNOSTIC_OPTIONAL_KEYS = DIAGNOSTIC_RANGE_KEYS | {"rule"}
REPO_ROOT = Path(__file__).resolve().parent.parent


class QualityCheckError(RuntimeError):
    """A fail-closed quality check error."""


@dataclass(frozen=True)
class Project:
    config: str
    baseline: str
    qa_marker: str


PROJECTS = {
    "quickadapter": Project(
        config="quickadapter/pyrightconfig.json",
        baseline="quickadapter/.basedpyright/diagnostics.json",
        qa_marker="quickadapter",
    ),
    "reforcexy": Project(
        config="ReforceXY/pyrightconfig.json",
        baseline="ReforceXY/.basedpyright/diagnostics.json",
        qa_marker="reforcexy",
    ),
}


def _is_unicode_scalar_string(value: object, *, nonempty: bool = False) -> bool:
    return (
        isinstance(value, str)
        and (not nonempty or bool(value))
        and not any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    )


def _strict_json_loads(text: str) -> object:
    def reject_constant(value: str) -> None:
        raise QualityCheckError(f"Non-standard JSON constant: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise QualityCheckError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        return json.loads(
            text, parse_constant=reject_constant, object_pairs_hook=reject_duplicate_keys
        )
    except QualityCheckError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise QualityCheckError(f"Invalid JSON: {error}") from error


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise QualityCheckError(f"{label} must be a JSON object")
    return value


def _require_scalar_string(value: object, label: str, *, nonempty: bool = False) -> str:
    if not _is_unicode_scalar_string(value, nonempty=nonempty):
        qualifier = "nonempty " if nonempty else ""
        raise QualityCheckError(f"{label} must be a {qualifier}Unicode scalar string")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    if frozenset(value) != expected:
        raise QualityCheckError(f"{label} has missing or unknown keys")


def _require_allowed_keys(
    value: Mapping[str, object],
    required: frozenset[str],
    optional: frozenset[str],
    label: str,
) -> frozenset[str]:
    keys = frozenset(value)
    missing = required - keys
    if missing:
        raise QualityCheckError(f"{label} is missing required keys: {', '.join(sorted(missing))}")
    unknown = keys - required - optional
    if unknown:
        raise QualityCheckError(f"{label} has unknown keys: {', '.join(sorted(unknown))}")
    return keys


def _require_nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QualityCheckError(f"{label} must be a non-negative integer")
    return value


def _require_nonnegative_number(value: object, label: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        raise QualityCheckError(f"{label} must be a finite non-negative number")
    return value


def _validate_range(
    start_line: int, start_character: int, end_line: int, end_character: int, label: str
) -> None:
    if (start_line, start_character) > (end_line, end_character):
        raise QualityCheckError(f"{label} start must not follow its end")


def _resolve_repo_path(path: str, *, must_exist: bool) -> Path:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else REPO_ROOT / candidate).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as error:
        raise QualityCheckError(f"Path is outside the repository: {path}") from error
    if must_exist and not resolved.is_file():
        raise QualityCheckError(f"Repository file does not exist: {path}")
    return resolved


def _resolve_snapshot_path(path: str) -> Path:
    candidate = Path(path)
    absolute = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    if not absolute.name or absolute.name in {".", ".."}:
        raise QualityCheckError(f"Snapshot path must name a file: {path}")

    parent = absolute.parent.resolve()
    try:
        parent.relative_to(REPO_ROOT)
    except ValueError as error:
        raise QualityCheckError(f"Path escapes repository: {path}") from error
    if not parent.is_dir():
        raise QualityCheckError(
            f"Snapshot directory does not exist: {parent.relative_to(REPO_ROOT)}"
        )

    snapshot = parent / absolute.name
    try:
        snapshot_stat = snapshot.stat(follow_symlinks=False)
    except FileNotFoundError:
        return snapshot
    except OSError as error:
        raise QualityCheckError(
            f"Cannot inspect snapshot {snapshot.relative_to(REPO_ROOT)}: {error}"
        ) from error
    if not stat.S_ISREG(snapshot_stat.st_mode):
        raise QualityCheckError(
            f"Snapshot target must be a regular file: {snapshot.relative_to(REPO_ROOT)}"
        )
    return snapshot


def _read_utf8_file(path: Path, label: str) -> str:
    try:
        return path.read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise QualityCheckError(
            f"Cannot read {label} {path.relative_to(REPO_ROOT)}: {error}"
        ) from error


def _reject_symlink_components(path: Path, anchor: Path, label: str) -> None:
    current = anchor
    for part in path.relative_to(anchor).parts:
        current /= part
        try:
            mode = current.stat(follow_symlinks=False).st_mode
        except OSError as error:
            raise QualityCheckError(f"Cannot inspect {label} {current}: {error}") from error
        if stat.S_ISLNK(mode):
            raise QualityCheckError(f"{label} must not contain symbolic links: {current}")


def _collect_source_files(directory: Path) -> list[Path]:
    source_files: list[Path] = []
    for root, directory_names, file_names in os.walk(directory, followlinks=False):
        root_path = Path(root)
        for name in directory_names:
            child = root_path / name
            if child.is_symlink():
                raise QualityCheckError(f"Source scope must not contain symbolic links: {child}")
        for name in file_names:
            child = root_path / name
            if child.suffix not in SOURCE_SUFFIXES:
                continue
            try:
                mode = child.stat(follow_symlinks=False).st_mode
            except OSError as error:
                raise QualityCheckError(f"Cannot inspect source file {child}: {error}") from error
            if not stat.S_ISREG(mode):
                raise QualityCheckError(f"Source file must be a regular file: {child}")
            source_files.append(child)
    return source_files


def _source_files_from_config(config: Path) -> list[str]:
    config_data = _require_mapping(
        _strict_json_loads(_read_utf8_file(config, "BasedPyright config")),
        "BasedPyright config",
    )
    raw_includes = config_data.get("include")
    if not isinstance(raw_includes, list) or not raw_includes:
        raise QualityCheckError("BasedPyright config include must be a nonempty array")

    scopes: list[tuple[Path, bool]] = []
    for index, value in enumerate(raw_includes):
        label = f"BasedPyright config include[{index}]"
        include = _require_scalar_string(value, label, nonempty=True)
        if any(character in include for character in INCLUDE_GLOB_CHARACTERS):
            raise QualityCheckError(f"{label} must not use glob syntax")
        pure_path = PurePosixPath(include)
        if pure_path.is_absolute() or ".." in pure_path.parts or pure_path.as_posix() != include:
            raise QualityCheckError(f"{label} must be a normalized relative POSIX path")

        candidate = config.parent.joinpath(*pure_path.parts)
        try:
            candidate.relative_to(REPO_ROOT)
        except ValueError as error:
            raise QualityCheckError(f"{label} escapes the repository") from error
        _reject_symlink_components(candidate, config.parent, label)
        try:
            mode = candidate.stat(follow_symlinks=False).st_mode
        except OSError as error:
            raise QualityCheckError(f"Cannot inspect {label} {include}: {error}") from error
        is_directory = stat.S_ISDIR(mode)
        if not is_directory and not stat.S_ISREG(mode):
            raise QualityCheckError(f"{label} must identify a regular file or directory")
        if not is_directory and candidate.suffix not in SOURCE_SUFFIXES:
            raise QualityCheckError(f"{label} file must end in .py or .pyi")

        for previous, previous_is_directory in scopes:
            overlaps = candidate == previous
            if previous_is_directory:
                overlaps = overlaps or candidate.is_relative_to(previous)
            if is_directory:
                overlaps = overlaps or previous.is_relative_to(candidate)
            if overlaps:
                raise QualityCheckError(f"{label} overlaps another include scope")
        scopes.append((candidate, is_directory))

    source_paths: list[Path] = []
    for scope, is_directory in scopes:
        source_paths.extend(_collect_source_files(scope) if is_directory else [scope])
    source_files = sorted(path.relative_to(REPO_ROOT).as_posix() for path in source_paths)
    if not source_files:
        raise QualityCheckError("BasedPyright config include contains no Python source files")
    if len(source_files) != len(set(source_files)):
        raise QualityCheckError("BasedPyright config include produces duplicate source files")
    return source_files


def _normalize_file(path: object, label: str) -> str:
    raw_path = _require_scalar_string(path, label, nonempty=True)
    return _resolve_repo_path(raw_path, must_exist=True).relative_to(REPO_ROOT).as_posix()


def _normalize_child_diagnostic(value: object, index: int) -> dict[str, object]:
    label = f"generalDiagnostics[{index}]"
    diagnostic = _require_mapping(value, label)
    keys = _require_allowed_keys(
        diagnostic,
        BASEDPYRIGHT_DIAGNOSTIC_REQUIRED_KEYS,
        BASEDPYRIGHT_DIAGNOSTIC_OPTIONAL_KEYS,
        label,
    )
    severity = _require_scalar_string(diagnostic["severity"], f"{label}.severity", nonempty=True)
    if severity not in ALLOWED_SEVERITIES:
        raise QualityCheckError(f"{label}.severity is unsupported: {severity}")

    normalized: dict[str, object] = {
        "file": _normalize_file(diagnostic["file"], f"{label}.file"),
        "severity": severity,
        "message": _require_scalar_string(diagnostic["message"], f"{label}.message", nonempty=True),
    }
    if "rule" in keys:
        normalized["rule"] = _require_scalar_string(
            diagnostic["rule"], f"{label}.rule", nonempty=True
        )
    if "range" in keys:
        source_range = _require_mapping(diagnostic["range"], f"{label}.range")
        _require_exact_keys(source_range, BASEDPYRIGHT_RANGE_KEYS, f"{label}.range")
        start = _require_mapping(source_range["start"], f"{label}.range.start")
        _require_exact_keys(start, BASEDPYRIGHT_POSITION_KEYS, f"{label}.range.start")
        end = _require_mapping(source_range["end"], f"{label}.range.end")
        _require_exact_keys(end, BASEDPYRIGHT_POSITION_KEYS, f"{label}.range.end")
        start_line = _require_nonnegative_integer(start["line"], f"{label}.range.start.line")
        start_character = _require_nonnegative_integer(
            start["character"], f"{label}.range.start.character"
        )
        end_line = _require_nonnegative_integer(end["line"], f"{label}.range.end.line")
        end_character = _require_nonnegative_integer(
            end["character"], f"{label}.range.end.character"
        )
        _validate_range(start_line, start_character, end_line, end_character, f"{label}.range")
        normalized.update(
            startLine=start_line,
            startCharacter=start_character,
            endLine=end_line,
            endCharacter=end_character,
        )
    return normalized


def _diagnostic_sort_key(diagnostic: Mapping[str, object]) -> tuple[object, ...]:
    has_range = "startLine" in diagnostic
    has_rule = "rule" in diagnostic
    return (
        diagnostic["file"],
        has_range,
        diagnostic.get("startLine", 0),
        diagnostic.get("startCharacter", 0),
        diagnostic.get("endLine", 0),
        diagnostic.get("endCharacter", 0),
        diagnostic["severity"],
        has_rule,
        diagnostic.get("rule", ""),
        diagnostic["message"],
    )


def _files_analyzed_from_summary(value: object, diagnostics: Sequence[Mapping[str, object]]) -> int:
    summary = _require_mapping(value, "BasedPyright summary")
    _require_exact_keys(summary, BASEDPYRIGHT_SUMMARY_KEYS, "BasedPyright summary")
    files_analyzed = _require_nonnegative_integer(
        summary["filesAnalyzed"], "BasedPyright summary.filesAnalyzed"
    )
    _require_nonnegative_number(summary["timeInSec"], "BasedPyright summary.timeInSec")

    counts = dict.fromkeys(ALLOWED_SEVERITIES, 0)
    for diagnostic in diagnostics:
        counts[diagnostic["severity"]] += 1
    for severity, key in (
        ("error", "errorCount"),
        ("warning", "warningCount"),
        ("information", "informationCount"),
    ):
        reported = _require_nonnegative_integer(summary[key], f"BasedPyright summary.{key}")
        if reported != counts[severity]:
            raise QualityCheckError(
                f"BasedPyright summary.{key} is {reported}, expected {counts[severity]}"
            )
    return files_analyzed


def _snapshot_from_child(value: object, source_files: Sequence[str]) -> dict[str, object]:
    result = _require_mapping(value, "BasedPyright output")
    _require_exact_keys(result, BASEDPYRIGHT_OUTPUT_KEYS, "BasedPyright output")
    version = _require_scalar_string(result["version"], "BasedPyright version", nonempty=True)
    _require_scalar_string(result["time"], "BasedPyright time", nonempty=True)
    raw_diagnostics = result["generalDiagnostics"]
    if not isinstance(raw_diagnostics, list):
        raise QualityCheckError("BasedPyright generalDiagnostics must be an array")
    diagnostics = [
        _normalize_child_diagnostic(item, index) for index, item in enumerate(raw_diagnostics)
    ]
    files_analyzed = _files_analyzed_from_summary(result["summary"], diagnostics)
    if files_analyzed != len(source_files):
        raise QualityCheckError(
            f"BasedPyright analyzed {files_analyzed} files, expected {len(source_files)} configured sources"
        )
    diagnostics.sort(key=_diagnostic_sort_key)
    snapshot = {
        "schemaVersion": SCHEMA_VERSION,
        "basedpyrightVersion": version,
        "filesAnalyzed": files_analyzed,
        "sourceFiles": list(source_files),
        "diagnostics": diagnostics,
    }
    return _validate_snapshot(snapshot)


def _validate_snapshot_diagnostic(value: object, index: int) -> dict[str, object]:
    label = f"diagnostics[{index}]"
    diagnostic = _require_mapping(value, label)
    keys = _require_allowed_keys(
        diagnostic, DIAGNOSTIC_REQUIRED_KEYS, DIAGNOSTIC_OPTIONAL_KEYS, label
    )
    present_range_keys = keys & DIAGNOSTIC_RANGE_KEYS
    if present_range_keys and present_range_keys != DIAGNOSTIC_RANGE_KEYS:
        raise QualityCheckError(f"{label} must include all range fields or none")

    file_name = _require_scalar_string(diagnostic["file"], f"{label}.file", nonempty=True)
    pure_path = PurePosixPath(file_name)
    if pure_path.is_absolute() or ".." in pure_path.parts or pure_path.as_posix() != file_name:
        raise QualityCheckError(f"{label}.file must be a normalized repository-relative POSIX path")
    _resolve_repo_path(file_name, must_exist=True)

    severity = _require_scalar_string(diagnostic["severity"], f"{label}.severity", nonempty=True)
    if severity not in ALLOWED_SEVERITIES:
        raise QualityCheckError(f"{label}.severity is unsupported: {severity}")
    normalized: dict[str, object] = {
        "file": file_name,
        "severity": severity,
        "message": _require_scalar_string(diagnostic["message"], f"{label}.message", nonempty=True),
    }
    if "rule" in keys:
        normalized["rule"] = _require_scalar_string(
            diagnostic["rule"], f"{label}.rule", nonempty=True
        )
    if present_range_keys:
        start_line = _require_nonnegative_integer(diagnostic["startLine"], f"{label}.startLine")
        start_character = _require_nonnegative_integer(
            diagnostic["startCharacter"], f"{label}.startCharacter"
        )
        end_line = _require_nonnegative_integer(diagnostic["endLine"], f"{label}.endLine")
        end_character = _require_nonnegative_integer(
            diagnostic["endCharacter"], f"{label}.endCharacter"
        )
        _validate_range(start_line, start_character, end_line, end_character, label)
        normalized.update(
            startLine=start_line,
            startCharacter=start_character,
            endLine=end_line,
            endCharacter=end_character,
        )
    return normalized


def _validate_snapshot(value: object) -> dict[str, object]:
    snapshot = _require_mapping(value, "Snapshot")
    if frozenset(snapshot) != SNAPSHOT_KEYS:
        raise QualityCheckError("Snapshot has missing or unknown keys")
    schema_version = snapshot["schemaVersion"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise QualityCheckError(f"Snapshot schemaVersion must be {SCHEMA_VERSION}")
    version = _require_scalar_string(
        snapshot["basedpyrightVersion"], "Snapshot basedpyrightVersion", nonempty=True
    )
    files_analyzed = _require_nonnegative_integer(
        snapshot["filesAnalyzed"], "Snapshot filesAnalyzed"
    )

    raw_source_files = snapshot["sourceFiles"]
    if not isinstance(raw_source_files, list) or not raw_source_files:
        raise QualityCheckError("Snapshot sourceFiles must be a nonempty array")
    source_files: list[str] = []
    for index, raw_source_file in enumerate(raw_source_files):
        label = f"sourceFiles[{index}]"
        source_file = _require_scalar_string(raw_source_file, label, nonempty=True)
        pure_path = PurePosixPath(source_file)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or pure_path.as_posix() != source_file
            or pure_path.suffix not in SOURCE_SUFFIXES
        ):
            raise QualityCheckError(f"{label} must be a normalized repository-relative Python path")
        resolved = _resolve_repo_path(source_file, must_exist=True)
        if resolved.relative_to(REPO_ROOT).as_posix() != source_file:
            raise QualityCheckError(f"{label} must not contain symbolic links")
        if source_files and source_file <= source_files[-1]:
            raise QualityCheckError("Snapshot sourceFiles must be strictly sorted and unique")
        source_files.append(source_file)
    if files_analyzed != len(source_files):
        raise QualityCheckError(
            f"Snapshot filesAnalyzed is {files_analyzed}, expected {len(source_files)} sourceFiles"
        )

    raw_diagnostics = snapshot["diagnostics"]
    if not isinstance(raw_diagnostics, list):
        raise QualityCheckError("Snapshot diagnostics must be an array")
    diagnostics = [
        _validate_snapshot_diagnostic(item, index) for index, item in enumerate(raw_diagnostics)
    ]
    diagnostics.sort(key=_diagnostic_sort_key)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "basedpyrightVersion": version,
        "filesAnalyzed": files_analyzed,
        "sourceFiles": source_files,
        "diagnostics": diagnostics,
    }


def _canonical_bytes(snapshot: Mapping[str, object]) -> bytes:
    try:
        return (json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise QualityCheckError(f"Snapshot cannot be serialized: {error}") from error


def _validate_environment(project: Project) -> Path:
    marker = os.environ.get(QA_PROJECT_ENV)
    if marker != project.qa_marker:
        raise QualityCheckError(
            f"This check requires the {project.qa_marker} QA image ({QA_PROJECT_ENV}={project.qa_marker})"
        )
    freqtrade_root = Path("/freqtrade").resolve()
    if not freqtrade_root.is_dir():
        raise QualityCheckError("The QA image must provide /freqtrade")
    executable = Path(sys.executable).resolve()
    try:
        executable.relative_to(Path("/usr/local/bin").resolve())
    except ValueError as error:
        raise QualityCheckError(
            "The QA image Python interpreter must be under /usr/local/bin"
        ) from error
    return executable


def _run_basedpyright(config: Path, executable: Path) -> object:
    command = [
        str(executable),
        "-m",
        "basedpyright",
        "--outputjson",
        "--warnings",
        "--pythonpath",
        str(executable),
        "--project",
        str(config),
    ]
    try:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, capture_output=True, check=False, timeout=CHECK_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired as error:
        raise QualityCheckError(f"BasedPyright exceeded {CHECK_TIMEOUT_SECONDS} seconds") from error
    except OSError as error:
        raise QualityCheckError(f"Could not execute BasedPyright: {error}") from error

    try:
        stderr = completed.stderr.decode("utf-8", errors="strict")
        stdout = completed.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise QualityCheckError(f"BasedPyright output is not valid UTF-8: {error}") from error
    if stderr:
        raise QualityCheckError(f"BasedPyright wrote to stderr:\n{stderr}")
    if completed.returncode not in (0, 1):
        raise QualityCheckError(f"BasedPyright exited with code {completed.returncode}")
    return _strict_json_loads(stdout)


def _read_snapshot(path: Path) -> dict[str, object]:
    relative_path = path.relative_to(REPO_ROOT)
    descriptor: int | None = None
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise QualityCheckError(f"Snapshot target must be a regular file: {relative_path}")
        snapshot_file = os.fdopen(descriptor, "rb")
        descriptor = None
        with snapshot_file:
            content = snapshot_file.read()
    except FileNotFoundError as error:
        raise QualityCheckError(f"Snapshot does not exist: {relative_path}") from error
    except OSError as error:
        raise QualityCheckError(f"Cannot read snapshot {relative_path}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise QualityCheckError(f"Cannot read snapshot {relative_path}: {error}") from error
    return _validate_snapshot(_strict_json_loads(text))


def _atomic_write(path: Path, content: bytes) -> None:
    temporary_path: Path | None = None
    try:
        try:
            target_stat = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            target_mode = DEFAULT_SNAPSHOT_MODE
        else:
            if not stat.S_ISREG(target_stat.st_mode):
                raise QualityCheckError(
                    f"Snapshot target must be a regular file: {path.relative_to(REPO_ROOT)}"
                )
            target_mode = stat.S_IMODE(target_stat.st_mode)

        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fchmod(temporary.fileno(), target_mode)
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
        temporary_path = None
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise QualityCheckError(
            f"Cannot atomically update {path.relative_to(REPO_ROOT)}: {error}"
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _check_project(project_name: str, *, write: bool) -> int:
    project = PROJECTS[project_name]
    config = _resolve_repo_path(project.config, must_exist=True)
    baseline = _resolve_snapshot_path(project.baseline)
    executable = _validate_environment(project)
    source_files = _source_files_from_config(config)
    current_snapshot = _snapshot_from_child(_run_basedpyright(config, executable), source_files)
    current_bytes = _canonical_bytes(current_snapshot)

    if write:
        _atomic_write(baseline, current_bytes)
        print(
            f"Updated {baseline.relative_to(REPO_ROOT)} with "
            f"{len(current_snapshot['diagnostics'])} accepted diagnostics across "
            f"{current_snapshot['filesAnalyzed']} analyzed files"
        )
        return 0

    stored_snapshot = _read_snapshot(baseline)
    stored_bytes = _canonical_bytes(stored_snapshot)
    if stored_bytes == current_bytes:
        print(
            f"BasedPyright snapshot matches for {project_name}: "
            f"{len(current_snapshot['diagnostics'])} diagnostics across "
            f"{current_snapshot['filesAnalyzed']} analyzed files"
        )
        return 0

    diff = difflib.unified_diff(
        stored_bytes.decode("utf-8").splitlines(),
        current_bytes.decode("utf-8").splitlines(),
        fromfile=str(baseline.relative_to(REPO_ROOT)),
        tofile=f"current:{project_name}",
        lineterm="",
    )
    print("BasedPyright diagnostics differ from the accepted snapshot:", file=sys.stderr)
    print("\n".join(diff), file=sys.stderr)
    return 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=sorted(PROJECTS), required=True)
    parser.add_argument(
        "--write", action="store_true", help="Atomically replace the selected project's snapshot"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return _check_project(args.project, write=args.write)
    except QualityCheckError as error:
        print(f"BasedPyright snapshot check failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
