from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

logger = logging.getLogger(__name__)

JOURNAL_TAIL_PROBE_BYTES: Final[int] = 64 * 1024
_JOURNAL_QUARANTINE_TAG: Final[str] = "corrupt"
_JOURNAL_QUARANTINE_TIE_BREAK_LIMIT: Final[int] = 99
_JOURNAL_OP_CODE_KEY: Final[str] = "op_code"
_JOURNAL_OPERATION_CODES: Final[frozenset[int]] = frozenset(range(10))
_JOURNAL_RECOVERABLE_ERRORS: Final[type[Exception] | tuple[type[Exception], ...]] = (
    KeyError,
    AssertionError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)
_StorageFactory = Callable[[Path], JournalStorage]


def journal_has_corrupt_tail(journal_path: Path) -> bool:
    try:
        if not journal_path.exists():
            return False
        size = journal_path.stat().st_size
        if size == 0:
            return False
        with journal_path.open("rb") as journal_file:
            journal_file.seek(max(0, size - JOURNAL_TAIL_PROBE_BYTES))
            tail = journal_file.read()
            if not tail.endswith(b"\n"):
                return True
            last_newline = tail.rfind(b"\n", 0, len(tail) - 1)
            while last_newline == -1 and len(tail) < size:
                read_end = size - len(tail)
                read_start = max(0, read_end - JOURNAL_TAIL_PROBE_BYTES)
                journal_file.seek(read_start)
                tail = journal_file.read(read_end - read_start) + tail
                last_newline = tail.rfind(b"\n", 0, len(tail) - 1)
    except OSError:
        return False

    if last_newline == -1:
        last_line = tail[:-1]
        fail_open_missing_op_code = size > JOURNAL_TAIL_PROBE_BYTES
    else:
        last_line = tail[last_newline + 1 : -1]
        fail_open_missing_op_code = False
    if not last_line:
        return True
    try:
        record = json.loads(last_line)
    except ValueError:
        return True
    if not isinstance(record, dict):
        return True
    op_code = record.get(_JOURNAL_OP_CODE_KEY)
    if op_code is None and fail_open_missing_op_code:
        return False
    return type(op_code) is not int or op_code not in _JOURNAL_OPERATION_CODES


def create_recovered_journal_storage(
    journal_path: Path,
    storage_factory: _StorageFactory | None = None,
) -> JournalStorage:
    build_storage = storage_factory or _create_journal_storage
    if journal_has_corrupt_tail(journal_path):
        _quarantine_journal(
            journal_path,
            ValueError("trailing journal record is truncated or malformed JSON"),
        )
        journal_path.touch()
    try:
        return build_storage(journal_path)
    except _JOURNAL_RECOVERABLE_ERRORS as exc:
        quarantined_path = _quarantine_journal(journal_path, exc)
        if quarantined_path is None:
            raise
        journal_path.touch()
        return build_storage(journal_path)


def _create_journal_storage(journal_path: Path) -> JournalStorage:
    return JournalStorage(JournalFileBackend(str(journal_path)))


def _quarantine_journal(journal_path: Path, cause: Exception) -> Path | None:
    if not journal_path.exists():
        return None
    quarantine_path = _quarantine_path(journal_path, datetime.now(timezone.utc))
    journal_path.rename(quarantine_path)
    logger.warning(
        "Optuna journal %s corrupt (%r); quarantined to %s; resuming with fresh journal",
        journal_path.name,
        cause,
        quarantine_path.name,
    )
    return quarantine_path


def _quarantine_path(journal_path: Path, now: datetime) -> Path:
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    base_name = f"{journal_path.name}.{_JOURNAL_QUARANTINE_TAG}-{stamp}"
    for index in range(_JOURNAL_QUARANTINE_TIE_BREAK_LIMIT + 1):
        suffix = "" if index == 0 else f"-{index}"
        candidate = journal_path.with_name(f"{base_name}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(journal_path)
