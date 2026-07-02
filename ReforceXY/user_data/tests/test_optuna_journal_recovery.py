from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import freqaimodels.ReforceXY as reforcexy_module
import freqaimodels.optuna_journal_recovery as recovery_module
from freqaimodels.optuna_journal_recovery import (
    JOURNAL_TAIL_PROBE_BYTES,
    create_recovered_journal_storage,
    journal_has_corrupt_tail,
)
from optuna.storages import JournalStorage, RDBStorage

ReforceXY = reforcexy_module.ReforceXY


def _write_bytes(path: Path, content: bytes) -> None:
    path.write_bytes(content)


def _valid_record(index: int) -> bytes:
    return json.dumps(
        {
            "op_code": 0,
            "worker_id": f"worker-{index}",
            "study_name": f"study-{index}",
            "directions": [1],
        },
        separators=(",", ":"),
    ).encode()


def _assert_no_quarantine(journal_path: Path) -> None:
    matches = list(journal_path.parent.glob(f"{journal_path.name}.corrupt-*"))
    assert matches == []


def _assert_single_quarantine(journal_path: Path, expected_content: bytes) -> Path:
    matches = list(journal_path.parent.glob(f"{journal_path.name}.corrupt-*"))
    assert len(matches) == 1
    quarantined_path = matches[0]
    assert quarantined_path.read_bytes() == expected_content
    assert quarantined_path.name.startswith(f"{journal_path.name}.corrupt-")
    assert journal_path.exists()
    assert journal_path.read_bytes() == b""
    return quarantined_path


def _assert_recovered_storage(journal_path: Path) -> None:
    storage = create_recovered_journal_storage(journal_path)
    assert isinstance(storage, JournalStorage)


def _make_reforcexy_storage_model(
    full_path: Path, storage_backend: str
) -> ReforceXY:
    model = object.__new__(ReforceXY)
    model.full_path = full_path
    model.rl_config_optuna = {"storage": storage_backend}
    return model


def test_returns_false_for_missing_file() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: no journal file exists.
        journal_path = Path(temp_dir) / "optuna-BTC.log"

        # When: the helper probes the missing file.
        has_corrupt_tail = journal_has_corrupt_tail(journal_path)

        # Then: missing journals are treated as clean and are not quarantined.
        assert has_corrupt_tail is False
        _assert_recovered_storage(journal_path)
        _assert_no_quarantine(journal_path)


def test_returns_false_for_empty_file() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: an empty journal exists.
        journal_path = Path(temp_dir) / "optuna-BTC.log"
        journal_path.touch()

        # When: the helper probes the empty file.
        has_corrupt_tail = journal_has_corrupt_tail(journal_path)

        # Then: empty journals are treated as clean and are not quarantined.
        assert has_corrupt_tail is False
        _assert_recovered_storage(journal_path)
        _assert_no_quarantine(journal_path)


def test_returns_false_for_valid_journal() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: a journal whose trailing record is complete JSON plus newline.
        journal_path = Path(temp_dir) / "optuna-BTC.log"
        content = _valid_record(1) + b"\n"
        _write_bytes(journal_path, content)

        # When: the helper probes and constructs storage.
        has_corrupt_tail = journal_has_corrupt_tail(journal_path)
        storage = create_recovered_journal_storage(journal_path)

        # Then: the journal is left untouched.
        assert has_corrupt_tail is False
        assert isinstance(storage, JournalStorage)
        assert journal_path.read_bytes() == content
        _assert_no_quarantine(journal_path)


def test_quarantines_missing_trailing_newline() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: the trailing journal record has no newline terminator.
        journal_path = Path(temp_dir) / "optuna-BTC.log"
        content = _valid_record(1)
        _write_bytes(journal_path, content)

        # When: recovered storage is constructed.
        assert journal_has_corrupt_tail(journal_path) is True
        _assert_recovered_storage(journal_path)

        # Then: the corrupt journal is quarantined and a fresh journal is used.
        _assert_single_quarantine(journal_path, content)


def test_quarantines_empty_trailing_record() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: the journal ends with an empty trailing record after a newline.
        journal_path = Path(temp_dir) / "optuna-BTC.log"
        content = _valid_record(1) + b"\n\n"
        _write_bytes(journal_path, content)

        # When: recovered storage is constructed.
        assert journal_has_corrupt_tail(journal_path) is True
        _assert_recovered_storage(journal_path)

        # Then: the corrupt journal is quarantined and a fresh journal is used.
        _assert_single_quarantine(journal_path, content)


def test_quarantines_malformed_trailing_json() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: the final journal line is newline-terminated but malformed JSON.
        journal_path = Path(temp_dir) / "optuna-BTC.log"
        content = _valid_record(1) + b"\n{bad-json}\n"
        _write_bytes(journal_path, content)

        # When: recovered storage is constructed.
        assert journal_has_corrupt_tail(journal_path) is True
        _assert_recovered_storage(journal_path)

        # Then: the corrupt journal is quarantined and a fresh journal is used.
        _assert_single_quarantine(journal_path, content)


def test_quarantines_invalid_utf8_trailing_record() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: the final journal line is invalid UTF-8 JSON bytes.
        journal_path = Path(temp_dir) / "optuna-BTC.log"
        content = _valid_record(1) + b"\n\xff\n"
        _write_bytes(journal_path, content)

        # When: recovered storage is constructed.
        assert journal_has_corrupt_tail(journal_path) is True
        _assert_recovered_storage(journal_path)

        # Then: the corrupt journal is quarantined and a fresh journal is used.
        _assert_single_quarantine(journal_path, content)


def test_fails_open_when_probe_window_cuts_single_line() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: a single newline-terminated record exceeds the probe window.
        journal_path = Path(temp_dir) / "optuna-BTC.log"
        content = b'{"payload":"' + (b"x" * JOURNAL_TAIL_PROBE_BYTES) + b'"}\n'
        _write_bytes(journal_path, content)

        # When: the helper probes only the bounded tail window.
        has_corrupt_tail = journal_has_corrupt_tail(journal_path)

        # Then: it fails open instead of treating a cut line as corruption.
        assert has_corrupt_tail is False
        assert journal_path.read_bytes() == content
        _assert_no_quarantine(journal_path)


def test_quarantines_oversized_single_line_missing_trailing_newline() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: a single oversized journal line is missing its newline terminator.
        journal_path = Path(temp_dir) / "optuna-BTC.log"
        content = b'{"payload":"' + (b"x" * JOURNAL_TAIL_PROBE_BYTES) + b'"}'
        _write_bytes(journal_path, content)

        # When: recovered storage is constructed.
        assert journal_has_corrupt_tail(journal_path) is True
        _assert_recovered_storage(journal_path)

        # Then: missing newline wins over probe-window fail-open and quarantines.
        _assert_single_quarantine(journal_path, content)


def test_returns_false_when_probe_raises_oserror() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: the journal parent directory cannot be searched by the runtime user.
        parent_path = Path(temp_dir) / "blocked"
        parent_path.mkdir()
        journal_path = parent_path / "optuna-BTC.log"
        _write_bytes(journal_path, _valid_record(1) + b"\n")
        parent_path.chmod(0)

        try:
            # When: probing the journal raises an OSError from filesystem access.
            has_corrupt_tail = journal_has_corrupt_tail(journal_path)
        finally:
            parent_path.chmod(0o700)

        # Then: probing fails open and leaves operator-actionable recovery untouched.
        assert has_corrupt_tail is False


def _assert_replay_time_error_is_quarantined(error: Exception) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: the tail probe passes but storage replay construction fails once.
        journal_path = Path(temp_dir) / "optuna-BTC.log"
        content = _valid_record(1) + b"\n"
        _write_bytes(journal_path, content)
        calls = 0

        def build_storage(path: Path) -> JournalStorage:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise error
            return JournalStorage.__new__(JournalStorage)

        # When: recovered storage catches the replay-time corruption class.
        storage = create_recovered_journal_storage(journal_path, build_storage)

        # Then: it quarantines once and retries with a fresh journal.
        assert isinstance(storage, JournalStorage)
        assert calls == 2
        _assert_single_quarantine(journal_path, content)


def test_quarantines_replay_time_journal_storage_construction_errors() -> None:
    _assert_replay_time_error_is_quarantined(KeyError("study_id"))
    _assert_replay_time_error_is_quarantined(ValueError("replay failed"))
    _assert_replay_time_error_is_quarantined(
        json.JSONDecodeError("replay failed", "{", 0)
    )


def test_create_storage_sqlite_does_not_call_journal_recovery_helper() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: ReforceXY is configured for the default sqlite Optuna storage.
        model = _make_reforcexy_storage_model(Path(temp_dir), "sqlite")

        def fail_if_called(journal_path: Path) -> JournalStorage:
            raise AssertionError(f"sqlite storage called recovery helper for {journal_path}")

        original_helper = getattr(
            reforcexy_module, "create_recovered_journal_storage", None
        )
        reforcexy_module.create_recovered_journal_storage = fail_if_called

        try:
            # When: create_storage builds sqlite storage.
            storage = model.create_storage("BTC/USDT")
        finally:
            if original_helper is None:
                del reforcexy_module.create_recovered_journal_storage
            else:
                reforcexy_module.create_recovered_journal_storage = original_helper

        # Then: sqlite keeps the existing RDBStorage path and never touches the helper.
        assert isinstance(storage, RDBStorage)


def test_create_storage_file_delegates_valid_journal_to_recovery_helper() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: ReforceXY is configured for file-backed Optuna storage.
        storage_dir = Path(temp_dir)
        model = _make_reforcexy_storage_model(storage_dir, "file")
        expected_path = storage_dir / "optuna-BTC.log"
        expected_path.write_bytes(_valid_record(1) + b"\n")
        calls: list[Path] = []

        def build_storage(journal_path: Path) -> JournalStorage:
            calls.append(journal_path)
            return create_recovered_journal_storage(journal_path)

        original_helper = getattr(
            reforcexy_module, "create_recovered_journal_storage", None
        )
        reforcexy_module.create_recovered_journal_storage = build_storage

        try:
            # When: create_storage builds file-backed storage.
            storage = model.create_storage("BTC/USDT")
        finally:
            if original_helper is None:
                del reforcexy_module.create_recovered_journal_storage
            else:
                reforcexy_module.create_recovered_journal_storage = original_helper

        # Then: the file branch delegates to the verified recovery helper.
        assert isinstance(storage, JournalStorage)
        assert calls == [expected_path]
        assert expected_path.read_bytes() == _valid_record(1) + b"\n"
        _assert_no_quarantine(expected_path)


def test_create_storage_file_replay_corruption_quarantines_once_and_retries_once() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        # Given: the journal tail is valid but Optuna replay construction fails once.
        storage_dir = Path(temp_dir)
        model = _make_reforcexy_storage_model(storage_dir, "file")
        journal_path = storage_dir / "optuna-BTC.log"
        content = _valid_record(1) + b"\n"
        journal_path.write_bytes(content)
        calls = 0
        original_journal_storage = recovery_module.JournalStorage

        def build_storage(_backend: types.SimpleNamespace) -> JournalStorage:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyError("study_id")
            return original_journal_storage.__new__(original_journal_storage)

        recovery_module.JournalStorage = build_storage

        try:
            # When: create_storage delegates through the recovery helper.
            storage = model.create_storage("BTC/USDT")
        finally:
            recovery_module.JournalStorage = original_journal_storage

        # Then: replay-time corruption is quarantined exactly once and retried once.
        assert isinstance(storage, JournalStorage)
        assert calls == 2
        _assert_single_quarantine(journal_path, content)


def main() -> None:
    test_returns_false_for_missing_file()
    test_returns_false_for_empty_file()
    test_returns_false_for_valid_journal()
    test_quarantines_missing_trailing_newline()
    test_quarantines_empty_trailing_record()
    test_quarantines_malformed_trailing_json()
    test_quarantines_invalid_utf8_trailing_record()
    test_fails_open_when_probe_window_cuts_single_line()
    test_quarantines_oversized_single_line_missing_trailing_newline()
    test_returns_false_when_probe_raises_oserror()
    test_quarantines_replay_time_journal_storage_construction_errors()
    test_create_storage_sqlite_does_not_call_journal_recovery_helper()
    test_create_storage_file_delegates_valid_journal_to_recovery_helper()
    test_create_storage_file_replay_corruption_quarantines_once_and_retries_once()
    print("optuna journal recovery checks passed")


if __name__ == "__main__":
    main()
