"""Durable Optuna studies, recovery and atomic best-parameter persistence."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import optuna
from optuna.distributions import FloatDistribution
from optuna.storages import RDBStorage
from optuna.trial import TrialState
from qa_support import PAIR, REPO_ROOT, QaTestCase, model_config

from ReforceXY.user_data.freqaimodels.ReforceXY import ReforceXY

MODULE = "ReforceXY.user_data.freqaimodels.ReforceXY"


class OptunaStorageTest(QaTestCase):
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = ReforceXY(config=model_config(temporary.name))
        self.addCleanup(self.model.close_envs)

    def storage(self, backend="file", pair=PAIR):
        self.model.rl_config_optuna["storage"] = backend
        storage = self.model.create_storage(pair)
        if isinstance(storage, RDBStorage):
            self.addCleanup(storage.engine.dispose)
        return storage

    def study(self, storage, name="winner"):
        study = optuna.create_study(storage=storage, study_name=name, direction="maximize")
        study.add_trial(
            optuna.create_trial(
                value=2.0,
                params={"gamma": 0.83},
                distributions={"gamma": FloatDistribution(0.0, 1.0)},
            )
        )
        return study

    def test_backends_reopen_winners_and_keep_studies_sharing_a_base_asset_isolated(self):
        for backend in ("sqlite", "file"):
            with self.subTest(backend=backend):
                first = self.study(self.storage(backend), "BTC/USDT")
                other = self.study(self.storage(backend, "BTC/EUR"), "BTC/EUR")
                first.add_trial(optuna.create_trial(value=4.0))
                reopened = optuna.load_study(study_name="BTC/USDT", storage=self.storage(backend))
                self.assertEqual(reopened.best_value, 4.0)
                self.assertEqual(len(reopened.trials), 2)
                self.assertEqual(other.best_value, 2.0)
                self.assertEqual(len(other.trials), 1)
                ReforceXY.delete_study("BTC/USDT", self.storage(backend))
                with self.assertRaises(KeyError):
                    optuna.load_study(study_name="BTC/USDT", storage=self.storage(backend))
                ReforceXY.delete_study("BTC/USDT", self.storage(backend))
                self.assertEqual(other.best_params, {"gamma": 0.83})

    def test_only_complete_trials_make_a_best_trial_available(self):
        storage = self.storage()
        study = optuna.create_study(storage=storage, study_name="availability")
        self.assertFalse(ReforceXY.study_has_best_trial(None))
        self.assertFalse(ReforceXY.study_has_best_trial(study))
        study.add_trial(optuna.create_trial(state=TrialState.FAIL))
        study.add_trial(optuna.create_trial(state=TrialState.PRUNED))
        self.assertFalse(ReforceXY.study_has_best_trial(study))
        study.add_trial(optuna.create_trial(value=3.0))
        self.assertTrue(ReforceXY.study_has_best_trial(study))
        self.assertEqual(study.best_value, 3.0)

    def test_retrain_counters_survive_restart_and_recover_invalid_payloads(self):
        model = self.model
        self.assertEqual(model._increment_optuna_retrain_counter("BTC/USDT:USDT"), 1)
        self.assertEqual(model._increment_optuna_retrain_counter("ETH/USDT"), 1)
        restarted = ReforceXY.__new__(ReforceXY)
        restarted.full_path = model.full_path
        self.assertEqual(restarted._increment_optuna_retrain_counter("BTC/USDT:USDT"), 2)
        self.assertEqual(
            restarted._load_optuna_retrain_counters(PAIR), {"BTC_USDT_USDT": 2, "ETH_USDT": 1}
        )
        path = model._optuna_retrain_counters_path()
        path.write_text(json.dumps({"BTC_USDT": 4, "bad": "4", "other": []}), encoding="utf-8")
        self.assertEqual(model._increment_optuna_retrain_counter(PAIR), 5)
        self.assertEqual(model._load_optuna_retrain_counters(PAIR), {"BTC_USDT": 5})
        for payload in ("[]", "{broken"):
            path.write_text(payload, encoding="utf-8")
            self.assertEqual(model._increment_optuna_retrain_counter(PAIR), 1)
        path.unlink()
        path.mkdir()
        self.assertEqual(model._increment_optuna_retrain_counter(PAIR), 1)
        self.assertTrue(path.is_dir())

    def test_corrupt_journal_tails_are_preserved_and_new_trials_survive_restart(self):
        path = self.model.full_path / "optuna-BTC.log"
        for index, tail in enumerate(
            (
                b"partial",
                b"{invalid}\n",
                b"\n",
                b"[]\n",
                b'{"op_code":true}\n',
                b'{"op_code":99}\n',
                b"{}\n",
            )
        ):
            with self.subTest(tail=tail):
                path.unlink(missing_ok=True)
                self.study(self.storage(), f"old-{index}")
                with path.open("ab") as stream:
                    stream.write(tail)
                corrupt = path.read_bytes()
                recovered = self.storage()
                quarantines = list(path.parent.glob(path.name + ".corrupt-*"))
                self.assertIn(corrupt, [item.read_bytes() for item in quarantines])
                with self.assertRaises(KeyError):
                    optuna.load_study(study_name=f"old-{index}", storage=recovered)
                self.study(recovered, f"fresh-{index}")
                reopened = optuna.load_study(study_name=f"fresh-{index}", storage=self.storage())
                self.assertEqual(reopened.best_params, {"gamma": 0.83})
                self.assertEqual(reopened.best_value, 2.0)

    def test_interior_corruption_is_detected_by_real_journal_replay(self):
        self.study(self.storage())
        path = self.model.full_path / "optuna-BTC.log"
        records = path.read_bytes().splitlines(keepends=True)
        corrupt = records[0] + b"{invalid}\n" + b"".join(records[1:])
        path.write_bytes(corrupt)
        self.assertFalse(ReforceXY._journal_has_corrupt_tail(path))
        recovered = self.storage()
        with self.assertRaises(KeyError):
            optuna.load_study(study_name="winner", storage=recovered)
        self.assertEqual(next(path.parent.glob(path.name + ".corrupt-*")).read_bytes(), corrupt)

    def test_large_journal_records_replay_without_false_quarantine(self):
        study = self.study(self.storage())
        padding = "x" * (ReforceXY._JOURNAL_TAIL_PROBE_BYTES + 1024)
        study.set_user_attr("padding", padding)
        path = self.model.full_path / "optuna-BTC.log"
        reopened = optuna.load_study(study_name="winner", storage=self.storage())
        self.assertEqual(reopened.user_attrs["padding"], padding)
        self.assertEqual(reopened.best_value, 2.0)
        self.assertEqual(list(path.parent.glob(path.name + ".corrupt-*")), [])
        # A single oversized malformed operation passes the bounded tail probe,
        # but must still be rejected by full replay and preserved for diagnosis.
        path.write_text(json.dumps({"padding": padding}) + "\n", encoding="utf-8")
        corrupt = path.read_bytes()
        recovered = self.storage()
        self.study(recovered, "replacement")
        self.assertEqual(next(path.parent.glob(path.name + ".corrupt-*")).read_bytes(), corrupt)
        self.assertEqual(
            optuna.load_study(study_name="replacement", storage=self.storage()).best_value, 2.0
        )

    def test_quarantine_collision_never_overwrites_existing_evidence(self):
        path = self.model.full_path / "optuna-BTC.log"
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with mock.patch(MODULE + ".datetime") as clock:
            clock.now.return_value = now
            path.write_bytes(b"first truncated operation")
            self.storage()
            first = next(path.parent.glob(path.name + ".corrupt-*"))
            path.write_bytes(b"second truncated operation")
            self.storage()
            self.assertEqual(first.read_bytes(), b"first truncated operation")
            self.assertEqual(
                {item.read_bytes() for item in path.parent.glob(path.name + ".corrupt-*")},
                {b"first truncated operation", b"second truncated operation"},
            )
            for index in range(2, ReforceXY._QUARANTINE_TIE_BREAK_LIMIT + 1):
                first.with_name(first.name + f"-{index}").write_bytes(b"evidence")
            path.write_bytes(b"third truncated operation")
            with self.assertRaises(FileExistsError):
                self.storage()
            self.assertEqual(path.read_bytes(), b"third truncated operation")
            self.assertEqual(first.read_bytes(), b"first truncated operation")

    def test_best_params_are_atomic_pair_specific_and_preserve_existing_permissions(self):
        model = self.model
        first = {"gamma": 0.83, "target_kl": None, "nested": {"width": [8, 16]}}
        model.save_best_trial_params(first, PAIR)
        model.save_best_trial_params({"gamma": 0.91}, "ETH/USDT:USDT")
        path = model._best_trial_params_path(PAIR)
        path.chmod(0o640)
        restarted = ReforceXY.__new__(ReforceXY)
        restarted.full_path = model.full_path
        self.assertEqual(restarted.load_best_trial_params(PAIR), first)
        model.save_best_trial_params({"gamma": 0.97}, PAIR)
        self.assertEqual(restarted.load_best_trial_params(PAIR), {"gamma": 0.97})
        self.assertEqual(path.stat().st_mode & 0o777, 0o640)
        self.assertEqual(restarted.load_best_trial_params("ETH/USDT:USDT"), {"gamma": 0.91})

    def test_invalid_identity_and_payloads_are_ignored_without_destroying_the_file(self):
        path = self.model._best_trial_params_path(PAIR)
        self.assertIsNone(self.model.load_best_trial_params(PAIR))
        for payload in (
            [],
            {},
            {"params": {"gamma": 0.1}},
            {"objective_identity": "legacy", "params": {"gamma": 0.1}},
            {"objective_identity": ReforceXY._OPTUNA_OBJECTIVE_IDENTITY, "params": []},
        ):
            with self.subTest(payload=payload):
                encoded = json.dumps(payload).encode()
                path.write_bytes(encoded)
                self.assertIsNone(self.model.load_best_trial_params(PAIR))
                self.assertEqual(path.read_bytes(), encoded)
                self.assertEqual(list(path.parent.glob(path.name + ".corrupt-*")), [])
        model = ReforceXY.__new__(ReforceXY)
        model.full_path = self.root / "absent"
        self.assertIsNone(model.load_best_trial_params(PAIR))

    def test_malformed_best_params_are_quarantined_and_can_be_replaced(self):
        path = self.model._best_trial_params_path(PAIR)
        for corrupt in (b"{broken", b"\xff"):
            with self.subTest(corrupt=corrupt):
                path.write_bytes(corrupt)
                self.assertIsNone(self.model.load_best_trial_params(PAIR))
                self.assertFalse(path.exists())
                self.assertIn(
                    corrupt,
                    [item.read_bytes() for item in path.parent.glob(path.name + ".corrupt-*")],
                )
                self.model.save_best_trial_params({"gamma": 0.83}, PAIR)
                self.assertEqual(self.model.load_best_trial_params(PAIR), {"gamma": 0.83})

    def test_failed_or_interrupted_save_keeps_previous_version_and_cleans_temporary_file(self):
        self.model.save_best_trial_params({"gamma": 0.83}, PAIR)
        with self.assertRaises(TypeError):
            self.model.save_best_trial_params({"bad": object()}, PAIR)
        for error in (OSError("disk unavailable"), KeyboardInterrupt()):
            with (
                self.subTest(error=type(error).__name__),
                mock.patch(MODULE + ".os.fsync", side_effect=error),
                self.assertRaises(type(error)),
            ):
                self.model.save_best_trial_params({"gamma": 0.97}, PAIR)
            self.assertEqual(self.model.load_best_trial_params(PAIR), {"gamma": 0.83})
            self.assertEqual(list(self.model.full_path.glob(".*.tmp")), [])

    def test_symlink_targets_and_nonregular_locks_are_rejected_without_modification(self):
        target = self.root / "witness.json"
        target.write_bytes(b"witness")
        path = self.model._best_trial_params_path(PAIR)
        path.symlink_to(target)
        with self.assertRaises(OSError):
            self.model.load_best_trial_params(PAIR)
        with self.assertRaises(OSError):
            self.model.save_best_trial_params({"gamma": 0.83}, PAIR)
        self.assertEqual(target.read_bytes(), b"witness")
        path.unlink()
        lock = self.model.full_path / ReforceXY._BEST_PARAMS_LOCK_FILENAME
        lock.unlink(missing_ok=True)
        lock.symlink_to(target)
        with self.assertRaises(OSError):
            self.model.save_best_trial_params({"gamma": 0.83}, PAIR)
        self.assertEqual(target.read_bytes(), b"witness")
        lock.unlink()
        os.mkfifo(lock)
        # A subprocess timeout also protects against a regression to blocking open().
        script = (
            "from pathlib import Path; import sys; "
            "from ReforceXY.user_data.freqaimodels.ReforceXY import ReforceXY; "
            "m=ReforceXY.__new__(ReforceXY); m.full_path=Path(sys.argv[1]); "
            "m.load_best_trial_params('BTC/USDT')"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.model.full_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("OSError", result.stderr)
        self.assertTrue(lock.is_fifo())

    def test_reader_without_existing_lock_does_not_create_one(self):
        self.model.save_best_trial_params({"gamma": 0.83}, PAIR)
        lock = self.model.full_path / ReforceXY._BEST_PARAMS_LOCK_FILENAME
        lock.unlink()
        self.assertEqual(self.model.load_best_trial_params(PAIR), {"gamma": 0.83})
        self.assertFalse(lock.exists())

    def test_writer_repair_or_removal_between_reads_is_not_quarantined(self):
        path = self.model._best_trial_params_path(PAIR)
        original_lock = ReforceXY._locked_best_trial_params
        for repair in (True, False):
            path.write_bytes(b"{broken")

            @contextmanager
            def intervening_writer(path, *, exclusive, repair=repair):
                with original_lock(path, exclusive=exclusive):
                    if exclusive:
                        if repair:
                            path.write_text(
                                json.dumps(
                                    {
                                        "objective_identity": ReforceXY._OPTUNA_OBJECTIVE_IDENTITY,
                                        "params": {"gamma": 0.97},
                                    }
                                ),
                                encoding="utf-8",
                            )
                        else:
                            path.unlink()
                    yield

            with (
                self.subTest(repair=repair),
                mock.patch.object(
                    ReforceXY, "_locked_best_trial_params", staticmethod(intervening_writer)
                ),
            ):
                self.assertEqual(
                    self.model.load_best_trial_params(PAIR), {"gamma": 0.97} if repair else None
                )
            self.assertEqual(list(path.parent.glob(path.name + ".corrupt-*")), [])

    def test_quarantine_failure_is_visible_and_leaves_corrupt_evidence_in_place(self):
        path = self.model._best_trial_params_path(PAIR)
        path.write_bytes(b"{broken")
        with (
            mock.patch.object(Path, "rename", side_effect=PermissionError("read-only evidence")),
            self.assertRaises(PermissionError),
        ):
            self.model.load_best_trial_params(PAIR)
        self.assertEqual(path.read_bytes(), b"{broken")

    def test_unsupported_storage_is_rejected_instead_of_falling_back(self):
        with self.assertRaises(ValueError):
            self.storage("unsupported")


if __name__ == "__main__":
    unittest.main()
