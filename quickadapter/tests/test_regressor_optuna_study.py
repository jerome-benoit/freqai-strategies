"""Optuna study lifecycle — create / optimize / validate / enqueue / load / delete / has-best; requires the Freqtrade QA image."""

import datetime
import json
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import numpy as np
import optuna
from EnumErrors import enum_error_message
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import FrozenTrial, TrialState
from qa_support import PAIR, QaTestCase, model_config, temporary_directory

from quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3 import (
    QuickAdapterRegressorV3,
)

REGRESSOR = QuickAdapterRegressorV3
LABEL_DIRECTIONS = list(REGRESSOR._OPTUNA_LABEL_DIRECTIONS)
HP_IDENTITY = REGRESSOR._OPTUNA_HP_OBJECTIVE_IDENTITY
LOGGER = "quickadapter.user_data.freqaimodels.QuickAdapterRegressorV3"
ALPHA = FloatDistribution(0.0, 1.0)
LABEL_CANDLES = IntDistribution(1, 20)
EPOCH = datetime.datetime(2024, 1, 1)
LABEL_PARAMS = {"label_period_candles": 3, "label_natr_multiplier": 1.25}


def regressor(
    tmp_path: Path, *, live: bool = False, **optuna_hyperopt: object
) -> QuickAdapterRegressorV3:
    """Build a regressor without running __init__, injecting only what the study lifecycle reads.

    ``model_config`` pins both samplers to ``tpe``; ``optuna_hyperopt`` overrides are
    deep-merged on top, so no test can reach ``optunahub``'s outbound ``auto`` fetch.
    """
    config = model_config(tmp_path, freqai={"optuna_hyperopt": optuna_hyperopt})
    model = object.__new__(REGRESSOR)
    model.config = config
    model.freqai_info = config["freqai"]
    model.ft_params = {"label_method": REGRESSOR.LABEL_METHOD_DEFAULT}
    model.live = live
    model.pairs = [PAIR]
    # set_full_path's own arithmetic; the directories optuna_create_storage needs must exist.
    model.full_path = Path(config["user_data_dir"]) / "models" / config["freqai"]["identifier"]
    model.full_path.mkdir(parents=True, exist_ok=True)
    model._optuna_hp_value = {PAIR: -1.0}
    model._optuna_hp_params = {PAIR: {}}
    model._optuna_label_values = {PAIR: [-1.0] * REGRESSOR._OPTUNA_LABEL_N_OBJECTIVES}
    model._optuna_label_params = {PAIR: {"label_period_candles": 1, "label_natr_multiplier": 1.0}}
    return model


def frozen(
    number: int = 0,
    *,
    state: TrialState = TrialState.COMPLETE,
    value: float | None = None,
    values: list[float] | None = None,
    params: dict | None = None,
    distributions: dict | None = None,
) -> FrozenTrial:
    """Build a FrozenTrial for optuna 5.0, whose __init__ requires all eleven positionals.

    ``trial_id`` is the eleventh and is required, so it is passed explicitly; ``values`` is
    keyword-only. ``datetime_complete`` is set only for a finished state, which is what
    optuna's own validator demands.
    """
    return FrozenTrial(
        number,
        state,
        value,
        EPOCH,
        EPOCH if state.is_finished() else None,
        params or {},
        distributions or {},
        {},
        {},
        {},
        number,
        values=values,
    )


def trial_numbers(study: optuna.study.Study) -> list[int]:
    return [trial.number for trial in study.get_trials(deepcopy=False)]


def waiting_params(study: optuna.study.Study) -> list[dict]:
    return [
        trial.system_attrs["fixed_params"]
        for trial in study.get_trials(deepcopy=False, states=(TrialState.WAITING,))
    ]


def tree(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def study_name(model: QuickAdapterRegressorV3, pair: str, namespace: str) -> str:
    return f"{model.freqai_info['identifier']}-{pair}-{namespace}"


def best_params_path(model: QuickAdapterRegressorV3, namespace: str) -> Path:
    return model.full_path / f"optuna-{namespace}-best-params-BTC_USDT.json"


def rewrite_payload(path: Path, mutate: Callable[[dict], None]) -> None:
    """Apply ``mutate`` to the on-disk best-params payload and write it back."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def assert_logged(test: unittest.TestCase, records: list[str], fragment: str) -> None:
    """Assert some captured record contains ``fragment``, not necessarily the first.

    A single ``optuna_create_study`` call can emit more than one warning (the marker
    inspection warning plus the drift warning), so pinning an index would make these
    assertions depend on emission order rather than on content.
    """
    test.assertTrue(
        any(fragment in record for record in records), f"{fragment!r} absent from {records}"
    )


def single_study(model: QuickAdapterRegressorV3, pair: str = PAIR) -> optuna.study.Study:
    return model.optuna_create_study(pair, "hp", direction=optuna.study.StudyDirection.MAXIMIZE)


def multi_study(model: QuickAdapterRegressorV3, pair: str = PAIR) -> optuna.study.Study:
    return model.optuna_create_study(pair, "label", directions=LABEL_DIRECTIONS)


def seed_trial(
    model: QuickAdapterRegressorV3, study: optuna.study.Study, *, legacy_identity: bool = False
) -> optuna.study.Study:
    """Add one completed trial to ``study``, stamping a drifted marker first if asked.

    The trial shape follows the study's own arity: optuna refuses a single-objective
    FrozenTrial on a multi-objective study and vice versa.
    """
    target = study
    if legacy_identity:
        target = optuna.load_study(
            study_name=study.study_name, storage=model.optuna_create_storage(PAIR)
        )
        target.set_user_attr("objective_identity", "legacy-identity-v0")
    if len(study.directions) > 1:
        study.add_trial(
            frozen(
                0,
                values=[0.5] * REGRESSOR._OPTUNA_LABEL_N_OBJECTIVES,
                params={"label_period_candles": 3},
                distributions={"label_period_candles": LABEL_CANDLES},
            )
        )
    else:
        study.add_trial(frozen(0, value=0.5, params={"alpha": 0.5}, distributions={"alpha": ALPHA}))
    return target


class RegressorOptunaStudyTest(QaTestCase):
    # ------------------------------------------------------------------ direction guards

    def test_create_refuses_both_a_direction_and_directions(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            with self.assertRaisesRegex(ValueError, "Cannot specify both"):
                model.optuna_create_study(
                    PAIR,
                    "hp",
                    direction=optuna.study.StudyDirection.MAXIMIZE,
                    directions=LABEL_DIRECTIONS,
                )

    def test_create_needs_either_one_direction_or_at_least_two(self):
        single = optuna.study.StudyDirection.MAXIMIZE
        for kwargs in ({}, {"directions": None}, {"directions": []}, {"directions": [single]}):
            with (
                self.subTest(kwargs=sorted(kwargs), value=kwargs.get("directions")),
                temporary_directory() as tmp,
            ):
                model = regressor(tmp)
                with self.assertRaisesRegex(ValueError, "at least 2 objectives"):
                    model.optuna_create_study(PAIR, "hp", **kwargs)

    def test_two_directions_yield_a_two_objective_study(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            multi = model.optuna_create_study(
                PAIR,
                "label",
                directions=[optuna.study.StudyDirection.MAXIMIZE] * 2,
            )
            self.assertEqual(len(multi.directions), 2)

    def test_a_bare_direction_in_place_of_a_list_is_a_type_error_not_a_creation(self):
        # optuna_create_study probes len(directions) unguarded once `direction` is None,
        # so a non-list slips past both of its own checks and fails at the length probe.
        with temporary_directory() as tmp:
            model = regressor(tmp)
            with self.assertRaises(TypeError):
                model.optuna_create_study(
                    PAIR, "hp", directions=optuna.study.StudyDirection.MAXIMIZE
                )

    def test_optimize_repeats_the_both_specified_guard(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            with self.assertRaisesRegex(ValueError, "Cannot specify both"):
                model.optuna_optimize(
                    PAIR,
                    "hp",
                    lambda trial: 1.0,
                    direction=optuna.study.StudyDirection.MAXIMIZE,
                    directions=LABEL_DIRECTIONS,
                )

    def test_optimize_refuses_a_one_element_list_before_creating_any_study(self):
        # optuna_optimize's guard is `isinstance(directions, list) and len(directions) < 2`,
        # so it refuses a one-element list itself, without reaching optuna_create_study.
        with temporary_directory() as tmp:
            model = regressor(tmp, n_trials=1, n_jobs=1, timeout=60)
            model.optuna_create_study = mock.Mock(return_value=None)
            with self.assertRaisesRegex(ValueError, "at least 2 objectives"):
                model.optuna_optimize(
                    PAIR,
                    "hp",
                    lambda trial: 1.0,
                    directions=[optuna.study.StudyDirection.MAXIMIZE],
                )
            model.optuna_create_study.assert_not_called()

    def test_optimize_hands_a_missing_directions_to_optuna_create_study(self):
        # The asymmetry: `directions=None` is not a list, so optuna_optimize's own guard
        # passes it down, where optuna_create_study's `directions is None` branch refuses it.
        for directions in (None, ()):
            with self.subTest(directions=directions), temporary_directory() as tmp:
                model = regressor(tmp, n_trials=1, n_jobs=1, timeout=60)
                with self.assertRaisesRegex(ValueError, "at least 2 objectives"):
                    model.optuna_optimize(PAIR, "hp", lambda trial: 1.0, directions=directions)

    # ------------------------------------------------------- reset semantics: non-live

    def test_a_non_live_study_is_deleted_and_recreated_on_every_creation(self):
        # `continuous` is forced true while not live, so even a populated study is dropped.
        for continuous in (True, False):
            with self.subTest(continuous=continuous), temporary_directory() as tmp:
                model = regressor(tmp, live=False, continuous=continuous)
                shared = optuna.storages.InMemoryStorage()
                model.optuna_create_storage = lambda pair, backend=shared: backend
                first = single_study(model)
                seed_trial(model, first)
                self.assertEqual(trial_numbers(first), [0])

                second = single_study(model)
                self.assertIsNotNone(second)
                self.assertNotEqual(second._study_id, first._study_id)
                self.assertEqual(trial_numbers(second), [])

    def test_a_live_study_with_continuous_configured_true_is_always_recreated(self):
        for namespace in ("hp", "label"):
            with self.subTest(namespace=namespace), temporary_directory() as tmp:
                model = regressor(tmp, live=True, continuous=True)
                study = single_study(model) if namespace == "hp" else multi_study(model)
                seed_trial(model, study)
                again = single_study(model) if namespace == "hp" else multi_study(model)
                self.assertNotEqual(again._study_id, study._study_id)
                self.assertEqual(trial_numbers(again), [])

    def test_a_live_label_study_is_recreated_even_when_the_policy_forbids_reset(self):
        with temporary_directory() as tmp:
            model = regressor(
                tmp, live=True, continuous=True, reset_label_study_on_schema_mismatch=False
            )
            first = multi_study(model)
            seed_trial(model, first)
            second = multi_study(model)
            self.assertNotEqual(second._study_id, first._study_id)
            self.assertEqual(trial_numbers(second), [])

    # --------------------------------------------------------- reset semantics: live hp

    def test_a_live_hp_study_survives_a_matching_objective_identity(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            first = single_study(model)
            self.assertEqual(first.user_attrs, {"objective_identity": HP_IDENTITY})
            seed_trial(model, first)
            second = single_study(model)
            self.assertEqual(second._study_id, first._study_id)
            self.assertEqual(trial_numbers(second), [0])

    def test_a_live_hp_study_resets_when_the_objective_identity_moves(self):
        # The hp marker's reset_on_mismatch is unconditional, so the preserve policy —
        # which only governs the label namespace — cannot keep a stale hp study.
        for reset_policy in (True, False):
            with (
                self.subTest(reset_label_study_on_schema_mismatch=reset_policy),
                temporary_directory() as tmp,
            ):
                model = regressor(
                    tmp,
                    live=True,
                    continuous=False,
                    reset_label_study_on_schema_mismatch=reset_policy,
                )
                single_study(model)
                legacy = seed_trial(model, single_study(model), legacy_identity=True)
                self.assertEqual(legacy.user_attrs["objective_identity"], "legacy-identity-v0")

                third = single_study(model)
                self.assertNotEqual(third._study_id, legacy._study_id)
                self.assertEqual(trial_numbers(third), [])
                self.assertEqual(third.user_attrs, {"objective_identity": HP_IDENTITY})

    # ------------------------------------------------------- reset semantics: live label

    def test_a_live_label_study_survives_a_matching_selection_schema(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            first = multi_study(model)
            self.assertEqual(
                first.user_attrs["selection_metadata"], model._optuna_label_selection_metadata()
            )
            seed_trial(model, first)
            second = multi_study(model)
            self.assertEqual(second._study_id, first._study_id)
            self.assertEqual(trial_numbers(second), [0])

    def test_a_live_label_study_resets_when_the_selection_schema_version_moves(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            first = multi_study(model)
            drifted = dict(model._optuna_label_selection_metadata())
            drifted["schema_version"] = drifted["schema_version"] - 1
            first.set_user_attr("selection_metadata", drifted)
            seed_trial(model, first)

            with self.assertLogs(LOGGER, "WARNING") as captured:
                second = multi_study(model)
            assert_logged(self, captured.output, "resetting study")
            self.assertNotEqual(second._study_id, first._study_id)
            self.assertEqual(trial_numbers(second), [])
            self.assertEqual(
                second.user_attrs["selection_metadata"], model._optuna_label_selection_metadata()
            )

    def test_a_live_label_study_is_preserved_when_the_policy_forbids_reset(self):
        with temporary_directory() as tmp:
            model = regressor(
                tmp, live=True, continuous=False, reset_label_study_on_schema_mismatch=False
            )
            first = multi_study(model)
            drifted = dict(model._optuna_label_selection_metadata())
            drifted["schema_version"] = drifted["schema_version"] - 1
            first.set_user_attr("selection_metadata", drifted)
            seed_trial(model, first)

            with self.assertLogs(LOGGER, "WARNING") as captured:
                second = multi_study(model)
            assert_logged(self, captured.output, "preserving study")
            self.assertEqual(second._study_id, first._study_id)
            self.assertEqual(trial_numbers(second), [0])
            # The preserved study must keep the marker it was created with: rewriting it
            # here is what would make the next run see a compatible schema over trials
            # selected by the old one.
            self.assertEqual(second.user_attrs["selection_metadata"], drifted)

    def test_a_live_label_study_keeps_its_trials_when_only_the_method_config_drifted(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            first = multi_study(model)
            seed_trial(model, first)
            stored = first.user_attrs["selection_metadata"]

            model.ft_params = {"label_method": "topsis"}
            current = model._optuna_label_selection_metadata()
            self.assertNotEqual(stored["method_config"], current["method_config"])
            self.assertEqual(stored["schema_version"], current["schema_version"])

            with self.assertLogs(LOGGER, "WARNING") as captured:
                second = multi_study(model)
            assert_logged(self, captured.output, "selection_metadata change detected")
            self.assertEqual(second._study_id, first._study_id)
            self.assertEqual(trial_numbers(second), [0])
            self.assertEqual(second.user_attrs["selection_metadata"], current)

    def test_the_hp_marker_always_resets_while_the_label_marker_follows_the_policy(self):
        for reset_policy in (True, False):
            with (
                self.subTest(reset_label_study_on_schema_mismatch=reset_policy),
                temporary_directory() as tmp,
            ):
                model = regressor(
                    tmp,
                    live=True,
                    continuous=False,
                    reset_label_study_on_schema_mismatch=reset_policy,
                )
                hp = model._optuna_study_marker("hp")
                self.assertEqual(hp.user_attr_key, "objective_identity")
                self.assertTrue(hp.reset_on_mismatch)
                self.assertEqual(hp.build_marker(), HP_IDENTITY)
                self.assertTrue(hp.is_compatible(HP_IDENTITY))
                self.assertFalse(hp.is_compatible("legacy-identity-v0"))

                label = model._optuna_study_marker("label")
                self.assertEqual(label.user_attr_key, "selection_metadata")
                self.assertEqual(label.reset_on_mismatch, reset_policy)
                self.assertEqual(label.build_marker(), model._optuna_label_selection_metadata())

    def test_an_unknown_namespace_has_no_study_marker(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            self.assertIsNone(model._optuna_study_marker("candle"))

    # ------------------------------------------------------------- storage selection

    def test_a_live_study_is_backed_by_a_journal_under_the_user_data_dir(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            study = single_study(model)
            self.assertIsInstance(study._storage, optuna.storages.JournalStorage)
            journal = model.full_path / "optuna-BTC.log"
            self.assertTrue(journal.is_file())
            self.assertEqual(journal.parent, model.full_path)
            self.assertEqual(model.config["user_data_dir"], tmp)
            self.assertEqual(
                [str(p.relative_to(tmp)) for p in tmp.rglob("*.log")],
                [str(journal.relative_to(tmp))],
            )

    def test_a_live_sqlite_study_is_backed_by_a_database_under_the_user_data_dir(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False, storage="sqlite")
            study = single_study(model)
            database = model.full_path / "optuna-BTC.sqlite"
            self.assertTrue(database.is_file())
            self.assertEqual(database.parent, model.full_path)
            self.assertEqual(
                [str(path.relative_to(tmp)) for path in tmp.rglob("*.sqlite")],
                [str(database.relative_to(tmp))],
            )
            self.assertEqual(list(tmp.rglob("*.log")), [])
            # A fresh storage over the same file sees the study, which is what makes the
            # sqlite backend persistent where the in-memory one is not.
            reopened = optuna.load_study(
                study_name=study.study_name, storage=model.optuna_create_storage(PAIR)
            )
            self.assertEqual(reopened.user_attrs, {"objective_identity": HP_IDENTITY})

    def test_a_non_live_study_is_backed_by_memory_and_leaves_no_file_behind(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=False)
            before = tree(tmp)
            study = single_study(model)
            self.assertIsInstance(study._storage, optuna.storages.InMemoryStorage)
            self.assertEqual(study.user_attrs, {"objective_identity": HP_IDENTITY})
            self.assertEqual(tree(tmp), before)
            self.assertEqual(list(tmp.rglob("*.log")), [])
            self.assertEqual(list(tmp.rglob("*.sqlite")), [])

    def test_an_unknown_storage_backend_yields_no_study(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False, storage="carrier-pigeon")
            with self.assertLogs(LOGGER, "ERROR"):
                self.assertIsNone(single_study(model))

    def test_an_unknown_sampler_is_refused_before_the_study_is_built(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, sampler="simulated-annealing")
            with self.assertRaises(ValueError) as raised:
                single_study(model)
            self.assertEqual(
                str(raised.exception),
                enum_error_message(
                    "optuna hp sampler",
                    "simulated-annealing",
                    tuple(REGRESSOR._OPTUNA_HPO_SAMPLERS_SET),
                ),
            )

    def test_a_label_sampler_outside_the_label_set_is_refused(self):
        # The two namespaces admit different samplers: nsgaii is valid for label only,
        # so the refusal has to be proven with a value outside both sets.
        with temporary_directory() as tmp:
            model = regressor(tmp, label_sampler="simulated-annealing")
            with self.assertRaises(ValueError) as raised:
                multi_study(model)
            self.assertEqual(
                str(raised.exception),
                enum_error_message(
                    "optuna label sampler",
                    "simulated-annealing",
                    tuple(REGRESSOR._OPTUNA_LABEL_SAMPLERS_SET),
                ),
            )

    def test_the_two_namespaces_admit_different_sampler_sets(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            self.assertIn("nsgaii", REGRESSOR._OPTUNA_LABEL_SAMPLERS_SET)
            self.assertNotIn("nsgaii", REGRESSOR._OPTUNA_HPO_SAMPLERS_SET)
            self.assertEqual(
                model.optuna_samplers_by_namespace("label"),
                (REGRESSOR._OPTUNA_LABEL_SAMPLERS_SET, "tpe"),
            )
            with self.assertRaises(ValueError):
                model.optuna_samplers_by_namespace("candle")

    # --------------------------------------------------------- creation failure modes

    def test_a_storage_failure_yields_no_study(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            model.optuna_create_storage = mock.Mock(side_effect=OSError("read-only mount"))
            with self.assertLogs(LOGGER, "ERROR"):
                self.assertIsNone(single_study(model))

    def test_an_uninspectable_study_yields_no_study(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            single_study(model)
            with (
                mock.patch.object(
                    REGRESSOR, "optuna_load_study", side_effect=OSError("truncated journal")
                ),
                self.assertLogs(LOGGER, "ERROR"),
            ):
                self.assertIsNone(single_study(model))

    def test_a_failed_reset_aborts_study_creation_rather_than_loading_a_stale_study(self):
        # optuna_delete_study returning False is the only signal the reset failed, and
        # continuing would load the very trials the identity check just rejected.
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            single_study(model)
            legacy = seed_trial(model, single_study(model), legacy_identity=True)

            with mock.patch.object(REGRESSOR, "optuna_delete_study", return_value=False):
                self.assertIsNone(single_study(model))

            # The rejected study is still on disk, untouched by the aborted creation.
            survivor = optuna.load_study(
                study_name=legacy.study_name, storage=model.optuna_create_storage(PAIR)
            )
            self.assertEqual(survivor.user_attrs["objective_identity"], "legacy-identity-v0")
            self.assertEqual(trial_numbers(survivor), [0])

    def test_a_sampler_failure_yields_no_study(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            model.optuna_create_sampler = mock.Mock(side_effect=RuntimeError("sampler exploded"))
            with self.assertLogs(LOGGER, "ERROR"):
                self.assertIsNone(single_study(model))

    # ---------------------------------------------------------------- validate params

    def test_validation_needs_a_study(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            self.assertFalse(model.optuna_validate_params(PAIR, "hp", None))

    def test_a_single_objective_study_accepts_exactly_one_finite_stored_value(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            study = single_study(model)
            model.set_optuna_value(PAIR, "hp", 0.75)
            self.assertTrue(model.optuna_validate_params(PAIR, "hp", study))

    def test_a_single_objective_study_rejects_a_non_finite_or_non_numeric_stored_value(self):
        for value in (np.nan, np.inf, -np.inf, "0.75", None):
            with self.subTest(value=repr(value)), temporary_directory() as tmp:
                model = regressor(tmp)
                study = single_study(model)
                model.set_optuna_value(PAIR, "hp", value)
                self.assertFalse(model.optuna_validate_params(PAIR, "hp", study))

    def test_a_multi_objective_study_needs_one_finite_value_per_objective(self):
        arity = REGRESSOR._OPTUNA_LABEL_N_OBJECTIVES
        for values, expected in (
            ([0.5] * arity, True),
            ([0.5] * (arity - 1), False),
            ([0.5] * (arity + 1), False),
            ([np.nan, *([0.5] * (arity - 1))], False),
            ([np.inf, *([0.5] * (arity - 1))], False),
            ("not-a-list", False),
        ):
            with self.subTest(values=values, expected=expected), temporary_directory() as tmp:
                model = regressor(tmp)
                study = multi_study(model)
                model.set_optuna_values(PAIR, "label", values)
                self.assertEqual(model.optuna_validate_params(PAIR, "label", study), expected)

    # ----------------------------------------------------------------- enqueue warm start

    def test_the_previous_best_params_are_pinned_into_the_next_trial(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            study = single_study(model)
            model.set_optuna_value(PAIR, "hp", 0.5)
            model.set_optuna_params(PAIR, "hp", {"alpha": 0.25})

            model.optuna_enqueue_previous_best_params(PAIR, "hp", study)
            self.assertEqual(waiting_params(study), [{"alpha": 0.25}])

            study.optimize(lambda trial: trial.suggest_float("alpha", 0.0, 1.0), n_trials=1)
            self.assertEqual(study.trials[-1].params, {"alpha": 0.25})
            self.assertEqual(study.trials[-1].value, 0.25)

    def test_label_params_are_enqueued_in_the_label_shape(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            study = multi_study(model)
            model.set_optuna_values(
                PAIR, "label", [float(i) for i in range(REGRESSOR._OPTUNA_LABEL_N_OBJECTIVES)]
            )
            model.set_optuna_params(PAIR, "label", dict(LABEL_PARAMS))

            model.optuna_enqueue_previous_best_params(PAIR, "label", study)
            self.assertEqual(waiting_params(study), [dict(LABEL_PARAMS)])

    def test_nothing_is_enqueued_without_a_study(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            model.set_optuna_value(PAIR, "hp", 0.5)
            model.set_optuna_params(PAIR, "hp", {"alpha": 0.25})
            self.assertIsNone(model.optuna_enqueue_previous_best_params(PAIR, "hp", None))

    def test_nothing_is_enqueued_when_no_previous_params_are_stored(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            study = single_study(model)
            model.set_optuna_value(PAIR, "hp", 0.5)
            model.set_optuna_params(PAIR, "hp", {})
            model.optuna_enqueue_previous_best_params(PAIR, "hp", study)
            self.assertEqual(waiting_params(study), [])

    def test_nothing_is_enqueued_while_the_stored_best_value_is_invalid(self):
        for value in (np.nan, np.inf, None):
            with self.subTest(value=repr(value)), temporary_directory() as tmp:
                model = regressor(tmp)
                study = single_study(model)
                model.set_optuna_value(PAIR, "hp", value)
                model.set_optuna_params(PAIR, "hp", {"alpha": 0.25})
                model.optuna_enqueue_previous_best_params(PAIR, "hp", study)
                self.assertEqual(waiting_params(study), [])

    def test_an_unusable_param_payload_is_dropped_with_a_warning(self):
        # The guard is `if not best_params`, so a truthy non-mapping reaches
        # enqueue_trial, which refuses it; the warm start must not abort the run for it.
        with temporary_directory() as tmp:
            model = regressor(tmp)
            study = single_study(model)
            model.set_optuna_value(PAIR, "hp", 0.5)
            model.set_optuna_params(PAIR, "hp", "alpha=0.25")

            with self.assertLogs(LOGGER, "WARNING") as captured:
                model.optuna_enqueue_previous_best_params(PAIR, "hp", study)
            assert_logged(self, captured.output, "failed to enqueue previous best params")
            self.assertEqual(waiting_params(study), [])

    # --------------------------------------------------------------------- has best trial

    def test_no_study_has_no_best_trial(self):
        self.assertFalse(REGRESSOR.optuna_study_has_best_trial(None))
        self.assertFalse(REGRESSOR.optuna_study_has_best_trials(None))

    def test_an_empty_study_has_no_best_trial(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            self.assertFalse(REGRESSOR.optuna_study_has_best_trial(single_study(model)))

    def test_a_study_whose_trials_never_completed_has_no_best_trial(self):
        for state in (TrialState.FAIL, TrialState.PRUNED, TrialState.WAITING, TrialState.RUNNING):
            with self.subTest(state=state.name), temporary_directory() as tmp:
                model = regressor(tmp)
                study = single_study(model)
                study.add_trial(frozen(0, state=state))
                self.assertEqual(len(study.trials), 1)
                self.assertFalse(REGRESSOR.optuna_study_has_best_trial(study))

    def test_a_completed_trial_makes_a_best_trial_available(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            study = single_study(model)
            study.add_trial(frozen(0, state=TrialState.FAIL))
            self.assertFalse(REGRESSOR.optuna_study_has_best_trial(study))

            study.add_trial(
                frozen(1, value=0.75, params={"alpha": 0.5}, distributions={"alpha": ALPHA})
            )
            self.assertTrue(REGRESSOR.optuna_study_has_best_trial(study))
            self.assertEqual(study.best_value, 0.75)

    def test_best_trials_is_the_multi_objective_gate_and_never_filters_on_state(self):
        # optuna's best_trials returns an empty list rather than raising, so this guard
        # passes even for an empty study; the real emptiness filter is the best_trials
        # comprehension in _get_multi_objective_study_best_trial.
        with temporary_directory() as tmp:
            model = regressor(tmp)
            study = multi_study(model)
            self.assertEqual(len(study.best_trials), 0)
            self.assertTrue(REGRESSOR.optuna_study_has_best_trials(study))

            study.add_trial(frozen(0, state=TrialState.FAIL))
            self.assertTrue(REGRESSOR.optuna_study_has_best_trials(study))

            study.add_trial(
                frozen(
                    1,
                    values=[1.0] * REGRESSOR._OPTUNA_LABEL_N_OBJECTIVES,
                    params={"label_period_candles": 3},
                    distributions={"label_period_candles": LABEL_CANDLES},
                )
            )
            self.assertEqual(len(study.best_trials), 1)
            self.assertTrue(REGRESSOR.optuna_study_has_best_trials(study))

    def test_has_best_trial_refuses_a_multi_objective_study(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            with self.assertRaisesRegex(RuntimeError, "multi-objective"):
                REGRESSOR.optuna_study_has_best_trial(multi_study(model))

    # -------------------------------------------------------------- save / load best params

    def test_hp_best_params_round_trip_under_the_objective_identity(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            model.set_optuna_params(PAIR, "hp", {"alpha": 0.25})
            model.optuna_save_best_params(PAIR, "hp")

            path = best_params_path(model, "hp")
            self.assertEqual(path.parent, model.full_path)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"objective_identity": HP_IDENTITY, "params": {"alpha": 0.25}},
            )
            self.assertEqual(model.optuna_load_best_params(PAIR, "hp"), {"alpha": 0.25})

    def test_hp_best_params_are_ignored_when_the_objective_identity_moves(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            model.set_optuna_params(PAIR, "hp", {"alpha": 0.25})
            model.optuna_save_best_params(PAIR, "hp")
            path = best_params_path(model, "hp")

            rewrite_payload(path, lambda payload: payload.update(objective_identity="legacy-v0"))
            with self.assertLogs(LOGGER, "WARNING") as captured:
                self.assertIsNone(model.optuna_load_best_params(PAIR, "hp"))
            assert_logged(self, captured.output, "objective identity does not match")

            rewrite_payload(path, lambda payload: payload.update(objective_identity=HP_IDENTITY))
            self.assertEqual(model.optuna_load_best_params(PAIR, "hp"), {"alpha": 0.25})

    def test_hp_best_params_are_ignored_when_the_params_payload_is_not_a_mapping(self):
        for payload in (0.25, "alpha", [0.25], None):
            with self.subTest(params=payload), temporary_directory() as tmp:
                model = regressor(tmp, live=True, continuous=False)
                model.set_optuna_params(PAIR, "hp", {"alpha": 0.25})
                model.optuna_save_best_params(PAIR, "hp")
                path = best_params_path(model, "hp")
                rewrite_payload(path, lambda saved, value=payload: saved.update(params=value))
                self.assertIsNone(model.optuna_load_best_params(PAIR, "hp"))

    def test_hp_best_params_are_ignored_when_the_payload_is_not_a_mapping(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            model.set_optuna_params(PAIR, "hp", {"alpha": 0.25})
            model.optuna_save_best_params(PAIR, "hp")
            path = best_params_path(model, "hp")
            path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            self.assertIsNone(model.optuna_load_best_params(PAIR, "hp"))

    def test_label_best_params_round_trip_under_the_selection_schema(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            model.set_optuna_params(PAIR, "label", dict(LABEL_PARAMS))
            model.optuna_save_best_params(PAIR, "label")

            path = best_params_path(model, "label")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(payload), ["params", "schema_version", "selection_metadata"])
            self.assertEqual(payload["params"], dict(LABEL_PARAMS))
            self.assertEqual(
                payload["selection_metadata"], model._optuna_label_selection_metadata()
            )
            self.assertEqual(model.optuna_load_best_params(PAIR, "label"), dict(LABEL_PARAMS))

    def test_label_best_params_are_ignored_when_the_wire_schema_version_moves(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            model.set_optuna_params(PAIR, "label", dict(LABEL_PARAMS))
            model.optuna_save_best_params(PAIR, "label")
            path = best_params_path(model, "label")

            def bump(saved: dict) -> None:
                saved["schema_version"] += 1

            rewrite_payload(path, bump)
            with self.assertLogs(LOGGER, "WARNING") as captured:
                self.assertIsNone(model.optuna_load_best_params(PAIR, "label"))
            assert_logged(self, captured.output, "incompatible schema_version=3")

    def test_label_best_params_are_ignored_when_the_selection_schema_version_moves(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            model.set_optuna_params(PAIR, "label", dict(LABEL_PARAMS))
            model.optuna_save_best_params(PAIR, "label")
            path = best_params_path(model, "label")

            def roll_back(saved: dict) -> None:
                saved["selection_metadata"]["schema_version"] -= 1

            rewrite_payload(path, roll_back)
            with self.assertLogs(LOGGER, "WARNING") as captured:
                self.assertIsNone(model.optuna_load_best_params(PAIR, "label"))
            assert_logged(self, captured.output, "incompatible selection_metadata.schema_version=2")

    def test_label_best_params_are_ignored_when_the_selection_metadata_drifts(self):
        # The wire schema and the selection schema version both still match; only the
        # whole-metadata comparison catches this, which is what the regressor wrapper adds
        # over the strategy-side load that passes no expected metadata.
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            model.set_optuna_params(PAIR, "label", dict(LABEL_PARAMS))
            model.optuna_save_best_params(PAIR, "label")
            path = best_params_path(model, "label")

            def swap_metric(saved: dict) -> None:
                saved["selection_metadata"]["method_config"]["distance_metric"] = "manhattan"

            rewrite_payload(path, swap_metric)
            with self.assertLogs(LOGGER, "WARNING") as captured:
                self.assertIsNone(model.optuna_load_best_params(PAIR, "label"))
            assert_logged(self, captured.output, "selection_metadata drift")

    def test_label_best_params_are_ignored_when_a_required_param_is_out_of_range(self):
        for params in (
            {"label_period_candles": 0, "label_natr_multiplier": 1.0},
            {"label_period_candles": 2, "label_natr_multiplier": 0.0},
            {"label_period_candles": 2},
            {"label_period_candles": 2, "label_natr_multiplier": 1.0, "label_horizon_candles": 0},
        ):
            with self.subTest(params=params), temporary_directory() as tmp:
                model = regressor(tmp, live=True, continuous=False)
                model.set_optuna_params(PAIR, "label", dict(LABEL_PARAMS))
                model.optuna_save_best_params(PAIR, "label")
                path = best_params_path(model, "label")
                rewrite_payload(path, lambda saved, value=params: saved.update(params=value))
                self.assertIsNone(model.optuna_load_best_params(PAIR, "label"))

    def test_absent_best_params_load_as_none_for_both_namespaces(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            self.assertIsNone(model.optuna_load_best_params(PAIR, "hp"))
            self.assertIsNone(model.optuna_load_best_params(PAIR, "label"))

    # ------------------------------------------------------------------- delete and load

    def test_loading_a_study_that_was_never_created_yields_none(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            storage = model.optuna_create_storage(PAIR)
            self.assertIsNone(REGRESSOR.optuna_load_study(study_name(model, PAIR, "hp"), storage))

    def test_a_present_study_can_be_loaded_deleted_and_gone(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            name = study_name(model, PAIR, "hp")
            storage = model.optuna_create_storage(PAIR)
            single_study(model)

            loaded = REGRESSOR.optuna_load_study(name, model.optuna_create_storage(PAIR))
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.user_attrs, {"objective_identity": HP_IDENTITY})

            self.assertTrue(REGRESSOR.optuna_delete_study(PAIR, "hp", name, storage))
            self.assertIsNone(REGRESSOR.optuna_load_study(name, model.optuna_create_storage(PAIR)))

    def test_deleting_an_absent_study_is_a_benign_no_op(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            self.assertTrue(
                REGRESSOR.optuna_delete_study(
                    PAIR, "hp", study_name(model, PAIR, "hp"), model.optuna_create_storage(PAIR)
                )
            )

    def test_a_failing_deletion_is_reported_and_not_swallowed(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False)
            storage = mock.Mock()
            storage.delete_study.side_effect = OSError("read-only mount")
            with self.assertLogs(LOGGER, "WARNING") as captured:
                self.assertFalse(
                    REGRESSOR.optuna_delete_study(
                        PAIR, "hp", study_name(model, PAIR, "hp"), storage
                    )
                )
            assert_logged(self, captured.output, "deletion failed")

    # ------------------------------------------------------------------------ optimize

    def test_optimize_records_the_best_value_and_params_of_a_single_objective_study(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False, n_trials=3, n_jobs=1, timeout=60)
            study = model.optuna_optimize(
                PAIR,
                "hp",
                lambda trial: trial.suggest_float("alpha", 0.0, 1.0),
                direction=optuna.study.StudyDirection.MAXIMIZE,
            )
            self.assertEqual(len(study.trials), 3)
            self.assertEqual(model.get_optuna_params(PAIR, "hp"), study.best_params)
            self.assertAlmostEqual(model.get_optuna_value(PAIR, "hp"), study.best_value)

    def test_optimize_seeds_the_study_only_when_warm_start_is_configured(self):
        # Warm start replays the previous cutoff's best params as the next trial. Both
        # branches share one study, so the enqueue is visible as a trial whose params are
        # exactly the stored best and not a fresh sampler draw.
        for warm_start in (True, False):
            with self.subTest(warm_start=warm_start), temporary_directory() as tmp:
                model = regressor(
                    tmp,
                    live=True,
                    continuous=False,
                    warm_start=warm_start,
                    n_trials=1,
                    n_jobs=1,
                    timeout=60,
                )
                model.set_optuna_value(PAIR, "hp", 0.5)
                model.set_optuna_params(PAIR, "hp", {"alpha": 0.25})
                study = model.optuna_optimize(
                    PAIR,
                    "hp",
                    lambda trial: trial.suggest_float("alpha", 0.0, 1.0),
                    direction=optuna.study.StudyDirection.MAXIMIZE,
                )
                self.assertEqual(len(study.trials), 1)
                trial = study.trials[0]
                # optuna records an enqueued trial's pinned params under fixed_params and
                # omits them for a sampler-drawn one, so the attribute discriminates the
                # two branches without depending on what the sampler happened to draw.
                self.assertEqual("fixed_params" in trial.system_attrs, warm_start)
                if warm_start:
                    self.assertEqual(trial.params, {"alpha": 0.25})
                    self.assertEqual(trial.value, 0.25)

    def test_optimize_persists_best_params_only_while_live(self):
        for live in (True, False):
            with self.subTest(live=live), temporary_directory() as tmp:
                model = regressor(
                    tmp, live=live, continuous=False, n_trials=1, n_jobs=1, timeout=60
                )
                model.optuna_optimize(
                    PAIR,
                    "hp",
                    lambda trial: trial.suggest_float("alpha", 0.0, 1.0),
                    direction=optuna.study.StudyDirection.MAXIMIZE,
                )
                path = model.full_path / "optuna-hp-best-params-BTC_USDT.json"
                self.assertEqual(path.is_file(), live)

    def test_optimize_yields_no_study_when_no_trial_completed(self):
        with temporary_directory() as tmp:
            model = regressor(tmp, live=True, continuous=False, n_trials=1, n_jobs=1, timeout=60)

            def exploding(trial: object) -> float:
                raise RuntimeError("objective exploded")

            with self.assertLogs(LOGGER, "ERROR"):
                self.assertIsNone(
                    model.optuna_optimize(
                        PAIR,
                        "hp",
                        exploding,
                        direction=optuna.study.StudyDirection.MAXIMIZE,
                    )
                )

    def test_optimize_refuses_to_persist_a_preserved_incompatible_label_study(self):
        # A preserved study carries the old selection schema, so the trials it now holds
        # were selected by that schema; persisting them would let the next start load them.
        with temporary_directory() as tmp:
            model = regressor(
                tmp,
                live=True,
                continuous=False,
                reset_label_study_on_schema_mismatch=False,
                n_trials=1,
                n_jobs=1,
                timeout=60,
            )
            first = multi_study(model)
            first.set_user_attr("selection_metadata", {"schema_version": 2})

            with self.assertLogs(LOGGER, "WARNING") as captured:
                model.optuna_optimize(
                    PAIR,
                    "label",
                    lambda trial: [
                        float(index) for index in range(REGRESSOR._OPTUNA_LABEL_N_OBJECTIVES)
                    ],
                    directions=LABEL_DIRECTIONS,
                )
            assert_logged(self, captured.output, "best params not persisted")
            self.assertFalse(best_params_path(model, "label").exists())

    # -------------------------------------------------- selection metadata compatibility

    def test_the_current_selection_schema_is_compatible(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            self.assertTrue(
                REGRESSOR._optuna_label_selection_metadata_compatible(
                    model._optuna_label_selection_metadata()
                )
            )

    def test_a_numpy_integer_selection_schema_version_is_compatible(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            metadata = dict(model._optuna_label_selection_metadata())
            metadata["schema_version"] = np.int64(metadata["schema_version"])
            self.assertTrue(REGRESSOR._optuna_label_selection_metadata_compatible(metadata))

    def test_a_moved_selection_schema_version_is_not_compatible(self):
        with temporary_directory() as tmp:
            model = regressor(tmp)
            metadata = dict(model._optuna_label_selection_metadata())
            metadata["schema_version"] = metadata["schema_version"] - 1
            self.assertFalse(REGRESSOR._optuna_label_selection_metadata_compatible(metadata))

    def test_metadata_drift_below_the_schema_version_is_still_compatible(self):
        # Only schema_version decides compatibility; the stored payload is left to the
        # best-params loader, which compares the whole metadata dict.
        with temporary_directory() as tmp:
            model = regressor(tmp)
            metadata = dict(model._optuna_label_selection_metadata())
            metadata["method_config"] = {"category": "distance", "method": "topsis"}
            metadata["label_weights"] = [0.5, 0.5]
            self.assertTrue(REGRESSOR._optuna_label_selection_metadata_compatible(metadata))

    def test_a_missing_or_non_dict_marker_is_not_compatible(self):
        for marker in ({}, {"schema_version": None}, None, "compromise_programming", 3, []):
            with self.subTest(marker=marker):
                self.assertFalse(REGRESSOR._optuna_label_selection_metadata_compatible(marker))

    def test_a_boolean_is_not_an_integer_selection_schema_version(self):
        for version in (True, False):
            with self.subTest(schema_version=version):
                self.assertFalse(
                    REGRESSOR._optuna_label_selection_metadata_compatible(
                        {"schema_version": version}
                    )
                )


if __name__ == "__main__":
    unittest.main()
