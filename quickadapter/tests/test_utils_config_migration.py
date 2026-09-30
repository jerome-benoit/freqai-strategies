"""Deprecated-config migration: in-place rewrite, precedence and warn-once."""

import logging
import unittest

from qa_support import QaTestCase
from Utils import (
    _MISSING,
    CONFIG_DEPRECATIONS,
    _delete_path,
    _get_path,
    _set_path,
    as_config_section,
    as_dict,
    migrate_config,
)

LOGGER = logging.getLogger("test-config-migration")


class UtilsConfigMigrationTest(QaTestCase):
    def test_a_renamed_section_moves_the_value_and_drops_the_old_path(self):
        # extrema_smoothing -> label_smoothing, with a key that chains into no further rename.
        config = {"freqai": {"extrema_smoothing": {"factor": 0.5}}}

        with self.assertLogs(LOGGER, level="WARNING") as captured:
            migrate_config(config, LOGGER)

        self.assertNotIn("extrema_smoothing", config["freqai"])
        self.assertEqual(config["freqai"]["label_smoothing"], {"factor": 0.5})
        self.assertIn("'label_smoothing' instead", captured.output[0])

    def test_b_migration_rewrites_the_callers_dict_in_place(self):
        config = {"freqai": {"extrema_weighting": {"gamma": 0.5}}}
        section = config["freqai"]["extrema_weighting"]

        result = migrate_config(config, LOGGER)

        # The shared-config-base hazard is the mutation, not a returned copy: the caller still
        # holds the very object the value lived in, and finds it drained, so every other holder
        # of that dict is affected too.
        self.assertIsNone(result)
        self.assertIs(config["freqai"]["label_weighting"], section)
        self.assertEqual(section, {})
        self.assertEqual(config["freqai"]["label_pipeline"], {"gamma": 0.5})

    def test_c_a_renamed_key_within_a_section_takes_the_short_name_in_its_warning(self):
        config = {"freqai": {"label_smoothing": {"window": 4}}}

        with self.assertLogs(LOGGER, level="WARNING") as captured:
            migrate_config(config, LOGGER)

        self.assertEqual(config["freqai"]["label_smoothing"], {"window_candles": 4})
        self.assertIn("'window_candles' instead", captured.output[0])

    def test_d_a_key_move_follows_the_section_rename_that_precedes_it(self):
        config = {"freqai": {"label_weighting": {"standardization": True}}}

        with self.assertLogs(LOGGER, level="WARNING") as captured:
            migrate_config(config, LOGGER)

        self.assertEqual(config["freqai"]["label_pipeline"], {"standardization": True})
        self.assertEqual(config["freqai"]["label_weighting"], {})
        # The sections differ, so the warning names the whole new path rather than the bare key.
        self.assertIn("'freqai.label_pipeline.standardization' instead", captured.output[0])

    def test_e_a_key_already_at_its_new_path_is_left_alone(self):
        config = {"freqai": {"label_pipeline": {"gamma": 0.7}}}

        with self.assertNoLogs(LOGGER, level="WARNING"):
            migrate_config(config, LOGGER)

        self.assertEqual(config, {"freqai": {"label_pipeline": {"gamma": 0.7}}})

    def test_f_a_deleted_key_is_removed_and_its_value_dropped(self):
        config = {"exit_pricing": {"thresholds_calibration": {"low": 0.5}}}

        with self.assertLogs(LOGGER, level="WARNING") as captured:
            migrate_config(config, LOGGER)

        self.assertEqual(config, {"exit_pricing": {}})
        self.assertIn("is obsolete and ignored", captured.output[0])
        self.assertIn("armed volatility-scaled retracement", captured.output[0])

    def test_g_a_predicated_entry_warns_without_touching_the_key(self):
        config = {"freqai": {"feature_parameters": {"causal_mode": False}}}

        with self.assertLogs(LOGGER, level="WARNING") as captured:
            migrate_config(config, LOGGER)

        # The one entry that warns about a value rather than about a path: it has no new path,
        # so the key survives the migration carrying the very setting the message is about.
        self.assertEqual(config, {"freqai": {"feature_parameters": {"causal_mode": False}}})
        self.assertIn("causal_mode=false is deprecated", captured.output[0])
        self.assertIn("label lookahead leakage possible", captured.output[0])

    def test_h_the_new_value_wins_when_both_paths_are_present(self):
        config = {"freqai": {"extrema_smoothing": {"old": 1}, "label_smoothing": {"new": 2}}}

        with self.assertLogs(LOGGER, level="WARNING") as captured:
            migrate_config(config, LOGGER)

        self.assertNotIn("extrema_smoothing", config["freqai"])
        self.assertEqual(config["freqai"]["label_smoothing"], {"new": 2})
        self.assertIn("using 'label_smoothing'", captured.output[0])

    def test_i_a_renamed_chain_resolves_to_the_final_key_carrying_the_current_value(self):
        config = {
            "exit_pricing": {
                "trade_price_target": "candle_open",
                "trade_price_target_method": "candle_close",
            }
        }

        with self.assertLogs(LOGGER, level="WARNING") as captured:
            migrate_config(config, LOGGER)

        # Two entries share a path: the first pass discards the superseded name and keeps the
        # value already sitting at the new one, which the second pass then carries on to the
        # final name. The winner survives the whole chain.
        self.assertEqual(config, {"exit_pricing": {"trade_natr_method": "candle_close"}})
        self.assertEqual(len(captured.output), 2)

    def test_j_a_discarded_rename_creates_no_destination_section(self):
        config = {"freqai": {"extrema_smoothing": {"old": 1}, "label_smoothing": {"new": 2}}}

        with self.assertLogs(LOGGER, level="WARNING"):
            migrate_config(config, LOGGER)

        self.assertEqual(sorted(config["freqai"]), ["label_smoothing"])

    def test_k_the_deprecation_table_is_a_tuple_of_well_formed_unique_entries(self):
        self.assertIsInstance(CONFIG_DEPRECATIONS, tuple)
        self.assertEqual(
            len({entry[0] for entry in CONFIG_DEPRECATIONS}),
            len(CONFIG_DEPRECATIONS),
            "two entries share an old path, so the second can never fire",
        )
        for old_path, new_path, predicate, guidance in CONFIG_DEPRECATIONS:
            with self.subTest(old_path=old_path):
                self.assertIsInstance(old_path, str)
                self.assertIn(".", old_path)
                self.assertNotEqual(old_path, new_path)
                self.assertTrue(new_path is None or (isinstance(new_path, str) and "." in new_path))
                self.assertTrue(predicate is None or callable(predicate))
                self.assertTrue(guidance is None or isinstance(guidance, str))
                # A key with nowhere to go must say why; a plain rename has nothing to explain.
                self.assertEqual(new_path is None, guidance is not None)

    def test_l_section_renames_are_ordered_before_the_key_moves_that_feed_on_them(self):
        # An entry whose old path sits under an earlier entry's new path is unreachable unless
        # the earlier one has already run, which is what the ordering comment on the table
        # guarantees. Reordering the table would break every pair this finds.
        inherited = 0
        for index, (old_path, _, _, _) in enumerate(CONFIG_DEPRECATIONS):
            for earlier_index, (_, earlier_new, _, _) in enumerate(CONFIG_DEPRECATIONS[:index]):
                if earlier_new is None:
                    continue
                if old_path == earlier_new or old_path.startswith(f"{earlier_new}."):
                    inherited += 1
                    self.assertLess(
                        earlier_index,
                        index,
                        f"{old_path} can only be reached after {earlier_index} is applied",
                    )
        self.assertGreater(inherited, 0, "the table no longer encodes a rename chain")

    def test_m_a_deprecation_path_warns_once_per_process_across_config_objects(self):
        first = {"freqai": {"extrema_smoothing": {"factor": 0.5}}}
        second = {"freqai": {"extrema_smoothing": {"factor": 0.8}}}

        with self.assertLogs(LOGGER, level="WARNING") as captured:
            migrate_config(first, LOGGER)
            migrate_config(second, LOGGER)

        # Both objects are migrated; the notice is keyed on the path, so only the first caller
        # to reach it is told. That is the whole point of the warned-once registry.
        self.assertEqual(len(captured.output), 1)
        self.assertEqual(second["freqai"]["label_smoothing"], {"factor": 0.8})

    def test_n_two_deprecation_paths_warn_independently(self):
        config = {"freqai": {"extrema_smoothing": {"factor": 0.5}, "predictions_extrema": {"q": 9}}}

        with self.assertLogs(LOGGER, level="WARNING") as captured:
            migrate_config(config, LOGGER)

        self.assertEqual(len(captured.output), 2)
        self.assertTrue(any("extrema_smoothing" in line for line in captured.output))
        self.assertTrue(any("predictions_extrema" in line for line in captured.output))
        self.assertEqual(config["freqai"]["label_smoothing"], {"factor": 0.5})
        self.assertEqual(config["freqai"]["label_prediction"], {"q": 9})

    def test_o_get_path_walks_nested_mappings_and_signals_a_miss_with_the_sentinel(self):
        config = {"freqai": {"label_pipeline": {"gamma": 0.5}}, "flat": 0, "none": None}

        self.assertEqual(_get_path(config, "freqai.label_pipeline.gamma"), 0.5)
        self.assertEqual(_get_path(config, "freqai"), {"label_pipeline": {"gamma": 0.5}})
        self.assertIs(_get_path(config, "freqai.label_pipeline.absent"), _MISSING)
        self.assertIs(_get_path(config, "absent.branch.key"), _MISSING)
        # A scalar is not a branch: the walk stops instead of subscripting it.
        self.assertIs(_get_path(config, "flat.deeper"), _MISSING)
        # A stored None is a value, not an absence, and must not be confused with the sentinel.
        self.assertIsNone(_get_path(config, "none"))
        self.assertIsNot(_get_path(config, "none"), _MISSING)

    def test_p_set_path_creates_the_intermediate_levels(self):
        config: dict = {}
        self.assertIsNone(_set_path(config, "a.b.c", 5))
        self.assertEqual(config, {"a": {"b": {"c": 5}}})

        config = {"enabled": False}
        _set_path(config, "enabled", True)
        self.assertEqual(config, {"enabled": True})

        config = {"a": {"keep": 1}}
        _set_path(config, "a.b", 2)
        self.assertEqual(config, {"a": {"keep": 1, "b": 2}})

    def test_q_delete_path_tolerates_a_missing_path(self):
        config = {"a": {"b": 1}, "flat": 0}

        self.assertTrue(_delete_path(config, "a.b"))
        self.assertEqual(config, {"a": {}, "flat": 0})
        self.assertFalse(_delete_path(config, "a.b"))
        self.assertFalse(_delete_path(config, "absent.branch.key"))
        self.assertFalse(_delete_path(config, "flat.deeper"))
        self.assertEqual(config, {"a": {}, "flat": 0})

    def test_r_as_config_section_normalizes_a_non_mapping_with_a_warning(self):
        section = {"gamma": 0.5}
        with self.assertNoLogs(LOGGER, level="WARNING"):
            self.assertIs(as_config_section(section, "freqai.label_weighting", LOGGER), section)
            # None is an absent section, not a malformed one, so it normalizes silently.
            self.assertEqual(as_config_section(None, "freqai.label_weighting", LOGGER), {})

        with self.assertLogs(LOGGER, level="WARNING") as captured:
            result = as_config_section(0.5, "freqai.label_weighting", LOGGER)
        self.assertEqual(result, {})
        self.assertIn("must be a mapping, using defaults", captured.output[0])

    def test_s_as_dict_passes_a_mapping_through_by_identity_and_normalizes_the_rest(self):
        section = {"gamma": 0.5}
        self.assertIs(as_dict(section), section)
        for value in (None, 0.5, "gamma", [("gamma", 0.5)]):
            with self.subTest(value=value):
                self.assertEqual(as_dict(value), {})


if __name__ == "__main__":
    unittest.main()
