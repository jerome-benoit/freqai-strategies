"""Schedule, conversion and network-architecture algebra; requires the Freqtrade QA image."""

import math
import unittest

import torch as th
from qa_support import QaTestCase

from ReforceXY.user_data.freqaimodels.ReforceXY import (
    ReforceXY,
    SimpleLinearSchedule,
    compute_gradient_steps,
    deepmerge,
    get_activation_fn,
    get_net_arch,
    get_optimizer_class,
    get_schedule,
    get_schedule_type,
    hours_to_seconds,
    steps_to_days,
)

LINEAR, CONSTANT, UNKNOWN = ReforceXY._SCHEDULE_TYPES


class LinearScheduleTest(QaTestCase):
    def test_it_falls_from_its_initial_value_to_nothing(self):
        schedule = SimpleLinearSchedule(0.25)
        self.assertEqual(0.25, schedule(1.0))
        self.assertEqual(0.0, schedule(0.0))

    def test_a_string_initial_value_is_read_as_a_number(self):
        self.assertEqual(0.5, SimpleLinearSchedule("0.5")(1.0))

    def test_progress_beyond_one_is_clamped_so_no_negative_rate_is_ever_emitted(self):
        # SB3 asks for negative progress when a DQN overshoots the aligned budget; a
        # negative learning rate would silently invert the update.
        schedule = SimpleLinearSchedule(0.3)
        self.assertEqual(0.3, schedule(1.5))
        self.assertEqual(0.0, schedule(-0.5))


class ScheduleClassificationTest(QaTestCase):
    def test_a_number_is_a_constant_schedule_holding_that_number(self):
        kind, initial, final = get_schedule_type(0.4)
        self.assertEqual(CONSTANT, kind)
        self.assertEqual(0.4, initial)
        self.assertEqual(0.4, final)

    def test_a_linear_schedule_is_classified_by_its_two_endpoints(self):
        kind, initial, final = get_schedule_type(SimpleLinearSchedule(0.2))
        self.assertEqual(LINEAR, kind)
        self.assertEqual(0.2, initial)
        self.assertEqual(0.0, final)

    def test_an_unrecognised_schedule_is_reported_as_unknown_rather_than_guessed(self):
        kind, initial, final = get_schedule_type("not-a-schedule")
        self.assertEqual(UNKNOWN, kind)
        self.assertTrue(math.isnan(initial))
        self.assertTrue(math.isnan(final))

    def test_a_linear_request_builds_the_linear_schedule(self):
        schedule = get_schedule(LINEAR, 0.1)
        self.assertIsInstance(schedule, SimpleLinearSchedule)
        self.assertEqual(0.1, schedule(1.0))
        self.assertEqual(0.0, schedule(0.0))

    def test_every_other_request_builds_a_constant_schedule(self):
        for kind in (CONSTANT, UNKNOWN):
            with self.subTest(kind=kind):
                self.assertEqual(0.1, get_schedule(kind, 0.1)(1.0))
                self.assertEqual(0.1, get_schedule(kind, 0.1)(0.0))


class ConversionTest(QaTestCase):
    def test_hours_to_seconds_scales_by_3600(self):
        self.assertEqual(7200.0, hours_to_seconds(2.0))

    def test_steps_to_days_divides_the_total_minutes_by_a_day_and_rounds(self):
        self.assertEqual(60.0, steps_to_days(86_400, "1m"))
        self.assertEqual(0.0, steps_to_days(1, "1m"))

    def test_deepmerge_recurses_and_leaves_both_inputs_untouched(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        src = {"a": {"c": 9}}
        merged = deepmerge(base, src)
        self.assertEqual({"a": {"b": 1, "c": 9}, "d": 3}, merged)
        self.assertEqual({"a": {"b": 1, "c": 2}, "d": 3}, base)
        self.assertEqual({"a": {"c": 9}}, src)


class GradientStepTest(QaTestCase):
    def test_it_is_the_ceiling_of_the_frequency_quotient_capped_by_the_frequency(self):
        self.assertEqual(4, compute_gradient_steps(8, 2))
        self.assertEqual(1, compute_gradient_steps(3, 8))

    def test_a_non_positive_frequency_reports_minus_one_rather_than_a_step_count(self):
        self.assertEqual(-1, compute_gradient_steps(0, 2))

    def test_a_non_positive_subsample_reports_minus_one(self):
        self.assertEqual(-1, compute_gradient_steps(8, 0))

    def test_a_non_numeric_frequency_reports_minus_one(self):
        self.assertEqual(-1, compute_gradient_steps("often", 2))

    def test_a_sequence_frequency_reads_its_first_element(self):
        self.assertEqual(4, compute_gradient_steps((8, "s"), 2))


class ArchitectureTest(QaTestCase):
    def test_the_on_policy_family_returns_an_entry_per_network_and_shares_one_width(self):
        # Production gives pi and vf the SAME width at every size, so the invariant is that
        # they agree, not that they differ; a test named for a separation that does not
        # exist would pass on the wrong widths.
        for size in ReforceXY._NET_ARCH_SIZES:
            with self.subTest(size=size):
                arch = get_net_arch("MaskablePPO", size)
                self.assertEqual({"pi", "vf"}, set(arch))
                self.assertEqual(arch["pi"], arch["vf"])
                if size == "small":
                    self.assertEqual([128, 128], arch["pi"])

    def test_the_value_based_family_gets_a_shared_width_list(self):
        for model_type in ("DQN", "QR-DQN"):
            with self.subTest(model_type=model_type):
                self.assertEqual([128, 128], get_net_arch(model_type, "small"))

    def test_a_grown_size_is_wider_than_a_small_one(self):
        self.assertGreater(
            get_net_arch("DQN", "large")[0],
            get_net_arch("DQN", "small")[0],
        )

    def test_an_unknown_size_falls_back_to_small_for_every_family(self):
        for model_type in ("MaskablePPO", "DQN"):
            with self.subTest(model_type=model_type):
                self.assertEqual(
                    get_net_arch(model_type, "small"), get_net_arch(model_type, "huge")
                )

    def test_activation_and_optimizer_map_to_their_sb3_symbols(self):
        self.assertIs(th.nn.ReLU, get_activation_fn("relu"))
        self.assertIs(th.nn.Tanh, get_activation_fn("tanh"))
        self.assertIs(th.optim.Adam, get_optimizer_class("adam"))

    def test_an_unknown_activation_or_optimizer_falls_back_to_the_default(self):
        self.assertIs(th.nn.ReLU, get_activation_fn("no-such-activation"))
        self.assertIs(th.optim.Adam, get_optimizer_class("no-such-optimizer"))


class EvalFrequencyTest(QaTestCase):
    def _model(self, *, model_type="MaskablePPO", n_envs=1, n_eval_steps=16):
        model = ReforceXY.__new__(ReforceXY)
        model.n_envs = n_envs
        model.n_eval_steps = n_eval_steps
        model.model_type = model_type
        return model

    def test_a_non_positive_budget_still_evaluates_once(self):
        # A boundary worth pinning as such. Deleting the early return would NOT turn
        # this red — max_n_calls floors at 1 anyway — so it is not claimed to cover it.
        self.assertEqual(1, self._model().get_eval_freq(0))

    def test_eval_freq_is_capped_by_the_calls_the_budget_allows(self):
        # n_eval_steps above the budget is the case where the final min() bites; with
        # the value-based branch the uncapped result is strictly greater.
        model = self._model(model_type="DQN", n_eval_steps=100)
        self.assertEqual(64, model.get_eval_freq(64))
        self.assertEqual(8, model.get_eval_freq(8))

    def test_the_on_policy_family_prefers_a_rollout_that_fits_the_budget(self):
        model = self._model()
        # With no model_params the search walks _PPO_N_STEPS downwards, so a budget
        # larger than every candidate yields the largest one, never the budget itself.
        self.assertLess(model.get_eval_freq(16_000), 16_000)
        self.assertEqual(16, model.get_eval_freq(16))

    def test_more_environments_mean_fewer_callback_calls_for_the_same_budget(self):
        self.assertLess(
            self._model(n_envs=4).get_eval_freq(16_000),
            self._model(n_envs=1).get_eval_freq(16_000),
        )

    def test_the_value_based_family_uses_the_evaluation_step_budget(self):
        self.assertEqual(16, self._model(model_type="DQN", n_eval_steps=16).get_eval_freq(16_000))

    def test_hyperopt_reduces_the_interval(self):
        # Two explicit outcomes rather than the production formula restated: changing
        # _HYPEROPT_EVAL_FREQ_REDUCTION_FACTOR moves both of them.
        model = self._model(model_type="DQN")
        for n_eval_steps, plain, reduced in ((16, 16, 4), (64, 64, 16)):
            with self.subTest(n_eval_steps=n_eval_steps):
                model.n_eval_steps = n_eval_steps
                self.assertEqual(plain, model.get_eval_freq(16_000))
                self.assertEqual(reduced, model.get_eval_freq(16_000, hyperopt=True))

    def test_the_reduced_interval_never_falls_below_one(self):
        model = self._model(model_type="DQN", n_eval_steps=1)
        self.assertEqual(1, model.get_eval_freq(16_000, hyperopt=True))


if __name__ == "__main__":
    unittest.main()
