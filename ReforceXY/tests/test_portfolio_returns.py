"""Runtime regressions for liquidation-equity returns; requires the RL QA image."""

import math
import unittest

import numpy as np
import pandas as pd

from ReforceXY.user_data.freqaimodels.ReforceXY import Actions, MyRLEnv


class PortfolioReturnsTest(unittest.TestCase):
    def make_env(self, prices, *, compound=True, fee=0.0015):
        frame = pd.DataFrame({"open": prices})
        env = MyRLEnv(
            df=frame.copy(),
            prices=frame,
            df_raw=frame.copy(),
            window_size=1,
            reward_kwargs={"rr": 2.0, "profit_aim": 0.03},
            fee=fee,
            can_short=True,
            config={
                "stake_amount": "unlimited" if compound else 100.0,
                "freqai": {
                    "rl_config": {
                        "add_state_info": False,
                        "max_training_drawdown_pct": 0.99,
                        "model_reward_parameters": {},
                    }
                },
            },
            live=True,
        )
        self.addCleanup(env.close)
        env.reset()
        return env

    def test_entry_does_not_capture_pre_entry_move(self):
        env = self.make_env([100.0, 110.0, 110.0, 110.0, 110.0])
        _, _, _, _, entry = env.step(Actions.Long_enter.value)
        self.assertAlmostEqual(entry["most_recent_return"], -2 * math.log1p(env.fee), places=5)
        for action in (Actions.Neutral, Actions.Long_exit):
            _, _, _, _, info = env.step(action.value)
            self.assertEqual(info["most_recent_return"], 0.0)
            self.assertEqual(info["most_recent_profit"], 0.0)
        self.assertAlmostEqual(np.expm1(env.portfolio_log_returns.sum()), env._total_profit - 1)

    def test_transitions_match_equity_for_both_staking_modes_and_directions(self):
        prices = [100.0, 100.0, 110.0, 105.0, 115.0, 120.0, 118.0, 117.0, 117.0]
        for short in (False, True):
            for compound in (False, True):
                with self.subTest(short=short, compound=compound):
                    env = self.make_env(prices, compound=compound)
                    enter = Actions.Short_enter if short else Actions.Long_enter
                    exit_action = Actions.Short_exit if short else Actions.Long_exit
                    capital = previous_equity = 1.0
                    entry_price = None
                    for action in (
                        enter,
                        Actions.Neutral,
                        exit_action,
                        Actions.Neutral,
                        enter,
                        exit_action,
                    ):
                        _, _, _, _, info = env.step(action.value)
                        fill_price = prices[info["tick"] - 1]
                        mark_price = prices[info["tick"]]
                        fee_factor = (1 + env.fee) ** 2
                        if action == enter:
                            entry_price = fill_price
                        if action == exit_action:
                            price_ratio = fill_price / entry_price
                            exit_pnl = (
                                1 - price_ratio * fee_factor
                                if short
                                else price_ratio / fee_factor - 1
                            )
                            equity = capital * (1 + exit_pnl) if compound else capital + exit_pnl
                            capital = equity
                            entry_price = None
                        else:
                            pnl = 0.0
                            if entry_price is not None:
                                price_ratio = mark_price / entry_price
                                pnl = (
                                    1 - price_ratio * fee_factor
                                    if short
                                    else price_ratio / fee_factor - 1
                                )
                            equity = capital * (1 + pnl) if compound else capital + pnl
                        expected_log = math.log(equity / previous_equity)
                        self.assertAlmostEqual(
                            env.portfolio_log_returns[info["tick"]], expected_log
                        )
                        self.assertEqual(info["most_recent_return"], round(expected_log, 5))
                        self.assertEqual(
                            info["most_recent_profit"], round(equity / previous_equity - 1, 5)
                        )
                        previous_equity = equity
                    self.assertAlmostEqual(np.expm1(env.portfolio_log_returns.sum()), capital - 1)
                    self.assertAlmostEqual(env._total_profit, capital)
                    self.assertEqual(env.get_most_recent_return(), 0.0)
                    self.assertEqual(env.get_most_recent_profit(), 0.0)

    def test_terminal_liquidation_realizes_reward_and_equity_once(self):
        for short in (False, True):
            for compound in (False, True):
                for drawdown in (False, True):
                    with self.subTest(short=short, compound=compound, drawdown=drawdown):
                        terminal_price = 110.0 if short else 90.0
                        prices = [100.0, 100.0, 100.0, terminal_price]
                        if drawdown:
                            prices.append(terminal_price)
                        env = self.make_env(prices, compound=compound)
                        env.max_drawdown = 0.95 if drawdown else 0.01
                        env.step((Actions.Short_enter if short else Actions.Long_enter).value)
                        _, reward, terminated, _, info = env.step(Actions.Neutral.value)
                        fee_factor = (1 + env.fee) ** 2
                        expected_pnl = (
                            1 - terminal_price / 100 * fee_factor
                            if short
                            else terminal_price / 100 / fee_factor - 1
                        )
                        self.assertTrue(terminated)
                        self.assertEqual(env._position.name, "Neutral")
                        self.assertEqual(env._position_history[-1].name, "Neutral")
                        self.assertLess(reward, 0.0)
                        self.assertAlmostEqual(env._total_profit, 1 + expected_pnl)
                        self.assertAlmostEqual(
                            np.expm1(env.portfolio_log_returns.sum()), expected_pnl
                        )
                        self.assertAlmostEqual(info["exit_pnl"], expected_pnl)
                        self.assertTrue(info["terminal_liquidation"])
                        self.assertEqual(len(env.trade_history), 2)
                        self.assertEqual(
                            env.trade_history[-1]["type"],
                            "short_exit" if short else "long_exit",
                        )
                        self.assertEqual(env.trade_history[-1]["tick"], info["execution_tick"])
                        self.assertEqual(env.trade_history[-1]["price"], terminal_price)
                        self.assertAlmostEqual(env.trade_history[-1]["profit"], expected_pnl)
                        self.assertAlmostEqual(
                            reward,
                            sum(
                                info[key]
                                for key in (
                                    "reward_exit",
                                    "reward_hold",
                                    "reward_idle",
                                    "reward_invalid",
                                    "reward_shaping",
                                    "reward_entry_additive",
                                    "reward_exit_additive",
                                )
                            ),
                            places=4,
                        )
                        history = env.get_env_history()
                        terminal_row = history.loc[
                            history["execution_tick"] == info["execution_tick"]
                        ].iloc[0]
                        self.assertEqual(terminal_row["reward_exit"], info["reward_exit"])
                        self.assertTrue(terminal_row["terminal_liquidation"])

    def test_history_keeps_transitions_separate_from_trade_events(self):
        env = self.make_env([100.0, 100.0, 110.0, 110.0])
        _, _, _, _, entry_info = env.step(Actions.Long_enter.value)
        _, _, _, _, exit_info = env.step(Actions.Long_exit.value)
        history = env.get_env_history()
        self.assertEqual(len(history), 2)
        self.assertNotIn("type", history.columns)
        self.assertNotIn("profit", history.columns)
        entry_row = history.loc[history["execution_tick"] == 1].iloc[0]
        self.assertEqual(entry_row["reward_exit"], entry_info["reward_exit"])
        self.assertEqual(entry_row["reward_exit"], 0.0)
        self.assertEqual(entry_row["action"], Actions.Long_enter.value)
        exit_row = history.loc[history["execution_tick"] == 2].iloc[0]
        self.assertGreater(exit_row["reward_exit"], 0.0)
        self.assertEqual(exit_row["reward_exit"], exit_info["reward_exit"])
        self.assertEqual(
            [(event["tick"], event["type"]) for event in env.trade_history],
            [(1, "long_enter"), (2, "long_exit")],
        )

    def test_terminal_entry_preserves_two_events_and_plot_markers(self):
        env = self.make_env([100.0, 100.0, 90.0])
        _, _, terminated, truncated, info = env.step(Actions.Long_enter.value)
        expected_pnl = 90.0 / 100.0 / (1.0 + env.fee) ** 2 - 1.0

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["terminal_liquidation"])
        history = env.get_env_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]["execution_tick"], info["execution_tick"])
        self.assertEqual(history.iloc[0]["open"], 90.0)

        self.assertEqual(len(env.trade_history), 2)
        self.assertEqual(
            [(event["tick"], event["type"]) for event in env.trade_history],
            [(info["execution_tick"], "long_enter"), (info["execution_tick"], "long_exit")],
        )
        self.assertEqual(
            [event["price"] for event in env.trade_history],
            [100.0, 90.0],
        )
        self.assertEqual(env.trade_history[0]["profit"], 0.0)
        self.assertAlmostEqual(env.trade_history[1]["profit"], expected_pnl)

        figure = env.get_env_plot()
        marker_lines = [
            line
            for line in figure.axes[0].lines
            if line.get_linestyle() == "None" and len(line.get_xdata()) == 1
        ]
        self.assertEqual([line.get_marker() for line in marker_lines], ["^", "."])
        self.assertEqual([line.get_xdata()[0] for line in marker_lines], [1, 1])

    def test_nonpositive_equity_is_not_reported_as_zero_return(self):
        env = self.make_env([100.0, 100.0, 100.0, 300.0, 300.0], fee=0.0)
        env.step(Actions.Short_enter.value)
        _, _, terminated, _, info = env.step(Actions.Neutral.value)
        self.assertTrue(terminated)
        self.assertTrue(math.isnan(info["most_recent_return"]))
        self.assertTrue(math.isnan(info["most_recent_profit"]))


if __name__ == "__main__":
    unittest.main()
