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
        env = self.make_env([100.0, 100.0, 110.0, 110.0, 110.0])
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
                        price = prices[info["tick"]]
                        if action == enter:
                            entry_price = price
                        pnl = 0.0
                        if entry_price is not None:
                            price_ratio = price / entry_price
                            fee_factor = (1 + env.fee) ** 2
                            pnl = (
                                1 - price_ratio * fee_factor
                                if short
                                else price_ratio / fee_factor - 1
                            )
                        equity = capital * (1 + pnl) if compound else capital + pnl
                        if action == exit_action:
                            capital = equity
                            entry_price = None
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
                    env.reset()
                    self.assertEqual(env.get_most_recent_return(), 0.0)
                    self.assertEqual(env.get_most_recent_profit(), 0.0)

    def test_nonpositive_equity_is_not_reported_as_zero_return(self):
        env = self.make_env([100.0, 100.0, 100.0, 300.0, 300.0], fee=0.0)
        env.step(Actions.Short_enter.value)
        _, _, terminated, _, info = env.step(Actions.Neutral.value)
        self.assertTrue(terminated)
        self.assertTrue(math.isnan(info["most_recent_return"]))
        self.assertTrue(math.isnan(info["most_recent_profit"]))


if __name__ == "__main__":
    unittest.main()
