"""Unit tests for the retention-offer economics. Run with:  python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from churn import Offer, expected_profit, policies, realised_profit  # noqa: E402


class TestOfferEconomics(unittest.TestCase):
    def setUp(self):
        self.offer = Offer(cost=60, success_rate=0.3, months_kept=12, margin=0.6)

    def test_value_if_saved(self):
        # 12 months x 60% margin x $100 bill = $720
        self.assertAlmostEqual(self.offer.value_if_saved(100), 720)

    def test_break_even_risk(self):
        # at a $100 bill the offer breaks even when risk x 0.3 x 720 = 60, i.e. risk = 0.2778
        p = 60 / (0.3 * 720)
        self.assertAlmostEqual(expected_profit(p, 100, self.offer), 0, places=9)
        self.assertGreater(expected_profit(p + 0.01, 100, self.offer), 0)

    def test_low_bills_never_worth_it(self):
        # an $18 bill can't repay a $60 credit even if the customer is certain to leave
        self.assertLess(expected_profit(1.0, 18, self.offer), 0)

    def test_targeting_nobody_earns_nothing(self):
        self.assertEqual(realised_profit(np.zeros(3, bool), [1, 0, 1], [80, 50, 90], self.offer), 0)

    def test_realised_profit_by_hand(self):
        # target a leaver ($100 bill) and a stayer: 0.3 x 720 - 60 = 156, then -60 for the stayer
        got = realised_profit([True, True], [1, 0], [100, 100], self.offer)
        self.assertAlmostEqual(got, 156 - 60)

    def test_hindsight_is_the_upper_bound(self):
        rng = np.random.default_rng(0)
        churned = rng.random(500) < 0.27
        p = np.clip(churned * 0.4 + rng.random(500) * 0.6, 0, 1)
        bills = rng.uniform(20, 110, 500)
        table = policies(p, churned, bills, self.offer)
        # hindsight targets exactly the customers whose offer pays off, so no rule can beat it
        best_rule = table.drop("Perfect hindsight").profit.max()
        self.assertGreaterEqual(table.loc["Perfect hindsight", "profit"], best_rule)
        self.assertEqual(table.loc["Nobody", "profit"], 0)


if __name__ == "__main__":
    unittest.main()
