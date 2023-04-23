from __future__ import annotations

import unittest
from value_investing_strategy.strategy_system.stocks.stock.Stock import Stock, GrahamNumberCalculator


class TestGrahamNumberCalculator(unittest.TestCase):

    def test_init(self):
        stock = Stock.from_alpha_vantage_data("AAPL")
        GrahamNumberCalculator(stock)

    def test_calculate(self):
        expected = 229.46
        out = GrahamNumberCalculator.calculate(45, 52)
        self.assertEqual(expected, out)
