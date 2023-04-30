from __future__ import annotations


import unittest

from value_investing_strategy.strategy_system.stocks.stock.stock import Stock
from pathlib import Path

from value_investing_strategy.strategy_system.stocks.stock.components.balance_sheet import BalanceReport
from value_investing_strategy.strategy_system.stocks.stock.components.income_statement import IncomeStatement, IncomeReport
from value_investing_strategy.strategy_system.stocks.stock.components.earnings import EarningsReport, EarningsStatement
from value_investing_strategy.strategy_system.stocks.stock.components.cash_flow import Cashflow, CashflowReport
from value_investing_strategy.strategy_system.stocks.stock.components.company_overview import CompanyOverview


TEST_PATH = Path(
    "tests/test_files/inputs/strategy_system/stocks/stock/components")


class TestStock(unittest.TestCase):
    def test_from_alpha_vantage_data(self):
        stock = Stock.from_alpha_vantage_data('AAPL', TEST_PATH)

        # balance sheet
        self.assertEqual(len(stock.balance_sheet.annual_reports), 5)
        self.assertIsInstance(
            stock.balance_sheet.annual_reports[0], BalanceReport)
        self.assertEqual(len(stock.balance_sheet.quarterly_reports), 20)
        self.assertIsInstance(
            stock.balance_sheet.quarterly_reports[0], BalanceReport)

        # income statement
        self.assertIsInstance(
            stock.income_statement, IncomeStatement)
        self.assertEqual(len(stock.income_statement.annual_reports), 5)
        self.assertIsInstance(
            stock.income_statement.annual_reports[0], IncomeReport)
        self.assertEqual(len(stock.income_statement.quarterly_reports), 20)
        self.assertIsInstance(
            stock.income_statement.quarterly_reports[0], IncomeReport)

        # Earnings statement
        self.assertIsInstance(stock.earnings, EarningsStatement)
        self.assertEqual(len(stock.earnings.annual_reports), 28)
        self.assertIsInstance(
            stock.earnings.annual_reports[0], EarningsReport)
        self.assertEqual(len(stock.earnings.quarterly_reports), 108)
        self.assertIsInstance(
            stock.earnings.quarterly_reports[0], EarningsReport)

        # cash flow
        self.assertIsInstance(stock.cash_flow, Cashflow)
        self.assertEqual(len(stock.cash_flow.annual_reports), 5)
        self.assertIsInstance(
            stock.cash_flow.annual_reports[0], CashflowReport)
        self.assertEqual(len(stock.cash_flow.quarterly_reports), 20)
        self.assertIsInstance(
            stock.cash_flow.quarterly_reports[0], CashflowReport)
        self.assertIsInstance(stock.company_overview, CompanyOverview)

    def test_graham_number(self):
        stock = Stock.from_alpha_vantage_data("AAPL", TEST_PATH)

        gn = stock.graham_number
        pass
