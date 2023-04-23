import json
import unittest
from pathlib import Path

from value_investing_strategy.strategy_system.stocks.stock.components.CashFlow import (
    CashflowReport, Cashflow)
TEST_PATH = Path(
    "tests/test_files/inputs/strategy_system/stocks/stock/components/AAPL.CashFlow.json")
DATA = {
    "symbol": "ABBV",
    "annualReports": [
        {
            "fiscalDateEnding": "2022-12-31",
            "reportedCurrency": "USD",
            "operatingCashflow": "24943000000",
            "paymentsForOperatingActivities": "5746000000",
            "proceedsFromOperatingActivities": "None",
            "changeInOperatingLiabilities": "1605000000",
            "changeInOperatingAssets": "2405000000",
            "depreciationDepletionAndAmortization": "8467000000",
            "capitalExpenditures": "695000000",
            "changeInReceivables": "1455000000",
            "changeInInventory": "686000000",
            "profitLoss": "11845000000",
            "cashflowFromInvestment": "-623000000",
            "cashflowFromFinancing": "-24803000000",
            "proceedsFromRepaymentsOfShortTermDebt": "None",
            "paymentsForRepurchaseOfCommonStock": "1487000000",
            "paymentsForRepurchaseOfEquity": "1487000000",
            "paymentsForRepurchaseOfPreferredStock": "None",
            "dividendPayout": "10043000000",
            "dividendPayoutCommonStock": "10043000000",
            "dividendPayoutPreferredStock": "None",
            "proceedsFromIssuanceOfCommonStock": "None",
            "proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet": "2000000000",
            "proceedsFromIssuanceOfPreferredStock": "None",
            "proceedsFromRepurchaseOfEquity": "-1225000000",
            "proceedsFromSaleOfTreasuryStock": "None",
            "changeInCashAndCashEquivalents": "-483000000",
            "changeInExchangeRate": "None",
            "netIncome": "11836000000"
        },

    ],
    "quarterlyReports": [
        {
            "fiscalDateEnding": "2022-12-31",
            "reportedCurrency": "USD",
            "operatingCashflow": "7428000000",
            "paymentsForOperatingActivities": "None",
            "proceedsFromOperatingActivities": "None",
            "changeInOperatingLiabilities": "1275000000",
            "changeInOperatingAssets": "790000000",
            "depreciationDepletionAndAmortization": "2157000000",
            "capitalExpenditures": "213000000",
            "changeInReceivables": "416000000",
            "changeInInventory": "170000000",
            "profitLoss": "2472000000",
            "cashflowFromInvestment": "-448000000",
            "cashflowFromFinancing": "-9634000000",
            "proceedsFromRepaymentsOfShortTermDebt": "0",
            "paymentsForRepurchaseOfCommonStock": "4000000",
            "paymentsForRepurchaseOfEquity": "4000000",
            "paymentsForRepurchaseOfPreferredStock": "None",
            "dividendPayout": "2506000000",
            "dividendPayoutCommonStock": "2506000000",
            "dividendPayoutPreferredStock": "None",
            "proceedsFromIssuanceOfCommonStock": "None",
            "proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet": "0",
            "proceedsFromIssuanceOfPreferredStock": "None",
            "proceedsFromRepurchaseOfEquity": "49000000",
            "proceedsFromSaleOfTreasuryStock": "None",
            "changeInCashAndCashEquivalents": "-2654000000",
            "changeInExchangeRate": "None",
            "netIncome": "2473000000"
        }
    ]
}


class TestCashFlowReport(unittest.TestCase):

    def test_load_from_dict(self):
        dict_report: dict[str, str] = DATA["quarterlyReports"][0]  # nopep8  #type:ignore
        statement = CashflowReport.from_dict(dict_report)

        self.assertIsInstance(statement, CashflowReport)

        self.assertEqual(statement.fiscal_date_ending, "2022-12-31")
        self.assertEqual(statement.reported_currency, "USD")
        self.assertEqual(statement.operating_cashflow, 7428000000)
        self.assertEqual(statement.payments_for_operating_activities, 0)
        self.assertEqual(statement.proceeds_from_operating_activities, 0)
        self.assertEqual(statement.change_in_operating_liabilities, 1275000000)
        self.assertEqual(statement.change_in_operating_assets, 790000000)
        self.assertEqual(
            statement.depreciation_depletion_and_amortization, 2157000000)
        self.assertEqual(statement.capital_expenditures, 213000000)
        self.assertEqual(statement.change_in_receivables, 416000000)
        self.assertEqual(statement.change_in_inventory, 170000000)
        self.assertEqual(statement.profit_loss, 2472000000)
        self.assertEqual(statement.cash_flow_from_investment, -448000000)
        self.assertEqual(statement.cash_flow_from_financing, -9634000000)
        self.assertEqual(
            statement.proceeds_from_repayments_of_short_term_debt, 0)
        self.assertEqual(
            statement.payments_for_repurchase_of_common_stock, 4000000)
        self.assertEqual(statement.payments_for_repurchase_of_equity, 4000000)
        self.assertEqual(
            statement.payments_for_repurchase_of_preferred_stock, 0)
        self.assertEqual(statement.dividend_payout, 2506000000)
        self.assertEqual(statement.dividend_payout_common_stock, 2506000000)
        self.assertEqual(statement.dividend_payout_preferred_stock, 0)
        self.assertEqual(
            statement.proceeds_from_issuance_of_common_stock, 0)
        self.assertEqual(
            statement.proceeds_from_issuance_of_long_term_debt_and_capital_securities_net, 0)
        self.assertEqual(
            statement.proceeds_from_issuance_of_preferred_stock, 0)
        self.assertEqual(
            statement.proceeds_from_repurchase_of_equity, 49000000)
        self.assertEqual(statement.proceeds_from_sale_of_treasury_stock, 0)
        self.assertEqual(
            statement.change_in_cash_and_cash_equivalents, -2654000000)
        self.assertEqual(statement.change_in_exchange_rate, 0)
        self.assertEqual(statement.net_income, 2473000000)


class TestCashFlow(unittest.TestCase):
    def test_from_dict(self):
        statement = Cashflow.from_dict(DATA)
        self.assertIsInstance(statement, Cashflow)
        self.assertEqual(len(statement.annual_reports), 1)
        self.assertIsInstance(statement.annual_reports[0], CashflowReport)
        self.assertEqual(len(statement.quarterly_reports), 1)
        self.assertIsInstance(statement.quarterly_reports[0], CashflowReport)

    def test_from_file(self):
        statement = Cashflow.from_json_file(TEST_PATH)
        self.assertIsInstance(statement, Cashflow)
        self.assertEqual(len(statement.annual_reports), 5)
        self.assertIsInstance(statement.annual_reports[0], CashflowReport)
        self.assertEqual(len(statement.quarterly_reports), 20)
        self.assertIsInstance(statement.quarterly_reports[0], CashflowReport)
