import json
import unittest
from pathlib import Path

from value_investing_strategy.strategy_system.stocks.stock.components.IncomeStatement import (
    IncomeReport, IncomeStatement)
from value_investing_strategy.strategy_system.stocks.StocksInUse import StocksInUse

TEST_FILE = Path(
    "tests/test_files/inputs/strategy_system/stocks/stock/components/sample_income_report.json")

DATA = {
    "symbol": "AAPL",
    "annualReports": [
        {
            "fiscalDateEnding": "2022-09-30",
            "reportedCurrency": "USD",
            "grossProfit": "170782000000",
            "totalRevenue": "391397000000",
            "costOfRevenue": "248640000000",
            "costofGoodsAndServicesSold": "223546000000",
            "operatingIncome": "119437000000",
            "sellingGeneralAndAdministrative": "25094000000",
            "researchAndDevelopment": "26251000000",
            "operatingExpenses": "51345000000",
            "investmentIncomeNet": "2825000000",
            "netInterestIncome": "-2931000000",
            "interestIncome": "106000000",
            "interestExpense": "2931000000",
            "nonInterestIncome": "394328000000",
            "otherNonOperatingIncome": "-228000000",
            "depreciation": "8700000000",
            "depreciationAndAmortization": "11104000000",
            "incomeBeforeTax": "119103000000",
            "incomeTaxExpense": "19300000000",
            "interestAndDebtExpense": "2931000000",
            "netIncomeFromContinuingOperations": "99803000000",
            "comprehensiveIncomeNetOfTax": "88531000000",
            "ebit": "122034000000",
            "ebitda": "133138000000",
            "netIncome": "99803000000"
        },
    ],
    "quarterlyReports": [
        {
            "fiscalDateEnding": "2022-12-31",
            "reportedCurrency": "USD",
            "grossProfit": "50332000000",
            "totalRevenue": "116151000000",
            "costOfRevenue": "73429000000",
            "costofGoodsAndServicesSold": "66822000000",
            "operatingIncome": "36016000000",
            "sellingGeneralAndAdministrative": "6607000000",
            "researchAndDevelopment": "7709000000",
            "operatingExpenses": "14316000000",
            "investmentIncomeNet": "868000000",
            "netInterestIncome": "-1003000000",
            "interestIncome": "135000000",
            "interestExpense": "1003000000",
            "nonInterestIncome": "117154000000",
            "otherNonOperatingIncome": "-258000000",
            "depreciation": "None",
            "depreciationAndAmortization": "2916000000",
            "incomeBeforeTax": "35623000000",
            "incomeTaxExpense": "5625000000",
            "interestAndDebtExpense": "1003000000",
            "netIncomeFromContinuingOperations": "29998000000",
            "comprehensiveIncomeNetOfTax": "28195000000",
            "ebit": "36626000000",
            "ebitda": "38539000000",
            "netIncome": "29998000000"
        }]
}


class TestIncomeReport(unittest.TestCase):

    def test_load_from_dict(self):
        dict_report: dict[str, str] = DATA["annualReports"][0]  # type:ignore
        statement = IncomeReport.from_dict(dict_report)

        self.assertIsInstance(statement, IncomeReport)
        self.assertEqual(statement.fiscal_date_ending, "2022-09-30")
        self.assertEqual(statement.reported_currency, "USD")
        self.assertEqual(statement.gross_profit, 170782000000)
        self.assertEqual(statement.total_revenue, 391397000000)
        self.assertEqual(statement.cost_of_revenue, 248640000000)
        self.assertEqual(
            statement.cost_of_goods_and_services_sold, 223546000000)
        self.assertEqual(statement.research_and_development, 26251000000)
        self.assertEqual(
            statement.operating_expenses, 51345000000)
        self.assertEqual(statement.investment_income_net, 2825000000)
        self.assertEqual(statement.net_interest_income, -2931000000)
        self.assertEqual(statement.interest_income, 106000000)
        self.assertEqual(statement.net_interest_income, -2931000000)
        self.assertEqual(statement.non_interest_income, 394328000000)
        self.assertEqual(statement.other_non_operating_income, -228000000)
        self.assertEqual(statement.depreciation, 8700000000)
        self.assertEqual(statement.depreciation_and_amortization, 11104000000)
        self.assertEqual(statement.income_before_tax, 119103000000)
        self.assertEqual(statement.income_tax_expense, 19300000000)
        self.assertEqual(statement.interest_and_debt_expense, 2931000000)
        self.assertEqual(
            statement.net_income_from_continuing_operations, 99803000000)
        self.assertEqual(
            statement.comprehensive_income_net_of_tax, 88531000000)
        self.assertEqual(statement.ebit, 122034000000)
        self.assertEqual(statement.ebitda, 133138000000)
        self.assertEqual(statement.net_income, 99803000000)


class TestIncomeStatement(unittest.TestCase):
    def test_from_dict(self):
        statement = IncomeStatement.from_dict(DATA)
        self.assertIsInstance(statement, IncomeStatement)
        self.assertEqual(len(statement.annual_reports), 1)
        self.assertIsInstance(statement.annual_reports[0], IncomeReport)
        self.assertEqual(len(statement.quarterly_reports), 1)
        self.assertIsInstance(statement.quarterly_reports[0], IncomeReport)

    def test_from_file(self):
        statement = IncomeStatement.from_json_file(Path(
            "tests/test_files/inputs/strategy_system/stocks/stock/components/AAPL.IncomeStatement.json"))
        self.assertIsInstance(statement, IncomeStatement)
        self.assertEqual(len(statement.annual_reports), 5)
        self.assertIsInstance(statement.annual_reports[0], IncomeReport)
        self.assertEqual(len(statement.quarterly_reports), 20)
        self.assertIsInstance(statement.quarterly_reports[0], IncomeReport)

    def test_all_cached(self):
        tickers = StocksInUse.list_cached_tickers(
            Path("value_investing_strategy/data/SimpleAlphaVantageCacher/output/json_cache/covered.txt"))

        pass
