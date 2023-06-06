# TODO

from pathlib import Path
import unittest

from src.value_investing_strategy.strategy_system.stocks.stock.components.earnings import EarningsReport, EarningsStatement
from src.value_investing_strategy.strategy_system.stocks.stocks_in_use import StocksInUse


TEST_PATH = Path(
    "tests/test_files/inputs/strategy_system/stocks/stock/components/AAPL.Earnings.json")
DATA = {
    "symbol": "AAPL",
    "annualEarnings": [
        {
            "fiscalDateEnding": "2022-12-31",
            "reportedEPS": "1.88"
        }
    ],
    "quarterlyEarnings": [
        {
            "fiscalDateEnding": "2022-12-31",
            "reportedDate": "2023-02-02",
            "reportedEPS": "1.88",
            "estimatedEPS": "1.94",
            "surprise": "-0.06",
            "surprisePercentage": "-3.0928"
        }
    ]
}


class TestEarningsReport(unittest.TestCase):

    def test_load_from_dict(self):
        # type:ignore
        dict_report: dict[str, str] = DATA["quarterlyEarnings"][0]
        statement = EarningsReport.from_dict(dict_report)

        self.assertIsInstance(statement, EarningsReport)

        self.assertEqual(statement.fiscal_date_ending, "2022-12-31")
        self.assertEqual(statement.reported_date, "2023-02-02")
        self.assertEqual(statement.reported_eps, 1.88)
        self.assertEqual(statement.estimated_eps, 1.94)
        self.assertEqual(statement.surprise, -0.06)
        self.assertEqual(statement.surprise_percentage, -3.0928)

        dict_report: dict[str, str] = DATA["annualEarnings"][0]
        statement = EarningsReport.from_dict(dict_report)

        self.assertIsInstance(statement, EarningsReport)

        self.assertEqual(statement.fiscal_date_ending, "2022-12-31")
        self.assertEqual(statement.reported_date, None)
        self.assertEqual(statement.reported_eps, 1.88)
        self.assertEqual(statement.estimated_eps, None)
        self.assertEqual(statement.surprise, None)
        self.assertEqual(statement.surprise_percentage, None)


class TestEarningsStatement(unittest.TestCase):
    def test_from_dict(self):
        statement = EarningsStatement.from_dict(DATA)
        self.assertIsInstance(statement, EarningsStatement)
        self.assertEqual(len(statement.annual_reports), 1)
        self.assertIsInstance(statement.annual_reports[0], EarningsReport)
        self.assertEqual(len(statement.quarterly_reports), 1)
        self.assertIsInstance(statement.quarterly_reports[0], EarningsReport)

    def test_from_file(self):
        statement = EarningsStatement.from_json_file(TEST_PATH)
        self.assertIsInstance(statement, EarningsStatement)
        self.assertEqual(len(statement.annual_reports), 28)
        self.assertIsInstance(statement.annual_reports[0], EarningsReport)
        self.assertEqual(len(statement.quarterly_reports), 108)
        self.assertIsInstance(statement.quarterly_reports[0], EarningsReport)

    def test_all_cached(self):
        tickers = StocksInUse.list_cached_tickers(
            Path("src/value_investing_strategy/data/SimpleAlphaVantageCacher/output/json_cache/covered.txt"))

        pass
