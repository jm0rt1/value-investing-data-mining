from pathlib import Path
import unittest
from value_investing_strategy.strategy_system.stocks.alpha_vantage.AlphaVantageClient import AlphaVantageClient, generate_json_file_path

TEST_SYMBOL = "AAPL"
TO_FILE_OUT_PATH = Path(
    "./tests/test_files/outputs/strategy_system/stocks/alpha_vantage/TestAlphaVantageClient")
TO_FILE_OUT_PATH.mkdir(exist_ok=True, parents=True)


class TestIncomeStatement(unittest.TestCase):

    def test_to_file(self):
        path = generate_json_file_path(
            TO_FILE_OUT_PATH, TEST_SYMBOL, AlphaVantageClient.IncomeStatement.TYPE_STR)
        path.unlink(missing_ok=True)
        self.assertFalse(path.exists())
        # AlphaVantageClient.IncomeStatement.to_json_file(
        # TEST_SYMBOL, TO_FILE_OUT_PATH)
        self.assertTrue(path.exists())


class TestBalanceSheet(unittest.TestCase):

    def test_to_file(self):
        path = generate_json_file_path(
            TO_FILE_OUT_PATH, TEST_SYMBOL, AlphaVantageClient.BalanceSheet.TYPE_STR)
        path.unlink(missing_ok=True)
        self.assertFalse(path.exists())
        # AlphaVantageClient.BalanceSheet.to_json_file(
        # TEST_SYMBOL, TO_FILE_OUT_PATH)
        self.assertTrue(path.exists())


class TestCashFlow(unittest.TestCase):

    def test_to_file(self):

        path = generate_json_file_path(
            TO_FILE_OUT_PATH, TEST_SYMBOL, AlphaVantageClient.CashFlow.TYPE_STR)
        path.unlink(missing_ok=True)
        self.assertFalse(path.exists())

        # AlphaVantageClient.CashFlow.to_json_file(
        # TEST_SYMBOL, TO_FILE_OUT_PATH)

        self.assertTrue(path.exists())


class TestEarnings(unittest.TestCase):

    def test_to_file(self):
        path = generate_json_file_path(
            TO_FILE_OUT_PATH, TEST_SYMBOL, AlphaVantageClient.Earnings.TYPE_STR)
        path.unlink(missing_ok=True)
        self.assertFalse(path.exists())
        # AlphaVantageClient.Earnings.to_json_file(
        # TEST_SYMBOL, TO_FILE_OUT_PATH)

        self.assertTrue(path.exists())


class TestCompanyOverview(unittest.TestCase):

    def test_to_file(self):
        path = generate_json_file_path(
            TO_FILE_OUT_PATH, TEST_SYMBOL, AlphaVantageClient.CompanyOverview.TYPE_STR)
        path.unlink(missing_ok=True)
        self.assertFalse(path.exists())
        # AlphaVantageClient.CompanyOverview.to_json_file(
        # TEST_SYMBOL, TO_FILE_OUT_PATH)
        self.assertTrue(path.exists())
