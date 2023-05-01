
from pathlib import Path
import unittest
from value_investing_strategy.strategy_system.stocks.stocks_in_use import StocksInUse


class TestStocksInUse(unittest.TestCase):
    def test_list_available(self):
        cached = StocksInUse.list_cached_tickers(
            Path("./tests/test_files/inputs/strategy_system/stocks/StocksInUse/covered.txt"))

        expected = [
            "MMM",
            "AOS",
            "ABT",
            "ABBV",
            "ACN",
            "ATVI",
            "ADM",
            "ADBE",
            "ADP",
            "AAP",
            "AES",
            "AFL",
            "A",
            "APD",
            "AKAM"
        ]
        self.assertEqual(cached, expected)

    def test_load_stocks(self):
        cached = StocksInUse.list_cached_tickers(
            Path("./tests/test_files/inputs/strategy_system/stocks/StocksInUse/covered.txt"))
        stocks = StocksInUse()
        stocks.load_stocks(cached)

        df = stocks.to_data_frame()
        df.to_csv("out.csv")
        pass
