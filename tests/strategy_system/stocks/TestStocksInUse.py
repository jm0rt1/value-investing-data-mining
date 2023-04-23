
from pathlib import Path
import unittest
from value_investing_strategy.strategy_system.stocks.StocksInUse import StocksInUse


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
