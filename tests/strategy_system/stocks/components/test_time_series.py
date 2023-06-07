import unittest
from datetime import datetime

from src.value_investing_strategy.strategy_system.stocks.stock.components.time_series import MetaData, MonthlyData, TimeSeriesMonthly

import json
from pathlib import Path


JSON_DATA = {
    "Meta Data": {
        "2. Symbol": "AAPL",
        "3. Last Refreshed": "2023-01-03",
        "4. Time Zone": "US/Eastern"
    },
    "Monthly Adjusted Time Series": {
        "2023-01-03": {
            "1. open": "100",
            "2. high": "110",
            "3. low": "90",
            "4. close": "105",
            "5. adjusted close": "10",
            "6. volume": "1000000",
            "7. dividend amount": "1"

        }
    }
}


class TestTimeSeriesMonthly(unittest.TestCase):

    def setUp(self):
        meta_data = MetaData(symbol="AAPL", last_refreshed=datetime(
            2023, 1, 3), time_zone="US/Eastern")
        monthly_time_series = [
            MonthlyData(date=datetime(2023, 1, 3), open=100,
                        high=110, low=90, close=105, volume=1000000, adjusted_close=200, dividend_amount=10),
            MonthlyData(date=datetime(2023, 2, 3), open=105,
                        high=115, low=95, close=110, volume=1100000, adjusted_close=200, dividend_amount=10),
            MonthlyData(date=datetime(2023, 3, 3), open=110,
                        high=120, low=100, close=115, volume=1200000, adjusted_close=210, dividend_amount=10),
        ]

        self.time_series_monthly = TimeSeriesMonthly(
            meta_data, monthly_time_series)

    def test_find_nearest_data(self):
        target_date = datetime(2023, 1, 10)
        nearest_data = self.time_series_monthly.find_nearest_data(target_date)

        self.assertEqual(nearest_data.date, datetime(2023, 1, 3))

    def test_calculate_stock_returns(self):
        initial_date = datetime(2023, 1, 3)
        final_date = datetime(2023, 3, 3)

        stock_returns = self.time_series_monthly.calculate_stock_returns(
            initial_date, final_date)

        self.assertAlmostEqual(stock_returns, 5)

    def test_calculate_stock_returns_no_data(self):
        initial_date = datetime(2021, 1, 1)
        final_date = datetime(2021, 12, 31)

        stock_returns = self.time_series_monthly.calculate_stock_returns(
            initial_date, final_date)

        self.assertEquals(stock_returns, 0)

    def test_from_dict(self):

        time_series_monthly = TimeSeriesMonthly.from_dict(JSON_DATA)
        self.assertEqual(time_series_monthly.meta_data.symbol, "AAPL")
        self.assertEqual(time_series_monthly.monthly_time_series[0].close, 105)

    def test_from_json_file(self):
        json_file_path = Path("test_time_series_monthly.json")

        with open(json_file_path, "w") as f:
            json.dump(JSON_DATA, f)

        time_series_monthly = TimeSeriesMonthly.from_json_file(json_file_path)
        self.assertEqual(time_series_monthly.meta_data.symbol, "AAPL")
        self.assertEqual(time_series_monthly.monthly_time_series[0].close, 105)

        json_file_path.unlink()  # Clean up the created test file


if __name__ == "__main__":
    unittest.main()
