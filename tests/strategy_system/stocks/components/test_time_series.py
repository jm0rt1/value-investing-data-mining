import unittest
from datetime import datetime

from value_investing_strategy.strategy_system.stocks.stock.components.time_series import MetaData, MonthlyData, TimeSeriesMonthly


class TestTimeSeriesMonthly(unittest.TestCase):

    def setUp(self):
        meta_data = MetaData(symbol="AAPL", last_refreshed=datetime(
            2023, 1, 3), time_zone="US/Eastern")
        monthly_time_series = [
            MonthlyData(date=datetime(2023, 1, 3), open=100,
                        high=110, low=90, close=105, volume=1000000),
            MonthlyData(date=datetime(2023, 2, 3), open=105,
                        high=115, low=95, close=110, volume=1100000),
            MonthlyData(date=datetime(2023, 3, 3), open=110,
                        high=120, low=100, close=115, volume=1200000),
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

        self.assertAlmostEqual(stock_returns, 0.09523809523809523)

    def test_calculate_stock_returns_no_data(self):
        initial_date = datetime(2021, 1, 1)
        final_date = datetime(2021, 12, 31)

        stock_returns = self.time_series_monthly.calculate_stock_returns(
            initial_date, final_date)

        self.assertIsNone(stock_returns)


if __name__ == "__main__":
    unittest.main()
