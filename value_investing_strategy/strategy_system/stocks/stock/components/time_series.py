from pathlib import Path
from value_investing_strategy.strategy_system.stocks.stock.components.stock_component import StockComponent

import pandas as pd
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MonthlyData:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class MetaData:
    symbol: str
    last_refreshed: str
    time_zone: str


@dataclass
class TimeSeriesMonthly(StockComponent):
    meta_data: MetaData
    monthly_time_series: list[MonthlyData]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimeSeriesMonthly":
        meta_data_dict = data.get("Meta Data", {})
        meta_data = MetaData(
            symbol=meta_data_dict.get("2. Symbol", ""),
            last_refreshed=meta_data_dict.get("3. Last Refreshed", ""),
            time_zone=meta_data_dict.get("4. Time Zone", ""),
        )
        raw_monthly_time_series = data.get("Monthly Time Series", {})
        monthly_time_series = [
            MonthlyData(
                date=date,
                open=float(info.get("1. open", 0)),
                high=float(info.get("2. high", 0)),
                low=float(info.get("3. low", 0)),
                close=float(info.get("4. close", 0)),
                volume=int(info.get("5. volume", 0)),
            )
            for date, info in raw_monthly_time_series.items()
        ]
        return cls(meta_data, monthly_time_series)

    @classmethod
    def from_json_file(cls, path: Path) -> "TimeSeriesMonthly":
        data = cls.load_json_dict(path)
        return cls.from_dict(data)

    def find_nearest_data(self, target_date: str) -> Optional[MonthlyData]:
        target_date = pd.to_datetime(target_date)
        dates = [pd.to_datetime(data.date)
                 for data in self.monthly_time_series]
        nearest_date = min(dates, key=lambda x: abs(x - target_date))
        nearest_data = next((data for data in self.monthly_time_series if pd.to_datetime(
            data.date) == nearest_date), None)
        return nearest_data

    def calculate_stock_returns(self, initial_date: str, final_date: str) -> Optional[float]:
        initial_data = self.find_nearest_data(initial_date)
        final_data = self.find_nearest_data(final_date)

        if initial_data is None or final_data is None:
            return None

        initial_price = initial_data.close
        final_price = final_data.close

        stock_returns = (final_price - initial_price) / initial_price
        return stock_returns
