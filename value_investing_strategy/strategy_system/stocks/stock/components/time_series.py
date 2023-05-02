from pathlib import Path
from value_investing_strategy.strategy_system.stocks.stock.components.stock_component import StockComponent


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
