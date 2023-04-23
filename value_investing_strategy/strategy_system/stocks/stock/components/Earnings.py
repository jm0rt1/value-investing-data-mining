

from pathlib import Path
from typing import Optional, Union
from value_investing_strategy.strategy_system.stocks.stock.components.StockComponent import StockComponent


from dataclasses import dataclass


@dataclass
class EarningsReport():
    fiscal_date_ending: str
    reported_eps: float
    reported_date: Optional[str] = None
    estimated_eps: Optional[float] = None
    surprise: Optional[float] = None
    surprise_percentage: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        try:
            reported_date = data.get("reportedDate", None)
        except ValueError as _:
            reported_date = None
        try:
            estimated_eps = float(data.get("estimatedEPS", ""))

        except ValueError as _:
            estimated_eps = None
        try:
            surprise = float(data.get("surprise", ""))
        except ValueError as _:
            surprise = None
        try:
            surprise_percentage = float(data.get("surprisePercentage", ""))
        except ValueError as _:
            surprise_percentage = None
        try:
            fiscal_date_ending = str(data.get("fiscalDateEnding", ""))

        except ValueError as _:
            fiscal_date_ending = ""
        try:
            reported_eps = float(data.get("reportedEPS", ""))
        except ValueError as _:
            reported_eps = 0
        return cls(
            fiscal_date_ending=fiscal_date_ending,
            reported_eps=reported_eps,
            reported_date=reported_date,
            estimated_eps=estimated_eps,
            surprise=surprise,
            surprise_percentage=surprise_percentage,
        )


@ dataclass
class EarningsStatement(StockComponent):
    symbol: str
    annual_reports: list[EarningsReport]
    quarterly_reports: list[EarningsReport]

    @ classmethod
    def from_dict(cls, data: dict[str, Union[str, list[dict[str, str]]]]):
        annual_reports: list[EarningsReport] = []
        quarterly_reports: list[EarningsReport] = []

        raw_annual_reports: list[dict[str, str]
                                 ] = data.get("annualEarnings", [])  # type:ignore

        raw_qtrly_reports: list[dict[str, str]
                                ] = data.get("quarterlyEarnings", [])  # type:ignore

        if type(raw_annual_reports) is str:
            raise TypeError("Annual reports should not be a string")
        if type(raw_qtrly_reports) is str:
            raise TypeError("Quarterly reports should not be a string")

        for report in raw_annual_reports:
            annual_reports.append(EarningsReport.from_dict(report))
        for report in raw_qtrly_reports:
            quarterly_reports.append(EarningsReport.from_dict(report))

        symbol = data.get("symbol", "")
        if type(symbol) is not str:
            raise TypeError("symbol needs to be a string")
        return cls(
            symbol=symbol,
            annual_reports=annual_reports,
            quarterly_reports=quarterly_reports
        )

    @classmethod
    def from_json_file(cls, path: Path):
        return cls.from_dict(cls.load_json_dict(path))
