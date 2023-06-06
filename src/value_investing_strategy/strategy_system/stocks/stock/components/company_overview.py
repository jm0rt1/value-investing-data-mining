

from pathlib import Path
from typing import TypeVar, Union
from src.value_investing_strategy.strategy_system.stocks.stock.components.stock_component import StockComponent


from dataclasses import dataclass


@dataclass
class CompanyOverview(StockComponent):
    symbol: str
    asset_type: str
    name: str
    description: str
    cik: int
    exchange: str
    currency: str
    country: str
    sector: str
    industry: str
    address: str
    fiscal_year_end: str
    latest_quarter: str
    market_capitalization: float
    ebitda: float
    pe_ratio: float
    peg_ratio: float
    book_value: float
    dividend_per_share: float
    dividend_yield: float
    eps: float
    revenue_per_share_ttm: float
    profit_margin: float
    operating_margin_ttm: float
    return_on_assets_ttm: float
    return_on_equity_ttm: float
    revenue_ttm: float
    gross_profit_ttm: float
    diluted_eps_ttm: float
    quarterly_earnings_growth_yoy: float
    quarterly_revenue_growth_yoy: float
    analyst_target_price: float
    trailing_pe: float
    forward_pe: float
    price_to_sales_ratio: float
    price_to_book_ratio: float
    ev_to_revenue: float
    ev_to_ebitda: float
    beta: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    fifty_day_moving_average: float
    two_hundred_day_moving_average: float
    shares_outstanding: float
    dividend_date: str
    ex_dividend_date: str

    @classmethod
    def from_dict(cls, data: dict[str, str]):
        try:
            ebitda = float(data.get("EBITDA", 0))
        except ValueError as _:
            ebitda = 0

        try:
            ev_to_revenue = float(data.get("EVToRevenue", 0))
        except ValueError as _:
            ev_to_revenue = 0

        try:
            ev_to_ebitda = float(data.get("EVToEBITDA", 0))
        except ValueError as _:
            ev_to_ebitda = 0

        try:
            symbol = data.get("Symbol", "")
        except ValueError as _:
            symbol = ""
        try:
            asset_type = data.get("AssetType", "")
        except ValueError as _:
            asset_type = ""
        try:
            name = data.get("Name", "")
        except ValueError as _:
            name = ""
        try:
            description = data.get("Description", "")
        except ValueError as _:
            description = ""
        try:
            cik = int(data.get("CIK", 0))
        except ValueError as _:
            cik = 0
        try:
            exchange = data.get("Exchange", "")
        except ValueError as _:
            exchange = ""
        try:
            currency = data.get("Currency", "")
        except ValueError as _:
            currency = ""
        try:
            country = data.get("Country", "")
        except ValueError as _:
            country = ""
        try:
            sector = data.get("Sector", "")
        except ValueError as _:
            sector = ""
        try:
            industry = data.get("Industry", "")
        except ValueError as _:
            industry = ""
        try:
            address = data.get("Address", "")
        except ValueError as _:
            address = ""
        try:
            fiscal_year_end = data.get("FiscalYearEnd", "")
        except ValueError as _:
            fiscal_year_end = ""
        try:
            latest_quarter = data.get("LatestQuarter", "")
        except ValueError as _:
            latest_quarter = ""
        try:
            market_capitalization = float(data.get("MarketCapitalization", 0))
        except ValueError as _:
            market_capitalization = 0

        try:
            pe_ratio = float(data.get("PERatio", 0))
        except ValueError as _:
            pe_ratio = 0
        try:
            peg_ratio = float(data.get("PEGRatio", 0))
        except ValueError as _:
            peg_ratio = 0
        try:
            book_value = float(data.get("BookValue", 0))
        except ValueError as _:
            book_value = 0
        try:
            dividend_per_share = float(data.get("DividendPerShare", 0))
        except ValueError as _:
            dividend_per_share = 0
        try:
            dividend_yield = float(data.get("DividendYield", 0))
        except ValueError as _:
            dividend_yield = 0
        try:
            eps = float(data.get("EPS", 0))
        except ValueError as _:
            eps = 0
        try:
            revenue_per_share_ttm = float(data.get("RevenuePerShareTTM", 0))
        except ValueError as _:
            revenue_per_share_ttm = 0
        try:
            profit_margin = float(data.get("ProfitMargin", 0))
        except ValueError as _:
            profit_margin = 0
        try:
            operating_margin_ttm = float(data.get("OperatingMarginTTM", 0))
        except ValueError as _:
            operating_margin_ttm = 0
        try:
            return_on_assets_ttm = float(data.get("ReturnOnAssetsTTM", 0))
        except ValueError as _:
            return_on_assets_ttm = 0
        try:
            return_on_equity_ttm = float(data.get("ReturnOnEquityTTM", 0))
        except ValueError as _:
            return_on_equity_ttm = 0
        try:
            revenue_ttm = float(data.get("RevenueTTM", 0))
        except ValueError as _:
            revenue_ttm = 0
        try:
            gross_profit_ttm = float(data.get("GrossProfitTTM", 0))
        except ValueError as _:
            gross_profit_ttm = 0
        try:
            diluted_eps_ttm = float(data.get("DilutedEPSTTM", 0))
        except ValueError as _:
            diluted_eps_ttm = 0
        try:
            quarterly_earnings_growth_yoy = float(
                data.get("QuarterlyEarningsGrowthYOY", 0))
        except ValueError as _:
            quarterly_earnings_growth_yoy = 0
        try:
            quarterly_revenue_growth_yoy = float(
                data.get("QuarterlyRevenueGrowthYOY", 0))
        except ValueError as _:
            quarterly_revenue_growth_yoy = 0
        try:
            analyst_target_price = float(data.get("AnalystTargetPrice", 0))
        except ValueError as _:
            analyst_target_price = 0
        try:
            trailing_pe = float(data.get("TrailingPE", 0))
        except ValueError as _:
            trailing_pe = 0
        try:
            forward_pe = float(data.get("ForwardPE", 0))
        except ValueError as _:
            forward_pe = 0
        try:
            price_to_sales_ratio = float(data.get("PriceToSalesRatioTTM", 0))
        except ValueError as _:
            price_to_sales_ratio = 0
        try:
            price_to_book_ratio = float(data.get("PriceToBookRatio", 0))
        except ValueError as _:
            price_to_book_ratio = 0
        try:
            ev_to_revenue = ev_to_revenue
        except ValueError as _:
            ev_to_revenue = 0
        try:
            beta = float(data.get("Beta", 0))
        except ValueError as _:
            beta = 0
        try:
            fifty_two_week_high = float(data.get("52WeekHigh", 0))
        except ValueError as _:
            fifty_two_week_high = 0
        try:
            fifty_two_week_low = float(data.get("52WeekLow", 0))
        except ValueError as _:
            fifty_two_week_low = 0
        try:
            fifty_day_moving_average = float(data.get("50DayMovingAverage", 0))
        except ValueError as _:
            fifty_day_moving_average = 0
        try:
            two_hundred_day_moving_average = float(
                data.get("200DayMovingAverage", 0))
        except ValueError as _:
            two_hundred_day_moving_average = 0
        try:
            shares_outstanding = float(data.get("SharesOutstanding", 0))
        except ValueError as _:
            shares_outstanding = 0
        try:
            dividend_date = data.get("DividendDate", "")
        except ValueError as _:
            dividend_date = ""
        try:
            ex_dividend_date = data.get("ExDividendDate", "")
        except ValueError as _:
            ex_dividend_date = ""

        return cls(
            symbol=symbol,
            asset_type=asset_type,
            name=name,
            description=description,
            cik=cik,
            exchange=exchange,
            currency=currency,
            country=country,
            sector=sector,
            industry=industry,
            address=address,
            fiscal_year_end=fiscal_year_end,
            latest_quarter=latest_quarter,
            market_capitalization=market_capitalization,
            ebitda=ebitda,
            pe_ratio=pe_ratio,
            peg_ratio=peg_ratio,
            book_value=book_value,
            dividend_per_share=dividend_per_share,
            dividend_yield=dividend_yield,
            eps=eps,
            revenue_per_share_ttm=revenue_per_share_ttm,
            profit_margin=profit_margin,
            operating_margin_ttm=operating_margin_ttm,
            return_on_assets_ttm=return_on_assets_ttm,
            return_on_equity_ttm=return_on_equity_ttm,
            revenue_ttm=revenue_ttm,
            gross_profit_ttm=gross_profit_ttm,
            diluted_eps_ttm=diluted_eps_ttm,
            quarterly_earnings_growth_yoy=quarterly_earnings_growth_yoy,
            quarterly_revenue_growth_yoy=quarterly_revenue_growth_yoy,
            analyst_target_price=analyst_target_price,
            trailing_pe=trailing_pe,
            forward_pe=forward_pe,
            price_to_sales_ratio=price_to_sales_ratio,
            price_to_book_ratio=price_to_book_ratio,
            ev_to_revenue=ev_to_revenue,
            ev_to_ebitda=ev_to_ebitda,
            beta=beta,
            fifty_two_week_high=fifty_two_week_high,
            fifty_two_week_low=fifty_two_week_low,
            fifty_day_moving_average=fifty_day_moving_average,
            two_hundred_day_moving_average=two_hundred_day_moving_average,
            shares_outstanding=shares_outstanding,
            dividend_date=dividend_date,
            ex_dividend_date=ex_dividend_date,

        )

    @classmethod
    def from_json_file(cls, path: Path):
        return cls.from_dict(cls.load_json_dict(path))  # type:ignore
