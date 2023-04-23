import unittest

from value_investing_strategy.strategy_system.stocks.stock.components.CompanyOverview import CompanyOverview
DATA = {
    "Symbol": "AAPL",
    "AssetType": "Common Stock",
    "Name": "Apple Inc",
    "Description": "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services. Apple is the world's largest technology company by revenue (totalling $274.5 billion in 2020) and, since January 2021, the world's most valuable company. As of 2021, Apple is the world's fourth-largest PC vendor by unit sales, and fourth-largest smartphone manufacturer. It is one of the Big Five American information technology companies, along with Amazon, Google, Microsoft, and Facebook.",
    "CIK": "320193",
    "Exchange": "NASDAQ",
    "Currency": "USD",
    "Country": "USA",
    "Sector": "TECHNOLOGY",
    "Industry": "ELECTRONIC COMPUTERS",
    "Address": "ONE INFINITE LOOP, CUPERTINO, CA, US",
    "FiscalYearEnd": "September",
    "LatestQuarter": "2022-12-31",
    "MarketCapitalization": "2340375495000",
    "EBITDA": "125287997000",
    "PERatio": "25.11",
    "PEGRatio": "2.75",
    "BookValue": "3.581",
    "DividendPerShare": "0.91",
    "DividendYield": "0.0062",
    "EPS": "5.89",
    "RevenuePerShareTTM": "24.08",
    "ProfitMargin": "0.246",
    "OperatingMarginTTM": "0.294",
    "ReturnOnAssetsTTM": "0.196",
    "ReturnOnEquityTTM": "1.479",
    "RevenueTTM": "387537011000",
    "GrossProfitTTM": "170782000000",
    "DilutedEPSTTM": "5.89",
    "QuarterlyEarningsGrowthYOY": "-0.105",
    "QuarterlyRevenueGrowthYOY": "-0.055",
    "AnalystTargetPrice": "168.21",
    "TrailingPE": "25.11",
    "ForwardPE": "23.09",
    "PriceToSalesRatioTTM": "5.51",
    "PriceToBookRatio": "44.63",
    "EVToRevenue": "5.92",
    "EVToEBITDA": "17.53",
    "Beta": "1.278",
    "52WeekHigh": "178.53",
    "52WeekLow": "123.98",
    "50DayMovingAverage": "140.56",
    "200DayMovingAverage": "147.2",
    "SharesOutstanding": "15821900000",
    "DividendDate": "2023-02-16",
    "ExDividendDate": "2023-02-10"
}


class TestCompanyOverview(unittest.TestCase):

    def test_load_from_dict(self):
        company_overview = CompanyOverview.from_dict(DATA)

        self.assertIsInstance(company_overview, CompanyOverview)
        self.assertEqual(company_overview.symbol, "AAPL")
        self.assertEqual(company_overview.asset_type, "Common Stock")
        self.assertEqual(company_overview.name, "Apple Inc")
        self.assertEqual(company_overview.description, "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services. Apple is the world's largest technology company by revenue (totalling $274.5 billion in 2020) and, since January 2021, the world's most valuable company. As of 2021, Apple is the world's fourth-largest PC vendor by unit sales, and fourth-largest smartphone manufacturer. It is one of the Big Five American information technology companies, along with Amazon, Google, Microsoft, and Facebook.")
        self.assertEqual(company_overview.cik, 320193)
        self.assertEqual(company_overview.exchange, "NASDAQ")
        self.assertEqual(company_overview.currency, "USD")
        self.assertEqual(company_overview.country, "USA")
        self.assertEqual(company_overview.sector, "TECHNOLOGY")
        self.assertEqual(company_overview.industry, "ELECTRONIC COMPUTERS")
        self.assertEqual(company_overview.address,
                         "ONE INFINITE LOOP, CUPERTINO, CA, US")
        self.assertEqual(company_overview.fiscal_year_end, "September")
        self.assertEqual(company_overview.latest_quarter, "2022-12-31")
        self.assertEqual(company_overview.market_capitalization, 2340375495000)
        self.assertEqual(company_overview.ebitda, 125287997000)
        self.assertEqual(company_overview.pe_ratio, 25.11)
        self.assertEqual(company_overview.peg_ratio, 2.75)
        self.assertEqual(company_overview.book_value, 3.581)
        self.assertEqual(company_overview.dividend_per_share, 0.91)
        self.assertEqual(company_overview.dividend_yield, 0.0062)
        self.assertEqual(company_overview.eps, 5.89)
        self.assertEqual(company_overview.revenue_per_share_ttm, 24.08)
        self.assertEqual(company_overview.profit_margin, 0.246)
        self.assertEqual(company_overview.operating_margin_ttm, 0.294)
        self.assertEqual(company_overview.return_on_assets_ttm, 0.196)
        self.assertEqual(company_overview.return_on_equity_ttm, 1.479)
        self.assertEqual(company_overview.revenue_ttm, 387537011000)
        self.assertEqual(company_overview.gross_profit_ttm, 170782000000)
        self.assertEqual(company_overview.diluted_eps_ttm, 5.89)
        self.assertEqual(
            company_overview.quarterly_earnings_growth_yoy, -0.105)
        self.assertEqual(company_overview.quarterly_revenue_growth_yoy, -0.055)
        self.assertEqual(company_overview.analyst_target_price, 168.21)
        self.assertEqual(company_overview.trailing_pe, 25.11)
        self.assertEqual(company_overview.forward_pe, 23.09)
        self.assertEqual(company_overview.price_to_sales_ratio, 5.51)
        self.assertEqual(company_overview.price_to_book_ratio, 44.63)
        self.assertEqual(company_overview.ev_to_revenue, 5.92)
        self.assertEqual(company_overview.ev_to_ebitda, 17.53)
        self.assertEqual(company_overview.beta, 1.278)
        self.assertEqual(company_overview.fifty_two_week_high, 178.53)
        self.assertEqual(company_overview.fifty_two_week_low, 123.98)
        self.assertEqual(company_overview.fifty_day_moving_average, 140.56)
        self.assertEqual(
            company_overview.two_hundred_day_moving_average, 147.2)
        self.assertEqual(company_overview.shares_outstanding, 15821900000)
        self.assertEqual(company_overview.dividend_date, "2023-02-16")
        self.assertEqual(company_overview.ex_dividend_date, "2023-02-10")
