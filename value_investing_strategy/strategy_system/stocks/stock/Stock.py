import os

from dataclasses import dataclass
import math
from pathlib import Path
from value_investing_strategy.strategy_system.stocks.stock.components.income_statement import IncomeStatement
from value_investing_strategy.strategy_system.stocks.stock.components.earnings import EarningsStatement
from value_investing_strategy.strategy_system.stocks.stock.components.balance_sheet import BalanceSheet
from value_investing_strategy.strategy_system.stocks.stock.components.cash_flow import Cashflow
from value_investing_strategy.strategy_system.stocks.stock.components.company_overview import CompanyOverview
import pandas as pd

from value_investing_strategy.strategy_system.stocks.stock.components.time_series import TimeSeriesMonthly


def get_valid_path():

    default_path = "./value_investing_strategy/data/SimpleAlphaVantageCacher/output/json_cache/DATA"

    if os.path.exists(default_path):
        return Path(default_path)
    else:
        module_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(module_dir)
        grandparent_dir = os.path.dirname(parent_dir)
        great_grandparent_dir = os.path.dirname(grandparent_dir)
        return Path(os.path.join(great_grandparent_dir, "data/SimpleAlphaVantageCacher/output/json_cache/DATA"))


PATH_TO_STOCK_DATA = get_valid_path()


@dataclass
class Stock:

    ticker: str
    income_statement: IncomeStatement
    earnings: EarningsStatement
    balance_sheet: BalanceSheet
    cash_flow: Cashflow
    company_overview: CompanyOverview
    time_series_monthly: TimeSeriesMonthly
    last_updated = None
    symbol = None
    intrinsic_value = None

    @classmethod
    def from_ticker(cls, ticker: str):
        pass

    @classmethod
    def from_alpha_vantage_data(cls, ticker: str, data_base_path: Path = PATH_TO_STOCK_DATA):
        paths = DataPaths.build_paths(ticker, data_base_path)
        income_statement: IncomeStatement = IncomeStatement.from_json_file(
            paths.income_statement_file_path)
        earnings: EarningsStatement = EarningsStatement.from_json_file(
            paths.earnings_file_path)
        balance_sheet: BalanceSheet = BalanceSheet.from_json_file(
            paths.balance_sheet_file_path)
        cash_flow: Cashflow = Cashflow.from_json_file(
            paths.cash_flow_file_path)
        company_overview: CompanyOverview = CompanyOverview.from_json_file(
            paths.company_overview_file_path)
        time_series_monthly: TimeSeriesMonthly = TimeSeriesMonthly.from_json_file(
            paths.time_series_monthly_file_path)

        return cls(ticker, income_statement, earnings,
                   balance_sheet, cash_flow, company_overview, time_series_monthly)

    @property
    def graham_number(self):
        return GrahamNumberCalculator(self).run()

    def to_dataframe(self):
        # Extracting relevant data from financial statements
        annual_reports = self.income_statement.annual_reports
        cf_annual_reports = self.cash_flow.annual_reports
        bs_annual_reports = self.balance_sheet.annual_reports

        # Initializing empty lists for each column in the DataFrame
        tickers = []
        fiscal_years = []
        cash_flows = []
        book_values = []
        earnings = []
        five_year_return = 0

        # Populating lists with data from financial statements
        for i, report in enumerate(annual_reports):
            fiscal_year = report.fiscal_date_ending

            # Find matching cash flow report
            cf_report = next(
                (r for r in cf_annual_reports if r.fiscal_date_ending == fiscal_year), None)
            if cf_report is None:
                continue

            # Find matching balance sheet report
            bs_report = next(
                (r for r in bs_annual_reports if r.fiscal_date_ending == fiscal_year), None)
            if bs_report is None:
                continue

            tickers.append(self.ticker)
            fiscal_years.append(fiscal_year)
            earnings.append(report.net_income)
            cash_flows.append(cf_report.operating_cashflow)
            book_values.append(bs_report.total_shareholder_equity)

        if len(fiscal_years) > 0:
            five_year_prior = pd.to_datetime(
                fiscal_years[0]) - pd.DateOffset(years=5)
            five_year_return = self.time_series_monthly.calculate_stock_returns(
                five_year_prior, pd.to_datetime(fiscal_years[0]))
        else:
            five_year_return = None

        if five_year_return is None:
            five_year_return = 0
        # Creating a DataFrame using the populated lists
        data = {
            'Ticker': tickers,
            'FiscalYear': fiscal_years,
            'CashFlow': cash_flows,
            'BookValue': book_values,
            'Earnings': earnings,
            '5YrReturn%': [five_year_return * 100] * len(tickers)
        }
        try:
            return pd.DataFrame(data)
        except Exception as e:
            raise e


@dataclass
class DataPaths():
    income_statement_file_path: Path
    earnings_file_path: Path
    balance_sheet_file_path: Path
    cash_flow_file_path: Path
    company_overview_file_path: Path
    time_series_monthly_file_path: Path

    @classmethod
    def build_paths(cls, ticker: str, base_path: Path):
        income_statement_file_name = f"{ticker}.IncomeStatement.json"
        earnings_file_name = f"{ticker}.Earnings.json"
        balance_sheet_file_name = f"{ticker}.BalanceSheet.json"
        cash_flow_file_name = f"{ticker}.CashFlow.json"
        company_overview_file_name = f"{ticker}.CompanyOverview.json"
        time_series_monthly_file_name = f"{ticker}.TimeSeriesMonthly.json"

        income_statement_file_path = base_path/income_statement_file_name
        earnings_file_path = base_path/earnings_file_name
        balance_sheet_file_path = base_path/balance_sheet_file_name
        cash_flow_file_path = base_path/cash_flow_file_name
        company_overview_file_path = base_path/company_overview_file_name
        time_series_monthly_file_path = base_path/time_series_monthly_file_name

        if not income_statement_file_path.exists():
            raise FileNotFoundError(
                f"{income_statement_file_path} does not exist, and cannot be found")
        if not earnings_file_path.exists():
            raise FileNotFoundError(
                f"{earnings_file_path} does not exist, and cannot be found")
        if not balance_sheet_file_path.exists():
            raise FileNotFoundError(
                f"{balance_sheet_file_path} does not exist, and cannot be found")
        if not cash_flow_file_path.exists():
            raise FileNotFoundError(
                f"{cash_flow_file_path} does not exist, and cannot be found")
        if not company_overview_file_path.exists():
            raise FileNotFoundError(
                f"{company_overview_file_path} does not exist, and cannot be found")
        if not time_series_monthly_file_path.exists():
            raise FileNotFoundError(
                f"{time_series_monthly_file_path} does not exist, and cannot be found")
        return cls(
            income_statement_file_path=income_statement_file_path,
            earnings_file_path=earnings_file_path,
            balance_sheet_file_path=balance_sheet_file_path,
            cash_flow_file_path=cash_flow_file_path,
            company_overview_file_path=company_overview_file_path,
            time_series_monthly_file_path=time_series_monthly_file_path
        )


class Calculator:
    def __init__(self, stock: Stock, stock_data: Stock):
        self.stock: Stock = stock_ticker
        self.stock_data = None

    def run(self, ):
        pass


class AssetBasedValuationCalculator(Calculator):

    def __init__(self, stock: Stock):
        pass

    def run(self, ):
        pass


class BookValueDividendProjectionsValuationCalculator(Calculator):

    def __init__(self, stock: Stock):
        pass

    def run(self, ):
        pass


class DiscountedCashFlowValuationCalculator(Calculator):

    def __init__(self, stock: Stock):
        pass

    def run(self, ):
        pass


class FinancialMetricAnalysisCalculator(Calculator):
    def __init__(self, stock: Stock):
        pass

    def run(self, ):
        pass


class GrahamNumberCalculator():

    def __init__(self, stock: Stock):
        self.stock = stock

    def run(self, ) -> float:

        return self.calculate(self.stock.company_overview.eps, self.stock.company_overview.book_value)

    @staticmethod
    def calculate(eps: float, book_value_per_share: float):
        multiplier = 22.5  # Graham's multiplier

        # Calculate the Graham Number
        graham_number = round(
            math.sqrt(multiplier * eps * book_value_per_share), 2)
        return graham_number


class SharpeRatioCalculator(Calculator):
    """
    The Sharpe ratio is a financial metric that measures the excess return of an investment per unit of its risk. It was developed by Nobel laureate William F. Sharpe.

    The Sharpe ratio is calculated by subtracting the risk-free rate of return (such as the yield on a government bond) from the average rate of return of the investment, and dividing the result by the standard deviation of the investment's returns. Mathematically, it can be represented as:

    Sharpe ratio = (Rp - Rf) / σp

    where:
    Rp is the average rate of return of the investment
    Rf is the risk-free rate of return
    σp is the standard deviation of the investment's returns

    A higher Sharpe ratio indicates that the investment has generated a higher return for the amount of risk it has taken on. In general, a Sharpe ratio of 1 or higher is considered good, while a Sharpe ratio of less than 1 may indicate that the investment is not generating enough return for the amount of risk it is taking on.

    It's worth noting that the Sharpe ratio has some limitations and assumptions, including the assumption that investment returns follow a normal distribution, and the fact that it only measures risk relative to the risk-free rate and does not account for other types of risk (such as market risk or liquidity risk). Nonetheless, it is a widely used and respected measure of risk-adjusted performance in finance.
    """

    def __init__(self):
        self.Rp = None
        self.Rf = None
        self.sigma_p = None

    def __init__(self, stock: Stock):
        pass

    def run(self, ):
        pass
