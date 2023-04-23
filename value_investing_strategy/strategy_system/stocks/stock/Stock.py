import os

from dataclasses import dataclass
import math
from pathlib import Path
from value_investing_strategy.strategy_system.stocks.stock.components.IncomeStatement import IncomeStatement
from value_investing_strategy.strategy_system.stocks.stock.components.Earnings import EarningsStatement
from value_investing_strategy.strategy_system.stocks.stock.components.BalanceSheet import BalanceSheet
from value_investing_strategy.strategy_system.stocks.stock.components.CashFlow import Cashflow
from value_investing_strategy.strategy_system.stocks.stock.components.CompanyOverview import CompanyOverview


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

        return cls(ticker, income_statement, earnings,
                   balance_sheet, cash_flow, company_overview)

    @property
    def graham_number(self):
        return GrahamNumberCalculator(self).run()


@dataclass
class DataPaths():
    income_statement_file_path: Path
    earnings_file_path: Path
    balance_sheet_file_path: Path
    cash_flow_file_path: Path
    company_overview_file_path: Path

    @classmethod
    def build_paths(cls, ticker: str, base_path: Path):
        income_statement_file_name = f"{ticker}.IncomeStatement.json"
        earnings_file_name = f"{ticker}.Earnings.json"
        balance_sheet_file_name = f"{ticker}.BalanceSheet.json"
        cash_flow_file_name = f"{ticker}.CashFlow.json"
        company_overview_file_name = f"{ticker}.CompanyOverview.json"

        income_statement_file_path = base_path/income_statement_file_name
        earnings_file_path = base_path/earnings_file_name
        balance_sheet_file_path = base_path/balance_sheet_file_name
        cash_flow_file_path = base_path/cash_flow_file_name
        company_overview_file_path = base_path/company_overview_file_name

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

        return cls(
            income_statement_file_path=income_statement_file_path,
            earnings_file_path=earnings_file_path,
            balance_sheet_file_path=balance_sheet_file_path,
            cash_flow_file_path=cash_flow_file_path,
            company_overview_file_path=company_overview_file_path
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
