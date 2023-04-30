import os

from pathlib import Path
from value_investing_strategy.strategy_system.strategy_interface import StrategyInterface
from value_investing_strategy.strategy_system.stocks.stock.stock import Stock, PATH_TO_STOCK_DATA
from value_investing_strategy.strategy_system.stocks.stocks_in_use import StocksInUse


def get_data_path():
    module_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(module_dir)
    result = Path(os.path.join(
        parent_dir, "data/SimpleAlphaVantageCacher/output/json_cache/covered.txt"))
    return result


class StrategySystemError(Exception):
    pass


class StrategySystem(StrategyInterface):
    def __init__(self):
        self.Attribute1 = None

    def get_QFAStrategy(self, uid: str):
        pass

    def get_RebalancingStrategy(self, uid: str):
        pass

    def run(self, ):
        pass

    def initialize(self, ):
        pass

    def reset(self, ):
        pass

    def calculate_book_value_dividend_projections_valuation(self, ticker: str):
        pass

    def compare_qfa(self, tickers: list[str]):
        pass

    def calculate_graham_number(self, ticker):
        pass

    def calculate_discounted_cash_flow_valuation(self, ticker):
        pass

    def calculate_asset_based_valuation(self, ticker):
        pass

    def calculator_financial_metric_analysis(self, ticker):
        pass

    def calculate_financial_metric_analysis_valuation(self, ticker):
        pass

    def generate_qf_analysis(self, ):
        pass

    def generate_rebalancing_analysis(self, ):
        pass

    def add_rebalancing_dataflow(self, analyzer):
        pass

    def add_qfa_dataflow(self, analyzer):
        pass

    def run_rebalance(self, uid):
        pass

    def run_qfa(self, uid):
        pass

    def get_settings(self, ):
        pass

    def compare_stocks(self, tickers):
        pass

    @staticmethod
    def raise_error_if_ticker_not_avaialable(ticker: str):
        if ticker not in StrategySystem.get_available_stocks():
            raise StrategySystemError("Ticker is not avaialable")

    @staticmethod
    def get_available_stocks() -> list[str]:

        try:
            return StocksInUse.list_cached_tickers(
                PATH_TO_STOCK_DATA.parent/"covered.txt")
        except FileNotFoundError:
            return StocksInUse.list_cached_tickers(get_data_path())

    @staticmethod
    def get_stock(ticker: str):
        StrategySystem.raise_error_if_ticker_not_avaialable(ticker)

        return Stock.from_alpha_vantage_data(ticker)
