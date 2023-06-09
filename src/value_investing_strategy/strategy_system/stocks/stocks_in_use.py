

from pathlib import Path
from typing import Optional

import pandas as pd
from src.value_investing_strategy.strategy_system.stocks.stock.stock import Stock
import tqdm


class StocksInUse:
    def __init__(self):
        self.stock_tickers: list[str]
        self.data_cache = None
        self.stocks: Optional[list[Stock]] = None

    def cache_all(self, ):
        pass

    def load_stock(self, ticker: str):
        pass

    def load_stocks(self, tickers: list[str]):
        self.stocks = []

        for ticker in tqdm.tqdm(tickers):
            self.stocks.append(Stock.from_alpha_vantage_data(ticker))

    def to_data_frame(self):
        if self.stocks is None:
            raise TypeError("Stocks is unitialized")
        try:
            stock_dataframes = []
            for stock in tqdm.tqdm(self.stocks):
                try:
                    stock_df = stock.to_dataframe()
                except Exception as e:
                    raise e
                stock_dataframes.append(stock_df)
            combined_data = pd.concat(  # type:ignore
                stock_dataframes, ignore_index=True)  # type:ignore
        except Exception as e:
            raise e
        return combined_data

    def is_stock_loaded(self, ticker: str):
        pass

    def get_stock(self, ticker: str):
        pass

    @staticmethod
    def list_cached_tickers(cache_file: Path) -> list[str]:
        with open(cache_file, "r") as fp:
            lines = fp.readlines()
            lines: list[str] = [line.strip() for line in lines]
        return lines
