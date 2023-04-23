

from pathlib import Path


class StocksInUse:
    def __init__(self):
        self.stock_tickers: list[str]
        self.data_cache = None
        self.stocks = None

    def cache_all(self, ):
        pass

    def load_stock(self, ticker: str):
        pass

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
