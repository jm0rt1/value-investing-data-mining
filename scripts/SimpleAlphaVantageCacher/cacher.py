

import logging
from pathlib import Path
import time
import pandas as pd
import numpy as np
import tomli

import os
import sys  # nopep8
sys.path.insert(
    0, "/Users/James/Library/Mobile Documents/com~apple~CloudDocs/James's Files/Education/Syracuse/Semesters/10. April 2023/CIS 600/Paper")
from value_investing_strategy.strategy_system.stocks.alpha_vantage.alpha_vantage_client import AlphaVantageClient  # nopep8


class Config:
    def __init__(self, config_path: str):
        self._config = self.load_config(config_path)

    @staticmethod
    def load_config(config_path: str) -> dict:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        return config

    def __getitem__(self, key: str):
        return self._config[key]


class CountFile:
    def __init__(self, config: Config):
        self.file_path = Path(config['cache']['count_file'])
        self.count = self.read()

    def write(self, count: int):
        with open(self.file_path, "w") as fp:
            fp.write(str(count))

    def read(self):
        with open(self.file_path, "r") as fp:
            count_str = fp.read()
            if count_str == "":
                count = 0
            else:
                count = int(count_str)
        return count

    def verify(self):
        saved_count = self.read()
        return self.count == saved_count

    def __add__(self, value: int):
        self.count += value
        self.write(self.count)
        return self.count

    def __iadd__(self, value: int):
        self.count += value
        self.write(self.count)
        return self

    def __mod__(self, value: int):
        result = self.count % value
        return result

    def reset(self):
        self.count = 0
        self.write(self.count)


class StockDataRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.data_cache_path = Path(config['cache']['data_cache_path'])
        self.covered_list_path = Path(config['cache']['covered_list_file'])

    def get_s_and_p_list(self) -> list[str]:
        sp500 = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_list = np.array(sp500[0]['Symbol'])
        return list(sp500_list)

    def component_file_exists(self, ticker: str, component_str: str):
        name = f"{ticker}.{component_str}.json"
        files = os.listdir(self.data_cache_path)
        if name in files:
            return True
        return False

    def get_covered_list(self):
        with open(self.covered_list_path, "r") as fp:
            covered = [item.strip() for item in fp.readlines()]
        return covered

    def add_ticker_to_covered_list(self, ticker: str):
        with open(self.covered_list_path, "a") as fp:
            return fp.write(f"{ticker}\n")

    def option_1(self, ticker: str, count: CountFile):
        calls = [
            AlphaVantageClient.BalanceSheet.to_json_file,
            AlphaVantageClient.IncomeStatement.to_json_file,
            AlphaVantageClient.CompanyOverview.to_json_file,
            AlphaVantageClient.Earnings.to_json_file,
            AlphaVantageClient.CashFlow.to_json_file,
            AlphaVantageClient.TimeSeriesMonthly.to_json_file
        ]

        if ticker not in self.get_covered_list():
            for func in calls:
                func(ticker, self.data_cache_path)
                count = count_and_wait(count)

            self.add_ticker_to_covered_list(ticker)
            logging.info(f"{ticker} retrieved - API count at: {count.count}")

    def option_2(self, ticker: str, count: CountFile):
        if ticker in self.get_covered_list() and not self.component_file_exists(ticker, AlphaVantageClient.TimeSeriesMonthly.TYPE_STR):
            AlphaVantageClient.TimeSeriesMonthly.to_json_file(
                ticker, self.data_cache_path)
            logging.info(f"time series data collected for {ticker}")
            count = count_and_wait(count)

    def main(self, option: int):
        list_ = self.get_s_and_p_list()
        count_file = CountFile(self.config)
        for ticker in list_:
            if option == 1:
                self.option_1(ticker, count_file)
            elif option == 2:
                self.option_2(ticker, count_file)

            if count_file.count == 500:
                logging.info("done.")
                break


def count_and_wait(count: CountFile):
    count += 1
    if count % 5 == 0:
        logging.info(f"count = {count.count}...  waiting for one minute")
        time.sleep(60 + 1)
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config_file_path = "scripts/SimpleAlphaVantageCacher/config/cacher_config.toml"
    config = Config(config_file_path)

    stock_data_retriever = StockDataRetriever(config)
    stock_data_retriever.main(2)
    stock_data_retriever.main(1)
