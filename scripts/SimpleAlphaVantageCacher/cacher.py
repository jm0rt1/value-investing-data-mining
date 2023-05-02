

import logging
from pathlib import Path
import time
from typing import Optional
import pandas as pd
import numpy as np
import tomli
import datetime
import os
import sys  # nopep8
sys.path.insert(
    0, "/Users/James/Library/Mobile Documents/com~apple~CloudDocs/James's Files/Education/Syracuse/Semesters/10. April 2023/CIS 600/Paper")
from value_investing_strategy.strategy_system.stocks.alpha_vantage.alpha_vantage_client import AlphaVantageClient  # nopep8


class Config:
    def __init__(self, config_path: str = 'config.toml'):

        with open(config_path, 'rb') as f:
            config = tomli.load(f)

        self.api_key = config['api']['api_key']

        self.daily_api_limit = config['limits']['daily_api_limit']

        self.cache_path = Path(config['cache']['cache_path'])
        self.data_cache_path = Path(config['cache']['data_cache_path'])
        self.pdb_path = Path(config['cache']['pdb_path'])
        self.covered_list_file = Path(config['cache']['covered_list_file'])
        self.count_file = Path(config['cache']['count_file'])

        self.wait_time = config['time']['wait_time']

        self.create_directories()

    def create_directories(self):
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.data_cache_path.mkdir(parents=True, exist_ok=True)
        self.pdb_path.mkdir(parents=True, exist_ok=True)
        self.count_file.touch(exist_ok=True)
        self.covered_list_file.touch(exist_ok=True)


class CountFile:
    def __init__(self, config: Config):
        self.file_path = Path(config.count_file)
        self.count, self.last_reset = self.read()

    def read(self):
        with open(self.file_path, "r") as fp:
            lines = fp.readlines()
            if not lines:
                count = 0
                last_reset = datetime.datetime.now()
            else:
                count = int(lines[0].strip())
                last_reset = datetime.datetime.fromisoformat(lines[1].strip())
        return count, last_reset

    def write(self, count: int, last_reset: Optional[datetime.datetime] = None):
        if last_reset is None:
            last_reset = self.last_reset
        with open(self.file_path, "w") as fp:
            fp.write(f"{count}\n{last_reset.isoformat()}")

    def verify(self):
        saved_count, _ = self.read()
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
        self.last_reset = datetime.datetime.now()
        self.write(self.count, self.last_reset)


class SAndPList:
    @staticmethod
    def get() -> list[str]:
        sp500 = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_list = np.array(sp500[0]['Symbol'])
        return list(sp500_list)


class DataCollector:
    def __init__(self, config: Config):
        self.config = config
        self.alpha_vantage_client = AlphaVantageClient()
        self.covered_list = CoveredList(config)
        self.s_and_p_list = SAndPList()

    def option_1(self, ticker: str, count: CountFile):
        calls = [
            self.alpha_vantage_client.BalanceSheet.to_json_file,
            self.alpha_vantage_client.IncomeStatement.to_json_file,
            self.alpha_vantage_client.CompanyOverview.to_json_file,
            self.alpha_vantage_client.Earnings.to_json_file,
            self.alpha_vantage_client.CashFlow.to_json_file,
            self.alpha_vantage_client.TimeSeriesMonthly.to_json_file
        ]

        if ticker not in self.covered_list.get():
            for func in calls:
                func(ticker, self.config.data_cache_path)
                count = count_and_wait(count)

            self.covered_list.add(ticker)
            logging.info(f"{ticker} retrieved - API count at: {count.count}")

    def option_2(self, ticker: str, count: CountFile):
        if ticker in self.covered_list.get() and not self.alpha_vantage_client.component_file_exists(ticker, self.config.data_cache_path, AlphaVantageClient.TimeSeriesMonthly.TYPE_STR):
            self.alpha_vantage_client.TimeSeriesMonthly.to_json_file(
                ticker, self.config.data_cache_path)
            logging.info(f"time series data collected for {ticker}")
            count = count_and_wait(count)


class CoveredList:
    def __init__(self, config: Config):
        self.config = config
        self.covered_list_path = config.covered_list_file
        self.covered_list_path.touch(exist_ok=True)

    def get(self) -> list[str]:
        with open(self.covered_list_path, "r") as fp:
            covered = [item.strip() for item in fp.readlines()]
        return covered

    def add(self, ticker: str):
        with open(self.covered_list_path, "a") as fp:
            fp.write(f"{ticker}\n")


class StockDataRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.data_collector = DataCollector(config)

    def main(self, option: int):
        list_ = self.data_collector.s_and_p_list.get()
        count_file = CountFile(self.config)
        for ticker in list_:
            if option == 1:
                self.data_collector.option_1(ticker, count_file)
            elif option == 2:
                self.data_collector.option_2(ticker, count_file)

            if count_file.count == 500:
                logging.info("done.")
                break


def count_and_wait(count: CountFile):
    count += 1
    current_date = datetime.datetime.now().date()
    last_reset_date = count.last_reset.date()
    if current_date > last_reset_date:
        count.reset()
        logging.info(f"Counter reset due to crossing 12 AM")

    if count.count % 5 == 0:
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
