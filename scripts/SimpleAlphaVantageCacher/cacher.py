

from pathlib import Path
import time
import pandas as pd
import numpy as np
import os
import sys  # nopep8
sys.path.insert(
    0, "/Users/James/Library/Mobile Documents/com~apple~CloudDocs/James's Files/Education/Syracuse/Semesters/10. April 2023/CIS 600/Paper")
from value_investing_strategy.strategy_system.stocks.alpha_vantage.alpha_vantage_client import AlphaVantageClient  # nopep8

CACHE_PATH = Path("./scripts/SimpleAlphaVantageCacher/output/json_cache")
DATA_CACHE_PATH = Path(
    "./scripts/SimpleAlphaVantageCacher/output/json_cache/DATA")
PDB = Path("./scripts/SimpleAlphaVantageCacher/pdb")
PDB.mkdir(parents=True, exist_ok=True)
COUNT_FILE_PATH = PDB/"count.txt"
CACHE_PATH.mkdir(parents=True, exist_ok=True)

DATA_CACHE_PATH.mkdir(parents=True, exist_ok=True)
COVERED_LIST_PATH = CACHE_PATH/"covered.txt"
COVERED_LIST_PATH.touch(exist_ok=True)
COUNT_FILE_PATH.touch(exist_ok=True)


class CountFile:

    def __init__(self, file_path: Path = COUNT_FILE_PATH):
        self.file_path = file_path
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
    def __init__(self, count_file: CountFile):
        self.count_file = count_file

    def count_and_wait(self):
        self.count_file += 1
        if self.count_file % 5 == 0:
            print(
                f"count = {self.count_file.count}...  waiting for one minute")
            time.sleep(60+1)
        return self.count_file

    @staticmethod
    def get_s_and_p_list() -> list[str]:
        sp500 = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_list = np.array(sp500[0]['Symbol'])
        return list(sp500_list)

    def component_file_exists(self, ticker: str, component_str: str):
        name = f"{ticker}.{component_str}.json"
        files = os.listdir(DATA_CACHE_PATH)
        if name in files:
            return True
        return False

    def get_covered_list(self):
        with open(COVERED_LIST_PATH, "r") as fp:
            covered = [item.strip() for item in fp.readlines()]

        return covered

    def add_ticker_to_covered_list(self, ticker: str):
        with open(COVERED_LIST_PATH, "a") as fp:
            return fp.write(f"{ticker}\n")

    def option_1(self, ticker: str):
        calls = [AlphaVantageClient.BalanceSheet.to_json_file,
                 AlphaVantageClient.IncomeStatement.to_json_file,
                 AlphaVantageClient.CompanyOverview.to_json_file,
                 AlphaVantageClient.Earnings.to_json_file,
                 AlphaVantageClient.CashFlow.to_json_file,
                 AlphaVantageClient.TimeSeriesMonthly.to_json_file]

        if ticker not in self.get_covered_list():
            for func in calls:
                func(ticker, DATA_CACHE_PATH)
                self.count_and_wait()

            self.add_ticker_to_covered_list(ticker)
            print(f"{ticker} retrieved - API count at: {self.count_file}")

    def option_2(self, ticker: str):
        if ticker in self.get_covered_list() and not self.component_file_exists(ticker, AlphaVantageClient.TimeSeriesMonthly.TYPE_STR):

            AlphaVantageClient.TimeSeriesMonthly.to_json_file(
                ticker, DATA_CACHE_PATH)
            print(f"time series data collected for {ticker}")
            self.count_and_wait()

    def retrieve_data_for_ticker(self, ticker: str, option: int):
        if option == 1:
            self.option_1(ticker)
        elif option == 2:
            self.option_2(ticker)

    def process_stock_list(self, option: int):
        stock_list = self.get_s_and_p_list()
        for ticker in stock_list:
            self.retrieve_data_for_ticker(ticker, option)
            if self.count_file.count == 500:
                print("done.")
                break


def main(option: int):
    count_file = CountFile()
    stock_data_retriever = StockDataRetriever(count_file)
    stock_data_retriever.process_stock_list(option)


if __name__ == "__main__":
    print(Path.cwd())
    main(2)
    main(1)
