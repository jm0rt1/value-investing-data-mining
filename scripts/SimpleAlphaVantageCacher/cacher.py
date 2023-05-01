

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


def count_and_wait(count: CountFile):
    count += 1
    if count % 5 == 0:
        print(f"count = {count.count}...  waiting for one minute")
        time.sleep(60+1)
    return count


def get_s_and_p_list() -> list[str]:
    sp500 = pd.read_html(  # type:ignore
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500_list = np.array(sp500[0]['Symbol'])  # type:ignore
    return list(sp500_list)  # type:ignore


def component_file_exists(ticker: str, component_str: str):
    name = f"{ticker}.{component_str}.json"
    files = os.listdir(DATA_CACHE_PATH)
    if name in files:
        return True
    return False


def get_covered_list():
    with open(COVERED_LIST_PATH, "r") as fp:
        covered = [item.strip() for item in fp.readlines()]

    return covered


def add_ticker_to_covered_list(ticker: str):
    with open(COVERED_LIST_PATH, "a") as fp:
        return fp.write(f"{ticker}\n")


def option_1(ticker: str, count: CountFile):

    calls = [AlphaVantageClient.BalanceSheet.to_json_file,
             AlphaVantageClient.IncomeStatement.to_json_file,
             AlphaVantageClient.CompanyOverview.to_json_file,
             AlphaVantageClient.Earnings.to_json_file,
             AlphaVantageClient.CashFlow.to_json_file,
             AlphaVantageClient.TimeSeriesMonthly.to_json_file]

    if ticker not in get_covered_list():
        for func in calls:
            func(ticker, DATA_CACHE_PATH)
            count = count_and_wait(count)

        add_ticker_to_covered_list(ticker)
        print(f"{ticker} retrieved - API count at: {count}")
    return count


def option_2(ticker: str, count: CountFile):
    if ticker in get_covered_list() and not component_file_exists(ticker, AlphaVantageClient.TimeSeriesMonthly.TYPE_STR):

        AlphaVantageClient.TimeSeriesMonthly.to_json_file(
            ticker, DATA_CACHE_PATH)
        print(f"time series data collected for {ticker}")
        count = count_and_wait(count)
        pass

    return count


def main(option: int):
    list_ = get_s_and_p_list()
    count_file: CountFile = CountFile()
    for ticker in list_:
        if option == 1:
            count_file = option_1(ticker, count_file)
        elif option == 2:
            count_file = option_2(ticker, count_file)

        if count_file.count == 500:
            print("done.")
            break


if __name__ == "__main__":
    print(Path.cwd())
    main(2)
    main(1)
