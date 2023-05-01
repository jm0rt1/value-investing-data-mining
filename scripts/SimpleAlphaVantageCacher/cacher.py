

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
CACHE_PATH.mkdir(parents=True, exist_ok=True)
DATA_CACHE_PATH.mkdir(parents=True, exist_ok=True)
COVERED_LIST_PATH = CACHE_PATH/"covered.txt"
COVERED_LIST_PATH.touch(exist_ok=True)


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


def option_1(ticker: str, count: int):
    if ticker not in get_covered_list():
        AlphaVantageClient.BalanceSheet.to_json_file(
            ticker, DATA_CACHE_PATH)
        AlphaVantageClient.IncomeStatement.to_json_file(
            ticker, DATA_CACHE_PATH)
        AlphaVantageClient.CompanyOverview.to_json_file(
            ticker, DATA_CACHE_PATH)
        AlphaVantageClient.Earnings.to_json_file(ticker, DATA_CACHE_PATH)
        AlphaVantageClient.CashFlow.to_json_file(ticker, DATA_CACHE_PATH)
        AlphaVantageClient.TimeSeriesMonthly.to_json_file(
            ticker, DATA_CACHE_PATH)
        add_ticker_to_covered_list(ticker)
        print(f"{ticker} retrieved - API count at: {count}")
        time.sleep(60+1)
        count += 5
    return count


def option_2(ticker: str, count: int):
    if ticker in get_covered_list() and not component_file_exists(ticker, AlphaVantageClient.TimeSeriesMonthly.TYPE_STR):

        AlphaVantageClient.TimeSeriesMonthly.to_json_file(
            ticker, DATA_CACHE_PATH)
        time.sleep(60+1)
        count += 1
        pass

    return count


def main(option: int):
    list_ = get_s_and_p_list()
    count: int = 0
    for ticker in list_:
        if option == 1:
            count = option_1(ticker, count)
        elif option == 2:
            count = option_2(ticker, count)

        if count == 480:
            print("done.")
            break


if __name__ == "__main__":
    print(Path.cwd())
    main(2)
