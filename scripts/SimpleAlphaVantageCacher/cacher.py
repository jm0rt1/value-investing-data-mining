

from pathlib import Path
import time
import pandas as pd
import numpy as np
import sys  # nopep8
sys.path.insert(
    0, "/Users/James/Library/Mobile Documents/com~apple~CloudDocs/James's Files/Education/Syracuse/Semesters/9. January 2023/CSE 682/Project")
from src.strategy_system.stocks.alpha_vantage.AlphaVantageClient import AlphaVantageClient  # nopep8

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


def get_covered_list():
    with open(COVERED_LIST_PATH, "r") as fp:
        covered = [item.strip() for item in fp.readlines()]

    return covered


def add_ticker_to_covered_list(ticker: str):
    with open(COVERED_LIST_PATH, "a") as fp:
        return fp.write(f"{ticker}\n")


def main():
    list_ = get_s_and_p_list()
    count = 0
    for ticker in list_:
        if ticker not in get_covered_list():

            AlphaVantageClient.BalanceSheet.to_json_file(
                ticker, DATA_CACHE_PATH)
            AlphaVantageClient.IncomeStatement.to_json_file(
                ticker, DATA_CACHE_PATH)
            AlphaVantageClient.CompanyOverview.to_json_file(
                ticker, DATA_CACHE_PATH)
            AlphaVantageClient.Earnings.to_json_file(ticker, DATA_CACHE_PATH)
            AlphaVantageClient.CashFlow.to_json_file(ticker, DATA_CACHE_PATH)
            add_ticker_to_covered_list(ticker)
            count += 5
            print(f"{ticker} retrieved - API count at: {count}")
            time.sleep(60+1)
        if count == 480:
            print("done.")
            break
    pass


if __name__ == "__main__":
    print(Path.cwd())
    main()
