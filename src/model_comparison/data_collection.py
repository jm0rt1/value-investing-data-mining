

from pathlib import Path

from src.value_investing_strategy.strategy_system.stocks.stocks_in_use import StocksInUse
from src.model_comparison.constants import Paths
import os
import shutil
import re


def main():
    cached = StocksInUse.list_cached_tickers(
        Path("src/value_investing_strategy/data/SimpleAlphaVantageCacher/output/json_cache/covered.txt"))
    stocks = StocksInUse()
    stocks.load_stocks(cached)

    df = stocks.to_data_frame()

    if Paths.INPUT_DATA.exists():
        rollover_input_data()

    df.to_csv(Paths.INPUT_DATA)


def rollover_input_data():
    """
    increments all of the files in the directory data/ in the form of input_data_#.csv, the most recent CSV file is in the form of input_data.csv, it needs to be incremented to input_data_1.csv
    """
    path = Path("data/")
    files = os.listdir(path)
    files = [f for f in files if re.match(r"input_data_\d+.csv", f)]
    files.sort()
    files = files[::-1]
    for i, file in enumerate(files):
        shutil.move(path / file, path / f"input_data_{i+1}.csv")
    shutil.move(path / "input_data.csv", path / "input_data_1.csv")


if __name__ == "__main__":
    main()
