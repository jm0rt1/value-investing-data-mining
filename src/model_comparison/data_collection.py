

from pathlib import Path

from src.value_investing_strategy.strategy_system.stocks.stocks_in_use import StocksInUse


def main():
    cached = StocksInUse.list_cached_tickers(
        Path("src/value_investing_strategy/data/SimpleAlphaVantageCacher/output/json_cache/covered.txt"))
    stocks = StocksInUse()
    stocks.load_stocks(cached)

    df = stocks.to_data_frame()
    df.to_csv("out.csv")


if __name__ == "__main__":
    main()
