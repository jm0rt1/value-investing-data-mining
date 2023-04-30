import json
from pathlib import Path
import requests
from typing import Optional

from value_investing_strategy.strategy_system.stocks.alpha_vantage.api import API  # nopep8

API_KEY = "UYOGYE4MI3DF16W2"


class AlphaVantageData():
    def __init__(self) -> None:
        pass

    @classmethod
    def load(cls, data: dict[str, str]):
        data_class = cls()
        data_class.__dict__ = data
        return data_class


class AlphaVantageClientError(Exception):
    pass


class AlphaVantageClient():

    class IncomeStatement(API):
        TYPE_STR = "IncomeStatement"
        FUNCTION_STR = "INCOME_STATEMENT"

        @staticmethod
        def print_json(symbol: str):
            data = request_data(
                symbol, AlphaVantageClient.IncomeStatement.FUNCTION_STR)
            print(data)

        @staticmethod
        def to_json_file(symbol: str, path: Path, data: Optional[dict[str, str]] = None):
            if not data:
                data = request_data(
                    symbol, AlphaVantageClient.IncomeStatement.FUNCTION_STR)
            save_component(symbol, path, data,
                           AlphaVantageClient.IncomeStatement.TYPE_STR)

        @staticmethod
        def from_local_file(symbol: str, cache_path: Path):
            pass

    class BalanceSheet(API):
        TYPE_STR = "BalanceSheet"
        FUNCTION_STR = "BALANCE_SHEET"

        @staticmethod
        def print_json(symbol: str):
            data = request_data(
                symbol, AlphaVantageClient.BalanceSheet.FUNCTION_STR)
            print(data)

        @staticmethod
        def to_json_file(symbol: str, path: Path, data: Optional[dict[str, str]] = None):
            if not data:

                data = request_data(
                    symbol, AlphaVantageClient.BalanceSheet.FUNCTION_STR)
            save_component(symbol, path, data,
                           AlphaVantageClient.BalanceSheet.TYPE_STR)

        @staticmethod
        def from_local_file(symbol: str, cache_path: Path):
            pass

    class CashFlow(API):

        TYPE_STR = "CashFlow"
        FUNCTION_STR = "CASH_FLOW"

        @staticmethod
        def print_json(symbol: str):
            data = request_data(
                symbol, AlphaVantageClient.CashFlow.FUNCTION_STR)
            print(data)

        @staticmethod
        def to_json_file(symbol: str, path: Path, data: Optional[dict[str, str]] = None):
            if not data:
                data = request_data(
                    symbol, AlphaVantageClient.CashFlow.FUNCTION_STR)
            save_component(symbol, path, data,
                           AlphaVantageClient.CashFlow.TYPE_STR)

        @staticmethod
        def from_local_file(symbol: str, cache_path: Path):
            pass

    class Earnings(API):

        TYPE_STR = "Earnings"
        FUNCTION_STR = "EARNINGS"

        @staticmethod
        def print_json(symbol: str):
            data = request_data(
                symbol, AlphaVantageClient.Earnings.FUNCTION_STR)
            print(data)

        @staticmethod
        def to_json_file(symbol: str, path: Path, data: Optional[dict[str, str]] = None):
            if not data:
                data = request_data(
                    symbol, AlphaVantageClient.Earnings.FUNCTION_STR)
            save_component(symbol, path, data,
                           AlphaVantageClient.Earnings.TYPE_STR)

        @staticmethod
        def from_local_file(symbol: str, cache_path: Path):
            pass

    class CompanyOverview(API):
        TYPE_STR = "CompanyOverview"
        FUNCTION_STR = "OVERVIEW"

        @staticmethod
        def print_json(symbol: str):
            data = request_data(
                symbol, AlphaVantageClient.CompanyOverview.FUNCTION_STR)
            print(data)

        @staticmethod
        def to_json_file(symbol: str, path: Path, data: Optional[dict[str, str]] = None):
            if not data:
                data = request_data(
                    symbol, AlphaVantageClient.CompanyOverview.FUNCTION_STR)
            save_component(symbol, path, data,
                           AlphaVantageClient.CompanyOverview.TYPE_STR)

        @staticmethod
        def from_local_file(symbol: str, cache_path: Path):
            pass


AVC = AlphaVantageClient


def request_data(symbol: str, function_str: str) -> dict[str, str]:
    url = f'https://www.alphavantage.co/query?function={function_str}&symbol={symbol}&apikey={API_KEY}'
    r = requests.get(url)
    data = r.json()
    return data


def save_component(symbol: str, base_path: Path, data: dict[str, str], type_str: str):
    with open(generate_json_file_path(base_path, symbol, type_str), "w") as fp:
        json.dump(data, fp, indent=4)


def generate_json_file_path(base_path: Path, symbol: str, type_str: str) -> Path:
    return Path(base_path/f"{symbol}.{type_str}.json")


def load_component(symbol: str, base_path: Path, data: dict[str, str], type_str: str):
    target_file = base_path/f"{symbol}.{type_str}.json"
    if target_file.exists():
        try:
            with open(target_file, "r") as fp:
                data = json.load(fp)
            return data
        except Exception as e:
            raise AlphaVantageClientError(
                "Unexpected error loading data from file") from e

    raise FileNotFoundError(
        f"target file not found: {target_file.resolve().as_posix()}")
