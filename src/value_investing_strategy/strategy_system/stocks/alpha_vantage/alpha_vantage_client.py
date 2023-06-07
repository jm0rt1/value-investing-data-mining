from enum import Enum, auto
import json
import os
from pathlib import Path
import requests
from typing import Optional

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

    class ComponentType(Enum):
        BalanceSheet = auto()
        IncomeStatement = auto()
        CompanyOverview = auto()
        Earnings = auto()
        CashFlow = auto()
        TimeSeriesMonthly = auto()
        TimeSeriesMonthlyAdjusted = auto()

    @staticmethod
    def component_file_exists(ticker: str, data_cache_path: Path, component_type: ComponentType):
        name = f"{ticker}.{component_type.name}.json"
        files = os.listdir(data_cache_path)
        if name in files:
            return True
        return False

    class APIComponent(API):
        TYPE_STR = ""
        FUNCTION_STR = ""

        @staticmethod
        def print_json(symbol: str):
            data = request_data(
                symbol, AlphaVantageClient.APIComponent.FUNCTION_STR)
            print(data)

        @staticmethod
        def to_json_file(symbol: str, path: Path, data: Optional[dict[str, str]] = None):
            if not data:
                data = request_data(
                    symbol, AlphaVantageClient.APIComponent.FUNCTION_STR)
            save_component(symbol, path, data,
                           AlphaVantageClient.APIComponent.TYPE_STR)

        @staticmethod
        def from_local_file(symbol: str, cache_path: Path):
            pass  # Add your implementation here.

    class IncomeStatement(APIComponent):
        TYPE_STR = "IncomeStatement"
        FUNCTION_STR = "INCOME_STATEMENT"

    class BalanceSheet(APIComponent):
        TYPE_STR = "BalanceSheet"
        FUNCTION_STR = "BALANCE_SHEET"

    class CashFlow(APIComponent):
        TYPE_STR = "CashFlow"
        FUNCTION_STR = "CASH_FLOW"

    class Earnings(APIComponent):
        TYPE_STR = "Earnings"
        FUNCTION_STR = "EARNINGS"

    class CompanyOverview(APIComponent):
        TYPE_STR = "CompanyOverview"
        FUNCTION_STR = "OVERVIEW"

    class TimeSeriesMonthly(APIComponent):
        TYPE_STR = "TimeSeriesMonthly"
        FUNCTION_STR = "TIME_SERIES_MONTHLY"

    class TimeSeriesMonthlyAdjusted(APIComponent):
        TYPE_STR = "TimeSeriesMonthlyAdjusted"
        FUNCTION_STR = "TIME_SERIES_MONTHLY_ADJUSTED"


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
    return base_path / f"{symbol}.{type_str}.json"


def load_component(symbol: str, base_path: Path, type_str: str):
    target_file = base_path / f"{symbol}.{type_str}.json"
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
