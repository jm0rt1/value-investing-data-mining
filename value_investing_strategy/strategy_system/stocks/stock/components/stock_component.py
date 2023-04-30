

from __future__ import annotations
from os import path
from pathlib import Path

from abc import ABC, abstractmethod
from typing import Any, Union
import json


class StockComponent(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> StockComponent:
        pass

    def from_alpha_vantage_data(self, cls, data):
        pass

    @classmethod
    @abstractmethod
    def from_json_file(cls, path: Path) -> StockComponent:
        pass

    @staticmethod
    def load_json_dict(path: Path) -> dict[str, Union[str, list[dict[str, str]]]]:

        with open(path, "r") as fp:
            data = json.load(fp)
        return data
