from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class API(ABC):
    TYPE_STR: str
    FUNCTION_STR: str

    @staticmethod
    @abstractmethod
    def print_json(symbol: str):
        pass

    @staticmethod
    @abstractmethod
    def to_json_file(symbol: str, path: Path, data: Optional[dict[str, str]] = None):
        pass

    @staticmethod
    @abstractmethod
    def to_dict(symbol: str):
        pass

    @staticmethod
    @abstractmethod
    def from_local_file(symbol: str, cache_path: Path):
        pass
