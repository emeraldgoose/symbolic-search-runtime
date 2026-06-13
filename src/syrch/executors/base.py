from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from syrch.core.models import TableSchema


class BaseExecutor(ABC):
    @abstractmethod
    def execute(self, sql: str) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_schema(self, table_name: str | None = None) -> TableSchema:
        ...

    @abstractmethod
    def list_tables(self) -> list[str]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
