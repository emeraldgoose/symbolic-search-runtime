from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from benchmark.generator.core.schema import TableDef


class Backend(ABC):
    @abstractmethod
    def create_table(self, table: TableDef) -> None:
        ...

    @abstractmethod
    def write_batch(self, table: str, df: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def finalize(self, table: str) -> None:
        ...

    @abstractmethod
    def verify(self) -> dict[str, dict[str, int | float]]:
        ...
