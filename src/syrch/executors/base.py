from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from syrch.core.models import ParentContext, TableSchema


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

    def materialize_context(self, context: ParentContext) -> str:
        """Materialize a ParentContext's data into a physical table named
        `context.table_name` (e.g. `_task_context_A`) so dependent tasks can
        JOIN it directly (v0.3.5b).

        Default implementation: no-op returning the table name. Executors
        that support temp tables/views override this. Empty or missing data
        is a no-op (returns the name without creating anything).
        """
        return context.table_name

    def drop_context(self, table_name: str) -> None:
        """Remove a previously materialized context table (cleanup)."""
