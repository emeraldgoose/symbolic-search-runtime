from __future__ import annotations

from io import StringIO

import pandas as pd

from syrch.executors.base import BaseExecutor
from syrch.core.models import TableSchema
from syrch.llm.cache import CentralCache


class CachedExecutor(BaseExecutor):
    def __init__(self, inner: BaseExecutor, cache: CentralCache):
        self._inner = inner
        self._cache = cache

    @property
    def cache(self) -> CentralCache:
        return self._cache

    def execute(self, sql: str) -> pd.DataFrame:
        cached = self._cache.get("sql", sql=sql)
        if cached is not None:
            return pd.read_json(StringIO(cached))
        result = self._inner.execute(sql)
        self._cache.set("sql", result.to_json(), sql=sql)
        return result

    def get_schema(self, table_name: str | None = None) -> TableSchema:
        return self._inner.get_schema(table_name)

    def list_tables(self) -> list[str]:
        return self._inner.list_tables()

    def close(self) -> None:
        self._inner.close()
