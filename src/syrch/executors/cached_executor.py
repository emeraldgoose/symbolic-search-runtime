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

    @property
    def db_id(self) -> str:
        return self._inner.db_id

    def execute(self, sql: str) -> pd.DataFrame:
        cached = self._cache.get("sql", db=self._inner.db_id, sql=sql)
        if cached is not None:
            return pd.read_json(StringIO(cached))
        result = self._inner.execute(sql)
        self._cache.set("sql", result.to_json(), db=self._inner.db_id, sql=sql)
        return result

    def get_schema(self, table_name: str | None = None) -> TableSchema:
        # `table_name=None` resolves to the DB's first table, so its result is
        # not stable across table-list changes — don't cache that case.
        if table_name is None:
            return self._inner.get_schema(None)
        cached = self._cache.get("schema", db=self._inner.db_id, table_name=table_name)
        if cached is not None:
            return cached
        result = self._inner.get_schema(table_name)
        self._cache.set("schema", result, db=self._inner.db_id, table_name=table_name)
        return result

    def list_tables(self) -> list[str]:
        return self._inner.list_tables()

    def materialize_context(self, context) -> str:
        return self._inner.materialize_context(context)

    def drop_context(self, table_name: str) -> None:
        self._inner.drop_context(table_name)

    def close(self) -> None:
        self._inner.close()
