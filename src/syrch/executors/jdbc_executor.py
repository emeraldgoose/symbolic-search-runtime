from __future__ import annotations

import pandas as pd

from syrch.core.models import TableSchema
from syrch.executors.base import BaseExecutor


class JDBCExecutor(BaseExecutor):
    def __init__(self, connection_string: str, **kwargs):
        self.connection_string = connection_string
        self.kwargs = kwargs
        self._conn = None

    def _connect(self):
        import sqlalchemy

        self._engine = sqlalchemy.create_engine(self.connection_string, **self.kwargs)
        self._conn = self._engine.connect()
        return self._conn

    def execute(self, sql: str) -> pd.DataFrame:
        if self._conn is None:
            self._connect()
        return pd.read_sql(sql, self._conn)

    def get_schema(self, table_name: str | None = None) -> TableSchema:
        if self._conn is None:
            self._connect()
        if table_name is None:
            table_name = self.list_tables()[0]
        result = self._conn.execute(f"SELECT * FROM {table_name} LIMIT 0")
        columns = [
            ColumnSchema(name=col.name, type=str(col.type))
            for col in result.cursor.description
        ]
        return TableSchema(name=table_name, columns=columns)

    def list_tables(self) -> list[str]:
        if self._conn is None:
            self._connect()
        from sqlalchemy import inspect

        inspector = inspect(self._engine)
        return inspector.get_table_names()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
