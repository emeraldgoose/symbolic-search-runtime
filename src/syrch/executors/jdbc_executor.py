from __future__ import annotations

import pandas as pd

from syrch.core.models import ColumnSchema, TableSchema, infer_layer
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
        assert self._conn is not None
        return pd.read_sql(sql, self._conn)

    def get_schema(self, table_name: str | None = None) -> TableSchema:
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        if table_name is None:
            table_name = self.list_tables()[0]

        from sqlalchemy import inspect as sa_inspect
        inspector = sa_inspect(self._engine)
        cols_info = inspector.get_columns(table_name)
        columns = [
            ColumnSchema(
                name=c["name"],
                type=str(c["type"]),
                description=c.get("comment", None),
            )
            for c in cols_info
        ]
        return TableSchema(name=table_name, columns=columns, layer=infer_layer(table_name))

    def list_tables(self) -> list[str]:
        if self._conn is None:
            self._connect()
        from sqlalchemy import inspect

        inspector = inspect(self._engine)
        return [t for t in inspector.get_table_names() if not t.startswith("_task_context_")]

    def materialize_context(self, context) -> str:
        table = context.table_name
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        if context.data is None or context.data.empty:
            return table
        context.data.to_sql(table, self._engine, if_exists="replace", index=False)
        return table

    def drop_context(self, table_name: str) -> None:
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        from sqlalchemy import text
        self._conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))

    def close(self) -> None:
        if self._conn:
            self._conn.close()
