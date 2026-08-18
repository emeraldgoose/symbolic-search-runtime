from __future__ import annotations

import sqlite3
import threading

import pandas as pd

from syrch.core.models import ColumnSchema, ParentContext, TableSchema, infer_layer
from syrch.executors.base import BaseExecutor

_CONTEXT_PREFIX = "_task_context_"


class SQLiteExecutor(BaseExecutor):
    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._main_conn: sqlite3.Connection | None = sqlite3.connect(path)

    def _get_conn(self) -> sqlite3.Connection:
        thread_id = threading.get_ident()
        if thread_id == threading.main_thread().ident:
            assert self._main_conn is not None
            return self._main_conn
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.path)
        return self._local.conn

    @property
    def db_id(self) -> str:
        return f"sqlite:{self.path}"

    def execute(self, sql: str) -> pd.DataFrame:
        conn = self._get_conn()
        return pd.read_sql(sql, conn)

    def get_schema(self, table_name: str | None = None) -> TableSchema:
        conn = self._get_conn()
        if table_name is None:
            table_name = self.list_tables()[0]
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = [
            ColumnSchema(name=row[1], type=row[2], nullable=not row[3], description=None)
            for row in cursor.fetchall()
        ]
        return TableSchema(name=table_name, columns=columns, layer=infer_layer(table_name))

    def list_tables(self) -> list[str]:
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return [
            row[0] for row in cursor.fetchall()
            if not row[0].startswith(_CONTEXT_PREFIX)
        ]

    def materialize_context(self, context: ParentContext) -> str:
        table = context.table_name
        conn = self._get_conn()
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        if context.data is None or context.data.empty:
            conn.commit()
            return table
        context.data.to_sql(table, conn, index=False)
        conn.commit()
        return table

    def drop_context(self, table_name: str) -> None:
        conn = self._get_conn()
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._main_conn:
                self._main_conn.close()
                self._main_conn = None
            if hasattr(self._local, "conn") and self._local.conn:
                self._local.conn.close()
                self._local.conn = None
