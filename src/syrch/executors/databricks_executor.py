from __future__ import annotations

import os

import pandas as pd

from syrch.core.models import ColumnSchema, TableSchema
from syrch.executors.base import BaseExecutor


class DatabricksExecutor(BaseExecutor):
    def __init__(
        self,
        server_hostname: str | None = None,
        http_path: str | None = None,
        access_token: str | None = None,
        catalog: str | None = None,
        schema: str | None = None,
    ):
        self.server_hostname = server_hostname or os.getenv("DATABRICKS_SERVER_HOSTNAME", "")
        self.http_path = http_path or os.getenv("DATABRICKS_HTTP_PATH", "")
        self.access_token = access_token or os.getenv("DATABRICKS_TOKEN", "")
        self.catalog = catalog or os.getenv("DATABRICKS_CATALOG")
        self.schema_name = schema or os.getenv("DATABRICKS_SCHEMA")
        self._conn = None

    def _connect(self):
        from databricks import sql

        kwargs = dict(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            access_token=self.access_token,
        )
        if self.catalog:
            kwargs["catalog"] = self.catalog
        if self.schema_name:
            kwargs["schema"] = self.schema_name
        self._conn = sql.connect(**kwargs)
        return self._conn

    def execute(self, sql: str) -> pd.DataFrame:
        if self._conn is None:
            self._connect()
        with self._conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            columns = [desc[0] for desc in cursor.description]
            return pd.DataFrame([list(r) for r in rows], columns=columns)

    def get_schema(self, table_name: str | None = None) -> TableSchema:
        if table_name is None:
            table_name = self.list_tables()[0]
        if self._conn is None:
            self._connect()
        with self._conn.cursor() as cursor:
            cursor.columns(catalog_name=self.catalog, schema_name=self.schema_name, table_name=table_name)
            columns = [
                ColumnSchema(name=row[2], type=row[5])
                for row in cursor.fetchall()
            ]
        return TableSchema(name=table_name, columns=columns)

    def list_tables(self) -> list[str]:
        if self._conn is None:
            self._connect()
        with self._conn.cursor() as cursor:
            cursor.tables(catalog_name=self.catalog, schema_name=self.schema_name)
            return sorted({row[2] for row in cursor.fetchall()})

    def close(self) -> None:
        if self._conn:
            self._conn.close()
