from __future__ import annotations

import logging
import os

import pandas as pd

from syrch.core.models import ColumnSchema, TableSchema
from syrch.executors.base import BaseExecutor

logger = logging.getLogger(__name__)


class DatabricksExecutor(BaseExecutor):
    def __init__(
        self,
        server_hostname: str | None = None,
        http_path: str | None = None,
        access_token: str | None = None,
        catalog: str | None = None,
        schema: str | None = None,
        auth_type: str = "pat",
        client_id: str | None = None,
        client_secret: str | None = None,
        azure_tenant_id: str | None = None,
        tables: list[str] | None = None,
    ):
        self.server_hostname = server_hostname or os.getenv("DATABRICKS_SERVER_HOSTNAME", "")
        self.http_path = http_path or os.getenv("DATABRICKS_HTTP_PATH", "")
        self.access_token = access_token or os.getenv("DATABRICKS_TOKEN", "")
        self.catalog = catalog or os.getenv("DATABRICKS_CATALOG")
        self.schema_name = schema or os.getenv("DATABRICKS_SCHEMA")
        self.auth_type = auth_type or os.getenv("DATABRICKS_AUTH_TYPE", "pat")
        self.client_id = client_id or os.getenv("DATABRICKS_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("DATABRICKS_CLIENT_SECRET")
        self.azure_tenant_id = azure_tenant_id or os.getenv("AZURE_TENANT_ID")
        self._tables = tables or []
        self._conn = None

    def _connect(self):
        if not self.server_hostname or not self.http_path:
            raise ValueError(
                "DatabricksExecutor requires DATABRICKS_SERVER_HOSTNAME and "
                "DATABRICKS_HTTP_PATH environment variables. "
                "Inside Databricks Runtime, use executor_type='spark' instead "
                "with pip install syrch[spark]."
            )

        from databricks import sql

        kwargs = dict(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
        )

        if self.auth_type == "pat":
            kwargs["access_token"] = self.access_token
        elif self.auth_type == "databricks-oauth":
            kwargs["auth_type"] = "databricks-oauth"
            kwargs["client_id"] = self.client_id
            kwargs["client_secret"] = self.client_secret
        elif self.auth_type == "azure":
            kwargs["auth_type"] = "azure"
            kwargs["client_id"] = self.client_id
            kwargs["client_secret"] = self.client_secret
            kwargs["azure_tenant_id"] = self.azure_tenant_id
        else:
            kwargs["access_token"] = self.access_token

        if self.catalog:
            kwargs["catalog"] = self.catalog
        if self.schema_name:
            kwargs["schema"] = self.schema_name

        logger.debug("Connecting to Databricks: server=%s http_path=%s auth_type=%s",
                     self.server_hostname, self.http_path, self.auth_type)
        self._conn = sql.connect(**kwargs)
        return self._conn

    def execute(self, sql: str) -> pd.DataFrame:
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        logger.debug("Executing SQL on Databricks (%s chars)", len(sql))
        with self._conn.cursor() as cursor:
            if self.catalog:
                cursor.execute(f"USE CATALOG {self.catalog}")
            if self.schema_name:
                cursor.execute(f"USE SCHEMA {self.schema_name}")
            cursor.execute(sql)
            rows = cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            columns = [desc[0] for desc in cursor.description]
            return pd.DataFrame([list(r) for r in rows], columns=columns)

    @staticmethod
    def _parse_fqn(fqn: str) -> tuple[str | None, str | None, str]:
        parts = fqn.split(".")
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        if len(parts) == 2:
            return None, parts[0], parts[1]
        return None, None, parts[0]

    def get_schema(self, table_name: str | None = None) -> TableSchema:
        if table_name is None:
            table_name = self.list_tables()[0]
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        catalog, schema, table = self._parse_fqn(table_name)
        logger.debug("Fetching schema for table: %s", table_name)
        with self._conn.cursor() as cursor:
            cursor.columns(catalog_name=catalog, schema_name=schema, table_name=table)
            columns = [
                ColumnSchema(name=row[2], type=row[5])
                for row in cursor.fetchall()
            ]
        return TableSchema(name=table_name, columns=columns)

    def list_tables(self) -> list[str]:
        if self._tables:
            return self._tables
        if self._conn is None:
            self._connect()
        assert self._conn is not None
        logger.debug("Listing tables in catalog=%s schema=%s", self.catalog, self.schema_name)
        with self._conn.cursor() as cursor:
            cursor.tables(catalog_name=self.catalog, schema_name=self.schema_name)
            return sorted({row[2] for row in cursor.fetchall()})

    def close(self) -> None:
        if self._conn:
            logger.debug("Closing Databricks connection")
            self._conn.close()
