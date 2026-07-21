from __future__ import annotations

import logging
import os
import re
from typing import Any

import pandas as pd

from syrch.core.models import ColumnSchema, TableSchema
from syrch.executors.base import BaseExecutor

logger = logging.getLogger(__name__)

_SESSION_EXPIRED_PATTERNS = [
    re.compile(r"\[NO_ACTIVE_SESSION\]"),
    re.compile(r"Session expired"),
    re.compile(r"Spark Connect Session expired"),
]


def _is_session_expired(exception: Exception) -> bool:
    msg = str(exception)
    return any(p.search(msg) for p in _SESSION_EXPIRED_PATTERNS)


class SparkExecutor(BaseExecutor):
    _spark: Any

    def __init__(
        self,
        catalog: str | None = None,
        schema: str | None = None,
        tables: list[str] | None = None,
    ):
        self._catalog = catalog or os.getenv("SPARK_CATALOG")
        self._schema_name = schema or os.getenv("SPARK_SCHEMA")
        self._tables = tables or []
        self._spark = None  # type: ignore[assignment]
        self._ensure_session()

    def _ensure_session(self):
        from pyspark.sql import SparkSession
        if self._spark is not None:
            try:
                self._spark.sql("SELECT 1").collect()
                return
            except Exception:
                logger.info("Spark session expired, creating a new one")
        self._spark = SparkSession.builder.getOrCreate()

    def execute(self, sql: str) -> pd.DataFrame:
        logger.debug("Executing SQL on Spark (%s chars)", len(sql))
        try:
            return self._spark.sql(sql).toPandas()
        except Exception as e:
            if _is_session_expired(e):
                logger.warning("Spark session expired, reconnecting...")
                self._ensure_session()
                return self._spark.sql(sql).toPandas()
            raise

    def get_schema(self, table_name: str | None = None) -> TableSchema:
        if table_name is None:
            table_name = self.list_tables()[0]
        logger.debug("Fetching schema for table: %s", table_name)
        try:
            rows = self._spark.sql(f"DESCRIBE {table_name}").collect()
        except Exception as e:
            if _is_session_expired(e):
                logger.warning("Spark session expired, reconnecting...")
                self._ensure_session()
                rows = self._spark.sql(f"DESCRIBE {table_name}").collect()
            else:
                raise
        columns = [ColumnSchema(name=r.col_name, type=r.data_type) for r in rows]
        return TableSchema(name=table_name, columns=columns)

    def list_tables(self) -> list[str]:
        if self._tables:
            return self._tables
        if self._catalog:
            self._spark.sql(f"USE CATALOG {self._catalog}")
        if self._schema_name:
            self._spark.sql(f"USE {self._schema_name}")
        try:
            rows = self._spark.sql("SHOW TABLES").collect()
        except Exception as e:
            if _is_session_expired(e):
                logger.warning("Spark session expired, reconnecting...")
                self._ensure_session()
                rows = self._spark.sql("SHOW TABLES").collect()
            else:
                raise
        return sorted({r.tableName for r in rows})

    def close(self) -> None:
        pass
