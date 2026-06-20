from __future__ import annotations

import logging
import os

import pandas as pd

from syrch.core.models import ColumnSchema, TableSchema
from syrch.executors.base import BaseExecutor

logger = logging.getLogger(__name__)


class SparkExecutor(BaseExecutor):
    def __init__(
        self,
        catalog: str | None = None,
        schema: str | None = None,
        tables: list[str] | None = None,
    ):
        from pyspark.sql import SparkSession

        self.spark = SparkSession.builder.getOrCreate()
        self._tables = tables or []
        if not self._tables:
            self.catalog = catalog or os.getenv("SPARK_CATALOG")
            self.schema_name = schema or os.getenv("SPARK_SCHEMA")
        else:
            self.catalog = None
            self.schema_name = None

    def execute(self, sql: str) -> pd.DataFrame:
        logger.debug("Executing SQL on Spark (%s chars)", len(sql))
        return self.spark.sql(sql).toPandas()

    def get_schema(self, table_name: str | None = None) -> TableSchema:
        if table_name is None:
            table_name = self.list_tables()[0]
        logger.debug("Fetching schema for table: %s", table_name)
        rows = self.spark.sql(f"DESCRIBE {table_name}").collect()
        columns = [ColumnSchema(name=r.col_name, type=r.data_type) for r in rows]
        return TableSchema(name=table_name, columns=columns)

    def list_tables(self) -> list[str]:
        if self._tables:
            return self._tables
        if self.catalog:
            self.spark.sql(f"USE CATALOG {self.catalog}")
        if self.schema_name:
            self.spark.sql(f"USE {self.schema_name}")
        rows = self.spark.sql("SHOW TABLES").collect()
        return sorted({r.tableName for r in rows})

    def close(self) -> None:
        pass
