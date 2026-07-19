from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.generator.backends.base import Backend
from benchmark.generator.core.schema import ColumnDef, TableDef

logger = logging.getLogger(__name__)

_TYPE_MAP = {
    "INTEGER": "INT", "INT": "INT", "BIGINT": "BIGINT", "LONG": "BIGINT",
    "REAL": "DOUBLE", "FLOAT": "DOUBLE", "DOUBLE": "DOUBLE", "DECIMAL": "DOUBLE",
    "TEXT": "STRING", "DATE": "DATE", "DATETIME": "TIMESTAMP",
    "TIMESTAMP": "TIMESTAMP", "TIMESTAMP_NTZ": "TIMESTAMP_NTZ",
    "BOOLEAN": "BOOLEAN", "BOOL": "BOOLEAN",
}


def _to_spark_type(t: str) -> str:
    return _TYPE_MAP.get(t.upper(), "STRING")


def _ddl(table: TableDef, catalog: str, schema: str) -> str:
    col_defs: list[str] = []
    for c in table.columns:
        spark_type = _to_spark_type(c.type)
        nullable = "" if c.pk else ""
        col_defs.append(f"  {c.name} {spark_type}{nullable}")

    tb_name = f"{catalog}.{schema}.{table.name}"
    return f"CREATE TABLE IF NOT EXISTS {tb_name} (\n" + ",\n".join(col_defs) + "\n) USING DELTA"


class DeltaBackend(Backend):
    def __init__(
        self,
        catalog: str = "benchmark",
        schema: str = "default",
        location: str | None = None,
        clustering: str | None = "liquid",
        cluster_by: dict[str, list[str]] | None = None,
        optimize: bool = True,
        tblproperties: dict[str, str] | None = None,
        mode: str = "local",
        spark_session=None,
    ):
        if spark_session is not None:
            self.spark = spark_session
        else:
            from pyspark.sql import SparkSession

            if mode == "local":
                saved = os.environ.pop("SPARK_REMOTE", None)
                builder = SparkSession.builder.appName("syrch-benchmark") \
                    .master("local[*]")
            else:
                builder = SparkSession.builder.appName("syrch-benchmark")

            if location:
                builder.config("spark.sql.warehouse.dir", location)
            self.spark = builder.config(
                "spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension"
            ).config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            ).getOrCreate()

            if mode == "local" and saved is not None:
                os.environ["SPARK_REMOTE"] = saved

        self.catalog = catalog
        self.schema_name = schema
        self.location = location
        self.clustering = clustering
        self.cluster_by = cluster_by or {}
        self.optimize = optimize
        self.tblproperties = tblproperties or {
            "delta.autoOptimize.optimizeWrite": "true",
            "delta.autoOptimize.autoCompact": "true",
        }

        try:
            self.spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
        except Exception:
            logger.warning("Could not create catalog '%s' — assuming it exists", catalog)
        self.spark.sql(f"USE CATALOG {catalog}")
        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        self.spark.sql(f"USE {schema}")

    def create_table(self, table: TableDef) -> None:
        tb_name = f"{self.catalog}.{self.schema_name}.{table.name}"
        self.spark.sql(f"DROP TABLE IF EXISTS {tb_name}")
        self.spark.sql(_ddl(table, self.catalog, self.schema_name))

        if table.comment:
            escaped = table.comment.replace("'", "\\'")
            self.spark.sql(f"COMMENT ON TABLE {tb_name} IS '{escaped}'")

        for key, val in self.tblproperties.items():
            self.spark.sql(
                f"ALTER TABLE {tb_name} SET TBLPROPERTIES ('{key}' = '{val}')"
            )

        cluster_cols = table.cluster_by or self.cluster_by.get(table.name)
        if cluster_cols and self.clustering == "liquid":
            cols_str = ", ".join(cluster_cols)
            self.spark.sql(f"ALTER TABLE {tb_name} CLUSTER BY ({cols_str})")
            logger.info("Liquid clustered %s by (%s)", table.name, cols_str)

    def write_batch(self, table: str, df: pd.DataFrame) -> None:
        tb_name = f"{self.catalog}.{self.schema_name}.{table}"
        sdf = self.spark.createDataFrame(df)
        sdf.write.format("delta").mode("append").saveAsTable(tb_name)

    def finalize(self, table: str) -> None:
        if not self.optimize:
            return
        tb_name = f"{self.catalog}.{self.schema_name}.{table}"
        try:
            self.spark.sql(f"OPTIMIZE {tb_name}")
        except Exception as e:
            logger.warning("OPTIMIZE failed for %s: %s", table, e)

    def verify(self) -> dict[str, dict[str, int | float]]:
        all_tables = self.spark.sql(
            f"SHOW TABLES IN {self.catalog}.{self.schema_name}"
        ).collect()
        results: dict[str, dict[str, int | float]] = {}
        for row in all_tables:
            tname = row.tableName
            cnt = self.spark.sql(
                f"SELECT COUNT(*) as cnt FROM {self.catalog}.{self.schema_name}.{tname}"
            ).collect()[0].cnt
            results[tname] = {"rows": cnt, "checksum": 0}
        return results

    def close(self) -> None:
        pass
