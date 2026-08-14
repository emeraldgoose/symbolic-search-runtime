# Databricks notebook source
# MAGIC %md
# MAGIC # syrch Benchmark — Delta Table Generation
# MAGIC
# MAGIC Run this notebook on a Databricks cluster (14.3 LTS or later) to generate benchmark Delta tables.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Repo cloned: `/Workspace/Users/<user>/symbolic-search-runtime`
# MAGIC - Unity Catalog catalog + schema: `syrch_benchmark.enterprise` (created automatically if permissions allow)
# MAGIC - Cluster: 14.3 LTS+, Delta Sharing enabled

# COMMAND ----------
# MAGIC %pip install pyyaml numpy pandas

# COMMAND ----------
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────
REPO_ROOT = Path("/Workspace/Users") / spark.sql("SELECT current_user()").collect()[0][0] / "symbolic-search-runtime"
CATALOG = "syrch_benchmark"
SCHEMA_NAME = "enterprise"
PROFILE = "databricks-enterprise"

sys.path.insert(0, str(REPO_ROOT))

from benchmark.generator.core.data import generate_table
from benchmark.generator.core.quality import apply_all_quality
from benchmark.generator.core.schema import load_schema_yaml, load_all_schemas
from benchmark.generator.core.seed import get_rng

# ── Catalog & Schema ──────────────────────────────────────────────
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
spark.sql(f"USE {SCHEMA_NAME}")

# ── Profile ───────────────────────────────────────────────────────
profile_path = REPO_ROOT / "benchmark" / "profiles" / f"{PROFILE}.yaml"
import yaml
with open(profile_path) as f:
    profile = yaml.safe_load(f)
while "extends" in profile:
    parent_name = profile.pop("extends")
    parent_path = REPO_ROOT / "benchmark" / "profiles" / f"{parent_name}.yaml"
    with open(parent_path) as f:
        parent = yaml.safe_load(f)
    parent.update(profile)
    profile = parent

seed = profile.get("seed", 42)
batch_size = profile.get("batch_size", 50000)
schema_include = profile.get("schema", {}).get("include", [])
quality_config = profile.get("quality", {})

# ── Load tables with topological sort ──────────────────────────────
SCHEMA_DIR = REPO_ROOT / "benchmark" / "schema"
all_tables = []
for s in schema_include:
    path = SCHEMA_DIR / s
    if path.exists():
        all_tables.extend(load_schema_yaml(path))
    else:
        logger.warning("Schema not found: %s", path)

table_map = {t.name: t for t in all_tables}
dep_graph = {t.name: set() for t in all_tables}
for t in all_tables:
    for c in t.columns:
        if c.fk:
            ref_table = c.fk.split(".")[0]
            if ref_table in dep_graph and ref_table != t.name:
                dep_graph[t.name].add(ref_table)

in_degree = {n: len(d) for n, d in dep_graph.items()}
queue = [n for n, d in in_degree.items() if d == 0]
sorted_names = []
while queue:
    name = queue.pop(0)
    sorted_names.append(name)
    for dn, deps in dep_graph.items():
        if name in deps:
            deps.remove(name)
            in_degree[dn] -= 1
            if in_degree[dn] == 0:
                queue.append(dn)

remaining = set(dep_graph.keys()) - set(sorted_names)
if remaining:
    logger.warning("Circular FK deps: %s", remaining)
    sorted_names.extend(remaining)

all_tables_sorted = [table_map[n] for n in sorted_names]

# ── Type mapping ──────────────────────────────────────────────────
TYPE_MAP = {
    "INTEGER": "INT", "INT": "INT", "BIGINT": "BIGINT", "LONG": "BIGINT",
    "REAL": "DOUBLE", "FLOAT": "DOUBLE", "DOUBLE": "DOUBLE", "DECIMAL": "DOUBLE",
    "TEXT": "STRING", "DATE": "DATE", "DATETIME": "TIMESTAMP",
    "TIMESTAMP": "TIMESTAMP", "TIMESTAMP_NTZ": "TIMESTAMP_NTZ",
    "BOOLEAN": "BOOLEAN", "BOOL": "BOOLEAN",
}

def spark_type(t: str) -> str:
    return TYPE_MAP.get(t.upper(), "STRING")

# ── Generate ──────────────────────────────────────────────────────
fk_registry: dict[str, pd.DataFrame] = {}

for table_def in all_tables_sorted:
    rng = get_rng(PROFILE, table_def.name)
    logger.info("Generating %s (%d rows)...", table_def.name, table_def.rows)
    start = time.time()

    tb_name = f"{CATALOG}.{SCHEMA_NAME}.{table_def.name}"
    spark.sql(f"DROP TABLE IF EXISTS {tb_name}")

    col_defs = [f"  {c.name} {spark_type(c.type)}" for c in table_def.columns]
    ddl = f"CREATE TABLE IF NOT EXISTS {tb_name} (\n" + ",\n".join(col_defs) + "\n) USING DELTA"
    spark.sql(ddl)

    if table_def.comment:
        escaped = table_def.comment.replace("'", "\\'")
        spark.sql(f"COMMENT ON TABLE {tb_name} IS '{escaped}'")

    tblproperties = profile.get("output", {}).get("delta", {}).get("tblproperties", {})
    for key, val in tblproperties.items():
        spark.sql(f"ALTER TABLE {tb_name} SET TBLPROPERTIES ('{key}' = '{val}')")

    cluster_cols = table_def.cluster_by
    if cluster_cols:
        cols_str = ", ".join(cluster_cols)
        spark.sql(f"ALTER TABLE {tb_name} CLUSTER BY ({cols_str})")

    n_rows = 0
    batch_dfs = []
    for batch_df in generate_table(rng, table_def, batch_size, fk_registry):
        batch_df = apply_all_quality(batch_df, quality_config, rng)
        sdf = spark.createDataFrame(batch_df)
        sdf.write.format("delta").mode("append").saveAsTable(tb_name)
        batch_dfs.append(batch_df)
        n_rows += len(batch_df)

    fk_registry[table_def.name] = pd.concat(batch_dfs, ignore_index=True)

    spark.sql(f"OPTIMIZE {tb_name}")

    elapsed = time.time() - start
    logger.info("%-30s %8d rows  %5.1fs  ✅", table_def.name, n_rows, elapsed)

# ── Verify ────────────────────────────────────────────────────────
print("\n=== Verification ===")
for t in all_tables_sorted:
    cnt = spark.sql(f"SELECT COUNT(*) as cnt FROM {CATALOG}.{SCHEMA_NAME}.{t.name}").collect()[0].cnt
    print(f"  {t.name:<30} {cnt:>8} rows")

print(f"\n✅ Done. {len(all_tables_sorted)} tables created in {CATALOG}.{SCHEMA_NAME}.")
