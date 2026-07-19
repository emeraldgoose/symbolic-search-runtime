#!/usr/bin/env python3
"""
Generate ground truth result CSVs by executing ground_truth_sql against the database.

Usage:
    python benchmark/evaluator/generate_ground_truth.py --profile small
    python benchmark/evaluator/generate_ground_truth.py --profile small --questions pilot --db generated/sqlite/benchmark_small.sqlite
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Support both local and Databricks Workspace repo layouts
_repo_root = Path(__file__).resolve().parent.parent.parent
if not (_repo_root / "pyproject.toml").exists():
    _repo_root = Path(__file__).resolve().parent.parent.parent.parent
if not (_repo_root / "pyproject.toml").exists():
    _repo_root = Path("/Workspace") / "Users" / os.getenv("USER", "") / "symbolic-search-runtime"
sys.path.insert(0, str(_repo_root))

_src = _repo_root / "src"
if _src.exists():
    sys.path.insert(0, str(_src))

_PROFILES_DIR = _repo_root / "benchmark" / "profiles"
_QUESTIONS_DIR = _repo_root / "benchmark" / "questions"
_GENERATED_DIR = _repo_root / "benchmark" / "generated"
_ANSWERS_DIR = _GENERATED_DIR / "answers"


def load_profile(name: str) -> dict:
    path = _PROFILES_DIR / f"{name}.yaml"
    with open(path) as f:
        profile = yaml.safe_load(f)
    while "extends" in profile:
        parent_name = profile.pop("extends")
        with open(_PROFILES_DIR / f"{parent_name}.yaml") as pf:
            parent = yaml.safe_load(pf)
        parent.update(profile)
        profile = parent
    return profile


def load_questions(name: str) -> list[dict]:
    path = _QUESTIONS_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)


def _connect_sqlite(profile, db_path: str | None):
    if db_path:
        db = Path(db_path)
    else:
        sqlite_cfg = profile.get("output", {}).get("sqlite", {})
        db = Path(sqlite_cfg["path"])
    if not db.exists():
        print(f"Database not found: {db}")
        sys.exit(1)
    import sqlite3
    return sqlite3.connect(str(db))


def _connect_databricks(
    server_hostname: str | None = None,
    http_path: str | None = None,
    access_token: str | None = None,
    catalog: str | None = None,
    schema: str | None = None,
):
    from databricks import sql as dbsql
    server_hostname = server_hostname or os.getenv("DATABRICKS_SERVER_HOSTNAME") or os.getenv("DATABRICKS_HOST")
    http_path = http_path or os.getenv("DATABRICKS_HTTP_PATH")
    access_token = access_token or os.getenv("DATABRICKS_TOKEN")
    catalog = catalog or os.getenv("DATABRICKS_CATALOG")
    schema = schema or os.getenv("DATABRICKS_SCHEMA")
    if not server_hostname or not http_path:
        raise ValueError(
            "Databricks connection requires server_hostname and http_path. "
            "Set DATABRICKS_SERVER_HOSTNAME and DATABRICKS_HTTP_PATH env vars "
            "or pass them directly."
        )
    conn = dbsql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=access_token,
        catalog=catalog,
        schema=schema,
    )
    return conn


def _connect_spark(catalog: str | None = None, schema: str | None = None):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    if catalog:
        spark.sql(f"USE CATALOG {catalog}")
    if schema:
        spark.sql(f"USE {schema}")
    return spark


def _execute_gt_sql(conn, gt_sql: str, executor_type: str) -> pd.DataFrame:
    if executor_type == "databricks-sql":
        with conn.cursor() as cursor:
            catalog = os.getenv("DATABRICKS_CATALOG")
            schema = os.getenv("DATABRICKS_SCHEMA")
            if catalog:
                cursor.execute(f"USE CATALOG {catalog}")
            if schema:
                cursor.execute(f"USE SCHEMA {schema}")
            cursor.execute(gt_sql)
            rows = cursor.fetchall()
            if not rows:
                return pd.DataFrame()
            columns = [desc[0] for desc in cursor.description]
            return pd.DataFrame([list(r) for r in rows], columns=columns)
    elif executor_type == "spark":
        return conn.sql(gt_sql).toPandas()
    else:
        import sqlite3
        return pd.read_sql_query(gt_sql, conn)


def generate_ground_truth(
    profile_name: str = "small",
    questions_name: str = "pilot",
    db_path: str | None = None,
    executor_type: str = "sqlite",
    output_dir: str | None = None,
    server_hostname: str | None = None,
    http_path: str | None = None,
    token: str | None = None,
    db_catalog: str | None = None,
    db_schema: str | None = None,
) -> list[dict]:
    profile = load_profile(profile_name)
    questions = load_questions(questions_name)

    if executor_type != "sqlite":
        delta_cfg = profile.get("output", {}).get("delta", {})
        db_catalog = db_catalog or delta_cfg.get("catalog")
        db_schema = db_schema or delta_cfg.get("schema")

    if executor_type == "databricks-sql":
        conn = _connect_databricks(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=token,
            catalog=db_catalog,
            schema=db_schema,
        )
    elif executor_type == "spark":
        conn = _connect_spark(catalog=db_catalog, schema=db_schema)
    else:
        conn = _connect_sqlite(profile, db_path)

    answers_dir = Path(output_dir) if output_dir else _ANSWERS_DIR / questions_name

    results = []
    try:
        for q in questions:
            qid = q["id"]
            # Prefer executor-specific SQL when available
            if executor_type in ("spark", "databricks-sql"):
                gt_sql = q.get("ground_truth_sql_spark") or q.get("ground_truth_sql", "")
            else:
                gt_sql = q.get("ground_truth_sql", "")
            if not gt_sql:
                print(f"  SKIP  {qid}: no ground_truth_sql")
                continue

            result_path_str = q.get("ground_truth_result_path", "")
            if result_path_str:
                relative = Path(result_path_str)
                if relative.parts[0] == "generated":
                    relative = Path(*relative.parts[1:])
                result_path = _GENERATED_DIR / relative
            else:
                result_path = answers_dir / f"{qid}.csv"

            result_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                df = _execute_gt_sql(conn, gt_sql, executor_type)
                df.to_csv(result_path, index=False)
                print(f"  OK    {qid}: {len(df)} rows → {result_path}")
                results.append({"id": qid, "rows": len(df), "columns": list(df.columns), "path": str(result_path)})
            except Exception as e:
                print(f"  ERROR {qid}: {e}")
                results.append({"id": qid, "error": str(e)})
    finally:
        if executor_type == "spark":
            conn.stop()
        else:
            conn.close()

    summary_path = answers_dir / "_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary: {summary_path}")

    ok = sum(1 for r in results if "error" not in r)
    total = len(results)
    print(f"Generated: {ok}/{total} ground truth results")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ground truth result CSVs")
    parser.add_argument("--profile", default="small", help="Profile name")
    parser.add_argument("--questions", default="pilot", help="Questions file (without .json)")
    parser.add_argument("--db", help="Override SQLite DB path (optional)")
    parser.add_argument("--executor", default="sqlite", choices=["sqlite", "databricks-sql", "spark"],
                        help="Executor type")
    parser.add_argument("--output-dir", help="Override output directory for CSVs")
    parser.add_argument("--server-hostname", help="Databricks server hostname")
    parser.add_argument("--http-path", help="Databricks HTTP path")
    parser.add_argument("--token", help="Databricks access token")
    parser.add_argument("--catalog", help="Databricks catalog name")
    parser.add_argument("--schema", help="Databricks schema name")
    args = parser.parse_args()
    generate_ground_truth(
        args.profile, args.questions, args.db,
        executor_type=args.executor, output_dir=args.output_dir,
        server_hostname=args.server_hostname, http_path=args.http_path,
        token=args.token, db_catalog=args.catalog, db_schema=args.schema,
    )


if __name__ == "__main__":
    main()
