#!/usr/bin/env python3
"""
End-to-end benchmark runner for the Text-to-SQL pipeline.

Loads questions → runs syrch.query() → compares against ground truth →
classifies errors → produces 3-axis report (Retriever / Planner / SQL).

Usage:
    python benchmark/evaluator/run_benchmark.py --profile small
    python benchmark/evaluator/run_benchmark.py --profile small --questions pilot --model qwen3.5-4b-4bit
    python benchmark/evaluator/run_benchmark.py --quick                         # L1-L2 only
    python benchmark/evaluator/run_benchmark.py --skip-cache                    # fresh LLM calls
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from syrch import query
from benchmark.evaluator.errors import classify, summarize


_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
_QUESTIONS_DIR = Path(__file__).resolve().parent.parent / "questions"
_GENERATED_DIR = Path(__file__).resolve().parent.parent / "generated"


@dataclass
class QuestionResult:
    id: str
    question: str
    escalation_group: str
    escalation_level: int

    retriever: dict = field(default_factory=dict)
    planner: dict = field(default_factory=dict)
    sql: dict = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    duration: float = 0.0


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


def load_ground_truth(questions_name: str) -> dict[str, pd.DataFrame]:
    answers_dir = _GENERATED_DIR / "answers" / questions_name
    results: dict[str, pd.DataFrame] = {}
    if not answers_dir.exists():
        return results
    for csv_path in answers_dir.glob("*.csv"):
        qid = csv_path.stem
        if qid == "_summary":
            continue
        try:
            results[qid] = pd.read_csv(csv_path)
        except Exception:
            pass
    return results


def load_ground_truth_summary(questions_name: str) -> dict:
    path = _GENERATED_DIR / "answers" / questions_name / "_summary.json"
    if path.exists():
        with open(path) as f:
            return {r["id"]: r for r in json.load(f)}
    return {}


def _simulate_query(q: dict, db_path: str | list[str], executor_type: str = "sqlite") -> tuple:
    """Dry-run: execute ground truth SQL and return as mock syrch result."""
    gt_sql = q.get("ground_truth_sql", "")
    if not gt_sql:
        return None, "", [], [], None

    try:
        if executor_type == "databricks-sql":
            from databricks import sql as dbsql
            conn = dbsql.connect(
                server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME", ""),
                http_path=os.getenv("DATABRICKS_HTTP_PATH", ""),
                access_token=os.getenv("DATABRICKS_TOKEN", ""),
                catalog=os.getenv("DATABRICKS_CATALOG"),
                schema=os.getenv("DATABRICKS_SCHEMA"),
            )
            with conn.cursor() as cursor:
                cursor.execute(gt_sql)
                rows = cursor.fetchall()
                if not rows:
                    df = pd.DataFrame()
                else:
                    columns = [desc[0] for desc in cursor.description]
                    df = pd.DataFrame([list(r) for r in rows], columns=columns)
            conn.close()
        else:
            import sqlite3
            conn = sqlite3.connect(db_path if isinstance(db_path, str) else db_path[0])
            df = pd.read_sql_query(gt_sql, conn)
            conn.close()

        tables = _extract_tables_from_sql(gt_sql)
        dag_nodes = _guess_dag_from_tables(tables, q)
        return df, gt_sql, tables, dag_nodes, None
    except Exception as e:
        return None, "", [], [], str(e)


def _extract_tables_from_sql(sql: str) -> list[str]:
    import re
    cte_names = _extract_cte_names(sql)
    pattern = re.compile(r'\b(?:FROM|JOIN)\s+(\w+)', re.IGNORECASE)
    all_refs = set(pattern.findall(sql))
    return sorted(all_refs - cte_names)


def _extract_cte_names(sql: str) -> set[str]:
    import re
    names: set[str] = set()
    idx = sql.upper().find("WITH ")
    if idx < 0:
        return names
    remainder = sql[idx + 5:]
    pos = 0
    while pos < len(remainder):
        remainder = remainder[pos:].lstrip()
        m = re.match(r'(\w+)\s+AS\s*\(', remainder)
        if not m:
            break
        names.add(m.group(1))
        depth = 1
        i = m.end()
        while i < len(remainder) and depth > 0:
            if remainder[i] == '(':
                depth += 1
            elif remainder[i] == ')':
                depth -= 1
            i += 1
        pos = i
        if pos < len(remainder) and remainder[pos] == ',':
            pos += 1
        else:
            break
    return names


def _dataframes_match(df_a: pd.DataFrame, df_b: pd.DataFrame) -> bool:
    if len(df_a) != len(df_b) or set(df_a.columns) != set(df_b.columns):
        return False
    cols = sorted(df_a.columns)
    a = df_a[cols].reset_index(drop=True)
    b = df_b[cols].reset_index(drop=True)
    try:
        for col in cols:
            if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
                diff = (a[col].fillna(0).values - b[col].fillna(0).values)
                if not np.all(np.abs(diff) < 1e-6):
                    return False
            else:
                if not a[col].fillna("").astype(str).equals(b[col].fillna("").astype(str)):
                    return False
        return True
    except Exception:
        return False


def _guess_dag_from_tables(tables: list[str], q: dict) -> list[dict]:
    dag_eval = q.get("dag_evaluation", {})
    optimal = dag_eval.get("optimal_nodes", 1)
    nodes = []
    for i, t in enumerate(tables):
        nodes.append({
            "id": chr(65 + i),
            "description": f"Query {t}",
            "depends_on": [chr(65 + j) for j in range(i)],
            "depth": 0,
            "is_atomic": True,
        })
    if not nodes:
        for i in range(max(optimal, 1)):
            nodes.append({
                "id": chr(65 + i),
                "description": f"Sub-task {chr(65 + i)}",
                "depends_on": [chr(65 + j) for j in range(i)],
                "depth": 0,
                "is_atomic": True,
            })
    return nodes


def run_benchmark(
    profile_name: str = "small",
    questions_name: str = "pilot",
    model: str = "qwen3.5-4b-4bit",
    executor_type: str = "sqlite",
    base_url: str | None = None,
    api_key: str | None = None,
    quick: bool = False,
    dry_run: bool = False,
    skip_cache: bool = False,
    verbose: bool = False,
    max_level: int = 5,
) -> list[QuestionResult]:
    profile = load_profile(profile_name)
    questions = load_questions(questions_name)

    if executor_type == "sqlite":
        sqlite_cfg = profile.get("output", {}).get("sqlite", {})
        db_path: str | list[str] = str(Path(sqlite_cfg["path"]))
        if not Path(db_path).exists():
            print(f"Database not found: {db_path}")
            print("Run: python benchmark/generator/gen_business_db.py --profile {profile_name}")
            sys.exit(1)
    else:
        _SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "benchmark" / "schema"
        fqns = []
        for f in sorted(_SCHEMA_DIR.glob("*.yaml")):
            with open(f) as fh:
                data = yaml.safe_load(fh)
            for tname in data.get("tables", {}).keys():
                fqns.append(f"{tname}")
        db_path = fqns

    ground_truth = load_ground_truth(questions_name)
    gt_summary = load_ground_truth_summary(questions_name)

    if quick:
        questions = [q for q in questions if q.get("escalation_level", 1) <= 2]
        print(f"Quick mode: {len(questions)} questions (L1-L2)")

    if skip_cache:
        print("Cache disabled: fresh LLM calls")

    results: list[QuestionResult] = []

    for q in questions:
        qid = q["id"]
        level = q.get("escalation_level", 1)
        if level > max_level:
            continue

        print(f"\n{'='*60}")
        print(f"  [{qid}] L{level} {q.get('question_en', q['question'])[:60]}")
        print(f"{'='*60}")

        t0 = time.time()

        if dry_run:
            result_df, result_sql, tables_used, dag_nodes, exec_error = _simulate_query(q, db_path, executor_type)
            duration = time.time() - t0
            print(f"  [DRY-RUN] Tables: {tables_used}, DAG nodes: {len(dag_nodes)}")
        else:
            try:
                sr = query(
                    question=q.get("question_en", q["question"]),
                    db_path=db_path,
                    executor_type=executor_type,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    cache=not skip_cache,
                    verbose=verbose,
                )
                duration = time.time() - t0
                result_df = sr.data
                result_sql = sr.sql
                tables_used = sr.tables_used
                dag_nodes = sr.dag_nodes
                exec_error = None

                print(f"  Confidence: {sr.confidence:.3f}, Tokens: {sr.token_cost}, Time: {duration:.1f}s")
                print(f"  Tables: {tables_used}")
                print(f"  DAG nodes: {len(dag_nodes)}")

            except Exception as e:
                duration = time.time() - t0
                result_df = None
                result_sql = ""
                tables_used = []
                dag_nodes = []
                exec_error = str(e)
                print(f"  ERROR: {exec_error}")

        gt_df = ground_truth.get(qid)
        gt_sql = q.get("ground_truth_sql", "")

        errors = classify(
            question=q,
            ground_truth_df=gt_df,
            ground_truth_sql=gt_sql,
            result_df=result_df,
            result_sql=result_sql,
            tables_used=tables_used,
            dag_nodes=dag_nodes,
            all_tables=[],
            execution_error=exec_error,
        )

        error_summary = summarize([errors])
        for e in errors:
            if e["type"] != "none":
                print(f"  ⚠ {e['type']}: {e['detail']}")

        required_tables = set(q.get("required_tables", []))
        used = set(tables_used)

        retriever_metrics = {
            "recall": len(required_tables & used) / max(len(required_tables), 1),
            "precision": len(required_tables & used) / max(len(used), 1),
            "noise_tables": sorted(used - required_tables),
            "missing_tables": sorted(required_tables - used),
            "tables_used": sorted(used),
            "tables_required": sorted(required_tables),
        }

        dag_eval = q.get("dag_evaluation", {})
        node_count = len(dag_nodes)
        planner_metrics = {
            "node_count": node_count,
            "minimal_nodes": dag_eval.get("minimal_nodes", 0),
            "optimal_nodes": dag_eval.get("optimal_nodes", 0),
            "is_minimal": node_count >= dag_eval.get("minimal_nodes", 0) if dag_eval else True,
            "is_acceptable": node_count <= dag_eval.get("optimal_nodes", 0) * 2 if dag_eval else True,
        }

        gt_rows = len(gt_df) if gt_df is not None else None
        res_rows = len(result_df) if result_df is not None else None

        q_confidence = sr.confidence if (not exec_error and not dry_run) else (1.0 if dry_run and exec_error is None else 0.0)
        q_token_cost = sr.token_cost if (not exec_error and not dry_run) else 0

        sql_metrics = {
            "execution_success": exec_error is None,
            "result_match": result_df is not None and gt_df is not None and _dataframes_match(result_df, gt_df),
            "confidence": q_confidence,
            "token_cost": q_token_cost,
            "ground_truth_rows": gt_rows,
            "result_rows": res_rows,
        }

        qr = QuestionResult(
            id=qid,
            question=q.get("question_en", q["question"]),
            escalation_group=q.get("escalation_group", ""),
            escalation_level=level,
            retriever=retriever_metrics,
            planner=planner_metrics,
            sql=sql_metrics,
            errors=errors,
            duration=duration,
        )
        results.append(qr)

    return results


def print_summary(results: list[QuestionResult]) -> None:
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)

    total = len(results)
    retriever_recalls = [r.retriever["recall"] for r in results]
    retriever_precisions = [r.retriever["precision"] for r in results]
    sql_success = sum(1 for r in results if r.sql["execution_success"])
    sql_match = sum(1 for r in results if r.sql.get("result_match"))
    total_conf = sum(r.sql["confidence"] for r in results if r.sql["execution_success"])
    total_tokens = sum(r.sql["token_cost"] for r in results)
    total_time = sum(r.duration for r in results)
    error_counts: dict[str, int] = {}
    for r in results:
        for e in r.errors:
            if e["type"] != "none":
                error_counts[e["type"]] = error_counts.get(e["type"], 0) + 1

    print(f"\n  Questions:    {total}")
    print(f"  Total time:   {total_time:.1f}s")
    print(f"  Avg time/q:   {total_time/max(total,1):.1f}s")
    print(f"  Total tokens: {total_tokens}")

    print(f"\n  ── Retriever ──")
    print(f"  Avg Recall:    {sum(retriever_recalls)/max(total,1):.3f}")
    print(f"  Avg Precision: {sum(retriever_precisions)/max(total,1):.3f}")

    print(f"\n  ── Planner ──")
    minimal_count = sum(1 for r in results if r.planner.get("is_minimal"))
    print(f"  Minimal DAG:   {minimal_count}/{total}")

    print(f"\n  ── SQL ──")
    print(f"  Execution:     {sql_success}/{total}")
    print(f"  Result Match:  {sql_match}/{total}")
    print(f"  Avg Confidence: {total_conf/max(sql_success,1):.3f}")

    if error_counts:
        print(f"\n  ── Error Distribution ──")
        for etype, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {etype:<20} {count}")

    print()

    print(f"  {'ID':<6} {'RetR':>5} {'RetP':>5} {'DAG':>4} {'Exec':>5} {'Match':>5} {'Conf':>5} {'Tok':>5} {'Time':>5}")
    print(f"  {'─'*6} {'─'*5} {'─'*5} {'─'*4} {'─'*5} {'─'*5} {'─'*5} {'─'*5} {'─'*5}")
    for r in results:
        rr = f"{r.retriever['recall']:.2f}"
        rp = f"{r.retriever['precision']:.2f}"
        dag = f"{r.planner['node_count']}"
        exec_s = "✓" if r.sql["execution_success"] else "✗"
        match_s = "✓" if r.sql.get("result_match") else "•" if r.sql["execution_success"] else " "
        conf = f"{r.sql['confidence']:.2f}"
        tok = str(r.sql["token_cost"])
        dur = f"{r.duration:.1f}"
        print(f"  {r.id:<6} {rr:>5} {rp:>5} {dag:>4} {exec_s:>5} {match_s:>5} {conf:>5} {tok:>5} {dur:>5}s")

    print("=" * 70)


def export_results(results: list[QuestionResult], path: str) -> None:
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": len(results),
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results exported to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run syrch benchmark evaluation")
    parser.add_argument("--profile", default="small", help="Profile name")
    parser.add_argument("--questions", default="pilot", help="Questions file (without .json)")
    parser.add_argument("--model", default="qwen3.5-4b-4bit", help="LLM model name")
    parser.add_argument("--executor", default="sqlite", choices=["sqlite", "databricks-sql", "spark"],
                        help="Executor type")
    parser.add_argument("--base-url", help="LLM API base URL (e.g. http://localhost:11434/v1)")
    parser.add_argument("--api-token", help="LLM API key / token")
    parser.add_argument("--quick", action="store_true", help="L1-L2 only")
    parser.add_argument("--dry-run", action="store_true", help="Execute ground truth SQL instead of syrch (no LLM needed)")
    parser.add_argument("--skip-cache", action="store_true", help="Bypass disk cache")
    parser.add_argument("--verbose", action="store_true", help="Verbose LLM output")
    parser.add_argument("--max-level", type=int, default=5, help="Maximum escalation level")
    parser.add_argument("--export", help="Export results to JSON file")
    args = parser.parse_args()

    results = run_benchmark(
        profile_name=args.profile,
        questions_name=args.questions,
        model=args.model,
        executor_type=args.executor,
        base_url=args.base_url,
        api_key=args.api_token,
        quick=args.quick,
        dry_run=args.dry_run,
        skip_cache=args.skip_cache,
        verbose=args.verbose,
        max_level=args.max_level,
    )

    print_summary(results)

    if args.export:
        export_results(results, args.export)


if __name__ == "__main__":
    main()
