"""
Rule-based error classification for benchmark evaluation.

Evaluates syrch pipeline outputs against ground truth to classify errors
into the standard Error Taxonomy:
  - Wrong Table, Wrong Column, Wrong Grain, Wrong Join, Wrong Filter
  - Wrong Semantic, Wrong Time, Wrong SCD
  - Planner Error, Retriever Error, Execution Error
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


_ERROR_WEIGHTS = {
    "wrong_table": 0.20,
    "wrong_column": 0.20,
    "wrong_grain": 0.25,
    "wrong_join": 0.20,
    "wrong_filter": 0.15,
    "wrong_time": 0.15,
    "wrong_scd": 0.25,
    "wrong_semantic": 0.20,
    "planner_error": 0.15,
    "retriever_error": 0.15,
    "execution_error": 0.30,
    "unknown": 0.10,
}


def classify(
    *,
    question: dict,
    ground_truth_df: pd.DataFrame | None,
    ground_truth_sql: str,
    result_df: pd.DataFrame | None,
    result_sql: str,
    tables_used: list[str],
    dag_nodes: list[dict],
    all_tables: list[str],
    execution_error: str | None,
) -> list[dict]:
    errors: list[dict] = []
    if execution_error:
        errors.append({
            "type": "execution_error",
            "detail": execution_error,
        })

    required_tables = set(question.get("required_tables", []))
    used = set(tables_used)

    retriever_missing = required_tables - used
    retriever_extra = used - required_tables
    if retriever_missing:
        errors.append({
            "type": "retriever_error",
            "detail": f"Missing required tables: {sorted(retriever_missing)}",
        })
    if retriever_extra:
        errors.append({
            "type": "retriever_error",
            "detail": f"Extra irrelevant tables used: {sorted(retriever_extra)}",
        })

    gt_tables = _extract_tables(ground_truth_sql)
    if gt_tables and used and used != gt_tables:
        missing_gt = gt_tables - used
        if missing_gt:
            errors.append({
                "type": "wrong_table",
                "detail": f"Missing tables vs ground truth: {sorted(missing_gt)}",
            })

    scd_required = question.get("scd_required", False)
    if scd_required and result_sql:
        has_scd = "valid_from" in result_sql.lower() and "valid_to" in result_sql.lower()
        if not has_scd:
            errors.append({
                "type": "wrong_scd",
                "detail": "SCD2 join expected (valid_from/valid_to) but not found in SQL",
            })

    if ground_truth_df is not None and result_df is not None:
        try:
            gt_cols = set(ground_truth_df.columns)
            res_cols = set(result_df.columns)
            if gt_cols != res_cols:
                missing_cols = gt_cols - res_cols
                extra_cols = res_cols - gt_cols
                detail_parts = []
                if missing_cols:
                    detail_parts.append(f"Missing: {sorted(missing_cols)}")
                if extra_cols:
                    detail_parts.append(f"Extra: {sorted(extra_cols)}")
                errors.append({
                    "type": "wrong_column",
                    "detail": "; ".join(detail_parts),
                })

            gt_rows = len(ground_truth_df)
            res_rows = len(result_df)
            if gt_rows != res_rows:
                row_ratio = res_rows / max(gt_rows, 1)
                if row_ratio < 0.5 or row_ratio > 1.5:
                    errors.append({
                        "type": "wrong_grain",
                        "detail": f"Row count mismatch: expected {gt_rows}, got {res_rows}",
                    })

            if not execution_error and gt_rows > 0 and res_rows > 0:
                if not _dataframes_tolerant_match(ground_truth_df, result_df):
                    errors.append({
                        "type": "wrong_filter",
                        "detail": "Result data mismatch with ground truth",
                    })

        except Exception:
            pass

    if dag_nodes:
        dag_errors = _evaluate_dag(question, dag_nodes)
        errors.extend(dag_errors)

    if ground_truth_sql and result_sql:
        time_errors = _check_time_conditions(ground_truth_sql, result_sql)
        errors.extend(time_errors)

        join_errors = _check_join_strategy(ground_truth_sql, result_sql)
        errors.extend(join_errors)

        semantic_errors = _check_semantic_mapping(question, result_sql)
        errors.extend(semantic_errors)

    if not errors:
        errors.append({"type": "none", "detail": "No errors detected"})

    for e in errors:
        w = _ERROR_WEIGHTS.get(e["type"], 0.10)
        e["weight"] = w

    return errors


def _dataframes_tolerant_match(df_a: pd.DataFrame, df_b: pd.DataFrame, tol: float = 1e-6) -> bool:
    if len(df_a) != len(df_b) or set(df_a.columns) != set(df_b.columns):
        return False
    cols = sorted(df_a.columns)
    a = df_a[cols].reset_index(drop=True)
    b = df_b[cols].reset_index(drop=True)
    try:
        for col in cols:
            if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
                diff = (a[col].fillna(0).values - b[col].fillna(0).values)
                if not np.all(np.abs(diff) < tol):
                    return False
            else:
                if not a[col].fillna("").astype(str).equals(b[col].fillna("").astype(str)):
                    return False
        return True
    except Exception:
        return False


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


def _extract_tables(sql: str) -> set[str]:
    if not sql:
        return set()
    cte_names = _extract_cte_names(sql)
    pattern = re.compile(r'\b(?:FROM|JOIN)\s+(\w+)', re.IGNORECASE)
    all_refs = set(pattern.findall(sql))
    return all_refs - cte_names


def _evaluate_dag(question: dict, dag_nodes: list[dict]) -> list[dict]:
    errors: list[dict] = []
    dag_eval = question.get("dag_evaluation", {})
    if not dag_eval:
        return errors
    node_count = len(dag_nodes)
    minimal = dag_eval.get("minimal_nodes", 0)
    optimal = dag_eval.get("optimal_nodes", 0)
    if node_count < minimal:
        errors.append({
            "type": "planner_error",
            "detail": f"DAG too small: {node_count} nodes, expected at least {minimal}",
        })
    elif node_count > optimal * 2:
        errors.append({
            "type": "planner_error",
            "detail": f"DAG too large: {node_count} nodes, optimal is {optimal}",
        })
    return errors


def _check_time_conditions(gt_sql: str, result_sql: str) -> list[dict]:
    errors: list[dict] = []
    gt_dates = re.findall(r"'\d{4}[-/]\d{2}[-/]\d{2}'", gt_sql)
    res_dates = re.findall(r"'\d{4}[-/]\d{2}[-/]\d{2}'", result_sql)
    if gt_dates and res_dates:
        if set(gt_dates) != set(res_dates):
            errors.append({
                "type": "wrong_time",
                "detail": f"Date range mismatch: GT has {gt_dates}, result has {res_dates}",
            })
    return errors


def _check_join_strategy(gt_sql: str, result_sql: str) -> list[dict]:
    errors: list[dict] = []
    gt_joins = re.findall(r'\b(LEFT|RIGHT|INNER|CROSS|FULL)\s+JOIN', gt_sql, re.IGNORECASE)
    res_joins = re.findall(r'\b(LEFT|RIGHT|INNER|CROSS|FULL)\s+JOIN', result_sql, re.IGNORECASE)
    if gt_joins and res_joins:
        gt_has_left = any("LEFT" in j.upper() for j in gt_joins)
        res_has_left = any("LEFT" in j.upper() for j in res_joins)
        if gt_has_left and not res_has_left:
            errors.append({
                "type": "wrong_join",
                "detail": "LEFT JOIN expected but not found in result SQL",
            })
    return errors


def _check_semantic_mapping(question: dict, result_sql: str) -> list[dict]:
    errors: list[dict] = []
    semantic = question.get("semantic_mapping", {})
    for concept, expected_col in semantic.items():
        if isinstance(expected_col, str):
            table_col = expected_col.split(".")
            col_name = table_col[-1] if len(table_col) > 1 else table_col[0]
            if col_name not in result_sql:
                errors.append({
                    "type": "wrong_semantic",
                    "detail": f"Semantic concept '{concept}' → expected column '{col_name}' not found in SQL",
                })
    return errors


def summarize(errors_list: list[list[dict]]) -> dict:
    type_counts: dict[str, int] = {}
    total_weight = 0.0
    total_errors = 0
    for errors in errors_list:
        for e in errors:
            if e["type"] == "none":
                continue
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
            total_weight += e.get("weight", 0.10)
            total_errors += 1

    if total_errors == 0:
        return {"error_count": 0, "error_types": {}, "avg_error_weight": 0.0, "is_clean": True}

    return {
        "error_count": total_errors,
        "error_types": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
        "avg_error_weight": round(total_weight / total_errors, 3),
        "is_clean": False,
    }
