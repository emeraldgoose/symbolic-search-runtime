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
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
_QUESTIONS_DIR = Path(__file__).resolve().parent.parent / "questions"
_GENERATED_DIR = Path(__file__).resolve().parent.parent / "generated"
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


def generate_ground_truth(profile_name: str, questions_name: str, db_path: str | None) -> list[dict]:
    profile = load_profile(profile_name)
    questions = load_questions(questions_name)

    if db_path:
        db = Path(db_path)
    else:
        sqlite_cfg = profile.get("output", {}).get("sqlite", {})
        db = Path(sqlite_cfg["path"])

    if not db.exists():
        print(f"Database not found: {db}")
        sys.exit(1)

    conn = sqlite3.connect(str(db))
    results = []

    for q in questions:
        qid = q["id"]
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
            result_path = _ANSWERS_DIR / questions_name / f"{qid}.csv"

        result_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.read_sql_query(gt_sql, conn)
            df.to_csv(result_path, index=False)
            print(f"  OK    {qid}: {len(df)} rows → {result_path}")
            results.append({"id": qid, "rows": len(df), "columns": list(df.columns), "path": str(result_path)})
        except Exception as e:
            print(f"  ERROR {qid}: {e}")
            results.append({"id": qid, "error": str(e)})

    conn.close()

    summary_path = _ANSWERS_DIR / questions_name / "_summary.json"
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
    parser.add_argument("--db", help="Override DB path (optional)")
    args = parser.parse_args()
    generate_ground_truth(args.profile, args.questions, args.db)


if __name__ == "__main__":
    main()
