#!/usr/bin/env python3
"""
Validate benchmark databases and evaluate question coverage.

Usage:
    python -m benchmark.evaluator.validate_business --profile small
    python benchmark/evaluator/validate_business.py --profile small --questions pilot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import sqlite3
import yaml


_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
_QUESTIONS_DIR = Path(__file__).resolve().parent.parent / "questions"
_GENERATED_DIR = Path(__file__).resolve().parent.parent / "generated"


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


class BenchmarkValidator:
    def __init__(self, db_path: str | Path, questions: list[dict], profile: dict):
        self.db_path = Path(db_path)
        self.questions = questions
        self.profile = profile
        self.conn = sqlite3.connect(str(self.db_path))
        self.results: list[dict] = []

    def run_all(self) -> dict:
        start = time.time()
        for q in self.questions:
            result = self._evaluate_question(q)
            self.results.append(result)
        elapsed = time.time() - start

        return self._summarize(elapsed)

    def _evaluate_question(self, q: dict) -> dict:
        qid = q["id"]
        required_tables = q.get("required_tables", [])
        result = {
            "id": qid,
            "question": q.get("question_en", q["question"]),
            "escalation_group": q.get("escalation_group", ""),
            "escalation_level": q.get("escalation_level", 1),
            "checks": {},
        }

        # 1. Required tables exist?
        existing_tables = self._list_tables()
        for t in required_tables:
            exists = t in existing_tables
            result["checks"][f"table_exists:{t}"] = exists

        # 2. Required domains available
        for domain in q.get("required_domains", []):
            has_domain = any(domain in t for t in existing_tables)
            result["checks"][f"domain_available:{domain}"] = has_domain

        # 3. Ground truth SQL executes?
        gt_sql = q.get("ground_truth_sql", "")
        if gt_sql:
            try:
                df = self.conn.execute(gt_sql).fetchall()
                result["checks"]["ground_truth_executes"] = True
                result["ground_truth_rows"] = len(df)
                result["ground_truth_cols"] = len(df[0]) if df else 0
                result["ground_truth_summary"] = str(df[:3]) if df else "empty"
            except Exception as e:
                result["checks"]["ground_truth_executes"] = False
                result["ground_truth_error"] = str(e)

        # 4. Grain selection test
        expected_grain = q.get("grain", "")
        result["grain"] = expected_grain

        # 5. Schema retrieval difficulty
        noise_count = len(existing_tables) - len(required_tables)
        result["retriever_difficulty"] = {
            "total_tables": len(existing_tables),
            "required_tables": len(required_tables),
            "noise_tables": noise_count,
            "recall_possible": all(
                t in existing_tables for t in required_tables
            ),
        }

        # 6. SQL pattern validation
        for pattern in q.get("expected_sql_patterns", []):
            found = pattern.lower() in gt_sql.lower()
            result["checks"][f"expected_pattern:{pattern}"] = found

        for pattern in q.get("forbidden_sql_patterns", []):
            found = pattern.lower() in gt_sql.lower()
            result["checks"][f"forbidden_pattern:{pattern}"] = not found

        return result

    def _list_tables(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return [r[0] for r in rows]

    def _summarize(self, elapsed: float) -> dict:
        total = len(self.results)
        domains: dict[str, dict] = {}
        groups: dict[str, list] = {}

        for r in self.results:
            # Domains
            for check_name, ok in r["checks"].items():
                if check_name.startswith("domain_available:"):
                    domain = check_name.split(":", 1)[1]
                    domains.setdefault(domain, {"total": 0, "ok": 0})
                    domains[domain]["total"] += 1
                    if ok:
                        domains[domain]["ok"] += 1

            # Escalation groups
            g = r.get("escalation_group", "")
            groups.setdefault(g, []).append(r)

        retriever_checks = []
        sql_checks = []
        for r in self.results:
            for cname, ok in r["checks"].items():
                if cname.startswith("table_exists:"):
                    retriever_checks.append(ok)
                if cname == "ground_truth_executes":
                    sql_checks.append(ok)

        return {
            "profile": str(self.db_path),
            "total_questions": total,
            "total_tables": len(self._list_tables()),
            "time_seconds": round(elapsed, 2),
            "retriever": {
                "recall_score": (
                    sum(retriever_checks) / len(retriever_checks)
                    if retriever_checks
                    else 0
                ),
                "checks_passed": sum(retriever_checks),
                "checks_total": len(retriever_checks),
            },
            "sql": {
                "execution_rate": (
                    sum(sql_checks) / len(sql_checks) if sql_checks else 0
                ),
                "queries_valid": sum(sql_checks),
                "queries_total": len(sql_checks),
            },
            "domain_coverage": {
                d: f"{v['ok']}/{v['total']}" for d, v in sorted(domains.items())
            },
            "escalation_groups": {
                g: len(v) for g, v in sorted(groups.items())
            },
            "questions": self.results,
        }


def print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  Benchmark: {report['profile']}")
    print("=" * 60)
    print(f"  Questions:      {report['total_questions']}")
    print(f"  Tables in DB:   {report['total_tables']}")
    print(f"  Time:           {report['time_seconds']}s")
    print()
    print(f"  Retriever Recall:  {report['retriever']['recall_score']:.0%} "
          f"({report['retriever']['checks_passed']}/{report['retriever']['checks_total']})")
    print(f"  SQL Execution:     {report['sql']['execution_rate']:.0%} "
          f"({report['sql']['queries_valid']}/{report['sql']['queries_total']})")
    print()

    print("  Domain Coverage:")
    for d, cov in sorted(report["domain_coverage"].items()):
        print(f"    {d:<20} {cov}")
    print()

    print("  Escalation Groups:")
    for g, cnt in sorted(report["escalation_groups"].items()):
        print(f"    {g:<25} {cnt} questions")
    print()

    print("  Per-Question Results:")
    for q in report["questions"]:
        checks_pass = sum(1 for v in q["checks"].values() if v)
        checks_total = len(q["checks"])
        status = "✅" if checks_pass == checks_total else "⚠️" if checks_pass > 0 else "❌"
        print(f"    {status} {q['id']:<5} {q['question'][:50]:<52} "
              f"{checks_pass}/{checks_total}")
        if q.get("ground_truth_error"):
            print(f"          SQL Error: {q['ground_truth_error'][:60]}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate syrch benchmark")
    parser.add_argument("--profile", default="small", help="Profile name")
    parser.add_argument("--questions", default="pilot", help="Questions file (without .json)")
    parser.add_argument("--db", help="Override DB path (optional)")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    questions = load_questions(args.questions)

    if args.db:
        db_path = Path(args.db)
    else:
        sqlite_cfg = profile.get("output", {}).get("sqlite", {})
        db_path = Path(sqlite_cfg["path"])

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run: python benchmark/generator/gen_business_db.py --profile {args.profile}")
        sys.exit(1)

    validator = BenchmarkValidator(db_path, questions, profile)
    report = validator.run_all()
    print_report(report)


if __name__ == "__main__":
    main()
