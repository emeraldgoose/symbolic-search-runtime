#!/usr/bin/env python3
"""
validate_real.py — Real end-to-end validation for syrch

Runs the full pipeline against real LLM + real databases across 5 difficulty levels.
Reports structured output for each test case.

Usage:
    python validate_real.py                           # all levels
    python validate_real.py --level 3                 # specific level
    python validate_real.py --quick                   # L1-L2 only
    python validate_real.py --interactive --level 5   # test clarification
    python validate_real.py --skip-cache              # bypass cache
    python validate_real.py --question "..." --db orders_10dim.sqlite  # custom
    python validate_real.py --verbose                 # detailed output
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from syrch.core.config import ExecutionConfig, LLMConfig
from syrch.core.models import (
    FinalSolution,
    NodeResult,
    ProblemSpec,
    TaskDAG,
)
from syrch.executors.sqlite_executor import SQLiteExecutor
from syrch.llm.openai_llm import OpenAILLM
from syrch.llm.base import BaseLLM
from syrch.llm.cache import CachedLLM, CentralCache
from syrch.executors.cached_executor import CachedExecutor
from syrch.search.pipeline import run_pipeline
from syrch.search.clarify import find_worst_ambiguity, generate_question

# ── Constants ──────────────────────────────────────────────────────────

LLM_MODEL = "minimax-m3:cloud"
LLM_FALLBACK_MODEL = "lfm2.5-thinking:latest"
LLM_BASE_URL = "http://localhost:8000/v1"
ORDERS_DB = "orders_10dim.sqlite"
WIKI_DB = "wikipedia_clickstream.sqlite"
AMBIGUITY_THRESHOLD = 0.25
AMBIGUOUS_KEYWORDS = {
    "best": 0.28,
    "most important": 0.28,
    "performance": 0.15,
    "good": 0.20,
    "bad": 0.20,
    "better": 0.25,
    "worse": 0.25,
    "meaningful": 0.20,
    "relevant": 0.20,
    "significant": 0.20,
    "important": 0.20,
}

# ── Test Cases ─────────────────────────────────────────────────────────

TEST_CASES = {
    "L1_easy": [
        {
            "question": "How many orders are in orders_10dim?",
            "db": ORDERS_DB,
            "expected_pattern": "COUNT",
            "min_confidence": 0.7,
            "description": "Simple COUNT aggregation",
        },
        {
            "question": "What is the total revenue across all orders in orders_10dim?",
            "db": ORDERS_DB,
            "expected_pattern": "SUM",
            "min_confidence": 0.7,
            "description": "Simple SUM aggregation",
        },
        {
            "question": "List all unique order priorities",
            "db": ORDERS_DB,
            "expected_pattern": "DISTINCT",
            "min_confidence": 0.5,
            "description": "SELECT DISTINCT",
        },
    ],
    "L2_medium": [
        {
            "question": "What is the average total price per order priority?",
            "db": ORDERS_DB,
            "expected_pattern": "GROUP BY",
            "min_confidence": 0.6,
            "description": "Grouped AVG with GROUP BY",
        },
        {
            "question": "Which clerk has the highest total sales?",
            "db": ORDERS_DB,
            "expected_pattern": "LIMIT",
            "min_confidence": 0.6,
            "description": "Top-N GROUP BY SUM",
        },
        {
            "question": "What is the total revenue by year?",
            "db": ORDERS_DB,
            "expected_pattern": "GROUP BY",
            "min_confidence": 0.6,
            "description": "GROUP BY year",
        },
    ],
    "L3_complex": [
        {
            "question": "What is the revenue trend by year and quarter?",
            "db": ORDERS_DB,
            "expected_pattern": "DAG",
            "min_confidence": 0.5,
            "min_nodes": 2,
            "description": "Multi-dimensional DAG: yearly → quarterly breakdown",
        },
        {
            "question": "Compare low-price vs high-price order volumes across years",
            "db": ORDERS_DB,
            "expected_pattern": "DAG",
            "min_confidence": 0.4,
            "min_nodes": 2,
            "description": "Price bucket → trend decomposition",
        },
    ],
    "L4_very_complex": [
        {
            "question": "Find top 10 clerks with highest avg price and their order status distribution",
            "db": ORDERS_DB,
            "expected_pattern": "DAG",
            "min_confidence": 0.3,
            "min_nodes": 2,
            "description": "Multi-layer DAG with status breakdown",
        },
        {
            "question": "What is the relationship between order status and total price trends over years?",
            "db": ORDERS_DB,
            "expected_pattern": "DAG",
            "min_confidence": 0.3,
            "min_nodes": 2,
            "description": "Cross-table relationship DAG",
        },
    ],
    "L5_ambiguous": [
        {
            "question": "Which orders are the best?",
            "db": ORDERS_DB,
            "expected_pattern": "AMBIGUOUS",
            "min_confidence": 0.0,
            "description": "Intentionally vague — should trigger low confidence",
        },
        {
            "question": "What is the most important metric for order performance?",
            "db": ORDERS_DB,
            "expected_pattern": "AMBIGUOUS",
            "min_confidence": 0.0,
            "description": "Vague metric — should trigger low confidence",
        },
    ],
}

LEVEL_ORDER = ["L1_easy", "L2_medium", "L3_complex", "L4_very_complex", "L5_ambiguous"]

# ── Test Result ────────────────────────────────────────────────────────

@dataclass
class TestResult:
    level: str
    question: str
    db: str
    status: str = "ERROR"
    execution_time: float = 0.0
    solution: FinalSolution | None = None
    dag: TaskDAG | None = None
    results: dict[str, NodeResult] | None = None
    error: str | None = None
    checks: list[tuple[str, bool]] = field(default_factory=list)
    ambiguity_result: tuple[str, float] | None = None
    clarification_question: str | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    keyword_ambiguity: float = 0.0


# ── Color helpers ─────────────────────────────────────────────────────

def _c(text: str, code: str) -> str:
    codes = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "cyan": "\033[96m",
        "dim": "\033[2m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"


def _status_tag(status: str) -> str:
    if status == "PASS":
        return _c("  PASS  ", "green")
    if status == "FAIL":
        return _c("  FAIL  ", "red")
    if status == "AMBIGUOUS":
        return _c("AMBIGUOUS", "yellow")
    return _c(f" {status} ", "red")


# ── Validation Logic ───────────────────────────────────────────────────

def _keyword_ambiguity(question: str) -> float:
    q_lower = question.lower()
    score = 0.0
    for kw, weight in AMBIGUOUS_KEYWORDS.items():
        if kw in q_lower:
            score = max(score, weight)
    return min(score, 0.6)


def _run_case(
    question: str,
    db_path: str,
    config: ExecutionConfig,
    llm: BaseLLM,
    executor: SQLiteExecutor,
    cache_obj: CentralCache | None = None,
) -> TestResult:
    result = TestResult(level="custom", question=question, db=db_path)
    result.keyword_ambiguity = _keyword_ambiguity(question)
    cache_hits_before = cache_obj.hit_count if cache_obj else 0
    t0 = time.time()
    try:
        schema = executor.get_schema()
        all_schemas = [executor.get_schema(t) for t in executor.list_tables()]
        problem = ProblemSpec(question=question, schema=schema, all_schemas=all_schemas)
        solution, dag, _results = run_pipeline(llm, executor, config, problem)
        result.execution_time = time.time() - t0
        result.solution = solution
        result.dag = dag
        result.results = _results

        if cache_obj:
            result.cache_hits = cache_obj.hit_count - cache_hits_before

        # Detect ambiguity (boost with keyword heuristic)
        worst = find_worst_ambiguity(_results)
        if worst:
            wid, wscore = worst
            wscore = max(wscore, result.keyword_ambiguity)
            result.ambiguity_result = (wid, wscore)
        else:
            result.ambiguity_result = worst

        if solution.confidence >= 0.5:
            result.status = "PASS"
        elif solution.confidence >= 0.3:
            result.status = "FAIL"
            result.error = f"Low confidence: {solution.confidence:.3f}"
        else:
            result.status = "AMBIGUOUS"
            result.error = f"Very low confidence: {solution.confidence:.3f}"
    except Exception as e:
        result.execution_time = time.time() - t0
        result.error = f"{type(e).__name__}: {e}"
        result.status = "ERROR"
    return result


def _run_case_with_check(
    case: dict,
    config: ExecutionConfig,
    llm: BaseLLM,
    executor: SQLiteExecutor,
    interactive: bool = False,
    cache_obj: CentralCache | None = None,
) -> TestResult:
    question = case["question"]
    db_path = case["db"]
    config.question = question
    config.db_path = db_path
    result = _run_case(question, db_path, config, llm, executor, cache_obj=cache_obj)

    # Determine expected behavior
    expected = case.get("expected_pattern", "")
    min_conf = case.get("min_confidence", 0.0)
    min_nodes = case.get("min_nodes", 1)

    checks: list[tuple[str, bool]] = []

    if expected == "AMBIGUOUS":
        # For ambiguous questions: expect low confidence or ambiguity detected
        is_ambiguous = result.status == "AMBIGUOUS" or (
            result.solution is not None and result.solution.confidence < 0.4
        )
        has_ambiguity_signal = (
            result.ambiguity_result is not None
            and result.ambiguity_result[1] > AMBIGUITY_THRESHOLD
        )
        checks.append(("low_confidence", is_ambiguous))
        checks.append(("ambiguity_detected", has_ambiguity_signal))

        if is_ambiguous or has_ambiguity_signal:
            result.status = "AMBIGUOUS"
            result.checks = checks
            if interactive and has_ambiguity_signal:
                _handle_interactive_clarification(result, llm, config, executor, case)
            return result

        # If unexpectedly high confidence, set as FAIL for investigation
        if result.solution is not None and result.solution.confidence >= 0.5:
            result.status = "FAIL"
            result.error = (
                f"Expected ambiguous but got confidence={result.solution.confidence:.3f}"
            )
            result.checks = checks
            return result
    else:
        # Normal questions: check confidence and DAG
        conf_ok = result.solution is not None and result.solution.confidence >= min_conf
        checks.append((f"confidence>={min_conf}", conf_ok or result.status == "PASS"))

        if expected == "DAG" and result.dag is not None:
            nodes_ok = len(result.dag.nodes) >= min_nodes
            checks.append((f"nodes>={min_nodes}", nodes_ok))
            if nodes_ok and result.status == "PASS":
                result.status = "PASS"
            elif nodes_ok:
                result.status = "PASS" if conf_ok else "FAIL"
            else:
                if result.status == "PASS":
                    result.status = "FAIL"
                    result.error = (result.error or "") + f" Only {len(result.dag.nodes)} nodes"
        else:
            if expected != "DAG" and result.status == "PASS" and conf_ok:
                result.status = "PASS"
            elif result.status == "PASS" and not conf_ok:
                result.status = "FAIL"
                result.error = (
                    result.error or ""
                ) + f" Confidence {result.solution.confidence:.3f} < {min_conf}"

        # Check for pattern in SQL
        if expected in ("COUNT", "SUM", "DISTINCT", "GROUP BY", "LIMIT"):
            sql = (result.solution.sql if result.solution else "").upper()
            pattern_ok = expected in sql
            checks.append((f"SQL contains '{expected}'", pattern_ok))

        result.checks = checks

    # Set error if no valid status
    if result.status == "PASS":
        pass
    elif result.status == "ERROR":
        pass
    else:
        result.status = "FAIL"
        if result.error is None:
            result.error = "Checks failed"

    return result


def _handle_interactive_clarification(
    result: TestResult,
    llm: BaseLLM,
    config: ExecutionConfig,
    executor: SQLiteExecutor,
    case: dict,
) -> None:
    """When interactive and ambiguous, generate question and re-run."""
    if not result.results or not result.dag:
        return

    worst_id, worst_score = result.ambiguity_result or (None, 0.0)
    if worst_id is None or worst_score <= AMBIGUITY_THRESHOLD:
        return

    schema = executor.get_schema()
    all_schemas = [executor.get_schema(t) for t in executor.list_tables()]
    problem = ProblemSpec(
        question=case["question"], schema=schema, all_schemas=all_schemas
    )
    q = generate_question(llm, problem, result.dag, worst_id, result.results)
    result.clarification_question = q

    print(f"\n{_c('? Clarification needed', 'yellow')}")
    print(f"  Question: {_c(q, 'cyan')}")
    print(f"  Worst node: {_c(worst_id, 'yellow')} (ambiguity: {worst_score:.3f})")
    answer = input("  > ")
    if not answer.strip():
        answer = "By total sales descending"
        print(f"  (using default: '{answer}')")

    new_question = f"{case['question']}\n[User clarification: {answer}]"
    new_config = ExecutionConfig(
        question=new_question,
        db_path=case["db"],
        executor_type=config.executor_type,
        max_depth=config.max_depth,
        max_attempts_per_node=config.max_attempts_per_node,
        high_confidence=config.high_confidence,
        token_budget=config.token_budget,
        verbose=config.verbose,
        cache_enabled=False,  # Don't cache the re-run
        cache_ttl=config.cache_ttl,
        calibration_enabled=config.calibration_enabled,
        interactive=config.interactive,
        ambiguity_threshold=config.ambiguity_threshold,
        llm=config.llm,
    )

    refined = _run_case(new_question, case["db"], new_config, llm, executor)
    result.solution = refined.solution
    result.dag = refined.dag
    result.results = refined.results
    result.execution_time += refined.execution_time
    result.status = refined.status
    result.error = refined.error
    if refined.solution and refined.solution.confidence >= 0.5:
        result.status = "PASS"
        result.error = f"Clarified: confidence improved to {refined.solution.confidence:.3f}"


# ── Print Helpers ──────────────────────────────────────────────────────

def print_case_result(result: TestResult, idx: int, total: int, verbose: bool) -> None:
    sol = result.solution
    conf = sol.confidence if sol else 0.0
    cost = sol.token_cost if sol else 0
    n_dag = len(result.dag.nodes) if result.dag else 0

    print(f"\n  [{idx}/{total}] {result.question}")
    print(f"       DB: {result.db}")

    if result.dag and n_dag > 0:
        deps = []
        for nid, node in result.dag.nodes.items():
            if node.depends_on:
                deps.append(f"{nid} ← {', '.join(node.depends_on)}")
            else:
                deps.append(nid)
        print(f"       DAG: {n_dag} node(s) — {' | '.join(deps)}")

    if verbose and result.results:
        for nid, nr in result.results.items():
            data_info = ""
            if nr.data is not None and not nr.data.empty:
                data_info = f" ({len(nr.data)} rows x {len(nr.data.columns)} cols)"
            print(
                f"         ├─ {nid}: conf={nr.confidence:.3f} cost={nr.cost_tokens}"
                f" amb={nr.ambiguity_score:.3f}{data_info}"
            )
            for rp in nr.reasoning_paths:
                sql_short = rp.sql[:100].replace("\n", " ")
                print(f"         │  path {rp.path_id}: conf={rp.confidence:.3f} {sql_short}")

    if result.ambiguity_result:
        wid, wscore = result.ambiguity_result
        print(f"       Worst ambiguity: {wid} (score={wscore:.3f})")

    if result.clarification_question:
        print(f"       Clarification Q: {result.clarification_question}")

    checks_str = " | ".join(
        f"{'✅' if ok else '❌'} {name}" for name, ok in result.checks
    )
    cache_tag = ""
    if result.cache_hits > 0:
        cache_tag = f" \033[2m[cache:{result.cache_hits}h]\033[0m"
    kw_tag = ""
    if result.keyword_ambiguity > 0:
        kw_tag = f" \033[2m[kw-amb:{result.keyword_ambiguity:.2f}]\033[0m"
    print(
        f"       {_status_tag(result.status)}"
        f" conf={conf:.3f} cost={cost} time={result.execution_time:.1f}s{cache_tag}{kw_tag}"
    )
    if checks_str:
        print(f"       Checks: {checks_str}")
    if result.error and result.status != "AMBIGUOUS":
        print(f"       {_c(result.error[:120], 'red')}")

    if verbose and sol and sol.answer:
        answer_short = sol.answer[:200].replace("\n", " ")
        print(f"       Answer: {_c(answer_short, 'dim')}")
    if verbose and sol and sol.sql:
        print(f"       SQL: {_c(sol.sql[:200].replace(chr(10), ' '), 'dim')}")


def print_summary(level_results: dict[str, list[TestResult]], elapsed: float) -> None:
    print(f"\n{_c('═' * 72, 'dim')}")
    print(_c("SUMMARY", "bold"))
    print(_c('═' * 72, 'dim'))

    total = 0
    passed = 0
    failed = 0
    errors = 0
    ambiguous = 0

    for level in LEVEL_ORDER:
        results = level_results.get(level, [])
        if not results:
            continue
        level_pass = sum(1 for r in results if r.status == "PASS")
        level_fail = sum(1 for r in results if r.status == "FAIL")
        level_err = sum(1 for r in results if r.status == "ERROR")
        level_amb = sum(1 for r in results if r.status == "AMBIGUOUS")
        total += len(results)
        passed += level_pass
        failed += level_fail
        errors += level_err
        ambiguous += level_amb

        tag = _c("✅", "green") if level_err == 0 and level_fail == 0 else _c("⚠️", "yellow")
        status_line = f"{tag} {level}: {level_pass}P/{level_fail}F/{level_err}E"
        if level_amb:
            status_line += f"/{level_amb}A"
        print(f"  {status_line}")

    print()
    rate = (passed / total * 100) if total > 0 else 0
    summary = (
        f"  Total: {total} | "
        f"{_c(f'{passed} PASS', 'green')} | "
        f"{_c(f'{failed} FAIL', 'red')} | "
        f"{_c(f'{errors} ERROR', 'red')} | "
        f"{_c(f'{ambiguous} AMBIGUOUS', 'yellow')}"
    )
    print(summary)
    print(f"  Pass rate: {rate:.1f}% | Total time: {elapsed:.1f}s")
    print(_c('═' * 72, 'dim'))

    if failed > 0 or errors > 0:
        print(_c("\n❌ Failures:", "red"))
        for level in LEVEL_ORDER:
            for r in level_results.get(level, []):
                if r.status in ("FAIL", "ERROR"):
                    print(f"  [{level}] {r.question}")
                    print(f"    {r.error[:150] if r.error else 'No error detail'}")
        sys.exit(1)
    elif ambiguous > 0:
        print(_c("\n⚠️  Ambiguous cases (expected for L5):", "yellow"))
        for level in LEVEL_ORDER:
            for r in level_results.get(level, []):
                if r.status == "AMBIGUOUS":
                    print(f"  [{level}] {r.question}")
        print(_c("These are expected for intentionally vague questions.", "dim"))


# ── Main ───────────────────────────────────────────────────────────────

def create_llm(
    skip_cache: bool = False,
    model: str = LLM_MODEL,
    base_url: str = LLM_BASE_URL,
    api_key: str | None = None,
    cache: CentralCache | None = None,
) -> BaseLLM:
    if api_key is None:
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    llm = OpenAILLM(model=model, api_key=api_key, base_url=base_url)
    if not skip_cache:
        if cache is None:
            cache = CentralCache(ttl=86400)
        llm = CachedLLM(llm, cache, model=model)
    return llm


def create_executor(
    db_path: str,
    skip_cache: bool = False,
    cache: CentralCache | None = None,
) -> SQLiteExecutor:
    executor = SQLiteExecutor(db_path)
    if not skip_cache:
        if cache is None:
            cache = CentralCache(ttl=86400)
        executor = CachedExecutor(executor, cache)
    return executor


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate syrch on real data")
    parser.add_argument("--level", type=int, choices=range(1, 6), help="Run specific level")
    parser.add_argument("--quick", action="store_true", help="Skip L4, L5")
    parser.add_argument("--skip-cache", action="store_true", help="Disable LLM/SQL cache")
    parser.add_argument(
        "--interactive", action="store_true", help="Test clarification flow (L5)"
    )
    parser.add_argument("--question", type=str, help="Custom question (single run)")
    parser.add_argument("--db", type=str, default=ORDERS_DB, help="Database for custom run")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")
    parser.add_argument("--max-depth", type=int, default=3, help="Max D&C recursion depth")
    parser.add_argument("--model", type=str, default=None, help="LLM model name (overrides default)")
    args = parser.parse_args()

    model = args.model or LLM_MODEL
    _cache = CentralCache(ttl=86400) if not args.skip_cache else None
    llm = create_llm(skip_cache=args.skip_cache, model=model, cache=_cache)
    executor = create_executor(args.db, skip_cache=args.skip_cache, cache=_cache)

    overall_start = time.time()
    level_results: dict[str, list[TestResult]] = {}

    # Single custom run
    if args.question:
        case = {
            "question": args.question,
            "db": args.db,
            "expected_pattern": "CUSTOM",
            "min_confidence": 0.0,
            "min_nodes": 0,
            "description": "Custom single run",
        }
        config = ExecutionConfig(
            question=args.question,
            db_path=args.db,
            max_depth=args.max_depth,
            verbose=args.verbose,
            calibration_enabled=True,
        )
        result = _run_case_with_check(case, config, llm, executor, cache_obj=_cache)
        print_case_result(result, 1, 1, verbose=args.verbose)
        print_summary({"custom": [result]}, time.time() - overall_start)
        executor.close()
        return

    # Select levels
    levels_to_run: list[str] = []
    if args.level:
        levels_to_run = [LEVEL_ORDER[args.level - 1]]
    else:
        levels_to_run = LEVEL_ORDER[:2] if args.quick else LEVEL_ORDER

    # Run each level
    for level_name in levels_to_run:
        cases = TEST_CASES[level_name]
        print(f"\n{_c('━' * 72, 'dim')}")
        print(f"{_c(f'  {level_name}', 'bold')} — {cases[0]['description'].split(' —')[0]}...")
        print(_c(f'  {len(cases)} test(s)', 'dim'))

        config = ExecutionConfig(
            question="",
            db_path=ORDERS_DB,
            max_depth=args.max_depth,
            verbose=args.verbose,
            calibration_enabled=True,
            interactive=args.interactive,
            ambiguity_threshold=AMBIGUITY_THRESHOLD,
        )

        results: list[TestResult] = []
        for i, case in enumerate(cases, 1):
            case_executor = create_executor(case["db"], skip_cache=args.skip_cache, cache=_cache)
            result = _run_case_with_check(
                case, config, llm, case_executor, interactive=args.interactive, cache_obj=_cache
            )
            case_executor.close()
            results.append(result)
            print_case_result(result, i, len(cases), verbose=args.verbose)

        level_results[level_name] = results

    executor.close()
    print_summary(level_results, time.time() - overall_start)


if __name__ == "__main__":
    main()
