from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pandas as pd

from syrch.core.config import ExecutionConfig, LLMConfig
from syrch.core.models import FinalSolution
from syrch.eval.metrics import EvaluationMetrics, evaluate
from syrch.executors.base import BaseExecutor
from syrch.executors.sqlite_executor import SQLiteExecutor


@dataclass
class BenchmarkProblem:
    id: str = ""
    question: str = ""
    db: str = ""
    expected_data: str | None = None
    expected_sql: str | None = None
    expected_answer: str | None = None


@dataclass
class BenchmarkResult:
    problem: BenchmarkProblem
    solution: FinalSolution | None = None
    metrics: EvaluationMetrics | None = None
    error: str | None = None
    duration: float = 0.0


def _normalize_db_path(db_path: str | list[str]) -> list[str]:
    if isinstance(db_path, str):
        return [p.strip() for p in db_path.split(",")]
    return list(db_path)


def _create_executor(executor_type: str, db_path: str | list[str]) -> BaseExecutor:
    tables = _normalize_db_path(db_path)
    if executor_type == "sqlite":
        return SQLiteExecutor(tables[0])
    if executor_type == "databricks-sql":
        try:
            from syrch.executors.databricks_executor import DatabricksExecutor
        except ImportError:
            raise ImportError(
                "executor_type='databricks-sql' requires: pip install syrch[databricks-sql]"
            )
        return DatabricksExecutor(tables=tables)
    if executor_type == "spark":
        try:
            from syrch.executors.spark_executor import SparkExecutor
        except ImportError:
            raise ImportError(
                "executor_type='spark' requires: pip install syrch[spark]"
            )
        return SparkExecutor(tables=tables)
    if executor_type == "jdbc":
        from syrch.executors.jdbc_executor import JDBCExecutor
        return JDBCExecutor(tables[0])
    raise ValueError(f"Unknown executor type: {executor_type}")


def load_benchmark(path: str) -> list[BenchmarkProblem]:
    problems: list[BenchmarkProblem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            problems.append(BenchmarkProblem(**data))
    return problems


def _make_config(
    problem: BenchmarkProblem,
    llm_config: LLMConfig | None = None,
    overrides: dict | None = None,
) -> ExecutionConfig:
    cfg = dict(
        question=problem.question,
        db_path=problem.db,
        verbose=False,
    )
    if overrides:
        cfg.update(overrides)
    config = ExecutionConfig(**cfg)  # type: ignore[arg-type]
    if llm_config is not None:
        config.llm = llm_config
    return config


def run_single(
    problem: BenchmarkProblem,
    llm_config: LLMConfig | None = None,
    config_overrides: dict | None = None,
) -> BenchmarkResult:
    from syrch.core.models import ProblemSpec
    from syrch.executors.cached_executor import CachedExecutor
    from syrch.llm.anthropic_llm import AnthropicLLM
    from syrch.llm.cache import CachedLLM, CentralCache
    from syrch.llm.openai_llm import OpenAILLM
    from syrch.search.pipeline import run_pipeline

    config = _make_config(problem, llm_config, config_overrides)
    t0 = time.time()
    try:
        llm_cfg = config.llm
        if llm_cfg.provider == "openai":
            llm = OpenAILLM(model=llm_cfg.model, api_key=llm_cfg.api_key, base_url=llm_cfg.base_url)
        elif llm_cfg.provider == "anthropic":
            llm = AnthropicLLM(model=llm_cfg.model, api_key=llm_cfg.api_key)  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown LLM provider: {llm_cfg.provider}")

        cache: CentralCache | None = None
        if config.cache_enabled:
            cache = CentralCache(ttl=config.cache_ttl)
            llm = CachedLLM(llm, cache, model=config.llm.model, temperature=config.llm.temperature)  # type: ignore[assignment]

        executor = _create_executor(config.executor_type, config.db_path)
        if cache:
            executor = CachedExecutor(executor, cache)
        schema = executor.get_schema()
        prob = ProblemSpec(question=problem.question, schema=schema)
        solution, _dag, _results = run_pipeline(llm, executor, config, prob)
        executor.close()
        duration = time.time() - t0

        expected_df = None
        if problem.expected_data:
            expected_df = pd.read_csv(problem.expected_data)
        metrics = evaluate(solution, expected_data=expected_df) if expected_df is not None else evaluate(solution)
        return BenchmarkResult(
            problem=problem, solution=solution, metrics=metrics, duration=duration
        )
    except Exception as e:
        duration = time.time() - t0
        return BenchmarkResult(problem=problem, error=str(e), duration=duration)


def run_benchmark(
    problems: list[BenchmarkProblem],
    llm_config: LLMConfig | None = None,
    config_overrides: dict | None = None,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for p in problems:
        result = run_single(p, llm_config, config_overrides)
        results.append(result)
    return results
