from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from syrch.core.config import LLMConfig, merge_config
from syrch.core.logging import LogConfig, setup_logging
from syrch.core.models import NodeResult, ProblemSpec
from syrch.eval.runner import _create_executor
from syrch.executors.cached_executor import CachedExecutor
from syrch.llm.base import BaseLLM
from syrch.llm.cache import CachedLLM, CentralCache
from syrch.llm.openai_llm import OpenAILLM
from syrch.llm.anthropic_llm import AnthropicLLM
from syrch.search.pipeline import run_pipeline


@dataclass
class SearchResult:
    answer: str
    sql: str
    confidence: float
    token_cost: int
    data: pd.DataFrame | None = None
    tree: list[NodeResult] = field(default_factory=list, repr=False)
    dag_nodes: list[dict] = field(default_factory=list, repr=False)
    tables_used: list[str] = field(default_factory=list, repr=False)


def _create_llm(config: LLMConfig) -> BaseLLM:
    if config.provider == "openai":
        return OpenAILLM(model=config.model, api_key=config.api_key, base_url=config.base_url)
    if config.provider == "anthropic":
        return AnthropicLLM(model=config.model, api_key=config.api_key)
    raise ValueError(f"Unknown LLM provider: {config.provider}")


def query(
    question: str,
    db_path: str | list[str] = "orders_10dim.sqlite",
    executor_type: str = "sqlite",
    *,
    model: str = "qwen3.5-4b-4bit",
    base_url: str | None = None,
    api_key: str | None = None,
    llm_provider: str = "openai",
    max_depth: int = 3,
    max_attempts: int = 3,
    high_confidence: float = 0.85,
    token_budget: int = 100_000,
    verbose: bool = False,
    cache: bool = True,
    cache_ttl: int = 86400,
    calibration: bool = True,
    interactive: bool = False,
    ambiguity_threshold: float = 0.35,
    max_concurrency: int = 5,
    config_file: str | None = None,
    log_level: str = "WARNING",
    log_json: bool = False,
    **kwargs: Any,
) -> SearchResult:
    cfg = merge_config(
        cli_overrides=dict(
            question=question,
            db_path=db_path,
            executor_type=executor_type,
            max_depth=max_depth,
            max_attempts_per_node=max_attempts,
            high_confidence=high_confidence,
            token_budget=token_budget,
            verbose=verbose,
            cache_enabled=cache,
            cache_ttl=cache_ttl,
            calibration_enabled=calibration,
            interactive=interactive,
            ambiguity_threshold=ambiguity_threshold,
            max_concurrency=max_concurrency,
            llm=dict(provider=llm_provider, model=model, api_key=api_key, base_url=base_url),
        ),
        config_file_path=config_file,
    )

    log_level = "INFO" if verbose else log_level
    setup_logging(LogConfig(level=log_level, format="json" if log_json else "text"))

    llm = _create_llm(cfg.llm)

    cache_obj: CentralCache | None = None
    if cfg.cache_enabled:
        cache_obj = CentralCache(ttl=cfg.cache_ttl)
        llm = CachedLLM(llm, cache_obj, model=cfg.llm.model, temperature=cfg.llm.temperature)

    executor = _create_executor(cfg.executor_type, cfg.db_path)
    if cache_obj:
        executor = CachedExecutor(executor, cache_obj)

    tables = executor.list_tables()
    all_schemas = [executor.get_schema(t) for t in tables]
    schema = all_schemas[0]

    problem = ProblemSpec(question=question, schema=schema, all_schemas=all_schemas)

    def _ask_user(question: str) -> str:
        return input(f"\n? {question}\n> ")

    user_callback = _ask_user if cfg.interactive else None

    try:
        solution, dag, results = run_pipeline(llm, executor, cfg, problem, user_callback=user_callback)
    finally:
        executor.close()

    tables_used = _extract_tables_from_tree(solution.tree)

    return SearchResult(
        answer=solution.answer,
        sql=solution.sql,
        confidence=solution.confidence,
        token_cost=solution.token_cost,
        data=solution.data,
        tree=solution.tree,
        dag_nodes=_dag_to_dict(dag),
        tables_used=tables_used,
    )


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


def _extract_tables_from_tree(tree: list[NodeResult]) -> list[str]:
    import re
    tables: set[str] = set()
    pattern = re.compile(r'\b(?:FROM|JOIN)\s+(\w+)', re.IGNORECASE)
    cte_names: set[str] = set()
    for node in tree:
        if node.sql:
            cte_names |= _extract_cte_names(node.sql)
    for node in tree:
        if node.sql:
            for match in pattern.finditer(node.sql):
                tbl = match.group(1)
                if tbl not in cte_names:
                    tables.add(tbl)
    return sorted(tables)


def _dag_to_dict(dag: Any) -> list[dict]:
    nodes = []
    for nid, node in dag.nodes.items():
        nodes.append({
            "id": node.id,
            "description": node.description,
            "depends_on": node.depends_on,
            "depth": node.depth,
            "is_atomic": node.is_atomic,
        })
    return nodes
