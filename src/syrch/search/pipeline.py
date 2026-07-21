from __future__ import annotations

from typing import Callable

from syrch.core.config import ExecutionConfig
from syrch.core.models import FinalSolution, NodeResult, ProblemSpec, TableSchema, TaskDAG
from syrch.executors.base import BaseExecutor
from syrch.llm.base import BaseLLM
from syrch.search.aggregator import Aggregator
from syrch.search.clarify import run_clarification
from syrch.search.planner import Planner
from syrch.search.retriever import Retriever
from syrch.search.scheduler import Scheduler


def compress_schema(dag: TaskDAG, all_schemas: list[TableSchema]) -> list[TableSchema]:
    hinted_names: set[str] = set()
    for node in dag.nodes.values():
        if node.hint_tables:
            hinted_names.update(node.hint_tables)
    if not hinted_names:
        return all_schemas
    return [s for s in all_schemas if s.name in hinted_names]


def run_pipeline(
    llm: BaseLLM,
    executor: BaseExecutor,
    config: ExecutionConfig,
    problem: ProblemSpec,
    user_callback: Callable[[str], str] | None = None,
) -> tuple[FinalSolution, TaskDAG, dict[str, NodeResult]]:
    if problem.all_schemas is None:
        problem.all_schemas = [executor.get_schema(t) for t in executor.list_tables()]
    if problem.scored_schemas is None:
        retriever = Retriever()
        problem.scored_schemas = retriever.score(problem.question, problem.all_schemas)

    planner = Planner(llm, config)
    dag = planner.decompose(problem)

    amended_question, qa_pairs = run_clarification(
        llm=llm,
        question=problem.question,
        dag=dag,
        scored_schemas=problem.scored_schemas,
        user_callback=user_callback if config.interactive else None,
    )

    problem = ProblemSpec(
        question=amended_question,
        schema=problem.schema,
        all_schemas=problem.all_schemas,
        scored_schemas=problem.scored_schemas,
    )

    compressed = compress_schema(dag, problem.all_schemas)
    scheduler = Scheduler(llm, executor, config, compressed_schemas=compressed)
    results = scheduler.run(dag)
    aggregator = Aggregator(llm, executor, config)
    solution = aggregator.merge(problem.question, dag, results)
    solution.clarified = bool(qa_pairs)
    solution.clarification_qa = qa_pairs
    return solution, dag, results
