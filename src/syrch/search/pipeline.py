from __future__ import annotations

from syrch.core.config import ExecutionConfig
from syrch.core.models import FinalSolution, NodeResult, ProblemSpec, TaskDAG
from syrch.executors.base import BaseExecutor
from syrch.llm.base import BaseLLM
from syrch.search.aggregator import Aggregator
from syrch.search.planner import Planner
from syrch.search.scheduler import Scheduler


def run_pipeline(
    llm: BaseLLM,
    executor: BaseExecutor,
    config: ExecutionConfig,
    problem: ProblemSpec,
) -> tuple[FinalSolution, TaskDAG, dict[str, NodeResult]]:
    if problem.all_schemas is None:
        problem.all_schemas = [executor.get_schema(t) for t in executor.list_tables()]
    planner = Planner(llm, config)
    dag = planner.decompose(problem)
    scheduler = Scheduler(llm, executor, config)
    results = scheduler.run(dag)
    aggregator = Aggregator(llm, executor, config)
    solution = aggregator.merge(problem.question, dag, results)
    return solution, dag, results
