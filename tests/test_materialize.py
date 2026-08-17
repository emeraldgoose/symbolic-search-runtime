"""Phase 1: context materialization tests (v0.3.5b).

Covers the Executor contract: materialize_context creates a real temp table
that the RLM can JOIN, the table is hidden from list_tables() so the
retriever never treats it as a physical candidate, and drop_context cleans up.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from syrch.core.config import ExecutionConfig
from syrch.core.models import ColumnSchema, NodeStatus, ParentContext, ProblemSpec
from syrch.executors.cached_executor import CachedExecutor
from syrch.executors.sqlite_executor import SQLiteExecutor
from syrch.search.aggregator import Aggregator
from syrch.search.planner import Planner
from syrch.search.rlm_engine import RLMAgent
from syrch.search.scheduler import Scheduler

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
ORDERS_DB = str(FIXTURES_DIR / "orders_10dim.sqlite")


def _context(table_name: str = "_task_context_A") -> ParentContext:
    return ParentContext(
        node_id="A",
        table_name=table_name,
        columns=[
            ColumnSchema(name="o_year", type="INTEGER"),
            ColumnSchema(name="o_orderpriority", type="TEXT"),
        ],
        source_tables=["orders_10dim"],
        row_count=2,
        data=pd.DataFrame(
            {"o_year": [1992, 1993], "o_orderpriority": ["1-URGENT", "5-LOW"]}
        ),
        materialized=True,
    )


def _require_db() -> str:
    if not Path(ORDERS_DB).exists():
        pytest.skip(f"Database file not found: {ORDERS_DB}")
    return ORDERS_DB


def test_sqlite_materialize_creates_joinable_table():
    db = _require_db()
    exec = SQLiteExecutor(db)
    try:
        name = exec.materialize_context(_context())
        assert name == "_task_context_A"
        rows = exec.execute(
            "SELECT o_year, o_orderpriority FROM _task_context_A ORDER BY o_year"
        )
        assert len(rows) == 2
        assert rows["o_orderpriority"].tolist() == ["1-URGENT", "5-LOW"]
    finally:
        exec.drop_context("_task_context_A")
        exec.close()


def test_materialized_context_hidden_from_list_tables():
    db = _require_db()
    exec = SQLiteExecutor(db)
    try:
        exec.materialize_context(_context())
        assert "_task_context_A" not in exec.list_tables()
        assert "orders_10dim" in exec.list_tables()
    finally:
        exec.drop_context("_task_context_A")
        exec.close()


def test_materialized_context_joinable_in_sql():
    db = _require_db()
    exec = SQLiteExecutor(db)
    try:
        exec.materialize_context(_context())
        df = exec.execute(
            "SELECT o.o_year FROM orders_10dim o "
            "JOIN _task_context_A c ON o.o_year = c.o_year"
        )
        assert len(df) > 0
    finally:
        exec.drop_context("_task_context_A")
        exec.close()


def test_materialize_empty_data_is_noop():
    db = _require_db()
    exec = SQLiteExecutor(db)
    try:
        ctx = _context()
        ctx.data = pd.DataFrame()
        name = exec.materialize_context(ctx)
        assert name == "_task_context_A"
        with pytest.raises(Exception):
            exec.execute("SELECT * FROM _task_context_A")
    finally:
        exec.drop_context("_task_context_A")
        exec.close()


def test_drop_context_removes_table():
    db = _require_db()
    exec = SQLiteExecutor(db)
    try:
        exec.materialize_context(_context())
        assert "_task_context_A" not in exec.list_tables()
        exec.drop_context("_task_context_A")
        with pytest.raises(Exception):
            exec.execute("SELECT * FROM _task_context_A")
    finally:
        exec.drop_context("_task_context_A")
        exec.close()


def test_materialize_replaces_existing_table():
    db = _require_db()
    exec = SQLiteExecutor(db)
    try:
        first = _context()
        first.data = pd.DataFrame({"o_year": [1992], "o_orderpriority": ["1-URGENT"]})
        exec.materialize_context(first)
        second = _context()
        second.data = pd.DataFrame(
            {"o_year": [1999], "o_orderpriority": ["5-LOW"]}
        )
        exec.materialize_context(second)
        rows = exec.execute("SELECT o_orderpriority FROM _task_context_A")
        assert rows["o_orderpriority"].tolist() == ["5-LOW"]
    finally:
        exec.drop_context("_task_context_A")
        exec.close()


def test_cached_executor_passes_materialize_through():
    db = _require_db()
    inner = SQLiteExecutor(db)
    exec = CachedExecutor(inner, cache=None)
    try:
        name = exec.materialize_context(_context())
        assert name == "_task_context_A"
        rows = inner.execute("SELECT o_orderpriority FROM _task_context_A")
        assert len(rows) == 2
        assert "_task_context_A" not in exec.list_tables()
    finally:
        exec.drop_context("_task_context_A")
        exec.close()


class _ContextJoinLLM:
    """Deterministic FakeLLM for the S3 acceptance test.

    Node A materializes a yearly aggregate; node B's SQL JOINs
    `_task_context_A` and must succeed only if the scheduler really
    materialized A's result into a joinable table.
    """

    def __init__(self, plan_response: dict | None = None):
        self.plan_calls = 0
        self.solve_calls = 0
        self.aggregate_calls = 0
        self._plan_response = plan_response

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        self.plan_calls += 1
        if self._plan_response:
            return self._plan_response
        return {"subtasks": [{"id": "A", "description": "yearly aggregate",
                              "depends_on": [], "is_atomic": True,
                              "expected_output": "per-year counts"}]}

    def generate(self, system: str, user: str, **kwargs):
        self.solve_calls += 1
        if "_task_context_A" in user:
            sql = (
                "SELECT o.o_year, c.cnt FROM orders_10dim o "
                "JOIN _task_context_A c ON o.o_year = c.o_year"
            )
        else:
            sql = (
                "SELECT o_year, COUNT(*) AS cnt FROM orders_10dim "
                "GROUP BY o_year"
            )
        return type(
            "Response",
            (),
            {
                "content": f"```sql\n{sql}\n```\nConfidence: 0.9",
                "model": "fake",
                "usage": {"completion_tokens": 15},
            },
        )()


def test_s3_acceptance_context_join_materialized():
    """Phase 3 acceptance: node B JOINs the materialized `_task_context_A`.

    B's SQL is only executable if the scheduler materialized A's SOLVED
    result into a real joinable table; otherwise execution fails with
    'no such table'. Also asserts the context table was cleaned up after
    the run and never surfaced as a physical candidate.
    """
    _require_db()
    plan_response = {
        "subtasks": [
            {
                "id": "A",
                "description": "Count orders by year",
                "depends_on": [],
                "is_atomic": True,
                "expected_output": "per-year counts",
            },
            {
                "id": "B",
                "description": "Join the yearly counts with orders",
                "depends_on": ["A"],
                "is_atomic": True,
                "expected_output": "per-year counts joined",
            },
        ]
    }
    config = ExecutionConfig(
        question="Count orders by year",
        db_path=ORDERS_DB,
        max_depth=2,
        max_attempts_per_node=1,
        verbose=False,
    )
    executor = SQLiteExecutor(config.db_path)
    llm = _ContextJoinLLM(plan_response=plan_response)
    schema = executor.get_schema("orders_10dim")
    problem = ProblemSpec(question=config.question, schema=schema)

    planner = Planner(llm, config)
    dag = planner.decompose(problem)
    assert "A" in dag.nodes and "B" in dag.nodes
    assert dag.nodes["B"].depends_on == ["A"]

    agent = RLMAgent(llm, executor, config)
    scheduler = Scheduler(llm, executor, config, agent=agent)
    results = scheduler.run(dag)

    assert results["A"].status == NodeStatus.SOLVED
    assert results["B"].status == NodeStatus.SOLVED
    assert results["B"].data is not None and not results["B"].data.empty
    assert results["B"].had_context is True
    assert results["B"].context_used is True

    table_names = set(executor.list_tables())
    assert not any(t.startswith("_task_context_") for t in table_names)

    aggregator = Aggregator(llm, executor, config)
    solution = aggregator.merge(config.question, dag, results)
    assert solution.answer is not None
    executor.close()


def test_s3_acceptance_ambiguous_dep_not_materialized():
    """AMBIGUOUS parents never materialize their context (user rule).

    A's result is not materialized, so B must not reference `_task_context_A`;
    the scope guard rejects the phantom context name even though the executor
    never created the table.
    """
    _require_db()
    plan_response = {
        "subtasks": [
            {"id": "A", "description": "Count orders by year",
             "depends_on": [], "is_atomic": True,
             "expected_output": "per-year counts"},
            {"id": "B", "description": "Join yearly counts with orders",
             "depends_on": ["A"], "is_atomic": True,
             "expected_output": "joined counts"},
        ]
    }
    config = ExecutionConfig(
        question="Count orders by year",
        db_path=ORDERS_DB,
        max_depth=2,
        max_attempts_per_node=1,
        verbose=False,
    )

    class AmbiguousLLM(_ContextJoinLLM):
        def generate(self, system: str, user: str, **kwargs):
            self.solve_calls += 1
            sql = "SELECT * FROM nonexistent_table_xyz"
            return type("Response", (), {
                "content": f"```sql\n{sql}\n```\nConfidence: 0.3",
                "model": "fake",
                "usage": {"completion_tokens": 5},
            })()

    executor = SQLiteExecutor(config.db_path)
    llm = AmbiguousLLM(plan_response=plan_response)
    schema = executor.get_schema("orders_10dim")
    problem = ProblemSpec(question=config.question, schema=schema)

    planner = Planner(llm, config)
    dag = planner.decompose(problem)
    agent = RLMAgent(llm, executor, config)
    scheduler = Scheduler(llm, executor, config, agent=agent)
    results = scheduler.run(dag)

    assert results["A"].status in (NodeStatus.AMBIGUOUS, NodeStatus.FAILED)
    assert results["B"].status in (NodeStatus.BLOCKED, NodeStatus.FAILED, NodeStatus.AMBIGUOUS)
    assert not any(
        t.startswith("_task_context_") for t in executor.list_tables()
    )
    executor.close()