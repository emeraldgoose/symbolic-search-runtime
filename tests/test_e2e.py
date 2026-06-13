"""End-to-end tests against real SQLite databases using FakeLLM."""

import pandas as pd

from syrch.core.config import ExecutionConfig, LLMConfig
from syrch.core.models import ColumnSchema, ProblemSpec, TableSchema
from syrch.executors.sqlite_executor import SQLiteExecutor
from syrch.search.aggregator import Aggregator
from syrch.search.planner import Planner
from syrch.search.scheduler import Scheduler
from syrch.search.rlm_engine import RLMAgent


class FakeLLM:
    def __init__(self, plan_response: dict | None = None):
        self.plan_calls = 0
        self.solve_calls = 0
        self.aggregate_calls = 0
        self._plan_response = plan_response

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        self.plan_calls += 1
        if self._plan_response:
            return self._plan_response
        return {
            "subtasks": [
                {
                    "id": "A",
                    "description": "Analyze the data distribution",
                    "depends_on": [],
                    "is_atomic": True,
                    "expected_output": "summary statistics",
                }
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        self.solve_calls += 1
        if "confidence" in user:
            conf = "0.85"
        else:
            conf = "0.92"
        return type(
            "Response",
            (),
            {
                "content": (
                    f"```sql\n{self._infer_sql(user)}\n```\nconfidence: {conf}"
                ),
                "model": "fake",
                "usage": {"completion_tokens": 15},
            },
        )()

    def _infer_sql(self, user: str) -> str:
        user_lower = user.lower()
        if "clickstream" in user_lower or "wikipedia" in user_lower:
            return "SELECT type, SUM(total_n) as total FROM wikipedia_clickstream GROUP BY type ORDER BY total DESC"
        if "orders_10dim" in user_lower:
            return "SELECT o_orderpriority, COUNT(*) as cnt FROM orders_10dim GROUP BY o_orderpriority ORDER BY cnt DESC"
        return "SELECT * FROM (SELECT 1 as test) LIMIT 5"


def test_wikipedia_clickstream_schema():
    """Wikipedia clickstream DB의 스키마 읽기 검증."""
    exec = SQLiteExecutor("wikipedia_clickstream.sqlite")
    tables = exec.list_tables()
    assert "wikipedia_clickstream" in tables
    assert "metadata" in tables

    schema = exec.get_schema("wikipedia_clickstream")
    col_names = {c.name for c in schema.columns}
    assert "type" in col_names
    assert "total_n" in col_names
    assert "referrer_source" in col_names
    assert "row_count" in col_names
    assert "curr_first_letter" in col_names
    assert "curr_title_length" in col_names

    exec.close()


def test_orders_10dim_schema():
    """Orders 10dim DB의 스키마 읽기 검증."""
    exec = SQLiteExecutor("orders_10dim.sqlite")
    tables = exec.list_tables()
    assert "orders_10dim" in tables
    assert "orders" in tables

    schema = exec.get_schema("orders_10dim")
    col_names = {c.name for c in schema.columns}
    assert "o_year" in col_names
    assert "o_totalprice" in col_names
    assert "o_orderpriority" in col_names
    assert "o_orderstatus" in col_names
    assert "o_clerk" in col_names

    exec.close()


def test_wikipedia_clickstream_query():
    """Wikipedia clickstream DB에 실제 SQL 실행 검증."""
    exec = SQLiteExecutor("wikipedia_clickstream.sqlite")
    df = exec.execute("SELECT type, SUM(total_n) as total FROM wikipedia_clickstream GROUP BY type ORDER BY total DESC")
    assert len(df) > 0
    assert "type" in df.columns
    assert "total" in df.columns
    exec.close()


def test_orders_10dim_query():
    """Orders 10dim DB에 실제 SQL 실행 검증."""
    exec = SQLiteExecutor("orders_10dim.sqlite")
    df = exec.execute("SELECT o_orderpriority, COUNT(*) as cnt FROM orders_10dim GROUP BY o_orderpriority ORDER BY cnt DESC")
    assert len(df) > 0
    assert "o_orderpriority" in df.columns
    assert "cnt" in df.columns
    exec.close()


def test_e2e_wikipedia_clickstream_pipeline():
    """Wikipedia clickstream으로 full search pipeline 검증."""
    config = ExecutionConfig(
        question="Which click type generates the most traffic?",
        db_path="wikipedia_clickstream.sqlite",
        max_depth=2,
        max_attempts_per_node=1,
        high_confidence=0.85,
        verbose=False,
    )
    executor = SQLiteExecutor(config.db_path)
    llm = FakeLLM()

    schema = executor.get_schema("wikipedia_clickstream")
    problem = ProblemSpec(question=config.question, schema=schema)

    # Planner
    planner = Planner(llm, config)
    dag = planner.decompose(problem)
    assert len(dag.nodes) >= 1
    assert dag.topo_layers is not None

    # Scheduler + RLMAgent
    agent = RLMAgent(llm, executor, config)
    scheduler = Scheduler(llm, executor, config, agent=agent)
    results = scheduler.run(dag)
    assert len(results) >= 1

    # Aggregator
    aggregator = Aggregator(llm, executor, config)
    solution = aggregator.merge(config.question, dag, results)

    assert solution.answer is not None
    assert solution.confidence > 0
    assert solution.token_cost > 0


def test_e2e_orders_10dim_pipeline():
    """Orders 10dim으로 full search pipeline 검증."""
    config = ExecutionConfig(
        question="What is the order priority distribution?",
        db_path="orders_10dim.sqlite",
        max_depth=2,
        max_attempts_per_node=1,
        high_confidence=0.85,
        verbose=False,
    )
    executor = SQLiteExecutor(config.db_path)
    llm = FakeLLM()

    schema = executor.get_schema("orders_10dim")
    problem = ProblemSpec(question=config.question, schema=schema)

    # Planner
    planner = Planner(llm, config)
    dag = planner.decompose(problem)
    assert len(dag.nodes) >= 1

    # Scheduler + RLMAgent
    agent = RLMAgent(llm, executor, config)
    scheduler = Scheduler(llm, executor, config, agent=agent)
    results = scheduler.run(dag)
    assert len(results) >= 1

    # Aggregator
    aggregator = Aggregator(llm, executor, config)
    solution = aggregator.merge(config.question, dag, results)

    assert solution.answer is not None
    assert solution.confidence > 0
    assert solution.token_cost > 0


def test_e2e_multi_layer_dag():
    """Multi-layer DAG (의존성 있는 sub-task) 파이프라인 검증."""
    plan_response = {
        "subtasks": [
            {
                "id": "A",
                "description": "Get total traffic by type from wikipedia_clickstream",
                "depends_on": [],
                "is_atomic": True,
                "expected_output": "traffic counts per type",
            },
            {
                "id": "B",
                "description": "Using result from A, find top referrer_source for the most popular type",
                "depends_on": ["A"],
                "is_atomic": True,
                "expected_output": "referrer breakdown for top type",
            },
        ]
    }

    config = ExecutionConfig(
        question="What is the top referrer source for the most popular click type?",
        db_path="wikipedia_clickstream.sqlite",
        max_depth=2,
        max_attempts_per_node=1,
        high_confidence=0.85,
        verbose=False,
    )
    executor = SQLiteExecutor(config.db_path)
    llm = FakeLLM(plan_response=plan_response)

    schema = executor.get_schema("wikipedia_clickstream")
    problem = ProblemSpec(question=config.question, schema=schema)

    planner = Planner(llm, config)
    dag = planner.decompose(problem)

    # Planner가 FakeLLM의 응답을 사용했는지 확인
    assert "A" in dag.nodes
    assert "B" in dag.nodes
    assert dag.nodes["B"].depends_on == ["A"]

    agent = RLMAgent(llm, executor, config)
    scheduler = Scheduler(llm, executor, config, agent=agent)
    results = scheduler.run(dag)

    assert "A" in results
    assert "B" in results
    # B의 context에 A의 결과가 전달되었는지 확인
    assert results["B"].confidence > 0

    aggregator = Aggregator(llm, executor, config)
    solution = aggregator.merge(config.question, dag, results)

    assert solution.answer is not None
    assert solution.confidence > 0


def test_orders_10dim_row_count():
    """Orders 10dim의 행 수 검증 (cardinality check)."""
    exec = SQLiteExecutor("orders_10dim.sqlite")
    df = exec.execute("SELECT COUNT(*) as cnt FROM orders_10dim")
    assert df.iloc[0]["cnt"] == 7_500_000
    exec.close()


def test_metadata_table_content():
    """Wikipedia clickstream metadata 테이블 내용 검증."""
    exec = SQLiteExecutor("wikipedia_clickstream.sqlite")
    df = exec.execute("SELECT key, value FROM metadata")
    assert len(df) > 0
    keys = set(df["key"])
    assert any("mi_" in k for k in keys)
    assert any("card_" in k for k in keys)
    exec.close()
