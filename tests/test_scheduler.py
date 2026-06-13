import pandas as pd
import pytest

from syrch.core.config import ExecutionConfig
from syrch.core.models import NodeResult, TaskDAG, TaskNode


class FakeExecutor:
    def __init__(self):
        self.executed_sqls = []

    def execute(self, sql: str) -> pd.DataFrame:
        self.executed_sqls.append(sql)
        return pd.DataFrame({"result": [1, 2, 3]})

    def get_schema(self, table_name=None):
        from syrch.core.models import ColumnSchema, TableSchema

        return TableSchema(
            name="test",
            columns=[ColumnSchema(name="result", type="INTEGER")],
        )

    def list_tables(self):
        return ["test"]

    def close(self):
        pass


class FakeLLM:
    def __init__(self):
        self.attempt = 0

    def generate(self, system: str, user: str, **kwargs):
        self.attempt += 1
        content = f"```sql\nSELECT * FROM test LIMIT {self.attempt}\n```\nconfidence: 0.9"
        return type("Response", (), {"content": content, "model": "test", "usage": {"completion_tokens": 10}})()

    def generate_json(self, system: str, user: str, **kwargs):
        return {}


def test_scheduler_executes_dag():
    from syrch.search.scheduler import Scheduler
    from syrch.search.rlm_engine import RLMAgent

    config = ExecutionConfig(
        question="test",
        db_path=":memory:",
        max_attempts_per_node=2,
        high_confidence=0.85,
    )
    executor = FakeExecutor()
    llm = FakeLLM()

    dag = TaskDAG(
        nodes={
            "A": TaskNode(id="A", description="test A", depends_on=[], is_atomic=True),
            "B": TaskNode(id="B", description="test B", depends_on=["A"], is_atomic=True),
        },
        root_id="A",
        topo_layers=[["A"], ["B"]],
    )

    agent = RLMAgent(llm, executor, config)
    scheduler = Scheduler(llm, executor, config, agent=agent)
    results = scheduler.run(dag)

    assert "A" in results
    assert "B" in results
    assert len(results["A"].reasoning_paths) >= 1
    assert len(results["B"].reasoning_paths) >= 1
