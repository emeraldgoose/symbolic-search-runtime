import pytest
import pandas as pd

from syrch.core.config import ExecutionConfig
from syrch.core.models import TaskNode


class FakeLLM:
    def __init__(self):
        self.call_count = 0

    def generate(self, system: str, user: str, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            content = "```sql\nSELECT * FROM test WHERE x > 10\n```\nconfidence: 0.7"
        else:
            content = "```sql\nSELECT * FROM test WHERE x > 100\n```\nconfidence: 0.92"
        return type("Response", (), {"content": content, "model": "test", "usage": {"completion_tokens": 10}})()

    def generate_json(self, system: str, user: str, **kwargs):
        return {}


class FakeExecutor:
    def execute(self, sql: str) -> pd.DataFrame:
        return pd.DataFrame({"x": [50, 200]})

    def get_schema(self, table_name=None):
        from syrch.core.models import ColumnSchema, TableSchema

        return TableSchema(
            name="test",
            columns=[ColumnSchema(name="x", type="INTEGER")],
        )

    def list_tables(self):
        return ["test"]

    def close(self):
        pass


def test_rlm_agent_stops_on_high_confidence():
    from syrch.search.rlm_engine import RLMAgent

    config = ExecutionConfig(
        question="test",
        db_path=":memory:",
        max_attempts_per_node=5,
        high_confidence=0.85,
    )
    llm = FakeLLM()
    executor = FakeExecutor()
    agent = RLMAgent(llm, executor, config)

    node = TaskNode(
        id="A",
        description="test task",
        is_atomic=True,
    )
    result = agent.solve(node)

    assert llm.call_count == 2
    assert result.confidence == pytest.approx(0.9108, rel=1e-3)
    assert result.sql == "SELECT * FROM test WHERE x > 100"
    assert len(result.reasoning_paths) == 2


def test_rlm_agent_uses_best_path_on_budget_exhausted():
    from syrch.search.rlm_engine import RLMAgent

    class LowConfLLM:
        def __init__(self):
            self.count = 0

        def generate(self, system: str, user: str, **kwargs):
            self.count += 1
            return type("Response", (), {"content": "```sql\nSELECT 1\n```\nconfidence: 0.3", "model": "t", "usage": {"completion_tokens": 5}})()

        def generate_json(self, *a, **kw):
            return {}

    config = ExecutionConfig(
        question="test", db_path=":memory:",
        max_attempts_per_node=3, high_confidence=0.85,
    )
    executor = FakeExecutor()
    llm = LowConfLLM()
    agent = RLMAgent(llm, executor, config)

    node = TaskNode(id="A", description="test", is_atomic=True)
    result = agent.solve(node)

    assert llm.count == 3
    assert result.confidence == pytest.approx(0.29, rel=1e-2)


class SyntaxErrorLLM:
    def __init__(self):
        self.count = 0

    def generate(self, system: str, user: str, **kwargs):
        self.count += 1
        if self.count == 1:
            content = "```sql\nSELECT COUNT( FROM test\n```\nconfidence: 0.4"
        else:
            content = "```sql\nSELECT COUNT(*) FROM test\n```\nconfidence: 0.9"
        return type("Response", (), {"content": content, "model": "test", "usage": {"completion_tokens": 10}})()

    def generate_json(self, *a, **kw):
        return {}


def test_rlm_agent_validates_sql_syntax():
    from syrch.search.rlm_engine import RLMAgent

    config = ExecutionConfig(
        question="test", db_path=":memory:",
        max_attempts_per_node=3, high_confidence=0.85,
    )
    llm = SyntaxErrorLLM()
    executor = FakeExecutor()
    agent = RLMAgent(llm, executor, config)

    node = TaskNode(id="A", description="test task", is_atomic=True)
    result = agent.solve(node)

    assert llm.count == 2
    assert result.confidence == pytest.approx(0.7965, rel=1e-3)
    assert len(result.reasoning_paths) == 2
    # First path should have the invalid SQL recorded
    assert result.reasoning_paths[0].sql == "SELECT COUNT( FROM test"


class SchemaErrorLLM:
    def __init__(self):
        self.count = 0

    def generate(self, system: str, user: str, **kwargs):
        self.count += 1
        if self.count == 1:
            content = "```sql\nSELECT y FROM test\n```\nconfidence: 0.4"
        else:
            content = "```sql\nSELECT x FROM test\n```\nconfidence: 0.9"
        return type("Response", (), {"content": content, "model": "test", "usage": {"completion_tokens": 10}})()

    def generate_json(self, *a, **kw):
        return {}


def test_rlm_agent_validates_schema():
    from syrch.search.rlm_engine import RLMAgent

    config = ExecutionConfig(
        question="test", db_path=":memory:",
        max_attempts_per_node=3, high_confidence=0.85,
    )
    llm = SchemaErrorLLM()
    executor = FakeExecutor()
    agent = RLMAgent(llm, executor, config)

    node = TaskNode(id="A", description="test task", is_atomic=True)
    result = agent.solve(node)

    assert llm.count == 2
    assert result.confidence == pytest.approx(0.8407, rel=1e-3)
    assert len(result.reasoning_paths) == 2
    assert result.reasoning_paths[0].sql == "SELECT y FROM test"


class EmptyResultExecutor:
    def execute(self, sql: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_schema(self, table_name=None):
        from syrch.core.models import ColumnSchema, TableSchema
        return TableSchema(
            name="test",
            columns=[ColumnSchema(name="x", type="INTEGER")],
        )

    def list_tables(self):
        return ["test"]

    def close(self):
        pass


def test_rlm_agent_warns_empty_result():
    from syrch.search.rlm_engine import RLMAgent

    class TwoAttemptLLM:
        def __init__(self):
            self.count = 0

        def generate(self, system: str, user: str, **kwargs):
            self.count += 1
            content = "```sql\nSELECT x FROM test\n```\nconfidence: 0.7"
            return type("Response", (), {"content": content, "model": "test", "usage": {"completion_tokens": 10}})()

        def generate_json(self, *a, **kw):
            return {}

    config = ExecutionConfig(
        question="test", db_path=":memory:",
        max_attempts_per_node=3, high_confidence=0.85,
    )
    llm = TwoAttemptLLM()
    executor = EmptyResultExecutor()
    agent = RLMAgent(llm, executor, config)

    node = TaskNode(id="A", description="test task", is_atomic=True)
    result = agent.solve(node)

    # Should exhaust all attempts because quality warning triggers retry
    assert llm.count == 3
    assert len(result.reasoning_paths) == 3


def test_validate_sql_direct():
    from syrch.search.rlm_engine import RLMAgent

    class FakeLLMBase:
        def generate(self, *a, **kw): raise NotImplementedError
        def generate_json(self, *a, **kw): return {}

    config = ExecutionConfig(question="t", db_path=":memory:")
    agent = RLMAgent(FakeLLMBase(), FakeExecutor(), config)

    assert agent._validate_sql("SELECT 1") is None
    assert agent._validate_sql("SELECT COUNT(*) FROM test") is None
    assert agent._validate_sql("SELECT COUNT( FROM test") is not None


def test_validate_schema_direct():
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ColumnSchema, TableSchema

    class MultiTableExecutor:
        def execute(self, sql: str):
            import pandas as pd
            return pd.DataFrame({"x": [1]})

        def get_schema(self, table_name=None):
            if table_name == "orders":
                return TableSchema(
                    name="orders",
                    columns=[ColumnSchema(name="id", type="INTEGER"), ColumnSchema(name="name", type="TEXT")],
                )
            return TableSchema(name="test", columns=[ColumnSchema(name="x", type="INTEGER")])

        def list_tables(self):
            return ["test", "orders"]

        def close(self):
            pass

    class FakeLLMBase:
        def generate(self, *a, **kw): raise NotImplementedError
        def generate_json(self, *a, **kw): return {}

    config = ExecutionConfig(question="t", db_path=":memory:")
    agent = RLMAgent(FakeLLMBase(), MultiTableExecutor(), config)

    assert agent._validate_schema("SELECT id FROM orders") is None
    assert agent._validate_schema("SELECT x FROM test") is None
    err = agent._validate_schema("SELECT nonexistent FROM orders")
    assert err is not None and "nonexistent" in err
    assert agent._validate_schema("SELECT * FROM orders") is None  # star is skipped


def test_check_result_quality_direct():
    from syrch.search.rlm_engine import RLMAgent
    import pandas as pd

    class FakeLLMBase:
        def generate(self, *a, **kw): raise NotImplementedError
        def generate_json(self, *a, **kw): return {}

    config = ExecutionConfig(question="t", db_path=":memory:")
    agent = RLMAgent(FakeLLMBase(), FakeExecutor(), config)

    assert agent._check_result_quality(pd.DataFrame({"x": [1, 2, 3]})) is None
    assert agent._check_result_quality(pd.DataFrame()) is not None
    assert "empty" in agent._check_result_quality(pd.DataFrame()).lower()
    assert agent._check_result_quality(pd.DataFrame({"x": [None, None]})) is not None
    assert "NULL" in agent._check_result_quality(pd.DataFrame({"x": [None, None]})).upper()
    assert agent._check_result_quality(pd.DataFrame({"x": range(2000)})) is not None
    msg = agent._check_result_quality(pd.DataFrame({"x": range(2000)}))
    assert msg is not None and "2000 rows" in msg
