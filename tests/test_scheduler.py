import pandas as pd

from syrch.core.config import ExecutionConfig
from syrch.core.models import TaskDAG, TaskNode


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


def test_scheduler_blocks_dependents_of_failed_node():
    """A FAILED dependency propagates as BLOCKED; dependents never execute SQL."""
    from syrch.search.scheduler import Scheduler
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ColumnSchema, NodeStatus, TableSchema

    class FailingExecutor:
        def __init__(self):
            self.executed_sqls = []

        def execute(self, sql: str) -> pd.DataFrame:
            self.executed_sqls.append(sql)
            raise RuntimeError("boom")

        def get_schema(self, table_name=None):
            return TableSchema(
                name="test",
                columns=[ColumnSchema(name="result", type="INTEGER")],
            )

        def list_tables(self):
            return ["test"]

        def close(self):
            pass

    config = ExecutionConfig(
        question="test", db_path=":memory:", max_attempts_per_node=1,
    )
    executor = FailingExecutor()
    llm = FakeLLM()

    dag = TaskDAG(
        nodes={
            "A": TaskNode(id="A", description="test A", depends_on=[], is_atomic=True),
            "B": TaskNode(id="B", description="test B", depends_on=["A"], is_atomic=True),
            "C": TaskNode(id="C", description="test C", depends_on=["B"], is_atomic=True),
        },
        root_id="A",
        topo_layers=[["A"], ["B"], ["C"]],
    )

    agent = RLMAgent(llm, executor, config)
    scheduler = Scheduler(llm, executor, config, agent=agent)
    results = scheduler.run(dag)

    assert results["A"].status == NodeStatus.FAILED
    assert results["B"].status == NodeStatus.BLOCKED
    assert results["C"].status == NodeStatus.BLOCKED
    # Only A ever reached the executor; B and C were short-circuited.
    assert len(executor.executed_sqls) == 1


def test_build_context_returns_parent_context_contract():
    """Dependency results are wrapped in first-class ParentContext with
    explicit column/type/source metadata, not a bare DataFrame preview."""
    from syrch.core.models import NodeResult
    from syrch.search.rlm_engine import RLMAgent
    from syrch.search.scheduler import Scheduler

    config = ExecutionConfig(question="test", db_path=":memory:")
    executor = FakeExecutor()
    llm = FakeLLM()
    agent = RLMAgent(llm, executor, config)
    scheduler = Scheduler(llm, executor, config, agent=agent)

    node = TaskNode(id="B", description="test B", depends_on=["A"], is_atomic=True)
    res_a = NodeResult(
        node_id="A",
        data=pd.DataFrame({"customer_id": [1, 2], "segment": ["vip", "standard"]}),
        sql="SELECT customer_id, segment FROM dw_customer",
        confidence=0.9,
    )

    ctx = scheduler._build_context(node, {"A": res_a})

    assert "A" in ctx
    pc = ctx["A"]
    assert pc.node_id == "A"
    assert pc.table_name == "_task_context_A"
    assert [c.name for c in pc.columns] == ["customer_id", "segment"]
    assert [c.type for c in pc.columns] == ["INTEGER", "TEXT"]
    assert pc.source_tables == ["dw_customer"]
    assert pc.row_count == 2
    rendered = pc.to_prompt()
    assert "_task_context_A" in rendered
    assert "customer_id (INTEGER)" in rendered
    assert "source tables: dw_customer" in rendered


def test_build_context_skips_empty_results():
    """A dependency that produced no data contributes no ParentContext."""
    from syrch.core.models import NodeResult
    from syrch.search.rlm_engine import RLMAgent
    from syrch.search.scheduler import Scheduler

    config = ExecutionConfig(question="test", db_path=":memory:")
    executor = FakeExecutor()
    llm = FakeLLM()
    agent = RLMAgent(llm, executor, config)
    scheduler = Scheduler(llm, executor, config, agent=agent)

    node = TaskNode(id="B", description="test B", depends_on=["A"], is_atomic=True)
    res_a = NodeResult(
        node_id="A",
        data=pd.DataFrame(),
        sql="SELECT * FROM dw_customer",
        confidence=0.0,
    )

    ctx = scheduler._build_context(node, {"A": res_a})
    assert ctx == {}


def test_attempt_schemas_include_dependency_source_tables():
    """A dependent node sees the physical tables that produced its dependency
    so it can re-derive the dependency's columns (S3/S4)."""
    from syrch.core.models import ColumnSchema, TableSchema
    from syrch.search.rlm_engine import RLMAgent

    config = ExecutionConfig(question="test", db_path=":memory:")
    executor = FakeExecutor()
    llm = FakeLLM()

    all_schemas = [
        TableSchema(name="dw_sales_order", columns=[ColumnSchema(name="amount", type="INTEGER")]),
        TableSchema(name="dw_customer", columns=[ColumnSchema(name="segment", type="TEXT")]),
        TableSchema(name="mart_sales_monthly", columns=[ColumnSchema(name="revenue", type="INTEGER")]),
    ]
    agent = RLMAgent(llm, executor, config, all_schemas=all_schemas)
    from syrch.core.models import ParentContext, ScoredTable

    pool = [ScoredTable(schema=s, score=1.0) for s in all_schemas]
    agent.set_candidate_pool(pool)

    node = TaskNode(
        id="B", description="revenue by segment", depends_on=["A"], is_atomic=True,
        hint_tables=["dw_sales_order"],
    )
    cand = pool[0]  # dw_sales_order
    ctx = {
        "A": ParentContext(
            node_id="A",
            table_name="_task_context_A",
            columns=[ColumnSchema(name="segment", type="TEXT")],
            source_tables=["dw_customer"],
            row_count=10,
        ),
    }

    scope = agent._build_attempt_schemas(node, cand, ctx)
    assert scope.primary_tables == {"dw_sales_order"}
    assert "dw_customer" in scope.join_available_tables  # dependency source table
    assert "mart_sales_monthly" not in scope.join_available_tables
    assert scope.allowed_tables == {"dw_customer", "dw_sales_order"}

