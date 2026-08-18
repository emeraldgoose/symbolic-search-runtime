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


def test_rlm_agent_single_candidate():
    """Single candidate: RLM runs once and returns the result; confidence is the
    model's explicit value (no auto-boost, no confidence threshold stop)."""
    from syrch.search.rlm_engine import RLMAgent

    config = ExecutionConfig(
        question="test",
        db_path=":memory:",
        max_attempts_per_node=5,
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

    assert llm.call_count == 1
    assert result.confidence == pytest.approx(0.7, rel=1e-2)
    assert result.sql == "SELECT * FROM test WHERE x > 10"
    assert len(result.reasoning_paths) == 1


def test_rlm_agent_prefers_higher_path_score():
    """When multiple candidates succeed, best path_score wins."""
    from syrch.search.rlm_engine import RLMAgent

    class PoolLLM:
        def __init__(self):
            self.count = 0
        def generate(self, system: str, user: str, **kwargs):
            self.count += 1
            return type("Response", (), {"content": "```sql\nSELECT 1 AS x\n```\nconfidence: 0.7", "model": "t", "usage": {"completion_tokens": 5}})()
        def generate_json(self, *a, **kw):
            return {}

    config = ExecutionConfig(
        question="test", db_path=":memory:",
        max_attempts_per_node=3,
    )
    executor = FakeExecutor()
    llm = PoolLLM()
    agent = RLMAgent(llm, executor, config)

    node = TaskNode(id="A", description="test", is_atomic=True)
    result = agent.solve(node)

    assert result.sql == "SELECT 1 AS x"
    assert result.path_score is not None


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
        max_attempts_per_node=3,
    )
    llm = SyntaxErrorLLM()
    executor = FakeExecutor()
    agent = RLMAgent(llm, executor, config)

    node = TaskNode(id="A", description="test task", is_atomic=True)
    result = agent.solve(node)

    assert llm.count == 2
    assert result.confidence == pytest.approx(0.9, rel=1e-2)
    assert len(result.reasoning_paths) == 2
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
        max_attempts_per_node=3,
    )
    llm = SchemaErrorLLM()
    executor = FakeExecutor()
    agent = RLMAgent(llm, executor, config)

    node = TaskNode(id="A", description="test task", is_atomic=True)
    result = agent.solve(node)

    assert llm.count == 2
    assert result.confidence == pytest.approx(0.9, rel=1e-2)
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
        max_attempts_per_node=3,
    )
    llm = TwoAttemptLLM()
    executor = EmptyResultExecutor()
    agent = RLMAgent(llm, executor, config)

    node = TaskNode(id="A", description="test task", is_atomic=True)
    result = agent.solve(node)

    # Empty result breaks to next candidate (no retry for quality warnings)
    assert llm.count == 1
    assert len(result.reasoning_paths) == 1


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


def test_validate_schema_from_scope_blocks_drift():
    """FROM tables outside the per-attempt scope are rejected — S5 drift guard.
    Tables unknown to the executor (e.g. synthetic node names) are not blocked."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ColumnSchema, ScoredTable, TableSchema, TaskNode

    class MultiTableExecutor:
        def execute(self, sql: str):
            import pandas as pd
            return pd.DataFrame({"x": [1]})

        def get_schema(self, table_name=None):
            if table_name == "dw_sales_order":
                return TableSchema(
                    name="dw_sales_order",
                    columns=[ColumnSchema(name="id", type="INTEGER"),
                             ColumnSchema(name="total_amount", type="REAL")],
                )
            return TableSchema(name="mart_sales_monthly",
                               columns=[ColumnSchema(name="total", type="REAL")])

        def list_tables(self):
            return ["dw_sales_order", "mart_sales_monthly"]

        def close(self):
            pass

    class FakeLLMBase:
        def generate(self, *a, **kw): raise NotImplementedError
        def generate_json(self, *a, **kw): return {}

    config = ExecutionConfig(question="t", db_path=":memory:")
    agent = RLMAgent(FakeLLMBase(), MultiTableExecutor(), config)
    agent._candidate_pool = [
        ScoredTable(schema=TableSchema(name="mart_sales_monthly",
                                       columns=[ColumnSchema(name="total", type="REAL")]),
                    score=1.0),
    ]
    # scope = the per-attempt boundary (primary | join-available), not the pool
    node = TaskNode(id="A", description="t", is_atomic=True)
    cand = agent._candidate_pool[0]
    agent._attempt_scope = agent._build_attempt_schemas(node, cand)
    agent._allowed_tables = agent._attempt_scope.allowed_tables

    assert agent._allowed_tables == {"mart_sales_monthly"}
    # in-scope table passes
    assert agent._validate_schema("SELECT total FROM mart_sales_monthly") is None
    # drift to a physical pool-external table is rejected (S5)
    err = agent._validate_schema(
        "SELECT SUM(total_amount) FROM dw_sales_order"
    )
    assert err is not None and "outside the allowed search scope" in err
    # CTE names are not treated as physical tables
    assert agent._validate_schema(
        "WITH tmp AS (SELECT total FROM mart_sales_monthly) SELECT * FROM tmp"
    ) is None


def test_validate_schema_fqn_drift_guard_works():
    """S5 drift guard must survive fully-qualified names: sqlglot parses
    `catalog.schema.table` into Table.name = base name, so the known/allowed
    comparison must be done on the base name too. Previously an FQN FROM table
    never matched the allowed scope and the guard silently passed everything."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ColumnSchema, ScoredTable, TableSchema, TaskNode

    class MultiTableExecutor:
        def execute(self, sql: str):
            import pandas as pd
            return pd.DataFrame({"x": [1]})

        def get_schema(self, table_name=None):
            if table_name == "syrch_benchmark.enterprise.dw_sales_order":
                return TableSchema(
                    name="syrch_benchmark.enterprise.dw_sales_order",
                    columns=[ColumnSchema(name="id", type="INTEGER"),
                             ColumnSchema(name="total_amount", type="REAL")],
                )
            return TableSchema(
                name="syrch_benchmark.enterprise.mart_sales_monthly",
                columns=[ColumnSchema(name="total", type="REAL")],
            )

        def list_tables(self):
            return ["syrch_benchmark.enterprise.dw_sales_order",
                    "syrch_benchmark.enterprise.mart_sales_monthly"]

        def close(self):
            pass

    class FakeLLMBase:
        def generate(self, *a, **kw): raise NotImplementedError
        def generate_json(self, *a, **kw): return {}

    config = ExecutionConfig(question="t", db_path=":memory:")
    agent = RLMAgent(FakeLLMBase(), MultiTableExecutor(), config)
    agent._candidate_pool = [
        ScoredTable(
            schema=TableSchema(
                name="syrch_benchmark.enterprise.mart_sales_monthly",
                columns=[ColumnSchema(name="total", type="REAL")],
            ),
            score=1.0,
        ),
    ]
    node = TaskNode(id="A", description="t", is_atomic=True)
    cand = agent._candidate_pool[0]
    agent._attempt_scope = agent._build_attempt_schemas(node, cand)
    agent._allowed_tables = agent._attempt_scope.allowed_tables

    # in-scope FQN table passes
    assert agent._validate_schema(
        "SELECT total FROM syrch_benchmark.enterprise.mart_sales_monthly"
    ) is None
    # FQN drift to a physical pool-external table is rejected (S5)
    err = agent._validate_schema(
        "SELECT SUM(total_amount) FROM syrch_benchmark.enterprise.dw_sales_order"
    )
    assert err is not None and "outside the allowed search scope" in err


def test_validate_scope_positions_fqn_primary_switch_rejected():
    """S3 position invariant must survive FQN: candidate = dw_customer but SQL
    anchors FROM on a JOIN-AVAILABLE table (rpt_customer_ltv) is a primary
    switch and must be rejected. Previously base-name comparison against the
    FQN scope let every candidate's SQL pass with the same FROM."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ColumnSchema, ScoredTable, TableSchema, TaskNode

    schemas = [
        TableSchema(name="syrch_benchmark.enterprise.dw_customer",
                    columns=[ColumnSchema(name="customer_id", type="INTEGER")]),
        TableSchema(name="syrch_benchmark.enterprise.rpt_customer_ltv",
                    columns=[ColumnSchema(name="customer_id", type="INTEGER"),
                             ColumnSchema(name="segment", type="STRING")]),
    ]
    agent = RLMAgent.__new__(RLMAgent)
    agent.all_schemas = schemas

    class Executor:
        def list_tables(self):
            return [s.name for s in schemas]
        def get_schema(self, table_name=None):
            return schemas[0] if table_name == schemas[0].name else schemas[1]
    agent.executor = Executor()

    pool = [ScoredTable(schema=s, score=1.0) for s in schemas]
    agent._candidate_pool = pool

    node = TaskNode(id="A", description="vip customers", is_atomic=True)
    scope = agent._build_attempt_schemas(node, pool[0])  # primary = dw_customer
    agent._attempt_scope = scope
    agent._allowed_tables = scope.allowed_tables
    assert "syrch_benchmark.enterprise.rpt_customer_ltv" in scope.join_available_tables

    # primary switch: candidate=dw_customer but FROM anchors on rpt_customer_ltv
    sql = (
        "SELECT customer_id FROM syrch_benchmark.enterprise.rpt_customer_ltv "
        "WHERE segment = 'VIP'"
    )
    err = agent._validate_schema(sql)
    assert err is not None and "appears in FROM" in err
    assert "dw_customer" in err

    # anchored on the candidate -> passes
    ok_sql = "SELECT customer_id FROM syrch_benchmark.enterprise.dw_customer"
    assert agent._validate_schema(ok_sql) is None


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


def test_rlm_candidate_traversal_s14_like():
    """Single-schema best-first traversal: rank-1 irrelevant table fails validation,
    then the traversal reaches the ground-truth table at rank-5."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ScoredTable, TableSchema, ColumnSchema

    tables = {
        "archive": TableSchema(name="archive", columns=[
            ColumnSchema(name="archived_at", type="TIMESTAMP"),
            ColumnSchema(name="total", type="REAL"),
        ]),
        "mart_sales_daily": TableSchema(name="mart_sales_daily", columns=[
            ColumnSchema(name="sale_date", type="DATE"),
            ColumnSchema(name="total_revenue", type="REAL"),
        ]),
        "mart_sales_monthly": TableSchema(name="mart_sales_monthly", columns=[
            ColumnSchema(name="month", type="DATE"),
            ColumnSchema(name="total_revenue", type="REAL"),
        ]),
        "rpt_daily": TableSchema(name="rpt_daily", columns=[
            ColumnSchema(name="report_date", type="DATE"),
            ColumnSchema(name="revenue", type="REAL"),
        ]),
        "dw_sales_order": TableSchema(name="dw_sales_order", columns=[
            ColumnSchema(name="status", type="TEXT"),
            ColumnSchema(name="total_amount", type="REAL"),
        ]),
    }

    class TraversalLLM:
        def __init__(self):
            self.shown: list[str] = []
        def generate(self, system: str, user: str, **kwargs):
            import re
            table = re.search(r"Table: (\w+)", system)
            shown = table.group(1) if table else ""
            self.shown.append(shown)
            if shown == "dw_sales_order":
                content = "```sql\nSELECT status, COUNT(*) AS cnt, SUM(total_amount) AS total FROM dw_sales_order WHERE status='refunded' GROUP BY status\n```\nconfidence: 0.9"
            else:
                content = "```sql\nSELECT * FROM " + shown + "\n```\nconfidence: 0.5"
            return type("Response", (), {
                "content": content, "model": "t", "usage": {"completion_tokens": 10},
            })()
        def generate_json(self, *a, **kw):
            return {}

    class Executor:
        def execute(self, sql: str) -> pd.DataFrame:
            if "dw_sales_order" in sql:
                return pd.DataFrame({"status": ["refunded"], "cnt": [3], "total": [120.0]})
            return pd.DataFrame()
        def get_schema(self, table_name=None):
            return tables.get(table_name, tables["dw_sales_order"])
        def list_tables(self):
            return list(tables.keys())
        def close(self):
            pass

    config = ExecutionConfig(
        question="refund counts by reason", db_path=":memory:",
        max_attempts_per_node=1,
    )
    llm = TraversalLLM()
    executor = Executor()
    pool = [
        ScoredTable(schema=tables["archive"], score=5.0),
        ScoredTable(schema=tables["mart_sales_daily"], score=2.0),
        ScoredTable(schema=tables["mart_sales_monthly"], score=2.0),
        ScoredTable(schema=tables["rpt_daily"], score=2.0),
        ScoredTable(schema=tables["dw_sales_order"], score=2.0),
    ]
    agent = RLMAgent(llm, executor, config, candidate_pool=pool, all_schemas=list(tables.values()))

    node = TaskNode(id="A", description="refund counts", is_atomic=True)
    result = agent.solve(node)

    assert llm.shown == ["archive", "mart_sales_daily", "mart_sales_monthly", "rpt_daily", "dw_sales_order"]
    assert result.sql == "SELECT status, COUNT(*) AS cnt, SUM(total_amount) AS total FROM dw_sales_order WHERE status='refunded' GROUP BY status"
    assert result.error is None


def test_rlm_zero_score_candidates_excluded():
    """Zero-score candidates are not part of the traversal order."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ScoredTable, TableSchema, ColumnSchema

    tables = {
        "dw_sales_order": TableSchema(name="dw_sales_order", columns=[
            ColumnSchema(name="status", type="TEXT"),
            ColumnSchema(name="total_amount", type="REAL"),
        ]),
        "sqlite_stat1": TableSchema(name="sqlite_stat1", columns=[
            ColumnSchema(name="tbl", type="TEXT"),
        ]),
    }

    agent = RLMAgent(
        FakeLLM(), FakeExecutor(), ExecutionConfig(question="t", db_path=":memory:"),
        candidate_pool=[
            ScoredTable(schema=tables["dw_sales_order"], score=2.0),
            ScoredTable(schema=tables["sqlite_stat1"], score=0.0),
        ],
        all_schemas=list(tables.values()),
    )
    node = TaskNode(id="A", description="t", is_atomic=True)
    order = agent._build_candidate_order(node)
    names = [c.schema.name for c in order]
    assert "dw_sales_order" in names
    assert "sqlite_stat1" not in names


def test_rlm_hint_tables_tried_first():
    """Hint tables come before score-ranked pool candidates, regardless of score."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ScoredTable, TableSchema, ColumnSchema

    tables = {
        "dw_sales_order": TableSchema(name="dw_sales_order", columns=[
            ColumnSchema(name="status", type="TEXT"),
            ColumnSchema(name="total_amount", type="REAL"),
        ]),
        "archive": TableSchema(name="archive", columns=[
            ColumnSchema(name="total", type="REAL"),
        ]),
    }

    agent = RLMAgent(
        FakeLLM(), FakeExecutor(), ExecutionConfig(question="t", db_path=":memory:"),
        candidate_pool=[
            ScoredTable(schema=tables["archive"], score=5.0),
            ScoredTable(schema=tables["dw_sales_order"], score=2.0),
        ],
        all_schemas=list(tables.values()),
    )
    node = TaskNode(id="A", description="t", is_atomic=True, hint_tables=["dw_sales_order"])
    order = agent._build_candidate_order(node)
    names = [c.schema.name for c in order]
    assert names == ["dw_sales_order", "archive"]


def test_rlm_explores_candidates_within_budget():
    """First high-confidence success does NOT stop the search: candidates are
    explored until the beam floor / budget is met. Two candidates with
    indistinguishable evidence are reported AMBIGUOUS — never resolved by
    execution order."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import NodeStatus, ScoredTable, TableSchema, ColumnSchema

    tables = {
        "t1": TableSchema(name="t1", columns=[
            ColumnSchema(name="amount", type="REAL"),
        ]),
        "t2": TableSchema(name="t2", columns=[
            ColumnSchema(name="amount", type="REAL"),
        ]),
    }

    class GreedyLLM:
        def __init__(self):
            self.count = 0
        def generate(self, system: str, user: str, **kwargs):
            import re
            self.count += 1
            m = re.search(r"Table: (\w+)", system)
            tbl = m.group(1) if m else "t1"
            return type("Response", (), {
                "content": f"```sql\nSELECT SUM(amount) AS total FROM {tbl}\n```\nconfidence: 0.95",
                "model": "t", "usage": {"completion_tokens": 10},
            })()
        def generate_json(self, *a, **kw):
            return {}

    class Executor:
        def execute(self, sql: str) -> pd.DataFrame:
            return pd.DataFrame({"total": [500.0]})
        def get_schema(self, table_name=None):
            return tables.get(table_name, tables["t1"])
        def list_tables(self):
            return list(tables.keys())
        def close(self):
            pass

    config = ExecutionConfig(
        question="test", db_path=":memory:",
        max_attempts_per_node=1,
    )
    llm = GreedyLLM()
    executor = Executor()
    pool = [
        ScoredTable(schema=tables["t1"], score=5.0),
        ScoredTable(schema=tables["t2"], score=2.0),
    ]
    agent = RLMAgent(llm, executor, config, candidate_pool=pool, all_schemas=list(tables.values()))

    node = TaskNode(id="A", description="test", is_atomic=True)
    result = agent.solve(node)

    # Both candidates are explored despite the first returning high confidence
    assert llm.count == 2
    assert result.confidence == pytest.approx(0.95, rel=1e-2)
    assert result.sql in (
        "SELECT SUM(amount) AS total FROM t1",
        "SELECT SUM(amount) AS total FROM t2",
    )
    # Both candidates produced identical evidence (same SQL shape, same data) →
    # the node must not fake a winner.
    assert result.status == NodeStatus.AMBIGUOUS
    assert result.selected_candidate is None
    assert len(result.candidates) == 2


def test_rlm_ambiguous_expands_then_solves():
    """Budget-exhausted AMBIGUOUS triggers candidate expansion; a distinct later
    candidate resolves the search to SOLVED (never a fake execution-order win)."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import (
        NodeStatus, RequirementSpec, ScoredTable, TableSchema, ColumnSchema,
    )

    tables = {
        "m0": TableSchema(name="m0", columns=[ColumnSchema(name="revenue", type="REAL")]),
        "m1": TableSchema(name="m1", columns=[ColumnSchema(name="revenue", type="REAL")]),
        "gt_monthly": TableSchema(name="gt_monthly", columns=[
            ColumnSchema(name="total_amount", type="REAL"),
            ColumnSchema(name="month", type="TEXT"),
        ]),
    }

    class ExpandLLM:
        def __init__(self):
            self.shown: list[str] = []
        def generate(self, system: str, user: str, **kwargs):
            import re
            table = re.search(r"Table: (\w+)", system)
            shown = table.group(1) if table else ""
            self.shown.append(shown)
            sql = (
                "SELECT month, SUM(total_amount) AS total FROM gt_monthly GROUP BY month"
                if shown == "gt_monthly"
                else f"SELECT SUM(revenue) AS total FROM {shown}"
            )
            return type("Response", (), {
                "content": f"```sql\n{sql}\n```\nconfidence: 0.9",
                "model": "t", "usage": {"completion_tokens": 10},
            })()
        def generate_json(self, *a, **kw):
            return {}

    class Executor:
        def execute(self, sql: str) -> pd.DataFrame:
            if "gt_monthly" in sql:
                return pd.DataFrame({"month": ["Jan"], "total": [300.0]})
            return pd.DataFrame({"total": [100.0]})
        def get_schema(self, table_name=None):
            return tables.get(table_name, tables["m0"])
        def list_tables(self):
            return list(tables.keys())
        def close(self):
            pass

    config = ExecutionConfig(
        question="total monthly", db_path=":memory:",
        max_attempts_per_node=1,
        search_policy="beam",
        beam_width=1,
        candidate_budget=2,
        max_candidate_expansion=2,
    )
    llm = ExpandLLM()
    executor = Executor()
    pool = [
        ScoredTable(schema=tables["m0"], score=3.0),
        ScoredTable(schema=tables["m1"], score=2.0),
        ScoredTable(schema=tables["gt_monthly"], score=1.0),
    ]
    agent = RLMAgent(llm, executor, config, candidate_pool=pool, all_schemas=list(tables.values()))

    node = TaskNode(
        id="A", description="total monthly", is_atomic=True,
        requirements=RequirementSpec(metrics=["total"], aggregation="sum", grain="monthly"),
    )
    result = agent.solve(node)

    # Expansion explored the 3rd candidate beyond the budget of 2
    assert llm.shown == ["m0", "m1", "gt_monthly"]
    assert result.status == NodeStatus.SOLVED
    assert result.selected_candidate is not None
    assert result.selected_candidate.table == "gt_monthly"


def test_rlm_ambiguous_persists_without_new_evidence():
    """When expansion adds no distinguishing evidence, the node stays AMBIGUOUS
    with selected_candidate=None (BLOCKED-safe output for the aggregator)."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import NodeStatus, RequirementSpec, ScoredTable, TableSchema, ColumnSchema

    tables = {
        "m0": TableSchema(name="m0", columns=[ColumnSchema(name="revenue", type="REAL")]),
        "m1": TableSchema(name="m1", columns=[ColumnSchema(name="revenue", type="REAL")]),
        "m2": TableSchema(name="m2", columns=[ColumnSchema(name="revenue", type="REAL")]),
    }

    class SameLLM:
        def __init__(self):
            self.shown: list[str] = []
        def generate(self, system: str, user: str, **kwargs):
            import re
            table = re.search(r"Table: (\w+)", system)
            shown = table.group(1) if table else ""
            self.shown.append(shown)
            sql = f"SELECT SUM(revenue) AS total FROM {shown}"
            return type("Response", (), {
                "content": f"```sql\n{sql}\n```\nconfidence: 0.9",
                "model": "t", "usage": {"completion_tokens": 10},
            })()
        def generate_json(self, *a, **kw):
            return {}

    class Executor:
        def execute(self, sql: str) -> pd.DataFrame:
            return pd.DataFrame({"total": [100.0]})
        def get_schema(self, table_name=None):
            return tables.get(table_name, tables["m0"])
        def list_tables(self):
            return list(tables.keys())
        def close(self):
            pass

    config = ExecutionConfig(
        question="total", db_path=":memory:",
        max_attempts_per_node=1,
        search_policy="beam",
        beam_width=1,
        candidate_budget=2,
        max_candidate_expansion=2,
    )
    agent = RLMAgent(
        SameLLM(), Executor(), config,
        candidate_pool=[ScoredTable(schema=t, score=float(i + 1)) for i, t in enumerate(tables.values())],
        all_schemas=list(tables.values()),
    )
    node = TaskNode(
        id="A", description="total", is_atomic=True,
        requirements=RequirementSpec(metrics=["total"], aggregation="sum"),
    )
    llm = agent.llm
    result = agent.solve(node)

    assert result.status == NodeStatus.AMBIGUOUS
    assert result.selected_candidate is None
    assert len(result.candidates) == 3  # expansion exhausted the pool
    assert llm.shown == ["m2", "m1", "m0"]  # score order, all explored


def test_rlm_constraint_failure_retries():
    """Constraint failures now retry with feedback instead of moving to next candidate."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import RequirementSpec

    class RetryLLM:
        def __init__(self):
            self.count = 0
        def generate(self, system: str, user: str, **kwargs):
            self.count += 1
            if self.count == 1:
                sql = "SELECT total_revenue FROM mart_sales_daily"  # missing SUM
            else:
                sql = "SELECT SUM(total_revenue) FROM mart_sales_daily"
            return type("Response", (), {
                "content": f"```sql\n{sql}\n```\nconfidence: 0.7",
                "model": "t", "usage": {"completion_tokens": 10},
            })()
        def generate_json(self, *a, **kw):
            return {}

    class Executor:
        def execute(self, sql: str) -> pd.DataFrame:
            return pd.DataFrame({"total_revenue": [100.0]})
        def get_schema(self, table_name=None):
            from syrch.core.models import ColumnSchema, TableSchema
            return TableSchema(name="mart_sales_daily", columns=[
                ColumnSchema(name="total_revenue", type="REAL"),
            ])
        def list_tables(self):
            return ["mart_sales_daily"]
        def close(self):
            pass

    config = ExecutionConfig(
        question="test", db_path=":memory:",
        max_attempts_per_node=3,
    )
    llm = RetryLLM()
    agent = RLMAgent(llm, Executor(), config)

    node = TaskNode(
        id="A", description="total revenue", is_atomic=True,
        requirements=RequirementSpec(metrics=["revenue"], aggregation="sum"),
    )
    result = agent.solve(node)

    assert llm.count == 2
    assert result.sql == "SELECT SUM(total_revenue) FROM mart_sales_daily"


def test_retriever_policy_filter_keeps_no_zero_score():
    """CandidatePolicy.filter must not let zero-score tables back in when the
    threshold pass returns fewer than max_candidates (S14 regression)."""
    from syrch.core.models import CandidatePolicy, ScoredTable, TableSchema

    table = TableSchema(name="t", columns=[])
    candidates = [
        ScoredTable(schema=table, score=s)
        for s in [5.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
    policy = CandidatePolicy(max_candidates=10)
    filtered = policy.filter(candidates)
    assert [c.score for c in filtered] == [5.0, 2.0, 2.0, 2.0, 2.0]
    assert all(c.score > 0 for c in filtered)


def test_hint_section_renders_structured_requirements():
    """REQUIREMENTS block surfaces metric semantics / filters / supporting
    relations to the RLM without dictating physical columns."""
    from syrch.core.models import (
        FilterRequirement,
        MetricRequirement,
        RequirementSpec,
        SupportingRelation,
    )
    from syrch.search.rlm_engine import RLMAgent

    config = ExecutionConfig(question="test", db_path=":memory:")
    llm = FakeLLM()
    executor = FakeExecutor()
    agent = RLMAgent(llm, executor, config)

    node = TaskNode(
        id="A",
        description="daily revenue excluding holidays",
        is_atomic=True,
        requirements=RequirementSpec(
            metrics=["revenue"],
            metric_details=[
                MetricRequirement(
                    name="delivery_time",
                    aggregation="avg",
                    expression_semantics="time between payment and shipment",
                )
            ],
            grain="day",
            filters=[
                FilterRequirement(semantic="holidays", predicate="is_holiday = false",
                                  relation="dim_date")
            ],
            supporting_relations=[
                SupportingRelation(table="dim_date", purpose="exclude holidays",
                                   join_hint="date = sale_date")
            ],
        ),
    )

    hint = agent._build_hint_section(node)
    assert "REQUIREMENTS" in hint
    assert "time between payment and shipment" in hint
    assert "is_holiday = false" in hint
    assert "dim_date" in hint
    assert "grain: day" in hint
    assert "paid_at" not in hint
    assert "shipped_at" not in hint


def test_diff_unit_argument_is_not_unknown_column():
    """S6: `TIMESTAMPDIFF(DAY, ...)` / `DATEDIFF(day, ...)` unit is a keyword,
    not a column — `_validate_schema` must not reject it as unknown."""
    from syrch.search.rlm_engine import RLMAgent

    agent = RLMAgent.__new__(RLMAgent)
    agent._allowed_tables = None
    agent._attempt_scope = None
    agent.all_schemas = None

    class Executor:
        def list_tables(self):
            return ["dw_sales_order"]
        def get_schema(self, table_name=None):
            from syrch.core.models import ColumnSchema, TableSchema
            return TableSchema(
                name="dw_sales_order",
                columns=[ColumnSchema(name="paid_at", type="TIMESTAMP"),
                         ColumnSchema(name="shipped_at", type="TIMESTAMP")],
            )
    agent.executor = Executor()

    sql = "SELECT TIMESTAMPDIFF(DAY, paid_at, shipped_at) AS dt FROM dw_sales_order"
    assert agent._validate_schema(sql) is None

    sql2 = "SELECT DATEDIFF(day, paid_at, shipped_at) AS dt FROM dw_sales_order"
    assert agent._validate_schema(sql2) is None


def test_two_arg_datediff_keeps_first_column():
    """S6 guard: `DATEDIFF(shipped_at, paid_at)` has no unit arg — the first
    column is a real column and must still be validated."""
    from syrch.search.rlm_engine import RLMAgent

    agent = RLMAgent.__new__(RLMAgent)
    agent._allowed_tables = None
    agent._attempt_scope = None
    agent.all_schemas = None

    class Executor:
        def list_tables(self):
            return ["t"]
        def get_schema(self, table_name=None):
            from syrch.core.models import ColumnSchema, TableSchema
            return TableSchema(
                name="t",
                columns=[ColumnSchema(name="paid_at", type="TIMESTAMP")],
            )
    agent.executor = Executor()

    # shipped_at not in schema -> still flagged
    sql = "SELECT DATEDIFF(shipped_at, paid_at) AS dt FROM t"
    assert agent._validate_schema(sql) is not None


def test_alias_with_within_is_not_question_text():
    """S7: `successful_purchases_within_24h` is a valid domain alias; 'within'
    is a time preposition, not question text."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import TaskNode

    agent = RLMAgent.__new__(RLMAgent)
    node = TaskNode(id="A", description="conversion rate within 24h", is_atomic=True)
    sql = "SELECT COUNT(*) AS successful_purchases_within_24h FROM fact_event_log"
    assert agent._validate_aliases(sql, node) is None


def test_alias_starting_with_question_word_still_rejected():
    """S7 guard: `what_is_revenue` genuinely reads like question text and must
    still be rejected."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import TaskNode

    agent = RLMAgent.__new__(RLMAgent)
    node = TaskNode(id="A", description="revenue", is_atomic=True)
    sql = "SELECT SUM(total_revenue) AS what_is_revenue FROM t"
    assert agent._validate_aliases(sql, node) is not None


def test_join_candidate_exposed_in_scope():
    """S3 Test 1: with primary=dw_customer, a pool table sharing a join key
    (customer_id) is exposed as JOIN-AVAILABLE."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ColumnSchema, ScoredTable, TableSchema, TaskNode

    schemas = [
        TableSchema(name="dw_customer", columns=[ColumnSchema(name="customer_id", type="INTEGER")]),
        TableSchema(name="dw_sales_order", columns=[ColumnSchema(name="customer_id", type="INTEGER")]),
        TableSchema(name="fact_event_log", columns=[ColumnSchema(name="event_id", type="INTEGER")]),
    ]
    agent = RLMAgent.__new__(RLMAgent)
    agent.all_schemas = schemas
    pool = [ScoredTable(schema=s, score=1.0) for s in schemas]
    agent._candidate_pool = pool

    node = TaskNode(id="A", description="customer orders", is_atomic=True)
    cand = pool[0]  # dw_customer
    scope = agent._build_attempt_schemas(node, cand)

    assert scope.primary_tables == {"dw_customer"}
    assert "dw_sales_order" in scope.join_available_tables
    # fact_event_log shares no key with dw_customer -> not exposed
    assert "fact_event_log" not in scope.join_available_tables
    assert scope.allowed_tables == {"dw_customer", "dw_sales_order"}


def test_pool_table_outside_scope_is_rejected():
    """S3 Test 2: a table in the pool but outside the attempt scope must be
    rejected by the FROM guard even though it exists."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ColumnSchema, ScoredTable, TableSchema, TaskNode

    schemas = [
        TableSchema(name="dw_customer", columns=[ColumnSchema(name="customer_id", type="INTEGER")]),
        TableSchema(name="dw_sales_order", columns=[ColumnSchema(name="customer_id", type="INTEGER")]),
        TableSchema(name="fact_event_log", columns=[ColumnSchema(name="event_id", type="INTEGER")]),
    ]
    agent = RLMAgent.__new__(RLMAgent)
    agent.all_schemas = schemas
    pool = [ScoredTable(schema=s, score=1.0) for s in schemas]
    agent._candidate_pool = pool

    node = TaskNode(id="A", description="customer orders", is_atomic=True)
    cand = pool[0]  # dw_customer
    scope = agent._build_attempt_schemas(node, cand)
    agent._attempt_scope = scope
    agent._allowed_tables = scope.allowed_tables

    # fact_event_log is in the pool but not in this attempt's scope
    err = agent._validate_schema("SELECT event_id FROM fact_event_log")
    assert err is not None and "outside the allowed search scope" in err


def test_join_across_scope_tables_is_allowed():
    """S3 Test 3: FROM primary JOIN join-available passes the FROM guard."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ColumnSchema, ScoredTable, TableSchema, TaskNode

    schemas = [
        TableSchema(name="dw_customer", columns=[ColumnSchema(name="customer_id", type="INTEGER")]),
        TableSchema(name="dw_sales_order", columns=[ColumnSchema(name="customer_id", type="INTEGER")]),
    ]
    agent = RLMAgent.__new__(RLMAgent)
    agent.all_schemas = schemas
    pool = [ScoredTable(schema=s, score=1.0) for s in schemas]
    agent._candidate_pool = pool

    node = TaskNode(id="A", description="customer orders", is_atomic=True)
    cand = pool[0]  # dw_customer
    scope = agent._build_attempt_schemas(node, cand)
    agent._attempt_scope = scope
    agent._allowed_tables = scope.allowed_tables

    sql = (
        "SELECT o.customer_id FROM dw_customer c "
        "JOIN dw_sales_order o ON c.customer_id = o.customer_id"
    )
    assert agent._validate_schema(sql) is None


def test_s5_drift_guard_preserved_under_scope():
    """S3 Test 4 (S5 regression): a candidate that cannot reach the real table
    through the scope must still be rejected — no re-broadening of the FROM
    guard to the whole pool."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ColumnSchema, ScoredTable, TableSchema, TaskNode

    schemas = [
        TableSchema(name="mart_sales_monthly", columns=[ColumnSchema(name="revenue", type="REAL")]),
        TableSchema(name="dw_sales_order", columns=[ColumnSchema(name="total_amount", type="REAL")]),
    ]
    agent = RLMAgent.__new__(RLMAgent)
    agent.all_schemas = schemas

    class Executor:
        def list_tables(self):
            return ["mart_sales_monthly", "dw_sales_order"]
        def get_schema(self, table_name=None):
            return schemas[0] if table_name == "mart_sales_monthly" else schemas[1]
    agent.executor = Executor()

    pool = [ScoredTable(schema=schemas[0], score=1.0)]
    agent._candidate_pool = pool

    node = TaskNode(id="A", description="monthly revenue", is_atomic=True)
    cand = pool[0]
    scope = agent._build_attempt_schemas(node, cand)
    agent._attempt_scope = scope
    agent._allowed_tables = scope.allowed_tables

    # mart_sales_monthly and dw_sales_order share no columns and no hint
    assert scope.join_available_tables == set()
    err = agent._validate_schema("SELECT SUM(total_amount) FROM dw_sales_order")
    assert err is not None and "outside the allowed search scope" in err


def test_join_available_table_in_from_is_primary_switch():
    """S3 invariant: a JOIN-AVAILABLE table used as the FROM anchor is a
    'primary switch' — rejected even though the table is within allowed_tables."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ColumnSchema, ScoredTable, TableSchema, TaskNode

    schemas = [
        TableSchema(name="mart_sales_daily", columns=[ColumnSchema(name="sale_date", type="DATE")]),
        TableSchema(name="mart_sales_monthly", columns=[ColumnSchema(name="sale_month", type="TEXT"), ColumnSchema(name="total_revenue", type="REAL")]),
    ]
    agent = RLMAgent.__new__(RLMAgent)
    agent.all_schemas = schemas

    class Executor:
        def list_tables(self):
            return ["mart_sales_daily", "mart_sales_monthly"]
        def get_schema(self, table_name=None):
            return schemas[0] if table_name == "mart_sales_daily" else schemas[1]
    agent.executor = Executor()

    pool = [ScoredTable(schema=s, score=1.0) for s in schemas]
    agent._candidate_pool = pool

    node = TaskNode(id="A", description="monthly revenue", is_atomic=True)
    node.hint_tables = ["mart_sales_monthly"]  # planner hint -> JOIN-AVAILABLE
    scope = agent._build_attempt_schemas(node, pool[0])  # primary = mart_sales_daily
    agent._attempt_scope = scope
    agent._allowed_tables = scope.allowed_tables
    assert "mart_sales_monthly" in scope.join_available_tables

    # S5 log signature: candidate=mart_sales_daily but FROM anchors on the
    # monthly table (hint in JOIN-AVAILABLE). Must be rejected as a switch.
    sql = "SELECT sale_month, total_revenue FROM mart_sales_monthly WHERE sale_month LIKE '2024%'"
    err = agent._validate_schema(sql)
    assert err is not None and "appears in FROM" in err
    assert "mart_sales_daily" in err

    # anchored on the candidate -> passes
    ok_sql = "SELECT sale_date FROM mart_sales_daily"
    assert agent._validate_schema(ok_sql) is None


def test_from_primary_with_join_available_passes_invariant():
    """S3 invariant: FROM primary + JOIN join-available passes the position
    guard (complements test_join_across_scope_tables_is_allowed)."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import ColumnSchema, ScoredTable, TableSchema, TaskNode

    schemas = [
        TableSchema(name="dw_customer", columns=[ColumnSchema(name="customer_id", type="INTEGER")]),
        TableSchema(name="dw_sales_order", columns=[ColumnSchema(name="customer_id", type="INTEGER")]),
    ]
    agent = RLMAgent.__new__(RLMAgent)
    agent.all_schemas = schemas

    class Executor:
        def list_tables(self):
            return ["dw_customer", "dw_sales_order"]
        def get_schema(self, table_name=None):
            return schemas[0] if table_name == "dw_customer" else schemas[1]
    agent.executor = Executor()

    pool = [ScoredTable(schema=s, score=1.0) for s in schemas]
    agent._candidate_pool = pool

    node = TaskNode(id="A", description="customer orders", is_atomic=True)
    scope = agent._build_attempt_schemas(node, pool[0])
    agent._attempt_scope = scope
    agent._allowed_tables = scope.allowed_tables

    sql = (
        "SELECT c.customer_id FROM dw_customer c "
        "JOIN dw_sales_order o ON c.customer_id = o.customer_id"
    )
    assert agent._validate_schema(sql) is None


def test_task_context_in_from_is_allowed_when_materialized():
    """S3 invariant v0.3.5b: a materialized TASK CONTEXT is a real table and
    MAY be used in FROM (scope matrix: FROM = PRIMARY | TASK CONTEXT)."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import (
        ColumnSchema, ParentContext, ScoredTable, TableSchema, TaskNode,
    )

    schemas = [
        TableSchema(name="dw_sales_order", columns=[ColumnSchema(name="total_amount", type="REAL")]),
    ]
    agent = RLMAgent.__new__(RLMAgent)
    agent.all_schemas = schemas

    class Executor:
        def list_tables(self):
            return ["dw_sales_order"]
        def get_schema(self, table_name=None):
            return schemas[0]
    agent.executor = Executor()

    pool = [ScoredTable(schema=schemas[0], score=1.0)]
    agent._candidate_pool = pool

    node = TaskNode(id="B", description="revenue by order", is_atomic=True)
    scope = agent._build_attempt_schemas(node, pool[0])
    scope.task_contexts = [
        ParentContext(
            node_id="A", table_name="_task_context_A",
            columns=[ColumnSchema(name="status", type="TEXT")],
            source_tables=["dw_sales_order"],
            materialized=True,
        )
    ]
    agent._attempt_scope = scope
    agent._allowed_tables = scope.allowed_tables

    err = agent._validate_schema("SELECT total_amount FROM _task_context_A")
    assert err is None


def test_task_context_in_join_is_allowed_when_materialized():
    """S3 invariant v0.3.5b: a materialized TASK CONTEXT may be JOINed
    (scope matrix: JOIN = PRIMARY | JOIN-AVAILABLE | TASK CONTEXT)."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import (
        ColumnSchema, ParentContext, ScoredTable, TableSchema, TaskNode,
    )

    schemas = [
        TableSchema(name="dw_sales_order", columns=[ColumnSchema(name="total_amount", type="REAL")]),
    ]
    agent = RLMAgent.__new__(RLMAgent)
    agent.all_schemas = schemas

    class Executor:
        def list_tables(self):
            return ["dw_sales_order"]
        def get_schema(self, table_name=None):
            return schemas[0]
    agent.executor = Executor()

    pool = [ScoredTable(schema=schemas[0], score=1.0)]
    agent._candidate_pool = pool

    node = TaskNode(id="B", description="revenue by order", is_atomic=True)
    scope = agent._build_attempt_schemas(node, pool[0])
    scope.task_contexts = [
        ParentContext(
            node_id="A", table_name="_task_context_A",
            columns=[ColumnSchema(name="order_id", type="INTEGER")],
            source_tables=["dw_sales_order"],
            materialized=True,
        )
    ]
    agent._attempt_scope = scope
    agent._allowed_tables = scope.allowed_tables

    sql = (
        "SELECT s.total_amount FROM dw_sales_order s "
        "JOIN _task_context_A c ON s.order_id = c.order_id"
    )
    assert agent._validate_schema(sql) is None


def test_unmaterialized_task_context_in_from_is_rejected():
    """S3 invariant v0.3.5b: a dependency result that was NOT materialized
    (AMBIGUOUS/FAILED parent) is not a real table and must not be FROM/JOINed.
    Only SOLVED parents materialize."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import (
        ColumnSchema, ParentContext, ScoredTable, TableSchema, TaskNode,
    )

    schemas = [
        TableSchema(name="dw_sales_order", columns=[ColumnSchema(name="total_amount", type="REAL")]),
    ]
    agent = RLMAgent.__new__(RLMAgent)
    agent.all_schemas = schemas

    class Executor:
        def list_tables(self):
            return ["dw_sales_order"]
        def get_schema(self, table_name=None):
            return schemas[0]
    agent.executor = Executor()

    pool = [ScoredTable(schema=schemas[0], score=1.0)]
    agent._candidate_pool = pool

    node = TaskNode(id="B", description="revenue by order", is_atomic=True)
    scope = agent._build_attempt_schemas(node, pool[0])
    scope.task_contexts = [
        ParentContext(
            node_id="A", table_name="_task_context_A",
            columns=[ColumnSchema(name="status", type="TEXT")],
            source_tables=["dw_sales_order"],
            materialized=False,
        )
    ]
    agent._attempt_scope = scope
    agent._allowed_tables = scope.allowed_tables

    err = agent._validate_schema("SELECT total_amount FROM _task_context_A")
    assert err is not None and "TASK CONTEXT" in err
    """S3 invariant v0.3.5b: a `_task_context_*` name that was never produced
    is a schema-level mistake, even though it is not a known physical table."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.models import (
        ColumnSchema, ParentContext, ScoredTable, TableSchema, TaskNode,
    )

    schemas = [
        TableSchema(name="dw_sales_order", columns=[ColumnSchema(name="total_amount", type="REAL")]),
    ]
    agent = RLMAgent.__new__(RLMAgent)
    agent.all_schemas = schemas

    class Executor:
        def list_tables(self):
            return ["dw_sales_order"]
        def get_schema(self, table_name=None):
            return schemas[0]
    agent.executor = Executor()

    pool = [ScoredTable(schema=schemas[0], score=1.0)]
    agent._candidate_pool = pool

    node = TaskNode(id="B", description="revenue by order", is_atomic=True)
    scope = agent._build_attempt_schemas(node, pool[0])
    scope.task_contexts = [
        ParentContext(
            node_id="A", table_name="_task_context_A",
            columns=[ColumnSchema(name="status", type="TEXT")],
            source_tables=["dw_sales_order"],
            materialized=True,
        )
    ]
    agent._attempt_scope = scope
    agent._allowed_tables = scope.allowed_tables

    err = agent._validate_schema("SELECT total_amount FROM _task_context_GHOST")
    assert err is not None and "TASK CONTEXT" in err


def test_prompt_includes_proxy_column_rule():
    """RLM_SYSTEM rule 11 guides value-based proxy approximation without
    leaking GT-specific column/value mappings."""
    from syrch.search.rlm_engine import RLM_SYSTEM

    assert "business state or reason" in RLM_SYSTEM
    assert "categorical column" in RLM_SYSTEM
    assert "Never invent a column name" in RLM_SYSTEM
    assert "refunded" not in RLM_SYSTEM
    assert "refund_reason" not in RLM_SYSTEM


def test_proxy_rule_rendered_in_attempt_prompt():
    """The proxy guidance reaches the LLM in the per-attempt system prompt."""
    from syrch.search.rlm_engine import RLMAgent, RLM_SYSTEM
    from syrch.core.models import (
        ColumnSchema, ScoredTable, TableSchema, TaskNode,
    )

    schemas = [
        TableSchema(name="dw_sales_order", columns=[
            ColumnSchema(name="status", type="TEXT"),
            ColumnSchema(name="total_amount", type="REAL"),
        ]),
    ]
    agent = RLMAgent.__new__(RLMAgent)
    agent.all_schemas = schemas

    class Executor:
        def list_tables(self):
            return ["dw_sales_order"]
        def get_schema(self, table_name=None):
            return schemas[0]
    agent.executor = Executor()
    agent.config = ExecutionConfig(question="refund by reason", db_path=":memory:", max_attempts_per_node=1)
    agent.llm = None
    agent.retriever = None
    agent.alias_map = {}
    agent._compressed_schemas = None
    agent._candidate_pool = [ScoredTable(schema=schemas[0], score=1.0)]

    node = TaskNode(id="A", description="refund by reason", is_atomic=True)
    scope = agent._build_attempt_schemas(node, agent._candidate_pool[0])
    agent._attempt_scope = scope
    agent._allowed_tables = scope.allowed_tables

    schema_str = agent._build_schema_str()
    system = RLM_SYSTEM.format(
        schema=schema_str,
        task_description=node.description,
        hint_section="",
    )
    assert "categorical column" in system
    assert "business state or reason" in system
    assert "refunded" not in system
    assert "refund_reason" not in system




