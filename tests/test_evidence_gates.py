"""Evidence-gate regression tests (Databricks S-run follow-up).

Covers the three gaps found by re-running the failing VIP net-revenue case
against real data:

- N1: an aggregate over an empty join set returns a 1-row all-NULL frame —
  it must be treated like an empty result (retry + signal), never as viable
  data.
- N3: an AMBIGUOUS leaf's provisional data must not become the final answer
  basis (the alphabetical tie-break winner used to leak upward).
- GT guard: `_guess_gt_table` exists for log annotation only and must have
  zero influence on planning, selection or output.
"""

import pandas as pd
import pytest

from syrch.core.config import ExecutionConfig
from syrch.core.models import (
    ColumnSchema,
    NodeResult,
    NodeStatus,
    RequirementSpec,
    TableSchema,
    TaskNode,
)


class StaticLLM:
    """Returns the same SQL for every attempt; records prompts."""

    def __init__(self, sql: str, confidence: str = "0.9"):
        self.sql = sql
        self.confidence = confidence
        self.prompts: list[str] = []

    def generate(self, system: str, user: str, **kwargs):
        self.prompts.append(user)
        content = f"```sql\n{self.sql}\n```\nconfidence: {self.confidence}"
        return type("Response", (), {
            "content": content, "model": "t", "usage": {"completion_tokens": 10},
        })()

    def generate_json(self, *a, **kw):
        return {}


def _schema(name: str, columns: list[tuple[str, str]]) -> TableSchema:
    return TableSchema(
        name=name,
        columns=[ColumnSchema(name=n, type=t) for n, t in columns],
    )


# ---------------------------------------------------------------------------
# N1: all-NULL aggregate results are not viable data
# ---------------------------------------------------------------------------

NULL_TOTAL_SQL = (
    "SELECT SUM(o.total) AS total_net_revenue FROM archive_orders o "
    "JOIN customers c ON o.customer_id = c.customer_id "
    "WHERE c.segment = 'VIP' AND o.order_date >= '2024-01-01'"
)


class NullThenRealExecutor:
    """archive-anchored SQL → 1-row NULL total; customers-anchored → real rows."""

    def __init__(self):
        self.queries: list[str] = []

    def execute(self, sql: str) -> pd.DataFrame:
        self.queries.append(sql)
        # Existence probes find nothing anywhere → no cross-table contrast →
        # the candidate-skip shortcut must not fire; exercises the feedback path.
        if sql.strip().upper().startswith("SELECT COUNT"):
            return pd.DataFrame({"n": [0]})
        if "FROM archive_orders" in sql or "FROM syrch.archive_orders" in sql:
            return pd.DataFrame({"total_net_revenue": [None]})
        return pd.DataFrame({"total_net_revenue": [81959.47]})

    def get_schema(self, table_name=None):
        if table_name and table_name.endswith("customers"):
            return _schema("customers", [
                ("customer_id", "INTEGER"), ("segment", "TEXT"),
                ("revenue_amount", "DOUBLE"),
            ])
        if table_name and table_name.endswith("archive_orders"):
            return _schema("archive_orders", [
                ("customer_id", "INTEGER"), ("order_date", "DATE"), ("total", "DOUBLE"),
            ])
        return _schema("customers", [("customer_id", "INTEGER")])

    def list_tables(self):
        return ["customers", "archive_orders"]

    def close(self):
        pass


def test_all_null_aggregate_retries_with_feedback_and_fails():
    """A NULL-total result is retried with explicit feedback (like empty) and,
    when it never recovers, does NOT become a viable candidate."""
    from syrch.core.models import ScoredTable
    from syrch.search.rlm_engine import RLMAgent

    config = ExecutionConfig(
        question="vip revenue", db_path=":memory:",
        max_attempts_per_node=3, candidate_budget=1,
    )
    llm = StaticLLM(NULL_TOTAL_SQL)
    schemas = [
        _schema("archive_orders", [
            ("customer_id", "INTEGER"), ("order_date", "DATE"), ("total", "DOUBLE"),
        ]),
        _schema("customers", [("customer_id", "INTEGER"), ("segment", "TEXT")]),
    ]
    agent = RLMAgent(
        llm, NullThenRealExecutor(), config,
        candidate_pool=[
            ScoredTable(schema=schemas[0], score=2.0),
            ScoredTable(schema=schemas[1], score=1.0),
        ],
        all_schemas=schemas,
    )

    node = TaskNode(
        id="A", description="VIP net revenue", is_atomic=True,
        requirements=RequirementSpec(
            metrics=["revenue"], aggregation="sum",
            time_range=("2024-01-01", "2024-12-31"),
        ),
    )
    result = agent.solve(node)

    assert result.status == NodeStatus.FAILED
    # All three attempts ran and the retry prompt carried the NULL guidance.
    assert len(llm.prompts) == 3
    assert "every result column is NULL" in llm.prompts[1]


def test_all_null_candidate_loses_to_real_data_candidate():
    """Two candidates: the one whose aggregation matches no rows (NULL total)
    is not viable; the one returning real rows is selected."""
    from syrch.core.models import ScoredTable
    from syrch.search.rlm_engine import RLMAgent

    schemas = [
        _schema("archive_orders", [
            ("customer_id", "INTEGER"), ("order_date", "DATE"), ("total", "DOUBLE"),
        ]),
        _schema("customers", [
            ("customer_id", "INTEGER"), ("segment", "TEXT"),
            ("revenue_amount", "DOUBLE"),
        ]),
    ]

    class TwoTableLLM(StaticLLM):
        def generate(self, system: str, user: str, **kwargs):
            self.prompts.append(user)
            primary_section = system.split("JOIN-AVAILABLE")[0]
            primary = (
                "archive_orders"
                if "PRIMARY TABLES" in system and "archive_orders" in primary_section
                else "customers"
            )
            sql = (
                "SELECT SUM(o.total) AS total_net_revenue FROM archive_orders o "
                "JOIN customers c ON o.customer_id = c.customer_id"
                if primary == "archive_orders"
                else "SELECT SUM(revenue_amount) AS total_net_revenue FROM customers"
            )
            content = f"```sql\n{sql}\n```\nconfidence: 0.9"
            return type("Response", (), {
                "content": content, "model": "t", "usage": {"completion_tokens": 10},
            })()

    executor = NullThenRealExecutor()
    config = ExecutionConfig(
        question="vip revenue", db_path=":memory:",
        max_attempts_per_node=2, candidate_budget=2,
    )
    agent = RLMAgent(
        TwoTableLLM(""), executor, config,
        candidate_pool=[ScoredTable(schema=s, score=float(i + 1)) for i, s in enumerate(schemas)],
        all_schemas=list(schemas),
    )
    node = TaskNode(
        id="A", description="VIP net revenue", is_atomic=True,
        requirements=RequirementSpec(metrics=["revenue"], aggregation="sum"),
    )
    result = agent.solve(node)

    assert result.status == NodeStatus.SOLVED
    assert result.sql == "SELECT SUM(revenue_amount) AS total_net_revenue FROM customers"
    by_table = {c.table: c for c in result.candidates}
    assert by_table["archive_orders"].viable is False


# ---------------------------------------------------------------------------
# N3: AMBIGUOUS leaf data is never the answer basis
# ---------------------------------------------------------------------------

def _agg_config():
    return ExecutionConfig(question="q", db_path=":memory:", materialize_context=False)


class AggLLM:
    def generate(self, *a, **kw):
        return type("Response", (), {
            "content": "final answer", "model": "t", "usage": {"completion_tokens": 5},
        })()

    def generate_json(self, *a, **kw):
        return {}


def test_ambiguous_leaf_data_not_used_as_answer_basis():
    """When the only leaf is AMBIGUOUS its provisional (NULL-total) frame must
    not become solution.data — but confidence still flows, discounted."""
    from syrch.search.aggregator import Aggregator
    from syrch.core.models import TaskDAG

    dag = TaskDAG(
        nodes={"C": TaskNode(id="C", description="final merge", is_atomic=True)},
        root_id="C", topo_layers=[["C"]],
    )
    ambiguous = NodeResult(
        node_id="C",
        data=pd.DataFrame({"total_net_revenue": [None]}),
        sql="SELECT SUM(ao.total) AS total_net_revenue FROM archive_orders ao",
        confidence=0.9,
        status=NodeStatus.AMBIGUOUS,
        ambiguity_score=1.0,
    )
    aggregator = Aggregator(AggLLM(), NullThenRealExecutor(), _agg_config())
    solution = aggregator.merge("q", dag, {"C": ambiguous})

    assert solution.data is None
    # Ambiguity discount applied to the leaf's own evidence (×0.5 at minimum).
    assert solution.confidence <= 0.45


def test_solved_leaf_keeps_answer_basis():
    """A SOLVED leaf keeps carrying data — the gate only fires on ambiguity."""
    from syrch.search.aggregator import Aggregator
    from syrch.core.models import TaskDAG

    dag = TaskDAG(
        nodes={"C": TaskNode(id="C", description="final", is_atomic=True)},
        root_id="C", topo_layers=[["C"]],
    )
    solved = NodeResult(
        node_id="C",
        data=pd.DataFrame({"total_net_revenue": [81959.47]}),
        sql="SELECT SUM(o.total) AS total_net_revenue FROM dw_sales_order o",
        confidence=0.95,
        status=NodeStatus.SOLVED,
    )
    aggregator = Aggregator(AggLLM(), NullThenRealExecutor(), _agg_config())
    solution = aggregator.merge("q", dag, {"C": solved})

    assert solution.data is not None
    assert solution.data.iloc[0, 0] == pytest.approx(81959.47)
    assert solution.confidence > 0.5


# ---------------------------------------------------------------------------
# Calculation basis must state WHICH criteria produced THIS number
# ---------------------------------------------------------------------------

class Scd2SchemaExecutor(NullThenRealExecutor):
    """Every lookup returns a customers-like schema carrying SCD2 columns."""

    def get_schema(self, table_name=None):
        return TableSchema(name=table_name or "customers", columns=[
            ColumnSchema(name="customer_id", type="INTEGER"),
            ColumnSchema(name="segment", type="TEXT"),
            ColumnSchema(name="valid_from", type="DATE"),
            ColumnSchema(name="valid_to", type="DATE"),
        ])


def _criteria_for(sql: str) -> str:
    from syrch.search.aggregator import Aggregator

    agg = Aggregator(AggLLM(), Scd2SchemaExecutor(), _agg_config())
    return agg._extract_criteria(sql)


def test_basis_reports_fixed_window_overlap_and_missing_refund_filter():
    sql = (
        "SELECT customer_id FROM dw_customer "
        "WHERE segment = 'VIP' AND valid_from <= '2024-12-31' "
        "AND (valid_to > '2024-01-01' OR valid_to IS NULL)"
    )
    basis = _criteria_for(sql)
    assert "time window: 2024-01-01 .. 2024-12-31" in basis
    assert "NOT APPLIED" in basis
    assert "fixed-window overlap approximation" in basis
    assert "NOT evaluated per order" in basis


def test_basis_reports_point_in_time_scd2_and_refund_exclusion():
    sql = (
        "SELECT SUM(o.total_amount) AS v FROM dw_sales_order o "
        "JOIN dw_customer c ON o.customer_id = c.customer_id "
        "AND o.order_date >= c.valid_from "
        "AND (c.valid_to IS NULL OR o.order_date <= c.valid_to) "
        "WHERE o.status != 'refunded' AND c.segment = 'VIP'"
    )
    basis = _criteria_for(sql)
    assert "point-in-time" in basis
    assert "applied (status != 'refunded')" in basis


def test_basis_reports_ignored_validity_window():
    sql = "SELECT SUM(o.total) FROM orders o JOIN customers c ON o.customer_id = c.customer_id"
    basis = _criteria_for(sql)
    assert "validity window ignored" in basis


def test_calculation_basis_prefixed_into_solution():
    """merge() output carries the criteria block ahead of the steps."""
    from syrch.search.aggregator import Aggregator
    from syrch.core.models import TaskDAG

    dag = TaskDAG(
        nodes={"C": TaskNode(id="C", description="final", is_atomic=True)},
        root_id="C", topo_layers=[["C"]],
    )
    solved = NodeResult(
        node_id="C",
        data=pd.DataFrame({"total_net_revenue": [81959.47]}),
        sql=(
            "SELECT SUM(o.total_amount) AS total_net_revenue "
            "FROM dw_sales_order o JOIN dw_customer c "
            "ON o.customer_id = c.customer_id AND o.order_date >= c.valid_from "
            "WHERE o.status != 'refunded'"
        ),
        confidence=0.95,
        status=NodeStatus.SOLVED,
    )
    aggregator = Aggregator(AggLLM(), Scd2SchemaExecutor(), _agg_config())
    solution = aggregator.merge("q", dag, {"C": solved})

    assert solution.calculation_basis.startswith("Criteria applied")
    assert "point-in-time" in solution.calculation_basis
    assert "Step 1" in solution.calculation_basis


# ---------------------------------------------------------------------------
# GT guesser must stay log-only
# ---------------------------------------------------------------------------

class _PipelineFakeExecutor(NullThenRealExecutor):
    def get_schema(self, table_name=None):  # keep both tables fully described
        if table_name and table_name.endswith("archive_orders"):
            return _schema("syrch.enterprise.archive_orders", [
                ("customer_id", "INTEGER"), ("order_date", "DATE"), ("total", "DOUBLE"),
            ])
        return _schema("syrch.enterprise.customers", [
            ("customer_id", "INTEGER"), ("segment", "TEXT"),
            ("signup_date", "DATE"), ("revenue_amount", "DOUBLE"),
        ])

    def list_tables(self):
        return ["syrch.enterprise.customers", "syrch.enterprise.archive_orders"]


class _PipelineLLM:
    def generate(self, system: str, user: str, **kwargs):
        content = (
            "```sql\nSELECT SUM(revenue_amount) AS total_revenue "
            "FROM syrch.enterprise.customers\n```\nconfidence: 0.9"
        )
        return type("Response", (), {
            "content": content, "model": "t", "usage": {"completion_tokens": 10},
        })()

    def generate_json(self, *a, **kw):
        return {
            "subtasks": [{
                "id": "A",
                "description": "net revenue",
                "depends_on": [],
                "is_atomic": True,
                "expected_output": "total",
                "hint_tables": ["syrch.enterprise.customers"],
            }]
        }


def test_gt_guesser_has_no_decision_influence(monkeypatch):
    """Pinning the contract: whether the log-only GT guesser returns the right
    table, a wrong table, or nothing, the produced solution and DAG hints are
    byte-identical."""
    import syrch.search.pipeline as pipeline_mod
    from syrch.core.models import ProblemSpec

    def _run(gt_return):
        monkeypatch.setattr(pipeline_mod, "_guess_gt_table", lambda q: gt_return)
        executor = _PipelineFakeExecutor()
        problem = ProblemSpec(
            question="total revenue",
            schema=executor.get_schema("syrch.enterprise.customers"),
            all_schemas=[executor.get_schema(t) for t in executor.list_tables()],
        )
        solution, dag, _ = pipeline_mod.run_pipeline(
            _PipelineLLM(), executor,
            ExecutionConfig(question="total revenue", db_path=":memory:", verbose=False),
            problem,
        )
        return solution, dag

    sol_gt, dag_gt = _run("syrch.enterprise.customers")
    sol_none, dag_none = _run(None)
    sol_wrong, _ = _run("syrch.enterprise.archive_orders")

    assert sol_gt.sql == sol_none.sql == sol_wrong.sql
    assert sol_gt.confidence == sol_none.confidence == sol_wrong.confidence
    hints_gt = [n.hint_tables for n in dag_gt.nodes.values()]
    hints_none = [n.hint_tables for n in dag_none.nodes.values()]
    assert hints_gt == hints_none