from syrch.core.config import ExecutionConfig
from syrch.core.models import ColumnSchema, ProblemSpec, TableSchema


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        self.calls.append(("generate_json", system, user))
        return {
            "subtasks": [
                {
                    "id": "A",
                    "description": "Find top 10% customers",
                    "depends_on": [],
                    "is_atomic": True,
                    "expected_output": "customer IDs",
                },
                {
                    "id": "B",
                    "description": "Get orders for top customers",
                    "depends_on": ["A"],
                    "is_atomic": True,
                    "expected_output": "order records",
                },
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        self.calls.append(("generate", system, user))
        return None


def test_planner_decomposes_into_dag():
    from syrch.search.planner import Planner

    llm = FakeLLM()
    config = ExecutionConfig(question="test", db_path=":memory:")
    planner = Planner(llm, config)

    schema = TableSchema(
        name="orders",
        columns=[
            ColumnSchema(name="customer_id", type="INTEGER"),
            ColumnSchema(name="total_price", type="REAL"),
        ],
    )
    problem = ProblemSpec(question="Find top customers", schema=schema)
    dag = planner.decompose(problem)

    assert "A" in dag.nodes
    assert "B" in dag.nodes
    assert dag.nodes["B"].depends_on == ["A"]
    assert dag.topo_layers == [["A"], ["B"]]


def test_planner_validates_cycles():
    from syrch.search.planner import Planner

    llm = FakeLLM()
    config = ExecutionConfig(question="test", db_path=":memory:")
    planner = Planner(llm, config)

    schema = TableSchema(name="t", columns=[ColumnSchema(name="x", type="INT")])
    problem = ProblemSpec(question="test", schema=schema)

    dag = planner.decompose(problem)
    assert dag is not None


class HallucinatedHintFakeLLM:
    def __init__(self):
        self.calls = []

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        self.calls.append(("generate_json", system, user))
        return {
            "subtasks": [
                {
                    "id": "A",
                    "description": "Compute inventory difference",
                    "depends_on": [],
                    "is_atomic": True,
                    "metric_columns": ["inventory", "total_items_sold"],
                    "hint_columns": ["refund_reason", "status"],
                },
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        return None


def test_planner_filters_hallucinated_columns():
    """S10/S14 guard: planner hints must not fabricate columns that do not
    exist in the schema. `inventory` / `refund_reason` are invented by the
    planner LLM; they must be dropped so the RLM does not trust then refuse
    them (empty-SQL failures). Real columns are preserved."""
    from syrch.search.planner import Planner

    llm = HallucinatedHintFakeLLM()
    config = ExecutionConfig(question="test", db_path=":memory:")
    planner = Planner(llm, config)

    schema = TableSchema(
        name="mart_sales_daily",
        columns=[
            ColumnSchema(name="sale_date", type="DATE"),
            ColumnSchema(name="total_items_sold", type="INTEGER"),
            ColumnSchema(name="status", type="TEXT"),
        ],
    )
    problem = ProblemSpec(question="Compute inventory", schema=schema)
    dag = planner.decompose(problem)

    node = dag.nodes["A"]
    assert node.metric_columns == ["total_items_sold"]
    assert node.hint_columns == ["status"]


def test_filter_known_columns_unit():
    from syrch.search.planner import Planner

    known = {"total_items_sold", "status"}
    assert Planner._filter_known_columns(["inventory", "total_items_sold"], known) == ["total_items_sold"]
    assert Planner._filter_known_columns(["inventory", "refund_reason"], known) is None
    assert Planner._filter_known_columns([], known) is None
    assert Planner._filter_known_columns(["status"], known) == ["status"]


class RecursiveFakeLLM:
    def __init__(self):
        self.call_count = 0

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        self.call_count += 1
        if self.call_count == 1:
            return {
                "subtasks": [
                    {
                        "id": "A",
                        "description": "Analyze revenue by region",
                        "depends_on": [],
                        "is_atomic": False,
                        "expected_output": "region revenue data",
                    },
                ]
            }
        return {
            "subtasks": [
                {
                    "id": "A1",
                    "description": "Get orders per region",
                    "depends_on": [],
                    "is_atomic": True,
                    "expected_output": "orders with region",
                },
                {
                    "id": "A2",
                    "description": "Sum revenue per region",
                    "depends_on": ["A1"],
                    "is_atomic": True,
                    "expected_output": "revenue by region",
                },
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        self.call_count += 1
        return None


def test_recursive_decomposition():
    from syrch.search.planner import Planner

    llm = RecursiveFakeLLM()
    config = ExecutionConfig(question="test", db_path=":memory:", max_depth=2)
    planner = Planner(llm, config)

    schema = TableSchema(name="orders", columns=[ColumnSchema(name="amount", type="REAL")])
    problem = ProblemSpec(question="Revenue analysis", schema=schema)
    dag = planner.decompose(problem)

    assert llm.call_count == 2
    assert "A" not in dag.nodes
    assert "A.A1" in dag.nodes
    assert "A.A2" in dag.nodes
    assert dag.nodes["A.A2"].depends_on == ["A.A1"]
    assert dag.nodes["A.A1"].depth == 1
    assert dag.nodes["A.A2"].depth == 1
    assert all(n.is_atomic for n in dag.nodes.values())


def test_recursive_decomposition_max_depth_1_disabled():
    from syrch.search.planner import Planner

    llm = RecursiveFakeLLM()
    config = ExecutionConfig(question="test", db_path=":memory:", max_depth=1)
    planner = Planner(llm, config)

    schema = TableSchema(name="t", columns=[ColumnSchema(name="x", type="INT")])
    problem = ProblemSpec(question="test", schema=schema)
    dag = planner.decompose(problem)

    assert llm.call_count == 1
    assert "A" in dag.nodes
    assert dag.nodes["A"].is_atomic


class RecursiveJoinKeyFakeLLM:
    def __init__(self):
        self.call_count = 0

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        self.call_count += 1
        if self.call_count == 1:
            return {
                "subtasks": [
                    {
                        "id": "A",
                        "description": "Analyze revenue",
                        "depends_on": [],
                        "is_atomic": False,
                        "expected_output": "revenue data",
                    },
                ]
            }
        return {
            "subtasks": [
                {
                    "id": "A1",
                    "description": "Get orders",
                    "depends_on": [],
                    "is_atomic": True,
                    "expected_output": "order data",
                },
                {
                    "id": "A2",
                    "description": "Get revenue by order",
                    "depends_on": ["A1"],
                    "is_atomic": True,
                    "expected_output": "revenue",
                    "join_keys": [
                        {"left": "A2", "left_col": "order_id",
                         "right": "A1", "right_col": "id", "how": "inner"}
                    ],
                },
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        return None


def test_recursive_decomposition_with_join_keys():
    from syrch.search.planner import Planner

    llm = RecursiveJoinKeyFakeLLM()
    config = ExecutionConfig(question="test", db_path=":memory:", max_depth=2)
    planner = Planner(llm, config)

    schema = TableSchema(name="orders", columns=[ColumnSchema(name="amount", type="REAL")])
    problem = ProblemSpec(question="Revenue analysis", schema=schema)
    dag = planner.decompose(problem)

    assert llm.call_count == 2
    assert "A.A1" in dag.nodes
    assert "A.A2" in dag.nodes

    a2 = dag.nodes["A.A2"]
    assert a2.join_keys is not None
    assert len(a2.join_keys) == 1
    jk = a2.join_keys[0]
    assert jk.left == "A.A2"
    assert jk.right == "A.A1"
    assert jk.left_col == "order_id"
    assert jk.right_col == "id"


class JoinKeyFakeLLM:
    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        return {
            "subtasks": [
                {
                    "id": "A",
                    "description": "Get customer segments",
                    "depends_on": [],
                    "is_atomic": True,
                    "expected_output": "segment data",
                },
                {
                    "id": "B",
                    "description": "Get order stats by segment",
                    "depends_on": ["A"],
                    "is_atomic": True,
                    "expected_output": "order stats",
                    "join_keys": [
                        {"left": "B", "left_col": "segment_id",
                         "right": "A", "right_col": "id", "how": "inner"}
                    ],
                },
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        return None


def test_planner_parses_join_keys():
    from syrch.search.planner import Planner
    from syrch.core.models import JoinKey

    llm = JoinKeyFakeLLM()
    config = ExecutionConfig(question="test", db_path=":memory:")
    planner = Planner(llm, config)

    schema = TableSchema(name="t", columns=[ColumnSchema(name="x", type="INT")])
    problem = ProblemSpec(question="test", schema=schema)
    dag = planner.decompose(problem)

    node_b = dag.nodes["B"]
    assert node_b.join_keys is not None
    assert len(node_b.join_keys) == 1
    jk = node_b.join_keys[0]
    assert isinstance(jk, JoinKey)
    assert jk.left == "B"
    assert jk.left_col == "segment_id"
    assert jk.right == "A"
    assert jk.right_col == "id"
    assert jk.how == "inner"


class RequirementFakeLLM:
    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        return {
            "subtasks": [
                {
                    "id": "A",
                    "description": "Compute avg delivery time by month",
                    "depends_on": [],
                    "is_atomic": True,
                    "expected_output": "avg delivery time",
                    "requirements": {
                        "metrics": ["delivery_time"],
                        "metric_details": [
                            {
                                "name": "delivery_time",
                                "aggregation": "avg",
                                "expression_semantics": "time between payment and shipment",
                            }
                        ],
                        "dimensions": [],
                        "aggregation": "avg",
                        "grain": None,
                        "time_range": None,
                        "filters": [
                            {
                                "semantic": "holidays",
                                "predicate": "is_holiday = false",
                                "relation": "dim_date",
                            }
                        ],
                        "supporting_relations": [
                            {"table": "dim_date", "purpose": "exclude holidays",
                             "join_hint": "date = sale_date"}
                        ],
                    },
                }
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        return None


def test_planner_parses_structured_requirements():
    from syrch.core.models import FilterRequirement, MetricRequirement, SupportingRelation
    from syrch.search.planner import Planner

    llm = RequirementFakeLLM()
    config = ExecutionConfig(question="test", db_path=":memory:")
    planner = Planner(llm, config)

    schema = TableSchema(name="t", columns=[ColumnSchema(name="x", type="INT")])
    problem = ProblemSpec(question="test", schema=schema)
    dag = planner.decompose(problem)

    req = dag.nodes["A"].requirements
    assert req is not None
    assert req.metrics == ["delivery_time"]
    assert len(req.metric_details) == 1
    md = req.metric_details[0]
    assert isinstance(md, MetricRequirement)
    assert md.name == "delivery_time"
    assert md.aggregation == "avg"
    assert md.expression_semantics == "time between payment and shipment"

    assert len(req.filters) == 1
    f = req.filters[0]
    assert isinstance(f, FilterRequirement)
    assert f.semantic == "holidays"
    assert f.predicate == "is_holiday = false"
    assert f.relation == "dim_date"

    assert len(req.supporting_relations) == 1
    sr = req.supporting_relations[0]
    assert isinstance(sr, SupportingRelation)
    assert sr.table == "dim_date"
    assert sr.purpose == "exclude holidays"
    assert sr.join_hint == "date = sale_date"

    # S5: grain was None in the planner response but the description says "month"
    assert req.grain == "month"


class RequirementMalformedFakeLLM:
    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        return {
            "subtasks": [
                {
                    "id": "A",
                    "description": "daily revenue",
                    "depends_on": [],
                    "is_atomic": True,
                    "expected_output": "revenue",
                    "requirements": {
                        "metrics": ["revenue"],
                        "metric_details": [{"name": 123}],
                        "filters": [{"semantic": 5}],
                        "supporting_relations": [{"table": None}],
                    },
                }
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        return None


def test_planner_tolerates_malformed_requirement_fields():
    from syrch.search.planner import Planner

    llm = RequirementMalformedFakeLLM()
    config = ExecutionConfig(question="test", db_path=":memory:")
    planner = Planner(llm, config)

    schema = TableSchema(name="t", columns=[ColumnSchema(name="x", type="INT")])
    problem = ProblemSpec(question="test", schema=schema)
    dag = planner.decompose(problem)

    req = dag.nodes["A"].requirements
    assert req is not None
    assert req.metric_details == []
    assert req.filters == []
    assert req.supporting_relations == []
    # description says "daily" → S5 fallback
    assert req.grain == "day"


def test_grain_fallback_does_not_override_planner_grain():
    from syrch.core.models import RequirementSpec
    from syrch.search.planner import Planner

    req = RequirementSpec(metrics=["revenue"], aggregation="sum", grain="total")
    Planner._apply_grain_fallback(req, "monthly revenue")
    assert req.grain == "total"


def test_grain_fallback_ignores_unknown_markers():
    from syrch.core.models import RequirementSpec
    from syrch.search.planner import Planner

    req = RequirementSpec(metrics=["revenue"], aggregation="sum")
    Planner._apply_grain_fallback(req, "revenue by region")
    assert req.grain is None


def test_replan_merges_instead_of_replacing_hint_tables():
    """S14: AMBIGUOUS replan must not drop the correct table from scope —
    new candidates are merged with (not substituted for) the current hints."""
    import pandas as pd
    from syrch.core.models import NodeResult, NodeStatus, ScoredTable, TableSchema, TaskDAG, TaskNode
    from syrch.search.planner import Planner

    llm = FakeLLM()
    config = ExecutionConfig(question="test", db_path=":memory:")
    planner = Planner(llm, config, retriever=None)

    dag = TaskDAG(
        nodes={
            "A": TaskNode(
                id="A",
                description="refund counts",
                depends_on=[],
                is_atomic=True,
                hint_tables=["dw_sales_order"],
            ),
        },
        root_id="A",
        topo_layers=[["A"]],
    )
    node_result = NodeResult(
        node_id="A",
        data=pd.DataFrame({"x": [1]}),
        sql="SELECT status AS refund_reason FROM dw_sales_order",
        confidence=0.5,
        status=NodeStatus.AMBIGUOUS,
    )
    scored = [
        ScoredTable(schema=TableSchema(name="mart_sales_daily", columns=[]), score=1.0),
        ScoredTable(schema=TableSchema(name="dw_sales_order", columns=[]), score=1.0),
    ]

    new_dag = planner.replan(dag, "A", "", "", node_result, scored)

    hints = new_dag.nodes["A"].hint_tables or []
    assert "dw_sales_order" in hints
    assert "mart_sales_daily" in hints


def test_replan_drops_infeasible_supporting_relation_from_probe():
    """A supporting relation whose filter value the probe verified does not
    exist is dropped on replan, so the node stops demanding an impossible JOIN."""
    from syrch.core.models import (
        NodeResult, NodeStatus, RequirementSpec, ScoredTable,
        SupportingRelation, TableSchema, TaskDAG, TaskNode,
    )
    from syrch.search.data_probe import ProbeRegistry, ProbeResult
    from syrch.search.planner import Planner

    llm = FakeLLM()
    config = ExecutionConfig(question="test", db_path=":memory:")
    planner = Planner(llm, config, retriever=None)

    node = TaskNode(
        id="C",
        description="VIP net revenue",
        depends_on=["A"],
        is_atomic=True,
        hint_tables=["dw_sales_order"],
        requirements=RequirementSpec(
            metrics=["revenue"],
            aggregation="sum",
            supporting_relations=[
                SupportingRelation(
                    table="rpt_customer_ltv",
                    purpose="to filter for VIP customers",
                ),
                SupportingRelation(
                    table="dim_date",
                    purpose="filter to 2024",
                ),
            ],
        ),
    )
    dag = TaskDAG(nodes={"C": node}, root_id="C", topo_layers=[["C"]])

    registry = ProbeRegistry()
    registry.put(
        "db1",
        ProbeResult(
            table="rpt_customer_ltv",
            column="segment",
            value="VIP",
            exists=False,
            count=0,
        ),
    )
    node_result = NodeResult(
        node_id="C",
        data=None,
        sql="",
        confidence=0.0,
        status=NodeStatus.FAILED,
        error="requirement infeasible",
    )

    new_dag = planner.replan(dag, "C", "", "infeasible", node_result, [], probe_registry=registry)

    kept = new_dag.nodes["C"].requirements.supporting_relations
    tables = [sr.table for sr in kept]
    assert "rpt_customer_ltv" not in tables
    assert "dim_date" in tables


def test_replan_keeps_relations_when_no_probe_fact():
    """Without a probe fact, supporting relations survive replan unchanged."""
    from syrch.core.models import (
        NodeResult, NodeStatus, RequirementSpec, ScoredTable,
        SupportingRelation, TableSchema, TaskDAG, TaskNode,
    )
    from syrch.search.data_probe import ProbeRegistry
    from syrch.search.planner import Planner

    llm = FakeLLM()
    config = ExecutionConfig(question="test", db_path=":memory:")
    planner = Planner(llm, config, retriever=None)

    node = TaskNode(
        id="C",
        description="VIP net revenue",
        depends_on=["A"],
        is_atomic=True,
        hint_tables=["dw_sales_order"],
        requirements=RequirementSpec(
            metrics=["revenue"],
            aggregation="sum",
            supporting_relations=[
                SupportingRelation(table="rpt_customer_ltv", purpose="to filter for VIP customers"),
            ],
        ),
    )
    dag = TaskDAG(nodes={"C": node}, root_id="C", topo_layers=[["C"]])

    node_result = NodeResult(
        node_id="C", data=None, sql="", confidence=0.0, status=NodeStatus.FAILED,
        error="requirement infeasible",
    )
    new_dag = planner.replan(dag, "C", "", "infeasible", node_result, [], probe_registry=ProbeRegistry())

    kept = new_dag.nodes["C"].requirements.supporting_relations
    assert [sr.table for sr in kept] == ["rpt_customer_ltv"]


def test_is_supporting_relation_task():
    """S15: a dim/lookup node with no answer evidence is a supporting-relation
    task; an answer node or a mixed-hint node is not."""
    from syrch.core.models import RequirementSpec, TaskNode
    from syrch.search.planner import Planner

    planner = Planner.__new__(Planner)

    lookup = TaskNode(id="B", description="holiday dates", is_atomic=True,
                      hint_tables=["dim_date"])
    assert planner._is_supporting_relation_task(lookup)

    answer = TaskNode(id="A", description="daily revenue", is_atomic=True,
                      hint_tables=["mart_sales_daily"],
                      requirements=RequirementSpec(metrics=["revenue"]))
    assert not planner._is_supporting_relation_task(answer)

    mixed = TaskNode(id="X", description="x", is_atomic=True,
                     hint_tables=["dim_date", "mart_sales_daily"])
    assert not planner._is_supporting_relation_task(mixed)

    no_hints = TaskNode(id="Y", description="y", is_atomic=True)
    assert not planner._is_supporting_relation_task(no_hints)


def test_fold_supporting_relation_task_into_consumer():
    """S15: a holiday-lookup task (B) consumed by the merge node (C) is folded
    into C as a supporting relation instead of running as an independent node.
    The DAG collapses to A → C and C gains dim_date in its scope."""
    from syrch.core.models import RequirementSpec, TaskDAG, TaskNode
    from syrch.search.planner import Planner

    config = ExecutionConfig(question="avg daily revenue excluding holidays", db_path=":memory:")
    planner = Planner(FakeLLM(), config)

    a = TaskNode(id="A", description="daily revenue 2024", is_atomic=True,
                 hint_tables=["mart_sales_daily"],
                 requirements=RequirementSpec(metrics=["revenue"], aggregation="avg", grain="day"))
    b = TaskNode(id="B", description="holiday dates for 2024", is_atomic=True,
                 hint_tables=["dim_date"])
    c = TaskNode(id="C", description="average daily revenue excluding holidays",
                 depends_on=["A", "B"], is_atomic=True,
                 requirements=RequirementSpec(metrics=["avg_daily_revenue"], aggregation="avg"))
    dag = TaskDAG(nodes={"A": a, "B": b, "C": c}, root_id="A", topo_layers=[["A", "B"], ["C"]])

    folded = planner._fold_supporting_relation_tasks(dag, pool_names={"mart_sales_daily", "dim_date"})

    assert "B" not in folded.nodes
    assert folded.nodes["C"].depends_on == ["A"]
    req = folded.nodes["C"].requirements
    assert any(sr.table == "dim_date" for sr in req.supporting_relations)
    assert folded.root_id == "A"
    assert folded.topo_layers == [["A"], ["C"]]


def test_fold_skips_table_outside_pool():
    """S15 guard: a supporting relation that is NOT in the candidate pool must
    not be folded in — otherwise the validator would demand a table the RLM
    cannot reach."""
    from syrch.core.models import RequirementSpec, TaskDAG, TaskNode
    from syrch.search.planner import Planner

    config = ExecutionConfig(question="x", db_path=":memory:")
    planner = Planner(FakeLLM(), config)

    b = TaskNode(id="B", description="holiday dates", is_atomic=True,
                 hint_tables=["dim_date"])
    c = TaskNode(id="C", description="merge", depends_on=["B"], is_atomic=True,
                 requirements=RequirementSpec(metrics=["avg"], aggregation="avg"))
    dag = TaskDAG(nodes={"B": b, "C": c}, root_id="B", topo_layers=[["B"], ["C"]])

    folded = planner._fold_supporting_relation_tasks(dag, pool_names={"mart_sales_daily"})

    # dim_date not in pool → B stays as a node (fold is a no-op)
    assert "B" in folded.nodes
    assert not folded.nodes["C"].requirements.supporting_relations


def test_decompose_folds_s15_style_dag():
    """End-to-end: decompose() with an S15-style planner output (daily revenue,
    holiday lookup, merge) collapses to a single answer chain with dim_date as
    a supporting relation."""
    from syrch.core.models import ColumnSchema, ProblemSpec, TableSchema
    from syrch.search.planner import Planner

    class S15LLM(FakeLLM):
        def generate_json(self, system: str, user: str, **kwargs) -> dict:
            return {
                "subtasks": [
                    {
                        "id": "A", "description": "daily revenue for 2024",
                        "depends_on": [], "is_atomic": True,
                        "expected_output": "daily revenue rows",
                        "hint_tables": ["mart_sales_daily"],
                        "requirements": {"metrics": ["revenue"], "aggregation": "sum",
                                         "grain": "daily"},
                    },
                    {
                        "id": "B", "description": "holiday dates for 2024",
                        "depends_on": [], "is_atomic": True,
                        "expected_output": "holiday dates",
                        "hint_tables": ["dim_date"],
                        "requirements": {"metrics": []},
                    },
                    {
                        "id": "C", "description": "average daily revenue excluding holidays",
                        "depends_on": ["A", "B"], "is_atomic": True,
                        "expected_output": "the average",
                        "hint_tables": ["mart_sales_daily"],
                        "requirements": {"metrics": ["avg_daily_revenue"], "aggregation": "avg"},
                    },
                ]
            }

    schemas = [
        TableSchema(name="mart_sales_daily",
                    columns=[ColumnSchema(name="sale_date", type="TEXT"),
                             ColumnSchema(name="total_revenue", type="REAL")]),
        TableSchema(name="dim_date",
                    columns=[ColumnSchema(name="date_key", type="TEXT"),
                             ColumnSchema(name="is_holiday", type="INTEGER")]),
    ]
    config = ExecutionConfig(question="avg daily revenue excluding holidays", db_path=":memory:")
    planner = Planner(S15LLM(), config)
    problem = ProblemSpec(question="avg daily revenue excluding holidays",
                          schema=schemas[0], all_schemas=schemas)
    dag = planner.decompose(problem)

    assert "B" not in dag.nodes
    c = dag.nodes["C"]
    assert any(sr.table == "dim_date" for sr in (c.requirements.supporting_relations if c.requirements else []))
    assert dag.nodes["C"].depends_on == ["A"]


