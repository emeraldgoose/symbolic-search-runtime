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
    from syrch.core.models import JoinKey

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
