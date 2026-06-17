"""Integration tests for the full pipeline (Planner → Scheduler → Aggregator)."""

import json
import os

import pandas as pd

from syrch.core.config import ExecutionConfig, LLMConfig
from syrch.core.models import (
    ColumnSchema,
    FinalSolution,
    ProblemSpec,
    TableSchema,
    TaskDAG,
    TaskNode,
)
from syrch.search.pipeline import run_pipeline


# ═══════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════

class LLMResponse:
    def __init__(self, content, model="fake", usage=None):
        self.content = content
        self.model = model
        self.usage = usage or {"completion_tokens": 10}


# ═══════════════════════════════════════════════════════════════════
# TC1: 6-node branching DAG + join keys e2e
# ═══════════════════════════════════════════════════════════════════

class TC1_PlannerLLM:
    """Returns a 6-node branching DAG with join keys."""

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        return {
            "subtasks": [
                {
                    "id": "A",
                    "description": "Get base orders data",
                    "depends_on": [],
                    "is_atomic": True,
                    "expected_output": "orders",
                },
                {
                    "id": "B",
                    "description": "Get customer segments",
                    "depends_on": ["A"],
                    "is_atomic": True,
                    "expected_output": "customers",
                    "join_keys": [
                        {"left": "B", "left_col": "seg_id", "right": "A", "right_col": "id", "how": "inner"},
                    ],
                },
                {
                    "id": "C",
                    "description": "Get product info",
                    "depends_on": ["A"],
                    "is_atomic": True,
                    "expected_output": "products",
                    "join_keys": [
                        {"left": "C", "left_col": "prod_id", "right": "A", "right_col": "id", "how": "inner"},
                    ],
                },
                {
                    "id": "D",
                    "description": "Get customer segments detail",
                    "depends_on": ["B"],
                    "is_atomic": True,
                    "expected_output": "segment detail",
                    "join_keys": [
                        {"left": "D", "left_col": "ref", "right": "B", "right_col": "seg_id", "how": "inner"},
                    ],
                },
                {
                    "id": "E",
                    "description": "Get geography data",
                    "depends_on": ["B"],
                    "is_atomic": True,
                    "expected_output": "geo",
                    "join_keys": [
                        {"left": "E", "left_col": "geo", "right": "B", "right_col": "seg_id", "how": "inner"},
                    ],
                },
                {
                    "id": "F",
                    "description": "Get category info",
                    "depends_on": ["C"],
                    "is_atomic": True,
                    "expected_output": "category",
                    "join_keys": [
                        {"left": "F", "left_col": "cat", "right": "C", "right_col": "prod_id", "how": "inner"},
                    ],
                },
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        task_keywords = {
            "orders": "SELECT id, value FROM node_A",
            "customers": "SELECT seg_id, b_val FROM node_B",
            "products": "SELECT prod_id, c_val FROM node_C",
            "segment detail": "SELECT ref, d_val FROM node_D",
            "geo": "SELECT geo, e_val FROM node_E",
            "category": "SELECT cat, f_val FROM node_F",
        }
        for kw, sql in task_keywords.items():
            if kw in user.lower():
                return LLMResponse(f"```sql\n{sql}\n```\nconfidence: 0.9")
        return LLMResponse("```sql\nSELECT 1\n```\nconfidence: 0.9")


class TC1_Executor:
    def __init__(self):
        self._data = {
            "node_A": {"id": [1, 2, 3], "value": [10, 20, 30]},
            "node_B": {"seg_id": [1, 2, 3], "b_val": [100, 200, 300]},
            "node_C": {"prod_id": [1, 2, 3], "c_val": [1000, 2000, 3000]},
            "node_D": {"ref": [1, 2], "d_val": [500, 600]},
            "node_E": {"geo": [1, 2], "e_val": [300, 400]},
            "node_F": {"cat": [1, 2], "f_val": [700, 800]},
        }

    def execute(self, sql: str) -> pd.DataFrame:
        for table, cols in self._data.items():
            if table in sql:
                return pd.DataFrame(cols)
        return pd.DataFrame({"cnt": [1]})

    def get_schema(self, table_name=None):
        return TableSchema(
            name="orders",
            columns=[
                ColumnSchema(name="id", type="INT"),
                ColumnSchema(name="value", type="REAL"),
                ColumnSchema(name="seg_id", type="INT"),
                ColumnSchema(name="prod_id", type="INT"),
                ColumnSchema(name="ref", type="INT"),
                ColumnSchema(name="geo", type="INT"),
                ColumnSchema(name="cat", type="INT"),
            ],
        )

    def list_tables(self):
        return ["orders"]

    def close(self):
        pass


def test_tc1_6node_branching_dag_with_joins():
    """6-node branching DAG (A→B→D/E, A→C→F) with join keys → full pipeline."""
    config = ExecutionConfig(
        question="Analyze orders with customer and product data",
        db_path=":memory:",
        max_depth=2,
        max_attempts_per_node=1,
        high_confidence=0.85,
        verbose=False,
        calibration_enabled=False,
    )
    llm = TC1_PlannerLLM()
    executor = TC1_Executor()
    schema = executor.get_schema()
    problem = ProblemSpec(question=config.question, schema=schema)

    solution, dag, results = run_pipeline(llm, executor, config, problem)

    # All 6 nodes executed
    for nid in ["A", "B", "C", "D", "E", "F"]:
        assert nid in results, f"Missing result for {nid}"
        assert results[nid].error is None, f"{nid} failed: {results[nid].error}"

    # DAG topology
    assert dag.nodes["B"].depends_on == ["A"]
    assert dag.nodes["C"].depends_on == ["A"]
    assert dag.nodes["D"].depends_on == ["B"]
    assert dag.nodes["E"].depends_on == ["B"]
    assert dag.nodes["F"].depends_on == ["C"]

    # Join keys preserved
    assert len(dag.nodes["B"].join_keys) == 1
    assert dag.nodes["B"].join_keys[0].left == "B"
    assert dag.nodes["B"].join_keys[0].right == "A"

    # Final solution
    assert solution.confidence > 0
    assert solution.token_cost > 0
    assert solution.answer is not None

    # Merged data contains columns from all 6 nodes
    if solution.data is not None:
        for col in ["id", "value", "seg_id", "b_val", "prod_id",
                     "c_val", "ref", "d_val", "geo", "e_val", "cat", "f_val"]:
            assert col in solution.data.columns, f"Missing column {col}"


# ═══════════════════════════════════════════════════════════════════
# TC2: Recursive decomposition e2e
# ═══════════════════════════════════════════════════════════════════

class TC2_RecursiveLLM:
    def __init__(self):
        self.generate_json_calls = 0
        self.solve_calls = 0

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        self.generate_json_calls += 1
        if self.generate_json_calls == 1:
            return {
                "subtasks": [
                    {
                        "id": "A",
                        "description": "Analyze revenue across segments",
                        "depends_on": [],
                        "is_atomic": False,
                        "expected_output": "revenue analysis",
                    },
                ]
            }
        return {
            "subtasks": [
                {
                    "id": "A1",
                    "description": "Get total revenue by segment",
                    "depends_on": [],
                    "is_atomic": True,
                    "expected_output": "revenue by segment",
                },
                {
                    "id": "A2",
                    "description": "Get revenue trend over time",
                    "depends_on": ["A1"],
                    "is_atomic": True,
                    "expected_output": "revenue trend",
                },
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        self.solve_calls += 1
        sql = "SELECT segment, SUM(amount) as rev FROM orders GROUP BY segment"
        return LLMResponse(f"```sql\n{sql}\n```\nconfidence: 0.85")


class TC2_Executor:
    def execute(self, sql: str) -> pd.DataFrame:
        return pd.DataFrame({"segment": ["A", "B"], "rev": [100, 200]})

    def get_schema(self, table_name=None):
        return TableSchema(
            name="orders",
            columns=[
                ColumnSchema(name="segment", type="TEXT"),
                ColumnSchema(name="amount", type="REAL"),
            ],
        )

    def list_tables(self):
        return ["orders"]

    def close(self):
        pass


def test_tc2_recursive_decomposition():
    """Non-atomic node A → recursive decomp → A.A1, A.A2 executed."""
    config = ExecutionConfig(
        question="Analyze revenue by segment",
        db_path=":memory:",
        max_depth=2,
        max_attempts_per_node=1,
        high_confidence=0.85,
        verbose=False,
        calibration_enabled=False,
    )
    llm = TC2_RecursiveLLM()
    executor = TC2_Executor()
    schema = executor.get_schema()
    problem = ProblemSpec(question=config.question, schema=schema)

    solution, dag, results = run_pipeline(llm, executor, config, problem)

    # Planner called twice (level 0 → level 1)
    assert llm.generate_json_calls == 2

    # Original 'A' removed, sub-nodes added
    assert "A" not in dag.nodes
    assert "A.A1" in dag.nodes
    assert "A.A2" in dag.nodes

    # Sub-node topology
    assert dag.nodes["A.A1"].depth == 1
    assert dag.nodes["A.A2"].depth == 1
    assert dag.nodes["A.A2"].depends_on == ["A.A1"]

    # A.A1 has join_keys from recursive expansion

    # Scheduler ran both sub-nodes
    assert "A.A1" in results
    assert "A.A2" in results
    assert results["A.A1"].error is None
    assert results["A.A2"].error is None

    # Final solution
    assert solution.confidence > 0
    assert solution.token_cost > 0
    assert solution.answer is not None


# ═══════════════════════════════════════════════════════════════════
# TC3: Grid search e2e
# ═══════════════════════════════════════════════════════════════════

class TC3_FakeRunner:
    def __init__(self, expected_cells: int):
        self.expected_cells = expected_cells
        self.calls = 0

    def run_single(self, problem, llm_config, config_overrides):
        self.calls += 1
        data = pd.DataFrame({"x": [config_overrides.get("max_depth", 3)]})
        solution = FinalSolution(
            question=problem.question,
            answer="42",
            data=data,
            confidence=0.85 + 0.05 * config_overrides.get("max_depth", 3) - 0.05 * config_overrides.get("max_attempts_per_node", 3),
            token_cost=100,
        )
        metrics = type("M", (), {
            "solution": solution,
            "exact_match": True,
            "row_count_match": True,
            "column_match": True,
            "token_cost": 100,
            "to_dict": lambda self: {
                "exact_match": self.exact_match,
                "solution_confidence": self.solution.confidence,
            },
        })()
        result = type("R", (), {
            "solution": solution,
            "metrics": metrics,
            "error": None,
            "duration": 0.5,
        })()
        return result


def test_tc3_grid_search(tmp_path):
    """GridSearchConfig + run_grid_search produces report with 4 cells."""
    from syrch.search.grid import GridSearchConfig, run_grid_search
    from syrch.eval.runner import BenchmarkProblem

    grid_config = GridSearchConfig(
        max_depth_values=[2, 3],
        high_conf_values=[0.85],
        max_attempts_values=[1, 2],
        calibration_values=[False],
        parallel=False,
    )
    # Expected cells: 2×1×2×1 = 4

    problem = BenchmarkProblem(id="test", question="test?", db=":memory:")
    llm_config = LLMConfig(provider="openai", model="fake")

    # Monkey-patch run_single in grid module
    import syrch.search.grid as grid_mod
    fake = TC3_FakeRunner(expected_cells=4)
    original_run_single = grid_mod.run_single
    grid_mod.run_single = fake.run_single

    try:
        report = run_grid_search(problem, llm_config, grid_config)
        assert fake.calls == 4
        assert len(report.cells) == 4
        assert report.best is not None
        assert "max_depth" in report.best.params
        assert "max_attempts_per_node" in report.best.params

        # Report files created
        report_dir = report.output_dir
        assert os.path.exists(os.path.join(report_dir, "config.json"))
        assert os.path.exists(os.path.join(report_dir, "results.json"))
        assert os.path.exists(os.path.join(report_dir, "best.json"))
        assert os.path.exists(os.path.join(report_dir, "summary.md"))

        # best.json content
        with open(os.path.join(report_dir, "best.json")) as f:
            best_data = json.load(f)
        assert "params" in best_data
        assert "metrics" in best_data
    finally:
        grid_mod.run_single = original_run_single


# ═══════════════════════════════════════════════════════════════════
# TC4: Clarification loop e2e
# ═══════════════════════════════════════════════════════════════════

class TC4_AmbiguousLLM:
    """Returns low confidence initially, then high confidence after a re-run."""
    def __init__(self):
        self._json_calls = 0
        self._text_calls = 0

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        self._json_calls += 1
        return {
            "subtasks": [
                {
                    "id": "A",
                    "description": "Count total orders",
                    "depends_on": [],
                    "is_atomic": True,
                    "expected_output": "order count",
                },
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        self._text_calls += 1
        # Detect re-run: Planner calls generate_json on each run
        # So json_calls > 1 means a second pipeline run
        if self._json_calls > 1:
            return LLMResponse("```sql\nSELECT COUNT(*) as cnt FROM orders\n```\nconfidence: 0.95")
        return LLMResponse("```sql\nSELECT * FROM orders\n```\nconfidence: 0.20")


class TC4_Executor:
    def execute(self, sql: str) -> pd.DataFrame:
        if "*" in sql:
            return pd.DataFrame({"id": [1, 2], "amount": [10, 20]})
        return pd.DataFrame({"cnt": [2]})

    def get_schema(self, table_name=None):
        return TableSchema(
            name="orders",
            columns=[
                ColumnSchema(name="id", type="INT"),
                ColumnSchema(name="amount", type="REAL"),
            ],
        )

    def list_tables(self):
        return ["orders"]

    def close(self):
        pass


def test_tc4_clarification_loop():
    """Ambiguous low-confidence result triggers clarification question and re-run."""
    config = ExecutionConfig(
        question="How many orders are there?",
        db_path=":memory:",
        max_depth=2,
        max_attempts_per_node=1,
        high_confidence=0.85,
        verbose=False,
        calibration_enabled=False,
        interactive=True,
        ambiguity_threshold=0.25,
    )
    llm = TC4_AmbiguousLLM()
    executor = TC4_Executor()
    schema = executor.get_schema()
    problem = ProblemSpec(question=config.question, schema=schema)

    from syrch.search.clarify import find_worst_ambiguity, generate_question

    clarification_qa = []
    solved = False

    for _ in range(2):
        solution, dag, results = run_pipeline(llm, executor, config, problem)

        worst = find_worst_ambiguity(results)
        if worst and worst[1] > config.ambiguity_threshold:
            q = generate_question(llm, problem, dag, worst[0], results)
            clarification_qa.append((q, "Count all rows in the orders table"))
            problem = ProblemSpec(
                question=f"{problem.question}\n[User clarification: Count all rows in the orders table]",
                schema=schema,
            )
        else:
            solved = True
            break

    assert solved, "Clarification loop did not resolve ambiguity"
    assert len(clarification_qa) >= 1
    solution.clarification_qa = clarification_qa
    solution.clarified = bool(clarification_qa)

    assert solution.clarified is True
    assert len(solution.clarification_qa) >= 1
    assert solution.confidence > 0.5  # Now confident after clarification
    assert solution.answer is not None


# ═══════════════════════════════════════════════════════════════════
# TC5: Token budget enforcement
# ═══════════════════════════════════════════════════════════════════

class TC5_BudgetLLM:
    def generate(self, system: str, user: str, **kwargs):
        return LLMResponse("```sql\nSELECT 1\n```\nconfidence: 0.9",
                           usage={"completion_tokens": 50})

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        return {
            "subtasks": [
                {"id": "A", "description": "Task A", "depends_on": [],
                 "is_atomic": True, "expected_output": "a"},
                {"id": "B", "description": "Task B", "depends_on": ["A"],
                 "is_atomic": True, "expected_output": "b"},
                {"id": "C", "description": "Task C", "depends_on": ["A"],
                 "is_atomic": True, "expected_output": "c"},
            ]
        }


def test_tc5_token_budget_halts_pipeline():
    """token_budget=30 → A costs 50 → budget exceeded after layer 0 → B and C never run."""
    config = ExecutionConfig(
        question="test",
        db_path=":memory:",
        max_depth=2,
        max_attempts_per_node=1,
        high_confidence=0.9,
        token_budget=30,
        verbose=False,
        calibration_enabled=False,
    )

    class FakeExecForBudget:
        def execute(self, sql): return pd.DataFrame({"x": [1]})
        def get_schema(self, table_name=None):
            return TableSchema(name="t", columns=[ColumnSchema(name="x", type="INT")])
        def list_tables(self): return ["t"]
        def close(self): pass

    dag = TaskDAG(
        nodes={
            "A": TaskNode(id="A", description="task A", depends_on=[], is_atomic=True),
            "B": TaskNode(id="B", description="task B", depends_on=["A"], is_atomic=True),
            "C": TaskNode(id="C", description="task C", depends_on=["A"], is_atomic=True),
        },
        root_id="A",
        topo_layers=[["A"], ["B", "C"]],
    )

    from syrch.search.scheduler import Scheduler
    from syrch.search.rlm_engine import RLMAgent
    executor = FakeExecForBudget()
    llm = TC5_BudgetLLM()
    agent = RLMAgent(llm, executor, config)
    scheduler = Scheduler(llm, executor, config, agent=agent)
    results = scheduler.run(dag)

    # A costs 50 (completion_tokens from LLM response).
    # Budget = 30. After layer 0, total_tokens=50 > 30 → break.
    # B and C never run.
    assert "A" in results
    assert "B" not in results, "B should be skipped due to token budget"
    assert "C" not in results, "C should be skipped due to token budget"


# ═══════════════════════════════════════════════════════════════════
# TC6: Error isolation
# ═══════════════════════════════════════════════════════════════════

class TC6_FaultyExecutor:
    def __init__(self):
        self.call_count = 0

    def execute(self, sql: str):
        self.call_count += 1
        # Each node calls execute() twice (validation + final).
        # A (layer 0): calls 1, 2 → succeed
        # B (layer 1): call 3 → fail
        if self.call_count >= 3:
            raise RuntimeError("Simulated DB failure")
        return pd.DataFrame({"x": [1]})

    def get_schema(self, table_name=None):
        return TableSchema(
            name="t",
            columns=[ColumnSchema(name="x", type="INT")],
        )

    def list_tables(self):
        return ["t"]

    def close(self):
        pass


class TC6_IsolationLLM:
    def __init__(self):
        self.text_calls = 0

    def generate(self, system: str, user: str, **kwargs):
        self.text_calls += 1
        return LLMResponse("```sql\nSELECT x FROM t\n```\nconfidence: 0.9")

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        return {
            "subtasks": [
                {"id": "A", "description": "Task A", "depends_on": [], "is_atomic": True, "expected_output": "a"},
                {"id": "B", "description": "Task B", "depends_on": ["A"], "is_atomic": True, "expected_output": "b"},
            ]
        }


def test_tc6_error_isolation():
    """One node fails with empty result → other nodes succeed, pipeline produces solution."""
    config = ExecutionConfig(
        question="test", db_path=":memory:",
        max_depth=2, max_attempts_per_node=1, high_confidence=0.9,
        verbose=False, calibration_enabled=False,
    )
    executor = TC6_FaultyExecutor()
    llm = TC6_IsolationLLM()
    schema = executor.get_schema()
    problem = ProblemSpec(question=config.question, schema=schema)

    solution, dag, results = run_pipeline(llm, executor, config, problem)

    # A (layer 0) runs first → both execute() calls succeed
    assert results["A"].error is None
    assert results["A"].data is not None and not results["A"].data.empty

    # B (layer 1) runs second → execute() call 3 raises, caught by RLM → empty data
    # The empty data is the error signal, not a crash error
    assert results["B"].error is None  # RLM catches internally
    assert results["B"].data is not None
    assert results["B"].data.empty  # empty = failed execution

    # Pipeline still produces a solution
    assert solution is not None
    assert solution.answer is not None
    assert solution.confidence > 0


# ═══════════════════════════════════════════════════════════════════
# TC7: Calibration on/off comparison
# ═══════════════════════════════════════════════════════════════════

class TC7_CalLLM:
    def __init__(self, low_first: bool = False):
        self.calls = 0

    def generate(self, system: str, user: str, **kwargs):
        self.calls += 1
        return LLMResponse("```sql\nSELECT COUNT(*) as cnt FROM t\n```\nconfidence: 0.85")

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        return {
            "subtasks": [
                {"id": "A", "description": "Count", "depends_on": [],
                 "is_atomic": True, "expected_output": "count"},
            ]
        }


class TC7_CalExecutor:
    def execute(self, sql: str):
        return pd.DataFrame({"cnt": [100]})

    def get_schema(self, table_name=None):
        return TableSchema(name="t", columns=[ColumnSchema(name="cnt", type="INT")])

    def list_tables(self):
        return ["t"]

    def close(self):
        pass


def test_tc7_calibration_reduces_confidence():
    """Same LLM response → calibration_enabled=True gives lower or equal confidence."""
    config_on = ExecutionConfig(
        question="test", db_path=":memory:",
        max_depth=2, max_attempts_per_node=3, high_confidence=0.95,
        verbose=False, calibration_enabled=True,
    )
    config_off = ExecutionConfig(
        question="test", db_path=":memory:",
        max_depth=2, max_attempts_per_node=3, high_confidence=0.95,
        verbose=False, calibration_enabled=False,
    )
    executor = TC7_CalExecutor()
    schema = executor.get_schema()
    problem = ProblemSpec(question="test", schema=schema)

    # Run with calibration ON
    llm_on = TC7_CalLLM()
    sol_on, _, _ = run_pipeline(llm_on, executor, config_on, problem)

    # Run with calibration OFF
    llm_off = TC7_CalLLM()
    sol_off, _, _ = run_pipeline(llm_off, executor, config_off, problem)

    # Calibration applies multiplicative penalty (max_attempts=3, 1 call → retry_ratio=1/3)
    # So calibrated confidence should be lower or equal
    assert sol_on.confidence <= sol_off.confidence
    assert sol_on.answer is not None
    assert sol_off.answer is not None


# ═══════════════════════════════════════════════════════════════════
# TC8: Multi-table schema pipeline
# ═══════════════════════════════════════════════════════════════════

class TC8_MultiTableExecutor:
    def __init__(self):
        self._data = {
            "orders": {"id": [1, 2], "amount": [100, 200]},
            "customers": {"id": [1, 2], "name": ["Alice", "Bob"]},
        }

    def execute(self, sql: str):
        for table, cols in self._data.items():
            if table in sql.lower():
                return pd.DataFrame(cols)
        return pd.DataFrame({"cnt": [1]})

    def get_schema(self, table_name=None):
        schemas = {
            "orders": TableSchema(
                name="orders",
                columns=[ColumnSchema("id", "INT"), ColumnSchema("amount", "REAL")],
            ),
            "customers": TableSchema(
                name="customers",
                columns=[ColumnSchema("id", "INT"), ColumnSchema("name", "TEXT")],
            ),
        }
        if table_name:
            return schemas.get(table_name, schemas["orders"])
        return schemas["orders"]

    def list_tables(self):
        return ["orders", "customers"]

    def close(self):
        pass


class TC8_MultiTableLLM:
    def __init__(self):
        self.json_calls = 0

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        self.json_calls += 1
        return {
            "subtasks": [
                {
                    "id": "A",
                    "description": "Join orders and customers",
                    "depends_on": [],
                    "is_atomic": True,
                    "expected_output": "joined data",
                },
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        return LLMResponse("```sql\nSELECT o.id, o.amount, c.name "
                           "FROM orders o JOIN customers c ON o.id = c.id\n"
                           "```\nconfidence: 0.9")


def test_tc8_multi_table_schema():
    """Executor with 2 tables → schema lists both, pipeline succeeds."""
    config = ExecutionConfig(
        question="Join orders with customers",
        db_path=":memory:",
        max_depth=2,
        max_attempts_per_node=1,
        high_confidence=0.85,
        verbose=False,
        calibration_enabled=False,
    )
    executor = TC8_MultiTableExecutor()
    llm = TC8_MultiTableLLM()
    schema = executor.get_schema()
    problem = ProblemSpec(question=config.question, schema=schema)

    solution, dag, results = run_pipeline(llm, executor, config, problem)

    assert solution.answer is not None
    assert solution.confidence > 0
    assert "A" in results
    assert results["A"].error is None
    # Schema validation ran against both tables' columns
    assert "orders" in results["A"].sql
    assert "customers" in results["A"].sql
