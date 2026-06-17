"""Tests for the evaluation harness (eval/runner, eval/report, search/pipeline)."""

import json
import tempfile

import pandas as pd
import pytest

from syrch.core.config import ExecutionConfig
from syrch.core.models import ColumnSchema, FinalSolution, ProblemSpec, TableSchema
from syrch.eval.metrics import evaluate
from syrch.eval.runner import BenchmarkProblem, BenchmarkResult, load_benchmark
from syrch.search.pipeline import run_pipeline


class FakeLLM:
    def __init__(self):
        self.plan_calls = 0
        self.solve_calls = 0

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        self.plan_calls += 1
        return {
            "subtasks": [
                {
                    "id": "A",
                    "description": "Analyze the data",
                    "depends_on": [],
                    "is_atomic": True,
                    "expected_output": "statistics",
                }
            ]
        }

    def generate(self, system: str, user: str, **kwargs):
        self.solve_calls += 1
        return type(
            "Response",
            (),
            {
                "content": "```sql\nSELECT COUNT(*) as cnt FROM wikipedia_clickstream\n```\nconfidence: 0.95",
                "model": "fake",
                "usage": {"completion_tokens": 15},
            },
        )()


class FakeExecutor:
    def execute(self, sql: str) -> pd.DataFrame:
        return pd.DataFrame({"cnt": [3138]})

    def get_schema(self, table_name=None):
        return TableSchema(
            name="wikipedia_clickstream",
            columns=[ColumnSchema(name="cnt", type="INTEGER")],
        )

    def list_tables(self):
        return ["wikipedia_clickstream"]

    def close(self):
        pass


class FakeLLMResponse:
    def __init__(self, content, model="fake", usage=None):
        self.content = content
        self.model = model
        self.usage = usage or {"completion_tokens": 10}


def test_pipeline_runner():
    """run_pipeline returns FinalSolution with the fake pipeline."""
    llm = FakeLLM()
    executor = FakeExecutor()
    config = ExecutionConfig(
        question="test", db_path=":memory:",
        max_depth=2, max_attempts_per_node=1, high_confidence=0.85,
    )
    schema = executor.get_schema()
    problem = ProblemSpec(question="test", schema=schema)

    solution, dag, results = run_pipeline(llm, executor, config, problem)

    assert solution is not None
    assert solution.answer is not None
    assert solution.confidence > 0
    assert solution.token_cost > 0
    assert len(dag.nodes) >= 1
    assert len(results) >= 1


def test_evaluate_exact_match():
    """evaluate() detects exact match correctly."""
    data = pd.DataFrame({"x": [1, 2, 3]})
    solution = FinalSolution(question="test", answer="42", data=data, confidence=0.9, token_cost=100)
    metrics = evaluate(solution, expected_data=data)

    assert metrics.exact_match is True
    assert metrics.row_count_match is True
    assert metrics.column_match is True
    assert metrics.expected_rows == 3
    assert metrics.actual_rows == 3


def test_evaluate_mismatch():
    """evaluate() detects mismatches correctly."""
    actual = pd.DataFrame({"x": [1, 2]})
    expected = pd.DataFrame({"x": [1, 2, 3]})
    solution = FinalSolution(question="test", answer="42", data=actual, confidence=0.9, token_cost=100)
    metrics = evaluate(solution, expected_data=expected)

    assert metrics.exact_match is False
    assert metrics.row_count_match is False
    assert metrics.actual_rows == 2
    assert metrics.expected_rows == 3


def test_evaluate_no_expected():
    """evaluate() works without expected data."""
    solution = FinalSolution(question="test", answer="42", token_cost=50)
    metrics = evaluate(solution)

    assert metrics.token_cost == 50
    assert metrics.exact_match is False
    assert metrics.row_count_match is False


def test_evaluate_tracks_reasoning_paths():
    """evaluate() counts reasoning paths and sub-tasks."""
    from syrch.core.models import NodeResult, ReasoningPath

    solution = FinalSolution(
        question="test", answer="42", confidence=0.8, token_cost=100,
        tree=[
            NodeResult(
                node_id="A", data=pd.DataFrame(), sql="SELECT 1",
                confidence=0.8,
                reasoning_paths=[
                    ReasoningPath(path_id="A-0", sql="SELECT 1", confidence=0.7, cost_tokens=10),
                    ReasoningPath(path_id="A-1", sql="SELECT 2", confidence=0.8, cost_tokens=15),
                ],
                cost_tokens=25,
            ),
        ],
    )
    metrics = evaluate(solution)

    assert metrics.num_reasoning_paths == 2
    assert metrics.num_subtasks == 1


def test_benchmark_problem_dataclass():
    """BenchmarkProblem creates correctly from dict."""
    p = BenchmarkProblem(id="q1", question="test?", db="test.db", expected_data="expected.csv")
    assert p.id == "q1"
    assert p.question == "test?"
    assert p.db == "test.db"
    assert p.expected_data == "expected.csv"


def test_benchmark_result_dataclass():
    """BenchmarkResult stores error correctly."""
    r = BenchmarkResult(
        problem=BenchmarkProblem(id="q1", question="test?", db="test.db"),
        error="API error",
        duration=1.5,
    )
    assert r.error == "API error"
    assert r.duration == 1.5


def test_load_benchmark():
    """load_benchmark parses JSONL correctly."""
    lines = [
        json.dumps({"id": "q1", "question": "Q1?", "db": "d1.sqlite"}),
        json.dumps({"id": "q2", "question": "Q2?", "db": "d2.sqlite", "expected_data": "e2.csv"}),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines))
        f.flush()
        problems = load_benchmark(f.name)

    assert len(problems) == 2
    assert problems[0].id == "q1"
    assert problems[1].expected_data == "e2.csv"


def test_load_benchmark_skips_empty_lines():
    """load_benchmark skips blank lines."""
    lines = [
        json.dumps({"id": "q1", "question": "Q1?", "db": "d1.sqlite"}),
        "",
        json.dumps({"id": "q2", "question": "Q2?", "db": "d2.sqlite"}),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines))
        f.flush()
        problems = load_benchmark(f.name)

    assert len(problems) == 2


def test_pipeline_with_real_db():
    """run_pipeline executes against a real SQLite DB and returns metrics."""
    import os
    db_path = "wikipedia_clickstream.sqlite"
    if not os.path.exists(db_path):
        pytest.skip("wikipedia_clickstream.sqlite not found")

    from syrch.executors.sqlite_executor import SQLiteExecutor
    from syrch.core.models import ProblemSpec

    config = ExecutionConfig(
        question="test", db_path=db_path,
        max_depth=2, max_attempts_per_node=1, high_confidence=0.85,
    )

    llm = FakeLLM()
    executor = SQLiteExecutor(db_path)
    schema = executor.get_schema()
    problem = ProblemSpec(question="test", schema=schema)

    solution, _, _ = run_pipeline(llm, executor, config, problem)
    executor.close()

    assert solution is not None
    assert solution.confidence > 0


def test_metrics_to_dict():
    """EvaluationMetrics.to_dict() returns serializable dict."""
    data = pd.DataFrame({"x": [1]})
    solution = FinalSolution(question="t", answer="a", data=data, confidence=0.9, token_cost=50)
    metrics = evaluate(solution, expected_data=data)
    d = metrics.to_dict()

    assert d["exact_match"] is True
    assert d["token_cost"] == 50
    assert d["solution_confidence"] == 0.9
    assert isinstance(d["num_reasoning_paths"], int)


def test_benchmark_result_duration():
    """BenchmarkResult records positive duration."""
    r = BenchmarkResult(
        problem=BenchmarkProblem(id="q1", question="?", db="d.db"),
        duration=2.34,
    )
    assert r.duration > 0
    assert r.error is None
    assert r.solution is None


def test_aggregator_try_join_merge():
    from syrch.search.aggregator import Aggregator
    from syrch.core.models import JoinKey, NodeResult

    class FakeNullLLM:
        def generate(self, *a, **kw):
            return type("R", (), {"content": "ok", "model": "t", "usage": {"completion_tokens": 1}})()
        def generate_json(self, *a, **kw):
            return {}

    llm = FakeNullLLM()
    config = ExecutionConfig(question="test", db_path=":memory:")
    agg = Aggregator(llm, None, config)

    results = {
        "A": NodeResult(node_id="A", data=pd.DataFrame({"id": [1, 2, 3], "name": ["x", "y", "z"]}),
                        sql="SELECT * FROM a", confidence=0.9),
        "B": NodeResult(node_id="B", data=pd.DataFrame({"segment_id": [1, 2], "value": [100, 200]}),
                        sql="SELECT * FROM b", confidence=0.8),
    }
    joins = [
        JoinKey(left="B", left_col="segment_id", right="A", right_col="id"),
    ]

    merged = agg._try_join_merge(["A", "B"], results, joins)
    assert merged is not None
    assert list(merged.columns) == ["id", "name", "segment_id", "value"]
    assert len(merged) == 2


def test_aggregator_try_join_merge_3node_chain():
    from syrch.search.aggregator import Aggregator
    from syrch.core.models import JoinKey, NodeResult

    class FakeNullLLM:
        def generate(self, *a, **kw):
            return type("R", (), {"content": "ok", "model": "t", "usage": {"completion_tokens": 1}})()
        def generate_json(self, *a, **kw):
            return {}

    llm = FakeNullLLM()
    config = ExecutionConfig(question="test", db_path=":memory:")
    agg = Aggregator(llm, None, config)

    results = {
        "A": NodeResult(node_id="A", data=pd.DataFrame({"id": [1, 2, 3], "name": ["x", "y", "z"]}),
                        sql="", confidence=0.9),
        "B": NodeResult(node_id="B", data=pd.DataFrame({"seg_id": [1, 2], "val": [10, 20]}),
                        sql="", confidence=0.8),
        "C": NodeResult(node_id="C", data=pd.DataFrame({"ref": [1, 2, 3], "label": ["a", "b", "c"]}),
                        sql="", confidence=0.7),
    }
    joins = [
        JoinKey(left="B", left_col="seg_id", right="A", right_col="id"),
        JoinKey(left="C", left_col="ref", right="B", right_col="seg_id"),
    ]

    merged = agg._try_join_merge(["C"], results, joins)
    assert merged is not None
    assert "id" in merged.columns
    assert "val" in merged.columns
    assert "label" in merged.columns
    assert len(merged) == 2


def test_aggregator_try_join_merge_no_match_returns_none():
    from syrch.search.aggregator import Aggregator
    from syrch.core.models import NodeResult

    class FakeNullLLM:
        def generate(self, *a, **kw):
            return type("R", (), {"content": "ok", "model": "t", "usage": {"completion_tokens": 1}})()
        def generate_json(self, *a, **kw):
            return {}

    llm = FakeNullLLM()
    config = ExecutionConfig(question="test", db_path=":memory:")
    agg = Aggregator(llm, None, config)

    results = {
        "A": NodeResult(node_id="A", data=pd.DataFrame({"id": [1]}), sql="", confidence=0.9),
        "B": NodeResult(node_id="B", data=pd.DataFrame({"id": [1]}), sql="", confidence=0.8),
    }
    merged = agg._try_join_merge(["A", "B"], results, [])
    assert merged is None
