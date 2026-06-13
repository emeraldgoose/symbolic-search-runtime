from __future__ import annotations

import pandas as pd

from syrch.core.models import NodeResult, ProblemSpec, ReasoningPath, TableSchema, ColumnSchema


def test_compute_ambiguity_score_low_variance_high_confidence():
    from syrch.search.clarify import compute_ambiguity_score

    result = NodeResult(
        node_id="A",
        data=pd.DataFrame({"x": [1]}),
        sql="SELECT 1",
        confidence=0.92,
        reasoning_paths=[
            ReasoningPath(path_id="A-0", sql="SELECT 1", confidence=0.92, cost_tokens=10),
        ],
    )
    score = compute_ambiguity_score(result)
    assert 0.0 <= score <= 0.1, f"Expected low score for high confidence, got {score}"


def test_compute_ambiguity_score_no_paths():
    from syrch.search.clarify import compute_ambiguity_score

    result = NodeResult(
        node_id="A",
        data=pd.DataFrame(),
        sql="",
        confidence=0.0,
        error="No valid SQL generated",
    )
    score = compute_ambiguity_score(result)
    assert score == 1.0, f"Expected 1.0 for no paths, got {score}"


def test_compute_ambiguity_score_low_confidence():
    from syrch.search.clarify import compute_ambiguity_score

    result = NodeResult(
        node_id="A",
        data=pd.DataFrame({"x": [1]}),
        sql="SELECT 1",
        confidence=0.3,
        reasoning_paths=[
            ReasoningPath(path_id="A-0", sql="SELECT 1", confidence=0.3, cost_tokens=10),
        ],
    )
    score = compute_ambiguity_score(result)
    assert score >= 0.2, f"Expected moderate score for low confidence, got {score}"


def test_compute_ambiguity_score_high_variance():
    from syrch.search.clarify import compute_ambiguity_score

    result = NodeResult(
        node_id="A",
        data=pd.DataFrame({"x": [1]}),
        sql="SELECT 1",
        confidence=0.9,
        reasoning_paths=[
            ReasoningPath(path_id="A-0", sql="SELECT 1", confidence=0.9, cost_tokens=10),
            ReasoningPath(path_id="A-1", sql="SELECT 2", confidence=0.1, cost_tokens=10),
        ],
    )
    score = compute_ambiguity_score(result)
    assert score > 0.03, f"Expected some ambiguity from variance, got {score}"


def test_compute_ambiguity_score_empty_result():
    from syrch.search.clarify import compute_ambiguity_score

    result = NodeResult(
        node_id="A",
        data=pd.DataFrame(),
        sql="SELECT 1 WHERE 1=0",
        confidence=0.5,
        reasoning_paths=[
            ReasoningPath(path_id="A-0", sql="SELECT 1", confidence=0.5, cost_tokens=10),
        ],
    )
    score = compute_ambiguity_score(result)
    assert score > 0, f"Expected penalty for empty result, got {score}"


def test_compute_ambiguity_score_with_error():
    from syrch.search.clarify import compute_ambiguity_score

    result = NodeResult(
        node_id="A",
        data=pd.DataFrame(),
        sql="",
        confidence=0.0,
        error="Execution failed: syntax error",
        reasoning_paths=[
            ReasoningPath(path_id="A-0", sql="BAD SQL", confidence=0.2, cost_tokens=10),
        ],
    )
    score = compute_ambiguity_score(result)
    assert score > 0.3, f"Expected high score for error, got {score}"


def test_find_worst_ambiguity():
    from syrch.search.clarify import find_worst_ambiguity

    results = {
        "A": NodeResult(
            node_id="A", data=pd.DataFrame(), sql="", confidence=0.9,
            reasoning_paths=[ReasoningPath(path_id="A-0", sql="OK", confidence=0.9, cost_tokens=5)],
            ambiguity_score=0.05,
        ),
        "B": NodeResult(
            node_id="B", data=pd.DataFrame(), sql="", confidence=0.0,
            error="Failed",
            reasoning_paths=[ReasoningPath(path_id="B-0", sql="BAD", confidence=0.0, cost_tokens=5)],
            ambiguity_score=0.85,
        ),
    }
    worst = find_worst_ambiguity(results)
    assert worst is not None
    wid, wscore = worst
    assert wid == "B"
    assert wscore == 0.85


def test_find_worst_ambiguity_all_low():
    from syrch.search.clarify import find_worst_ambiguity

    results = {
        "A": NodeResult(
            node_id="A", data=pd.DataFrame(), sql="", confidence=0.95,
            reasoning_paths=[ReasoningPath(path_id="A-0", sql="OK", confidence=0.95, cost_tokens=5)],
            ambiguity_score=0.02,
        ),
    }
    worst = find_worst_ambiguity(results)
    assert worst is not None
    assert worst[1] == 0.02


def test_generate_question():
    from syrch.search.clarify import generate_question

    class FakeClarifyLLM:
        def generate(self, system: str, user: str, **kwargs):
            from syrch.llm.base import LLMResponse
            return LLMResponse(
                content="Which table do you mean: orders or customers?",
                model="fake",
                usage={"completion_tokens": 10},
            )
        def generate_json(self, *a, **kw):
            return {}

    llm = FakeClarifyLLM()
    schema = TableSchema(name="t", columns=[ColumnSchema(name="x", type="INT")])
    problem = ProblemSpec(question="Test question?", schema=schema)

    from syrch.core.models import TaskDAG, TaskNode
    dag = TaskDAG(
        nodes={
            "A": TaskNode(id="A", description="Find revenue", depends_on=[]),
        },
        root_id="A",
    )
    results = {
        "A": NodeResult(
            node_id="A",
            data=pd.DataFrame(),
            sql="SELECT * FROM t",
            confidence=0.3,
            reasoning_paths=[
                ReasoningPath(path_id="A-0", sql="SELECT * FROM t", confidence=0.3, cost_tokens=10),
            ],
            ambiguity_score=0.7,
        ),
    }

    q = generate_question(llm, problem, dag, "A", results)
    assert len(q) > 0
    assert "orders" in q or "customers" in q or "table" in q


def test_pipeline_with_clarify_integration():
    """RLMAgent.solve() stores ambiguity_score on NodeResult."""
    from syrch.search.rlm_engine import RLMAgent
    from syrch.core.config import ExecutionConfig

    class FakeExec:
        def execute(self, sql):
            return pd.DataFrame({"cnt": [1]})
        def list_tables(self):
            return ["t"]
        def get_schema(self, table_name=None):
            return TableSchema(name="t", columns=[ColumnSchema(name="cnt", type="INT")])
        def close(self):
            pass

    class FakeLLM:
        def generate(self, system, user, **kwargs):
            from syrch.llm.base import LLMResponse
            return LLMResponse(
                content="```sql\nSELECT cnt FROM t\n```\nconfidence: 0.95",
                model="fake",
                usage={"completion_tokens": 20},
            )
        def generate_json(self, *a, **kw):
            return {}

    config = ExecutionConfig(question="test", db_path=":memory:", max_attempts_per_node=2)
    agent = RLMAgent(FakeLLM(), FakeExec(), config)

    from syrch.core.models import TaskNode
    node = TaskNode(id="A", description="Count rows", depends_on=[], is_atomic=True)
    result = agent.solve(node)
    assert hasattr(result, "ambiguity_score")
    assert isinstance(result.ambiguity_score, float)
    assert 0.0 <= result.ambiguity_score <= 1.0
