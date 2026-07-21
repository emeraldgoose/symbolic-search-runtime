from __future__ import annotations

from typing import Callable

from syrch.core.models import NodeResult, ProblemSpec, ScoredTable, TaskDAG
from syrch.llm.base import BaseLLM

CLARIFICATION_SYSTEM = """You identify ambiguity in SQL query generation.
Given a failed attempt, ask a single concise question to resolve the ambiguity."""

SEMANTIC_CLARIFY_SYSTEM = """You identify ambiguity in a user's natural language question for a Text-to-SQL system.
Given the question and the planner's decomposition plan, determine if the question is ambiguous.

Common ambiguities:
- Missing time range: "What is the churn rate?" needs a period (year, quarter, etc.)
- Ambiguous metric: "What is the revenue?" could be net, gross, total, or monthly
- Missing threshold: "Top customers" needs a count (top 10? top 100?)
- Vague comparison: "Best performing" needs a metric (by revenue? by growth?)
- Relative reference: "Last month" without a reference date
- Unclear granularity: "Sales by region" without specifying region level (country? city?)

If the question is sufficiently clear, respond with exactly: CLEAR
If ambiguous, ask ONE concise question to resolve the most critical ambiguity.

Question: {question}

Planner decomposition:
{dag_summary}

Relevant tables:
{table_summary}

Respond with CLEAR or a single question."""

CLARIFICATION_PROMPT = """Original question: {question}
Sub-task: {task}
Best SQL: {best_sql}
Best confidence: {best_confidence}

Symptoms:
{symptoms}

Ask ONE concise question to clarify what the user wants."""


def compute_ambiguity_score(result: NodeResult) -> float:
    confidences = [p.confidence for p in result.reasoning_paths]
    if not confidences:
        return 1.0

    best_conf = max(confidences)
    base_penalty = 1.0 - best_conf

    variance_penalty = 0.0
    if len(confidences) >= 2:
        mean = sum(confidences) / len(confidences)
        variance_penalty = min(
            sum((c - mean) ** 2 for c in confidences) / len(confidences),
            1.0,
        )

    error_penalty = 0.0
    if result.error is not None:
        error_penalty = 0.5

    empty_penalty = 0.3 if result.data is not None and result.data.empty and not result.error else 0.0

    score = (
        base_penalty * 0.35
        + variance_penalty * 0.15
        + error_penalty * 0.30
        + empty_penalty * 0.20
    )
    return min(score, 1.0)


def _format_symptoms(result: NodeResult) -> str:
    parts = []
    paths = result.reasoning_paths
    if not paths:
        return "No valid SQL was generated."

    confidences = [p.confidence for p in paths]
    parts.append(f"Confidences per attempt: {[round(c, 2) for c in confidences]}")
    parts.append(f"Best SQL attempt: {result.sql}")

    if result.error is not None:
        parts.append(f"Error: {result.error}")
    if result.data is not None and result.data.empty:
        parts.append("Result returned 0 rows (empty dataset)")

    return "\n".join(parts)


def generate_question(
    llm: BaseLLM,
    problem: ProblemSpec,
    dag: TaskDAG,
    worst_id: str,
    results: dict[str, NodeResult],
) -> str:
    node = dag.nodes[worst_id]
    result = results[worst_id]
    symptoms = _format_symptoms(result)

    prompt = CLARIFICATION_PROMPT.format(
        question=problem.question,
        task=node.description,
        best_sql=result.sql or "(none)",
        best_confidence=result.confidence,
        symptoms=symptoms,
    )
    response = llm.generate(CLARIFICATION_SYSTEM, prompt)
    return response.content.strip()


def find_worst_ambiguity(results: dict[str, NodeResult]) -> tuple[str, float] | None:
    worst_id: str | None = None
    worst_score = 0.0
    for nid, result in results.items():
        if result.ambiguity_score > worst_score:
            worst_score = result.ambiguity_score
            worst_id = nid
    if worst_id is None:
        return None
    return worst_id, worst_score


def _summarize_dag(dag: TaskDAG) -> str:
    lines: list[str] = []
    for layer in dag.topo_layers:
        for nid in layer:
            node = dag.nodes[nid]
            hints = ""
            if node.hint_tables:
                hints = f" [tables: {', '.join(node.hint_tables)}]"
            lines.append(f"  - {nid}: {node.description}{hints}")
    return "\n".join(lines)


def _summarize_tables(scored: list[ScoredTable]) -> str:
    lines: list[str] = []
    for s in scored[:8]:
        cols = ", ".join(c.name for c in s.schema.columns[:5])
        reasons = "; ".join(s.match_reasons[:2])
        lines.append(f"  {s.schema.name:25s} score={s.score:.2f}  [{reasons}]")
        lines.append(f"  {'':25s}  columns: {cols}")
    return "\n".join(lines)


class SemanticClarifier:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def check(
        self,
        question: str,
        dag: TaskDAG,
        scored_schemas: list[ScoredTable] | None = None,
    ) -> str | None:
        dag_summary = _summarize_dag(dag)
        table_summary = _summarize_tables(scored_schemas) if scored_schemas else "N/A"

        prompt = SEMANTIC_CLARIFY_SYSTEM.format(
            question=question,
            dag_summary=dag_summary,
            table_summary=table_summary,
        )
        response = self.llm.generate(
            "You identify ambiguity in user questions for SQL generation.",
            prompt,
        )
        result = response.content.strip().rstrip(".")
        if result.upper() == "CLEAR":
            return None
        return result


def run_clarification(
    llm: BaseLLM,
    question: str,
    dag: TaskDAG,
    scored_schemas: list[ScoredTable] | None,
    user_callback: Callable[[str], str] | None,
    max_rounds: int = 2,
) -> tuple[str, list[tuple[str, str]]]:
    """Run pre-execution clarification loop.
    
    Returns (amended_question, qa_pairs).
    """
    if user_callback is None:
        return question, []

    clarifier = SemanticClarifier(llm)
    qa_pairs: list[tuple[str, str]] = []
    current_q = question

    for _ in range(max_rounds):
        q = clarifier.check(current_q, dag, scored_schemas)
        if q is None:
            break
        answer = user_callback(q)
        qa_pairs.append((q, answer))
        current_q = f"{current_q}\n[User clarification: {answer}]"

    return current_q, qa_pairs
