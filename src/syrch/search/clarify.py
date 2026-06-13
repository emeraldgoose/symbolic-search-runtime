from __future__ import annotations

from syrch.core.models import NodeResult, ProblemSpec, TaskDAG
from syrch.llm.base import BaseLLM

CLARIFICATION_SYSTEM = """You identify ambiguity in SQL query generation.
Given a failed attempt, ask a single concise question to resolve the ambiguity."""

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
