from __future__ import annotations

import logging
from typing import Callable

from syrch.core.config import ExecutionConfig
from syrch.core.models import FinalSolution, NodeResult, ProblemSpec, TableSchema, TaskDAG
from syrch.executors.base import BaseExecutor
from syrch.llm.base import BaseLLM
from syrch.search.aggregator import Aggregator
from syrch.search.planner import Planner
from syrch.search.question_norm import normalize_question
from syrch.search.retriever import Retriever, ScoredSchemaEvidence
from syrch.search.scheduler import Scheduler
from syrch.search.semantic_index import SemanticIndex
from syrch.search.data_probe import ProbeRegistry

logger = logging.getLogger(__name__)


def compress_node_schemas(
    dag: TaskDAG,
    all_schemas: list[TableSchema],
) -> None:
    for node in dag.nodes.values():
        if node.hint_tables:
            compressed = [s for s in all_schemas if s.name in node.hint_tables]
            node._compressed_schemas = compressed if compressed else None
        else:
            node._compressed_schemas = None


def _normalized_confidence(evidence: ScoredSchemaEvidence) -> float:
    candidates = evidence.candidates
    if not candidates:
        return 0.0
    top_k = candidates[:10]
    total = sum(c.score for c in top_k) or 1
    return candidates[0].score / total


def _terminal_ambiguity(evidence: ScoredSchemaEvidence, min_tables: int = 5) -> bool:
    candidates = evidence.candidates
    if not candidates:
        return bool(evidence.all_schemas) and len(evidence.all_schemas) >= min_tables
    norm_conf = _normalized_confidence(evidence)
    has_matched_cols = bool(evidence.matched_columns)
    total_tables = len(evidence.all_schemas) if evidence.all_schemas else len(candidates)
    if total_tables < min_tables:
        return False
    if has_matched_cols:
        return False
    if norm_conf >= 0.35:
        return False
    return True


def _planner_ambiguity(dag: TaskDAG) -> bool:
    root = dag.nodes.get(dag.root_id)
    if root is None:
        return True
    hint_tables = root.hint_tables or []
    if len(hint_tables) < 1:
        return True
    if root.selection_reason and root.selection_reason.used_features:
        return False
    return True


def run_pipeline(
    llm: BaseLLM,
    executor: BaseExecutor,
    config: ExecutionConfig,
    problem: ProblemSpec,
    user_callback: Callable[[str], str] | None = None,
) -> tuple[FinalSolution, TaskDAG, dict[str, NodeResult]]:
    if problem.all_schemas is None:
        problem.all_schemas = [executor.get_schema(t) for t in executor.list_tables()]

    semantic_index = SemanticIndex.load()
    retriever = Retriever(problem.all_schemas, semantic_index=semantic_index if not semantic_index.is_empty else None)

    if problem.scored_schemas is None or problem.evidence is None:
        search_question = normalize_question(llm, problem.question)
        evidence = retriever.score(search_question)
        problem.scored_schemas = evidence.candidates
        problem.evidence = evidence
    else:
        evidence = problem.evidence

    # Phase 1.4: log retriever info
    if evidence.candidates:
        gt_table = _guess_gt_table(problem.question)
        for rank, c in enumerate(evidence.candidates[:20]):
            is_gt = gt_table and gt_table in c.schema.name
            logger.info(
                "retriever rank=%-2d score=%-6.2f table=%-25s features=%s%s",
                rank + 1, c.score, c.schema.name,
                dict(matched_keywords=c.features.matched_keywords[:3],
                     matched_columns=bool(c.features.matched_columns),
                     layer=c.features.layer),
                " [GT]" if is_gt else "",
            )

    # Stage 1 Gate: terminal ambiguity -> early exit
    if _terminal_ambiguity(evidence):
        msg = "Terminal ambiguity: retriever found no grounded columns"
        logger.warning(msg)
        if config.interactive and user_callback:
            from syrch.search.clarify import SEMANTIC_CLARIFY_SYSTEM
            table_summary = "\n".join(
                f"  {c.schema.name:25s} score={c.score:.2f}" for c in evidence.candidates[:8]
            )
            prompt = SEMANTIC_CLARIFY_SYSTEM.format(
                question=problem.question,
                dag_summary="(no plan yet)",
                table_summary=table_summary,
            )
            resp = llm.generate(
                "You identify ambiguity in user questions for SQL generation.",
                prompt,
            )
            q = resp.content.strip()
            if q.upper() != "CLEAR" and user_callback is not None:
                answer = user_callback(q)
                problem = ProblemSpec(
                    question=f"{problem.question}\n[User clarification: {answer}]",
                    schema=problem.schema,
                    all_schemas=problem.all_schemas,
                    scored_schemas=problem.scored_schemas,
                    evidence=problem.evidence,
                )
                evidence = retriever.score(normalize_question(llm, problem.question))
                problem.scored_schemas = evidence.candidates
                problem.evidence = evidence
        else:
            solution = FinalSolution(
                question=problem.question,
                answer=f"I could not find the right tables for: {problem.question}",
                data=None,
                sql="",
                confidence=0.3,
                token_cost=0,
                tree=[],
                clarified=False,
                clarification_qa=[],
            )
            dag = TaskDAG(nodes={}, root_id="")
            return solution, dag, {}

    planner = Planner(llm, config, retriever=retriever)

    # Phase 4: Adaptive K
    norm_conf = _normalized_confidence(evidence)
    if norm_conf >= 0.7:
        retriever.policy.max_candidates = 5
    elif norm_conf >= 0.35:
        retriever.policy.max_candidates = 10
    else:
        retriever.policy.max_candidates = 15

    dag = planner.decompose(problem)

    # Stage 2 Gate: planner ambiguity -> clarification
    amended_question = problem.question
    qa_pairs: list[tuple[str, str]] = []
    if _planner_ambiguity(dag):
        logger.warning("Planner ambiguity: weak table selection")
        if config.interactive and user_callback:
            from syrch.search.clarify import SEMANTIC_CLARIFY_SYSTEM
            table_summary = "\n".join(
                f"  {c.schema.name:25s} score={c.score:.2f}" for c in evidence.candidates[:8]
            )
            prompt = SEMANTIC_CLARIFY_SYSTEM.format(
                question=problem.question,
                dag_summary=_summarize_dag_simple(dag),
                table_summary=table_summary,
            )
            resp = llm.generate(
                "You identify ambiguity in user questions for SQL generation.",
                prompt,
            )
            q = resp.content.strip()
            if q.upper() != "CLEAR" and user_callback is not None:
                answer = user_callback(q)
                qa_pairs.append((q, answer))
                amended_question = f"{problem.question}\n[User clarification: {answer}]"
                problem = ProblemSpec(
                    question=amended_question,
                    schema=problem.schema,
                    all_schemas=problem.all_schemas or [problem.schema],
                    scored_schemas=problem.scored_schemas,
                    evidence=problem.evidence,
                )
                dag = planner.decompose(problem)
        else:
            pass

    all_schemas_list: list[TableSchema] = problem.all_schemas or [problem.schema]
    problem = ProblemSpec(
        question=amended_question,
        schema=problem.schema,
        all_schemas=all_schemas_list,
        scored_schemas=problem.scored_schemas,
    )

    compress_node_schemas(dag, all_schemas_list)

    probe_registry = ProbeRegistry()

    def _on_replan(current_dag: TaskDAG, failed_node_id: str, node_result: NodeResult) -> TaskDAG:
        new_dag = planner.replan(
            dag=current_dag,
            failed_node_id=failed_node_id,
            sql=node_result.sql,
            error=node_result.error or "",
            node_result=node_result,
            scored_schemas=evidence.candidates if evidence.candidates else [],
            probe_registry=probe_registry,
        )
        compress_node_schemas(new_dag, all_schemas_list)
        return new_dag

    scheduler = Scheduler(
        llm, executor, config,
        compressed_schemas=None,
        replan_callback=_on_replan,
        retriever=retriever,
        all_schemas=all_schemas_list,
        alias_map=evidence.alias_map if problem.evidence else None,
        candidate_pool=evidence.candidates if evidence.candidates else None,
        probe_registry=probe_registry,
    )
    results = scheduler.run(dag)
    aggregator = Aggregator(llm, executor, config)
    solution = aggregator.merge(problem.question, dag, results)
    solution.clarified = bool(qa_pairs)
    solution.clarification_qa = qa_pairs
    return solution, dag, results


def _guess_gt_table(question: str) -> str | None:
    known: dict[str, str] = {
        "total revenue": "dw_sales_order",
        "net revenue": "dw_sales_order",
        "vip customers": "dw_customer",
        "cart": "fact_event_log",
        "conversion": "fact_event_log",
        "marketing": "dw_marketing",
        "gmv": "dw_marketing",
        "abandonment": "fact_event_log",
        "monthly active": "fact_event_log",
        "mau": "fact_event_log",
        "churn": "dw_customer",
        "customer": "dw_customer",
        "payment": "dw_sales_order",
        "delivery": "dw_sales_order",
        "fulfillment": "dw_sales_order",
    }
    ql = question.lower()
    for pattern, table in known.items():
        if pattern in ql:
            return table
    return None


def _summarize_dag_simple(dag: TaskDAG) -> str:
    lines: list[str] = []
    for nid, node in dag.nodes.items():
        hints = ""
        if node.hint_tables:
            hints = f" [tables: {', '.join(node.hint_tables)}]"
        lines.append(f"  - {nid}: {node.description}{hints}")
    return "\n".join(lines)
