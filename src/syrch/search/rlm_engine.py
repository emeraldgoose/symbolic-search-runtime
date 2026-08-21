from __future__ import annotations

import logging

import pandas as pd

import re

from typing import Any

from syrch.core.config import ExecutionConfig
from syrch.core.models import (
    AttemptSchemaContext,
    CandidateEvaluation,
    NodeResult,
    NodeStatus,
    ParentContext,
    ReasoningPath,
    ReplanType,
    ScoredTable,
    TableSchema,
    TaskNode,
    base_table_name,
)
from syrch.executors.base import BaseExecutor
from syrch.llm.base import BaseLLM
from syrch.search.calibrator import ExecutionSignals
from syrch.search.clarify import compute_ambiguity_score
from syrch.search.data_probe import DataProbe, ProbeRegistry, ProbeResult
from syrch.search.path_evaluator import PathEvaluator
from syrch.search.retriever import Retriever
from syrch.search.search_policy import build_policy
from syrch.search.validator import Validator

logger = logging.getLogger(__name__)

RLM_SYSTEM = """You are a constraint-based SQL generator.

CRITICAL RULES — Follow ALL:

1. USE ONLY COLUMNS SHOWN IN THE SCHEMA BELOW. Never invent column names.
    Map business terms to physical column names. Never use a business term as a column name directly.

2. Aggregate/metric questions MUST include GROUP BY. If the question asks for
   total, average, sum, count, per, by, or a metric column, use aggregation
   functions + GROUP BY. Never return raw rows for an aggregate question.

3. Time-filtered questions MUST include WHERE on a date/time column. If the
   question mentions a year, month, quarter, date range, trend, or "recent",
   add a date filter condition.

4. SCD2 tables (having valid_from, valid_to columns): always filter by
   valid_from <= reference_date AND (valid_to > reference_date OR valid_to IS NULL).

5. Prefer hint_columns and metric_columns over other columns. These are the
   columns most likely to answer the question correctly.

6. Column aliases must be business-domain terms, not question phrases.

7. Never embed dates, years, or time ranges in column aliases.

8. Do not invent column names. Every column name you SELECT must exist
   in the schema. If you need a derived value, give it a short domain alias.

9. TASK CONTEXT tables (`_task_context_*`) are real materialized tables
   produced by earlier tasks in this run. You MAY use them in FROM or JOIN,
   exactly like physical tables. They are listed under "Task context tables"
   with their columns. Use them as intermediate results when joining
   dependent steps.

10. FROM must anchor on a PRIMARY table or a TASK CONTEXT table. Never anchor
    FROM on a JOIN-AVAILABLE table — that switches the primary source and the
    attempt is testing a different candidate. JOIN-AVAILABLE tables may be
    used ONLY in JOIN clauses.

11. If the question describes a business state or reason that has no dedicated
    column, look for an existing categorical column whose values encode that
    state and filter on those values. Never invent a column name for a state
    that is not explicitly present in the schema — approximate it with the
    values that already exist.

{hint_section}
Available tables:
{schema}

Task: {task_description}

Output your SQL query, then end with "Confidence: <0.0-1.0>" on its own line.
When satisfied, output FINAL(result_var_name)."""


class RLMAgent:
    MAX_ROWS_WARNING = 1000

    def __init__(
        self,
        llm: BaseLLM,
        executor: BaseExecutor,
        config: ExecutionConfig,
        retriever: Retriever | None = None,
        all_schemas: list | None = None,
        alias_map: dict[str, list[tuple[str, str, str | None]]] | None = None,
        candidate_pool: list[ScoredTable] | None = None,
        probe_registry: ProbeRegistry | None = None,
    ):
        self.llm = llm
        self.executor = executor
        self.config = config
        self.retriever = retriever
        self.all_schemas = all_schemas
        self.alias_map = alias_map or {}
        self._compressed_schemas: list | None = None
        self._candidate_pool: list[ScoredTable] = candidate_pool or []
        self._allowed_tables: set[str] | None = None
        self._attempt_scope: AttemptSchemaContext | None = None
        self.probe = DataProbe(executor, probe_registry)

    def set_compressed_schemas(self, schemas: list | None) -> None:
        self._compressed_schemas = schemas

    def set_candidate_pool(self, pool: list[ScoredTable]) -> None:
        self._candidate_pool = pool

    def _build_schema_str(self) -> str:
        if self._attempt_scope is not None:
            return self._render_scope(self._attempt_scope)
        if self._compressed_schemas is not None:
            return self._format_schemas(list(self._compressed_schemas))
        tables = self.executor.list_tables()
        parts = []
        for t in tables:
            schema = self.executor.get_schema(t)
            parts.append(self._format_table(schema))
        return "\n\n".join(parts)

    @staticmethod
    def _format_table(s: TableSchema) -> str:
        col_parts = []
        for c in s.columns:
            desc = f" ({c.description})" if c.description else ""
            col_parts.append(f"{c.name} ({c.type}){desc}")
        return f"Table: {s.name}\nColumns: {', '.join(col_parts)}"

    @staticmethod
    def _format_schemas(schemas: list[TableSchema]) -> str:
        return "\n\n".join(RLMAgent._format_table(s) for s in schemas)

    def _resolve_schemas(self, names: set[str]) -> list[TableSchema]:
        by_name: dict[str, TableSchema] = {}
        if self.all_schemas:
            by_name = {s.name: s for s in self.all_schemas}
        else:
            for t in self.executor.list_tables():
                by_name[t] = self.executor.get_schema(t)
        return [by_name[n] for n in sorted(names) if n in by_name]

    def _render_scope(self, scope: AttemptSchemaContext) -> str:
        parts: list[str] = []
        primary = self._resolve_schemas(scope.primary_tables)
        if primary:
            parts.append(
                "PRIMARY TABLES\n"
                "FROM must reference exactly these tables (the candidate under test). "
                "Never anchor FROM on a JOIN-AVAILABLE table."
            )
            parts.append(self._format_schemas(primary))
        join = self._resolve_schemas(scope.join_available_tables)
        if join:
            parts.append(
                "JOIN-AVAILABLE TABLES\n"
                "These tables are available only in JOIN clauses. They may never "
                "appear in FROM (that would switch the primary source)."
            )
            parts.append(self._format_schemas(join))
        contexts = [c for c in scope.task_contexts if c.materialized]
        if contexts:
            ctx_parts: list[str] = []
            for c in contexts:
                cols = ", ".join(f"{col.name} ({col.type})" for col in c.columns)
                ctx_parts.append(f"{c.table_name}: {cols}")
            parts.append(
                "TASK CONTEXT TABLES (materialized)\n"
                "These are real tables produced by earlier tasks. You MAY use them "
                "in FROM or JOIN."
            )
            parts.append("\n".join(ctx_parts))
        return "\n\n".join(parts) if parts else "No tables available."

    def _build_valid_columns(self) -> set[str]:
        cols: set[str] = set()
        if self.all_schemas:
            for s in self.all_schemas:
                cols.update(c.name.lower() for c in s.columns)
        else:
            for t in self.executor.list_tables():
                schema = self.executor.get_schema(t)
                cols.update(c.name.lower() for c in schema.columns)
        scope = self._attempt_scope
        if scope is not None:
            for c in scope.task_contexts:
                cols.update(col.name.lower() for col in c.columns)
        cols.add("*")
        return cols

    def _build_hint_section(self, node: TaskNode) -> str:
        shown = set()
        for s in self._compressed_schemas or []:
            shown.add(s.name)
        parts: list[str] = []
        if node.hint_tables:
            hinted = [t for t in node.hint_tables if t in shown] or []
            if hinted:
                score_info = ""
                if node.selection_reason and node.selection_reason.score > 0:
                    score_info = f" (confidence: {node.selection_reason.score:.2f})"
                parts.append(f"Recommended tables: {', '.join(hinted)}{score_info}")
        if node.hint_columns:
            parts.append("Prefer these columns: " + ", ".join(node.hint_columns))
        if node.metric_columns:
            parts.append("Metric columns (use these for aggregation): " + ", ".join(node.metric_columns))
        if node.grain:
            parts.append("Row granularity: " + node.grain)
        if node.time_columns:
            parts.append("Time columns for date filtering: " + ", ".join(node.time_columns))
        if node.requirements and node.requirements.render():
            parts.append("REQUIREMENTS (what this task must compute):" + node.requirements.render())
        if self.alias_map:
            alias_lines: list[str] = []
            for term, mappings in sorted(self.alias_map.items()):
                cols_str = "; ".join(
                    f"{agg}({col}) in {tbl}" if agg else f"{col} in {tbl}"
                    for col, tbl, agg in mappings
                )
                alias_lines.append(f"  {term} → {cols_str}")
            parts.append("ALIAS MAP — Business terms → physical columns:\n" + "\n".join(alias_lines))
        if parts:
            return "\n".join(parts) + "\n"
        return ""

    def solve(
        self,
        node: TaskNode,
        context: dict[str, ParentContext] | None = None,
    ) -> NodeResult:
        infeasible = self._requirement_infeasible(node)
        if infeasible is not None:
            if self.config.verbose:
                logger.warning("  [%s] requirement infeasible (probe): %s", node.id, infeasible)
            result = NodeResult(
                node_id=node.id,
                data=pd.DataFrame(),
                sql="",
                confidence=0.0,
                status=NodeStatus.FAILED,
                error=infeasible,
                cost_tokens=0,
            )
            result.replan_request = (ReplanType.STRUCTURAL, infeasible)
            return result

        order = self._build_candidate_order(node)
        policy = build_policy(
            self.config.search_policy,
            order,
            beam_width=self.config.beam_width,
            max_candidates=self.config.candidate_budget,
            stop_margin=self.config.stop_margin,
        )
        evaluator = PathEvaluator()

        evaluations: dict[str, CandidateEvaluation] = {}
        result_by_table: dict[str, NodeResult] = {}
        all_paths: list[ReasoningPath] = []
        total_cost = 0
        expansions = 0

        while True:
            if self.config.verbose:
                viable_now = [e for e in evaluations.values() if e.viable]
                logger.info(
                    "  [%s] policy index=%d/%d viable=%d eval=%d remaining=%d",
                    node.id, policy._index, len(policy._candidates),
                    len(viable_now), len(evaluations), len(policy.remaining()),
                )
            if not policy.has_next():
                viable = [e for e in evaluations.values() if e.viable]
                if (
                    self._is_ambiguous(viable)
                    and expansions < self.config.max_candidate_expansion
                    and policy.remaining()
                ):
                    policy.expand(1)
                    expansions += 1
                    if self.config.verbose:
                        logger.info(
                            "  [%s] ambiguous; expanding search (%d/%d)",
                            node.id, expansions, self.config.max_candidate_expansion,
                        )
                    continue
                if self.config.verbose:
                    logger.info(
                        "  [%s] policy exhausted index=%d/%d viable=%d",
                        node.id, policy._index, len(policy._candidates), len(viable),
                    )
                break

            cand = policy.next()
            scope = self._build_attempt_schemas(node, cand, context)
            self._attempt_scope = scope
            self._allowed_tables = scope.allowed_tables
            self._compressed_schemas = self._resolve_schemas(scope.allowed_tables)
            if self.config.verbose:
                logger.info(
                    "  [%s] attempt scope: candidate=%s PRIMARY={%s} JOIN_AVAILABLE={%s} "
                    "TASK_CONTEXT={%s} ALLOWED_FROM={%s}",
                    node.id, cand.schema.name,
                    ", ".join(sorted(scope.primary_tables)) or "-",
                    ", ".join(sorted(scope.join_available_tables)) or "-",
                    ", ".join(t.table_name for t in scope.task_contexts) or "-",
                    ", ".join(sorted(scope.allowed_tables)) or "-",
                )
            result, ok = self._attempt(node, cand.score, context)
            total_cost += result.cost_tokens
            all_paths.extend(result.reasoning_paths)

            ev = self._build_candidate_evaluation(node, cand, result, ok, evaluator)
            evaluations[cand.schema.name] = ev
            policy.update(ev)
            if ok:
                result_by_table[cand.schema.name] = result
            if self.config.verbose:
                logger.info(
                    "  [%s] eval %s ok=%s viable=%s ps=%s rq=%.3f sm=%.3f gm=%.3f dm=%.3f tm=%.3f",
                    node.id, cand.schema.name, ok, ev.viable,
                    result.path_score.total if result.path_score else "None",
                    ev.result_quality, ev.structural_match, ev.grain_match,
                    ev.dimension_match, ev.time_match,
                )

        viable = [e for e in evaluations.values() if e.viable]

        if not viable:
            result = NodeResult(
                node_id=node.id,
                data=pd.DataFrame(),
                sql="",
                confidence=0.0,
                status=NodeStatus.FAILED,
                candidates=list(evaluations.values()),
                reasoning_paths=all_paths,
                cost_tokens=total_cost,
                error="No valid SQL generated",
            )
            result.ambiguity_score = 1.0
            if len(order) > 1:
                tried = ", ".join(c.schema.name for c in order)
                result.replan_request = (
                    ReplanType.STRUCTURAL,
                    f"Local search exhausted {len(order)} candidates [{tried}]",
                )
            return result

        ranked = sorted(viable, key=self._rank_key, reverse=True)
        top = ranked[0]
        best_result = result_by_table[top.table]

        if len(ranked) >= 2 and top.ranking_signals == ranked[1].ranking_signals:
            best_result.status = NodeStatus.AMBIGUOUS
            best_result.selected_candidate = None
            best_result.ambiguity_score = 1.0
            best_result.replan_request = (
                ReplanType.STRUCTURAL,
                f"Ambiguous: {top.table} == {ranked[1].table}",
            )
        else:
            best_result.status = NodeStatus.SOLVED
            best_result.selected_candidate = top

        best_result.candidates = list(evaluations.values())
        best_result.cost_tokens = total_cost
        best_result.reasoning_paths = all_paths
        return best_result

    @staticmethod
    def _rank_key(e: CandidateEvaluation):
        """Deterministic lexicographic selection key (candidate_id last).

        Discrimination only — lexical relevance (`semantic_match`) and token
        cost never select a candidate over a genuinely indistinguishable one.
        """
        return (
            e.structural_match,
            e.grain_match,
            e.dimension_match,
            e.time_match,
            e.result_quality,
            e.candidate_id,
        )

    def _requirement_infeasible(self, node: TaskNode) -> str | None:
        """Node-start feasibility gate driven by shared probe facts.

        The planner picks supporting relations from schema alone, so it can
        demand a relation whose filter value the probe already verified does
        not exist (e.g. rpt_customer_ltv demanded 'to filter for VIP
        customers' while the registry holds 'VIP NOT FOUND in
        rpt_customer_ltv.segment'). Such a requirement is unsatisfiable by
        construction — no amount of SQL retrying can satisfy it — so surface
        a STRUCTURAL replan instead of burning attempts on it.
        """
        req = node.requirements
        if req is None or not req.supporting_relations:
            return None
        facts = self.probe.registry.all()
        if not facts:
            return None
        for sr in req.supporting_relations:
            base = sr.table.split(".")[-1].lower()
            purpose = (sr.purpose or "").lower()
            if not purpose:
                continue
            for fact in facts:
                if fact.exists:
                    continue
                if fact.table.split(".")[-1].lower() != base:
                    continue
                if fact.value.lower() in purpose:
                    return (
                        f"Supporting relation {sr.table} (purpose: {sr.purpose}) "
                        f"is unsatisfiable: probe verified {fact.render()}"
                    )
        return None

    @staticmethod
    def _is_ambiguous(viable: list[CandidateEvaluation]) -> bool:
        """True when the top two viable candidates are indistinguishable on the
        ranking signals (execution order never breaks the tie)."""
        if len(viable) < 2:
            return False
        ranked = sorted(viable, key=RLMAgent._rank_key, reverse=True)
        return ranked[0].ranking_signals == ranked[1].ranking_signals

    def _build_candidate_evaluation(
        self,
        node: TaskNode,
        cand: ScoredTable,
        result: NodeResult,
        ok: bool,
        evaluator: PathEvaluator,
    ) -> CandidateEvaluation:
        ps = result.path_score
        return CandidateEvaluation(
            table=cand.schema.name,
            ok=ok,
            execution_valid=ok and result.error is None,
            requirement_pass=bool(
                result.validation is not None and result.validation.passed
            ),
            semantic_match=evaluator.semantic_match(node, cand.schema, result.data),
            result_quality=ps.execution_signal if ps else 0.0,
            structural_match=evaluator.structural_match(
                result.validation, node, result.data
            ),
            grain_match=evaluator.grain_match(node, cand.schema),
            dimension_match=evaluator.dimension_match(node, cand.schema),
            time_match=evaluator.time_match(node, cand.schema),
            cost_tokens=result.cost_tokens,
            candidate_id=cand.schema.name,
            confidence=result.confidence,
            error=result.error,
            attempts=len(result.reasoning_paths),
            has_data=result.data is not None and not result.data.empty,
            path_score=ps,
        )

    def _build_candidate_order(self, node: TaskNode) -> list[ScoredTable]:
        pool_by_name: dict[str, ScoredTable] = {
            c.schema.name: c for c in self._candidate_pool
        }
        order: list[ScoredTable] = []
        seen: set[str] = set()

        hint_names = node.hint_tables or []
        hint_cands: list[ScoredTable] = []
        for name in hint_names:
            cand = pool_by_name.get(name)
            if cand is not None:
                hint_cands.append(cand)
        hint_cands.sort(key=lambda c: -c.score)
        for cand in hint_cands:
            if cand.schema.name not in seen:
                seen.add(cand.schema.name)
                order.append(cand)

        pool_cands = [
            c for c in self._candidate_pool
            if c.score > 0 and c.schema.name not in seen
        ]
        pool_cands.sort(key=lambda c: -c.score)
        order.extend(pool_cands)

        if not order:
            schemas = (self._compressed_schemas or []) or (self.all_schemas or [])
            for s in schemas:
                if s.name not in seen:
                    seen.add(s.name)
                    order.append(ScoredTable(schema=s, score=0.0))

        if not order:
            for t in self.executor.list_tables():
                schema = self.executor.get_schema(t)
                if schema.name not in seen:
                    seen.add(schema.name)
                    order.append(ScoredTable(schema=schema, score=0.0))

        if self.config.verbose:
            names = ", ".join(f"{c.schema.name}:{c.score:.2f}" for c in order)
            logger.info(
                "  [%s] candidate_order (%d) hint=%s pool=%d -> %s",
                node.id, len(order), hint_names,
                len(self._candidate_pool), names or "-",
            )
        return order

    def _search_scope(self) -> set[str]:
        """The retriever candidate pool — the bounded source for join-available
        tables. This is NOT the FROM boundary: the per-attempt boundary is
        `AttemptSchemaContext.allowed_tables` (S3), computed in
        `_build_attempt_schemas`. Drift to a pool-external physical table is
        still rejected by `_validate_schema` (S5)."""
        if self._candidate_pool:
            return {c.schema.name for c in self._candidate_pool}
        if self.all_schemas:
            return {s.name for s in self.all_schemas}
        return set()

    def _build_attempt_schemas(
        self,
        node: TaskNode,
        cand: ScoredTable,
        context: dict[str, ParentContext] | None = None,
    ) -> AttemptSchemaContext:
        """Build the explicit per-attempt SQL scope (S3).

        PRIMARY = the candidate under test. JOIN-AVAILABLE = pool-bounded
        supporting relations: tables sharing a join key (column name) with the
        primary, tables the planner hinted for this node, requirement
        supporting_relations, and the physical source tables of any dependency
        context. Every join candidate must be in the candidate pool — the pool
        is the only source, so an out-of-pool table stays unreferenceable (S5
        drift guard) and the v0.3.3 schema-dump problem does not resurface.
        TASK CONTEXT entries are carried as materialized table references
        (v0.3.5b); materialized ones may appear in FROM/JOIN.
        """
        pool_names = self._search_scope()
        primary: set[str] = {cand.schema.name}

        join: set[str] = set()
        if node.hint_tables:
            join.update(node.hint_tables)
        if node._compressed_schemas:
            join.update(s.name for s in node._compressed_schemas)
        if node.requirements and node.requirements.supporting_relations:
            join.update(r.table for r in node.requirements.supporting_relations)
        if context:
            for ctx in context.values():
                join.update(ctx.source_tables)

        primary_cols = {c.name.lower() for c in cand.schema.columns}
        for t in self._candidate_pool:
            if t.schema.name == cand.schema.name:
                continue
            cols = {c.name.lower() for c in t.schema.columns}
            if cols & primary_cols:
                join.add(t.schema.name)

        join &= pool_names
        join -= primary

        task_contexts = list(context.values()) if context else []
        return AttemptSchemaContext(
            primary_tables=primary,
            join_available_tables=join,
            task_contexts=task_contexts,
        )

    def _attempt(
        self,
        node: TaskNode,
        retriever_score: float,
        context: dict[str, ParentContext] | None = None,
    ) -> tuple[NodeResult, bool]:
        hint_section = self._build_hint_section(node)
        context_vars = ""
        if context:
            rendered = [ctx.to_prompt() for ctx in context.values()]
            if rendered:
                context_vars = "\n" + "\n".join(rendered) + "\n"

        max_tokens = self.config.llm.max_tokens_per_call
        requirements = node.requirements
        validator = Validator()
        evaluator = PathEvaluator()

        best_result: NodeResult | None = None
        best_pscore = -1.0
        all_paths: list[ReasoningPath] = []
        total_cost = 0
        max_attempts = self.config.max_attempts_per_node
        attempt_feedback: list[str] = []

        for attempt in range(max_attempts):
            signals = ExecutionSignals(max_attempts=max_attempts)
            signals.num_attempts = attempt + 1

            schema_str = self._build_schema_str()
            system = RLM_SYSTEM.format(
                schema=schema_str,
                task_description=node.description,
                hint_section=hint_section,
            )
            user_prompt = (
                f"Task: {node.description}\n"
                f"Expected output: {node.expected_output_desc}\n"
            )
            if context_vars:
                user_prompt += (
                    "\nTASK CONTEXT (materialized results of earlier tasks; "
                    "use them in FROM/JOIN when they are real tables):\n"
                    f"{context_vars}\n"
                )
            user_prompt += "\nGenerate a SQL query and confidence score."
            if attempt_feedback:
                user_prompt += (
                    "\n\nPrevious attempt failed. Fix the reported problems "
                    "and produce a corrected SQL query:\n"
                    + "\n".join(f"- {msg}" for msg in attempt_feedback)
                )

            response = self.llm.generate(system, user_prompt, max_tokens=max_tokens)
            sql = self._extract_sql(response.content)
            confidence, _ = self._extract_confidence(response.content)
            tokens = response.usage.get("completion_tokens", 0) if response.usage else 0
            total_cost += tokens

            path = ReasoningPath(
                path_id=f"{node.id}-a{attempt}",
                sql=sql or "",
                confidence=confidence,
                cost_tokens=tokens,
            )

            if not sql:
                # DIAGNOSTIC: capture why extraction failed so empty-SQL
                # failures can be grouped by root cause (prose / wrong fence
                # / refusal) instead of collapsing into one "empty SQL".
                snippet = response.content.strip().replace("\n", " \\n ")[:600]
                logger.warning(
                    "  [%s#a%d] empty SQL (tokens=%d) raw: %s",
                    node.id, attempt, tokens, snippet,
                )
                all_paths.append(path)
                break

            if self.config.verbose:
                logger.info("  [%s#a%d] SQL: %s", node.id, attempt, sql[:300])

            syntax_err = self._validate_sql(sql)
            if syntax_err:
                logger.warning("  [%s#a%d] SYNTAX FAIL (attempt %d/%d): %s", node.id, attempt, attempt + 1, max_attempts, syntax_err)
                signals.syntax_errors += 1
                all_paths.append(path)
                if attempt < max_attempts - 1:
                    attempt_feedback.append(f"SQL syntax error: {syntax_err}")
                    continue
                break

            schema_err = self._validate_schema(sql)
            if schema_err:
                logger.warning("  [%s#a%d] SCHEMA FAIL (attempt %d/%d): %s", node.id, attempt, attempt + 1, max_attempts, schema_err)
                signals.schema_errors += 1
                all_paths.append(path)
                if attempt < max_attempts - 1:
                    probe_msg, skip_candidate = self._probe_empty_feedback(sql)
                    if skip_candidate:
                        logger.warning(
                            "  [%s#a%d] CANDIDATE SKIP: PRIMARY provably lacks "
                            "the filter value (verified NOT FOUND); moving to "
                            "the next candidate",
                            node.id, attempt,
                        )
                        break
                    feedback = f"SQL semantic error: {schema_err}"
                    if probe_msg:
                        feedback = f"{feedback}\n{probe_msg}"
                    attempt_feedback.append(feedback)
                    continue
                break

            alias_err = self._validate_aliases(sql, node)
            if alias_err:
                logger.warning("  [%s#a%d] ALIAS FAIL: %s", node.id, attempt, alias_err)
                signals.quality_warnings.append(alias_err)
                all_paths.append(path)
                attempt_feedback.append(f"SQL alias error: {alias_err}")
                continue

            try:
                data = self.executor.execute(sql)
            except Exception as e:
                error_msg = str(e).lower()
                is_wrong_table = any(phrase in error_msg for phrase in
                    ["no such table", "table not found", "doesn't exist",
                     "table does not exist", "relation", "not found", "ambiguous column"])
                logger.warning("  [%s#a%d] EXEC FAIL (attempt %d/%d): %s", node.id, attempt, attempt + 1, max_attempts, str(e)[:200])
                signals.execution_errors += 1
                all_paths.append(path)
                if attempt < max_attempts - 1:
                    if is_wrong_table:
                        attempt_feedback.append(
                            f"SQL execution error: {e}. Use only the tables shown in the schema above."
                        )
                    else:
                        attempt_feedback.append(f"SQL execution error: {e}")
                    continue
                path.path_score = evaluator.evaluate(None, signals, retriever_score)
                node_result = NodeResult(
                    node_id=node.id,
                    data=pd.DataFrame(),
                    sql=sql or "",
                    confidence=confidence or 0.0,
                    reasoning_paths=[path],
                    cost_tokens=total_cost,
                    path_score=path.path_score,
                )
                node_result.ambiguity_score = compute_ambiguity_score(node_result)
                if path.path_score and path.path_score.total > best_pscore:
                    best_pscore = path.path_score.total
                    best_result = node_result
                break

            valid_cols = self._build_valid_columns()
            vresult = validator.validate(sql, requirements, valid_cols)
            path.path_score = evaluator.evaluate(vresult, signals, retriever_score)
            all_paths.append(path)

            if not vresult.passed:
                logger.warning("  [%s#a%d] CONSTRAINT FAIL (attempt %d/%d): %s", node.id, attempt, attempt + 1, max_attempts, "; ".join(vresult.details)[:300])
                if self.config.verbose:
                    logger.info("  [%s#a%d] constraint fail: %s", node.id, attempt, vresult.details)
                if attempt < max_attempts - 1:
                    detail_str = "; ".join(vresult.details) or "requirements not met"
                    user_prompt = (
                        f"Constraint check failed: {detail_str}\n\n"
                        f"Fix the query to satisfy the constraints and try again."
                    )
                    continue
                break

            quality = self._check_result_quality(data)
            if quality and "0 rows" in quality:
                logger.warning("  [%s#a%d] EMPTY RESULT (attempt %d/%d)", node.id, attempt, attempt + 1, max_attempts)
                signals.had_empty_result = True
                if attempt < max_attempts - 1:
                    probe_msg, skip_candidate = self._probe_empty_feedback(sql)
                    if skip_candidate:
                        logger.warning(
                            "  [%s#a%d] CANDIDATE SKIP: PRIMARY provably lacks "
                            "the filter value (verified NOT FOUND); moving to "
                            "the next candidate",
                            node.id, attempt,
                        )
                        break
                    attempt_feedback.append(
                        probe_msg
                        or "SQL executed successfully but returned 0 rows. The filter "
                        "columns, filter values, or join keys may be wrong — "
                        "reconsider which columns hold the required data and retry."
                    )
                    continue
                path.path_score = evaluator.evaluate(vresult, signals, retriever_score)
                break

            node_result = NodeResult(
                node_id=node.id,
                data=data,
                sql=sql,
                confidence=confidence,
                reasoning_paths=[path],
                cost_tokens=total_cost,
                path_score=path.path_score,
                validation=vresult,
            )
            node_result.ambiguity_score = compute_ambiguity_score(node_result)

            if path.path_score and path.path_score.total > best_pscore:
                best_pscore = path.path_score.total
                best_result = node_result

            if self.config.verbose:
                logger.info("  [%s#a%d] path_score=%.3f", node.id, attempt, best_pscore)
            break

        if best_result is None:
            if self.config.verbose:
                logger.info("  [%s] attempt FAILED: no valid SQL (attempts=%d)", node.id, len(all_paths))
            result = NodeResult(
                node_id=node.id,
                data=pd.DataFrame(),
                sql="",
                confidence=0.0,
                reasoning_paths=all_paths,
                cost_tokens=total_cost,
                error="No valid SQL generated",
            )
            result.ambiguity_score = 1.0
            return result, False

        best_result.reasoning_paths = all_paths
        return best_result, True

    def _validate_sql(self, sql: str) -> str | None:
        from sqlglot import parse_one
        from sqlglot.errors import ParseError

        try:
            parse_one(sql)
            return None
        except ParseError as e:
            return str(e)

    def _validate_schema(self, sql: str) -> str | None:
        import sqlglot
        from sqlglot import parse_one
        from sqlglot.expressions import Alias, Column, Table

        try:
            tree = parse_one(sql)
        except Exception:
            return None

        allowed = self._allowed_tables
        known: set[str] = set()
        if allowed:
            known = {
                base_table_name(s.name).lower()
                for s in (self.all_schemas or [])
            } or {base_table_name(t).lower() for t in self.executor.list_tables()}
            allowed_base = {base_table_name(a).lower() for a in allowed}
            for t in self._extract_table_names(sql):
                if t in known and t not in allowed_base:
                    return (
                        f"Table '{t}' is outside the allowed search scope. "
                        f"Use only: {', '.join(sorted(allowed))}"
                    )

        if self._attempt_scope is not None:
            err = self._validate_scope_positions(tree, self._attempt_scope, known)
            if err is not None:
                return err

        valid_columns = self._build_valid_columns()

        select_aliases: set[str] = set()
        for alias in tree.find_all(Alias):
            select_aliases.add(alias.alias.lower())

        order_by_aliases: set[str] = set()
        having_aliases: set[str] = set()
        for statement in (tree for tree in [tree] if isinstance(tree, sqlglot.expressions.Select)):
            for order in statement.args.get("order", sqlglot.expressions.Order(expressions=[])).expressions:
                if isinstance(order, Column) and order.name.lower() in select_aliases:
                    order_by_aliases.add(order.name.lower())
            having = statement.args.get("having")
            if having:
                for col in having.find_all(Column):
                    if col.name.lower() in select_aliases:
                        having_aliases.add(col.name.lower())

        exempt = select_aliases | order_by_aliases | having_aliases

        schemas_by_base: dict[str, TableSchema] = {}
        for s in (self.all_schemas or []) + (getattr(self, "_compressed_schemas", None) or []):
            schemas_by_base[base_table_name(s.name).lower()] = s

        qualifier_to_table: dict[str, str] = {}
        for tbl in tree.find_all(Table):
            base = base_table_name(tbl.name).lower()
            qualifier_to_table.setdefault(base, base)
            if tbl.alias:
                qualifier_to_table.setdefault(tbl.alias.lower(), base)

        for col in tree.find_all(Column):
            if self._is_diff_unit_arg(col):
                continue
            col_name = col.name.lower()
            qualifier = (col.table or "").lower()
            if qualifier:
                # A qualified reference `alias.col` must resolve against the
                # columns of THAT table — the global valid-columns union is too
                # permissive (a column present only on `dw_customer` would let
                # `mart_sales_daily.customer_id` pass here and fail only at
                # execution with UNRESOLVED_COLUMN).
                schema = schemas_by_base.get(qualifier_to_table.get(qualifier, qualifier))
                if schema is not None:
                    table_cols = {c.name.lower() for c in schema.columns}
                    if col_name not in table_cols:
                        avail = ", ".join(sorted(table_cols)) or "(no columns)"
                        return (
                            f"Unknown column '{col.table}.{col.name}': table "
                            f"'{base_table_name(schema.name)}' has no column named "
                            f"'{col.name}'. Columns in {base_table_name(schema.name)}: {avail}"
                        )
            if col_name in valid_columns or col_name in exempt:
                continue
            suggestions = ", ".join(sorted(valid_columns - {"*"}))
            return f"Unknown column '{col.name}'. Available columns: {suggestions}"

        return None

    def _validate_scope_positions(
        self, tree: Any, scope: AttemptSchemaContext, known: set[str]
    ) -> str | None:
        """Enforce the candidate execution invariant (S3).

        FROM  -> PRIMARY | TASK CONTEXT (materialized)
        JOIN  -> PRIMARY | JOIN-AVAILABLE | TASK CONTEXT
        OUT   -> never

        A JOIN-AVAILABLE table used as the FROM anchor is a 'primary switch':
        the model picks candidate X but anchors the SQL on Y, so the attempt
        stops testing X. The search boundary then degenerates — every candidate
        runs identical SQL and selection ties (the S5 log signature). Reject
        the switch with actionable feedback instead. Only *known physical*
        tables are enforced for drift: unknown tables keep surfacing as
        execution errors (documented contract), never schema errors.

        v0.3.5b: TASK CONTEXT (`_task_context_*`) is a *materialized* namespace
        and is allowed in both FROM and JOIN — but only when the name exists in
        this attempt's `task_contexts`. A context name that was never produced
        (or a dependency that did not materialize) is rejected.
        """
        from sqlglot.expressions import CTE, From, Join, Table

        primary = {base_table_name(p).lower() for p in scope.primary_tables}
        join_avail = {base_table_name(j).lower() for j in scope.join_available_tables}
        context_names = {c.table_name.lower() for c in scope.task_contexts}
        materialized_ctx = {
            c.table_name.lower() for c in scope.task_contexts if c.materialized
        }

        cte_names: set[str] = set()
        for cte in tree.find_all(CTE):
            cte_names.add(cte.alias_or_name.lower())

        from_tables: set[str] = set()
        join_tables: set[str] = set()
        for tbl in tree.find_all(Table):
            name = tbl.name.lower()
            if name in cte_names:
                continue
            if isinstance(tbl.parent, From):
                from_tables.add(name)
            elif isinstance(tbl.parent, Join):
                join_tables.add(name)

        referenced = from_tables | join_tables

        # A `_task_context_*` name that was never materialized (either not
        # produced at all, or its dependency did not end SOLVED) is rejected.
        # Unmaterialized contexts are not real tables — the dependent task must
        # re-derive from the physical sources (AMBIGUOUS/FAILED parents never
        # materialize, so their result must not be treated as ground truth).
        unknown_ctx = {
            t for t in referenced
            if t.startswith("_task_context_") and t not in materialized_ctx
        }
        if unknown_ctx:
            name = sorted(unknown_ctx)[0]
            avail = ", ".join(sorted(materialized_ctx)) or "(none)"
            return (
                f"Table '{name}' is not a materialized TASK CONTEXT available "
                f"to this attempt. Available contexts: {avail}. Use only "
                f"materialized dependency results from your parent tasks."
            )

        bad_from = {t for t in from_tables - primary - materialized_ctx if t in known}
        if bad_from:
            name = sorted(bad_from)[0]
            rest = ", ".join(sorted(join_avail)) or "none available"
            return (
                f"Table '{name}' appears in FROM, but the primary table for this "
                f"attempt is '{sorted(primary)[0]}'. FROM must anchor on the "
                f"candidate table under test or a TASK CONTEXT. You may JOIN "
                f"supporting tables only: {rest}."
            )

        bad_join = {t for t in join_tables - (primary | join_avail | materialized_ctx) if t in known}
        if bad_join:
            name = sorted(bad_join)[0]
            return (
                f"Table '{name}' is used in JOIN but is outside this attempt's "
                f"scope. Allowed: {', '.join(sorted(primary | join_avail | context_names)) or '(none)'}."
            )

        return None

    @staticmethod
    def _is_diff_unit_arg(col: Any) -> bool:
        """True when a Column is the unit argument of DATEDIFF/TIMESTAMPDIFF
        (e.g. `DAY` in `TIMESTAMPDIFF(DAY, paid_at, shipped_at)`).

        The unit is parsed as a Column but is a keyword/literal, not a real
        column reference. Rejecting it as "unknown column" (S6) forced the
        RLM away from correct date-diff SQL. The 2-arg form
        (`DATEDIFF(shipped_at, paid_at)`) has no `unit` arg, so its `this`
        column is preserved.
        """
        from sqlglot.expressions import DateDiff, TimestampDiff

        parent = col.parent
        if isinstance(parent, (DateDiff, TimestampDiff)):
            unit = parent.args.get("unit")
            if unit is not None and parent.args.get("this") is col:
                return True
        return False

    @staticmethod
    def _extract_table_names(sql: str) -> set[str]:
        """Physical tables referenced by a SQL statement (CTEs excluded)."""
        from sqlglot import parse_one
        from sqlglot.expressions import Table

        try:
            tree = parse_one(sql)
        except Exception:
            return set()
        ctes: set[str] = set()
        with_clause = tree.args.get("with_") or tree.args.get("with")
        if with_clause is not None:
            for cte in with_clause.expressions:
                ctes.add(cte.alias_or_name.lower())
        names: set[str] = set()
        for t in tree.find_all(Table):
            if t.name.lower() not in ctes:
                names.add(t.name.lower())
        return names

    @staticmethod
    def _extract_context_refs(sql: str) -> bool:
        """True if a SQL statement references a materialized `_task_context_*`
        table in FROM/JOIN (CTEs excluded). Context SQL Usage metric."""
        from sqlglot import parse_one
        from sqlglot.expressions import CTE, From, Join, Table

        try:
            tree = parse_one(sql)
        except Exception:
            return False
        cte_names: set[str] = set()
        for cte in tree.find_all(CTE):
            cte_names.add(cte.alias_or_name.lower())
        for tbl in tree.find_all(Table):
            if tbl.name.lower() in cte_names:
                continue
            if tbl.name.lower().startswith("_task_context_"):
                if isinstance(tbl.parent, (From, Join)):
                    return True
        return False

    def _validate_aliases(self, sql: str, node: TaskNode) -> str | None:
        from sqlglot import parse_one
        from sqlglot.expressions import Alias

        try:
            tree = parse_one(sql)
        except Exception:
            return None

        year_pattern = re.compile(r'\b(19|20)\d{2}\b')

        for alias_node in tree.find_all(Alias):
            raw = alias_node.alias
            if isinstance(raw, str):
                alias_name = raw
            else:
                alias_name = getattr(raw, 'name', '')
            if not alias_name:
                continue

            if year_pattern.search(alias_name):
                return (
                    f"Alias '{alias_name}' contains a year or date. "
                    f"Do not embed dates or years in column aliases. "
                    f"Use a short business-domain term."
                )

            if len(alias_name) > 50:
                return (
                    f"Alias '{alias_name}' is too long ({len(alias_name)} chars). "
                    f"Use a short business-domain term."
                )

            tokens = alias_name.split("_")
            # Only reject when the alias *starts* with a question word
            # (e.g. "what_is_revenue"). "within"/"between" are legitimate
            # domain prepositions (within_24h, between_dates) and must not
            # be treated as question text (S7 ALIAS FAIL loop).
            question_words = {"what", "which", "show", "give", "find", "get",
                              "list", "calculate", "compute"}
            if tokens and tokens[0] in question_words:
                return (
                    f"Alias '{alias_name}' reads like question text. "
                    f"Use a business-domain term instead."
                )

        return None

    def _probe_empty_feedback(self, sql: str) -> tuple[str, bool]:
        """Probe the failing SQL's filter literals against the candidate pool.

        Returns (rendered, skip_candidate):
          - rendered: a VERIFIED DATA FACTS block when a filter value exists in
            some candidate table, so the model learns *where* the data lives.
          - skip_candidate: True when a verified NOT FOUND holds for the current
            PRIMARY table while the same value EXISTS in another pool table —
            deterministic evidence the PRIMARY cannot produce the required rows,
            so retrying it only repeats the same drift.
        """
        if not sql or not self._candidate_pool:
            if self.config.verbose:
                logger.info(
                    "  [probe] skipped (sql=%s pool=%d)",
                    bool(sql), len(self._candidate_pool),
                )
            return "", False
        pool_names = [c.schema.name for c in self._candidate_pool]
        facts = self.probe.probe_sql_filters(sql, pool_names)
        anchored = DataProbe._extract_from_tables(sql)
        rendered = self.probe.render_facts(facts, anchored_tables=anchored)
        skip_candidate = self._probe_suggests_skip(facts)
        if self.config.verbose:
            logger.info(
                "  [probe] filters=%d facts=%d anchored=%s pool=%d skip=%s%s",
                len(facts), len([f for f in facts if f.exists]),
                anchored or "-", len(pool_names), skip_candidate,
                f" -> {rendered.replace(chr(10), ' | ')}" if rendered else "",
            )
        return rendered, skip_candidate

    def _probe_suggests_skip(self, facts: list[ProbeResult]) -> bool:
        """True when the current PRIMARY provably lacks the filter value while
        the same value EXISTS in another pool table — a verified 'wrong table'
        signal, not a wrong literal (which stays retryable)."""
        scope = getattr(self, "_attempt_scope", None)
        if not scope or not scope.primary_tables:
            return False
        primary = {DataProbe._norm(t) for t in scope.primary_tables}
        exists_vals = {(f.column, f.value) for f in facts if f.exists}
        for f in facts:
            if (
                not f.exists
                and DataProbe._norm(f.table) in primary
                and (f.column, f.value) in exists_vals
            ):
                return True
        return False

    def _check_result_quality(self, data: pd.DataFrame) -> str | None:
        if data.empty:
            return "WARNING: Query returned 0 rows. The result may be empty."
        null_cols = [col for col in data.columns if data[col].isna().all()]
        if null_cols:
            return (
                f"WARNING: Column(s) {null_cols} are all NULL in the result. "
                f"The query may be incorrect."
            )
        if len(data) > self.MAX_ROWS_WARNING:
            return (
                f"NOTE: Query returned {len(data)} rows. "
                f"Consider adding LIMIT or aggregation."
            )
        return None

    def _extract_sql(self, content: str) -> str:
        import re

        patterns = [
            r"```sql\n(.*?)```",
            r"```\n(.*?)```",
            r"(?:WITH|SELECT).*?;",
        ]
        for pat in patterns:
            match = re.search(pat, content, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1) if match.lastindex else match.group(0)
                sql = sql.strip()
                if re.match(r"(?:WITH|SELECT)\b", sql, re.IGNORECASE):
                    return sql
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if re.match(r"(?:WITH|SELECT)\b", stripped, re.IGNORECASE):
                return stripped
        return ""

    def _extract_confidence(self, content: str) -> tuple[float, bool]:
        import re

        match = re.search(r"confidence[:\s]+(\d+\.?\d*)", content, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if 0.0 <= val <= 1.0:
                return val, True
        return 0.7, False
