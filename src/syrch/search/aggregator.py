from __future__ import annotations

from collections import deque
import re
from typing import Any

import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

from syrch.core.config import ExecutionConfig
from syrch.core.models import FinalSolution, JoinKey, NodeResult, NodeStatus, TaskDAG
from syrch.executors.base import BaseExecutor
from syrch.llm.base import BaseLLM

_AGG_RE = re.compile(r"\b(SUM|AVG|AVERAGE|COUNT|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?([^)]*)\)", re.IGNORECASE)
_FROM_RE = re.compile(r"\bFROM\s+([\w\.]+)(?:\s+(?:AS\s+)?([\w]+))?", re.IGNORECASE)
_JOIN_RE = re.compile(r"\bJOIN\s+([\w\.]+)", re.IGNORECASE)
_JOIN_ON_RE = re.compile(r"\bJOIN\s+[\w\.]+\s+(?:\w+\s+)?ON\s+(.+?)(?=\bWHERE\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|;|$)", re.IGNORECASE | re.DOTALL)
_WHERE_RE = re.compile(r"\bWHERE\s+(.+?)(?=\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|;|$)", re.IGNORECASE | re.DOTALL)
_GROUPBY_RE = re.compile(r"\bGROUP\s+BY\s+(.+?)(?=\bORDER\s+BY\b|\bHAVING\b|\bLIMIT\b|;|$)", re.IGNORECASE | re.DOTALL)
_REFUND_RE = re.compile(r"\brefunded\b", re.IGNORECASE)

DetectorFactory.seed = 0

_LANG_MAP = {
    "ko": "Korean", "ja": "Japanese", "zh-cn": "Chinese", "zh-tw": "Chinese",
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "pt": "Portuguese", "ru": "Russian", "ar": "Arabic", "vi": "Vietnamese",
    "th": "Thai", "id": "Indonesian", "hi": "Hindi", "tr": "Turkish",
    "it": "Italian", "nl": "Dutch", "pl": "Polish", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "cs": "Czech", "ro": "Romanian",
    "hu": "Hungarian", "el": "Greek", "he": "Hebrew", "uk": "Ukrainian",
}


def _detect_language(text: str) -> str:
    try:
        code = detect(text)
        return _LANG_MAP.get(code, "English")
    except LangDetectException:
        return "English"


AGGREGATE_SYSTEM = """You are a result aggregator. You have received results from sub-tasks.
Synthesize them into a final answer to the original question.

Original question: {question}

Results from sub-tasks:
{results}

{language_instruction}
Provide:
1. A concise answer to the original question
2. Supporting evidence from the data
3. The SQL queries that produced the key results (if applicable)
"""

NUM_PATTERN = re.compile(r"(top|bottom)\s+(\d+)", re.IGNORECASE)
EXPECTED_ROW_PATTERNS = [
    (re.compile(r"average|avg|mean|count\s+all|total\s+(revenue|sales|count)", re.IGNORECASE), 1),
    (re.compile(r"(top|bottom)\s+(\d+)", re.IGNORECASE), None),  # dynamic
    (re.compile(r"unique|distinct", re.IGNORECASE), None),
]


class Aggregator:
    def __init__(self, llm: BaseLLM, executor: BaseExecutor, config: ExecutionConfig):
        self.llm = llm
        self.executor = executor
        self.config = config
        self._schema_memo: dict[str, Any | None] = {}

    def merge(
        self,
        question: str,
        dag: TaskDAG,
        results: dict[str, NodeResult],
    ) -> FinalSolution:
        leaf_ids = self._find_leaves(dag)
        result_summaries: list[str] = []
        all_data = None
        total_tokens = 0
        best_conf = 0.0

        all_joins = [jk for node in dag.nodes.values() if node.join_keys for jk in node.join_keys]

        # The Aggregator does NOT re-rank candidates. Each node already chose
        # its own result (Local Selection). Here we only pick which leaf result
        # to synthesize around, preferring SOLVED leaves by their selected
        # candidate's evidence and preserving ambiguity instead of faking a
        # winner.
        leaf_results: list[NodeResult] = []
        for nid in leaf_ids:
            res = results.get(nid)
            if res is None:
                continue
            leaf_results.append(res)
            total_tokens += res.cost_tokens
            summary = f"[{nid}] {res.sql}"
            if res.data is not None and not res.data.empty:
                ncols = len(res.data.columns)
                nrows = len(res.data)
                preview = res.data.head(5).to_string()
                summary += f"\n  → {nrows} rows, {ncols} cols\n  Preview:\n{preview}\n"
            else:
                summary += "\n  → (no data)"
            result_summaries.append(summary)

        primary = self._pick_primary(leaf_results)

        if (primary is None
                or primary.status != NodeStatus.SOLVED
                or primary.data is None
                or primary.data.empty):
            fallback = self._pick_solved_any(list(results.values()))
            if fallback is not None:
                primary = fallback

        # An AMBIGUOUS/FAILED leaf's provisional data must never become the
        # answer basis (S-run: a fully-tied node leaked its alphabetical
        # winner's NULL-total SQL upward as if it were the answer). Only a
        # SOLVED selection may carry data; ambiguity keeps lowering
        # confidence (below) instead of masquerading as a solved number.
        if (
            primary is not None
            and primary.status == NodeStatus.SOLVED
            and primary.data is not None
            and not primary.data.empty
        ):
            all_data = primary.data
        if primary is not None:
            best_conf = primary.confidence
            if primary.selected_candidate is not None:
                best_conf = max(best_conf, primary.selected_candidate.confidence)

        merged = self._try_join_merge(leaf_ids, results, all_joins) if all_joins else None
        if merged is not None:
            all_data = merged

        lang_name = _detect_language(question)
        lang_instruction = f"Answer in {lang_name}." if lang_name != "English" else ""

        system = AGGREGATE_SYSTEM.format(
            question=question,
            results="\n".join(result_summaries),
            language_instruction=lang_instruction,
        )

        response = self.llm.generate(system, "Provide the final answer.")
        token_usage = response.usage.get("completion_tokens", 0) if response.usage else 0
        total_tokens += token_usage

        sql_lines: list[str] = []
        for res in results.values():
            if res.sql and res.sql not in sql_lines:
                sql_lines.append(res.sql)

        # Confidence: preserve ambiguity instead of masking it. A leaf that the
        # local search could not resolve lowers confidence; a BLOCKED/FAILED
        # leaf is penalized so the answer reflects the missing dependency.
        max_ambiguity = max(
            (getattr(res, 'ambiguity_score', 0.0) or 0.0) for res in results.values()
        )
        heuristic_penalty = self._check_result_heuristics(question, results)
        if any(r.status == NodeStatus.AMBIGUOUS for r in leaf_results):
            heuristic_penalty = min(1.0, heuristic_penalty + 0.10)
        adjusted_conf = best_conf * (1.0 - max_ambiguity * 0.5) * (1.0 - heuristic_penalty)
        adjusted_conf = max(0.0, min(1.0, round(adjusted_conf, 3)))

        cand_ps = (
            primary.selected_candidate.path_score.total
            if primary is not None
            and primary.selected_candidate is not None
            and primary.selected_candidate.path_score is not None
            else -1.0
        )
        path_score = cand_ps if cand_ps >= 0 else adjusted_conf

        self._last_question = question
        self._last_dag: TaskDAG | None = dag

        return FinalSolution(
            question=question,
            answer=response.content,
            data=all_data,
            sql="\n\n".join(sql_lines),
            confidence=adjusted_conf,
            path_score=path_score,
            token_cost=total_tokens,
            tree=list(results.values()),
            calculation_basis=self._build_calculation_basis(sql_lines, results),
        )

    def _build_calculation_basis(
        self, sql_lines: list[str], results: dict[str, NodeResult]
    ) -> str:
        """Step-by-step explanation of how the final numbers were computed.

        Derived from executed SQL + row counts only (no LLM re-interpretation).
        Each step names the table, filter criteria and how the result was
        aggregated, so the user can verify the calculation from observable
        facts alone. Preceded by a deterministic criteria block stating which
        semantic decisions the executed queries encode (time window, refund/
        status exclusion, SCD2 entity-state semantics) — different legitimate
        readings of the same question yield different numbers, so the basis
        must say which reading produced THIS number.
        """
        steps = self._build_step_explanation(sql_lines, results)
        criteria = self._extract_criteria("\n\n".join(sql_lines))
        parts: list[str] = []
        if criteria:
            parts.append("Criteria applied (derived from executed SQL):\n" + criteria)
        if steps:
            parts.append(steps)
        return "\n\n".join(parts)

    def _extract_criteria(self, sql_text: str) -> str:
        """Deterministic semantic-criteria summary of the executed SQL.

        Observable facts only: literal date bounds, status/refund predicates,
        and how SCD2 validity columns are actually compared (per-order column
        reference vs constant bound vs ignored). Never consults ground truth;
        unknowns are stated as unknown instead of guessed."""
        lines: list[str] = []

        iso_dates = sorted(set(re.findall(r"'(\d{4}(?:-\d{2}){0,2})(?:-\d{2})?'", sql_text)))
        if iso_dates:
            lines.append(
                f"- time window: {iso_dates[0]} .. {iso_dates[-1]}"
            )
        else:
            lines.append("- time window: none explicitly bounded in SQL")

        excl = self._detect_status_exclusion(sql_text)
        lines.append(f"- refund/status exclusion: {excl}")

        for note in self._detect_scd2_semantics(sql_text):
            lines.append(f"- entity state (SCD2): {note}")

        return "\n".join(lines)

    @staticmethod
    def _detect_status_exclusion(sql_text: str) -> str:
        """Report whether a status-like predicate excludes outcome values."""
        m = re.search(
            r"\b(\w*status\w*)\s*(?:!=|<>\s*)\s*'([a-z_]+)'",
            sql_text,
            re.IGNORECASE,
        )
        if m:
            return f"applied ({m.group(1)} != '{m.group(2)}')"
        neg_in = re.search(
            r"\b(\w*status\w*)\s+not\s+in\s*\(([^)]+)\)",
            sql_text,
            re.IGNORECASE,
        )
        if neg_in:
            return f"applied ({neg_in.group(1)} not in {neg_in.group(2).strip()})"
        return (
            "NOT APPLIED — totals include refunded/cancelled orders "
            "if such rows exist"
        )

    def _detect_scd2_semantics(self, sql_text: str) -> list[str]:
        """Classify how each SCD2 source table's validity window was applied.

        point-in-time  — validity columns compared against another COLUMN
                         (e.g. o.order_date >= c.valid_from or
                         o.order_date <= c.valid_to): the state as of each
                         transaction decides membership.
        fixed-window   — validity columns compared only against LITERAL bounds
                         (e.g. valid_from <= '2024-12-31'): an overlap
                         approximation, not per-order state.
        ignored        — table carries validity columns but the SQL never
                         references them.
        """
        notes: list[str] = []
        seen_tables: set[str] = set()
        table_pattern = re.compile(
            r"\b(?:FROM|JOIN)\s+([\w\.]+)"
            r"(?:\s+(?:AS\s+)?)?"
            r"(?!(?:JOIN|WHERE|GROUP|ORDER|ON|AS|LEFT|RIGHT|INNER|OUTER|FULL|CROSS|LIMIT|HAVING)\b)"
            r"(\w+)?",
            re.IGNORECASE,
        )
        for match in table_pattern.finditer(sql_text):
            table = match.group(1)
            alias = match.group(2)
            base = table.split(".")[-1]
            if base.lower() in seen_tables:
                continue
            schema = self._cached_schema(table)
            if schema is None:
                continue
            cols = {c.name.lower() for c in schema.columns}
            if not {"valid_from", "valid_to"} <= cols:
                continue
            seen_tables.add(base.lower())
            refs = self._validity_comparisons(sql_text, [q for q in {alias, base} if q])
            if any(rhs_is_column for _, rhs_is_column in refs):
                notes.append(
                    f"{base}: point-in-time — validity compared against the "
                    f"transaction date per row"
                )
            elif refs:
                bounds = sorted({v.strip("'") for v, is_col in refs if not is_col})
                notes.append(
                    f"{base}: fixed-window overlap approximation — validity "
                    f"filtered with constant bounds {bounds}, NOT evaluated "
                    f"per order"
                )
            else:
                notes.append(
                    f"{base}: validity window ignored — rows used regardless "
                    f"of state at transaction time"
                )
        return notes

    def _cached_schema(self, table: str):
        if table not in self._schema_memo:
            try:
                self._schema_memo[table] = self.executor.get_schema(table)
            except Exception:
                self._schema_memo[table] = None
        return self._schema_memo[table]

    @staticmethod
    def _validity_comparisons(
        sql_text: str, qualifiers: list[str]
    ) -> list[tuple[str, bool]]:
        """Find validity-column comparisons in either orientation:
        `q.valid_(from|to) <op> rhs` or `rhs <op> q.valid_(from|to)`.

        Returns (rhs_token, rhs_is_column_ref) pairs. A quoted literal is not
        a column ref; anything else (dotted or bare identifier) is treated as
        one."""
        out: list[tuple[str, bool]] = []
        qual = "|".join(re.escape(q) for q in qualifiers)
        # Qualifier is optional: SQL frequently references validity columns
        # unqualified (single-table WHERE clauses).
        col = rf"(?:(?:{qual})\.)?valid_(?:from|to)"
        op = r"\s*(?:>=|<=|<>|=|>|<)\s*"
        for m in re.finditer(rf"\b(?:{col}){op}([^\s(),]+)", sql_text, re.IGNORECASE):
            rhs = m.group(1)
            out.append((rhs, not rhs.startswith("'")))
        for m in re.finditer(rf"\b([\w\.]+){op}(?:{col})\b", sql_text, re.IGNORECASE):
            lhs = m.group(1)
            if not lhs.lower().endswith("valid_from") and not lhs.lower().endswith("valid_to"):
                out.append((lhs, True))
        return out

    def _build_step_explanation(
        self, sql_lines: list[str], results: dict[str, NodeResult]
    ) -> str:
        dag = getattr(self, '_last_dag', None)

        # Walk DAG layers in execution order when available; fall back to
        # sql_lines order for callers that don't provide a DAG.
        ordered_ids: list[str] = []
        if dag is not None and hasattr(dag, 'topo_layers'):
            for layer in dag.topo_layers:
                ordered_ids.extend(layer)

        # Map node_id -> NodeResult for row-count lookup
        by_id: dict[str, NodeResult] = {r.node_id: r for r in results.values()}
        # Map sql -> node_id for reverse lookup
        sql_to_id: dict[str, str] = {r.sql: r.node_id for r in results.values() if r.sql}

        steps: list[str] = []

        if ordered_ids:
            step_no = 0
            for nid in ordered_ids:
                res = by_id.get(nid)
                if res is None or not res.sql:
                    continue
                step_no += 1
                text = self._render_step(step_no, nid, res, by_id)
                if text:
                    steps.append(text)
            # Any results not in DAG order (orphan nodes)
            for sql in sql_lines:
                orphan_nid: str | None = sql_to_id.get(sql)
                if orphan_nid is None or orphan_nid in ordered_ids:
                    continue
                res = by_id.get(orphan_nid)
                if res is None:
                    continue
                step_no += 1
                text = self._render_step(step_no, orphan_nid, res, by_id)
                if text:
                    steps.append(text)
        else:
            for idx, sql in enumerate(sql_lines):
                if not sql:
                    continue
                nid = sql_to_id.get(sql, f"step{idx + 1}")
                res = results.get(nid) if nid in results else None
                # Find any result whose sql matches
                if res is None:
                    for r in results.values():
                        if r.sql == sql:
                            res = r
                            nid = r.node_id
                            break
                if res is not None:
                    text = self._render_step(idx + 1, nid, res, by_id)
                else:
                    text = self._summarize_sql_fallback(sql, sql_to_id)
                if text:
                    steps.append(text)

        return "\n\n".join(s for s in steps if s)

    def _render_step(
        self,
        step_no: int,
        node_id: str,
        res: NodeResult,
        by_id: dict[str, NodeResult],
    ) -> str:
        sql = res.sql
        row_count = len(res.data) if res.data is not None and not res.data.empty else 0

        # --- Parse structural info from SQL ---
        aggs = _AGG_RE.findall(sql)
        tables: list[str] = []
        for m in _FROM_RE.finditer(sql):
            tables.append(self._short_table(m.group(1)))
        for j in _JOIN_RE.finditer(sql):
            tbl = self._short_table(j.group(1))
            if tbl not in tables:
                tables.append(tbl)

        # Separate validity-window condition from other filters
        where_conds: list[str] = []
        for m in _WHERE_RE.finditer(sql):
            cond = m.group(1).strip()
            if cond:
                where_conds.append(" ".join(cond.split()))

        join_conds: list[str] = []
        for m in _JOIN_ON_RE.finditer(sql):
            cond = m.group(1).strip()
            if cond:
                join_conds.append(" ".join(cond.split()))

        # --- Build description ---
        # Check if this step's source is a task context (dependent)
        depends_label = ""
        for jc in join_conds:
            ctx_m = re.search(r'_task_context_(\w+)', jc)
            if ctx_m:
                dep_id = ctx_m.group(1)
                dep_res = by_id.get(dep_id)
                dep_rows = len(dep_res.data) if dep_res and dep_res.data is not None and not dep_res.data.empty else 0
                if dep_rows:
                    depends_label = f" (joined with {dep_rows:,} rows from step {dep_id})"
                else:
                    depends_label = f" (joined with result of step {dep_id})"
                break
        if not depends_label:
            for tbl in tables:
                if tbl.startswith("_task_context_"):
                    dep_id = tbl.replace("_task_context_", "")
                    dep_res = by_id.get(dep_id)
                    dep_rows = len(dep_res.data) if dep_res and dep_res.data is not None and not dep_res.data.empty else 0
                    if dep_rows:
                        depends_label = f" (joined with {dep_rows:,} rows from step {dep_id})"
                    break

        # Physical tables (exclude _task_context_*)
        phys_tables = [t for t in tables if not t.startswith("_task_context_")]
        table_label = " + ".join(phys_tables) if phys_tables else (tables[0] if tables else "?")

        lines: list[str] = []
        lines.append(f"Step {step_no} — {table_label}{depends_label}")

        if where_conds:
            lines.append(f"  filter: {' AND '.join(where_conds)}")

        if join_conds:
            lines.append(f"  join: {' AND '.join(join_conds)}")

        if aggs:
            agg_str = ", ".join(f"{a.upper()}({c.strip()})" for a, c in aggs)
            if aggs and row_count == 1 and res.data is not None:
                try:
                    val = res.data.iloc[0].iloc[0]
                    if isinstance(val, float):
                        lines.append(f"  aggregate: {agg_str} → {val:,.2f} ({row_count} row)")
                    else:
                        lines.append(f"  aggregate: {agg_str} → {val} ({row_count} row)")
                except Exception:
                    lines.append(f"  aggregate: {agg_str} ({row_count} row)")
            elif row_count > 0:
                lines.append(f"  aggregate: {agg_str} ({row_count:,} rows)")
            else:
                lines.append(f"  aggregate: {agg_str}")
        elif row_count > 0:
            lines.append(f"  result: {row_count:,} rows")

        return "\n".join(lines)

    @staticmethod
    def _short_table(name: str) -> str:
        return name.split(".")[-1]

    def _summarize_sql_fallback(self, sql: str, sql_to_id: dict[str, str]) -> str:
        """Fallback for SQL not linked to a NodeResult."""
        source = ""
        nid = sql_to_id.get(sql)
        if nid:
            source = f" (task {nid})"
        else:
            ctx = re.search(r"_task_context_(\w+)", sql)
            if ctx:
                source = f" (JOINs task {ctx.group(1)} context)"
        aggs = _AGG_RE.findall(sql)
        tables: list[str] = []
        for m in _FROM_RE.finditer(sql):
            tables.append(self._short_table(m.group(1)))
        for j in _JOIN_RE.finditer(sql):
            tbl = self._short_table(j.group(1))
            if tbl not in tables:
                tables.append(tbl)
        filters: list[str] = []
        for m in _WHERE_RE.finditer(sql):
            cond = m.group(1).strip()
            if cond:
                filters.append(" ".join(cond.split()))
        for m in _JOIN_ON_RE.finditer(sql):
            cond = m.group(1).strip()
            if cond:
                filters.append(" ".join(cond.split()))
        lines = [f"[{len(tables) and ' + '.join(tables) or '?'}]" + source]
        if aggs:
            agg_str = ", ".join(f"{a.upper()}({c.strip()})" for a, c in aggs)
            lines.append(f"  aggregate: {agg_str}")
        if filters:
            lines.append(f"  filters: {' AND '.join(filters)}")
        return "\n".join(lines)

    _summarize_sql = _summarize_sql_fallback  # compat alias (tests call the old name)

    @staticmethod
    def _pick_primary(leaf_results: list[NodeResult]) -> NodeResult | None:
        """Trust node-level Local Selection; never re-rank candidates.

        SOLVED leaves (with a `selected_candidate`) outrank AMBIGUOUS leaves;
        among SOLVED leaves the selected candidate's evidence is used. Only a
        SOLVED leaf is answer-capable: an AMBIGUOUS leaf's provisional data is
        never promoted to the primary just because it happens to be non-empty
        (that would launder an arbitrary tie-break into a winner).
        """

        def key(res: NodeResult):
            sel = res.selected_candidate
            if sel is not None:
                return (
                    1,
                    sel.semantic_match,
                    sel.result_quality,
                    sel.structural_match,
                    -sel.cost_tokens,
                )
            if res.status == NodeStatus.AMBIGUOUS:
                return (0, res.confidence, 0.0, 0.0, 0)
            return (-1, 0.0, 0.0, 0.0, 0)

        ranked = sorted(leaf_results, key=key, reverse=True)
        for res in ranked:
            if (
                res.status == NodeStatus.SOLVED
                and res.data is not None
                and not res.data.empty
            ):
                return res
        return ranked[0] if ranked else None

    @staticmethod
    def _pick_solved_any(results: list[NodeResult]) -> NodeResult | None:
        """Fallback when no leaf carries usable data: surface the best SOLVED
        node's result (dependency evidence) instead of returning nothing.
        The FAILED/BLOCKED status still drives the confidence penalty below."""
        solved = [
            r for r in results
            if r.status == NodeStatus.SOLVED
            and r.selected_candidate is not None
            and r.data is not None
            and not r.data.empty
        ]

        def key(res: NodeResult):
            sel = res.selected_candidate
            if sel is None:
                return (0.0, 0.0, 0.0, 0)
            return (
                sel.semantic_match,
                sel.result_quality,
                sel.structural_match,
                -sel.cost_tokens,
            )

        return max(solved, key=key) if solved else None

    def _check_result_heuristics(
        self, question: str, results: dict[str, NodeResult]
    ) -> float:
        """Return a penalty between 0.0 and 1.0 based on result quality."""
        total_penalty = 0.0
        q_lower = question.lower()
        has_top_n = NUM_PATTERN.search(q_lower)
        expected_rows = None
        if has_top_n:
            expected_rows = int(has_top_n.group(2))

        global_by_year = "by year" in q_lower
        applied_by_year = False

        for res in results.values():
            if res.status in (NodeStatus.FAILED, NodeStatus.BLOCKED):
                total_penalty += 0.15
            if res.data is not None and res.data.empty:
                total_penalty += 0.15
            if res.error is not None:
                total_penalty += 0.15
            if res.data is not None and not res.data.empty and expected_rows is not None:
                if len(res.data) != expected_rows:
                    total_penalty += 0.05
            if res.data is not None and not res.data.empty and global_by_year and not applied_by_year:
                has_year = any("year" in c.lower() for c in res.data.columns)
                if not has_year:
                    total_penalty += 0.10
                    applied_by_year = True

        return min(total_penalty, 0.4)

    def _try_join_merge(
        self,
        leaf_ids: list[str],
        results: dict[str, NodeResult],
        joins: list[JoinKey],
    ) -> pd.DataFrame | None:
        anchor = None
        for lid in leaf_ids:
            r = results.get(lid)
            if r is not None and r.data is not None and not r.data.empty:
                anchor = lid
                break
        if anchor is None:
            return None

        graph: dict[str, list[tuple[str, JoinKey]]] = {}
        for jk in joins:
            graph.setdefault(jk.left, []).append((jk.right, jk))
            graph.setdefault(jk.right, []).append((jk.left, jk))

        df = results[anchor].data.copy()
        merged_ids: set[str] = {anchor}
        queue: deque[str] = deque([anchor])

        while queue:
            current = queue.popleft()
            for neighbor, jk in graph.get(current, []):
                if neighbor in merged_ids:
                    continue
                if jk.left == current and jk.right == neighbor:
                    right = results.get(jk.right)
                    if right is not None and right.data is not None and not right.data.empty:
                        df = df.merge(right.data, left_on=jk.left_col, right_on=jk.right_col, how=jk.how)
                elif jk.left == neighbor and jk.right == current:
                    left = results.get(jk.left)
                    if left is not None and left.data is not None and not left.data.empty:
                        df = df.merge(left.data, left_on=jk.right_col, right_on=jk.left_col, how=jk.how)
                merged_ids.add(neighbor)
                queue.append(neighbor)

        return df if len(merged_ids) > 1 else None

    def _find_leaves(self, dag: TaskDAG) -> list[str]:
        all_dependents: set[str] = set()
        for node in dag.nodes.values():
            all_dependents.update(node.depends_on)
        leaves = [nid for nid in dag.nodes if nid not in all_dependents]
        return leaves or [dag.root_id]
