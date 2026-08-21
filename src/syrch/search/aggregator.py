from __future__ import annotations

from collections import deque
import re

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
                or primary.data is None
                or primary.data.empty):
            fallback = self._pick_solved_any(list(results.values()))
            if fallback is not None:
                primary = fallback

        if primary is not None and primary.data is not None and not primary.data.empty:
            all_data = primary.data
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
        """Explain *how* the final numbers were computed, derived from the
        executed SQL alone (no LLM re-interpretation).

        Domain-ambiguous questions (e.g. SCD "purchase-time" semantics) cannot
        be resolved by the system; surfacing the exact join/filter/aggregate
        lets the user detect that the interpretation differs from their intent
        and re-ask.
        """
        node_desc: dict[str, str] = {
            res.node_id: res.sql for res in results.values()
        }
        parts: list[str] = []
        for sql in sql_lines:
            if not sql:
                continue
            part = self._summarize_sql(sql, node_desc)
            if part:
                parts.append(part)
        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _short_table(name: str) -> str:
        return name.split(".")[-1]

    def _summarize_sql(self, sql: str, node_desc: dict[str, str]) -> str:
        # Which node produced this SQL (context names map back to their task).
        source = ""
        for nid, nsql in node_desc.items():
            if nsql == sql:
                source = f" (task {nid})"
                break
        if not source:
            ctx = re.search(r"_task_context_(\w+)", sql)
            if ctx:
                source = f" (JOINs task {ctx.group(1)} context)"

        aggs = _AGG_RE.findall(sql)
        agg_str = ", ".join(
            f"{a.upper()}({c.strip()})" for a, c in aggs
        ) if aggs else "raw row selection (no aggregate)"

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
            lines.append(f"  aggregate: {agg_str}")
        if filters:
            lines.append(f"  filters: {' AND '.join(filters)}")

        return "\n".join(lines)

    @staticmethod
    def _pick_primary(leaf_results: list[NodeResult]) -> NodeResult | None:
        """Trust node-level Local Selection; never re-rank candidates.

        SOLVED leaves (with a `selected_candidate`) outrank AMBIGUOUS leaves;
        among SOLVED leaves the selected candidate's evidence is used. An
        AMBIGUOUS leaf keeps its provisional data only if no SOLVED leaf
        exists — it is not fabricated as a winner.
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
            if res.data is not None and not res.data.empty:
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
