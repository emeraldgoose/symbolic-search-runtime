from __future__ import annotations

from collections import deque
import re

import pandas as pd

from syrch.core.config import ExecutionConfig
from syrch.core.models import FinalSolution, JoinKey, NodeResult, TaskDAG
from syrch.executors.base import BaseExecutor
from syrch.llm.base import BaseLLM
AGGREGATE_SYSTEM = """You are a result aggregator. You have received results from sub-tasks.
Synthesize them into a final answer to the original question.

Original question: {question}

Results from sub-tasks:
{results}

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
        best_data_cost: int | None = None

        all_joins = [jk for node in dag.nodes.values() if node.join_keys for jk in node.join_keys]

        for nid in leaf_ids:
            res = results.get(nid)
            if res is None:
                continue
            summary = f"[{nid}] {res.sql}"
            if res.data is not None and not res.data.empty:
                ncols = len(res.data.columns)
                nrows = len(res.data)
                preview = res.data.head(5).to_string()
                summary += f"\n  → {nrows} rows, {ncols} cols\n  Preview:\n{preview}\n"
            else:
                summary += "\n  → (no data)"
            result_summaries.append(summary)
            total_tokens += res.cost_tokens
            better = res.confidence > best_conf
            tie = res.confidence == best_conf and (best_data_cost is None or res.cost_tokens < best_data_cost)
            if (better or tie) and res.data is not None:
                best_conf = res.confidence
                best_data_cost = res.cost_tokens
                all_data = res.data

        merged = self._try_join_merge(leaf_ids, results, all_joins) if all_joins else None
        if merged is not None:
            all_data = merged

        system = AGGREGATE_SYSTEM.format(
            question=question,
            results="\n".join(result_summaries),
        )

        response = self.llm.generate(system, "Provide the final answer.")
        token_usage = response.usage.get("completion_tokens", 0) if response.usage else 0
        total_tokens += token_usage

        sql_lines: list[str] = []
        for res in results.values():
            if res.sql and res.sql not in sql_lines:
                sql_lines.append(res.sql)

        # Confidence calibration: adjust by ambiguity + heuristics
        max_ambiguity = max(
            (getattr(res, 'ambiguity_score', 0.0) or 0.0) for res in results.values()
        )
        heuristic_penalty = self._check_result_heuristics(question, results)
        adjusted_conf = best_conf * (1.0 - max_ambiguity * 0.5) * (1.0 - heuristic_penalty)
        adjusted_conf = max(0.0, min(1.0, round(adjusted_conf, 3)))

        return FinalSolution(
            question=question,
            answer=response.content,
            data=all_data,
            sql="\n\n".join(sql_lines),
            confidence=adjusted_conf,
            token_cost=total_tokens,
            tree=list(results.values()),
        )

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

        for res in results.values():
            if res.data is not None and res.data.empty:
                total_penalty += 0.25
            if res.error is not None:
                total_penalty += 0.20
            if res.data is not None and not res.data.empty and expected_rows is not None:
                if len(res.data) != expected_rows:
                    total_penalty += 0.10
            if res.data is not None and not res.data.empty and "by year" in q_lower:
                has_year = any("year" in c.lower() for c in res.data.columns)
                if not has_year:
                    total_penalty += 0.15

        return min(total_penalty, 0.5)

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
