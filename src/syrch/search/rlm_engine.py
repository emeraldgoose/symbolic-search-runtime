from __future__ import annotations

import logging

import pandas as pd

import re

from syrch.core.config import ExecutionConfig
from syrch.core.models import NodeResult, ReasoningPath, TaskNode
from syrch.executors.base import BaseExecutor
from syrch.llm.base import BaseLLM
from syrch.search.calibrator import ExecutionSignals, calibrate
from syrch.search.clarify import compute_ambiguity_score
from syrch.search.retriever import Retriever

logger = logging.getLogger(__name__)

RLM_SYSTEM = """You are a constraint-based SQL generator.

CRITICAL RULES — Follow ALL:

1. USE ONLY COLUMNS SHOWN IN THE SCHEMA BELOW. Never invent column names.
   Map business terms (e.g. "revenue" → total_amount, "items" → quantity)
   to physical column names. Never use a business term as a column name directly.

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
    ):
        self.llm = llm
        self.executor = executor
        self.config = config
        self.retriever = retriever
        self.all_schemas = all_schemas
        self.alias_map = alias_map or {}
        self._compressed_schemas: list | None = None

    def set_compressed_schemas(self, schemas: list | None) -> None:
        self._compressed_schemas = schemas

    def _build_schema_str(self) -> str:
        if self._compressed_schemas is not None:
            parts: list[str] = []
            for s in self._compressed_schemas:
                col_parts = []
                for c in s.columns:
                    desc = f" ({c.description})" if c.description else ""
                    col_parts.append(f"{c.name} ({c.type}){desc}")
                parts.append(f"Table: {s.name}\nColumns: {', '.join(col_parts)}")
            return "\n\n".join(parts) if parts else "No tables available."
        tables = self.executor.list_tables()
        parts = []
        for t in tables:
            schema = self.executor.get_schema(t)
            col_parts = []
            for c in schema.columns:
                desc = f" ({c.description})" if c.description else ""
                col_parts.append(f"{c.name} ({c.type}){desc}")
            parts.append(f"Table: {schema.name}\nColumns: {', '.join(col_parts)}")
        return "\n\n".join(parts)

    def _build_valid_columns(self) -> set[str]:
        if self._compressed_schemas is not None:
            cols: set[str] = set()
            for s in self._compressed_schemas:
                cols.update(c.name.lower() for c in s.columns)
            cols.add("*")
            return cols
        cols = set()
        for t in self.executor.list_tables():
            schema = self.executor.get_schema(t)
            cols.update(c.name.lower() for c in schema.columns)
        cols.add("*")
        return cols

    def _build_hint_section(self, node: TaskNode) -> str:
        parts: list[str] = []
        if node.hint_tables:
            parts.append("Recommended tables: " + ", ".join(node.hint_tables))
        if node.hint_columns:
            parts.append("Prefer these columns: " + ", ".join(node.hint_columns))
        if node.metric_columns:
            parts.append("Metric columns (use these for aggregation): " + ", ".join(node.metric_columns))
        if node.grain:
            parts.append("Row granularity: " + node.grain)
        if node.time_columns:
            parts.append("Time columns for date filtering: " + ", ".join(node.time_columns))
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
        context: dict[str, NodeResult] | None = None,
    ) -> NodeResult:
        paths: list[ReasoningPath] = []
        best_path: ReasoningPath | None = None
        total_cost = 0
        signals = ExecutionSignals(max_attempts=self.config.max_attempts_per_node)

        schema_str = self._build_schema_str()
        hint_section = self._build_hint_section(node)

        context_vars = ""
        if context:
            for nid, res in context.items():
                if res.data is not None and not res.data.empty:
                    preview = res.data.head(3).to_string()
                    context_vars += f"\nresult_{nid} =\n{preview}\n"

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
            user_prompt += f"\nAvailable results from dependencies:\n{context_vars}\n"
        user_prompt += "\nGenerate a SQL query and confidence score."

        max_tokens = self.config.llm.max_tokens_per_call

        for attempt in range(self.config.max_attempts_per_node):
            signals.num_attempts = attempt + 1
            response = self.llm.generate(system, user_prompt, max_tokens=max_tokens)
            sql = self._extract_sql(response.content)
            confidence, confidence_found = self._extract_confidence(response.content)
            tokens = response.usage.get("completion_tokens", 0) if response.usage else 0
            total_cost += tokens

            if not sql:
                path = ReasoningPath(
                    path_id=f"{node.id}-{attempt}",
                    sql="",
                    confidence=confidence,
                    cost_tokens=tokens,
                )
                paths.append(path)
                if attempt < self.config.max_attempts_per_node - 1:
                    user_prompt = (
                        f"Attempt {attempt + 1}: No SQL found. "
                        f"Please generate a valid SQL query."
                    )
                continue

            if self.config.verbose:
                logger.info(
                    "  [%s#%d] SQL (%d chars): %s",
                    node.id, attempt, len(sql), sql[:400],
                )

            path = ReasoningPath(
                path_id=f"{node.id}-{attempt}",
                sql=sql,
                confidence=confidence,
                cost_tokens=tokens,
            )

            syntax_error = self._validate_sql(sql)
            if syntax_error:
                signals.syntax_errors += 1
                paths.append(path)
                if best_path is None or confidence > best_path.confidence:
                    best_path = path
                user_prompt = (
                    f"SQL syntax error: {syntax_error}\n\n"
                    f"Fix the SQL syntax and try again."
                )
                continue

            schema_error = self._validate_schema(sql)
            if schema_error:
                signals.schema_errors += 1
                paths.append(path)
                if best_path is None or confidence > best_path.confidence:
                    best_path = path
                user_prompt = (
                    f"SQL semantic error: {schema_error}\n\n"
                    f"Fix the query and try again."
                )
                continue

            col_selection = self._validate_column_selection(sql, node)
            if col_selection:
                signals.quality_warnings.append(col_selection)
                paths.append(path)
                if best_path is None or confidence > best_path.confidence:
                    best_path = path
                user_prompt = f"{col_selection}\n\nTry again."
                continue

            try:
                data = self.executor.execute(sql)
            except Exception as e:
                error_msg = str(e).lower()
                # Non-recoverable errors → signal replan
                if self._is_non_recoverable(error_msg, node):
                    result = NodeResult(
                        node_id=node.id,
                        data=pd.DataFrame(),
                        sql=sql,
                        confidence=0.0,
                        reasoning_paths=paths,
                        cost_tokens=total_cost,
                        error=f"Non-recoverable: {e}",
                        replan_request=self._build_replan_reason(error_msg, node),
                    )
                    result.ambiguity_score = 1.0
                    return result
                signals.execution_errors += 1
                paths.append(path)
                if best_path is None or confidence > best_path.confidence:
                    best_path = path
                user_prompt = (
                    f"SQL execution error: {e}\n\n"
                    f"Try again with a corrected SQL query."
                )
                continue

            paths.append(path)

            if best_path is None or confidence > best_path.confidence:
                best_path = path

            if not confidence_found and data is not None and not data.empty:
                confidence = 0.85
            best_path = path
            best_path.confidence = confidence

            quality_feedback = self._check_result_quality(data)
            if quality_feedback:
                signals.quality_warnings.append(quality_feedback)
                if "0 rows" in quality_feedback:
                    signals.had_empty_result = True
                    diagnosis = self._diagnose_empty_result(sql, node)
                    if diagnosis:
                        user_prompt = f"{quality_feedback}\n{diagnosis}\n\nFix the issue and try again."
                    else:
                        user_prompt = f"{quality_feedback}\n\nTry a different SQL approach."
                elif "all NULL" in quality_feedback:
                    signals.had_null_columns = True
                    user_prompt = f"{quality_feedback}\n\nTry a different SQL approach."
                elif f"{self.MAX_ROWS_WARNING}" in quality_feedback:
                    signals.had_overflow_result = True
                    user_prompt = f"{quality_feedback}\n\nTry a different SQL approach."
                else:
                    user_prompt = f"{quality_feedback}\n\nTry a different SQL approach."
                continue

            semantic_feedback = self._check_semantic_match(data, sql, node)
            if semantic_feedback:
                signals.quality_warnings.append(semantic_feedback)
                user_prompt = (
                    f"Semantic mismatch: {semantic_feedback}\n\n"
                    f"Try a different SQL query. "
                    f"Use the correct tables and columns to match the question."
                )
                if attempt < self.config.max_attempts_per_node - 1:
                    self._expand_compressed_schemas(semantic_feedback, data)
                continue

            if confidence >= self.config.high_confidence:
                break

            if attempt < self.config.max_attempts_per_node - 1:
                user_prompt = (
                    f"Attempt {attempt + 1} confidence was {confidence:.2f} "
                    f"(target: {self.config.high_confidence}). "
                    f"Try a different SQL approach to improve confidence."
                )

            if len(user_prompt) > 2000:
                user_prompt = user_prompt[-2000:]

        if best_path is None:
            result = NodeResult(
                node_id=node.id,
                data=pd.DataFrame(),
                sql="",
                confidence=0.0,
                reasoning_paths=paths,
                cost_tokens=total_cost,
                error="No valid SQL generated",
            )
            result.ambiguity_score = 1.0
            return result

        if self.config.calibration_enabled:
            best_path.confidence = calibrate(best_path.confidence, signals)

        try:
            final_data = self.executor.execute(best_path.sql)
        except Exception:
            final_data = pd.DataFrame()

        result = NodeResult(
            node_id=node.id,
            data=final_data,
            sql=best_path.sql,
            confidence=best_path.confidence,
            reasoning_paths=paths,
            cost_tokens=total_cost,
        )
        result.ambiguity_score = compute_ambiguity_score(result)
        return result

    def _validate_sql(self, sql: str) -> str | None:
        from sqlglot import parse_one
        from sqlglot.errors import ParseError

        try:
            parse_one(sql)
            return None
        except ParseError as e:
            return str(e)

    def _validate_schema(self, sql: str) -> str | None:
        from sqlglot import parse_one
        from sqlglot.expressions import Column

        try:
            tree = parse_one(sql)
        except Exception:
            return None

        valid_columns = self._build_valid_columns()

        for col in tree.find_all(Column):
            col_name = col.name.lower()
            if col_name in valid_columns:
                continue
            suggestions = ", ".join(sorted(valid_columns - {"*"}))
            return f"Unknown column '{col.name}'. Available columns: {suggestions}"

        return None

    def _validate_column_selection(self, sql: str, node: TaskNode) -> str | None:
        from sqlglot import parse_one
        from sqlglot.expressions import Column

        expected: set[str] = set()
        if node.hint_columns:
            expected.update(c.lower() for c in node.hint_columns)
        if node.metric_columns:
            expected.update(c.lower() for c in node.metric_columns)
        if not expected:
            return None

        try:
            tree = parse_one(sql)
        except Exception:
            return None

        used = set()
        for col in tree.find_all(Column):
            name = col.name.lower()
            if name != "*" and not name.startswith("result_"):
                used.add(name)

        overlap = used & expected
        if not overlap and used:
            return (
                f"Column selection issue: SQL uses columns {sorted(used)} "
                f"but preferred columns are {sorted(expected)}. "
                f"Use at least one of the preferred columns."
            )
        return None

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

    def _diagnose_empty_result(self, sql: str, node: TaskNode) -> str | None:
        sql_upper = sql.upper()
        findings: list[str] = []

        if "WHERE" in sql_upper:
            if node.time_columns:
                date_cols = [c.lower() for c in node.time_columns]
                has_date_where = any(c in sql_upper for c in date_cols)
                if not has_date_where:
                    findings.append(
                        f"Question has time context but WHERE does not filter on date columns {node.time_columns}"
                    )

            if node.hint_columns:
                from sqlglot import parse_one
                from sqlglot.expressions import Column

                try:
                    tree = parse_one(sql)
                    used = {c.name.lower() for c in tree.find_all(Column)}
                    hinted = set(c.lower() for c in node.hint_columns)
                    if used and not (used & hinted):
                        findings.append(
                            f"SQL uses columns {used - {'*'}} but none match "
                            f"recommended columns {node.hint_columns}. "
                            f"Try using these columns instead."
                        )
                except Exception:
                    pass
        else:
            if node.time_columns:
                findings.append(
                    f"Question has time context but SQL lacks a WHERE clause. "
                    f"Add WHERE filtering on date columns {node.time_columns}."
                )

        has_aggregation = any(kw in sql_upper for kw in ["COUNT(", "SUM(", "AVG("])
        question_lower = node.description.lower()
        aggregate_keywords = ["total", "average", "sum", "count", "per", "by"]
        if any(kw in question_lower for kw in aggregate_keywords):
            if not has_aggregation:
                findings.append(
                    "Question asks for aggregation but SQL has no aggregate function. "
                    "Use SUM(), COUNT(), or AVG()."
                )
            elif "GROUP BY" not in sql_upper:
                findings.append(
                    "Aggregation used without GROUP BY. "
                    "Add GROUP BY for the dimension columns."
                )

        if findings:
            return "Diagnosis of 0-row result:\n- " + "\n- ".join(findings)
        return None

    def _check_semantic_match(
        self,
        data: pd.DataFrame,
        sql: str,
        node: TaskNode,
    ) -> str | None:
        result_cols = set(c.lower() for c in data.columns)

        if node.hint_columns:
            hinted = set(c.lower() for c in node.hint_columns)
            overlap = result_cols & hinted
            if not overlap:
                return (
                    f"Result columns {list(data.columns)} do not include "
                    f"expected hint columns {node.hint_columns}"
                )

        if node.metric_columns:
            metric = set(c.lower() for c in node.metric_columns)
            overlap = result_cols & metric
            if not overlap:
                return (
                    f"Result columns {list(data.columns)} do not include "
                    f"expected metric columns {node.metric_columns}"
                )

        sql_upper = sql.upper()
        has_group_by = "GROUP BY" in sql_upper
        has_aggregation = any(
            kw in sql_upper for kw in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]
        )
        question_lower = node.description.lower()

        if has_group_by and len(data) == 1:
            return (
                "Query uses GROUP BY but only returned 1 row. "
                "Expected multiple rows for grouped result."
            )

        time_keywords = ["year", "month", "quarter", "date", "trend", "recent",
                         "over time", "daily", "weekly", "monthly", "annually",
                         "since", "between", "from", "to", "last"]
        if any(kw in question_lower for kw in time_keywords):
            date_cols = node.time_columns or []
            if not date_cols:
                if self._compressed_schemas:
                    for s in self._compressed_schemas:
                        for c in s.columns:
                            if any(t in c.name.lower() for t in ["date", "time", "_at", "timestamp"]):
                                date_cols.append(c.name)
            if date_cols:
                has_date_where = any(
                    c.lower() in sql_upper for c in date_cols
                    if "WHERE" in sql_upper
                )
                if not has_date_where:
                    return (
                        f"Question mentions time ({[kw for kw in time_keywords if kw in question_lower][:3]}) "
                        f"but SQL has no WHERE condition on date columns {date_cols}. "
                        f"Add a date filter."
                    )

        aggregate_phrases = ["total", "average", "sum", "count", "per", "by",
                             "aggregate", "metric", "trend", "overall"]
        is_aggregate_question = any(
            p in question_lower for p in aggregate_phrases
        ) or bool(node.metric_columns)

        if is_aggregate_question and has_aggregation and not has_group_by:
            return (
                "Question asks for an aggregate/metric but SQL uses aggregation "
                "without GROUP BY. Add GROUP BY for the dimension columns."
            )

        if is_aggregate_question and not has_aggregation and not has_group_by:
            if len(data) > 3:
                return (
                    f"Question expects aggregation but query returned {len(data)} raw rows. "
                    f"Use aggregation functions (SUM, COUNT, AVG) with GROUP BY."
                )

        if not has_group_by and not has_aggregation and len(data) > 3:
            singular_phrases = ["what is", "what's", "how many", "how much", "total", "average"]
            if any(p in question_lower for p in singular_phrases):
                return (
                    f"Question seems to expect a single value "
                    f"but query returned {len(data)} rows. "
                    f"Add aggregation without GROUP BY or use LIMIT."
                )

        if any(p in sql_upper for p in ["valid_from", "valid_to"]):
            pass
        elif any(p in question_lower for p in ["as of", "point in time",
                                                "current", "snapshot"]):
            if self._compressed_schemas:
                has_scd2 = any(
                    any(c.name.lower() in ("valid_from", "valid_to") for c in s.columns)
                    for s in self._compressed_schemas
                )
                if has_scd2 and "valid_from" not in sql_upper:
                    return (
                        "SCD2 tables detected but SQL does not filter by "
                        "valid_from/valid_to. Add temporal filtering."
                    )

        return None

    def _expand_compressed_schemas(
        self,
        reason: str,
        data: pd.DataFrame,
    ) -> None:
        if not self.retriever:
            return
        question_words = set(re.findall(r"[a-zA-Z0-9_]\w*", reason.lower()))
        missing_col = None
        for phrase in ["column", "hint column", "expected", "missing"]:
            if phrase in reason.lower():
                for w in question_words:
                    if len(w) > 2 and w not in ("the", "not", "for", "with", "are"):
                        missing_col = w
                        break
        if missing_col and self.retriever:
            broad = self.retriever.score(missing_col)
            new_names = set(st.schema.name for st in broad.matched_tables[:3])
            if self._compressed_schemas:
                existing = set(s.name for s in self._compressed_schemas)
                missing_schemas = [s for s in self.all_schemas or []
                                   if s.name in new_names and s.name not in existing]
                if missing_schemas:
                    self._compressed_schemas.extend(missing_schemas)

    def _is_non_recoverable(self, error_msg: str, node: TaskNode) -> bool:
        """Determine if an execution error requires replanning vs simple retry."""
        # Table not found → replan (missing table in available schemas)
        if any(phrase in error_msg for phrase in
               ["no such table", "table not found", "doesn't exist",
                "table does not exist", "relation", "not found"]):
            return True
        # Join path impossible
        if "ambiguous column" in error_msg:
            return True
        # If hint_tables are available but the error references a missing table
        if node.hint_tables:
            for t in node.hint_tables:
                if t.lower() in error_msg and "not found" in error_msg:
                    return True
        return False

    @staticmethod
    def _build_replan_reason(error_msg: str, node: TaskNode) -> str:
        reason = f"Execution failed for node {node.id}: {error_msg[:200]}"
        if node.hint_tables:
            reason += f" | hint_tables: {node.hint_tables}"
        return reason

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
