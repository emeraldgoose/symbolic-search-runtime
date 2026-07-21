from __future__ import annotations

import pandas as pd

from syrch.core.config import ExecutionConfig
from syrch.core.models import NodeResult, ReasoningPath, TaskNode
from syrch.executors.base import BaseExecutor
from syrch.llm.base import BaseLLM
from syrch.search.calibrator import ExecutionSignals, calibrate
from syrch.search.clarify import compute_ambiguity_score

RLM_SYSTEM = """You are a search agent working inside a SQL query environment.
You have access to:
- A database with the following tables:
{schema}
- The executor to run SQL queries
- Previous sub-task results as variables (e.g., result_A, result_B, etc.)

Your task:
{task_description}

{hint_section}
For each attempt, produce a SQL query that:
1. Constructs a SQL query addressing the task
2. Stores the result in a variable

After each attempt, output a confidence score between 0.0 and 1.0.

When you are satisfied, output FINAL(result_var_name) to submit your answer.

IMPORTANT: You MUST end your response with "Confidence: <0.0-1.0>" on its own line.
Do NOT forget the confidence line. A confidence of 1.0 means absolutely certain.
"""


class RLMAgent:
    MAX_ROWS_WARNING = 1000

    def __init__(self, llm: BaseLLM, executor: BaseExecutor, config: ExecutionConfig):
        self.llm = llm
        self.executor = executor
        self.config = config
        self._compressed_schemas: list | None = None

    def set_compressed_schemas(self, schemas: list | None) -> None:
        self._compressed_schemas = schemas

    def _build_schema_str(self) -> str:
        if self._compressed_schemas is not None:
            parts: list[str] = []
            for s in self._compressed_schemas:
                cols = ", ".join(f"{c.name} ({c.type})" for c in s.columns)
                parts.append(f"Table: {s.name}\nColumns: {cols}")
            return "\n\n".join(parts) if parts else "No tables available."
        tables = self.executor.list_tables()
        parts = []
        for t in tables:
            schema = self.executor.get_schema(t)
            cols = ", ".join(f"{c.name} ({c.type})" for c in schema.columns)
            parts.append(f"Table: {schema.name}\nColumns: {cols}")
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

    @staticmethod
    def _build_hint_section(node: TaskNode) -> str:
        parts: list[str] = []
        if node.hint_tables:
            parts.append("Recommended tables (strongly prefer these): " + ", ".join(node.hint_tables))
        if node.hint_columns:
            parts.append("Planner hint — columns you may need (verify actual names): " + ", ".join(node.hint_columns))
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
                if "all NULL" in quality_feedback:
                    signals.had_null_columns = True
                if f"{self.MAX_ROWS_WARNING}" in quality_feedback:
                    signals.had_overflow_result = True
                user_prompt = f"{quality_feedback}\n\nTry a different SQL approach."
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
