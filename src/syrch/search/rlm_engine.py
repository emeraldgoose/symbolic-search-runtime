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
- A database with the following schema:
{schema}
- The executor to run SQL queries
- Previous sub-task results as variables (e.g., result_A, result_B, etc.)

Your task:
{task_description}

For each attempt, produce a SQL query that:
1. Constructs a SQL query addressing the task
2. Stores the result in a variable

After each attempt, output a confidence score between 0.0 and 1.0.

When you are satisfied, output FINAL(result_var_name) to submit your answer.
"""


class RLMAgent:
    MAX_ROWS_WARNING = 1000

    def __init__(self, llm: BaseLLM, executor: BaseExecutor, config: ExecutionConfig):
        self.llm = llm
        self.executor = executor
        self.config = config

    def _build_schema_str(self) -> str:
        tables = self.executor.list_tables()
        parts: list[str] = []
        for t in tables:
            schema = self.executor.get_schema(t)
            cols = ", ".join(f"{c.name} ({c.type})" for c in schema.columns)
            parts.append(f"Table: {schema.name}\nColumns: {cols}")
        return "\n\n".join(parts)

    def _build_valid_columns(self) -> set[str]:
        cols: set[str] = set()
        for t in self.executor.list_tables():
            schema = self.executor.get_schema(t)
            cols.update(c.name.lower() for c in schema.columns)
        cols.add("*")
        return cols

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

        context_vars = ""
        if context:
            for nid, res in context.items():
                if res.data is not None and not res.data.empty:
                    preview = res.data.head(3).to_string()
                    context_vars += f"\nresult_{nid} =\n{preview}\n"

        system = RLM_SYSTEM.format(
            schema=schema_str,
            task_description=node.description,
        )
        user_prompt = (
            f"Task: {node.description}\n"
            f"Expected output: {node.expected_output_desc}\n"
        )
        if context_vars:
            user_prompt += f"\nAvailable results from dependencies:\n{context_vars}\n"
        user_prompt += "\nGenerate a SQL query and confidence score."

        for attempt in range(self.config.max_attempts_per_node):
            signals.num_attempts = attempt + 1
            response = self.llm.generate(system, user_prompt)
            sql = self._extract_sql(response.content)
            confidence = self._extract_confidence(response.content)
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

    def _extract_confidence(self, content: str) -> float:
        import re

        match = re.search(r"confidence[:\s]+(\d+\.?\d*)", content, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if 0.0 <= val <= 1.0:
                return val
        return 0.5
