from __future__ import annotations

import logging
from typing import Any

from sqlglot import parse_one
from sqlglot.expressions import Alias, Column, Table

from syrch.core.models import RequirementSpec, ValidationResult
from syrch.search.path_evaluator import _canonical_grain, infer_native_grain

logger = logging.getLogger(__name__)


class Validator:
    def validate(
        self,
        sql: str,
        requirements: RequirementSpec | None,
        valid_columns: set[str],
    ) -> ValidationResult:
        result = ValidationResult()
        if not sql:
            result.passed = False
            result.details.append("Empty SQL")
            return result

        try:
            tree = parse_one(sql)
        except Exception as e:
            result.passed = False
            result.details.append(f"SQL parse error: {e}")
            return result

        select_columns = self._get_select_columns(tree)
        sql_upper = sql.upper()
        sql_lower = sql.lower()

        if requirements is None:
            return result

        for metric in requirements.metrics:
            col_name = metric.lower()
            found = any(col_name in c.lower() for c in select_columns)
            if not found:
                found = any(col_name in valid_columns and col_name in sql for c in valid_columns)
            if not found:
                result.missing_metrics.append(metric)

        for sr in requirements.supporting_relations:
            if sr.table and sr.table.lower() not in sql_lower:
                result.missing_filters.append(
                    f"supporting_relation={sr.table} (purpose: {sr.purpose or 'required'})"
                )

        for vc in requirements.value_constraints:
            for val in vc.values:
                if f"'{val.lower()}'" not in sql_lower and f"{val.lower()}" not in sql_lower:
                    result.missing_filters.append(
                        f"value_constraint={vc.render()}"
                    )
                    break

        if requirements.time_range:
            has_filter = "WHERE" in sql_upper
            date_strs = [requirements.time_range[0], requirements.time_range[1]]
            has_date = any(d in sql_upper for d in date_strs if d)
            if not has_date and has_filter:
                for col in valid_columns:
                    if any(t in col.lower() for t in ["date", "time", "_at", "_ts"]):
                        if col.lower() in sql.lower():
                            has_date = True
                            break
            if not has_date and has_filter:
                for d in date_strs:
                    if not d or len(d) < 4:
                        continue
                    year = d[:4]
                    if (
                        year
                        and (
                            f"'{year}%" in sql
                            or f"= {year}" in sql
                            or f"{year}" in sql_upper
                        )
                    ):
                        has_date = True
                        break
            if not has_date:
                result.missing_filters.append(f"time_range={requirements.time_range}")

        if requirements.aggregation:
            agg_funcs = {"sum", "count", "avg", "min", "max"}
            sql_lower = sql.lower()
            agg_keywords = {f"{f}(" for f in agg_funcs}
            has_agg = any(kw in sql_lower for kw in agg_keywords)
            if not has_agg:
                grain = (requirements.grain or "").lower()
                pre_aggregated_grain = grain in {"monthly", "daily", "weekly", "quarterly"}
                select_text = " ".join(select_columns).lower()
                metrics_present = bool(requirements.metrics) and all(
                    m.lower() in select_text for m in requirements.metrics
                )
                if pre_aggregated_grain and metrics_present:
                    pass
                else:
                    result.aggregation_ok = False
                    result.details.append(f"Missing aggregation: {requirements.aggregation}")

        if requirements.grain and requirements.grain != "total":
            if "GROUP BY" in sql_upper:
                result.grain_mismatch = False
            else:
                req_canon = _canonical_grain(requirements.grain)
                collapse_kws = {f"{f}(" for f in ("sum", "count", "avg", "min", "max")}
                collapses_rows = any(kw in sql_lower for kw in collapse_kws)
                native_at_grain = (
                    req_canon is not None
                    and not collapses_rows
                    and any(
                        infer_native_grain(t) == req_canon
                        for t in self._from_table_names(tree)
                    )
                )
                result.grain_mismatch = not native_at_grain

        result.passed = (
            len(result.missing_metrics) == 0
            and len(result.missing_filters) == 0
            and result.aggregation_ok
        )
        if not result.passed:
            if result.missing_metrics:
                result.details.append(f"Missing metrics: {result.missing_metrics}")
            if result.missing_filters:
                result.details.append(f"Missing filters: {result.missing_filters}")
            if not result.aggregation_ok:
                result.details.append(f"Missing {requirements.aggregation} aggregation")

        return result

    @staticmethod
    def _from_table_names(tree: Any) -> set[str]:
        """Physical FROM/JOIN tables referenced by the parsed query."""
        return {t.name for t in tree.find_all(Table)}

    @staticmethod
    def _get_select_columns(tree: Any) -> list[str]:
        cols: list[str] = []
        try:
            for node in tree.find_all(Column):
                name = node.name
                if name != "*":
                    cols.append(name)
            for alias in tree.find_all(Alias):
                raw = alias.alias
                if isinstance(raw, str):
                    cols.append(raw)
                else:
                    cols.append(getattr(raw, "name", ""))
        except Exception:
            pass
        return list(set(cols))
