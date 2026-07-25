from __future__ import annotations

import re
from dataclasses import dataclass, field

from syrch.core.models import ColumnSchema, ScoredSchemaEvidence, ScoredTable, TableSchema


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_]\w*", text.lower())
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "in", "on", "at", "of", "for", "to", "by", "with", "from",
        "and", "or", "not", "but", "what", "which", "who", "how",
        "show", "give", "find", "get", "list", "calculate", "compute",
        "do", "does", "did", "has", "have", "had",
    }
    return {w for w in words if w not in stopwords and len(w) > 1}


_GRAIN_KEYWORDS = {
    "year": "year", "month": "month", "quarter": "quarter", "daily": "day", "weekly": "week",
    "hourly": "hour", "date": "date", "time": "time",
    "category": "category", "segment": "segment", "region": "region",
    "department": "department", "store": "store", "supplier": "supplier",
    "top": "ranking", "bottom": "ranking", "rank": "ranking",
    "customer": "customer_entity", "product": "product_entity",
    "trend": "trend", "over_time": "trend", "history": "trend",
    "distribution": "distribution", "breakdown": "breakdown",
    "average": "average", "avg": "average", "mean": "average",
    "percentage": "percentage", "percent": "percentage", "ratio": "ratio", "rate": "rate",
}

_METRIC_KEYWORDS = {
    "revenue": "revenue", "sales": "revenue", "amount": "amount",
    "count": "count", "quantity": "quantity",
    "cost": "cost", "expense": "cost", "spend": "cost",
    "profit": "profit", "margin": "profit",
    "discount": "discount", "refund": "refund",
    "price": "price", "rate": "rate",
}


@dataclass
class SchemaIndex:
    tables: dict[str, TableSchema] = field(default_factory=dict)
    col_by_name: dict[str, list[tuple[str, ColumnSchema]]] = field(default_factory=dict)
    numeric_cols: dict[str, list[ColumnSchema]] = field(default_factory=dict)
    entity_keys: list[str] = field(default_factory=list)
    time_columns: dict[str, list[ColumnSchema]] = field(default_factory=dict)
    description_tokens: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, schemas: list[TableSchema]) -> SchemaIndex:
        tables = {}
        col_by_name: dict[str, list[tuple[str, ColumnSchema]]] = {}
        numeric_cols: dict[str, list[ColumnSchema]] = {}
        entity_keys: list[str] = []
        time_cols: dict[str, list[ColumnSchema]] = {}
        desc_tokens: dict[str, set[str]] = {}

        for s in schemas:
            tables[s.name] = s
            numeric_cols[s.name] = []
            time_cols[s.name] = []
            for c in s.columns:
                col_by_name.setdefault(c.name.lower(), []).append((s.name, c))
                if c.description:
                    tokens = _tokenize(c.description)
                    desc_tokens.setdefault(s.name, set()).update(tokens)
                ctype = c.type.upper()
                if ctype in ("INT", "INTEGER", "BIGINT", "REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC"):
                    numeric_cols[s.name].append(c)
                if ctype in ("DATE", "DATETIME", "TIMESTAMP", "TIMESTAMP_NTZ"):
                    time_cols[s.name].append(c)
                name_lower = c.name.lower()
                if name_lower.endswith("_id") or name_lower.endswith("_sk"):
                    entity_keys.append(c.name)

        return cls(
            tables=tables,
            col_by_name=col_by_name,
            numeric_cols=numeric_cols,
            entity_keys=sorted(set(entity_keys)),
            time_columns=time_cols,
            description_tokens=desc_tokens,
        )


class Retriever:
    def __init__(self, schemas: list[TableSchema]):
        self.index = SchemaIndex.build(schemas)
        self.all_schemas = schemas

    @classmethod
    def from_executor(cls, executor) -> Retriever:
        schemas = [executor.get_schema(t) for t in executor.list_tables()]
        return cls(schemas)

    def score(
        self,
        question: str,
        tables: list[str] | None = None,
    ) -> ScoredSchemaEvidence:
        keywords = _tokenize(question)

        target_schemas = (
            [s for s in self.all_schemas if s.name in tables]
            if tables
            else self.all_schemas
        )

        if not keywords:
            return ScoredSchemaEvidence(
                matched_tables=[
                    ScoredTable(schema=s, score=0.5, match_reasons=["no keywords extracted"])
                    for s in target_schemas
                ],
                all_schemas=self.all_schemas,
            )

        matched_tables: list[ScoredTable] = []
        matched_columns: set[str] = set()
        found_grain: set[str] = set()
        found_metrics: set[str] = set()
        found_time_cols: list[str] = []

        for schema in target_schemas:
            total = 0.0
            reasons: list[str] = []

            s, r = _table_name_score(keywords, schema.name)
            total += s
            reasons.extend(r)

            s, r, matched, metrics, time_cols = _column_score(
                keywords, schema.columns, self.index
            )
            total += s
            reasons.extend(r)
            matched_columns.update(matched)
            found_metrics.update(metrics)
            found_time_cols.extend(
                f"{schema.name}.{c}" for c in time_cols
            )

            s, r = _description_score(keywords, schema, self.index)
            total += s
            reasons.extend(r)

            matched_tables.append(ScoredTable(
                schema=schema,
                score=round(total, 2),
                match_reasons=reasons[:5],
            ))

        matched_tables.sort(key=lambda x: -x.score)

        for kw in keywords:
            grain = _GRAIN_KEYWORDS.get(kw)
            if grain:
                found_grain.add(grain)
            metric = _METRIC_KEYWORDS.get(kw)
            if metric:
                found_metrics.add(metric)

        return ScoredSchemaEvidence(
            matched_tables=matched_tables,
            matched_columns=[
                col for name, col in _resolve_matched_columns(
                    matched_columns, self.index
                )
            ],
            grain_hints=sorted(found_grain),
            metric_hints=sorted(found_metrics),
            time_columns=sorted(set(found_time_cols)),
            all_schemas=self.all_schemas,
        )


def _table_name_score(keywords: set[str], table_name: str) -> tuple[float, list[str]]:
    name_lower = table_name.lower()
    name_parts = set(name_lower.split("_"))
    matched = keywords & name_parts
    if matched:
        ratio = len(matched) / max(len(name_parts), 1)
        reasons = [f"table name matched '{m}' (boost +{ratio * 3.0:.1f})" for m in matched]
        return ratio * 3.0, reasons
    return 0.0, []


def _column_score(
    keywords: set[str],
    columns: list[ColumnSchema],
    index: SchemaIndex,
) -> tuple[float, list[str], set[str], set[str], set[str]]:
    col_name_parts: set[str] = set()
    col_full_names: set[str] = set()
    col_types: set[str] = set()
    col_desc_tokens: set[str] = set()

    for c in columns:
        lowered = c.name.lower()
        col_full_names.add(lowered)
        col_name_parts.update(lowered.split("_"))
        col_types.add(c.type.lower())
        if c.description:
            col_desc_tokens.update(_tokenize(c.description))

    reasons: list[str] = []
    score = 0.0
    matched_cols: set[str] = set()
    metrics: set[str] = set()
    time_cols: set[str] = set()

    exact_match = keywords & col_full_names
    score += len(exact_match) * 3.0
    for m in exact_match:
        matched_cols.add(m)
        reasons.append(f"column '{m}' exact match (boost +3.0)")

    for kw in keywords:
        for col in col_full_names:
            if kw in col.split("_"):
                score += 2.0
                matched_cols.add(col)
                reasons.append(f"column part '{kw}' in '{col}' (boost +2.0)")
                break

    type_match = keywords & col_types
    score += len(type_match) * 0.5
    for m in type_match:
        reasons.append(f"column type '{m}' matched (boost +0.5)")

    desc_match = keywords & col_desc_tokens
    score += len(desc_match) * 1.0
    for m in desc_match:
        reasons.append(f"column description matched '{m}' (boost +1.0)")

    for c in columns:
        if c.type.upper() in ("DATE", "DATETIME", "TIMESTAMP", "TIMESTAMP_NTZ"):
            time_cols.add(c.name)

    for kw in keywords:
        metric = _METRIC_KEYWORDS.get(kw)
        if metric:
            metrics.add(metric)

    return score, reasons, matched_cols, metrics, time_cols


def _description_score(
    keywords: set[str],
    schema: TableSchema,
    index: SchemaIndex,
) -> tuple[float, list[str]]:
    desc_tokens = index.description_tokens.get(schema.name, set())
    matched = keywords & desc_tokens
    if matched:
        reasons = [f"column description matched '{m}' (boost +1.0)" for m in matched]
        return len(matched) * 1.0, reasons
    return 0.0, []


def _resolve_matched_columns(
    col_names: set[str], index: SchemaIndex
) -> list[tuple[str, ColumnSchema]]:
    result: list[tuple[str, ColumnSchema]] = []
    seen: set[str] = set()
    for name in col_names:
        if name in index.col_by_name:
            for table, col in index.col_by_name[name]:
                key = f"{table}.{col.name}"
                if key not in seen:
                    seen.add(key)
                    result.append((table, col))
    return result
