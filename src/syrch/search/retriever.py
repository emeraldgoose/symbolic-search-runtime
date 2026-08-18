from __future__ import annotations

import re
from dataclasses import dataclass, field

from syrch.core.models import (
    CandidateFeatures,
    CandidatePolicy,
    ColumnSchema,
    ScoredSchemaEvidence,
    ScoredTable,
    TableSchema,
    base_table_name,
)
from syrch.search.semantic_index import SemanticIndex


def _tokenize(text: str) -> set[str]:
    raw_words = re.findall(r"[a-zA-Z0-9]+(?:[_\-][a-zA-Z0-9]+)*", text.lower())
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "in", "on", "at", "of", "for", "to", "by", "with", "from",
        "and", "or", "not", "but", "what", "which", "who", "how",
        "show", "give", "find", "get", "list", "calculate", "compute",
        "do", "does", "did", "has", "have", "had",
    }
    words: set[str] = set()
    for w in raw_words:
        parts = re.split(r"[_\-]", w)
        for part in parts:
            if part not in stopwords and len(part) > 1:
                words.add(part)
        if w not in stopwords and len(w) > 1:
            words.add(w)
    return words


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
    "inventory": "inventory", "stock": "inventory",
}

_COLUMN_SYNONYMS: dict[str, set[str]] = {
    "inventory": {"items", "stock"},
    "payment": {"paid"},
    "delivery": {"shipped"},
    "fulfillment": {"shipped", "paid", "delivery"},
    "salary": {"compensation"},
    "employee": {"staff"},
    "churn": {"inactive"},
}


def _stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ves") and len(word) > 4:
        return word[:-3] + "f"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word

_GRAIN_TO_LAYER: dict[str, str] = {
    "daily": "mart", "monthly": "mart", "weekly": "mart", "trend": "mart",
    "day": "mart", "month": "mart", "week": "mart", "year": "mart",
    "quarter": "mart", "hour": "mart",
    "order": "dw", "customer": "dw", "product": "dw", "employee": "dw",
    "event": "fact", "session": "fact", "click": "fact", "impression": "fact",
    "category": "dim", "segment": "dim", "region": "dim",
    "ranking": "rpt",
}

_LAYER_PENALTY: dict[str, float] = {
    "staging": -0.5, "legacy": -0.5,
    "config": -1.0, "audit": -1.0, "archive": -0.5,
}


@dataclass
class SchemaIndex:
    tables: dict[str, TableSchema] = field(default_factory=dict)
    col_by_name: dict[str, list[tuple[str, ColumnSchema]]] = field(default_factory=dict)
    numeric_cols: dict[str, list[ColumnSchema]] = field(default_factory=dict)
    entity_keys: list[str] = field(default_factory=list)
    time_columns: dict[str, list[ColumnSchema]] = field(default_factory=dict)
    description_tokens: dict[str, set[str]] = field(default_factory=dict)
    table_to_layer: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, schemas: list[TableSchema]) -> SchemaIndex:
        tables = {}
        col_by_name: dict[str, list[tuple[str, ColumnSchema]]] = {}
        numeric_cols: dict[str, list[ColumnSchema]] = {}
        entity_keys: list[str] = []
        time_cols: dict[str, list[ColumnSchema]] = {}
        desc_tokens: dict[str, set[str]] = {}
        table_to_layer: dict[str, str] = {}

        for s in schemas:
            tables[s.name] = s
            table_to_layer[s.name] = s.layer
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
            table_to_layer=table_to_layer,
        )


class Retriever:
    def __init__(
        self,
        schemas: list[TableSchema],
        policy: CandidatePolicy | None = None,
        semantic_index: SemanticIndex | None = None,
    ):
        self.index = SchemaIndex.build(schemas)
        self.all_schemas = schemas
        self.policy = policy or CandidatePolicy()
        self.semantic_index = semantic_index

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
                candidates=[
                    ScoredTable(schema=s, score=0.5, match_reasons=["no keywords extracted"])
                    for s in target_schemas
                ],
                all_schemas=self.all_schemas,
            )

        candidates: list[ScoredTable] = []
        matched_columns_set: set[str] = set()
        found_metrics: set[str] = set()
        found_time_cols: list[str] = []

        grain_hints: set[str] = set()
        for kw in keywords:
            grain = _GRAIN_KEYWORDS.get(kw)
            if grain:
                grain_hints.add(grain)
            metric = _METRIC_KEYWORDS.get(kw)
            if metric:
                found_metrics.add(metric)

        for schema in target_schemas:
            total = 0.0
            reasons: list[str] = []

            s, r = _table_name_score(keywords, schema.name)
            total += s
            reasons.extend(r)

            s, r, matched_col_names, metrics, time_cols = _column_score(
                keywords, schema.columns, self.index
            )
            total += s
            reasons.extend(r)
            matched_columns_set.update(matched_col_names)
            found_time_cols.extend(f"{schema.name}.{c}" for c in time_cols)

            if self.semantic_index and not self.semantic_index.is_empty:
                s, r, semantic_cols = _semantic_score(keywords, schema, self.semantic_index)
                total += s
                reasons.extend(r)
                matched_columns_set.update(semantic_cols)

            s, r = _description_score(keywords, schema, self.index)
            total += s
            reasons.extend(r)

            features = _build_features(keywords, schema, matched_col_names, metrics, grain_hints)
            total = _apply_layer_adjustment(total, schema.layer, grain_hints, reasons)

            candidates.append(ScoredTable(
                schema=schema,
                score=round(total, 2),
                match_reasons=reasons[:5],
                features=features,
            ))

        candidates.sort(key=lambda x: -x.score)
        candidates = self.policy.filter(candidates)

        alias_map = _build_alias_map(keywords, candidates)

        return ScoredSchemaEvidence(
            candidates=candidates,
            matched_columns=[
                col for name, col in _resolve_matched_columns(
                    matched_columns_set, self.index
                )
            ],
            grain_hints=sorted(grain_hints),
            metric_hints=sorted(found_metrics),
            time_columns=sorted(set(found_time_cols)),
            alias_map=alias_map,
            all_schemas=self.all_schemas,
        )


def _table_name_score(keywords: set[str], table_name: str) -> tuple[float, list[str]]:
    name_lower = base_table_name(table_name).lower()
    name_parts = set(name_lower.split("_"))
    matched = keywords & name_parts
    if not matched:
        matched = {k for k in keywords if _stem(k) in name_parts}
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
            if _stem(kw) in col.split("_"):
                score += 2.0
                matched_cols.add(col)
                reasons.append(f"column part '{_stem(kw)}' (stem of '{kw}') in '{col}' (boost +2.0)")
                break
        else:
            synonyms = _COLUMN_SYNONYMS.get(kw, set())
            if synonyms:
                for col in col_full_names:
                    col_parts = set(col.split("_"))
                    if synonyms & col_parts:
                        score += 1.0
                        matched_cols.add(col)
                        reasons.append(f"synonym '{kw}'→{synonyms} matched '{col}' (boost +1.0)")
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


def _build_features(
    keywords: set[str],
    schema: TableSchema,
    matched_col_names: set[str],
    metrics: set[str],
    grain_hints: set[str],
) -> CandidateFeatures:
    matched_kw: list[str] = []
    matched_cols_list: list[str] = []
    matched_descs: list[str] = []

    for kw in keywords:
        if any(kw in col.lower() for col in matched_col_names):
            matched_kw.append(kw)

    for col_name in matched_col_names:
        matched_cols_list.append(col_name)
        for c in schema.columns:
            if c.name.lower() == col_name and c.description:
                matched_descs.append(c.description)

    for c in schema.columns:
        name_lower = c.name.lower()
        if any(kw in name_lower for kw in keywords):
            if c.description and any(kw in c.description.lower() for kw in keywords):
                matched_descs.append(c.description)

    rep_columns: list[str] = []
    seen = set()
    for col in matched_cols_list:
        if col not in seen:
            rep_columns.append(col)
            seen.add(col)
    for c in schema.columns:
        if c.name.lower() not in seen:
            rep_columns.append(c.name.lower())
            seen.add(c.name.lower())
            if len(rep_columns) >= 5:
                break

    return CandidateFeatures(
        matched_keywords=matched_kw[:8],
        matched_columns=matched_cols_list[:8],
        matched_descriptions=list(set(matched_descs))[:3],
        matched_metrics=list(metrics)[:5],
        matched_grains=list(grain_hints)[:3],
        representative_columns=rep_columns[:5],
        layer=schema.layer,
    )


def _apply_layer_adjustment(
    score: float, layer: str, grain_hints: set[str], reasons: list[str]
) -> float:
    layer_lower = layer.lower()

    base_penalty = _LAYER_PENALTY.get(layer_lower, 0.0)
    if base_penalty:
        score += base_penalty
        reasons.append(f"layer '{layer}' penalty ({base_penalty:+.1f})")

    for hint in grain_hints:
        preferred_layer = _GRAIN_TO_LAYER.get(hint)
        if preferred_layer == layer_lower:
            score += 0.3
            reasons.append(f"grain '{hint}' matches layer '{layer}' (boost +0.3)")

    return score


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


def _semantic_score(
    keywords: set[str],
    schema: TableSchema,
    semantic_index: SemanticIndex,
) -> tuple[float, list[str], set[str]]:
    score = 0.0
    reasons: list[str] = []
    matched: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    for kw in keywords:
        hits = semantic_index.lookup(kw)
        for tbl, col in hits:
            if tbl != schema.name:
                continue
            pair = (tbl, col)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            score += 2.0
            matched.add(col)
            reasons.append(f"semantic index '{kw}'→'{col}' (boost +2.0)")

    return score, reasons, matched


def _build_alias_map(
    keywords: set[str],
    candidates: list[ScoredTable],
) -> dict[str, list[tuple[str, str, str | None]]]:
    alias_map: dict[str, list[tuple[str, str, str | None]]] = {}

    for kw in keywords:
        category = _METRIC_KEYWORDS.get(kw)
        if not category:
            continue

        for st in candidates:
            for col in st.schema.columns:
                name_lower = col.name.lower()
                parts = set(name_lower.split("_"))

                is_numeric = col.type.upper() in (
                    "INT", "INTEGER", "BIGINT", "REAL", "FLOAT",
                    "DOUBLE", "DECIMAL", "NUMERIC",
                )

                if kw in parts or category in parts:
                    agg = "SUM" if is_numeric else None
                    entry = (col.name, st.schema.name, agg)
                    if entry not in alias_map.setdefault(kw, []):
                        alias_map[kw].append(entry)
                    continue

                synonyms = _COLUMN_SYNONYMS.get(kw, set())
                if synonyms & parts:
                    agg = "SUM" if is_numeric else None
                    entry = (col.name, st.schema.name, agg)
                    if entry not in alias_map.setdefault(kw, []):
                        alias_map[kw].append(entry)
                    continue

                if col.description and kw in col.description.lower():
                    agg = "SUM" if is_numeric else None
                    entry = (col.name, st.schema.name, agg)
                    if entry not in alias_map.setdefault(kw, []):
                        alias_map[kw].append(entry)

    return alias_map
