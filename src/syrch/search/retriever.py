from __future__ import annotations

import re

from syrch.core.models import ScoredTable, TableSchema


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


def _table_name_score(keywords: set[str], table_name: str) -> tuple[float, list[str]]:
    name_lower = table_name.lower()
    name_parts = set(name_lower.split("_"))
    matched = keywords & name_parts
    if matched:
        ratio = len(matched) / max(len(name_parts), 1)
        reasons = [f"table name matched '{m}' (boost +{ratio * 3.0:.1f})" for m in matched]
        return ratio * 3.0, reasons
    return 0.0, []


def _column_score(keywords: set[str], columns: list) -> tuple[float, list[str]]:
    col_name_parts: set[str] = set()
    col_full_names: set[str] = set()
    col_types: set[str] = set()
    for c in columns:
        if hasattr(c, "name"):
            lowered = c.name.lower()
            col_full_names.add(lowered)
            col_name_parts.update(lowered.split("_"))
        if hasattr(c, "type"):
            col_types.add(c.type.lower())

    reasons = []
    score = 0.0

    # exact column name match (e.g., keyword "total_revenue" → column "total_revenue")
    exact_match = keywords & col_full_names
    score += len(exact_match) * 3.0
    for m in exact_match:
        reasons.append(f"column '{m}' exact match (boost +3.0)")

    # partial match: keyword appears as part of column name (e.g., "revenue" → "total_revenue")
    for kw in keywords:
        for col in col_full_names:
            if kw in col.split("_"):
                score += 2.0
                reasons.append(f"column part '{kw}' in '{col}' (boost +2.0)")
                break

    # column type match
    type_match = keywords & col_types
    score += len(type_match) * 0.5
    for m in type_match:
        reasons.append(f"column type '{m}' matched (boost +0.5)")

    return score, reasons


def _description_score(keywords: set[str], description: str) -> tuple[float, list[str]]:
    if not description:
        return 0.0, []
    desc_lower = description.lower()
    desc_words = _tokenize(desc_lower)
    matched = keywords & desc_words
    if matched:
        reasons = [f"description matched '{m}' (boost +1.0)" for m in matched]
        return len(matched) * 1.0, reasons
    return 0.0, []


class Retriever:
    def score(
        self,
        question: str,
        schemas: list[TableSchema],
    ) -> list[ScoredTable]:
        keywords = _tokenize(question)
        if not keywords:
            return [ScoredTable(schema=s, score=0.5, match_reasons=["no keywords extracted"]) for s in schemas]

        results: list[ScoredTable] = []
        for schema in schemas:
            total = 0.0
            all_reasons: list[str] = []

            s, r = _table_name_score(keywords, schema.name)
            total += s
            all_reasons.extend(r)

            s, r = _column_score(keywords, schema.columns)
            total += s
            all_reasons.extend(r)

            if hasattr(schema, "description"):
                s, r = _description_score(keywords, schema.description)
                total += s
                all_reasons.extend(r)

            results.append(ScoredTable(
                schema=schema,
                score=round(total, 2),
                match_reasons=all_reasons[:5],
            ))

        results.sort(key=lambda x: -x.score)
        return results
