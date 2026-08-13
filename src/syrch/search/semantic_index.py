from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from syrch.core.models import ColumnSchema, TableSchema
from syrch.llm.base import BaseLLM

SEMANTIC_INDEX_PATH = Path.home() / ".syrch" / "semantic_index.json"

_BUILD_PROMPT = """You are a SQL schema analyst. Given a table and column,
list the business concepts this column represents.

Be precise: distinguish revenue from profit, cost from expense, etc.

Return comma-separated terms only, no explanation or numbering.

Table: {table}
Column: {column} ({col_type}, nullable={nullable})
Description: {description}

Business concepts:"""


def _build_column_tags(llm: BaseLLM, table: str, column: ColumnSchema) -> list[str]:
    desc = column.description or "(none)"
    prompt = _BUILD_PROMPT.format(
        table=table,
        column=column.name,
        col_type=column.type,
        nullable=column.nullable,
        description=desc,
    )
    resp = llm.generate(
        "You are a precise SQL schema analyst. Return comma-separated business terms only.",
        prompt,
    )
    text = resp.content.strip()
    tags = [t.strip().lower() for t in text.split(",") if t.strip()]
    return tags[:10]


class SemanticIndex:
    def __init__(self, data: dict[str, list[tuple[str, str]]] | None = None):
        self._data: dict[str, list[tuple[str, str]]] = data or {}

    @classmethod
    def build(
        cls,
        schemas: list[TableSchema],
        llm: BaseLLM,
        progress_callback: Any | None = None,
    ) -> SemanticIndex:
        data: dict[str, list[tuple[str, str]]] = {}
        for schema in schemas:
            for col in schema.columns:
                tags = _build_column_tags(llm, schema.name, col)
                for tag in tags:
                    data.setdefault(tag, []).append((schema.name, col.name))
        return cls(data)

    @classmethod
    def load(cls, path: Path = SEMANTIC_INDEX_PATH) -> SemanticIndex:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(data)
        return cls()

    def save(self, path: Path = SEMANTIC_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._data, f, indent=2)

    def lookup(self, keyword: str) -> list[tuple[str, str]]:
        kl = keyword.lower()
        return self._data.get(kl, [])

    @property
    def is_empty(self) -> bool:
        return not self._data
