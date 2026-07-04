from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "schema"


@dataclass
class ColumnDef:
    name: str
    type: str
    nullable: bool = True
    pk: bool = False
    fk: str | None = None
    comment: str | None = None
    values: list[str] | None = None
    distribution: str = "uniform"
    dist_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableDef:
    name: str
    columns: list[ColumnDef]
    comment: str | None = None
    rows: int = 10000
    cluster_by: list[str] | None = None
    layer: str = "dw"
    grain: str = ""
    scd_columns: list[str] | None = None
    pk: str | None = None
    fks: list[dict[str, str]] = field(default_factory=list)


def _parse_type(raw: str | dict) -> str:
    if isinstance(raw, str):
        return raw
    return raw.get("type", "TEXT")


def _parse_nullable(raw: str | dict) -> bool:
    if isinstance(raw, dict):
        return raw.get("nullable", True)
    return True


def _parse_pk(raw: str | dict) -> bool:
    if isinstance(raw, dict):
        return raw.get("pk", False)
    return False


def _parse_fk(raw: str | dict) -> str | None:
    if isinstance(raw, dict):
        return raw.get("fk", None)
    return None


def load_schema_yaml(path: str | Path) -> list[TableDef]:
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    tables: list[TableDef] = []
    for table_name, tdef in raw.get("tables", {}).items():
        cols = []
        for col_name, cdef in tdef.get("columns", {}).items():
            cols.append(
                ColumnDef(
                    name=col_name,
                    type=_parse_type(cdef),
                    nullable=_parse_nullable(cdef),
                    pk=_parse_pk(cdef),
                    fk=_parse_fk(cdef),
                    comment=cdef.get("comment") if isinstance(cdef, dict) else None,
                    values=cdef.get("values") if isinstance(cdef, dict) else None,
                    distribution=cdef.get("distribution", "uniform") if isinstance(cdef, dict) else "uniform",
                    dist_params=cdef.get("dist_params", {}) if isinstance(cdef, dict) else {},
                )
            )
        tables.append(
            TableDef(
                name=table_name,
                columns=cols,
                comment=tdef.get("comment"),
                rows=tdef.get("distribution", {}).get("rows", 10000),
                cluster_by=tdef.get("cluster_by"),
                layer=tdef.get("layer", "dw"),
                grain=tdef.get("grain", ""),
                scd_columns=tdef.get("scd_columns"),
                pk=tdef.get("pk"),
                fks=tdef.get("fks", []),
            )
        )
    return tables


def load_all_schemas(schema_dir: str | Path | None = None) -> list[TableDef]:
    if schema_dir is None:
        schema_dir = _SCHEMA_DIR
    path = Path(schema_dir)
    all_tables: list[TableDef] = []
    for yaml_file in sorted(path.glob("*.yaml")):
        all_tables.extend(load_schema_yaml(yaml_file))
    return all_tables
