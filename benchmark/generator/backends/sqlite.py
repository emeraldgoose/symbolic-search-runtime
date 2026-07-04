from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from benchmark.generator.backends.base import Backend
from benchmark.generator.core.schema import ColumnDef, TableDef

_TYPE_MAP = {
    "INTEGER": "INTEGER",
    "INT": "INTEGER",
    "BIGINT": "INTEGER",
    "LONG": "INTEGER",
    "REAL": "REAL",
    "FLOAT": "REAL",
    "DOUBLE": "REAL",
    "DECIMAL": "REAL",
    "TEXT": "TEXT",
    "DATE": "TEXT",
    "DATETIME": "TEXT",
    "TIMESTAMP": "TEXT",
    "TIMESTAMP_NTZ": "TEXT",
    "BOOLEAN": "INTEGER",
    "BOOL": "INTEGER",
}


def _to_sql_type(t: str) -> str:
    return _TYPE_MAP.get(t.upper(), "TEXT")


def _ddl(table: TableDef) -> str:
    col_defs: list[str] = []
    for c in table.columns:
        sql_type = _to_sql_type(c.type)
        nullable = "" if c.pk else (" NOT NULL" if not c.nullable else "")
        col_defs.append(f"  {c.name} {sql_type}{nullable}")
    fks: list[str] = []
    for c in table.columns:
        if c.fk:
            ref_table, ref_col = c.fk.split(".", 1)
            fks.append(f"  FOREIGN KEY ({c.name}) REFERENCES {ref_table}({ref_col})")
    ddl_parts = ["CREATE TABLE IF NOT EXISTS", table.name, "("]
    ddl_parts.append(",\n".join(col_defs))
    if fks:
        ddl_parts.append(",\n" + ",\n".join(fks))
    ddl_parts.append("\n)")
    return " ".join(ddl_parts)


class SQLiteBackend(Backend):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA synchronous = OFF")
        self._conn.execute("PRAGMA journal_mode = MEMORY")
        self._conn.execute("PRAGMA foreign_keys = OFF")

    def create_table(self, table: TableDef) -> None:
        if table.comment:
            self._conn.execute(f"-- {table.comment}")
        self._conn.execute(_ddl(table))
        self._conn.commit()

    def write_batch(self, table: str, df: pd.DataFrame) -> None:
        df.to_sql(table, self._conn, if_exists="append", index=False)

    def finalize(self, table: str) -> None:
        self._conn.execute(f"ANALYZE {table}")
        self._conn.commit()

    def verify(self) -> dict[str, dict[str, int | float]]:
        tables = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        results: dict[str, dict[str, int | float]] = {}
        for (tname,) in tables:
            cnt = self._conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
            sum_val: int | float = 0
            numeric_cols = self._conn.execute(
                f"PRAGMA table_info({tname})"
            ).fetchall()
            for col in numeric_cols:
                cname = col[1]
                ctype = col[2].upper()
                if ctype in ("REAL", "FLOAT", "DOUBLE", "INTEGER", "INT", "BIGINT"):
                    try:
                        val = self._conn.execute(
                            f"SELECT COALESCE(SUM({cname}), 0) FROM {tname}"
                        ).fetchone()[0]
                        sum_val += val if val else 0
                    except sqlite3.OperationalError:
                        pass
            results[tname] = {"rows": cnt, "checksum": sum_val}
        return results

    def close(self) -> None:
        self._conn.close()

    def verify_cross_backend(self, other: dict[str, dict[str, int | float]]) -> dict[str, bool]:
        self_stats = self.verify()
        result: dict[str, bool] = {}
        all_tables = set(self_stats) | set(other)
        for t in all_tables:
            s = self_stats.get(t, {})
            o = other.get(t, {})
            rows_match = s.get("rows") == o.get("rows")
            sum_match = s.get("checksum") == o.get("checksum")
            result[t] = rows_match and sum_match
        return result
