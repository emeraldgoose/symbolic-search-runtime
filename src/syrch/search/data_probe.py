from __future__ import annotations

import logging
from dataclasses import dataclass

from syrch.executors.base import BaseExecutor

logger = logging.getLogger(__name__)

_PROBE_LIMIT = 8
_MAX_PROBE_FACTS = 3


@dataclass
class ProbeResult:
    """A verified data-level fact: does `value` exist in `table.column`.

    This is evidence about the database, not about any single node. It is RLM
    reasoning input only — it never enters selection score (same contract as
    ValueConstraint in models.py).
    """

    table: str
    column: str
    value: str
    exists: bool
    count: int = 0
    source: str = "sql-filter"

    def render(self) -> str:
        state = "EXISTS" if self.exists else "NOT FOUND"
        suffix = f" (count={self.count})" if self.exists else ""
        return f"value '{self.value}' {state} in {self.table}.{self.column}{suffix}"


class ProbeRegistry:
    """Run-wide shared cache of verified data facts.

    Probe results are database truths, not node state — every node that needs
    the same (table, column, value) fact reads the same answer, so probes run
    at most once per run (the executor's SQL cache provides a second layer).
    Keyed by (db_id, table, column, value) so two databases never share facts.
    """

    def __init__(self) -> None:
        self._facts: dict[tuple[str, str, str, str], ProbeResult] = {}

    def get(
        self, db_id: str, table: str, column: str, value: str
    ) -> ProbeResult | None:
        return self._facts.get((db_id, table, column, value))

    def put(self, db_id: str, fact: ProbeResult) -> None:
        self._facts[(db_id, fact.table, fact.column, fact.value)] = fact

    def all(self) -> list[ProbeResult]:
        return list(self._facts.values())


class DataProbe:
    """Probes candidate-pool tables for whether a filter literal actually
    exists in a column.

    The probed value always comes from the failing SQL's own literals (the
    model already used it), never discovered by scanning data — so the probe
    is bounded to `|pool| x |filter literals| indexed equality checks and
    never introduces values the model could copy into SQL.
    """

    def __init__(self, executor: BaseExecutor, registry: ProbeRegistry | None = None):
        self.executor = executor
        self.registry = registry or ProbeRegistry()

    @property
    def _db_id(self) -> str:
        return getattr(self.executor, "db_id", type(self.executor).__name__)

    def probe(self, table: str, column: str, value: str) -> ProbeResult:
        cached = self.registry.get(self._db_id, table, column, value)
        if cached is not None:
            return cached
        lit = self._literal(value)
        sql = (
            f"SELECT COUNT(*) AS n FROM {table} "
            f"WHERE {column} = {lit}"
        )
        try:
            df = self.executor.execute(sql)
            count = int(df.iloc[0, 0]) if not df.empty else 0
        except Exception as e:
            logger.debug("probe failed table=%s col=%s val=%s: %s", table, column, value, e)
            count = 0
        fact = ProbeResult(
            table=table,
            column=column,
            value=value,
            exists=count > 0,
            count=count,
        )
        self.registry.put(self._db_id, fact)
        return fact

    def probe_sql_filters(
        self,
        sql: str,
        candidate_tables: list[str],
        max_tables: int = _PROBE_LIMIT,
    ) -> list[ProbeResult]:
        """Extract `column = 'literal'` predicates from a failing SQL and probe
        which candidate-pool table actually holds each value.

        Only the FIRST `max_tables` candidates are probed (the pool is already
        retriever-bounded; probing all physical tables would be combinatorial).
        """
        filters = self._extract_eq_filters(sql)
        if not filters:
            return []
        facts: list[ProbeResult] = []
        tables = candidate_tables[:max_tables]
        for column, value in filters:
            for table in tables:
                facts.append(self.probe(table, column, value))
        return facts

    def render_facts(
        self,
        facts: list[ProbeResult],
        anchored_tables: list[str] | None = None,
    ) -> str:
        """Render VERIFIED DATA FACTS for RLM feedback.

        Positive facts (value exists somewhere) are always surfaced. A negative
        fact is surfaced ONLY for a table the failing SQL anchored on (FROM/JOIN)
        AND only when the same (column, value) EXISTS in another probed table —
        that separates "wrong table" (value lives elsewhere) from "wrong
        literal" (value absent everywhere, already covered by the generic
        empty-result message). Bounded to the most relevant hits.
        """
        positive = [f for f in facts if f.exists]
        if not positive:
            return ""
        anchored = {self._norm(t) for t in (anchored_tables or [])}
        ok = {(f.column, f.value) for f in positive}
        lines = ["VERIFIED DATA FACTS (from actual data):"]
        shown = 0
        for fact in positive:
            if shown >= _MAX_PROBE_FACTS:
                break
            lines.append(f"  - {fact.render()}")
            shown += 1
        if anchored:
            for fact in facts:
                if shown >= _MAX_PROBE_FACTS:
                    break
                if fact.exists or self._norm(fact.table) not in anchored:
                    continue
                if (fact.column, fact.value) not in ok:
                    continue
                lines.append(f"  - {fact.render()}")
                shown += 1
        return "\n".join(lines)

    @staticmethod
    def _norm(name: str) -> str:
        return name.split(".")[-1].lower()

    @staticmethod
    def _extract_from_tables(sql: str) -> list[str]:
        """Base names of physical FROM/JOIN tables in a failing SQL (task
        contexts excluded — they are materialized, not candidates)."""
        try:
            from sqlglot import parse_one
            from sqlglot.expressions import Table
        except ImportError:
            return []
        try:
            tree = parse_one(sql)
        except Exception:
            return []
        out = []
        for tbl in tree.find_all(Table):
            name = tbl.name
            if name and not name.lower().startswith("_task_context"):
                out.append(name)
        return out

    @staticmethod
    def _extract_eq_filters(sql: str) -> list[tuple[str, str]]:
        """Extract `column = 'literal'` predicates (base column name only)."""
        try:
            from sqlglot import parse_one
            from sqlglot.expressions import Column, EQ, Literal
        except ImportError:
            return []
        try:
            tree = parse_one(sql)
        except Exception:
            return []
        out: list[tuple[str, str]] = []
        for eq in tree.find_all(EQ):
            left, right = eq.this, eq.expression
            if isinstance(left, Column) and isinstance(right, Literal):
                out.append((left.name, str(right.this)))
        return out

    @staticmethod
    def _literal(value: str) -> str:
        return "'" + str(value).replace("'", "''") + "'"