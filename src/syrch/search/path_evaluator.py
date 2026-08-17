from __future__ import annotations

from collections.abc import Iterable

from syrch.core.models import PathScore, TaskNode, ValidationResult
from syrch.search.calibrator import ExecutionSignals


_GRAIN_MARKERS: dict[str, set[str]] = {
    "hour": {"hourly", "hour"},
    "day": {"daily", "day"},
    "week": {"weekly", "week"},
    "month": {"monthly", "month"},
    "quarter": {"quarterly", "quarter"},
    "year": {"yearly", "annual", "year"},
}


def infer_native_grain(table_name: str, layer: str | None = None) -> str | None:
    """Best-effort native grain of a table, inferred from its name.

    `mart_sales_daily` → "day", `mart_sales_monthly` → "month". Transactional
    / dimension / staging tables return None (any grain is derivable by
    aggregation, but none is native).
    """
    tokens = set(table_name.lower().split("_"))
    for grain, markers in _GRAIN_MARKERS.items():
        if tokens & markers:
            return grain
    return None


def _canonical_grain(grain: str) -> str | None:
    """Map a requirement grain token ("monthly", "per_order") to the canonical
    native-grain key ("month") used by `_GRAIN_MARKERS`."""
    g = (grain or "").lower()
    for canonical, markers in _GRAIN_MARKERS.items():
        if g == canonical or g in markers or g in {f"per_{m}" for m in markers}:
            return canonical
    return None


class PathEvaluator:
    """Evidence-based evaluation of a candidate.

    Selection is NOT driven by `PathScore.total` (retriever prior is excluded
    from it and only kept as metadata). Local selection uses the
    multi-signal `CandidateEvaluation` in `solve()`; `PathScore` is retained
    for logging / back-compat.
    """

    WEIGHT_REQUIREMENT = 0.5
    WEIGHT_EXECUTION = 0.5

    def evaluate(
        self,
        validation: ValidationResult | None,
        signals: ExecutionSignals,
        retriever_score: float = 0.0,
    ) -> PathScore:
        if validation is None:
            req_score = 1.0
        elif not validation.passed:
            return PathScore(total=0.0)
        else:
            req_score = 1.0

        exec_score = self._execution_signal(signals)
        ret_score = min(1.0, max(0.0, retriever_score / 10.0))

        total = (
            self.WEIGHT_REQUIREMENT * req_score
            + self.WEIGHT_EXECUTION * exec_score
        )

        return PathScore(
            total=total,
            requirement_coverage=req_score,
            execution_signal=exec_score,
            retriever_evidence=ret_score,
        )

    def semantic_match(
        self,
        node: TaskNode,
        schema,
        data=None,
    ) -> float:
        """Requirement↔candidate semantic evidence.

        Deliberately NOT the retriever's question↔table keyword score. This
        measures whether the candidate's schema and executed result actually
        satisfy the task's structured requirements (metrics, time range).
        """
        req = node.requirements
        if req is None:
            return 0.5

        scores: list[float] = []
        result_cols = (
            {str(c).lower() for c in data.columns}
            if data is not None and not data.empty
            else set()
        )

        for metric in req.metrics:
            col_hit = self._token_overlap(metric, [c.name for c in schema.columns])
            res_hit = self._token_overlap(metric, result_cols)
            if col_hit and res_hit:
                scores.append(1.0)
            elif col_hit or res_hit:
                scores.append(0.5)
            else:
                scores.append(0.0)

        if req.time_range:
            has_date = any(
                any(tok in c.name.lower() for tok in ("date", "time", "_at", "_ts"))
                for c in schema.columns
            )
            scores.append(1.0 if has_date else 0.0)

        return (sum(scores) / len(scores)) if scores else 0.5

    def structural_match(
        self,
        validation: ValidationResult | None,
        node: TaskNode,
        data=None,
    ) -> float:
        """Grain / aggregation / result-shape consistency (soft evidence)."""
        s = 1.0
        if validation is not None:
            if validation.grain_mismatch:
                s -= 0.3
            if not validation.aggregation_ok:
                s -= 0.5

        req = node.requirements
        if (
            req is not None
            and req.aggregation
            and req.grain in (None, "total")
            and data is not None
            and not data.empty
            and len(data) != 1
        ):
            s -= 0.1
        return max(0.0, s)

    def grain_match(self, node: TaskNode, schema) -> float:
        """Requirement grain vs candidate native grain (discrimination).

        A native match is strong evidence the candidate is built for the
        requested grain. Transactional/dimension tables are derivable but not
        native. A "total" requirement accepts any table."""
        req = node.requirements
        req_grain = (req.grain or "").lower() if req else ""
        if not req_grain or req_grain in ("total", "overall"):
            return 1.0
        native = infer_native_grain(schema.name, schema.layer)
        if native and native == _canonical_grain(req_grain):
            return 1.0
        return 0.5

    def dimension_match(self, node: TaskNode, schema) -> float:
        """Requirement dimensions vs candidate columns (discrimination).

        Best-effort lexical check; unused when the requirement has no
        dimensions (returns 1.0)."""
        req = node.requirements
        dims = req.dimensions if req else []
        if not dims:
            return 1.0
        cols = [c.name for c in schema.columns]
        hits = sum(1.0 for d in dims if self._token_overlap(d, cols))
        return hits / len(dims)

    def time_match(self, node: TaskNode, schema) -> float:
        """Requirement time range vs candidate time column (discrimination)."""
        req = node.requirements
        if not req or not req.time_range:
            return 1.0
        has_date = any(
            any(tok in c.name.lower() for tok in ("date", "time", "_at", "_ts"))
            for c in schema.columns
        )
        return 1.0 if has_date else 0.0

    @staticmethod
    def _token_overlap(term: str, candidates: Iterable[str]) -> bool:
        tokens = set(str(term).lower().replace("_", " ").split())
        if not tokens:
            return False
        for cand in candidates:
            cand_tokens = set(str(cand).lower().replace("_", " ").split())
            if cand_tokens and (tokens & cand_tokens or tokens <= cand_tokens):
                return True
        return False

    @staticmethod
    def _execution_signal(signals: ExecutionSignals) -> float:
        if signals.num_attempts == 0:
            return 0.0
        penalties = 0.0
        if signals.syntax_errors > 0:
            penalties += 0.10 * min(signals.syntax_errors, 3)
        if signals.schema_errors > 0:
            penalties += 0.05 * min(signals.schema_errors, 3)
        if signals.execution_errors > 0:
            penalties += 0.10 * min(signals.execution_errors, 3)
        if signals.had_empty_result:
            penalties += 0.15
        if signals.had_null_columns:
            penalties += 0.05
        if signals.had_overflow_result:
            penalties += 0.05
        return max(0.0, 1.0 - penalties)
