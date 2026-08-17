from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class ReplanType(Enum):
    LOCAL = "local"
    STRUCTURAL = "structural"


class NodeStatus(str, Enum):
    SOLVED = "solved"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class ColumnSchema:
    name: str
    type: str
    nullable: bool = True
    description: str | None = None


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnSchema]
    description: str | None = None
    layer: str = "unknown"


_LAYER_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("dw_", "dw"),
    ("fact_", "fact"),
    ("dim_", "dim"),
    ("mart_", "mart"),
    ("agg_", "mart"),
    ("stg_", "staging"),
    ("stage_", "staging"),
    ("rpt_", "rpt"),
    ("report_", "rpt"),
    ("archive", "archive"),
    ("view_", "view"),
    ("v_", "view"),
)


def infer_layer(table_name: str) -> str:
    """Infer a table's layer from its name prefix.

    Feeds the retriever's grain-layer preference and layer penalties
    (`_apply_layer_adjustment`). Without it every table is `unknown` and the
    grain-aware machinery never fires.
    """
    name = table_name.lower()
    for prefix, layer in _LAYER_BY_PREFIX:
        if name.startswith(prefix):
            return layer
    return "unknown"


@dataclass
class CandidateFeatures:
    matched_keywords: list[str] = field(default_factory=list)
    matched_columns: list[str] = field(default_factory=list)
    matched_descriptions: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    matched_metrics: list[str] = field(default_factory=list)
    matched_grains: list[str] = field(default_factory=list)
    representative_columns: list[str] = field(default_factory=list)
    layer: str = "unknown"


@dataclass
class ScoredTable:
    schema: TableSchema
    score: float = 0.0
    match_reasons: list[str] = field(default_factory=list)
    features: CandidateFeatures = field(default_factory=CandidateFeatures)


@dataclass
class ScoredSchemaEvidence:
    candidates: list[ScoredTable] = field(default_factory=list)
    matched_columns: list[ColumnSchema] = field(default_factory=list)
    grain_hints: list[str] = field(default_factory=list)
    metric_hints: list[str] = field(default_factory=list)
    time_columns: list[str] = field(default_factory=list)
    alias_map: dict[str, list[tuple[str, str, str | None]]] = field(default_factory=dict)
    # {business_term: [(physical_column, table_name, suggested_agg), ...]}
    all_schemas: list[TableSchema] = field(default_factory=list)


@dataclass
class CandidatePolicy:
    max_candidates: int = 10
    min_score: float = 0.0
    relative_threshold: float = 0.9

    def filter(self, candidates: list[ScoredTable]) -> list[ScoredTable]:
        if not candidates:
            return candidates
        top_score = candidates[0].score
        if top_score == 0.0:
            return candidates[:self.max_candidates]
        threshold = max(self.min_score, top_score * (1.0 - self.relative_threshold))
        filtered = [c for c in candidates if c.score >= threshold]
        return filtered[:self.max_candidates]


@dataclass
class SelectionReason:
    selected_from_candidate: bool = True
    top_k_rank: int = 0
    score: float = 0.0
    used_features: list[str] = field(default_factory=list)


@dataclass
class ProblemSpec:
    question: str
    schema: TableSchema
    all_schemas: list[TableSchema] | None = None
    scored_schemas: list[ScoredTable] | None = None
    goal_metric: str | None = None
    evidence: ScoredSchemaEvidence | None = None


@dataclass
class MetricRequirement:
    """A business metric the SQL must compute, expressed structurally.

    Planner owns the *semantic* structure (what to solve); it must NOT
    dictate physical column names (`paid_at`/`shipped_at`) — those are
    candidate evidence the RLM resolves via alias_map / schema descriptions.
    `expression_semantics` describes the metric's meaning in business terms,
    giving the RLM a target to map onto physical columns.
    """

    name: str
    aggregation: str | None = None
    expression_semantics: str | None = None

    def render(self) -> str:
        parts = [self.name]
        if self.aggregation:
            parts.append(f"aggregated by {self.aggregation}")
        if self.expression_semantics:
            parts.append(f"semantics: {self.expression_semantics}")
        return " ".join(parts)


@dataclass
class FilterRequirement:
    """A semantic filter the SQL must apply (e.g. exclude holidays).

    `relation`/`predicate` are hints about *which table/expression* carries
    the filter — provided by the planner so the RLM knows the requirement
    exists and why a supporting table is needed, not to select the answer
    table (S15).
    """

    semantic: str
    relation: str | None = None
    predicate: str | None = None

    def render(self) -> str:
        parts = [f"exclude/filter by {self.semantic}"]
        if self.predicate:
            parts.append(f"({self.predicate})")
        if self.relation:
            parts.append(f"via {self.relation}")
        return " ".join(parts)


@dataclass
class ValueConstraint:
    """A value-domain constraint on a column (e.g. status = 'refunded').

    The value domain is evidence provided by the system (column value
    profile), NOT guessed by the planner. This is RLM reasoning input only —
    it never enters selection score.
    """

    column: str
    values: list[str] = field(default_factory=list)
    negate: bool = False

    def render(self) -> str:
        op = "not in" if self.negate else "in"
        vals = ", ".join(f"'{v}'" for v in self.values)
        return f"{self.column} {op} ({vals})"


@dataclass
class SupportingRelation:
    """A supporting table the RLM should consider joining (e.g. dim_date).

    Unlike hint_tables (which are answer candidates), a supporting relation is
    a helper the planner signals as necessary to satisfy a filter/grain — the
    RLM gets a *reason* to use it, not just "it's available" (S15).
    """

    table: str
    purpose: str | None = None
    join_hint: str | None = None

    def render(self) -> str:
        parts = [self.table]
        if self.join_hint:
            parts.append(f"(join: {self.join_hint})")
        if self.purpose:
            parts.append(f"purpose: {self.purpose}")
        return " ".join(parts)


@dataclass
class RequirementSpec:
    metrics: list[str] = field(default_factory=list)
    metric_details: list[MetricRequirement] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    aggregation: str | None = None
    grain: str | None = None
    time_range: tuple[str, str] | None = None
    must_use_columns: list[str] = field(default_factory=list)
    filters: list[FilterRequirement] = field(default_factory=list)
    value_constraints: list[ValueConstraint] = field(default_factory=list)
    supporting_relations: list[SupportingRelation] = field(default_factory=list)

    def render(self) -> str:
        """Structural rendering of the requirements for the RLM prompt.

        Metric semantics, filters and supporting relations are surfaced here
        so the RLM knows *what* to compute and *why* a supporting table is
        needed — without dictating physical columns (those come from evidence).
        """
        lines: list[str] = []
        for md in self.metric_details:
            lines.append(f"  metric: {md.render()}")
        for m in self.metrics:
            if not any(md.name == m for md in self.metric_details):
                lines.append(f"  metric: {m}")
        if self.aggregation and not self.metric_details:
            lines.append(f"  aggregation: {self.aggregation}")
        if self.dimensions:
            lines.append(f"  dimensions: {', '.join(self.dimensions)}")
        if self.grain:
            lines.append(f"  grain: {self.grain}")
        if self.time_range:
            lines.append(f"  time range: {self.time_range[0]} to {self.time_range[1]}")
        for f in self.filters:
            lines.append(f"  filter: {f.render()}")
        for vc in self.value_constraints:
            lines.append(f"  value constraint: {vc.render()}")
        for sr in self.supporting_relations:
            lines.append(f"  supporting relation: {sr.render()}")
        if not lines:
            return ""
        return "\n" + "\n".join(lines)


@dataclass
class ValidationResult:
    passed: bool = True
    missing_metrics: list[str] = field(default_factory=list)
    missing_filters: list[str] = field(default_factory=list)
    grain_mismatch: bool = False
    aggregation_ok: bool = True
    details: list[str] = field(default_factory=list)


@dataclass
class PathScore:
    total: float = 0.0
    requirement_coverage: float = 0.0
    execution_signal: float = 0.0
    retriever_evidence: float = 0.0


@dataclass
class CandidateEvaluation:
    """Outcome of running the RLM REPL loop against one candidate table.

    Hard filters gate ranking entry: `ok` (recoverability), `has_data`,
    `execution_valid` (recovered + executed) and `requirement_pass`
    (Validator). Any candidate failing a hard filter is not ranked.

    Selection uses *discrimination* signals only — requirement↔candidate
    compatibility (`structural_match → grain_match → dimension_match →
    time_match → result_quality`). Lexical relevance (`semantic_match`) and
    token cost are NOT discriminating evidence: a candidate that merely
    *names* the requested metric better is not thereby more likely correct.
    If the top two *viable* candidates have identical ranking signals there
    is no evidence to choose between them → the node is marked AMBIGUOUS
    (never resolved by execution / retriever / cost order).

    `posterior` is execution/semantic evidence only (retriever prior and
    lexical relevance excluded) and is used solely for the SearchPolicy
    termination margin.
    """

    table: str
    ok: bool = False
    execution_valid: bool = False
    requirement_pass: bool = False
    semantic_match: float = 0.0
    result_quality: float = 0.0
    structural_match: float = 0.0
    grain_match: float = 0.0
    dimension_match: float = 0.0
    time_match: float = 0.0
    cost_tokens: int = 0
    candidate_id: str = ""
    confidence: float = 0.0
    error: str | None = None
    attempts: int = 0
    has_data: bool = False
    path_score: PathScore | None = None

    @property
    def viable(self) -> bool:
        return (
            self.ok
            and self.has_data
            and self.execution_valid
            and self.requirement_pass
        )

    @property
    def ranking_signals(self) -> tuple[float, float, float, float, float]:
        """Discrimination signals compared for selection. Equality on all
        ⇒ AMBIGUOUS. Lexical relevance and token cost are excluded."""
        return (
            self.structural_match,
            self.grain_match,
            self.dimension_match,
            self.time_match,
            self.result_quality,
        )

    @property
    def posterior(self) -> float:
        """Execution/semantic evidence, excluding the retriever prior and
        lexical relevance.

        Used only for termination-margin decisions. The retriever prior never
        participates in termination, so a ground-truth table ranked far down
        the candidate list is never pruned on retriever grounds (S1 guard).
        """
        if not self.viable:
            return 0.0
        return (
            self.result_quality
            + self.structural_match
            + self.grain_match
            + self.dimension_match
            + self.time_match
        ) / 5.0


@dataclass
class JoinKey:
    left: str
    left_col: str
    right: str
    right_col: str
    how: str = "inner"


@dataclass
class TaskNode:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    parent_id: str | None = None
    depth: int = 0
    is_atomic: bool = False
    expected_output_desc: str = ""
    join_type: str = "all_of"
    join_keys: list[JoinKey] | None = None
    hint_tables: list[str] | None = None
    hint_columns: list[str] | None = None
    metric_columns: list[str] | None = None
    grain: str | None = None
    time_columns: list[str] | None = None
    requirements: RequirementSpec | None = None
    selection_reason: SelectionReason | None = None
    _children: list[str] | None = None
    _compressed_schemas: list[TableSchema] | None = None

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class TaskDAG:
    nodes: dict[str, TaskNode]
    root_id: str
    topo_layers: list[list[str]] = field(default_factory=list)


@dataclass
class ReasoningPath:
    path_id: str
    sql: str
    confidence: float
    cost_tokens: int = 0
    path_score: PathScore | None = None


@dataclass
class NodeResult:
    node_id: str
    data: pd.DataFrame
    sql: str
    confidence: float
    status: NodeStatus = NodeStatus.SOLVED
    selected_candidate: CandidateEvaluation | None = None
    candidates: list[CandidateEvaluation] = field(default_factory=list)
    reasoning_paths: list[ReasoningPath] = field(default_factory=list)
    cost_tokens: int = 0
    error: str | None = None
    ambiguity_score: float = 0.0
    replan_request: tuple[ReplanType, str] | None = None
    path_score: PathScore | None = None
    validation: ValidationResult | None = None
    had_context: bool = False
    context_used: bool = False


@dataclass
class ParentContext:
    """First-class contract describing a dependency node's result.

    Unlike a bare DataFrame preview, a ParentContext carries explicit metadata
    the RLM can reason about: the logical result name, column names/types, the
    physical tables the result was derived from, row count and a bounded
    preview. This lets a dependent task know *what* its dependency produced and
    *where the columns come from*, instead of guessing that `result_A` is a
    real database table (S3/S4 root cause).

    v0.3.5a: contract only — the RLM must not use `table_name` as a FROM
    table; dependent tasks re-derive the columns from the physical source
    tables. v0.3.5b materializes the context (Executor.materialize_context)
    so `_task_context_<id>` becomes a real temp table the dependent task can
    JOIN directly.
    """

    node_id: str
    table_name: str
    columns: list[ColumnSchema] = field(default_factory=list)
    source_tables: list[str] = field(default_factory=list)
    row_count: int = 0
    preview: str = ""
    sql: str = ""
    data: pd.DataFrame | None = None
    materialized: bool = False

    def to_prompt(self) -> str:
        cols = ", ".join(f"{c.name} ({c.type})" for c in self.columns) or "(empty)"
        lines = [
            f"  {self.table_name}  <- result of task {self.node_id}",
            f"    columns: {cols}",
            f"    source tables: {', '.join(self.source_tables) or '(unknown)'}",
            f"    row count: {self.row_count}",
        ]
        if self.preview:
            lines.append(f"    preview (first rows):\n{self.preview}")
        return "\n".join(lines)


@dataclass
class AttemptSchemaContext:
    """Explicit per-attempt SQL scope (S3).

    PRIMARY vs JOIN-AVAILABLE vs TASK CONTEXT are separated so the search
    boundary (the tables the RLM may reference) exactly equals the schemas
    shown to it. A pool table that is not part of this attempt's scope must be
    rejected by the FROM guard even though it exists in the candidate pool.

    allowed_tables = primary_tables | join_available_tables (physical scope).
    TASK CONTEXT entries are dependency results. In v0.3.5b they are
    materialized (Executor.materialize_context) into real `_task_context_*`
    tables when the dependency ends SOLVED; materialized contexts may appear
    in FROM/JOIN, unmaterialized ones are rejected by the scope guard.
    """

    primary_tables: set[str] = field(default_factory=set)
    join_available_tables: set[str] = field(default_factory=set)
    task_contexts: list[ParentContext] = field(default_factory=list)

    @property
    def allowed_tables(self) -> set[str]:
        return self.primary_tables | self.join_available_tables


@dataclass
class FinalSolution:
    question: str
    answer: str
    data: pd.DataFrame | None = None
    sql: str = ""
    confidence: float = 0.0
    path_score: float = 0.0
    token_cost: int = 0
    tree: list[NodeResult] = field(default_factory=list)
    clarified: bool = False
    clarification_qa: list[tuple[str, str]] = field(default_factory=list)
