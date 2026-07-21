from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ColumnSchema:
    name: str
    type: str
    nullable: bool = True


@dataclass
class TableSchema:
    name: str
    columns: list[ColumnSchema]


@dataclass
class ScoredTable:
    schema: TableSchema
    score: float = 0.0
    match_reasons: list[str] = field(default_factory=list)


@dataclass
class ProblemSpec:
    question: str
    schema: TableSchema
    all_schemas: list[TableSchema] | None = None
    scored_schemas: list[ScoredTable] | None = None
    goal_metric: str | None = None


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
    _children: list[str] | None = None

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


@dataclass
class NodeResult:
    node_id: str
    data: pd.DataFrame
    sql: str
    confidence: float
    reasoning_paths: list[ReasoningPath] = field(default_factory=list)
    cost_tokens: int = 0
    error: str | None = None
    ambiguity_score: float = 0.0


@dataclass
class FinalSolution:
    question: str
    answer: str
    data: pd.DataFrame | None = None
    sql: str = ""
    confidence: float = 0.0
    token_cost: int = 0
    tree: list[NodeResult] = field(default_factory=list)
    clarified: bool = False
    clarification_qa: list[tuple[str, str]] = field(default_factory=list)
