from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from syrch.core.models import FinalSolution


@dataclass
class EvaluationMetrics:
    solution: FinalSolution
    exact_match: bool = False
    row_count_match: bool = False
    column_match: bool = False
    expected_rows: int = 0
    actual_rows: int = 0
    token_cost: int = 0
    num_reasoning_paths: int = 0
    num_subtasks: int = 0
    max_depth: int = 0
    expected_provided: bool = False
    error_types: list[str] = field(default_factory=list)
    error_count: int = 0
    retriever_recall: float = 0.0
    retriever_precision: float = 0.0
    dag_node_count: int = 0
    dag_minimal: bool = True

    def to_dict(self) -> dict:
        return {
            "exact_match": self.exact_match,
            "row_count_match": self.row_count_match,
            "column_match": self.column_match,
            "expected_rows": self.expected_rows,
            "actual_rows": self.actual_rows,
            "expected_provided": self.expected_provided,
            "token_cost": self.token_cost,
            "num_reasoning_paths": self.num_reasoning_paths,
            "num_subtasks": self.num_subtasks,
            "max_depth": self.max_depth,
            "solution_confidence": self.solution.confidence,
            "error_types": self.error_types,
            "error_count": self.error_count,
            "retriever_recall": self.retriever_recall,
            "retriever_precision": self.retriever_precision,
            "dag_node_count": self.dag_node_count,
            "dag_minimal": self.dag_minimal,
        }


def evaluate(
    solution: FinalSolution,
    expected_data: pd.DataFrame | None = None,
) -> EvaluationMetrics:
    metrics = EvaluationMetrics(solution=solution)
    metrics.token_cost = solution.token_cost
    metrics.num_reasoning_paths = sum(
        len(r.reasoning_paths) for r in (solution.tree or [])
    )
    metrics.num_subtasks = len(solution.tree or [])
    max_depth_val = 0
    for res in solution.tree or []:
        for path in res.reasoning_paths:
            if path.path_id:
                parts = path.path_id.split("-")
                if len(parts) > 1:
                    try:
                        d = int(parts[1])
                        max_depth_val = max(max_depth_val, d)
                    except ValueError:
                        pass
    metrics.max_depth = max_depth_val

    if expected_data is not None and solution.data is not None:
        metrics.expected_provided = True
        metrics.expected_rows = len(expected_data)
        metrics.actual_rows = len(solution.data)
        metrics.row_count_match = metrics.expected_rows == metrics.actual_rows
        try:
            pd.testing.assert_frame_equal(
                solution.data.reset_index(drop=True),
                expected_data.reset_index(drop=True),
            )
            metrics.exact_match = True
        except AssertionError:
            metrics.exact_match = False
        expected_cols = set(expected_data.columns)
        actual_cols = set(solution.data.columns)
        metrics.column_match = expected_cols == actual_cols

    return metrics
