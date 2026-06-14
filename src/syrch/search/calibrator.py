from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExecutionSignals:
    num_attempts: int = 0
    max_attempts: int = 3
    syntax_errors: int = 0
    schema_errors: int = 0
    execution_errors: int = 0
    had_empty_result: bool = False
    had_null_columns: bool = False
    had_overflow_result: bool = False
    quality_warnings: list[str] = field(default_factory=list)


@dataclass
class CalibrationWeights:
    retry_ratio: float = 0.05
    empty_result: float = 0.15
    null_column: float = 0.05
    overflow: float = 0.05
    syntax_error: float = 0.10
    schema_error: float = 0.05
    execution_error: float = 0.10


def calibrate(
    raw_confidence: float,
    signals: ExecutionSignals,
    weights: CalibrationWeights | None = None,
) -> float:
    if weights is None:
        weights = CalibrationWeights()

    penalty = 1.0

    if signals.max_attempts > 1:
        retry_ratio = signals.num_attempts / signals.max_attempts
        penalty *= 1.0 - weights.retry_ratio * (retry_ratio - 1.0 / signals.max_attempts)

    if signals.syntax_errors > 0:
        penalty *= 1.0 - weights.syntax_error

    if signals.schema_errors > 0:
        penalty *= 1.0 - weights.schema_error

    if signals.execution_errors > 0:
        penalty *= 1.0 - weights.execution_error

    if signals.had_empty_result:
        penalty *= 1.0 - weights.empty_result

    if signals.had_null_columns:
        penalty *= 1.0 - weights.null_column

    if signals.had_overflow_result:
        penalty *= 1.0 - weights.overflow

    return max(0.0, min(1.0, raw_confidence * penalty))
