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
