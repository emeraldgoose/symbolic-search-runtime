from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens_per_call: int = 4096
    timeout_seconds: int = 120


@dataclass
class ExecutionConfig:
    question: str
    db_path: str
    executor_type: str = "sqlite"
    max_depth: int = 3
    max_attempts_per_node: int = 3
    high_confidence: float = 0.85
    token_budget: int = 100_000
    verbose: bool = False
    cache_enabled: bool = True
    cache_ttl: int = 86400
    calibration_enabled: bool = True
    interactive: bool = False
    ambiguity_threshold: float = 0.35
    max_concurrency: int = 5
    llm: LLMConfig = field(default_factory=LLMConfig)
