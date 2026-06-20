from __future__ import annotations

import os
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, cast


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
    db_path: str | list[str]
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


_CONFIG_FILE_PATHS = [
    Path("syrch.yml"),
    Path("syrch.yaml"),
    Path.home() / ".syrch" / "config.yml",
    Path.home() / ".syrch" / "config.yaml",
]


def find_config_file(path: str | None = None) -> Path | None:
    if path:
        p = Path(path)
        return p if p.exists() else None
    for p in _CONFIG_FILE_PATHS:
        if p.exists():
            return p
    return None


def load_config_from_file(path: Path) -> dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_secret(key: str) -> str | None:
    val = os.environ.get(key)
    if val is not None:
        return val
    try:
        from databricks.sdk.runtime import dbutils
        scope = os.environ.get("DATABRICKS_SECRET_SCOPE", "syrch")
        return dbutils.secrets.get(scope=scope, key=key.lower())
    except (ImportError, Exception):
        return None


_ENV_MAP: dict[str, str] = {
    "model": "SYRCH_MODEL",
    "api_key": "SYRCH_API_KEY",
    "base_url": "SYRCH_BASE_URL",
    "provider": "SYRCH_LLM_PROVIDER",
    "temperature": "SYRCH_TEMPERATURE",
    "max_tokens_per_call": "SYRCH_MAX_TOKENS",
    "timeout_seconds": "SYRCH_TIMEOUT",
    "executor_type": "SYRCH_EXECUTOR",
    "max_depth": "SYRCH_MAX_DEPTH",
    "max_attempts_per_node": "SYRCH_MAX_ATTEMPTS",
    "high_confidence": "SYRCH_HIGH_CONFIDENCE",
    "token_budget": "SYRCH_TOKEN_BUDGET",
    "verbose": "SYRCH_VERBOSE",
    "cache_enabled": "SYRCH_CACHE",
    "cache_ttl": "SYRCH_CACHE_TTL",
    "calibration_enabled": "SYRCH_CALIBRATION",
    "interactive": "SYRCH_INTERACTIVE",
    "ambiguity_threshold": "SYRCH_AMBIGUITY_THRESHOLD",
    "max_concurrency": "SYRCH_MAX_CONCURRENCY",
}


def _resolve_env(key: str, field_type: type[Any] | str) -> Any:
    env_var = _ENV_MAP.get(key)
    if not env_var:
        return None
    val = os.environ.get(env_var)
    if val is None:
        return None
    if field_type is bool:
        return val.lower() in ("1", "true", "yes")
    if field_type is int:
        return int(val)
    if field_type is float:
        return float(val)
    return val


def merge_config(
    cli_overrides: dict[str, Any] | None = None,
    config_file_path: str | None = None,
) -> ExecutionConfig:
    data: dict[str, Any] = {}

    config_path = find_config_file(config_file_path)
    if config_path:
        file_cfg = load_config_from_file(config_path)
        exec_cfg = file_cfg.get("execution", file_cfg)
        llm_cfg = file_cfg.get("llm", {})
        if "question" in exec_cfg:
            data["question"] = exec_cfg.pop("question")
        if "db_path" in exec_cfg:
            data["db_path"] = exec_cfg.pop("db_path")
        llm_overrides: dict[str, Any] = {}
        for f in dataclass_fields(LLMConfig):
            if f.name in llm_cfg:
                llm_overrides[f.name] = llm_cfg[f.name]
        for f in dataclass_fields(ExecutionConfig):
            if f.name == "llm":
                continue
            if f.name in exec_cfg:
                data[f.name] = exec_cfg[f.name]
        data["llm"] = llm_overrides

    for f in dataclass_fields(ExecutionConfig):
        if f.name == "llm":
            continue
        val = _resolve_env(f.name, f.type)
        if val is not None:
            data.setdefault(f.name, val)
    for f in dataclass_fields(LLMConfig):
        val = _resolve_secret(f"SYRCH_{f.name.upper()}")
        if val is not None:
            data.setdefault("llm", {}).setdefault(f.name, val)

    if cli_overrides:
        llm_cli = cli_overrides.pop("llm", {})
        for k, v in cli_overrides.items():
            if v is not None:
                data[k] = v
        if llm_cli:
            data.setdefault("llm", {}).update({k: v for k, v in llm_cli.items() if v is not None})

    llm_dict = data.pop("llm", {})
    llm = LLMConfig(**{f.name: llm_dict.get(f.name, getattr(LLMConfig(), f.name)) for f in dataclass_fields(LLMConfig)})
    exec_kwargs = dict(cast(dict, {f.name: data.get(f.name) for f in dataclass_fields(ExecutionConfig) if f.name in data}))
    exec_kwargs["llm"] = llm
    return ExecutionConfig(**exec_kwargs)
