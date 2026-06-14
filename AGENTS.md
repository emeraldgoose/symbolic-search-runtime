# syrch — Symbolic Search Runtime

## Architecture

```
NL Problem → ProblemSpec → Search(D&C+RLM) → SQL Executor → Optimal Solution
```

## Directory

```
src/syrch/
├── cli/app.py           # Typer CLI
├── core/models.py       # Pydantic-style dataclasses
├── core/config.py       # ExecutionConfig
├── executors/           # BaseExecutor + SQLite/JDBC/Databricks
│   └── cached_executor.py
├── llm/                 # BaseLLM + OpenAI/Anthropic
│   ├── cache.py         # CentralCache (diskcache) + CachedLLM
│   └── openai_llm.py
├── search/
│   ├── planner.py       # D&C decomposition
│   ├── scheduler.py     # DAG execution engine
│   ├── rlm_engine.py    # Node-level REPL loop (syntax/schema/execution/quality validation)
│   ├── aggregator.py    # Result merge + confidence calibration + heuristics
│   ├── clarify.py       # Ambiguity detection & question generation
│   ├── calibrator.py    # ConfidenceCalibrator (execution signal penalties)
│   ├── pipeline.py      # run_pipeline() orchestrator
│   └── grid.py          # Hyperparameter grid search
└── eval/metrics.py
```

## Commands

```bash
syrch search -q "What discount × shipping combo maximizes revenue?"
syrch search -q "..." --db orders_10dim.sqlite --max-depth 3 -v
syrch search -q "..." --interactive        # clarification flow
syrch search --grid --grid-max-workers 3    # hyperparameter search
syrch schema --db orders_10dim.sqlite
syrch config
```

## Real Validation

```bash
python validate_real.py                      # all levels (L1-L5)
python validate_real.py --level 3            # specific level
python validate_real.py --quick              # L1-L2 only
python validate_real.py --question "..." --db orders_10dim.sqlite  # custom
python validate_real.py --skip-cache         # bypass disk cache
python validate_real.py --model "..."        # override LLM model
python validate_real.py --verbose            # detailed output
```

## Config

Key `ExecutionConfig` / `LLMConfig` fields:

| Field | Default | Description |
|-------|---------|-------------|
| `max_attempts_per_node` | 3 | Max REPL retries per DAG node |
| `high_confidence` | 0.85 | Threshold to early-stop REPL loop |
| `token_budget` | 100_000 | Global token cap across all nodes |
| `max_concurrency` | 5 | Max parallel threads in scheduler |
| `calibration_enabled` | True | Apply execution-signal penalties |
| `max_tokens_per_call` | 4096 | Max output tokens per LLM call (prevents runaway generation) |
| `timeout_seconds` | 120 | Per-LLM-call HTTP timeout |

## Test

```bash
pytest tests/ -v                             # 69 unit + integration tests
python validate_real.py --quick              # real LLM validation
```

## Pipeline Confidence Flow

```
RLM Agent (per node):
  raw_confidence → ConfidenceCalibrator (execution signals) → node.confidence

Aggregator (final):
  best_conf = max(leaf_node.confidence)
  adjusted_conf = best_conf × (1.0 - max_ambiguity × 0.5) × (1.0 - heuristic_penalty)
  heuristic_penalty from: empty results, errors, TOP-N mismatch, missing columns
```

### Confidence Details

**Raw confidence extraction** (`rlm_engine.py:_extract_confidence`):
- Scans LLM output for `Confidence: <0.0-1.0>` pattern
- Returns `(value, found)` tuple — `found=False` when model omits the line
- **Default** when not found: `0.7`

**Execution-based boost** (`rlm_engine.py`):
- When SQL executes successfully with non-empty data AND confidence was not explicitly stated by the LLM:
  `confidence = 0.85` (overrides default 0.7)
- Successful execution always wins as `best_path`, regardless of prior failed attempts

**Calibration weights** (`calibrator.py`):

| Signal | Weight | Effect |
|--------|--------|--------|
| `syntax_error` | 0.10 | ×0.90 per occurrence |
| `execution_error` | 0.10 | ×0.90 per occurrence |
| `empty_result` | 0.15 | ×0.85 if result is empty |
| `schema_error` | 0.05 | ×0.95 per occurrence |
| `null_column` | 0.05 | ×0.95 if result has all-NULL columns |
| `retry_ratio` | 0.05 | Scales with attempts used |

**Heuristic penalties** (`aggregator.py`):
- Empty result: +0.15 per node
- Error present: +0.15 per node  
- TOP-N mismatch: +0.05 per node
- "by year" without year column: +0.10 (once, global)
- **Capped at 0.40 total**

## Cache

- `diskcache`-backed `CentralCache` at `~/.syrch/cache` (TTL 24h)
- Cache hit count displayed in validate_real.py output: `[cache:Nh]`
- `--skip-cache` flag to bypass for fresh measurements

## Small Model Tuning

When using small quantized models (e.g., `qwen3.5-4b-4bit`):

- Set `max_tokens_per_call=4096` (prevents 504 timeouts from runaway generation)
- The confidence system auto-boosts to `0.85` when SQL executes successfully, compensating for models that don't output confidence scores
- Calibration weights are tuned lower than default to avoid over-penalizing syntax/execution errors common in smaller models
- Planner includes type guards (`isinstance` checks) for malformed JSON responses
