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

## Cache

- `diskcache`-backed `CentralCache` at `~/.syrch/cache` (TTL 24h)
- Cache hit count displayed in validate_real.py output: `[cache:Nh]`
- `--skip-cache` flag to bypass for fresh measurements
