# syrch — Symbolic Search Runtime

**English** | [[한국어]](README_ko.md)

NL Problem → ProblemSpec → Search(D&C+RLM) → SQL Executor → Optimal Solution

## Project Goal

`symbolic-search-runtime` (syrch) is a **search harness** that finds optimal solutions to natural language problems over structured data. Unlike a simple QA agent that answers in one shot, syrch **explores multiple reasoning paths** using Divide & Conquer decomposition and Recursive Language Models, executing candidate solutions against real databases to select the best result.

### Key Ideas

- **Divide & Conquer**: Decompose a problem into logically independent sub-problems (sub-tasks), solve each independently, then merge results. Sub-problems can depend on each other forming a DAG.
- **RLM (Recursive Language Model)**: Each sub-task runs its own REPL loop — generate code → validate syntax → validate schema → execute SQL → check quality → evaluate confidence → refine or stop. Multiple reasoning paths are explored per node.
- **Confidence Calibration**: LLM self-assessed confidence is discounted by execution signals (retries, errors, empty results) for more reliable scoring.
- **Grid Search**: Systematic hyperparameter testing (`max_depth`, `high_confidence`, `max_attempts`, `calibration_enabled`) to find optimal configs.
- **Multi-table Schema**: Planner and RLM see all database tables, not just one.
- **Search over reasoning, not execution**: D&C splits the *problem space*, not the SQL. Each sub-problem is a complete reasoning unit (think → code → validate → execute → evaluate).
- **Pluggable Executors**: Abstract `BaseExecutor` with SQLite, JDBC, and Databricks implementations — PEP 249 compatible.

## Architecture

```
User Question
    │
    ▼
┌──────────────────┐
│    Planner       │  ← LLM decomposes question into sub-task DAG
│  (D&C)           │     (depends_on, is_atomic, expected_output)
│                  │     Multi-table schema: all tables visible
└──────┬───────────┘
       │ TaskDAG (topo_layers)
       ▼
┌──────────────────┐
│    Scheduler     │  ← Layer-by-layer DAG execution
│                   │
│  For each node:   │
│  ┌─────────────┐  │
│  │ RLM Agent    │  │  ← 3-step validation loop:
│  │ 1. SQLGlot   │  │     1. Syntax check (sqlglot.parse_one)
│  │    syntax    │  │     2. Schema AST check (valid columns)
│  │ 2. Schema    │  │     3. Execute + quality check
│  │    AST check │  │     Confidence calibration applied
│  │ 3. Execute   │  │
│  │ 4. Quality   │  │
│  │ 5. Calibrate │  │
│  └─────────────┘  │
│                   │
│  Pruning:          │
│  conf ≥ threshold → greedy stop
│  max_attempts hit → best path selected
└──────┬───────────┘
       │ NodeResults (DataFrames + SQL + confidence)
       ▼
┌──────────────────┐
│   Aggregator     │  ← Merge leaf results → final answer
│                   │     Tiebreaker: equal confidence → lower token_cost
└──────┬───────────┘
       │ FinalSolution
       ▼
 Optimal Answer + SQL + Reasoning Trace

 ═══════ Optional: RLM Clarification ═══════
       │
       ▼ (if --interactive)
┌──────────────────┐
│  RLM Clarifier   │  ← RLM exhaustion detected
│                   │     ambiguity score >= threshold
│  Node-level:      │     → ask user question
│  ┌─────────────┐  │     → refine ProblemSpec
│  │ no_sql       │  │     → re-run pipeline
│  │ empty_result │  │
│  │ quality_fail │  │
│  │ low_confidence│  │
│  └─────────────┘  │
└──────┬───────────┘
       │ Clarification answer → refined problem
       ▼
     Back to Planner (retry)

 ═══════ Optional: Grid Search ═══════
       │
       ▼
┌──────────────────┐
│   Grid Search    │  ← 27-54 cells (Product of params)
│                   │     ProcessPoolExecutor (max_workers=3)
│                   │     Reports: config.json, results.json,
│                   │              best.json, summary.md
└──────┬───────────┘
       │ Best config → run_pipeline again
```

### How a Sub-Task Executes (RLM Node)

```
Node "Find top 10% customers"
    │
    ├── Attempt 1: SQL path A
    │   ├── [PASS] Syntax check (sqlglot)
    │   ├── [PASS] Schema column check
    │   ├── [PASS] Execute → 5,234 rows
    │   ├── [WARN] Quality: returned 5234 rows (>1000)
    │   └── confidence: 0.72 (below threshold, retry)
    │
    ├── Attempt 2: SQL path B
    │   ├── [PASS] Syntax check
    │   ├── [PASS] Schema column check
    │   ├── [PASS] Execute → 534 rows
    │   ├── [PASS] Quality: OK
    │   └── confidence: 0.91 → calibrated: 0.86 (above threshold, stop)
    │
    └── Return best (calibrated) result to parent
```

## Directory Structure

```
syrch/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── PLAN.md
├── LICENSE
├── .gitignore
├── benchmarks/example.jsonl
├── src/syrch/
│   ├── __init__.py
│   ├── cli/app.py                # Typer CLI
│   ├── core/
│   │   ├── models.py             # Data types (dataclasses)
│   │   ├── config.py             # ExecutionConfig + config loader
│   │   └── logging.py            # Structured logging
│   ├── executors/
│   │   ├── base.py               # BaseExecutor (ABC)
│   │   ├── sqlite_executor.py    # SQLite
│   │   ├── jdbc_executor.py      # JDBC via SQLAlchemy
│   │   ├── databricks_executor.py # Databricks SQL
│   │   └── cached_executor.py    # diskcache-backed SQL cache
│   ├── llm/
│   │   ├── base.py               # BaseLLM (ABC)
│   │   ├── openai_llm.py         # OpenAI
│   │   ├── anthropic_llm.py      # Anthropic Claude
│   │   └── cache.py              # CachedLLM + CentralCache
│   ├── search/
│   │   ├── planner.py            # D&C: NL -> TaskDAG
│   │   ├── scheduler.py          # DAG execution engine
│   │   ├── rlm_engine.py         # RLM REPL loop
│   │   ├── aggregator.py         # Result merge
│   │   ├── calibrator.py         # Confidence calibration
│   │   ├── clarify.py            # Ambiguity detection
│   │   ├── grid.py               # Grid search
│   │   └── pipeline.py           # Orchestrator
│   └── eval/
│       ├── runner.py             # Benchmark harness
│       ├── metrics.py            # Evaluation metrics
│       └── report.py             # Report export
├── validate_real.py              # Real LLM validation
├── orders_10dim.sqlite           # TPC-H derived (7.5M rows)
├── wikipedia_clickstream.sqlite  # Clickstream data (3K rows)
└── tests/
    ├── test_cache.py
    ├── test_clarify.py
    ├── test_e2e.py
    ├── test_eval.py
    ├── test_integration.py
    ├── test_planner.py
    ├── test_rlm_engine.py
    └── test_scheduler.py
```

## Data Model

```
ProblemSpec { question, schema, all_schemas, goal_metric }
    │
    ▼
TaskDAG { nodes: {A, B, C, ...}, root_id, topo_layers }
    │  각 TaskNode: { id, description, depends_on, is_atomic, join_type }
    ▼
Scheduler → NodeResult { node_id, data(DataFrame), sql, confidence,
                         reasoning_paths, cost_tokens, error }
    │
    ▼
Aggregator → FinalSolution { answer, sql, confidence, data, token_cost, tree }
             (tiebreaker: equal confidence → lower cost_tokens wins)
```

## Installation

```bash
# Core (CLI + SQLite)
pip install syrch

# Databricks SQL Warehouse
pip install "syrch[databricks]"

# PySpark executor (inside Databricks Runtime)
pip install "syrch[pyspark]"

# Development (tests + lint)
pip install -e ".[dev]"

# Everything
pip install "syrch[all]"
```

## Python API (Library Mode)

Use directly from Databricks notebooks or Python scripts:

```python
from syrch import query

result = query(
    question="What discount × shipping combo maximizes revenue?",
    executor_type="databricks",
    model="gpt-4o",
)
print(result.answer)
print(result.sql)
print(result.confidence)
print(result.data)
```

## CLI Usage

```bash
# Install
pip install syrch

# Inspect database schema
syrch schema wikipedia_clickstream.sqlite
syrch schema orders_10dim.sqlite -t orders_10dim

# Show default config
syrch config

# Solve a problem (requires LLM API key)
export OPENAI_API_KEY="sk-..."
syrch search -q "What discount × shipping combo maximizes revenue for top 10% customers?"

# With config file
syrch search -q "..." --config syrch.yml

# With options
syrch search -q "Which click type generates the most traffic?" \
  --db wikipedia_clickstream.sqlite \
  --max-depth 3 \
  --high-conf 0.85 \
  --max-attempts 3 \
  --verbose

# Grid search over hyperparameters (54 cells)
syrch search -q "..." --db orders_10dim.sqlite --grid

# Benchmark against expected results
syrch eval -q "..." --db orders_10dim.sqlite --expected expected.csv

# Run benchmark suite
syrch benchmark benchmarks/orders.jsonl
```

### CLI Reference

| Command | Option | Description |
|---------|--------|-------------|
| `search` | `-q` / `--question` | Natural language problem (required) |
| | `--db` | Database path (default: `orders_10dim.sqlite`) |
| | `--max-depth` | Max D&C recursion depth (default: 3) |
| | `--executor` | `sqlite` / `databricks` / `jdbc` |
| | `--max-attempts` | Max RLM attempts per node (default: 3) |
| | `--high-conf` | Confidence threshold for greedy stop (default: 0.85) |
| | `--budget` | Token budget (default: 100000) |
| | `--llm` | `openai` / `anthropic` |
| | `--model` | LLM model name (default: `qwen3.5-4b-4bit`) |
| | `-v` / `--verbose` | Show reasoning traces |
| | `--cache/--no-cache` | Enable/disable LLM + SQL cache (default: on) |
| | `--cache-ttl` | Cache TTL in seconds (default: 86400) |
| | `--grid` | Run grid search over hyperparameters |
| | `--grid-parallel/--grid-sequential` | Parallel vs sequential grid execution |
| | `--grid-max-workers` | Max concurrent API calls (default: 3) |
| | `--max-concurrency` | Max concurrent LLM calls (default: 5; use 1 for local models) |
| | `--interactive` | Ask clarification questions when SQL cannot solve the task |
| | `--non-interactive` | One-shot mode with no clarification (default) |
| | `--config` | Path to YAML config file (`syrch.yml` or `~/.syrch/config.yml`) |
| `eval` | `-q` | Question |
| | `--db` | Database path |
| | `--executor` | Executor type |
| | `--expected` | Expected results CSV |
| | `--report-format` | `md` / `json` |
| `benchmark` | `PATH` | JSONL benchmark file (positional) |
| | `--executor` | Executor type |
| | `--report` | Output report path |
| `schema` | `DB` | Database path (positional) |
| | `-t` / `--table` | Specific table |
| `config` | `--db` | Database path |

## Configuration

Config loaded from (priority order): **CLI args > env vars (`SYRCH_*`) > config file > Databricks Secrets > defaults**.

### Config File (`syrch.yml`)

```yaml
llm:
  provider: openai
  model: qwen3.5-4b-4bit
  base_url: http://100.88.35.18:11434/v1
  temperature: 0.7
  max_tokens_per_call: 4096
  timeout_seconds: 120

execution:
  executor_type: sqlite
  max_depth: 3
  max_attempts_per_node: 3
  high_confidence: 0.85
  token_budget: 100000
  cache_enabled: true
  cache_ttl: 86400
  verbose: false
```

Search paths: `./syrch.yml` > `~/.syrch/config.yml` > `--config <path>` explicit override

### Environment Variables

| Variable | Maps to | Example |
|----------|---------|---------|
| `SYRCH_MODEL` | `llm.model` | `gpt-4o` |
| `SYRCH_API_KEY` | `llm.api_key` | `sk-...` |
| `SYRCH_BASE_URL` | `llm.base_url` | `http://100.88.35.18:11434/v1` |
| `SYRCH_MAX_DEPTH` | `execution.max_depth` | `3` |
| `SYRCH_VERBOSE` | `execution.verbose` | `true` |

### Databricks Connection

| Variable | Auth type | Description |
|----------|-----------|-------------|
| `DATABRICKS_SERVER_HOSTNAME` | all | Databricks workspace URL |
| `DATABRICKS_HTTP_PATH` | all | SQL Warehouse HTTP path |
| `DATABRICKS_TOKEN` | `pat` | Personal Access Token |
| `DATABRICKS_AUTH_TYPE` | all | `pat` (default), `databricks-oauth`, or `azure` |
| `DATABRICKS_CLIENT_ID` | oauth/azure | OAuth client ID |
| `DATABRICKS_CLIENT_SECRET` | oauth/azure | OAuth client secret |
| `AZURE_TENANT_ID` | azure | Azure AD tenant ID |

## Structured Logging

Internal diagnostics go to **stderr** via `logging`. User-facing output (Solution, SQL) goes to **stdout** via `rich`.

```bash
# Default: WARNING+ only to stderr
syrch search -q "..."

# Verbose: INFO level
syrch search -q "..." -v

# Library mode
python -c "
from syrch import query
result = query('Total revenue?', verbose=True)
"
```

Log format: `LEVEL:logger_name:message` (stdlib `logging` default)

```
INFO:syrch.scheduler:Layer 0: dispatching 2 nodes
WARNING:syrch.rlm_engine:Empty result, confidence penalized
```

## CI

GitHub Actions (`push`/`PR` → `main`):

| Step | Command |
|------|---------|
| Lint | `ruff check src/syrch/` |
| Type check | `mypy src/syrch/ --ignore-missing-imports` |
| Test | `pytest tests/ -v --cov=src/syrch/` (Python 3.11 + 3.12) |

## Confidence Calibration

LLM self-assessed confidence is adjusted by execution signals:

| Signal | Weight | Effect |
|--------|--------|--------|
| `syntax_error` | 0.10 | ×0.90 per occurrence |
| `execution_error` | 0.10 | ×0.90 per occurrence |
| `empty_result` | 0.15 | ×0.85 if result is empty |
| `schema_error` | 0.05 | ×0.95 per occurrence |
| `null_column` | 0.05 | ×0.95 if result has all-NULL columns |
| `retry_ratio` | 0.05 | Scales with attempts used |

**Heuristic penalties** (aggregator):
- Empty result: +0.15 per node
- Error present: +0.15 per node
- TOP-N mismatch: +0.05 per node
- "by year" without year column: +0.10 (once, global)
- **Capped at 0.40 total**

Formula: `calibrated = raw × Π(1 - weight_if_applicable)`

Disabled by passing `--no-cache` (sets `calibration_enabled=False` in `ExecutionConfig`).

## Grid Search

Automated hyperparameter search for optimal configuration:

```bash
syrch search -q "What discount × shipping combo maximizes revenue?" \
  --db orders_10dim.sqlite --grid
```

Default parameter grid (54 cells):
| Parameter | Values |
|-----------|--------|
| max_depth | 1, 3, 5 |
| high_confidence | 0.7, 0.85, 0.95 |
| max_attempts_per_node | 1, 3, 5 |
| calibration_enabled | True, False |

Output: `autoresearch/reports/{YYYYMMDD-HHMMSS}/{config,results,best}.json` + `summary.md`

Best config selection: `exact_match > confidence` (cells with errors are skipped).

## Pruning Strategy

The RLM engine uses a confidence-based pruning strategy:

1. Generate first reasoning path → SQL → 3-step validation (syntax → schema → quality)
2. Execute → score
3. Apply confidence calibration (if enabled)
4. If calibrated confidence ≥ `HIGH_CONFIDENCE` (0.85) → **greedy accept**, stop
5. If below threshold → generate alternative path
6. After `max_attempts` → pick **best path** by calibrated confidence

This balances search thoroughness with token budget. Simple problems resolve quickly (greedy path), while complex ones explore multiple candidates.

## Executor Abstraction

All executors conform to `BaseExecutor`:

```python
class BaseExecutor(ABC):
    def execute(sql: str) -> DataFrame: ...
    def get_schema(table_name?: str) -> TableSchema: ...
    def list_tables() -> list[str]: ...
    def close(): ...
```

| Executor | Backend | Connection |
|----------|---------|------------|
| `SQLiteExecutor` | SQLite | `sqlite3` (thread-safe via `threading.local`) |
| `JDBCExecutor` | Any JDBC | SQLAlchemy |
| `DatabricksExecutor` | Databricks SQL | `databricks-sql-connector` (PEP 249) |

## Caching

All LLM and SQL calls are cached via `diskcache` (24h TTL):

| Layer | Cache | Key |
|-------|-------|-----|
| LLM `generate()` | `CachedLLM` | SHA256(system + user + model + temperature) |
| LLM `generate_json()` | `CachedLLM` | SHA256(system + user + model + temperature) |
| SQL `execute()` | `CachedExecutor` | SHA256(sql) |

Toggle with `--cache/--no-cache` flag; TTL configurable with `--cache-ttl`.

## Datasets

| Dataset | Rows | Size | Description |
|---------|------|------|-------------|
| `wikipedia_clickstream.sqlite` | 3,138 | 280 KB | Aggregated Wikipedia clickstream data with mutual information metadata |
| `orders_10dim.sqlite` | 7,500,000 | 1.3 GB | TPC-H derived synthetic orders with 10 dimension columns |

## Testing

```bash
# Unit tests (FakeLLM, no API key required)
pytest tests/ -v

# All 69 tests pass:
#   7  cache tests (CentralCache, CachedLLM, CachedExecutor)
#   9  clarify tests (ambiguity score, question generation, worst detection)
#   9  e2e tests (real SQLite DBs + pipeline)
#   14 eval tests (runner, metrics, benchmark, join merge)
#   8  integration tests (DAG, grid, clarification loop, multi-table)
#   8  planner tests (decompose, cycle, join keys, recursive)
#   10 rlm_engine tests (validation, calibration, quality, calibrator)
#   4  scheduler tests (DAG execution)
```

### Real-world Validation

```bash
# Run full validation (requires LLM API key)
python validate_real.py

# Specific level
python validate_real.py --level 3 --verbose

# Custom question
python validate_real.py --question "Total revenue by year?" --db orders_10dim.sqlite

# With local model
python validate_real.py --model qwen3.5-4b --max-concurrency 1

# Results (2026-06-15, minimax-m3:cloud):
#   L1 Easy           3/3 PASS
#   L2 Medium         3/3 PASS
#   L3 Complex        2/2 PASS
#   L4 Very Complex   2/2 PASS
#   L5 Ambiguous      2/2 AMBIGUOUS (expected)
#   ─────────────────────────────
#   Total             10/10 PASS  100% (2 AMBIGUOUS)
```

## Research Background

- **RLM (Recursive Language Model)**: MIT CSAIL OASYS Lab, 2025. Inference paradigm where LLMs recursively decompose input via REPL environments. [`paper`](https://arxiv.org/abs/2512.24601) [`code`](https://github.com/alexzhang13/rlm)
- **RDD (Recursive Decomposition with Dependencies)**: Formal D&C framework with dependency DAGs. [`paper`](https://arxiv.org/abs/2505.02576)
- **PAC-MCTS**: Bias-aware pruning with formal guarantees for tree search. [`paper`](https://arxiv.org/abs/2604.14345)
- **ROMA**: Recursive meta-agent framework with Atomizer/Planner/Executor/Aggregator roles. [`paper`](https://arxiv.org/abs/2602.01848)
- **Graph Harness**: Structured DAG execution with immutable plan versions. [`paper`](https://arxiv.org/abs/2604.11378)
- **AdaptOrch**: Topology-aware multi-agent orchestration (parallel/sequential/hierarchical/hybrid). [`paper`](https://arxiv.org/abs/2602.16873)
- **DST**: Adaptive tree search with confidence-based pruning (26-75% computation reduction). [`paper`](https://arxiv.org/abs/2603.20267)
- **LLM Compiler**: Parallel task scheduling via dependency graphs; closely related to syrch's DAG scheduler and layer-by-layer execution. [`paper`](https://arxiv.org/abs/2312.13311)

## Open Research Questions

| Question | Approach |
|----------|----------|
| **When to stop dividing?** (Unit case detection) | Experiment with LLM self-assessment + complexity heuristics |
| **How to merge sub-task results?** | DAG-based REPL variable passing + Aggregator role |
| **How to prune search space?** | Confidence-based pruning + uncertainty-aware allocation |
| **Optimal D&C strategy?** | Topology routing (AdaptOrch) based on DAG structure metrics |
| **Optimal calibration weights?** | Grid search over penalty coefficients per signal |
| **Join key inference?** | Planner emits join_keys between sub-tasks |
| **Recursive decomposition?** | Planner recurses on non-atomic sub-tasks |
| **When SQL cannot solve?** | RLM clarification: ambiguity score → interactive feedback → re-decompose |
| **Optimal clarification threshold?** | Grid search over score weights + decision boundary |
