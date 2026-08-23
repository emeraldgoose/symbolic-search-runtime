# syrch — Symbolic Search Runtime

**English** | [[한국어]](README_ko.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](#ci)
[![Version](https://img.shields.io/badge/version-0.3.7-blueviolet)](pyproject.toml)

> **Natural language → verifiable SQL → optimal answer.**
> syrch explores multiple reasoning paths over real databases and picks the winner by evidence, not by guess.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Confidence & Evidence](#confidence--evidence)
- [Directory Structure](#directory-structure)
- [Development](#development)
- [Benchmark & Evaluation](#benchmark--evaluation)
- [Contributing](#contributing)
- [Research Background](#research-background)
- [License](#license)

---

## Overview

`syrch` (symbolic-search-runtime) is a **search harness** for natural-language analytics over structured data. Instead of answering in one shot, it **decomposes**, **searches**, and **verifies**:

```
NL Problem → ProblemSpec → Search(D&C + RLM) → SQL Executor → Optimal Solution
```

- **Decompose** the question into a DAG of sub-tasks via Divide & Conquer.
- **Search** each sub-task over a candidate table pool with a Recursive Language Model (RLM) REPL loop.
- **Verify** every candidate by executing SQL against the real database and ranking by evidence.

The result is not just an answer — it is an answer **with provenance**: which tables were used, which filters applied, and how the number was computed.

---

## Features

| Area | What it does |
|------|--------------|
| **Divide & Conquer Planner** | Turns a question into a `TaskDAG` of `RequirementSpec` nodes (metrics, dimensions, grain, filters, supporting relations). |
| **Retriever (ordering only)** | Scores every table by lexical/semantic relevance. The score decides *what to try next* — it never decides the winner. |
| **RLM Agent** | Per-candidate REPL loop: `generate SQL → validate syntax → validate schema → execute → quality check → evaluate`. |
| **Candidate Scope** | Strict per-attempt namespace: `PRIMARY` (FROM only), `JOIN-AVAILABLE` (JOIN only), `TASK CONTEXT` (materialized parent results). Prevents drift to out-of-scope tables. |
| **Pluggable Executors** | `SQLite`, `JDBC`, `Spark`, `Databricks SQL` behind a single `BaseExecutor` interface. |
| **Evidence-based Selection** | Lexicographic ranking over `structural_match → grain_match → dimension_match → time_match → result_quality`. Full tie → `AMBIGUOUS` (never broken by cost or order). |
| **Capability Gates** | `metric_feasible` and non-empty/non-null hard filters gate ranking entry; `time_match` uses data-aware coverage (MIN/MAX) to exclude tables whose data cannot cover the requested window. |
| **Context Materialization** | `SOLVED` parents are materialized as `_task_context_<id>` real tables that dependents can `JOIN`. `AMBIGUOUS`/`FAILED` never materialize. |
| **Calculation Basis** | Deterministic, localized explanation of *which reading* produced the number: time window, refund/status exclusion, and per-source SCD2 semantics (point-in-time vs fixed-window overlap vs ignored), plus step-by-step evidence. |

---

## Architecture

```
                USER QUESTION
                     │
                     ▼
           ┌──────────────────┐
           │  Planner (D&C)   │  "what to solve?"
           │  → RequirementSpec + TaskDAG
           └────────┬─────────┘
                    │ requirements + DAG
                    ▼
           ┌──────────────────┐
           │   Retriever      │  "where to look?"
           │  → candidate pool (ranked)
           └────────┬─────────┘
                    │ pool
                    ▼
           ┌──────────────────┐
           │   Scheduler      │  runs DAG layer by layer
           │  ┌────────────┐  │
           │  │ RLM Agent  │  │  candidate → SQL → validate → execute → evaluate
           │  │ beam / exhaustive, lexicographic
           │  └─────┬──────┘  │
           └────────┼─────────┘
                    │ SOLVED → materialize _task_context_<id>
                    ▼
           ┌──────────────────┐
           │   Aggregator     │  picks primary leaf (SOLVED > AMBIGUOUS),
           │  no re-ranking   │  builds localized calculation basis
           └────────┬─────────┘
                    ▼
        FINAL ANSWER + SQL + confidence + step-by-step evidence
```

**Node outcomes:** `SOLVED` → materialized, joinable. `AMBIGUOUS` → not materialized (uncertain results never become facts). `FAILED`/`BLOCKED` → consumers short-circuit to `BLOCKED`.

<details>
<summary><b>How a sub-task executes (click to expand)</b></summary>

```
Node "Refund counts by reason"
  ├─ candidate dw_sales_order (prior 0.8)
  │   ├─ [PASS] Syntax (sqlglot)
  │   ├─ [PASS] Schema (status, total_amount)
  │   ├─ [PASS] Scope (FROM = PRIMARY)
  │   ├─ [PASS] Execute → 3 rows
  │   └─ viable: structural=1.0, grain=1.0, result_quality=1.0
  ├─ candidate mart_sales_daily (prior 0.6)
  │   ├─ [PASS] Syntax
  │   ├─ [FAIL] Schema: no status column → non-viable
  │   └─ rejected
  └─ top-1 vs top-2 differ → SOLVED (dw_sales_order)
     full tie → AMBIGUOUS (never broken by cost)
```

</details>

---

## Installation

```bash
# Core (CLI + SQLite)
pip install syrch

# Databricks SQL Warehouse
pip install "syrch[databricks-sql]"

# Spark (Databricks Runtime / EMR / standalone)
pip install "syrch[spark]"

# Development (tests + lint + type check)
pip install -e ".[dev]"

# Everything
pip install "syrch[all]"
```

**Requirements:** Python 3.11+

---

## Quick Start

### Library mode

```python
from syrch import query

result = query(
    question="What discount × shipping combo maximizes revenue?",
    executor_type="databricks-sql",
    model="gpt-4o",
)

print(result.answer)              # Final answer text (localized)
print(result.sql)                 # Executed SQL (all steps)
print(result.confidence)          # 0.0 – 1.0
print(result.data)                # Result DataFrame
print(result.calculation_basis)   # Step-by-step evidence (localized labels, English internals)
print(result.tables_used)         # Physical tables actually queried
print(result.tree)                # Per-task NodeResult list
print(result.dag_nodes)           # Task graph structure
```

Works directly inside **Databricks notebooks** and any Python script.

### CLI mode

```bash
# Inspect schema
syrch schema orders_10dim.sqlite
syrch schema wikipedia_clickstream.sqlite -t wikipedia_clickstream

# Show default config
syrch config

# Solve (requires API key)
export OPENAI_API_KEY="sk-..."
syrch search -q "What discount × shipping combo maximizes revenue for top 10% customers?"

# With options
syrch search -q "Which click type generates the most traffic?" \
  --db wikipedia_clickstream.sqlite --max-depth 3 --verbose

# Grid search
syrch search -q "..." --db orders_10dim.sqlite --grid
```

---

## CLI Usage

| Command | Key Options | Description |
|---------|-------------|-------------|
| `search` | `-q` / `--question` | Natural language problem (**required**) |
| | `--db` | Database path (default: `orders_10dim.sqlite`) |
| | `--executor` | `sqlite` / `databricks-sql` / `spark` / `jdbc` |
| | `--max-depth` | Max D&C recursion depth (default: 3) |
| | `--max-attempts` | Max RLM attempts per node (default: 3) |
| | `--search-policy` | `beam` / `exhaustive` (default: `beam`) |
| | `--beam-width` | Min candidates before early stop (default: 3) |
| | `--candidate-budget` | Max candidates per node (default: 8) |
| | `--stop-margin` | Posterior gap to stop early (default: 0.15) |
| | `--budget` | Token budget (default: 100000) |
| | `--llm` | `openai` / `anthropic` |
| | `--model` | Model name (default: `qwen3.5-4b-4bit`) |
| | `-v` / `--verbose` | Show reasoning traces |
| | `--cache / --no-cache` | LLM + SQL cache (default: on) |
| | `--grid` | Run hyperparameter grid search |
| | `--config` | YAML config file |
| `schema` | `DB` | Database path (positional) |
| | `-t` / `--table` | Specific table |
| `config` | | Show resolved config |
| `eval` | | Benchmark against expected CSV |
| `benchmark` | | Run JSONL benchmark suite |

---

## Configuration

Priority: **CLI args > config file > env vars (`SYRCH_*`) > Databricks Secrets > defaults**.

### Config file (`syrch.yml`)

```yaml
llm:
  provider: openai
  model: qwen3.5-4b-4bit
  base_url: http://localhost:11434/v1
  temperature: 0.7
  max_tokens_per_call: 4096
  timeout_seconds: 120

execution:
  executor_type: sqlite
  max_depth: 3
  max_attempts_per_node: 3
  search_policy: beam
  beam_width: 3
  candidate_budget: 8
  stop_margin: 0.15
  max_candidate_expansion: 2
  max_replans: 1
  token_budget: 100000
  cache_enabled: true
  cache_ttl: 86400
  calibration_enabled: true
  materialize_context: true
  verbose: false
```

Search paths: `./syrch.yml` → `./syrch.yaml` → `~/.syrch/config.yml` → `~/.syrch/config.yaml` (`--config <path>` overrides all).

### Environment variables

| Variable | Maps to | Example |
|----------|---------|---------|
| `SYRCH_MODEL` | `llm.model` | `gpt-4o` |
| `SYRCH_API_KEY` | `llm.api_key` | `sk-...` |
| `SYRCH_BASE_URL` | `llm.base_url` | `http://localhost:11434/v1` |
| `SYRCH_EXECUTOR` | `execution.executor_type` | `sqlite` |
| `SYRCH_MAX_DEPTH` | `execution.max_depth` | `3` |
| `SYRCH_SEARCH_POLICY` | `execution.search_policy` | `beam` |
| `SYRCH_BEAM_WIDTH` | `execution.beam_width` | `3` |
| `SYRCH_CANDIDATE_BUDGET` | `execution.candidate_budget` | `8` |
| `DATABRICKS_SERVER_HOSTNAME` | Databricks workspace URL | `dbc-...cloud.databricks.com` |
| `DATABRICKS_HTTP_PATH` | SQL Warehouse path | `/sql/1.0/warehouses/...` |
| `DATABRICKS_TOKEN` | Personal Access Token | `dapi...` |

Full list: see [Configuration section in AGENTS.md](AGENTS.md#config).

---

## How It Works

### Pipeline

```
ProblemSpec → Retriever → Planner → Scheduler → Aggregator → FinalSolution
```

### Candidate Scope (per-attempt)

```
PRIMARY         → candidate under test (FROM only)
JOIN-AVAILABLE  → pool-bounded supporting tables (JOIN only)
TASK CONTEXT    → _task_context_* materialized parents (FROM or JOIN)
```

- `JOIN-AVAILABLE` in `FROM` → rejected as **primary switch**.
- Unmaterialized `_task_context_*` (AMBIGUOUS/FAILED parent) → rejected.
- Only *known physical* tables are enforced; unknown tables surface as execution errors.

### Search Policy

1. Candidates ordered best-first by retriever prior; consumed one at a time.
2. Each candidate runs the REPL loop; failures surface as `ok=False` without terminating search.
3. `BeamSearchPolicy`: explores at least `beam_width`, at most `candidate_budget`; early stop only when best viable posterior beats second-best by `≥ stop_margin`.
4. Retriever prior never drives termination — a GT table at rank 5 is still explored.
5. Confidence is an aggregator output, never a search signal.

### Evidence Tiers (anti-cheat contract)

- **T1 schema** (`grain_match`, `dimension_match`, lexical feasibility)
- **T2 capability** (data the candidate *can* serve: time coverage via `TimeCoverage` probe, metric feasibility) — uniform, GT-free; `TimeCoverage` is the documented exception where probe facts *may* enter `time_match`
- **T3 outcome** (`result_quality`, `has_data`, all-NULL treated like empty) — VALUE probe facts ("which table holds value X") stay RLM-only and never enter ranking

### Calculation Basis

Deterministic, localized explanation derived from executed SQL + row counts — not from the LLM.

```
적용 기준 (실행된 SQL에서 도출):          ← localized header
- 시간 범위: 2024-01-01 .. 2024-12-31     ← literal bounds
- 환불/상태 제외: 미적용                   ← predicate presence
- 엔티티 상태 (SCD2): dw_customer: 고정 창 겹침 근사 ...

단계 1 — dw_customer
  필터: segment = 'VIP' AND valid_from <= '2024-12-31' ...
  결과: 782 rows
단계 2 — dw_sales_order (joined with 782 rows from step 1)
  조인: so.customer_id = ca.customer_id
  집계: SUM(total_amount) → 140,866.70 (1 row)
```

- Time-range detection, status-exclusion detection (`status != 'refunded'`), and SCD2 semantics classification (point-in-time vs fixed-window overlap vs ignored) are structural parses of the SQL.
- Step labels are localized to the question language; internals (table/column names, literal values) stay in English/schema form.
- Available via `result.calculation_basis` (library) and dim print in CLI.

---

## Confidence & Evidence

**Raw confidence** is the model's `Confidence: <0.0-1.0>` line (default `0.7` when omitted).

**Execution-signal penalties** → `CandidateEvaluation.result_quality`:

| Signal | Weight |
|--------|--------|
| `syntax_error` | −0.10 × count (cap 3) |
| `execution_error` | −0.10 × count (cap 3) |
| `empty_result` | −0.15 |
| `schema_error` | −0.05 × count (cap 3) |
| `null_column` | −0.05 |
| `overflow_result` | −0.05 |

**Aggregator confidence:**

```
best_conf = max(primary.confidence, selected_candidate.confidence)
adjusted  = best_conf × (1 − max_ambiguity × 0.5) × (1 − heuristic_penalty)
```

Heuristic penalties: empty +0.15, error +0.15, TOP-N mismatch +0.05, `by year` without year +0.10, AMBIGUOUS +0.10, FAILED/BLOCKED +0.15 each (cap 0.40).

---

## Directory Structure

<details>
<summary><b>Click to expand</b></summary>

```
src/syrch/
├── __init__.py          # Public API: query
├── api.py               # query() high-level function
├── cli/app.py           # Typer CLI
├── core/
│   ├── models.py        # Dataclasses (ProblemSpec, TaskDAG, CandidateEvaluation …)
│   ├── config.py        # ExecutionConfig + loader
│   └── logging.py       # Structured logging
├── executors/           # BaseExecutor + SQLite/JDBC/Databricks/Spark (+ CachedExecutor)
├── llm/                 # BaseLLM + OpenAI/Anthropic (+ CachedLLM / CentralCache)
├── search/
│   ├── planner.py       # D&C decomposition + replan
│   ├── scheduler.py     # DAG execution + context materialization
│   ├── rlm_engine.py    # Candidate search REPL loop
│   ├── search_policy.py # Beam / exhaustive
│   ├── path_evaluator.py# Quality scores + ranking signals
│   ├── validator.py     # Hard constraint checks
│   ├── retriever.py     # Keyword scoring + candidate pool
│   ├── semantic_index.py# Optional embedding-based index
│   ├── data_probe.py    # Shared probe cache (VALUE vs COVERAGE)
│   ├── question_norm.py # Non-English question normalization
│   ├── aggregator.py    # Result merge + localized calculation basis
│   ├── calibrator.py    # Execution penalties
│   ├── clarify.py       # Ambiguity detection
│   ├── grid.py          # Hyperparameter grid search
│   └── pipeline.py      # Orchestrator
└── eval/                # Benchmark harness + metrics
tests/                  # Unit + integration tests
scripts/                # validate_real.py, gen_fixtures.py
```

</details>

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
pytest tests/ -v --cov=src/syrch/

# Lint & type check
ruff check src/syrch/
mypy src/syrch/ --ignore-missing-imports

# Real LLM validation (requires API key)
python scripts/validate_real.py --quick
python scripts/validate_real.py --question "..." --db orders_10dim.sqlite --verbose
```

**CI** (push/PR → `main`): `ruff` → `mypy` → `pytest` (Python 3.11 + 3.12).

---

## Benchmark & Evaluation

```bash
# Single question with expected CSV
syrch eval -q "Which click type generates the most traffic?" \
  --db wikipedia_clickstream.sqlite --expected expected.csv

# Full suite
syrch benchmark --file benchmarks/orders.jsonl --report report.md
```

Grid search sweeps `max_depth`, `beam_width`, `max_attempts_per_node`, `calibration_enabled` (54 cells) and reports `exact_match > confidence`.

---

## Contributing

See [AGENTS.md](AGENTS.md) for module responsibilities and design principles.

Branch policy: `main` (stable, PR only) → `feat/*` / `fix/*` (squash merge) → `release/v*` (merge commit, version bump in `pyproject.toml` + `src/syrch/__init__.py`, tag `v*` → PyPI via `publish.yml`).

---

## Research Background

- **RLM** (Recursive Language Model) — MIT CSAIL OASIS 2025. [`paper`](https://arxiv.org/abs/2512.24601) [`code`](https://github.com/alexzhang13/rlm)
- **RDD** — Recursive Decomposition with Dependencies. [`paper`](https://arxiv.org/abs/2505.02576)
- **PAC-MCTS**, **ROMA**, **Graph Harness**, **AdaptOrch**, **DST**, **LLM Compiler** — see [AGENTS.md](AGENTS.md#research-background) for full list.

---

## License

[MIT](LICENSE)
