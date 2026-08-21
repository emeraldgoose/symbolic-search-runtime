# syrch — Symbolic Search Runtime

**English** | [[한국어]](README_ko.md)

NL Problem → ProblemSpec → Search(D&C+RLM) → SQL Executor → Optimal Solution

## Project Goal

`symbolic-search-runtime` (syrch) is a **search harness** that finds optimal solutions to natural language problems over structured data. Unlike a simple QA agent that answers in one shot, syrch **explores multiple reasoning paths** using Divide & Conquer decomposition and Recursive Language Models, executing candidate solutions against real databases to select the best result.

### Key Ideas

- **Divide & Conquer**: Decompose a problem into logically independent sub-problems (sub-tasks), solve each independently, then merge results. Sub-problems can depend on each other forming a DAG.
- **RLM (Recursive Language Model)**: Each sub-task runs its own REPL loop — generate SQL → validate syntax → validate schema → execute → quality check → evaluate evidence. Multiple reasoning paths are explored per node.
- **Evidence-based Selection**: The RLM explores a candidate pool (best-first by retriever prior), evaluates each candidate on discrimination signals (`structural_match`, `grain_match`, `dimension_match`, `time_match`, `result_quality`), and selects lexicographically. Ties on all signals → **AMBIGUOUS** (never resolved by execution/retriever/cost order).
- **Candidate Scope**: Each attempt is confined to `PRIMARY` (the candidate under test), `JOIN-AVAILABLE` (pool-bounded supporting tables), and `TASK CONTEXT` (materialized parent results). FROM must anchor on PRIMARY/TASK CONTEXT; JOIN may use PRIMARY/JOIN-AVAILABLE/TASK CONTEXT.
- **Grid Search**: Systematic hyperparameter testing (`max_depth`, `beam_width`, `max_attempts_per_node`, `calibration_enabled`) to find optimal configs.
- **Multi-table Schema**: Retriever scores all tables by relevance, Planner selects per subtask, and RLM sees only the per-attempt scope (not all tables).
- **Search over reasoning, not execution**: D&C splits the *problem space*, not the SQL. Each sub-problem is a complete reasoning unit (think → code → validate → execute → evaluate).
- **Pluggable Executors**: Abstract `BaseExecutor` with SQLite, JDBC, Spark, and Databricks implementations.
  Any database that speaks a standard query interface can be plugged in.
- **Step-by-step Evidence**: The final answer includes a trace that walks through the
  task graph in execution order — which table was queried, what filter was applied,
  how many rows were found, and how the result was aggregated — so the calculation
  can be verified without trusting the language model.

## Architecture

```
                    USER QUESTION
                         │
                         ▼
┌──────────────────────────────────────────┐
│  Planner (D&C) — "what to solve?"        │
│  Question → RequirementSpec + TaskDAG    │
│  (each task: metrics, dims, grain,       │
│   filters, supporting relations)         │
└────────────────────┬─────────────────────┘
                     │ requirements + task graph
                     ▼
┌──────────────────────────────────────────┐
│  Retriever — "where to look?"            │
│  scores every table → candidate pool     │
│  (ordering only — never the final pick)  │
└────────────────────┬─────────────────────┘
                     │ candidate pool
                     ▼
┌──────────────────────────────────────────┐
│  Scheduler — runs tasks in DAG order     │
│                                         │
│  for each node:                         │
│  ┌───────────────────────────────────┐  │
│  │  RLM Agent — "how to run it?"     │  │
│  │  candidate → SQL → validate →     │  │
│  │  execute → evaluate → select      │  │
│  │  (beam/exhaustive, lexicographic) │  │
│  └──────────────────┬────────────────┘  │
└─────────────────────┼────────────────────┘
                      │ SOLVED (best viable candidate)
                      ▼
┌──────────────────────────────────────────┐
│  materialize → _task_context_<node_id>   │
└────────────────────┬─────────────────────┘
                     │ real table; dependents JOIN it
                     ▼
┌──────────────────────────────────────────┐
│  Aggregator — "compose the answer"       │
│  primary leaf, no re-ranking;            │
│  builds step-by-step evidence chain      │
└────────────────────┬─────────────────────┘
                     ▼
        FINAL ANSWER + SQL + step-by-step evidence + reasoning trace
```

Node outcomes: `SOLVED` → a real `_task_context_<id>` table is materialized
that dependent tasks can JOIN. `AMBIGUOUS` → **no materialization** (uncertain
results are never made into a downstream fact). `FAILED`/`BLOCKED` →
consumers short-circuit to `BLOCKED` without running SQL.

### How a Sub-Task Executes (RLM Node)

Each node explores candidates under a pluggable search policy (`beam` by default, or `exhaustive`):

```
Node "Refund counts by reason"
    │
    ├── Candidate dw_sales_order (retriever prior 0.8)
    │   ├── [PASS] Syntax check (sqlglot)
    │   ├── [PASS] Schema column check (status, total_amount)
    │   ├── [PASS] Scope check (FROM = PRIMARY)
    │   ├── [PASS] Execute → 3 rows
    │   └── viable: structural=1.0, grain=1.0, result_quality=1.0
    │
    ├── Candidate mart_sales_daily (retriever prior 0.6)
    │   ├── [PASS] Syntax check
    │   ├── [FAIL] Schema: no status column → non-viable
    │   └── rejected (no "refund" dimension)
    │
    └── Top-1 vs top-2 differ → SOLVED (dw_sales_order)
        If top-2 ties on all signals → AMBIGUOUS (never resolved by cost)
```

## Directory Structure

```
syrch/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── LICENSE
├── .gitignore
├── src/syrch/
│   ├── __init__.py               # Public API: query, SearchResult
│   ├── api.py                    # query() high-level function
│   ├── cli/
│   │   ├── __init__.py
│   │   └── app.py                # Typer CLI
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py             # Data types (dataclasses)
│   │   ├── config.py             # ExecutionConfig + config loader
│   │   └── logging.py            # Structured logging
│   ├── executors/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseExecutor (ABC)
│   │   ├── sqlite_executor.py    # SQLite
│   │   ├── jdbc_executor.py      # JDBC via SQLAlchemy
│   │   ├── databricks_executor.py # Databricks SQL
│   │   ├── spark_executor.py     # SparkSession (Databricks/EMR/standalone)
│   │   └── cached_executor.py    # diskcache-backed SQL cache
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseLLM (ABC)
│   │   ├── openai_llm.py         # OpenAI
│   │   ├── anthropic_llm.py      # Anthropic Claude
│   │   └── cache.py              # CachedLLM + CentralCache
│   ├── search/
│   │   ├── __init__.py
│   │   ├── planner.py            # D&C: NL -> TaskDAG (+ replan)
│   │   ├── scheduler.py          # DAG execution + context materialization
│   │   ├── rlm_engine.py         # RLM candidate search loop
│   │   ├── search_policy.py      # Beam / exhaustive candidate policy
│   │   ├── path_evaluator.py     # Quality scores + ranking signals
│   │   ├── validator.py          # Hard constraint checks
│   │   ├── retriever.py          # Keyword scoring + candidate pool
│   │   ├── semantic_index.py     # Optional embedding-based semantic index
│   │   ├── data_probe.py         # Data-level fact verification (shared probe cache)
│   │   ├── question_norm.py      # Non-English question normalization
│   │   ├── aggregator.py         # Result merge + step-by-step evidence
│   │   ├── calibrator.py         # Execution penalties
│   │   ├── clarify.py            # Ambiguity detection + clarification
│   │   ├── grid.py               # Grid search
│   │   └── pipeline.py           # Orchestrator
│   └── eval/
│       ├── __init__.py
│       ├── runner.py             # Benchmark harness
│       ├── metrics.py            # Evaluation metrics
│       └── report.py             # Report export
├── scripts/
│   ├── gen_fixtures.py           # Generate test fixture DBs
│   └── validate_real.py          # Real LLM validation
└── tests/
    ├── test_api.py
    ├── test_cache.py
    ├── test_clarify.py
    ├── test_discrimination.py
    ├── test_e2e.py
    ├── test_eval.py
    ├── test_integration.py
    ├── test_materialize.py
    ├── test_planner.py
    ├── test_rlm_engine.py
    ├── test_scheduler.py
    ├── test_search_policy.py
    └── test_validator.py
```

## Data Model

```
ProblemSpec { question, schema, all_schemas, goal_metric }
    │
    ▼
TaskDAG { nodes: {A, B, C, ...}, root_id, topo_layers }
    │  Each TaskNode: { id, description, depends_on, is_atomic,
    │                   requirements (metrics/dimensions/filters/grain),
    │                   hint_tables, hint_columns, join_keys }
    ▼
Scheduler → NodeResult { node_id, data(DataFrame), sql, confidence,
                         selected_candidate, reasoning_paths,
                         cost_tokens, status(SOLVED/AMBIGUOUS/FAILED/BLOCKED),
                         had_context, context_used }
    │
    ▼
Aggregator → FinalSolution { answer, sql, confidence, data, token_cost, tree,
                              calculation_basis }
             (primary leaf = SOLVED with evidence; no re-ranking;
              calculation_basis = step-by-step evidence chain)
```

`calculation_basis` is a plain-text trace derived from the executed SQL and row counts
(walking the DAG layer by layer), not from the language model. For example:

```
Step 1 — dw_customer
  filter: segment = 'VIP' AND valid_from <= '2024-12-31' ...
  result: 1,247 rows
Step 2 — dw_sales_order (joined with 1,247 rows from step 1)
  join: v.customer_id = dso.customer_id
  filter: order_date BETWEEN '2024-01-01' AND '2024-12-31'
  aggregate: SUM(total_amount) → 140,866.70 (1 row)
```

It lets users verify *how* a number was computed by showing the table, columns,
filter criteria and aggregation at each step — with no assumption about table
structure or SCD modeling.

Node-level selection produces a `CandidateEvaluation` per explored candidate:

```
CandidateEvaluation {
    table, ok, execution_valid, requirement_pass, has_data,
    semantic_match, result_quality,
    structural_match, grain_match, dimension_match, time_match,
    cost_tokens, candidate_id, confidence, error, attempts, path_score
}
viable = ok && has_data && execution_valid && requirement_pass
ranking order: structural_match → grain_match → dimension_match →
               time_match → result_quality → candidate_id
```

The first five signals are called *ranking signals* — they compare how well a
candidate matches the task requirements. A tie between the top two viable
candidates on all ranking signals ⇒ **AMBIGUOUS** — the system does not pick a
winner based on run order, keyword overlap, or how cheap the query was.

## Installation

```bash
# Core (CLI + SQLite)
pip install syrch

# Databricks SQL Warehouse (external connection)
pip install "syrch[databricks-sql]"

# Spark executor (Databricks Runtime, EMR, standalone)
pip install "syrch[spark]"

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
    executor_type="databricks-sql",
    model="gpt-4o",
)
print(result.answer)              # Final answer text
print(result.sql)                 # Executed SQL (all steps)
print(result.confidence)          # Confidence score (0-1)
print(result.data)                # Result DataFrame
print(result.calculation_basis)   # Step-by-step evidence chain
print(result.tables_used)         # Physical tables actually queried
print(result.tree)                # Per-task results (NodeResult list)
print(result.dag_nodes)           # Task graph structure
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
  --max-attempts 3 \
  --search-policy beam \
  --verbose

# Grid search over hyperparameters
syrch search -q "..." --db orders_10dim.sqlite --grid

# Benchmark against expected results
syrch eval -q "..." --db orders_10dim.sqlite --expected expected.csv

# Run benchmark suite
syrch benchmark --file benchmarks/orders.jsonl
```

### CLI Reference

| Command | Option | Description |
|---------|--------|-------------|
| `search` | `-q` / `--question` | Natural language problem (required) |
| | `--db` | Database path (default: `orders_10dim.sqlite`) |
| | `--max-depth` | Max D&C recursion depth (default: 3) |
| | `--executor` | `sqlite` / `databricks-sql` / `spark` / `jdbc` |
| | `--max-attempts` | Max RLM attempts per node (default: 3) |
| | `--search-policy` | Candidate search policy: `beam` / `exhaustive` (default: beam) |
| | `--beam-width` | Min candidates explored before early stop (default: 3) |
| | `--candidate-budget` | Max candidates explored per node (default: 8) |
| | `--stop-margin` | Posterior gap required to stop early (default: 0.15) |
| | `--budget` | Token budget (default: 100000) |
| | `--llm` | `openai` / `anthropic` |
| | `--model` | LLM model name (default: `qwen3.5-4b-4bit`) |
| | `-v` / `--verbose` | Show reasoning traces |
| | `--cache/--no-cache` | Enable/disable LLM + SQL cache (default: on) |
| | `--cache-ttl` | Cache TTL in seconds (default: 86400) |
| | `--grid` | Run grid search over hyperparameters |
| | `--grid-parallel/--grid-sequential` | Parallel vs sequential grid execution |
| | `--grid-max-workers` | Max concurrent API calls (default: 3) |
| | `--interactive/--no-interactive` | Ask clarification questions when SQL cannot solve the task |
| | `--config` | Path to YAML config file (`syrch.yml` or `~/.syrch/config.yml`) |
| `eval` | `-q` | Question |
| | `--db` | Database path |
| | `--executor` | Executor type |
| | `--expected` | Expected results CSV |
| `benchmark` | `--file` | JSONL benchmark file |
| | `--executor` | Executor type |
| | `--report` | Output report path |
| | `--report-format` | `md` / `json` |
| `schema` | `DB` | Database path (positional) |
| | `-t` / `--table` | Specific table |
| `config` | `--db` | Database path |

## Configuration

Config loaded from (priority order): **CLI args > config file > env vars (`SYRCH_*`) > Databricks Secrets > defaults**.

### Config File (`syrch.yml`)

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

Search paths: `./syrch.yml` > `./syrch.yaml` > `~/.syrch/config.yml` > `~/.syrch/config.yaml` (`--config <path>` overrides all)

### Environment Variables

| Variable | Maps to | Example |
|----------|---------|---------|
| `SYRCH_MODEL` | `llm.model` | `gpt-4o` |
| `SYRCH_API_KEY` | `llm.api_key` | `sk-...` |
| `SYRCH_BASE_URL` | `llm.base_url` | `http://localhost:11434/v1` |
| `SYRCH_LLM_PROVIDER` | `llm.provider` | `openai` |
| `SYRCH_TEMPERATURE` | `llm.temperature` | `0.7` |
| `SYRCH_MAX_TOKENS` | `llm.max_tokens_per_call` | `4096` |
| `SYRCH_TIMEOUT` | `llm.timeout_seconds` | `120` |
| `SYRCH_EXECUTOR` | `execution.executor_type` | `sqlite` |
| `SYRCH_MAX_DEPTH` | `execution.max_depth` | `3` |
| `SYRCH_MAX_ATTEMPTS` | `execution.max_attempts_per_node` | `3` |
| `SYRCH_TOKEN_BUDGET` | `execution.token_budget` | `100000` |
| `SYRCH_VERBOSE` | `execution.verbose` | `true` |
| `SYRCH_CACHE` | `execution.cache_enabled` | `true` |
| `SYRCH_CACHE_TTL` | `execution.cache_ttl` | `86400` |
| `SYRCH_SEARCH_POLICY` | `execution.search_policy` | `beam` |
| `SYRCH_BEAM_WIDTH` | `execution.beam_width` | `3` |
| `SYRCH_CANDIDATE_BUDGET` | `execution.candidate_budget` | `8` |
| `SYRCH_STOP_MARGIN` | `execution.stop_margin` | `0.15` |
| `SYRCH_MAX_CANDIDATE_EXPANSION` | `execution.max_candidate_expansion` | `2` |
| `SYRCH_MAX_REPLANS` | `execution.max_replans` | `1` |
| `SYRCH_CALIBRATION` | `execution.calibration_enabled` | `true` |
| `SYRCH_MATERIALIZE_CONTEXT` | `execution.materialize_context` | `true` |
| `SYRCH_INTERACTIVE` | `execution.interactive` | `true` |
| `SYRCH_AMBIGUITY_THRESHOLD` | `execution.ambiguity_threshold` | `0.35` |
| `SYRCH_MAX_CONCURRENCY` | `execution.max_concurrency` | `5` |

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
| `DATABRICKS_CATALOG` | all | Catalog name (optional) |
| `DATABRICKS_SCHEMA` | all | Schema name (optional) |

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

## Confidence

Confidence is an **output signal for the aggregator** — it no longer drives
search termination. The RLM reads the model's `Confidence: <0.0-1.0>` line
(default `0.7` when omitted) and stores it as raw confidence.

### Execution-signal penalties

Applied via `PathEvaluator` and surfaced as `CandidateEvaluation.result_quality`
(execution signals no longer multiply confidence directly):

| Signal | Weight | Effect |
|--------|--------|--------|
| `syntax_error` | 0.10 | −0.10 per occurrence (capped ×3) |
| `execution_error` | 0.10 | −0.10 per occurrence (capped ×3) |
| `empty_result` | 0.15 | −0.15 if result is empty |
| `schema_error` | 0.05 | −0.05 per occurrence (capped ×3) |
| `null_column` | 0.05 | −0.05 if result has all-NULL columns |
| `overflow_result` | 0.05 | −0.05 if result is oversized |

### Aggregator confidence

```
primary = leaf with selected_candidate evidence (SOLVED > AMBIGUOUS; no re-rank)
best_conf = max(primary.confidence, selected_candidate.confidence)
adjusted_conf = best_conf × (1.0 − max_ambiguity × 0.5) × (1.0 − heuristic_penalty)
```

**Heuristic penalties** (aggregator):
- Empty result: +0.15 per node
- Error present: +0.15 per node
- TOP-N mismatch: +0.05 per node
- "by year" without year column: +0.10 (once, global)
- AMBIGUOUS leaf: +0.10; FAILED/BLOCKED leaf: +0.15 each
- **Capped at 0.40 total**

`calibration_enabled` (default `True`, env `SYRCH_CALIBRATION`) toggles the
execution-signal path in the evaluator.

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
| beam_width | 2, 3, 5 |
| max_attempts_per_node | 1, 3, 5 |
| calibration_enabled | True, False |

Output: `autoresearch/reports/{YYYYMMDD-HHMMSS}/{config,results,best}.json` + `summary.md`

Best config selection: `exact_match > confidence` (cells with errors are skipped).

## Search Policy (candidate termination)

The RLM engine explores candidate tables under a pluggable `SearchPolicy`
(`beam` by default, or `exhaustive`):

1. Candidates are ordered by retriever prior (best-first) and consumed one at a time.
2. Each candidate runs the RLM REPL loop: generate SQL → validate syntax → validate schema → execute → quality check → evaluate `PathScore`.
3. Recoverability and requirement failures surface as `ok=False` / zero PathScore; they do not terminate the search.
4. `BeamSearchPolicy` explores at least `beam_width` candidates and at most `candidate_budget`. After the beam floor, it stops early only when the best *viable* posterior beats the second-best viable posterior by ≥ `stop_margin`.
5. The retriever prior is used only to order candidates, never to terminate — so a ground-truth table ranked far down the pool (e.g. `dw_sales_order` at rank 5) is still explored.
6. Confidence is an output for the aggregator; it no longer drives termination (no confidence threshold, no execution-based auto-boost).

This replaces the old "calibrated confidence ≥ 0.85 → greedy accept" rule. Simple problems still resolve quickly (small pools), while ambiguous ones explore more candidates within the budget.

## Candidate Scope

Each RLM attempt builds an explicit per-attempt schema scope (`AttemptSchemaContext`):

```
PRIMARY         → the candidate under test (FROM anchor only)
JOIN-AVAILABLE  → pool-bounded supporting tables (JOIN only, never FROM)
TASK CONTEXT    → materialized parent results (_task_context_*, FROM or JOIN)
```

Position rules:

```
FROM  → PRIMARY | TASK CONTEXT (materialized)
JOIN  → PRIMARY | JOIN-AVAILABLE | TASK CONTEXT
```

- A JOIN-AVAILABLE table in FROM is rejected as a **primary switch** with actionable feedback.
- `_task_context_*` names that were never materialized (dependency AMBIGUOUS/FAILED) are rejected — only SOLVED parents materialize.
- Only *known physical* tables are enforced for drift; unknown tables surface as execution errors.
- The candidate pool is the only source of join candidates — drift to an out-of-pool table is blocked.

## Module Responsibilities

The whole design rests on one principle: **each layer never invades the
responsibility of the layers above or below it.**

| Module | Core question | Input | Output |
|--------|---------------|-------|--------|
| Planner | What to solve? | Question + Schema | RequirementSpec + DAG |
| Retriever | Where to look? | Requirement + Schema | Candidate Pool (ranked by relevance) |
| SemanticIndex | What schema evidence exists? | DB metadata | Semantic hints for scoring |
| DataProbe | Does this value actually exist in the database? | Candidate tables + filter values | Verified facts (shared across all tasks) |
| Pipeline | How do all stages connect? | LLM + Executor + Config + Problem | FinalSolution + DAG + per-task results |
| TaskDAG | How to split work? | RequirementSpec | DAG (layers of dependent tasks) |
| Scheduler | In what order? | DAG + Executor | Node execution (parallel within each layer) |
| RLM | How to execute? | Task + Candidate scope | SQL candidates (one attempt per candidate) |
| Validator | Is the SQL allowed? | SQL + Schema + Scope | Pass/Fail |
| Executor | Run the SQL | Valid SQL | Data (rows + columns) |
| Materializer | Make parent results reusable | Task result | Temporary table (`_task_context_<id>`) |
| Evaluator | Does the candidate meet the requirement? | Requirement + SQL + Result | Quality scores (ranking signals) |
| Selection | Which candidate is adopted? | Quality scores | Chosen candidate or AMBIGUOUS (tie) |
| Replanner | Should we try a different plan? | Failure / Ambiguity | Expanded candidate pool |
| Aggregator | What is the final answer? | All task results | Answer + SQL + step-by-step evidence |

Four core responsibilities are always kept separate:

```
Planner     → "what to solve?"      (what the question requires)
RLM         → "how to execute?"     (how to get it from the database)
Evaluator   → "does it satisfy?"    (how well the result matches the requirement)
Aggregator  → "how to compose?"     (combine task results into one answer)
```

v0.3.5b adds a fifth layer between RLM and Aggregator:

```
Scheduler / Executor → "how to pass results between tasks?" (Context/Dataflow)
```

Node states: a dependency that is `FAILED`/`BLOCKED` short-circuits its
consumers to `BLOCKED` (no SQL run); `AMBIGUOUS` never materializes a context,
so uncertain results are never made into a downstream fact.

## Caching

All LLM and SQL calls are cached via `diskcache` (24h TTL):

| Layer | Cache | Key |
|-------|-------|-----|
| LLM `generate()` | `CachedLLM` | SHA256(system + user + model + temperature) |
| LLM `generate_json()` | `CachedLLM` | SHA256(system + user + model + temperature) |
| SQL `execute()` | `CachedExecutor` | SHA256(sql) |

Toggle with `--cache/--no-cache` flag; TTL configurable with `--cache-ttl`.

## Research Background

- **RLM (Recursive Language Model)**: MIT CSAIL OASYS Lab, 2025. Inference paradigm where LLMs recursively decompose input via REPL environments. [`paper`](https://arxiv.org/abs/2512.24601) [`code`](https://github.com/alexzhang13/rlm)
- **RDD (Recursive Decomposition with Dependencies)**: Formal D&C framework with dependency DAGs. [`paper`](https://arxiv.org/abs/2505.02576)
- **PAC-MCTS**: Bias-aware pruning with formal guarantees for tree search. [`paper`](https://arxiv.org/abs/2604.14345)
- **ROMA**: Recursive meta-agent framework with Atomizer/Planner/Executor/Aggregator roles. [`paper`](https://arxiv.org/abs/2602.01848)
- **Graph Harness**: Structured DAG execution with immutable plan versions. [`paper`](https://arxiv.org/abs/2604.11378)
- **AdaptOrch**: Topology-aware multi-agent orchestration (parallel/sequential/hierarchical/hybrid). [`paper`](https://arxiv.org/abs/2602.16873)
- **DST**: Adaptive tree search with confidence-based pruning (26-75% computation reduction). [`paper`](https://arxiv.org/abs/2603.20267)
- **LLM Compiler**: Parallel task scheduling via dependency graphs; closely related to syrch's DAG scheduler and layer-by-layer execution. [`paper`](https://arxiv.org/abs/2312.13311)
