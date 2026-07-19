# syrch Benchmark Platform — Project Plan

## Project Goal

End-to-end Text-to-SQL Benchmark Platform that evaluates the full pipeline:
```
Question → Schema Retriever → Planner (DAG) → SQL Generator → Executor → Evaluator
```

Evaluation targets: **Retriever**, **Planner**, and **SQL Generator** independently — not just the final SQL.

---

## Current Status (Phase 1 Complete ✅, Phase 2 In Progress)

### What exists

#### Generator (`benchmark/generator/`)
- **35 tables** across 5 YAML schema files (Sales, Customer, Marketing, Metadata, Noise)
- **ODS/DW/Mart/Legacy** layer structure with real-world complexity
- Quality injection: nulls, FK breaks, outliers, duplicates, late-arriving data
- **SQLiteBackend** (working) + **DeltaBackend** (working, local & Databricks)
- Profiles: small (50K rows), medium (3M), enterprise (10M), databricks-enterprise
- Fully generated: `benchmark/generated/sqlite/benchmark_small.sqlite` (104MB, 31 tables)

#### Questions (`benchmark/questions/`)
- `pilot.json` — 15 questions (L1-L5) covering revenue, funnel, SCD, churn, cross-domain, temporal, semantic
- Each question includes: `ground_truth_sql`, `required_tables`, `dag_evaluation`, `grain`, error taxonomy fields

#### Evaluator (`benchmark/evaluator/`)
| Component | File | Status |
|-----------|------|--------|
| Ground Truth Generator | `generate_ground_truth.py` | ✅ Executes `ground_truth_sql` → CSV saved to `generated/answers/pilot/{id}.csv` |
| Error Classifier | `errors.py` | ✅ Rule-based Error Taxonomy (10 types) |
| Benchmark Runner | `run_benchmark.py` | ✅ Multi-executor (sqlite/databricks-sql/spark), `--base-url`, `--api-key`, `gt_dir`, importable from notebooks |
| Old Validator | `validate_business.py` | ✅ Backward compatible |

#### DeltaBackend (`benchmark/generator/backends/delta.py`)
- `spark_session` param for Databricks cluster reuse
- `mode="local"` bypasses Databricks Connect for local Spark
- `CREATE CATALOG` wrapped in try/except for Unity Catalog permissions
- Fully qualified table names in DDL (`catalog.schema.table`)
- Databricks generation notebook: `benchmark/generator/gen_on_databricks.py`

#### DatabricksExecutor (`src/syrch/executors/databricks_executor.py`)
- `USE CATALOG` / `USE SCHEMA` before each SQL execution for short-name resolution
- Connection params from env vars (`DATABRICKS_*`)

#### OpenAILLM (`src/syrch/llm/openai_llm.py`)
- Local model support: dummy key (`sk-no-key-required`) when `base_url` is set

#### Data Generation Fixes
| Issue | Fix | Status |
|-------|-----|--------|
| S5: `sale_month` had 'A','B','C' | Added 36 month values to `sales.yaml` + `replace=False` sampling + improved TEXT fallback | ✅ |
| S13: FK `product→category` 0% match | FK resolution via `fk_registry` + topological sort (Kahn's algorithm) | ✅ |

### Validation Results

#### SQLite dry-run (15 questions)
```
Retriever: Avg Recall 1.000, Avg Precision 1.000
Planner:   Minimal DAG 13/15 (dry-run limitation)
SQL:       Execution 15/15, Result Match 15/15
```

#### Databricks live run (qwen3.5-4b-4bit, 15 questions)
```
Retriever: Avg Recall ~0.70, Avg Precision ~0.57
SQL:       Execution 15/15, Result Match 0/15 (GT CSV not available on Databricks)
Avg Confidence: ~0.65
Avg Duration: ~185s/q
```

### Known Issues

| Category | Issue | Impact |
|----------|-------|--------|
| Ground Truth | GT CSV not on Databricks → `result_match` always False | Fixed via `gt_dir` param |
| Retriever | Model selects wrong tables (e.g. `mart_sales_daily` instead of `dw_sales_order` for revenue) | Needs better schema descriptions |
| Time Range | Model uses `<= '2024-12-31'` vs ground truth `< '2025-01-01'` | Creates false `wrong_time` errors |
| Planner | D&C over-decomposition (8 nodes for simple query) | Needs `max_depth` tuning or node merging |
| SCD | SCD2 join (`valid_from`/`valid_to`) not detected | Missing from training/system prompt |

---

## Next Steps (Planned Phases)

### Phase 2 — Schema & Question Expansion (In Progress)

**Goal**: Scale from 35→~90 tables, 15→120 questions

1. **Schema expansion**
   - Finance domain (`schema/finance.yaml`)
   - Inventory domain (`schema/inventory.yaml`)
   - SCD Type 2 expansion (Product, Supplier, Campaign)
   - Naming drift tables (customer_id / cust_id / client_id / member_no)
   - Additional ODS/DW/Mart/Legacy layer tables

2. **Question bank expansion**
   - `questions/` directory with escalating difficulty files
   - New groups: cross-domain, temporal, semantic, ambiguous
   - Each question in easy→medium→hard→very-hard progression

3. **Generate medium/enterprise databases**
   - `benchmark_medium.sqlite` (3M rows)
   - `benchmark_enterprise.sqlite` (10M rows)

### Phase 3 — Databricks Validation (Next)

1. **Ground truth generation on Databricks**
   - Script to execute `ground_truth_sql` against Databricks SQL Warehouse
   - Save CSVs to DBFS/Volumes for `gt_dir` consumption

2. **Cross-backend consistency**
   - SQLite result == Delta result (same seed)
   - Validation run against both backends

3. **Benchmark report automation**
   - Per-model comparison: qwen3.5 vs gpt-4o vs claude
   - JSON export with full error taxonomy breakdown

### Phase 4 — CI & Report System

1. **GitHub Actions workflow** (`.github/workflows/benchmark.yml`)
   - On PR: generate DB, run `run_benchmark.py --quick`
   - Output results to GitHub Summary
   - Regression detection vs previous run

2. **Model quality improvements**
   - Improved planner system prompt for better table selection
   - Few-shot examples for time range boundary handling
   - SCD2 join detection
   - D&C decomposition depth limits

3. **Hyperparameter grid** (`search/grid.py` integration)
   - Cross-product runs: model × max_depth × max_attempts × calibration

---

## Branch Strategy

Follows `AGENTS.md` convention:

| Branch | Purpose |
|--------|---------|
| `main` | Stable releases |
| `feat/*` | Feature branches → squash merge to `main` |
| `fix/*` | Bug fixes → squash merge to `main` |
| `release/v*` | Version bumps → merge commit to `main` |

## Usage

### Local (SQLite)
```bash
# Ground truth generation
python benchmark/evaluator/generate_ground_truth.py --profile small

# Dry-run evaluation
python benchmark/evaluator/run_benchmark.py --profile small --questions pilot --dry-run

# Full evaluation (requires API key)
python benchmark/evaluator/run_benchmark.py --profile small --model gpt-4o --api-key $OPENAI_API_KEY

# Quick check (L1-L2 only)
python benchmark/evaluator/run_benchmark.py --quick

# Export results
python benchmark/evaluator/run_benchmark.py --dry-run --export reports/pilot.json
```

### Databricks
```bash
# Dry-run against Databricks SQL Warehouse
python benchmark/evaluator/run_benchmark.py \
    --profile databricks-enterprise \
    --executor databricks-sql \
    --dry-run

# Full evaluation
python benchmark/evaluator/run_benchmark.py \
    --profile databricks-enterprise \
    --executor databricks-sql \
    --model qwen3.5-4b-4bit \
    --base-url http://localhost:11434/v1
```

### From Notebook
```python
from benchmark.evaluator.run_benchmark import run_benchmark, print_summary

results = run_benchmark(
    profile_name="databricks-enterprise",
    executor_type="databricks-sql",
    dry_run=True,
    gt_dir="/Workspace/Users/me/benchmark/generated/answers/pilot",
)
print_summary(results)
```

### Delta Table Generation
```bash
# Local Spark (pyspark + delta-spark required)
python benchmark/generator/gen_business_db.py --profile databricks-enterprise --backend delta

# Databricks notebook: upload benchmark/generator/gen_on_databricks.py and run
```
