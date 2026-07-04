# syrch Benchmark Platform — Project Plan

## Project Goal

End-to-end Text-to-SQL Benchmark Platform that evaluates the full pipeline:
```
Question → Schema Retriever → Planner (DAG) → SQL Generator → Executor → Evaluator
```

Evaluation targets: **Retriever**, **Planner**, and **SQL Generator** independently — not just the final SQL.

---

## Current Status (Phase 1 Complete ✅)

### What exists

#### Generator (`benchmark/generator/`)
- **31 tables** across 5 YAML schema files (Sales, Customer, Marketing, Metadata, Noise)
- **ODS/DW/Mart/Legacy** layer structure with real-world complexity
- Quality injection: nulls, FK breaks, outliers, duplicates, late-arriving data
- **SQLiteBackend** (working) + **DeltaBackend** (working, needs pyspark)
- Profiles: small (50K rows), medium (3M), enterprise (10M)
- Fully generated: `benchmark/generated/sqlite/benchmark_small.sqlite` (104MB, 31 tables)

#### Questions (`benchmark/questions/`)
- `pilot.json` — 15 questions (L1-L5) covering revenue, funnel, SCD, churn, cross-domain, temporal, semantic
- Each question includes: `ground_truth_sql`, `required_tables`, `dag_evaluation`, `grain`, error taxonomy fields

#### Evaluator — Phase 1 (`benchmark/evaluator/`)
| Component | File | Status |
|-----------|------|--------|
| Ground Truth Generator | `generate_ground_truth.py` | ✅ Executes `ground_truth_sql` → CSV saved to `generated/answers/pilot/{id}.csv` |
| Error Classifier | `errors.py` | ✅ Rule-based Error Taxonomy (10 types): wrong_table, wrong_column, wrong_grain, wrong_join, wrong_filter, wrong_time, wrong_scd, wrong_semantic, planner_error, retriever_error |
| Benchmark Runner | `run_benchmark.py` | ✅ End-to-end: loads questions → runs `syrch.query()` → compares results → classifies errors → 3-axis report. Supports `--dry-run`, `--quick`, `--export JSON` |
| Old Validator | `validate_business.py` | ✅ Backward compatible (ground truth SQL validation only) |

#### syrch Core Changes
- `SearchResult` now includes `dag_nodes` and `tables_used`
- CTE-aware table extraction in all regex-based table parsers
- Metrics extended with error taxonomy fields

### Validation Results (dry-run, 15 questions)

```
Retriever: Avg Recall 1.000, Avg Precision 1.000
Planner:   Minimal DAG 13/15 (dry-run limitation)
SQL:       Execution 15/15, Result Match 15/15
Errors:    2 planner_error (dry-run: CTE-based SQL can't infer multi-node DAG)
```

### Known Data Issues

| Issue | Impact | Root Cause |
|-------|--------|------------|
| S5: `sale_month` has 'A','B','C' instead of date strings | Ground truth SQL `LIKE '2024%'` returns 0 rows | `mart_sales_monthly.sale_month` is TEXT without distribution params |
| S13: FK chain `product→category` 0% match | Category revenue query returns 0 rows | Data generation order issue in quality injection |

---

## Next Steps (Planned Phases)

### Phase 2 — Schema & Question Expansion

**Goal**: Scale from 31→~90 tables, 15→120 questions

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

3. **Fix known data issues**
   - `mart_sales_monthly.sale_month` distribution → generate proper month strings
   - Fix FK generation order for `product→category` chain

4. **Generate medium/enterprise databases**
   - `benchmark_medium.sqlite` (3M rows)
   - `benchmark_enterprise.sqlite` (10M rows)

### Phase 3 — CI & Automation

1. **GitHub Actions workflow** (`.github/workflows/benchmark.yml`)
   - On PR: generate `benchmark_small.sqlite`, run `run_benchmark.py --quick`
   - Output results to GitHub Summary
   - Regression detection vs previous run

2. **Report system**
   - Markdown/JSON export with escalation group breakdown
   - Per-model comparison (`qwen3.5-4b-4bit` vs `gpt-4o` vs `claude-3-opus`)

3. **Hyperparameter grid** (`search/grid.py` integration)
   - Cross-product runs: model × max_depth × max_attempts × calibration

### Phase 4 — Databricks Validation

1. **Delta DB generation** via `databricks-enterprise` profile
2. **Cross-backend consistency**: SQLite result == Delta result (same seed)
3. **Databricks executor benchmark**: `syrch.query(executor_type="databricks-sql")`

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

```bash
# Ground truth generation
python benchmark/evaluator/generate_ground_truth.py --profile small

# Dry-run evaluation (no LLM needed)
python benchmark/evaluator/run_benchmark.py --profile small --questions pilot --dry-run

# Full evaluation (requires API key)
python benchmark/evaluator/run_benchmark.py --profile small --model gpt-4o

# Quick check (L1-L2 only)
python benchmark/evaluator/run_benchmark.py --quick

# Export results
python benchmark/evaluator/run_benchmark.py --dry-run --export reports/pilot.json
```
