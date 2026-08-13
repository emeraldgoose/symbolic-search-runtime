# syrch — Symbolic Search Runtime

[English](README.md) | **한국어**

NL Problem → ProblemSpec → Search(D&C+RLM) → SQL Executor → Optimal Solution

## 프로젝트 목표

`syrch`는 정형 데이터에 대한 자연어 문제의 최적 해결책을 탐색하는 **검색 도구**입니다. 단순 QA 에이전트처럼 한 번에 답하는 대신, **분할 정복(divide & conquer) 분해와 재귀 언어 모델(RLM)**을 사용해 여러 추론 경로를 탐색하고, 실제 데이터베이스에서 후보 해결책을 실행하여 가장 좋은 결과를 선택합니다.

### 핵심 아이디어

- **분할 정복**: 문제를 논리적으로 독립적인 하위 문제(sub-task)로 분해하고, 각각 독립적으로 해결한 뒤 결과를 병합합니다. 하위 문제 간 의존성은 DAG로 표현됩니다.
- **RLM (Recursive Language Model)**: 각 하위 문제는 자체 REPL 루프에서 실행됩니다 — SQL 생성 → 구문 검증 → 스키마 검증 → 실행 → 품질 검사 → evidence 평가. 노드당 여러 후보를 탐색합니다.
- **Evidence 기반 선택**: RLM은 retriever prior(best-first)로 후보 풀을 탐색하고, 각 후보를 판별 신호(`structural_match`, `grain_match`, `dimension_match`, `time_match`, `result_quality`)로 평가해 사전식(lexicographic)으로 선택합니다. 모든 신호가 동률이면 **AMBIGUOUS** — 실행/retriever/cost 순서로는 절대 해소되지 않습니다.
- **후보 범위(Candidate Scope, S3)**: 각 시도는 `PRIMARY`(현재 후보), `JOIN-AVAILABLE`(풀에 속한 보조 테이블), `TASK CONTEXT`(materialized 부모 결과)로 제한됩니다. FROM은 PRIMARY/TASK CONTEXT만 앵커 가능, JOIN은 PRIMARY/JOIN-AVAILABLE/TASK CONTEXT만 사용 가능.
- **격자 탐색(Grid Search)**: 하이퍼파라미터(`max_depth`, `beam_width`, `max_attempts_per_node`, `calibration_enabled`)를 체계적으로 테스트하여 최적 설정을 찾습니다.
- **다중 테이블 스키마**: Retriever가 모든 테이블을 관련성별로 점수화하고, Planner가 sub-task별로 테이블을 선택하며, RLM은 시도별 범위(scope)의 테이블만 참조합니다.
- **실행이 아닌 추론에 대한 탐색**: D&C는 *문제 공간*을 분할하며, SQL을 분할하지 않습니다. 각 하위 문제는 완전한 추론 단위입니다 (생각 → 코드 → 검증 → 실행 → 평가).
- **플러그형 Executor**: 추상 `BaseExecutor`에 SQLite, JDBC, Spark, Databricks 구현 — PEP 249 호환.

## 아키텍처

```
User Question
    │
    ▼
┌──────────────────────┐
│    Retriever         │  ← 키워드 매칭으로 전체 테이블 점수화
│  (keyword match)     │     후보 풀 출력 (ordering 전용 —
│  + match_reason      │     최종 선택은 하지 않음)
└─────────┬────────────┘
          │ 후보 풀 (점수화된 테이블)
          ▼
┌──────────────────────┐
│  Schema-aware        │  ← LLM이 TaskDAG로 분해
│  Planner (D&C)       │     hint_tables: 강한 제약
│                      │     hint_columns: 소프트 힌트 (검증)
└─────────┬────────────┘
          │ DAG + sub-task별 요구사항
          ▼
┌──────────────────────┐
│    Scheduler         │  ← 위상 정렬 DAG 실행
│                      │     SOLVED → context materialize
│                      │     AMBIGUOUS/FAILED/BLOCKED 전파
│  각 노드:            │
│  ┌──────────────┐   │
│  │ RLM Agent    │   │  ← 후보별 탐색 루프:
│  │ 1. 후보      │   │     retriever prior best-first
│  │ 2. 생성      │   │     beam/exhaustive 정책 하에서
│  │ 3. 검증      │   │     (구문 → 스키마 → 범위)
│  │ 4. 실행      │   │
│  │ 5. 평가      │   │     CandidateEvaluation (5개 신호)
│  │ 6. 선택      │   │     사전식 → SOLVED/AMBIGUOUS
│  └──────────────┘   │
│                     │
│  replan_request ────→ Planner.replan() → 확장된 DAG
└──────┬───────────────┘
       │ NodeResults (DataFrames + SQL + confidence)
       ▼
┌──────────────────┐
│   Aggregator     │  ← primary 리프 선택, 재랭킹 없음
│                   │     confidence 조정 (모호성 + 휴리스틱)
└──────┬───────────┘
       │ FinalSolution
       ▼
 Optimal Answer + SQL + Reasoning Trace
```

### 하위 태스크 실행 방식 (RLM 노드)

각 노드는 플러그형 검색 정책(`beam` 기본, 또는 `exhaustive`)으로 후보를 탐색합니다:

```
Node "Refund counts by reason"
    │
    ├── 후보 dw_sales_order (retriever prior 0.8)
    │   ├── [PASS] 구문 검사 (sqlglot)
    │   ├── [PASS] 스키마 컬럼 검사 (status, total_amount)
    │   ├── [PASS] 범위 검사 (FROM = PRIMARY)
    │   ├── [PASS] 실행 → 3 rows
    │   └── viable: structural=1.0, grain=1.0, result_quality=1.0
    │
    ├── 후보 mart_sales_daily (retriever prior 0.6)
    │   ├── [PASS] 구문 검사
    │   ├── [FAIL] 스키마: status 컬럼 없음 → non-viable
    │   └── 거부 ("refund" 차원 없음)
    │
    └── 1위 vs 2위가 다르면 → SOLVED (dw_sales_order)
        모든 신호 동률이면 → AMBIGUOUS (cost로 해소하지 않음)
```

## 디렉토리 구조

```
syrch/
├── pyproject.toml
├── README.md
├── README_ko.md
├── AGENTS.md
├── PLAN.md
├── LICENSE
├── .gitignore
├── src/syrch/
│   ├── __init__.py               # Public API: query, SearchResult
│   ├── api.py                    # query() 고수준 함수
│   ├── cli/
│   │   ├── __init__.py
│   │   └── app.py                # Typer CLI
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py             # 데이터 타입 (dataclasses)
│   │   ├── config.py             # ExecutionConfig + 설정 로더
│   │   └── logging.py            # 구조화된 로깅
│   ├── executors/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseExecutor (ABC)
│   │   ├── sqlite_executor.py    # SQLite
│   │   ├── jdbc_executor.py      # JDBC via SQLAlchemy
│   │   ├── databricks_executor.py # Databricks SQL
│   │   ├── spark_executor.py     # SparkSession (Databricks/EMR/standalone)
│   │   └── cached_executor.py    # diskcache 기반 SQL 캐시
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py               # BaseLLM (ABC)
│   │   ├── openai_llm.py         # OpenAI
│   │   ├── anthropic_llm.py      # Anthropic Claude
│   │   └── cache.py              # CachedLLM + CentralCache
│   ├── search/
│   │   ├── __init__.py
│   │   ├── retriever.py          # 키워드 매칭 Retriever (후보 풀 + ordering)
│   │   ├── semantic_index.py     # 스키마 기반 evidence
│   │   ├── planner.py            # D&C: NL -> TaskDAG
│   │   ├── scheduler.py          # DAG 실행 엔진 (+ context materialize)
│   │   ├── rlm_engine.py         # RLM 후보 탐색 루프
│   │   ├── search_policy.py      # 후보 검색 정책 (beam/exhaustive)
│   │   ├── path_evaluator.py     # PathScore + 판별 신호
│   │   ├── validator.py          # 하드 제약 검사 (구문/스키마/범위)
│   │   ├── aggregator.py         # 결과 병합 + 휴리스틱
│   │   ├── calibrator.py         # 실행 신호 (execution-signal) penalties
│   │   ├── clarify.py            # 모호성 감지
│   │   ├── grid.py               # Grid search
│   │   └── pipeline.py           # 오케스트레이터
│   └── eval/
│       ├── __init__.py
│       ├── runner.py             # 벤치마크 하네스
│       ├── metrics.py            # 평가 메트릭
│       └── report.py             # 리포트 내보내기
├── scripts/
│   ├── gen_fixtures.py           # 테스트 fixture DB 생성
│   └── validate_real.py          # 실제 LLM 검증 (L1-L5)
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

## 데이터 모델

```
ProblemSpec { question, schema, all_schemas, goal_metric }
    │
    ▼
TaskDAG { nodes: {A, B, C, ...}, root_id, topo_layers }
    │  각 TaskNode: { id, description, depends_on, is_atomic,
    │                 requirements (metrics/dimensions/filters/grain),
    │                 hint_tables, hint_columns, join_keys }
    ▼
Scheduler → NodeResult { node_id, data(DataFrame), sql, confidence,
                         selected_candidate, reasoning_paths,
                         cost_tokens, status(SOLVED/AMBIGUOUS/FAILED/BLOCKED) }
    │
    ▼
Aggregator → FinalSolution { answer, sql, confidence, data, token_cost, tree }
             (primary = evidence가 있는 SOLVED 리프; 재랭킹 없음)
```

노드 선택은 후보당 `CandidateEvaluation`을 생성합니다:

```
CandidateEvaluation {
    table, ok, execution_valid, requirement_pass,
    semantic_match, result_quality,
    structural_match, grain_match, dimension_match, time_match,
    cost_tokens, candidate_id, confidence, path_score
}
```

선택은 판별 신호만으로 사전식 비교:
`structural_match → grain_match → dimension_match → time_match → result_quality → candidate_id`.
두 후보가 모든 신호에서 동률이면 **AMBIGUOUS** — 실행 순서/retriever prior/token cost로 절대 해소하지 않습니다.

## 설치

```bash
# 기본 (CLI + SQLite)
pip install syrch

# Databricks SQL Warehouse (외부 접속)
pip install "syrch[databricks-sql]"

# Spark executor (Databricks Runtime, EMR, standalone 공용)
pip install "syrch[spark]"

# 개발용 (테스트 + 린트)
pip install -e ".[dev]"

# 전체 설치
pip install "syrch[all]"
```

## Python API (라이브러리 모드)

Databricks notebook이나 Python 스크립트에서 직접 import하여 사용:

```python
from syrch import query

result = query(
    question="What discount × shipping combo maximizes revenue?",
    executor_type="databricks-sql",
    model="gpt-4o",
)
print(result.answer)      # 최종 답변
print(result.sql)         # 실행된 SQL
print(result.confidence)  # 신뢰도
print(result.data)        # 결과 DataFrame
```

## CLI 사용법

```bash
# 데이터베이스 스키마 확인
syrch schema wikipedia_clickstream.sqlite
syrch schema orders_10dim.sqlite -t orders_10dim

# 기본 설정 확인
syrch config

# 문제 해결 (LLM API 키 필요)
export OPENAI_API_KEY="sk-..."
syrch search -q "What discount × shipping combo maximizes revenue for top 10% customers?"

# Config file 사용
syrch search -q "..." --config syrch.yml

# 옵션 사용
syrch search -q "Which click type generates the most traffic?" \
  --db wikipedia_clickstream.sqlite \
  --max-depth 3 \
  --max-attempts 3 \
  --search-policy beam \
  --verbose

# 하이퍼파라미터 격자 탐색
syrch search -q "..." --db orders_10dim.sqlite --grid

# 예상 결과 대비 벤치마크
syrch eval -q "..." --db orders_10dim.sqlite --expected expected.csv

# 벤치마크 스위트 실행
syrch benchmark --file benchmarks/orders.jsonl
```

### CLI 참조

| Command | Option | 설명 |
|---------|--------|------|
| `search` | `-q` / `--question` | 자연어 문제 (필수) |
| | `--db` | 데이터베이스 경로 (기본값: `orders_10dim.sqlite`) |
| | `--max-depth` | 최대 D&C 재귀 깊이 (기본값: 3) |
| | `--executor` | `sqlite` / `databricks-sql` / `spark` / `jdbc` |
| | `--max-attempts` | 노드당 최대 RLM 시도 (기본값: 3) |
| | `--search-policy` | 후보 검색 정책: `beam` / `exhaustive` (기본값: beam) |
| | `--beam-width` | 조기 중단 전 최소 탐색 후보 수 (기본값: 3) |
| | `--candidate-budget` | 노드당 최대 후보 수 (기본값: 8) |
| | `--stop-margin` | 조기 중단에 필요한 posterior 격차 (기본값: 0.15) |
| | `--budget` | 토큰 예산 (기본값: 100000) |
| | `--llm` | `openai` / `anthropic` |
| | `--model` | LLM 모델명 (기본값: `qwen3.5-4b-4bit`) |
| | `-v` / `--verbose` | 추론 과정 출력 |
| | `--cache/--no-cache` | LLM + SQL 캐시 활성화/비활성화 (기본값: on) |
| | `--cache-ttl` | 캐시 TTL (초) (기본값: 86400) |
| | `--grid` | 하이퍼파라미터 격자 탐색 실행 |
| | `--grid-parallel/--grid-sequential` | 병렬 vs 순차 격자 실행 |
| | `--grid-max-workers` | 최대 동시 API 호출 (기본값: 3) |
| | `--interactive/--no-interactive` | SQL로 해결 불가능시 명확화 질문 활성화 |
| | `--config` | YAML 설정 파일 경로 (`syrch.yml` 또는 `~/.syrch/config.yml`) |
| `eval` | `-q` | 질문 |
| | `--db` | 데이터베이스 경로 |
| | `--executor` | Executor 유형 |
| | `--expected` | 예상 결과 CSV |
| | `--report-format` | `md` / `json` |
| `benchmark` | `--file` | JSONL 벤치마크 파일 |
| | `--executor` | Executor 유형 |
| | `--report` | 출력 리포트 경로 |
| `schema` | `DB` | 데이터베이스 경로 (positional) |
| | `-t` / `--table` | 특정 테이블 |
| `config` | `--db` | 데이터베이스 경로 |

## 설정 (Configuration)

설정은 다음 우선순위로 로드됩니다: **CLI 인자 > 환경변수 (`SYRCH_*`) > 설정 파일 > Databricks Secrets > 기본값**.

### 설정 파일 (`syrch.yml`)

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

검색 위치: `./syrch.yml` > `~/.syrch/config.yml` > `--config <path>` 명시 지정

### 환경변수

| 변수 | 매핑 | 예시 |
|------|------|------|
| `SYRCH_MODEL` | `llm.model` | `gpt-4o` |
| `SYRCH_API_KEY` | `llm.api_key` | `sk-...` |
| `SYRCH_BASE_URL` | `llm.base_url` | `http://localhost:11434/v1` |
| `SYRCH_MAX_DEPTH` | `execution.max_depth` | `3` |
| `SYRCH_VERBOSE` | `execution.verbose` | `true` |
| `SYRCH_SEARCH_POLICY` | `execution.search_policy` | `beam` |
| `SYRCH_BEAM_WIDTH` | `execution.beam_width` | `3` |
| `SYRCH_CANDIDATE_BUDGET` | `execution.candidate_budget` | `8` |
| `SYRCH_STOP_MARGIN` | `execution.stop_margin` | `0.15` |
| `SYRCH_MAX_CANDIDATE_EXPANSION` | `execution.max_candidate_expansion` | `2` |
| `SYRCH_MAX_REPLANS` | `execution.max_replans` | `1` |
| `SYRCH_CALIBRATION` | `execution.calibration_enabled` | `true` |
| `SYRCH_MATERIALIZE_CONTEXT` | `execution.materialize_context` | `true` |

### Databricks 연결

| 변수 | 인증 방식 | 설명 |
|------|-----------|------|
| `DATABRICKS_SERVER_HOSTNAME` | 전체 | Databricks workspace URL |
| `DATABRICKS_HTTP_PATH` | 전체 | SQL Warehouse HTTP path |
| `DATABRICKS_TOKEN` | `pat` | Personal Access Token |
| `DATABRICKS_AUTH_TYPE` | 전체 | `pat` (기본), `databricks-oauth`, `azure` |
| `DATABRICKS_CLIENT_ID` | oauth/azure | OAuth 클라이언트 ID |
| `DATABRICKS_CLIENT_SECRET` | oauth/azure | OAuth 클라이언트 시크릿 |
| `AZURE_TENANT_ID` | azure | Azure AD 테넌트 ID |

## 구조화된 로깅 (Structured Logging)

내부 진단 메시지는 `logging`을 통해 **stderr**로 출력됩니다. 사용자 결과(Solution, SQL)는 `rich`를 통해 **stdout**으로 출력됩니다.

```bash
# 기본: WARNING+만 stderr 출력
syrch search -q "..."

# 상세 로그 (INFO 레벨)
syrch search -q "..." -v

# 라이브러리 모드
python -c "
from syrch import query
result = query('Total revenue?', verbose=True)
"
```

로그 포맷: `LEVEL:logger_name:message`

```
INFO:syrch.scheduler:Layer 0: dispatching 2 nodes
WARNING:syrch.rlm_engine:Empty result, confidence penalized
```

## CI

GitHub Actions (`push`/`PR` → `main`):

| 단계 | 명령 |
|------|------|
| Lint | `ruff check src/syrch/` |
| 타입 검사 | `mypy src/syrch/ --ignore-missing-imports` |
| 테스트 | `pytest tests/ -v --cov=src/syrch/` (Python 3.11 + 3.12) |

## 신뢰도 (Confidence)

Confidence는 **aggregator의 출력 신호**이며 더 이상 탐색 종료를 주도하지 않습니다.
RLM은 모델의 `Confidence: <0.0-1.0>` 줄을 읽어(raw, 미기재시 기본 `0.7`) 노드 confidence로 저장합니다.

### 실행 신호 감점 (execution-signal penalties)

`PathEvaluator`를 통해 적용되며 `CandidateEvaluation.result_quality`로 표면화됩니다
(실행 신호는 더 이상 confidence를 직접 곱하지 않습니다):

| 신호 | 가중치 | 효과 |
|------|--------|------|
| `syntax_error` | 0.10 | 발생당 −0.10 (최대 ×3) |
| `execution_error` | 0.10 | 발생당 −0.10 (최대 ×3) |
| `empty_result` | 0.15 | 결과가 비면 −0.15 |
| `schema_error` | 0.05 | 발생당 −0.05 (최대 ×3) |
| `null_column` | 0.05 | 결과에 전부 NULL 컬럼이면 −0.05 |
| `overflow_result` | 0.05 | 결과가 과대하면 −0.05 |

### Aggregator 신뢰도

```
primary = selected_candidate evidence가 있는 리프 (SOLVED > AMBIGUOUS; 재랭킹 없음)
best_conf = max(primary.confidence, selected_candidate.confidence)
adjusted_conf = best_conf × (1.0 − max_ambiguity × 0.5) × (1.0 − heuristic_penalty)
```

**Heuristic penalties** (aggregator):
- Empty result: +0.15 per node
- Error present: +0.15 per node
- TOP-N mismatch: +0.05 per node
- "by year" without year column: +0.10 (once, global)
- AMBIGUOUS 리프: +0.10; FAILED/BLOCKED 리프: 각 +0.15
- **Capped at 0.40 total**

`calibration_enabled` (기본 `True`, env `SYRCH_CALIBRATION`)는 evaluator의
실행 신호 경로를 토글합니다.

## 격자 탐색 (Grid Search)

최적 설정을 위한 자동 하이퍼파라미터 탐색:

```bash
syrch search -q "What discount × shipping combo maximizes revenue?" \
  --db orders_10dim.sqlite --grid
```

기본 파라미터 그리드 (54 cells):
| Parameter | 값 |
|-----------|-----|
| max_depth | 1, 3, 5 |
| beam_width | 2, 3, 5 |
| max_attempts_per_node | 1, 3, 5 |
| calibration_enabled | True, False |

출력: `autoresearch/reports/{YYYYMMDD-HHMMSS}/{config,results,best}.json` + `summary.md`

최적 설정 선택: `exact_match > confidence` (오류 셀은 건너뜀).

## 탐색 정책 (후보 종료)

RLM 엔진은 플러그형 `SearchPolicy`(기본 `beam`, 또는 `exhaustive`)에 따라 후보 테이블을 탐색합니다:

1. 후보는 retriever 사전 확률 순서로 정렬(best-first)되어 하나씩 소비됩니다.
2. 각 후보는 RLM REPL 루프를 실행합니다: SQL 생성 → 구문 검증 → 스키마 검증 → 실행 → 품질 검사 → `PathScore` 평가.
3. 복구 실패(parse/schema/execution)와 요구사항 미충족은 `ok=False` / 0 PathScore로 표면화되며, 탐색을 종료시키지 않습니다.
4. `BeamSearchPolicy`는 최소 `beam_width`개, 최대 `candidate_budget`개 후보를 탐색합니다. beam 하한 이후에는 최선 *viable* 사후확률이 2번째 viable을 `stop_margin` 이상 앞설 때만 조기 종료합니다.
5. retriever 사전 확률은 후보 순서 결정에만 쓰이고 **종료에는 절대 사용되지 않습니다** — 풀의 하위 순위(예: rank 5 `dw_sales_order`)에 있는 정답 테이블도 계속 탐색됩니다.
6. confidence는 aggregator의 출력일 뿐이며 더 이상 탐색 종료에 사용되지 않습니다 (신뢰도 임계값 없음, 실행 기반 자동 상향 없음).

기존의 "보정 신뢰도 ≥ 0.85 → 즉시 수락" 규칙을 대체합니다. 단순한 문제는 여전히 빠르게 해결되고(작은 풀), 모호한 문제는 예산 내에서 더 많은 후보를 탐색합니다.

## 후보 범위 (Candidate Scope, S3)

각 RLM 시도는 명시적인 시도별 스키마 범위(`AttemptSchemaContext`)를 구성합니다:

```
PRIMARY         → 현재 테스트 중인 후보 (FROM 앵커 전용)
JOIN-AVAILABLE  → 풀에 속한 보조 테이블 (JOIN 전용, FROM 불가)
TASK CONTEXT    → materialized 부모 결과 (_task_context_*, FROM 또는 JOIN)
```

위치 규칙:

```
FROM  → PRIMARY | TASK CONTEXT (materialized)
JOIN  → PRIMARY | JOIN-AVAILABLE | TASK CONTEXT
```

- JOIN-AVAILABLE 테이블이 FROM에 오면 **primary switch**로 거부되며, JOIN 가능한 보조 테이블을 안내합니다.
- materialize되지 않은 `_task_context_*` 이름(의존성 AMBIGUOUS/FAILED)은 거부 — SOLVED 부모만 materialize합니다.
- drift 방지는 *알려진 물리* 테이블에만 적용됩니다; 미지 테이블은 실행 오류로 표면화됩니다.
- 후보 풀이 JOIN 후보의 유일한 소스 — 풀 밖 테이블로의 drift는 차단됩니다.

## Executor 추상화

모든 Executor는 `BaseExecutor`를 따릅니다:

```python
class BaseExecutor(ABC):
    def execute(sql: str) -> DataFrame: ...
    def get_schema(table_name?: str) -> TableSchema: ...
    def list_tables() -> list[str]: ...
    def close(): ...
```

| Executor | 백엔드 | 연결 방식 |
|----------|--------|----------|
| `SQLiteExecutor` | SQLite | `sqlite3` (thread-safe via `threading.local`) |
| `JDBCExecutor` | Any JDBC | SQLAlchemy |
| `DatabricksExecutor` | Databricks SQL | `databricks-sql-connector` (PEP 249) |
| `SparkExecutor` | SparkSession | `pyspark` (`SparkSession.builder.getOrCreate()`) |

### Context Materialization (v0.3.5b)

SOLVED 노드의 결과는 실제 테이블 `_task_context_<id>`로 쓰여져 의존 노드가 JOIN할 수 있습니다.
각 Executor는 `materialize_context(context)` / `drop_context(table)`를 오버라이드합니다:

| Executor | 방식 |
|----------|------|
| `SQLiteExecutor` | `DROP` + `to_sql` + commit |
| `SparkExecutor` | `createOrReplaceTempView` |
| `JDBCExecutor` | `to_sql(if_exists="replace")` |
| `DatabricksExecutor` | `CREATE OR REPLACE TEMP VIEW` |

`_task_context_*` 테이블은 `list_tables()`에서 제외되어 retriever가 물리 후보로 취급하지 않습니다.
스케줄러는 각 SOLVED 노드 후 materialize하고(AMBIGUOUS/FAILED/BLOCKED/empty는 건너뜀),
실행 종료 시 모든 materialized 테이블을 삭제합니다. `ExecutionConfig.materialize_context`
(기본 `True`, env `SYRCH_MATERIALIZE_CONTEXT`)가 전체 기능을 게이트합니다.

## 캐싱

모든 LLM 및 SQL 호출은 `diskcache`를 통해 캐시됩니다 (24h TTL):

| 레이어 | 캐시 | 키 |
|--------|------|-----|
| LLM `generate()` | `CachedLLM` | SHA256(system + user + model + temperature) |
| LLM `generate_json()` | `CachedLLM` | SHA256(system + user + model + temperature) |
| SQL `execute()` | `CachedExecutor` | SHA256(sql) |

`--cache/--no-cache` 플래그로 전환; TTL은 `--cache-ttl`로 설정 가능.

## 데이터셋

`scripts/gen_fixtures.py`로 `tests/fixtures/`에 생성됩니다:

| Dataset | Rows | Size | 설명 |
|---------|------|------|------|
| `wikipedia_clickstream.sqlite` | ~200 | ~36 KB | 위키백과 클릭스트림 집계 데이터 |
| `orders_10dim.sqlite` | 1000+ | ~90 KB | 합성 주문 데이터 (10개 차원 컬럼) |

## 테스트

```bash
# 단위 + 통합 테스트 (FakeLLM, API 키 불필요)
pytest tests/ -v

# 현재: 155개 테스트 통과
#   api, cache, clarify, discrimination, e2e, eval, integration,
#   materialize, planner, rlm_engine, scheduler, search_policy, validator
```

### 실제 환경 검증

```bash
# 전체 검증 실행 (LLM API 키 필요)
python scripts/validate_real.py

# 특정 레벨
python scripts/validate_real.py --level 3 --verbose

# 커스텀 질문
python scripts/validate_real.py --question "Total revenue by year?" --db orders_10dim.sqlite

# 로컬 모델 사용
python scripts/validate_real.py --model qwen3.5-4b --max-concurrency 1

# 결과 (2026-06-15, minimax-m3:cloud):
#   L1 Easy           3/3 PASS
#   L2 Medium         3/3 PASS
#   L3 Complex        2/2 PASS
#   L4 Very Complex   2/2 PASS
#   L5 Ambiguous      2/2 AMBIGUOUS (expected)
#   ─────────────────────────────
#   Total             10/10 PASS  100% (2 AMBIGUOUS)
```

## 연구 배경

- **RLM (Recursive Language Model)**: MIT CSAIL OASYS Lab, 2025. LLM이 REPL 환경을 통해 입력을 재귀적으로 분해하는 추론 패러다임. [`paper`](https://arxiv.org/abs/2512.24601) [`code`](https://github.com/alexzhang13/rlm)
- **RDD (Recursive Decomposition with Dependencies)**: 의존성 DAG를 사용한 공식 D&C 프레임워크. [`paper`](https://arxiv.org/abs/2505.02576)
- **PAC-MCTS**: 편향 인식 가지치기를 통한 트리 탐색 공식 보장. [`paper`](https://arxiv.org/abs/2604.14345)
- **ROMA**: Atomizer/Planner/Executor/Aggregator 역할의 재귀 메타에이전트 프레임워크. [`paper`](https://arxiv.org/abs/2602.01848)
- **Graph Harness**: 불변 계획 버전을 사용한 구조화된 DAG 실행. [`paper`](https://arxiv.org/abs/2604.11378)
- **AdaptOrch**: 토폴로지 인식 멀티에이전트 오케스트레이션 (병렬/순차/계층/혼합). [`paper`](https://arxiv.org/abs/2602.16873)
- **DST**: 신뢰도 기반 가지치기를 사용한 적응형 트리 탐색 (26-75% 계산량 감소). [`paper`](https://arxiv.org/abs/2603.20267)
- **LLM Compiler**: 의존성 그래프를 통한 병렬 태스트 스케줄링; syrch의 DAG 스케줄러 및 레이어별 실행과 밀접한 관련. [`paper`](https://arxiv.org/abs/2312.13311)

## 공개 연구 질문

| 질문 | 접근 방식 |
|------|----------|
| **분할을 언제 멈출까?** (단위 케이스 감지) | LLM 자체 평가 + 복잡도 휴리스틱 실험 |
| **하위 태스크 결과를 어떻게 병합할까?** | DAG 기반 REPL 변수 전달 + Aggregator 역할 |
| **탐색 공간을 어떻게 가지치기할까?** | 신뢰도 기반 가지치기 + 불확실성 인식 할당 |
| **최적 D&C 전략은?** | DAG 구조 메트릭 기반 토폴로지 라우팅 (AdaptOrch) |
| **최적 보정 가중치는?** | 신호별 패널티 계수에 대한 격자 탐색 |
| **조인 키 추론?** | Planner가 sub-task 간 join_keys 생성 |
| **재귀 분해?** | Planner가 비원자 sub-task에 재귀 적용 |
| **SQL로 해결 불가능할 때?** | RLM 명확화: 모호성 점수 → 대화형 피드백 → 재분해 |
| **최적 명확화 임계값은?** | 점수 가중치 + 결정 경계에 대한 격자 탐색 |
