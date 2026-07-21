# syrch — Symbolic Search Runtime

[English](README.md) | **한국어**

NL Problem → ProblemSpec → Search(D&C+RLM) → SQL Executor → Optimal Solution

## 프로젝트 목표

`syrch`는 정형 데이터에 대한 자연어 문제의 최적 해결책을 탐색하는 **검색 도구**입니다. 단순 QA 에이전트처럼 한 번에 답하는 대신, **분할 정복(divide & conquer) 분해와 재귀 언어 모델(RLM)**을 사용해 여러 추론 경로를 탐색하고, 실제 데이터베이스에서 후보 해결책을 실행하여 가장 좋은 결과를 선택합니다.

### 핵심 아이디어

- **분할 정복**: 문제를 논리적으로 독립적인 하위 문제(sub-task)로 분해하고, 각각 독립적으로 해결한 뒤 결과를 병합합니다. 하위 문제 간 의존성은 DAG로 표현됩니다.
- **RLM (Recursive Language Model)**: 각 하위 문제는 자체 REPL 루프에서 실행됩니다 — 코드 생성 → 구문 검증 → 스키마 검증 → SQL 실행 → 품질 검사 → 신뢰도 평가 → 개선 또는 중단. 노드당 여러 추론 경로를 탐색합니다.
- **신뢰도 보정(Confidence Calibration)**: LLM이 스스로 평가한 신뢰도를 실행 신호(재시도, 오류, 빈 결과)로 할인하여 더 신뢰할 수 있는 점수를 산출합니다.
- **격자 탐색(Grid Search)**: 하이퍼파라미터(`max_depth`, `high_confidence`, `max_attempts`, `calibration_enabled`)를 체계적으로 테스트하여 최적 설정을 찾습니다.
- **다중 테이블 스키마**: Retriever가 모든 테이블을 관련성별로 점수화하고, Planner가 sub-task별로 테이블을 선택하며, RLM은 압축된 스키마(2-5개 테이블, 전체 31개 아님)만 참조합니다.
- **실행이 아닌 추론에 대한 탐색**: D&C는 *문제 공간*을 분할하며, SQL을 분할하지 않습니다. 각 하위 문제는 완전한 추론 단위입니다 (생각 → 코드 → 검증 → 실행 → 평가).
- **플러그형 Executor**: 추상 `BaseExecutor`에 SQLite, JDBC, Databricks 구현 — PEP 249 호환.

## 아키텍처

```
User Question
    │
    ▼
┌──────────────────────┐
│    Retriever         │  ← 키워드 매칭으로 전체 테이블 점수화
│  (keyword match)     │     scored_schemas 출력 (고정 K 없음)
│  + match_reason      │
└─────────┬────────────┘
          │ scored_schemas (점수 + 사유)
          ▼
┌──────────────────────┐
│  Schema-aware        │  ← LLM 분해 + 테이블 선택
│  Planner (D&C)       │     hint_tables: 강한 제약
│                      │     hint_columns: 소프트 힌트 (검증)
│                      │     L1 깊이 자동 제한 (최대 2)
└─────────┬────────────┘
          │ DAG + sub-task별 힌트
          ▼
┌──────────────────────┐
│  Semantic Clarifier  │  ← 실행 전 모호성 감지
│  (Question + DAG)    │     Planner 출력 활용
│  interactive only    │
└─────────┬────────────┘
          │ (or skip if clear)
          ▼
┌──────────────────────┐
│  Schema Compression  │  ← hint_tables 스키마만 추출
│                      │     RLM은 2-5개 테이블만 참조 (전체 31 아님)
└─────────┬────────────┘
          │ 압축된 스키마
          ▼
┌──────────────────────┐
│    Scheduler         │  ← 레이어별 DAG 실행
│                      │     복구 가능 오류 → RLM 재시도
│  각 노드:            │     구조적 오류 → Planner.replan()
│  ┌──────────────┐   │
│  │ RLM Agent    │   │  ← 5단계 검증 루프:
│  │ 1. 구문 검사  │   │     1. SQLGlot 구문 검사
│  │ 2. 스키마 검사│   │     2. 스키마 AST 검사
│  │ 3. 실행       │   │     3. SQL 실행
│  │ 4. 품질 검사  │   │     4. 품질 검사
│  │ 5. 보정       │   │     5. 신뢰도 보정
│  └──────────────┘   │
│                     │
│  replan_request ────→ Planner.replan(dag, node, error, trace)
│                     │   → DAG 업데이트 → 재압축 → 계속
│  가지치기:           │
│  conf ≥ 임계값 → 즉시 중단
└──────┬───────────────┘
       │ NodeResults (DataFrames + SQL + confidence)
       ▼
┌──────────────────┐
│   Aggregator     │  ← 리프 결과 병합 → 최종 답변
│                   │     동점: 동일 confidence → 낮은 token_cost
└──────┬───────────┘
       │ FinalSolution
       ▼
 Optimal Answer + SQL + Reasoning Trace

 ═══════ Grid Search ═══════
       │
       ▼
┌──────────────────┐
│   Grid Search    │  ← 27-54 cells (파라미터 조합)
│                   │     ProcessPoolExecutor (max_workers=3)
│                   │     Reports: config.json, results.json,
│                   │              best.json, summary.md
└──────┬───────────┘
       │ Best config → run_pipeline again
```

### 하위 태스크 실행 방식 (RLM 노드)

```
Node "Find top 10% customers"
    │
    ├── 시도 1: SQL 경로 A
    │   ├── [PASS] 구문 검사 (sqlglot)
    │   ├── [PASS] 스키마 컬럼 검사
    │   ├── [PASS] 실행 → 5,234 rows
    │   ├── [WARN] 품질: 5234 rows 반환 (>1000)
    │   └── confidence: 0.72 (임계값 미달, 재시도)
    │
    ├── 시도 2: SQL 경로 B
    │   ├── [PASS] 구문 검사
    │   ├── [PASS] 스키마 컬럼 검사
    │   ├── [PASS] 실행 → 534 rows
    │   ├── [PASS] 품질: OK
    │   └── confidence: 0.91 → 보정: 0.86 (임계값 초과, 중단)
    │
    └── 최고(보정된) 결과를 부모에 반환
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
│   │   ├── retriever.py          # 키워드 매칭 Retriever
│   │   ├── planner.py            # D&C: NL -> TaskDAG
│   │   ├── scheduler.py          # DAG 실행 엔진
│   │   ├── rlm_engine.py         # RLM REPL 루프
│   │   ├── aggregator.py         # 결과 병합
│   │   ├── calibrator.py         # 신뢰도 보정
│   │   ├── clarify.py            # 모호성 감지
│   │   ├── grid.py               # Grid search
│   │   └── pipeline.py           # 오케스트레이터
│   └── eval/
│       ├── __init__.py
│       ├── runner.py             # 벤치마크 하네스
│       ├── metrics.py            # 평가 메트릭
│       └── report.py             # 리포트 내보내기
├── validate_real.py              # 실제 LLM 검증
├── orders_10dim.sqlite           # TPC-H 기반 (750만 행)
├── wikipedia_clickstream.sqlite  # Clickstream (3K 행)
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

## 데이터 모델

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
             (동점: 동일 confidence → 낮은 cost_tokens 우선)
```

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
  --high-conf 0.85 \
  --max-attempts 3 \
  --verbose

# 하이퍼파라미터 격자 탐색 (54 cells)
syrch search -q "..." --db orders_10dim.sqlite --grid

# 예상 결과 대비 벤치마크
syrch eval -q "..." --db orders_10dim.sqlite --expected expected.csv

# 벤치마크 스위트 실행
syrch benchmark benchmarks/orders.jsonl
```

### CLI 참조

| Command | Option | 설명 |
|---------|--------|------|
| `search` | `-q` / `--question` | 자연어 문제 (필수) |
| | `--db` | 데이터베이스 경로 (기본값: `orders_10dim.sqlite`) |
| | `--max-depth` | 최대 D&C 재귀 깊이 (기본값: 3) |
| | `--executor` | `sqlite` / `databricks-sql` / `spark` / `jdbc` |
| | `--max-attempts` | 노드당 최대 RLM 시도 (기본값: 3) |
| | `--high-conf` | 조기 중단을 위한 신뢰도 임계값 (기본값: 0.85) |
| | `--budget` | 토큰 예산 (기본값: 100000) |
| | `--llm` | `openai` / `anthropic` |
| | `--model` | LLM 모델명 (기본값: `qwen3.5-4b-4bit`) |
| | `-v` / `--verbose` | 추론 과정 출력 |
| | `--cache/--no-cache` | LLM + SQL 캐시 활성화/비활성화 (기본값: on) |
| | `--cache-ttl` | 캐시 TTL (초) (기본값: 86400) |
| | `--grid` | 하이퍼파라미터 격자 탐색 실행 |
| | `--grid-parallel/--grid-sequential` | 병렬 vs 순차 격자 실행 |
| | `--grid-max-workers` | 최대 동시 API 호출 (기본값: 3) |
| | `--max-concurrency` | 최대 동시 LLM 호출 (기본값: 5; 로컬 모델은 1 권장) |
| | `--interactive` | SQL로 해결 불가능시 명확화 질문 활성화 |
| | `--non-interactive` | 명확화 없는 원샷 모드 (기본값) |
| | `--config` | YAML 설정 파일 경로 (`syrch.yml` 또는 `~/.syrch/config.yml`) |
| `eval` | `-q` | 질문 |
| | `--db` | 데이터베이스 경로 |
| | `--executor` | Executor 유형 |
| | `--expected` | 예상 결과 CSV |
| | `--report-format` | `md` / `json` |
| `benchmark` | `PATH` | JSONL 벤치마크 파일 (positional) |
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
  high_confidence: 0.85
  token_budget: 100000
  cache_enabled: true
  cache_ttl: 86400
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

## 신뢰도 보정 (Confidence Calibration)

LLM이 스스로 평가한 신뢰도를 실행 신호로 조정합니다:

| 신호 | 가중치 | 효과 |
|------|--------|------|
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

공식: `calibrated = raw × Π(1 - weight_if_applicable)`

`--no-cache` 전달시 비활성화 (`calibration_enabled=False` in `ExecutionConfig`).

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
| high_confidence | 0.7, 0.85, 0.95 |
| max_attempts_per_node | 1, 3, 5 |
| calibration_enabled | True, False |

출력: `autoresearch/reports/{YYYYMMDD-HHMMSS}/{config,results,best}.json` + `summary.md`

최적 설정 선택: `exact_match > confidence` (오류 셀은 건너뜀).

## 가지치기 전략 (Pruning)

RLM 엔진은 신뢰도 기반 가지치기 전략을 사용합니다:

1. 첫 번째 추론 경로 생성 → SQL → 3단계 검증 (구문 → 스키마 → 품질)
2. 실행 → 점수 산출
3. 신뢰도 보정 적용 (활성화된 경우)
4. 보정된 신뢰도 ≥ `HIGH_CONFIDENCE` (0.85) → **즉시 수락**, 중단
5. 임계값 미만 → 대체 경로 생성
6. `max_attempts` 후 → 보정된 신뢰도 기준 **최고 경로** 선택

이는 탐색의 철저함과 토큰 예산 사이의 균형을 유지합니다. 단순한 문제는 빠르게 해결(탐욕 경로)되고, 복잡한 문제는 여러 후보를 탐색합니다.

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

## 캐싱

모든 LLM 및 SQL 호출은 `diskcache`를 통해 캐시됩니다 (24h TTL):

| 레이어 | 캐시 | 키 |
|--------|------|-----|
| LLM `generate()` | `CachedLLM` | SHA256(system + user + model + temperature) |
| LLM `generate_json()` | `CachedLLM` | SHA256(system + user + model + temperature) |
| SQL `execute()` | `CachedExecutor` | SHA256(sql) |

`--cache/--no-cache` 플래그로 전환; TTL은 `--cache-ttl`로 설정 가능.

## 데이터셋

| Dataset | Rows | Size | 설명 |
|---------|------|------|------|
| `wikipedia_clickstream.sqlite` | 3,138 | 280 KB | 위키백과 클릭스트림 집계 데이터 |
| `orders_10dim.sqlite` | 7,500,000 | 1.3 GB | TPC-H 기반 합성 주문 데이터 (10개 차원 컬럼) |

## 테스트

```bash
# 단위 테스트 (FakeLLM, API 키 불필요)
pytest tests/ -v

# 전체 69개 테스트 통과:
#   7  cache tests (CentralCache, CachedLLM, CachedExecutor)
#   9  clarify tests (ambiguity score, question generation, worst detection)
#   9  e2e tests (실제 SQLite DB + pipeline)
#   14 eval tests (runner, metrics, benchmark, join merge)
#   8  integration tests (DAG, grid, clarification loop, multi-table)
#   8  planner tests (decompose, cycle, join keys, recursive)
#   10 rlm_engine tests (validation, calibration, quality, calibrator)
#   4  scheduler tests (DAG execution)
```

### 실제 환경 검증

```bash
# 전체 검증 실행 (LLM API 키 필요)
python validate_real.py

# 특정 레벨
python validate_real.py --level 3 --verbose

# 커스텀 질문
python validate_real.py --question "Total revenue by year?" --db orders_10dim.sqlite

# 로컬 모델 사용
python validate_real.py --model qwen3.5-4b --max-concurrency 1

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
