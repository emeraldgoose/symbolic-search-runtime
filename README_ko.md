# syrch — Symbolic Search Runtime

[[English]](README.md) | **한국어**

> **자연어 → 검증 가능한 SQL → 최적의 답.**
> syrch는 실제 데이터베이스에서 여러 추론 경로를 탐색하고, 근거(evidence)로 승자를 가립니다.

---

## 목차

- [개요](#개요)
- [기능](#기능)
- [아키텍처](#아키텍처)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [CLI 사용법](#cli-사용법)
- [설정](#설정)
- [동작 원리](#동작-원리)
- [신뢰도 & 근거](#신뢰도--근거)
- [디렉토리 구조](#디렉토리-구조)
- [개발](#개발)
- [벤치마크 & 평가](#벤치마크--평가)
- [기여](#기여)
- [연구 배경](#연구-배경)
- [라이선스](#라이선스)

---

## 개요

`syrch`(symbolic-search-runtime)는 정형 데이터에 대한 자연어 분석을 위한 **탐색 하네스(search harness)**입니다. 한 번에 답하는 대신 **분해하고, 탐색하고, 검증**합니다:

```
NL Problem → ProblemSpec → Search(D&C + RLM) → SQL Executor → Optimal Solution
```

- **분해(Decompose)** — 질문을 Divide & Conquer로 DAG 형태의 하위 작업으로 나눕니다.
- **탐색(Search)** — 각 하위 작업을 후보 테이블 풀 위에서 RLM REPL 루프로 탐색합니다.
- **검증(Verify)** — 모든 후보를 실제 DB에서 SQL로 실행하고 근거로 순위를 매깁니다.

결과는 단순한 답이 아니라 **출처가 있는 답**입니다 — 어떤 테이블을 썼는지, 어떤 필터를 걸었는지, 어떻게 계산했는지가 함께 제공됩니다.

---

## 기능

| 영역 | 설명 |
|------|------|
| **Divide & Conquer Planner** | 질문을 `TaskDAG`의 `RequirementSpec` 노드로 변환 (metrics, dimensions, grain, filters, supporting relations). |
| **Retriever (ordering only)** | 모든 테이블을 어휘/의미 관련도로 점수화. 점수는 *다음에 무엇을 시도할지*만 정하고 승자를 결정하지 않습니다. |
| **RLM Agent** | 후보별 REPL 루프: `SQL 생성 → 구문 검증 → 스키마 검증 → 실행 → 품질 검사 → 평가`. |
| **Candidate Scope** | 시도별 엄격한 네임스페이스: `PRIMARY`(FROM 전용), `JOIN-AVAILABLE`(JOIN 전용), `TASK CONTEXT`(materialized 부모 결과). 범위 밖 테이블로의 드리프트를 차단합니다. |
| **Pluggable Executors** | `SQLite`, `JDBC`, `Spark`, `Databricks SQL`을 단일 `BaseExecutor` 인터페이스로 지원. |
| **Evidence 기반 선택** | `structural_match → grain_match → dimension_match → time_match → result_quality` 사전식(lexicographic) 랭킹. 완전 동점 → `AMBIGUOUS`(cost나 순서로 해소하지 않음). |
| **Capability Gates** | `metric_feasible`와 non-empty/non-null 하드 필터가 랭킹 진입을 통제; `time_match`는 데이터 기반 커버리지(MIN/MAX)로 요청 기간을 커버할 수 없는 테이블을 제외합니다. |
| **Context Materialization** | `SOLVED` 부모는 `_task_context_<id>` 실제 테이블로 materialize되어 하위 작업이 `JOIN` 가능. `AMBIGUOUS`/`FAILED`는 materialize하지 않습니다. |
| **Calculation Basis** | *어떤 해석*으로 숫자가 나왔는지를 결정론적으로, 현지화된 언어로 설명: 시간 범위, 환불/상태 제외 여부, 소스별 SCD2 의미(point-in-time vs 고정 창 겹침 vs 무시) + 단계별 근거. |

---

## 아키텍처

```
                USER QUESTION
                     │
                     ▼
           ┌──────────────────┐
           │  Planner (D&C)   │  "무엇을 풀까?"
           │  → RequirementSpec + TaskDAG
           └────────┬─────────┘
                    │ requirements + DAG
                    ▼
           ┌──────────────────┐
           │   Retriever      │  "어디를 찾아볼까?"
           │  → candidate pool (ranked)
           └────────┬─────────┘
                    │ pool
                    ▼
           ┌──────────────────┐
           │   Scheduler      │  DAG를 층별로 실행
           │  ┌────────────┐  │
           │  │ RLM Agent  │  │  candidate → SQL → validate → execute → evaluate
           │  │ beam / exhaustive, lexicographic
           │  └─────┬──────┘  │
           └────────┼─────────┘
                    │ SOLVED → materialize _task_context_<id>
                    ▼
           ┌──────────────────┐
           │   Aggregator     │  primary leaf 선택 (SOLVED > AMBIGUOUS),
           │  no re-ranking   │  현지화된 calculation basis 생성
           └────────┬─────────┘
                    ▼
        FINAL ANSWER + SQL + confidence + step-by-step evidence
```

**노드 결과:** `SOLVED` → materialize되어 JOIN 가능. `AMBIGUOUS` → materialize하지 않음(불확실한 결과는 사실이 되지 않음). `FAILED`/`BLOCKED` → 하위 노드는 `BLOCKED`로 단락.

<details>
<summary><b>하위 작업 실행 방식 (클릭하여 펼치기)</b></summary>

```
Node "Refund counts by reason"
  ├─ 후보 dw_sales_order (prior 0.8)
  │   ├─ [PASS] 구문 검사 (sqlglot)
  │   ├─ [PASS] 스키마 검사 (status, total_amount)
  │   ├─ [PASS] 범위 검사 (FROM = PRIMARY)
  │   ├─ [PASS] 실행 → 3 rows
  │   └─ viable: structural=1.0, grain=1.0, result_quality=1.0
  ├─ 후보 mart_sales_daily (prior 0.6)
  │   ├─ [PASS] 구문 검사
  │   ├─ [FAIL] 스키마: status 컬럼 없음 → non-viable
  │   └─ 제외
  └─ 1위 vs 2위가 다르면 → SOLVED (dw_sales_order)
     완전 동점 → AMBIGUOUS (cost로 해소하지 않음)
```

</details>

---

## 설치

```bash
# 기본 (CLI + SQLite)
pip install syrch

# Databricks SQL Warehouse
pip install "syrch[databricks-sql]"

# Spark (Databricks Runtime / EMR / standalone)
pip install "syrch[spark]"

# 개발용 (테스트 + 린트 + 타입 체크)
pip install -e ".[dev]"

# 전체
pip install "syrch[all]"
```

**요구사항:** Python 3.11+

---

## 빠른 시작

### 라이브러리 모드

```python
from syrch import query

result = query(
    question="What discount × shipping combo maximizes revenue?",
    executor_type="databricks-sql",
    model="gpt-4o",
)

print(result.answer)              # 최종 답변 텍스트 (현지화)
print(result.sql)                 # 실행된 SQL (모든 단계)
print(result.confidence)          # 0.0 – 1.0
print(result.data)                # 결과 DataFrame
print(result.calculation_basis)   # 단계별 근거 (라벨 현지화, 내부는 영어 유지)
print(result.tables_used)         # 실제 조회한 물리 테이블 목록
print(result.tree)                # 작업별 NodeResult 리스트
print(result.dag_nodes)           # 작업 그래프 구조
```

Databricks 노트북과 일반 Python 스크립트에서 바로 동작합니다.

### CLI 모드

```bash
# 스키마 확인
syrch schema orders_10dim.sqlite
syrch schema wikipedia_clickstream.sqlite -t wikipedia_clickstream

# 기본 설정 확인
syrch config

# 문제 해결 (API 키 필요)
export OPENAI_API_KEY="sk-..."
syrch search -q "What discount × shipping combo maximizes revenue for top 10% customers?"

# 옵션 사용
syrch search -q "Which click type generates the most traffic?" \
  --db wikipedia_clickstream.sqlite --max-depth 3 --verbose

# 격자 탐색
syrch search -q "..." --db orders_10dim.sqlite --grid
```

---

## CLI 사용법

| 명령 | 주요 옵션 | 설명 |
|------|-----------|------|
| `search` | `-q` / `--question` | 자연어 문제 (**필수**) |
| | `--db` | 데이터베이스 경로 (기본값: `orders_10dim.sqlite`) |
| | `--executor` | `sqlite` / `databricks-sql` / `spark` / `jdbc` |
| | `--max-depth` | 최대 D&C 재귀 깊이 (기본값: 3) |
| | `--max-attempts` | 노드당 최대 RLM 시도 (기본값: 3) |
| | `--search-policy` | `beam` / `exhaustive` (기본값: `beam`) |
| | `--beam-width` | 조기 중단 전 최소 탐색 후보 수 (기본값: 3) |
| | `--candidate-budget` | 노드당 최대 후보 수 (기본값: 8) |
| | `--stop-margin` | 조기 중단에 필요한 posterior 격차 (기본값: 0.15) |
| | `--budget` | 토큰 예산 (기본값: 100000) |
| | `--llm` | `openai` / `anthropic` |
| | `--model` | 모델명 (기본값: `qwen3.5-4b-4bit`) |
| | `-v` / `--verbose` | 추론 과정 출력 |
| | `--cache / --no-cache` | LLM + SQL 캐시 (기본값: on) |
| | `--grid` | 하이퍼파라미터 격자 탐색 실행 |
| | `--config` | YAML 설정 파일 |
| `schema` | `DB` | 데이터베이스 경로 (positional) |
| | `-t` / `--table` | 특정 테이블 |
| `config` | | 해석된 설정 출력 |
| `eval` | | 예상 CSV 대비 벤치마크 |
| `benchmark` | | JSONL 벤치마크 스위트 실행 |

---

## 설정

우선순위: **CLI 인자 > 설정 파일 > 환경변수 (`SYRCH_*`) > Databricks Secrets > 기본값**.

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

탐색 경로: `./syrch.yml` → `./syrch.yaml` → `~/.syrch/config.yml` → `~/.syrch/config.yaml` (`--config <path>`가 모두 덮어씀).

### 환경변수

| 변수 | 매핑 | 예시 |
|------|------|------|
| `SYRCH_MODEL` | `llm.model` | `gpt-4o` |
| `SYRCH_API_KEY` | `llm.api_key` | `sk-...` |
| `SYRCH_BASE_URL` | `llm.base_url` | `http://localhost:11434/v1` |
| `SYRCH_EXECUTOR` | `execution.executor_type` | `sqlite` |
| `SYRCH_MAX_DEPTH` | `execution.max_depth` | `3` |
| `SYRCH_SEARCH_POLICY` | `execution.search_policy` | `beam` |
| `SYRCH_BEAM_WIDTH` | `execution.beam_width` | `3` |
| `SYRCH_CANDIDATE_BUDGET` | `execution.candidate_budget` | `8` |
| `DATABRICKS_SERVER_HOSTNAME` | Databricks workspace URL | `dbc-...cloud.databricks.com` |
| `DATABRICKS_HTTP_PATH` | SQL Warehouse 경로 | `/sql/1.0/warehouses/...` |
| `DATABRICKS_TOKEN` | Personal Access Token | `dapi...` |

전체 목록: [AGENTS.md 설정 섹션](AGENTS.md#config) 참조.

---

## 동작 원리

### 파이프라인

```
ProblemSpec → Retriever → Planner → Scheduler → Aggregator → FinalSolution
```

### Candidate Scope (시도별)

```
PRIMARY         → 테스트 중인 후보 (FROM 전용)
JOIN-AVAILABLE  → 풀에 속한 보조 테이블 (JOIN 전용)
TASK CONTEXT    → _task_context_* materialized 부모 결과 (FROM 또는 JOIN)
```

- `JOIN-AVAILABLE`이 `FROM`에 오면 **primary switch**로 거부.
- materialize되지 않은 `_task_context_*`(AMBIGUOUS/FAILED 부모) → 거부.
- *알려진 물리* 테이블만 drift 검사; 미지 테이블은 실행 오류로 표면화.

### 탐색 정책

1. 후보는 retriever prior 순서대로 하나씩 소비.
2. 각 후보는 REPL 루프 실행; 실패는 `ok=False`로 표면화되며 탐색을 종료시키지 않음.
3. `BeamSearchPolicy`: 최소 `beam_width`, 최대 `candidate_budget` 탐색; beam 하한 이후 최선 viable posterior가 2번째를 `≥ stop_margin` 앞설 때만 조기 종료.
4. Retriever prior는 종료에 절대 사용되지 않음 — rank 5의 GT 테이블도 계속 탐색됨.
5. Confidence는 aggregator의 출력이며 탐색 신호가 아님.

### Evidence Tiers (anti-cheat 계약)

- **T1 스키마** (`grain_match`, `dimension_match`, 어휘적 타당성)
- **T2 능력** (후보가 *제공할 수 있는* 데이터: `TimeCoverage` probe 기반 시간 커버리지, metric 타당성) — 균일하고 GT-free; `TimeCoverage`는 probe 사실이 `time_match`에 들어갈 수 있는 문서화된 예외
- **T3 결과** (`result_quality`, `has_data`, all-NULL은 empty처럼 처리) — VALUE probe 사실("값 X가 어디에 있나")은 RLM 전용이며 랭킹에 절대 유입되지 않음

### Calculation Basis

실행된 SQL + 행 수에서 도출된 결정론적, 현지화된 설명 — LLM이 아닌 코드가 생성.

```
적용 기준 (실행된 SQL에서 도출):
- 시간 범위: 2024-01-01 .. 2024-12-31
- 환불/상태 제외: 미적용
- 엔티티 상태 (SCD2): dw_customer: 고정 창 겹침 근사 ...

단계 1 — dw_customer
  필터: segment = 'VIP' AND valid_from <= '2024-12-31' ...
  결과: 782 rows
단계 2 — dw_sales_order (joined with 782 rows from step 1)
  조인: so.customer_id = ca.customer_id
  집계: SUM(total_amount) → 140,866.70 (1 row)
```

- 시간 범위 탐지, 상태 제외 탐지(`status != 'refunded'`), SCD2 의미 분류(point-in-time vs 고정 창 겹침 vs 무시)는 SQL의 구조적 파싱.
- 단계 라벨은 질문 언어로 현지화; 내부(테이블/컬럼명, 리터럴 값)는 영어/스키마 그대로 유지.
- `result.calculation_basis`(라이브러리)와 CLI의 dim 출력으로 제공.

---

## 신뢰도 & 근거

**Raw confidence**는 모델의 `Confidence: <0.0-1.0>` 줄(미기재시 기본 `0.7`).

**실행 신호 페널티** → `CandidateEvaluation.result_quality`:

| 신호 | 가중치 |
|------|--------|
| `syntax_error` | −0.10 × 횟수 (cap 3) |
| `execution_error` | −0.10 × 횟수 (cap 3) |
| `empty_result` | −0.15 |
| `schema_error` | −0.05 × 횟수 (cap 3) |
| `null_column` | −0.05 |
| `overflow_result` | −0.05 |

**Aggregator 신뢰도:**

```
best_conf = max(primary.confidence, selected_candidate.confidence)
adjusted  = best_conf × (1 − max_ambiguity × 0.5) × (1 − heuristic_penalty)
```

Heuristic: empty +0.15, error +0.15, TOP-N 불일치 +0.05, `by year`인데 year 없음 +0.10, AMBIGUOUS +0.10, FAILED/BLOCKED 각 +0.15 (cap 0.40).

---

## 디렉토리 구조

<details>
<summary><b>클릭하여 펼치기</b></summary>

```
src/syrch/
├── __init__.py          # Public API: query
├── api.py               # query() 고수준 함수
├── cli/app.py           # Typer CLI
├── core/
│   ├── models.py        # Dataclasses (ProblemSpec, TaskDAG, CandidateEvaluation …)
│   ├── config.py        # ExecutionConfig + 로더
│   └── logging.py       # 구조화된 로깅
├── executors/           # BaseExecutor + SQLite/JDBC/Databricks/Spark (+ CachedExecutor)
├── llm/                 # BaseLLM + OpenAI/Anthropic (+ CachedLLM / CentralCache)
├── search/
│   ├── planner.py       # D&C 분해 + replan
│   ├── scheduler.py     # DAG 실행 + context materialization
│   ├── rlm_engine.py    # 후보 탐색 REPL 루프
│   ├── search_policy.py # Beam / exhaustive
│   ├── path_evaluator.py# 품질 점수 + 랭킹 신호
│   ├── validator.py     # 하드 제약 검사
│   ├── retriever.py     # 키워드 스코어링 + 후보 풀
│   ├── semantic_index.py# 선택적 임베딩 기반 인덱스
│   ├── data_probe.py    # 공유 probe 캐시 (VALUE vs COVERAGE)
│   ├── question_norm.py # 비영어 질문 정규화
│   ├── aggregator.py    # 결과 병합 + 현지화된 calculation basis
│   ├── calibrator.py    # 실행 페널티
│   ├── clarify.py       # 모호성 감지
│   ├── grid.py          # 하이퍼파라미터 격자 탐색
│   └── pipeline.py      # 오케스트레이터
└── eval/                # 벤치마크 하네스 + 메트릭
tests/                  # 유닛 + 통합 테스트
scripts/                # validate_real.py, gen_fixtures.py
```

</details>

---

## 개발

```bash
# 개발 의존성 설치
pip install -e ".[dev]"

# 테스트 실행
pytest tests/ -v
pytest tests/ -v --cov=src/syrch/

# 린트 & 타입 체크
ruff check src/syrch/
mypy src/syrch/ --ignore-missing-imports

# 실제 LLM 검증 (API 키 필요)
python scripts/validate_real.py --quick
python scripts/validate_real.py --question "..." --db orders_10dim.sqlite --verbose
```

**CI** (push/PR → `main`): `ruff` → `mypy` → `pytest` (Python 3.11 + 3.12).

---

## 벤치마크 & 평가

```bash
# 예상 CSV 대비 단일 질문
syrch eval -q "Which click type generates the most traffic?" \
  --db wikipedia_clickstream.sqlite --expected expected.csv

# 전체 스위트
syrch benchmark --file benchmarks/orders.jsonl --report report.md
```

격자 탐색은 `max_depth`, `beam_width`, `max_attempts_per_node`, `calibration_enabled`(54 cells)를 스윕하고 `exact_match > confidence`로 보고합니다.

---

## 기여

[AGENTS.md](AGENTS.md)에서 모듈별 책임과 설계 원칙을 확인하세요.

브랜치 정책: `main`(안정, PR 전용) → `feat/*` / `fix/*`(squash merge) → `release/v*`(merge commit, `pyproject.toml` + `src/syrch/__init__.py` 버전 bump, 태그 `v*` → `publish.yml`로 PyPI 배포).

---

## 연구 배경

- **RLM** (Recursive Language Model) — MIT CSAIL OASIS 2025. [`paper`](https://arxiv.org/abs/2512.24601) [`code`](https://github.com/alexzhang13/rlm)
- **RDD** — Recursive Decomposition with Dependencies. [`paper`](https://arxiv.org/abs/2505.02576)
- **PAC-MCTS**, **ROMA**, **Graph Harness**, **AdaptOrch**, **DST**, **LLM Compiler** — 전체 목록은 [AGENTS.md](AGENTS.md#research-background) 참조.

---

## 라이선스

[MIT](LICENSE)
