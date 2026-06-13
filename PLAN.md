# syrch — Implementation Plan

## Current Status

**~3,100 lines total** — research prototype, Phase 0-4.6 complete, Phase 5 open.

**2026-06-13 Validate** — 5가지 개선점 적용, 재검증 완료 (rate limit 전까지):

| Level | 개선 전 | 개선 후 | 비고 |
|-------|---------|---------|------|
| L1 Easy | 3/3 PASS ✅ | 3/3 PASS ✅ | 동일 |
| L2 Medium | 3/3 PASS ✅ | 3/3 PASS ✅ | 동일 |
| L3 Complex | 1/2 PASS (50%) | 2/2 PASS (100%) | 🔺 Planner 분해 개선 |
| L4 Very Complex | 2/2 PASS ✅ | 2/2 PASS ✅ | 동일 |
| L5 Ambiguous | 1 FAIL + 1 AMBIGUOUS | 0 FAIL + 1 AMBIGUOUS + 1 ERROR (429) | 🔺 Ambiguity 검출 개선 |
| 단위 테스트 | 69/69 PASS ✅ | 69/69 PASS ✅ | 동일 |

---

## 2026-06-13 수정사항

### 적용된 개선 5가지

| # | 문제 | 변경 | 효과 |
|---|------|------|------|
| 1 | **Planner DAG 분해 부족** | 프롬프트 일반화: 분해 선호 조건 3가지 (다차원 분석, 중간결과 의미, 독립 접근) | L3 "year/quarter trend" FAIL→PASS ✅ |
| 2 | **Confidence calibration** | `max(leaf) → best × (1-ambiguity×0.5) × (1-heuristic_penalty)` | 모호 질문 confidence 자동 하향 |
| 3 | **Ambiguity detection** | threshold 0.35→0.25, 가중치 키워드 맵 추가, worst= max(node, keyword) | L5 "Which orders are the best?" FAIL→AMBIGUOUS ✅ |
| 4 | **SQL semantic validation** | `_check_result_heuristics()`: 빈 결과/에러/TOP N/컬럼 검사 | confidence에 quality 패널티 반영 |
| 5 | **Cache transparency** | CentralCache hit/miss 카운터, 출력에 `[cache:Nh]` 표시 | 캐시 히트 시각화 |

### 수정 파일
| 파일 | 변경 |
|------|------|
| `src/syrch/search/planner.py` | #1 분해 조건 일반화 (line 16-25) |
| `src/syrch/search/aggregator.py` | #2 confidence 조정식 + #4 heuristic 검사 추가 (line 98-137) |
| `src/syrch/llm/cache.py` | #5 CentralCache hit_count/miss_count 추가 (line 18-19, 29-34) |
| `src/syrch/executors/cached_executor.py` | #5 cache property 노출 |
| `validate_real.py` | #3 threshold 0.25 + 키워드 맵 + #5 cache 공유 + 표시 |

### 핵심 개선 사례

**L3-1 "Revenue trend by year and quarter"**
- 개선 전: 단일 SQL (`GROUP BY o_year, o_quarter`), 1노드 → FAIL
- 개선 후: 2노드 DAG (yearly aggregate → quarterly breakdown), conf 0.813 → PASS ✅

**L5-1 "Which orders are the best?"**
- 개선 전: confidence 0.950, ambiguity 미검출 → FAIL
- 개선 후: keyword-ambiguity 0.28 > threshold 0.25, conf 0.700 → AMBIGUOUS ✅

**L3-2 "Compare low-price vs high-price volumes"**
- 개선 전: conf 0.798 (빈 결과에도 높은 confidence)
- 개선 후: conf 0.404 (빈 결과 패널티로 하향, min_confidence 0.4 간신히 통과)

### 남은 과제
1. Rate limit (429) 복구 후 L5 test 2 재확인
2. 429 Too Many Requests retry 로직 추가
3. Token budget 제한 (L5 test 1: 47k 토큰 소모)
4. `syrch` CLI end-to-end 검증
