"""Tests for FQN-aware retriever scoring and LLM question normalization."""

from syrch.core.models import ColumnSchema, TableSchema, infer_layer
from syrch.llm.base import BaseLLM, LLMResponse
from syrch.search.retriever import Retriever, _table_name_score
from syrch.search.question_norm import normalize_question


def test_table_name_score_matches_fqn_table():
    keywords = {"dw", "sales", "order"}
    score, reasons = _table_name_score(keywords, "syrch_benchmark.enterprise.dw_sales_order")
    assert score > 0
    assert reasons


def test_table_name_score_does_not_match_catalog_part():
    keywords = {"syrch", "benchmark"}
    score, _ = _table_name_score(keywords, "syrch_benchmark.enterprise.dw_sales_order")
    assert score == 0.0


def test_retriever_scores_fqn_table_by_base_name():
    name = "syrch_benchmark.enterprise.dw_sales_order"
    table = TableSchema(
        name=name,
        columns=[ColumnSchema(name="customer_id", type="INT")],
        layer=infer_layer(name),
    )
    retriever = Retriever([table])
    evidence = retriever.score("what is dw sales order revenue")
    assert evidence.candidates[0].score > 0
    assert evidence.candidates[0].features.layer == "dw"


class _FakeLLM(BaseLLM):
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def generate(self, system: str, user: str, **kwargs):
        self.calls += 1
        return LLMResponse(content=self.text, model="fake")

    def generate_json(self, system: str, user: str, **kwargs):
        return {}


def test_normalize_question_returns_llm_keywords():
    llm = _FakeLLM("vip customer net revenue")
    out = normalize_question(llm, "2024년 VIP 고객의 순매출은 얼마인가?")
    assert out == "vip customer net revenue"
    assert llm.calls == 1


def test_normalize_question_falls_back_on_empty():
    llm = _FakeLLM("   ")
    out = normalize_question(llm, "2024년 VIP 고객의 순매출은 얼마인가?")
    assert out == "2024년 VIP 고객의 순매출은 얼마인가?"
    assert llm.calls == 1


def test_normalize_question_falls_back_on_exception():
    class _Boom(BaseLLM):
        def generate(self, system: str, user: str, **kwargs):
            raise RuntimeError("boom")

        def generate_json(self, system: str, user: str, **kwargs):
            raise RuntimeError("boom")

    out = normalize_question(_Boom(), "2024년 VIP 고객의 순매출은 얼마인가?")
    assert out == "2024년 VIP 고객의 순매출은 얼마인가?"


def test_normalize_question_keeps_english_unchanged():
    llm = _FakeLLM("should not be used")
    out = normalize_question(llm, "What is total revenue?")
    assert llm.calls == 0
    assert out == "What is total revenue?"