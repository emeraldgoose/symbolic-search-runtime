from __future__ import annotations

import logging

from syrch.llm.base import BaseLLM

logger = logging.getLogger(__name__)

_NORMALIZE_SYSTEM = (
    "You translate analytics questions into concise English search keywords "
    "for a database retriever. Preserve every business term (metrics, "
    "dimensions, filters, time ranges, entities) but strip conversational "
    "wording. Reply with the English keywords only."
)

_NORMALIZE_PROMPT = """Translate this question into English search keywords.

Question: {question}

English keywords:"""


def normalize_question(llm: BaseLLM, question: str, max_tokens: int = 128) -> str:
    """Rewrite a (possibly non-English) question into English search keywords.

    The retriever's `_tokenize` only extracts ASCII word tokens, so a Korean
    question like "2024년 VIP 고객의 순매출은?" yields no usable keywords and
    every table scores 0.0. This step translates the question so the retriever
    can match `vip`, `customer`, `net`, `revenue`. Falls back to the original
    question when the LLM call fails or returns unusable output.
    """
    stripped = (question or "").strip()
    if not stripped:
        return stripped
    if stripped.isascii():
        return stripped
    try:
        resp = llm.generate(
            _NORMALIZE_SYSTEM,
            _NORMALIZE_PROMPT.format(question=stripped),
            max_tokens=max_tokens,
        )
        text = (resp.content or "").strip()
        if not text:
            logger.warning("question_norm: LLM returned empty output; using original")
            return stripped
        if len(text) > 500:
            logger.warning("question_norm: output too long (%d chars); using original", len(text))
            return stripped
        return text
    except Exception as exc:  # noqa: BLE001
        logger.warning("question_norm: LLM failed (%s); using original", exc)
        return stripped
