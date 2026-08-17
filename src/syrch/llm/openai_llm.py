from __future__ import annotations

import json
import os
import re
import time

from openai import OpenAI, RateLimitError

from syrch.llm.base import BaseLLM, LLMResponse


_MAX_RETRIES = 5
_INITIAL_BACKOFF = 1.0
_BACKOFF_MULTIPLIER = 2.0


def _message_content(message) -> str:
    """Extract assistant text, falling back to tool-call arguments.

    Reasoning models (e.g. LFM 2.5 on rapid-mlx) can emit their answer as a
    tool call (`sql_query(query=...)`) while leaving `message.content` empty.
    Without this fallback every such response was read as "empty SQL".
    """
    content = getattr(message, "content", None) or ""
    if content.strip():
        return content
    tool_calls = getattr(message, "tool_calls", None) or []
    parts: list[str] = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        raw = (getattr(fn, "arguments", None) or "") if fn is not None else ""
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parts.append(raw)
            continue
        if isinstance(parsed, dict):
            for key in ("query", "sql", "statement"):
                if key in parsed:
                    parts.append(str(parsed[key]))
                    break
            else:
                parts.append(json.dumps(parsed))
        else:
            parts.append(str(parsed))
    return "\n".join(parts)


def _rate_limit_retry(fn, *args, **kwargs):
    last_exc = None
    delay = _INITIAL_BACKOFF
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except RateLimitError as e:
            last_exc = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= _BACKOFF_MULTIPLIER
    raise last_exc


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, base_url: str | None = None):
        self.model = model
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key and base_url:
            api_key = "sk-no-key-required"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, system: str, user: str, **kwargs) -> LLMResponse:
        max_tokens = kwargs.get("max_tokens", 4096)
        timeout = kwargs.get("timeout", 120)
        response = _rate_limit_retry(
            self.client.chat.completions.create,
            model=kwargs.get("model", self.model),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=max_tokens,
            timeout=timeout,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        choice = response.choices[0]
        return LLMResponse(
            content=_message_content(choice.message),
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        )

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        max_tokens = kwargs.get("max_tokens", 4096)
        timeout = kwargs.get("timeout", 120)
        messages = [
            {"role": "system", "content": f"{system}\nRespond in JSON."},
            {"role": "user", "content": user},
        ]
        kw = dict(
            model=kwargs.get("model", self.model),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=max_tokens,
            timeout=timeout,
        )
        try:
            response = _rate_limit_retry(
                self.client.chat.completions.create,
                **kw, response_format={"type": "json_object"}, messages=messages,
            )
        except Exception:
            response = _rate_limit_retry(
                self.client.chat.completions.create, **kw, messages=messages,
            )
        content = _message_content(response.choices[0].message)
        if not content or not content.strip():
            return {}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    return {}
            return {}
