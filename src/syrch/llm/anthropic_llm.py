from __future__ import annotations

import json
import os

from anthropic import Anthropic

from syrch.llm.base import BaseLLM, LLMResponse


class AnthropicLLM(BaseLLM):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        self.model = model
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, system: str, user: str, **kwargs) -> LLMResponse:
        response = self.client.messages.create(
            model=kwargs.get("model", self.model),
            system=system,
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
            messages=[{"role": "user", "content": user}],
        )
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        response = self.client.messages.create(
            model=kwargs.get("model", self.model),
            system=f"{system}\nRespond with a valid JSON object only.",
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.7),
            messages=[{"role": "user", "content": user}],
        )
        text = response.content[0].text
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return {}
