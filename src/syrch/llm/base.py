from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict | None = None


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, **kwargs) -> LLMResponse:
        ...

    @abstractmethod
    def generate_json(self, system: str, user: str, **kwargs) -> dict:
        ...
