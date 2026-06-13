from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import diskcache

from syrch.llm.base import BaseLLM, LLMResponse


class CentralCache:
    def __init__(self, cache_dir: str | None = None, ttl: int = 86400):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.syrch/cache")
        self._cache = diskcache.Cache(cache_dir)
        self.ttl = ttl
        self.hit_count = 0
        self.miss_count = 0

    def _hash_key(self, prefix: str, **parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str)
        h = hashlib.sha256(raw.encode()).hexdigest()
        return f"{prefix}:{h}"

    def get(self, prefix: str, **parts: Any) -> Any | None:
        key = self._hash_key(prefix, **parts)
        val = self._cache.get(key)
        if val is not None:
            self.hit_count += 1
        else:
            self.miss_count += 1
        return val

    def set(self, prefix: str, value: Any, **parts: Any) -> None:
        key = self._hash_key(prefix, **parts)
        self._cache.set(key, value, expire=self.ttl)

    def clear(self, prefix: str | None = None) -> None:
        if prefix is None:
            self._cache.clear()
        else:
            keys = [k for k in self._cache.iterkeys() if k.startswith(f"{prefix}:")]
            for k in keys:
                del self._cache[k]

    def close(self) -> None:
        self._cache.close()


class CachedLLM(BaseLLM):
    def __init__(
        self,
        inner: BaseLLM,
        cache: CentralCache,
        model: str | None = None,
        temperature: float | None = None,
    ):
        self._inner = inner
        self._cache = cache
        self._model = model or getattr(inner, "model", "unknown")
        self._temperature = temperature

    @property
    def cache(self) -> CentralCache:
        return self._cache

    def _build_key_parts(
        self, system: str, user: str, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        parts: dict[str, Any] = dict(
            system=system, user=user, model=self._model
        )
        temp = kwargs.get("temperature", self._temperature)
        if temp is not None:
            parts["temperature"] = temp
        parts.update(kwargs)
        return parts

    def generate(self, system: str, user: str, **kwargs: Any) -> LLMResponse:
        key_parts = self._build_key_parts(system, user, kwargs)
        cached = self._cache.get("llm", **key_parts)
        if cached is not None:
            return LLMResponse(**cached)
        result = self._inner.generate(system, user, **kwargs)
        self._cache.set(
            "llm",
            dict(content=result.content, model=result.model, usage=result.usage),
            **key_parts,
        )
        return result

    def generate_json(self, system: str, user: str, **kwargs: Any) -> dict:
        key_parts = self._build_key_parts(system, user, kwargs)
        cached = self._cache.get("llm_json", **key_parts)
        if cached is not None:
            return cached
        result = self._inner.generate_json(system, user, **kwargs)
        self._cache.set("llm_json", result, **key_parts)
        return result
