"""Tests for the caching layer (CentralCache, CachedLLM, CachedExecutor)."""

import time
import tempfile

import pandas as pd
import pytest

from syrch.llm.cache import CachedLLM, CentralCache
from syrch.executors.cached_executor import CachedExecutor


class FakeLLM:
    def __init__(self):
        self.generate_calls = 0
        self.generate_json_calls = 0

    def generate(self, system: str, user: str, **kwargs):
        self.generate_calls += 1
        return type("Response", (), {
            "content": f"result-{system[:5]}-{user[:5]}",
            "model": "fake",
            "usage": {"completion_tokens": 10},
        })()

    def generate_json(self, system: str, user: str, **kwargs):
        self.generate_json_calls += 1
        return {"answer": f"json-{system[:5]}-{user[:5]}"}


class FakeExecutor:
    def __init__(self):
        self.execute_calls = 0

    def execute(self, sql: str) -> pd.DataFrame:
        self.execute_calls += 1
        return pd.DataFrame({"result": [1, 2, 3]})

    def get_schema(self, table_name=None):
        return None

    def list_tables(self):
        return []

    def close(self):
        pass


@pytest.fixture
def tmp_cache():
    with tempfile.TemporaryDirectory() as d:
        yield CentralCache(cache_dir=d, ttl=3600)


class TestCentralCache:
    def test_set_and_get(self, tmp_cache):
        tmp_cache.set("test", 42, key="value")
        assert tmp_cache.get("test", key="value") == 42

    def test_miss(self, tmp_cache):
        assert tmp_cache.get("test", key="nonexistent") is None

    def test_overwrite(self, tmp_cache):
        tmp_cache.set("test", 1, key="v")
        tmp_cache.set("test", 2, key="v")
        assert tmp_cache.get("test", key="v") == 2

    def test_clear_prefix(self, tmp_cache):
        tmp_cache.set("a", 1, k=1)
        tmp_cache.set("b", 2, k=2)
        tmp_cache.set("a", 3, k=3)
        tmp_cache.clear(prefix="a")
        assert tmp_cache.get("a", k=1) is None
        assert tmp_cache.get("a", k=3) is None
        assert tmp_cache.get("b", k=2) == 2

    def test_clear_all(self, tmp_cache):
        tmp_cache.set("a", 1, k=1)
        tmp_cache.set("b", 2, k=2)
        tmp_cache.clear()
        assert tmp_cache.get("a", k=1) is None
        assert tmp_cache.get("b", k=2) is None

    def test_ttl_expiry(self, tmp_cache):
        tmp_cache.ttl = 1
        tmp_cache.set("test", 42, key="v")
        assert tmp_cache.get("test", key="v") == 42
        time.sleep(1.1)
        assert tmp_cache.get("test", key="v") is None

    def test_different_prefixes(self, tmp_cache):
        tmp_cache.set("a", 1, key="v")
        tmp_cache.set("b", 2, key="v")
        assert tmp_cache.get("a", key="v") == 1
        assert tmp_cache.get("b", key="v") == 2


class TestCachedLLM:
    def test_caches_generate(self, tmp_cache):
        inner = FakeLLM()
        cached = CachedLLM(inner, tmp_cache)
        r1 = cached.generate("sys", "usr")
        assert inner.generate_calls == 1
        r2 = cached.generate("sys", "usr")
        assert inner.generate_calls == 1
        assert r1.content == r2.content

    def test_caches_generate_json(self, tmp_cache):
        inner = FakeLLM()
        cached = CachedLLM(inner, tmp_cache)
        r1 = cached.generate_json("sys", "usr")
        assert inner.generate_json_calls == 1
        r2 = cached.generate_json("sys", "usr")
        assert inner.generate_json_calls == 1
        assert r1 == r2

    def test_different_inputs_miss(self, tmp_cache):
        inner = FakeLLM()
        cached = CachedLLM(inner, tmp_cache)
        cached.generate("sys1", "usr1")
        cached.generate("sys2", "usr2")
        assert inner.generate_calls == 2


class TestCachedExecutor:
    def test_caches_execute(self, tmp_cache):
        inner = FakeExecutor()
        cached = CachedExecutor(inner, tmp_cache)
        df1 = cached.execute("SELECT 1")
        assert inner.execute_calls == 1
        df2 = cached.execute("SELECT 1")
        assert inner.execute_calls == 1
        assert df1.equals(df2)

    def test_different_sql_miss(self, tmp_cache):
        inner = FakeExecutor()
        cached = CachedExecutor(inner, tmp_cache)
        cached.execute("SELECT 1")
        cached.execute("SELECT 2")
        assert inner.execute_calls == 2
