from __future__ import annotations

import typing

__version__ = "0.2.0"


def __getattr__(name: str) -> typing.Any:
    if name == "query":
        from syrch.api import query
        return query
    if name == "SearchResult":
        from syrch.api import SearchResult
        return SearchResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return ["query", "SearchResult"]
