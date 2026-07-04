from __future__ import annotations

import numpy as np

SEEDS = {"small": 42, "medium": 123, "enterprise": 456}


def _make_seed(base: int, table: str | None) -> int:
    if table:
        return abs(hash(f"{base}:{table}")) % (2**31)
    return base % (2**31)


def get_rng(profile: str, table: str | None = None) -> np.random.Generator:
    base = SEEDS.get(profile, 42)
    return np.random.Generator(np.random.PCG64(_make_seed(base, table)))

