from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from benchmark.generator.core.schema import ColumnDef, TableDef


def _generate_column(rng: np.random.Generator, col: ColumnDef, n: int,
                     fk_registry: dict[str, pd.DataFrame] | None = None) -> np.ndarray | pd.Series:
    if col.fk is not None and fk_registry is not None:
        ref_table, ref_col = col.fk.split(".", 1)
        if ref_table in fk_registry and ref_col in fk_registry[ref_table].columns:
            valid = fk_registry[ref_table][ref_col].dropna().unique()
            if len(valid) > 0:
                return rng.choice(valid, size=n).astype(valid.dtype)

    if col.values is not None:
        probs = col.dist_params.get("probs", None)
        if isinstance(probs, list) and len(probs) == len(col.values):
            p = np.array(probs, dtype=float)
            p /= p.sum()
            replace = True
        else:
            p = None
            replace = n > len(col.values)
        return pd.Categorical(rng.choice(col.values, size=n, p=p, replace=replace))

    dtype = col.type.upper()

    if dtype in ("INTEGER", "INT", "BIGINT", "LONG"):
        lo = col.dist_params.get("min", 1)
        hi = col.dist_params.get("max", 1000000)
        if col.distribution == "zipf":
            a = col.dist_params.get("a", 2.0)
            samples = rng.zipf(a, size=n)
            samples = np.clip(samples, lo, hi).astype(np.int64)
            return samples
        return rng.integers(lo, hi + 1, size=n).astype(np.int64)

    if dtype in ("REAL", "FLOAT", "DOUBLE", "DECIMAL"):
        if col.distribution == "lognormal":
            mean = col.dist_params.get("mean", 4.0)
            sigma = col.dist_params.get("sigma", 0.5)
            return rng.lognormal(mean, sigma, size=n).astype(np.float64)
        if col.distribution == "normal":
            mean = col.dist_params.get("mean", 100.0)
            std = col.dist_params.get("std", 20.0)
            return rng.normal(mean, std, size=n).astype(np.float64)
        lo = col.dist_params.get("min", 0.0)
        hi = col.dist_params.get("max", 1.0)
        return rng.uniform(lo, hi, size=n).astype(np.float64)

    if dtype in ("DATE",):
        start = col.dist_params.get("start", "2022-01-01")
        end = col.dist_params.get("end", "2024-12-31")
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        days = (end_ts - start_ts).days
        if days <= 0:
            return pd.Series([start_ts.date()] * n)
        offsets = rng.integers(0, days + 1, size=n)
        dates = start_ts + pd.to_timedelta(offsets, unit="D")
        return dates

    if dtype in ("DATETIME", "TIMESTAMP", "TIMESTAMP_NTZ"):
        start = col.dist_params.get("start", "2022-01-01")
        end = col.dist_params.get("end", "2024-12-31")
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        span = (end_ts - start_ts).total_seconds()
        if span <= 0:
            return pd.Series([start_ts] * n)
        offsets = rng.uniform(0, span, size=n)
        return start_ts + pd.to_timedelta(offsets, unit="s")

    if dtype in ("BOOLEAN", "BOOL"):
        p_true = col.dist_params.get("p_true", 0.5)
        return rng.choice([True, False], size=n, p=[p_true, 1 - p_true])

    return pd.Series([f"v_{i:05d}" for i in range(n)])


import datetime as _dt  # noqa: E402


def _holiday_set() -> set[tuple[int, int]]:
    """Real fixed-date holidays (KR + major international)."""
    return {
        (1, 1), (1, 2), (3, 1), (5, 5), (6, 6), (8, 15), (10, 3), (10, 9), (12, 25),
    }


def _season(month: int) -> str:
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    if month in (9, 10, 11):
        return "Fall"
    return "Winter"


def generate_date_dimension(table: TableDef, start: int = 0, count: int | None = None) -> pd.DataFrame:
    """Generate a real calendar for a date-grain dimension table.

    The first ``count`` rows start at 2022-01-01. Column values are derived
    from the actual date instead of random generation, so ``date_key`` can be
    joined against DATE columns and ``year``/``month``/``is_holiday`` are real.
    """
    n = table.rows
    if count is None:
        count = n
    start_date = _dt.date(2022, 1, 1)
    holidays = _holiday_set()
    data: dict[str, Any] = {}
    for col in table.columns:
        col_name = col.name
        values: list[Any] = []
        for i in range(start, start + count):
            d = start_date + _dt.timedelta(days=i)
            if col_name == "date_key":
                values.append(d.strftime("%Y-%m-%d"))
            elif col_name == "year":
                values.append(d.year)
            elif col_name == "quarter":
                values.append((d.month - 1) // 3 + 1)
            elif col_name == "month":
                values.append(d.month)
            elif col_name == "day_of_week":
                values.append(d.weekday())
            elif col_name == "is_weekend":
                values.append(d.weekday() >= 5)
            elif col_name == "is_holiday":
                values.append((d.month, d.day) in holidays)
            elif col_name == "season":
                values.append(_season(d.month))
            else:
                # Unknown columns in a date dimension fall back to raw values.
                values.append(f"v_{i:05d}")
        data[col.name] = values
    return pd.DataFrame(data)


def generate_table(
    rng: np.random.Generator,
    table: TableDef,
    batch_size: int = 10000,
    fk_registry: dict[str, pd.DataFrame] | None = None,
) -> Iterator[pd.DataFrame]:
    n = table.rows
    if batch_size <= 0:
        batch_size = n

    if table.grain == "date":
        # Calendar dimensions are fully deterministic (no RNG) and span
        # 2022-01-01 + table.rows days.
        for start in range(0, n, batch_size):
            yield generate_date_dimension(table, start, min(batch_size, n - start))
        return

    start = 0
    while start < n:
        end = min(start + batch_size, n)
        m = end - start
        data: dict[str, Any] = {}
        for col in table.columns:
            data[col.name] = _generate_column(rng, col, m, fk_registry)
        df = pd.DataFrame(data)
        yield df
        start = end


def inject_null(
    df: pd.DataFrame,
    col: str,
    null_ratio: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if null_ratio <= 0:
        return df
    mask = rng.random(len(df)) < null_ratio
    df.loc[mask, col] = None
    return df


def inject_outliers(
    df: pd.DataFrame,
    col: str,
    outlier_ratio: float,
    rng: np.random.Generator,
    multiplier: float = 10.0,
) -> pd.DataFrame:
    if outlier_ratio <= 0:
        return df
    mask = rng.random(len(df)) < outlier_ratio
    if df[col].dtype.kind in ("i", "f"):
        median = df[col].median()
        df.loc[mask, col] = median * multiplier
    return df
