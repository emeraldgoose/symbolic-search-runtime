from __future__ import annotations

import datetime
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
    if table.scd_columns and table.natural_key:
        yield from _generate_scd_table(rng, table, batch_size, fk_registry)
        return
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


def _generate_scd_table(
    rng: np.random.Generator,
    table: TableDef,
    batch_size: int = 10000,
    fk_registry: dict[str, pd.DataFrame] | None = None,
) -> Iterator[pd.DataFrame]:
    """Generate an SCD Type 2 table.

    Each natural key gets 1-3 versions. Versions for the same key form a
    contiguous validity chain: version i's valid_to == version i+1's valid_from
    (the final version has valid_to = NULL and is_current = True). SCD columns
    are re-sampled per version so attribute changes are real.
    """
    n_keys = table.rows
    if batch_size <= 0:
        batch_size = n_keys

    col_map = {c.name: c for c in table.columns}
    scd_cols = [c for c in table.scd_columns or [] if c in col_map]
    has_is_current = "is_current" in col_map
    has_valid_from = "valid_from" in col_map
    has_valid_to = "valid_to" in col_map
    natural_col = col_map[table.natural_key]
    pk_col = table.pk

    # A key (customer_id etc.) is a natural id. Per key, decide version count:
    # 40% 1 version, 40% 2 versions, 20% 3 versions -> avg ~1.8
    version_probs = np.array([0.40, 0.40, 0.20])
    version_counts = rng.choice([1, 2, 3], size=n_keys, p=version_probs)

    # Generate natural key values (must be unique per key).
    if natural_col.fk is not None and fk_registry is not None:
        ref_table, ref_col = natural_col.fk.split(".", 1)
        if ref_table in fk_registry and ref_col in fk_registry[ref_table].columns:
            valid = fk_registry[ref_table][ref_col].dropna().unique()
            if len(valid) > 0:
                nats = rng.choice(valid, size=n_keys, replace=False)
            else:
                nats = np.arange(1, n_keys + 1)
        else:
            nats = np.arange(1, n_keys + 1)
    elif natural_col.values is not None:
        nats = rng.choice(natural_col.values, size=n_keys, replace=True)
    else:
        lo = natural_col.dist_params.get("min", 1)
        hi = natural_col.dist_params.get("max", max(lo + n_keys, n_keys * 2))
        if hi - lo + 1 < n_keys:
            hi = lo + n_keys
        nats = rng.choice(np.arange(lo, hi + 1), size=n_keys, replace=False)

    # Stable per-key columns: pk value + non-SCD attributes sampled once per key.
    base_rows: list[dict[str, Any]] = []
    for i in range(n_keys):
        base: dict[str, Any] = {}
        for col in table.columns:
            if col.name == natural_col.name:
                base[col.name] = nats[i]
            elif pk_col and col.name == pk_col:
                pass  # surrogate pk filled below only if distinct from natural key
            elif col.name in scd_cols:
                pass  # sampled per version
            elif col.name == "valid_from" or col.name == "valid_to" or col.name == "is_current":
                pass
            else:
                base[col.name] = _generate_column(rng, col, 1, fk_registry)[0]
        base["_versions"] = int(version_counts[i])
        base_rows.append(base)

    # Build validity windows per key. Version windows tile [signup, 2024-12-31].
    ref_start = pd.Timestamp("2020-01-01")
    ref_end = pd.Timestamp("2024-12-31")

    out: list[pd.DataFrame] = []
    buf: list[dict[str, Any]] = []
    pk_counter = 0
    for base in base_rows:
        n_ver = base["_versions"]
        # Anchor the first version at the key's signup date when available.
        anchor = base.get("signup_date")
        anchor_ts = pd.Timestamp(anchor) if anchor is not None else ref_start
        if anchor_ts < ref_start:
            anchor_ts = ref_start
        if anchor_ts >= ref_end:
            anchor_ts = ref_end - datetime.timedelta(days=1)
        span_days = max(1, (ref_end - anchor_ts).days)
        # A version needs at least one full day; cap version count to the span.
        n_ver = min(n_ver, span_days)
        # Pick n_ver-1 strictly increasing cut points within [1, span_days).
        # This guarantees each window is non-empty and the chain stays contiguous.
        if n_ver > 1:
            pool = np.arange(1, span_days)
            if len(pool) < n_ver - 1:
                cuts = np.linspace(1, span_days - 1, n_ver - 1).astype(int)
            else:
                cuts = np.sort(rng.choice(pool, size=n_ver - 1, replace=False))
        else:
            cuts = np.array([], dtype=int)
        boundaries = [anchor_ts] + [anchor_ts + datetime.timedelta(days=int(c)) for c in cuts] + [ref_end]
        for v in range(n_ver):
            pk_counter += 1
            row = dict(base)
            del row["_versions"]
            if pk_col and pk_col != natural_col.name:
                row[pk_col] = pk_counter
            if has_valid_from:
                row["valid_from"] = boundaries[v]
            if has_valid_to:
                if v == n_ver - 1:
                    row["valid_to"] = None
                else:
                    row["valid_to"] = boundaries[v + 1]
            if has_is_current:
                row["is_current"] = v == n_ver - 1
            for scd_name in scd_cols:
                row[scd_name] = _generate_column(rng, col_map[scd_name], 1, fk_registry)[0]
            buf.append(row)
            if len(buf) >= batch_size:
                out.append(pd.DataFrame(buf))
                buf = []
    if buf:
        out.append(pd.DataFrame(buf))
    for df in out:
        yield df


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
