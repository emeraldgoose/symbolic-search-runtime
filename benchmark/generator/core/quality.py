from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def break_fk(
    df: pd.DataFrame,
    fk_col: str,
    break_ratio: float,
    rng: np.random.Generator,
    replacement_val: int | None = -1,
) -> pd.DataFrame:
    if break_ratio <= 0:
        return df
    mask = rng.random(len(df)) < break_ratio
    if replacement_val is not None:
        df.loc[mask, fk_col] = replacement_val
    else:
        df.loc[mask, fk_col] = None
    return df


def inject_duplicates(
    df: pd.DataFrame,
    dup_ratio: float,
    rng: np.random.Generator,
    key_cols: list[str] | None = None,
) -> pd.DataFrame:
    if dup_ratio <= 0:
        return df
    n = len(df)
    n_dup = int(n * dup_ratio)
    if n_dup == 0:
        return df
    idx = rng.choice(n, size=n_dup, replace=False)
    dup_rows = df.iloc[idx].copy()
    if key_cols:
        for c in key_cols:
            dup_rows[c] = dup_rows[c] + rng.integers(10000, 99999, size=n_dup)
    return pd.concat([df, dup_rows], ignore_index=True)


def inject_late_arrival(
    df: pd.DataFrame,
    date_col: str,
    etl_col: str,
    late_ratio: float,
    rng: np.random.Generator,
    max_lag_days: int = 7,
) -> pd.DataFrame:
    if late_ratio <= 0:
        return df
    mask = rng.random(len(df)) < late_ratio
    lag = rng.integers(1, max_lag_days + 1, size=mask.sum())
    df.loc[mask, etl_col] = pd.to_datetime(df.loc[mask, date_col]) + pd.to_timedelta(lag, unit="D")
    return df


def apply_all_quality(
    df: pd.DataFrame,
    table_config: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    quality = table_config.get("quality", {})
    for col, rules in quality.items():
        if col not in df.columns:
            continue
        null_ratio = rules.get("null_ratio", 0)
        if null_ratio > 0:
            df = inject_null(df, col, null_ratio, rng)
        outlier_ratio = rules.get("outlier_ratio", 0)
        if outlier_ratio > 0:
            multiplier = rules.get("outlier_multiplier", 10.0)
            df = inject_outliers(df, col, outlier_ratio, rng, multiplier)
        fk_break = rules.get("fk_break_ratio", 0)
        if fk_break > 0:
            replacement = rules.get("fk_replacement", -1)
            df = break_fk(df, col, fk_break, rng, replacement)
    dup_ratio = table_config.get("duplicate_ratio", 0)
    if dup_ratio > 0:
        key_cols = table_config.get("duplicate_key_cols")
        df = inject_duplicates(df, dup_ratio, rng, key_cols)
    late_arrival = table_config.get("late_arrival", {})
    if late_arrival.get("enabled", False):
        df = inject_late_arrival(
            df,
            late_arrival.get("date_col", ""),
            late_arrival.get("etl_col", "etl_loaded_at"),
            late_arrival.get("ratio", 0),
            rng,
            late_arrival.get("max_lag_days", 7),
        )
    return df


def inject_null(df: pd.DataFrame, col: str, null_ratio: float, rng: np.random.Generator) -> pd.DataFrame:
    if null_ratio <= 0:
        return df
    mask = rng.random(len(df)) < null_ratio
    df.loc[mask, col] = None
    return df


def inject_outliers(df: pd.DataFrame, col: str, outlier_ratio: float, rng: np.random.Generator, multiplier: float = 10.0) -> pd.DataFrame:
    if outlier_ratio <= 0:
        return df
    mask = rng.random(len(df)) < outlier_ratio
    if df[col].dtype.kind in ("i", "f"):
        median = df[col].median()
        df.loc[mask, col] = median * multiplier
    return df
