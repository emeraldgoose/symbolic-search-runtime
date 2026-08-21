"""SCD Type 2 generation invariants for the benchmark generator.

These tests guard the generator so a regenerated benchmark database keeps
physically valid SCD data: no inverted windows, contiguous version chains,
exactly one current version per natural key, and real attribute drift.
"""
import pandas as pd

from benchmark.generator.core.data import generate_table
from benchmark.generator.core.schema import load_all_schemas
from benchmark.generator.core.seed import get_rng

_SCD_TABLES = ["dw_customer", "dw_product", "dim_employee"]


def _scd_frames(table_name: str, profile: str = "small") -> tuple[pd.DataFrame, object]:
    tables = load_all_schemas()
    table = next(t for t in tables if t.name == table_name)
    rng = get_rng(profile, table_name)
    df = pd.concat(generate_table(rng, table, 5000), ignore_index=True)
    return df, table


def test_scd_no_inverted_windows():
    for name in _SCD_TABLES:
        df, table = _scd_frames(name)
        inv = df[df["valid_to"].notna() & (df["valid_from"] > df["valid_to"])]
        assert len(inv) == 0, f"{name}: {len(inv)} inverted valid windows"


def test_scd_no_zero_length_windows():
    for name in _SCD_TABLES:
        df, table = _scd_frames(name)
        zero = df[df["valid_to"].notna() & (df["valid_from"] == df["valid_to"])]
        assert len(zero) == 0, f"{name}: {len(zero)} zero-length windows"


def test_scd_contiguous_chain():
    """Version i's valid_to must equal version i+1's valid_from per natural key."""
    for name in _SCD_TABLES:
        df, table = _scd_frames(name)
        nk = table.natural_key
        srt = df.sort_values([nk, "valid_from"]).copy()
        srt["prev_to"] = srt.groupby(nk)["valid_to"].shift()
        breaks = srt[srt["prev_to"].notna() & (srt["prev_to"] != srt["valid_from"])]
        assert len(breaks) == 0, f"{name}: {len(breaks)} chain breaks"


def test_scd_exactly_one_current_version_per_key():
    df, table = _scd_frames("dw_customer")
    assert "is_current" in df.columns
    keys = df["customer_id"].nunique()
    assert df["is_current"].sum() == keys
    last = df.sort_values(["customer_id", "valid_from"]).groupby("customer_id").tail(1)
    assert last["is_current"].all()
    assert last["valid_to"].isna().all()


def test_scd_valid_window_covers_reference_end():
    """The final version must be open-ended (valid_to NULL) through ref_end."""
    df, _ = _scd_frames("dw_customer")
    current = df[df["is_current"]]
    assert current["valid_to"].isna().all()
    # no window may start after the reference horizon
    horizon = pd.Timestamp("2024-12-31")
    starts = pd.to_datetime(df["valid_from"])
    assert (starts <= horizon).all()


def test_scd_versions_reuse_natural_key_and_unique_surrogate():
    df, table = _scd_frames("dw_customer")
    assert df["customer_id"].nunique() < len(df)  # keys repeat across versions
    assert df["customer_sk"].nunique() == len(df)  # surrogate stays unique


def test_scd_attribute_drift_is_real():
    """SCD columns should actually change across versions (not identical per key)."""
    df, table = _scd_frames("dw_customer")
    multi = df[df.duplicated("customer_id", keep=False)]
    assert len(multi) > 0
    # at least some multi-version keys change their segment
    grouped = multi.groupby("customer_id")["segment"].nunique()
    assert (grouped > 1).any()


def test_scd_version_distribution_approx_two():
    df, _ = _scd_frames("dw_customer")
    vc = df.groupby("customer_id").size()
    avg = vc.mean()
    assert 1.5 <= avg <= 2.5
    assert (vc >= 1).all()
    assert (vc <= 4).all()