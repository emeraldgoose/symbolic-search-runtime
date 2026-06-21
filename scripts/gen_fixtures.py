#!/usr/bin/env python3
"""Generate test fixture databases for syrch E2E tests.

Generates synthetic data from scratch (no source DB needed).

Usage:
    python scripts/gen_fixtures.py                        # both fixtures, 1000 rows each
    python scripts/gen_fixtures.py --rows 5000            # custom row count
    python scripts/gen_fixtures.py --seed 42              # reproducible
"""
from __future__ import annotations

import argparse
import random
import sqlite3
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
ORDERS_DB = FIXTURES_DIR / "orders_10dim.sqlite"
WIKI_DB = FIXTURES_DIR / "wikipedia_clickstream.sqlite"

DEFAULT_ROWS = 1000
YEARS = list(range(1992, 1999))
QUARTERS = [1, 2, 3, 4]
MONTHS = list(range(1, 13))
DAYS_OF_WEEK = list(range(1, 8))
PRICE_BUCKETS = ["high", "medium", "low"]
QUARTILES = [1, 2, 3, 4]
DECILES = list(range(1, 11))
PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
STATUSES = ["F", "O", "P"]
CLERKS = [f"Clerk#{i:09d}" for i in range(1, 1001)]

TYPES = ["external", "internal", "other"]
REFERRERS = ["empty", "google", "facebook", "twitter", "bing", "yahoo", "wikipedia", "reddit"]
YES_NO = ["yes", "no"]
FIRST_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
TITLE_LENGTHS = ["short", "medium", "long"]


def _bucket(price: float) -> str:
    if price > 300_000:
        return "high"
    if price > 150_000:
        return "medium"
    return "low"


def _quartile(price: float) -> int:
    if price > 375_000:
        return 4
    if price > 250_000:
        return 3
    if price > 125_000:
        return 2
    return 1


def _decile(price: float) -> int:
    return min(int(price / 50_000) + 1, 10)


def gen_orders(output: Path, rows: int, rng: random.Random) -> None:
    conn = sqlite3.connect(str(output))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders_10dim (
            o_year INTEGER,
            o_quarter INTEGER,
            o_month INTEGER,
            o_day_of_week INTEGER,
            o_totalprice_bucket TEXT,
            o_totalprice_quartile INTEGER,
            o_totalprice_decile INTEGER,
            o_orderpriority TEXT,
            o_orderstatus TEXT,
            o_clerk TEXT,
            o_totalprice REAL
        )
    """)

    data: list[tuple] = []
    for _ in range(rows):
        year = rng.choice(YEARS)
        totalprice = round(rng.uniform(1000.0, 500_000.0), 2)
        data.append((
            year, rng.choice(QUARTERS), rng.choice(MONTHS), rng.choice(DAYS_OF_WEEK),
            _bucket(totalprice), _quartile(totalprice), _decile(totalprice),
            rng.choice(PRIORITIES), rng.choice(STATUSES), rng.choice(CLERKS), totalprice,
        ))

    conn.executemany(
        "INSERT INTO orders_10dim VALUES (?,?,?,?,?,?,?,?,?,?,?)", data
    )
    conn.commit()
    actual = conn.execute("SELECT COUNT(*) FROM orders_10dim").fetchone()[0]
    conn.close()
    print(f"Generated {actual} rows in {output.name}")


def gen_wikipedia_clickstream(output: Path, rng: random.Random) -> None:
    conn = sqlite3.connect(str(output))
    conn.execute("""
        CREATE TABLE wikipedia_clickstream (
            type TEXT, referrer_source TEXT, is_prev_article TEXT,
            curr_first_letter TEXT, curr_title_length TEXT,
            total_n INTEGER, row_count INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value REAL)
    """)

    data: list[tuple] = []
    for _ in range(200):
        total_n = rng.randint(1000, 200_000_000)
        rc = rng.randint(100, 200_000)
        data.append((
            rng.choice(TYPES), rng.choice(REFERRERS), rng.choice(YES_NO),
            rng.choice(FIRST_LETTERS), rng.choice(TITLE_LENGTHS),
            total_n, rc,
        ))

    conn.executemany(
        "INSERT INTO wikipedia_clickstream VALUES (?,?,?,?,?,?,?)", data
    )

    meta = [
        ("mi_type_referrer_source", round(rng.random(), 4)),
        ("mi_type_is_prev_article", round(rng.random(), 4)),
        ("mi_type_curr_first_letter", round(rng.random(), 4)),
        ("mi_type_curr_title_length", round(rng.random(), 4)),
        ("mi_referrer_source_is_prev_article", round(rng.random(), 4)),
        ("card_type", float(rng.randint(3, 10))),
        ("card_referrer_source", float(rng.randint(5, 20))),
    ]
    conn.executemany("INSERT INTO metadata VALUES (?,?)", meta)

    conn.commit()
    conn.close()
    print(f"Generated 200 wiki-clickstream rows + metadata in {output.name}")


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Generate E2E test fixtures")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Orders sample row count")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    gen_orders(ORDERS_DB, args.rows, rng)
    gen_wikipedia_clickstream(WIKI_DB, rng)

    total = (ORDERS_DB.stat().st_size + WIKI_DB.stat().st_size) / 1024
    print(f"Total fixture size: {total:.0f} KB")


if __name__ == "__main__":
    main()
