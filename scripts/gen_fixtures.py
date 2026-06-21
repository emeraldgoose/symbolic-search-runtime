#!/usr/bin/env python3
"""Generate test fixture databases for syrch E2E tests.

Generates synthetic TPC-H-like orders data from scratch (no source DB needed).

Usage:
    python scripts/gen_fixtures.py                        # 1000 rows
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


def gen_orders(output: Path, rows: int, seed: int) -> None:
    rng = random.Random(seed)

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
    for i in range(rows):
        year = rng.choice(YEARS)
        quarter = rng.choice(QUARTERS)
        month = rng.choice(MONTHS)
        day_of_week = rng.choice(DAYS_OF_WEEK)
        totalprice = round(rng.uniform(1000.0, 500_000.0), 2)
        bucket = _bucket(totalprice)
        quartile = _quartile(totalprice)
        decile = _decile(totalprice)
        priority = rng.choice(PRIORITIES)
        status = rng.choice(STATUSES)
        clerk = f"Clerk#{rng.randint(1, 1000):09d}"

        data.append((
            year, quarter, month, day_of_week,
            bucket, quartile, decile,
            priority, status, clerk, totalprice,
        ))

    conn.executemany(
        "INSERT INTO orders_10dim VALUES (?,?,?,?,?,?,?,?,?,?,?)", data
    )
    conn.commit()
    actual = conn.execute("SELECT COUNT(*) FROM orders_10dim").fetchone()[0]
    conn.close()

    print(f"Generated {actual} rows of synthetic orders data")
    print(f"Written: {output} ({output.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Generate E2E test fixtures")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Number of rows")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    gen_orders(ORDERS_DB, args.rows, args.seed)


if __name__ == "__main__":
    main()
