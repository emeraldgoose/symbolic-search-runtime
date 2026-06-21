#!/usr/bin/env python3
"""Generate test fixture databases for syrch E2E tests.

Usage:
    python scripts/gen_fixtures.py                         # sample from full DB if exists
    python scripts/gen_fixtures.py --source path/to/orders_10dim.sqlite
    python scripts/gen_fixtures.py --rows 500              # override sample size
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
ORDERS_SAMPLE = FIXTURES_DIR / "orders_10dim.sqlite"
ORDERS_FULL = Path("orders_10dim.sqlite")
SAMPLE_ROWS = 1000


def gen_orders_sample(source: Path, output: Path, rows: int) -> None:
    if not source.exists():
        print(f"Source not found: {source}")
        print("Creating schema-only fixture (no data).")
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
        conn.commit()
        conn.close()
        print(f"Created schema-only: {output} ({os.path.getsize(output)} bytes)")
        return

    src = sqlite3.connect(str(source))
    dst = sqlite3.connect(str(output))

    dst.execute("""
        CREATE TABLE orders_10dim (
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

    count = src.execute("SELECT COUNT(*) FROM orders_10dim").fetchone()[0]
    sample = src.execute(
        f"SELECT * FROM orders_10dim WHERE rowid IN "
        f"(SELECT rowid FROM orders_10dim ORDER BY RANDOM() LIMIT {rows})"
    ).fetchall()

    dst.executemany(
        "INSERT INTO orders_10dim VALUES (?,?,?,?,?,?,?,?,?,?,?)", sample
    )
    dst.commit()

    actual = dst.execute("SELECT COUNT(*) FROM orders_10dim").fetchone()[0]
    src.close()
    dst.close()

    print(f"Sampled {actual}/{count} rows from {source.name}")
    print(f"Written: {output} ({os.path.getsize(output) / 1024:.0f} KB)")


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Generate E2E test fixtures")
    parser.add_argument("--source", default=str(ORDERS_FULL), help="Path to full orders_10dim.sqlite")
    parser.add_argument("--rows", type=int, default=SAMPLE_ROWS, help="Number of sample rows")
    args = parser.parse_args()

    gen_orders_sample(Path(args.source), ORDERS_SAMPLE, args.rows)


if __name__ == "__main__":
    main()
