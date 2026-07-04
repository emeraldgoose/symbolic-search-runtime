#!/usr/bin/env python3
"""
Generate benchmark databases for syrch Text-to-SQL evaluation.

Usage:
    python -m benchmark.generator.gen_business_db --profile small
    python benchmark/generator/gen_business_db.py --profile small
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import yaml

from benchmark.generator.backends.base import Backend
from benchmark.generator.backends.sqlite import SQLiteBackend
from benchmark.generator.core.data import generate_table
from benchmark.generator.core.quality import apply_all_quality
from benchmark.generator.core.schema import load_all_schemas, load_schema_yaml
from benchmark.generator.core.seed import get_rng

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"
_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"


def _load_profile(name: str) -> dict:
    path = _PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        available = [p.stem for p in sorted(_PROFILES_DIR.glob("*.yaml"))]
        print(f"Profile '{name}' not found. Available: {available}")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_extend(profile: dict, name: str) -> dict:
    while "extends" in profile:
        parent_name = profile.pop("extends")
        parent = _load_profile(parent_name)
        parent.update(profile)
        profile = parent
    return profile


def generate(profile_name: str, backends: list[str] | None = None, verify: bool = False) -> None:
    profile = _load_profile(profile_name)
    profile = _resolve_extend(profile, profile_name)

    seed = profile.get("seed", 42)
    batch_size = profile.get("batch_size", 50000)
    schema_include = profile.get("schema", {}).get("include", [])
    quality_config = profile.get("quality", {})
    output_config = profile.get("output", {})

    # Load all schema tables
    all_tables = []
    for s in schema_include:
        path = _SCHEMA_DIR / s
        if path.exists():
            all_tables.extend(load_schema_yaml(path))
        else:
            logger.warning("Schema file not found: %s", path)

    # Clean up existing SQLite DB before generation
    if "sqlite" in output_config:
        sqlite_path = Path(output_config["sqlite"]["path"])
        if sqlite_path.exists():
            sqlite_path.unlink()
            logger.info("Removed existing SQLite DB: %s", sqlite_path)

    # Determine backends
    if backends is None:
        backends = []
        if "sqlite" in output_config:
            backends.append("sqlite")
        if "delta" in output_config:
            backends.append("delta")

    active_backends: list[Backend] = []
    try:
        for b in backends:
            if b == "sqlite" and "sqlite" in output_config:
                sqlite_cfg = output_config["sqlite"]
                active_backends.append(SQLiteBackend(sqlite_cfg["path"]))
                logger.info("SQLite output: %s", sqlite_cfg["path"])
            elif b == "delta" and "delta" in output_config:
                delta_cfg = output_config["delta"]
                try:
                    from benchmark.generator.backends.delta import DeltaBackend
                    active_backends.append(DeltaBackend(
                        catalog=delta_cfg.get("catalog", "benchmark"),
                        schema=delta_cfg.get("schema", "default"),
                        location=delta_cfg.get("location"),
                        clustering=delta_cfg.get("clustering", "liquid"),
                        optimize=delta_cfg.get("optimize", True),
                        tblproperties=delta_cfg.get("tblproperties"),
                    ))
                    logger.info("Delta output: %s.%s", delta_cfg.get("catalog"), delta_cfg.get("schema"))
                except ImportError as e:
                    logger.error("Delta backend not available: %s", e)
                    logger.error("Install: pip install pyspark delta-spark")

        if not active_backends:
            logger.error("No backends configured for profile '%s'", profile_name)
            sys.exit(1)

        # Generate each table
        for table_def in all_tables:
            rng = get_rng(profile_name, table_def.name)
            start = time.time()

            # Create table schema in each backend
            for be in active_backends:
                be.create_table(table_def)

            # Generate and write data in batches
            n_rows = 0
            for batch_df in generate_table(rng, table_def, batch_size):
                batch_df = apply_all_quality(batch_df, quality_config, rng)
                for be in active_backends:
                    be.write_batch(table_def.name, batch_df)
                n_rows += len(batch_df)

            # Finalize (OPTIMIZE for Delta, ANALYZE for SQLite)
            for be in active_backends:
                be.finalize(table_def.name)

            elapsed = time.time() - start
            logger.info("%-30s %8d rows  %5.1fs", table_def.name, n_rows, elapsed)

        # Verify
        if verify:
            verify_all(active_backends)

    finally:
        for be in active_backends:
            try:
                be.close()
            except Exception:
                pass


def verify_all(backends: list[Backend]) -> None:
    print("\n=== Cross-Backend Verification ===")
    results = [be.verify() for be in backends]
    names = [type(be).__name__ for be in backends]

    all_tables: set[str] = set()
    for r in results:
        all_tables.update(r.keys())

    if len(results) == 2:
        r0, r1 = results
        all_ok = True
        for t in sorted(all_tables):
            s0 = r0.get(t, {})
            s1 = r1.get(t, {})
            rows_match = s0.get("rows") == s1.get("rows")
            sum_match = s0.get("checksum") == s1.get("checksum")
            ok = rows_match and sum_match
            status = "✅" if ok else "❌"
            if not ok:
                all_ok = False
            print(f"  {status} {t:<30} rows={s0.get('rows'):>8}/{s1.get('rows'):<8}  "
                  f"sum={'OK' if sum_match else 'MISMATCH'}  ({names[0]} vs {names[1]})")
        if all_ok:
            print("ALL TABLES VERIFIED ✅")
        else:
            print("SOME TABLES MISMATCHED ❌")
    else:
        for i, r in enumerate(results):
            print(f"\n  {names[i]}:")
            for t in sorted(r.keys()):
                s = r[t]
                print(f"    {t:<30} rows={s.get('rows'):>8}  checksum={s.get('checksum')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate syrch benchmark databases")
    parser.add_argument("--profile", default="small", help="Benchmark profile (small/medium/enterprise)")
    parser.add_argument("--backend", nargs="*", choices=["sqlite", "delta", "all"],
                        help="Output backends (default: from profile)")
    parser.add_argument("--verify", action="store_true", help="Verify cross-backend consistency")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-5s %(message)s",
    )

    if args.backend and "all" in args.backend:
        backend_list = ["sqlite", "delta"]
    else:
        backend_list = args.backend

    generate(args.profile, backends=backend_list, verify=args.verify)


if __name__ == "__main__":
    main()
