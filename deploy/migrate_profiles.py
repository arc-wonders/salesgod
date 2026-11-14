#!/usr/bin/env python
"""One-shot: import data/voice/profiles/*.json into Postgres (ACIE_PG_DSN). Safe to re-run (upsert).

    ACIE_PG_DSN=postgresql://acie:acie@localhost:15432/acie python deploy/migrate_profiles.py [dir] [--dry-run]
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acie.schemas import CustomerProfile  # noqa: E402


async def migrate(src: Path, store, dry_run: bool = False) -> tuple[int, int]:
    ok = bad = 0
    for p in sorted(src.glob("*.json")):
        try:
            prof = CustomerProfile.model_validate_json(p.read_text())
        except Exception as e:
            print(f"skip {p.name}: {e}", file=sys.stderr)
            bad += 1
            continue
        if not dry_run:
            await store.save(prof)
        ok += 1
    return ok, bad


async def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    src = Path(args[0] if args else "data/voice/profiles")
    dry = "--dry-run" in argv
    store = None
    if not dry:
        from acie.personalization.pg_store import PostgresProfileStore
        store = PostgresProfileStore(os.environ["ACIE_PG_DSN"])
    ok, bad = await migrate(src, store, dry)
    print(f"{'would import' if dry else 'imported'} {ok} profiles, skipped {bad}")
    if store:
        await store.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
