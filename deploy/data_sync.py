#!/usr/bin/env python3
"""Bring a build's data files onto the live server WITHOUT overwriting anything (run ON the VM, by deploy_vm.sh).

    data_sync.py --app /opt/salesgod/app --build-data /tmp/salesgod-data-<ts> [--dry-run]
    data_sync.py --app /opt/salesgod/app --restore <backup-dir>

Why: deploys ship code only (data/ holds calls, recordings, profiles and console edits). On 15 Nov the deploy of
3e6c1f9 left the live products.json without FitZone's economics/ladder/tone, so live haggling was dead.

What it does (additive only):
  1. products.json: deploy/merge_products.merge(live, build) adds missing products, missing keys and missing FAQ items;
     every live value (console edits) is kept as it is.
  2. Seed files the live server does not have yet are copied: data/voice/knowledge/**, data/voice/playbooks/**,
     data/voice/calltypes.json. An existing live file is never replaced.
  3. Check (fails the deploy BEFORE the restart): every product validates against the code's Product model, and every
     product that is not a real company's (no "real_company"/allow_synthetic-false KB) and that the build gives an
     economics block also has a usable one live (list_price > 0 and a floor below it).
Backups of every file it changes go to <app>/data/voice/.sync-backup-<ts>/ (printed as BACKUP=<dir>).
Exit codes: 0 ok, 2 check failed (nothing written), 1 error.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

SEED_DIRS = ("knowledge", "playbooks")
SEED_FILES = ("calltypes.json",)


def _load_merge(app: Path):
    sys.path.insert(0, str(app / "deploy"))
    sys.path.insert(0, str(app))
    from merge_products import merge  # noqa: PLC0415
    return merge


def _usable_economics(e: dict | None) -> bool:
    if not isinstance(e, dict):
        return False
    lp = float(e.get("list_price") or 0)
    floor = max(float(e.get("cost") or 0) * (1 + float(e.get("min_margin_pct") or 0) / 100.0), float(e.get("floor_price") or 0))
    return lp > 0 and 0 < floor < lp


def check(app: Path, merged: dict, build: dict) -> list[str]:
    problems = []
    try:
        sys.path.insert(0, str(app))
        from acie.decision.domains.product_sales import Product  # noqa: PLC0415
    except Exception as e:  # the code tree itself is broken: refuse
        return [f"cannot import Product from the deployed code: {type(e).__name__}: {e}"]
    for pid, raw in merged.items():
        try:
            Product.model_validate(raw)
        except Exception as e:
            problems.append(f"{pid}: does not validate: {str(e)[:200]}")
    for pid, b in build.items():
        if pid in merged and _usable_economics(b.get("economics")) and not _usable_economics(merged[pid].get("economics")):
            problems.append(f"{pid}: the build has a negotiation range but the live product would not (haggling would be dead)")
    return problems


def run(app: Path, build_data: Path, dry: bool) -> int:
    live_voice = app / "data" / "voice"
    bvoice = build_data / "voice"
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup = live_voice / f".sync-backup-{ts}"
    changes: list[str] = []
    merge = _load_merge(app)

    live_p, build_p = live_voice / "products.json", bvoice / "products.json"
    live = json.loads(live_p.read_text(encoding="utf-8")) if live_p.exists() else {}
    build = json.loads(build_p.read_text(encoding="utf-8")) if build_p.exists() else {}
    added: list[str] = []
    merged = merge(live, build, added) if build else live
    problems = check(app, merged, build)
    if problems:
        print("DATA CHECK FAILED (nothing written):", *problems, sep="\n  ", file=sys.stderr)
        return 2
    if added:
        changes.append(f"products.json: +{len(added)} keys ({', '.join(added[:12])}{' …' if len(added) > 12 else ''})")
    seeds: list[tuple[Path, Path]] = []
    for d in SEED_DIRS:
        for src in sorted((bvoice / d).rglob("*")) if (bvoice / d).exists() else []:
            if src.is_file():
                dst = live_voice / d / src.relative_to(bvoice / d)
                if not dst.exists():
                    seeds.append((src, dst))
    for f in SEED_FILES:
        if (bvoice / f).exists() and not (live_voice / f).exists():
            seeds.append((bvoice / f, live_voice / f))
    changes += [f"seed: {dst.relative_to(live_voice)}" for _, dst in seeds]
    print("\n".join(changes) if changes else "data already up to date")
    if dry or not changes:
        return 0
    backup.mkdir(parents=True, exist_ok=True)
    if added:
        if live_p.exists():
            shutil.copy2(live_p, backup / "products.json")
        tmp = live_p.with_suffix(".json.sync-tmp")
        tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(live_p)
    (backup / "seeded.txt").write_text("\n".join(str(dst) for _, dst in seeds), encoding="utf-8")
    for src, dst in seeds:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"BACKUP={backup}")
    return 0


def restore(app: Path, backup: Path) -> int:
    live_voice = app / "data" / "voice"
    if (backup / "products.json").exists():
        shutil.copy2(backup / "products.json", live_voice / "products.json")
    seeded = backup / "seeded.txt"
    for line in (seeded.read_text(encoding="utf-8").splitlines() if seeded.exists() else []):
        p = Path(line)
        if p.exists() and live_voice in p.parents:
            p.unlink()
    print(f"restored data from {backup}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    ap.add_argument("--build-data")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore")
    a = ap.parse_args()
    app = Path(a.app)
    try:
        if a.restore:
            return restore(app, Path(a.restore))
        return run(app, Path(a.build_data), a.dry_run)
    except Exception as e:
        print(f"data_sync error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
