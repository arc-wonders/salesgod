"""Build data/voice/knowledge/default/fitzone-gym.json from the hand-authored modules (synthetic demo data)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from acie.knowledge.store import save_kb, validate  # noqa: E402
from scripts.kb.authored import fitzone_e1, fitzone_e2, fitzone_e3, fitzone_e4, fitzone_meta as M  # noqa: E402
from scripts.kb.authored.common import build  # noqa: E402


def make():
    entries = fitzone_e1.ENTRIES + fitzone_e2.ENTRIES + fitzone_e3.ENTRIES + fitzone_e4.ENTRIES
    return build(M.PROD, "default", True, "synthetic",
                 "Operator's own demo gym. Every entry is synthetic demo data; it must not be used for a real company's product.",
                 M.TOPICS, entries, M.OFFERINGS, M.TAGS, M.ROOT)


if __name__ == "__main__":
    kb = make()
    prod = json.loads((ROOT / "data/voice/products.json").read_text(encoding="utf-8"))[M.PROD]
    issues = validate(kb, prod)
    print(len(kb.entries), "entries", len(kb.topics), "topics", len(kb.offerings), "offerings", issues)
    if issues:
        sys.exit(1)
    print(save_kb(kb, ROOT / "data/voice/knowledge/default" / f"{M.PROD}.json"))
