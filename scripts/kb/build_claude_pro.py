"""Build data/voice/knowledge/default/claude-pro.json: public facts only, allow_synthetic=false."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from acie.knowledge.store import save_kb, validate  # noqa: E402
from scripts.kb.authored import claude_pro as C  # noqa: E402
from scripts.kb.authored.common import build  # noqa: E402


def make():
    return build(C.PROD, "default", False, "public", "Real company's product: public facts only; no synthetic entries may exist or be served.",
                 C.TOPICS, C.ENTRIES, C.OFFERINGS, C.TAGS, C.ROOT)


if __name__ == "__main__":
    kb = make()
    prod = json.loads((ROOT / "data/voice/products.json").read_text(encoding="utf-8"))[C.PROD]
    issues = validate(kb, prod)
    print(len(kb.entries), "entries", len(kb.topics), "topics", len(kb.offerings), "offerings", issues)
    if issues:
        sys.exit(1)
    print(save_kb(kb, ROOT / "data/voice/knowledge/default" / f"{C.PROD}.json"))
