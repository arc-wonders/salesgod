"""Generate a product from the command line (same pipeline as the UI), e.g. on the server where the Gemini settings live:

  # on the voice server, as the salesgod user (the Gemini settings are in /opt/salesgod/.env)
  cd /opt/salesgod/app && set -a && . /opt/salesgod/.env && set +a
  python -m scripts.productgen.run --name "ZenBrew Coffee Subscription" --list 1200 --offer 999 --unit "per month" --floor 600
  python -m scripts.productgen.run --spec my_spec.json --publish          # --publish writes products.json / KB / playbook
  # repair an existing version in place of regenerating everything (saves v<N+1>; nothing is regenerated that already passes):
  python -m scripts.productgen.run --repair-version 1 --product-id zenbrew-coffee-subscription [--tenant default] [--publish]

Env: PRODUCTGEN_BASE_URL / PRODUCTGEN_MODEL / PRODUCTGEN_API_KEY / PRODUCTGEN_EXTRA_BODY override ACIE_EXPR_*. Keys are never printed.
Versions land in <data>/productgen/<tenant>/<product_id>/v<N>.json (default data dir: VOICE_DATA_DIR or data/voice).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from acie.productgen import store as S  # noqa: E402
from acie.productgen.generate import generate, repair_version  # noqa: E402
from acie.productgen.llm import configured, default_llm  # noqa: E402
from acie.productgen.spec import Spec  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--spec", help="a JSON file with the full spec (see acie/productgen/spec.py)")
    ap.add_argument("--list", type=float, dest="list_price")
    ap.add_argument("--offer", type=float)
    ap.add_argument("--unit", default="")
    ap.add_argument("--floor", type=float)
    ap.add_argument("--authorised", action="store_true", help="I own / am authorised to sell this")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--data", default=os.getenv("VOICE_DATA_DIR") or str(ROOT / "data" / "voice"))
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--repair-version", type=int, metavar="N", help="only run the repair loop on saved version N and save N+1 (needs --product-id, or --name / --spec)")
    ap.add_argument("--product-id", help="the product id (the folder name under productgen/<tenant>/)")
    a = ap.parse_args(argv)
    if a.repair_version:
        pid = a.product_id or (Spec.model_validate({"name": a.name}).pid if a.name else "")
        if not pid:
            ap.error("--repair-version needs --product-id (or --name)")
        if not configured():
            print("set PRODUCTGEN_BASE_URL and PRODUCTGEN_MODEL (or ACIE_EXPR_BASE_URL and ACIE_EXPR_MODEL, plus the _API_KEY)", file=sys.stderr)
            return 3
        llm = default_llm()
        print("model:", llm.model)
        try:
            doc = asyncio.run(repair_version(Path(a.data), a.tenant, pid, a.repair_version, llm, lambda st, status, detail="": print(f"  [{st}] {status} {detail}")))
        except KeyError as ex:
            print(str(ex), file=sys.stderr)
            return 2
        rep = doc["report"]
        print(f"repaired v{a.repair_version} -> saved v{doc['version']}; counts {rep['counts']}")
        for x in rep.get("dropped") or []:
            print("  removed:", x)
        for c in rep["checks"]:
            print(("  PASS " if c["passed"] else "  FAIL ") + c["label"] + ("" if c["passed"] else f" -- {c['detail']}"))
        if a.publish and rep["publishable"]:
            print(json.dumps(S.publish(Path(a.data), a.tenant, pid, doc["version"]), indent=1))
        return 0 if rep["publishable"] else 1
    if a.spec:
        d = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    elif a.name:
        d = {"name": a.name, "authorised": a.authorised, "tenant": a.tenant}
        price = {k: v for k, v in (("list_price", a.list_price), ("offer_price", a.offer), ("unit", a.unit)) if v not in (None, "")}
        if price:
            d["price"] = price
        if a.floor is not None:
            d["economics"] = {"floor": a.floor}
    else:
        ap.error("give --name or --spec")
    spec = Spec.model_validate(d)
    if not configured():
        print("set PRODUCTGEN_BASE_URL and PRODUCTGEN_MODEL (or ACIE_EXPR_BASE_URL and ACIE_EXPR_MODEL, plus the _API_KEY)", file=sys.stderr)
        return 3
    llm = default_llm()
    print("model:", llm.model)
    doc = asyncio.run(generate(spec, llm, Path(a.data), lambda st, status, detail="": print(f"  [{st}] {status} {detail}")))
    rep = doc["report"]
    print(f"version v{doc['version']} saved; counts {rep['counts']}")
    for c in rep["checks"]:
        print(("  PASS " if c["passed"] else "  FAIL ") + c["label"] + ("" if c["passed"] else f" -- {c['detail']}"))
    if a.publish:
        print(json.dumps(S.publish(Path(a.data), spec.tenant, doc["effective"]["product_id"], doc["version"]), indent=1))
    return 0 if rep["publishable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
