#!/usr/bin/env python3
"""Merge a build's products.json into the live one WITHOUT overwriting anything edited on the console.

    deploy/merge_products.py <live.json> <build.json> [-o merged.json]

Adds only what the live file is missing: whole products, missing keys of a product, missing keys inside dict fields
(info_source, economics.price_terms ...) and missing list items of faqs / benefits-like lists (a FAQ counts as present when
its English answer, or the same keyword set, is already there). Every existing live value is kept as it is. Prints a summary
of what was added to stderr; writes merged JSON to -o or stdout.
"""

import json
import sys


def _faq_key(f: dict):
    return (f.get("answer", "").strip().lower(), tuple(sorted(f.get("keywords", []))))


def merge(live: dict, build: dict, added: list[str] | None = None, path: str = "") -> dict:
    added = added if added is not None else []
    out = json.loads(json.dumps(live))
    for k, v in build.items():
        here = f"{path}{k}"
        if k not in out:
            out[k] = json.loads(json.dumps(v))
            added.append(here)
        elif isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge(out[k], v, added, here + ".")
        elif k == "faqs" and isinstance(out[k], list) and isinstance(v, list):
            have_a = {x.get("answer", "").strip().lower() for x in out[k]}
            have_k = {tuple(sorted(x.get("keywords", []))) for x in out[k]}
            for f in v:
                if f.get("answer", "").strip().lower() not in have_a and tuple(sorted(f.get("keywords", []))) not in have_k:
                    out[k].append(json.loads(json.dumps(f)))
                    added.append(f"{here}[+{f.get('answer', '')[:30]}]")
    return out


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "-o"]
    out_path = argv[argv.index("-o") + 1] if "-o" in argv else None
    if out_path:
        args.remove(out_path)
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    live, build = (json.load(open(p)) for p in args)
    added: list[str] = []
    merged = merge(live, build, added)
    text = json.dumps(merged, ensure_ascii=False, indent=1)
    (open(out_path, "w") if out_path else sys.stdout).write(text)
    print(f"merge_products: {len(added)} additions" + ("".join(f"\n  + {a}" for a in added[:60])), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
