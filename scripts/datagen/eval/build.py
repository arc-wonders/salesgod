import json, sys, random, collections, importlib
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import common
from common import ROWS, timing_for

batches = sys.argv[1:] or ["b1", "b2", "b3", "b4", "b5", "b6", "b7"]
for b in batches:
    try:
        importlib.import_module(b)
    except ModuleNotFoundError as e:
        if e.name != b:
            raise

out = []
for i, r in enumerate(ROWS, 1):
    tm = timing_for(r, i)
    if tm is None and r["b"]:
        r2 = dict(r, notime=False)
        tm = timing_for(r2, i + 7777) or {"barge_in": True, "response_latency_ms": 400, "speech_rate_wps": 3.2}
        tm["barge_in"] = True
    d = {
        "id": f"t{i:04d}",
        "lang": r["lang"],
        "text": r["text"],
        "context": {
            "prev_customer_text": r["prev"],
            "prev_agent_text": r["agent"],
            "dialog_pending": r["pend"],
            "pending_slot": r["slot"],
        },
    }
    if tm is not None:
        d["timing"] = tm
    d.update({
        "intent": r["intent"],
        "labels": {"frustration": r["f"], "confusion": r["c"], "urgency": r["u"], "engagement": r["e"]},
        "explicit": r["ex"],
        "amount": r["amt"],
        "notes": r["notes"],
    })
    out.append(d)

with open("/home/arkin/salesgod/data/eval/turns.jsonl", "w", encoding="utf-8") as fh:
    for d in out:
        fh.write(json.dumps(d, ensure_ascii=False) + "\n")
print(len(out), collections.Counter(d["lang"] for d in out), collections.Counter(d["intent"] for d in out))
L = lambda k, v: sum(1 for d in out if d["labels"][k] >= v)
print("f>=1", L("frustration", 1), "c>=1", L("confusion", 1), "u>=1", L("urgency", 1),
      "e0", sum(1 for d in out if d["labels"]["engagement"] == 0), "e2", sum(1 for d in out if d["labels"]["engagement"] == 2),
      "explicit", sum(1 for d in out if d["explicit"]), "barge", sum(1 for d in out if d.get("timing", {}).get("barge_in")))
