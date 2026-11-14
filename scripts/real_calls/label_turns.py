"""Label the real customer turns -> data/real/turns_labeled.jsonl (git-ignored; never commit).

Schema = data/eval/turns.jsonl (+ `state14` 0/1/2 per HumanState dim, `need`, `text_careful`). The 14 levels come
from the repo's rule estimator on the turn in its call context (low < 0.34 <= mid < 0.67 <= high); the need comes
from the deterministic table, then a hand-read OVERRIDES entry where reading the turn in context disagrees.
Sorted by id. Run: python scripts/real_calls/label_turns.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import ROOT, load_calls, script  # noqa: E402

from acie.psyche.needs import DIMS, NeedInput, table_need  # noqa: E402
from acie.schemas import SessionState, TurnInput, TurnTiming, Utterance  # noqa: E402
from acie.state.estimator import HumanStateEstimator  # noqa: E402

# hand-read corrections: (call id6, turn index) -> need
OVERRIDES = {
    ("018a0b", 1): "clarity", ("018a0b", 3): "clarity", ("018a0b", 5): "be_heard", ("018a0b", 7): "be_heard",
    ("07b508", 9): "information", ("07b508", 11): "closure",
    ("4b10be", 7): "be_heard", ("4b10be", 11): "be_heard",
    ("f8ad3e", 3): "be_heard",
    ("339d4a", 3): "clarity", ("339d4a", 13): "be_heard", ("339d4a", 15): "be_heard", ("339d4a", 17): "autonomy",
    ("7abf60", 7): "autonomy", ("7abf60", 9): "answer", ("7abf60", 11): "answer", ("7abf60", 13): "answer",
    ("7abf60", 15): "answer", ("7abf60", 17): "answer",
    ("9bde11", 13): "answer", ("9bde11", 15): "answer", ("9bde11", 17): "answer", ("9bde11", 19): "answer",
    ("9bde11", 21): "reassurance", ("9bde11", 23): "reassurance",
    ("899b50", 3): "reassurance", ("899b50", 5): "reassurance", ("899b50", 7): "reassurance", ("899b50", 9): "reassurance",
    ("912c39", 11): "reassurance", ("912c39", 15): "reassurance", ("912c39", 17): "reassurance",
    ("194d15", 9): "reassurance", ("194d15", 11): "answer", ("194d15", 13): "reassurance", ("194d15", 18): "be_heard",
    ("7920ed", 5): "reassurance", ("7920ed", 7): "clarity", ("7920ed", 9): "reassurance",
    ("a65aee", 9): "be_heard", ("e45d62", 5): "clarity", ("e45d62", 7): "clarity",
    ("937e13", 3): "clarity", ("bed9a1", 1): "space", ("a7152c", 3): "closure",
    ("1a6306", 3): "autonomy", ("1a6306", 5): "answer", ("1a6306", 7): "autonomy",
    ("8d88aa", 5): "autonomy", ("8d88aa", 7): "autonomy", ("899b50", 1): "autonomy",
    ("b3dbfe", 5): "autonomy", ("5b478a", 5): "clarity", ("77b155", 3): "clarity", ("912c39", 1): "clarity",
    ("9d69a3", 5): "clarity", ("e45d62", 3): "clarity",
}


def level(v: float) -> int:
    return 0 if v < 0.34 else (1 if v < 0.67 else 2)


def main() -> None:
    est = HumanStateEstimator()
    rows = []
    for c in load_calls():
        sess = SessionState(session_id=c["_id6"], customer_id="x")
        prev_c = prev_a = None
        careful = {}
        if c.get("_review"):
            for t in c["_review"]["turns"]:
                if t["role"] == "customer":
                    careful.setdefault(len([1 for k in careful]), t.get("second_opinion"))
        n_c = 0
        for i, t in enumerate(c["turns"]):
            if t["role"] == "agent":
                prev_a = t["text"]
                sess.history.append(Utterance(role="agent", text=t["text"]))
                continue
            turn = TurnInput(session_id=c["_id6"], customer_id="x", text=t["text"],
                             timing=TurnTiming(barge_in=bool(t.get("barge_in"))))
            st = est.estimate(turn, sess, None)
            dims = {k: level(getattr(st, k)) for k in DIMS}
            nxt = c["turns"][i + 1] if i + 1 < len(c["turns"]) else {}
            res = table_need(NeedInput(text=t["text"], state={k: getattr(st, k) for k in DIMS}, intent="",
                                       has_link=any(u.role == "agent" and "SMS" in u.text and "भेज रहा" in u.text for u in sess.history)))
            need = OVERRIDES.get((c["_id6"], i), res.need)
            rows.append({
                "id": f"r{c['_id6']}_{i:02d}", "lang": {"hi": "hi", "en": "en"}.get(t.get("stt_lang"), "hinglish") if script(t["text"]) != "mix" else "hinglish",
                "text": t["text"], "text_careful": careful.get(n_c) if careful.get(n_c) not in (None, t["text"]) else None,
                "context": {"prev_customer_text": prev_c, "prev_agent_text": prev_a, "dialog_pending": None, "pending_slot": None},
                "timing": {"barge_in": bool(t.get("barge_in")), "response_latency_ms": None, "speech_rate_wps": None},
                "intent": nxt.get("intent") or "unknown",
                "labels": {"frustration": dims["frustration"], "confusion": dims["confusion"], "urgency": dims["urgency"],
                           "engagement": dims["engagement"]},
                "explicit": {}, "amount": None, "state14": dims, "need": need, "need_table": res.need,
                "notes": f"real call turn; need {'hand-read' if (c['_id6'], i) in OVERRIDES else 'table'}"})
            sess.history.append(Utterance(role="customer", text=t["text"]))
            sess.state = st
            prev_c = t["text"]
            n_c += 1
    rows.sort(key=lambda r: r["id"])
    out = ROOT / "data/real/turns_labeled.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    from collections import Counter
    print(len(rows), "turns;", dict(Counter(r["need"] for r in rows)), "; hand-read:", sum(r["need"] != r["need_table"] for r in rows))


if __name__ == "__main__":
    main()
