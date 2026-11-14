import json
import subprocess
import sys
from pathlib import Path

from acie.knowledge import retrieve
from acie.knowledge.store import load_kb

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "data/voice/products.json").read_text(encoding="utf-8"))
FZ, CP = P["fitzone-gym"], P["claude-pro"]


def ids(q, prod=FZ, k=3, **kw):
    r = retrieve(prod, q, product=prod, k=k, **kw)
    return [h.id for h in r.hits], r


def test_powerlifting_in_three_languages():
    for q in ("what do you have for powerlifting", "powerlifting ke liye kya hai", "पावरलिफ्टिंग के लिए क्या है"):
        got, r = ids(q)
        assert r.confident and got[0] == "pl-overview", (q, got)


def test_bench_press_and_competition_prep():
    assert ids("bench press")[0][0] == "sz-bench"
    assert ids("bench press kaise badhaun")[0][0] == "pl-bench-tip"
    assert ids("competition prep")[0][0] == "pl-meet-prep"
    assert ids("Do you have any provisions for specific competitions?")[0][0] == "pl-meet-prep"
    assert ids("कॉम्पिटिशन की तैयारी होती है क्या")[0][0] == "pl-meet-prep"
    assert ids("competition ki taiyari karwate ho")[0][0] == "pl-meet-prep"


def test_reply_language():
    r = retrieve(FZ, "powerlifting ke liye kya hai", "hi", product=FZ)
    assert "पावरलिफ्टिंग" in r.hits[0].text
    r = retrieve(FZ, "powerlifting", "hinglish", product=FZ)
    assert "Haan" in r.hits[0].text
    r = retrieve(FZ, "powerlifting", "en", product=FZ)
    assert r.hits[0].text.startswith("Yes")


def test_price_slots_are_filled_from_config_and_offerings():
    r = retrieve(FZ, "how much is the membership", "en", product=FZ)
    assert "₹2,999 for 3 months" in r.hits[0].text
    r = retrieve(FZ, "how much is the powerlifting coaching program", "en", product=FZ)
    assert "₹3,000 per month" in r.hits[0].text


def test_not_found_gives_overview_never_contact_owner():
    r = retrieve(FZ, "zzqx plorf wibble", "en", product=FZ)
    assert r.fallback and not r.confident and r.hits and r.hits[0].overview
    assert "owner" not in r.hits[0].text.lower() and "website" not in r.hits[0].text.lower()
    r = retrieve(FZ, "tell me about nutrition stuff idk", "en", product=FZ)
    assert r.hits and r.hits[0].text


def test_exclude_skips_given_entries():
    first, _ = ids("powerlifting")
    again, _ = ids("powerlifting", exclude=set(first[:1]))
    assert first[0] not in again


def test_claude_pro_never_yields_synthetic_entries():
    kb = load_kb(CP)
    for q in ("can it write code", "coding help", "image generation", "how much", "powerlifting", "squat rack", "random words here"):
        r = retrieve(CP, q, product=CP, k=5)
        assert r.hits
        assert all(h.entry.provenance == "public" for h in r.hits), q
    assert all(e.provenance == "public" for e in kb.usable_entries())
    assert ids("can it help me debug my code", CP)[0][0] == "cp-coding"
    assert "$8 per month" in retrieve(CP, "what is the price", "en", product=CP).hits[0].text


def test_deterministic_across_processes():
    code = ("import json,sys;sys.path.insert(0,'.');from acie.knowledge import retrieve;"
            "P=json.load(open('data/voice/products.json'))['fitzone-gym'];"
            "qs=['powerlifting ke liye kya hai','bench press','competition prep','kitne ka hai','पावरलिफ्टिंग के लिए क्या है','zzz'];"
            "print(json.dumps([[(h.id,round(h.score,6)) for h in retrieve(P,q,product=P,k=5).hits] for q in qs]))")
    outs = {subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env={"PYTHONHASHSEED": str(s), "PATH": "/usr/bin"}).stdout
            for s in (1, 2, 3)}
    assert len(outs) == 1 and next(iter(outs)).strip()
