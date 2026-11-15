import asyncio
import json
from types import SimpleNamespace

from acie.pronounce import (GLOBAL, PronunciationStore, Resolver, merged, number_words, resolver_for, speak_rupees)
from acie.pronounce_suggest import candidates, phonetic, score


def _lex():
    lex = merged(None)
    lex["Aujla"] = {"hi": "औजला", "hinglish": "Aujlaa", "en": "Ouj-laa"}
    lex["FitZone"] = {"hi": "फिटज़ोन"}
    lex["Fan Pit"] = {"hi": "फ़ैन पिट"}
    return lex


def test_layer_order(tmp_path):
    st = PronunciationStore(tmp_path / "pronunciations.json")
    st.save({"global": {"VIP": {"hi": "G"}}, "tenants": {"t": {"VIP": {"hi": "T"}, "Zed": {"en": "zee"}}},
             "products": {"p": {"VIP": {"hi": "P"}}}, "agents": {"a": {"VIP": {"hi": "A"}}}})
    prod = SimpleNamespace(id="p", tenant="t", pronunciations={"VIP": "PP", "Karan": "करन"})
    assert merged(st)["VIP"]["hi"] == "G"
    assert merged(st, "t")["VIP"]["hi"] == "T"
    assert merged(st, "t", SimpleNamespace(id="x", tenant="t", pronunciations={"VIP": "PP"}))["VIP"]["hi"] == "PP"
    assert merged(st, "t", prod)["VIP"]["hi"] == "P"
    m = merged(st, "t", prod, "a")
    assert m["VIP"]["hi"] == "A" and m["Karan"]["hi"] == "करन" and m["Zed"]["en"] == "zee"
    assert m["VIP"]["hinglish"] == GLOBAL["VIP"]["hinglish"]  # per-language merge keeps the lower layer's other languages


def test_whole_word_and_case():
    r = Resolver(_lex())
    assert r.apply("VIPs ke liye", "hinglish") == "VIPs ke liye"  # not a whole word
    assert r.apply("ai kya hai, AI hai", "hinglish") == "ai kya hai, A.I. hai"  # ALL-CAPS keys are case-sensitive
    assert r.apply("fitzone में आइए", "hi") == "फिटज़ोन में आइए"  # mixed-case keys match any case
    assert r.apply("FitZoneX में", "hi") == "FitZoneX में"
    assert r.apply("Fan  Pit का टिकट", "hi") == "फ़ैन पिट का टिकट"  # multi-word, longest first


def test_scripts_pick_the_voice_value():
    r = Resolver(_lex())
    assert r.apply("करन Aujla के VIP टिकट", "hi") == "करन औजला के वी-आई-पी टिकट"
    assert r.apply("Karan Aujla ke VIP tickets", "hinglish") == "Karan Aujlaa ke V.I.P. tickets"
    assert r.apply("Karan Aujla is on tour", "en") == "Karan Ouj-laa is on tour"
    # a Roman line spoken by the Hindi voice inside a "hi" reply still gets Roman values
    assert r.apply("Aujla", "hi") == "Aujlaa"


def test_devanagari_keys():
    r = Resolver({"औजला": {"hi": "औज-ला"}})
    assert r.apply("करन औजला के", "hi") == "करन औज-ला के"
    assert r.apply("करनऔजला", "hi") == "करनऔजला"


def test_numbers_untouched_and_rupees_exact():
    r = Resolver(_lex())
    assert r.apply("Call 9876543210 at 5 pm, 15 days", "en") == "Call 9876543210 at 5 pm, 15 days"
    assert r.apply("₹1,499 में", "hi") == "एक हज़ार चार सौ निन्यानवे रुपये में"
    assert r.apply("sirf ₹4,999", "hinglish") == "sirf chaar hazaar nau sau ninyaanve rupaye"
    assert r.apply("only ₹1,999/month", "en") == "only 1,999 rupees a month"
    assert speak_rupees("₹12,50,000", "hi") == "बारह लाख पचास हज़ार रुपये"
    assert speak_rupees("Rs. 99.50", "hi") == "निन्यानवे रुपये पचास पैसे"
    assert speak_rupees("₹2,00,00,00,000", "hi").startswith("2,00,00,00,000")  # too big for words: digits kept
    for n in (0, 7, 19, 99, 100, 101, 999, 1000, 1001, 99999, 100000, 1234567, 10**7):
        assert number_words(n, "hi") and number_words(n, "hinglish")
    assert number_words(2800, "hi") == "दो हज़ार आठ सौ"


def test_resolver_for_never_raises(tmp_path):
    (tmp_path / "pronunciations.json").write_text("{broken")
    r = resolver_for(SimpleNamespace(id="p", tenant="t", pronunciations={"Aujla": "औजला"}), "a", tmp_path)
    assert r.apply("Aujla के VIP", "hi") == "औजला के वी-आई-पी"


def test_candidates_and_score():
    prod = {"product_name": "Karan Aujla Concert Tickets", "benefits": ["Fan Pit access", "pay by UPI"],
            "pitch_hi": "करन का शो, Fan Pit में जगह", "knowledge": "Tickets on District by Zomato."}
    words = {c["word"] for c in candidates(prod, {"Karan": {}})}
    assert {"Aujla", "Fan Pit", "Zomato", "District"} <= words and "Karan" not in words and "UPI" not in words
    assert phonetic("औजला") == phonetic("Aujala") or score("करण औजला के", "Aujla", {"hi": "औजला"}) > 0.9
    assert score("Tickets for the Quran Ojla concert", "Aujla", {"en": "Ouj-laa"}) > score("Tickets for the concert", "Aujla")


def test_call_path_applies_lexicon(monkeypatch):
    from voice.call import CallSession
    spoken = []

    class TTS:
        accepts_voice = False

        async def stream(self, text, language, **kw):
            spoken.append((text, language))
            yield b"\x00\x00"

    cs = CallSession.__new__(CallSession)
    cs.app = SimpleNamespace(tts=TTS())
    cs.call = SimpleNamespace(call_id="t")
    cs.agent = None
    cs.speaking_epoch = 1
    cs.pronounce = Resolver(_lex())

    async def play(pcm, epoch):
        return True
    cs.play_pcm = play
    cs.event = lambda *a, **k: None
    asyncio.run(cs._speak(["करन Aujla के VIP टिकट ₹1,499 में", "Karan Aujla tickets for ₹1,499"], "hi", 1, None, None))
    asyncio.run(cs._speak(["Karan Aujla tickets for ₹1,499"], "en", 1, None, None))
    assert spoken[0] == ("करन औजला के वी-आई-पी टिकट एक हज़ार चार सौ निन्यानवे रुपये में", "hi")
    assert spoken[1][0] == "Karan Aujlaa tickets for ek hazaar chaar sau ninyaanve rupaye"
    assert spoken[2] == ("Karan Ouj-laa tickets for 1,499 rupees", "en")


def test_live_seed_file_is_valid():
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "data" / "voice" / "pronunciations.json"
    if p.exists():
        d = json.loads(p.read_text())
        assert isinstance(d.get("products", {}), dict)
