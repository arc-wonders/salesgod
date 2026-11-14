"""ACIE_HI_EARLY_SPEAK (default off): a Hindi/Hinglish reply may speak its first validated content sentence while the rest
is still generating. Whatever happens to the rest of the reply, what was already spoken must stay true."""
import re

from acie.decision.domains.bank_demo import BankDemoDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import TurnInput

HI_Q = "मेरी ट्रांसफर लिमिट कितनी है"
FIRST = "आपकी रोज़ की ट्रांसफर लिमिट {daily_limit} है और आज के लिए अभी {remaining_limit} बाकी है। "


class Streamer:
    """Streams fixed sentences and records, for each spoken sentence, whether generation had finished yet."""

    def __init__(self, parts):
        self.parts, self.done, self.log = parts, False, []

    async def stream(self, prompt, plan, expr):
        self.done = False
        for p in self.parts:
            yield p
        self.done = True

    def sink(self, s):
        self.log.append((s, self.done))


async def run(parts, monkeypatch, on="1", text=HI_Q):
    monkeypatch.setenv("ACIE_HI_EARLY_SPEAK", on)
    llm = Streamer(parts)
    pipe = ACIEPipeline(DecisionEngine(BankDemoDomain()), backend=llm, buffer_reply=True)
    r = await pipe.run_turn(TurnInput(session_id="e1", customer_id="c1", text=text), on_sentence=llm.sink)
    return r, llm.log


async def test_off_by_default_nothing_spoken_before_generation_ends(monkeypatch):
    r, log = await run([FIRST, "क्या आप कुछ और जानना चाहेंगे?"], monkeypatch, on="0")
    assert log and all(done for _, done in log)


async def test_on_hindi_first_content_sentence_spoken_early_and_once(monkeypatch):
    r, log = await run([FIRST, "क्या आप कुछ और जानना चाहेंगे?"], monkeypatch)
    assert log[0][1] is False and "₹40,000" in log[0][0]  # spoken while generating, slots filled
    spoken = [s for s, _ in log]
    assert sum("₹40,000" in s for s in spoken) == 1  # never twice


async def test_short_opener_is_not_spoken_early(monkeypatch):
    r, log = await run(["जी, बिल्कुल। ", FIRST, "क्या आप कुछ और जानना चाहेंगे?"], monkeypatch)
    assert not any(s.startswith("जी, बिल्कुल") and not done for s, done in log)


async def test_english_reply_is_never_spoken_early(monkeypatch):
    parts = ["Your daily transfer limit is {daily_limit} and {remaining_limit} is left for today. ", "Anything else?"]
    r, log = await run(parts, monkeypatch, text="what is my transfer limit")
    assert all(done for _, done in log)


async def test_failure_later_keeps_what_was_spoken_true(monkeypatch):
    # the second sentence writes a raw number -> validator fails -> safe template; the early sentence was validated
    r, log = await run([FIRST, "और चालीस हज़ार 40000 रुपये की सीमा है।"], monkeypatch)
    assert r.validation.used_fallback
    early = [s for s, done in log if not done]
    assert early and "₹40,000" in early[0] and not re.search(r"\b40000\b", early[0])
    assert not any(re.search(r"\b40000\b", s) for s, _ in log)  # the bad sentence is never spoken
