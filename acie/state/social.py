"""Social / human-state signals: HOW the caller is talking to us, beyond the 14 affect dims.

  sarcasm     0..1  says the opposite of what they mean ("oh wow, another sales call, my lucky day")
  humour      0..1  playful, joking, teasing ("robot ho kya bhai, chai piyoge?")
  hostility   0..1  rude / insulting / abusive at the agent ("bakwas mat karo", "stupid bot"); distinct from frustration
                    (frustration = annoyed at the conversation going wrong; hostility = aggression aimed at us)
  interest    0..1  real curiosity about the product (specific questions, "tell me more", "how do I join")
  impatience  0..1  wants it short NOW ("30 seconds", "seedha batao", "get to the point")
  scepticism  0..1  doubts it is genuine ("scam?", "hidden charges?", "number kahan se mila")
  rapport     0..1  slow running warmth of the call (rises with humour/interest/thanks, falls with hostility/sarcasm)

Two readers, fused per turn: a lexicon here (always on, works when the NLU is skipped or times out) and the NLU's own
reading of the same dims (acie/understanding.py, ACIE_NLU_SOCIAL). Smoothed like the HumanState (rise fast, decay slowly)
and stored as JSON in session.dialog["_social"]; acie/psyche/style.py and the domain read it with `load(dialog)`.
Pure functions, no I/O. ACIE_SOCIAL=0 turns the lexicon + storage off (load() then returns the neutral reading).
"""

from __future__ import annotations

import json
import os
import re

KEY = "_social"
NLU_KEY = "_nlu_social"
DIMS = ("sarcasm", "humour", "hostility", "interest", "impatience", "scepticism", "rapport")
NLU_DIMS = DIMS[:6]
NEUTRAL = {"sarcasm": 0.0, "humour": 0.0, "hostility": 0.0, "interest": 0.3, "impatience": 0.0, "scepticism": 0.0, "rapport": 0.5}
_I = re.I


def enabled() -> bool:
    return os.getenv("ACIE_SOCIAL", "1") != "0"


# exaggerated praise / mock gratitude: sarcastic only next to a complaint target (a sales call, an AI, "another", "again") or
# a flat follow-up ("...", "right.")
_SARC_PRAISE = re.compile(r"\b(oh (wow|great|joy|lovely|fantastic|perfect|wonderful|brilliant|goody)|wow,? (just )?(great|amazing|fantastic)|"
                          r"my lucky day|lucky me|just what i (needed|wanted)|just what my \w+ (was|is) missing|exactly what i needed|"
                          r"how (original|exciting|thrilling|wonderful|revolutionary)|revolutionary|groundbreaking|what a surprise|"
                          r"because i have (so much|all the) (free )?time|sure,? because|yeah,? right|as if|thanks for nothing|great dodge|"
                          r"nice try|oh sure|how generous|generous\.|wow,? generous|so generous|big deal|whoop.?de.?doo|badi meherbani|bahut meherbani|"
                          r"of course you (do|are|can)|totally|obviously|can.?t wait|i.?m (so|super) (excited|thrilled))\b|"
                          r"kya baat hai|wah(h)? (wah|ji wah)|waah ji waah|bada aaya|badi aayi|haan haan,? (bilkul|zaroor)|bahut (badhiya|accha),? (ab|phir)|"
                          r"वाह (वाह|जी वाह)|क्या बात है|बड़ा आया|बड़ी आई|हाँ हाँ,? (बिल्कुल|ज़रूर)", _I)
_SARC_TARGET = re.compile(r"another|again|sales|call|ai\b|robot|bot|pitch|offer|marketing|spam|telecaller|phir se|fir se|ek aur|फिर से|"
                          r"एक और|कॉल|\.\.\.|…", _I)
_SARC_STRONG = re.compile(r"my lucky day|lucky me|just what (i|my \w+) (needed|wanted|was missing)|thanks for nothing|great dodge|"
                          r"because i have (so much|all the)|yeah,? right|how original|revolutionary|bada aaya|badi aayi|बड़ा आया", _I)

_HUMOUR = re.compile(r"\b(haha+|hehe+|hihi|lol|lmao|rofl|just kidding|kidding|jk|joke|mazaak|mazak|mazaq|majak|masti|"
                     r"chai pi(yoge|yega|oge|lo)|robot (ho|hai) kya|aalsi|aalas|lazy (guy|me|bum)|baap re|arre (yaar|bhai|boss)|"
                     r"pakka\?|sach me|bindaas|mast hai|ha ha)\b|😂|🤣|😄|😜|मज़ाक|मजाक|हाहा|हा हा|हीही|आलसी|बाप रे", _I)

_HOSTILE = re.compile(r"\b(stupid|idiot|dumb|moron|shut up|get lost|go to hell|bloody|nonsense|rubbish|useless|pathetic|damn you|f+u?c?k|"
                      r"bakwas|bakwaas|bakvas|bewakoof|bevakoof|ullu|gadha|pagal|chup|chup kar|dimaag mat kha\w*|dimag mat kha\w*|"
                      r"dimaag kharab|dimag kharab|faltu|ghatiya|tameez|badtameez|kamine|saala|sala\b|chal hat|nikal|bhaag|"
                      r"stupid bot|dumb bot|bot jaisa|time pass mat|bhejafry|bheja fry)\b|बकवास|बेवकूफ़|बेवकूफ|उल्लू|गधा|पागल|चुप|"
                      r"दिमाग (मत खा|ख़राब|खराब)|फ़ालतू|फालतू|घटिया|बदतमीज़|बदतमीज|तमीज़|निकल|भाग यहाँ", _I)

_INTEREST = re.compile(r"\b(tell me more|sounds (good|interesting|great)|interesting|how do i (join|sign|buy|start|pay)|how does it work|"
                       r"what (exactly )?(is|does it) include|what.?s included|what do i get|can i (try|use|get|join|share)|is there (a )?trial|"
                       r"when (does|can) (it|i) start|which (plan|branch|days?)|timings?|i.?m interested|i want (it|to join|to try)|"
                       r"aur batao|aur bataiye|batao (aur|zara)|kya kya milega|kya milta hai|kaise (join|le|milega|kare|karu|karein)|"
                       r"achha (hai|laga)|interested hu|interest hai|chahiye mujhe|le sakta|le sakti|try kar sak|kab se)\b|"
                       r"और बताइए|और बताओ|क्या क्या मिलेगा|क्या मिलता है|कैसे (जॉइन|लें|मिलेगा|करें)|अच्छा लगा|लेना है|ट्राई कर सक", _I)

_IMPATIENT = re.compile(r"\b(quick(ly)?|be quick|make it (quick|fast|short)|hurry|in short|get to the point|to the point|cut to|straight(away)?|"
                        r"\d+ seconds?|one minute|no time|busy|i.?m in a meeting|gotta run|jaldi|jaldi bolo|seedha (batao|bolo|point)|"
                        r"point (par|pe) aao|short mein|time nahi|time nahin|samay nahi|bas itna|fatafat|ek line mein)\b|"
                        r"जल्दी|सीधा (बताओ|बोलो)|पॉइंट पर|टाइम नहीं|समय नहीं|फटाफट", _I)

_SCEPTIC = re.compile(r"\b(scam|fraud|fake|froud|spam|cheat|genuine|legit|real company|guarantee|hidden (charge|fee|cost)s?|catch|"
                      r"too good to be true|why so cheap|kaise (maan|bharosa)|bharosa|vishwas|sach me|sach mein|pakka\b|dhokha|"
                      r"otp|card details?|number kahan se|where did you get my number)\b|फ्रॉड|फ़्रॉड|धोखा|भरोसा|विश्वास|नकली|"
                      r"हिडन|कैसे मान|नंबर कहाँ से|नंबर कहां से|ओटीपी", _I)
_THANKS = re.compile(r"\b(thanks?|thank you|shukriya|dhanyavaad|nice|great|good point|fair)\b|धन्यवाद|शुक्रिया|बढ़िया", _I)


def rule_read(text: str) -> dict[str, float]:
    """The lexicon's reading of ONE turn (0 when silent; interest defaults to its neutral)."""
    t = (text or "").strip()
    out = {d: 0.0 for d in NLU_DIMS}
    if not t:
        return out
    if _SARC_STRONG.search(t):
        out["sarcasm"] = 0.85
    elif _SARC_PRAISE.search(t) and _SARC_TARGET.search(t):
        out["sarcasm"] = 0.7
    elif _SARC_PRAISE.search(t):
        out["sarcasm"] = 0.4
    if _HUMOUR.search(t):
        out["humour"] = 0.7 if not out["sarcasm"] else 0.3
    n_host = len(_HOSTILE.findall(t))
    if n_host:
        out["hostility"] = min(1.0, 0.65 + 0.15 * (n_host - 1))
    if _INTEREST.search(t):
        out["interest"] = 0.75
    elif "?" in t and not out["hostility"] and not out["sarcasm"]:
        out["interest"] = 0.5
    if _IMPATIENT.search(t):
        out["impatience"] = 0.7
    if _SCEPTIC.search(t):
        out["scepticism"] = 0.7
    return out


def load(dialog: dict | None) -> dict[str, float]:
    if not dialog or not enabled():
        return dict(NEUTRAL)
    try:
        v = json.loads(dialog.get(KEY) or "{}")
    except (ValueError, TypeError):
        v = {}
    return {d: float(v.get(d, NEUTRAL[d])) for d in DIMS}


def update(dialog: dict, text: str, nlu: dict | None = None, alpha_up: float = 0.75, alpha_down: float = 0.35) -> dict[str, float]:
    """Fuse the lexicon and the NLU for this turn (the stronger cue wins, the NLU can also lower a lexicon guess a bit), smooth,
    update rapport, store. Returns the new reading (also kept as dialog[KEY])."""
    if not enabled():
        return dict(NEUTRAL)
    prev = load(dialog)
    raw = rule_read(text)
    nlu = {k: float(v) for k, v in (nlu or {}).items() if k in NLU_DIMS and v is not None}
    now: dict[str, float] = {}
    for d in NLU_DIMS:
        r, m = raw[d], nlu.get(d)
        if m is None:
            x = r
        elif r >= 0.6:  # the lexicon heard an explicit cue: keep it, nudged by the model
            x = 0.7 * r + 0.3 * max(r, m) if m >= 0.3 else 0.6 * r + 0.4 * m
        else:
            x = max(r, m)
        now[d] = round(min(1.0, max(0.0, x)), 3)
    out: dict[str, float] = {}
    for d in NLU_DIMS:
        p, r = prev[d], now[d]
        a = alpha_up if r > p else alpha_down
        b = NEUTRAL[d]
        out[d] = round(b + a * (r - b) + (1 - a) * (p - b), 3)
    # rapport: slow, from the turn's tone (humour, interest, thanks up; hostility, sarcasm down)
    up = 0.35 * now["humour"] + 0.25 * max(0.0, now["interest"] - 0.3) + (0.1 if _THANKS.search(text or "") and not now["sarcasm"] else 0.0)
    down = 0.45 * now["hostility"] + 0.3 * now["sarcasm"]
    out["rapport"] = round(min(1.0, max(0.0, prev["rapport"] + 0.5 * (up - down))), 3)
    out["turn_sarcasm"], out["turn_hostility"], out["turn_humour"] = now["sarcasm"], now["hostility"], now["humour"]
    try:
        old = json.loads(dialog.get(KEY) or "{}")
    except (ValueError, TypeError):
        old = {}
    for d in ("sarcasm", "hostility", "impatience"):  # the strongest this call (a sarcastic caller's later flat "no thanks" is still final)
        out[f"peak_{d}"] = round(max(float(old.get(f"peak_{d}", 0.0) or 0.0), now[d]), 3)
    dialog[KEY] = json.dumps(out, sort_keys=True)
    return out


def peak(dialog: dict | None, dim: str) -> float:
    """The strongest single-turn reading of sarcasm / hostility / impatience in this call so far."""
    if not dialog or not enabled():
        return 0.0
    try:
        return float(json.loads(dialog.get(KEY) or "{}").get(f"peak_{dim}", 0.0))
    except (ValueError, TypeError):
        return 0.0


def turn_value(dialog: dict | None, dim: str) -> float:
    """This turn's own (unsmoothed) reading for sarcasm / hostility / humour: style reacts to THIS line, not the average."""
    if not dialog or not enabled():
        return 0.0
    try:
        return float(json.loads(dialog.get(KEY) or "{}").get(f"turn_{dim}", 0.0))
    except (ValueError, TypeError):
        return 0.0
