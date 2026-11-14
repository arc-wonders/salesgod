"""Claims the agent must never make up (persona test 14 Nov): weather it did not look up, health or safety promises,
"cancel any time" for a product whose facts do not say so, and the agent speaking about itself with the wrong gender.

Rule packs only judge LLM-written sentences (operator templates are trusted). Pure regex, deterministic.
"""

from __future__ import annotations

import re

from ..schemas import ExpressionParams, TurnPlan

_I = re.IGNORECASE
_DEV = r"[ऀ-ॿ]"

_WEATHER = re.compile(r"(मौसम|mausam|weather).{0,40}(साफ़|साफ|सुहाना|अच्छा|धूप|बारिश|गर्मी|ठंड|sunny|clear|pleasant|rain|hot|cold|nice)|"
                      r"(साफ़|साफ|सुहाना|धूप|बारिश).{0,20}(मौसम)|(it'?s|it is|today'?s?) (sunny|raining|clear|hot|cold|pleasant)|"
                      r"\b\d+ ?(degrees|°|डिग्री)", _I)
_HEALTH = re.compile(r"(घुटन|knee|joint|जोड़|दर्द|\bpain\b|\bbp\b|blood pressure|sugar|diabet|heart|दिल|back|कमर|बुजुर्ग|बुज़ुर्ग|बूढ़|senior|elderly|age).{0,70}"
                     r"(सुरक्षित|safe|safely|मददगार|helpful|suitable|उपयुक्त|मदद मिलेगी|help you|relief|आराम|cure|ठीक हो|हिसाब से|good for|perfect)|"
                     r"(सुरक्षित|safe|safely|मददगार|helpful|suitable|उपयुक्त|relief|आराम|good for).{0,40}"
                     r"(घुटन|knee|joint|दर्द|pain|बुजुर्ग|बुज़ुर्ग|बूढ़|senior|elderly)", _I)
_CANCEL = re.compile(r"cancel (any ?time|whenever|kabhi bhi)|(कभी भी|kabhi bhi|any ?time) (cancel|कैंसल)|कैंसल कर सकते|cancel kar sakte|cancel कर सकते|"
                     r"cancel (कभी भी|any)", _I)
_LIMITED = re.compile(r"limited[- ](time|period|offer)|सीमित (समय|अवधि)|कुछ ही दिन|only for a few days|ऑफर (खत्म|ख़त्म) हो", _I)


_MEMORY = re.compile(r"(we|हमारे पास|hamare paas|humare paas).{0,40}(your (details|record|information|data)|आपकी (डिटेल|जानकारी|details)|aapki (details|jaankari))|"
                     r"(in|हमारे) (our|system|सिस्टम)|(previous|earlier|पिछली|pehle ki|last) (inquiry|enquiry|call|query|बातचीत|पूछताछ)|"
                     r"(you|आपने) (called|inquired|enquired|asked|filled|visited).{0,20}(before|earlier|last time|पहले|pichli)|"
                     r"your (previous|earlier|last) (visit|inquiry|enquiry|call|booking|membership)|आपकी (पिछली|पहले की) (विज़िट|पूछताछ|कॉल|बुकिंग)|hamare system mein", _I)


class ClaimsPack:
    name = "claims"

    def check(self, sentence: str, plan: TurnPlan, expr: ExpressionParams) -> str | None:
        ref = (plan.reference or "")
        if _WEATHER.search(sentence) and "weather now" not in ref.lower():
            return "invented weather (nothing was looked up: say you can't check it right now)"
        if _HEALTH.search(sentence):
            return "health or safety claim (never promise what is safe or helpful for a health condition)"
        if _CANCEL.search(sentence) and not re.search(r"cancel", plan.context or "", _I) and not any(
                re.search(r"cancel", f.value or "", _I) for f in plan.facts.values()):
            return "cancellation claim that is not in the product facts"
        if _MEMORY.search(sentence):
            return "invented memory of the customer (no CRM, previous inquiry or saved details exist; never claim them)"
        if _LIMITED.search(sentence):
            return "limited-time claim without an end date in the facts"
        return None


# first-person feminine forms: Devanagari and Roman Hindi (the agent Arjun is male)
_FEM = re.compile(r"%s+ती\s+(हूँ|हूं|हु)|(रही|गयी|गई)\s+(हूँ|हूं)|%s+(ूँगी|ूंगी|ऊँगी|ऊंगी)(?!%s)|माफ़ी चाहती|"
                  r"\b\w+ti (hoon|hun|hu)\b|\b(rahi|gayi|gai) (hoon|hun)\b|\b(karungi|bataungi|dungi|bhejungi|lungi|chahungi|kahungi|samjhaungi|karoongi|bataoongi)\b" % (_DEV, _DEV, _DEV), _I)
_MASC = re.compile(r"%s+ता\s+(हूँ|हूं|हु)|(रहा|गया)\s+(हूँ|हूं)|\b\w+ta (hoon|hun|hu)\b|\b(raha|gaya) (hoon|hun)\b|\b(karunga|bataunga|dunga|bhejunga|lunga)\b" % _DEV, _I)


class AgentGenderPack:
    """The agent speaks of itself in the gender the campaign gives it ("माफ़ी चाहता हूँ", never "चाहती")."""

    def __init__(self, gender: str = "male"):
        self.gender = gender
        self.name = f"agent_{gender}"

    def check(self, sentence: str, plan: TurnPlan, expr: ExpressionParams) -> str | None:
        rx = _FEM if self.gender == "male" else _MASC
        m = rx.search(sentence)
        return f"wrong gender for the agent ({self.gender}): {m.group(0)!r}" if m else None
