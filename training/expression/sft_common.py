"""Shared scenario machinery for Expression-Engine (Layer 6) SFT data.

Everything here uses the real runtime code (acie.compiler.PromptCompiler,
acie.schemas, BankDemoDomain + MockBankAPI) so that training prompts are
byte-identical to what the pipeline sends to vLLM at serve time.

Only acie + stdlib are required.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from dataclasses import dataclass, field

from acie.compiler import PromptCompiler
from acie.decision.domains.bank_demo import BankDemoDomain, MockBankAPI
from acie.decision.engine import DecisionEngine
from acie.expression import render_fallback
from acie.schemas import ExpressionParams, Fact, SessionState, TurnInput, TurnPlan, Utterance
from acie.text import inr, parse_amount

LANGS = ("en", "hi", "hinglish")
LEVELS = ("low", "med", "high")
_SLOT = re.compile(r"\{(\w+)\}")

COMPILER = PromptCompiler()


# =========================================================================== expression params

_SHORT = {"l": "low", "m": "med", "h": "high"}


def parse_expr(spec: str) -> ExpressionParams:
    """Compact spec used by the seed set: "<lang> F V C W P [ack]".

    F/V/C/W/P = formality, verbosity, complexity, warmth, proactivity, each l|m|h.
    Example: "hi h l m h m ack".
    """
    parts = spec.split()
    lang, levels, rest = parts[0], parts[1:6], parts[6:]
    keys = ("formality", "verbosity", "complexity", "warmth", "proactivity")
    return ExpressionParams(language=lang, acknowledge_emotion="ack" in rest,
                            **{k: _SHORT[v] for k, v in zip(keys, levels)})


def expr_spec(e: ExpressionParams) -> str:
    s = " ".join([e.language] + [getattr(e, k)[0] for k in
                                 ("formality", "verbosity", "complexity", "warmth", "proactivity")])
    return s + (" ack" if e.acknowledge_emotion else "")


def _w(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


LANG_WEIGHTS = {"hi": 0.45, "en": 0.40, "hinglish": 0.15}


def sample_expr(rng: random.Random, lang: str | None = None, clarify: bool = False) -> tuple[ExpressionParams, dict]:
    """Sample A_t roughly the way acie.personalization.policy produces it at runtime.

    ~75%: emulate the rule policy (profile prefs mostly 'med', then state flags
    override). ~25%: broad random coverage so every combo is seen.
    Returns (params, state_flags) - flags drive the customer-utterance flavour.
    """
    lang = lang or _w(rng, LANG_WEIGHTS)
    flags = {"confused": clarify or rng.random() < 0.15, "frustrated": rng.random() < 0.2,
             "urgent": rng.random() < 0.1, "disengaged": rng.random() < 0.06}
    if rng.random() < 0.25:
        e = ExpressionParams(language=lang, **{k: rng.choice(LEVELS) for k in
                             ("formality", "verbosity", "complexity", "warmth", "proactivity")})
        e.acknowledge_emotion = flags["frustrated"] and rng.random() < 0.7
        if clarify:
            e.complexity = "low" if rng.random() < 0.8 else "med"
        return e, flags

    e = ExpressionParams(
        language=lang,
        formality=_w(rng, {"med": 0.55, "high": 0.3, "low": 0.15}),
        verbosity=_w(rng, {"med": 0.55, "low": 0.3, "high": 0.15}),
        complexity=_w(rng, {"med": 0.6, "low": 0.3, "high": 0.1}),
        warmth=_w(rng, {"med": 0.7, "high": 0.2, "low": 0.1}),
        proactivity=_w(rng, {"med": 0.7, "low": 0.2, "high": 0.1}),
    )
    if flags["confused"]:
        e.complexity = "low"
        e.verbosity = "med" if e.verbosity == "high" else e.verbosity
        e.proactivity = "low"
        e.warmth = "high" if rng.random() < 0.6 else e.warmth
    if flags["frustrated"]:
        e.warmth, e.verbosity, e.proactivity = "high", "low", "low"
        e.formality = "high" if e.formality != "low" else "med"
        e.acknowledge_emotion = rng.random() < 0.75  # policy has an ack cooldown
    if flags["urgent"]:
        e.verbosity, e.proactivity = "low", "low"
    if flags["disengaged"] and not (flags["confused"] or flags["frustrated"]):
        e.verbosity = "low"
    return e, flags


# =========================================================================== customer utterances

CUSTOMER = {
    "greet": {"en": ["Hello?", "Hi", "Hello, good morning", "Hi, is this the bank?", "Hey, hello"],
              "hi": ["नमस्ते", "हेलो, नमस्ते", "हाँ हेलो", "नमस्ते जी", "हेलो, बैंक से बात हो रही है?"],
              "hinglish": ["Hello ji", "Namaste", "Haan hello", "Hello, bank se baat ho rahi hai?"]},
    "check_balance": {"en": ["What's my balance?", "Can you tell me my account balance?", "How much money do I have?",
                             "balance please", "Check my balance"],
                      "hi": ["मेरा बैलेंस कितना है?", "बैलेंस बता दीजिए", "मेरे अकाउंट में कितने पैसे हैं?",
                             "ज़रा बैलेंस चेक कर दो", "अकाउंट बैलेंस क्या है?"],
                      "hinglish": ["Mera balance kitna hai?", "Balance bata do", "Account mein kitne paise hain?",
                                   "Zara balance check karo"]},
    "transfer_limit": {"en": ["What's my transfer limit?", "How much can I send today?", "What is the daily limit?"],
                       "hi": ["मेरी ट्रांसफर लिमिट कितनी है?", "आज कितना भेज सकता हूँ?", "डेली लिमिट क्या है?"],
                       "hinglish": ["Meri transfer limit kitni hai?", "Aaj kitna bhej sakta hoon?",
                                    "Daily limit kya hai?"]},
    "transfer_noamt": {"en": ["I want to send some money", "I need to make a transfer", "Transfer money please"],
                       "hi": ["पैसे भेजने हैं", "मुझे ट्रांसफर करना है", "किसी को पैसे भेजने हैं"],
                       "hinglish": ["Paise bhejne hain", "Mujhe transfer karna hai", "Bhai ko paise bhejne hain"]},
    "transfer_amt": {"en": ["Send {amt}", "I want to transfer {amt}", "Please transfer {amt} to my brother",
                            "transfer {amt} rupees"],
                     "hi": ["{amt} भेजने हैं", "{amt} रुपये ट्रांसफर कर दो", "मुझे {amt} ट्रांसफर करना है",
                            "भाई को {amt} भेज दीजिए"],
                     "hinglish": ["{amt} bhejne hain", "{amt} transfer kar do", "Mujhe {amt} bhejna hai"]},
    "affirm": {"en": ["Yes", "Yes, go ahead", "Okay, do it", "Sure", "Yes please"],
               "hi": ["हाँ", "हाँ, कर दीजिए", "जी हाँ", "ठीक है, कर दो", "हाँ जी"],
               "hinglish": ["Haan", "Haan kar do", "Theek hai, kar do", "Ji haan"]},
    "deny": {"en": ["No", "No, leave it", "Not interested", "No thanks", "Cancel it"],
             "hi": ["नहीं", "नहीं, रहने दो", "नहीं चाहिए", "अभी नहीं", "मत करो"],
             "hinglish": ["Nahi", "Nahi, rehne do", "Nahi chahiye", "Abhi nahi"]},
    "card_offer": {"en": ["Tell me about credit cards", "Do you have a credit card for me?", "Can I get a credit card?"],
                   "hi": ["क्रेडिट कार्ड मिल सकता है?", "मेरे लिए कोई क्रेडिट कार्ड है?", "क्रेडिट कार्ड के बारे में बताओ"],
                   "hinglish": ["Credit card mil sakta hai?", "Mere liye koi credit card hai?",
                                "Credit card ke baare mein batao"]},
    "human_agent": {"en": ["Let me talk to a real person", "I want a human agent", "Connect me to a manager"],
                    "hi": ["किसी इंसान से बात कराओ", "मुझे एजेंट से बात करनी है", "मैनेजर से बात कराइए"],
                    "hinglish": ["Kisi insaan se baat karao", "Agent se baat karni hai", "Manager se connect karo"]},
    "goodbye": {"en": ["Thanks, that's all", "Thank you, bye", "Okay thanks"],
                "hi": ["धन्यवाद", "ठीक है, शुक्रिया", "बस इतना ही, धन्यवाद"],
                "hinglish": ["Thank you, bas itna hi", "Theek hai, shukriya", "Ok thanks, bye"]},
    "unknown": {"en": ["Umm, the thing with the...", "What about the other thing?", "Hmm, wait"],
                "hi": ["वो जो पिछली बार...", "अरे वो वाला काम", "हम्म, एक मिनट"],
                "hinglish": ["Woh jo pichli baar...", "Arre woh wala kaam", "Hmm ek minute"]},
    "clarify": {"en": ["I didn't understand", "What do you mean?", "Sorry, can you repeat that?", "Huh? Say that again"],
                "hi": ["समझ नहीं आया", "क्या मतलब?", "फिर से बताइए", "मैं समझा नहीं, दोबारा बोलिए"],
                "hinglish": ["Samajh nahi aaya", "Kya matlab?", "Phir se batao", "Samjha nahi, dobara bolo"]},
}

FRUSTRATED_PREFIX = {
    "en": ["This is the third time I'm asking. ", "Why is this so hard? ", "I've been waiting forever. ",
           "Seriously, "],
    "hi": ["कितनी बार बोलना पड़ेगा, ", "क्या बकवास है, ", "आधे घंटे से लगा हूँ, ", "यार, "],
    "hinglish": ["Kitni baar bolna padega, ", "Kya bakwas hai yaar, ", "Aadhe ghante se laga hoon, ", "Yaar, "],
}

NAMES = ["Rahul", "Priya", "Amit", "Sunita", "Vikram", "Neha", "Arjun", "Kavita", "Rohit", "Pooja",
         "Suresh", "Anjali", "Manoj", "Deepa", "Imran", "Fatima", "Harpreet", "Lakshmi", "Karthik", "Meera"]


def format_amount(rng: random.Random, amount: int, lang: str) -> str:
    """How STT would plausibly transcribe an amount, guaranteed to round-trip through parse_amount."""
    opts = [str(amount), inr(amount).lstrip("₹")]
    if amount % 1000 == 0 and lang != "hi":
        opts += [f"{amount // 1000}k", f"{amount // 1000} thousand" if lang == "en" else f"{amount // 1000} hazaar"]
    for s in rng.sample(opts, len(opts)):
        if parse_amount(s) == amount:
            return s
    return str(amount)


def customer_text(rng: random.Random, key: str, lang: str, frustrated: bool = False, amt: str | None = None) -> str:
    t = rng.choice(CUSTOMER[key][lang])
    if amt is not None:
        t = t.replace("{amt}", amt)
    if frustrated:
        t = rng.choice(FRUSTRATED_PREFIX[lang]) + t[0].lower() + t[1:] if lang != "hi" else \
            rng.choice(FRUSTRATED_PREFIX[lang]) + t
    return t


# =========================================================================== bank demo scenarios

# name -> (customer_id, prior steps [(intent, utterance_key)], final intent, final utterance key)
BANK_RECIPES: dict[str, tuple[str, list[tuple[str, str]], str, str]] = {
    "greet": ("c1", [], "greet", "greet"),
    "check_balance": ("c1", [], "check_balance", "check_balance"),
    "transfer_limit": ("c1", [], "transfer_limit", "transfer_limit"),
    "transfer_ask_amount": ("c1", [], "transfer", "transfer_noamt"),
    "transfer_confirm": ("c1", [], "transfer", "transfer_amt"),
    "transfer_over_limit": ("c1", [], "transfer", "transfer_amt"),
    "transfer_insufficient": ("c2", [], "transfer", "transfer_amt"),
    "transfer_done": ("c1", [("transfer", "transfer_amt")], "affirm", "affirm"),
    "cancelled": ("c1", [("transfer", "transfer_amt")], "deny", "deny"),
    "card_offer": ("c1", [], "card_offer", "card_offer"),
    "card_ineligible": ("c2", [], "card_offer", "card_offer"),
    "card_declined": ("c1", [("card_offer", "card_offer")], "deny", "deny"),
    "card_declined_again": ("c1", [("card_offer", "card_offer"), ("deny", "deny")], "card_offer", "card_offer"),
    "card_apply_started": ("c1", [("card_offer", "card_offer")], "affirm", "affirm"),
    "human_agent": ("c1", [], "human_agent", "human_agent"),
    "goodbye": ("c1", [], "goodbye", "goodbye"),
    "unknown": ("c1", [], "unknown", "unknown"),
}
# re-explanations: "clarify:<recipe>" replays <recipe> then the customer says they didn't understand
CLARIFIABLE = ["check_balance", "transfer_limit", "transfer_over_limit", "transfer_insufficient", "transfer_done",
               "card_offer", "card_ineligible", "transfer_confirm", "card_apply_started"]

DEFAULT_ACCOUNTS = MockBankAPI().accounts  # c1: limit 40k, used 15k, bal 82.5k; c2: bal 12k, limit 25k


def default_amount(recipe: str) -> int:
    """Amount that drives the default demo accounts into the branch the recipe names."""
    return {"transfer_confirm": 10_000, "transfer_over_limit": 30_000, "transfer_insufficient": 20_000,
            "transfer_done": 10_000, "cancelled": 10_000}.get(recipe, 10_000)


def random_account(rng: random.Random, eligible: bool | None = None) -> dict:
    limit = rng.choice([25_000, 40_000, 50_000, 100_000, 200_000])
    used = rng.randrange(0, limit, 5_000)
    elig = rng.random() < 0.6 if eligible is None else eligible
    return {"name": rng.choice(NAMES), "balance": rng.randrange(2_000, 400_000, 500), "daily_limit": limit,
            "used_today": used, "card_eligible": elig,
            "card_limit": rng.choice([50_000, 75_000, 100_000, 150_000, 200_000, 300_000]) if elig else 0}


def amount_for(rng: random.Random, recipe: str, acc: dict) -> int | None:
    """Pick a transfer amount producing the recipe's branch for this account, or None if impossible."""
    bal, rem = acc["balance"], acc["daily_limit"] - acc["used_today"]
    step = rng.choice([500, 1000, 5000])
    if recipe in ("transfer_confirm", "transfer_done", "cancelled"):
        hi = min(bal, rem)
        return rng.randrange(step, hi + 1, step) if hi >= step else None
    if recipe == "transfer_over_limit":
        lo = rem + step
        return rng.randrange(lo, bal + 1, step) if bal >= lo else None
    if recipe == "transfer_insufficient":
        return bal + rng.choice([500, 1000, 2500, 5000, 10_000, 50_000])
    return None


def _fill(plan: TurnPlan, text: str) -> str:
    return _SLOT.sub(lambda m: plan.facts[m.group(1)].value if m.group(1) in plan.facts else "", text)


def agent_history_text(plan: TurnPlan, lang: str) -> str:
    """What the pipeline would have stored as the agent's previous turn (filled fallback text)."""
    return _fill(plan, render_fallback(plan, ExpressionParams(language=lang)))


@dataclass
class Scenario:
    plan: TurnPlan
    history: list[Utterance]
    tag: str  # recipe/synthetic name, "clarify:..." for re-explanations
    customer_id: str = "c1"
    meta: dict = field(default_factory=dict)


def run_bank_recipe(name: str, lang: str, final_text: str, rng: random.Random,
                    accounts: dict | None = None, prior_texts: list[str] | None = None,
                    amount: int | None = None, pre_history: list[Utterance] | None = None) -> Scenario:
    """Drive a *fresh* BankDemoDomain through a recipe and return the final TurnPlan + history.

    `name` may be "clarify:<recipe>". Transfer amounts are taken from `final_text`
    (or the first prior transfer step) exactly like at runtime.
    """
    clarify = name.startswith("clarify:")
    base = name.split(":", 1)[1] if clarify else name
    cid, prior, final_intent, final_key = BANK_RECIPES[base]
    steps = list(prior)
    if clarify:
        steps.append((final_intent, final_key))
        final_intent, final_key = "clarify", "clarify"

    api = MockBankAPI()
    if accounts:
        api.accounts.update(json.loads(json.dumps(accounts)))
    engine = DecisionEngine(BankDemoDomain(api))
    session = SessionState(session_id="sft", customer_id=cid)
    amt = amount if amount is not None else default_amount(base)

    async def go() -> TurnPlan:
        history: list[Utterance] = list(pre_history or [])
        for i, (intent, key) in enumerate(steps):
            if prior_texts and i < len(prior_texts):
                text = prior_texts[i]
            else:
                text = customer_text(rng, key, lang, amt=format_amount(rng, amt, lang))
            p = await engine.decide(intent, TurnInput(session_id="sft", customer_id=cid, text=text), session)
            history += [Utterance(role="customer", text=text), Utterance(role="agent", text=agent_history_text(p, lang))]
        session.history = history
        # history is passed to the compiler *before* the current turn, exactly like ACIEPipeline
        return await engine.decide(final_intent, TurnInput(session_id="sft", customer_id=cid, text=final_text), session)

    plan = asyncio.run(go())
    return Scenario(plan=plan, history=list(session.history), tag=name, customer_id=cid)


# =========================================================================== synthetic retail-banking plans

def _date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28)} {rng.choice(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])}"


def _days(rng: random.Random, lo: int, hi: int) -> str:
    return f"{rng.randint(lo, hi)} working days"


def _amt(rng: random.Random, lo: int, hi: int, step: int = 100) -> str:
    return inr(rng.randrange(lo, hi, step))


def _rate(rng: random.Random, lo: float, hi: float) -> str:
    return f"{rng.uniform(lo, hi):.2f}".rstrip("0").rstrip(".") + "%"


@dataclass
class Synth:
    intent: str
    objective: str
    facts: dict  # key -> (description, generator(rng) -> value)
    fallback: dict
    decision: str | None = None
    allowed_actions: list = field(default_factory=list)
    must_not_say: list = field(default_factory=list)
    ask: str | None = None
    end_call: bool = False
    customer: dict = field(default_factory=dict)  # lang -> utterances

    def build(self, rng: random.Random) -> TurnPlan:
        return TurnPlan(intent=self.intent, objective=self.objective,
                        facts={k: Fact(value=gen(rng), description=d) for k, (d, gen) in self.facts.items()},
                        decision=self.decision, allowed_actions=list(self.allowed_actions),
                        must_not_say=list(self.must_not_say), ask=self.ask, fallback=dict(self.fallback),
                        end_call=self.end_call)


SYNTH: dict[str, Synth] = {s.intent: s for s in [
    Synth("emi_due", "Tell the customer their next loan EMI amount and due date",
          {"emi_amount": ("next EMI amount", lambda r: _amt(r, 3_000, 60_000)),
           "emi_due_date": ("EMI due date", _date)},
          {"en": "Your next EMI of {emi_amount} is due on {emi_due_date}.",
           "hi": "आपकी अगली EMI {emi_amount} की है, जो {emi_due_date} को ड्यू है।",
           "hinglish": "Aapki next EMI {emi_amount} ki hai, {emi_due_date} ko due hai."},
          allowed_actions=["offer to set up auto-debit for the EMI"],
          customer={"en": ["When is my EMI due?", "What's my next EMI?", "How much is my loan EMI this month?"],
                    "hi": ["मेरी EMI कब है?", "अगली EMI कितनी है?", "इस महीने की EMI कितनी कटेगी?"],
                    "hinglish": ["Meri EMI kab hai?", "Next EMI kitni hai?", "Is mahine EMI kitni kategi?"]}),
    Synth("emi_bounce", "Explain that the last EMI bounced because the balance was low and a bounce charge was applied",
          {"emi_amount": ("EMI amount that bounced", lambda r: _amt(r, 3_000, 60_000)),
           "bounce_charge": ("bounce charge applied", lambda r: inr(r.choice([350, 500, 590, 750])))},
          {"en": "Your EMI of {emi_amount} bounced due to low balance, and a charge of {bounce_charge} was applied. Should I send a payment link?",
           "hi": "बैलेंस कम होने की वजह से आपकी {emi_amount} की EMI बाउंस हो गई, और {bounce_charge} का चार्ज लगा है। क्या मैं पेमेंट लिंक भेज दूँ?",
           "hinglish": "Balance kam hone ki wajah se aapki {emi_amount} ki EMI bounce ho gayi, aur {bounce_charge} ka charge laga hai. Payment link bhej doon?"},
          decision="bounce charge stands", allowed_actions=["offer to send a payment link"],
          must_not_say=["waiver of the charge", "blame the customer"], ask="should I send a payment link?",
          customer={"en": ["Why was I charged extra on my loan?", "My EMI failed?", "What is this bounce charge?"],
                    "hi": ["मेरे लोन पर एक्स्ट्रा चार्ज क्यों लगा?", "मेरी EMI फेल हो गई क्या?", "ये बाउंस चार्ज क्या है?"],
                    "hinglish": ["Loan pe extra charge kyun laga?", "Meri EMI fail ho gayi kya?", "Ye bounce charge kya hai?"]}),
    Synth("kyc_due", "Tell the customer their KYC update is due and that it can be done in the app or at a branch",
          {"kyc_due_date": ("last date to update KYC", _date)},
          {"en": "Your KYC update is due by {kyc_due_date}. You can do it in the app or at any branch. Should I send the link by SMS?",
           "hi": "आपका KYC अपडेट {kyc_due_date} तक करना है। आप ऐप से या किसी भी ब्रांच में कर सकते हैं। क्या मैं SMS पर लिंक भेज दूँ?",
           "hinglish": "Aapka KYC update {kyc_due_date} tak karna hai. App se ya kisi bhi branch mein ho jayega. SMS pe link bhej doon?"},
          allowed_actions=["offer to send the KYC link by SMS"], must_not_say=["threats of account closure"],
          ask="should I send the link by SMS?",
          customer={"en": ["I got a message about KYC", "Do I need to update my KYC?", "What's this KYC thing?"],
                    "hi": ["KYC का मैसेज आया है", "क्या मुझे KYC अपडेट करना है?", "ये KYC वाला क्या है?"],
                    "hinglish": ["KYC ka message aaya hai", "Kya mujhe KYC update karna hai?", "Ye KYC wala kya hai?"]}),
    Synth("kyc_received", "Confirm the KYC documents were received and say how long verification takes",
          {"verification_time": ("time needed to verify documents", lambda r: _days(r, 2, 5))},
          {"en": "We have received your KYC documents. Verification will take {verification_time}.",
           "hi": "आपके KYC डॉक्यूमेंट्स हमें मिल गए हैं। वेरिफ़िकेशन में {verification_time} लगेंगे।",
           "hinglish": "Aapke KYC documents mil gaye hain. Verification mein {verification_time} lagenge."},
          decision="documents received, verification pending", must_not_say=["verification is already complete"],
          customer={"en": ["Did you get my KYC documents?", "I uploaded my KYC, is it done?"],
                    "hi": ["मेरे KYC डॉक्यूमेंट मिल गए?", "KYC अपलोड किया था, हो गया?"],
                    "hinglish": ["Mere KYC documents mil gaye?", "KYC upload kiya tha, ho gaya?"]}),
    Synth("upi_failed", "Explain that the failed UPI payment was debited and will be refunded automatically",
          {"upi_amount": ("amount of the failed UPI payment", lambda r: _amt(r, 100, 25_000, 50)),
           "refund_time": ("time for the automatic refund", lambda r: _days(r, 2, 5))},
          {"en": "Your UPI payment of {upi_amount} failed but was debited. It will be refunded automatically within {refund_time}.",
           "hi": "आपका {upi_amount} का UPI पेमेंट फेल हुआ पर पैसे कट गए। ये {refund_time} में अपने आप वापस आ जाएँगे।",
           "hinglish": "Aapka {upi_amount} ka UPI payment fail hua par paise kat gaye. Ye {refund_time} mein automatically wapas aa jayenge."},
          decision="auto-refund in progress", allowed_actions=["offer to raise a complaint if the refund does not arrive"],
          must_not_say=["refund already done", "blame the merchant"],
          customer={"en": ["My UPI payment failed but money got deducted", "Money was cut but the payment failed",
                           "UPI failed, where is my money?"],
                    "hi": ["UPI फेल हो गया पर पैसे कट गए", "पेमेंट हुआ नहीं और पैसे कट गए", "मेरे पैसे कहाँ गए, UPI फेल हुआ था"],
                    "hinglish": ["UPI fail ho gaya par paise kat gaye", "Payment hua nahi aur paise kat gaye",
                                 "Mere paise kahan gaye, UPI fail hua tha"]}),
    Synth("upi_complaint", "Confirm a complaint was raised for the failed UPI payment and give the complaint number and timeline",
          {"complaint_id": ("complaint number", lambda r: f"CMP{r.randint(100000, 999999)}"),
           "resolution_time": ("time to resolve the complaint", lambda r: _days(r, 3, 7))},
          {"en": "I have raised a complaint. Your complaint number is {complaint_id}, and it will be resolved within {resolution_time}.",
           "hi": "मैंने कंप्लेंट दर्ज कर दी है। आपका कंप्लेंट नंबर {complaint_id} है, और {resolution_time} में इसका हल हो जाएगा।",
           "hinglish": "Maine complaint raise kar di hai. Complaint number {complaint_id} hai, aur {resolution_time} mein resolve ho jayegi."},
          decision="complaint raised",
          customer={"en": ["Still no refund, raise a complaint", "I want to file a complaint about that UPI payment"],
                    "hi": ["अभी तक पैसे नहीं आए, कंप्लेंट करो", "उस UPI पेमेंट की शिकायत दर्ज करनी है"],
                    "hinglish": ["Abhi tak refund nahi aaya, complaint karo", "Us UPI payment ki complaint karni hai"]}),
    Synth("fd_rates", "Share the FD interest rate for the tenure asked and the senior citizen rate",
          {"fd_tenure": ("FD tenure asked about", lambda r: r.choice(["1 year", "2 years", "3 years", "5 years", "444 days"])),
           "fd_rate": ("FD interest rate for that tenure", lambda r: _rate(r, 6.0, 7.6)),
           "senior_rate": ("rate for senior citizens", lambda r: _rate(r, 6.5, 8.1))},
          {"en": "For {fd_tenure}, the FD rate is {fd_rate}, and for senior citizens it is {senior_rate}.",
           "hi": "{fd_tenure} की FD पर ब्याज {fd_rate} है, और सीनियर सिटिज़न के लिए {senior_rate} है।",
           "hinglish": "{fd_tenure} ki FD pe rate {fd_rate} hai, aur senior citizens ke liye {senior_rate} hai."},
          allowed_actions=["offer to open an FD in the app"],
          must_not_say=["guaranteed returns beyond the stated rate", "advice to break existing FDs"],
          customer={"en": ["What are your FD rates?", "What interest do I get on a fixed deposit?", "FD rate for my father?"],
                    "hi": ["FD पर कितना ब्याज मिलेगा?", "FD के रेट क्या हैं?", "मेरे पापा के लिए FD का रेट?"],
                    "hinglish": ["FD pe kitna interest milega?", "FD rates kya hain?", "Papa ke liye FD ka rate?"]}),
    Synth("card_block_confirm", "Confirm which debit card to block before blocking it",
          {"card_last4": ("last four digits of the debit card", lambda r: f"{r.randint(0, 9999):04d}")},
          {"en": "I can block your debit card ending in {card_last4}. Should I block it now?",
           "hi": "मैं आपका {card_last4} पर ख़त्म होने वाला डेबिट कार्ड ब्लॉक कर सकता हूँ। क्या अभी ब्लॉक कर दूँ?",
           "hinglish": "Main aapka {card_last4} wala debit card block kar sakta hoon. Abhi block kar doon?"},
          decision="block only after the customer confirms", ask="should I block it now?",
          customer={"en": ["I lost my debit card", "Block my card please", "My wallet got stolen, my card was in it"],
                    "hi": ["मेरा डेबिट कार्ड खो गया", "कार्ड ब्लॉक कर दो", "पर्स चोरी हो गया, उसमें कार्ड था"],
                    "hinglish": ["Mera debit card kho gaya", "Card block kar do", "Wallet chori ho gaya, card usme tha"]}),
    Synth("card_blocked", "Confirm the debit card is blocked and a replacement card will be sent",
          {"card_last4": ("last four digits of the blocked card", lambda r: f"{r.randint(0, 9999):04d}"),
           "delivery_time": ("replacement card delivery time", lambda r: _days(r, 5, 10)),
           "replacement_fee": ("replacement card fee", lambda r: inr(r.choice([150, 200, 250, 300])))},
          {"en": "Your card ending in {card_last4} is blocked. A new card will reach you in {delivery_time}, with a fee of {replacement_fee}.",
           "hi": "आपका {card_last4} वाला कार्ड ब्लॉक हो गया है। नया कार्ड {delivery_time} में आ जाएगा, इसका चार्ज {replacement_fee} है।",
           "hinglish": "Aapka {card_last4} wala card block ho gaya hai. Naya card {delivery_time} mein aa jayega, charge {replacement_fee} hai."},
          decision="card blocked", must_not_say=["fee waiver"],
          customer={"en": ["Yes, block it", "Yes please, block it now"],
                    "hi": ["हाँ, ब्लॉक कर दो", "हाँ जी, अभी ब्लॉक कीजिए"],
                    "hinglish": ["Haan, block kar do", "Haan abhi block karo"]}),
    Synth("loan_offer", "Present the pre-approved personal loan offer honestly and ask if the customer wants the details",
          {"loan_amount": ("pre-approved personal loan amount", lambda r: inr(r.choice([100_000, 200_000, 300_000, 500_000, 750_000]))),
           "interest_rate": ("interest rate per year", lambda r: _rate(r, 10.5, 16.0)),
           "max_tenure": ("maximum tenure", lambda r: r.choice(["3 years", "4 years", "5 years"]))},
          {"en": "You have a pre-approved personal loan of up to {loan_amount} at {interest_rate} per year, for up to {max_tenure}. Would you like the details?",
           "hi": "आपके लिए {loan_amount} तक का पर्सनल लोन pre-approved है, {interest_rate} सालाना ब्याज पर, {max_tenure} तक के लिए। क्या आप डिटेल्स जानना चाहेंगे?",
           "hinglish": "Aapke liye {loan_amount} tak ka personal loan pre-approved hai, {interest_rate} saalana rate pe, {max_tenure} tak. Details chahiye?"},
          must_not_say=["guaranteed approval", "limited time pressure", "fees other than stated"],
          ask="would you like the details?",
          customer={"en": ["Do I have any loan offers?", "I need some money for a wedding, any loan?", "Tell me about personal loans"],
                    "hi": ["मेरे लिए कोई लोन ऑफ़र है?", "शादी के लिए पैसे चाहिए, लोन मिलेगा?", "पर्सनल लोन के बारे में बताइए"],
                    "hinglish": ["Mere liye koi loan offer hai?", "Shaadi ke liye paise chahiye, loan milega?", "Personal loan ke baare mein batao"]}),
    Synth("loan_ineligible", "Say the customer is not eligible for a personal loan right now",
          {}, {"en": "Right now you are not eligible for a personal loan. I can check again later.",
               "hi": "अभी आप पर्सनल लोन के लिए eligible नहीं हैं। मैं बाद में फिर से देख सकता हूँ।",
               "hinglish": "Abhi aap personal loan ke liye eligible nahi hain. Baad mein phir check kar sakta hoon."},
          decision="not eligible", must_not_say=["reasons from credit score", "promises of approval"],
          customer={"en": ["Can I get a personal loan?", "Any loan for me?"],
                    "hi": ["मुझे पर्सनल लोन मिल सकता है?", "मेरे लिए लोन है कोई?"],
                    "hinglish": ["Mujhe personal loan mil sakta hai?", "Mere liye loan hai koi?"]}),
    Synth("cc_bill", "Tell the customer their credit card bill amount, minimum due and due date",
          {"total_due": ("total amount due", lambda r: _amt(r, 2_000, 150_000)),
           "min_due": ("minimum amount due", lambda r: _amt(r, 200, 8_000)),
           "due_date": ("payment due date", _date)},
          {"en": "Your credit card bill is {total_due}, the minimum due is {min_due}, and the due date is {due_date}.",
           "hi": "आपका क्रेडिट कार्ड बिल {total_due} है, मिनिमम ड्यू {min_due} है, और ड्यू डेट {due_date} है।",
           "hinglish": "Aapka credit card bill {total_due} hai, minimum due {min_due}, aur due date {due_date} hai."},
          must_not_say=["advice to pay only the minimum"],
          customer={"en": ["How much is my credit card bill?", "When do I have to pay my card bill?"],
                    "hi": ["मेरा क्रेडिट कार्ड बिल कितना है?", "कार्ड का बिल कब तक भरना है?"],
                    "hinglish": ["Mera credit card bill kitna hai?", "Card ka bill kab tak bharna hai?"]}),
    Synth("fraud_check", "Say a suspicious card transaction was stopped and the card is temporarily blocked, then ask if the customer made it",
          {"txn_amount": ("amount of the suspicious transaction", lambda r: _amt(r, 1_000, 90_000)),
           "merchant": ("merchant name", lambda r: r.choice(["an online electronics store", "a jewellery shop", "an international website", "a fuel station"]))},
          {"en": "We stopped a transaction of {txn_amount} at {merchant} and temporarily blocked your card. Did you make this transaction?",
           "hi": "{merchant} पर {txn_amount} का एक ट्रांज़ैक्शन हमने रोका है और आपका कार्ड कुछ समय के लिए ब्लॉक किया है। क्या ये ट्रांज़ैक्शन आपने किया था?",
           "hinglish": "{merchant} pe {txn_amount} ka ek transaction humne roka hai aur card temporarily block kiya hai. Kya ye transaction aapne kiya tha?"},
          decision="card temporarily blocked pending confirmation", must_not_say=["accusing the customer"],
          ask="did you make this transaction?",
          customer={"en": ["Why is my card not working?", "My card got declined just now"],
                    "hi": ["मेरा कार्ड चल क्यों नहीं रहा?", "अभी मेरा कार्ड डिक्लाइन हो गया"],
                    "hinglish": ["Mera card chal kyun nahi raha?", "Abhi mera card decline ho gaya"]}),
    Synth("statement_sent", "Confirm the account statement for the requested period has been sent to the registered email",
          {"statement_period": ("statement period", lambda r: r.choice(["last 3 months", "last 6 months", "the last financial year", "last month"])),
           "email_masked": ("masked registered email", lambda r: f"{r.choice('arspmkn')}****{r.randint(10, 99)}@gmail.com")},
          {"en": "I have sent your statement for {statement_period} to {email_masked}.",
           "hi": "मैंने {statement_period} का स्टेटमेंट {email_masked} पर भेज दिया है।",
           "hinglish": "Maine {statement_period} ka statement {email_masked} pe bhej diya hai."},
          decision="statement emailed",
          customer={"en": ["I need my bank statement", "Can you email me my statement?"],
                    "hi": ["मुझे बैंक स्टेटमेंट चाहिए", "स्टेटमेंट ईमेल कर दीजिए"],
                    "hinglish": ["Mujhe bank statement chahiye", "Statement email kar do"]}),
]}


def greet_history(rng: random.Random, lang: str, name: str) -> list[Utterance]:
    """A short opening exchange to prepend, so the model sees mid-call turns too."""
    fb = {"en": f"Hello {name}, how can I help you today?", "hi": f"नमस्ते {name}, मैं आपकी क्या मदद कर सकता हूँ?",
          "hinglish": f"Namaste {name}, main aapki kya help kar sakta hoon?"}[lang]
    return [Utterance(role="customer", text=rng.choice(CUSTOMER["greet"][lang])), Utterance(role="agent", text=fb)]


def synth_scenario(name: str, lang: str, rng: random.Random, history: list[Utterance] | None = None) -> Scenario:
    return Scenario(plan=SYNTH[name].build(rng), history=history or [], tag=f"synth:{name}")


# =========================================================================== record assembly

def compile_record(sc: Scenario, expr: ExpressionParams, customer_said: str, source: str,
                   target: str | None = None, rid: str | None = None) -> dict:
    session = SessionState(session_id="sft", customer_id=sc.customer_id, history=sc.history)
    cp = COMPILER.compile(sc.plan, expr, session, customer_said)
    messages = [{"role": "system", "content": cp.system}, *cp.messages]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    if rid is None:
        h = hashlib.sha1(json.dumps(messages[:len(cp.messages) + 1], ensure_ascii=False).encode()).hexdigest()[:12]
        rid = f"{source}-{h}"
    return {"id": rid, "source": source, "messages": messages,
            "meta": {"tag": sc.tag, "intent": sc.plan.intent, "lang": expr.language, "expr_spec": expr_spec(expr),
                     "customer_text": customer_said, "plan": sc.plan.model_dump(), "expr": expr.model_dump(), **sc.meta}}


def record_plan_expr(rec: dict) -> tuple[TurnPlan, ExpressionParams]:
    return TurnPlan.model_validate(rec["meta"]["plan"]), ExpressionParams.model_validate(rec["meta"]["expr"])


def prompt_messages(rec: dict) -> list[dict]:
    """Messages without the assistant target (what the model sees at inference)."""
    msgs = rec["messages"]
    return msgs[:-1] if msgs and msgs[-1]["role"] == "assistant" else list(msgs)


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
