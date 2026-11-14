"""Robustness of the rule layers on raw ASR text in English, Hindi (Devanagari) and Hinglish."""

import pytest

from acie.decision.domains.bank_demo import BankDemoDomain
from acie.decision.engine import DecisionEngine
from acie.personalization.policy import detect_explicit_requests
from acie.personalization.profile_store import MemoryProfileStore
from acie.pipeline import ACIEPipeline
from acie.schemas import SessionState, TurnInput, TurnTiming, Utterance
from acie.state.estimator import HumanStateEstimator, RuleScorer
from acie.text import _HI_1_99, detect_language, find_amounts, normalize, parse_amount


def T(text, sid="s1", cid="c1", **timing):
    return TurnInput(session_id=sid, customer_id=cid, text=text, timing=TurnTiming(**timing))


def S(**dialog):
    return SessionState(session_id="s", customer_id="c", dialog=dict(dialog))


# ------------------------------------------------------------------ normalization

def test_normalize_folds_devanagari_variants():
    assert normalize("हज़ार") == normalize("हजार")
    assert normalize("हाँ") == normalize("हां")
    assert normalize("५०,०००") == "50,000"
    assert normalize("Don't") == "dont"


# ------------------------------------------------------------------ amounts

@pytest.mark.parametrize("text,amount", [
    # English
    ("fifty thousand", 50_000),
    ("i want to send twenty five thousand rupees", 25_000),
    ("twenty-five thousand", 25_000),
    ("one lakh and fifty thousand", 150_000),
    ("five hundred twenty five", 525),
    ("twenty five thousand five hundred and fifty", 25_550),
    ("a thousand rupees", 1_000),
    ("send 25k", 25_000),
    ("fifty k", 50_000),
    ("rs. 500", 500),
    ("₹50,000 bhejo", 50_000),
    ("two point five lakh", 250_000),
    # Hinglish
    ("pachas hazaar bhej do", 50_000),
    ("paanch sau", 500),
    ("dedh lakh", 150_000),
    ("dhai lakh", 250_000),
    ("sava lakh", 125_000),
    ("sava do lakh", 225_000),
    ("saadhe teen hazaar", 3_500),
    ("paune do lakh", 175_000),
    ("ek lakh pachas hazaar", 150_000),
    ("do hazaar paanch sau", 2_500),
    ("das hazaar transfar karna hai", 10_000),
    ("chhappan hazaar", 56_000),
    ("pachchis hajar rupaye", 25_000),
    ("50 hazaar", 50_000),
    # Devanagari
    ("पचास हज़ार", 50_000),
    ("पच्चीस हजार रुपये", 25_000),
    ("डेढ़ लाख", 150_000),
    ("ढाई लाख", 250_000),
    ("एक लाख", 100_000),
    ("साढ़े तीन हजार", 3_500),
    ("अट्ठावन हजार", 58_000),
    ("इक्कीस हज़ार भेजने हैं", 21_000),
    ("पैंतीस सौ", 3_500),
    ("निन्यानबे हजार", 99_000),
    ("५०,०००", 50_000),
    ("50 हजार", 50_000),
    ("मुझे दस हजार ट्रांसफर करने हैं", 10_000),
    ("दो लाख", 200_000),
])
def test_spoken_amounts(text, amount):
    assert parse_amount(text) == amount


def test_full_hindi_1_99_table():
    for n, forms in _HI_1_99.items():
        assert parse_amount(f"{forms[0]} हजार") == n * 1000, forms[0]
        romans = [f for f in forms if f.isascii()]
        assert parse_amount(f"{romans[0]} hazaar") == n * 1000, romans[0]


@pytest.mark.parametrize("text", [
    "mera otp 482910 hai",
    "otp is 4 5 3 2",
    "last four digits 4532",
    "card ke last 4 digit 4532",
    "pin 1234",
    "my phone number is 9876543210",
    "2 baar try kiya",
    "5 minute mein call karo",
    "15 august ko",
    "date 12 ko",
    "transfer kar do",  # "do" = give, not 2
    "please do so",
    "ek baat batao",
    "no number here",
    "पिन 1234",
])
def test_non_amounts_are_ignored(text):
    assert parse_amount(text) is None


def test_amount_after_identifier_still_found():
    assert parse_amount("account number 1234 mein 5000 bhejo") == 5_000
    assert parse_amount("15 august ko 5000 bhejna") == 5_000
    assert parse_amount("pachas hazaar do") == 50_000  # trailing "do" = give
    assert find_amounts("pachas hazaar nahi bees hazaar") == [50_000, 20_000]


# ------------------------------------------------------------------ language

@pytest.mark.parametrize("text,lang", [
    ("मेरा बैलेंस कितना है", "hi"),
    ("मुझे पचास हज़ार ट्रांसफर करने हैं", "hi"),
    ("मेरा balance check करो", "hi"),
    ("mujhe balance check karna hai", "hinglish"),
    ("balance batao", "hinglish"),
    ("aur limit kitni hai bhai", "hinglish"),
    ("can you tell me mera balance kitna hai", "hinglish"),
    ("check my balance yaar", "en"),  # one Hindi word in an English sentence
    ("i want to transfer money jaldi", "en"),
    ("the main problem is my card", "en"),  # "main" is not Hindi here
    ("व्हाट इज माय बैलेंस", "en"),  # English speech through Hindi ASR
    ("send fifty thousand to my brother", "en"),
])
def test_language_detection_asr(text, lang):
    assert detect_language(text) == lang


def test_loanword_only_devanagari_is_neutral():
    assert detect_language("ओके थैंक यू", neutral="en") == "en"
    assert detect_language("ओके थैंक यू") == "hi"  # default: Hindi ASR channel
    assert detect_language("ok", neutral="hi") == "hi"
    assert detect_language("मेरा बैलेंस बताओ", neutral="en") == "hi"


# ------------------------------------------------------------------ intents

CLF = BankDemoDomain().classifier
PENDING = {"pending": "transfer"}


@pytest.mark.parametrize("text", ["हाँ जी", "हां", "जी हां", "ठीक है", "कर दो", "हाँ भेज दो", "ok", "yeah",
                                  "yep", "sure", "go ahead", "haan ji bilkul", "theek hai kar do",
                                  "koi baat nahi bhej do", "no problem go ahead", "nahi theek hai bhej do",
                                  "ओके", "why not"])
def test_affirm(text):
    assert CLF.classify(text, S(**PENDING)) == "affirm"


@pytest.mark.parametrize("text", ["नहीं", "मत करो", "रहने दो", "nope", "don't", "cancel it", "no not interested",
                                  "ok mat bhejo", "nahi nahi", "theek nahi hai", "abhi nahi", "no thanks"])
def test_deny(text):
    assert CLF.classify(text, S(**PENDING)) == "deny"


def test_affirm_deny_gated_and_filler_haan():
    assert CLF.classify("haan", S()) != "affirm"
    assert CLF.classify("haan mera balance kitna hai", S(**PENDING)) == "check_balance"
    assert CLF.classify("koi baat nahi", S(**PENDING)) != "deny"


@pytest.mark.parametrize("text", ["customer care se baat karni hai", "कस्टमर केयर से बात करवा दो",
                                  "किसी आदमी से बात करनी है", "i want a representative",
                                  "manager se baat karao", "card block karna hai", "mera debit card kho gaya",
                                  "कार्ड ब्लॉक करना है", "atm card ka pin bhool gaya"])
def test_human_agent(text):
    assert CLF.classify(text, S()) == "human_agent"


@pytest.mark.parametrize("text", ["मतलब?", "matlab", "umm matlab", "समझ नहीं आया", "क्या बोला", "फिर से बोलिए",
                                  "sorry what", "come again", "which one", "sorry?", "nahi samjha",
                                  "samajh nahi aaya", "कौन सा"])
def test_clarify(text):
    assert CLF.classify(text, S(**PENDING)) == "clarify"


@pytest.mark.parametrize("text,intent", [
    ("matlab mujhe paise bhejne hai", "transfer"),  # filler "matlab" is not clarify
    ("transfer limit kitni hai", "transfer_limit"),
    ("transfar limit", "transfer_limit"),
    ("कितना भेज सकता हूं", "transfer_limit"),
    ("मेरी लिमिट कितनी है", "transfer_limit"),
    ("mujhe paanch hazaar transfar karna hai", "transfer"),
    ("ट्रांसफर करना है", "transfer"),
    ("जल्दी से पैसे भेजो", "transfer"),
    ("मेरा बैलेंस कितना है", "check_balance"),
    ("account mein kitna paisa hai", "check_balance"),
    ("credit card ke baare mein batao", "card_offer"),
    ("क्रेडिट कार्ड चाहिए", "card_offer"),
    ("नमस्ते जी", "greet"),
    ("हेलो", "greet"),
    ("थैंक यू", "goodbye"),
    ("bas itna hi shukriya", "goodbye"),
])
def test_open_intents(text, intent):
    assert CLF.classify(text, S()) == intent


# ------------------------------------------------------------------ human state

def score(text, history=None, **timing):
    sess = SessionState(session_id="s", customer_id="c",
                        history=[Utterance(role="customer", text=h) for h in (history or [])])
    s, _ = RuleScorer().score(T(text, **timing), sess)
    return s


@pytest.mark.parametrize("text", ["i already told you", "how many times do i have to say this",
                                  "kya bakwas hai", "मैंने पहले ही बोला था", "कितनी बार बोलूं", "हद है",
                                  "pathetic", "this is ridiculous", "bekaar service hai", "time waste ho raha hai"])
def test_frustration_fires(text):
    assert score(text)["frustration"] >= 0.8


@pytest.mark.parametrize("text", ["how many times can i transfer in a day", "kitni baar transfer kar sakte hain",
                                  "thanks yaar", "worst case kitna lagega", "haan yaar theek hai",
                                  "no worries"])
def test_frustration_neutral_uses(text):
    assert score(text)["frustration"] < 0.5


def test_frustration_graded():
    assert 0.3 <= score("arre yaar")["frustration"] < 0.8  # mild: needs support to flip the policy
    assert 0.3 <= score("ये क्या है")["frustration"] < 0.8


@pytest.mark.parametrize("text", ["मतलब?", "कौन सा", "समझ नहीं आया", "matlab", "which limit", "huh", "sorry?",
                                  "kya matlab", "i'm confused"])
def test_confusion_fires(text):
    assert score(text)["confusion"] >= 0.5


@pytest.mark.parametrize("text", ["matlab mujhe paise bhejne hain", "समझ गया", "ok theek hai", "phir se transfer karo"])
def test_confusion_neutral_uses(text):
    assert score(text)["confusion"] < 0.5


@pytest.mark.parametrize("text", ["जल्दी", "अभी के अभी", "urgent hai", "emergency hai hospital mein paise chahiye",
                                  "right now please", "jaldi karo", "तुरंत भेजो"])
def test_urgency_fires(text):
    assert score(text)["urgency"] >= 0.7


@pytest.mark.parametrize("text", ["no rush", "koi jaldi nahi", "जल्दी नहीं है", "take your time", "आराम से बताइए",
                                  "abhi nahi baad mein", "not urgent"])
def test_urgency_negations(text):
    assert score(text)["urgency"] == 0.0


def test_abhi_is_weak_urgency():
    assert score("abhi mera balance kitna hai")["urgency"] < 0.5


def test_engagement_minimal_devanagari():
    assert score("हम्म")["engagement"] < 0.5
    assert score("हां जी")["engagement"] < 0.5
    assert score("mera balance kitna hai aur limit")["engagement"] > 0.5


def test_smoothing_on_hindi_frustration():
    est = HumanStateEstimator()
    sess = SessionState(session_id="s", customer_id="c")
    s = est.estimate(T("कितनी बार बोलूं, हद है"), sess)
    assert s.frustration >= 0.5


# ------------------------------------------------------------------ explicit preferences

@pytest.mark.parametrize("text,expected", [
    ("हिंदी में बात कीजिए", {"language": "hi"}),
    ("हिंदी में बात करो", {"language": "hi"}),
    ("can you speak hindi", {"language": "hi"}),
    ("अंग्रेज़ी में बोलिए", {"language": "en"}),
    ("इंग्लिश में बात करो", {"language": "en"}),
    ("english please", {"language": "en"}),
    ("can you speak in english", {"language": "en"}),
    ("mujhe hindi nahi aati english mein bolo", {"language": "en"}),
    ("english nahi aati", {"language": "hi"}),
    ("hindi mein nahi english mein", {"language": "en"}),
    ("आसान भाषा में समझाइए", {"complexity": "low"}),
    ("थोड़ा धीरे बोलिए", {"complexity": "low"}),
    ("short mein batao", {"verbosity": "low"}),
    ("जल्दी जल्दी बताओ", {"verbosity": "low"}),
    ("in detail explain karo", {"verbosity": "high"}),
    ("विस्तार से बताइए", {"verbosity": "high"}),
])
def test_explicit_requests(text, expected):
    assert detect_explicit_requests(text) == expected


@pytest.mark.parametrize("text", ["mera balance kitna hai", "simple interest kitna hai", "the transfer was brief",
                                  "hindi", "जल्दी बताओ"])
def test_explicit_requests_negatives(text):
    assert detect_explicit_requests(text) == {}


# ------------------------------------------------------------------ end to end

def make_pipe():
    return ACIEPipeline(DecisionEngine(BankDemoDomain()), store=MemoryProfileStore())


async def test_hindi_over_limit_transfer_then_accept():
    pipe = make_pipe()
    r = await pipe.run_turn(T("मुझे पचास हज़ार ट्रांसफर करने हैं"))
    assert r.plan.intent == "transfer_over_limit" and r.expression.language == "hi"
    r = await pipe.run_turn(T("हाँ जी भेज दो"))
    assert r.plan.intent == "transfer_done" and "₹25,000" in r.text


async def test_hinglish_ask_amount_spoken():
    pipe = make_pipe()
    r = await pipe.run_turn(T("paise transfar karna hai"))
    assert r.plan.intent == "transfer_ask_amount"
    r = await pipe.run_turn(T("das hazaar"))
    assert r.plan.intent == "transfer_confirm" and "₹10,000" in r.text


async def test_otp_not_taken_as_amount_in_slot():
    pipe = make_pipe()
    await pipe.run_turn(T("send money"))
    r = await pipe.run_turn(T("otp 482910"))
    assert r.plan.intent != "transfer_confirm"


async def test_card_block_routes_to_human_not_offer():
    r = await make_pipe().run_turn(T("mera card block karna hai"))
    assert r.plan.intent == "human_agent"
