"""Honest answers: what the agent says when it does not have a fact, and fresh wordings for facts it already said.

Persona test (14 Nov, 24 calls): a question that was re-asked, or that the catalog did not cover, got "that's all I
have, or shall I let you go?" / "tell me more" / a made-up value. The rule here is simple and deterministic:

  * a question ALWAYS gets an answer: the fact again in other words, or an honest "I don't have that detail, so I
    won't guess; <where to find it>";
  * listening cues and "shall I let you go?" are for non-questions only;
  * nothing is invented: a topic the product facts do not cover (refund terms, UPI, branch list, parking, AC,
    women's batch, trial pass, activation time ...) is answered with the honest line, never with a guess.

Pure functions over the Product config; three languages (en / hi / hinglish); no randomness. The variant of a line is
chosen by what the call ledger already said, so asking the same thing three times gives three different honest lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

LANGS = ("en", "hi", "hinglish")
_I = re.IGNORECASE


def _t(en: str, hi: str, hl: str) -> dict[str, str]:
    return {"en": en, "hi": hi, "hinglish": hl}


# ---------------------------------------------------------------- is it a question?
_Q = re.compile(r"\?|\b(what|how|why|when|where|which|who|is it|are you|can i|can you|do you|does it|can he|can she|will it|"
                r"kya|kaise|kab|kyun|kyon|kaun|kahan|kitna|kitne|kitni|hai kya|milega|milta|hota hai|batao|bataiye|bataye)\b|"
                r"^\W*(can|could|does|do|is|are|will|would|should)\b|क्या|कैसे|कब|क्यों|कौन|कहां|कहाँ|कितना|कितने|कितनी|"
                r"बताओ|बताइए|बताइये|बताएं|बता दो|मिलेगा|मिलता|होता है|सकता है|सकती है", _I)


def is_question(text: str) -> bool:
    """A real question or a request for information (a listening cue is the wrong reply to it)."""
    return bool(text) and bool(_Q.search(text))


# ---------------------------------------------------------------- product helpers
def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def can_cancel(p) -> bool:
    """May the agent say "you can cancel any time"? Only if the product's own facts say so."""
    flag = getattr(p, "cancel_any_time", None)
    if flag is not None:
        return bool(flag)
    texts = [getattr(p, "knowledge", "") or ""] + [f.answer for f in getattr(p, "faqs", []) if getattr(f, "answer", "")]
    return any(re.search(r"cancel", t, _I) for t in texts)


def cancel_tail(p, lang: str) -> str:
    """', and you can cancel any time' (or '' when the product does not say so)."""
    if not can_cancel(p):
        return ""
    return {"en": ", and you can cancel any time", "hi": ", और कभी भी cancel कर सकते हैं",
            "hinglish": ", aur kabhi bhi cancel kar sakte hain"}.get(lang, "")


def source(p, lang: str) -> str:
    """Where the customer finds a detail the agent does not have, as a short clause."""
    src = getattr(p, "info_source", None) or {}
    return src.get(lang) or src.get("en") or {"en": "the official website will have it", "hi": "ऑफिशियल वेबसाइट पर मिल जाएगा",
                                                  "hinglish": "official website par mil jaayega"}[lang if lang in LANGS else "en"]


def terms_text(p, lang: str) -> str:
    """The complete offer as one phrase (the same text the domain speaks): price, how long, price after."""
    if getattr(p, "terms", None) and p.terms.get(lang):
        return p.terms[lang]
    months = p.offer_validity
    return {
        "hi": f"{months.replace('for the first', 'पहले').replace('months', 'महीने')} {p.offer_price}, उसके बाद {p.original_price}",
        "hinglish": f"{months.replace('for the first', 'pehle').replace('months', 'mahine')} {p.offer_price}, uske baad {p.original_price}",
    }.get(lang, f"{p.offer_price} {months}, then {p.original_price}")


_SPLIT = re.compile(r",\s*(?:उसके बाद|फिर|then|uske baad|phir)\s+", _I)


def terms_parts(p, lang: str) -> tuple[str, str]:
    """(what you pay first, what it is after) when the phrase has the usual ', then ...' shape, else (whole, '')."""
    whole = terms_text(p, lang)
    m = _SPLIT.search(whole)
    if not m:
        return whole, ""
    return whole[:m.start()].strip(), whole[m.end():].strip()


def price_variants(p, lang: str) -> list[str]:
    """The price said again in different shapes. Each one carries the whole fact; none is a copy of the first line."""
    first, after = terms_parts(p, lang)
    if not after:
        return {"en": [f"To put it plainly: {first}."], "hi": [f"सीधे शब्दों में: {first}।"],
                "hinglish": [f"Seedhe shabdon mein: {first}."]}[lang]
    return {
        "en": [f"Sure. First {first}; after that, {after}.",
               f"Short version: you start at {first}, and later it is {after}.",
               f"In two parts: {first}; and then {after}."],
        "hi": [f"जी, {'' if first.startswith('पहले') else 'पहले '}{first}; उसके बाद {after}।",
               f"छोटे में: शुरू में {first}, और आगे चलकर {after}।",
               f"दो हिस्से हैं: {first}; और फिर {after}।"],
        "hinglish": [f"Ji, {'' if first.lower().startswith('pehle') else 'pehle '}{first}; uske baad {after}.",
                     f"Chhote mein: shuru mein {first}, aur aage chalkar {after}.",
                     f"Do hisse hain: {first}; aur phir {after}."],
    }[lang]


def pick(variants: list[str], ledger) -> str:
    """The first variant the call has not said yet; if every one was said, the last (still an answer, never silence)."""
    for v in variants:
        if ledger is None or not ledger.repeats(v):
            return v
    return variants[-1]


# ---------------------------------------------------------------- topics the facts may not cover
@dataclass(frozen=True)
class Topic:
    id: str
    rx: re.Pattern
    name: dict[str, str]
    beats_faq: bool = False                       # answered honestly even if a general FAQ keyword also matches
    applies: Callable[[object], bool] | None = None


def _monthly_priced(p) -> bool:
    return bool(re.search(r"per month|a month|/month|monthly", f"{p.offer_price} {p.original_price}", _I))


def _fixed_validity(p) -> bool:
    return bool(re.search(r"week|today|till|until|date", getattr(p, "offer_validity", ""), _I))


_PLACE = (r"sector ?\d+|सेक्टर ?\d+|gurgaon|gurugram|गुड़गांव|गुरुग्राम|गुरगांव|noida|नोएडा|delhi|दिल्ली|mumbai|मुंबई|pune|पुणे|"
          r"bangalore|bengaluru|बैंगलोर|hyderabad|चेन्नई|chennai|kolkata|jaipur|lucknow|ghaziabad|faridabad|dwarka|द्वारका")
TOPICS: list[Topic] = [
    Topic("branch", re.compile(r"(branch|ब्रांच|gym|जिम|centre|center|where|kahan|कहाँ|कहां|near|nearest|पास|address|located).{0,60}(%s)|"
                               r"(%s).{0,60}(branch|ब्रांच|gym|जिम|centre|center|where|kahan|कहाँ|कहां|near|nearest|पास|address|located|hai kya|है क्या|hai ya nahi)|"
                               r"which (branch|area|location)|branch list|how many branch|kitni branch|कितनी ब्रांच|कौन सी ब्रांच|kaun si branch" % (_PLACE, _PLACE), _I),
          _t("the branch list", "ब्रांच की लिस्ट", "branch ki list")),
    Topic("refund", re.compile(r"refund|रिफंड|रिफण्ड|money back|paise (wapas|vapas)|पैसे वापस|पैसा वापस|paisa (wapas|vapas)", _I),
          _t("the refund policy", "रिफंड पॉलिसी", "refund policy")),
    Topic("payment_method", re.compile(r"\bupi\b|यूपीआई|यू पी आई|gpay|google pay|phonepe|paytm|net ?banking|\bemi\b|debit|credit card|cash (de|se|mein)|cheque|"
                                       r"payment method|कार्ड से|card se", _I),
          _t("the payment method details", "पेमेंट के तरीकों", "payment ke tareekon")),
    Topic("activation", re.compile(r"activat|एक्टिवेट|एक्टिव (हो|होगा)|active ho|कितने दिन में (चालू|शुरू|एक्टिव)|kitne din (me|mein) (chalu|shuru|active)", _I),
          _t("the activation time", "एक्टिवेट होने के समय", "activate hone ke time")),
    Topic("monthly_plan", re.compile(r"monthly (option|plan|payment|basis)|month[- ]to[- ]month|मंथली|महीने के हिसाब|mahine ke hisab|ek mahine ka|"
                                     r"1 month plan|one month plan|month by month|3 (mahine|महीने) (ek saath|एक साथ) nahi", _I),
          _t("details on a monthly plan", "मंथली प्लान", "monthly plan"), applies=lambda p: not _monthly_priced(p)),
    Topic("womens_batch", re.compile(r"\bwomen\b|womens|ladies|female|महिला|लेडीज|महिलाओं|\bgirls\b|women.?only", _I),
          _t("details on a women-only batch", "महिलाओं के अलग बैच", "women-only batch")),
    Topic("trial_pass", re.compile(r"\btrial\b|ट्रायल|\bdemo\b|guest pass|day pass|free pass|free (visit|day|class)|एक दिन (का )?(फ्री|free)", _I),
          _t("details on a free trial pass", "फ्री ट्रायल पास", "free trial pass")),
    Topic("parking", re.compile(r"parking|पार्किंग|park (my|the)? ?(car|bike)", _I), _t("the parking details", "पार्किंग", "parking")),
    Topic("facilities", re.compile(r"\bac\b|a\.c\.?|air.?condition|एसी|steam|sauna|स्टीम|सौना|सॉना|jacuzzi|swimming|\bpool\b|पूल|\bspa\b|crèche|creche", _I),
          _t("details on that facility", "इस सुविधा", "is facility")),
    Topic("offer_end", re.compile(r"kab tak (hai|valid|chalega|chalu|milega)|कब तक (है|वैध|चलेगा|valid|मिलेगा)|valid (till|until|for how long)|how long (is|does) (this|the) offer|"
                                  r"offer (end|expire|ends|expires)|last date|end date|कब (खत्म|ख़त्म)|offer kab", _I),
          _t("the end date of the offer", "ऑफर की आख़िरी तारीख़", "offer ki aakhri date")),
    Topic("context_window", re.compile(r"context (window|length|size)|\btokens?\b|टोकन|कॉन्टेक्स्ट|कंटेक्स्ट|message limit|message count|kitne messages", _I),
          _t("the context size", "कॉन्टेक्स्ट की सीमा", "context size")),
    Topic("group_discount", re.compile(r"(friend|दोस्त|\bgroup\b|ग्रुप|couple|family|corporate|senior citizen|referral|refer).{0,40}(discount|डिस्काउंट|छूट|offer|ऑफर|\boff\b|plan|सस्ता|sasta|कम|kam)|"
                                       r"(discount|डिस्काउंट|छूट|offer|ऑफर|plan|कम|kam).{0,40}(friend|दोस्त|\bgroup\b|ग्रुप|couple|family|corporate|senior citizen)", _I),
          _t("details on a friend or group discount", "दोस्त या ग्रुप डिस्काउंट", "friend ya group discount")),
    Topic("payer", re.compile(r"(pota|पोता|beta|बेटा|son|husband|pati|पति|wife|daughter|grandson|dost|दोस्त|friend|papa|पापा|bhai|भाई)[^।?.!,]{0,40}"
                              r"(pay|(paise|पैसे|paisa|पैसा) (de|दे)|payment|पेमेंट|de (sakta|sakti|dega|degi)|दे (सकता|सकती|देगा|देगी))", _I),
          _t("details on someone else paying", "किसी और के पेमेंट करने", "kisi aur ke payment karne")),
]


def find_topics(p, text: str) -> list[Topic]:
    """Topics the question touches that the product facts do not cover (table order, at most the first two)."""
    out = []
    low = text.lower()
    for t in TOPICS:
        if t.applies is not None and not t.applies(p):
            continue
        if t.id == "offer_end" and _fixed_validity(p):
            continue  # the facts give a window ("if you join this week"): the offer item answers it
        if t.rx.search(low):
            out.append(t)
    return out[:2]


def covered_by_faq(p, text: str, skip_beats: bool = True) -> bool:
    """A product FAQ answer exists for this question (its keywords match): the approved answer is used, not the unknown line."""
    low = text.lower()
    for f in getattr(p, "faqs", []):
        if not getattr(f, "answer", ""):
            continue
        for k in f.keywords:
            if not k.strip():
                continue
            pat = k[3:] if k.startswith("re:") else re.escape(k.lower())
            if re.search(pat, low):
                return True
    return False


def end_variants(p, lang: str) -> list[str]:
    """The last rungs when a fact was already given and asked again: nothing more to add, said honestly in three
    different ways (only the first names where to look, so the three never read as one sentence)."""
    S = source(p, lang)
    return {"en": [f"That is all I have on this; {S}.",
                   "I don't have anything more on that, and I don't want to guess.",
                   "I'm sorry, that is really all I know.",
                   "I'd rather not guess on that one, so I'll leave it there."],
            "hi": [f"इस बारे में मेरे पास बस इतना ही है; {S}।",
                   "इससे ज़्यादा मेरे पास नहीं है, और मैं अंदाज़ा नहीं लगाना चाहता।",
                   "माफ़ कीजिए, मैं बस इतना ही जानता हूँ।",
                   "इस पर मैं अंदाज़ा नहीं लगाना चाहता, इसलिए यहीं रुकता हूँ।"],
            "hinglish": [f"Is baare mein mere paas bas itna hi hai; {S}.",
                         "Isse zyada mere paas nahi hai, aur main andaza nahi lagana chahta.",
                         "Sorry, main bas itna hi jaanta hoon.",
                         "Is par main andaza nahi lagana chahta, isliye yahin rukta hoon."]}[lang]


def unknown_variants(p, lang: str, topic: Topic | None = None) -> list[str]:
    S = source(p, lang)
    if topic is None:
        return {
            "en": [f"I don't have that detail with me, so I won't guess; {S}.",
                   "Sorry, I really am not sure about that, and I don't want to give you a wrong answer.",
                   "That is one I can't answer for sure."],
            "hi": [f"वो जानकारी मेरे पास नहीं है, इसलिए मैं अंदाज़ा नहीं लगाऊँगा; {S}।",
                   "माफ़ कीजिए, इस बारे में मुझे पक्का नहीं पता, और मैं ग़लत जवाब नहीं देना चाहता।",
                   "ये बात मैं पक्का नहीं बता सकता।"],
            "hinglish": [f"Wo jaankari mere paas nahi hai, isliye main andaza nahi lagaunga; {S}.",
                         "Sorry, is baare mein mujhe pakka nahi pata, aur main galat jawab nahi dena chahta.",
                         "Ye baat main pakka nahi bata sakta."],
        }[lang]
    n = topic.name[lang]
    return {
        "en": [f"I don't have {n} with me, so I won't guess; {S}.",
               f"Sorry, I really don't know about {n}, and I don't want to give you a wrong answer.",
               f"I can't say anything sure about {n}."],
        "hi": [f"{n} की सही जानकारी मेरे पास नहीं है, इसलिए मैं अंदाज़ा नहीं लगाऊँगा; {S}।",
               f"माफ़ कीजिए, {n} के बारे में मुझे पक्का नहीं पता, और मैं ग़लत जवाब नहीं देना चाहता।",
               f"{n} वाली बात मैं पक्का नहीं बता सकता।"],
        "hinglish": [f"{cap(n)} ki sahi jaankari mere paas nahi hai, isliye main andaza nahi lagaunga; {S}.",
                     f"Sorry, {n} ke baare mein mujhe pakka nahi pata, aur main galat jawab nahi dena chahta.",
                     f"{cap(n)} wali baat main pakka nahi bata sakta."],
    }[lang]


def unknown_combo(p, lang: str, topics: list[Topic], idx: int = 0) -> str:
    """Two things asked at once ("monthly plan? and a friend discount?"): the full honest line for the first, a short one for the
    second, so neither question is ignored."""
    first = unknown_variants(p, lang, topics[0])
    line = first[min(idx, len(first) - 1)]
    if len(topics) > 1:
        line += " " + unknown_variants(p, lang, topics[1])[2]
    return line


def payer_variants(p, lang: str) -> list[str]:
    """"Can my grandson pay?": common sense, no pointing at a payment page. Anyone can pay; nothing is taken on the call."""
    return {"en": ["Yes, anyone can pay for it, your grandson too; nothing is charged on this call, and our team can send you the details if you like.",
                   "Of course, whoever is paying can do it at the gym or through the link; just tell me if you want our team to send you the details."],
            "hi": ["जी हाँ, पेमेंट कोई भी कर सकता है, आपके पोते भी; इस कॉल पर कुछ नहीं कटता, और आप चाहें तो हमारी टीम आपको जानकारी भेज सकती है।",
                   "बिल्कुल, जो भी पेमेंट करना चाहे वो जिम पर या लिंक से कर सकता है; चाहें तो बताइए, हमारी टीम आपको जानकारी भेज देगी।"],
            "hinglish": ["Ji haan, payment koi bhi kar sakta hai, aapke pote bhi; is call par kuch nahi katta, aur aap chahein to hamari team aapko jaankari bhej sakti hai.",
                         "Bilkul, jo bhi payment karna chahe wo gym par ya link se kar sakta hai; chahein to bataiye, hamari team aapko jaankari bhej degi."]}[lang]


def offer_end_variants(p, lang: str) -> list[str]:
    """'How long is the offer?' answered from the facts: how long the price lasts, and honestly that no end date is known."""
    first, after = terms_parts(p, lang)
    S = source(p, lang)
    if _fixed_validity(p):
        return {"en": [f"The offer price is for {p.offer_validity}; I don't have any other end date, {S}."],
                "hi": [f"ऑफर की कीमत {first} के लिए है; इसके अलावा मेरे पास कोई आख़िरी तारीख़ नहीं है, {S}।"],
                "hinglish": [f"Offer ki keemat {first} ke liye hai; iske alawa mere paas koi aakhri date nahi hai, {S}."]}[lang]
    return {
        "en": [f"The {p.offer_price} price lasts {p.offer_validity} once you start, and then it is {p.original_price}; I don't have a last date for the offer itself, and {S}."],
        "hi": [f"शुरू करने के बाद {p.offer_price} वाली कीमत {p.offer_validity.replace('for the first', 'पहले').replace('months', 'महीने')} रहती है, फिर {p.original_price}; ऑफर की आख़िरी तारीख़ मेरे पास नहीं है, और {S}।"],
        "hinglish": [f"Shuru karne ke baad {p.offer_price} wali keemat {p.offer_validity.replace('for the first', 'pehle').replace('months', 'mahine')} rehti hai, phir {p.original_price}; offer ki aakhri date mere paas nahi hai, aur {S}."],
    }[lang]


def negotiate_variants(p, lang: str) -> list[str]:
    return {"en": ["I can't lower the price; it is fixed, and I don't have any extra discount to offer.",
                   "There is no extra discount I can give; the price stays as it is."],
            "hi": ["मैं कीमत कम नहीं कर सकता, ये तय है, और मेरे पास कोई अतिरिक्त डिस्काउंट नहीं है।",
                   "अलग से कोई डिस्काउंट मैं नहीं दे सकता; कीमत वैसी ही रहेगी।"],
            "hinglish": ["Main keemat kam nahi kar sakta, ye fixed hai, aur mere paas koi extra discount nahi hai.",
                         "Alag se koi discount main nahi de sakta; keemat waisi hi rahegi."]}[lang]


NEGOTIATE = re.compile(r"कम (नहीं |नही )?(कर|हो)|और कम|थोड़ा कम|kam (nahi |nhi )?(kar|ho)|thoda kam|cheaper|lower (the )?price|\breduce\b|bargain|"
                        r"discount (do|de|aur|mil)|(some|any|a little|more) discount|सस्ता कर|sasta kar|और डिस्काउंट|\d+ ?(kam|कम)", _I)


# ---------------------------------------------------------------- health: never promise anything
HEALTH = re.compile(r"घुटन|knee|joint|जोड़ों|दर्द|\bpain\b|\bbp\b|blood pressure|sugar|diabet|heart|दिल की|surgery|operation|ऑपरेशन|pregnan|injur|चोट|"
                    r"back pain|कमर|arthritis|asthma|doctor|डॉक्टर|bimari|बीमारी|illness", _I)
AGE = re.compile(r"बूढ़|बुज़ुर्ग|बुजुर्ग|senior|elderly|old (people|age)|\b\d{2} (saal|साल|years?)\b|उम्र|umar", _I)


def health_question(text: str) -> bool:
    return bool(HEALTH.search(text)) and is_question(text)


def health_variants(p, lang: str) -> list[str]:
    S = source(p, lang)
    return {
        "en": [f"I can't say what is safe for a health problem like that, so please ask your doctor first. For the rest, {S}.",
               f"For anything about your health, your doctor's word comes first; I can't promise what suits you. {cap(S)}."],
        "hi": [f"सेहत की ऐसी परेशानी में क्या सुरक्षित है ये मैं नहीं बता सकता, इसलिए पहले डॉक्टर से पूछ लीजिए। बाकी के लिए, {S}।",
               f"सेहत की किसी भी बात में पहले डॉक्टर की सलाह ज़रूरी है; आपके लिए क्या ठीक रहेगा ये मैं वादा नहीं कर सकता। {S}।"],
        "hinglish": [f"Sehat ki aisi pareshani mein kya safe hai ye main nahi bata sakta, isliye pehle doctor se pooch lijiye. Baaki ke liye, {S}.",
                     f"Sehat ki kisi bhi baat mein pehle doctor ki salah zaroori hai; aapke liye kya theek rahega ye main vaada nahi kar sakta. {cap(S)}."],
    }[lang]


# ---------------------------------------------------------------- requests for something we cannot send
_SEND = r"(send|bhej|bhejo|भेज|sms|message|मैसेज|whatsapp|व्हाट्सएप|व्हाट्सऐप|mail|मेल)"
_INFO_NOUN = {
    "trial": r"trial|ट्रायल|demo|guest pass|day pass",
    "branch": r"branch|ब्रांच|address|पते|पता|location|लोकेशन",
    "whatsapp": r"whatsapp|व्हाट्सएप|व्हाट्सऐप|\bwa\b|e-?mail|ईमेल|इमेल|\bmail\b",
    "free_signup": r"(free|फ्री|muft|मुफ्त)\W+(\w+\W+){0,2}(link|लिंक|signup|sign up|साइनअप|account|अकाउंट)|(link|लिंक)\W+(\w+\W+){0,3}(free|फ्री|muft|मुफ्त)",
}
_PAID = re.compile(r"payment link|पेमेंट लिंक|pay kar|पे कर|\bpro\b|premium|subscribe|सब्सक्राइब", _I)


def info_request(text: str) -> str | None:
    """The caller wants something sent that this agent cannot send (a trial pass, branch addresses, a WhatsApp/email message,
    a free sign-up link). The only thing it can send is the payment/joining link by SMS."""
    low = text.lower()
    if not re.search(_SEND, low, _I) and not re.search(_INFO_NOUN["free_signup"], low, _I):
        return None
    if re.search(r"\b(likh\w*|write|writes|writing|draft\w*|reply to)\b|लिख", low, _I) and not re.search(r"\bsend\b|bhej|भेज", low, _I):
        return None  # "will it write my emails?" asks what the PRODUCT does, not for an email from us
    for kind in ("free_signup", "trial", "branch", "whatsapp"):
        if re.search(_INFO_NOUN[kind], low, _I) and not (kind == "free_signup" and _PAID.search(low)):
            return kind
    return None


def info_request_variants(p, kind: str, lang: str) -> list[str]:
    S = source(p, lang)
    site = getattr(p, "official_site", "") or ""
    where = {"en": site or "the official website", "hi": site or "ऑफिशियल वेबसाइट", "hinglish": site or "official website"}[lang]
    link = {"en": "the payment link", "hi": "पेमेंट लिंक", "hinglish": "payment link"}[lang]
    if kind == "free_signup":
        return {"en": [f"I can't send a separate free sign-up link; the only thing our team sends is {link}, and only if you ask for it. You can start the free plan yourself on {where}."],
                "hi": [f"मैं अलग से फ्री साइन-अप का लिंक नहीं भेज सकता; मैं सिर्फ़ {link} भेजता हूँ, और वो भी तभी जब आप कहें। फ्री प्लान आप ख़ुद {where} पर शुरू कर सकते हैं।"],
                "hinglish": [f"Main alag se free sign-up ka link nahi bhej sakta; main sirf {link} bhejta hoon, aur wo bhi tabhi jab aap kahein. Free plan aap khud {where} par shuru kar sakte hain."]}[lang]
    if kind == "whatsapp":
        return {"en": [f"I can't send a WhatsApp or an email from this call. If you'd like, our team can send you {link}; otherwise {S}."],
                "hi": [f"मैं इस कॉल से WhatsApp या ईमेल नहीं भेज सकता। आप चाहें तो हमारी टीम आपको {link} भेज सकती है; वरना {S}।"],
                "hinglish": [f"Main is call se WhatsApp ya email nahi bhej sakta. Aap chahein to hamari team aapko {link} bhej sakti hai; warna {S}."]}[lang]
    what = {"trial": ("a trial pass", "ट्रायल पास", "trial pass"), "branch": ("branch details", "ब्रांच की जानकारी", "branch ki jaankari")}[kind]
    return {"en": [f"I can't send {what[0]} from here; the only thing our team sends is {link}, and only if you ask for it. {cap(S)}."],
            "hi": [f"मैं यहाँ से {what[1]} नहीं भेज सकता; मैं सिर्फ़ {link} भेजता हूँ, और वो भी तभी जब आप कहें। {S}।"],
            "hinglish": [f"Main yahan se {what[2]} nahi bhej sakta; main sirf {link} bhejta hoon, aur wo bhi tabhi jab aap kahein. {cap(S)}."]}[lang]


# ---------------------------------------------------------------- hard times: warm, short, no selling
SAD = re.compile(
    r"lost my job|job (nahi|chali|gayi|gai|chhut|chut)|naukri (nahi|chali|gayi|chhut|chut)|jobless|unemployed|no job|laid off|got fired|"
    r"(abhi |ab )?(paise|paisa|pese) (nahi|nhi)|no money|can.?t afford|cannot afford|afford nahi|i.?m broke|paise ki (tangi|dikkat|kami)|money (problem|trouble|issue)|"
    r"kuch (accha|achha) nahi chal|sab kharab|money (is )?(very |really )?tight|tight on money|(really|very) low|paise ki (kami|tangi|dikkat)|ghar mein (bimari|bimaar)|pata nahi kya karun|depress|udaas|udas|feeling low|thak (gaya|gayi)|tired of everything|tension mein|stress mein|"
    r"dukhi|dukh mein|mushkil (waqt|time|din)|going through a (hard|tough|bad)|"
    r"नौकरी (नहीं|चली|छूट)|जॉब (नहीं|चली|छूट)|(अभी )?पैसे नहीं|पैसा नहीं|पैसों की (तंगी|दिक्कत|कमी)|पैसे की (तंगी|दिक्कत|कमी)|घर में (बीमारी|बीमार)|बीमारी चल|पता नहीं क्या करूँ|पता नहीं क्या करूं|उदास|दुखी|थक (गया|गई|गयी)|मुश्किल (समय|वक्त|दिन)|"
    r"कुछ अच्छा नहीं चल|सब खराब|डिप्रेस", _I)


def empathy_variants(p, lang: str, first: bool) -> list[str]:
    if first:
        return {"en": ["I'm really sorry to hear that, that sounds hard. Please don't worry about anything from my side, I won't push anything. Take care of yourself, and if it helps I can call another time."],
                "hi": ["ये सुनकर सच में बुरा लगा, ये मुश्किल समय होगा। मेरी तरफ़ से कोई दबाव नहीं है, आप बस अपना ख़याल रखिए; चाहें तो मैं किसी और समय फ़ोन कर लूँगा।"],
                "hinglish": ["Ye sunkar sach mein bura laga, ye mushkil time hoga. Meri taraf se koi pressure nahi hai, aap bas apna khayal rakhiye; chahein to main kisi aur time phone kar loonga."]}[lang]
    return {"en": ["I understand. Take your time; there is no hurry from my side.",
                   "I'm here if you want to ask anything, and we can talk another day too.",
                   "Please look after yourself; I hope things get easier soon."],
            "hi": ["जी, समझता हूँ। आराम से, मेरी तरफ़ से कोई जल्दी नहीं है।",
                   "आपको कुछ पूछना हो तो मैं यहीं हूँ, और किसी और दिन भी बात कर सकते हैं।",
                   "अपना ख़याल रखिएगा; उम्मीद है जल्दी सब आसान हो जाएगा।"],
            "hinglish": ["Ji, samajhta hoon. Aaram se, meri taraf se koi jaldi nahi hai.",
                         "Aapko kuch poochna ho to main yahin hoon, aur kisi aur din bhi baat kar sakte hain.",
                         "Apna khayal rakhiyega; umeed hai jaldi sab aasaan ho jaayega."]}[lang]


# ---------------------------------------------------------------- reading the caller's words
FREE_WANT = re.compile(r"\b(free|muft)\b|फ्री|मुफ्त", _I)
NO_LINKS = re.compile(r"(link|लिंक)\W+(\w+\W+){0,4}(nahi|नहीं|नही)\W+(\w+\W+){0,2}(chal|खोल|khol|aata|आता|use|इस्तेमाल|click|क्लिक)|"
                      r"can.?t (use|open|click)( the| a)? links?|don.?t know how to (use|open|click)|links? (nahi|नहीं) (chala|chal|khol)|"
                      r"i don.?t click|(link|लिंक) (मुझे )?नहीं (चलाना|खोलना)", _I)
SEND_REQ = re.compile(r"send (me )?(it|the link|link|me the link)|send the link|link (bhej|bhejo|भेज)|(लिंक|link) (भेज|भेजो|भेजिए|भेजिये|भेज दो|bhej do|bhejiye)|"
                      r"bhej (do|dijiye|dena|de|den|dein|dijiye)|भेज (दो|दीजिए|दीजिये|देना|दे|दें|दीजिए)|bhej dijiye|send it", _I)
NOT_SENDING = re.compile(r"(bhej|भेज) (dega|degi|denge|dunga|dungi|dūnga|देगा|देगी|देंगे|दूँगा|दूंगा|दूँगी|दूंगी)|(pay|पेमेंट|paise|पैसे) (kar )?(dega|degi|denge|देगा|देगी)", _I)
