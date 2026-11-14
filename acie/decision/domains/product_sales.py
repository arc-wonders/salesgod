"""Outbound product-selling domain, driven by a `Product` config editable from the console.

Call flow (FSM in session.dialog):
  greet (opening) --yes/what--> pitch --yes--> close --yes--> link sent, end
        |                         |              |
        +--no--> one soft re-offer (price) ---no--> polite goodbye, end   (max 1 re-offer: no pressure)
  price / benefits / FAQ questions are answered from the config at any point.
  "call me later" -> callback, end.   Questions outside the config -> LLM answers from config facts only.
"""

from __future__ import annotations

import asyncio
import json
import os
import re

from pydantic import BaseModel, Field, model_validator

from ...business import Customer, MockBusinessAPI
from ...context import _OUTCOMES
from ...schemas import Fact, SessionState, TurnInput, TurnPlan
from .. import constraints as cons
from .. import honest
from .. import negotiation
from .. import nba
from ..engine import KeywordIntentClassifier
from ..hybrid import HybridIntentClassifier, load_turn_model
from .bank_demo import KEYWORDS as BANK_KEYWORDS, MASKS as BANK_MASKS


class FAQ(BaseModel):
    keywords: list[str] = Field(default_factory=list)  # any of these in the customer's words
    answer: str = ""
    answer_hi: str = ""  # if given, Hindi calls speak this as written (fast); else the LLM translates
    answer_hinglish: str = ""  # same for Hinglish (Roman script); never speak the English answer in a Hindi/Hinglish call
    not_eligible: bool = False  # this answer means the offer doesn't apply to them: stop selling


class Step(BaseModel):
    """One rung of the concession ladder: a value perk (kind="perk", operator-written text) or a lower price (kind="price")."""
    kind: str = Field("perk", pattern="^(perk|price)$")
    id: str = ""
    price: float = 0.0           # price steps: the new price of one offer unit (must be at or above the profit floor)
    text: dict[str, str] = Field(default_factory=dict)   # perk sentence per language (price steps are worded by the state machine)
    short: dict[str, str] = Field(default_factory=dict)  # perk as a short phrase for the final terms ("plus one extra month free")


class Economics(BaseModel):
    """Per-product economics behind the concession ladder (acie/decision/negotiation.py). All zero = no concessions at all."""
    currency: str = "₹"
    unit: str = ""               # what one price covers, for the wording ("3 months")
    list_price: float = 0.0      # the price the call opens with (the offer price)
    cost: float = 0.0
    min_margin_pct: float = 0.0
    price_terms: dict[str, str] = Field(default_factory=dict)  # the full offer at a new price, per language, with {price}
    ladder: list[Step] = Field(default_factory=list)           # value perks first, then price steps; each price at or above the floor

    @property
    def profit_floor(self) -> float:
        return round(self.cost * (1 + self.min_margin_pct / 100.0), 2)


class Negotiation(BaseModel):
    """What the operator allows when a caller bargains (acie/decision/negotiation.py). Default: nothing, the price is the floor.
    levers: any of trial_pass, cheaper_plan, annual_saving, referral -> {en, hi, hinglish} text the agent may say. They are
    offered in that fixed order, each once, and only if configured here."""
    levers: dict[str, dict[str, str]] = Field(default_factory=dict)


class Product(BaseModel):
    id: str = "claude-pro"
    description: str = ""  # who wrote the facts: e.g. "operator demo facts" (authored by the operator for a demo; see docs/PRODUCT_FACTS_TODO.md)
    seller_name: str = "Anthropic"
    agent_name: str = "Arjun"
    product_name: str = "Claude Pro"
    one_liner: str = "an AI assistant for writing, coding, research and analysis"
    original_price: str = "$20 per month"
    offer_price: str = "$8 per month"
    discount: str = "60% off"
    offer_validity: str = "for the first 3 months"
    benefits: list[str] = Field(default_factory=lambda: [
        "much higher usage than the free plan",
        "access to the most capable Claude models",
        "Projects to organise your documents and chats",
    ])
    call_to_action: str = "send you a payment link on SMS"
    # The benefits above in Hindi / Hinglish (same order). Spoken as written; if a language has none, the LLM
    # paraphrases the English fact instead (an English fragment is never pasted into a Hindi or Hinglish reply).
    benefits_hi: list[str] = Field(default_factory=lambda: [
        "फ्री प्लान से कहीं ज़्यादा इस्तेमाल", "सबसे अच्छे Claude मॉडल्स का एक्सेस",
        "Projects, जिनमें आप अपने documents और chats एक जगह रख सकते हैं"])
    benefits_hinglish: list[str] = Field(default_factory=lambda: [
        "free plan se kahin zyada usage", "sabse capable Claude models ka access",
        "Projects, jisme aap apne documents aur chats ek jagah rakh sakte hain"])
    # The agent's gender decides the first-person verb forms in Hindi ("चाहता हूँ", never "चाहती हूँ"). Arjun is male.
    agent_gender: str = Field("male", pattern="^(male|female)$")
    # Where the customer finds a detail the agent does not have (spoken after "I don't have that detail"), per language
    info_source: dict[str, str] = Field(default_factory=lambda: {
        "en": "it will be shown on the payment page or the official website",
        "hi": "वो पेमेंट पेज या ऑफिशियल वेबसाइट पर दिख जाएगा",
        "hinglish": "wo payment page ya official website par dikh jaayega"})
    # Can the customer cancel any time? None = only if the product's own facts (FAQ answers, background) say so.
    cancel_any_time: bool | None = None
    # the official site, named when the agent can't send something ("the free plan: sign up yourself on claude.ai")
    official_site: str = "claude.ai"
    negotiation: Negotiation = Field(default_factory=Negotiation)
    economics: Economics = Field(default_factory=Economics)

    @model_validator(mode="after")
    def _no_borrowed_content(self):
        """A product that is not the Claude demo never inherits the Claude demo's content: every content field its own config
        does not set is emptied (so a missing field goes down the honest "I don't have that" path, never into another product's
        words). `Product()` with nothing set is still the Claude Pro demo (`Product.claude_demo()`)."""
        given = self.model_fields_set
        if not given or self.product_name == "Claude Pro" and "product_name" not in given and "seller_name" not in given:
            return self
        if "product_name" not in given and "seller_name" not in given:
            return self
        neutral = {"one_liner": "", "benefits": [], "benefits_hi": [], "benefits_hinglish": [], "faqs": [], "knowledge": "", "pitch_hi": "",
                   "pitch_hinglish": "", "pitch_en": "", "price_note": "", "rupee_terms": {}, "renewal_note": {}, "annual_note": {},
                   "pronunciations": {}, "payment_link": "", "official_site": "", "topic_words": [], "call_to_action": "send you a payment link on SMS",
                   "discovery_question": {"en": "What would you mainly like it for?", "hi": "आप इसे मुख्य रूप से किस लिए चाहते हैं?",
                                          "hinglish": "Aap ise mukhya roop se kis liye chahte hain?"},
                   "info_source": {"en": "the official website will have it", "hi": "ऑफिशियल वेबसाइट पर मिल जाएगा", "hinglish": "official website par mil jaayega"}}
        for k, v in neutral.items():
            if k not in given:
                object.__setattr__(self, k, v)
        return self

    @classmethod
    def claude_demo(cls) -> "Product":
        """The Claude Pro demo product (what `Product()` is with nothing set)."""
        return cls()
    # One-sentence pitch written by the operator, per language. Spoken as-is (fast, natural);
    # if empty the LLM writes the pitch from the benefits.
    pitch_hi: str = ("Claude Pro एक AI असिस्टेंट है जो लिखने, कोडिंग और रिसर्च में मदद करता है, "
                     "और इसमें फ्री प्लान से कहीं ज़्यादा इस्तेमाल मिलता है।")
    pitch_hinglish: str = ("Claude Pro ek AI assistant hai jo likhne, coding aur research mein madad karta hai, "
                           "aur isme free plan se kahin zyada usage milta hai.")
    pitch_en: str = "Claude Pro is an AI assistant for writing, coding and research, with far more usage than the free plan."
    # What the product/entity actually is, so the agent understands questions about it.
    # Background for the LLM, not a script. Keep it factual.
    knowledge: str = (
        "Claude is an AI assistant made by Anthropic. People use it by typing or talking to it to write and edit text, "
        "write and debug code, analyse documents and data, research topics and brainstorm. It works on the web "
        "(claude.ai), on iPhone and Android apps and on desktop. Claude Pro is the paid individual plan: much more "
        "usage than the free plan, access to more models, and Projects to keep documents and chats together. "
        "It is software, not a physical product; nothing is shipped. Limits: it cannot create images or videos (it can "
        "read and understand images, PDFs and files you upload); it needs the internet, there is no offline mode; it "
        "cannot make bookings, payments or phone calls for you. A Pro subscription is for one person's account and "
        "should not be shared; groups use the separate Team plan. Students, writers, developers and small businesses "
        "use it; it can explain homework step by step. This offer is for new Pro subscriptions: someone who already pays "
        "for Pro keeps their current plan and price, and someone on the free plan can take the offer. Many people use Claude alongside other assistants such as ChatGPT; "
        "people often pick Claude for careful writing, working with long documents and coding help. Coding: it explains "
        "problems step by step, helps debug, and can read long code files; good for practising DSA. It writes and edits "
        "reports and assignments and works in Hindi and many other languages. Privacy: Anthropic may use chats to improve "
        "its models only if the user allows it; each user chooses this in their privacy settings, and the full policy is at "
        "anthropic.com/privacy (say exactly this; do not claim it never uses data). Billing: "
        "there is also yearly billing that works out cheaper per month (exact price on the pricing page). There is no "
        "separate student discount in this offer; the discount already applies. Cancelling stops the next renewal and "
        "access continues until the end of the paid month; refunds follow the policy on the website. Anthropic is a real "
        "AI safety and research company, founded in 2021 and based in San Francisco, that makes Claude; the official site "
        "is anthropic.com. The payment page shows the payment methods; card works, and other methods depend on what the "
        "page shows (do not promise UPI or any specific method). The payment link comes by SMS to the number we called. "
        "Renewal: the subscription renews automatically every month; after the first 3 months it renews at $20 a month "
        "until the user cancels (say this plainly if asked whether money is taken automatically). Models: Pro includes "
        "Anthropic's latest Claude models, including the most capable Opus and the fast Sonnet; usage is roughly five times "
        "the free plan, counted in sessions that reset every five hours; the exact number of messages depends on their "
        "length, so there is no fixed message count (say so plainly). The $8 offer is only for the monthly plan. Yearly "
        "billing is listed at about $200 a year (about $17 a month) with no extra offer; Team plan prices are on the "
        "website. The SMS with the link usually arrives within a couple of minutes and opens Anthropic's official "
        "claude.ai page. Receipts and "
        "invoices for each payment are available in the account's billing settings. For a shop or small business it "
        "helps: GST calculations, bill formats, lists of dues (udhaari), and drafting WhatsApp or SMS messages to customers "
        "(it writes them, the user sends them); the user should check the numbers. Tally (टैली/टेली) is accounting "
        "software; Claude is not a replacement for Tally, but helps alongside it. "
        "OTP: on the payment page the user's own bank may send an OTP to approve the payment; it must never be shared "
        "with anyone, including this call. "
        "If an existing subscriber asks whether cancelling and signing up again with a new account would get the offer: "
        "you don't know that and must not promise it. Colleagues or friends who don't have Pro yet can each take this "
        "offer on their own account. "
        "The SMS comes from Anthropic, not from a bank, and the link opens Anthropic's own claude.ai payment page. "
        "Never suggest making another account or email to get the offer, and never claim what is or isn't allowed for "
        "second accounts; you can't confirm those rules. "
        "If asked why they are getting this call: Anthropic is running an intro offer on Claude Pro and you're calling to "
        "let people know about it; say that in one line and offer to remove their number if they'd prefer no calls. "
        "If asked where we got their number: you don't have details of the list's source; offer to remove the number.")
    # Words that mean the customer is talking about this product (anything else is off-topic)
    topic_words: list[str] = Field(default_factory=lambda: [
        "claude", "क्लॉड", "क्लाउड", "cloud", "pro", "प्रो", "ai", "एआई", "assistant", "असिस्टेंट", "subscription",
        "सब्सक्रिप्शन", "plan", "प्लान", "offer", "ऑफर", "app", "ऐप", "chatgpt", "anthropic", "model", "मॉडल",
        # pronouns pointing back at what we just pitched ("does it work on iPhone?", "इसमें क्या है")
        "it", "this", "ये", "यह", "इसमें", "इसका", "इसको", "इसे", "isme", "iska", "isko", "ise", "ye", "yeh"])
    faqs: list[FAQ] = Field(default_factory=lambda: [
        # live call 912c391917: "I'm already using it for six months, why so cheap?"
        FAQ(keywords=["already using claude", "already use claude", "already have claude", "already have pro",
                      "already subscribed", "already a subscriber", "pehle se le rakha", "पहले से ले रखा",
                      "पहले से इस्तेमाल", "पहले से यूज़", "existing", "already pay", "already paying", "current plan",
                      "pro le rakha", "pro पहले से", r"re:already .{0,25}(claude|pro|क्लॉड|प्रो)", r"re:(claude|pro|क्लॉड|प्रो) .{0,25}(use|यूज़|यूज|इस्तेमाल) (kar|कर)", "pehle se pro", "पहले से pro", "पहले से प्रो", "claude use kar raha", "क्लॉड इस्तेमाल कर रहा", "क्लॉड यूज़ कर रहा"],
            answer="Thanks for using Claude! This offer is for new Pro subscriptions; if you already pay for Pro, your plan stays as it is",
            answer_hi="Claude इस्तेमाल करने के लिए शुक्रिया! ये ऑफर नए Pro सब्सक्रिप्शन के लिए है; अगर आप पहले से Pro लेते हैं, तो आपका प्लान वैसा ही रहेगा",
            answer_hinglish="Claude use karne ke liye shukriya! Ye offer naye Pro subscriptions ke liye hai; agar aap pehle se Pro lete hain, to aapka plan waisa hi rahega",
            not_eligible=True),
        # objection heard on live call 4, asked three times and never answered
        FAQ(keywords=["fake", "nakli", "नकली", "scam", "fraud", "फ्रॉड", "genuine", "असली", "asli", "real hai",
                      "so cheap", "सस्ता", "sasta", "itna kam", "इतना कम", "धोखा", "dhokha"],
            answer="Fair question. It's a genuine intro discount for the first 3 months; you pay only through a secure link, and can cancel any time",
            answer_hi="अच्छा सवाल है। ये पहले तीन महीने का असली intro डिस्काउंट है, पेमेंट सिर्फ़ सुरक्षित लिंक से होता है, और आप कभी भी cancel कर सकते हैं",
            answer_hinglish="Achha sawal hai. Ye pehle teen mahine ka asli intro discount hai, payment sirf secure link se hota hai, aur aap kabhi bhi cancel kar sakte hain"),
        FAQ(keywords=["how to pay", "how do i pay", "how can i pay", "how to buy", "how do i buy", "kaise kharid",
                      "कैसे खरीद", "कैसे पेमेंट", "पेमेंट कैसे", "पेवेंट कैसे", "kaise pay", "payment kaise", "kaise le"],
            answer="I'll send a secure payment link by SMS to this number; you open it and pay on the official page with the methods shown there, usually a card; nothing is taken on this call",
            answer_hi="मैं इसी नंबर पर SMS से एक सुरक्षित पेमेंट लिंक भेजूँगा, उसे खोलकर ऑफिशियल पेज पर वहाँ दिखाए गए तरीके से, आमतौर पर कार्ड से, पेमेंट होता है; इस कॉल पर कुछ नहीं लिया जाता",
            answer_hinglish="Main isi number par SMS se ek secure payment link bhejunga, use kholkar official page par wahan dikhaye gaye tareeke se, aamtaur par card se, payment hota hai; is call par kuch nahi liya jaata"),
        FAQ(keywords=["train", "training", "my data", "privacy", "प्राइवेसी", "डेटा", "डाटा", "data use", "chats use"],
            answer="Only if you allow it: your privacy settings have an option that controls whether your chats are used to improve the models, and you can switch it off any time",
            answer_hi="सिर्फ़ तब, जब आप इजाज़त दें: आपकी privacy settings में एक option है जिससे तय होता है कि आपकी chats मॉडल सुधारने में इस्तेमाल हों या नहीं, और आप उसे कभी भी बंद कर सकते हैं",
            answer_hinglish="Sirf tab, jab aap ijazat dein: aapki privacy settings mein ek option hai jisse tay hota hai ki aapki chats model sudharne mein use hon ya nahi, aur aap use kabhi bhi band kar sakte hain"),
        FAQ(keywords=["cancel", "band kar", "बंद कर", "रद्द", "कैंसल"],
            answer="You can cancel any time in your claude.ai settings, under billing; you keep access until the end of the month you've paid for, and nothing more is charged",
            answer_hi="आप कभी भी claude.ai की settings में billing से cancel कर सकते हैं; जिस महीने का पैसा दिया है उसके आख़िर तक इस्तेमाल चलता रहेगा, और आगे कुछ नहीं कटेगा",
            answer_hinglish="Aap kabhi bhi claude.ai ki settings mein billing se cancel kar sakte hain; jis mahine ka paisa diya hai uske aakhir tak use chalta rahega, aur aage kuch nahi katega"),
        FAQ(keywords=["re:(student|स्टूडेंट|छात्र).{0,25}(discount|डिस्काउंट|offer|ऑफर|छूट|sasta|सस्ता|kam|कम)",
                      "re:(discount|डिस्काउंट|छूट).{0,25}(student|स्टूडेंट|छात्र)"],
            answer="There's no separate student discount; this offer is the same for everyone, and it's already 60% off for the first 3 months",
            answer_hi="अलग से student discount नहीं है; ये ऑफर सबके लिए एक जैसा है, और पहले 3 महीने पहले से 60% off है",
            answer_hinglish="Alag se student discount nahi hai; ye offer sabke liye ek jaisa hai, aur pehle 3 mahine pehle se 60% off hai"),
        FAQ(keywords=["hindi", "हिंदी", "language", "bhasha", "भाषा"], answer="It understands and replies in Hindi and many other languages",
            answer_hi="ये हिंदी समेत कई भाषाएँ समझता है और उनमें जवाब देता है",
            answer_hinglish="Ye Hindi samet kai bhashayein samajhta hai aur unmein jawab deta hai"),
        FAQ(keywords=["free", "फ्री", "muft", "मुफ्त"], answer="There is a free plan, but Pro gives much more usage and the best models",
            answer_hi="फ्री प्लान भी है, पर Pro में कहीं ज़्यादा इस्तेमाल और सबसे अच्छे मॉडल मिलते हैं",
            answer_hinglish="Free plan bhi hai, par Pro mein kahin zyada usage aur sabse achhe models milte hain"),
        # persona test 14 Nov: UPI / refund / "why switch from ChatGPT" were answered with filler or invented values
        FAQ(keywords=["upi", "यूपीआई", "यू पी आई", "gpay", "phonepe", "paytm", "netbanking", "net banking", "payment method", "payment methods"],
            answer="The payment page shows which payment methods work; a card works, but I can't promise UPI or any other method",
            answer_hi="पेमेंट पेज पर दिखेगा कि कौन-से तरीके चलते हैं; कार्ड चलता है, पर UPI या किसी और तरीके का मैं वादा नहीं कर सकता",
            answer_hinglish="Payment page par dikhega ki kaun se tareeke chalte hain; card chalta hai, par UPI ya kisi aur tareeke ka main vaada nahi kar sakta"),
        FAQ(keywords=["refund", "रिफंड", "रिफण्ड", "money back", "paise wapas", "पैसे वापस", "पैसा वापस", "paisa wapas"],
            answer="Refunds follow the policy on the website; I don't have the exact terms with me",
            answer_hi="रिफंड वेबसाइट की पॉलिसी के हिसाब से होता है; उसकी सही शर्तें मेरे पास नहीं हैं",
            answer_hinglish="Refund website ki policy ke hisaab se hota hai; uski sahi sharten mere paas nahi hain"),
        FAQ(keywords=[r"re:(chatgpt|gemini|copilot|gpt).{0,40}(better|vs|versus|compar|differen|alag|अलग|फर्क|फ़र्क|बेहतर|switch|kyun|क्यों)",
                      r"re:(better|vs|versus|compar|differen|alag|अलग|फर्क|फ़र्क|बेहतर|switch).{0,40}(chatgpt|gemini|copilot|gpt)",
                      "plus se", "kyun switch", "क्यों switch"],
            answer="Many people use Claude alongside other assistants such as ChatGPT; people often pick Claude for careful writing, working with long documents and coding help. I can't say it is better for everyone, that depends on what you do",
            answer_hi="बहुत से लोग Claude को ChatGPT जैसे दूसरे असिस्टेंट्स के साथ भी इस्तेमाल करते हैं; लोग अक्सर careful writing, लंबे documents और coding की मदद के लिए Claude चुनते हैं। मैं ये नहीं कह सकता कि ये सबके लिए बेहतर है, ये आपके काम पर निर्भर करता है",
            answer_hinglish="Bahut se log Claude ko ChatGPT jaise dusre assistants ke saath bhi use karte hain; log aksar careful writing, lambe documents aur coding ki madad ke liye Claude chunte hain. Main ye nahi keh sakta ki ye sabke liye behtar hai, ye aapke kaam par nirbhar karta hai"),
    ])
    max_reoffers: int = Field(1, ge=0, le=1)  # compliance: at most one soft re-offer after a "no"
    # natural: the LLM phrases every turn to fit the conversation (the decision layer still sets
    # what to say; prices stay in locked slots; templates are the fallback). scripted: the reviewed
    # templates are spoken word for word (fastest, most rigid).
    style: str = Field("natural", pattern="^(natural|scripted)$")
    tone: str = "warm, respectful and conversational, like a helpful person, never pushy"
    # optional: the offer in the customer's currency, said only as an approximation
    price_note: str = ("about ₹700 a month for the first 3 months, then about ₹1,700 a month, at today's exchange rate; "
                       "the exact amount with taxes is shown on the payment page")
    # If not empty, this campaign may only dial these numbers. Used when the seller is a real
    # company that hasn't authorised the calls (the Claude demo uses Anthropic's name): test phones only.
    # spoken word for word after the price on a price question (the LLM never phrases prices itself)
    rupee_terms: dict[str, str] = Field(default_factory=lambda: {
        "en": "that's about ₹700 a month, then about ₹1,700", "hi": "यानी लगभग 700 रुपये महीना, फिर लगभग 1,700 रुपये",
        "hinglish": "yaani lagbhag 700 rupaye mahina, phir lagbhag 1,700"})
    renewal_note: dict[str, str] = Field(default_factory=lambda: {
        "en": "The discount is only for the first 3 months; after that it renews automatically at $20 a month until you cancel, and you can cancel any time.",
        "hi": "छूट सिर्फ़ पहले 3 महीने है; उसके बाद अपने आप $20 महीना लगेगा, जब तक आप cancel न करें, और cancel कभी भी कर सकते हैं।",
        "hinglish": "Discount sirf pehle 3 mahine hai; uske baad apne aap $20 mahina lagega jab tak aap cancel na karein, aur cancel kabhi bhi kar sakte hain."})
    annual_note: dict[str, str] = Field(default_factory=lambda: {
        "en": "Yearly billing is about $200 a year, around $17 a month; this $8 offer is only for the monthly plan.",
        "hi": "सालाना प्लान लगभग $200 साल का है, यानी करीब $17 महीना; ये $8 वाला ऑफर सिर्फ़ मंथली प्लान पर है।",
        "hinglish": "Yearly plan lagbhag $200 saal ka hai, yaani kareeb $17 mahina; ye $8 wala offer sirf monthly plan pe hai."})
    allowed_numbers: list[str] = Field(default_factory=list)  # empty = any number; the Claude demo sets its test phones
    # Spoken respellings for the TTS on Hindi turns (whole word -> what the voice reads). The
    # transcript keeps the written form. Same idea as voice_platform/pronunciation.yaml.
    pronunciations: dict[str, str] = Field(default_factory=lambda: {
        "Anthropic": "एन्थ्रॉपिक", "Claude": "क्लॉड", "SMS": "एस एम एस", "AI": "ए आई", "Pro": "प्रो"})
    # Text message sent after a "yes" (voice_platform F4: the agent says "sending the link", so send
    # it). {offer_price} {product} {link} {seller} are filled in; empty = no SMS.
    payment_link: str = "https://claude.ai/upgrade"
    sms_text: str = ""  # hackathon demo: no SMS is sent
    # the full offer as one spoken phrase per language; empty = built from offer_price/offer_validity/original_price
    terms: dict[str, str] = Field(default_factory=dict)
    # asked after the price, to learn their need (per language)
    discovery_question: dict[str, str] = Field(default_factory=lambda: {
        "en": "What would you mainly use it for?", "hi": "आप इसे ज़्यादातर किस काम के लिए इस्तेमाल करेंगे?",
        "hinglish": "Aap ise zyadatar kis kaam ke liye use karenge?"})
    tenant: str = "default"  # whose campaign this is: playbooks are stored per tenant and product
    use_playbook: bool = True  # use the pre-generated decision tree (data/voice/playbooks/<tenant>/<id>.json)
    # Declarative rules checked before every plan (acie/decision/constraints.py). Empty = no rules, unchanged behaviour.
    constraints: list[dict] = Field(default_factory=list)
    playbook_tree: str = ""  # optional path of a hand-written tree file (acie.playbook.load_tree), used when no playbook is passed


KEYWORDS = {
    "clarify": BANK_KEYWORDS["clarify"],
    "human_agent": [k for k in BANK_KEYWORDS["human_agent"] if not re.search(r"card|कार्ड|fraud|फ्रॉड|फ़्रॉड|धोखा|scam", str(k))]
                   + ["pass the phone", "a person who", "someone who can", "real human", "kisi aur ko do",
                      "किसी और को दो", "किसी इंसान", "insaan se baat"],
    "affirm": BANK_KEYWORDS["affirm"] + ["interested", "sign me up", "send the link", "link bhej*",
                                         "लिंक भेज*", "le lunga", "le lenge", "ले लूंगा", "ले लेंगे",
                                         "chahiye", "चाहिए", "buy", "kharid*", "खरीद*", "subscribe kar*",
                                         "सब्सक्राइब कर*", "le leta", "ले लेता", "ले लेती", "le leti", "ले लो", "le lo"],
    # "go on / tell me" means keep talking, never "yes, sell it to me" (the first live call closed
    # a sale on "बताइए")
    "tell_more": ["tell me", "go on", "go ahead and tell", "bataiye", "batao", "boliye", "bolo", "haan bolo",
                  "बताइए", "बताइये", "बताओ", "बोलिए", "बोलो", "हां बोलो", "हाँ बोलिए", "सुनाइए", "kya hai"],
    "deny": BANK_KEYWORDS["deny"] + ["not interested", "no thanks", "no thank you", "dont want",
                                     "dont need", "nahi lena", "नहीं लेना", "जरूरत नहीं", "zarurat nahi",
                                     "zaroorat nahi", "need nahi"],
    "price": ["price", "cost", "kitne ka", "kitna hai", "kitne ki", "kitne mein", "how much", "charges",
              "fees", "fee", "paisa", "paise", "rate", "kya rate", "discount", "offer", "कितने का",
              "कितने की", "कितना है", "कितने में", "कीमत", "दाम", "प्राइस", "चार्ज", "फीस", "डिस्काउंट",
              "ऑफर", "पैसे"],
    "benefits": ["what is", "what does", "features", "benefit", "kya hai", "kya milega", "kya karta",
                 "kya kar sakta", "fayda", "faayda", "use kya", "kaise kaam", "explain", "details",
                 "क्या है", "क्या मिलेगा", "क्या करता", "फायदा", "फ़ायदा", "क्या कर सकता", "कैसे काम",
                 "डिटेल", "जानकारी"],
    "callback": ["busy", "baad mein", "baad me", "later", "call back", "callback", "abhi time nahi",
                 "time nahi", "meeting", "driving", "kal call", "कल कॉल", "बाद में", "बिजी", "अभी टाइम नहीं",
                 "टाइम नहीं", "मीटिंग", "ड्राइव"],
    "goodbye": ["bye", "bye bye", "बेबाई", "बायबाय", "बाय बाय", "thank you", "thanks", "dhanyavad", "shukriya", "धन्यवाद", "शुक्रिया", "बाय",
                "रखता हूं", "रखती हूं", "rakhta hoon", "rakhti hoon"],
}
MASKS = {"deny": BANK_MASKS["deny"] + ["time nahi", "टाइम नहीं", "abhi time nahi",
                                      # "I don't know" is uncertainty, not a no (live call 10)
                                      "dont know", "do not know", "not sure", "pata nahi", "पता नहीं", "maloom nahi",
                                      "मालूम नहीं", "dont understand", "not able to understand"],
         "affirm": BANK_MASKS["affirm"] + ["nahi chahiye", "नहीं चाहिए", "zarurat nahi", "जरूरत नहीं"]}
GATED = {"affirm": "pending", "deny": "pending"}
PRIORITY = ["clarify", "human_agent", "callback", ("deny", "affirm")]
# affirm only wins over tell_more when it scores higher (the gated-priority rule in the classifier)


_QUESTION = re.compile(r"\?|\b(what|how|why|when|where|which|who|kya|kaise|kab|kyun|kyon|kaun|kahan|kitna|kitne)\b|"
                       # "do/is/can" only open a question at the start ("okay fine, let's do it" is not one)
                       r"^\W*(can|could|does|do|is|are|will|would|should)\b|क्या|कैसे|कब|क्यों|कौन|कहां|कहाँ|कितना|कितने|कितनी",
                       re.IGNORECASE)


_HYPOTHETICAL = re.compile(r"\b(agar|if|suppose|maan lo|man lo)\b|अगर|मान लो|मान लीजिए|यदि", re.IGNORECASE)
_PRICE_WORD = re.compile(r"dollar|डॉलर|डालर|rupe|रुपए|रुपये|रुपया|₹|\$|bucks|rs\b", re.IGNORECASE)
_HAGGLE = re.compile(r"कम कर|kam kar|kam me|कम में|cheaper|lower|less|discount (do|de|aur)|aur discount|"
                     r"और डिस्काउंट|सस्ता कर|sasta kar|free trial|फ्री ट्रायल|trial", re.IGNORECASE)


# asking for a lower price is a negotiation, not a decline, even when it contains "नहीं" ("कम नहीं कर सकते?")
_NEGOTIATE = re.compile(r"कम (नहीं |नही )?(कर|हो)|और कम|थोड़ा कम|थोडा कम|बोरा सा कम|कुछ कम|kam (nahi |nhi )?kar|thoda kam|\bcheaper\b|lower (the )?price|"
                        r"\breduce\b|bargain|negotiat|discount (do|de|aur|mil)|(some|any|a little|more) discount|can i get (a )?discount|"
                        r"सस्ता कर|sasta kar|और डिस्काउंट|डिस्काउंट (दो|दे|मिल)", re.I)

from ...signals import NEGOTIATE_MORE as _NEGOTIATE_MORE  # noqa: E402
_NEGOTIATE = re.compile(_NEGOTIATE.pattern + "|" + _NEGOTIATE_MORE, re.I)


def _is_counter_offer(text: str, product) -> bool:
    """The customer proposes their own price ("2 डॉलर में चाहिए") or asks for a lower one."""
    from ...text import parse_amount
    if _HAGGLE.search(text):
        return True
    if not _PRICE_WORD.search(text):
        return False
    offered = re.findall(r"\d+", product.offer_price)
    said = re.findall(r"\d+", text)
    words = re.search(r"\b(ek|do|teen|char|paanch|one|two|three|four|five)\b|एक|दो|तीन|चार|पांच|पाँच", text.lower())
    if said:
        return any(n not in offered for n in said)
    return bool(words) or parse_amount(text) is not None


_WEAK_YES = {"ok", "okay", "okk", "k", "haan", "han", "ha", "haa", "hmm", "hm", "ji", "acha", "achha", "accha",
             "ओके", "हां", "हाँ", "हम्म", "जी", "अच्छा", "yeah", "ya", "yes"}


def _weak_yes(text: str) -> bool:
    from ...text import tokens
    toks = tokens(text)
    return 0 < len(toks) <= 2 and all(t in _WEAK_YES for t in toks)


def _similar(a: str, b: str) -> bool:
    from ...text import jaccard
    return bool(a and b) and jaccard(a, b) >= 0.4


def _is_question(text: str) -> bool:
    return bool(_QUESTION.search(text.lower()))


# ---------------------------------------------------------------- compliance (policy floor)
# From voice_platform's locked platform rules: admit to being an AI, honour an opt-out at once,
# never take OTP/PIN/card details on a call. These run before any selling logic.
_OPT_OUT = re.compile(
    r"\b(do ?n.?t|do not|never|stop) call|stop calling|remove my number|take me off|"
    r"\b(call|phone|fon|kol) (mat|na) (karo|karna|kijiye|kare|karein)|(dobara|dubara) (call|phone) (mat|na|nahi)|"
    r"number hata|number delete|"
    r"(कॉल|फोन|फ़ोन) (मत|ना|न) (करो|करना|कीजिए|करें|करे)|(दोबारा|दुबारा|फिर से) (कॉल|फोन|फ़ोन) (मत|ना|न|नहीं)|नंबर हटा|नंबर डिलीट",
    re.IGNORECASE)
_FIRM_NO = re.compile(r"नहीं चाहिए|नही चाहिए|nahi chahiye|nahin chahiye|not interested|don.?t want|do not want|no thanks|"
                      r"नहीं लेना|nahi lena|ज़रूरत नहीं|जरूरत नहीं|zarurat nahi", re.I)
_TIME_WORDS = re.compile(r"\d|tomorrow|tonight|evening|morning|afternoon|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
                         r"kal\b|parso|shaam|subah|dopahar|baje|कल|परसों|शाम|सुबह|दोपहर|बजे|रात", re.I)
# a time to call back ("tomorrow at 5", "शाम को", "in two hours"); "20 seconds" or "one minute" is how long THEY have, not when to call
_CB_TIME = re.compile(r"tomorrow|tonight|today|evening|morning|afternoon|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\bkal\b|parso|shaam|subah|dopahar|"
                      r"baje|बजे|कल|परसों|शाम|सुबह|दोपहर|रात|\b\d{1,2}(:\d\d)?\s?(am|pm)\b|\bat \d|\bin \d+ (minutes?|mins?|hours?|hrs?|days?)|\b\d+\s*(hours?|hrs?|घंटे|ghante|din|days?|दिन)\b|"
                      r"after \d|\d+ (ke )?baad|\d+ बजे के बाद|\d+ के बाद|after (office|lunch|dinner|work)|next week|weekend|सोमवार|मंगलवार|बुधवार|गुरुवार|शुक्रवार|शनिवार|रविवार|somvar|mangalvar|budhvar|guruvar|shukravar|shanivar|ravivar", re.I)
_LATER = re.compile(r"baad|later|kal\b|abhi nahi|बाद|कल |अभी नहीं|अभी मत", re.IGNORECASE)
_ARE_YOU_AI = re.compile(
    r"are you (a |an )?(bot|robot|ai|human|real|machine|computer|recording|person)|is this (a )?(bot|robot|recording|recorded)|"
    r"(aap|tum|ap) (koi )?(robot|bot|ai|machine|computer|insaan|insan|asli)|(robot|bot|machine|insaan|insan) (ho|hai kya|hain kya)|"
    r"(आप|तुम) (कोई )?(रोबोट|बॉट|मशीन|कंप्यूटर|इंसान|असली|एआई|AI)|(रोबोट|बॉट|मशीन|इंसान) (हो|हैं|है क्या)|रिकॉर्डिंग है",
    re.IGNORECASE)
_WHO = re.compile(
    r"who (is this|are you|is calling|is speaking)|which company|calling from where|where are you calling from|"
    r"kaun (bol|baat)|kaun ho|kahan se (bol|call)|kaha se (bol|call)|kis company|"
    r"कौन (बोल|बात)|(आप|तुम) कौन|कौन हो\b|कहाँ से (बोल|कॉल)|कहां से (बोल|कॉल)|किस कंपनी|"
    r"who.?s this|who is it|^\W*(\S+\W+){0,4}(कौन है|kaun hai)\W*(\S+\W*){0,4}$", re.IGNORECASE)
_NUMBER_SOURCE = re.compile(
    r"(where|how) did you get my number|got my number|my number from|number kaha|number kahan|number kaise mila|"
    r"नंबर (कहाँ|कहां|कैसे)|मेरा नंबर (कहाँ|कहां|किसने|कैसे)", re.IGNORECASE)
_SENSITIVE_DIGITS = re.compile(r"\d{6,}|(\d[ -]?){8,}")
_SENSITIVE = re.compile(r"\botp\b|ओटीपी|\bcvv\b|सीवीवी|card number|कार्ड नंबर|card ka number|कार्ड का नंबर|"
                        r"upi pin|यूपीआई पिन|\bpin\b|पिन (नंबर|है)|password|पासवर्ड|\d{6,}|(\d[ -]?){8,}", re.IGNORECASE)

# "answer me" / "you didn't answer": go back to the customer's last question instead of moving on
# (live call 194d15aef7: "Okay, then answer me." was taken as a yes and the sale was closed)
_ANSWER_ME = re.compile(r"answer (me|my question|the question)|(did ?n.?t|did not|never) answer|"
                        r"jawab (do|dijiye|nahi diya)|जवाब (दो|दीजिए|दीजिये|नहीं दिया)|मेरे सवाल|mere sawal", re.IGNORECASE)
# the caller can't follow the language: say the last thing again (now in their language), never a "no"
_LANG_TROUBLE = re.compile(r"(do ?n.?t|do not|did ?n.?t|can.?t|cannot|could ?n.?t) (speak|understand|follow|know (hindi|english))|not able to understand|"
                           r"(hindi|english) (nahi|nahin) (aati|aata|samajh)|(हिंदी|इंग्लिश|अंग्रेज़ी|अंग्रेजी) (नहीं|नही) (आती|आता|समझ)",
                           re.IGNORECASE)
# an explicit "buy" - required before the payment link goes out (or a short clean yes, confirmed once)
_BUY = re.compile(r"\bsend\b|bhej|भेज|le (lunga|lenge|leta|leti|lo)|ले (लूंगा|लूँगा|लेंगे|लेता|लेती|लो)|interested|"
                  r"chahiye|चाहिए|\bbuy\b|kharid|खरीद|subscribe|सब्सक्राइब|sign me up|go ahead|kar do|कर दो|"
                  r"\bdeal\b|\bdone\b|yes please|let.?s do it|let.?s go|go for it|i.?ll take it|sounds good|"
                  r"try (it|kar)|ट्राई कर|कोशिश कर|join (karna|kar|krna)|जॉइन (करना|कर)|join करना|join कर|"
                  r"lena hai|लेना है|le leta|sign up", re.IGNORECASE)
_PRICE_CORE = re.compile(r"price|cost|how much|kitn|कितन|कीमत|दाम|\brate\b|रेट|\bfees?\b|paise kitne|पैसे कितने", re.I)
_PRICE_ASK = re.compile(r"price|cost|how much|kitn|\bfees?\b|\brate\b|कितन|कीमत|दाम|पैसे|paise|paisa|rupe|rupay|रुपए|रुपये|रुपया|dollar|डॉलर|डालर|\$|"
                        r"permanent|हमेशा|hamesha|\bkat(ega|ta|enge)\b|कट|auto.?(renew|charge|debit)|60|साठ|\boff\b|छूट|"
                        r"(after|baad|बाद).{0,12}(3|three|teen|तीन)|(3|three|teen|तीन).{0,20}(after|baad|बाद)", re.I)
_RENEWAL_ASK = re.compile(r"permanent|हमेशा|hamesha|kat(ega|ta|enge)|कट|auto|renew|deduct|after|baad|बाद|always|forever|"
                          r"(20|बीस|twenty).{0,20}(bhi|भी|too|also)", re.I)  # not "why is it 60% off?" (live call: a wall of terms)
_ANNUAL = (r"annual|yearly|सालाना|saalana|salana|(saal|साल) (ka|का|wala|वाला|bhar|भर|ke|के) (plan|प्लान|liye|लिए|kitna|कितना)|"
           r"(per|a|every) year|(plan|प्लान) (saal|साल)")
_RUPEE_ASK = re.compile(r"rupe|rupay|रुपए|रुपये|रुपया|₹|inr|indian|डॉलर क्या|dollar kya|what.?s a dollar", re.I)
_GOODBYE_NOW = re.compile(r"रखते हैं|रखती हूँ|रखता हूँ|rakhte hain|rakhti hoon|rakhta hoon|चलते हैं|chalte hain|रखो (फोन|फ़ोन)|(फोन|फ़ोन) रखो|रख दो|bye|बाय|phone rakh|rakho phone|hang up|गुड बाय|good ?bye", re.I)
_NOT_NOW_LINK = re.compile(r"(link|लिंक) (abhi )?(mat|मत|ना|nahi|नहीं)|(abhi|अभी) (mat|मत) bhej|मत भेजो|mat bhejo|don.?t send|"
                           r"not now|abhi nahi|अभी नहीं|i don.?t click|नहीं भेजना|nahi bhejna|मत भेजना|mat bhejna|कुछ नहीं भेज", re.I)
# a price question with a twist goes to the conversation turn, not the price line ("is there an annual plan?")
_PRICE_DETAIL = re.compile(r"annual|yearly|\byear|saal|साल|student|स्टूडेंट|after|baad|बाद|rupee|rupay|रुपए|रुपये|₹|kab tak|"
                           r"permanent|forever|hamesha|हमेशा|refund|रिफंड|team|share|शेयर|compare|chatgpt|existing|switch|"
                           r"कैसे|kaise|how (do|to|can)|otp|ओटीपी", re.I)
_PAID_WORDS = re.compile(r"\b(paid|pro|premium|plus|subscribe|subscription)\b|पेड|प्रो|सब्सक्", re.I)
_REPEAT_ASK = re.compile(r"(फिर|दोबारा|दुबारा|एक बार और|एक बार फिर)\s*(से )?\s*(बोल|बता|सुना|कह)|(phir|dobara|dubara|ek baar aur|ek baar phir)\s*(se )?\s*(bol|bata|suna|kah)|"
                         r"\bsay (that |it )?again|\brepeat\b|once more|kya bola|क्या बोला|क्या कहा|सुनाई नहीं|sunai nahi|(धीरे|dheere|dhire|slowly|slow)\s*(से )?\s*(बोल|bol|speak|say|बता|bata)|"
                         r"(speak|talk) (more )?slowly|ठीक से सुनाई|can.?t hear", re.I)
_WEATHER_Q = re.compile(r"weather|temperature|mausam|मौसम|तापमान|baarish|बारिश|garmi|गर्मी|thand|ठंड", re.I)
_NOT_ARRIVED = re.compile(r"(nahi|नहीं|not|didn.?t).{0,20}(aaya|आया|received|mila|मिला|come|got|arrive)|where is (the |my )?link|link kahan|लिंक कहाँ", re.I)
_CLEAN_YES = _WEAK_YES | {"please", "sure", "bilkul", "बिल्कुल", "zaroor", "जरूर", "ज़रूर", "theek", "ठीक", "hai",
                          "है", "yes", "yup", "haanji", "हांजी"}


def _strip_lead(answer: str) -> str:
    """'Fair question. It's ...' -> 'It's ...' (when repeating, the pleasantry sounds robotic)."""
    return re.sub(r"^(fair question|good question|अच्छा सवाल है)[.!।,]\s*", "", answer.strip(), flags=re.I)


def _clean_yes(text: str) -> bool:
    from ...text import tokens
    toks = tokens(text)
    return 0 < len(toks) <= 3 and all(t in _CLEAN_YES for t in toks)


_WHAT_LINK = re.compile(r"(what|which) link|link (kya|kaun|kaisa)|kaun sa link|kaunsa link|कौन सा लिंक|कौनसा लिंक|क्या लिंक|लिंक क्या|कैसा लिंक", re.I)
_PURPOSE = re.compile(r"what do you want|why (are|did) you call|what is this (call|about)|what.?s this about|kyu call|kyon call|"
                      r"kya chahiye aapko|kis liye call|क्यों कॉल|किस लिए कॉल|क्या चाहिए आपको|किसलिए", re.I)
_LEAD_YES = re.compile(r"^\W*(haan|han|haa|ha|yes|yeah|ji|haanji|हाँ|हां|जी|हांजी|ok|okay|theek hai|ठीक है)\b", re.I)
_NOT_AGREED = re.compile(r"कहा ही नहीं|नहीं कहा|हाँ नहीं|हां नहीं|didn.?t (say|agree|ask)|never (said|agreed|asked)|"
                         r"did not (say|agree|ask)|maine (kab|nahi) (kaha|bola)|मैंने (कब|नहीं) (कहा|बोला)|lene ko (nahi|kab)|"
                         r"लेने को (नहीं|कब)|(पूछना|पूछ के|पूछकर|poochna|puchna|pooch ke|puch ke) (पड़ेगा|padega|बताएँगे|bataenge)|"
                         r"ask my (son|daughter|wife|husband|family|grandson)|talk to my (son|daughter|wife|husband|family)", re.I)
_CANCEL = re.compile(r"cancel|don.?t send|do not send|mat bhej|मत भेज|nahi chahiye|नहीं चाहिए|changed my mind|rehne do|रहने दो", re.I)
_SALE_WORDS = re.compile(r"pay|payment|पेमेंट|पेवेंट|पेमेन्ट|भुगतान|buy|खरीद|kharid|link|लिंक|price|कीमत|offer|ऑफर|"
                         r"discount|डिस्काउंट|subscri|सब्सक्रा|upi|यूपीआई|card|कार्ड|refund|रिफंड|privacy|प्राइवेसी|\bdata\b|डेटा|डाटा|"
                         r"account|अकाउंट|cancel|कैंसल|annual|yearly|सालाना|student|स्टूडेंट|band kar|बंद कर|report|रिपोर्ट|coding|code|कोडिंग|"
                         r"hindi|हिंदी|chatgpt|share|शेयर|limit|model|मॉडल|invoice|bill|बिल|gst|जीएसटी|tally|टैली", re.I)
_HELLO = re.compile(r"^\W*(hello|hallo|helo|hi|हेलो|हैलो|हलो|हल्लो)(\W+(hello|हेलो|हैलो|ji|जी|sir|सर))*\W*$", re.I)
ANYTHING_ELSE = ("Is there anything else you'd like to know?", "और कुछ जानना चाहेंगे?", "Aur kuch jaanna chahenge?")


# the question to come back to after a side answer, by FSM state
_RESUME = {
    "intro": ("Is this a good time for two minutes?", "क्या अभी दो मिनट बात हो सकती है?", "Kya abhi do minute baat ho sakti hai?"),
    "pitch": ("Shall I tell you the price?", "कीमत बताऊँ?", "Price bataun?"),
    "close": ("Shall I send you the link?", "क्या मैं लिंक भेज दूँ?", "Kya main link bhej doon?"),
    "reoffer": ("Would you like to try it?", "क्या आप ट्राई करना चाहेंगे?", "Try karna chahenge?"),
    "confirm": ("Shall I send the link?", "लिंक भेज दूँ?", "Link bhej doon?"),
}


def _tpl(en: str, hi: str, hinglish: str) -> dict[str, str]:
    return {"en": en, "hi": hi, "hinglish": hinglish}


class ProductSalesDomain:
    """Policy + conversation.

    Six persona reviews (2026-11-14) all found the same thing: the understanding model read the
    customer correctly ("asks what happens after 3 months", "asks about data privacy"), and a
    keyword state machine then overrode it with the price line, a re-offer, or a hang-up. So the
    domain is now split in two:

      POLICY (deterministic, decides): safety and compliance lines, consent before the link
        (explicit yes + one confirmation that states the full terms), at most one re-offer after a
        clear no, ending the call, callbacks, opt-out. Prices only ever come from locked slots.
      CONVERSATION (the LLM, phrases): every other turn. It must answer what the customer just
        said first, from the product facts / approved answers / world lookup, then make at most
        one small step forward depending on where the sale stands (from the call's context
        ledger), and never push when the customer is annoyed, sad, or said "not now".
    """

    name = "product_sales"

    intent_guide = {
        "affirm": "clearly agrees to BUY or to receive the link (not just 'ok', not agreeing to listen, not a question)",
        "deny": "clearly declines the offer or says they don't want it",
        "tell_more": "agrees to listen / says go on / asks what this is about",
        "price": "asks only the price or cost",
        "benefits": "asks what the product does or its features",
        "callback": "busy now, asks to be called later",
        "human_agent": "explicitly asks to talk to a real person",
        "clarify": "didn't hear or understand the words, asks to repeat, or says their question wasn't answered",
        "goodbye": "ending the call, bye, put the phone down",
        "slow_down": "says the agent is going too fast / rushing / pushy / not listening / didn't even ask if they are interested",
        "personal": "talks about their own life or feelings (lonely, family, health, mood), not about the offer",
        "small_talk": "chit-chat or a question unrelated to the product (weather, time, news, facts, recipes)",
    }
    confident_intents = {"callback", "human_agent"}
    # foreign intent label -> this domain's label (added to acie.signals.DEFAULT_ALIASES); unmapped labels are "unknown"
    intent_aliases: dict[str, str] = {}

    def __init__(self, product: Product, customer_name: str = "", world=None, customer_age: int | None = None,
                 playbook=None, business=None, customer: Customer | dict | None = None, turn_model=None,
                 now_fn=None, dnc_lookup=None):
        self.business = business if business is not None else MockBusinessAPI()  # default mock = today's behaviour
        cust = Customer.model_validate(customer) if isinstance(customer, dict) else (customer or Customer())
        if customer_age is not None and cust.age is None:
            cust = cust.model_copy(update={"age": customer_age})
        if customer_name and not cust.name:
            cust = cust.model_copy(update={"name": customer_name})
        self.customer = cust
        self._now_fn = now_fn
        self._dnc_lookup = dnc_lookup
        self._tasks: set = set()
        self.p = product
        self.about = (f"{product.product_name} ({product.one_liner}) from {product.seller_name}, offered at "
                      f"{product.offer_price} instead of {product.original_price} {product.offer_validity}")
        self.customer_name = customer_name.strip()
        self.customer_age = customer_age
        if "₹" in product.offer_price or "rs" in product.offer_price.lower():
            # already priced in rupees: dollar-conversion notes (defaults written for the Claude demo) don't apply
            product = product.model_copy(update={"price_note": "", "rupee_terms": {}})
        if "$" not in product.offer_price + product.original_price:
            product = product.model_copy(update={
                "renewal_note": {k: v for k, v in product.renewal_note.items() if "$" not in v},
                "annual_note": {k: v for k, v in product.annual_note.items() if "$" not in v}})
        if product.product_name not in product.pitch_hinglish and product.product_name in product.pitch_hi:
            # the Hinglish pitch is still the Claude default while the Hindi one was written for this product
            product = product.model_copy(update={"pitch_hinglish": ""})
        self.p = product
        if playbook is None and product.playbook_tree:
            from ... import playbook as pbm
            try:
                playbook = pbm.load_tree(product.playbook_tree)
            except Exception:
                playbook = None  # a broken tree file never stops a call
        self.playbook = playbook if product.use_playbook else None  # acie.playbook.Playbook (pre-call decision tree)
        self.world = world  # acie.world.WorldKnowledge: weather / clock / encyclopedia for off-topic questions
        self.interim = None  # set by the pipeline: speak a short line now ("one second, let me check")
        self.classifier = KeywordIntentClassifier(KEYWORDS, gated=GATED, priority=PRIORITY, masks=MASKS,
                                                  weak=["goodbye"])
        model = turn_model if turn_model is not None else load_turn_model()  # ACIE_TURN_MODEL; None = keywords only
        if model is not None:
            self.classifier = HybridIntentClassifier(self.classifier, model, known=set(KEYWORDS) | set(self.intent_guide))
        self._faq = [(f, [k[3:] if k.startswith("re:") else re.escape(k.lower())
                                       for k in f.keywords if k.strip()]) for f in product.faqs if f.answer]

    # ------------------------------------------------------------------ facts

    def _base(self) -> dict[str, Fact]:
        return {"product": Fact(value=self.p.product_name, description="product name", must_mention=False)}

    def _offer(self) -> dict[str, Fact]:
        return {"offer_price": Fact(value=self.p.offer_price, description="special offer price"),
                "original_price": Fact(value=self.p.original_price, description="normal price"),
                "discount": Fact(value=self.p.discount, description="discount"),
                "validity": Fact(value=self.p.offer_validity, description="how long the offer price applies",
                                 must_mention=False, verbatim=False)}

    def _terms(self, lang: str) -> str:
        """The complete offer in one locked phrase: price, how long, and what it costs after."""
        p = self.p
        if p.terms.get(lang):  # operator-written phrase (FitZone: "₹2,999 for 3 months if you join this week, then ...")
            return p.terms[lang]
        months = p.offer_validity
        return {
            "hi": f"{months.replace('for the first', 'पहले').replace('months', 'महीने')} {p.offer_price}, उसके बाद {p.original_price}",
            "hinglish": f"{months.replace('for the first', 'pehle').replace('months', 'mahine')} {p.offer_price}, uske baad {p.original_price}",
        }.get(lang, f"{p.offer_price} {months}, then {p.original_price}")

    def _playbook_turn(self, text: str, intent: str, d: dict, u, frustration: float) -> tuple[str, str, str] | None:
        """Pick the playbook node the conversation is heading into and hand its unused talking points to the
        paraphraser. Returns (context block, stage instruction) or None."""
        if self.playbook is None or d.get("existing_user") == "1":
            return None
        node, score = self.playbook.pick(text, u, intent)
        if node is None:
            return None
        used = set(json.loads(d.get("pb_used", "[]")))
        lang = d.get("_lang", "hi")
        pts = node.points_hi if lang == "hi" and len(node.points_hi) == len(node.points) else node.points
        ask = node.ask_hi if lang == "hi" and node.ask_hi else node.ask
        fresh = [(i, pt) for i, pt in enumerate(pts) if f"{node.id}:{i}" not in used] or list(enumerate(pts))
        picks = fresh[:3]
        used.update(f"{node.id}:{i}" for i, _ in picks[:2])
        d["pb_used"] = json.dumps(sorted(used))
        d["pb_node"] = node.id
        d["pb_node_turn"], d["pb_title_turn"] = node.id, node.title
        block = (f"SITUATION (from the pre-call playbook): {node.title}\nTALKING POINTS for this situation:\n"
                 + "\n".join(f"- {pt}" for _, pt in picks))
        if frustration >= 0.4:
            stage = ("They sound annoyed: apologise in a few words, answer directly using at most ONE talking point, "
                     "and do not add a selling question.")
        else:
            stage = (f"Goal of this reply: {node.goal or node.title}. After answering what they said, use one or two of the "
                     f"TALKING POINTS in your own words, made specific to this customer. End with a short question like: "
                     f"\"{ask}\" (in the reply language).") if ask else ""
        fallback = (picks[0][1] + (" " + ask if ask and frustration < 0.4 else "")) if picks else ""
        return block, stage, fallback

    def _other_topic(self, text: str, u, intent: str) -> bool:
        if self.playbook is None:
            return False
        node, _ = self.playbook.pick(text, u, intent)
        return node is not None and "price" not in node.id

    def _audience(self) -> str:
        """How to talk to this customer, from the operator-given age (live call: a 21-year-old was spoken to
        like an elderly person). Unknown age: a normal adult, never assume old."""
        a = self.customer_age
        if a is None:
            return ("Don't assume their age: talk like you would to a normal adult, don't over-explain basics unless they "
                    "ask, and don't use 'माँजी', 'बेटा' or 'अंकल'. ")
        if a < 30:
            return (f"The customer is about {a} years old: talk like a friendly peer, casual and quick, with examples "
                    "from college, coding, work or content; never over-explain basics; no 'जी/माँजी/सर' formality. ")
        if a < 55:
            return (f"The customer is about {a} years old: talk like a professional to a working adult, clear and "
                    "direct, with examples from work and family. ")
        return (f"The customer is about {a} years old: be respectful and patient (जी / सर / माँजी, never 'बेटा'), "
                "use simple everyday words, explain new terms briefly, and say amounts in rupees too. ")

    def _price_lead(self, text: str, lang: str) -> str:
        """The exact price answer, spoken word for word before the LLM's reply."""
        p = self.p
        terms = self._terms(lang)
        rupees = p.rupee_terms.get(lang, "") if (lang != "en" or _RUPEE_ASK.search(text)) else ""
        if _RUPEE_ASK.search(text) and rupees and lang != "en":  # asked in rupees: say rupees first
            r = re.sub(r"^(यानी|yaani)\s*", "", rupees)
            out = {"hi": f"रुपये में: पहले 3 महीने {r}। डॉलर में ये {terms} है।",
                   "hinglish": f"Rupaye mein: pehle 3 mahine {r}. Dollar mein ye {terms} hai."}[lang]
            if _RENEWAL_ASK.search(text) and p.renewal_note.get(lang):
                out += " " + p.renewal_note[lang]
            return out
        if lang == "hi":
            out = f"ये {terms} है" + (f", {rupees}" if rupees else "") + "।"
        elif lang == "hinglish":
            out = f"Ye {terms} hai" + (f", {rupees}" if rupees else "") + "."
        else:
            out = f"It's {terms}" + (f", {rupees}" if rupees else "") + "."
        if re.search(r"डॉलर क्या|dollar kya|what.?s a dollar|डालर क्या", text, re.I) and "$" in terms:
            out = {"hi": "डॉलर अमेरिका का पैसा है। ", "hinglish": "Dollar America ka paisa hai. "}.get(lang, "") + out
        if _RENEWAL_ASK.search(text) and p.renewal_note.get(lang):
            out += " " + p.renewal_note[lang]
        return out

    def _benefits(self, n: int = 3) -> dict[str, Fact]:
        return {f"benefit_{i + 1}": Fact(value=b, description=f"key benefit {i + 1}", verbatim=False, must_mention=False)
                for i, b in enumerate(self.p.benefits[:n])}

    # ------------------------------------------------------------------ signals

    @staticmethod
    def _nlu(d: dict):
        """The learned providers' fused signals for this turn (acie.signals.TurnSignals), or None when only the
        rules heard it. The decision layer never sees a provider's own object."""
        from ...signals import signals_from_dialog
        return signals_from_dialog(d)

    def _asks(self, text: str, d: dict) -> bool:
        u = self._nlu(d)
        return _is_question(text) or (u is not None and u.question_topic not in ("none",))

    def _general(self, text: str, d: dict) -> bool:
        u = self._nlu(d)
        if u is not None:
            return (u.question_topic == "general" or u.heard("small_talk")) and not _SALE_WORDS.search(text.lower())
        return _is_question(text) and not self._on_topic(text)

    def _on_topic(self, text: str) -> bool:
        low = text.lower()
        if _SALE_WORDS.search(low):
            return True
        words = [w.lower() for w in self.p.topic_words] + [self.p.product_name.lower(), self.p.seller_name.lower()]
        return any(re.search(r"(?<![\wऀ-ॿ])" + re.escape(w) + r"(?![\wऀ-ॿ])", low) for w in words if w)

    def _faq_match(self, text: str) -> FAQ | None:
        low = text.lower()
        for f, kws in self._faq:
            if any(re.search(k, low) for k in kws):
                return f
        return None

    def _yes(self, text: str, intent: str, d: dict) -> str | None:
        """'explicit' (buy words or a short clean yes), 'model' (the understanding model is sure), or None.
        A question is never consent ("UPI se ho jayega?"), nor a filler while asking something else."""
        if _HYPOTHETICAL.search(text):
            return None
        if _is_question(text) and not honest.SEND_REQ.search(text):
            return None  # a question is never consent (the model's question_topic alone is too eager)
        if honest.FREE_WANT.search(text) and not _PAID_WORDS.search(text) and not honest.SEND_REQ.search(text):
            return None  # "I'll try the free one": that is the free plan, never a yes to the payment link
        if d.get("no_link") == "1" and not honest.SEND_REQ.search(text):
            return None  # they said no links (can't use them / not now): only an explicit "send it" brings them back
        if _BUY.search(text) and intent != "deny" and not _CANCEL.search(text):
            return "explicit"
        if _clean_yes(text) and intent in ("affirm", "tell_more", "unknown", "goodbye"):
            return "explicit"
        u = self._nlu(d)
        if u is not None and u.heard("affirm") and u.buying_interest >= 0.8 and not _weak_yes(text):
            return "model"
        return None

    def _no(self, text: str, intent: str, d: dict) -> bool:
        if _FIRM_NO.search(text) and not _PRICE_ASK.search(text) and self._faq_match(text) is None:
            return True  # "बोला ना नहीं चाहिए! समझ नहीं आता?" is a refusal, even with a question mark
        if self._asks(text, d):
            return False  # "is it fake? will it get costlier later?" is a question, not a no
        u = self._nlu(d)
        if u is not None and u.heard("personal", "small_talk"):
            return False  # loneliness / weather chat is not a refusal
        return intent == "deny"

    # ------------------------------------------------------------------ entry

    async def plan(self, intent: str, turn: TurnInput, session: SessionState) -> TurnPlan:
        d = session.dialog
        d.pop("pb_node_turn", None)
        d.pop("pb_title_turn", None)
        plan = await self._guarded_plan(intent, turn, session)
        plan = await self._answer_before_goodbye(plan, intent, turn, session)
        u = self._nlu(d)
        if u is not None and u.objection != "none":
            d.setdefault("_objection", u.objection)
        if plan.llm_allowed:
            male = self.p.agent_gender != "female"
            plan.persona = ((self.p.tone + ". ") if self.p.tone else "") + self._audience() + (
                f"You ({self.p.agent_name}) are {'a man' if male else 'a woman'}: in Hindi always speak about yourself in the "
                f"{'masculine' if male else 'feminine'} form (e.g. "
                f"{'चाहता हूँ, कर सकता हूँ, बताऊँगा, माफ़ी चाहता हूँ' if male else 'चाहती हूँ, कर सकती हूँ, बताऊँगी, माफ़ी चाहती हूँ'}). ")
            plan.rule_packs = ["sales_compliance", "claims", "agent_male" if male else "agent_female"]
        d["last_intent"] = plan.intent
        d["last_customer"] = turn.text
        self._outcome(plan, session)
        return plan

    # ------------------------------------------------------------------ business API, constraints, next best action

    async def _api(self, coro, default=None, budget: float = 1.5):
        """A business-API call that can never stall a turn longer than `budget` seconds or raise."""
        try:
            return await asyncio.wait_for(coro, budget)
        except Exception:
            return default

    def _constraint_ctx(self, d: dict, session: SessionState) -> cons.ConstraintContext:
        c = self.customer
        sub = c.existing_subscriber
        if d.get("_elig_reason") == "existing_subscriber":
            sub = True
        return cons.ConstraintContext(product=self.p, phone=c.id or session.customer_id, age=c.age, region=c.region,
                                      existing_subscriber=sub, reoffers=int(d.get("reoffers", "0")),
                                      now=self._now_fn() if self._now_fn else None, dnc_lookup=self._dnc_lookup)

    async def _guarded_plan(self, intent: str, turn: TurnInput, session: SessionState) -> TurnPlan:
        d = session.dialog
        ctx = getattr(session, "context", None)
        rules = self.p.constraints
        if intent == "greet":
            await self._opening_lookups(d)
        if (rules or d.get("_elig_blocked")) and not (ctx is not None and ctx.outcome):
            if d.get("_elig_blocked"):
                v = cons.Violation("eligibility", d["_elig_blocked"])
            else:
                v = cons.evaluate(rules, self._constraint_ctx(d, session), types=cons.PRE)
            if v is not None:
                d.pop("pending", None)
                return cons.violation_plan(v)
        if nba.enabled() and intent != "greet":
            self._nba(d, ctx)
        plan = await self._plan(intent, turn, session)
        if rules and plan.intent == "reoffer":
            v = cons.evaluate(rules, self._constraint_ctx(d, session), types=("max_reoffers",))
            if v is not None:  # the rule forbids another offer: respect the no
                d.pop("pending", None)
                d["reoffers"] = str(int(d.get("reoffers", "1")) - 1)
                return self._declined_plan()
        if plan.intent == "link_sent":
            await self._payment_link_fact(plan, d, session)
        return plan

    def _declined_plan(self) -> TurnPlan:
        return TurnPlan(intent="declined", llm_allowed=False, decision="customer declined", end_call=True,
                        objective="Thank them and say goodbye; do not pitch again",
                        fallback=_tpl("No problem at all. Thanks for your time, have a good day!",
                                      "कोई बात नहीं। आपके समय के लिए धन्यवाद, आपका दिन शुभ हो!",
                                      "Koi baat nahi. Aapke time ke liye thank you, aapka din accha ho!"))

    async def _opening_lookups(self, d: dict) -> None:
        """Before the first line: is this person eligible, and what is the verified offer? Facts only."""
        elig, quote = await asyncio.gather(self._api(self.business.eligibility(self.customer)),
                                           self._api(self.business.quote(self.p, self.customer)))
        if elig is not None and not elig.eligible:
            d["_elig_reason"] = elig.reason
            if elig.verified:  # an unverified (mock/fallback) "no" never ends a call by itself
                d["_elig_blocked"] = elig.reason or "not eligible"
        if quote is not None and quote.verified:
            upd = {k: v for k, v in (("offer_price", quote.price), ("original_price", quote.original_price),
                                     ("offer_validity", quote.validity)) if v}
            if quote.terms:
                upd["terms"] = dict(quote.terms)
            self.p = self.p.model_copy(update=upd)
            d["quote_verified"] = "1"

    async def _payment_link_fact(self, plan: TurnPlan, d: dict, session: SessionState) -> None:
        link = await self._api(self.business.create_payment_link(self.p, self.customer))
        url = link.url if link is not None and link.url else self.p.payment_link
        d["payment_link_url"] = url  # voice/call.py sends this one by SMS
        d["payment_link_verified"] = "1" if link is not None and link.verified else "0"
        if url:
            plan.facts["payment_link"] = Fact(value=url, description="the payment link (sent by SMS, never spoken)",
                                              must_mention=False, hidden=True)

    def _nba(self, d: dict, ctx) -> None:
        u = self._nlu(d)
        stage = d.get("pending") or "intro"
        action, scores = nba.next_best_action(stage, u.buying_interest if u is not None else 0.0,
                                              list(ctx.objections) if ctx is not None else [], ctx,
                                              benefit_turns=int(d.get("benefit_turns", "0")))
        d["nba"] = action
        d["nba_scores"] = json.dumps(scores)

    def _outcome(self, plan: TurnPlan, session: SessionState) -> None:
        """Every terminal path leaves one consistent outcome label (the same one the ledger folds into
        session.context.outcome / decision / promised) and tells the business API once."""
        d = session.dialog
        label = _OUTCOMES.get(plan.intent)
        if label is None:
            if plan.intent in ("link_cancelled", "confirm_close"):
                d["outcome"] = ""
                d.pop("_outcome_sent", None)
            return
        d["outcome"] = label
        if d.get("_outcome_sent") == label:
            return
        d["_outcome_sent"] = label
        try:
            t = asyncio.get_running_loop().create_task(
                self._api(self.business.record_outcome(session.session_id, label)))
        except RuntimeError:
            return
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    async def _plan(self, intent: str, turn: TurnInput, session: SessionState) -> TurnPlan:
        d, p = session.dialog, self.p
        text = honest.NOT_SENDING.sub(" ", turn.text)  # "वो ऑनलाइन भेज देगा" (he will send the money) is not "send me the link"
        ctx = getattr(session, "context", None)
        lang = d.get("_lang", "hi")
        pending = d.get("pending")

        if intent == "greet":
            d["pending"] = "intro"
            name = f" {self.customer_name}" if self.customer_name else ""
            facts = {**self._base(), "agent": Fact(value=p.agent_name, description="your name"),
                     "seller": Fact(value=p.seller_name, description="your company")}
            return TurnPlan(
                intent="opening", facts=facts, ask="do you have a minute?", llm_allowed=False,
                objective="Say who you are (an AI assistant calling for the company), that the call is recorded, why you're calling, and ask permission to continue",
                fallback=_tpl(
                    f"Hi{name}, I'm {{agent}}, an AI assistant calling for {{seller}}. This call is recorded. I'm calling about an offer on {{product}}. Do you have two minutes?",
                    f"नमस्ते{name}, मैं {{seller}} की तरफ़ से {{agent}}, एक AI असिस्टेंट। ये कॉल रिकॉर्ड हो रही है। {{product}} के एक ऑफर के लिए कॉल किया है, दो मिनट बात कर सकते हैं?",
                    f"Namaste{name}, main {{seller}} ki taraf se {{agent}}, ek AI assistant. Ye call record ho rahi hai. {{product}} ke ek offer ke liye call kiya hai, do minute baat kar sakte hain?"))

        # ---------------- policy: compliance (word for word)
        if (side := self._compliance(text, intent, d, ctx)) is not None:
            return side

        # ---------------- policy: the customer already decided
        if ctx is not None and (after := self._after_decision(intent, text, d, ctx, lang)) is not None:
            return after

        # ---------------- the offer doesn't apply to them (already a subscriber): answer honestly, never sell
        f0 = self._faq_match(text)
        if f0 is not None and f0.not_eligible:
            d["existing_user"] = "1"
            d.pop("pending", None)
            return await self._converse(text, intent, d, ctx, lang, float(d.get("_frustration", "0") or 0))

        # ---------------- the caller can't use links / wants the free plan / is having a hard time / asks for something we can't send
        if honest.NO_LINKS.search(text):
            d["no_link"] = "1"  # "I can't use links": never offer the link again unless they ask for it
        if (sad := self._sad_step(text, intent, d, lang)) is not None:
            return sad
        if (info := self._info_request_step(text, d, ctx, lang)) is not None:
            return info
        if (rep_ := self._repeat_request_step(text, d, ctx, lang)) is not None:
            return rep_

        # ---------------- "yes" to a negotiated price: lock it in the terms, then the usual confirmation (terms + link) with the agreed price
        if d.get("pending") == "neg_close" and not _is_question(text) and (self._yes(text, intent, d) or _clean_yes(text) or intent == "affirm") \
                and not _FIRM_NO.search(text):
            self._apply_neg_offer(d)
            d["pending"] = "confirm"
            d["requested_link"] = "0"
            return self._confirm_plan(lang, d)

        # ---------------- a counter-offer ("2,500 mein de do") is price talk, never a yes to the link: the negotiation flow answers it
        if negotiation.detect(text) == "counter_offer" and (neg0 := self._negotiate_step(text, d, ctx, lang)) is not None:
            return neg0

        # ---------------- policy: consent and closing
        u_ = self._nlu(d)
        if u_ is not None and u_.garbled:
            d["unclear"] = str(int(d.get("unclear", "0")) + 1)  # a bad line / noise: never close without a read-back
        after_confirm = pending == "confirm" or d.get("last_intent") in ("confirm_close", "confirm_buy")
        if u_ is not None and u_.garbled \
                and u_.heard("unknown", "clarify", "tell_more", "affirm", "small_talk") and not _BUY.search(text):
            # a noisy line: don't build an answer (or a sale) on words the caller probably never said
            return TurnPlan(intent="clarify", llm_allowed=False, objective="Ask them to repeat",
                            fallback=_tpl("Sorry, the line isn't clear. Could you say that again?",
                                          "माफ़ कीजिए, आवाज़ साफ़ नहीं आई। एक बार फिर बोलेंगे?",
                                          "Sorry, awaaz saaf nahi aayi. Ek baar phir bolenge?"))
        if after_confirm and ctx is not None and (_LEAD_YES.match(text) or _BUY.search(text)) and _is_question(text) \
                and not _NOT_NOW_LINK.search(text) and not _HYPOTHETICAL.search(text) \
                and not re.search(r"mat bhej|मत भेज|nahi chahiye|नहीं चाहिए|don.?t send|rehne do|रहने दो|changed my mind", text, re.I):
            return self._link_plan(d, lang, question=text)  # "haan isi number pe. aur cancel kaise karte hain?"
        yes = self._yes(text, intent, d)
        if after_confirm and not yes and ctx is not None and not _CANCEL.search(text) and not _FIRM_NO.search(text) \
                and (_LEAD_YES.match(text) or re.search(r"this number|isi number|इसी नंबर|same number", text, re.I)
                     or (d.get("requested_link") == "1" and _GOODBYE_NOW.search(text))):
            yes = "explicit"  # "Yes, this number." / they asked for the link, then "Okay. Bye."
        if yes and ctx is not None and "offer_price" not in ctx.quoted and pending in ("intro", "pitch") \
                and (not _BUY.search(text) or re.search(r"go ahead|tell|bata|bol", text, re.I)) and not re.search(r"send|bhej|भेज|\bbuy\b|kharid|खरीद|subscribe|sign me up|i.?ll take|le (lunga|lenge|leti|leta)|"
                                  r"ले (लूँगा|लूंगा|लेंगे|लेती|लेता)", text, re.I):
            yes = None  # "yes, go ahead" to "shall I tell you the price?" means tell me, not buy
        if yes and ctx is not None and (ctx.has("payment_link") or d.get("sad_hold") == "1"):
            yes = None  # the link already went out (never confirm or offer it again), or they are in a hard time: nothing to close
        if yes and ctx is not None and not _NOT_NOW_LINK.search(text):
            strong = bool(_BUY.search(text))
            asked_link = d.get("link_asked_last") == "1" or d.get("last_intent") in ("price", "reoffer")
            # the price line already stated the full terms, so "yes, send it" right after it is consent
            if after_confirm or (strong and d.get("last_intent") == "price" and d.get("unclear", "0") == "0") \
                    or (d.get("terms_confirmed") == "1" and (strong or asked_link)):
                u0 = self._nlu(d)
                # "हाँ हाँ" from someone who sounds unsure (the elderly persona got a link she never wanted)
                if not strong and u0 is not None and u0.buying_interest < 0.6 and d.get("last_intent") != "confirm_buy":
                    d["pending"] = "confirm"
                    return TurnPlan(intent="confirm_buy", llm_allowed=False, objective="Make sure they really want to buy",
                                    fallback=_tpl("Just to be sure: would you like to buy it? It's completely fine to talk to your family first.",
                                                  "बस पक्का कर लूँ: क्या आप सच में इसे लेना चाहते हैं? पहले घर में किसी से बात करनी हो तो कोई बात नहीं।",
                                                  "Bas pakka kar loon: kya aap sach mein ise lena chahte hain? Pehle ghar mein kisi se baat karni ho to koi baat nahi."))
                return self._link_plan(d, lang, question=text if honest.is_question(text) else "")
            just_asked = d.get("last_intent") in ("price", "reoffer") or d.get("link_asked_last") == "1"
            if not strong and not just_asked and (_weak_yes(text) or _clean_yes(text)):
                pass  # "हाँ हाँ" after "sorry, say again?" is a filler, not a yes to buying
            elif pending in ("close", "reoffer", "pitch", "intro", None, "discovery") and (yes == "explicit" or pending in ("close", "reoffer", "discovery")):
                if pending in ("close", "reoffer") or "offer_price" in ctx.quoted or yes == "explicit":
                    if pending in ("intro", "pitch") and "offer_price" not in ctx.quoted and _weak_yes(text):
                        pass  # "हाँ" to "shall I tell you?" = go on, handled below
                    else:
                        d["pending"] = "confirm"
                        d["requested_link"] = "1" if strong else "0"
                        return self._confirm_plan(lang, d)

        timed = bool(_CB_TIME.search(text))
        callback_reply = d.get("pending") == "callback_time" and timed and not _FIRM_NO.search(text)  # their answer to "when should I call?"
        if callback_reply or (intent == "callback" and not _FIRM_NO.search(text) and (not self._asks(text, d) or (re.search(r"call|कॉल|phone|फ़ोन|फोन", text, re.I)
                                                             and (_LATER.search(text) or _TIME_WORDS.search(text))))):
            d.pop("pending", None)
            if timed or not d.get("callback_note"):
                d["callback_note"] = text[:120]
            if timed:
                d["callback_time"] = "1"
            if not timed and d.get("callback_asked") != "1" and d.get("callback_time") != "1":
                # "I'm in a meeting": not a goodbye. Ask when to call back (the answer is recorded), offer the link, then close
                d["callback_asked"] = "1"
                d["pending"] = "callback_time"
                return TurnPlan(intent="callback_time", llm_allowed=False, decision="ask when to call back",
                                objective="Say of course; ask what time to call back; offer to send the link by SMS instead; no pitch",
                                fallback=(_tpl("Of course. What time should I call you back?", "ज़रूर। आपको किस समय वापस कॉल करूँ?",
                                               "Zaroor. Aapko kis time wapas call karoon?") if d.get("no_link") == "1" else
                                          _tpl("Of course, I'll keep it short. What time should I call you back? Or, if it's easier, I can send the link by SMS.",
                                               "ज़रूर, मैं छोटा रखूँगा। आपको किस समय वापस कॉल करूँ? या आसान हो तो SMS से लिंक भेज दूँ।",
                                               "Zaroor, main chhota rakhunga. Aapko kis time wapas call karoon? Ya aasaan ho to SMS se link bhej doon.")))
            when = d["callback_note"].strip(" .!?।") if timed and len(d["callback_note"].split()) <= 6 else ""  # the time they said, repeated back
            return TurnPlan(intent="callback", end_call=True, decision="schedule callback",
                            context=f"Customer said: {text}\nWhen they asked to be called: {d['callback_note']}",
                            objective="Say of course; if they mentioned a time or day, repeat it to confirm you'll call then "
                                      "(otherwise say you'll call later); then a short goodbye",
                            fallback=(_tpl(f"Of course, I'll call you back then: {when}. Have a good day!",
                                           f"ज़रूर, मैं तभी वापस कॉल करूँगा: {when}। आपका दिन शुभ हो!",
                                           f"Zaroor, main tabhi wapas call karunga: {when}. Aapka din accha ho!") if when else
                                      _tpl("Of course, I'll call you back later. Have a good day!",
                                           "ज़रूर, मैं बाद में कॉल करूँगा। आपका दिन शुभ हो!",
                                           "Zaroor, main baad mein call karunga. Aapka din accha ho!")))

        u = self._nlu(d)
        if d.get("_language_request") and not _OPT_OUT.search(text) and len(text.split()) <= 12 \
                and not _PRICE_ASK.search(text):
            # "I feel like talking to someone who can speak Hindi" = switch language, not hand off (live call)
            lr = d["_language_request"]
            q = self.p.discovery_question.get(lr, "") if ctx is not None and "offer_price" in ctx.quoted else ""
            return TurnPlan(intent="language_switch", llm_allowed=False, objective="Switch language and carry on",
                            fallback=_tpl("Sure, let's talk in English. " + (q or "Please go on."),
                                          "ज़रूर, हिंदी में बात करते हैं। " + (q or "बताइए।"),
                                          "Zaroor, Hinglish mein baat karte hain. " + (q or "Bataiye.")))
        if intent == "human_agent" and ((u is not None and u.heard("human_agent")) or re.search(
                r"human|person|insaan|insan|इंसान|व्यक्ति|aadmi|आदमी|manager|मैनेजर|supervisor|real agent", text, re.I)):
            return TurnPlan(intent="human_agent", llm_allowed=False, decision="handoff_to_human", end_call=True,
                            objective="Say a team member will call them back",
                            fallback=_tpl("Sure, a person from our team will call you back. Thank you!",
                                          "ज़रूर, हमारी टीम से कोई व्यक्ति आपको वापस कॉल करेगा। धन्यवाद!",
                                          "Zaroor, hamari team se koi aapko call karega. Thank you!"))

        if _GOODBYE_NOW.search(text) and (not self._asks(text, d) or _FIRM_NO.search(text)
                                          or re.search(r"(bye|बाय)\W*$", text, re.I)):
            d.pop("pending", None)
            d["outcome"] = d.get("outcome") or "declined"
            return TurnPlan(intent="declined", llm_allowed=False, end_call=True, objective="Goodbye",
                            fallback=_tpl("Sure, thank you for your time. Have a good day!",
                                          "ठीक है, आपके समय के लिए धन्यवाद। आपका दिन शुभ हो!",
                                          "Theek hai, aapke time ke liye thank you. Aapka din accha ho!"))

        if (neg := self._negotiate_step(text, d, ctx, lang)) is not None:
            return neg

        frustration = float(d.get("_frustration", "0") or 0)
        mv = self._psyche_move(d)
        if mv is not None and mv.name == "soft_check":  # "not now" is not "never": one gentle check before closing
            return self._move_plan(mv, d)
        if self._no(text, intent, d):
            reoffers = int(d.get("reoffers", "0"))
            if mv is None and reoffers < p.max_reoffers and frustration < 0.5 and pending in ("intro", "pitch", "close", "confirm") \
                    and negotiation.NegState.load(d).rounds == 0 \
                    and d.get("existing_user") != "1":
                d["reoffers"] = str(reoffers + 1)
                d["pending"] = "reoffer"
                return TurnPlan(intent="reoffer", decision="one soft re-offer", llm_allowed=False,
                                facts={"offer": Fact(value=self._terms(lang), description="the full offer, said as one phrase")},
                                objective="Respect their answer. Mention once, gently, the offer (say it as one phrase), and ask "
                                          "if they'd like to try it. No pressure",
                                ask="would you like to try it?", must_not_say=["pressure", "urgency", "guilt"],
                                fallback=self._cx(_tpl("I understand. Just so you know, it's {offer}@@. Would you like to try it?",
                                                       "समझ सकता हूँ। बस बता दूँ, ये {offer} है@@। क्या आप ट्राई करना चाहेंगे?",
                                                       "Samajh sakta hoon. Bas bata doon, ye {offer} hai@@. Try karna chahenge?")))
            d.pop("pending", None)
            d["outcome"] = "declined"
            return TurnPlan(intent="declined", llm_allowed=False, decision="customer declined", end_call=True,
                            objective="Thank them and say goodbye; do not pitch again",
                            fallback=_tpl("No problem at all. Thanks for your time, have a good day!",
                                          "कोई बात नहीं। आपके समय के लिए धन्यवाद, आपका दिन शुभ हो!",
                                          "Koi baat nahi. Aapke time ke liye thank you, aapka din accha ho!"))

        if intent == "goodbye" and not self._asks(text, d) and not _weak_yes(text):
            return TurnPlan(intent="goodbye", llm_allowed=False, end_call=True, objective="Goodbye",
                            fallback=_tpl("Thank you, have a great day!", "धन्यवाद, आपका दिन शुभ हो!", "Thank you, aapka din accha ho!"))

        if _HELLO.match(text):
            return TurnPlan(intent="hello", llm_allowed=False, objective="Confirm you're there",
                            fallback=_tpl("Yes, I'm here, I can hear you. Please go on.", "जी, मैं सुन रहा हूँ, बताइए।",
                                          "Ji, main sun raha hoon, bataiye."))

        if intent == "slow_down":
            d["pending"] = "discovery"
            return TurnPlan(intent="slow_down", decision="stop pitching; ask about their needs",
                            objective="Apologise for rushing in a few words and ask ONE open question about what they are looking for; "
                                      "do not mention price or the link",
                            fallback={l: v + " " + p.discovery_question.get(l, "") for l, v in _tpl(
                                "You're right, sorry for rushing.", "आप सही कह रहे हैं, माफ़ कीजिए मैं जल्दी कर रहा था।",
                                "Aap sahi keh rahe hain, sorry main jaldi kar raha tha.").items()})

        # ---------------- a question the facts do not cover (refund, UPI, parking, a branch near them, a health condition): honest
        if (hq := self._honest_step(text, d, lang)) is not None:
            return hq

        # ---------------- a price question about something the offer includes ("trainer कितने का?"): from the facts
        from ...psyche.moves import item_price_answer
        ip = item_price_answer(p, text, lang)
        if ip is not None and mv is None and ctx is not None and not ctx.has("payment_link"):
            d["pending"] = "close" if "offer_price" in ctx.quoted else "pitch"
            return TurnPlan(intent="answer", llm_allowed=False, decision="answer from the facts",
                            objective="Say the item is included; do not quote the membership price again",
                            fallback=_tpl(*(item_price_answer(p, text, l)[1] for l in ("en", "hi", "hinglish"))))

        # ---------------- psyche (ACIE_PSYCHE=1): the move that serves the caller's need, never a repeat
        if mv is not None and mv.name.split(":")[0] != "no_detail":
            return self._move_plan(mv, d)

        # ---------------- fast scripted steps for the plain cases
        asks = self._asks(text, d)
        pitched = ctx is not None and "what_the_product_is" in ctx.told
        # a short "yes, go on" - not "I'm a college student, assignments and coding" (that deserves an answer)
        plain_go_on = ((intent in ("tell_more", "affirm") and len(text.split()) <= 4) or _weak_yes(text) or _clean_yes(text)) and not asks
        quoted = ctx is not None and "offer_price" in ctx.quoted
        short_what = asks and len(text.split()) <= 6 and (u is None or u.question_topic in ("product", "none"))
        if not pitched and not quoted and pending in ("intro", None) and self._faq_match(text) is None \
                and (plain_go_on or (intent in ("tell_more", "benefits") and (not asks or short_what))) \
                and not (u is not None and u.heard("personal", "small_talk")) and p.pitch_hi and p.pitch_en:
            d["pending"] = "pitch"
            return TurnPlan(intent="pitch", llm_allowed=False, facts={"discount": self._offer()["discount"]},
                            objective="Operator-written pitch, then the discount and a short question", ask="shall I tell you the price?",
                            fallback=_tpl(p.pitch_en + " Right now it's {discount}. Shall I tell you the price?",
                                          p.pitch_hi + " अभी इस पर {discount} चल रहा है। कीमत बताऊँ?",
                                          (p.pitch_hinglish or p.pitch_hi) + " Abhi is pe {discount} chal raha hai. Price bataun?"))
        decided = ctx is not None and (ctx.has("payment_link") or ctx.decision != "undecided")
        price_core = bool(_PRICE_CORE.search(text)) and asks
        pure_price = ((intent == "price" or price_core) and len(text.split()) <= 6 and bool(_PRICE_ASK.search(text)) and self._faq_match(text) is None
                      and d.get("last_intent") != "price"  # asked again = about something else (trainer cost)
                      and not decided and not _PRICE_DETAIL.search(text)
                      and not self._other_topic(text, u, intent)  # "personal trainer session कितने का?" is about the trainer
                      and (u is None or u.question_topic in ("offer_terms", "none")))
        if pure_price or (pending == "pitch" and plain_go_on):
            d["pending"] = "discovery"
            return self._price_plan(lang, known=bool(d.get("need_known")))
        if price_core and quoted and len(text.split()) <= 10 and not decided and self._faq_match(text) is None and not _PRICE_DETAIL.search(text) and not _NEGOTIATE.search(text) \
                and not _HAGGLE.search(text) and not self._other_topic(text, u, intent) and (u is None or u.question_topic in ("offer_terms", "none")):
            # asked for the price again ("आपने बताया था, कितना था?"): the same fact, a new shape each time, never "I don't have it"
            n = int(d.get("price_again", "0"))
            d["price_again"] = str(n + 1)
            variants = self._tpl3(lambda l: (honest.price_variants(p, l) + honest.end_variants(p, l))[min(n, len(honest.price_variants(p, l)) + 3)])
            return TurnPlan(intent="price", llm_allowed=False, decision="the price again, in other words", fallback=variants,
                            objective="They asked for the price again: say it again in different words; no new question")

        # ---------------- conversation: everything else
        return await self._converse(text, intent, d, ctx, lang, frustration)

    # ------------------------------------------------------------------ psyche hooks

    @staticmethod
    def _psyche_move(d: dict):
        raw = d.get("_move")
        if not raw:
            return None
        from ...psyche.moves import Move
        try:
            return Move.from_json(raw)
        except (ValueError, TypeError):
            return None

    def _move_plan(self, mv, d: dict) -> TurnPlan:
        """The psyche move as a plan: its own deterministic wording (three languages), never the LLM."""
        if mv.slot == "what":
            intent = "pitch"
        elif mv.slot == "offer":
            intent = "price"
        elif mv.name == "respect_close":
            intent = "declined"
        elif mv.end_call:
            intent = "goodbye"
        else:
            intent = {"soft_check": "acknowledge", "ack": "acknowledge", "simplify": "clarify", "ask_repeat": "clarify", "ask_unclear": "clarify",
                      "offer_stop": "clarify", "reassure": "answer", "connect": "personal"}.get(mv.name, "answer")
        if mv.name == "respect_close":
            d["outcome"] = d.get("outcome") or "declined"
        d["pending"] = "pitch" if mv.slot == "what" else ("discovery" if mv.slot == "offer" else d.get("pending", "") or "close")
        if mv.end_call:
            d.pop("pending", None)
        return TurnPlan(intent=intent, llm_allowed=False, decision="psyche_move", end_call=mv.end_call,
                        objective=mv.note or mv.name, fallback=dict(mv.text))

    # ------------------------------------------------------------------ conversation turn

    async def _converse(self, text: str, intent: str, d: dict, ctx, lang: str, frustration: float) -> TurnPlan:
        p = self.p
        u = self._nlu(d)
        general = self._general(text, d)
        ref = ""
        if general and self.world is not None:
            if self.interim is not None:
                await self.interim({"en": "One second, let me check.", "hi": "एक सेकंड, देखकर बताता हूँ।",
                                    "hinglish": "Ek second, dekh ke batata hoon."})
            ref = await self.world.lookup(text)
        if general and _WEATHER_Q.search(text) and "weather now" not in ref.lower() and "no city" not in ref.lower():
            # nothing was looked up (no world service, a timeout, no result): never invent the weather ("दिल्ली में मौसम साफ़")
            return TurnPlan(intent="small_talk", llm_allowed=False, decision="cannot check the weather: say so",
                            objective="Say honestly you can't check the weather right now",
                            fallback=_tpl("I can't check the weather right now, sorry; your phone will show it faster.",
                                          "मैं अभी मौसम चेक नहीं कर पा रहा, माफ़ कीजिए; आपके फ़ोन पर जल्दी दिख जाएगा।",
                                          "Main abhi mausam check nahi kar paa raha, sorry; aapke phone par jaldi dikh jaayega."))

        if d.get("pending") == "discovery" and not _NOT_NOW_LINK.search(text):  # they just told us what they use: tie the product to that
            d["pending"] = "close" if ctx is not None and "offer_price" in ctx.quoted else "pitch"
            d["benefit_turns"] = str(int(d.get("benefit_turns", "0")) + 1)
            stage = ("They just told you how they work or what they use (or answered your question). Connect the product to "
                     "exactly that: give one or two CONCRETE examples of how it would help them in that work (a task it "
                     "does for them, time it saves). Then ask one short question about their work or what else they need. "
                     "Do not ask about the link in this reply.")
        else:
            stage = self._stage(text, d, ctx, frustration, u)

        knowledge = [p.knowledge] if p.knowledge else []
        approved = [f"- {f.answer}" for f in p.faqs if f.answer]
        if approved:
            knowledge.append("Approved answers you can use:\n" + "\n".join(approved))
        if p.price_note:
            knowledge.append(f"Price in rupees: {p.price_note}")
        gaps = sorted({t.name["en"] for t in honest.TOPICS if (t.applies is None or t.applies(p)) and not (t.id == "offer_end" and re.search(r"week|today", p.offer_validity, re.I))})
        knowledge.append("You do NOT have these details, so if asked say honestly that you don't have them (and where to look: "
                         + honest.source(p, "en") + "); never guess or invent a value, a yes or a no for them: " + "; ".join(gaps)
                         + ". You also cannot say what is safe or good for anyone's health, and you never claim a facility, policy or "
                         "cancellation/refund term that is not in the background above.")
        if u is not None and u.gist:
            knowledge.append(f"What the customer most likely means: {u.gist}")
        if d.get("callback_note"):
            knowledge.append(f"Callback the customer asked for: {d['callback_note']}")

        facts = {**self._base(),
                 "offer": Fact(value=self._terms(lang), description="the monthly offer as one phrase; it already says for how long and the "
                                                                  "price after, so don't add 'for 3 months' again; never use it for yearly or Team plans",
                               must_mention=False),
                 "validity": Fact(value=f"{p.offer_validity}, then {p.original_price}", description="offer duration",
                                  verbatim=False, must_mention=False),
                 **self._benefits(len(p.benefits))}
        if p.price_note:
            facts["rupees"] = Fact(value=p.price_note, description="price in rupees", verbatim=False, must_mention=False)

        lead = None
        quoted_last = d.get("last_intent") in ("price", "confirm_close", "reoffer") or d.get("quoted_last") == "1"
        other_plan = re.search(_ANNUAL + r"|team|टीम", text, re.I)
        complaint = re.search(r"(just|already) (said|told|gave|mentioned)|told me the price|(said|told).{0,15}price|"
                              r"bata (diya|chuke)|बता (दिया|चुके)|फिर से price|again", text, re.I)
        explicit_terms = _RUPEE_ASK.search(text) or _RENEWAL_ASK.search(text)
        haggle = _HAGGLE.search(text) or re.search(r"student|स्टूडेंट|(mein|me|में|for) (do|de do|दो|दे दो)\b|do na\b|दो ना|"
                                                   r"zyada hai|jyada hai|ज़्यादा है|ज्यादा है|mehenga|mehnga|महंगा|expensive|too much", text, re.I)
        if _PRICE_ASK.search(text) and not other_plan and not complaint and not haggle and (explicit_terms or not quoted_last) \
                and self._faq_match(text) is None \
                and not self._other_topic(text, u, intent) \
                and (u is None or u.question_topic in ("offer_terms", "none", "product") or intent == "price"):
            lead = self._price_lead(text, lang)
            for k in ("offer", "validity", "rupees"):
                facts.pop(k, None)
        elif re.search(_ANNUAL, text, re.I) and self.p.annual_note.get(lang) and not (u is not None and u.heard("personal")):
            lead = self.p.annual_note[lang]
            for k in ("offer", "validity", "rupees"):
                facts.pop(k, None)
        elif ctx is not None and "offer_price" in ctx.quoted and not _PRICE_ASK.search(text):
            facts.pop("validity", None)
            if not (haggle or other_plan or complaint):
                for k in ("offer", "rupees"):
                    facts.pop(k, None)  # the price was told: no price line in a turn that did not ask for it (the sceptic got it four times)
        d["quoted_last"] = "1" if lead else "0"
        node_note = self._playbook_turn(text, intent, d, u, frustration)
        if node_note:
            knowledge.append(node_note[0])
            stage = node_note[1]
        sad = u is not None and u.heard("personal")
        faq = self._faq_match(text)
        if faq is not None:
            d["_objection"] = (faq.keywords or ["question"])[0]
            if faq.not_eligible:
                d["existing_user"] = "1"
        # when the LLM is down or its reply fails a check: the approved answer if there is one
        asked = honest.is_question(text)
        if (ctx is None or "what_the_product_is" not in ctx.told) and p.pitch_en and p.pitch_hi and int(d.get("answers", "0")) <= 1:
            generic = _tpl(p.pitch_en + " Shall I tell you the price?", p.pitch_hi + " कीमत बताऊँ?",
                           (p.pitch_hinglish or p.pitch_hi) + " Price bataun?")
        elif asked:  # a real question: the catalog item it matches, else the honest line; never "I didn't get that"
            from ...psyche.moves import build_catalog, _matches
            hit = next((i for i in build_catalog(p) if i.kind != "faq" and _matches(i, text)), None)
            generic = self._tpl3(lambda l: (hit.text.get(l) if hit is not None else None) or self._unknown_answer(text, d, l))
        else:
            generic = _tpl("Sorry, I didn't quite get that. Could you say it once more?",
                           "माफ़ कीजिए, मैं ठीक से समझ नहीं पाया। एक बार फिर से बताएँगे?",
                           "Sorry, main theek se samajh nahi paya. Ek baar phir se bataenge?")
        if faq is None and node_note and node_note[2]:
            fb = _tpl(node_note[2], node_note[2], node_note[2])  # the playbook's own line beats "say that again?"
        elif faq is not None:
            texts = {"en": faq.answer + ".", "hi": faq.answer_hi + "।" if faq.answer_hi else "",
                     "hinglish": faq.answer_hinglish + "." if faq.answer_hinglish else ""}
            fb = {l: texts[l] or generic[l] for l in texts}  # never the English answer in a Hindi/Hinglish call
        else:
            fb = generic
        objective = (
            "Reply to what the customer just said FIRST, directly and truthfully, like a helpful person on the phone: "
            "answer their question from the background, approved answers and offer facts (use {offer} for anything about price "
            "or duration). If the background does not cover it, say honestly you're not sure and that the details are on the "
            "official website - never say a team will send details. "
            + ("They shared something personal or sad: respond with warmth and kindness first, no selling in this reply. " if sad else "")
            + ("If the question is unclear, ask in a few words what they mean. ")
            + "Never ask whether they want or need it, and never end with the same question as your last reply. "
            + "Never call the customer by a name unless they clearly told you their name. "
            + "You called them: never ask 'how can I help you'. " + self._audience()
            + "Never claim personal experiences (you are an AI: you didn't watch the match or feel the weather). "
            + ("They seem confused: use very simple everyday words, no English or technical terms, and say amounts in rupees. "
               if u is not None and (u.confusion >= 0.7 or (u.confusion >= 0.5 and (self.customer_age or 0) >= 55)) else "")
            + stage
            + (f" {psy.note} Do not sell in this reply." if (psy := self._psyche_move(d)) is not None and psy.need in
               ("autonomy", "space", "reassurance", "be_heard", "connection", "clarity") else "")
            + (f" They already told you what they need or use it for: \"{d['need_known']}\". NEVER ask again what they use it for or "
               "what their goal is." if d.get("need_known") else "")
            + " Never repeat a question you already asked earlier in the call. Speak only in the reply language: never an English "
              "sentence in a Hindi or Hinglish reply."
            + (" The lookup returned nothing: if the question needs live information (today's weather, news, a score, a price) say "
               "you can't check it right now; never guess it." if general and not ref else "")
        )
        if faq is not None and not lead:
            # the operator-approved answer is spoken word for word (the 14B model paraphrased "only if you allow
            # it" into "we never train on your data", and ignored "you already have Pro")
            lead = {"hi": faq.answer_hi + "।" if faq.answer_hi else "", "hinglish": faq.answer_hinglish + "." if faq.answer_hinglish else "",
                    "en": faq.answer + "."}.get(lang) or None
            # a language with no written answer: the LLM paraphrases the approved English answer (it is in the background)
        if lead and lead == d.get("last_lead") and not _RUPEE_ASK.search(text):
            lead = None  # said it word for word last turn: don't paste it again
            facts["offer"] = Fact(value=self._terms(lang), description="the monthly offer as one phrase", must_mention=False)
        d["last_lead"] = lead or ""
        if lead:
            objective = (f"You have ALREADY said this exact sentence: \"{lead}\". Do NOT state any price, discount or duration "
                         "again. Answer only the OTHER parts of what they asked, briefly and truthfully from the background; "
                         "if there is nothing else, reply with one short friendly question (or nothing more than 'Okay?'). " + stage)
            fb = ({"en": faq.answer + ".", "hi": faq.answer_hi + "।" if faq.answer_hi else ANYTHING_ELSE[1],
                   "hinglish": faq.answer_hinglish + "." if faq.answer_hinglish else ANYTHING_ELSE[2]} if faq is not None
                  else _tpl(ANYTHING_ELSE[0], ANYTHING_ELSE[1], ANYTHING_ELSE[2]))
        return TurnPlan(intent="small_talk" if general else ("personal" if sad else "answer"),
                        facts=facts, context="\n\n".join(knowledge) or None, reference=ref or None,
                        general_knowledge=general, objective=objective,
                        lead={lang: lead} if lead else None,
                        must_not_say=["that a team will send details", "made-up features or prices", "'lowest price ever' or other superlatives not in the background", "'up to' discounts",
                                      "claims not in the background", "'limited time' (no end date is known)", "promises about health or safety",
                                      "facilities, policies, refunds or cancellation terms that are not in the background"],
                        fallback=fb)

    def _stage(self, text: str, d: dict, ctx, frustration: float, u) -> str:
        base = self._stage_base(text, d, ctx, frustration, u)
        hint = nba.HINT.get(d.get("nba", "")) if nba.enabled() and d.get("nba") != "confirm_close" else None
        return f"{base} {hint}" if hint else base

    def _stage_base(self, text: str, d: dict, ctx, frustration: float, u) -> str:
        """One small step forward after answering, from where the sale stands (context ledger)."""
        if d.get("existing_user") == "1":
            return ("They already pay for Pro, so this offer (new subscriptions only) does not apply to them: say so plainly "
                    "if relevant, never offer the link or the price, and don't ask what they'd use it for.")
        if d.get("sad_hold") == "1":
            return ("They are going through a hard time (job, money, mood): answer warmly and briefly, and do NOT sell at all: no price, "
                    "no link, no question about what they need. You may say they can take their time or you can call another day.")
        asked_link_last = d.get("last_intent") in ("price", "confirm_close", "reoffer") or d.get("link_asked_last") == "1"
        d["link_asked_last"] = "0"
        if ctx is not None and ctx.has("payment_link"):
            return "They already agreed and the link is on its way by SMS to this number. Don't sell. You may ask if there's anything else."
        if ctx is not None and ctx.decision in ("declined", "later"):
            return "They already said no. Don't sell or offer again."
        if frustration >= 0.4:
            return "They sound annoyed: apologise in a few words, answer directly, keep it short, and do NOT add any selling line or question."
        if _NOT_NOW_LINK.search(text) or d.get("no_link") == "1":
            d["no_link"] = "1"
            return "They said not to send the link now: do not offer the link; you may ask if they have any other question."
        last_nudge = d.get("last_nudge", "")
        quoted = ctx is not None and "offer_price" in ctx.quoted
        if (ctx is None or "what_the_product_is" not in ctx.told) and not quoted and int(d.get("answers", "0")) == 0:
            d["answers"] = "1"
            d["pending"] = "intro"
            d["last_nudge"] = "intro"
            return "Then, only if it fits, ask if they'd like to hear briefly what the product is."
        d["answers"] = str(int(d.get("answers", "0")) + 1)
        if not quoted:
            d["pending"] = "pitch"
            if last_nudge == "pitch":
                d["last_nudge"] = ""
                return "Don't add a question about the price this time (you just offered); just answer, or ask what they'll use it for."
            d["last_nudge"] = "pitch"
            return "Then, only if it fits, offer to tell them the price."
        d["last_nudge"] = ""
        d["pending"] = "close"
        since = int(d.get("since_link_ask", "9")) + 1
        d["since_link_ask"] = str(since)
        benefits = int(d.get("benefit_turns", "0"))
        if asked_link_last or since < 3 or benefits < 1:
            d["benefit_turns"] = str(benefits + 1)
            return ("Do NOT ask about the link in this reply. After answering, explain one concrete benefit that fits what "
                    "you know about them (how it helps in their work, study or daily life, with a specific example), "
                    "and ask one short question about their needs.")
        d["link_asked_last"] = "1"
        d["since_link_ask"] = "0"
        return "Then, since you've shown how it helps them, you may ask once, lightly, whether they'd like to try it."

    # ------------------------------------------------------------------ honest answers, hard times, requests we can't serve

    def _tpl3(self, fn) -> dict[str, str]:
        return {l: fn(l) for l in ("en", "hi", "hinglish")}

    def _unknown_answer(self, text: str, d: dict | None, lang: str, n: int = 0) -> str:
        """The ONE place where the product has nothing to say about a question: the honest-but-helpful line (rotating wording).
        The knowledge-base lane swaps this function for its lookup; nothing else in this domain writes an "I don't have that" line."""
        p = self.p
        topics = honest.find_topics(p, text)
        if topics:
            return honest.unknown_combo(p, lang, topics, min(n, 2))
        return honest.unknown_variants(p, lang)[min(n, 2)]

    def _answer_text(self, text: str, lang: str, n: int = 0) -> str | None:
        """What the agent says to this question when the facts decide it: the operator's approved FAQ answer, or the honest
        "I don't have that detail" for a topic the facts do not cover. None = the conversation turn (LLM) answers."""
        p = self.p
        if honest.health_question(text):
            return honest.health_variants(p, lang)[0]
        topics = honest.find_topics(p, text)
        if topics and topics[0].id == "payer":
            return honest.payer_variants(p, lang)[n % 2]
        if topics and (topics[0].beats_faq or not honest.covered_by_faq(p, text)):
            return self._unknown_answer(text, None, lang, n)  # a different wording each time the call hits a gap
        faq = self._faq_match(text)
        if faq is not None and not faq.not_eligible:
            return ({"hi": faq.answer_hi, "hinglish": faq.answer_hinglish}.get(lang) or (faq.answer if lang == "en" else "")) or None
        if re.search(r"kab tak|कब तक|how long|valid", text, re.I) and re.search(r"offer|ऑफर|discount|डिस्काउंट|price|कीमत", text, re.I):
            line = honest.offer_end_variants(p, lang)[0]
            if _RUPEE_ASK.search(text) and p.rupee_terms.get(lang):
                line += " " + re.sub(r"^(यानी|yaani|that's)\s*", "", p.rupee_terms[lang]).strip().capitalize() + ("।" if lang == "hi" else ".")
            return line
        return None

    def _honest_step(self, text: str, d: dict, lang: str) -> TurnPlan | None:
        """A question about something the facts do not cover (refund, UPI, parking, a branch near them ...) or a health
        condition: the honest line, deterministic, never a guess and never "shall I let you go?"."""
        if not honest.is_question(text) or d.get("existing_user") == "1":
            return None
        p = self.p
        health = honest.health_question(text)
        topics = honest.find_topics(p, text)
        unknown = bool(topics) and (topics[0].beats_faq or not honest.covered_by_faq(p, text))
        ended = bool(re.search(r"kab tak|कब तक|how long|valid", text, re.I)) and bool(re.search(r"offer|ऑफर|discount|डिस्काउंट|price|कीमत", text, re.I)) \
            and not honest.covered_by_faq(p, text) and not topics
        if not (health or unknown or ended):
            return None
        n = int(d.get("unk_n", "0"))
        d["unk_n"] = str(n + 1)
        line = self._tpl3(lambda l: self._answer_text(text, l, n) or "")
        d["pending"] = d.get("pending") or "close"
        return TurnPlan(intent="answer", llm_allowed=False, decision="honest answer: the facts do not cover it", fallback=line,
                        objective=("Never promise what is safe for a health condition; send them to their doctor" if health
                                   else "Say honestly you do not have that detail and where they will find it; do not guess"))

    def _info_request_step(self, text: str, d: dict, ctx, lang: str) -> TurnPlan | None:
        """"Send me the trial link / the branch address / WhatsApp it / the free sign-up link": answered honestly. The only SMS
        this agent sends is the payment link, so these are never turned into "shall I send the payment link?"."""
        kind = honest.info_request(text)
        if kind is None or (ctx is not None and ctx.has("payment_link")):
            return None
        if kind != "whatsapp":
            d["no_link"] = "1"  # they did not ask for the payment link: do not push it
        d["pending"] = d.get("pending") or "close"
        lines = self._tpl3(lambda l: honest.info_request_variants(self.p, kind, l)[0])
        bye = bool(_GOODBYE_NOW.search(text))
        if bye:
            d.pop("pending", None)
            d["outcome"] = d.get("outcome") or "declined"
        return TurnPlan(intent="answer", llm_allowed=False, end_call=bye, decision="honest answer: cannot send that", fallback=lines,
                        objective="Say honestly you can only send the payment link by SMS; do not turn their request into a payment-link offer")

    def _negotiate_step(self, text: str, d: dict, ctx, lang: str) -> TurnPlan | None:
        """Price talk (haggling, a counter-offer, a competitor, budget, value doubt, a group discount, a free trial) goes through the
        deterministic negotiation state machine (acie/decision/negotiation.py): the floor once, then only the levers the product config
        allows, then a warm walk-away. State lives in dialog["_neg"] so it never loops or repeats."""
        if d.get("existing_user") == "1" or d.get("sad_hold") == "1" or _FIRM_NO.search(text) or _OPT_OUT.search(text):
            return None
        if ctx is not None and ctx.has("payment_link"):
            return None
        faq = self._faq_match(text)
        faq_line = ((lambda l: {"hi": faq.answer_hi, "hinglish": faq.answer_hinglish, "en": faq.answer}.get(l) or "")
                    if faq is not None and not faq.not_eligible else None)
        r = negotiation.step(self.p, text, negotiation.NegState.load(d), d.get("need_known", ""), faq_line)
        if r is None:
            return None
        r.state.save(d)
        d["pending"] = "neg_close" if r.offer else "close"
        if r.offer:
            d["neg_offer"] = json.dumps(r.offer)
        return TurnPlan(intent=r.intent, llm_allowed=False, decision="negotiation: " + r.state.last, end_call=False, fallback=r.text,
                        facts={"offer": Fact(value=self._terms(lang), description="the offer as one phrase", must_mention=False)},
                        objective=r.note or "Answer the price talk honestly; no pressure, no invented discount")

    def _apply_neg_offer(self, d: dict) -> None:
        """The caller said yes to a concession: the agreed price goes into the terms, and into the ledger (closed price vs the profit floor)."""
        try:
            o = json.loads(d.get("neg_offer") or "{}")
        except ValueError:
            return
        eco = self.p.economics
        price = float(o.get("price") or eco.list_price or 0)
        if eco.list_price and price < eco.profit_floor:
            price = eco.profit_floor  # never below the floor, whatever was said
        upd = {}
        if eco.price_terms and price and price != eco.list_price:
            upd["terms"] = {l: eco.price_terms[l].format(price=f"{eco.currency}{int(round(price)):,}") for l in eco.price_terms}
        perk = next((x for x in eco.ladder if x.id == o.get("perk") and x.kind == "perk"), None)
        if perk is not None:
            base = upd.get("terms") or {l: honest.terms_text(self.p, l) for l in ("en", "hi", "hinglish")}
            upd["terms"] = {l: base[l] + ", " + (perk.short.get(l) or perk.text.get(l, "")) for l in base}
        if upd:
            self.p = self.p.model_copy(update=upd)
        if price:
            d["closed_price"] = f"{price:.2f}"
            d["list_price"] = f"{eco.list_price:.2f}"
            d["profit_floor"] = f"{eco.profit_floor:.2f}"
            d["margin_pct"] = f"{(price - eco.cost) / price * 100:.1f}" if eco.cost and price else ""

    def _repeat_request_step(self, text: str, d: dict, ctx, lang: str) -> TurnPlan | None:
        """"फिर से बोलिए / धीरे बोलो / say that again": the AGENT says the last thing again, shorter and slower, in plain words
        (it used to say "I'll tell it again" and then not tell it)."""
        if not _REPEAT_ASK.search(text) or len(text.split()) > 16 or _OPT_OUT.search(text):
            return None
        if self._psyche_move(d) is not None and self._psyche_move(d).need == "be_heard" and re.search(r"repeat|रिपीट|रिपेट|बार बार|bar bar", text, re.I):
            return None  # "stop repeating yourself" is a complaint, handled by the be_heard move
        p = self.p
        quoted = ctx is not None and "offer_price" in ctx.quoted
        told = ctx is not None and "what_the_product_is" in ctx.told
        if quoted and (_PRICE_ASK.search(text) or d.get("last_intent") in ("price", "reoffer", "confirm_close", "answer")):
            n = int(d.get("repeat_n", "0"))
            d["repeat_n"] = str(n + 1)
            lines = self._tpl3(lambda l: (honest.price_variants(p, l) + honest.end_variants(p, l))[min(n, len(honest.price_variants(p, l)) + 3)])
            intent = "price"
        elif not told and not quoted:
            lines = _tpl("Sure, slowly. I'm {agent}, an AI assistant from {seller}. I'm calling about an offer on {product}. Is this a good time for two minutes?",
                         "जी, धीरे बताता हूँ। मैं {agent} हूँ, {seller} से एक AI असिस्टेंट। {product} के एक ऑफर के बारे में फ़ोन किया है। क्या दो मिनट बात हो सकती है?",
                         "Ji, dheere batata hoon. Main {agent} hoon, {seller} se ek AI assistant. {product} ke ek offer ke baare mein phone kiya hai. Kya do minute baat ho sakti hai?")
            intent = "clarify"
        else:
            lines = _tpl(p.pitch_en, p.pitch_hi or p.pitch_en, p.pitch_hinglish or p.pitch_hi or p.pitch_en)
            intent = "clarify"
        facts = {"agent": Fact(value=p.agent_name, description="your name", must_mention=False),
                 "seller": Fact(value=p.seller_name, description="your company", must_mention=False),
                 "product": Fact(value=p.product_name, description="product name", must_mention=False)}
        return TurnPlan(intent=intent, llm_allowed=False, facts=facts, decision="say the last thing again, slowly and simply", fallback=lines,
                        objective="They could not follow: say the last thing again, slowly, in plain words")

    def _sad_step(self, text: str, intent: str, d: dict, lang: str) -> TurnPlan | None:
        """Lost a job / no money / tired: pause the selling for the rest of the call (no price, no link, no re-offer, no
        discovery question), be warm and short, answer what they ask, offer another time. Explicit buying words re-open it."""
        now = bool(honest.SAD.search(text)) or d.get("_row") == "sad"
        if now:
            d["sad_hold"] = "1"
            d["reoffers"] = str(max(int(d.get("reoffers", "0")), self.p.max_reoffers))  # no soft re-offer to someone in trouble
        if d.get("sad_hold") != "1":
            return None
        if not now and (_BUY.search(text) and not honest.FREE_WANT.search(text)) and not _HYPOTHETICAL.search(text):
            d.pop("sad_hold", None)  # they re-opened it themselves
            return None
        if _GOODBYE_NOW.search(text) or _FIRM_NO.search(text) or _OPT_OUT.search(text) or (
                not now and intent in ("deny", "goodbye", "callback", "human_agent")):
            return None  # the caller is closing: the normal goodbye (never a re-offer) handles it
        if honest.is_question(text) and not now:
            return None  # a real question: the conversation answers it (its stage says: no selling)
        n = int(d.get("sad_n", "0"))
        d["sad_n"] = str(n + 1)
        variants = [self._tpl3(lambda l: honest.empathy_variants(self.p, l, True)[0])] + \
            [dict(zip(("en", "hi", "hinglish"), t)) for t in zip(*(honest.empathy_variants(self.p, l, False) for l in ("en", "hi", "hinglish")))]
        return TurnPlan(intent="personal", llm_allowed=False, decision="hard time: pause selling", fallback=variants[min(n, len(variants) - 1)],
                        objective="Be warm, short and human; no selling at all; offer to call another time")

    async def _answer_before_goodbye(self, plan: TurnPlan, intent: str, turn: TurnInput, session: SessionState) -> TurnPlan:
        """Never say goodbye with an unanswered question pending ("कितने दिन में activate होगा?" got a goodbye)."""
        d, text = session.dialog, turn.text
        if not (plan.end_call and plan.intent in ("goodbye", "declined", "link_sent")):
            return plan
        if not honest.is_question(text) or _FIRM_NO.search(text) or _OPT_OUT.search(text) or _GOODBYE_NOW.search(text) \
                or re.search(r"cancel|mat bhej|मत भेज|rehne do|रहने दो", text, re.I):
            return plan
        ctx = getattr(session, "context", None)
        lang = d.get("_lang", "hi")
        return await self._converse(text, intent, d, ctx, lang, float(d.get("_frustration", "0") or 0))

    # ------------------------------------------------------------------ policy pieces

    def _compliance(self, text: str, intent: str, d: dict, ctx=None) -> TurnPlan | None:
        p = self.p
        if d.get("pending") == "optout_offer":  # we asked "shall I remove your number?"
            d["pending"] = d.pop("before_optout", "") or "intro"
            if intent == "affirm" or re.search(r"hata (do|dijiye)|हटा (दो|दीजिए|दीजिये)|remove it|yes", text, re.I):
                return self._opt_out_plan(d)
        if _OPT_OUT.search(text) and not (_LATER.search(text) and not re.search(r"dobara|dubara|दोबारा|दुबारा|never|कभी", text, re.I)):
            return self._opt_out_plan(d)
        ident = {"agent": Fact(value=p.agent_name, description="your name", must_mention=False),
                 "seller": Fact(value=p.seller_name, description="your company", must_mention=False),
                 "product": Fact(value=p.product_name, description="product name", must_mention=False)}
        if _SENSITIVE_DIGITS.search(text) or (_SENSITIVE.search(text) and not _is_question(text)
                                              and d.get("last_intent") != "sensitive_stop"):
            return TurnPlan(intent="sensitive_stop", llm_allowed=False, decision="never take card/OTP details on a call",
                            objective="Stop them sharing payment details",
                            fallback=_tpl("Please stop, don't share any card number, OTP or PIN on this call. I will never ask for them; payment only happens on the secure payment page.",
                                          "रुकिए, कृपया इस कॉल पर कार्ड नंबर, OTP या पिन मत बताइए। मैं कभी ये नहीं माँगूँगा, पेमेंट सिर्फ़ सुरक्षित पेमेंट पेज पर होता है।",
                                          "Rukiye, please is call pe card number, OTP ya PIN mat bataiye. Main kabhi ye nahi maangunga, payment sirf secure page pe hota hai."))
        told = ctx.told if ctx is not None else []
        if _ARE_YOU_AI.search(text) and "agent_is_an_ai" not in told:
            return TurnPlan(intent="are_you_ai", llm_allowed=False, facts=ident, decision="always admit to being an AI",
                            objective="Say honestly you're an AI assistant calling for the company",
                            fallback=_tpl("Yes, I'm an AI assistant calling on behalf of {seller}, not a human. If you'd prefer, a person from the team can call you instead.",
                                          "जी हाँ, मैं इंसान नहीं, {seller} की तरफ़ से एक AI असिस्टेंट हूँ। आप चाहें तो टीम का कोई व्यक्ति आपको कॉल कर सकता है।",
                                          "Ji haan, main insaan nahi, {seller} ki taraf se ek AI assistant hoon. Aap chahein to team ka koi insaan aapko call kar sakta hai."))
        about_product = re.search(r"claude|क्लॉड|क्लाउड|anthropic|एन्थ्रॉपिक|ए ?आई|\bai\b|" + re.escape(p.product_name.lower()), text, re.I)
        if _WHO.search(text) and not _NUMBER_SOURCE.search(text) and not _ARE_YOU_AI.search(text) and not about_product:
            intro = d.get("pending") in ("intro", None)
            sorry = re.search(r"फिर से|again|dobara|दोबारा|kal bhi|कल भी|phir se", text, re.I)
            pre = _tpl("Sorry for calling again. ", "माफ़ कीजिए, दोबारा कॉल के लिए। ", "Sorry, dobara call ke liye. ") if sorry else _tpl("", "", "")
            again = ctx is not None and "who_is_calling" in ctx.told
            return TurnPlan(intent="identify", llm_allowed=again, facts=ident,
                            objective=("In a few words remind them who you are ({agent}, an AI assistant calling for {seller}), "
                                       "and respond warmly to anything else they said; don't repeat your opening line, and never claim you called before"
                                       if again else "Say who is calling and why"),
                            fallback=_tpl(pre["en"] + "I'm {agent}, an AI assistant calling for {seller} about an offer on {product}. " + ("Do you have two minutes?" if intro else "Please go on."),
                                          pre["hi"] + "मैं {agent} हूँ, {seller} की तरफ़ से एक AI असिस्टेंट, {product} के एक ऑफर के बारे में कॉल किया है। " + ("दो मिनट बात कर सकते हैं?" if intro else "बताइए।"),
                                          pre["hinglish"] + "Main {agent} hoon, {seller} ki taraf se ek AI assistant, {product} ke ek offer ke baare mein call kiya hai. " + ("Do minute baat kar sakte hain?" if intro else "Bataiye.")))
        if _NUMBER_SOURCE.search(text) and d.get("asked_number_source") != "1":
            d["asked_number_source"] = "1"
            d["before_optout"] = d.get("pending", "")
            d["pending"] = "optout_offer"
            return TurnPlan(intent="number_source", llm_allowed=False, facts=ident,
                            objective="Say where the number came from honestly and offer to remove it",
                            fallback=_tpl("Honestly, your number was on a calling list I was given, and I don't have details of where the list came from. If you don't want these calls, I can remove it right now. Shall I remove it?",
                                          "सच कहूँ तो आपका नंबर मुझे मिली एक कॉलिंग लिस्ट में था, और वो लिस्ट कहाँ से आई इसकी जानकारी मेरे पास नहीं है। अगर आप ऐसी कॉल नहीं चाहते, तो मैं अभी नंबर हटा सकता हूँ। हटा दूँ?",
                                          "Sach kahoon to aapka number mujhe mili ek calling list mein tha, aur wo list kahan se aayi iski jaankari mere paas nahi hai. Agar aap aisi calls nahi chahte to main abhi hata sakta hoon. Hata doon?"))
        return None

    def _opt_out_plan(self, d: dict) -> TurnPlan:
        d.pop("pending", None)
        d["outcome"] = "opt_out"
        return TurnPlan(intent="opt_out", llm_allowed=False, end_call=True, decision="add to do-not-call list",
                        objective="Confirm the number is removed, apologise for the disturbance, goodbye",
                        fallback=_tpl("Of course. I've removed your number, you won't get these calls again. Sorry for the trouble, have a good day.",
                                      "ज़रूर। मैंने आपका नंबर हटा दिया है, अब आपको ये कॉल नहीं आएँगी। परेशानी के लिए माफ़ी, आपका दिन शुभ हो।",
                                      "Zaroor. Maine aapka number hata diya hai, ab aapko ye calls nahi aayengi. Pareshani ke liye maafi, aapka din accha ho."))

    def _after_decision(self, intent: str, text: str, d: dict, ctx, lang: str) -> TurnPlan | None:
        if ctx.has("number_removed_no_more_calls"):
            return TurnPlan(intent="opt_out_confirmed", llm_allowed=False, end_call=True, objective="Reconfirm removal",
                            fallback=_tpl("Yes, your number has been removed, you won't get these calls again. Goodbye.",
                                          "जी, आपका नंबर हटा दिया गया है, अब ये कॉल नहीं आएँगी। नमस्ते।",
                                          "Ji, aapka number hata diya gaya hai, ab ye calls nahi aayengi. Namaste."))
        if ctx.has("payment_link"):
            if _NOT_AGREED.search(text):
                ctx.promised.remove("payment_link")
                ctx.decision = "undecided"
                d["outcome"] = ""
                d["no_link"] = "1"
                d.pop("terms_confirmed", None)
                d["pending"] = "close"
                return TurnPlan(intent="link_cancelled", llm_allowed=False, decision="customer had not agreed: nothing is sent",
                                objective="Apologise, nothing will be sent, no pressure",
                                fallback=_tpl("Sorry, my mistake. I won't send anything. Please take your time and talk it over with your family. Is there anything you'd like to know?",
                                              "माफ़ कीजिए, मेरी गलती। मैं कुछ नहीं भेजूँगा। आराम से घर में बात कर लीजिए। कुछ और जानना चाहेंगे?",
                                              "Sorry, meri galti. Main kuch nahi bhejunga. Aaram se ghar mein baat kar lijiye. Kuch aur jaanna chahenge?"))
            # only an explicit cancel ("don't send it", "नहीं चाहिए") - "नहीं बस, ठीक है। Thank you" means "that's all"
            if _CANCEL.search(text) and not self._asks(text, d):
                d["outcome"] = "declined"
                return TurnPlan(intent="declined", end_call=True, llm_allowed=False, decision="customer cancelled after agreeing",
                                objective="Confirm you won't send the link; goodbye",
                                fallback=_tpl("No problem, I won't send the link. Thanks for your time, have a good day!",
                                              "कोई बात नहीं, मैं लिंक नहीं भेजूँगा। आपके समय के लिए धन्यवाद, आपका दिन शुभ हो!",
                                              "Koi baat nahi, main link nahi bhejunga. Aapke time ke liye thank you!"))
            if self._asks(text, d) or self._faq_match(text) or intent == "price" or _NOT_ARRIVED.search(text):
                return None  # answered by the conversation turn ("the link is on its way")
            d["outcome"] = "sold"
            return TurnPlan(intent="goodbye", end_call=True, llm_allowed=False, objective="Link is on its way; goodbye",
                            fallback=_tpl("The link is on its way to this number. Thank you, have a great day!",
                                          "लिंक इसी नंबर पर भेजा जा रहा है। धन्यवाद, आपका दिन शुभ हो!",
                                          "Link isi number pe bheja ja raha hai. Thank you, aapka din accha ho!"))
        if ctx.decision == "later" or ctx.has("callback_later"):
            if not (_BUY.search(text) and intent != "deny"):
                note = d.get("callback_note", "")
                return TurnPlan(intent="callback", end_call=True, decision="callback already agreed",
                                context=f"The customer asked to be called back. What they said then: {note}",
                                objective="Confirm by repeating the exact day and time from what they said then (e.g. 'tomorrow at 7 PM'), "
                                          "say you'll call then, and say goodbye",
                                fallback=_tpl("Yes, I've noted it, I'll call you back then. Goodbye!",
                                              "जी, मैंने नोट कर लिया है, तभी कॉल करूँगा। नमस्ते!",
                                              "Ji, maine note kar liya hai, tabhi call karunga. Namaste!"))
        if ctx.decision in ("declined", "later"):
            if _BUY.search(text) and intent != "deny" and not _CANCEL.search(text) and ctx.decision == "declined":
                ctx.decision = "undecided"  # changed their mind: confirm once, then send
                d["pending"] = "confirm"
                return self._confirm_plan(lang, d)
            if self._asks(text, d) or self._faq_match(text) or intent == "price":
                return None
            return TurnPlan(intent="goodbye", end_call=True, llm_allowed=False, objective="Goodbye",
                            fallback=_tpl("Thank you for your time, have a good day!", "आपके समय के लिए धन्यवाद, आपका दिन शुभ हो!",
                                          "Aapke time ke liye thank you, aapka din accha ho!"))
        return None

    def _confirm_plan(self, lang: str = "en", d: dict | None = None) -> TurnPlan:
        """Before any link: one confirmation that states the complete terms (persona reviews: it never said
        that $8 lasts only 3 months). Once confirmed, a later clear yes sends the link (no confirm loop)."""
        if d is not None:
            d["terms_confirmed"] = "1"
        return TurnPlan(intent="confirm_close", llm_allowed=False,
                        facts={"offer": Fact(value=self._terms(lang), description="the full offer")},
                        objective="Confirm once, with the full terms, before sending the payment link", ask="confirm?",
                        fallback=self._cx(_tpl("Just to confirm: it's {offer}@@. Shall I send the payment link to this number?",
                                               "बस कन्फ़र्म कर दूँ: ये {offer} है@@। क्या मैं इसी नंबर पर पेमेंट लिंक भेज दूँ?",
                                               "Bas confirm kar doon: ye {offer} hai@@. Kya main isi number pe payment link bhej doon?")))

    def _link_plan(self, d: dict, lang: str, question: str = "") -> TurnPlan:
        d.pop("pending", None)
        d["outcome"] = "sold"
        if question:  # a yes with a question attached: answer it, then confirm the link
            p = self.p
            return TurnPlan(intent="link_sent", decision="payment link sent", end_call=False,
                            facts={"offer": Fact(value=self._terms(lang), description="the full offer", must_mention=False)},
                            context="\n\n".join([p.knowledge] + [f"- {f.answer}" for f in p.faqs if f.answer]),
                            lead=_tpl("Great, I'm sending the payment link by SMS to this number now.",
                                      "बढ़िया, मैं अभी इसी नंबर पर SMS से पेमेंट लिंक भेज रहा हूँ।",
                                      "Badhiya, main abhi isi number pe SMS se payment link bhej raha hoon."),
                            objective="You've ALREADY said the payment link is being sent by SMS. Now answer their question in "
                                      "one or two short sentences from the background, and ask if there's anything else. "
                                      "Don't mention any price",
                            # the template speaks the answer (approved FAQ line, or the honest "I don't have that"), never a bare "anything else?"
                            fallback={l: self._answer_text(question, l) or ANYTHING_ELSE[i] for i, l in enumerate(("en", "hi", "hinglish"))})
        site = self.p.official_site
        return TurnPlan(intent="link_sent", decision="payment link sent", end_call=False, llm_allowed=False,
                        facts={},
                        objective="Confirm the link is coming by SMS to this number; thank them",
                        fallback=_tpl("Great! I'm sending the payment link by SMS to this number now. Open it to pay securely" + (f"; if it doesn't arrive in a few minutes, the offer is also on {site}" if site else "") + ". Take your time, I'm here if you have a question.",
                                      "बढ़िया! मैं अभी इसी नंबर पर SMS से पेमेंट लिंक भेज रहा हूँ। उसे खोलकर सुरक्षित पेमेंट कर सकते हैं। आराम से देखिए, कोई सवाल हो तो मैं यहीं हूँ।",
                                      "Badhiya! Main abhi isi number pe SMS se payment link bhej raha hoon. Use khol ke secure payment kar sakte hain. Aaram se dekhiye, koi sawal ho to main yahin hoon."))

    def _price_plan(self, lang: str = "en", known: bool = False) -> TurnPlan:
        """The full offer, then (only if we still don't know) what they'd use it for. `known`: they already told us."""
        p = self.p
        q = (lambda l: "" if known else " " + p.discovery_question.get(l, ""))
        if p.terms.get(lang):  # the operator's phrase already says the normal price after the offer: don't say it twice
            fb = self._cx(_tpl("It's {offer}, that's {discount}@@." + q("en"), "ये {offer} है, यानी {discount}@@।" + q("hi"),
                               "Ye {offer} hai, yaani {discount}@@." + q("hinglish")))
        else:
            fb = self._cx(_tpl("Normally it's " + p.original_price + ", but for you it's {offer}, that's {discount}@@." + q("en"),
                               "वैसे ये " + p.original_price + " का है, पर आपके लिए {offer}, यानी {discount}@@।" + q("hi"),
                               "Waise ye " + p.original_price + " ka hai, par aapke liye {offer}, yaani {discount}@@." + q("hinglish")))
        return TurnPlan(
            intent="price", facts={"offer": Fact(value=self._terms(lang), description="the full offer"),
                                   "discount": self._offer()["discount"]},
            decision="quote offer price with its full terms, then learn what they need", llm_allowed=False,
            objective="Give the full offer terms" + ("" if known else " and ask what they'd use it for"),
            ask=None if known else "what would you use it for?",
            must_not_say=["prices not in the facts", "hidden charges"],
            # live call: it kept asking "shall I send the link?" without ever saying what you get. Ask about
            # their needs instead; the next turns tie a benefit to that, and the link comes after.
            fallback=fb)

    def _cx(self, tpl: dict[str, str]) -> dict[str, str]:
        """Fill the '@@' marker with ', and you can cancel any time' only when the product's own facts say so."""
        return {l: v.replace("@@", honest.cancel_tail(self.p, l)) for l, v in tpl.items()}
