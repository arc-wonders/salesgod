"""Demo domain: retail-bank voice agent (account support + one compliant sales offer).

Shows the pattern every domain follows:
  classifier -> FSM state in session.dialog -> backend call -> TurnPlan with
  facts as slots + safe fallback templates in each language.
"""

from __future__ import annotations

import random

from ...schemas import Fact, SessionState, TurnInput, TurnPlan
from ...text import inr, parse_amount
from ..engine import KeywordIntentClassifier

# --------------------------------------------------------------------------- mock backend


class MockBankAPI:
    def __init__(self) -> None:
        self.accounts = {
            "c1": {"name": "Rahul", "balance": 82_500, "daily_limit": 40_000, "used_today": 15_000,
                   "card_eligible": True, "card_limit": 150_000},
            "c2": {"name": "Priya", "balance": 12_000, "daily_limit": 25_000, "used_today": 0,
                   "card_eligible": False, "card_limit": 0},
        }

    async def account(self, cid: str) -> dict:
        return self.accounts.get(cid, {"name": "", "balance": 0, "daily_limit": 0, "used_today": 0,
                                       "card_eligible": False, "card_limit": 0})

    async def transfer(self, cid: str, amount: int) -> str:
        acc = self.accounts[cid]
        acc["balance"] -= amount
        acc["used_today"] += amount
        return f"TXN{random.randint(100000, 999999)}"


# --------------------------------------------------------------------------- intents

# Keyword syntax: see KeywordIntentClassifier (whole words; "x*" = prefix; "=x" = whole
# utterance; (kw, weight)). Text and keywords are normalized (nukta/chandrabindu folded),
# so "हाँ"/"हां" and "हज़ार"/"हजार" are the same keyword.
KEYWORDS = {
    "clarify": [
        "samajh nahi", "samjha nahi", "samjhi nahi", "samaj nahi", "samajh nai", "samajh nhi",
        "samjh nahi", "nahi samjha", "nahi samjhi", "nahi samajh", "samajh mein nahi", "samajh me nahi",
        "kya matlab", "matlab kya", "kya bola", "kya kaha", "kya bole", "kya bol rahe", "kya bol rahi",
        "phir se bol*", "phir se bata*", "phir se samjha*", "fir se bol*", "fir se bata*", "firse bol*",
        "dobara bol*", "dobara bata*", "dubara bol*", "ek baar aur bol*", "ek baar phir se",
        "awaaz nahi", "awaz nahi", "sunai nahi",
        "what do you mean", "didnt understand", "dont understand", "did not understand",
        "do not understand", "not clear", "repeat", "say again", "say that again", "come again",
        "sorry what", "pardon", "which one", "i dont get it", "didnt get that", "didnt catch",
        "couldnt hear", "cant hear you", "which limit", "kaun sa wala", "konsa wala",
        "समझ नहीं", "समझा नहीं", "समझी नहीं", "नहीं समझ*", "समझ में नहीं", "क्या मतलब", "मतलब क्या",
        "क्या बोला", "क्या बोले", "क्या कहा", "क्या बोल रहे", "फिर से बोल*", "फिर से बता*",
        "फिर से समझा*", "दोबारा बोल*", "दोबारा बता*", "एक बार और", "आवाज नहीं", "सुनाई नहीं",
        "कौन सा वाला", "कौनसा वाला", "रिपीट",
        "=matlab", "=matlab kya", "=मतलब", "=kya", "=क्या", "=huh", "=what", "=sorry", "=hain", "=हैं",
        "=phir se", "=fir se", "=dobara", "=फिर से", "=दोबारा", "=kaun sa", "=konsa", "=कौन सा",
        "=कौनसा", "=sorry what", "=what what", "=excuse me", "=haan kya", "=हां क्या",
    ],
    "human_agent": [
        "agent", "human", "insaan", "insan", "manager", "real person", "representative",
        "customer care", "customer service", "customer support", "executive", "supervisor",
        "operator", "kisi aadmi", "aadmi se", "kisi insaan", "kisi se baat", "kisi aur se baat",
        "baat karao", "baat karwao", "baat karvao", "baat karwa do", "talk to a person",
        "talk to someone", "speak to someone", "live person", "connect me to",
        # out-of-scope card servicing -> a human handles it, never the card *offer*
        "card block", "block card", "block my card", "card band", "card blocked", "lost my card",
        "card lost", "card kho", "card chori", "stolen", "fraud", "complaint", "debit card",
        "atm card", "card ka pin", "card pin",
        "इंसान", "किसी आदमी", "आदमी से", "कस्टमर केयर", "कस्टमर सर्विस", "कस्टमर सपोर्ट", "एजेंट",
        "मैनेजर", "अधिकारी", "ऑपरेटर", "बात कराओ", "बात करवाओ", "बात करवा दो", "किसी से बात",
        "किसी और से बात", "कार्ड ब्लॉक", "ब्लॉक कर*", "कार्ड बंद", "कार्ड खो", "कार्ड चोरी", "चोरी",
        "फ्रॉड", "धोखा", "शिकायत", "कंप्लेंट", "डेबिट कार्ड", "एटीएम कार्ड",
    ],
    "affirm": [
        ("haan", 0.5), ("han", 0.5), ("ha", 0.5), ("haa", 0.5), ("ji", 0.5), ("ok", 0.5),
        ("okay", 0.5), ("okk", 0.5), ("okey", 0.5), ("achha", 0.5), ("acha", 0.5), ("accha", 0.5),
        "=k", "=ok", "=okay", "=haan", "=han", "=ha", "=ji",
        "yes", "yeah", "yea", "yep", "yup", "sure", "haan ji", "haanji", "ji haan", "ji ha",
        "theek hai", "thik hai", "theek h", "thik h", "confirm", "confirmed", "kar do", "kardo",
        "kar dijiye", "kar dijie", "kar dena", "bhej do", "bhejdo", "bhej dijiye", "bhej dijie",
        "send it", "transfer it", "do it", "go ahead", "proceed", "please do", "sounds good",
        "alright", "all right", "correct", "of course", "definitely", "absolutely", "why not",
        "chalega", "chalo", "bilkul", "zaroor", "jarur", "sahi hai", "done", "apply kar*",
        ("हां", 0.5), ("हा", 0.5), "जी हां", "हां जी", "ठीक है", "ठीक हैं", "कर दो", "कर दीजिए", "कर दीजिये",
        "कर दें", "भेज दो", "भेज दीजिए", "भेज दीजिये", "यस", "बिल्कुल", "बिलकुल", "जरूर", "सही है",
        "चलेगा", "कन्फर्म", "डन", ("जी", 0.5), ("ओके", 0.5), ("ओक", 0.5), ("अच्छा", 0.5),
        "=ओके", "=जी", "=हां",
    ],
    "deny": [
        "nahi", "nahin", "nhi", "nai", "no", "nope", "nah", "mat", "matt", "cancel", "rehne do",
        "rahne do", "rehene do", "not interested", "dont", "do not", "not now", "nahi chahiye",
        "nahi karna", "mat karo", "stop", "never mind", "nevermind", "leave it", "chhodo", "chodo",
        "chhod do", "chod do", "ruko", "ruk jao", "abhi nahi", "interest nahi",
        "नहीं", "नही", "मत", "रहने दो", "कैंसल", "नो", "छोड़ो", "छोडो", "छोड़ दो", "छोड दो", "रुको",
        "रुकिए", "नहीं चाहिए", "मना",
    ],
    "greet": ["hello", "hi", "hey", "helo", "hallo", "hellow", "namaste", "namaskar", "good morning",
              "good afternoon", "good evening", "pranam", "नमस्ते", "नमस्कार", "हेलो", "हैलो", "हलो",
              "हाय", "प्रणाम"],
    "check_balance": ["balance", "balence", "ballance", "belance", "balanc", "kitna paisa",
                      "kitne paise", "paise kitne", "kitna paise", "account mein kitna",
                      "account me kitna", "how much money", "how much do i have", "बैलेंस", "बेलेंस",
                      "बैलन्स", "शेष", "कितने पैसे", "कितना पैसा", "पैसे कितने", "खाते में कितना",
                      "अकाउंट में कितना"],
    "transfer": ["transfer", "transfar", "tranfer", "trasfer", "transfor", "tansfer", "transffer",
                 "bhej*", "send", "sending", "neft", "imps", "rtgs", "paise dal*", "paisa dal*",
                 "ट्रांसफर", "ट्रान्सफर", "ट्रांसफार", "ट्रांस्फर", "भेज*", "सेंड",
                 "ट्रांसवर", "ट्रान्सवर", "ट्रांसफ़र", "ट्रंसफर"],  # Whisper mishearings
    "transfer_limit": ["limit", "limt", "limet", "lmit", "transfer limit", "daily limit", "maximum",
                       "max kitna", "kitna bhej sak*", "kitne bhej sak*", "how much can i send",
                       "how much can i transfer", "सीमा", "लिमिट", "कितना भेज सक*", "कितने भेज सक*",
                       "kitna transfer kar sak*", "कितना ट्रांसफर कर सक*"],
    "card_offer": ["credit card", "card", "cradit card", "kredit card", "card offer", "offer",
                   "कार्ड", "क्रेडिट कार्ड", "ऑफर"],
    "goodbye": ["bye", "bye bye", "thank you", "thankyou", "thanks", "thank u", "thx",
                "dhanyavad", "dhanyawad", "dhanyabad", "shukriya", "thats all", "bas itna hi",
                "bas itna", "nothing else", "aur kuch nahi", "बाय", "थैंक यू", "थैंक्स", "थैंकयू",
                "धन्यवाद", "शुक्रिया", "बस इतना ही", "और कुछ नहीं"],
}
MASKS = {
    "deny": ["koi baat nahi", "koi nahi", "no problem", "not a problem", "koi problem nahi",
             "problem nahi", "dikkat nahi", "koi dikkat nahi", "kyun nahi", "kyon nahi", "why not",
             "no worries", "dont worry", "dont mind", "no doubt",
             "कोई बात नहीं", "कोई प्रॉब्लम नहीं", "प्रॉब्लम नहीं", "दिक्कत नहीं", "कोई दिक्कत नहीं",
             "क्यों नहीं", "कोई नहीं"],
    "affirm": ["not interested", "not ok", "not okay", "theek nahi", "thik nahi", "ok nahi",
               "ठीक नहीं", "ओके नहीं"],
    "transfer": ["transfer limit", "transfar limit", "sending limit", "send limit",
                 "ट्रांसफर लिमिट", "भेजने की लिमिट", "भेजने की सीमा", "kitna bhej sak*", "kitne bhej sak*",
                 "how much can i send", "how much can i transfer", "कितना भेज सक*", "कितने भेज सक*",
                 "kitna transfer kar sak*", "कितना ट्रांसफर कर सक*"],
    "transfer_limit": ["card limit", "कार्ड लिमिट", "कार्ड की लिमिट", "card ki limit", "card ka limit"],
    "card_offer": ["debit card", "atm card", "card block", "block card", "block my card",
                   "card blocked", "card band", "card lost", "lost my card", "card kho", "card chori",
                   "card number", "card ka pin", "card pin", "डेबिट कार्ड", "एटीएम कार्ड",
                   "कार्ड ब्लॉक", "कार्ड बंद", "कार्ड खो", "कार्ड चोरी", "कार्ड नंबर"],
}
GATED = {"affirm": "pending", "deny": "pending"}
# deny before affirm on ties: when unsure, nothing gets executed.
PRIORITY = ["clarify", "human_agent", ("deny", "affirm")]


def _plan(intent: str, objective: str, fallback: dict[str, str], **kw) -> TurnPlan:
    return TurnPlan(intent=intent, objective=objective, fallback=fallback, **kw)


class BankDemoDomain:
    name = "bank_demo"

    def __init__(self, api: MockBankAPI | None = None):
        self.api = api or MockBankAPI()
        self.classifier = KeywordIntentClassifier(KEYWORDS, gated=GATED, priority=PRIORITY, masks=MASKS,
                                                  weak=["greet"])

    async def plan(self, intent: str, turn: TurnInput, session: SessionState) -> TurnPlan:
        d = session.dialog
        acc = await self.api.account(turn.customer_id)
        pending = d.get("pending")

        topic = d.pop("topic", None)  # set again below only by transfer-related plans

        # A new amount while we're in a transfer flow ("अच्छा दस हज़ार कर दो फिर", "haan 10000 bhej do",
        # "nahi, das hazaar") is a new transfer request: re-quote and re-confirm, never execute the
        # previously pending amount.
        in_transfer_flow = pending == "transfer" or d.get("pending_slot") == "amount" or topic == "transfer"
        if (in_transfer_flow and intent in ("unknown", "affirm", "deny", "transfer")
                and parse_amount(turn.text) is not None):
            intent = "transfer"
            d.pop("pending", None)
            pending = None

        # Safety: only start a card application on a clean yes. "ok then balance please" while the
        # offer is pending is a topic change, not consent.
        if intent == "affirm" and pending == "card_offer":
            scores = self.classifier.scores(turn.text, session)
            other = max((i for i in scores if i not in ("affirm", "deny") and scores[i] > 0),
                        key=lambda i: scores[i], default=None)
            if other is not None:
                d.pop("pending", None)
                pending, intent = None, other

        if intent == "clarify" and "last_plan" in d:
            p = TurnPlan.model_validate_json(d["last_plan"])
            prefix = "The customer did not understand. Re-explain more simply: "
            p.objective = p.objective if p.objective.startswith(prefix) else prefix + p.objective
            return p

        if intent == "affirm" and pending:
            return await self._on_affirm(pending, turn, session, acc)
        if intent == "deny" and pending:
            d.pop("pending", None)
            d.pop("amount", None)
            if pending == "card_offer":
                d["card_declined"] = "1"
                return _plan("card_declined", "Accept the decline politely and ask if there is anything else",
                             {"en": "No problem at all. Is there anything else I can help with?",
                              "hi": "कोई बात नहीं। क्या मैं किसी और चीज़ में मदद कर सकता हूँ?",
                              "hinglish": "Koi baat nahi. Kya main kisi aur cheez mein help kar sakta hoon?"},
                             must_not_say=["pressure to reconsider", "repeat the offer", "limited time urgency"],
                             ask="anything else?")
            return _plan("cancelled", "Confirm nothing was done and ask if there is anything else",
                         {"en": "Okay, I have not made any transfer. Anything else?",
                          "hi": "ठीक है, मैंने कोई ट्रांसफर नहीं किया। और कुछ?",
                          "hinglish": "Theek hai, maine koi transfer nahi kiya. Aur kuch?"},
                         ask="anything else?")

        if intent == "greet":
            return _plan("greet", "Greet the customer by name and ask how you can help",
                         {"en": "Hello {name}, how can I help you today?",
                          "hi": "नमस्ते {name}, मैं आपकी क्या मदद कर सकता हूँ?",
                          "hinglish": "Namaste {name}, main aapki kya help kar sakta hoon?"},
                         facts={"name": Fact(value=acc["name"], description="customer's first name")},
                         ask="how can I help?")

        if intent == "check_balance":
            return _plan("check_balance", "Tell the customer their current account balance",
                         {"en": "Your current balance is {balance}.",
                          "hi": "आपका मौजूदा बैलेंस {balance} है।",
                          "hinglish": "Aapka current balance {balance} hai."},
                         facts={"balance": Fact(value=inr(acc["balance"]), description="available balance")})

        if intent in ("transfer_limit",):
            d["topic"] = "transfer"
            remaining = acc["daily_limit"] - acc["used_today"]
            return _plan("transfer_limit", "Explain the daily transfer limit and what is left today",
                         {"en": "Your daily transfer limit is {daily_limit}, and {remaining_limit} is left for today.",
                          "hi": "आपकी डेली ट्रांसफर लिमिट {daily_limit} है, और आज {remaining_limit} बाकी है।",
                          "hinglish": "Aapki daily transfer limit {daily_limit} hai, aur aaj {remaining_limit} baaki hai."},
                         facts={"daily_limit": Fact(value=inr(acc["daily_limit"]), description="daily transfer limit"),
                                "remaining_limit": Fact(value=inr(remaining), description="limit remaining today")})

        if intent == "transfer":
            return self._transfer(turn, session, acc)

        if intent == "card_offer":
            if d.get("card_declined"):
                return _plan("card_declined_again", "Customer already declined the card in this call; only answer, do not pitch",
                             {"en": "Sure. You had said no earlier, so I won't push it. Anything else?",
                              "hi": "ज़रूर। आपने पहले मना किया था, तो मैं ज़ोर नहीं दूँगा। और कुछ?",
                              "hinglish": "Zaroor. Aapne pehle mana kiya tha, toh main push nahi karunga. Aur kuch?"},
                             must_not_say=["repeat the offer"])
            if not acc["card_eligible"]:
                return _plan("card_ineligible", "Say the customer is not eligible for a credit card right now",
                             {"en": "Right now you are not eligible for a credit card. I can check again later.",
                              "hi": "अभी आप क्रेडिट कार्ड के लिए eligible नहीं हैं। मैं बाद में फिर से देख सकता हूँ।",
                              "hinglish": "Abhi aap credit card ke liye eligible nahi hain. Main baad mein phir check kar sakta hoon."},
                             decision="not eligible", must_not_say=["reasons from credit score", "promises of approval"])
            d["pending"] = "card_offer"
            return _plan("card_offer", "Present the pre-approved credit card offer honestly and ask if interested",
                         {"en": "You are pre-approved for a credit card with a limit of {card_limit} and {card_benefit}. Would you like to apply?",
                          "hi": "आपके लिए {card_limit} लिमिट वाला क्रेडिट कार्ड pre-approved है, {card_benefit}। क्या आप apply करना चाहेंगे?",
                          "hinglish": "Aapke liye {card_limit} limit ka credit card pre-approved hai, {card_benefit}. Kya aap apply karna chahenge?"},
                         facts={"card_limit": Fact(value=inr(acc["card_limit"]), description="pre-approved credit limit"),
                                "card_benefit": Fact(value="no annual fee in the first year", description="main benefit")},
                         must_not_say=["guaranteed savings", "limited time pressure", "fees other than stated"],
                         ask="would you like to apply?")

        if intent == "human_agent":
            return _plan("human_agent", "Tell the customer you are connecting them to a human agent now",
                         {"en": "Sure, I am connecting you to an agent now.",
                          "hi": "ज़रूर, मैं आपको अभी एक एजेंट से जोड़ रहा हूँ।",
                          "hinglish": "Zaroor, main aapko abhi agent se connect kar raha hoon."},
                         decision="handoff_to_human", end_call=True)

        if intent == "goodbye":
            return _plan("goodbye", "Thank the customer and close the call warmly",
                         {"en": "Thank you for calling. Have a good day!",
                          "hi": "कॉल करने के लिए धन्यवाद। आपका दिन शुभ हो!",
                          "hinglish": "Call karne ke liye thank you. Aapka din accha ho!"},
                         end_call=True)

        return _plan("unknown", "Say you did not catch that and list what you can help with: balance, transfers, limits, credit card",
                     {"en": "Sorry, I didn't get that. I can help with balance, transfers, limits or a credit card.",
                      "hi": "माफ़ कीजिए, मैं समझ नहीं पाया। मैं बैलेंस, ट्रांसफर, लिमिट या क्रेडिट कार्ड में मदद कर सकता हूँ।",
                      "hinglish": "Sorry, main samjha nahi. Main balance, transfer, limit ya credit card mein help kar sakta hoon."},
                     ask="what would you like to do?")

    # ---------------------------------------------------------------- transfer FSM

    def _transfer(self, turn: TurnInput, session: SessionState, acc: dict) -> TurnPlan:
        d = session.dialog
        d["topic"] = "transfer"
        amount = parse_amount(turn.text)
        if amount is None:
            d["pending_slot"] = "amount"
            return _plan("transfer_ask_amount", "Ask how much money the customer wants to transfer",
                         {"en": "Sure. How much would you like to transfer?",
                          "hi": "ज़रूर। आप कितने पैसे ट्रांसफर करना चाहते हैं?",
                          "hinglish": "Zaroor. Aap kitne paise transfer karna chahte hain?"},
                         ask="how much?")
        d.pop("pending_slot", None)
        remaining = acc["daily_limit"] - acc["used_today"]
        facts = {"requested_amount": Fact(value=inr(amount), description="amount the customer asked to send")}

        if amount > acc["balance"]:
            return _plan("transfer_insufficient", "Explain the transfer cannot go through because the balance is too low",
                         {"en": "I can't send {requested_amount} because your balance is {balance}.",
                          "hi": "मैं {requested_amount} नहीं भेज सकता क्योंकि आपका बैलेंस {balance} है।",
                          "hinglish": "Main {requested_amount} nahi bhej sakta kyunki aapka balance {balance} hai."},
                         facts={**facts, "balance": Fact(value=inr(acc["balance"]), description="available balance")},
                         decision="transfer declined: insufficient balance")

        if amount > remaining:
            d["pending"], d["amount"] = "transfer", str(remaining)
            return _plan("transfer_over_limit",
                         "Explain the transfer exceeds today's remaining limit and offer to send the remaining amount instead",
                         {"en": "Your daily limit is {daily_limit} and only {remaining_limit} is left today, so I can't send {requested_amount}. Should I send {remaining_limit} instead?",
                          "hi": "आपकी डेली लिमिट {daily_limit} है और आज सिर्फ़ {remaining_limit} बाकी है, इसलिए {requested_amount} नहीं भेज सकते। क्या मैं {remaining_limit} भेज दूँ?",
                          "hinglish": "Aapki daily limit {daily_limit} hai aur aaj sirf {remaining_limit} baaki hai, isliye {requested_amount} nahi bhej sakte. Kya main {remaining_limit} bhej doon?"},
                         facts={**facts,
                                "daily_limit": Fact(value=inr(acc["daily_limit"]), description="daily transfer limit"),
                                "remaining_limit": Fact(value=inr(remaining), description="limit remaining today")},
                         decision="transfer of requested amount declined: exceeds remaining daily limit",
                         allowed_actions=["offer to send remaining_limit instead"],
                         ask="send the remaining amount instead?")

        d["pending"], d["amount"] = "transfer", str(amount)
        return _plan("transfer_confirm", "Ask the customer to confirm the transfer amount",
                     {"en": "Just to confirm, you want to transfer {requested_amount}?",
                      "hi": "कन्फ़र्म कर दूँ, आप {requested_amount} ट्रांसफर करना चाहते हैं?",
                      "hinglish": "Confirm kar doon, aap {requested_amount} transfer karna chahte hain?"},
                     facts=facts, ask="confirm?")

    async def _on_affirm(self, pending: str, turn: TurnInput, session: SessionState, acc: dict) -> TurnPlan:
        d = session.dialog
        d.pop("pending", None)
        if pending == "transfer":
            amount = int(d.pop("amount"))
            ref = await self.api.transfer(turn.customer_id, amount)
            return _plan("transfer_done", "Confirm the transfer succeeded and give the reference number",
                         {"en": "Done. {amount} has been transferred. Your reference number is {reference}.",
                          "hi": "हो गया। {amount} ट्रांसफर हो गए हैं। आपका रेफ़रेंस नंबर {reference} है।",
                          "hinglish": "Ho gaya. {amount} transfer ho gaye hain. Aapka reference number {reference} hai."},
                         facts={"amount": Fact(value=inr(amount), description="amount transferred"),
                                "reference": Fact(value=ref, description="transaction reference number")},
                         decision="transfer executed")
        if pending == "card_offer":
            return _plan("card_apply_started", "Say the application has started and an SMS link is coming to finish it",
                         {"en": "Great, I have started your application. You'll get an SMS link to complete it.",
                          "hi": "बढ़िया, मैंने आपकी एप्लिकेशन शुरू कर दी है। इसे पूरा करने के लिए आपको SMS पर लिंक मिलेगा।",
                          "hinglish": "Badhiya, maine aapki application start kar di hai. Complete karne ke liye SMS pe link aayega."},
                         decision="application started", must_not_say=["approval is guaranteed"])
        return _plan("unknown", "Ask what the customer would like to do", {"en": "What would you like to do?",
                     "hi": "आप क्या करना चाहेंगे?", "hinglish": "Aap kya karna chahenge?"})

