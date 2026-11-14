"""The built-in call types, as data. Operators duplicate and edit these (data/voice/calltypes.json for their own).

Placeholders in any text: {agent} {seller} {product} {name}. Hindi lines avoid gendered verb forms on purpose.
"""

from __future__ import annotations

from .schema import CallType, Choice, Step


def T(en: str, hi: str, hinglish: str) -> dict[str, str]:
    return {"en": en, "hi": hi, "hinglish": hinglish}


_REC = T("This call is recorded.", "ये कॉल रिकॉर्ड हो रही है।", "Ye call record ho rahi hai.")
_CLOSE_OK = T("Thank you for your time{name}. Have a good day!",
              "आपके समय के लिए धन्यवाद। आपका दिन शुभ हो!",
              "Aapke time ke liye thank you. Aapka din accha ho!")
_CLOSE_NO = T("No problem at all. Thank you for your time, have a good day!",
              "कोई बात नहीं। आपके समय के लिए धन्यवाद, आपका दिन शुभ हो!",
              "Koi baat nahi. Aapke time ke liye thank you, aapka din accha ho!")


def _lead() -> CallType:
    return CallType(
        id="lead_qualification", name="Lead qualification", builtin=True, pack="lead_qualification",
        description="Understand the need, budget, authority and timeline; book a demo or a callback. No hard sell.",
        objective="Find out if this person is a fit (need, budget, decision maker, timeline) and book a demo if they are.",
        opening=T("Hi {name}, I'm {agent}, an AI assistant calling for {seller}. This call is recorded. I'm calling about {product}. "
                  "I'd just like to understand what you need, only a few quick questions, no selling. Is this a good time for two minutes?",
                  "नमस्ते {name}, मैं {agent} हूँ, {seller} की तरफ़ से एक AI असिस्टेंट। ये कॉल रिकॉर्ड हो रही है। {product} के बारे में कॉल किया है। "
                  "बस ये समझना चाहता हूँ कि आपको क्या चाहिए, कुछ छोटे सवाल, कोई सेल्स पिच नहीं। क्या अभी दो मिनट बात हो सकती है?",
                  "Namaste {name}, main {agent} hoon, {seller} ki taraf se ek AI assistant. Ye call record ho rahi hai. {product} ke baare mein call kiya hai. "
                  "Bas ye samajhna chahta hoon ki aapko kya chahiye, kuch chhote sawal, koi sales pitch nahi. Kya abhi do minute baat ho sakti hai?"),
        steps=[
            Step(id="need", answer_type="text", keywords=["need", "looking for", "requirement", "zaroorat", "chahiye", "जरूरत", "ज़रूरत"],
                 text=T("What are you mainly looking for help with?", "आप मुख्य रूप से किस काम के लिए मदद ढूँढ रहे हैं?",
                        "Aap mainly kis kaam ke liye madad dhoondh rahe hain?"), score_answered=20),
            Step(id="budget", answer_type="money", keywords=["budget", "kharcha", "spend", "बजट", "खर्चा"],
                 text=T("Roughly what budget do you have in mind for this?", "इसके लिए आपका लगभग कितना बजट है?",
                        "Iske liye aapka lagbhag kitna budget hai?"), score_answered=10),
            Step(id="authority", answer_type="yes_no", keywords=["decision", "decide", "faisla", "फ़ैसला", "फैसला"],
                 text=T("Will you be the one making the final decision on this?", "क्या इसका अंतिम फ़ैसला आप खुद लेते हैं?",
                        "Kya iska final decision aap khud lete hain?"), score={"true": 20, "false": 5}),
            Step(id="timeline", answer_type="choice", keywords=["when", "timeline", "kab tak", "कब तक"],
                 text=T("When are you looking to get started: right away, this month, or later on?",
                        "आप कब शुरू करना चाहेंगे: अभी, इसी महीने, या थोड़ा बाद में?",
                        "Aap kab shuru karna chahenge: abhi, isi mahine, ya thoda baad mein?"),
                 choices=[Choice(id="now", keywords=["right away", "immediately", "asap", "this week", "turant", "jaldi", "is hafte", "abhi shuru", "urgent", "aaj", "तुरंत", "जल्दी", "इसी हफ्ते"]),
                          Choice(id="this_month", keywords=["this month", "soon", "next month", "few weeks", "isi mahine", "agle mahine", "kuch hafte", "is mahine", "इसी महीने", "अगले महीने"]),
                          Choice(id="later", keywords=["later", "baad mein", "few months", "next quarter", "kuch mahine", "abhi nahi", "not now", "next year", "बाद में", "अभी नहीं"]),
                          Choice(id="exploring", keywords=["just looking", "exploring", "just checking", "dekh raha", "dekh rahe", "sirf dekh", "no plan", "सिर्फ देख", "बस देख"])],
                 score={"now": 30, "this_month": 20, "later": 8, "exploring": 0}),
            Step(id="book_demo", answer_type="date_time", required=False, skip_if="score < 40", action="book_demo", outcome="demo_booked", score_answered=0,
                 score={"booked": 20},
                 text=T("It sounds like a quick demo could help. When would suit you, say tomorrow evening or the day after?",
                        "लगता है एक छोटा डेमो आपके काम आ सकता है। आपके लिए कब ठीक रहेगा, जैसे कल शाम या परसों?",
                        "Lagta hai ek chhota demo aapke kaam aa sakta hai. Aapke liye kab theek rahega, jaise kal shaam ya parso?")),
        ],
        allowed_actions=["book_demo", "book_callback", "transfer_human", "opt_out"],
        success_outcomes=["demo_booked", "qualified", "interested", "callback_booked", "not_qualified"],
        closing=_CLOSE_OK, closing_declined=_CLOSE_NO, tone="friendly, curious and unhurried; asks, never pushes",
        max_call_s=300, selling_allowed=False, negotiation_allowed=False)


def _follow() -> CallType:
    return CallType(
        id="follow_up", name="Follow-up call", builtin=True, pack="follow_up",
        description="Continue from the last call: its open questions, the agreed price, the promised callback; then close or reschedule.",
        objective="Pick up where the last call ended, answer what was left open, and either move forward or agree a time to talk again.",
        opening=T("Hi {name}, I'm {agent}, an AI assistant calling for {seller}. This call is recorded. I'm following up on {product}. Is this a good time for two minutes?",
                  "नमस्ते {name}, मैं {agent} हूँ, {seller} की तरफ़ से एक AI असिस्टेंट। ये कॉल रिकॉर्ड हो रही है। {product} के बारे में फ़ॉलो-अप के लिए कॉल किया है। क्या अभी दो मिनट बात हो सकती है?",
                  "Namaste {name}, main {agent} hoon, {seller} ki taraf se ek AI assistant. Ye call record ho rahi hai. {product} ke baare mein follow-up ke liye call kiya hai. Kya abhi do minute baat ho sakti hai?"),
        steps=[
            Step(id="decision", answer_type="choice", keywords=["decide", "decision", "soch", "फ़ैसला", "फैसला"],
                 text=T("Have you had a chance to think it over? Are you ready to go ahead, would you like a little more time, or is it not for you?",
                        "क्या आपने इस बारे में सोचा? आप आगे बढ़ना चाहते हैं, थोड़ा और समय चाहिए, या ये आपके लिए नहीं है?",
                        "Kya aapne is baare mein socha? Aap aage badhna chahte hain, thoda aur time chahiye, ya ye aapke liye nahi hai?"),
                 choices=[Choice(id="already_done", outcome="paid", ends_call=True,
                                 keywords=["already paid", "payment kar diya", "pay kar diya", "paid already", "ho gaya", "kar diya payment", "done payment", "le liya", "already bought", "pehle hi le liya", "पेमेंट कर दिया", "ले लिया", "हो गया"]),
                          Choice(id="go_ahead", outcome="ready_to_buy",
                                 keywords=["go ahead", "ready", "proceed", "lena hai", "le lunga", "le lenge", "karna hai", "kar do", "bhej do", "send it", "tayyar", "तैयार", "आगे बढ़", "आगे बढ", "ले लूंगा", "लेना है", "lena chahta", "lena chahti",
                                             "~interested", "~yes", "~haan", "~chahiye", "~हाँ", "~हां", "~चाहिए", "~ok", "~okay", "~theek hai", "~ठीक है"]),
                          Choice(id="need_time", outcome="undecided",
                                 keywords=["more time", "need time", "thoda time", "time chahiye", "thoda aur time", "think", "soch", "sochna", "baad mein", "later", "not yet", "wait", "abhi nahi", "थोड़ा समय", "थोड़ा टाइम", "सोचना", "बाद में", "अभी नहीं", "रुकिए"]),
                          Choice(id="not_interested", outcome="declined", ends_call=True,
                                 keywords=["not interested", "no thanks", "nahi chahiye", "interest nahi", "nahi lena", "not for me", "cancel", "~no", "~nahi", "नहीं चाहिए", "नहीं लेना", "रुचि नहीं", "~नहीं"])]),
            Step(id="reschedule", answer_type="date_time", skip_if="decision != need_time", action="book_callback", outcome="callback_booked",
                 text=T("No problem. When should I call you again, say tomorrow evening or later this week?",
                        "कोई बात नहीं। मैं आपको दोबारा कब कॉल करूँ, जैसे कल शाम या इस हफ़्ते के आख़िर में?",
                        "Koi baat nahi. Main aapko dobara kab call karoon, jaise kal shaam ya is hafte ke end mein?")),
            Step(id="confirm_link", answer_type="yes_no", skip_if="decision != go_ahead", action="send_link", outcome="link_sent",
                 text=T("Shall I send you the payment link on SMS now?", "क्या मैं अभी आपको SMS पर पेमेंट लिंक भेज दूँ?",
                        "Kya main abhi aapko SMS par payment link bhej doon?")),
        ],
        allowed_actions=["send_link", "book_callback", "transfer_human", "opt_out"],
        success_outcomes=["link_sent", "ready_to_buy", "callback_booked", "paid", "declined"],
        closing=_CLOSE_OK, closing_declined=_CLOSE_NO, tone="warm, remembers the last call, never repeats itself, never pushes",
        max_call_s=240, selling_allowed=True, negotiation_allowed=False)


def _payment() -> CallType:
    return CallType(
        id="payment_reminder", name="Payment reminder", builtin=True, pack="steps",
        description="A polite reminder about a pending payment; get a date or hand over to a person.",
        objective="Find out if the payment is done, will be done, or needs help, and get a date.",
        opening=T("Hi {name}, I'm {agent}, an AI assistant calling for {seller}. This call is recorded. This is a friendly reminder about a pending payment for {product}. Do you have a minute?",
                  "नमस्ते {name}, मैं {agent} हूँ, {seller} की तरफ़ से एक AI असिस्टेंट। ये कॉल रिकॉर्ड हो रही है। {product} के एक पेंडिंग पेमेंट के बारे में रिमाइंडर है। क्या एक मिनट बात हो सकती है?",
                  "Namaste {name}, main {agent} hoon, {seller} ki taraf se ek AI assistant. Ye call record ho rahi hai. {product} ke ek pending payment ka friendly reminder hai. Kya ek minute baat ho sakti hai?"),
        steps=[
            Step(id="status", answer_type="choice",
                 text=T("Has the payment already been made, or are you planning to pay?", "क्या पेमेंट हो चुका है, या आप करने वाले हैं?",
                        "Kya payment ho chuka hai, ya aap karne wale hain?"),
                 choices=[Choice(id="paid", outcome="paid", ends_call=True, keywords=["already paid", "paid", "kar diya", "ho gaya", "done", "कर दिया", "हो गया", "चुका"]),
                          Choice(id="will_pay", keywords=["will pay", "pay karunga", "karunga", "karungi", "kar dunga", "kar dungi", "plan", "tomorrow", "kal", "pay", "कर दूंगा", "करूंगा", "करूँगा", "कर दूँगा"]),
                          Choice(id="dispute", outcome="human_requested", ends_call=True, keywords=["wrong", "galat", "not mine", "dispute", "mera nahi", "never took", "गलत", "मेरा नहीं"]),
                          Choice(id="cannot_pay", outcome="callback_requested", keywords=["can't pay", "cannot pay", "no money", "paise nahi", "nahi ho payega", "mushkil", "tangi", "पैसे नहीं", "नहीं हो पाएगा", "मुश्किल"])]),
            Step(id="pay_date", answer_type="date_time", skip_if="status == paid", action="take_payment_promise", outcome="promised_to_pay",
                 text=T("By when can you make the payment?", "आप पेमेंट कब तक कर सकेंगे?", "Aap payment kab tak kar payenge?")),
        ],
        allowed_actions=["take_payment_promise", "send_link", "book_callback", "transfer_human", "opt_out"],
        success_outcomes=["paid", "promised_to_pay", "callback_requested", "human_requested"],
        closing=_CLOSE_OK, closing_declined=_CLOSE_NO, tone="calm, respectful, never threatening",
        max_call_s=180, selling_allowed=False, negotiation_allowed=False)


def _feedback() -> CallType:
    return CallType(
        id="feedback_survey", name="Feedback survey", builtin=True, pack="steps",
        description="A short satisfaction survey after a purchase or a service.",
        objective="Get a rating, whether they would recommend it, and one comment.",
        opening=T("Hi {name}, I'm {agent}, an AI assistant calling for {seller}. This call is recorded. I'd like to ask two quick questions about {product}. Do you have a minute?",
                  "नमस्ते {name}, मैं {agent} हूँ, {seller} की तरफ़ से एक AI असिस्टेंट। ये कॉल रिकॉर्ड हो रही है। {product} के बारे में दो छोटे सवाल पूछने हैं। क्या एक मिनट है?",
                  "Namaste {name}, main {agent} hoon, {seller} ki taraf se ek AI assistant. Ye call record ho rahi hai. {product} ke baare mein do chhote sawal poochne hain. Kya ek minute hai?"),
        steps=[
            Step(id="rating", answer_type="number", keywords=["rating", "rate", "stars", "star", "out of"],
                 text=T("On a scale of 1 to 5, how would you rate it?", "1 से 5 के बीच, आप इसे कितनी रेटिंग देंगे?", "1 se 5 ke beech, aap ise kitni rating denge?")),
            Step(id="recommend", answer_type="yes_no", keywords=["recommend", "suggest"],
                 text=T("Would you recommend it to a friend?", "क्या आप इसे किसी दोस्त को सुझाएँगे?", "Kya aap ise kisi dost ko recommend karenge?")),
            Step(id="comment", answer_type="text", required=False,
                 text=T("Is there one thing we could do better?", "ऐसी कोई एक बात जो हम बेहतर कर सकें?", "Aisi koi ek baat jo hum behtar kar sakein?")),
        ],
        allowed_actions=["transfer_human", "opt_out"], success_outcomes=["feedback_given", "completed"], closing=_CLOSE_OK,
        closing_declined=_CLOSE_NO, tone="grateful and brief", max_call_s=150, selling_allowed=False, negotiation_allowed=False)


def _appointment() -> CallType:
    return CallType(
        id="appointment_booking", name="Appointment booking", builtin=True, pack="steps",
        description="Book a visit, a consultation or a service slot.",
        objective="Agree what the appointment is for and a date and time.",
        opening=T("Hi {name}, I'm {agent}, an AI assistant calling for {seller}. This call is recorded. I'm calling to help you book an appointment for {product}. Do you have a minute?",
                  "नमस्ते {name}, मैं {agent} हूँ, {seller} की तरफ़ से एक AI असिस्टेंट। ये कॉल रिकॉर्ड हो रही है। {product} के लिए अपॉइंटमेंट बुक करने में मदद के लिए कॉल किया है। क्या एक मिनट है?",
                  "Namaste {name}, main {agent} hoon, {seller} ki taraf se ek AI assistant. Ye call record ho rahi hai. {product} ke liye appointment book karne mein madad ke liye call kiya hai. Kya ek minute hai?"),
        steps=[
            Step(id="purpose", answer_type="text", keywords=["for", "purpose", "reason", "liye"],
                 text=T("What would you like the appointment for?", "आप अपॉइंटमेंट किस काम के लिए चाहते हैं?", "Aap appointment kis kaam ke liye chahte hain?")),
            Step(id="slot", answer_type="date_time", action="book_appointment", outcome="appointment_booked",
                 text=T("When would suit you, which day and time?", "आपके लिए कब ठीक रहेगा, कौन सा दिन और समय?", "Aapke liye kab theek rahega, kaun sa din aur time?")),
        ],
        allowed_actions=["book_appointment", "book_callback", "transfer_human", "opt_out"],
        success_outcomes=["appointment_booked", "callback_booked", "declined"], closing=_CLOSE_OK, closing_declined=_CLOSE_NO,
        tone="helpful and efficient", max_call_s=180, selling_allowed=False, negotiation_allowed=False)


def _sales() -> CallType:
    return CallType(
        id="sales_pitch", name="Sales pitch", builtin=True, pack="sales_pitch",
        description="Today's product sales call: pitch, handle objections, close at a profitable price.",
        objective="Sell the product honestly at a profitable price (link on consent).",
        success_outcomes=["link_sent", "callback_requested", "declined"],
        allowed_actions=["send_link", "book_callback", "transfer_human", "opt_out"],
        tone="persuasive with real value, natural, never pushy", max_call_s=420, selling_allowed=True, negotiation_allowed=True)


def builtin_call_types() -> dict[str, CallType]:
    out = {}
    for f in (_sales, _lead, _follow, _payment, _feedback, _appointment):
        ct = f()
        out[ct.id] = ct
    return out
