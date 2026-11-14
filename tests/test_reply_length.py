"""Reply length budget (acie/reply_length): the 8B writes an extra empty opener in ~15% of turns (eval 15 Nov 2026)."""
from types import SimpleNamespace as NS

from acie.reply_length import budget, enough, fit

MED_ASK = NS(ask="ask if they want the link", facts={"offer": 1})
EXPR = lambda lang="en", v="med": NS(verbosity=v, language=lang, acknowledge_emotion=False)  # noqa: E731


def test_budget_follows_eval_rule():
    assert budget(MED_ASK, EXPR()) == (3, 48)
    assert budget(NS(ask=None, facts={}), EXPR("hi", "low")) == (1, 14)  # Hindi cap (ACIE_HI_MAX_WORDS)
    assert budget(MED_ASK, EXPR("hi")) == (3, 24)


def test_empty_opener_dropped_first_question_kept():
    r = ["Aapka sawal samajh gaya.", "Aapke liye {product} ka offer hai, jo pehle 3 mahino ke liye {offer} hai.",
         "Yoga aur zumba classes bhi milti hain roz subah aur shaam.", "Kya main link bhej doon?"]
    out = fit(r, MED_ASK, EXPR("hinglish"))
    assert out[0].startswith("Aapke liye") and out[-1] == "Kya main link bhej doon?"
    assert any("{offer}" in s for s in out)


def test_fact_sentence_never_dropped():
    r = ["जी, ये पहले तीन महीने का {offer} वाला ऑफर है।", "पेमेंट सिर्फ़ सुरक्षित लिंक से होता है और आप कभी भी cancel कर सकते हैं।",
         "आपको {benefit} भी मिलता है हर हफ़्ते।", "क्या लिंक भेज दूँ?"]
    out = fit(r, MED_ASK, EXPR("hi"))
    assert any("{offer}" in s for s in out) and any("{benefit}" in s for s in out) and out[-1] == "क्या लिंक भेज दूँ?"


def test_short_reply_untouched():
    r = ["It's {offer} for three months.", "Shall I send the link?"]
    assert fit(r, MED_ASK, EXPR()) == r


def test_enough_stops_after_the_question_at_budget():
    assert not enough(["It's {offer} for three months, with a free trainer session."], MED_ASK, EXPR())
    assert enough(["It's {offer} for three months, with a free trainer session.", "Classes run every single day.",
                   "Shall I send the link?"], MED_ASK, EXPR())
