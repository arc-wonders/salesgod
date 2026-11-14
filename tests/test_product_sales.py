"""Product-sales domain: the policy decides (consent, compliance, closing), the conversation turn
answers everything else. These run on the template backend (no LLM), so a conversation turn shows
its fallback: the approved FAQ answer if one matches, else "could you say that once more?"."""

from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import TurnInput


def T(text):
    return TurnInput(session_id="s", customer_id="+910000000000", text=text)


async def run(lines, name="Yash", nlu=None, world=None):
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(Product(), customer_name=name, world=world)), nlu=nlu)
    sess = pipe.session(T(""))
    opening = await pipe.decision.decide("greet", T(""), sess)
    sess.context.record_turn(0, "", opening.intent, False)
    out, last = [opening.intent], None
    for line in lines:
        last = await pipe.run_turn(T(line))
        out.append(last.plan.intent)
    return out, last, pipe


class FakeNLU:
    def __init__(self, table):
        self.table = table

    async def understand(self, text, history, intents, about=""):
        from acie.understanding import Understanding
        return Understanding(**self.table[text]) if text in self.table else None


# ---------------------------------------------------------------- the sale path and consent

async def test_happy_path_states_full_terms_and_confirms_before_link():
    out, last, _ = await run(["हाँ बोलिए", "हाँ बताइए", "हाँ भेज दो"])
    assert out == ["opening", "pitch", "price", "link_sent"]
    out, last, _ = await run(["हाँ बोलिए", "हाँ", "हाँ ठीक है"])
    assert out[-1] in ("confirm_close", "link_sent")


async def test_price_always_says_how_long_and_the_price_after():
    out, last, _ = await run(["kitne ka hai ye"])
    assert out[-1] == "price" and "3" in last.text and "$20" in last.text and "$8" in last.text
    out, last, _ = await run(["yes tell me", "okay fine I guess"])
    assert out[-1] in ("price", "confirm_close")


async def test_confirmation_states_terms():
    out, last, _ = await run(["yes tell me", "how much", "okay"])
    assert out[-1] in ("confirm_close", "link_sent")
    out, last, _ = await run(["yes tell me", "sure let's do it"])  # yes before any price: confirm with terms
    assert out[-1] == "confirm_close" and "then $20" in last.text


async def test_questions_and_hypotheticals_are_never_consent():
    out, _, _ = await run(["हाँ बोलिए", "कितने का है", "UPI se ho jayega?"])
    assert out[-1] not in ("confirm_close", "link_sent")
    out, _, _ = await run(["हाँ बोलिए", "कितने का है", "अगर मैं washing machine खरीदूँ तो तुम ले लोगे?"])
    assert out[-1] not in ("confirm_close", "link_sent")
    q = "मेरे पास पहले से Claude है, क्या मैं ये offer अभी भी ले सकता हूँ?"
    out, _, _ = await run(["हाँ बोलिए", "कितने का है", q],
                          nlu=FakeNLU({q: dict(intent="affirm", buying_signal=0.8, question_topic="offer_terms")}))
    assert out[-1] not in ("confirm_close", "link_sent")


async def test_model_heard_yes_is_confirmed_once():
    line = "कोशिश तो करना चाहते हैं"
    out, _, _ = await run(["हाँ बोलिए", "कितने का है", line],
                          nlu=FakeNLU({line: dict(intent="affirm", buying_signal=0.85, gist="wants to try it")}))
    assert out[-1] in ("confirm_close", "link_sent")


async def test_one_soft_reoffer_then_respect_no():
    # OWNER UPDATE 2: the first soft no gets a friendly "what is holding you back?", then ONE re-offer, then the no is respected
    out, last, _ = await run(["yes tell me", "no not interested", "nahi chahiye", "nahi chahiye"])
    assert out[-3:] == ["reason_ask", "reoffer", "declined"] and last.plan.end_call


async def test_questions_are_not_refusals():
    q = "बाद में बढ़ जाएगा, धोखा तो नहीं है?"
    out, _, _ = await run(["हाँ बोलिए", "कितने का है", q], nlu=FakeNLU({q: dict(intent="deny", question_topic="offer_terms")}))
    assert out[-1] not in ("reoffer", "declined")


async def test_personal_and_small_talk_are_not_refusals():
    line = "दिन भर घर में अकेले बैठे रहते हैं, बहुत अकेलापन लगता है"
    out, last, _ = await run(["हाँ बोलिए", line], nlu=FakeNLU({line: dict(intent="personal", gist="feels lonely")}))
    assert out[-1] == "personal" and not last.plan.end_call
    w = "मौसम कैसा है, ठंड है क्या"
    out, last, _ = await run(["हाँ बोलिए", w], nlu=FakeNLU({w: dict(intent="small_talk", question_topic="general")}))
    assert out[-1] == "small_talk" and not last.plan.end_call


# ---------------------------------------------------------------- conversation turns

async def test_faq_question_gets_the_approved_answer_without_llm():
    out, last, _ = await run(["yes tell me", "Why is it so cheap? Is it fake?"])
    assert out[-1] == "answer" and "cancel any time" in last.text


async def test_off_topic_is_looked_up_with_interim_line():
    class World:
        asked = []

        async def lookup(self, q):
            self.asked.append(q)
            return "Weather now in Mumbai: 31°C, humid."
    world = World()
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(Product(), world=world)))
    spoken = []
    await pipe.run_turn(T("yes tell me"))
    r = await pipe.run_turn(T("what is the weather in Mumbai today?"), on_sentence=spoken.append)
    assert r.plan.intent == "small_talk" and r.plan.reference and r.plan.general_knowledge and world.asked
    assert spoken and "check" in spoken[0].lower()


async def test_conversation_turn_gets_full_offer_and_knowledge():
    out, last, _ = await run(["yes tell me", "do you train your models on my data?"])
    assert out[-1] == "answer"
    assert "offer" in last.plan.facts and "then $20" in last.plan.facts["offer"].value
    assert "privacy" in (last.plan.context or "").lower()


async def test_rushing_complaint_slows_down_then_tailors():
    c = "आप बहुत जल्दी कर रहे हो"
    a = "हाँ मैं कॉलेज में ChatGPT यूज़ करता हूँ"
    out, last, _ = await run(["हाँ बोलिए", "कितने का है", c, a],
                             nlu=FakeNLU({c: dict(intent="slow_down", frustration=0.6), a: dict(intent="affirm", buying_signal=0.4)}))
    assert out[-2] == "slow_down" and out[-1] == "answer"


async def test_hello_mid_call():
    out, last, _ = await run(["हाँ बोलिए", "हेलो"])
    assert out[-1] == "hello" and "सुन रहा" in last.text


# ---------------------------------------------------------------- whole-call memory

async def test_after_sale_no_reselling():
    out, last, _ = await run(["yes tell me", "how much", "yes send the link", "how much was it again?"])
    assert out[-2] == "link_sent" and out[-1] == "answer" and "already agreed" in last.plan.objective
    out, last, _ = await run(["yes tell me", "how much", "yes send the link", "no don't send it"])
    assert out[-1] == "declined" and last.plan.end_call
    out, last, _ = await run(["yes tell me", "how much", "yes send the link", "ok thanks"])
    assert out[-1] == "goodbye"


async def test_changed_mind_after_no_is_confirmed():
    out, _, _ = await run(["yes tell me", "no not interested", "no thanks", "actually okay, send it", "yes"])
    assert out[-2:] == ["confirm_close", "link_sent"]


async def test_not_now_means_no_link_offer():
    out, last, _ = await run(["yes tell me", "how much", "ruko, abhi link mat bhejo. ChatGPT se better kaise hai?"])
    assert out[-1] == "answer" and "do not offer the link" in last.plan.objective


async def test_ledger_records_promises_and_questions():
    out, last, pipe = await run(["yes tell me", "is it fake?", "how much", "yes send the link"])
    ctx = pipe.sessions["s"].context
    assert ctx.has("payment_link") and "offer_price" in ctx.quoted and ctx.decision == "accepted"
    assert ctx.questions and ctx.objections and "payment_link" in ctx.summary()


# ---------------------------------------------------------------- compliance

async def test_opening_discloses_ai_and_recording():
    out, _, pipe = await run([])
    plan = await pipe.decision.decide("greet", T(""), pipe.session(T("")))
    for lang in ("en", "hi", "hinglish"):
        assert "AI" in plan.fallback[lang] and "record" in plan.fallback[lang].replace("रिकॉर्ड", "record")


async def test_opt_out_ends_call_but_later_and_again_are_not_opt_out():
    for line in ("mujhe dobara call mat karna", "please don't call me again", "मेरा नंबर हटा दो", "stop calling me"):
        out, last, _ = await run(["हाँ बोलिए", line])
        assert out[-1] == "opt_out" and last.plan.end_call, line
    out, _, _ = await run(["abhi call mat karo baad mein karna"])
    assert out[-1] == "callback_time"  # (was "callback": a hang-up with no time; now it asks when to call back)
    out, _, _ = await run(["abhi call mat karo baad mein karna", "kal shaam 6 baje"])
    assert out[-1] == "callback"
    out, _, _ = await run(["फिर से कॉल? कौन है?"])
    assert out[-1] != "opt_out"


async def test_after_opt_out_reconfirms_removal():
    out, last, _ = await run(["हाँ बोलिए", "मेरा नंबर हटा दो", "पक्का हटा दिया?"])
    assert out[-1] == "opt_out_confirmed" and "हटा" in last.text


async def test_admits_ai_and_number_source():
    out, last, _ = await run(["क्या आप रोबोट हो?"])
    assert out[-1] == "are_you_ai" and "AI" in last.text and not last.plan.end_call
    out, _, _ = await run(["मेरा नंबर कहाँ से मिला?", "हाँ हटा दो"])
    assert out[-2:] == ["number_source", "opt_out"]


async def test_never_takes_card_details():
    out, last, _ = await run(["हाँ बोलिए", "मेरा कार्ड नंबर 4111 2222 3333 4444 है"])
    assert out[-1] == "sensitive_stop" and not last.plan.end_call


async def test_put_the_phone_down_is_a_goodbye():
    out, last, _ = await run(["हाँ बोलिए", "नहीं चाहिए, रखो फ़ोन"])
    assert out[-1] in ("declined", "reoffer")
    out, last, _ = await run(["हाँ बोलिए", "बस रखो फ़ोन"])
    assert out[-1] == "declined" and last.plan.end_call


async def test_callback_and_human():
    out, last, _ = await run(["abhi busy hoon baad mein call karo"])
    assert out[-1] == "callback_time" and not last.plan.end_call  # (was a hang-up with no time; now asks when)
    out, last, _ = await run(["abhi busy hoon baad mein call karo", "kal shaam 6 baje"])
    assert out[-1] == "callback" and last.plan.end_call
    out, last, _ = await run(["mujhe kisi insaan se baat karni hai"])
    assert out[-1] == "human_agent"


# ---------------------------------------------------------------- persona round 2

async def test_filler_after_price_confirms_and_unsure_yes_is_checked():
    out, _, _ = await run(["हाँ बोलिए", "कितने का है", "हाँ हाँ"])
    assert out[-1] == "confirm_close"
    h = "हाँ हाँ"
    out, last, _ = await run(["हाँ बोलिए", "कितने का है", h, h],
                             nlu=FakeNLU({h: dict(intent="affirm", buying_signal=0.5, gist="filler, not a clear purchase")}))
    assert out[-1] == "confirm_buy" and not last.plan.end_call


async def test_i_never_said_yes_cancels_the_link():
    out, last, pipe = await run(["yes tell me", "how much", "yes send the link", "कौन सा लिंक? हमने तो कुछ लेने को कहा ही नहीं"])
    assert out[-1] == "link_cancelled" and not last.plan.end_call
    assert not pipe.sessions["s"].context.has("payment_link")


async def test_yes_after_confirm_with_a_question_sends_and_answers():
    out, last, _ = await run(["yes tell me", "sure let's do it", "haan isi number pe. aur cancel kaise karte hain?"])
    assert out[-1] == "link_sent" and last.plan.llm_allowed


async def test_confirmed_terms_then_question_then_yes_sends():
    out, _, _ = await run(["yes tell me", "sure let's do it", "is it fake?", "okay send it"])
    assert out[-2:] == ["answer", "link_sent"]


async def test_asking_about_otp_is_not_sharing_it():
    out, _, _ = await run(["हाँ बोलिए", "link पर OTP तो बैंक का आएगा ना? cancel कहाँ से करूँ?"])
    assert out[-1] != "sensitive_stop"


async def test_who_is_it_identifies():
    out, last, _ = await run(["फिर से कॉल? कौन है?"])
    assert out[-1] == "identify" and "AI" in last.text


async def test_question_about_models_is_not_the_scripted_pitch():
    out, _, _ = await run(["Which models do I actually get? And what are the usage limits?"])
    assert out[-1] != "pitch"
    out, _, _ = await run(["haan bolo, kya offer hai?"])
    assert out[-1] == "pitch"


# ---------------------------------------------------------------- persona round 3

async def test_price_questions_get_the_exact_terms_word_for_word():
    out, last, _ = await run(["हाँ बोलिए", "डॉलर क्या होता है? रुपये में कितना?"])
    assert last.plan.lead and "700" in last.text and "$8" in last.text and "offer" not in last.plan.facts
    out, last, _ = await run(["yes tell me", "how much", "is the eight dollars permanent? will it auto charge 20 later?"])
    assert "renews automatically" in last.text


async def test_callback_time_is_confirmed_on_follow_up():
    out, last, _ = await run(["I'm busy, call me tomorrow evening around 7", "wait, did you note the time?"])
    assert out[-2:] == ["callback", "callback"] and "noted" in last.text


async def test_filler_after_a_side_answer_is_not_a_yes():
    out, _, _ = await run(["हाँ बोलिए", "कितने का है", "मौसम कैसा है?", "हाँ हाँ"])
    assert out[-1] not in ("confirm_close", "link_sent")


async def test_already_pro_is_not_pitched():
    out, last, _ = await run(["हाँ बोलो, पर मैं तो already Claude Pro use करता हूँ"])
    assert out[-1] != "pitch" and "नए" in last.text


async def test_asking_for_hindi_switches_language_not_handoff():
    out, last, _ = await run(["yes tell me", "how much", "I feel like talking to someone who can speak Hindi."])
    assert out[-1] == "language_switch" and not last.plan.end_call


def test_other_product_does_not_inherit_claude_hinglish_pitch():
    from acie.decision.domains.product_sales import Product, ProductSalesDomain
    p = Product(id="gym", product_name="FitZone", seller_name="FitZone", offer_price="₹2,999",
                original_price="₹6,000", pitch_hi="FitZone एक जिम है।", pitch_en="FitZone is a gym.")
    assert ProductSalesDomain(p).p.pitch_hinglish == ""
