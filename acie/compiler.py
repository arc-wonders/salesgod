"""Layer 5 - Structured Prompt Compiler.

Turns (TurnPlan, ExpressionParams, history) into chat messages. Key design
choice: facts are given to the LLM as {slot} placeholders only. The LLM never
sees or writes the actual numbers, so it cannot get them wrong; the validator
fills them in afterwards.
"""

from __future__ import annotations

import json

from .schemas import CompiledPrompt, ExpressionParams, SessionState, TurnPlan

LANGUAGE_RULES = {
    "en": "Speak in simple Indian English.",
    "hi": "Speak in natural spoken Hindi written in Devanagari. Common English banking words "
          "(balance, transfer, limit, card, account) may stay in English as people say them.",
    "hinglish": "Speak in natural Hinglish written in Roman script, mixing Hindi and English the way "
                "people in Delhi/Mumbai talk on the phone.",
}
LEVEL_RULES = {
    "verbosity": {"low": "One short sentence, under 15 words, plus the required question if any.",
                  "med": "At most two sentences.",
                  "high": "Up to four sentences; you may add one helpful clarification."},
    "complexity": {"low": "Use very simple everyday words. No jargon. Short sentences.",
                   "med": "Plain language; basic banking terms are fine.",
                   "high": "Precise terminology is fine."},
    "formality": {"low": "Casual and friendly.", "med": "Polite and natural.",
                  "high": "Respectful and formal (use 'aap', 'ji' where natural)."},
    "warmth": {"low": "Neutral.", "med": "Friendly.", "high": "Calm, patient and reassuring."},
    "proactivity": {"low": "Answer only what was asked. Do not suggest anything extra.",
                    "med": "Answer what was asked; a brief next-step hint is fine.",
                    "high": "You may suggest a relevant next step from allowed_actions."},
}

SYSTEM_TEMPLATE = """You are the voice of a customer-service phone agent. You only turn a given plan into natural speech.

HARD RULES
1. Communicate exactly the objective, decision and facts in PLAN. Do not add new facts, offers, numbers, dates or promises.
2. Write every fact ONLY as its placeholder in curly braces, e.g. {{balance}}. Never write digits or amounts yourself.
3. Mention every fact listed under "facts". Cover the "talking_points" in your own words, in the reply language (no numbers).
4. Never say anything listed under "must_not_say".
5. If PLAN has "ask", end with that question in your own words.
6. Output only the words to be spoken. No lists, emojis, markdown, or stage directions.
7. "background_knowledge" is for understanding what is being discussed; use it to answer, never recite it.
8. If "general_question" is true, the customer asked about something unrelated. Answer it like a well-informed
   person who just checked: use "looked_up" only if it directly answers the question (search results are often
   about something else; then ignore them and use your own general knowledge), in one or two
   short sentences. Here you MAY say numbers (years, temperatures, scores), written as words in the reply
   language, directly in the sentence (no {{placeholders}} for these). Never prices or promises. Never say you can't help; if you truly don't know say so briefly, and if
   the question is unclear ask one short clarifying question. Never claim it relates to the product. Then come
   back to the plan's ask in a few words.
10. Never repeat, quote or echo the customer's words back ("हाँ, बताइए" / "okay, let's try" are THEIR words, not yours).
    If "Reference wording" is given in the objective, say everything it says (same facts, same closing question).
9. You are an AI assistant. If asked, say so honestly. Never ask for OTP, PIN, CVV, passwords or card or bank
   details. No pressure, no false urgency, no guilt, no claims you can't back. Answer what the customer just said first.

STYLE
- Language: {language}
- Length: {verbosity}
- Words: {complexity}
- Tone: {formality} {warmth}
- Proactivity: {proactivity}{ack}
"""


def _plan_for_llm(plan: TurnPlan) -> dict:
    return {
        "objective": plan.objective,
        "decision": plan.decision,
        "facts": {f"{{{k}}}": f.description for k, f in plan.facts.items() if f.verbatim and not f.hidden},
        "talking_points": [f.value for f in plan.facts.values() if not f.verbatim and not f.hidden],
        "allowed_actions": plan.allowed_actions,
        "must_not_say": plan.must_not_say,
        "ask": plan.ask,
        **({"background_knowledge": plan.context} if plan.context else {}),
        **({"general_question": True} if plan.general_knowledge else {}),
        **({"looked_up": plan.reference} if plan.reference else {}),
        **({"tone": plan.persona} if plan.persona else {}),
    }


class PromptCompiler:
    def __init__(self, history_turns: int = 10):
        self.history_turns = history_turns

    def compile(self, plan: TurnPlan, expr: ExpressionParams, session: SessionState,
                customer_text: str) -> CompiledPrompt:
        ack = ("\n- Start with one brief, sincere acknowledgement of the customer's frustration, then get to the point."
               if expr.acknowledge_emotion else "")
        system = SYSTEM_TEMPLATE.format(
            language=LANGUAGE_RULES[expr.language],
            verbosity=LEVEL_RULES["verbosity"][expr.verbosity],
            complexity=LEVEL_RULES["complexity"][expr.complexity],
            formality=LEVEL_RULES["formality"][expr.formality],
            warmth=LEVEL_RULES["warmth"][expr.warmth],
            proactivity=LEVEL_RULES["proactivity"][expr.proactivity],
            ack=ack,
        )
        messages: list[dict[str, str]] = []
        for u in session.history[-self.history_turns:]:
            messages.append({"role": "user" if u.role == "customer" else "assistant", "content": u.text})
        plan_json = json.dumps(_plan_for_llm(plan), ensure_ascii=False, indent=1)
        reminder = {"en": "Reply only in English.", "hi": "Reply only in Hindi (Devanagari script).",
                    "hinglish": "Reply only in Hinglish (Roman script)."}[expr.language]
        ctx = session.context.summary() if getattr(session, "context", None) else ""
        ctx_block = f"CALL SO FAR (true for the whole call, never contradict it):\n{ctx}\n\n" if ctx else ""
        messages.append({"role": "user", "content": f"{ctx_block}CUSTOMER SAID: {customer_text}\n\nPLAN:\n{plan_json}\n\n{reminder} "
                                                    "Use the {placeholders} for every fact, never the values."})
        return CompiledPrompt(system=system, messages=messages)
