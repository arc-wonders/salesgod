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

# Regional flavour per language. Only used when ExpressionParams.regional_variant is set.
REGION_RULES = {
    "delhi": {
        "hi": "Delhi-style spoken Hindi: relaxed and direct, 'आप' with a friendly 'जी' now and then, words like 'बिल्कुल', 'एक मिनट', 'ठीक है'.",
        "hinglish": "Delhi-style Hinglish: relaxed, direct, 'ji' now and then, words like 'bilkul', 'ek minute', 'theek hai', 'chalo'.",
        "en": "Indian English with a Delhi tone: direct and friendly, light 'ji', 'bilkul', 'one minute'.",
    },
    "mumbai": {
        "hi": "Mumbai-style spoken Hindi: brisk and to the point, casual words like 'चलो', 'बोले तो', 'सही है'; no heavy slang.",
        "hinglish": "Mumbai-style Hinglish: brisk, to the point, casual words like 'chalo', 'bole toh', 'sahi hai', 'ekdum'; no heavy slang.",
        "en": "Indian English with a Mumbai tone: brisk and practical, short sentences, an occasional 'na' or 'only'.",
    },
    "lucknow": {
        "hi": "Lucknow-style Hindi: courteous and soft ('आप', 'जनाब' sparingly, 'तशरीफ़', 'जी हाँ', 'ज़रूर'), polite phrasing.",
        "hinglish": "Lucknow-style Hinglish: very courteous and soft ('aap', 'ji haan', 'zaroor', 'farmaiye'), polite phrasing.",
        "en": "Courteous Indian English with a Lucknow politeness: 'please', 'sure', 'kindly' used lightly.",
    },
    "bihar": {
        "hi": "Bihar-style Hindi: warm and plain, simple words, respectful 'आप' and 'जी', 'ठीक है न', 'बताइए'; no dialect spelling.",
        "hinglish": "Bihar-style Hinglish: warm and plain, simple words, respectful 'aap' and 'ji', 'theek hai na', 'bataiye'; no dialect spelling.",
        "en": "Simple, warm Indian English, short plain words, respectful 'ji'.",
    },
    "south_indian_english": {
        "hi": "Slow, clear, standard Hindi with simple words; avoid idioms and slang.",
        "hinglish": "Slow, clear Hinglish with mostly English words and a few simple Hindi ones; avoid idioms and slang.",
        "en": "South Indian English: clear, neutral and slightly formal, full sentences, standard words, no North Indian slang.",
    },
}
EXPLANATION_RULES = {
    "example": "Explain with one short, concrete everyday example (no numbers of your own).",
    "step_by_step": "Explain in order, one short step at a time ('first ..., then ...').",
}
DENSITY_RULES = {
    "low": "Give only the one key point; leave details for when they ask.",
    "high": "Cover the facts completely, but keep every sentence short.",
}
REASSURANCE_RULES = {
    "med": "Add a brief reassurance that they are in control and can decide at their own pace.",
    "high": "Be clearly reassuring: say there is no hurry and no pressure, and that you will help step by step.",
}


def _extra_style(expr: ExpressionParams) -> str:
    """STYLE lines for the newer knobs. Empty at their defaults, so today's prompt is unchanged."""
    lines = []
    region = getattr(expr, "regional_variant", None)
    if region:
        rules = REGION_RULES.get(str(region).lower())
        lang = expr.language
        lines.append("- Regional flavour: " + (rules[lang] if rules else
                     f"{str(region).replace('_', ' ')} regional flavour, light touch, standard words."))
    for attr, table in (("explanation_style", EXPLANATION_RULES), ("info_density", DENSITY_RULES),
                        ("reassurance", REASSURANCE_RULES)):
        line = table.get(getattr(expr, attr, None))
        if line:
            lines.append(f"- {attr.replace('_', ' ').capitalize()}: {line}")
    return "".join(line + "\n" for line in lines)


def _clean(text, limit: int) -> str:
    return " ".join(str(text).replace("{", "").replace("}", "").split())[:limit]


def _profile_block(profile) -> str:
    """Soft long-term hints (likes/topics/dislikes of CustomerProfile). Empty when nothing is known."""
    parts = []
    for attr, label in (("likes", "likes"), ("topics", "often asks about"), ("dislikes", "dislikes")):
        items = [_clean(i, 40) for i in (getattr(profile, attr, None) or [])[:5] if str(i).strip()]
        if items:
            parts.append(f"{label}: {', '.join(items)}")
    if not parts:
        return ""
    return ("CUSTOMER PROFILE (soft hints from earlier calls; use only if it fits naturally, never say you "
            "remember it, never add facts or numbers from it):\n" + "\n".join(parts) + "\n\n")


def _memory_block(memories) -> str:
    items = [_clean(m, 160) for m in (memories or [])[:3] if str(m).strip()]
    if not items:
        return ""
    return ("RELEVANT MEMORIES (earlier calls; background only, may be out of date; do not state them as new "
            "facts and do not write numbers from them):\n" + "\n".join(f"- {m}" for m in items) + "\n\n")


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
   Never say you have the customer's details, a record, a previous inquiry or an earlier call in your system: you only know what they said in THIS call.

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
                customer_text: str, *, profile=None, memories: list[str] | None = None) -> CompiledPrompt:
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
        ) + _extra_style(expr)
        messages: list[dict[str, str]] = []
        for u in session.history[-self.history_turns:]:
            messages.append({"role": "user" if u.role == "customer" else "assistant", "content": u.text})
        plan_json = json.dumps(_plan_for_llm(plan), ensure_ascii=False, indent=1)
        reminder = {"en": "Reply only in English.", "hi": "Reply only in Hindi (Devanagari script).",
                    "hinglish": "Reply only in Hinglish (Roman script)."}[expr.language]
        ctx = session.context.summary() if getattr(session, "context", None) else ""
        ctx_block = f"CALL SO FAR (true for the whole call, never contradict it):\n{ctx}\n\n" if ctx else ""
        ctx_block += _profile_block(profile) + _memory_block(memories)
        messages.append({"role": "user", "content": f"{ctx_block}CUSTOMER SAID: {customer_text}\n\nPLAN:\n{plan_json}\n\n{reminder} "
                                                    "Use the {placeholders} for every fact, never the values."})
        return CompiledPrompt(system=system, messages=messages)
