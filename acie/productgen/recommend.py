"""Recommendations for a studio draft: deterministic, instant, no model call.

1. Coverage: a library of canonical sales-call situations. A situation is "covered" when any node's id / title / cue words match its
   pattern; an uncovered one becomes an "add" recommendation whose description is fed to playbook.generate_node in one call.
2. Checks: a node with no linked facts, a trust node with no proof fact, a node without Hindi, a number nobody confirmed."""

from __future__ import annotations

import re

from .studio import _text, find_node, ungrounded, walk

# id, title, match pattern (on node id + title + cue words), description for the generator
LIBRARY = [
    ("why_buy", "Asks why they should buy / what they get", r"why|what.*get|benefit|kya milega|fayda", "The customer asks what's in it for them and why they should buy."),
    ("price_high", "Says it's too expensive", r"expens|price.*high|costly|mehenga|महंगा|too much|afford", "The customer says the price is too high."),
    ("haggle", "Asks for a discount / haggles", r"discount|haggl|kam kar|bargain|less price|negotiat|counter", "The customer asks for a lower price or a bigger discount."),
    ("trust", "Doubts it's genuine (scam, fake, who are you)", r"trust|scam|fake|genuine|fraud|asli|bharosa|why so cheap", "The customer doubts the offer is genuine and asks if it's a scam."),
    ("competitor", "Compares with a competitor / cheaper option", r"compet|other|cheaper|alternative|dusra|compare|free version", "The customer compares with a competitor or a cheaper alternative."),
    ("already_have", "Already uses something similar / member elsewhere", r"already|member elsewhere|have one|pehle se|existing", "The customer says they already have a similar product or are a member somewhere else."),
    ("no_need", "Says they don't need it", r"no need|don.?t need|zaroorat nahi|not interested|need", "The customer says they don't need it."),
    ("busy", "Busy / call later", r"busy|later|call back|callback|baad mein|time nahi", "The customer is busy and asks to be called later."),
    ("think", "Will think about it", r"think|soch|decide later|consider", "The customer says they will think about it."),
    ("spouse", "Needs to ask spouse / family / boss", r"spouse|wife|husband|family|parent|boss|ghar|partner|ask someone", "The customer says they must ask their spouse, family or boss first."),
    ("send_details", "Asks to send details on WhatsApp / SMS", r"whatsapp|sms|send.*detail|message|bhej", "The customer asks to just send the details on WhatsApp or SMS."),
    ("how_to_start", "Asks how to join / pay / start", r"how to|join|pay|start|sign ?up|kaise|ready", "The customer asks how to join, pay or get started."),
    ("ready", "Ready to buy", r"ready|buy now|let.?s do|haan kar|book|sign me", "The customer is ready to buy."),
    ("refund", "Asks about cancellation / refund", r"cancel|refund|money back|freeze|wapas", "The customer asks about cancellation or refunds."),
    ("beginner", "Beginner / unsure it's for them", r"beginner|new to|never|first time|nervous|pehli baar", "The customer is a beginner and unsure if it is for them."),
    ("past_bad", "Had a bad experience before", r"bad experience|waste|before.*never|pehle.*bekar|didn.?t work", "The customer had a bad experience with something similar before."),
    ("location", "Asks where / how far / availability", r"where|location|far|branch|address|kahan|available", "The customer asks where it is available or how far it is."),
    ("privacy", "Asks how you got their number", r"number|privacy|data|kahan se mila|how did you get", "The customer asks how you got their number and worries about privacy."),
    ("confused", "Confused / didn't understand", r"confus|understand|samajh|clarify|repeat", "The customer is confused and didn't understand the offer."),
    ("annoyed", "Annoyed / irritated by the call", r"annoy|irritat|stop calling|pareshan|angry|frustrat", "The customer is annoyed by the call."),
]
_TRUST = re.compile(r"trust|scam|fake|genuine|fraud|bharosa", re.I)


def _hay(n: dict) -> str:
    w = n.get("when") or {}
    return " ".join([n.get("id", ""), n.get("title", ""), " ".join(w.get("keywords") or []), " ".join(w.get("objections") or [])]).lower()


def recommendations(draft: dict) -> list[dict]:
    nodes = list(walk((draft.get("tree") or {}).get("nodes") or []))
    hay = [_hay(n) for n in nodes]
    out: list[dict] = []
    for sid, title, pat, desc in LIBRARY:
        rx = re.compile(pat, re.I)
        if not any(rx.search(h) for h in hay):
            out.append({"id": "add:" + sid, "type": "missing_situation", "severity": "suggest", "title": f"No situation for: {title}",
                        "detail": desc, "action": "add", "description": desc})
    facts = {f["id"]: f for f in draft.get("facts") or []}
    for n in nodes:
        nid = n.get("id")
        if not n.get("cites"):
            out.append({"id": f"cite:{nid}", "type": "no_cites", "severity": "warn", "node": nid, "title": f"'{n.get('title')}' cites no facts",
                        "detail": "Link the facts this situation may quote, or the agent can only speak generally here.", "action": "link"})
        if _TRUST.search(_hay(n)) and not any(facts.get(c, {}).get("kind") == "proof" for c in n.get("cites") or []):
            out.append({"id": f"proof:{nid}", "type": "trust_without_proof", "severity": "warn", "node": nid, "title": f"'{n.get('title')}' has no proof to point to",
                        "detail": "Add a proof fact (reviews, years in business, members, official site) and link it here.", "action": "add_fact", "kind": "proof"})
        if not n.get("points_hi") or not str(n.get("ask_hi", "")).strip():
            out.append({"id": f"hindi:{nid}", "type": "missing_hindi", "severity": "warn", "node": nid, "title": f"'{n.get('title')}' has no Hindi lines",
                        "detail": "Hindi callers will hear a machine translation. Regenerate this situation to get Hindi.", "action": "regenerate",
                        "description": f"Rewrite the situation '{n.get('title')}' with Hindi points."})
    for u in ungrounded(draft):
        out.append({"id": f"num:{u['node']}", "type": "ungrounded_number", "severity": "warn", "node": u["node"],
                    "title": f"Unconfirmed number in '{u['node']}': {', '.join(u['numbers'])}",
                    "detail": "Every number the agent says must be a confirmed fact. Confirm the fact or edit the point.", "action": "edit"})
    if not any(f.get("kind") == "proof" for f in facts.values()):
        out.append({"id": "fact:proof", "type": "missing_fact", "severity": "suggest", "title": "No proof facts yet",
                    "detail": "Customers ask 'why should I trust you?'. Add reviews, years in business, member count or the official site.", "action": "add_fact", "kind": "proof"})
    if not any(f.get("kind") == "competitor" for f in facts.values()):
        out.append({"id": "fact:competitor", "type": "missing_fact", "severity": "suggest", "title": "No competitor comparison",
                    "detail": "Add one fair line on how you differ from the usual alternative (never run them down).", "action": "add_fact", "kind": "competitor"})
    return out


def find(draft: dict, rid: str) -> dict | None:
    return next((r for r in recommendations(draft) if r["id"] == rid), None)


__all__ = ["LIBRARY", "recommendations", "find", "find_node", "_text"]
