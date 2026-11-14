"""Conversation playbook: a per-tenant, per-product decision tree written BEFORE the call.

A frontier model (configurable, see `generate`) reads the product config and lists the situations a sales
call is likely to head into - "why should I even buy this?", "it's too expensive", "is this a scam?",
"I'm a student", "I'm busy" - and for each one the talking points a great salesperson would use and the
question to end on. On the call, the picker matches the live signals (NLU intent, objection, question
topic, buying signal, frustration, confusion, keywords) to the most likely node, and the paraphraser
(Qwen or Gemini, the expression layer) says 1-2 of that node's points in its own words, in the caller's
language, without repeating points already used.

Files: data/voice/playbooks/<tenant>/<product_id>.json
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
from pydantic import BaseModel, Field


class When(BaseModel):
    """Signals that point to this node. Any match adds to the score; ranges must hold if given."""
    keywords: list[str] = Field(default_factory=list)  # regex fragments, matched on the customer's words
    intents: list[str] = Field(default_factory=list)  # NLU / rule intents
    objections: list[str] = Field(default_factory=list)  # trust | price | need | timing | language | other
    topics: list[str] = Field(default_factory=list)  # product | offer_terms | general | conversation
    min_buying: float | None = None
    max_buying: float | None = None
    min_frustration: float | None = None
    min_confusion: float | None = None


class Node(BaseModel):
    id: str
    title: str  # the situation, e.g. "asks why they should buy it"
    when: When = Field(default_factory=When)
    points: list[str] = Field(default_factory=list)  # talking points (facts only from the product config)
    ask: str = ""  # the question to end on (never "shall I send the link" unless the node is ready_to_buy)
    # the same in Hindi (Devanagari, natural spoken Hindi with common English words) - the 14B paraphraser
    # translates poorly ("पैशन ट्यूटर"), so the planner writes the Hindi too
    points_hi: list[str] = Field(default_factory=list)
    ask_hi: str = ""
    goal: str = ""  # what this turn should achieve
    # hand-written decision tree: once this node matches, a matching child is more specific and wins
    children: list["Node"] = Field(default_factory=list)


Node.model_rebuild()


class Playbook(BaseModel):
    tenant: str = "default"
    product_id: str
    generated_by: str = ""
    nodes: list[Node] = Field(default_factory=list)

    @staticmethod
    def _score(n: "Node", low: str, nlu, intent: str) -> float | None:
        """A node's match score, or None when one of its signal ranges rules it out."""
        from .signals import as_signals
        nlu = as_signals(nlu)  # TurnSignals (or a legacy Understanding, coerced): the playbook never sees a provider's object
        w, score = n.when, 0.0
        if any(re.search(k, low, re.I) for k in w.keywords if k):
            score += 2.5
        if intent and intent in w.intents:
            score += 1.0
        if nlu is not None:
            if any(v in w.intents for v in nlu.votes.values()):
                score += 1.0
            if nlu.objection != "none" and nlu.objection in w.objections:
                score += 2.0
            if nlu.question_topic != "none" and nlu.question_topic in w.topics:
                score += 0.5
            ok = ((w.min_buying is None or nlu.buying_signal >= w.min_buying)
                  and (w.max_buying is None or nlu.buying_signal <= w.max_buying)
                  and (w.min_frustration is None or nlu.frustration >= w.min_frustration)
                  and (w.min_confusion is None or nlu.confusion >= w.min_confusion))
            if not ok:
                return None
            if any(v is not None for v in (w.min_buying, w.min_frustration, w.min_confusion)):
                score += 1.5  # a signal-defined node whose signal holds
        return score

    def pick(self, text: str, nlu=None, intent: str = "") -> tuple[Node | None, float]:
        low = text.lower()
        best, best_score = None, 0.0
        for n in self.nodes:
            score = self._score(n, low, nlu, intent)
            if score is not None and score > best_score:
                best, best_score = n, score
        if best_score < 2.0:
            return None, best_score
        # tree: walk down while a child matches on its own (score >= 2.0); the child replaces its parent
        while best is not None and best.children:
            kid, kid_score = None, 0.0
            for c in best.children:
                sc = self._score(c, low, nlu, intent)
                if sc is not None and sc > kid_score:
                    kid, kid_score = c, sc
            if kid is None or kid_score < 2.0:
                break
            best, best_score = kid, kid_score
        return best, best_score


def playbook_dir(data: Path) -> Path:
    return data / "playbooks"


SEED = Path(__file__).resolve().parent.parent / "playbooks"  # versioned playbooks shipped with the repo


def load(data: Path, tenant: str, product_id: str) -> Playbook | None:
    """The tenant's saved playbook, else the default tenant's, else the seed shipped in the repo."""
    for root in (playbook_dir(data), SEED):
        for t in (tenant, "default"):
            # a hand-written tree file (<product>.tree.json) beats the generated flat playbook
            for name in (f"{product_id}.tree.json", f"{product_id}.json"):
                f = root / t / name
                if f.exists():
                    return Playbook.model_validate_json(f.read_text())
    return None


def load_tree(path: Path | str) -> Playbook:
    """Load one hand-written tree file: {"product_id", "tenant"?, "nodes": [{... "children": [...]}]}."""
    return Playbook.model_validate_json(Path(path).read_text())


def save(data: Path, pb: Playbook) -> Path:
    f = playbook_dir(data) / pb.tenant / f"{pb.product_id}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(pb.model_dump_json(indent=1))
    return f


GEN_PROMPT = """You are a top sales coach. Build the conversation playbook for an outbound phone agent selling the
product below, in India (customers speak Hindi, Hinglish or English).

List 14-20 SITUATIONS the call is likely to head into, e.g.: asks what's in it for them / why buy; compares with
a competitor or the free version; price is too high / haggles; doubts it's genuine (scam, fake, who are you);
says they don't need it; busy / will think about it; specific customer types for THIS product (student, developer,
small business, professional, parent, beginner who doesn't know the technology...); privacy; how to start / pay;
confused; annoyed; ready to buy. For each situation give:
- when: keywords (short regex fragments in English, Roman Hindi AND Devanagari), intents (from: affirm, deny,
  tell_more, price, benefits, callback, clarify, goodbye, personal, small_talk), objections (trust, price, need,
  timing, language, other), topics (product, offer_terms, general, conversation), optional min_buying /
  max_buying / min_frustration / min_confusion (0..1)
- points: 3-5 concrete talking points a great salesperson would say - specific benefits and examples for that
  situation. Use ONLY facts in the product config; never invent prices, features or guarantees.
- ask: one short question to end on that moves the conversation forward (learn their need, or offer to try only
  in the ready-to-buy situation)
- goal: what this turn should achieve.
- points_hi / ask_hi: the same points and question in natural spoken Hindi (Devanagari), keeping common English
  words people actually say on the phone (plan, offer, coding, assignment, app, cancel, link).

Return JSON only: {"nodes": [{"id", "title", "when": {...}, "points": [...], "ask", "goal", "points_hi": [...],
"ask_hi"}, ...]}

PRODUCT CONFIG:
"""


async def generate(product, tenant: str = "default", timeout_s: float = 180.0) -> Playbook:
    """Ask the planner model (a big frontier model, any OpenAI-compatible endpoint) for the playbook.
    ACIE_PLANNER_BASE_URL / ACIE_PLANNER_MODEL / ACIE_PLANNER_API_KEY."""
    base = os.getenv("ACIE_PLANNER_BASE_URL", "").rstrip("/")
    model = os.getenv("ACIE_PLANNER_MODEL", "")
    if not base or not model:
        raise RuntimeError("set ACIE_PLANNER_BASE_URL and ACIE_PLANNER_MODEL to generate a playbook")
    cfg = product.model_dump(exclude={"allowed_numbers", "pronunciations", "sms_text", "payment_link"})
    body = {"model": model, "temperature": 0.4, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": GEN_PROMPT + json.dumps(cfg, ensure_ascii=False, indent=1)}]}
    headers = {"Authorization": f"Bearer {os.getenv('ACIE_PLANNER_API_KEY', 'EMPTY')}"}
    async with httpx.AsyncClient(timeout=timeout_s) as c:
        r = await c.post(f"{base}/chat/completions", json=body, headers=headers)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"]
    raw = raw[raw.find("{"): raw.rfind("}") + 1]
    nodes = [Node.model_validate(n) for n in json.loads(raw)["nodes"]]
    return Playbook(tenant=tenant, product_id=product.id, generated_by=model, nodes=nodes)
