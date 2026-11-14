"""A deterministic fake LLM for the productgen tests (canned JSON by stage). Not a test module itself."""
from __future__ import annotations

import copy

HI = "यह एक जवाब है"
FLAWS_ALL = {"missing_hi", "banned", "cheap_offering", "few_entries", "short_ladder", "few_nodes"}


def t3(en: str, n: str = "") -> dict:
    return {"en": en, "hi": f"{HI} {n}".strip() if n else HI, "hinglish": f"yeh ek jawab hai {n}".strip()}


class FakeLLM:
    model = "fake-llm-1"

    def __init__(self, topics=16, per_topic=10, offerings=6, nodes=15, ladder=5, flaws=(), fixable=True, variant=""):
        self.topics, self.per_topic, self.n_off, self.n_nodes, self.n_ladder = topics, per_topic, offerings, nodes, ladder
        self.flaws, self.fixable, self.variant = set(flaws), fixable, variant
        self.calls: list[tuple[str, dict]] = []

    # -- helpers
    def _entry(self, path: str, k: int, flaw: bool = False) -> dict:
        slug = path.replace("/", "-")
        e = {"id": f"{slug}-q{k}{self.variant}", "topic_path": path,
             "question_variants": {"en": [f"What about item {k} in {path}{self.variant}?", f"Tell me {k} of {path}{self.variant}"],
                                   "hi": [f"{path} के आइटम {k} के बारे में बताइए{self.variant}"], "hinglish": [f"{path} ke item {k} ke baare mein batao{self.variant}"]},
             "answer": t3(f"Item {k} of {path} works like this, and the price is {{offer_price}}.", str(k)),
             "interests": ["tag-1", "tag-2"], "offerings": ["off-1"], "confidence": 0.9, "teach": k == 0}
        if flaw and "missing_hi" in self.flaws and k % 5 == 0:
            e["answer"]["hi"] = ""
        if flaw and "banned" in self.flaws and k == 1:
            e["answer"]["en"] = "This is guaranteed to lose weight, only today!"
        return e

    async def json(self, stage: str, payload: dict) -> dict:
        self.calls.append((stage, copy.deepcopy(payload)))
        return getattr(self, "_" + stage)(payload)

    # -- stages
    def _profile(self, p):
        return {"one_liner": "a fresh coffee subscription", "pitch": t3("ZenBrew delivers fresh coffee.", "1"),
                "benefits": {"en": ["fresh beans", "free delivery", "skip any month"], "hi": ["ताज़ा बीन्स", "फ्री डिलीवरी", "कभी भी स्किप"],
                             "hinglish": ["taaza beans", "free delivery", "kabhi bhi skip"]},
                "faqs": [{"keywords": [f"kw{i}", "delivery"], "answer": t3(f"FAQ {i} answer", str(i))} for i in range(8)],
                "knowledge": "ZenBrew is a coffee subscription.", "topic_words": ["coffee", "कॉफ़ी", "zenbrew"],
                "discovery_question": t3("How do you take your coffee?", "d"), "pronunciations": {"ZenBrew": "ज़ेनब्रू"},
                "tone": "warm", "info_source": t3("the website will have it", "i"), "call_to_action": "send you the link on SMS"}

    def _kb_plan(self, p):
        if p.get("only") == "offerings":
            have = set(p.get("have_offerings", []))
            return {"offerings": [self._off(j) for j in range(1, 12) if f"off-{j}" not in have]}
        if p.get("only") == "fix_text":
            return {"topics": [{"path": t["path"], "title": t3("Fixed " + t["path"], "x"), "overview": t3("Fixed overview", "x")} for t in p.get("topics", [])],
                    "offerings": [{"id": o["id"], "name": t3("Fixed offering", "x")} for o in p.get("offerings", [])]}
        nt = self.topics
        topics = [{"path": f"topic-{i}", "title": t3(f"Topic {i}", str(i)), "overview": t3(f"Overview {i}", str(i)), "interests": ["tag-1"]} for i in range(nt)]
        topics.append({"path": "topic-0/sub", "title": t3("Sub topic", "s"), "overview": t3("Sub overview", "s"), "interests": []})
        tags = [{"id": f"tag-{i}", "label": t3(f"Tag {i}", str(i)), "keywords": [f"kw{i}", f"word{i}"], "related": [f"tag-{(i % 9) + 1}"],
                 "discovery": {"en": [f"Question about tag {i}?"], "hi": ["टैग के बारे में सवाल?"], "hinglish": ["tag ke baare mein sawal?"]}} for i in range(1, 10)]
        return {"topics": topics, "interests": tags, "offerings": [self._off(j) for j in range(1, self.n_off + 1)]}

    def _off(self, j):
        price = 100 if ("cheap_offering" in self.flaws and j == 1) else 700 + 100 * j
        return {"id": f"off-{j}", "name": t3(f"Offering {j}", str(j)), "who_suits": [f"tag-{(j % 9) + 1}"], "price_inr": price,
                "price_label": {"en": f"Rs {price}", "hi": f"{price} रुपये", "hinglish": f"{price} rupaye"},
                "pitch": {"en": [f"You may like offering {j}"], "hi": ["आपको पसंद आएगा"], "hinglish": ["aapko pasand aayega"]}, "upsell_from": ["topic-1"], "min_fit": 1.0}

    def _kb_entries(self, p):
        out = []
        per = min(p.get("per_topic", 8), self.per_topic)
        if "few_entries" in self.flaws and not p.get("errors"):
            per = 3
        for t in p["topics"]:
            if isinstance(t, dict) and "/" not in t["path"] or isinstance(t, dict) and t["path"].count("/") == 0:
                for k in range(per):
                    out.append(self._entry(t["path"], k + (50 if p.get("errors") else 0), flaw=not p.get("errors")))
        return {"entries": out}

    def _kb_fix(self, p):
        out = []
        for e in p["fix"]:
            e = copy.deepcopy(e)
            if not e["answer"].get("hi"):
                e["answer"]["hi"] = HI + " ठीक"
            if "guaranteed" in e["answer"]["en"]:
                e["answer"]["en"] = "This is a plain, honest answer."
            if not self.fixable:
                e = copy.deepcopy(next(x for x in p["fix"] if x["id"] == e["id"]))
            out.append(e)
        return {"entries": out}

    def _playbook(self, p):
        have = set(p.get("have", []))
        n = p.get("need") or self.n_nodes
        if "few_nodes" in self.flaws and not p.get("errors"):
            n = 4
        nodes = [{"id": f"sit-{i}", "title": f"Situation {i}", "when": {"keywords": [f"kw{i}", "price"], "intents": ["price", "bogus"], "objections": ["price"], "topics": ["product"]},
                  "points": ["point one", "point two", "point three"], "ask": "What matters most?", "goal": "learn", "points_hi": ["बिंदु एक", "बिंदु दो", "बिंदु तीन"],
                  "ask_hi": "आपके लिए क्या ज़रूरी है?"} for i in range(40) if f"sit-{i}" not in have]
        return {"nodes": nodes[:n]}

    def _ladder(self, p):
        n = self.n_ladder
        if "short_ladder" in self.flaws and not p.get("errors"):
            n = 2
        levels = ["goal", "sub_goal", "experience", "schedule", "preference", "budget"]
        return {"ladder": [{"id": f"r-{i}", "level": levels[i % 6], "question": t3(f"Ladder question {i}?", str(i)), "purpose": "narrow", "interests": ["tag-1", "nope"]} for i in range(n)],
                "fit_rules": [{"offering": "off-1", "who_suits": ["tag-2"], "min_fit": 1.5}, {"offering": "ghost", "who_suits": ["tag-1"]}]}

    def _negotiation(self, p):
        f = p["floor"]
        return {"perks": [{"id": "free-month", "text": t3("an extra week free", "p"), "short": t3("extra week", "p")}],
                "price_steps": [{"id": "a", "price": f + 100}, {"id": "b", "price": f - 50}, {"id": "c", "price": f + 40}, {"id": "d", "price": p["current_price"] + 5}]}

    def _expand_topic(self, p):
        path = p["topic"]
        return {"topics": [{"path": f"{path}/new-a", "title": t3("New A", "a"), "overview": t3("Overview A", "a")}, {"path": "elsewhere/x", "title": t3("X", "x"), "overview": t3("X", "x")}],
                "entries": [self._entry(f"{path}/new-a", k) for k in range(6)] + [self._entry("topic-9", 99)]}

    def _expand_node(self, p):
        nid = p["node"]["id"]
        return {"children": [{"id": f"{nid}-kid-{i}", "title": f"Kid {i}", "when": {"keywords": [f"kid{i}"]}, "points": ["a"], "points_hi": ["अ"], "ask": "q?", "ask_hi": "क्या?"} for i in range(3)]}

    def _gap_entries(self, p):
        ents = []
        for c in p["clusters"]:
            e = self._entry("topic-3", 70 + int(c["id"][-2:], 16) % 20)
            e["id"] = "gap-" + c["id"][4:]
            e["question_variants"]["en"] = [c["questions"][0]]
            e["cluster"] = c["id"]
            ents.append(e)
        return {"entries": ents, "topics": [], "nodes": []}
