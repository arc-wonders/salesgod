"""Per-call coverage state, kept in session.dialog["_kb"] (dialog is dict[str,str]; the value is canonical JSON).

  asked    entry ids retrieved for something the caller asked (explored)
  given    entry ids the agent has already told / handed to the LLM (given)
  offered  offering id -> times pitched
  topics   topic path -> number of entries asked about under it
  total / topic_total  size of the KB, for coverage ratios
"""

from __future__ import annotations

import json
from typing import Any

KEY = "_kb"


def load(session: Any) -> dict:
    raw = (getattr(session, "dialog", None) or {}).get(KEY, "")
    try:
        d = json.loads(raw) if raw else {}
    except ValueError:
        d = {}
    d.setdefault("asked", [])
    d.setdefault("given", [])
    d.setdefault("offered", {})
    d.setdefault("topics", {})
    d.setdefault("total", 0)
    d.setdefault("topic_total", {})
    d.setdefault("seen", "")  # "<turn_index>|<text hash>" of the last turn folded in (idempotent re-calls)
    return d


def save(session: Any, d: dict) -> None:
    session.dialog[KEY] = json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def add_unique(lst: list, items) -> None:
    for i in items:
        if i not in lst:
            lst.append(i)


def coverage_summary(session: Any) -> dict:
    """What has been explored and given so far, as plain numbers (for logs, the console and the prompt)."""
    d = load(session)
    topics: dict[str, dict[str, int]] = {}
    for path, n in sorted(d["topics"].items()):
        topics[path] = {"asked": n, "total": d["topic_total"].get(path, 0)}
    total = d["total"] or 0
    given = len(d["given"])
    return {"kb_total": total, "asked": len(d["asked"]), "given": given,
            "given_pct": round(100.0 * given / total, 1) if total else 0.0,
            "topics_explored": topics, "offered": dict(sorted(d["offered"].items())),
            "given_ids": list(d["given"]), "asked_ids": list(d["asked"])}


def mark_given(session: Any, entry_ids, offering_id: str | None = None) -> None:
    d = load(session)
    add_unique(d["given"], entry_ids)
    if offering_id:
        d["offered"][offering_id] = d["offered"].get(offering_id, 0) + 1
    save(session, d)
