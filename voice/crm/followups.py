"""Post-call outcome application (stage, score, next_action, DNC) and the follow-up due list.

Source of truth for a call's result, in order:
  1. rec["result"] = {outcome, stage_suggestion, next_action, answers, score_delta}   (CALLTYPES lane)
  2. derived from rec["context"] (decision / promised / objections) + status / machine
Canonical outcomes: accepted, interested, callback, later, undecided, declined, opt_out, no_answer, voicemail, failed.
Applying a call twice does nothing (idempotent per call_id).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .models import Customer, NextAction, Project, STAGES, TERMINAL
from .scoring import UNREACHED, score
from .store import JsonStore

DAY = 86400
NURTURE_DAYS = int(os.getenv("CRM_NURTURE_DAYS", "30"))
ALIASES = {"callback_requested": "callback", "callback_booked": "callback", "qualified": "interested", "not_qualified": "undecided",
           "link_sent": "accepted", "promised_to_pay": "accepted", "paid": "accepted", "ready_to_buy": "accepted", "not_interested": "declined",
           "human_requested": "interested", "appointment_booked": "interested", "completed": "undecided", "feedback_given": "undecided",
           "ended_early": "undecided",
           "accepted": "accepted", "won": "accepted", "sold": "accepted", "link_sent": "accepted", "interested": "interested",
           "callback": "callback", "callback_later": "callback", "later": "later", "undecided": "undecided",
           "declined": "declined", "lost": "declined", "opt_out": "opt_out", "dnc": "opt_out", "do_not_call": "opt_out",
           "no_answer": "no_answer", "voicemail": "voicemail", "machine": "voicemail", "failed": "failed",
           "demo_booked": "interested", "safety": "opt_out"}


_NA_TYPE = {"follow_up_call": "follow_up", "human_call": "follow_up", "book_demo": "follow_up", "book_callback": "callback",
            "book_appointment": "follow_up", "send_link": "follow_up", "nurture": "nurture"}


def _na_from_result(na) -> dict | None:
    """CallResult.next_action {type, due_at ISO, call_type} -> the CRM's next_action dict (epoch due_at, None = not scheduled)."""
    if not isinstance(na, dict) or not na.get("type") or na["type"] == "none":
        return None
    due = na.get("due_at")
    if isinstance(due, str) and due:
        try:
            due = datetime.fromisoformat(due).timestamp()
        except ValueError:
            due = None
    elif not isinstance(due, (int, float)):
        due = None
    out = {"type": _NA_TYPE.get(na["type"], na["type"]), "due_at": due, "call_type": na.get("call_type") or None}
    return {k: v for k, v in out.items() if v is not None or k == "due_at"}


def derive(rec: dict) -> dict:
    """Structured result for a call record: use rec['result'] when valid, else read the context ledger."""
    res = (rec.get("context") or {}).get("result")  # acie.calltypes.CallResult, rebuilt by the call-type pack every turn
    if not (isinstance(res, dict) and res.get("outcome")):
        res = rec.get("result")  # older records / direct callers
    if isinstance(res, dict) and res.get("outcome"):
        out = dict(res)
        out["outcome"] = ALIASES.get(str(res["outcome"]).lower(), str(res["outcome"]).lower())
        out["answers"] = {**(res.get("facts") or {}), **(res.get("answers") or {})}
        out["next_action"] = _na_from_result(res.get("next_action"))
        out["derived"] = False
        return out
    ctx = rec.get("context") or {}
    promised = list(ctx.get("promised") or [])
    decision = ctx.get("decision", "undecided")
    co = ctx.get("outcome", "")
    out: dict = {"derived": True, "answers": {}, "score_delta": 0}
    if rec.get("status") == "failed":
        out["outcome"] = "failed"
    elif not rec.get("answered_at"):
        out["outcome"] = "no_answer"
    elif rec.get("machine"):
        out["outcome"] = "voicemail"
    elif "number_removed_no_more_calls" in promised or co == "opt_out":
        out["outcome"] = "opt_out"
    elif decision == "accepted" or "payment_link" in promised:
        out["outcome"] = "accepted"
        out["answers"]["payment_link_sent"] = "payment_link" in promised
    elif decision == "declined":
        out["outcome"] = "declined"
    elif "callback_later" in promised or decision == "later" or co == "later":
        out["outcome"] = "callback"
    elif "team_member_will_call" in promised:
        out["outcome"] = "callback"
        out["next_action"] = {"type": "follow_up", "note": "a team member was promised to call"}
    elif any(t.get("role") == "customer" for t in rec.get("turns", [])):
        out["outcome"] = "undecided"
    else:
        out["outcome"] = "no_answer"
    return out


def _forward(cur: str, new: str) -> str:
    if new == "do_not_call":
        return new
    if cur in TERMINAL or new not in STAGES:
        return cur
    if new in ("lost", "won"):
        return new
    return new if STAGES.index(new) > STAGES.index(cur) else cur


STAGE_FOR = {"accepted": "negotiating", "interested": "interested", "callback": "contacted", "later": "contacted",
             "undecided": "contacted", "declined": "lost", "opt_out": "do_not_call"}


def _ev(c: Customer, t: float, type_: str, **kw) -> None:
    c.events.append({"t": t, "type": type_, **kw})
    del c.events[:-500]


def apply_call(store: JsonStore, rec: dict, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    cid_call = rec.get("call_id", "")
    link = {k: rec.get(k) for k in ("project_id", "campaign_id", "call_type") if rec.get(k)}
    if rec.get("crm_customer_id"):
        link["customer_id"] = rec["crm_customer_id"]
    link = {**(store.call_link(cid_call) or {}), **link}
    cust = store.get_customer(link["customer_id"]) if link.get("customer_id") else None
    if cust is None:
        return {"applied": False, "reason": "no CRM customer linked to this call"}
    if rec.get("status") not in ("ended", "failed"):
        return {"applied": False, "reason": "call not finished"}
    if any(o.get("call_id") == cid_call for o in cust.history.outcomes):
        return {"applied": False, "reason": "already applied", "customer_id": cust.id}
    project = store.get_project(cust.project_id) or Project(id=cust.project_id, name=cust.project_id)
    res = derive(rec)
    outcome = res["outcome"]
    t = rec.get("ended_at") or now
    h = cust.history
    h.call_ids.append(cid_call)
    h.outcomes.append({"call_id": cid_call, "outcome": outcome, "at": t, "call_type": link.get("call_type")})
    del h.outcomes[:-100]
    ctx = rec.get("context") or {}
    for o in ctx.get("objections") or []:
        if o not in h.objections:
            h.objections.append(o)
    answers = res.get("answers") or {}
    if isinstance(answers, dict):
        for k, v in answers.items():
            h.facts[k] = v
        if answers.get("agreed_price") is not None:
            try:
                h.agreed_price = float(answers["agreed_price"])
            except (TypeError, ValueError):
                pass
        for like in answers.get("likes") or []:
            if like not in h.likes:
                h.likes.append(like)
    try:
        cust.score_delta_total += int(res.get("score_delta") or 0)
    except (TypeError, ValueError):
        pass
    if outcome not in UNREACHED:
        cust.last_contact_at = t
    _ev(cust, t, "call", call_id=cid_call, outcome=outcome, call_type=link.get("call_type"), campaign_id=link.get("campaign_id"))

    old_stage = cust.stage
    sug = res.get("stage_suggestion")
    if outcome == "opt_out":
        new = "do_not_call"
    elif sug in STAGES and sug != "do_not_call":
        new = _forward(old_stage, sug)
    else:
        new = _forward(old_stage, STAGE_FOR.get(outcome, old_stage))
    if outcome == "opt_out":
        store.add_dnc(cust.phone)
        cust.consent.dnc, cust.consent.reason, cust.consent.at = True, "asked not to be called (call " + cid_call + ")", t
    if new != old_stage:
        cust.stage = new
        _ev(cust, t, "stage", **{"from": old_stage, "to": new, "by": "call"})

    campaign = store.get_campaign(link["campaign_id"]) if link.get("campaign_id") else None
    cust.next_action = _next_action(cust, project, campaign, res, outcome, link, now)
    cust.updated_at = now
    cust.score = score(cust, now)
    store.put_customer(cust)
    if campaign:
        campaign.stats["calls"] = campaign.stats.get("calls", 0) + 1
        campaign.stats[outcome] = campaign.stats.get(outcome, 0) + 1
        store.put_campaign(campaign)
    return {"applied": True, "customer_id": cust.id, "outcome": outcome, "stage": cust.stage, "score": cust.score,
            "next_action": cust.next_action.model_dump() if cust.next_action else None, "derived": res["derived"]}


def _next_action(c: Customer, project: Project, campaign, res: dict, outcome: str, link: dict, now: float) -> NextAction | None:
    if c.stage in ("won", "do_not_call"):
        return None
    na = res.get("next_action")
    if isinstance(na, dict) and na.get("type"):
        try:
            return NextAction(**{**{"campaign_id": link.get("campaign_id")}, **na, "due_estimated": not na.get("due_at")})
        except Exception:
            pass
    cid = link.get("campaign_id")
    if outcome == "callback":
        return NextAction(type="callback", due_at=now + DAY, call_type="follow_up", campaign_id=cid, due_estimated=True,
                          note="the customer asked to be called later")
    if outcome == "accepted":
        return NextAction(type="follow_up", due_at=now + DAY, call_type="follow_up", campaign_id=cid, due_estimated=True,
                          note="payment link sent: check if it was paid")
    if outcome == "declined":
        return NextAction(type="nurture", due_at=now + NURTURE_DAYS * DAY, call_type="follow_up", campaign_id=cid,
                          note="declined: one gentle check-in later")
    if outcome in UNREACHED:
        retry = campaign.schedule.retry if campaign else None
        attempts = sum(1 for o in c.history.outcomes if o.get("outcome") in UNREACHED)
        if retry and attempts >= retry.max_attempts:
            return None
        mins = retry.retry_after_minutes if retry else 240
        return NextAction(type="retry", due_at=now + mins * 60, call_type=link.get("call_type"), campaign_id=cid,
                          note=f"retry after {outcome}")
    if outcome in ("later", "undecided", "interested"):
        return NextAction(type="follow_up", due_at=now + 2 * DAY, call_type="follow_up", campaign_id=cid, due_estimated=True,
                          note="continue from the last call")
    return None


def sweep(store: JsonStore, calls_dir: Path, now: float | None = None) -> int:
    """Apply every finished, linked call not yet applied. Cheap: only walks the link index."""
    n = 0
    for call_id, link in store.all_call_links().items():
        if link.get("applied"):
            continue
        p = Path(calls_dir) / f"{call_id}.json"
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        r = apply_call(store, rec, now)
        if r.get("applied") or r.get("reason") == "already applied":
            store.link_call(call_id, {"applied": True})
            n += r.get("applied", False)
    return n


def due_list(store: JsonStore, project_id: str, due: str = "now", now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    proj = store.get_project(project_id)
    tz = ZoneInfo(proj.timezone if proj else "Asia/Kolkata")
    if due == "today":
        d = datetime.fromtimestamp(now, tz)
        limit = (d.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).timestamp()
    elif due == "all":
        limit = float("inf")
    elif due == "week":
        limit = now + 7 * DAY
    else:
        limit = now
    dnc = store.dnc()
    out = []
    for c in store.customers(project_id):
        na = c.next_action
        if not na or c.stage in TERMINAL or c.consent.dnc or c.phone in dnc:
            continue
        at = na.due_at if na.due_at is not None else now
        if at <= limit:
            out.append({"customer_id": c.id, "name": c.name, "phone": c.phone, "stage": c.stage, "score": c.score,
                        "next_action": na.model_dump(), "overdue": at < now, "due_at": at})
    return sorted(out, key=lambda r: (r["due_at"], -r["score"], r["customer_id"]))
