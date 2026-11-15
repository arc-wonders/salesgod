"""Campaign queue: who is next and why. Pure function of (store, campaign, now, active calls); never dials."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .models import Campaign, Customer, Project, TERMINAL, is_demo
from .scoring import UNREACHED
from .store import JsonStore

DEFAULT_STAGES = ["new", "contacted", "interested", "qualified", "demo_booked", "negotiating"]


def window_state(project: Project, campaign: Campaign, now: float) -> dict:
    tz = ZoneInfo(project.timezone)
    a, b = campaign.schedule.window or project.call_hours
    days = campaign.schedule.days
    d = datetime.fromtimestamp(now, tz)
    if d.weekday() in days and a <= d.hour < b:
        return {"open": True, "next_open_at": None, "window": [a, b], "timezone": project.timezone}
    for off in range(0, 9):
        day = (d + timedelta(days=off)).replace(hour=a, minute=0, second=0, microsecond=0)
        if day.weekday() in days and day.timestamp() > now:
            return {"open": False, "next_open_at": day.timestamp(), "window": [a, b], "timezone": project.timezone}
    return {"open": False, "next_open_at": None, "window": [a, b], "timezone": project.timezone}


def _attempts(c: Customer, campaign_id: str) -> list[dict]:
    return [e for e in c.events if e.get("type") == "call" and e.get("campaign_id") == campaign_id]


def compute(store: JsonStore, campaign: Campaign, now: float | None = None, active_customer_ids: set[str] | None = None,
            active_calls: int = 0) -> dict:
    now = time.time() if now is None else now
    project = store.get_project(campaign.project_id) or Project(id=campaign.project_id, name=campaign.project_id)
    win = window_state(project, campaign, now)
    dnc, active = store.dnc(), active_customer_ids or set()
    aud, sch = campaign.audience, campaign.schedule
    stages = aud.stages or DEFAULT_STAGES
    eligible, skipped = [], []
    for c in sorted(store.customers(campaign.project_id), key=lambda x: (x.created_at, x.id)):
        def skip(why: str) -> None:
            skipped.append({"customer_id": c.id, "name": c.name, "reason": why})
        if c.consent.dnc or c.stage == "do_not_call" or c.phone in dnc:
            skip("do not call"); continue
        if not c.phone:
            skip("no phone number"); continue
        if aud.customer_ids and c.id not in aud.customer_ids:
            continue
        if c.stage in TERMINAL or c.stage not in stages:
            skip(f"stage {c.stage} is not in the audience"); continue
        if aud.tags and not set(aud.tags) & set(c.tags):
            continue
        if aud.languages and c.language not in aud.languages:
            continue
        if aud.min_score is not None and c.score < aud.min_score:
            skip(f"score {c.score} below {aud.min_score}"); continue
        if c.id in active:
            skip("a call with this person is in progress"); continue
        att = _attempts(c, campaign.id)
        na = c.next_action
        due_fu = bool(na and na.type != "retry" and na.due_at is not None and na.due_at <= now and na.campaign_id in (None, campaign.id))
        if na and na.due_at is not None and na.due_at > now and not (att and att[-1].get("outcome") in sch.retry.retry_on):
            skip("scheduled for later"); continue
        if due_fu and att:
            reason = f"follow-up due ({na.type})"
        elif not att:
            reason = f"not called yet in this campaign (score {c.score})"
        elif att[-1].get("outcome") in sch.retry.retry_on:
            unreached = [e for e in att if e.get("outcome") in UNREACHED]
            if len(unreached) >= sch.retry.max_attempts:
                skip(f"gave up after {len(unreached)} unreached attempts"); continue
            if now < att[-1]["t"] + sch.retry.retry_after_minutes * 60:
                skip("waiting before the next retry"); continue
            reason = f"retry {len(unreached) + 1}/{sch.retry.max_attempts} after {att[-1]['outcome']}"
        else:
            skip("already reached; waits for a follow-up"); continue
        ct = (na.call_type if due_fu and na and na.call_type else campaign.call_type_id)
        eligible.append({"customer_id": c.id, "name": c.name, "phone": c.phone, "stage": c.stage, "score": c.score,
                         "call_type": ct, "reason": reason, **({"demo": True, "note": "demo record - no real number"} if is_demo(c) else {}), "_k": (0 if due_fu else 1, -c.score, c.created_at, c.id)})
    eligible.sort(key=lambda r: r["_k"])
    for r in eligible:
        r.pop("_k", None)
    dialable = [r for r in eligible if not r.get("demo")]  # demo records (fake numbers) are listed but never dialled
    last = campaign.stats.get("last_dial_at", 0)
    pacing_wait = max(0, int(last + sch.pacing_seconds - now)) if last else 0
    slots = max(0, sch.max_parallel - active_calls)
    can_now = win["open"] and campaign.status == "running" and slots > 0 and pacing_wait == 0
    return {"campaign_id": campaign.id, "status": campaign.status, "dial_mode": campaign.dial_mode, "window": win,
            "slots": slots, "pacing_wait_seconds": pacing_wait, "dialable_now": can_now,
            "next": dialable[:slots] if can_now else [], "eligible": eligible, "skipped": skipped[:200]}
