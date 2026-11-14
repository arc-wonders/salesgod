"""CSV import: column mapping, E.164 normalisation, dedupe by phone, DNC respected. preview() writes nothing; commit() does."""

from __future__ import annotations

import csv
import io
import time

from voice.plivo import e164

from .models import Customer, Project, STAGES, LANGS
from .store import JsonStore

MAX_ROWS = 5000
MAX_BYTES = 2_000_000
CORE = ["phone", "name", "email", "language", "city", "tags", "stage", "notes"]
_GUESS = {
    "phone": ("phone", "mobile", "mobile no", "mobile number", "phone number", "contact", "contact number", "number", "whatsapp", "cell", "tel"),
    "name": ("name", "full name", "customer name", "contact name", "lead name"),
    "email": ("email", "e-mail", "email id", "mail"),
    "language": ("language", "lang", "preferred language"),
    "city": ("city", "town", "location"),
    "tags": ("tags", "tag", "labels", "segment"),
    "stage": ("stage", "status"),
    "notes": ("notes", "note", "remarks", "comment", "comments"),
}


class ImportError_(ValueError):
    pass


def parse_csv(text: str) -> tuple[list[str], list[dict]]:
    if len(text.encode("utf-8", "ignore")) > MAX_BYTES:
        raise ImportError_("file too large (max 2 MB)")
    text = text.lstrip("﻿")
    if "," in text.split("\n", 1)[0]:  # a comma in the header row: plain CSV, no guessing (the console always sends commas)
        dialect = csv.excel
    else:
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
    rd = csv.reader(io.StringIO(text), dialect)
    rows = [r for r in rd if any(c.strip() for c in r)]
    if not rows:
        raise ImportError_("the file is empty")
    header = [h.strip() for h in rows[0]]
    if len(rows) - 1 > MAX_ROWS:
        raise ImportError_(f"too many rows (max {MAX_ROWS})")
    out = []
    for r in rows[1:]:
        out.append({h: (r[i].strip() if i < len(r) else "") for i, h in enumerate(header) if h})
    return [h for h in header if h], out


def guess_mapping(columns: list[str], project: Project | None = None) -> dict[str, str]:
    m: dict[str, str] = {}
    norm = {c.lower().strip(): c for c in columns}
    for field, names in _GUESS.items():
        for n in names:
            if n in norm and norm[n] not in m.values():
                m[field] = norm[n]
                break
    for fd in (project.fields if project else []):
        for n in (fd.key, fd.label.lower()):
            if n and n in norm and norm[n] not in m.values():
                m[f"custom.{fd.key}"] = norm[n]
                break
    return m


def _split_tags(v: str) -> list[str]:
    return [t.strip().lower() for t in v.replace(";", ",").replace("|", ",").split(",") if t.strip()]


def _row_to_customer(row: dict, mapping: dict[str, str], project: Project, default_tags: list[str]) -> tuple[Customer | None, str]:
    def g(f: str) -> str:
        col = mapping.get(f)
        return (row.get(col, "") if col else "").strip()

    phone = e164(g("phone"))
    if not phone:
        return None, "not a valid phone number"
    c = Customer(project_id=project.id, phone=phone, name=g("name")[:80], email=g("email")[:120], city=g("city")[:60],
                 source="csv", notes=g("notes")[:1000])
    lang = {"hindi": "hi", "english": "en"}.get(g("language").lower(), g("language").lower())
    c.language = lang if lang in LANGS else (project.languages[0] if project.languages else "hi")
    st = g("stage").lower().replace(" ", "_")
    if st in STAGES and st != "do_not_call":  # an import can never un-DNC or force DNC through a column
        c.stage = st
    c.tags = sorted(set(_split_tags(g("tags")) + default_tags))
    for f in project.fields:
        v = g(f"custom.{f.key}")
        if not v:
            continue
        if f.type == "number":  # same typing as a hand-edited customer (api._check_custom)
            try:
                v = float(v.replace(",", ""))
            except ValueError:
                continue
        elif f.type == "bool":
            v = v.lower() in ("1", "true", "yes", "y")
        c.custom[f.key] = v
    return c, ""


def run(store: JsonStore, project: Project, text: str, mapping: dict[str, str] | None = None, commit: bool = False,
        on_duplicate: str = "skip", default_tags: list[str] | None = None, now: float | None = None) -> dict:
    columns, rows = parse_csv(text)
    mapping = mapping or guess_mapping(columns, project)
    bad = [v for v in mapping.values() if v not in columns]
    if bad:
        raise ImportError_(f"mapped columns not in the file: {bad}")
    if "phone" not in mapping:
        raise ImportError_("map a phone column")
    if on_duplicate not in ("skip", "update"):
        raise ImportError_("on_duplicate: skip or update")
    now = now or time.time()
    dnc = store.dnc()
    existing = {c.phone: c for c in store.customers(project.id)}
    seen: dict[str, int] = {}
    report, to_write = [], []
    counts = {"new": 0, "updated": 0, "duplicate": 0, "invalid": 0, "dnc": 0}
    for i, row in enumerate(rows, start=2):  # row 1 is the header
        c, err = _row_to_customer(row, mapping, project, default_tags or [])
        if c is None:
            counts["invalid"] += 1
            report.append({"row": i, "status": "invalid", "error": err, "raw": {k: v for k, v in list(row.items())[:6]}})
            continue
        if c.phone in seen:
            counts["duplicate"] += 1
            report.append({"row": i, "status": "duplicate_in_file", "phone": c.phone, "first_row": seen[c.phone]})
            continue
        seen[c.phone] = i
        is_dnc = c.phone in dnc or (c.phone in existing and (existing[c.phone].consent.dnc or existing[c.phone].stage == "do_not_call"))
        if c.phone in existing:
            ex = existing[c.phone]
            if on_duplicate == "update" and not is_dnc:
                for f in ("name", "email", "city"):
                    if getattr(c, f) and not getattr(ex, f):
                        setattr(ex, f, getattr(c, f))
                ex.tags = sorted(set(ex.tags) | set(c.tags))
                for k, v in c.custom.items():
                    ex.custom.setdefault(k, v)
                ex.updated_at = now
                to_write.append(ex)
                counts["updated"] += 1
                report.append({"row": i, "status": "updated", "phone": c.phone, "customer_id": ex.id})
            else:
                counts["dnc" if is_dnc else "duplicate"] += 1
                report.append({"row": i, "status": "dnc_existing" if is_dnc else "duplicate_existing", "phone": c.phone, "customer_id": ex.id})
            continue
        if is_dnc:  # kept on file for the record, flagged, never callable
            c.stage = "do_not_call"
            c.consent.dnc, c.consent.reason, c.consent.at = True, "on the do-not-call list at import", now
            counts["dnc"] += 1
            report.append({"row": i, "status": "dnc", "phone": c.phone, "customer_id": c.id})
        else:
            counts["new"] += 1
            report.append({"row": i, "status": "new", "phone": c.phone, "name": c.name})
        c.created_at = c.updated_at = now
        c.events.append({"t": now, "type": "created", "source": "csv"})
        to_write.append(c)
    if commit:
        from .scoring import score
        for c in to_write:
            c.score = score(c, now)
        store.put_customers(to_write)
    return {"committed": commit, "columns": columns, "mapping": mapping, "total_rows": len(rows), "counts": counts,
            "rows": report[:200], "rows_truncated": len(report) > 200}
