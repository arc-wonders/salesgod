"""Call-type registry: the built-ins plus the operator's own (data/voice/calltypes.json), validated.

calltypes.json is a list of CallType objects, or {"calltypes": [...]}, or {id: {...}}. A bad entry is reported by
`load_custom` and skipped (it never breaks a call); `get` returns None for an unknown id so callers fall back to today's flow.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from .builtins import builtin_call_types
from .schema import CallType, validate_call_type

log = logging.getLogger("acie.calltypes")


def custom_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "calltypes.json"


def parse_custom(raw) -> list[dict]:
    if isinstance(raw, dict) and isinstance(raw.get("calltypes"), list):
        raw = raw["calltypes"]
    if isinstance(raw, dict):
        raw = [{**v, "id": v.get("id", k)} for k, v in raw.items() if isinstance(v, dict)]
    return [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []


def load_custom(data_dir: Path | str) -> tuple[list[CallType], dict[str, list[str]]]:
    """(valid custom call types, {id: [problems]} for the rejected ones). A missing or unreadable file is just empty."""
    path = custom_path(data_dir)
    if not path.exists():
        return [], {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        return [], {"<file>": [f"calltypes.json is unreadable: {e}"]}
    ok: list[CallType] = []
    bad: dict[str, list[str]] = {}
    builtin_ids = set(builtin_call_types())
    for item in parse_custom(raw):
        cid = str(item.get("id", "?"))
        try:
            ct = CallType.model_validate({**item, "builtin": False})
        except ValidationError as e:
            bad[cid] = [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()]
            continue
        errs = validate_call_type(ct)
        if ct.id in builtin_ids and not ct.project_id:
            errs.append(f"id {ct.id!r} is a built-in; duplicate it under another id or give a project_id")
        if errs:
            bad[ct.id] = errs
        else:
            ok.append(ct)
    return ok, bad


class CallTypeRegistry:
    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.builtins = builtin_call_types()
        self.custom: list[CallType] = []
        self.rejected: dict[str, list[str]] = {}
        self.reload()

    def reload(self) -> None:
        if self.data_dir is not None:
            self.custom, self.rejected = load_custom(self.data_dir)
            for cid, errs in self.rejected.items():
                log.warning("call type %s ignored: %s", cid, "; ".join(errs))

    def all(self, project_id: str = "") -> list[CallType]:
        """Built-ins, then this project's own (and project-less custom ones)."""
        mine = [c for c in self.custom if not c.project_id or c.project_id == project_id]
        return list(self.builtins.values()) + mine

    def get(self, call_type_id: str, project_id: str = "") -> CallType | None:
        """This project's own type, else a project-less custom one, else the built-in. Another project's type is never returned."""
        mine = [c for c in self.custom if c.id == call_type_id]
        for c in mine:
            if project_id and c.project_id == project_id:
                return c
        for c in mine:
            if not c.project_id:
                return c
        return self.builtins.get(call_type_id)

    @staticmethod
    def validate(ct: CallType) -> list[str]:
        return validate_call_type(ct)
