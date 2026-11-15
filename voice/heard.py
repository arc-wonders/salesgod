"""What the caller actually HEARD of an agent turn (barge-in cuts playback).

Every TTS request is one entry: its text and the line time its audio occupies at Plivo (start of the first chunk,
end of the last one, both on the monotonic clock as scheduled by CallSession.play_pcm: Plivo plays the queue in
order at 8 kHz, so scheduled time = playback time). Bytes are sent ahead of playback, so "sent" is not "heard": at a
cut (clearAudio) at time `cut`, the caller heard audio up to `cut - margin` (network + jitter buffer). A sentence
counts as heard only when all of it fits before that point (time spread over the entry by characters).
"""

from __future__ import annotations

import re

HEARD_MARGIN_S = 0.35
_SENT = re.compile(r"(?<=[.!?।])\s+")


class PlayLog:
    def __init__(self, keep: int = 60) -> None:
        self.entries: list[dict] = []
        self.keep = keep

    def begin(self, epoch: int, text: str) -> dict:
        e = {"epoch": epoch, "text": text, "start": None, "end": None, "done": False}
        self.entries = self.entries[-self.keep:] + [e]
        return e

    @staticmethod
    def sent(e: dict | None, start: float, end: float) -> None:
        if e is None:
            return
        if e["start"] is None:
            e["start"] = start
        e["end"] = end

    def resumed(self, epoch: int, at: float, resume_start: float, new_end: float, backup_s: float) -> None:
        """A possible barge-in paused playback at `at`; the held audio was re-sent from `backup_s` before the cut."""
        shift = max(0.0, resume_start - at) + backup_s
        es = [e for e in self.entries if e["epoch"] == epoch and e["end"] is not None and e["end"] > at]
        for e in es:
            if e["start"] >= at:
                e["start"] += shift
            e["end"] += shift
        if es:
            es[-1]["end"] = max(es[-1]["end"], new_end)

    def heard(self, epoch: int, cut: float, margin: float = HEARD_MARGIN_S) -> tuple[str, str, float]:
        """-> (heard text, full text, heard share of the characters) for the turn spoken in `epoch`."""
        until = cut - margin
        heard, full = [], []
        for e in self.entries:
            if e["epoch"] != epoch or not e["text"]:
                continue
            full.append(e["text"])
            if e["start"] is None or e["start"] >= until:
                continue
            if e["done"] and e["end"] is not None and e["end"] <= until:
                heard.append(e["text"])
                continue
            total = max((e["end"] or e["start"]) - e["start"], len(e["text"]) / 15.0)  # still streaming: ~15 chars/s
            frac = (until - e["start"]) / total if total > 0 else 0.0
            parts = [x for x in _SENT.split(e["text"].strip()) if x]
            n = sum(len(x) for x in parts) or 1
            acc = 0
            for x in parts:
                acc += len(x)
                if acc / n <= frac + 1e-9:
                    heard.append(x)
                else:
                    break
        h, f = " ".join(heard).strip(), " ".join(full).strip()
        return h, f, (len(h) / len(f)) if f else 1.0
