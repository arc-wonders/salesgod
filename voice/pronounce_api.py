"""/api/pronunciations: the TTS pronunciation lexicon (acie/pronounce.py), suggestions and round-trip checks.

GET  /api/pronunciations?product=ID          layers (defaults, global, tenant, product, agent), effective lexicon, suggestions
PUT  /api/pronunciations/global              {lexicon: {word: {hi, hinglish, en}}}
GET  /api/pronunciations/product/{pid}       the product's own layer
PUT  /api/pronunciations/product/{pid}       {lexicon}  (Hindi values are mirrored into Product.pronunciations)
POST /api/pronunciations/suggest             {product, verify?: bool, limit?: int} -> candidates + LLM respellings (suggestions, not applied)
POST /api/pronunciations/accept              {product, words: [word], picks?: {word: {lang: value}}}
POST /api/pronunciations/verify              {product?, word, lang, candidates?: [str], agent?} -> round-trip score per candidate
POST /api/pronunciations/preview             {text, lang, agent?, product?, raw?: bool} -> audio/wav (what the call would say)

Round trips (synthesise with the agent's voice -> the service's own Whisper -> score) are on demand only and refused
while a call is live, so they never take GPU time from a caller.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
from fastapi import HTTPException, Request, Response

from acie.pronounce import GLOBAL, LANGS, Resolver, _wrap, default_store, merged, norm_lexicon
from acie import pronounce_suggest as sug

log = logging.getLogger("voice.pronounce")


def register(api, state, _require, DATA) -> None:
    store = default_store(DATA)

    def _product(pid: str):
        p = state.products().get(pid)
        if p is None:
            raise HTTPException(404, f"unknown product {pid!r}")
        return p

    def _agent(agent_id: str = "", product=None):
        a = None
        if agent_id:
            a = state.agents.get(agent_id)
        if a is None:
            a = state.agent_for(SimpleNamespace(agent_id=""), product)
        return a

    def _live() -> bool:
        return any(c.status in ("dialing", "ringing", "live") for c in state.calls.values()) or bool(state.live_sessions)

    async def _synth(text: str, lang: str, agent) -> bytes:
        if state.tts is None:
            raise HTTPException(503, "TTS is still loading")
        from acie.agents import voice_chain
        kw = {"voices": voice_chain(agent, lang), "speed": agent.speaking_rate} \
            if agent is not None and getattr(state.tts, "accepts_voice", False) else {}
        pcm = bytearray()
        async for chunk in state.tts.stream(text, lang, **kw):
            pcm += chunk
        return bytes(pcm)

    async def _hear(pcm: bytes, lang: str) -> str:
        """The live service's Whisper (already loaded): large-v3 for English, the Hindi model for Hindi/Hinglish."""
        if state.stt is None:
            raise HTTPException(503, "STT is still loading")
        from .audio import upsample_8k_to_16k
        a = upsample_8k_to_16k(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)
        stt = state.stt

        def run() -> str:
            # Hindi audio: the Hindi model's Devanagari AND large-v3's English reading (it keeps names in Latin)
            parts = []
            if lang != "en":
                parts.append(stt._run_hi(a) if stt.hi_model is not None else stt._run(a, "hi", None))
            parts.append(stt._run(a, "en", None))
            return " | ".join(" ".join(s.text.strip() for s in segs) for segs in parts)
        async with stt.lock:
            return await asyncio.to_thread(run)

    async def _round_trip(word: str, values: list[str], lang: str, agent, entry: dict) -> list[dict]:
        out = []
        for v in values[:4]:
            line = sug.sentence_for(word, _wrap(v) if v else word, lang)
            try:
                heard = await _hear(await _synth(line, lang, agent), lang)
            except HTTPException:
                raise
            except Exception as e:  # noqa: BLE001
                out.append({"value": v, "error": str(e)[:160]})
                continue
            out.append({"value": v, "spoken": line, "heard": heard, "score": sug.score(heard, word, entry, lang)})
        return out

    def _wav(pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(pcm)
        return buf.getvalue()

    @api.get("/api/pronunciations")
    async def pron_get(request: Request, product: str = "", agent: str = "") -> dict:
        _require(request)
        p = _product(product) if product else None
        a = _agent(agent, p) if p is not None or agent else None
        tenant = getattr(p, "tenant", "") if p is not None else ""
        return {"defaults": GLOBAL, "global": store.scope("global"),
                "tenant": {"id": tenant, "lexicon": store.scope("tenants", tenant) if tenant else {}},
                "product": {"id": product, "lexicon": {**norm_lexicon(getattr(p, "pronunciations", {}) or {}),
                                                       **store.scope("products", product)} if p is not None else {}},
                "agent": {"id": getattr(a, "id", ""), "lexicon": store.scope("agents", getattr(a, "id", "")) if a else {}},
                "effective": merged(store, tenant, p, getattr(a, "id", "")),
                "suggestions": (store.data().get("suggestions") or {}).get(product, []) if product else [],
                "languages": list(LANGS)}

    @api.put("/api/pronunciations/global")
    async def pron_put_global(request: Request) -> dict:
        _require(request)
        body = await request.json()
        return {"global": store.put("global", "", body.get("lexicon") if isinstance(body, dict) else {})}

    @api.get("/api/pronunciations/product/{pid}")
    async def pron_get_product(pid: str, request: Request) -> dict:
        _require(request)
        p = _product(pid)
        return {"product": pid, "lexicon": {**norm_lexicon(p.pronunciations or {}), **store.scope("products", pid)}}

    def _save_product_layer(pid: str, lex: dict) -> dict:
        clean = store.put("products", pid, lex)
        products = state.products()
        if pid in products:  # Hindi values mirrored into Product.pronunciations (the studio's facts read it)
            products[pid] = products[pid].model_copy(update={"pronunciations": {k: e["hi"] for k, e in clean.items() if e.get("hi")}})
            state.save_products(products)
        return clean

    @api.put("/api/pronunciations/product/{pid}")
    async def pron_put_product(pid: str, request: Request) -> dict:
        _require(request)
        _product(pid)
        body = await request.json()
        return {"product": pid, "lexicon": _save_product_layer(pid, body.get("lexicon") if isinstance(body, dict) else {})}

    @api.post("/api/pronunciations/suggest")
    async def pron_suggest(request: Request) -> dict:
        """Auto-detected words + LLM respellings, stored as suggestions (not applied until accepted). verify=true also
        runs the round trip for each language and orders the candidates by score (refused while a call is live)."""
        _require(request)
        body = await request.json()
        pid = str(body.get("product") or "")
        p = _product(pid)
        eff = merged(store, p.tenant, p)
        pb = ""
        try:
            raw = state.products()[pid].model_dump()
        except Exception:  # noqa: BLE001
            raw = {}
        cands = sug.candidates(raw, eff, pb, limit=int(body.get("limit") or 20))
        words = [{"word": w, "kind": "known", "example": ""} for w in body.get("words") or [] if isinstance(w, str)]
        try:
            resp = await sug.llm_respellings(cands + words, p.product_name)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"LLM failed: {str(e)[:200]}")
        verify = bool(body.get("verify"))
        if verify and _live():
            raise HTTPException(409, "a call is live: round-trip checks wait until it ends")
        agent = _agent("", p)
        new = []
        for c in cands + words:
            r = resp.get(c["word"]) or next((v for k, v in resp.items() if k.lower() == c["word"].lower()), None)
            if not r:
                continue
            item = {"word": c["word"], "kind": c["kind"], "example": c.get("example", ""), "candidates": r,
                    "values": {lang: vals[0] for lang, vals in r.items() if vals}, "status": "suggested", "at": time.time()}
            if verify:
                item["checks"] = {}
                for lang in LANGS:
                    vals = [v for v in r.get(lang) or [] if v]
                    if not vals:
                        continue
                    res = await _round_trip(c["word"], vals, lang, agent, {})
                    good = [x for x in res if "score" in x]
                    if good:
                        best = max(good, key=lambda x: x["score"])
                        item["values"][lang] = best["value"]
                        item["checks"][lang] = {"score": best["score"], "heard": best["heard"], "tried": res}
            new.append(item)
        d = store.data()
        d = {**d, "suggestions": {**(d.get("suggestions") or {}), pid: sug.merge_suggestions((d.get("suggestions") or {}).get(pid, []), new)}}
        store.save(d)
        return {"product": pid, "suggestions": new, "candidates": cands}

    @api.post("/api/pronunciations/accept")
    async def pron_accept(request: Request) -> dict:
        _require(request)
        body = await request.json()
        pid = str(body.get("product") or "")
        _product(pid)
        words = {str(w).lower() for w in body.get("words") or []}
        picks = body.get("picks") if isinstance(body.get("picks"), dict) else {}
        d = store.data()
        pending = (d.get("suggestions") or {}).get(pid, [])
        lex = {**norm_lexicon(_product(pid).pronunciations or {}), **store.scope("products", pid)}
        keep = []
        for s in pending:
            w = str(s.get("word", ""))
            if w.lower() in words:
                lex[w] = {**lex.get(w, {}), **norm_lexicon({w: s.get("values") or {}}).get(w, {}),
                          **norm_lexicon({w: picks.get(w) or {}}).get(w, {})}
            else:
                keep.append(s)
        d = store.data()
        store.save({**d, "suggestions": {**(d.get("suggestions") or {}), pid: keep}})
        return {"product": pid, "lexicon": _save_product_layer(pid, lex), "suggestions": keep}

    @api.post("/api/pronunciations/verify")
    async def pron_verify(request: Request) -> dict:
        """'Sounds right?': synthesise the word (each candidate) in a short line with the agent's voice, transcribe it
        with the service's Whisper and score whether the word came back. Picks the best candidate."""
        _require(request)
        body = await request.json()
        word, lang = str(body.get("word") or "").strip()[:60], str(body.get("lang") or "hi")
        if not word or lang not in LANGS:
            raise HTTPException(422, "word and lang (hi|hinglish|en)")
        if _live():
            raise HTTPException(409, "a call is live: round-trip checks wait until it ends")
        p = _product(str(body["product"])) if body.get("product") else None
        eff = merged(store, getattr(p, "tenant", "") if p else "", p)
        entry = eff.get(word) or next((v for k, v in eff.items() if k.lower() == word.lower()), {})
        given = body.get("candidates") if isinstance(body.get("candidates"), list) else []
        vals = list(dict.fromkeys(str(v).strip() for v in given)) or [entry.get(lang, "")]
        if "" not in vals and body.get("include_plain", True):
            vals = [v for v in vals if v][:2] + [""]  # the unmodified word as the baseline
        vals = vals[:4]
        res = await _round_trip(word, vals, lang, _agent(str(body.get("agent") or ""), p), entry)
        good = [x for x in res if "score" in x]
        best = max(good, key=lambda x: x["score"]) if good else None
        return {"word": word, "lang": lang, "results": res, "best": best}

    @api.post("/api/pronunciations/preview")
    async def pron_preview(request: Request) -> Response:
        """The line as a call would say it: lexicon applied (unless raw), the agent's voice. Returns audio/wav."""
        _require(request)
        body = await request.json()
        lang = str(body.get("lang") or body.get("language") or "hi")
        if lang not in LANGS:
            raise HTTPException(422, "lang: hi|hinglish|en")
        text = str(body.get("text") or "").strip()[:300]
        if not text:
            raise HTTPException(422, "text")
        p = _product(str(body["product"])) if body.get("product") else None
        a = _agent(str(body.get("agent") or ""), p)
        spoken = text if body.get("raw") else Resolver(merged(store, getattr(p, "tenant", "") if p else "", p,
                                                              getattr(a, "id", ""))).apply(text, lang)
        try:
            pcm = await _synth(spoken, lang, a)
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"TTS failed: {str(e)[:200]}")
        return Response(_wav(pcm), media_type="audio/wav", headers={"Cache-Control": "no-store", "X-Spoken": quote(spoken)})
