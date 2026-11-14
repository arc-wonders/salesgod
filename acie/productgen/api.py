"""/api/productgen/* routes and the background job runner. One self-contained block: `register(api, require, DATA, CALLS_DIR)`."""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from . import expand as X
from . import store as S
from .generate import STAGES, generate
from .llm import LLM, LLMError, configured, default_llm
from .spec import Criteria, Spec, resolve

_SAFE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,39}$")
PAGE = Path(__file__).resolve().parents[2] / "voice" / "static" / "productgen.html"


@dataclass
class Job:
    id: str
    kind: str
    tenant: str
    product_id: str
    status: str = "queued"  # queued | running | done | failed | cancelled
    stages: list[dict] = field(default_factory=list)
    version: int | None = None
    error: str = ""
    result: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    finished: float | None = None
    task: asyncio.Task | None = None

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "tenant": self.tenant, "product_id": self.product_id, "status": self.status, "stages": self.stages,
                "version": self.version, "error": self.error, "result": self.result, "created": self.created, "finished": self.finished}


def register(api: FastAPI, require: Callable[[Request], None], data: Path, calls_dir: Path | None = None, kb_root: Path | None = None,
             llm_factory: Callable[[], LLM] | None = None) -> dict[str, Job]:
    data = Path(data)
    calls_dir = Path(calls_dir) if calls_dir else data / "calls"
    jobs: dict[str, Job] = {}
    make_llm = llm_factory or default_llm

    def _ids(tenant: str, pid: str) -> tuple[str, str]:
        if not _SAFE.match(tenant or "") or not _SAFE.match(pid or ""):
            raise HTTPException(400, "tenant and product_id: lower-case letters, digits, - and _")
        return tenant, pid

    def _llm() -> LLM:
        try:
            return make_llm()
        except LLMError as e:
            raise HTTPException(503, str(e))

    def _start(job: Job, coro_fn: Callable[[Job], "asyncio.Future"]) -> Job:
        jobs[job.id] = job

        async def run() -> None:
            job.status = "running"
            try:
                await coro_fn(job)
                if job.status == "running":
                    job.status = "done"
            except asyncio.CancelledError:
                job.status = "cancelled"
            except Exception as e:  # noqa: BLE001  (messages never contain keys: the client redacts them)
                job.status, job.error = "failed", f"{type(e).__name__}: {str(e)[:300]}"
            finally:
                job.finished = time.time()
                for s in job.stages:
                    if s["status"] == "running":
                        s["status"] = "failed" if job.status == "failed" else "cancelled"
        job.task = asyncio.get_running_loop().create_task(run())
        while len(jobs) > 200:  # forget the oldest finished jobs
            old = next((k for k, j in jobs.items() if j.finished), None)
            if old is None:
                break
            jobs.pop(old)
        return job

    def _progress(job: Job):
        def cb(stage: str, status: str, detail: str = "") -> None:
            for s in job.stages:
                if s["id"] == stage:
                    s["status"], s["detail"] = status, detail
                    return
        return cb

    @api.get("/productgen")
    async def productgen_page() -> FileResponse:
        return FileResponse(PAGE, headers={"Cache-Control": "no-cache"})

    @api.get("/api/productgen/defaults")
    async def pg_defaults(request: Request) -> dict:
        require(request)
        return {"criteria": Criteria().model_dump(), "languages": ["en", "hi", "hinglish"], "llm_configured": bool(llm_factory) or configured(),
                "auto_approve": os.getenv("ACIE_PRODUCTGEN_AUTO_APPROVE") == "1"}

    @api.post("/api/productgen/brand-check")
    async def pg_brand(request: Request) -> dict:
        require(request)
        b = await request.json()
        try:
            eff = resolve(Spec(name=str(b.get("name", "")) or "xx", seller=str(b.get("seller", "")), authorised=bool(b.get("authorised")),
                               real_brand=b.get("real_brand")))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, str(e)[:300])
        return {"real_brand": eff.real_brand, "brand_match": eff.brand_match, "provenance": eff.provenance, "allow_synthetic": eff.allow_synthetic, "notes": eff.notes,
                "product_id": eff.product_id}

    @api.post("/api/productgen/start")
    async def pg_start(request: Request) -> dict:
        require(request)
        body = await request.json()
        try:
            spec = Spec.model_validate(body.get("spec", body))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, str(e)[:600])
        llm = _llm()
        eff = resolve(spec)
        job = Job(uuid.uuid4().hex[:10], "generate", spec.tenant, eff.product_id, stages=[{"id": i, "label": l, "status": "pending", "detail": ""} for i, l in STAGES])

        async def go(j: Job) -> None:
            doc = await generate(spec, llm, data, _progress(j))
            j.version = doc["version"]
            j.result = {"passed": doc["report"]["passed"], "publishable": doc["report"]["publishable"], "counts": doc["report"]["counts"]}
        _start(job, go)
        return {"job_id": job.id}

    @api.get("/api/productgen/jobs")
    async def pg_jobs(request: Request) -> dict:
        require(request)
        return {"jobs": [j.public() for j in sorted(jobs.values(), key=lambda j: -j.created)][:50]}

    @api.get("/api/productgen/jobs/{job_id}")
    async def pg_job(job_id: str, request: Request) -> dict:
        require(request)
        j = jobs.get(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        return j.public()

    @api.post("/api/productgen/jobs/{job_id}/cancel")
    async def pg_cancel(job_id: str, request: Request) -> dict:
        require(request)
        j = jobs.get(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        if j.task and not j.task.done():
            j.task.cancel()
        return {"ok": True, "status": j.status}

    @api.get("/api/productgen/products")
    async def pg_products(request: Request) -> dict:
        require(request)
        return {"products": S.list_products(data)}

    @api.get("/api/productgen/versions")
    async def pg_versions(product_id: str, request: Request, tenant: str = "default") -> dict:
        require(request)
        t, p = _ids(tenant, product_id)
        return {"versions": S.list_versions(data, t, p), "published": S.state(data, t, p).get("published")}

    @api.get("/api/productgen/version")
    async def pg_version(product_id: str, n: int, request: Request, tenant: str = "default") -> dict:
        require(request)
        t, p = _ids(tenant, product_id)
        v = S.load_version(data, t, p, n)
        if not v:
            raise HTTPException(404, "unknown version")
        d = S._read(S.pdir(data, t, p) / f"v{n}.diff.json")
        return {"version": v, "diff": (d or {}).get("diff"), "published": S.state(data, t, p).get("published")}

    @api.get("/api/productgen/diff")
    async def pg_diff(product_id: str, a: int, b: int, request: Request, tenant: str = "default") -> dict:
        require(request)
        t, p = _ids(tenant, product_id)
        try:
            return S.get_diff(data, t, p, a, b)
        except KeyError:
            raise HTTPException(404, "unknown version")

    @api.post("/api/productgen/publish")
    async def pg_publish(request: Request) -> dict:
        require(request)
        b = await request.json()
        t, p = _ids(b.get("tenant", "default"), b.get("product_id", ""))
        try:
            return S.publish(data, t, p, int(b.get("version")), kb_root)
        except S.PublishError as e:
            raise HTTPException(409, str(e))
        except (TypeError, ValueError):
            raise HTTPException(400, "version: a number")

    @api.post("/api/productgen/expand")
    async def pg_expand(request: Request) -> dict:
        require(request)
        b = await request.json()
        t, p = _ids(b.get("tenant", "default"), b.get("product_id", ""))
        node = b.get("node") or {}
        if node.get("kind") not in ("topic", "entry", "playbook") or not node.get("id"):
            raise HTTPException(400, "node: {kind: topic|entry|playbook, id}")
        if not S.latest_full(data, t, p):
            raise HTTPException(404, "generate the product first")
        llm = _llm()
        job = Job(uuid.uuid4().hex[:10], "expand", t, p, stages=[{"id": "expand", "label": "Expand " + str(node["id"]), "status": "running", "detail": ""}])

        async def go(j: Job) -> None:
            doc = await X.expand_node(data, t, p, llm, node, str(b.get("instruction", ""))[:600], b.get("version"))
            j.version = doc["version"]
            j.stages[0].update(status="done", detail=doc["note"])
            j.result = {"note": doc["note"], "publishable": doc["report"]["publishable"]}
        _start(job, go)
        return {"job_id": job.id}

    def _live_gaps(t: str, p: str) -> list[dict]:
        v = S.latest_full(data, t, p)
        if not v:
            return []
        from acie.knowledge.schema import KnowledgeBase
        pub = S.state(data, t, p).get("published")
        pv = S.load_version(data, t, p, pub) if pub else None
        kb = KnowledgeBase.model_validate(((pv or v)["bundle"])["kb"])
        return X.mine_gaps(calls_dir, p, kb, (pv or v)["bundle"]["product"], skip=set(S.state(data, t, p).get("gap_seen", {})))

    @api.get("/api/productgen/gaps")
    async def pg_gaps(product_id: str, request: Request, tenant: str = "default") -> dict:
        require(request)
        t, p = _ids(tenant, product_id)
        pending = []
        for n in S.version_numbers(data, t, p):
            v = S.load_version(data, t, p, n)
            if v and v.get("kind") == "gap_proposals" and any(x["status"] == "pending" for x in v["proposals"]):
                pending.append({"version": n, "created_at": v.get("created_at"), "note": v.get("note"), "proposals": v["proposals"]})
        return {"unanswered": _live_gaps(t, p), "pending": pending}

    @api.post("/api/productgen/gaps/scan")
    async def pg_gaps_scan(request: Request) -> dict:
        require(request)
        b = await request.json()
        t, p = _ids(b.get("tenant", "default"), b.get("product_id", ""))
        if not S.latest_full(data, t, p):
            raise HTTPException(404, "generate the product first")
        llm = _llm()
        job = Job(uuid.uuid4().hex[:10], "gaps", t, p, stages=[{"id": "gaps", "label": "Mine calls for gaps", "status": "running", "detail": ""}])

        async def go(j: Job) -> None:
            doc = await X.propose_gaps(data, t, p, llm, calls_dir, kb_root)
            if doc is None:
                j.stages[0].update(status="done", detail="nothing new")
                j.result = {"proposals": 0}
                return
            j.version = doc["version"]
            j.stages[0].update(status="done", detail=doc["note"])
            j.result = {"proposals": len(doc["proposals"]), "auto_approved": bool(doc.get("auto_approved"))}
        _start(job, go)
        return {"job_id": job.id}

    @api.post("/api/productgen/approve-gaps")
    async def pg_approve(request: Request) -> dict:
        require(request)
        b = await request.json()
        t, p = _ids(b.get("tenant", "default"), b.get("product_id", ""))
        try:
            return X.approve_gaps(data, t, p, int(b.get("version")), list(b.get("approve") or []), list(b.get("reject") or []), bool(b.get("publish")), kb_root)
        except KeyError as e:
            raise HTTPException(404, str(e))
        except S.PublishError as e:
            raise HTTPException(409, str(e))
        except (TypeError, ValueError):
            raise HTTPException(400, "version: a number")

    return jobs
