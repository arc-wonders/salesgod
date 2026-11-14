"""Interactive text demo of the full pipeline.

  python -m acie.cli                       # template backend, no GPU needed
  python -m acie.cli --backend openai      # vLLM / any OpenAI-compatible server (ACIE_LLM_* env vars)
  python -m acie.cli --customer c2 --debug
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from .decision.domains.bank_demo import BankDemoDomain
from .decision.engine import DecisionEngine
from .expression import OpenAICompatibleBackend, TemplateBackend
from .personalization.profile_store import JsonProfileStore
from .pipeline import ACIEPipeline
from .schemas import TurnInput


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["template", "openai"], default="template")
    ap.add_argument("--customer", default="c1")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    backend = OpenAICompatibleBackend() if args.backend == "openai" else TemplateBackend()
    pipe = ACIEPipeline(DecisionEngine(BankDemoDomain()), backend=backend, store=JsonProfileStore())
    sid = uuid.uuid4().hex[:8]
    print(f"session {sid} | customer {args.customer} | backend {args.backend}  (Ctrl-D to quit)\n")

    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text:
            continue
        r = await pipe.run_turn(TurnInput(session_id=sid, customer_id=args.customer, text=text))
        print(f"bot> {r.text}")
        if args.debug:
            s, e = r.state, r.expression
            print(f"  intent={r.plan.intent}  state: frus={s.frustration} conf={s.confusion} "
                  f"urg={s.urgency} eng={s.engagement}  signals={s.signals}")
            print(f"  expr: {e.model_dump()}")
            print(f"  valid={r.validation.ok} fallback={r.validation.used_fallback} {r.validation.issues}")
            print(f"  timings_ms={r.timings_ms}")
        if r.plan.end_call:
            break


if __name__ == "__main__":
    asyncio.run(main())
