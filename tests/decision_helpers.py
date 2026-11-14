from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import TurnInput


def T(text, cid="+910000000000"):
    return TurnInput(session_id="s", customer_id=cid, text=text)


async def run(lines, product=None, nlu=None, **kw):
    dom = ProductSalesDomain(product or Product(), customer_name="Yash", **kw)
    pipe = ACIEPipeline(DecisionEngine(dom), nlu=nlu)
    sess = pipe.session(T(""))
    opening = await pipe.decision.decide("greet", T(""), sess)
    sess.context.record_turn(0, "", opening.intent, False)
    out, last = [opening.intent], opening
    for line in lines:
        r = await pipe.run_turn(T(line))
        out.append(r.plan.intent)
        last = r
    return out, last, pipe, dom, opening
