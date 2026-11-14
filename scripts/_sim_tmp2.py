import asyncio
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.expression import OpenAICompatibleBackend
from acie.schemas import TurnInput, ExpressionParams, Utterance
async def run():
    pipe=ACIEPipeline(DecisionEngine(ProductSalesDomain(Product())),backend=OpenAICompatibleBackend(),buffer_reply=True)
    T=lambda t: TurnInput(session_id="s",customer_id="c",text=t)
    sess=pipe.session(T("")); sess.expression=ExpressionParams(language="hi")
    await pipe.decision.decide("greet",T(""),sess); sess.history.append(Utterance(role="agent",text="hi"))
    sess.dialog["pending"]="discovery"
    for l in ["हाँ मैं कॉलेज में असाइनमेंट के लिए ChatGPT यूज़ करता हूँ","हाँ बताइए"]:
        r=await pipe.run_turn(T(l)); print(r.plan.intent, r.text, r.validation.issues)
asyncio.run(run())
