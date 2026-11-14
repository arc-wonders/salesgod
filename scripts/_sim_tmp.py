import asyncio, time, json
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.expression import OpenAICompatibleBackend
from acie.schemas import TurnInput, ExpressionParams, Utterance
from acie.world import WorldKnowledge
from acie.understanding import LLMUnderstanding
W = WorldKnowledge(); N = LLMUnderstanding()
async def run(lines, lang="hi"):
    pipe=ACIEPipeline(DecisionEngine(ProductSalesDomain(Product(), world=W)),backend=OpenAICompatibleBackend(),buffer_reply=True, nlu=N)
    T=lambda t: TurnInput(session_id="s",customer_id="c",text=t)
    sess=pipe.session(T("")); sess.expression=ExpressionParams(language=lang)
    g=await pipe.decision.decide("greet",T(""),sess)
    from acie.validator import StreamingValidator
    op=" ".join(StreamingValidator(g, sess.expression).finish())
    sess.history.append(Utterance(role="agent",text=op)); sess.context.record_turn(0,"","opening",False)
    print(" A:",op)
    for l in lines:
        t=time.time()
        r=await pipe.run_turn(T(l))
        n=json.loads(pipe.sessions["s"].dialog.get("_nlu") or "{}")
        print(f" C: {l}\n   nlu: {n.get('intent')} buy={n.get('buying_signal')} topic={n.get('question_topic')} obj={n.get('objection')} gist={n.get('gist')!r} rule={pipe.sessions['s'].dialog.get('_rule_intent')}")
        print(f" A[{r.plan.intent} {time.time()-t:.1f}s{' FALLBACK' if r.validation.used_fallback else ''}]: {r.text}")
    print("---")
async def main():
    await run(["हाँ बताइए","कि यह बताएं कीमत क्या है","क्या आपको नहीं लगता ये बहुत जल्दी जा रहा है, आपने जाना ही नहीं कि मुझे इंटरेस्ट है कि नहीं","हाँ मैं कॉलेज में असाइनमेंट के लिए ChatGPT यूज़ करता हूँ","अच्छा, ये उससे बेहतर कैसे है?","ठीक है, कोशिश करके देखते हैं"])
    await run(["yes, what is it?","hmm I already use ChatGPT, why should I switch?","how much is it?","okay fine, let's do it","yes"],"en")
asyncio.run(main())
