from pathlib import Path

from acie.playbook import Playbook, load
from acie.understanding import Understanding

PB = load(Path("/nonexistent"), "default", "claude-pro")


def test_seed_playbook_loads_and_picks_the_situation():
    assert PB is not None and len(PB.nodes) >= 14
    pick = lambda t, **k: PB.pick(t, Understanding(**k) if k else None)[0]
    assert pick("why should I even buy this?").id == "why_buy"
    assert pick("ChatGPT se better kaise hai?").id == "compare_chatgpt"
    assert pick("बहुत महंगा है यार").id == "price_high"
    assert pick("कहीं fraud तो नहीं?").id == "trust"
    assert pick("main college student hoon, assignments ke liye").id == "student"
    assert pick("दुकान का GST बिल बनेगा?").id == "business"
    assert pick("ok", intent="affirm", buying_signal=0.3) is None


def test_signal_nodes_need_their_signal():
    hit = PB.pick("tum baar baar same bol rahe ho", Understanding(frustration=0.7))[0]
    assert hit.id == "annoyed"
    miss = PB.pick("tum baar baar same bol rahe ho", Understanding(frustration=0.1))[0]
    assert miss is None or miss.id != "annoyed"


async def test_domain_uses_playbook_points_and_rotates_them():
    from acie.decision.domains.product_sales import Product, ProductSalesDomain
    from acie.decision.engine import DecisionEngine
    from acie.pipeline import ACIEPipeline
    from acie.schemas import TurnInput
    T = lambda x: TurnInput(session_id="s", customer_id="+910000000000", text=x)
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(Product(), playbook=PB)))
    await pipe.run_turn(T("yes tell me"))
    r1 = await pipe.run_turn(T("why should I buy it, what's in it for me?"))
    assert "TALKING POINTS" in (r1.plan.context or "") and "why_buy" == pipe.sessions["s"].dialog["pb_node"]
    first = r1.plan.context.split("TALKING POINTS")[1]
    r2 = await pipe.run_turn(T("but why should I buy it?"))
    assert r2.plan.context.split("TALKING POINTS")[1] != first  # used points rotate out
