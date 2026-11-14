import random

ROWS = []


def T(lang, text, intent, f=0, c=0, u=0, e=1, amt=None, ex=None, pend=None, slot=None,
      agent=None, prev=None, b=False, notime=False, notes=""):
    ROWS.append(dict(lang=lang, text=text, intent=intent, f=f, c=c, u=u, e=e, amt=amt,
                     ex=ex or {}, pend=pend, slot=slot, agent=agent, prev=prev, b=b,
                     notime=notime, notes=notes))


def timing_for(r, idx):
    rnd = random.Random(1000 + idx)
    if r["notime"] or rnd.random() < 0.12:
        return None
    lat = rnd.gauss(950, 250)
    rate = rnd.gauss(2.6 if r["lang"] != "en" else 2.4, 0.3)
    if r["e"] == 0:
        lat += rnd.uniform(350, 1100)
        rate -= rnd.uniform(0.3, 0.7)
    if r["c"] >= 1:
        lat += rnd.uniform(150, 600)
        rate -= rnd.uniform(0.0, 0.4)
    if r["u"] >= 1 or r["f"] == 2:
        lat -= rnd.uniform(150, 450)
        rate += rnd.uniform(0.2, 0.8)
    lat = int(max(300, min(2500, lat)) // 10 * 10)
    rate = round(max(1.3, min(4.2, rate)), 1)
    return {"barge_in": bool(r["b"]), "response_latency_ms": lat, "speech_rate_wps": rate}
