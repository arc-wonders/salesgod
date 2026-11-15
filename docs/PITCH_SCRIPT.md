# SalesGod pitch script (Team Coffee Please) · ~5.5 minutes

**Slide 1 · Hook (0:00–0:20)**
Play 10 seconds of a real call where the agent haggles in Hinglish (backup: say the hook).
"What if your best salesperson spoke Hinglish, never forgot a word you said, and could never be talked below the floor price? We built that. It's SalesGod."

**Slide 2 · Problem (0:20–1:00)**
"Half of all cold calls in India never connect. About 2% convert. A human telecaller costs ₹7–8 a minute. AI voice bots are cheaper, but they break in two ways. One: they let the LLM decide the price, and a chatbot once agreed to sell a Chevy Tahoe for one dollar. Two: they forget. Interrupt them mid-price and the deal is gone."

**Slide 3 · Insight (1:00–1:40)**
"Our insight: LLMs are brilliant talkers and terrible negotiators. So we split the job. A deterministic brain decides every move. The LLM only puts it into words, and a validator checks every number before it's spoken. That gives three guarantees: never below the floor, never a made-up number, never losing the thread."

**Slide 4 · The brain (1:40–2:40)**
Walk left to right. "We hear: phone audio, Whisper, and a tracker of how much of our sentence the caller actually heard. We sense: intent and objection from a 14B model, plus 14 mood dimensions, 7 social signals like sarcasm and impatience, and a distress score. We decide, in code: nine gates with safety first, the playbook tree, the haggle engine with a hidden floor, and a style policy. Then we speak: the LLM words the plan, the validator checks it, Fish voices it. NATS remembers the deal and the caller across interruptions and calls. And every turn is visible live on our console."

**Slide 5 · Live demo (2:40–4:00)**
Call the agent on speaker with the console on screen.
1. "₹500 de do" → it jokes, no discount.
2. "₹2,500 final" → one computed concession.
3. "Theek hai" → the price locks.
4. "Where were we?" → it reads the deal back.
5. Interrupt it mid-price → it repeats only what you missed.
Backup: a recorded call, in case the network drops.

**Slide 6 · Proof (4:00–4:30)**
"We attacked it ourselves: 12 difficult personas, rude, sarcastic, elderly, hagglers, across three products, 144 conversations. Faults fell across the board, and refund questions misread as price went to zero. 735 automated tests, and 25+ real phone calls by the team today."

**Slide 7 · Market & model (4:30–5:10)**
"Voice AI agents are going from $2.5B to $35B. AI minutes already cost less than people. We charge per minute, take a small share of each closed deal (we can, because our agreed price is a logged field, not free text), and sell a monthly studio plan. Others sell the voice plumbing. We built the sales brain."

**Slide 8 · Vision & ask (5:10–5:30)**
"Next 90 days: pilots with gyms, D2C brands and edtech. Our ask: three pilot businesses, compute credits and mentors. Code decides the deal. The voice closes it. Thank you, we're Coffee Please."

## Rules for the room
- One speaker per section; hand over at slide 4 and slide 6.
- Rehearse the demo on the live build right before the pitch.
- Don't claim numbers that aren't on the slides. Latency, if asked: median 1.9 s scripted, 3.2 s LLM turns (from 382 turns in 60 real calls). Cost per call: not measured yet.
- Slide 9 (appendix) has answers to the 8 likely judge questions.
