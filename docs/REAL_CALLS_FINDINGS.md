# Real-call findings (sanitised)

Source: the 38 most recent calls on the live voice server, JSON transcripts only (no audio), read-only.
28 calls had a real conversation (169 caller turns, 207 agent turns); the rest were
unanswered, busy, voicemail or an immediate hang-up. 23 of 38 calls ended by the caller hanging up.
No numbers, names or recordings are kept in git; quotes below are shortened and generic.

## What went wrong (counts over the 38 calls)

| failure | instances |
|---|---|
| agent reply almost a copy (SequenceMatcher > 0.6) of an earlier agent reply in the same call | 37 replies in 15 calls |
| same selling / answering intent said again | 57 |
| same question asked again after the caller had answered | 46 |
| caller's question not answered (reply was "didn't catch that", a re-offer, the price line, ...) | 22 |
| offer pushed right after a no, or a second re-offer | 10 |
| reply in the wrong language (caller asked for the other one) | 16 |
| agent had to ask again / speech-to-text disagreed with a careful second pass | 31 |
| caller used frustrated words ("you keep repeating", "is it fake", swearing) | 15 |

Typical patterns: a garbled turn ("Hello", noise, a word repeated) got the same "sorry, say that again?" or the same pitch five
times; "is it fake / why so cheap?" got the same sales sentence several times; a price question about something already
described (a trainer session) got the membership price four times; "can you reduce it a little?" was read as a no and the call
ended, then the caller said they wanted to join; a caller who could not follow Hindi got the price line in Hindi again; a remark
about feeling low got the offer pitched after a sympathy phrase (the safety check was added to the code afterwards).
After a caller said "no" the agent re-offered; after the link had gone out it said the closing line twice or more.

## Root causes, ranked (file:line in the code at the time of this work)

1. No memory of what was already said. Every plan is chosen from the current turn and the FSM state only; the only guards compare
   with the previous reply (acie/pipeline.py:353-358, ratio > 0.7) or the previous lead (product_sales.py:1117). Four replies back
   is invisible. Most repeats (37) come from here.
2. The catch-all conversation fallback is a single fixed sentence (product_sales.py:1093, "Sorry, I didn't quite get that") and
   the "ask the price / the link" stage line is chosen every turn (product_sales.py:1140 `_stage_base`), so a run of garbled
   speech-to-text turns produces the identical reply each time.
3. Price lines are chosen from the words alone, not from what the caller is asking about (product_sales.py:962 `pure_price`, the
   `_PRICE_ASK` branch in `_converse`): "trainer कितने का" matches the price pattern, so the membership price is repeated.
4. "No" is detected from a word, not from meaning: a negotiation with "नहीं" in it is a decline (product_sales.py:892 `_no`,
   `_HAGGLE`), so the call ends and the sale that was still open is lost.
5. Emotion handling is a special case, not a table: distress is its own code path (acie/pipeline.py, acie/safety.py) and every other
   feeling (angry, anxious, confused, overwhelmed, suspicious, busy) is only a threshold on one dimension (frustration) in a prompt line.
6. Language: the reply language follows the last explicit request only for some plans; the price lines are chosen by the earlier
   language setting (policy), so a caller asking for English can still get the Hindi price template.

## What was built (all deterministic; ACIE_PSYCHE=1 unless noted)

* acie/psyche/needs.py: the caller's need (be_heard, clarity, answer, space, reassurance, autonomy, information, connection, closure)
  from a fixed decision table over the 14 state dimensions and word signals; optional XGBoost refinement (fixed params, hashed crc32 features,
  only among the soft needs, >= 0.70 sure; the table wins on autonomy/closure/crisis).
* acie/psyche/moves.py + ledger.py: need -> move, content from the product facts, a per-call SaidLedger of every sentence/slot/ask/move;
  chosen text that was already said is replaced by the next unsaid alternative, then an honest "that is all I have", then an offer to stop,
  then a polite close, then silence. Profit-free: only `information` can sell, once per call, after interest, never after a no.
* acie/psyche/emotion_policy.py: one table of 11 rows (crisis, angry, frustrated, anxious, sad, confused, overwhelmed, suspicious, busy, excited, calm)
  with fixed priority and thresholds; crisis is the first row and keeps the reviewed helpline wording and sticky behaviour. The crisis row is ON by
  default; the others need ACIE_PSYCHE=1.
* Always-on bug fixes: a deterministic reply that is a near copy of an earlier whole reply is replaced (ACIE_REPEAT_GUARD=0 turns it off, for
  measuring); a price question naming something the offer includes is answered from the facts (included, no separate price); a price negotiation
  is answered "the price is fixed" and the call continues; the newer live-server edits (language switch, narrower renewal pattern, call-reason line,
  no superlatives) were ported.
* Expression LLM temperature is 0 with ACIE_PSYCHE=1.

## Replay before / after (scripts/real_calls/replay.py: caller turns replayed through the pipeline, template backend, no network; every mode run twice and asserted identical)

| | agent replies that repeat an earlier reply in the call |
|---|---|
| as logged on the live server (older code during the day) | 37 (in 15 calls) |
| today's base code replayed, repeat guard off | 56 (in 16 calls) |
| with the repeat guard (always on now) | 0 |
| with ACIE_PSYCHE=1 (needs, moves, ledger, emotion rows) | 0 |

The need classifier: 5-fold cross-validation on the labelled real turns 0.83 vs the table alone
0.69 against the labels (the labels start from the table and are hand-corrected in
15 turns, so this is only a rough guide with 169 real turns).
Caveat: the replay feeds the recorded caller words, which were themselves answers to the older replies; it measures repetition and determinism, not
sales or satisfaction.
