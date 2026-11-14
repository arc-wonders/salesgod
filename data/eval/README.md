# Eval data: bank phone-agent turns and conversations

Contract: `docs/DATA_SCHEMA.md`. Validate and print stats with:

```
.venv/bin/python scripts/validate_data.py
```

## How it was made

- Hand-written by one LLM author (Claude), working from general knowledge of how Indian retail-bank callers
  talk on the phone and what ASR transcripts of that speech look like. **No real call logs were used.**
- Written blind to the detector code: nothing under `acie/` or `tests/` was read, so the examples are not
  built around any keyword list. Use the set to audit the rule-based detectors.
- Text is meant to look like raw ASR: lowercase, almost no punctuation, fillers, false starts and repeats
  ("in my in my account"), ASR mistakes ("transfar", "balence", "limited", "a legible"), English bank words in
  Devanagari ("बैलेंस", "ट्रांसफर", "लिमिट", "ओके"), and spoken numbers ("डेढ़ लाख", "पौने दो लाख", "पैंतीस सौ",
  "one point five lakh", "25k").
- Hard negatives were added on purpose: "nahi" as a deny vs inside "samajh nahi aaya" (clarify) vs
  "koi problem nahi" / "nahi nahi theek hai bhej do" (affirm); "card" meaning a blocked debit card (`unknown`);
  credit-card limit vs transfer limit; "ok"/"haan"/"ठीक है" as an affirm when something is pending vs a
  low-engagement backchannel (`unknown`) when nothing is; numbers that aren't amounts (OTP, last-4 digits,
  dates, times, numbers the agent said that the caller repeats back). Each ambiguous label has its reasoning in `notes`.
- Timing in `turns.jsonl` comes from a seeded generator that loosely tracks the labels (slower when disengaged
  or confused, faster when urgent or angry, with noise). Barge-in is set by hand. Treat timing as plausible,
  not measured.
- Conversation `expect` fields are set only where the behaviour is clear-cut (plan intent from the c1/c2 facts;
  a language switch on the turn where the caller asks for one; `complexity: low` on the response to a clear
  confusion turn; `acknowledge_emotion: true` + `verbosity: low` on the response to a clear frustration turn,
  and `false` on the turn after it). Clarify turns, and turns whose plan intent depends on implementation order
  (for example c2 asking for more than both balance and limit), have no `intent` expectation.

## Distribution (output of `scripts/validate_data.py`)

**turns.jsonl: 369 turns**
- lang: hi 154 (41.7%), en 145 (39.3%), hinglish 70 (19.0%)
- intent: unknown 69, transfer 60, clarify 42, check_balance 39, affirm 29, deny 26, transfer_limit 24,
  card_offer 24, human_agent 20, greet 18, goodbye 18
- frustration 0/1/2 = 308/42/19 · confusion 308/37/24 · urgency 325/17/27 · engagement 0/1/2 = 50/278/41
- explicit preferences: 32 turns (8.7%): language=hi 8, en 6, hinglish 3; complexity=low 7; verbosity=low 6, high 3
- amount non-null: 44; turns with an agent prompt in context: 220; dialog_pending transfer 62, card_offer 43
- timing present on 328 turns; barge_in 31 (9.5%); latency median 960 ms (300–2400)

**conversations.jsonl: 28 calls** (c1: 18, c2: 10; 4–7 turns, 142 turns in total)
- By hand: 11 open in Hindi (10 stay Hindi, 1 switches to English), 11 open in English (10 stay English, 1 switches
  to Hindi), 6 Hinglish/mixed (3 with an explicit switch)
- expect keys: intent 122, language 15, complexity 7, acknowledge_emotion 5, verbosity 3
- Boundary cases covered: c1 asking for exactly ₹25,000 (assumed allowed), c1 ₹30,000/₹40,000 (under the
  ₹40k daily cap but over the ₹25k left), c2 ₹20,000 (within limit, over balance)

## Known gaps and biases

- **Synthetic, single author.** Phrasing reflects one model's idea of real speech. The roughness (fillers,
  ASR errors) is applied by hand and is more regular than real ASR. Real transcripts also have far more
  garbage tokens, words cut off at the start of the turn, and agent speech leaking into the transcript.
- **Hindi is Delhi/UP-style Hindustani.** No Bhojpuri, Marathi, Tamil-accented English, Bengali-Hindi, etc.
  The "rural" and "elderly" personas show mostly in word choice ("राम राम जी", "बेटा", "पेंशन"), not in dialect.
- **Hindi ASR script is idealised.** Real Hindi ASR mixes digits and words more chaotically and sometimes outputs
  Romanised words inside Devanagari (or the reverse). There are only a few such cases here.
- **`unknown` is large (69)** because it holds both out-of-domain requests and pure backchannels/complaints
  ("हम्म", "ok", "ये क्या है यार"), plus pure language-preference turns with no task (the schema has no
  `preference` or `backchannel` intent). Score those subsets separately if it matters.
- **Acceptance of the card offer** ("sign me up", "हां बनवा दो कार्ड") is labelled `card_offer`, not `affirm`.
  This is arguable and noted per row.
- **Label calibration** for 1 vs 2 is subjective, and frustration=2 is rarer than it is in real escalated calls.
  Labels are per-turn and do not use earlier turns except through `context`.
- Explicit language requests are mostly short commands. Implicit switching (the caller just starts answering
  in Hindi) is labelled with an empty `explicit`, which is correct by the schema, but there are few such cases.
- The amount set leans towards round numbers; there are few odd amounts (19,900, 10,500, 3,500).
