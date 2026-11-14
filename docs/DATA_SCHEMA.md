# Data schema (shared contract for data, eval and training)

Demo languages: **English and Hindi** are primary. Hinglish (Roman-script Hindi-English mix) is secondary but common in real calls.

All text simulates **raw ASR (speech-to-text) output** as it arrives from a real call. It is not written prose:
- often lowercase, little or no punctuation
- fillers: "umm", "haan", "matlab", "you know", "acha"
- spoken numbers: "fifty thousand", "pachas hazaar", "पचास हज़ार", "ek lakh", "25k"
- typical ASR errors: dropped words, homophones, "transfer" heard as "transfar", Hindi words in Roman script inside Devanagari output and vice versa

## data/eval/turns.jsonl — one customer turn per line

```json
{
  "id": "t0001",
  "lang": "en | hi | hinglish",
  "text": "mera paanch hazaar transfer karna hai jaldi",
  "context": {
    "prev_customer_text": "string or null",
    "prev_agent_text": "string or null",
    "dialog_pending": "null | transfer | card_offer",
    "pending_slot": "null | amount"
  },
  "timing": {"barge_in": false, "response_latency_ms": 900, "speech_rate_wps": 2.6},
  "intent": "greet | check_balance | transfer | transfer_limit | card_offer | human_agent | affirm | deny | clarify | goodbye | unknown",
  "labels": {"frustration": 0, "confusion": 0, "urgency": 2, "engagement": 1},
  "explicit": {"language": "hi"},
  "amount": 5000,
  "notes": "why this example is hard / what it tests"
}
```

- `labels` use 0 = absent, 1 = mild, 2 = clear. **Engagement:** 0 = disengaged (minimal or one-word replies, slow), 1 = normal, 2 = highly engaged (asking questions, giving detail).
- `explicit` holds only preferences the customer states directly. Allowed keys: `language` (en/hi/hinglish), `verbosity` (low/high), `complexity` (low).
- `amount` is the rupee amount the customer means, as an integer, or null.
- `timing` fields are optional. When present, they must be plausible: real barge-in rate is about 5–15% of turns, and response latency is usually 300–2500 ms.

## data/eval/conversations.jsonl — scripted multi-turn calls for end-to-end checks

```json
{
  "id": "c001",
  "customer_id": "c1",
  "description": "Hindi caller asks over-limit transfer, gets confused, accepts remaining",
  "turns": [
    {"text": "...", "timing": {}, "expect": {"intent": "transfer_over_limit", "language": "hi",
                                              "complexity": "low", "acknowledge_emotion": false}}
  ]
}
```

`expect` keys are optional. Only list the ones that turn is meant to test. Plan intents come from `acie/decision/domains/bank_demo.py`.

Demo customers:
- `c1`: Rahul. Balance ₹82,500, daily limit ₹40,000, ₹15,000 used today (₹25,000 left), card-eligible with a ₹1,50,000 limit.
- `c2`: Priya. Balance ₹12,000, daily limit ₹25,000, nothing used today, not card-eligible.
