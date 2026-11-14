# Evaluating ACIE's signals

The question this answers: **are the signals ACIE generates per turn (intent, human state, explicit preferences, language, amount) actually correct on realistic call input, or do they only look plausible?** Everything below is measured against labeled data, and none of it is self-reported by a model.

## How to run

```bash
# per-turn signal audit (data/eval/turns.jsonl)
.venv/bin/python scripts/eval_signals.py
.venv/bin/python scripts/eval_signals.py --lang hi            # one language only
.venv/bin/python scripts/eval_signals.py --data scripts/fixtures/smoke_turns.jsonl --out /tmp/x   # smoke

# end-to-end scripted calls through ACIEPipeline (data/eval/conversations.jsonl)
.venv/bin/python scripts/eval_conversations.py
.venv/bin/python scripts/eval_conversations.py --data scripts/fixtures/smoke_conversations.jsonl --out /tmp/x
```

Outputs go to `data/eval/results/`:
- `signals_<ts>.json` has every metric overall and per language, plus threshold sweeps and confusion matrices.
- `errors_<ts>.md` lists the 15 worst errors per category, with text, gold label and prediction.
- `conversations_<ts>.json` has pass rates, every failure, and per-turn state, signals, replies and timings.

Each results file records `acie_fingerprint`, a hash of `acie/**/*.py`, so a number can be traced back to the code that produced it. Records that fail schema validation (`docs/DATA_SCHEMA.md`) are skipped and reported. They are never repaired silently.

The training code for the learned replacement is in `training/state_classifier/` (see its README). It trains on `data/train/turns.jsonl`. **`data/eval/` is held out and must never be used for training, tuning or picking checkpoints.**

## What each metric means

| Signal | What is run | Metric | Why it matters |
|---|---|---|---|
| Intent | `BankDemoDomain().classifier.classify(text, session)`. The session carries `dialog_pending` / `pending_slot` from the record, so `affirm`/`deny` gating behaves as in a live call. The headline number also applies bank_demo's slot-fill rule (`unknown` + pending amount slot + parsable number → `transfer`). Classifier-only accuracy is reported separately. | accuracy, macro-F1, per-intent P/R/F1, confusion matrix | A wrong intent means a wrong action. `affirm` executes a transfer, `deny` cancels it, and `human_agent`/`goodbye` end the call. |
| Human state | `HumanStateEstimator().estimate` on one turn from the neutral prior. This is what the policy sees on the first turn of a topic: with `alpha_up = 0.7`, the smoothed value is `0.7 × raw` for rising dims. Raw `RuleScorer` scores are reported too. | For frustration, confusion and urgency, two positive classes: label ≥ 1 (any) and label = 2 (clear). P/R/F1 at the policy's `ENTER = 0.5` threshold (the value at which the policy changes behaviour), plus a sweep from 0.20 to 0.80 for the best achievable F1. **Engagement:** positive = label 0 (disengaged), score = `1 − engagement`. This mirrors `policy._flag("disengaged", 1 − engagement)`. | The policy only reacts when a state crosses ENTER. A score that is correct in direction but never crosses ENTER changes nothing. |
| Explicit prefs | `detect_explicit_requests(text)` | exact match of the whole dict; per-key (`language`, `verbosity`, `complexity`) P/R; false-request rate on turns that contain no request | Explicit requests are *locked* into the long-term profile, so a false positive is sticky. |
| Language | `detect_language(text)` | accuracy vs the record's `lang` | Decides the reply language when nothing explicit is set. |
| Amount | `parse_amount(text)` | exact match on turns with a gold amount; false-positive rate on turns whose gold amount is null | The amount goes straight into a money movement. |
| End-to-end | `ACIEPipeline` + `TemplateBackend`, with a fresh domain and profile store per conversation | pass rate per `expect` key (intent → `plan.intent`; language, complexity, verbosity, acknowledge_emotion → `expression.*`), per language; p50/p95 of `timings_ms.total` | Shows how the per-turn errors compound across a call: FSM state, hysteresis, the language streak, ack cooldown. |

## Caveats on the data

- `data/eval/turns.jsonl` has 369 turns: en 145, hi 154, hinglish 70. It is **synthetic**: written to look like ASR output, not transcribed from real calls, and labeled by one annotator (an LLM agent). There is no inter-annotator agreement figure, so some disagreements below will be label noise.
- Sample sizes are small. The 95% interval on intent accuracy is about ±8 points for en and hi, and ±11 for hinglish. Per-intent and per-state numbers rest on 4 to 30 positives each, so treat them as directional.
- The set was written to be *hard*: ASR misspellings, spoken numbers, indirect phrasing, frustration without swear words. That matches real calls, but it is harsher than a hand-picked demo script. The v1 keyword and phrase rules were not written against this set.
- The state numbers are single-turn. In a live call the EMA accumulates over turns, so recall on sustained states is somewhat higher than shown here, and reaction on the first turn is exactly what is shown.

## Current numbers (v1 rules, acie fingerprint as of 2026-11-14)

From `data/eval/results/signals_20261114_153535.json`:

| metric | all | en | hi | hinglish |
|---|---|---|---|---|
| n turns | 369 | 145 | 154 | 70 |
| **intent accuracy** | **0.51** | 0.50 | **0.46** | 0.64 |
| intent macro-F1 | 0.53 | 0.53 | 0.46 | 0.65 |
| share of turns predicted `unknown` (gold `unknown` ≈ 19%) | 0.52 | 0.51 | 0.58 | 0.39 |
| frustration (label ≥ 1) F1 @ENTER | 0.15 | 0.15 | 0.07 | 0.31 |
| frustration (label ≥ 1) P / R @ENTER | 1.00 / 0.08 | 1.00 / 0.08 | 1.00 / 0.04 | 1.00 / 0.18 |
| frustration (label = 2) F1 @ENTER | 0.42 | 0.50 | 0.17 | 1.00 (n=2) |
| confusion (label ≥ 1) F1 @ENTER | 0.20 | 0.23 | 0.13 | 0.29 |
| confusion (label ≥ 1) P / R @ENTER | 0.70 / 0.12 | 0.75 / 0.14 | 0.67 / 0.07 | 0.67 / 0.18 |
| urgency (label ≥ 1) F1 @ENTER | **0.13** | **0.00** | 0.11 | 0.44 |
| urgency (label ≥ 1) best F1 at any threshold (raw) | 0.58 | 0.51 | 0.62 | 0.77 |
| disengaged F1 @ENTER | 0.32 | 0.39 | 0.34 | 0.13 |
| disengaged P / R @ENTER | **0.19** / 1.00 | 0.24 / 1.00 | 0.20 / 1.00 | 0.07 / 1.00 |
| explicit prefs: exact match on turns with a request (n = 32 / 11 / 12 / 9) | 0.50 | 0.36 | 0.50 | 0.67 |
| explicit prefs: false-request rate on other turns | 0.00 | 0.00 | 0.00 | 0.00 |
| language detection accuracy | 0.93 | 1.00 | 1.00 | **0.61** |
| amount exact match (n = 44 / 14 / 22 / 8) | **0.11** | 0.21 | **0.05** | 0.12 |
| amount false-positive rate | 0.012 | 0.023 | 0.000 | 0.016 |

Precision is high for frustration, urgency and explicit preferences: when a phrase rule fires, it is almost always right. Recall is the problem.

### Interpretation: what is weak and why

1. **Every normal turn is flagged "disengaged". This is a policy bug, not a scorer weakness.** The neutral engagement prior is 0.5. The policy flags disengaged when `1 − engagement ≥ ENTER (0.5)`, so a plain turn like "what is my balance" (engagement 0.5) sets `disengaged = True` and forces `verbosity = low` on every turn that is not confused or frustrated. That explains recall 1.00 with precision 0.19, and it was confirmed end-to-end through `ACIEPipeline` (`active_flags.disengaged == True`, `verbosity == "low"` on a neutral first turn). The fix belongs to whoever owns `acie/personalization/policy.py`: use a strict `>` or a separate disengagement threshold of about 0.7. Separately, minimal replies are only recognised in Roman script: `हाँ`, `जी`, `yes` and `ji` all score as normal engagement.
2. **Most urgency rules can never trigger on their own (a calibration cliff).** An urgency phrase gives raw 0.7. From the neutral prior, the smoothed value is `0.7 × 0.7 = 0.49`, just under ENTER = 0.5. So "need to send money urgently my father is in hospital", "balance please quickly i'm driving" and "सवा लाख रुपए भेजने हैं जल्दी करो" all score 0.49 and are ignored, unless fast speech or a barge-in adds to them. English urgency F1 at ENTER is therefore 0.00, while the same scores reach F1 0.51 to 0.77 at a lower threshold. Frustration and confusion phrases (raw 0.8 → 0.56) cross the threshold; urgency (0.7) does not.
3. **Intent recall collapses to `unknown`, worst for Hindi.** 52% of all turns are predicted `unknown`, against a true rate of about 19%. The top confusions are transfer → unknown (33), clarify → unknown (27), check_balance → unknown (17) and human_agent → unknown (14). Concrete misses:
   - Hindi: `पचास हज़ार ट्रांसफर करना है` (only Latin "transfer" and `भेज` are keywords; `ट्रांसफर` is missing) and `25000 रुपए मेरे बेटे के खाते में डाल दो`.
   - English: ASR spelling `transfar money to my wife's account`, paraphrase `can i move some money to my other account`.
   - human_agent phrasing: `i don't want to talk to a machine get me someone real`, `transfer me to an executive` (that one is classified as a money transfer).
   - On ties, the keyword classifier picks whichever intent comes first in the dict: `नमस्ते मेरी ट्रांसफर लिमिट कितनी है` → `greet`.
4. **Amounts in words are not parsed at all.** `parse_amount` only reads digits. 0 of 38 spoken-word amounts are parsed: `fifty thousand`, `पचास हज़ार`, `डेढ़ लाख`, `ढाई हज़ार`, `पौने दो लाख`, `chalis hajar`. Real ASR output is mostly words, so the transfer flow loops on "how much?". This also cancels out the slot-fill rule: pipeline-effective intent accuracy is only 0.512 against 0.509 without the rule. In the other direction, digits that are not amounts get parsed: `the otp is 4 8 2 9 1 6` → 4, `my account number ends with 4572` → 4572, `आखिरी के चार नंबर 0091 वाले खाते में दस हज़ार` → 91 (gold 10000). An LLM could not have done better here, but a deterministic Indian-number-words parser (en + hi + romanized hi, with fractions like सवा/डेढ़/ढाई/पौने) would.
5. **The affirm/deny gate can execute the opposite action.** `no problem go ahead`, `yeah no that's fine send it` and `कोई दिक्कत नहीं चलेगा` are all classified as `deny`. "no" / "नहीं" match first, because `deny` is checked by priority before hits are compared. On a pending transfer this cancels a confirmed request. The reverse error (deny → affirm) did not occur in this set.
6. **Frustration and confusion depend on exact phrases.** Recall at ENTER is 8% for frustration (label ≥ 1) and 12% for confusion. Indirect but clear cases score 0: `ये क्या है यार हर बार यही बोल रही हो`, `तीसरी बार बोल रहा हूं बीस हज़ार`, `okay this is going nowhere`, `मतलब`, `कौन सी लिमिट`, `wait i'm confused you said forty earlier`. The jaccard "repeat" signal only fires on near-verbatim repeats. Paraphrase is the main gap here, and the multilingual classifier in `training/state_classifier/` is aimed at it.
7. **Language: Hinglish is detected as English 39% of the time.** The marker list misses common romanized words (`hazaar`, `yaar`, `matlab`, `banda`, `tha`, `bolna`, `kaunsa`) and needs two hits, so short turns like `rehne do yaar` and `matlab` fall through to `en`. English and Devanagari Hindi score 100%.
8. **Explicit requests: high precision, half recall.** Nothing fires falsely, so profile locks stay safe. But `english please`, `इंग्लिश में बोलो`, `hindi please my english is not so good`, `just tell me the number` and every request to switch to Hinglish are missed.

### What this means for the demo

- **Safe today:** language mirroring for English and Devanagari Hindi; frustration and confusion responses *when the customer uses a stock phrase*; explicit "Hindi mein / simple / short" requests; digit amounts.
- **Not reliable yet:** spoken amounts (the biggest blocker for a Hindi demo), Hindi intent recognition beyond the exact keywords, urgency adaptation, the disengagement-driven brevity (currently always on), and Hinglish detection on short turns.
- The fixes for 1 and 2 are a few lines in the policy and the scorer weights. The fixes for 3, 4 and 7 are keyword, marker and number-word coverage. For 6, and for 3 in general, the trained classifier is the real fix. Re-run `scripts/eval_signals.py` after each change and compare the JSON. Do **not** tune against the same eval set indefinitely: once the rules are adjusted on it, these numbers become optimistic, and a fresh held-out batch is needed.

## End-to-end conversations (v1, TemplateBackend)

From `data/eval/results/conversations_20261114_153738.json`. 28 scripted calls with 142 turns and 152 `expect` checks. Calls are bucketed by the caller's **opening** language, detected from the first two turns.

| expect key | all | en (13 calls) | hi (12) | hinglish (3) |
|---|---|---|---|---|
| intent (`plan.intent`) | 87/122 **71%** | 42/54 78% | 31/52 **60%** | 14/16 88% |
| language | 11/15 73% | 8/9 89% | 3/6 50% | – |
| complexity = low after confusion | **2/7 29%** | 0/3 | 1/2 | 1/2 |
| acknowledge_emotion | 5/5 100% | 1/1 | 4/4 | – |
| verbosity | 3/3 100% | 1/1 | 2/2 | – |
| **all checks** | **108/152 71%** | 52/68 76% | 41/66 62% | 15/18 83% |
| calls with every check passing | 6/28 | | | |

- 22 of the 44 failures are the first failure in their call. The other 22 follow an earlier failure and are mostly knock-on damage. The typical chain: a Hindi spoken amount (`पंद्रह हज़ार ट्रांसफर करना है`) is not understood, so no transfer is pending, so the following `हां कर दो` and `ओके बाय` fail as well (c002).
- **Safety-relevant chain (c021):** `my debit card got blocked` is classified as `card_offer` because of the keyword " card ". That pitches the credit card and sets `pending = card_offer`. The caller then says `ok then balance please`, which is read as affirm, so the agent **starts a credit-card application the customer never asked for** (`card_apply_started`). The same bug fires in Hindi on `और मेरा एटीएम कार्ड ब्लॉक हो गया है` (c008).
- Confusion only lowers complexity when a stock phrase is used. `which limit`, `what i never set any limit`, `wait so the money went or not` and `ये limit wimit kya hai mujhe nahi pata` all leave complexity at `med`.
- Explicit language switches fail when they are phrased differently from the regex (`इंग्लिश में बोलो प्लीज़`, `sorry english please`, `thoda mix karke bolo na hindi english dono`). The implicit-switch streak then needs two more turns before it follows the caller.
- Bare greetings in Devanagari (`हेलो`, `हेलो जी`) are `unknown`, because only the Latin "hello" and `नमस्ते` are keywords. So Hindi calls often open with "Sorry, I didn't get that".
- Frustration handling works when the caller is explicit. `acknowledge_emotion` and the short reply fired on every expected turn (5/5), and the ack cooldown held on the turn after.
- Latency without an LLM is p50 0.14 ms / p95 0.20 ms per turn for the whole pipeline (understand + adapt + decide + compile + validate). The decision and interaction layers add effectively zero latency. Real end-to-end latency is set by the LLM backend and TTS, and this eval does not measure it.

## Progress log

Same held-out data (369 turns / 28 calls). The Hindi/ASR rule rewrite was done **without** access to this data. The later structural fixes were informed by the conversation failures; they are rule bugs, not lexicon tuning.

| metric (all langs) | baseline v1 | + Hindi/ASR rules | + structural fixes |
|---|---|---|---|
| intent accuracy | 0.51 | 0.70 | 0.70 |
| amount exact match | 0.11 | 0.86 | 0.86 |
| language detection | 0.93 | 0.98 | 0.98 |
| explicit prefs (turns with a request) | 0.50 | 0.75 | 0.75 |
| urgency F1 @ENTER (label ≥1) | 0.13 | 0.40 | 0.40 |
| frustration F1 @ENTER (label ≥1) | 0.15 | 0.26 | 0.26 |
| confusion F1 @ENTER (label ≥1) | 0.20 | 0.23 | 0.23 |
| conversation checks passed | 71% | — | 89% (en 90 / hi 91 / hinglish 83) |
| conversations fully passing | 6/28 | — | 15/28 |

What the structural fixes were:
- **Disengaged:** neutral engagement (0.5) used to count as disengaged, which forced short replies on every turn. It now has its own band (enter at 0.7, exit at 0.5). One "ok" no longer counts; it takes about 2–3 minimal replies in a row. Because of that, the single-turn disengaged F1 reads 0 by design.
- **Clarify:** a clarify intent now counts as confusion, so complexity drops to low.
- **New amount during a transfer:** this is re-quoted and re-confirmed. The previously pending amount is never executed.
- **Card offer:** a topic change while the offer is pending ("ok then balance please") is no longer treated as consent to apply.
- **Language:** loanword-only turns ("ओके थैंक यू") don't push toward a language switch.

**Honest read:**
- Intent, amount, language and explicit preferences are good enough for a demo in English and Hindi.
- **Frustration and confusion detection are high-precision but low-recall** (P = 1.0, R ≈ 0.12–0.18). Rules only catch stock phrases. Real callers paraphrase ("तीसरी बार बोल रहा हूं", "wait so the money went or not").
- That is the case for training the MuRIL classifier in `training/state_classifier/`. It should be trained on a separate synthetic training set generated by the teacher model on the voice box, and **never** on this eval set.
