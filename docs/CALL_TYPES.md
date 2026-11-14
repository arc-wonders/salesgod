# Call types

A call type is a template of one kind of call: what it is for, how it opens, the questions it asks, what it may do, how it ends.
The ACIE decision layer runs it as a deterministic step machine on top of the same layers as the sales call (understanding signals,
call ledger, needs/emotion table with the crisis row, compliance). The LLM only phrases answers to questions; everything that decides
what to do (which question next, what was answered, the score, the outcome) is plain code and gives the same output for the same input.

## Files

| file | what |
|---|---|
| `acie/calltypes/schema.py` | `CallType`, `Step`, `Choice`, `CallResult`, `NextAction`, `validate_call_type`. The CRM / API lane imports these. |
| `acie/calltypes/builtins.py` | the six built-ins as data |
| `acie/calltypes/registry.py` | `CallTypeRegistry(data_dir)`: built-ins + `data/voice/calltypes.json`, validated; bad entries are reported and skipped |
| `acie/calltypes/extract.py` | answers from the caller's words: yes/no, number, money, date/time, choice, text (en / hi / hinglish) |
| `acie/calltypes/memory.py` | previous-call memory for follow-ups, read from `data/voice/calls/*.json` |
| `acie/calltypes/factory.py` | `build_domain(call_type_id, product, ...)` picks the pack; unknown or broken id = today's sales domain |
| `acie/decision/domains/custom_steps.py` | `StepCallDomain`: the generic step machine (custom, payment_reminder, feedback_survey, appointment_booking) |
| `acie/decision/domains/lead_qualification.py` | `LeadQualificationDomain` (score-based outcome) |
| `acie/decision/domains/follow_up.py` | `FollowUpDomain` (continuity, open questions first, agreed price) |

`sales_pitch` is today's `ProductSalesDomain`, unchanged.

## Built-in types

| id | pack | selling | steps | outcomes |
|---|---|---|---|---|
| `sales_pitch` | sales_pitch | yes (negotiation within the profit floor) | today's flow | link_sent, callback_requested, declined |
| `lead_qualification` | lead_qualification | no | need, budget, authority, timeline, book_demo (skipped when score < 40) | demo_booked, qualified, interested, callback_booked, not_qualified |
| `follow_up` | follow_up | yes, no negotiation | decision (go ahead / more time / not interested / already paid), reschedule, confirm_link | link_sent, ready_to_buy, callback_booked, paid, declined |
| `payment_reminder` | steps | no | status (paid / will pay / dispute / cannot pay), pay_date | paid, promised_to_pay, callback_requested, human_requested |
| `feedback_survey` | steps | no | rating, recommend, comment (optional) | feedback_given, completed |
| `appointment_booking` | steps | no | purpose, slot | appointment_booked, callback_booked, declined |

## CallType fields

`id, name, objective, pack, languages, opening{en,hi,hinglish}, steps[], allowed_actions[], success_outcomes[], closing{}, closing_declined{},
tone, max_call_s, selling_allowed, negotiation_allowed, compliance_lines[{en,hi,hinglish}], project_id, builtin`.

Any text field accepts a plain string (the same line in every language). Placeholders: `{agent} {seller} {product} {name}`.
The opening should say who is calling and why and end with the permission question ("is this a good time for two minutes?");
the first reply is read as the answer to it. **The AI disclosure and the recording notice are always added if the operator's opening
lacks them.** `compliance_lines` are spoken right after the opening.

### Step

`id, text{en,hi,hinglish}, answer_type (yes_no|choice|number|text|date_time|money), choices[], required, skip_if` plus optional extras:

| field | meaning |
|---|---|
| `keywords` | topic words; one sentence can answer this step early ("mera budget 10 hazaar hai" fills `budget`) |
| `choices[]` | `{id, label, keywords, outcome, ends_call}`; a keyword written `~word` is weak (used only if no strong keyword matched) |
| `skip_if` | `answered:x`, `unanswered:x`, or `<step id or score> <op> <value>` (`authority == no`, `budget < 5000`, `score < 40`) |
| `score`, `score_answered` | deterministic points: value -> points (`{"yes": 20, "no": 5}`, choice ids), plus points for any real answer |
| `action`, `outcome` | what a real answer earns: the next action the CRM schedules (`book_demo`, `book_callback`, `book_appointment`, `take_payment_promise`, `send_link`, `transfer_human`) and the call outcome |

A step is asked at most twice (an optional one once), never again once answered, "I don't know" is an answer. Questions the caller asks are
answered first (approved FAQ, knowledge base or the locked price line word for word; else the LLM from the product facts), then the same
question is asked again.

## Result

Rebuilt after every turn into `session.dialog["_result"]` (JSON), `session.context.result` (so it lands in the call record's `context.result`)
and `TurnResult.result`. `complete` is false while the call goes on (a hang-up leaves a partial result with outcome `ended_early`).

```json
{"outcome": "demo_booked", "stage_suggestion": "demo_booked",
 "next_action": {"type": "book_demo", "due_at": "2026-11-16T18:00:00+05:30", "call_type": ""},
 "answers": {"need": "CRM for sales", "budget": 10000, "authority": true, "timeline": "this_month", "book_demo": "2026-11-16T18:00:00+05:30"},
 "score_delta": 90, "call_type": "lead_qualification", "facts": {"book_demo_at": "..."}, "complete": true}
```

Stage by outcome: demo_booked/appointment_booked -> demo_booked, qualified -> qualified, interested -> interested, callback/not_qualified/completed -> contacted,
link_sent/ready_to_buy/promised_to_pay -> negotiating, paid -> won, declined -> lost, opt_out -> do_not_call (`STAGE_FOR_OUTCOME`).
`next_action.type`: `follow_up_call` (call_type `follow_up`; for a callback, or 2 days after a link was sent), `book_demo`, `book_appointment`,
`payment_followup` (call_type `payment_reminder`), `human_call`. Times are ISO 8601 in Asia/Kolkata, resolved against the call's clock.

## Hooks

* `voice/server.py make_pipeline`: `getattr(call, "call_type", "")`; empty or `sales_pitch` = today's flow, otherwise `acie.calltypes.factory.build_domain` with the project's
  `calltypes.json`, the customer id / phone (follow-up memory) and the call id. (The CRM lane should set `call.call_type` and `call.project_id`; the `/api/sim` start can set it the same way.)
* `acie/context.py CallContext`: additive `call_type`, `result`. `acie/schemas.py TurnResult`: additive `result`. `acie/pipeline.py`: passes it through.

## Rules the packs keep

Compliance comes first on every turn and is word for word: opt-out (number removed, call ends, outcome `opt_out`), "where did you get my number",
AI disclosure, never take OTP / card digits (a budget said in digits is not mistaken for a card number). The crisis row (pipeline) wins before any domain.
A firm no / stop / goodbye is a polite close, never a pitch. A busy caller gets a callback (time read from their words, or asked once).
Not applied here (the sales pack's own pieces): the product's pre-call eligibility / constraints and the negotiation flow.
Calling hours, allowed numbers and the DNC list stay at the dial gate.

## Operator types (`data/voice/calltypes.json`)

A list of CallType objects (or `{"calltypes": [...]}` or `{id: {...}}`). Validation: unknown actions or languages, a missing language in the opening / steps / closing,
a step-driven type with no steps, duplicate step ids, choices on a non-choice step, an action not in `allowed_actions`, an unreadable `skip_if`, an id that shadows a built-in
(give it a `project_id`). Invalid entries are reported by `load_custom` / `CallTypeRegistry.rejected` and ignored at call time.
