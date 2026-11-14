# CRM backend (voice/crm)

Projects, customers/leads, campaigns, follow-ups and AI draft assist. Contract: `PLATFORM_SPEC.md`. All routes need the admin cookie
except the lead webhook (token). JSON in, JSON out. Times are epoch seconds. Errors are `{"detail": "..."}` with 400/401/404/409/422/502/503.

## Files (data/voice/)
`projects.json` (kept in sync with `tenants.json`, so `/api/tenants` works), `customers/<project>.json` ({id: customer}),
`campaigns.json`, `calltypes.json` (custom only), `crm_calls.json` (call_id -> project/customer/campaign/call_type, `applied`),
`dnc.json` (shared with the console). All writes are atomic (temp file + rename). A corrupt file is moved to `*.corrupt`.
`voice/crm/store.py` defines the `Store` protocol; `JsonStore` is the only implementation (Postgres later).

## Calls
Calls started from the CRM are saved with `project_id`, `crm_customer_id`, `campaign_id`, `call_type` (server.py `State.save` adds the
`call.crm` dict; the console's `customer_id` stays the phone number so the profile store is unchanged). The call object also gets
`call.call_type` for the CALLTYPES lane. When a call ends, `State.finish` calls `on_call_end`, which applies the outcome. Reading
`GET /api/followups` also applies any finished linked call that was missed. Applying is idempotent per call id.

## Outcome -> customer
Result source: `call.result` `{outcome, stage_suggestion, next_action, answers, score_delta}` if present, else derived from the context
ledger. Canonical outcomes: accepted, interested, callback, later, undecided, declined, opt_out, no_answer, voicemail, failed.

| outcome | stage | next_action |
|---|---|---|
| accepted (link sent) | negotiating (won is set by the operator or a result suggestion) | follow_up in 1 day: "check if it was paid" |
| interested | interested | follow_up in 2 days |
| callback (callback_later / team member promised) | contacted | callback (follow_up call type) in 1 day, `due_estimated: true` (the ledger has no time) |
| later / undecided | contacted | follow_up in 2 days |
| declined | lost | nurture in 30 days (`CRM_NURTURE_DAYS`) |
| opt_out | do_not_call, written to `dnc.json` | none |
| no_answer / voicemail / failed | unchanged | `retry` after the campaign's `retry_after_minutes`, until `max_attempts` |

Stages only move forward from a call (never back); won / lost are not changed by calls; do_not_call always applies.

## Lead score (0-100, `scoring.py`, deterministic)
stage (new 5, contacted 15, interested 40, qualified 55, demo_booked 70, negotiating 80; won 100; lost/do_not_call 0) + engagement
(+5 per connected call, max 15) + last outcome (accepted +15, interested +10, callback +8, later +3, declined -20) - 3 per unreached
among the last 3 outcomes (max 9) + recency (<=2 d +10, <=7 d +6, <=14 d +3, <=30 d 0, older -5) + call score_delta (clamped +-20).

## Campaign queue
`GET /api/campaigns/{id}/queue` is read-only and never dials. A customer is eligible when: not DNC (flag or `dnc.json`), in the audience
(stages default to all non-terminal; tags any-of; languages; min_score; explicit ids), not on a live call, and either never called in this
campaign, or a follow-up is due, or the last attempt was unreached and `retry_after_minutes` passed (max `max_attempts`).
Order: due follow-ups first, then score desc, then oldest. `dialable_now` needs status running + inside the window (campaign
`schedule.window` or the project's `call_hours`, in the project's timezone, on `schedule.days`, Monday=0) + a free slot (`max_parallel`
minus live calls) + pacing elapsed. `next` is the first `slots` rows of `eligible`.

Dial mode is `manual` by default: nothing dials except `POST /api/campaigns/{id}/call/{customer_id}` (operator). `auto` needs
`confirm_auto: true` when setting it, status running, and `CRM_AUTO_DIAL=1` in the server environment; then `POST .../tick` dials the
queue's `next`. Every dial goes through `State.dial_block` (allowed_numbers, DNC, `CALL_HOURS`) and the max-parallel cap, and the CRM also
refuses DNC customers itself (`allow_dnc` is ignored). Nothing calls `tick` automatically: schedule it from outside if wanted.

## AI assist
`POST /api/ai/assist {kind, prompt, context}`, kinds: product, calltype, campaign, customer_fields, step, reply. The LLM is the
OpenAI-compatible endpoint from `PRODUCTGEN_BASE_URL/MODEL/API_KEY` or `ACIE_EXPR_*`: temperature 0, JSON mode, no `seed`, 60 s timeout.
The reply is validated against the kind's pydantic model; on failure one repair round is sent with the errors; still invalid -> 422.
Returns `{kind, draft, repaired, saved: false}` and never saves. Keys are never logged or returned.

## Call types
The schema is `acie.calltypes.CallType` (see docs/CALL_TYPES.md): ids are `lower_snake` (`^[a-z][a-z0-9_]{0,39}$`, derived from the name),
`pack` picks the domain, steps carry `answer_type` (yes_no|choice|number|text|date_time|money), `choices`, `keywords`, and every language in
`languages` needs opening/closing/step text. Create and edit run `validate_call_type` and answer 422 with the problems. Built-ins are listed
with the project's own; built-ins cannot be edited, only duplicated. A call's result is read from `call.context.result` (`CallResult`).

## Single call
`POST /api/customers/{id}/call {product_id, call_type?, language?}`: operator-triggered call without a campaign, same dial gates (409 with the
reason when refused), linked to the customer so the outcome is applied. Calls set `call.call_type` and `call.project_id`.

## Stored for the console
Project: `stages` (own stage names), `default_calltypes`, `onboarding` (wizard state). Campaign: `tone_level` (professional|mid|casual).
CSV import: `{project, csv, mapping {field: column | "custom.<key>"}, commit, on_duplicate, tags}`; custom values are typed like hand edits.
Customer do-not-call is one-way: `consent:{dnc:true, reason}` (a stage move out of do_not_call is a 409).
