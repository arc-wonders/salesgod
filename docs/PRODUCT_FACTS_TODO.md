# Product facts: what was authored, what the owner confirms

Both products live in `data/voice/products.json`. A fact that is not in a product's `faqs`, `knowledge` or `benefits` is
never invented: the agent says "I don't have that detail" and points to `info_source` (see `acie/decision/honest.py`).

## fitzone-gym (operator demo facts; `description` field says so)
FitZone is the operator's own demo gym, not a real company. Everything below was AUTHORED as demo data and must be confirmed
or replaced by the operator before any real use:

| Fact | Authored value |
|---|---|
| Plans | monthly Rs 2,200; quarterly (3 months) Rs 6,000; annual Rs 21,000 (about Rs 1,750 a month); launch offer Rs 2,999 for 3 months if you join this week |
| Payment methods | UPI, debit/credit card, net banking, cash at the front desk |
| Refund | within 3 days of joining and gym unused: refund minus Rs 500 admin charge; otherwise none (freeze instead) |
| Cancellation | any time at the front desk; paid fees follow the refund rule |
| Activation | same day: after payment walk in with the SMS and a photo ID |
| Branches | Saket (Delhi), Sector 29 (Gurugram), Sector 62 (Noida); none in Sector 49 Gurugram |
| Timings | every day 5 am to 11 pm |
| Women's batch | every branch, 10 am-12 noon and 4-6 pm, woman trainer |
| Trial pass | one-day free pass by walking in with an ID (cannot be sent by SMS) |
| Facilities | parking free at all branches; lockers and showers included; all branches AC; no steam or sauna |
| Trainers | 1 free personal session a week; extra sessions Rs 600 each |
| Documents / age | one government photo ID; age 16+ (under 18 needs parent consent); no upper limit; doctor first for health conditions |
| Discounts | none for students/friends/groups; referral: both get one extra free PT session |
| Freeze | up to 15 days |

Owner to confirm: every row above; also whether "the SMS shows the nearest branch" (older FAQ text) is true: the SMS template
(`sms_text`) only carries the payment link.

## claude-pro (real company name: public facts only)
Kept: Pro is $20/month (about $17/month billed yearly, about $200 a year), subscribe on claude.ai or in the iPhone/Android apps,
cancel any time in settings, a free plan exists, Pro has higher usage than free. Everything else stays the honest
"I don't have that detail; it will be shown on the payment page or the official website".

Owner to confirm / not verified by us:
- The **$8 for the first 3 months (60% off)** intro offer: not a public Anthropic price; it is the operator's demo offer.
- Refund terms (the agent says "refunds follow the website policy; I don't have the exact terms").
- UPI / local payment methods in India (agent only promises a card and "the payment page shows the methods").
- Context window size, message limits (agent says it does not have the number).
- Activation time after payment, offer end date, the rupee conversion (`price_note`: about Rs 700 then Rs 1,700, at an exchange rate).
- The longer `knowledge` text (privacy wording, Team plan, GST help for shops) was written earlier by the operator; re-check it.

## Gaps handled by the generic "unknown" lines (any product)
refund policy, payment methods, monthly plan, activation time, branch list, women's batch, trial pass, parking, facilities (AC, steam),
offer end date, context window, friend/group discount, who can pay. Add an FAQ with `answer`, `answer_hi`, `answer_hinglish` for any of them to
give a real answer; until then the agent says it does not have it.
