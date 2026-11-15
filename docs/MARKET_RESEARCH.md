# SalesGod: Market and Competitor Research (hackathon pitch)

Team: Coffee Please. Product: SalesGod, an AI voice agent that makes outbound sales calls in India (Hindi, Hinglish and English phone calls over Plivo).
Research date: 2026-11-15. [V] = checked against the cited page today. [VENDOR] = the vendor's own claim or marketing. [UNVERIFIED] = not confirmed in a primary source, or our own estimate.
Internal sources: team/research/REPORT.md (Sarvam, Bolna, Vaani, Vapi), team/human/PLAN.md, team/INTERFACE.md, team/studio/API.md, team/voices/RUST_NOTES.md.

## 0. What we actually built (claim only this)
- **Decisions are deterministic.** Each turn runs policy gates, then a playbook decision tree, then a negotiation state machine. The LLM (Gemini 3.5 Flash, or a tuned Qwen3-8B on vLLM) only writes the wording of the reply. A validator drops any number that has not been confirmed and any empty slot, so the agent never invents a price.
- **Haggling engine.** It has a hard floor that is never revealed, concessions that only move in one direction, and an agreed price that stays fixed once the caller accepts (deal.agreed_price; the agent never offers more than the agreed price and never reopens it). An absurd lowball gets a joke instead of a counter.
- **Durable call context on NATS JetStream.** The deal, the sale checklist (product explained, price quoted, offer terms read, objections, counter-offers, consent, link) and the last line are restored if the call is interrupted. "Repeat that" and "where were we" are routed so they can never trigger a concession.
- **"Heard-only" context.** When the caller interrupts (barge-in), only the part of the agent's line the caller actually heard counts as said.
- **Long-term caller memory.** It stores the caller's language mix, style, objections and past deals.
- **Style policy chosen per turn** (13 rows, first match wins). Styles include held-back, curious, warm, playful, reassuring and brief. There is a question budget (questions in at most 50% of turns), mirroring of at most 4 words, backchannels, and humour limits (at most 2 per call, none when the caller is in distress, elderly or angry). The agent always uses aap, never तू.
- **Named agents with a gender and per-language voices.** Hindi verbs are gendered to match the agent.
- **Pronunciation layer.** It is verified by a round trip (TTS output transcribed back with Whisper); spoken ₹ amounts in Hindi were fixed this way.
- **Crisis and distress detection** stops the sales pitch.
- **Product studio.**
  - AI drafts product facts and records a source for each.
  - Critical facts (price, offer, guarantees, floor) must be confirmed by a human before the product can be published.
  - It recommends playbook situations that are missing.
  - AI-simulated customers grade each node of the decision tree (pass, fail or not reached per persona), with "fix with AI" on failed nodes.
  - Products are versioned with a publish gate.
- **Live console** shows the decision path for each turn and the live haggle.
- **Multi-tenant, multi-product.** Demo products: FitZone gym, Claude Pro demo, and concert tickets generated from one line.
- **Stack:** Whisper large-v3 (STT), Fish Audio (TTS), vLLM Qwen, Gemini, FastAPI, NATS, Plivo, a GCP A100.
- **Limits to state honestly:** 3 languages (Hindi, Hinglish, English) against Sarvam's 11 Indian languages plus English. This is a hackathon prototype with no production call volume or conversion data yet. Self-hosted cost per minute has not been measured.

## 1. Market
### 1.1 Market size
| Market | Size | Growth | Source |
|---|---|---|---|
| Global voice AI agents | ~$2.54B (2025) to ~$35.2B (2033) | 39.0% CAGR | Grand View Research via https://www.gii.tw/report/grvi2018348-ai-voice-agents-market-size-share-trends-analysis.html |
| Global voice AI agents (alt.) | $3.40B (2025) to $43.8B (2036) | 24.0% CAGR | https://www.meticulousresearch.com/product/voice-ai-agents-market-6938 |
| Global voice AI agents (alt.) | +$10.96B 2024-29 | 37.2% CAGR | https://www.technavio.com/report/voice-ai-agents-market-industry-analysis |
| India conversational AI | $565.8M (2025) to $2.33B (2030) | 26.6% CAGR | https://www.marketsandmarkets.com/Market-Reports/geography/conversational-ai-market/India |
| India conversational AI (alt.) | INR 38.1B (2024) to INR 152.3B (2030) | ~26.2% CAGR | https://www.marketresearch.com/Netscribes-India-Pvt-Ltd-v3676/Conversational-AI-India-42513034 |
| India voice AI | $153M (2024) to $958M (2030) | 35.7% CAGR | https://caller.digital/blog/india-voice-ai-conversational-ai-market-size-segments-2026 [VENDOR, analyst not named] |
| India BPO sector | ~$40B; voice slice ~$12-15B | - | https://news.outsourceaccelerator.com/voice-ai-dismantling-india/ ; caller.digital [UNVERIFIED] |
| Contact-centre labour savings | $80B by 2026 (global); AI-handled interactions 1.6% to 10% | - | Gartner 2022 https://www.gartner.com/en/newsroom/press-releases/2022-08-31-gartner-predicts-conversational-ai-will-reduce-contac |

Caveat: forecasts differ widely (one aggregator counts 53 forecasts ranging from 8% to 39% CAGR: https://select.keenable.ai/r/voice-ai-market-growth-forecasts/OOaxIvbEPDuayhy6a3Aiyg). Quote ranges.

### 1.2 India telesales economics
- **Human call centre:** about ₹7-8 per minute, excluding hiring, training and attrition. Bolna self-serve is about ₹5.5/min, and large enterprise contracts get ₹1.75-2/min. Inc42, 15 Jul 2026: https://inc42.com/features/the-unit-economics-of-indias-voice-ai-boom/
- **Telecaller pay:** Teleperformance agents take home ₹19.2-20.9k per month; BPO+ telecallers ₹13.4-14.9k (AmbitionBox, self-reported): https://ambitionbox.com/salaries/teleperformance-salaries/telecaller ; https://ambitionbox.com/salaries/bpo-plus-salaries/telecaller
- **Effective cost of voice AI:** list prices are ₹2-12/min but effective cost is ₹6-25/min. Example: ₹41 per resolved contact against a ₹95 human baseline. https://caller.digital/blog/voice-ai-vendor-pricing-teardown-india-2026 [VENDOR, illustrative]
- **Human agent cost in USD:** $10-15/hour fully loaded (Outsource Accelerator, May 2026).
- **Cost trend:** voice AI cost per minute has fallen about 60-70% since early 2023 (caller.digital [VENDOR, illustrative]).

### 1.3 Pain points in outbound sales
- **Low connect rate:** 46-48% average in India across 5 crore calls, so about 53% go unanswered. The best slot (5 PM on weekdays) connects 61.8%. A connected call averages 1.6 minutes. Most leads need 12-13 touchpoints. https://callyzer.co/blog/the-ultimate-list-of-cold-calling-statistics-in-india/ [VENDOR dataset]
- **Connect rate by hour:** 22.4% (08:00-09:30) to 43.8% (11:00-13:00): https://caller.digital/blog/connect-rate-benchmark-outbound-calling-india-2026 [VENDOR]
- **Low conversion:** about 2% for cold calls, and often more than 18 dials to reach a prospect (https://squadstack.com/blog/cold-calling-services [VENDOR, unsourced]). B2B cold-call-to-meeting rate is 1-3% (LiveHelpIndia [VENDOR]). No verified India-specific benchmark for insurance or loan conversion.
- **Attrition:** recruiters and vendors call it high (https://taggd.in/blogs/call-center-hiring-india/), but no figure is given [UNVERIFIED]. BPO wage inflation runs 8-10% a year (caller.digital [UNVERIFIED]).
- **Compliance:**
  - From 12 Feb 2025, TRAI bans 10-digit numbers for telemarketing. Promotional calls must come from the 140 series on DLT, service calls from 1600, and Do Not Disturb (DND) preferences apply. https://cms.trai.gov.in/sites/default/files/2025-02/PR_No.11of2025.pdf
  - In 2026 TRAI and Truecaller disputed whether these number series can be tagged or blocked by apps: https://indiantelevision.com/regulators/trai-defends-140-1600-call-rules-amid-spam-row-with-truecaller/
  - India's data protection law (the DPDP Act) has rules notified 14 Nov 2025. Consent and data-fiduciary obligations take effect 18 months later (about May 2027). https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf ; https://www.india-briefing.com/news/dpdp-rules-2025-india-data-protection-law-compliance-40769.html
  - Spam is concentrated in financial services (51%) and real estate (29%) (Statista, via Callyzer).
- **Language diversity:** 57% of internet users prefer Indic-language content, and the user base was 886M in 2024 (IAMAI-Kantar 2024: https://www.iamai.in/sites/default/files/research/Kantar_%20IAMAI%20report_2024_.pdf). Reports conflict on whether the 57% is urban or rural. Code-mixing is normal (Haptik [VENDOR]: https://www.haptik.ai/blog/state-of-voice-ai-adoption-india).
- **Trust and liability when an LLM makes things up:**
  - Moffatt v. Air Canada, 2024 BCCRT 149: the airline was held liable for wrong fare advice from its chatbot. https://www.theregister.com/2024/02/15/air_canada_chatbot_fine/
  - A Chevy dealership ChatGPT bot agreed to sell a 2024 Tahoe for $1 (Dec 2023). https://incidentdatabase.ai/es/reports/3673/
  - FTC v. Air AI (Aug 2025; settled Mar 2026 with an $18M judgment, largely suspended): https://search.ftc.gov/news-events/news/press-releases/2025/08/ftc-sues-stop-air-ai-using-deceptive-claims-about-business-growth-earnings-potential-refund

### 1.4 Adoption trends, 2025-2026
- **Funding:**
  - Bolna: $6.3M seed led by General Catalyst (Jan 2026): https://yourstory.com/ai-story/voice-ai-startup-bolna-raises-63m-led-by-general-catalyst
  - Ringg: Series A closed at $15.5M, including $10M from Peak XV: https://inc42.com/?p=545709
  - Vaani: self-serve launch on 2 Nov 2026 at ₹5/min: https://cxotoday.com/media-coverage/vaani-ai-research-launches-self-serve-platform-to-build-and-deploy-voice-ai-agents/
  - Gnani: Inya VoiceOS speech-to-speech model: https://www.digit.in/news/general/india-ai-impact-summit-2026-gnaniai-unveils-indias-first-5b-voice-to-voice-ai-model-inya-voiceos.html
- **Price convergence:** Sarvam ₹3.50/min (https://docs.sarvam.ai/voice-agents/pricing.md), Bolna 6.0¢ (₹5.52)/min (https://www.bolna.ai/pricing), Vaani ₹5/min, Retell $0.07-0.31, Vapi $0.05 plus providers, Bland $0.12-0.14, ElevenLabs $0.08 plus LLM. Voice infrastructure is commoditising; the sales layer is where the value is.
- **Vertical mix of India voice AI (2024, illustrative):** BFSI ~40%, D2C/e-commerce ~18%, healthcare ~12%, insurance ~10% (caller.digital [VENDOR]).
- **Platforms moving from graphs to LLM routing:**
  - Vapi retired its deterministic Workflows graph on 18 Aug 2026 in favour of Squads, where routing is done by the LLM and extraction is "best-effort" (https://docs.vapi.ai/workflows/legacy-migration).
  - Retell's docs say its conversation flows "do not make every part deterministic" (https://docs.retellai.com/build/choose-agent-type.md).
  - This leaves room for a deterministic sales core.

## 2. Competitors
| Vendor | Positioning | Price per min (public) | Languages | Build style | Gaps for SALES |
|---|---|---|---|---|---|
| Sarvam Voice Agents (formerly Samvaad) | Indian full stack (Bulbul TTS), Genie builder, campaigns, DND, A/B | ₹3.50 PAYG; number ₹59/30 days; simulated test ₹2 [V] | 11 Indian + English, code-mix, switches language mid-call [V] | Block prompt + variables + multi-state | Genie "asks rather than inventing prices"; KB source tags; AI-judge tests per scenario. Negotiation, heard-only context, live decision path, gendered verbs: not documented |
| Bolna | India orchestration; Auto Build; sales templates (credit card, loans, COD, cart) | 6.0¢ (₹5.52) Pilots [V]; enterprise ₹1.75-2 (Inc42) | Indian (count not documented); gender picks the voice | Form-generated prompt; graph agents and workflows | Post-call Debug view only; negotiation and number validation not documented |
| Vaani AI Research | Indic speech; self-serve since Nov 2026; Vibe-Code | ₹5 all-in | Indic + code-mix; pronunciation dictionary | Prompt + 5-stage pipeline (incl. Memory) | "Write exact numbers" KB tip; post-call evals; deterministic pricing not documented |
| Gnani.ai (Inya) | Enterprise BFSI; own models; claims >80% gross margin | Quote only | 12+ to 15+ Indic (sources conflict) | No-code agents | Not documented |
| Skit.ai | BFSI collections outbound since 2017 | Not public | Indian | Collections dialler | Sales and negotiation not documented |
| Haptik (Jio) / Yellow.ai | Omnichannel, chat-first, adding voice | Not public | Haptik "100+ dialects" [VENDOR] | Bot platform | Mainly inbound; outbound sales depth not documented |
| Exotel / Ozonetel | CPaaS + contact centre voicebots | Not public | Exotel 15+ Indian [VENDOR] | Bot builder | Not documented |
| Smallest.ai Atoms | Low-latency TTS + agents | $0.09-0.21 (https://smallest.ai/pricing) | Hindi among others | Prompt | Not documented |
| Vapi | Developer API, bring your own models | $0.05 + pass-through (~$0.08-0.13 all in) [V] https://vapi.ai/pricing | Via providers | Prompt + Squads (LLM handoffs); Workflows retired Aug 2026 | Extraction "best-effort"; nothing India-specific |
| Retell | Developer/business; Conversation Flow | $0.07-0.31 ($0.055 infrastructure) [V] https://www.retellai.com/pricing | Not on pricing page | Nodes; equation transitions deterministic, the rest LLM | Community: "agent enters nodes out of order, causing hallucinations" https://community.retellai.com/t/agent-enters-nodes-out-of-order-causing-hallucinations-and-flow-failure/825 ; AI QA post-call $0.10/min |
| Bland | Enterprise US outbound; Pathways | $0.14 Start; $0.12 + $299/mo Build [V] https://www.bland.ai/pricing | Not listed | Pathways graph, LLM picks edges by label https://docs.bland.ai/tutorials/pathways | Hindi and negotiation not documented |
| Synthflow | No-code; enterprise annual only | Not public [V] | "30+ countries" | Not documented | Not documented |
| ElevenLabs Agents | Top TTS + agents; Workflow Builder | $0.08/min extra + LLM [V] https://elevenlabs.io/pricing/agents | Multilingual; Hindi voices | Prompt + workflows | Not documented |
| PolyAI | Enterprise inbound customer service; $86M raised (review) | Not public | 45 (Gartner) | Managed service | Not for SMBs or outbound sales (https://assembled.com/page/polyai) |
| Air.ai | "AI replaces staff" marketing | n/a | n/a | n/a | FTC lawsuit Aug 2025, settled Mar 2026: a warning about overclaiming |

**Common sales gaps** (from public docs; absence in the docs does not prove absence in the product):
1. Price and negotiation are left to the prompt. No vendor documents a floor, monotonic concessions or a locked agreed price.
2. Numbers come from the LLM drawing on a KB. No vendor documents a filter that drops unconfirmed numbers after generation.
3. Barge-in handling is exposed as "interruption sensitivity"; none documents counting only what the caller heard.
4. Explainability is post-call only (Sarvam Log Analyser, Bolna Debug, Retell QA).
5. No vendor documents gendered Hindi verbs tied to the persona, or pronunciation checked by a round trip.

## 3. Differentiation matrix
Legend: ✅ documented, ◐ partial or similar, — not documented in public docs.

| Capability | SalesGod | Sarvam | Bolna | Vaani | Retell | Bland | Vapi |
|---|---|---|---|---|---|---|---|
| Deterministic price and negotiation (floor, monotonic concessions, locked agreed price) | ✅ | — | — | — | ◐ equation transitions | ◐ pathways, LLM-routed | — (Squads LLM-routed) |
| No invented facts (validator + human-confirmed critical facts) | ✅ | ◐ Genie won't invent prices; source tags | — | ◐ KB tip | — | — | — |
| Heard-only barge-in context + durable mid-call state | ✅ | — | — | ◐ Memory stage | — | — | — |
| Long-term caller memory | ✅ | — | — | ◐ | — | — | — |
| Live decision path + haggle console | ✅ | ◐ post-call | ◐ post-call | ◐ post-call evals | ◐ post-call QA | — | — |
| Indian languages + code-mix | ◐ 3 | ✅ 11 + English | ✅ | ✅ | — | — | — |
| Gendered Hindi verbs + round-trip pronunciation | ✅ | ◐ pronunciation | ◐ gender picks voice | ◐ pronunciation dictionary | — | — | — |
| Per-turn human-ness policy | ✅ | ◐ speaking-style block | — | ◐ backchannel | ◐ backchannel | — | — |
| Crisis detection stops selling | ✅ | — | — | ◐ escalation eval | — | — | — |
| AI customers grade each tree node + fix with AI | ✅ | ◐ AI-judge per scenario | — | ◐ evals | ◐ QA | — | — |
| Product studio from one line, facts + sources + publish gate | ✅ | ◐ Genie | ◐ Auto Build | ◐ Vibe-Code | — | — | — |
| Production scale, campaigns, DND, A/B | — prototype | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## 4. Pitch material
### Problem
- About half of outbound sales calls in India go unanswered (46-48% connect rate across 5 crore calls, Callyzer), and cold calls convert at about 2% (SquadStack [VENDOR]). A human costs ₹7-8 per connected minute before hiring and attrition (Inc42).
- 57% of India's 886M internet users prefer Indic content (IAMAI-Kantar 2024), and real calls are Hinglish.
- LLM bots can't be trusted with money: a chatbot "sold" a $1 Tahoe, and Air Canada was held liable for its chatbot's wrong fare advice.

### Why now
- Voice AI costs have collapsed: ₹3.50-5.52/min (Sarvam, Bolna, Vaani) against ₹7-8/min for a human. Open models run on one A100.
- Hinglish STT and TTS are now good enough for phone calls, and capital is flowing in (Bolna $6.3M, Ringg $15.5M).
- Regulation demands accountability: TRAI 140/1600 series (Feb 2025); DPDP Rules notified Nov 2025, consent obligations due ~May 2027.
- Platforms are moving to LLM routing (Vapi retired Workflows in Aug 2026), which leaves room for a deterministic sales core.

### Unique insight
"Let code decide the deal and let the LLM only say it": the price, the concession and the next step come from a deterministic playbook and negotiation engine, so SalesGod sounds human in Hinglish but can never make up a number or a discount.

### Target customers
- **D2C:** cart recovery, COD-to-prepaid, upsell (~18% of spend [VENDOR]).
- **Gyms and fitness:** renewal and trial-to-paid, where haggling is normal (FitZone demo).
- **Edtech:** lead qualification and counsellor-style closing.
- **BFSI lead generation:** cards, loans, insurance (~40% of spend [VENDOR]); needs explainability and human handover.
- **Events and ticketing:** urgency-driven selling (concert demo generated from one line).

### Business model (proposal) [UNVERIFIED]
- **Usage fee:** ₹4-5 per connected minute, the same band as competitors and below the human ₹7-8.
- **Outcome fee:** a flat fee per closed deal or 2-5% of agreed value. This works because the agreed price is a logged, structured field (deal.agreed_price).
- **Studio plan:** a monthly fee per tenant for products, playbooks, simulated-customer testing and the live console.
- Self-hosted cost per minute is not measured yet.

### Three demo moments
1. **Live haggle (FitZone).** "₹500 de do" → joke. "₹2,800?" → one controlled concession, never below the floor. "Theek hai" → price locked. "Where were we?" → "We agreed on ₹2,800." The console shows each node and the deal state live.
2. **Interrupt and recover.** Interrupt mid-pitch; the agent continues from the words actually heard, with no repeated price. Ask for an unlisted price: the validator refuses to invent one.
3. **One line to a product.** Type "concert tickets for X". The studio drafts facts with sources, and price stays locked until a human confirms. Simulated customers grade every node, "fix with AI" repairs a failed node, publish, call. A female agent uses gendered Hindi verbs and pronounces ₹ amounts correctly.

### Risks and answers judges may ask
| Question | Answer |
|---|---|
| "Just Sarvam or Bolna with a prompt?" | They are infrastructure plus a prompt builder. Our layer is the deterministic sales brain (negotiation engine, number validator, heard-only context, live decision path), none of which they document. |
| "Deterministic means robotic?" | The wording comes from the LLM, and a per-turn style policy controls mirroring, backchannels, humour and question budget. Our agent spoke 78% of the words vs Gong's best 43-57%, and we fixed it. |
| "Only 3 languages?" | Hindi, Hinglish and English cover the core market first. Grammar and pronunciation are per language, and the TTS/STT layer is swappable (e.g. Bulbul). |
| "Compliance?" | 140-series/DLT numbers via the telephony provider, a DND scrub, consent in the checklist, and crisis detection that stops selling. Calling windows and DND are on the roadmap. |
| "What if the LLM still says a wrong number?" | The validator drops unconfirmed numbers before TTS, and critical facts can't be published without a human. |
| "Latency?" | One A100 runs Whisper + vLLM + Fish. Measure and show the real turn latency [UNVERIFIED]. |
| "Unit economics?" | Market price ₹3.50-5.52/min vs ₹7-8/min human; our cost per minute still needs measuring; an outcome fee aligns incentives. |
| "Why will people pick up?" | Connect rate is a dialling problem (5 PM = 61.8%). We improve conversion per connected minute; best-time scheduling is on the roadmap. |
| "AI-washing (Air.ai)?" | We claim only measured behaviour, show the decision path live, and keep humans confirming critical facts. |

## 5. Could not verify
- India BPO attrition percentage.
- India insurance or loan telesales conversion benchmarks.
- Gnani, Skit, Haptik, Yellow, Exotel and Ozonetel per-minute prices.
- Retell, Bland and Synthflow language lists.
- PolyAI's 75-language claim; Smallest.ai funding (sources conflict).
- Whether the IAMAI 57% figure is urban or rural.
- Whether capabilities marked "not documented" exist privately in competitor products.
