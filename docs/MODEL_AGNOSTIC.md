# Model-agnostic Layer 4

Layer 4 (decision, psyche, playbook) reads ONE object, `acie.signals.TurnSignals`. It never sees a provider's own
object (`Understanding`, a classifier, an HTTP reply) and never parses `dialog["_nlu"]`. (`_nlu` is still written for
the console and logs; `dialog["_signals"]` holds the real contract.)

```
turn -> providers (rules, llm, model, http, fake) -> PartialSignals each -> fuse() -> TurnSignals -> decision / psyche / playbook
```

## TurnSignals
| field | meaning |
|---|---|
| `intent`, `intent_scores`, `confidence` | top intent (domain vocabulary), distribution, confidence of the provider that supplied it |
| `objection`, `question_topic` | trust/price/need/timing/language/other/none ; product/offer_terms/general/conversation/none |
| `buying_interest` (alias `buying_signal`) | 0..1 readiness |
| `garbled`, `garbled_reason`, `gist` | bad line / unclear speech flag, one-line meaning |
| `state`, `learned_state` | the 14 dims fused with the rules / what learned providers read alone |
| `sources` | field -> provider that produced it |
| `votes`, `learned`, `heard(*labels)` | each learned provider's own mapped intent; `heard` = some learned provider said it |

## Providers (`acie/providers/`, env `ACIE_SIGNAL_PROVIDERS=rules,llm`)
| name | what | config |
|---|---|---|
| `rules` | domain keyword classifier + lexicon scorer (always first) | none |
| `llm` | any OpenAI-compatible model via `acie/understanding.py` | `ACIE_LLM_BASE_URL`, `ACIE_LLM_MODEL`, `ACIE_NLU_TIMEOUT_S` |
| `model` | trained classifier from the loader registry (XGBoost JSON dir, MuRIL dir, your loader) | `ACIE_TURN_MODEL`, `ACIE_TURN_MODEL_THRESHOLD` |
| `http` | your service returning TurnSignals JSON | `ACIE_SIGNAL_HTTP_URL`, `_TOKEN`, `_TIMEOUT_S` |
| `fake` | scripted, tests only | code |

Unset = today: `rules`, plus `llm` when the pipeline has an NLU. Late or failing providers are skipped (rules-only turn).
Explicit list = the rules are the plain rules and models come in only as providers (no hybrid in the domain).

## Fusion (`acie/signals_fusion.py`, one config dict, no randomness)
| what | rule |
|---|---|
| intent | rules are the base. `model` replaces any intent the rules do not protect (clarify, human_agent, callback) at >= 0.6 and respects affirm/deny gates. Others (`llm`, `http`) only fill unknown / tell_more / benefits / goodbye / affirm / deny |
| objection, topic, buying, garbled, gist | first provider in `field_priority` (llm, http, model, fake) that has a value |
| state dims | rules value, then each learned provider blended 0.5/0.5 in provider order |

## Labels
The domain declares `intent_guide`, keyword intents and `intent_aliases` (plus `ACIE_INTENT_ALIASES` JSON). A provider's
label is mapped through the alias table; anything unknown becomes `unknown` and is ignored.

## Plug in a new model
1. HTTP: serve `POST` -> `{"intent": "...", "confidence": 0.9, "buying_signal": 0.7, "state": {...}}`, set `ACIE_SIGNAL_HTTP_URL`, add `http` to the list.
2. Local file: `register_loader(name, match(path), load(path))` in `acie/providers/loaders.py`; the object needs `classify_with_confidence(text, session)` and optionally `score(turn, session)`.
3. New kind: subclass `ProviderBase` (`name`, `budget_s`, `provide`) and `register_provider(name, factory)`.
