# Project Context: SalesGod (Adaptive Voice AI Engine)

## 1. Overall objective

We are designing a voice AI agent, specifically its **intelligence and response-generation engine**.

Out of scope: Speech-to-Text (STT), Text-to-Speech (TTS), SIP trunking / telephony, audio streaming infrastructure.

The focus is everything between receiving a transcript and producing the final response.

Fundamental idea: **the LLM handles natural-language realization, paraphrasing and conversational speech generation — it does not determine what the system responds with.** Response determination happens through deterministic systems, specialized ML models, business logic and other fast algorithms.

## 2. Core architecture

1. **STT Input** — incoming transcript stream
2. **Human State Estimator** — satisfaction · frustration · confusion · engagement
3. **Personalization Engine** — language · tone · vocabulary · complexity · information preferences
4. **Response Determination Engine** — deterministic algorithms · ML · business rules · APIs
5. **Structured Prompt Compiler** — combines response specification and communication parameters
6. **LLM Expression Engine** — paraphrasing · NLG · conversational continuity
7. **Response Validation → TTS**

## 3. The three major intelligence components

### A. Human State Estimator

| Dimension     | Parameters                                 |
| ------------- | ------------------------------------------ |
| Satisfaction  | Liking, disliking, satisfaction            |
| Emotion       | Frustration, anxiety, excitement, calmness |
| Engagement    | Attention, participation, receptiveness    |
| Comprehension | Confusion, cognitive load                  |
| Interaction   | Urgency, trust, certainty                  |

These are observable conversational estimates, not direct measurements of neurological states.

### B. Personalization Engine

**Long-term customer preference vector:** historical language preferences, regional language preferences, preferred tone and formality, technical depth, vocabulary complexity, preferred explanation style, information preferences, likes and dislikes, past interactions.

**Short-term conversational adaptation vector:** current verbosity, current tone, reassurance requirements, language adjustments, explanation complexity, proactivity, information density.

The system adapts the second vector using the first, the current human state and feedback:

$$A_t = \pi(H_t, P_t, C_t)$$

- $H_t$: current human interaction state
- $P_t$: historical customer preferences
- $C_t$: current conversation context
- $A_t$: adaptation parameters for the next response

### C. Response Determination Engine

Determines *what* to communicate. Candidate implementations: finite-state machines, decision trees, rule engines, intent classifiers, ML models, database queries, business APIs, constraint-based systems.

Output is a structured response specification, not prose.

## 4. Structured Prompt Compiler

Bridge between the decision engine and the LLM. Inputs:

1. Response specification — what to communicate
2. Expression parameters — how to communicate it

```json
{
  "response": {
    "objective": "Explain transfer limit",
    "facts": ["Daily limit is ₹40,000", "Remaining limit is ₹25,000"],
    "decision": "₹50,000 transfer cannot proceed"
  },
  "expression": {
    "language": "Hindi",
    "regional_variant": "Delhi Hindi",
    "tone": "calm",
    "formality": 0.5,
    "verbosity": 0.1,
    "complexity": "simple",
    "proactivity": "low"
  },
  "constraints": {
    "preserve_facts": true,
    "no_invented_information": true,
    "do_not_change_decision": true
  }
}
```

The LLM acts as a linguistic realization engine: it may change wording, structure, transitions and naturalness, but not the underlying decision. A deterministic validator checks output before TTS.

## 5. Runtime architecture

Event-driven and parallelized:

- Stream A: intent extraction and human-state estimation
- Stream B: customer profile and memory retrieval
- Context merger
- Response engine
- Prompt compiler
- LLM
- Validator
- Feedback engine (updates customer state and memory)

| Component             | Technology                  |
| --------------------- | --------------------------- |
| Orchestration         | Python asyncio              |
| Session state         | Redis                       |
| Customer database     | PostgreSQL                  |
| Semantic memory       | pgvector                    |
| Event processing      | NATS                        |
| Intent classification | Small transformer           |
| State estimation      | Lightweight ML classifier   |
| Decision engine       | Custom rules/state machines |
| Schema validation     | Pydantic                    |
| LLM                   | Qwen / GPT / Claude         |

## 6. Architectural principles

1. The LLM is not the source of business truth.
2. Response determination and linguistic generation are independent.
3. Long-term preferences and temporary emotional states stay separate.
4. Preferences are learned with confidence scores and explicit overrides.
5. Personalization is multidimensional, not a single satisfaction score.
6. Adaptation must not oscillate between tones or languages every turn.
7. Customer feedback updates memory and improves future interactions.
8. Critical business facts are validated before speech generation.
9. Personalization optimizes comprehension and satisfaction, not manipulation.

## 7. Roadmap

| Phase | Deliverable                                    |
| ----- | ---------------------------------------------- |
| 1     | Deterministic response engine                  |
| 2     | Customer profile and personalization vectors   |
| 3     | Human state estimation and adaptation policy   |
| 4     | Structured prompt compiler and LLM integration |
| 5     | Response validation                            |
| 6     | Feedback and long-term memory                  |
| 7     | Continuous personalization improvements        |

## 8. Conceptual direction

- **Decision Intelligence:** what should the system communicate?
- **Interaction Intelligence:** how should it communicate with this human at this moment?

The LLM primarily handles the second. The potentially novel component is the adaptive communication policy — learning which linguistic and interaction parameters work best for different customers and conversational states.

Working name: **Adaptive Conversational Intelligence Engine (ACIE)**.

## Next steps

Design the Human State Estimator and Personalization Engine in depth:

1. Complete state-vector schema
2. Learned vs. explicitly configured parameters
3. Which ML models estimate each parameter
4. How historical preferences are stored and updated
5. How the adaptation policy maps state changes to communication changes
6. How to evaluate whether personalization improves conversations
