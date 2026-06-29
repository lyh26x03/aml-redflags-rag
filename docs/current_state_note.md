# Current State Note — Multi-Turn Memory Redesign

Branch: `memory-router-state-fix`  
Date: 2026-06-29  
Test status: **156 passed, 0 failed, 2 harmless deprecation warnings**

---

## What was wrong

`record_retrieval_turn()` did `active_scenario_summary = current_user_query` on every turn.
A short follow-up like "What about cross-border transfers?" (7 words) would overwrite the
full case description ("Funds show rapid movement through a virtual asset exchange.").
Turn 2 still worked because it composed the *old* summary before overwriting it.
From Turn 3 onward the retrieval query was a chain of bare follow-up fragments; the
original case context had silently vanished.

This is **not** a retrieval or LLM bug — it is a state-representation bug.
One field was being asked to hold three different things: the stable case, the current
query, and the next-turn retrieval context.

---

## Multi-turn memory design (backbone + deltas)

The active case is split into three roles:

| Role | Field | Update rule |
|---|---|---|
| stable case backbone | `active_scenario_summary` | set once (SEED), only changed by REPLACE |
| bounded follow-up refinements | `active_case_deltas` (≤ 6) | appended by PRESERVE |
| raw per-turn query (audit only) | `recent_turns[].user_query` | always appended, never used for retrieval |

`rag_core/memory/scenario_policy.py` — pure, deterministic, zero-LLM — chooses exactly
one action per retrieval turn:

| Action | Condition | Effect |
|---|---|---|
| `seed` | first substantive case; no backbone yet | adopt query as backbone, set `case_seed_text` |
| `preserve` | `retrieve_with_memory` route (always); or additive cue ("also", "另外…") | append a distilled delta; backbone unchanged |
| `replace` | plain `retrieve` with disjoint AML topics, substantive query (≥ 6 words or ≥ 2 topics), Jaccard similarity ≤ 0.20, no additive cue | replace backbone, clear deltas, **start fresh evidence scope** |
| `repair` | backbone detected as degraded (fragment cue, < 12 chars, or < 34 % term retention) | restore backbone from `case_seed_text` |
| `noop` | empty query | no change |

`retrieve_with_memory` composes the retrieval query as:
`backbone + deltas (joined) + current_query`

BM25/dense/RRF math is completely unchanged.

---

## Evidence contamination policy (explicit)

This is the invariant that must hold across all scenarios:

**Flags, citations, and retrieved chunk IDs from Case A must never appear in
the retrieval context or response for Case B.**

How each route enforces this:

### refuse / out-of-scope
`_handle_refuse` reads memory (to populate debug fields) but calls **no** `record_*`
or `_update_scenario` method. `memory_updated = False` in every debug response.
The active scenario, flags, citations, and all other fields are completely untouched.
An out-of-scope request is a no-op to memory.

### replace (new standalone case)
When `_update_scenario` fires action `REPLACE`, it clears:
- `active_flags` → `[]`
- `active_citations` → `[]`
- `retrieved_chunk_ids` → `[]`
- `active_entities_or_context_terms` → `[]`
- `active_case_deltas` → `[]`

The current turn then immediately repopulates them with Case B's own evidence.
Result: zero Case A evidence is accessible when Case B's retrieval happens.

### preserve (follow-up to the same case)
Backbone unchanged. Evidence fields accumulate. No clearing. Correct — the
follow-up adds to the same investigation.

### DELETE /sessions/{id}/memory (user-initiated full reset)
Wipes the entire session object. Clean slate for a new conversation.

---

## How a 4-turn conversation flows (example)

```
Turn 1  query="Funds show rapid movement through a virtual asset exchange."
        route=retrieve  →  action=seed
        backbone: "Funds show rapid movement through a virtual asset exchange."
        deltas: []

Turn 2  query="What about profile mismatch?"
        route=retrieve_with_memory  →  action=preserve
        retrieval_query: "Funds show rapid movement...  profile mismatch"
        backbone: unchanged
        deltas: ["profile mismatch"]

Turn 3  query="What about cross-border transfers?"
        route=retrieve_with_memory  →  action=preserve
        retrieval_query: "Funds show rapid movement...  profile mismatch  cross-border transfers"
        backbone: unchanged
        deltas: ["profile mismatch", "cross-border transfers"]

Turn 4  query="What are the combined red flags for this case?"
        route=retrieve_with_memory  (CASE_SUMMARY_QUESTION regex routes this here)
        → action=preserve
        retrieval_query: "Funds show rapid movement...  profile mismatch  cross-border transfers  combined red flags"
        backbone: unchanged
        scenario_health.drift: false, term_retention: 1.0
```

Before the redesign, Turn 3's retrieval query was:
`"What about profile mismatch?  What about cross-border transfers?"`
The original case was gone.

---

## Debug visibility

Every response includes in `debug`:

| Field | Meaning |
|---|---|
| `intent_route` | fine-grained route (`retrieve` / `retrieve_with_memory` / `answer_from_history` / `ask_clarifying_question` / `refuse`) |
| `route_family` | collapsed 3-outcome view (`retrieve` / `refuse` / `no_retrieval_response`) |
| `route_reason` | human-readable reason string |
| `memory_used` | whether the retrieval query was composed with memory context |
| `memory_updated` | whether memory state changed this turn |
| `scenario_update_action` | `seed` / `preserve` / `replace` / `repair` / `noop` |
| `case_delta_count` | number of accumulated deltas |

Memory snapshot (`GET /sessions/{id}/memory`) adds:
- `active_case_deltas` — the bounded follow-up refinements
- `case_seed_text` — the original first-turn case text
- `scenario_origin_turn` — which turn set the current backbone
- `last_scenario_action` — the last policy action
- `scenario_health` — `{drift, severity, reasons, term_retention}` — deterministic drift report

---

## Test coverage

| File | Tests | What it covers |
|---|---|---|
| `test_active_scenario_summary_overwrite.py` | 3 | Original regressions: memory state, pipeline query, API snapshot after 3 turns |
| `test_memory_scenario_policy.py` | 24 | Policy units, distillation, drift detector, state multi-turn, router, 4-turn pipeline, all 6 requirement scenarios (T1–T6) |
| `test_memory_hardening.py` | 8 | Drift corruption + repair, replace-then-followup, 50-turn bounded growth, delta dedup, Chinese connectors, conservative no-drop safety bias |
| `test_memory.py` | (pre-existing) | Existing memory + routing contract tests |
| `test_api_contract.py` | (pre-existing) | Single-turn API contract; unchanged |

**Total: 156 passed, 0 failed.**

The six requirement scenarios are explicitly covered by `test_api_*` tests in `test_memory_scenario_policy.py`:
- T1: follow-up does not overwrite original case
- T2: Turn-3 retrieval query preserves original case terms
- T3: Turn-4 combined-risk question retrieves with full accumulated context
- T4: new standalone case replaces backbone and resets evidence scope
- T5: history recall (`answer_from_history`) does not change the scenario
- T6: refusal does not pollute memory

---

## Known limitations

1. **Scenario summaries are truncated query text**, not model-generated abstractions.
   A case stated in 1,000 words is truncated to 500 chars. The backbone may miss
   low-priority terms, which the drift detector might not catch.

2. **New-case detection is rule-based (Jaccard ≤ 0.20 + topic count + word count).**
   It is conservative: a follow-up with a genuinely new case phrased as a question
   ("What about structuring? Our client also set up shell companies.") will be
   PRESERVEd, not REPLACEd. The `new_case_scorer` seam lets a future learned
   classifier override this without changing callers.

3. **In-process, non-persistent.** Memory is lost on restart and not shared across
   workers. Not suitable for production AML systems.

4. **Delta distillation strips only known leading-connector phrases** ("What about",
   "那跟", etc.). A follow-up phrased without a connector is stored verbatim as
   the delta.

5. **Delta list bounded at 6.** After 6 follow-ups the oldest deltas are dropped.
   In a very long conversation the retrieval query may lose early refinements —
   but never the backbone.

6. **`retrieve_with_memory` route ALWAYS PRESERVEs.** Even if a follow-up
   introduces disjoint AML topics, the backbone is not replaced. This is a
   deliberate conservative safety bias: a follow-up phrased as a question can
   never silently replace the case. If the user explicitly starts a new case,
   they should start a new turn without a follow-up connector.

---

## Interview explanation

> The multi-turn failure was not in BM25, dense retrieval, or LLM generation.
> It was a **state-representation bug**: a single field in the structured memory
> object was assigned three different jobs — stable case identifier, current
> query, and next-turn retrieval context — with no explicit policy for when to
> overwrite vs. accumulate. I fixed it by separating these three roles into
> distinct fields and making the update logic an explicit deterministic policy
> (`seed / preserve / replace / repair / noop`) in a dedicated module with no
> LLM dependency. Evidence contamination across cases is prevented by clearing
> flags, citations, and chunk IDs on `replace`, and by making `refuse` a
> guaranteed memory no-op. The design is tested at four layers: policy unit
> tests, state-level multi-turn, pipeline retrieval query verification, and
> API snapshot tests that reproduce all six requirement scenarios.
