# Structured Conversation Memory

Status: **implemented in the service** (mock-mode default, no API key required).

This document describes the intent routing and structured conversation memory
feature added to the FastAPI service. It is a **local, in-process, bounded**
demo memory — not a production memory store.

## What it is (and is not)

- It **is** *structured* conversation state: a bounded, deduplicated object that
  preserves the useful parts of an AML conversation (active scenario, active red
  flags, previous citations, retrieved chunk IDs, prior assessment, a
  referenceable prior-answer summary, unresolved clarification needs, and a
  bounded list of recent turn summaries).
- It **is not** an unlimited raw transcript. Every list is bounded and every
  free-text field is truncated.
- It **is not** a production memory store. State lives in the process, is never
  persisted, and is lost on restart.
- It does **not** introduce LangChain, LlamaIndex, Redis, SQL, a vector
  database, background workers, or any external memory service.
- It does **not** change retrieval math. Memory only *composes a richer query
  string* for follow-up retrieval; BM25/dense/RRF scoring is untouched.

## Backward compatibility

Existing single-turn clients are unaffected:

- `POST /query` without `session_id` / `use_memory` behaves exactly as before.
- The top-level `QueryResponse` keys are unchanged. All new routing/memory
  fields are additive and live inside `debug`.
- Intent routing's memory-only routes (`ask_clarifying_question`,
  `answer_from_history`, `retrieve_with_memory`, and the router's own non-AML
  refusal) only activate when memory is enabled, so single-turn behavior is
  preserved.

Memory is enabled for a request only when **all** of these hold:
`use_memory == true`, a non-empty `session_id` is supplied, and `memory_mode`
is not `"off"`.

## Request fields (additive)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `session_id` | string? | `null` | Conversation key for memory |
| `use_memory` | bool | `false` | Opt in to structured memory |
| `memory_mode` | `"off"`\|`"structured"`? | `null` (→ structured) | Force-disable with `"off"` |
| `reset_memory` | bool? | `null` | Clear this session before processing |

## Intent routes

Routing is deterministic and rule-based (no live-LLM dependency).

### Three high-level outcomes (reviewer-facing)

Every turn resolves to exactly one of three outcomes, surfaced as
`debug.route_family`. This is the simple story for a demo or a reviewer:

| `route_family` | Meaning |
|---|---|
| `retrieve` | Evidence retrieval happened (single-turn or memory-aware follow-up). |
| `refuse` | The request was out of scope; **memory is not touched**. |
| `no_retrieval_response` | Answered deterministically from conversation state — either recalling prior analysis or asking the user to clarify — with no new retrieval. |

### Five fine-grained routes (debug / tests)

Internally the router keeps five deterministic routes (surfaced as
`debug.intent_route`) so multi-turn behavior stays specific and testable. Each
maps onto exactly one outcome above:

| `intent_route` | `route_family` | When | Behavior |
|---|---|---|---|
| `retrieve` | `retrieve` | Normal AML query (incl. a follow-up that introduces a new signal) | Evidence retrieval; updates memory |
| `retrieve_with_memory` | `retrieve` | Follow-up connector (e.g. "那跟…有關嗎") with memory available | Composes prior scenario + new question, retrieves, updates memory |
| `answer_from_history` | `no_retrieval_response` | Recall/explain prior answer/flags/citations (e.g. "剛剛", "上一題", "those flags", "剛剛引用的是哪些來源") | Answers from structured memory; no new retrieval |
| `ask_clarifying_question` | `no_retrieval_response` | Vague/under-specified first-turn input, no AML topic (e.g. "這樣有沒有問題？") | Asks the user for the missing detail; stores an unresolved clarification need |
| `refuse` | `refuse` | Gate out-of-scope (sanctions/TBML/tax) **or** clearly non-AML request (e.g. "幫我推薦晚餐") | Structured refusal; **does not touch memory** |

Two routes the old notebook conflated under a single "clarification" label are
deliberately kept distinct here:

- `ask_clarifying_question` — the *user's input* is too vague to assess; the
  system asks **the user** for more detail and records the unresolved need.
- `answer_from_history` — the user asks the *system* to explain or recall a
  prior answer; the system restates what is already in memory.

`retrieve_with_memory` is the route for a follow-up that needs the previous
scenario context *and* fresh retrieval (as opposed to `answer_from_history`,
which restates what is already known).

## Memory schema

Per `session_id`, `ConversationMemory` holds:

- `session_id`, `turn_count`, `created_at`, `updated_at`
- `active_scenario_summary` — the **stable case backbone** (truncated); set once
  (SEED) and only changed when a new standalone case is introduced (REPLACE)
- `active_case_deltas` — bounded, ordered **follow-up refinements** of the case
- `case_seed_text`, `scenario_origin_turn` — the seed case text and the turn it
  was set on (audit + drift reference)
- `last_scenario_action` — the scenario-policy action taken on the last
  retrieval turn (`seed`/`preserve`/`replace`/`repair`/`noop`)
- `active_entities_or_context_terms` — detected AML topic terms (bounded, deduped)
- `active_flags` — deduplicated red flags (`code`/`name`/`name_zh`)
- `active_citations` — bounded list of `{chunk_id, source, excerpt}` (excerpt truncated)
- `retrieved_chunk_ids` — bounded, deduped, most-recent-kept
- `last_assessment`, `last_answer_summary`
- `unresolved_questions` — bounded clarification needs
- `recent_turns` — bounded list of concise per-turn summaries

The memory snapshot also reports a derived `scenario_health` diagnostic (see
[Scenario-state update policy](#scenario-state-update-policy)).

### Bounds

| Bound | Value |
|---|---|
| `recent_turns` | 8 |
| `active_case_deltas` | 6 |
| `active_citations` | 10 |
| `active_flags` | 12 (deduped by code) |
| `retrieved_chunk_ids` | 30 |
| `active_entities_or_context_terms` | 20 |
| `unresolved_questions` | 10 |
| citation excerpt | 200 chars |
| case delta | 200 chars |
| scenario / answer / query summaries | 500 / 400 / 300 chars |
| retained sessions (per process) | 256 (LRU-evicted) |

## Memory behavior by route

- `retrieve` — runs the scenario policy (SEED the first case, PRESERVE a
  refinement of the same case, or REPLACE with a new standalone case), then
  updates flags, citations, chunk IDs, last assessment/answer, and appends a
  recent turn.
- `retrieve_with_memory` — the retrieval query is composed from the case
  backbone **plus accumulated deltas plus** the new question, and the policy
  always PRESERVEs (a follow-up never overwrites the backbone). `memory_used = true`.
- `answer_from_history` — answers from memory (or returns a clear no-context
  message when nothing is stored); appends a recent turn; does not change the
  active scenario.
- `ask_clarifying_question` — stores the unresolved clarification need and
  appends a recent turn; no flags/citations are added.
- `refuse` — returns a structured refusal and **does not modify memory at all**,
  so out-of-scope requests cannot pollute the active AML scenario state.

## Scenario-state update policy

The active AML case is split into three roles so a short follow-up can never
erase the case being analyzed (the `active_scenario_summary` overwrite defect;
see [`active_scenario_summary_overwrite_test_report.md`](active_scenario_summary_overwrite_test_report.md)):

| Role | Field | Meaning |
|---|---|---|
| case backbone | `active_scenario_summary` | stable summary of the active case |
| case deltas | `active_case_deltas` | bounded follow-up refinements |
| raw turn query | `recent_turns[].user_query` | per-turn input, for audit only |

`rag_core/memory/scenario_policy.py` decides exactly one action on the backbone
per retrieval turn (surfaced as `debug.scenario_update_action`):

| Action | When | Effect on backbone |
|---|---|---|
| `seed` | first substantive case (no backbone yet) | adopt this query as the backbone |
| `preserve` | a follow-up (`retrieve_with_memory`) or an additive/related plain retrieve | keep backbone, append a bounded delta |
| `replace` | a plain retrieve whose topics are disjoint from the case (a new standalone case) | replace backbone, clear deltas, start a fresh evidence scope |
| `repair` | a degraded backbone is detected and recovered from the seed | restore backbone from `case_seed_text` |
| `noop` | nothing actionable (e.g. empty query) | unchanged |

The policy is deterministic (no live LLM). The new-case judgement is factored
behind a pluggable `new_case_scorer` seam, with the rule-based scorer as the
always-present fallback, so a learned scorer could replace it without changing
callers. Retrieval for `retrieve_with_memory` composes
`backbone + deltas + current_query`; the BM25/dense/RRF math is unchanged.

A whole-case question with memory present ("What are the combined red flags?",
"綜合來看…") is routed to `retrieve_with_memory` so it retrieves *with* the
accumulated case context instead of as a context-free query.

### Drift detection and recovery (memory failure detector)

`detect_scenario_drift` is a deterministic token-overlap diagnostic (a
lightweight stand-in for a representation-space probe) that flags a backbone
which has become empty, too short, a follow-up fragment, or has lost the
original case terms. It is exposed as `scenario_health` in the memory snapshot.
`ConversationMemory.repair_scenario()` uses it to restore a corrupted backbone
from the recorded seed — defensive, since the update policy already prevents
drift in normal operation.

## Debug fields (inside `debug`)

`intent_route`, `route_family`, `route_reason`, `memory_used`,
`memory_available`, `memory_updated`, `memory_turn_count`, `session_id`,
`referenced_previous_answer`, `referenced_previous_evidence`, `active_flags`,
`active_citation_count`, `scenario_update_action`, `case_delta_count`.

`scenario_update_action` reports the scenario-policy action taken this turn
(`seed`/`preserve`/`replace`/...) and `case_delta_count` the number of
accumulated follow-up deltas, so the case-state evolution is auditable per turn.

`route_family` is the high-level (three-outcome) view of `intent_route`; it is
present for every request (including single-turn) and is derived
deterministically, so a reviewer can read the outcome without learning the five
internal labels.

## Inspection endpoints (demo/debug only, no auth)

- `GET /sessions/{session_id}/memory` — bounded snapshot, or `404`
  `SESSION_NOT_FOUND`.
- `DELETE /sessions/{session_id}/memory` — clears the session; reports whether
  anything was removed.

## Evaluation

`scripts/run_multiturn_eval.py` drives four fixed sessions against a running
service (deterministic mock generation, BM25 retrieval) and checks routing +
memory behavior:

- Session A: AML scenario → follow-up recalls prior flags (`answer_from_history`)
- Session B: vague first-turn query → `ask_clarifying_question`
- Session C: out-of-scope query → refusal with no scenario pollution
- Session D: AML scenario → follow-up asks for the previous citations

Each turn declares both `expected_route` and `expected_family`, so the harness
checks the fine-grained route *and* the three-outcome view.

Outputs (gitignored): `eval/results/multiturn_latest.jsonl`,
`eval/reports/multiturn_latest.md`.

## Limitations

- In-process and non-persistent; memory is lost on restart and is not shared
  across workers/replicas.
- Routing is rule-based; it will not capture every phrasing of a follow-up.
- Scenario summaries are truncated query text, not model-generated abstractions.
- This is an educational demo, not a production conversation-memory system.
