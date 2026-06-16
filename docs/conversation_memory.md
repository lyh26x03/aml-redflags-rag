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
- Intent routing's memory-only routes (`clarify`, `answer_from_history`,
  `retrieve_with_memory`, and the router's own non-AML refusal) only activate
  when memory is enabled, so single-turn behavior is preserved.

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

| Route | When | Behavior |
|---|---|---|
| `retrieve` | Normal AML query | Evidence retrieval; updates memory |
| `retrieve_with_memory` | Follow-up connector (e.g. "那跟…有關嗎") with memory available | Composes prior scenario + new question, retrieves, updates memory |
| `answer_from_history` | Recall of prior answer/flags/citations (e.g. "剛剛", "上一題", "those flags", "剛剛引用的是哪些來源") | Answers from structured memory; no new retrieval |
| `clarify` | Under-specified, no AML topic (e.g. "這樣有沒有問題？") | Asks for detail; stores an unresolved clarification need |
| `refuse` | Gate out-of-scope (sanctions/TBML/tax) **or** clearly non-AML request (e.g. "幫我推薦晚餐") | Structured refusal; **does not touch memory** |

`retrieve_with_memory` is implemented: it is the route for a follow-up that
needs the previous scenario context *and* fresh retrieval (as opposed to
`answer_from_history`, which restates what is already known).

## Memory schema

Per `session_id`, `ConversationMemory` holds:

- `session_id`, `turn_count`, `created_at`, `updated_at`
- `active_scenario_summary` — latest retrieval scenario (truncated)
- `active_entities_or_context_terms` — detected AML topic terms (bounded, deduped)
- `active_flags` — deduplicated red flags (`code`/`name`/`name_zh`)
- `active_citations` — bounded list of `{chunk_id, source, excerpt}` (excerpt truncated)
- `retrieved_chunk_ids` — bounded, deduped, most-recent-kept
- `last_assessment`, `last_answer_summary`
- `unresolved_questions` — bounded clarification needs
- `recent_turns` — bounded list of concise per-turn summaries

### Bounds

| Bound | Value |
|---|---|
| `recent_turns` | 8 |
| `active_citations` | 10 |
| `active_flags` | 12 (deduped by code) |
| `retrieved_chunk_ids` | 30 |
| `active_entities_or_context_terms` | 20 |
| `unresolved_questions` | 10 |
| citation excerpt | 200 chars |
| scenario / answer / query summaries | 500 / 400 / 300 chars |
| retained sessions (per process) | 256 (LRU-evicted) |

## Memory behavior by route

- `retrieve` — updates the active scenario, flags, citations, chunk IDs, last
  assessment/answer, and appends a recent turn.
- `retrieve_with_memory` — same, but the retrieval query is composed from the
  prior `active_scenario_summary` plus the new question. `memory_used = true`.
- `answer_from_history` — answers from memory (or returns a clear no-context
  message when nothing is stored); appends a recent turn; does not change the
  active scenario.
- `clarify` — stores the unresolved clarification need and appends a recent
  turn; no flags/citations are added.
- `refuse` — returns a structured refusal and **does not modify memory at all**,
  so out-of-scope requests cannot pollute the active AML scenario state.

## Debug fields (inside `debug`)

`intent_route`, `route_reason`, `memory_used`, `memory_available`,
`memory_updated`, `memory_turn_count`, `session_id`,
`referenced_previous_answer`, `referenced_previous_evidence`, `active_flags`,
`active_citation_count`.

## Inspection endpoints (demo/debug only, no auth)

- `GET /sessions/{session_id}/memory` — bounded snapshot, or `404`
  `SESSION_NOT_FOUND`.
- `DELETE /sessions/{session_id}/memory` — clears the session; reports whether
  anything was removed.

## Evaluation

`scripts/run_multiturn_eval.py` drives four fixed sessions against a running
service (deterministic mock generation, BM25 retrieval) and checks routing +
memory behavior:

- Session A: AML scenario → follow-up recalls prior flags
- Session B: vague first-turn query → clarification
- Session C: out-of-scope query → refusal with no scenario pollution
- Session D: AML scenario → follow-up asks for the previous citations

Outputs (gitignored): `eval/results/multiturn_latest.jsonl`,
`eval/reports/multiturn_latest.md`.

## Limitations

- In-process and non-persistent; memory is lost on restart and is not shared
  across workers/replicas.
- Routing is rule-based; it will not capture every phrasing of a follow-up.
- Scenario summaries are truncated query text, not model-generated abstractions.
- This is an educational demo, not a production conversation-memory system.
