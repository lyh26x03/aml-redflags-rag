# Demo Evidence Pack

## Purpose of the evidence pack

This document is a reviewer-facing summary of what the current main branch
demonstrates, what was validated locally, and how to reproduce that validation.
It is written for interview, walkthrough, and code-review contexts.

The repository is a demo and evaluation-oriented AML red-flags RAG service. It
is not a production AML compliance system, not a transaction-monitoring
platform, and not legal advice.

## Current main-branch feature summary

- FastAPI service with `/health`, `/query`, and `/sources`.
- Single-turn RAG flow with rule-based refusal, retrieval, evidence-bound
  generation, citations, and optional debug fields.
- Retrieval modes for BM25, dense, and hybrid RRF, with explicit fallback
  reporting when dense/hybrid is unavailable.
- Deterministic mock generation as the default execution mode.
- Opt-in structured conversation memory for multi-turn demo sessions.
- Deterministic intent routing for memory-enabled follow-up behavior.
- Optional `llm_mode="ollama"` path for local generation verification.
- Local evaluation scripts for tests, smoke checks, consistency checks, and
  multi-turn routing/memory checks.

PR #13 is already merged on main and adds opt-in structured conversation memory
plus deterministic intent routing. PR #14 is already merged on main and adds
the optional Ollama local generation mode.

## Single-turn RAG behavior

The default `/query` flow is single-turn. A request is checked by the rule-based
gate first. Out-of-scope queries are refused before retrieval. Allowed queries
then use the requested retrieval mode, pass retrieved evidence to generation,
and return a structured response with assessment, identified flags, citations,
refusal information, and optional debug fields.

This single-turn path remains the baseline reviewer path because it does not
depend on conversation state.

## Optional structured conversation memory

Memory is opt-in via `session_id` and `use_memory=true`.

Default `/query` behavior remains single-turn and backward compatible. If the
new memory fields are omitted, the service behaves like the earlier single-turn
API.

The memory implementation stores structured state, not raw transcript storage.
It keeps bounded conversation state such as the active scenario summary,
identified flags, citations, retrieved chunk IDs, prior answer summary, and
unresolved clarification needs.

The memory store is bounded and local in-process. It is not persisted, not
shared across workers, and is lost on restart.

## Intent routing overview

Intent routing is deterministic and rule-based. It is used for memory-enabled
requests and does not require a live model.

Public route families:

- `retrieve`
- `refuse`
- `no_retrieval_response`

Internal debug routes:

- `retrieve`
- `retrieve_with_memory`
- `answer_from_history`
- `ask_clarifying_question`
- `refuse`

In practical terms:

- `retrieve` means the system performed evidence retrieval.
- `refuse` means the request was outside the demo scope.
- `no_retrieval_response` means the system answered from structured state or
  asked the user for clarification without new retrieval.

## Optional Ollama local generation mode

`llm_mode="ollama"` is optional.

Mock remains the default.

Ollama requires a local Ollama server and a local model. The service calls the
local Ollama HTTP endpoint directly.

If Ollama is unavailable, times out, or returns malformed output, the service
falls back to mock.

This is a local verification path, not a model-quality benchmark.

## Validation evidence

Local validation evidence from main:

- `pytest`: `121 passed, 2 warnings`
- `multi-turn eval`: `4 / 4 sessions passed`

The multi-turn eval means that four fixed demo conversations completed without
route or memory mismatches. In plain English, the service handled the expected
follow-up patterns correctly: recalling prior flags, asking for clarification
when a prompt was too vague, refusing out-of-scope requests, and recalling
prior citations when asked.

## How to reproduce the validation

Run the automated tests:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Run the multi-turn evaluator against a running local service:

```powershell
.venv\Scripts\python.exe scripts\run_multiturn_eval.py
```

Expected results:

- `121 passed, 2 warnings`
- `multi-turn eval: 4 / 4 sessions passed`

## Demo commands

Start the service in mock mode:

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Single-turn query:

```powershell
$body = @{
  query = "Funds show rapid movement through a virtual asset exchange."
  retrieval_mode = "hybrid"
  llm_mode = "mock"
  include_debug = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/query `
  -Method Post -ContentType "application/json" -Body $body
```

Memory-enabled query:

```powershell
$body = @{
  query = "Funds show rapid movement through a virtual asset exchange."
  retrieval_mode = "bm25"
  llm_mode = "mock"
  include_debug = $true
  session_id = "demo-1"
  use_memory = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/query `
  -Method Post -ContentType "application/json" -Body $body
```

Inspect session memory:

```powershell
Invoke-RestMethod http://localhost:8000/sessions/demo-1/memory
```

Optional Ollama run:

```powershell
$body = @{
  query = "Funds show rapid movement through a virtual asset exchange."
  retrieval_mode = "hybrid"
  llm_mode = "ollama"
  include_debug = $true
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/query `
  -Method Post -ContentType "application/json" -Body $body
```

## Known limitations

- This repository is a demo and evaluation-oriented service, not a production
  AML system.
- The default corpus is small and is intended for reviewability, not coverage.
- Structured memory is local, in-process, bounded, and non-persistent.
- Intent routing is rule-based and will not cover every phrasing.
- Optional live generation paths are verification paths, not benchmark claims.
- Ollama support depends on a local operator setup outside the repository.

## Suggested reviewer walkthrough

1. Read the README overview and this evidence pack.
2. Run `pytest` to confirm the current branch is stable.
3. Start the API in default mock mode.
4. Send one single-turn AML query and inspect `assessment`, `identified_flags`,
   `citations`, and `debug`.
5. Run one memory-enabled two-turn session to see routing and bounded memory.
6. Run `scripts/run_multiturn_eval.py` and confirm `4 / 4 sessions passed`.
7. If desired, test the optional Ollama path as a local integration check, not
   as a quality claim.
