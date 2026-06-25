# ADR 0001: Demo Scope and Honesty Boundaries

## Status

Accepted — amended 2026-06-24 (see Amendment below)

## Context

The original work developed through notebooks. The current repository turns a
focused part of that work into a runnable single-turn FastAPI demo.

The historical retrieval experiments used a private 226-chunk corpus that
cannot be fully committed. Raw PDFs, private corpus chunks, and binary retrieval
artifacts should not enter the repository.

At initial migration time, multi-turn routing and conversation memory were
notebook-only and outside the service API. These were subsequently implemented
as opt-in features (see Amendment below).

## Decision

- Ship 12 hand-written sample chunks as the committed demo corpus.
- Rebuild BM25 and optional dense indexes in memory.
- Keep deterministic mock generation as the default.
- Expose fallback behavior and retrieval debug signals.
- Document historical benchmarks separately from claims about the committed
  demo corpus.
- Include CQC-RAG Lite only as a cross-query consistency evaluation harness,
  not as a full CQC-RAG implementation.

## Consequences

- A reviewer can run the project without API keys.
- The demo is smaller than the historical research environment, but its scope
  and behavior are explicit.
- Historical claims remain traceable, although they are not fully reproducible
  on the committed sample corpus.
- Future work can extend evaluation and user experience without corrupting the
  core demo scope.

---

## Amendment — 2026-06-24

Post-migration additions that extend the original scope without violating its
honesty principles:

**Multi-turn structured conversation memory (PR #13):** Opt-in, local,
in-process, bounded memory store (`rag_core/memory/`) with deterministic rule-based
intent routing (`rag_core/intent_router.py`). Single-turn behavior is
backward-compatible. Memory is not persisted and is never an unlimited transcript.

**Ollama local generation mode (PR #14):** Optional `llm_mode="ollama"` path for
local verification. Mock remains the default. Failures fall back to mock.

**Public 226-chunk corpus profile:** `data/public_corpus_226/` with committed
source PDFs. The original 12-chunk sample profile remains the default.

**Additional live generation paths:** Gemma via Google AI Studio (`llm_mode="gemma"`)
and configurable live LLM timeout. All key-gated; mock is still the keyless reviewer
path.

**Evaluation scripts:** `scripts/run_cqc_eval.py`, `run_multiturn_eval.py`,
`run_failure_diagnostics.py`, `run_model_matrix.py`, `run_reviewer_pack.py`.
These are diagnostic tooling; they do not change the core honesty constraints above.
