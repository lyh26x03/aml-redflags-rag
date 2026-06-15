# ADR 0001: Demo Scope and Honesty Boundaries

## Status

Accepted

## Context

The original work developed through notebooks. The current repository turns a
focused part of that work into a runnable single-turn FastAPI demo.

The historical retrieval experiments used a private 226-chunk corpus that
cannot be fully committed. Raw PDFs, private corpus chunks, and binary retrieval
artifacts should not enter the repository. Multi-turn routing remains
notebook-only and is outside the service API.

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
