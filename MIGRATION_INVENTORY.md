# Migration Inventory

> Preflight recorded 2026-06-11 on branch `repo-consolidation`.
> Companion documents: [docs/migration_notes.md](docs/migration_notes.md) and
> [docs/implementation_plan.md](docs/implementation_plan.md).

## Source preflight

| Check | Result |
|---|---|
| Branch | `repo-consolidation` |
| Primary experiment source | `migration_staging/experiment_rag_v4_source.py` present |
| Primary indexing source | `migration_staging/build_data_v2_source.py` present |
| Fallback source | `notebooks_archive/*.ipynb` present; not activated |
| Python | `3.12.3` (plan targets 3.10/3.11; current native checks use 3.12) |
| Docker Compose | Unavailable in the current environment; Docker verification remains pending |

The preflight found untracked continuation work in `rag_core/retrieval.py` and
`rag_core/gate.py`. Both were validated, completed, committed, and pushed in
their respective stages. `.claude/` remains local tooling configuration and is
not part of the migration.

## Migration ledger

| Target | Notebook source | Status |
|---|---|---|
| `rag_core/config.py` | New service configuration per demo spec | Ported and committed (Stage 1) |
| `rag_core/schemas.py` | New API contract models per demo spec | Ported and committed (Stage 1) |
| `rag_core/loaders.py` | `load_all_indexes` (reshaped for JSON-only artifacts) | Ported and committed (Stage 2) |
| `artifacts/index/chunks.json` | Hand-written demo sample data | Added and committed (Stage 2) |
| `artifacts/index/manifest.json` | New demo artifact manifest | Added and committed (Stage 2) |
| `rag_core/retrieval.py` | `dense_search`, `bm25_search`, `hybrid_search`, `retrieve_contexts`; BM25 construction from indexing notebook | Ported and committed (Stage 3) |
| `rag_core/gate.py` | `KnowledgeManifest`, `TopicDetector`, `GateDecision`, `GateResult`, `SemanticScopeClassifier`, `pre_llm_gate` | Ported and committed (Stage 4) |
| `rag_core/generation.py` | `SYSTEM_PROMPT`, `build_user_prompt`, `call_llm`; deterministic mock is new demo code | Ported and committed (Stage 5) |
| `rag_core/pipeline.py` | `analyze_scenario` single-turn flow | Ported and committed (Stage 6) |
| `api/main.py` | New FastAPI wrapper per demo spec | Implemented and committed (Stage 6) |
| `tests/test_api_contract.py` | New contract tests per demo spec | Implemented and committed (Stage 6) |
| Docker packaging and smoke test | New demo packaging | Implemented and committed; native smoke verified, Docker runtime pending (Stage 7) |
| `indexing/build_data_v2.py` | PDF metadata/loading, chunking, FAISS/BM25 creation, artifact saving | Ported and committed; private-PDF E2E intentionally not run (Stage 8) |
| README and final documentation | Existing notebook narrative plus runnable service documentation | Completed (Stage 9) |

## Deferred and experimental

> **Note (updated 2026-06-24):** Several items marked "Planned" at migration time have
> since been implemented via post-migration PRs. See the table below for current status.

| Feature | Disposition |
|---|---|
| Semantic scope classifier | Experimental; opt-in and dense-backend dependent |
| Gemini and Groq generation | Experimental; key-gated with mock fallback |
| Gemma via Google AI Studio | Implemented — PR #8 (`feat: add Gemma Google AI Studio generation mode`) |
| Configurable live LLM timeout | Implemented — PR #11 |
| Ollama local generation mode | Implemented — PR #14 (`feat: add optional Ollama generation mode`) |
| Public 226-chunk corpus profile | Implemented — integrated via `feat: integrate public corpus model matrix` |
| Model matrix runner | Implemented — scripts/run_model_matrix.py |
| Intent routing | Implemented — PR #13 (`feat: add structured conversation memory and intent routing`); deterministic rule-based, no LLM dependency; see `rag_core/intent_router.py` and `docs/conversation_memory.md` |
| Structured conversation memory (multi-turn) | Implemented — PR #13; opt-in, local, in-process, bounded; see `rag_core/memory/` |
| Multi-turn chat and query rewriting | Partially implemented — structured memory and deterministic routing are live; open-ended LLM-based query rewriting remains notebook-only |
| Evaluation framework and experiment logging | Partial — API smoke, CQC-RAG Lite, multi-turn eval, failure diagnostics, and model matrix scripts committed; historical notebook-era experiment logger remains notebook-only |
| Full private-corpus rebuild | Offline-only; raw PDFs are not available or committed |

## Artifact and safety policy

- Raw PDFs, private data, `.env`, API keys, virtual environments, pickle files,
  and large FAISS artifacts are not committed.
- The repository ships only small demo `chunks.json` and `manifest.json`
  artifacts. Runtime indexes are rebuilt in memory.
- Display notebooks and archived source notebooks remain untouched.

## Verification summary

- Native compile, contract tests, Uvicorn HTTP checks, and HTTP smoke test pass.
- Dense requests degrade honestly to BM25 when the embedding model cannot be
  downloaded in the restricted environment.
- Docker Compose runtime verification is pending because Docker is unavailable.
- Live Groq/Gemini calls and private-PDF indexing were not run because no keys
  or private PDFs are available.
