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
`rag_core/gate.py`. These files are treated as in-progress migration work and
must be validated before their stage commits. `.claude/` is local tooling
configuration and is not part of the migration.

## Migration ledger

| Target | Notebook source | Status |
|---|---|---|
| `rag_core/config.py` | New service configuration per demo spec | Ported and committed (Stage 1) |
| `rag_core/schemas.py` | New API contract models per demo spec | Ported and committed (Stage 1) |
| `rag_core/loaders.py` | `load_all_indexes` (reshaped for JSON-only artifacts) | Ported and committed (Stage 2) |
| `artifacts/index/chunks.json` | Hand-written demo sample data | Added and committed (Stage 2) |
| `artifacts/index/manifest.json` | New demo artifact manifest | Added and committed (Stage 2) |
| `rag_core/retrieval.py` | `dense_search`, `bm25_search`, `hybrid_search`, `retrieve_contexts`; BM25 construction from indexing notebook | In progress (Stage 3) |
| `rag_core/gate.py` | `KnowledgeManifest`, `TopicDetector`, `GateDecision`, `GateResult`, `SemanticScopeClassifier`, `pre_llm_gate` | In progress (Stage 4) |
| `rag_core/generation.py` | `SYSTEM_PROMPT`, `build_user_prompt`, `call_llm`; deterministic mock is new demo code | Pending (Stage 5) |
| `rag_core/pipeline.py` | `analyze_scenario` single-turn flow | Pending (Stage 6) |
| `api/main.py` | New FastAPI wrapper per demo spec | Pending (Stage 6) |
| `tests/test_api_contract.py` | New contract tests per demo spec | Pending (Stage 6) |
| Docker packaging and smoke test | New demo packaging | Pending (Stage 7) |
| `indexing/build_data_v2.py` | PDF metadata/loading, chunking, FAISS/BM25 creation, artifact saving | Pending (Stage 8) |
| README and final documentation | Existing notebook narrative plus runnable service documentation | Pending (Stage 9) |

## Deferred and experimental

| Feature | Disposition |
|---|---|
| Semantic scope classifier | Experimental; opt-in and dense-backend dependent |
| Gemini and Groq generation | Experimental; key-gated with mock fallback |
| Multi-turn chat and query rewriting | Planned; not exposed by the demo API |
| Intent routing | Planned; remains notebook-only |
| Evaluation framework and experiment logging | Deferred; remains notebook-only |
| Full private-corpus rebuild | Offline-only; raw PDFs are not available or committed |

## Artifact and safety policy

- Raw PDFs, private data, `.env`, API keys, virtual environments, pickle files,
  and large FAISS artifacts are not committed.
- The repository ships only small demo `chunks.json` and `manifest.json`
  artifacts. Runtime indexes are rebuilt in memory.
- Display notebooks and archived source notebooks remain untouched.
