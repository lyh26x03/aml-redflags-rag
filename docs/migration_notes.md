# Migration Notes

> Status: implementation complete on `repo-consolidation` (2026-06-11). Companion document: [implementation_plan.md](implementation_plan.md).
> Scope of this document: an honest inventory of what exists, what migrates, what stays experimental, and what remains planned.

## 1. Current repo state

- Branch: `repo-consolidation` (PR #1 open against `main` — do not merge during migration).
- 17 tracked files. Working tree clean at time of audit.
- Scaffold directories exist but are empty (`.gitkeep` only): `api/`, `rag_core/`, `artifacts/index/`, `data/sample_docs/`, `tests/`.
- No `requirements.txt`, `Dockerfile`, `docker-compose.yml`, or `.env.example` yet.
- `docs/demo_spec.md` (957 lines) is the binding API/behavior contract for the demo.
- `MIGRATION_INVENTORY.md` is a one-line stub (to be filled during implementation Stage 0).
- `README.md` (21 KB, bilingual EN/ZH) describes the notebook experiments, including evaluation numbers. Relative to currently runnable code it overclaims — see Risks (§10). Fixed only in the final implementation stage.

```text
aml-redflags-rag/
├── api/                          (empty scaffold)
├── rag_core/                     (empty scaffold)
├── artifacts/index/              (empty scaffold)
├── data/sample_docs/             (empty scaffold)
├── tests/                        (empty scaffold)
├── docs/
│   ├── demo_spec.md              (binding demo contract)
│   ├── migration_notes.md        (this file)
│   └── implementation_plan.md    (staged execution plan)
├── notebooks_archive/
│   ├── build_data_v2_source.ipynb
│   └── experiment_rag_v4_source.ipynb
├── migration_staging/            (GITIGNORED — local-only, see §3)
│   ├── build_data_v2_source.py
│   ├── experiment_rag_v4_source.py
│   └── *.ipynb copies
├── build_data_v2_display.ipynb
├── experiment_rag_v2_display.ipynb
├── experiment_rag_v3.5_display.ipynb
├── experiment_rag_v4_display.ipynb
├── MIGRATION_INVENTORY.md
├── .gitignore
└── README.md
```

## 2. Display notebooks (do not touch)

These four root-level notebooks are GitHub display artifacts (outputs cleared / curated for reading). They are **not** migration sources and must not be deleted, modified, or moved:

| Notebook | Covers |
|---|---|
| `build_data_v2_display.ipynb` | Indexing pipeline v2: PDF → chunking → FAISS + BM25 |
| `experiment_rag_v2_display.ipynb` | Baseline hybrid RAG (dense + BM25 + RRF), priority weighting, pre-LLM gate |
| `experiment_rag_v3.5_display.ipynb` | Multi-turn: query rewrite + conversation-state decoupling |
| `experiment_rag_v4_display.ipynb` | Intent routing (rule-based + LLM), multi-turn A/B testing |

## 3. Migration sources

| Priority | Source | Location | Notes |
|---|---|---|---|
| Primary | `build_data_v2_source.py` (14 KB), `experiment_rag_v4_source.py` (226 KB) | `migration_staging/` | Colab `.py` exports — the cleanest extraction sources. **Gitignored and local-only**: they exist on this machine but NOT in any clone/worktree. Implementation Stage 0 must verify presence. |
| Fallback | `build_data_v2_source.ipynb`, `experiment_rag_v4_source.ipynb` | `notebooks_archive/` | Committed. If staging is absent, extract code cells by parsing the `.ipynb` JSON. |

The `.py` exports are raw Colab dumps: they contain `!pip` magics, `drive.mount()`, `userdata.get()` API key access, and interactive `input()` — usable as readable source, not importable as-is.

## 4. Useful logic in `build_data_v2_source` (indexing pipeline)

Line references are into `migration_staging/build_data_v2_source.py`.

| Function | Line | What it does |
|---|---|---|
| `get_pdf_metadata(pdf_name)` | 130 | Assigns per-document metadata: source (`FATF`/`TW_Gov`), language (`en`/`zh`), `doc_type` (`red_flag`/`training`), `retrieval_priority` (1.0/0.9/0.8), `doc_category` (`core`/`sector_specific`/`knowledge_bridge`), `explanation_style` |
| `load_pdfs(folder_path)` | 187 | `pypdf.PdfReader` page-by-page text extraction, skips blank pages |
| `create_chunks(pages, chunk_size, chunk_overlap)` | 241 | `RecursiveCharacterTextSplitter` (langchain-text-splitters), size 400 / overlap 80, Chinese-aware separators `["\n\n","\n","。",".","！","!","？","?","；",";"," "]` |
| `create_faiss_index(chunks, embedding_model_name)` | 302 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` → 384-dim normalized vectors → `faiss.IndexFlatIP` |
| `create_bm25_index(chunks)` | 342 | `rank_bm25.BM25Okapi`; jieba tokenization for `language == "zh"`, `lower().split()` for English |
| `save_all_indexes(...)` | 378 | Writes 5 artifacts: `faiss_index.bin`, `chunks.json`, `bm25_index.pkl`, `tokenized_corpus.pkl`, `metadata.json` |

Real chunk schema (per chunk, 9 fields):

```json
{
  "text": "...", "page": 1,
  "chunk_id": "fatf_tbm_laundering_red_flags.pdf_p1_c0",
  "source": "FATF", "language": "en", "doc_type": "red_flag",
  "retrieval_priority": 1.0, "doc_category": "core",
  "explanation_style": "authoritative"
}
```

`metadata.json` schema: `{version, created_at, config: {embedding_model, chunk_size, chunk_overlap}, stats: {total_chunks, total_vectors, vector_dimension}}`.

Original corpus: 3 private PDFs (FATF TBML red flags, FATF virtual assets red flags, TW AML training slides) → 226 chunks. **The PDFs are not in this repo and are assumed unavailable** (Google Drive only).

Colab-only parts to strip: `drive.mount`, hardcoded `/content/drive/MyDrive/AML/...` paths, `!pip` magics, trailing git cell.

## 5. Useful logic in `experiment_rag_v4_source` (retrieval/QA pipeline)

Line references are into `migration_staging/experiment_rag_v4_source.py`.

| Component | Line | What it does |
|---|---|---|
| `load_all_indexes(index_dir)` | 907 | Loads all 5 artifacts; validates FAISS vector count == chunk count |
| `dense_search(...)` | 1035 | Encode query (normalized) → `faiss_index.search` |
| `bm25_search(...)` | 1077 | CJK-character detection → jieba tokenization, else lowercase split; `bm25.get_scores` |
| `hybrid_search(...)` | 1119 | **Genuine RRF**: `score += 1/(rrf_k + rank)` over dense + BM25 rank lists, `rrf_k = 60`; then metadata priority weighting: `rrf_score * chunk["retrieval_priority"]` |
| `retrieve_contexts(...)` | 1218 | Mode dispatch wrapper (dense / bm25 / hybrid) |
| `KnowledgeManifest` | 423 | Covered topics: virtual_assets, cash_structuring, rapid_movement, third_party, cross_border, identity_mismatch, shell_company. Explicitly NOT covered: TBML, sanctions, tax_evasion |
| `TopicDetector` | 456 | Rule-based keyword matching for topic + evidence detection (no ML deps) |
| `GateDecision` / `GateResult` | 541 / 546 | Structured ALLOW/REFUSE decision with `reason_code` and `detected_topics` |
| `SemanticScopeClassifier` | 570 | Embedding-similarity out-of-scope detection, threshold 0.35 — **requires the embedding model** |
| `pre_llm_gate(...)` | 638 | Orchestrates rules + (optional) semantic check before any LLM call |
| `SYSTEM_PROMPT` | 2738 | ~350-line AML assistant prompt defining red flags RF-01..RF-08, assessment levels CONFIRMED/POSSIBLE/UNLIKELY/REFUSE, JSON output contract |
| `build_user_prompt(...)` | 2807 | Scenario + retrieved contexts + style tags |
| `call_llm(...)` | 2853 | Unified Groq (`llama-3.3-70b-versatile` etc.) / Gemini (`gemini-2.0-flash`) interface, JSON response mode, temperature 0.1 |
| `analyze_scenario(...)` | 2963 | Full single-turn pipeline: gate → retrieve → prompt → LLM → structured result with `_retrieved_chunks` |
| Multi-turn (Part 6) | 3811+ | `rewrite_query` (3811), `TurnIntent` enum (4006), rule/LLM intent classification (4019/4107), `chat()` (4244), `chat_loop()` (4615), multi-turn logging/A-B testing (4734–5692) |
| Eval framework | 2012–2228 | `precision_at_k`, `recall_at_k`, `mrr`, `evaluate_single_method`, `run_eval_comparison` + annotation helpers |

Colab-only parts to strip: `drive.mount`, `userdata.get()` for API keys, interactive `input()` model selection, Drive experiment-logging paths, `ExperimentLogger` (L165).

## 6. Target module mapping

| Target module | Source (function @ line in staging `.py`) | Notes |
|---|---|---|
| `indexing/build_data_v2.py` | `get_pdf_metadata` 130, `load_pdfs` 187, `create_chunks` 241, `create_faiss_index` 302, `create_bm25_index` 342, `save_all_indexes` 378 | Offline rebuild script (argparse). Compiles + `--help` only — cannot run E2E without private PDFs |
| `rag_core/loaders.py` | `load_all_indexes` 907 (reshaped) | Loads `chunks.json` + `manifest.json`; degrades on missing artifacts, never raises at startup |
| `rag_core/retrieval.py` | `dense_search` 1035, `bm25_search` 1077, `hybrid_search` 1119, `retrieve_contexts` 1218; BM25 build mirrors `create_bm25_index` 342 | BM25 + in-memory FAISS rebuilt at startup from `chunks.json` (no pickle/bin artifacts). RRF math ported verbatim |
| `rag_core/gate.py` | `KnowledgeManifest` 423, `TopicDetector` 456, `GateDecision`/`GateResult` 541/546, `pre_llm_gate` 638; `SemanticScopeClassifier` 570 (opt-in) | Rule-based baseline has zero ML deps |
| `rag_core/generation.py` | `SYSTEM_PROMPT` 2738, `build_user_prompt` 2807, `call_llm` 2853 (reworked to httpx REST, no SDKs) + new deterministic mock generator | Mock is new code by necessity (no mock existed in notebooks) but assembles answers only from retrieved chunk evidence |
| `rag_core/pipeline.py` | `analyze_scenario` 2963 (single-turn shape) | validate → gate → retrieve → generate → assemble |
| `api/main.py` | New (per `docs/demo_spec.md`) | Thin HTTP layer over `pipeline.py`; lifespan loads artifacts once |

## 7. Features implemented in notebook form only (not migrating now)

- Retrieval evaluation framework (P@k / Recall@k / MRR, annotation helpers, 3-way method comparison).
- `ExperimentLogger` and Drive-based experiment run logging.
- Multi-turn chat: `chat()`, `chat_loop()`, query rewriting, conversation state.
- Intent routing: `TurnIntent`, rule-based and LLM-based classification.
- Multi-turn A/B testing and session annotation tooling.

## 8. Features to mark **experimental** in the demo

- **Dense retrieval in lite profile**: with `requirements-lite.txt` the dense backend is absent; service runs BM25-only with `fallback_used=true` and `fallback_reason` in debug output. (With the default full profile, dense + RRF hybrid is implemented and runnable.)
- **`SemanticScopeClassifier`**: ported but env-gated **off** by default (threshold 0.35 was never tuned beyond the notebook).
- **Gemini / Groq live backends**: code present, key-gated; any error falls back to mock with an explicit flag. Untestable in this environment (no keys).

## 9. Features to mark **planned** (not ported)

- Multi-turn conversation + intent routing (v4 Part 6) — no API surface in the demo spec; pointer to `experiment_rag_v4_display.ipynb`.
- Evaluation automation (eval endpoint or CLI re-running P@k/Recall/MRR).
- `/ingest` or any document upload path.
- Full-corpus index rebuild as part of the service (the offline script exists but needs the private PDFs).

## 10. Risks and assumptions

1. **`migration_staging/` is local-only** (gitignored). If the implementation run happens in a fresh clone/worktree, the `.py` sources are missing → fall back to parsing `notebooks_archive/*.ipynb`. This is the single most likely failure of an unattended implementation run.
2. **Raw PDFs are private and unavailable.** `indexing/build_data_v2.py` can be ported and syntax-checked but never executed end-to-end here. Do not fake its outputs.
3. **No API keys available.** `LLM_MODE=mock` is the default and the only path verified in acceptance tests.
4. **README currently overclaims** relative to runnable code (eval numbers are notebook-experiment results on the private 226-chunk corpus; multi-turn/intent routing exists only in notebooks). The honesty rewrite happens in one coherent commit at the final stage — do not partially edit README earlier.
5. **`.gitignore` blocks `artifacts/index/*`, `*.pkl`, `*.faiss`, `*.pdf`.** Committing the sample `chunks.json`/`manifest.json` requires explicit `!` exceptions placed after the `artifacts/index/*` rule, verified with `git check-ignore`. FAISS/pickle artifacts are never committed — both indexes are rebuilt in memory at startup.
6. **`docs/demo_spec.md` contradictions** (resolved decisions, see implementation plan §"Spec reconciliation"):
   - Spec lists `faiss.index` / `bm25.pkl` under `artifacts/index/`, but committing them is banned → treated as local-only outputs of the offline indexing script.
   - `GET /sources` is "optional but allowed" in the spec → implemented (cheap, manifest-driven).
   - RF flag names: spec uses English, notebook catalog is Chinese → bilingual catalog, `name` (EN) + additive `name_zh`.
   - Spec shows `gate_decision: "allow"` lowercase vs notebook enum `ALLOW` → serialize lowercase.
7. **Mock honesty choice**: mock generation never emits `assessment: "confirmed"` (reserved for real LLM judgment); it returns `possible`, `unlikely`, or `refuse`, with empty citations when no evidence is retrieved. This is deliberate, not a contract violation.
8. **Windows host**: implementation runs on Windows 11 / PowerShell. No curl-with-JSON examples (escaping); use FastAPI `TestClient`, `tests/smoke_test.py`, and `Invoke-RestMethod`. Docker healthcheck uses `python -c` with urllib (slim image has no curl).
9. **Sample artifacts are demo data, not FATF source text.** Hand-written, marked `source_type: "sample"`, and described as such in manifest, /sources, and README.

## 11. Final implementation and acceptance ledger

| Area | Final status |
|---|---|
| Stages 0–8 | Implemented, committed separately, and pushed to `repo-consolidation` |
| Config, schemas, sample artifacts, loader | Implemented |
| BM25, dense FAISS, RRF, priority weighting | Implemented; dense runtime degraded honestly when model download was blocked |
| Rule-based gate and optional semantic classifier | Implemented; semantic mode remains experimental/off by default |
| Deterministic mock and key-gated Groq/Gemini REST paths | Implemented; mock verified, live providers not called without keys |
| Single-turn pipeline and FastAPI endpoints | Implemented and contract-tested |
| Dockerfile, Compose, smoke test | Implemented; smoke test verified natively, Docker runtime unavailable on this host |
| Offline indexing CLI | Implemented; compiles and `--help` works; private-PDF E2E intentionally not run |
| README honesty rewrite | Completed; notebook-only results and features are clearly separated |

Final native verification:

- `python -m compileall` passes for service, tests, and indexing modules.
- `pytest tests -q` passes.
- Native Uvicorn `/health`, `/query`, and `/sources` checks pass.
- `tests/smoke_test.py` passes against native Uvicorn.
- No tracked `.env`, PDF, pickle, or FAISS artifacts were found.

The literal plan command `python -m compileall .` also descends into the local,
gitignored `migration_staging/*.py` Colab exports. Those source-reference files
intentionally contain notebook magics such as `!pip` and are not importable
Python modules, so the targeted runnable-module compile command is the valid
acceptance check. The staging sources were not modified or committed.

Unverified by design or environment:

- Docker Compose build/container health: Docker command unavailable.
- Full dense/hybrid runtime: embedding model download blocked; fallback behavior verified.
- Groq/Gemini live calls: no API keys.
- Full-corpus index build: private PDFs unavailable.
