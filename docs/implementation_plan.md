# Implementation Plan — Notebook → FastAPI + Docker Compose Demo

> Written 2026-06-11 for an implementation agent running on Windows 11 / PowerShell with Docker Compose available, on branch `repo-consolidation`. One commit + push per stage. Companion inventory: [migration_notes.md](migration_notes.md). Binding API contract: [demo_spec.md](demo_spec.md).

---

## Autonomy policy (binding)

**The implementation agent MAY automatically:**
- Create Python modules, edit docs, create `.env.example`, `requirements.txt`, `requirements-lite.txt`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`.
- Create small, clearly labeled sample demo artifacts (`artifacts/index/chunks.json`, `manifest.json`).
- Add `.gitignore` exceptions for exactly those two files.
- Port notebook logic with minimal de-Colab edits (strip `drive.mount`, `userdata`, `!pip`, `input()`).
- Run tests, fix syntax errors, commit stage-by-stage, push to `repo-consolidation`.
- Fall back from `migration_staging/*.py` to parsing `notebooks_archive/*.ipynb` if staging is absent.
- Add **additive** fields to API responses (never remove or rename spec fields).

**The implementation agent MUST NOT automatically:**
- Merge the open PR into `main`.
- Delete, modify, or move any notebook (root display notebooks or `notebooks_archive/`).
- Commit `.env`, API keys, raw PDFs, private data, or `*.pkl`/`*.pickle`/`*.faiss` artifacts, or anything from `migration_staging/`.
- Introduce LangChain or LlamaIndex into the service (`langchain-text-splitters` is permitted only inside `indexing/build_data_v2.py`, and a stdlib reimplementation is preferred even there).
- Introduce a database or Kubernetes.
- Rewrite retrieval math or invent new RAG logic where notebook logic exists.
- Claim full FAISS/BM25/RRF support in any mode where it is not actually callable — degraded modes must be labeled in debug output (`fallback_used`, `fallback_reason`).
- Fabricate benchmark results or re-attribute notebook eval numbers to the demo.
- Require API keys for demo success: `LLM_MODE=mock` must fully work with zero keys.
- Port multi-turn / intent routing into the API surface (planned-only, see D7).

---

## Design decisions

### D1. Dependencies: full ML demo profile by default + lightweight fallback (user-decided)
- `requirements.txt` (default; used by the Dockerfile): `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `pydantic-settings`, `httpx`, `pytest`, `rank_bm25`, `jieba`, `sentence-transformers`, `faiss-cpu`, `pypdf` (for the indexing script). Image ~2–3 GB; dense + RRF hybrid genuinely works out of the box.
- `requirements-lite.txt`: `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `pydantic-settings`, `httpx`, `pytest`, `rank_bm25`, `jieba`. For quick native runs; the service degrades honestly (see D2).
- No binary index artifacts in git: at startup the service builds **BM25 from `chunks.json` text** (mirrors `create_bm25_index`, staging `build_data_v2_source.py:342`) and, when the dense backend imports succeed, an **in-memory `faiss.IndexFlatIP`** by embedding the chunk texts (sample scale → seconds). Feature detection via `try: import sentence_transformers, faiss`.

### D2. Retrieval honesty matrix (requested vs effective mode)
`debug.retrieval_mode` = **effective** mode per spec; additive fields `requested_mode`, `fallback_reason`.

| requested | dense available? | effective | dense_used | bm25_used | rrf_used | fallback_used |
|---|---|---|---|---|---|---|
| hybrid | yes | hybrid | true | true | true | false |
| hybrid | no | bm25 | false | true | false | **true** (`"dense backend unavailable — install requirements.txt (full profile)"`) |
| dense | yes | dense | true | false | false | false |
| dense | no | bm25 | false | true | false | **true** |
| bm25 | — | bm25 | false | true | false | false |

Degraded modes never error (spec: don't crash) and are never silently relabeled. RRF math ported verbatim from staging `experiment_rag_v4_source.py:1119` (`rrf_k=60`, then `rrf_score * chunk["retrieval_priority"]`).

### D3. Gate: rule-based baseline; semantic gate experimental
Port `KnowledgeManifest` (L423), `TopicDetector` (L456), `GateDecision`/`GateResult` (L541/546), `pre_llm_gate` (L638) — deterministic, zero ML deps. `SemanticScopeClassifier` (L570, threshold 0.35) is ported but activated only when `ENABLE_SEMANTIC_GATE=true` **and** the dense backend is available; default **off**. Debug reports which classifiers ran. Conservative behavior: allow AML-related queries; refuse only obvious out-of-scope (TBML, sanctions, tax evasion, explicit knowledge gaps); when uncertain, allow retrieval.

### D4. Mock generation: deterministic, evidence-assembled, conservative
Static bilingual `RF_CATALOG` (RF-01 Structuring/門檻拆分 … RF-08 Opaque Ownership/公司不透明, from `SYSTEM_PROMPT` at L2738). Mock algorithm:
1. Candidate flags = `TopicDetector` hits on the query ∪ flag metadata on retrieved chunks, intersected with the catalog.
2. Citations = top retrieved chunks (`chunk_id`, `source`, excerpt ≈ first 200 chars).
3. Assessment: gate refused → `refuse`; no chunks or no flags → `unlikely` with an explicit insufficient-evidence answer and **empty citations** (never fabricate); ≥1 flag with citation → `possible`.
4. Mock **never** emits `confirmed` (reserved for real LLM judgment), never mentions eval metrics, never uses large Markdown headings, and is deterministic (same input → same output, testable).

### D5. Sample artifacts: ~12 hand-written chunks, superset schema
Per chunk: `chunk_id` (e.g. `sample_va_p1_c0`), `text`, `page`, `source`, `source_type: "sample"`, `language` (zh/en mix), `doc_type`, `doc_category` (mix of core/sector_specific/knowledge_bridge), `retrieval_priority` (mix 1.0/0.9/0.8 so priority weighting is observable in rankings), `explanation_style`, optional `related_flags: ["RF-xx"]` (loader treats as optional so real rebuilt chunks stay valid). Cover ≥6 of RF-01/02/04/06/07/08; content may paraphrase the scenario bank (staging `experiment_rag_v4_source.py:3134–3311`) — **not** copied FATF source text, clearly described as demo samples. `manifest.json`: `{version: "demo-sample-v1", artifact_type: "sample", created_at, config: {embedding_model: null, chunk_size: 400, chunk_overlap: 80}, stats: {total_chunks, vector_dimension: null}, sources: [...]}` — `sources` drives `GET /sources`.

### D6. Source preflight + commit discipline
Stage 0 verifies `migration_staging/*.py` exists (it is gitignored and exists only in the main checkout at `C:\Users\USER\Documents\aml-redflags-rag`); if absent, extract code cells from `notebooks_archive/*.ipynb` via JSON parsing. Commit + push to `repo-consolidation` after each green stage. Rollback unit = `git revert <stage-commit>`; never force-push.

### D7. Multi-turn / intent routing: NOT ported
Part 6 (`chat`, `TurnIntent`, `rewrite_query`, intent classification) depends on LLM intent classification and conversation state; no API surface in the spec. Marked **Planned** in README with a pointer to `experiment_rag_v4_display.ipynb`. Honesty beats feature count.

### D8. Windows-friendly verification
No curl-with-JSON examples. Contract tests use `fastapi.testclient.TestClient`; `tests/smoke_test.py` uses `httpx` against `SMOKE_BASE_URL` (default `http://localhost:8000`); README shows `Invoke-RestMethod` plus bash equivalents. Docker healthcheck = `python -c` urllib (slim image has no curl). Use `.venv\Scripts\python.exe` directly instead of profile-dependent activation.

---

## Spec reconciliation ledger (read before coding)

1. `demo_spec.md` lists `faiss.index`/`bm25.pkl` under `artifacts/index/` but `.gitignore` and policy ban committing them → they are local-only outputs of `indexing/build_data_v2.py`; the repo ships `chunks.json` + `manifest.json` only; indexes rebuild in memory at startup. State this in README.
2. `GET /sources` is spec-optional → implemented (cheap, manifest-driven).
3. RF flag names EN (spec) vs ZH (notebook) → bilingual catalog: `name` (EN per spec) + additive `name_zh`.
4. `gate_decision` serialized lowercase (`"allow"`/`"refuse"`) although the notebook enum is uppercase.
5. Mock restricts `assessment` to `possible|unlikely|refuse` — deliberate, documented, not a contract violation.

---

## Stages

### Stage 0 — Preflight + inventory
- **Files:** `MIGRATION_INVENTORY.md` (fill stub with the function→module ledger from migration_notes §6 + ported/deferred status), `docs/migration_notes.md` (append a "fallback activated" note only if staging is missing).
- **Commands:**
  ```powershell
  git branch --show-current            # expect repo-consolidation
  git status --porcelain               # expect clean
  Test-Path migration_staging\experiment_rag_v4_source.py   # True → primary source; False → ipynb fallback
  python --version                     # expect 3.10/3.11
  docker compose version
  ```
- **Acceptance:** branch/clean confirmed; source path decided and recorded.
- **Commit:** `docs: record migration inventory and source preflight`
- **Rollback:** `git revert HEAD` (docs-only).

### Stage 1 — Requirements, config, schemas
- **Files:** `requirements.txt` (full, per D1), `requirements-lite.txt`, `.env.example` (exactly: `APP_ENV=local`, `API_HOST=0.0.0.0`, `API_PORT=8000`, `ARTIFACT_DIR=artifacts/index`, `LLM_MODE=mock`, `MODEL_NAME=mock-local`, `DEFAULT_TOP_K=5`, `DEFAULT_RETRIEVAL_MODE=hybrid`, `ENABLE_DEBUG=true`, `ENABLE_SEMANTIC_GATE=false`, `GEMINI_API_KEY=`, `GROQ_API_KEY=`), `rag_core/__init__.py`, `rag_core/config.py` (pydantic-settings; must not fail when `.env` is missing), `rag_core/schemas.py` (Pydantic v2: `QueryRequest`, `Citation`, `IdentifiedFlag`, `RefusalInfo`, `RetrievalDebug` incl. `requested_mode`/`fallback_reason`, `QueryResponse`, `HealthResponse`, `SourceSummary`, `SourcesResponse` — field names exactly per demo_spec §6).
- **Commands:**
  ```powershell
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  .venv\Scripts\python.exe -m compileall rag_core
  .venv\Scripts\python.exe -c "from rag_core.config import get_settings; s=get_settings(); print(s.llm_mode, s.default_top_k, s.artifact_dir)"
  ```
- **Acceptance:** compiles; defaults match spec with no `.env` present (`mock`, `5`, `artifacts/index`).
- **Commit:** `feat: add config, API schemas, and tiered requirements`
- **Rollback:** revert commit; `.venv` is untracked.

### Stage 2 — Sample artifacts + loaders + gitignore exceptions
- **Files:** `artifacts/index/chunks.json` (~12 chunks per D5), `artifacts/index/manifest.json`; **edit** `.gitignore` adding, after the `artifacts/index/*` rule:
  ```gitignore
  !artifacts/index/chunks.json
  !artifacts/index/manifest.json
  ```
  `rag_core/loaders.py` (returns an `ArtifactState` dataclass/model with `loaded: bool`, `chunks`, `manifest`, `message`; missing dir/files → degraded state, never raises).
- **Commands:**
  ```powershell
  git check-ignore -v artifacts/index/chunks.json     # MUST match nothing (exit code 1)
  .venv\Scripts\python.exe -c "from rag_core.loaders import load_artifacts; s=load_artifacts('artifacts/index'); print(s.loaded, len(s.chunks))"
  .venv\Scripts\python.exe -c "from rag_core.loaders import load_artifacts; print(load_artifacts('nonexistent').loaded)"   # False, no exception
  ```
- **Acceptance:** both JSONs tracked by git; loader returns all sample chunks; missing-dir path degrades without exception.
- **Commit:** `feat: add sample index artifacts and degradable artifact loader`
- **Rollback:** revert (also restores `.gitignore`).

### Stage 3 — Retrieval (BM25 + RRF + dense)
- **Files:** `rag_core/retrieval.py`. Port from staging `experiment_rag_v4_source.py`: `bm25_search` (L1077, CJK-detect → jieba), `hybrid_search` RRF + priority weighting (L1119, math verbatim), `dense_search` (L1035) behind try-import; BM25 corpus built at startup from chunk texts (mirrors `build_data_v2_source.py:342`); in-memory FAISS built at startup when dense available. Public API: `retrieve(query, top_k, requested_mode, artifacts) -> RetrievalResult` carrying `effective_mode` + flags per D2.
- **Commands:**
  ```powershell
  .venv\Scripts\python.exe -c "from rag_core import retrieval, loaders; s=loaders.load_artifacts('artifacts/index'); r=retrieval.Retriever(s); out=r.retrieve('學生帳戶頻繁轉入虛擬資產交易所', 5, 'hybrid'); print(out.effective_mode, out.rrf_used, out.fallback_used, [c['chunk_id'] for c in out.chunks])"
  ```
- **Acceptance (full profile):** `effective_mode=hybrid`, `rrf_used=True`, `fallback_used=False`; a virtual-asset sample chunk ranks top-3; a priority-1.0 chunk outranks an equally matched 0.8 chunk. **Lite check (optional second venv):** same call → `effective_mode=bm25`, `fallback_used=True` with reason.
- **Commit:** `feat: port BM25, RRF hybrid, and dense retrieval with honest fallback labeling`

### Stage 4 — Pre-LLM gate
- **Files:** `rag_core/gate.py` — port `KnowledgeManifest` (L423), `TopicDetector` (L456), `GateDecision`/`GateResult` (L541–567), `pre_llm_gate` (L638); `SemanticScopeClassifier` (L570) env-gated off by default. Expose `check_scope(query, retrieved_chunks=None)` returning `{allow|refuse, reason}`.
- **Commands:**
  ```powershell
  .venv\Scripts\python.exe -c "from rag_core.gate import check_scope; print(check_scope('客戶以貿易發票虛報價格進行 TBML').decision)"     # refuse
  .venv\Scripts\python.exe -c "from rag_core.gate import check_scope; print(check_scope('學生帳戶大量轉入虛擬資產交易所').decision)"   # allow
  ```
- **Acceptance:** TBML/sanctions/tax-evasion → refuse with `reason_code`; covered topics → allow; uncertain → allow; works with lite deps.
- **Commit:** `feat: port rule-based pre-LLM scope gate`

### Stage 5 — Generation (mock default + optional live backends)
- **Files:** `rag_core/generation.py` (`RF_CATALOG`, `mock_generate()` per D4, `call_llm()` ported from L2853 reworked to plain `httpx` REST for Groq/Gemini — no provider SDKs; key missing or API error → mock fallback with flag), optionally `rag_core/prompts.py` (`SYSTEM_PROMPT` L2738, `build_user_prompt` L2807).
- **Commands:**
  ```powershell
  .venv\Scripts\python.exe -c "from rag_core.generation import mock_generate; import json; r=mock_generate(query='虛擬資產快速轉帳', chunks=[], gate_allowed=True); print(json.dumps(r, ensure_ascii=False)[:300])"
  ```
- **Acceptance:** mock produces the full response shape with zero env keys and zero network calls; empty-chunks → `unlikely` + insufficient-evidence text + empty citations; `LLM_MODE=gemini` without key → mock fallback flagged, no crash; deterministic across repeat calls.
- **Commit:** `feat: add deterministic mock generation and key-gated Groq/Gemini backends`

### Stage 6 — Pipeline + FastAPI + contract tests
- **Files:** `rag_core/pipeline.py` (validate → load artifacts → gate → retrieve → generate → assemble, mirroring `analyze_scenario` L2963; gate-refuse short-circuits with `refusal.refused=true`, `assessment="refuse"`); `api/__init__.py`; `api/main.py` (lifespan loads artifacts + retriever once; `GET /health` ok/degraded per spec; `POST /query` returns 503 `{error:"ARTIFACTS_NOT_FOUND", ...}` when degraded; `GET /sources` from manifest; debug omitted when `include_debug=false` or `ENABLE_DEBUG=false`); `tests/test_api_contract.py` (TestClient: health, query happy path field names, refusal path, missing-artifacts behavior via temp `ARTIFACT_DIR`, fallback labeling); `tests/smoke_test.py` may be stubbed here, finalized Stage 7.
- **Commands:**
  ```powershell
  .venv\Scripts\python.exe -m pytest tests -q
  .venv\Scripts\python.exe -m uvicorn api.main:app --port 8000   # in background
  Invoke-RestMethod http://localhost:8000/health
  Invoke-RestMethod -Uri http://localhost:8000/query -Method Post -ContentType "application/json" -Body (@{query="客戶短時間內多次將資金轉入虛擬資產交易所，且金額與其學生身分不符"; top_k=5; retrieval_mode="hybrid"; llm_mode="mock"; include_debug=$true} | ConvertTo-Json)
  Invoke-RestMethod http://localhost:8000/sources
  ```
- **Acceptance:** uvicorn starts; `/health` ok with artifacts / degraded without; `/query` returns `answer, assessment, identified_flags, citations, refusal, debug` with citations from sample chunks; contract tests green.
- **Commit:** `feat: wire RAG pipeline into FastAPI service with contract tests`

### Stage 7 — Docker packaging + smoke test
- **Files:** `Dockerfile` (`python:3.11-slim`; install `requirements.txt`; pre-download the embedding model in a build layer so startup is offline-fast; copy `api/ rag_core/ artifacts/`; `EXPOSE 8000`; `CMD uvicorn api.main:app --host 0.0.0.0 --port 8000`), `.dockerignore` (`.git`, `.venv`, `migration_staging`, `notebooks_archive`, `*.ipynb`, `__pycache__`, `.env`), `docker-compose.yml` (single service `rag-api`, `8000:8000`, `env_file: .env`, volumes `./artifacts:/app/artifacts` and `./data:/app/data`, python-urllib healthcheck; no DB, no extra services), `tests/smoke_test.py` (httpx vs `SMOKE_BASE_URL`: health, query contract, refusal path, /sources).
- **Commands:**
  ```powershell
  Copy-Item .env.example .env        # never commit .env
  docker compose up --build -d
  Invoke-RestMethod http://localhost:8000/health
  .venv\Scripts\python.exe tests\smoke_test.py
  docker compose down
  ```
- **Acceptance:** image builds; container healthy; smoke test green against the container; expect a large image (~2–3 GB, full ML profile — note in README).
- **Commit:** `build: add Dockerfile, docker compose, and smoke test`
- **Rollback:** revert; `docker compose down --rmi local` to clean images.

### Stage 8 — Offline indexing script
- **Files:** `indexing/build_data_v2.py` (+ `indexing/__init__.py`). Port `get_pdf_metadata` (L130), `load_pdfs` (L187), `create_chunks` (L241 — prefer a ~40-line stdlib reimplementation of the recursive split, separators `["\n\n","\n","。",".","！","!","？","?","；",";"," "]`, size 400 / overlap 80; else keep `langchain-text-splitters` as a documented indexing-only dep), `create_faiss_index` (L302), `create_bm25_index` (L342), `save_all_indexes` (L378). Argparse `--pdf-dir --out-dir`; heavy imports inside `main()`. Header docstring: requires full profile + private PDFs (not in repo); outputs land in gitignored `artifacts/index/`.
- **Commands:**
  ```powershell
  .venv\Scripts\python.exe -m compileall indexing
  .venv\Scripts\python.exe indexing\build_data_v2.py --help
  ```
- **Acceptance:** compiles; `--help` works without heavy imports; **not executed end-to-end** (no PDFs — do not fake outputs).
- **Commit:** `feat: port offline index build script from build_data_v2 notebook`

### Stage 9 — README rewrite + final sweep
- **Files:** `README.md` (rewrite, preserve the bilingual project story), `docs/migration_notes.md` (final ported/deferred ledger update), `MIGRATION_INVENTORY.md`, optional `data/sample_queries.json` (paraphrased from the scenario bank, staging L3134–3311).
- **README must contain:** project overview; **Current Implementation Status matrix** (Implemented / Demo sample only / Experimental / Planned); native Python quick start (full + lite profiles); Docker Compose quick start; API examples (`Invoke-RestMethod` + bash); artifact policy (raw PDFs not included; sample artifacts demo-only; full artifacts rebuildable via `indexing/build_data_v2.py`); known limitations; roadmap. Existing P@5/Recall@5/MRR numbers relabeled as **notebook experiment results (v4, private 226-chunk corpus)** — kept, but never presented as demo runtime claims. Keep notebook experiment narrative in a clearly separated section.
- **Commands (final acceptance checklist):**
  ```powershell
  .venv\Scripts\python.exe -m compileall .
  .venv\Scripts\python.exe -m pytest tests -q
  docker compose up --build -d
  Invoke-RestMethod http://localhost:8000/health
  .venv\Scripts\python.exe tests\smoke_test.py
  docker compose down
  git ls-files | Select-String -Pattern '\.pdf$|\.pkl$|\.faiss$|^\.env$'   # MUST be empty
  ```
- **Acceptance:** every demo_spec acceptance criterion checked and recorded in migration_notes; no private/large/secret files tracked.
- **Commit:** `docs: rewrite README with honest feature matrix and runnable quick starts`

---

## Risks

1. **`migration_staging/` absent in fresh clone/worktree** (gitignored, local-only) — highest-probability failure; Stage 0 preflight + ipynb fallback covers it.
2. **`faiss-cpu`/`torch` wheels on Windows** — heavy installs; if a wheel fails natively, Docker (Linux) remains the verified full-profile path and the lite profile keeps native dev unblocked. Never block a stage on the optional native dense check.
3. **`.gitignore` precedence** — exceptions must follow `artifacts/index/*`; Stage 2 verifies with `git check-ignore -v`.
4. **Schema drift vs demo_spec** — spec field names are the contract; only additive changes allowed.
5. **Port 8000 collisions / stale containers** — `docker compose down` between stages.
6. **Embedding model download at Docker build** requires network; if blocked, move the download to first startup and document the delay.

## Rollback

- Unit of rollback = one stage commit: `git revert <sha>` (history is pushed; never force-push, never rebase published commits).
- Stages are ordered so each leaves the repo importable and the API (once it exists) runnable; reverting stage N does not break stages < N.
- Docker cleanup: `docker compose down --rmi local`.
