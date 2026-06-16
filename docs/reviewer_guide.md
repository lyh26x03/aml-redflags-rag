# Reviewer Demo Pack v1

This guide provides a practical path for reviewing the repository as a runnable,
evidence-oriented AML red-flag RAG demo.

## What This Repo Demonstrates

- Migration from notebook experiments into a FastAPI service.
- Evidence-bound AML red-flag assessment with citations and debug traces.
- Deterministic mock-mode operation with no API key required.
- A committed 12-chunk sample corpus for a runnable demo.
- Opt-in multi-turn analysis via deterministic intent routing and bounded,
  local, in-process **structured conversation memory** (single-turn clients are
  unaffected). See [`conversation_memory.md`](conversation_memory.md).
- Evaluation artifacts through the API smoke eval, CQC-RAG Lite cross-query
  consistency harness, and the multi-turn memory evaluator.

## What To Run In 10 Minutes

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lite.txt
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

In another PowerShell window:

```powershell
Invoke-RestMethod http://localhost:8000/health
.venv\Scripts\python.exe scripts\run_api_smoke_eval.py
.venv\Scripts\python.exe scripts\run_cqc_eval.py --report-md eval\reports\cqc_latest.md
.venv\Scripts\python.exe scripts\run_multiturn_eval.py
.venv\Scripts\python.exe scripts\run_failure_diagnostics.py
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\run_reviewer_pack.py
```

The full profile in `requirements.txt` enables optional dense retrieval when its
embedding model is available. The lite profile runs with honest BM25 fallback.
Optional live Gemma mode uses Google AI Studio through `GEMINI_API_KEY`, but it
is not required for this 10-minute reviewer path or any automated test.

## Suggested Reviewer Path

1. Start with `README.md`.
2. Run `/health`.
3. Run one `/query` request from the README.
4. Open the generated `eval/reports/cqc_latest.md`.
5. Run Failure Diagnostics Lite and open
   `eval/reports/failure_diagnostics_latest.md`.
6. Inspect `debug.retrieved_chunk_ids` and `citations`.

## What To Inspect

- `README.md`
- `rag_core/`
- `api/main.py`
- `rag_core/intent_router.py`
- `rag_core/memory/`
- `scripts/run_api_smoke_eval.py`
- `scripts/run_cqc_eval.py`
- `scripts/run_multiturn_eval.py`
- `scripts/run_failure_diagnostics.py`
- `scripts/run_reviewer_pack.py`
- `eval/queries/cqc_scenarios_5.json`
- `eval/queries/multiturn_sessions_4.json`
- `docs/conversation_memory.md`
- `docs/cqc_rag_lite_notes.md`
- `docs/failure_diagnostics_lite.md`
- `docs/evaluation_notes.md`

## Review Order

1. `README.md`
2. `docs/demo_walkthrough.md`
3. `docs/reviewer_guide.md`
4. `docs/adr/0001-demo-scope-and-honesty.md`
5. `scripts/run_reviewer_pack.py`
6. `scripts/run_cqc_eval.py`
7. `tests/`

## What This Repo Does Not Claim

- It is not legal advice, transaction monitoring, or a production AML system.
- It is not a full CQC-RAG implementation or a model-quality benchmark.
- It does not release the full private 226-chunk corpus.
- Conversation memory is implemented, but it is local, in-process, bounded demo
  memory — not a persistent or production conversation-memory store, and not an
  unlimited transcript.
- Optional live Gemma mode requires an operator-provided Google AI Studio key
  and an available model ID; the keyless reviewer path remains the baseline.
- Docker configuration is included, but Docker has not been verified on the
  documented Windows development host.
