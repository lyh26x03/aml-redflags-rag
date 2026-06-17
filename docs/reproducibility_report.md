# Reproducibility Report

## Scope

This report records a local reproducibility check for the AML Red Flag RAG repository.

The goal of this run was to verify the deterministic reviewer baseline:

- Native Python execution
- FastAPI service startup
- Sample corpus profile
- Mock generation mode
- API contract behavior
- CQC-RAG Lite consistency checks
- Failure Diagnostics Lite report generation
- Multi-turn structured memory evaluation

This report does **not** claim AML correctness, model quality, production readiness, Docker reproducibility, or live-provider quality.

## Environment

| Field | Value |
|---|---|
| Repository path | `C:\Users\USER\Documents\aml-redflags-rag` |
| Date | 2026-06-17 |
| OS | Windows / PowerShell |
| Dependency profile | Local `.venv` |
| Corpus profile | `sample` |
| LLM mode | `mock` |
| Branch | Not captured in pasted log |
| Commit | Not captured in pasted log |
| Python version | Not captured in pasted log |

## Static Validation

| Check | Command | Result | Notes |
|---|---|---|---|
| Pytest | `.venv\Scripts\python.exe -m pytest tests -q` | PASS | `121 passed, 2 warnings in 10.51s` |

## FastAPI Runtime Validation

### `/health`

| Field | Observed value |
|---|---|
| `status` | `ok` |
| `service` | `aml-redflags-rag-api` |
| `corpus_profile` | `sample` |
| `artifacts_loaded` | `True` |
| `llm_mode` | `mock` |
| `model_name` | `mock-local` |
| `index_version` | `demo-sample-v1` |
| `chunk_count` | `12` |
| `source_count` | `3` |

Observed sources:

- `Demo Core Red Flag Indicators (EN)`
- `Demo Virtual Asset Red Flags (ZH)`
- `Demo AML Training Notes (ZH)`

Interpretation: the service started successfully using the committed sample corpus and deterministic mock generation.

### Single-turn `/query`

Test query:

```text
Funds show rapid movement through a virtual asset exchange.
```

Observed behavior:

| Field | Observed value |
|---|---|
| `assessment` | `possible` |
| `identified_flags` | `RF-02 Rapid Movement`, `RF-07 Virtual Asset Anonymity` |
| `citations_count` | `2` |
| `refusal.refused` | `False` |
| `debug.retrieval_mode` | `hybrid` |
| `debug.dense_used` | `True` |
| `debug.bm25_used` | `True` |
| `debug.rrf_used` | `True` |
| `debug.llm_mode` | `mock` |
| `debug.fallback_used` | `False` |
| `debug.intent_route` | `retrieve` |
| `debug.route_family` | `retrieve` |
| `debug.route_reason` | `single_turn_default` |
| `debug.memory_used` | `False` |

Interpretation: the default single-turn path worked as expected. The request was allowed by the gate, retrieved evidence through hybrid retrieval, generated a deterministic mock answer, returned citations, and did not use conversation memory.

### `/sources`

| Field | Observed value |
|---|---|
| `corpus_profile` | `sample` |
| `index_version` | `demo-sample-v1` |
| `chunk_count` | `12` |
| `total_chunks` | `12` |
| `source_count` | `3` |

Interpretation: the API correctly exposed the sample corpus source metadata.

## API Smoke Evaluation

Command:

```powershell
.venv\Scripts\python.exe scripts\run_api_smoke_eval.py
```

Observed result: all 8 API smoke cases passed.

| Case | Assessment | Refusal | Citations | Result |
|---|---:|---:|---:|---|
| `normal-aml-structuring` | `possible` | `false` | 1 | PASS |
| `virtual-asset-rapid-movement` | `possible` | `false` | 2 | PASS |
| `third-party-control` | `possible` | `false` | 1 | PASS |
| `insufficient-evidence` | `unlikely` | `false` | 0 | PASS |
| `out-of-scope-sanctions` | `refuse` | `true` | 0 | PASS |
| `out-of-scope-tax-evasion` | `refuse` | `true` | 0 | PASS |
| `debug-disabled` | `possible` | `false` | 2 | PASS |
| `ambiguous-valid-query` | `unlikely` | `false` | 0 | PASS |

Interpretation: the running API satisfied the smoke-test contract for normal AML queries, insufficient-evidence queries, explicit out-of-scope refusals, and debug-disabled behavior.

## CQC-RAG Lite Evaluation

Command:

```powershell
.venv\Scripts\python.exe scripts\run_cqc_eval.py
```

Observed result: all 5 CQC groups passed.

| Group | Expected behavior | Assessment consistency | Refusal consistency | Flag Jaccard avg | Citation Jaccard avg | Retrieved chunk Jaccard avg | Result |
|---|---|---:|---:|---:|---:|---:|---|
| `virtual-asset-rapid-movement` | `stable_possible` | 1.0 | 1.0 | 0.6667 | 0.6667 | 0.7222 | PASS |
| `third-party-control` | `stable_possible` | 1.0 | 1.0 | 1.0 | 0.75 | 0.8333 | PASS |
| `profile-mismatch` | `stable_possible` | 1.0 | 1.0 | 1.0 | 1.0 | 0.7143 | PASS |
| `insufficient-evidence-unlikely` | `stable_unlikely` | 1.0 | 1.0 | 1.0 | 1.0 | 0.6825 | PASS |
| `out-of-scope-refusal` | `stable_refuse` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | PASS |

Interpretation: semantically related query variants produced stable high-level behavior. Some groups showed partial variation in flags, citations, or retrieved chunks, which is expected for a consistency diagnostic and should be interpreted as retrieval/answer stability evidence, not as an AML quality benchmark.

## Failure Diagnostics Lite

Command:

```powershell
.venv\Scripts\python.exe scripts\run_failure_diagnostics.py
```

Observed report:

| Field | Value |
|---|---|
| Timestamp | `2026-06-17T06:50:29.423498+00:00` |
| Total issues | `4` |
| Severity | `INFO` only |
| Category | `OUT_OF_SCOPE_REFUSAL` |
| API smoke diagnostic issues | `0` |

Interpretation: the only diagnostic findings were four expected out-of-scope refusals from the CQC refusal group. No API smoke diagnostic issues were found. These are informational records, not failures.

## Multi-turn Evaluation

Command:

```powershell
.venv\Scripts\python.exe scripts\run_multiturn_eval.py
```

Observed result:

```text
multi-turn eval: 4 / 4 sessions passed
- mt-session-a: passed=True, turn_errors=0, memory_errors=0
- mt-session-b: passed=True, turn_errors=0, memory_errors=0
- mt-session-c: passed=True, turn_errors=0, memory_errors=0
- mt-session-d: passed=True, turn_errors=0, memory_errors=0
```

Interpretation: the four fixed multi-turn demo sessions completed without routing or memory mismatches. This validates the expected behavior of the deterministic intent router and bounded structured memory for the covered demo scenarios.

## Known Limitations of This Run

- This run used the `sample` corpus, not the `public_226` profile.
- This run used `mock` generation, not live Groq, Gemini/Gemma, or Ollama quality comparison.
- This run does not reproduce the historical private-corpus retrieval benchmark.
- This run does not claim production AML compliance capability.
- Branch, commit, and Python version were not captured in the pasted log and should be added in a future reproducibility run.
- PowerShell displayed some Traditional Chinese strings with encoding corruption in the pasted output; this appears to be terminal rendering / encoding, not necessarily API data corruption.

## Summary

The local reproducibility baseline passed.

The repository successfully demonstrated:

- a stable FastAPI service in mock mode,
- successful sample-corpus artifact loading,
- working single-turn RAG behavior with citations and debug metadata,
- passing API smoke cases,
- passing CQC-RAG Lite consistency groups,
- Failure Diagnostics Lite report generation with only expected informational refusal records,
- passing multi-turn structured memory evaluation.

This is sufficient evidence for a reviewer-facing deterministic demo baseline. The next recommended reproducibility extension is to repeat a narrower subset on `CORPUS_PROFILE=public_226`, then optionally document Docker and live-provider checks separately.
