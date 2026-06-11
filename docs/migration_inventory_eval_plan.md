# Migration Inventory: Eval Evidence Chain

> Recorded 2026-06-11, branch `repo-consolidation`.  
> Purpose: identify which `_incoming_drive/` materials should enter the repo to make the
> evaluation evidence chain traceable by a reviewer or interviewer.  
> **This document is read-only.** No files were moved, deleted, or committed as part of
> this audit.

---

## A. Executive Summary

The `_incoming_drive/` directory contains three categories of material: (1) a 20-query
bilingual AML retrieval test set with full chunk-level annotations and a complete set of
P@3 / P@5 / Recall@5 / MRR results that directly back the README benchmark table;
(2) a structured experiment log of seven pipeline runs spanning three model/config versions
with per-case pass/fail records and a clear improvement arc on the 16-case historical test harness (0 % → 62.5 % → 87.5 % → 100 %); and (3) eighteen multi-turn conversation session logs that evidence intent routing,
coreference resolution, and per-turn degradation analysis referenced in the README.

The retrieval benchmark numbers in the README (Dense P@3 = 0.267, BM25 P@3 = 0.083,
Hybrid P@3 = 0.250) are directly traceable to
`eval/scenario_queries_eval_result.json`, which should be committed first.  The annotated
test set `eval/scenario_queries_annotated.json` provides the ground-truth that makes those
numbers reproducible.  Both files are self-contained, contain no raw PDFs, and carry no
apparent copyright risk.

The most important things to **not** commit are the FAISS binary (`faiss_index.bin`), the
two pickle files (`bm25_index.pkl`, `tokenized_corpus.pkl`), and the full 226-chunk
`chunks.json` from the private corpus — the `tw_aml_training_slides.pdf` source has
unclear copyright status and the binary artifacts are large and unreadable.  The two index
directories (`index_v2/v2/` and `indices/v2/`) appear to be duplicates of each other; at
most the smaller `metadata.json` from one of them merits inclusion as provenance.

The currently weakest link in the evidence chain is the absence of a rerunnable eval
script.  The raw results exist, but there is no committed code that re-executes the
retrieval benchmark against the test set.  Adding `scripts/run_retrieval_eval.py` (Phase 2
or 3 below) would close this gap and let a reviewer regenerate the README table
independently.

---

## B. Inventory Table

> `proposed_repo_path` is left blank for DO_NOT_COMMIT or DISCARD entries.  
> All source paths are relative to `_incoming_drive/`.

| source_file | file_type | proposed_repo_path | category | supported_claim | reason | risk | action |
|---|---|---|---|---|---|---|---|
| `eval/scenario_queries_annotated.json` | JSON test set | `eval/queries/scenario_20_annotated.json` | MUST_COMMIT_EVAL | README retrieval benchmark (P@3/P@5/Recall@5/MRR for dense, BM25, hybrid) | Ground-truth chunk annotations for all 20 queries; enables benchmark reproduction | None identified; synthetic queries, no real PII | Commit in Phase 2 |
| `eval/scenario_queries_eval_result.json` | JSON benchmark results | `eval/results/retrieval_scenario20_results.json` | MUST_COMMIT_EVAL | README table row values (Dense 0.267/0.18/0.825/0.67; BM25 0.083/0.05/0.25/0.25; Hybrid 0.25/0.18/0.825/0.649) | Direct source of the three-row README table; includes per-query breakdown and raw scores | None identified | Commit in Phase 2 |
| `experiments/runs/v1.0_20260130_baseline/metadata.json` | JSON metadata | `eval/results/runs/v1.0_20260130_baseline/metadata.json` | MUST_COMMIT_EVAL | Failure analysis: API-credit depletion caused 0 % accuracy in earliest run | Explains why v1.0 baseline shows 0 % accuracy — a documented failure, not a broken pipeline | None | Commit in Phase 2 |
| `experiments/runs/v1.0_20260130_baseline/metrics.json` | JSON metrics | `eval/results/runs/v1.0_20260130_baseline/metrics.json` | MUST_COMMIT_EVAL | Same as above | Summary: 0/16 passed | None | Commit in Phase 2 |
| `experiments/runs/v1.0_20260130_baseline/results.json` | JSON results | `eval/results/runs/v1.0_20260130_baseline/results.json` | MUST_COMMIT_EVAL | Same as above | Case-level error records (all 16 = API credit error); primary failure evidence | None; no real scenario text, only error strings | Commit in Phase 2 |
| `experiments/runs/v1.1_20260130_fix_priority_enabled/metadata.json` | JSON metadata | `eval/results/runs/v1.1_20260130_fix_priority_enabled/metadata.json` | MUST_COMMIT_EVAL | Priority-weighting fix improved accuracy from 0 % to 62.5 % on same date | Immediate A/B comparison of config change on identical infrastructure | None | Commit in Phase 2 |
| `experiments/runs/v1.1_20260130_fix_priority_enabled/metrics.json` | JSON metrics | `eval/results/runs/v1.1_20260130_fix_priority_enabled/metrics.json` | MUST_COMMIT_EVAL | Same as above | Summary: 10/16 passed | None | Commit in Phase 2 |
| `experiments/runs/v1.1_20260130_fix_priority_enabled/results.json` | JSON results | `eval/results/runs/v1.1_20260130_fix_priority_enabled/results.json` | MUST_COMMIT_EVAL | Same as above; also shows which case types (confirmed/possible/refuse) failed | Case-level pass/fail with expected vs actual | None | Commit in Phase 2 |
| `experiments/runs/v1.1_20260130_fix_priority_enabled/config.json` | JSON config | `eval/results/runs/v1.1_20260130_fix_priority_enabled/config.json` | MUST_COMMIT_EVAL | `retrieval_priority` weighting design decision | Shows `use_priority_weighting: true` vs baseline config | None | Commit in Phase 2 |
| `experiments/runs/v2.0_20260212_baseline_expand_KS/metadata.json` | JSON metadata | `eval/results/runs/v2.0_20260212_baseline_expand_KS/metadata.json` | MUST_COMMIT_EVAL | Claim: 100 % accuracy achieved after expanding knowledge scope | Milestone run; direct evidence of best result | None | Commit in Phase 2 |
| `experiments/runs/v2.0_20260212_baseline_expand_KS/metrics.json` | JSON metrics | `eval/results/runs/v2.0_20260212_baseline_expand_KS/metrics.json` | MUST_COMMIT_EVAL | Same as above | 16/16 passed, 1 gate_refused | None | Commit in Phase 2 |
| `experiments/runs/v2.0_20260212_baseline_expand_KS/results.json` | JSON results | `eval/results/runs/v2.0_20260212_baseline_expand_KS/results.json` | MUST_COMMIT_EVAL | Same as above; shows gate decision routing per case | Full case-level gate + outcome record | None | Commit in Phase 2 |
| `experiments/runs/v2.0_20260212_baseline_expand_KS/config.json` | JSON config | `eval/results/runs/v2.0_20260212_baseline_expand_KS/config.json` | MUST_COMMIT_EVAL | Pipeline config for the 100 % run | Config provenance for the milestone | None | Commit in Phase 2 |
| `experiments/runs/v2.0_20260210_baseline_L8b/metrics.json` | JSON metrics | `eval/results/runs/v2.0_20260210_baseline_L8b/metrics.json` | MUST_COMMIT_EVAL | Gate refused 3 cases out of 16; 87.5 % accuracy with L8b model | Shows rule-based gate functioning (3 refused) at v2.0 baseline | None | Commit in Phase 2 |
| `experiments/runs/v2.0_20260210_baseline_L8b/metadata.json` | JSON metadata | `eval/results/runs/v2.0_20260210_baseline_L8b/metadata.json` | MUST_COMMIT_EVAL | Same as above | Shows model = llama-3.1-8b and accuracy progression to 87.5 % | None | Commit in Phase 2 |
| `eval/multiturn/session_A_20260317_llama_3.1_8b_instant_llm.json` | JSON session log | `eval/multiturn/session_A_20260317_llm.json` | MUST_COMMIT_EVAL | Multi-turn intent routing (RETRIEVE vs ANSWER_FROM_HISTORY); coreference resolution; turn-level degradation | Final session-A iteration with LLM intent mode, including Turn 3 degradation analysis | No real PII; synthetic scenarios | Commit in Phase 2 |
| `eval/multiturn/session_B_20260317_llama_3.1_8b_instant_llm.json` | JSON session log | `eval/multiturn/session_B_20260317_llm.json` | MUST_COMMIT_EVAL | Same claim: multi-turn routing on session-B scenario type | Completes the A/B/C scenario set | Same as above | Commit in Phase 2 |
| `eval/multiturn/session_C_20260317_llama_3.1_8b_instant_llm.json` | JSON session log | `eval/multiturn/session_C_20260317_llm.json` | MUST_COMMIT_EVAL | Same claim: session-C covers a distinct multi-turn scenario type | Three-scenario coverage with intent classification | Same as above | Commit in Phase 2 |
| `eval/retrieval_eval_v1.json` | JSON eval | `eval/results/retrieval_basic2_v1.json` | MUST_COMMIT_EVAL | BM25 cross-language failure: 0 % P@3/P@5/Recall@5 on 2 basic queries; dense succeeds | Earliest retrieval evidence; documents that BM25 returns wrong-language docs on cross-language queries | None | Commit in Phase 2 |
| `eval/scenario_queries.json` | JSON queries (unannotated) | — | MAYBE_ARCHIVE | — | Un-annotated version; superseded by `scenario_queries_annotated.json`. No `relevant_chunks` field is populated | None | Archive in `notebooks_archive/` or omit; not needed if annotated version is committed |
| `eval/query_relevance_v1.json` | JSON queries (partial annotation) | — | MAYBE_ARCHIVE | — | Intermediate version: has `expected_document` but no chunk IDs. Superseded by annotated version | None | Archive or omit |
| `experiments/runs/v1.0_20260202_baseline/` (all files) | JSON run | — | MAYBE_ARCHIVE | — | Repeat of v1.0 on a second date (02-02); shows 93.75 % with no priority bug — but the 01-30 pair already tells the story. Adds a date-consistency data point | None | Keep locally; may commit if wanted as a second date checkpoint |
| `experiments/runs/v1.1_20260202_fix_priority_enabled/` (all files) | JSON run | — | MAYBE_ARCHIVE | — | Repeat of v1.1 on 02-02; same 93.75 % as v1.0_0202, showing priority weighting parity | None | Same as above |
| `experiments/runs/v2.0_20260212_baseline_gate_gray_area/` (all files) | JSON run | — | MAYBE_ARCHIVE | — | Variant of v2.0 testing gate gray-area; same 87.5 % metrics as L8b baseline. Interesting only if gate-threshold tuning is documented elsewhere | None | Archive; commit only if gate gray-area analysis is added to docs |
| `experiments/runs/*/cases/` (all 7 × 16 = 112 individual case files) | JSON case records | — | MAYBE_ARCHIVE | — | Case-level records contain full scenario text + LLM responses; rich for debugging but verbose. The `results.json` in each run summarizes the same pass/fail data | No PII; synthetic scenarios | Commit `cases/` only for the three key runs (v1.0, v1.1, v2.0_expand_KS) in Phase 2; omit the rest |
| `experiments/runs/v2.0_20260210_baseline_L8b/單題情境測試.docx` | Word document | — | MAYBE_ARCHIVE | — | Appears to be a test scenario worksheet (8.5 KB); format unclear without parsing. If it contains the 16 test case definitions, it is the original test-set source document | Possible copyright on scenario wording if sourced externally | Review content before committing; convert to markdown if committing |
| `index_v2/v2/metadata.json` | JSON | `eval/provenance/corpus_index_v2_metadata.json` | MUST_COMMIT_DOCS | "Private 226-chunk corpus" claim in README; model = paraphrase-multilingual-MiniLM-L12-v2, dim = 384 | Establishes that benchmark ran on 226-chunk corpus with this embedding model | None; no text content | Commit in Phase 1 |
| `index_v2/v2/chunks.json` | JSON corpus text | — | DO_NOT_COMMIT_PRIVATE | — | Contains 226 extracted text chunks from `fatf_tbm_laundering_red_flags.pdf`, `fatf_virtual_assets_red_flags.pdf`, and `tw_aml_training_slides.pdf`. The third source (TW Gov training slides) has unclear copyright status | Copyright risk for `tw_aml_training_slides.pdf` content; also large for demo repo | Do not commit; use demo `artifacts/index/chunks.json` (12-chunk hand-written sample) |
| `index_v2/v2/bm25_index.pkl` | Binary pickle | — | DO_NOT_COMMIT_PRIVATE | — | Large binary; not human-readable; pickle is a security vector if untrusted | Binary + security risk | Never commit |
| `index_v2/v2/faiss_index.bin` | Binary FAISS | — | DO_NOT_COMMIT_PRIVATE | — | Large binary; embeds full 226-chunk vector space | Binary + large | Never commit |
| `index_v2/v2/tokenized_corpus.pkl` | Binary pickle | — | DO_NOT_COMMIT_PRIVATE | — | Same concerns as bm25_index.pkl | Binary + security risk | Never commit |
| `indices/v2/` (all 5 files) | Mixed | — | DO_NOT_COMMIT_PRIVATE | — | Appears to be a near-duplicate of `index_v2/v2/` (same stats, slightly earlier `created_at`). Same copyright and binary concerns apply | Same as index_v2 | Never commit; discard duplicate if confirmed identical |
| `eval/multiturn/session_A_20260313_llama_3.1_8b_instant.json` (and B, C) | JSON session log | — | MAYBE_ARCHIVE | — | Earliest multi-turn sessions; note shows "Turn 3 DEGRADED (coreference across two turns failed, structuring lost)" — historically valuable failure analysis | None | Archive; not needed in main evidence chain once 03/17 sessions are committed |
| `eval/multiturn/session_A_20260314_llama_3.1_8b_instant.json` (and B, C) | JSON session log | — | MAYBE_ARCHIVE | — | Intermediate iteration; improvements over 03/13 but superseded by 03/17 | None | Archive |
| `eval/multiturn/session_A_20260315_llama_3.1_8b_instant.json` (and B, C) | JSON session log | — | MAYBE_ARCHIVE | — | Same as 03/14 — intermediate, superseded by final 03/17 versions | None | Archive |
| `eval/multiturn/session_*_A_20260317_*_offline.json` (3 files) | JSON session log | — | MAYBE_ARCHIVE | — | 03/17 offline-intent variant; useful for ablation (offline vs LLM intent mode) but not the primary narrative | None | Commit only if ablation comparison is documented |
| `eval/multiturn/session_*_B_20260317_*_llm.json` (3 files) | JSON session log | — | DISCARD | — | Apparent variant B of the LLM intent sessions; same date and model as the primary 03/17 files. Check if content differs from `session_*_20260317_*_llm.json`; if duplicate, discard | None | Verify before discarding |

---

## C. Recommended Repo Structure

The following structure is proposed **only for eval evidence artifacts**. Existing
`api/`, `rag_core/`, `tests/`, `docs/`, and `notebooks_archive/` paths are unchanged.

```
eval/
  queries/
    scenario_20_annotated.json        ← 20-query AML test set with chunk-level ground truth
  results/
    retrieval_basic2_v1.json          ← 2-query cross-language baseline (BM25 failure evidence)
    retrieval_scenario20_results.json ← Source of README P@3/P@5/Recall@5/MRR table
    runs/
      v1.0_20260130_baseline/
        metadata.json
        metrics.json                  ← 0 % accuracy (API credit failure)
        results.json                  ← All 16 = error records
        config.json
      v1.1_20260130_fix_priority_enabled/
        metadata.json
        metrics.json                  ← 62.5 % (priority_weighting fix)
        results.json
        config.json
      v2.0_20260210_baseline_L8b/
        metadata.json
        metrics.json                  ← 87.5 %, gate_refused=3
      v2.0_20260212_baseline_expand_KS/
        metadata.json
        metrics.json                  ← 100 % accuracy milestone
        results.json
        config.json
  multiturn/
    session_A_20260317_llm.json       ← Final A/B/C multi-turn sessions
    session_B_20260317_llm.json
    session_C_20260317_llm.json
  provenance/
    corpus_index_v2_metadata.json     ← 226-chunk corpus config (model, dim, chunk_size)

scripts/
  run_retrieval_eval.py               ← (Phase 3: new file to write; re-runs retrieval
                                          benchmark on annotated test set and writes results)
docs/
  evaluation_notes.md                 ← (Phase 1: new file to write; narrative connecting
                                          experiment log to README claims)
notebooks_archive/
  (existing .gitkeep, source notebooks — no change)
  eval_archive/                       ← (optional) store superseded eval files here
    scenario_queries_unannotated.json
    query_relevance_v1.json
    multiturn_sessions_intermediate/  ← 03/13, 03/14, 03/15 sessions
```

---

## D. Do Not Commit List

| File type | Examples | Reason |
|---|---|---|
| FAISS binary index | `faiss_index.bin` | Large binary; not human-readable; embeds private corpus vectors |
| BM25 pickle | `bm25_index.pkl` | Binary pickle; security risk if deserialized without trust; not reproducible by diff |
| Tokenized corpus pickle | `tokenized_corpus.pkl` | Same as above |
| Full 226-chunk corpus JSON | `chunks.json` (from index_v2 or indices) | Contains extracted text from `tw_aml_training_slides.pdf` whose copyright status is unclear; also 139 KB of prose that does not constitute a hand-curated demo sample |
| Raw source PDFs | `fatf_tbm_laundering_red_flags.pdf`, `fatf_virtual_assets_red_flags.pdf`, `tw_aml_training_slides.pdf` (not in drive, referenced only) | Not stored in `_incoming_drive/` but must not be added if they appear; PDFs are large and the TW-gov slide may have copyright restrictions |
| `.env`, API keys | Any `.env`, key files | Secrets; already in `.gitignore` |
| Duplicate index directory | `indices/v2/` (appears identical to `index_v2/v2/`) | Redundant; committing only increases confusion |
| Individual case JSON files (non-key runs) | `experiments/runs/v1.0_20260202_baseline/cases/*.json` | Verbose; pass/fail summary in `results.json` is sufficient; 96 of 112 case files add no incremental claim |

---

## E. Next Step Plan

### Phase 1 — Safe documentation only (no new code, no test re-runs)

1. Write `docs/evaluation_notes.md`:
   - Explain the 16-case LLM pipeline test harness (what scenarios it covers, which model,
     what the accuracy metric means).
   - Explain the 20-query retrieval benchmark (how queries were annotated, what P@3 means
     in this context, why dense beats BM25 on cross-language queries).
   - Note the multi-turn degradation findings (coreference resolution, intent routing
     design decisions) with forward pointers to the session logs.
   - Establish a table linking each README claim to its evidence file.
2. Commit `eval/provenance/corpus_index_v2_metadata.json` (0.3 KB; confirms the 226-chunk
   corpus config used for all benchmarks).

### Phase 2 — Move selected eval artifacts

Commit the following in a single PR named `feat: add eval evidence chain`:

- `eval/queries/scenario_20_annotated.json`
- `eval/results/retrieval_basic2_v1.json`
- `eval/results/retrieval_scenario20_results.json`
- `eval/results/runs/v1.0_20260130_baseline/{metadata,metrics,results,config}.json`
- `eval/results/runs/v1.1_20260130_fix_priority_enabled/{metadata,metrics,results,config}.json`
- `eval/results/runs/v2.0_20260210_baseline_L8b/{metadata,metrics}.json`
- `eval/results/runs/v2.0_20260212_baseline_expand_KS/{metadata,metrics,results,config}.json`
- `eval/multiturn/session_A/B/C_20260317_llm.json` (3 files)

Before committing the cases/ directories from the three key runs, review `單題情境測試.docx`
to confirm the 16 scenarios are original and carry no external copyright.

### Phase 3 — Build eval runner / rerun tests

1. Write `scripts/run_retrieval_eval.py`:
   - Takes `eval/queries/scenario_20_annotated.json` and the local retrieval artifacts or retrieval module as input.
   - Re-runs retrieval-level evaluation for dense, BM25, and hybrid modes.
   - Outputs a results file in the same schema as `retrieval_scenario20_results.json`.
   - Documents the command in README under a "Reproducing the retrieval benchmark" section.

2. Write a separate `scripts/run_api_smoke_eval.py`:
   - Calls the running FastAPI `/query` endpoint with a small sample query set.
   - Verifies API response contract fields such as `answer`, `assessment`, `refusal`, `citations`, and `debug`.
   - Does not claim to reproduce P@K / MRR retrieval benchmark numbers.

3. Add smoke-level pytest coverage:
   - For retrieval eval, test only a tiny fixture or mocked artifact path so CI does not require private corpus files.
   - For API smoke eval, test the mock API response contract separately.
   - Do not assert private-corpus Dense Recall@5 in public CI unless the required artifacts are committed or generated in the test.

4. Optionally: add `eval/multiturn/` session replay tooling once the multi-turn API contract is stabilised.

---

*End of inventory. No files were moved, deleted, or committed during this audit.*
