# Evaluation Notes

> These are historical results from notebook experiments on a private 226-chunk corpus.
> They are **not** benchmark claims for the 12-chunk API demo shipped in this repository,
> and they are **not** re-run by the current test suite.  The purpose of this document is
> to connect every quantitative claim in the README to a traceable evidence file.

---

## Corpus and Configuration

All retrieval experiments used a private corpus built from three source PDFs:

| Source | Type | Chunks |
|---|---|---|
| `fatf_tbm_laundering_red_flags.pdf` | FATF guidance (public) | ~120 |
| `fatf_virtual_assets_red_flags.pdf` | FATF guidance (public) | ~76 |
| `tw_aml_training_slides.pdf` | TW Gov training slides | ~30 |

**Total: 226 chunks**, chunk size 400 tokens, overlap 80 tokens.  
Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (dim 384).  
Corpus provenance: [`eval/provenance/corpus_index_v2_metadata.json`](../eval/provenance/corpus_index_v2_metadata.json).

The raw PDFs and binary index artifacts (FAISS `.bin`, BM25 `.pkl`) are not committed.
The demo shipped in this repository uses a hand-written 12-chunk sample corpus and
rebuilds indexes in memory at startup.

---

## Retrieval Benchmark: 20-Query Scenario Set

### Test set

[`eval/queries/scenario_20_annotated.json`](../eval/queries/scenario_20_annotated.json)
contains 20 bilingual Chinese AML queries with chunk-level ground-truth annotations.
Each query specifies:

- `query_id` (S01–S20)
- `query` — a Chinese-language AML scenario or paraphrased question
- `category` — `場景描述` (direct scenario) or `去關鍵字/語義轉換` (keyword-removed paraphrase)
- `relevant_chunks` — one ground-truth chunk ID per query
- `expected_document` — source PDF
- `lexical_overlap_risk` — high / medium / low
- `semantic_distance_level` — 1 (direct) to 3 (high semantic shift)

Queries span three document domains: trade-based money laundering (TBML), virtual asset
red flags, and Taiwan AML regulatory training material.

### Results

[`eval/results/retrieval_scenario20_results.json`](../eval/results/retrieval_scenario20_results.json)
contains per-query scores (P@3, P@5, Recall@5, MRR, raw cosine/BM25 scores) and
aggregate metrics for all three retrieval methods.  Evaluated 2026-02-28.

| Retrieval strategy | P@3 | P@5 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Dense (FAISS) | 0.267 | 0.180 | 0.825 | 0.670 |
| BM25 | 0.083 | 0.050 | 0.250 | 0.250 |
| Hybrid (RRF, k=60) | 0.250 | 0.180 | 0.825 | 0.649 |

These are the numbers in the README "Notebook Experiment Results" table.

### Interpretation

**Dense dominates BM25 on cross-language queries.**  The corpus is primarily English
(FATF guidance documents), while the queries are Chinese.  BM25 tokenises on character
n-grams and cannot bridge the vocabulary gap; dense retrieval via the multilingual
sentence-transformer model handles it naturally.

**Hybrid RRF inherits BM25 noise.**  When BM25 retrieves wrong-language documents at
high rank, RRF fuses them into the hybrid list and suppresses correct dense results.
This is most visible in P@3 (hybrid 0.250 vs dense 0.267) and MRR (0.649 vs 0.670).
Hybrid still matches dense on Recall@5 because correct chunks are present in the dense
sub-list regardless.

**`retrieval_priority` weighting partially mitigates BM25 noise.**  Documents tagged
`doc_category = "core"` (FATF source text) receive a score multiplier.  This is why the
service exposes `retrieval_priority` as a per-chunk field in `manifest.json`.

---

## Retrieval Baseline: 2-Query Cross-Language Sanity Check

[`eval/results/retrieval_basic2_v1.json`](../eval/results/retrieval_basic2_v1.json)
is an earlier evaluation (2026-02-24) on two basic queries:

- B01: `FATF 的全稱是什麼？` (Chinese; ground truth is an English FATF chunk)
- B02: `什麼是 Trade-Based Money Laundering？` (Chinese; ground truth is an English FATF chunk)

| Method | P@3 | Recall@5 | MRR |
|---|---:|---:|---:|
| Dense | 0.333 | 1.000 | 0.417 |
| BM25 | 0.000 | 0.000 | 0.000 |
| Hybrid | 0.167 | 1.000 | 0.625 |

BM25 returns 0 % across all metrics on both queries, confirming that it retrieves
only from the Chinese `tw_aml_training_slides.pdf` chunks when the query is Chinese — even
when the correct answer is in the English FATF documents.  This is the baseline evidence
for the design decision to default to dense and to use BM25 only as a component of hybrid
RRF.

---

## README Claim → Evidence File Map

| README claim | Evidence file | Notes |
|---|---|---|
| Dense P@3 = 0.267, P@5 = 0.180, Recall@5 = 0.825, MRR = 0.670 | `eval/results/retrieval_scenario20_results.json` → `results.dense.aggregate` | 20-query set, evaluated 2026-02-28 |
| BM25 P@3 = 0.083, P@5 = 0.050, Recall@5 = 0.250, MRR = 0.250 | `eval/results/retrieval_scenario20_results.json` → `results.bm25.aggregate` | Same file |
| Hybrid P@3 = 0.250, P@5 = 0.180, Recall@5 = 0.825, MRR = 0.649 | `eval/results/retrieval_scenario20_results.json` → `results.hybrid_rrf.aggregate` | Same file |
| "multilingual dense retrieval dominated BM25 for cross-language queries" | `eval/results/retrieval_basic2_v1.json` (BM25 Recall@5 = 0.000) | Two-query cross-language sanity check |
| "RRF could inherit systematic BM25 noise" | `eval/results/retrieval_scenario20_results.json` — hybrid P@3 (0.250) < dense P@3 (0.267) | Noise visible in P@3 and MRR, not Recall@5 |
| "226-chunk corpus" | `eval/provenance/corpus_index_v2_metadata.json` → `stats.total_chunks` | Also records embedding model and vector dim |
| `retrieval_priority` weighting | `eval/provenance/corpus_index_v2_metadata.json` (config) + experiment run `v1.1` config | Priority enabled by default in all post-v1.0 runs |
| "Later notebook versions explored query rewriting, state decoupling, and intent routing" | `_incoming_drive/eval/multiturn/` (18 session logs, not yet committed) | Phase 2 of migration plan |

---

## What Is Not Covered Here

- **LLM pipeline accuracy (16-case test harness):** Separate from retrieval eval; covered
  by the experiment run logs in `_incoming_drive/experiments/runs/`.  The arc is
  0 % → 62.5 % → 87.5 % → 100 % across v1.0 / v1.1 / v2.0 runs.  Those artifacts are
  documented in [`docs/migration_inventory_eval_plan.md`](migration_inventory_eval_plan.md)
  and are candidates for Phase 2 migration.
- **Multi-turn intent routing and coreference resolution:** Documented in 18 session logs
  under `_incoming_drive/eval/multiturn/`.  The final 03/17 sessions (A/B/C) are the most
  complete.  Not yet committed; see `migration_inventory_eval_plan.md`.
- **Rerunnable eval script:** No committed script yet reproduces the retrieval benchmark.
  A `scripts/run_retrieval_eval.py` is planned for Phase 3.
