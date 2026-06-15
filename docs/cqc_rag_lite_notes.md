# CQC-RAG Lite Notes

This note explains how this repository translates the CQC-RAG idea into a small, testable evaluation harness for the AML Red Flag RAG demo.

This is not a full implementation of the CQC-RAG paper.

## 1. Why this exists

The AML Red Flag RAG demo is an evidence-oriented single-turn FastAPI service. For this type of system, one practical reliability risk is query sensitivity:

> If the same AML scenario is phrased in several semantically equivalent ways, the service should not produce unstable assessments, unrelated red flags, or radically different evidence traces.

CQC-RAG Lite turns that risk into a small regression check.

Instead of asking only “does one query produce a plausible answer?”, this evaluation asks:

> Across equivalent query variants, does the system remain consistent enough in assessment, identified flags, citations, and retrieved chunk IDs?

This is useful for detecting fragile retrieval behavior, brittle keyword dependence, and evidence instability in a demo-sized RAG system.

## 2. Relation to the original CQC-RAG idea

The original CQC-RAG framework is based on cross-query consistency: semantically equivalent but syntactically diverse query formulations can expose whether an answer is stably supported across multiple query-conditioned evidence views.

At a high level, the full method includes:

- generating meaning-preserving query rewrites;
- reranking a shared document pool for each query view;
- producing answer-evidence pairs under each query-conditioned context;
- estimating answer confidence across query views;
- selecting the answer with high mean confidence and low variance.

This repository does not implement that full pipeline.

## 3. What this repository implements

This repository implements a lightweight evaluation harness only.

The current CQC-RAG Lite flow is:

```text
fixed synthetic AML scenario group
        |
        v
4 semantically equivalent query variants
        |
        v
existing FastAPI /query endpoint
        |
        v
compare response stability across variants
        |
        v
write JSONL result + summary
```


The evaluator checks whether each scenario group remains stable across equivalent variants.

Current evaluation inputs:

```text
eval/queries/cqc_scenarios_5.json
```

Current runner:

```text
scripts/run_cqc_eval.py
```

Default runtime output:

```text
eval/results/cqc_latest.jsonl
```

The runtime result file is intentionally ignored by git because it is a generated local evaluation artifact.

## 4. What is checked

CQC-RAG Lite focuses on response consistency signals that already exist in the current API response.

The evaluator may inspect:

* `assessment`
* `identified_flags`
* `citations`
* `debug.retrieved_chunk_ids`
* refusal behavior, when relevant

The goal is not to prove answer correctness. The goal is to detect when semantically equivalent inputs cause suspiciously different system behavior.

## 5. What this does not claim

CQC-RAG Lite is not:

* a full reproduction of the CQC-RAG paper;
* a model-quality benchmark;
* a historical benchmark over the private 226-chunk corpus;
* a replacement for retrieval metrics such as P@K, Recall@K, or MRR;
* a logits-based confidence estimator;
* an answer-selection algorithm;
* a query rewriting system;
* a reranker implementation;
* a multi-turn routing evaluation.

It does not modify the main RAG flow.

It does not change:

* retrieval logic;
* generation logic;
* gate logic;
* pipeline behavior;
* historical benchmark results;
* notebook-only multi-turn experiments.

## 6. Why this is appropriate for the current repo

The current repository is a runnable single-turn FastAPI demo with deterministic mock mode. That makes it suitable for a small consistency regression harness.

For this stage, the most useful engineering translation of CQC-RAG is not “implement the entire paper.” The useful slice is:

> Use controlled query variants to test whether the existing service behaves stably under equivalent phrasing.

This fits the project because AML red-flag analysis is evidence-sensitive. A reviewer should be able to inspect not only the final answer, but also whether the same scenario retrieves comparable evidence and produces comparable structured judgments.

## 7. How to run

Start the FastAPI service in mock mode, then run:

```powershell
.venv\Scripts\python.exe scripts\run_cqc_eval.py
```

The runner writes a JSONL result file under:

```text
eval/results/cqc_latest.jsonl
```

The console summary reports how many scenario groups passed the consistency checks.

## 8. How to interpret failures

A failed CQC group does not automatically mean the system is wrong.

It means at least one equivalent query variant caused a meaningful change in the response trace.

Possible causes include:

* retrieval depends too strongly on surface keywords;
* BM25 and dense retrieval disagree under paraphrase;
* the mock generator is over-sensitive to retrieved chunk metadata;
* citations shift even when assessment and flags remain stable;
* the scenario group itself contains variants that are not truly equivalent.

A failure should be treated as a debugging entry point, not as a final quality score.

## 9. Future extensions

Possible next steps, still outside the current main service flow:

* add per-group diagnostic summaries;
* track citation overlap and retrieved-chunk overlap separately;
* compare BM25, dense, and hybrid consistency;
* add a small report table for PR review;
* add optional generated paraphrases, clearly separated from the fixed synthetic test set;
* experiment with a true reranking layer in a separate branch;
* add confidence-style scoring only if the model/provider exposes reliable token-level or verifier scores.

These should remain evaluation-layer changes unless the project explicitly moves beyond the current single-turn demo scope.
