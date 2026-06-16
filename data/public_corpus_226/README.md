# Public 226-Chunk AML Corpus

This directory contains the public AML corpus profile used when the service is
started with `CORPUS_PROFILE=public_226`.

## Files

- `chunks.json`: canonical service corpus, imported from the legacy notebook
  handoff. It contains 226 chunk objects.
- `source_manifest.json`: service-facing manifest transformed from the handoff
  `metadata.json` plus source summaries derived from `chunks.json`.
- `sources/`: committed public PDFs referenced by the corpus.

## Chunk Schema

Each imported chunk contains:

- `chunk_id`
- `text`
- `page`
- `source`
- `language`
- `doc_type`
- `doc_category`
- `explanation_style`
- `retrieval_priority`

The handoff chunks do not contain a `layer` field. The manifest maps source
layers from each source's `doc_category`.

## Source Summary

| Source file | Chunks | Language | Layer | Priority |
|---|---:|---|---|---:|
| `fatf_tbm_laundering_red_flags.pdf` | 51 | en | core | 1.0 |
| `fatf_virtual_assets_red_flags.pdf` | 165 | en | sector_specific | 0.9 |
| `tw_aml_training_slides.pdf` | 10 | zh | knowledge_bridge | 0.8 |

## Notebook Weighting

The legacy notebook used `retrieval_priority` as a per-chunk multiplier during
hybrid RRF retrieval. The service preserves those values in `chunks.json`; the
current retriever already applies `retrieval_priority` when hybrid retrieval is
available.

Legacy FAISS/BM25/tokenized pickle artifacts were not copied here. The service
rebuilds BM25 from `chunks.json` and optionally builds an in-memory dense FAISS
index at startup when the dense dependencies and embedding model are available.
