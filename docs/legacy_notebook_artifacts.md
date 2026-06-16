# Legacy Notebook Artifacts

The handoff for the public 226-chunk AML corpus included three index artifacts
from the old notebook workflow:

- `faiss_index.bin`
- `bm25_index.pkl`
- `tokenized_corpus.pkl`

They are useful provenance references, but they are not required for default
FastAPI service startup.

## `faiss_index.bin`

`faiss_index.bin` is the serialized dense vector index produced by the notebook
after embedding each corpus chunk with
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The associated
metadata reports 226 vectors with dimension 384.

The service does not load this file by default. When dense dependencies are
available, it rebuilds an in-memory FAISS `IndexFlatIP` from `chunks.json` at
startup. If dense initialization fails or is disabled, retrieval falls back to
BM25 and reports the fallback in debug metadata.

## `bm25_index.pkl`

`bm25_index.pkl` is the notebook's serialized BM25 object. It depends on the
Python object layout and library versions used when the pickle was created.

The service does not require this pickle. It rebuilds BM25 directly from
`chunks.json` at startup using the same chunk text that the API serves.

## `tokenized_corpus.pkl`

`tokenized_corpus.pkl` is the tokenized text corpus paired with the notebook
BM25 index. English chunks are split on whitespace and Chinese chunks are
tokenized with `jieba`.

The service rebuilds this tokenized corpus alongside BM25 from `chunks.json`.
Keeping this derived pickle out of the runtime path avoids stale tokenization
and pickle compatibility issues.

## Why They Are Not Runtime Dependencies

`chunks.json` is the canonical corpus content for the FastAPI service. The
runtime indexes are derived from that file so startup remains transparent,
testable, and consistent across the `sample` and `public_226` profiles.

The legacy artifacts are not committed because they are derived binary/pickle
outputs, not source data. Requiring them would make the default service startup
more brittle without improving the corpus contract.
