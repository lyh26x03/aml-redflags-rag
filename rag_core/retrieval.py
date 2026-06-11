"""Dense / BM25 / hybrid retrieval, ported from experiment_rag_v4.

Source functions (migration_staging/experiment_rag_v4_source.py):
- dense_search (L1035), bm25_search (L1077), hybrid_search w/ RRF +
  priority weighting (L1119), retrieve_contexts (L1218).
BM25 corpus construction mirrors create_bm25_index
(migration_staging/build_data_v2_source.py L342).

No binary index artifacts are loaded from disk: BM25 is rebuilt from
chunk texts at startup, and (when the dense backend is installed) an
in-memory FAISS IndexFlatIP is built by embedding the chunk texts.

Honesty contract: when a requested mode is unavailable, retrieval
degrades to the best available mode and labels it (`effective_mode`,
`fallback_used`, `fallback_reason`) — it never pretends.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from rag_core.loaders import ArtifactState

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_RRF_K = 60
DEFAULT_USE_PRIORITY_WEIGHTING = True
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- optional dense backend (full profile only) ---
try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    _DENSE_IMPORTS_OK = True
    _DENSE_IMPORT_ERROR = ""
except Exception as exc:  # lite profile or incompatible native backend
    faiss = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    _DENSE_IMPORTS_OK = False
    _DENSE_IMPORT_ERROR = str(exc)


@dataclass
class RetrievalResult:
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    requested_mode: str = "hybrid"
    effective_mode: str = "bm25"
    dense_used: bool = False
    bm25_used: bool = False
    rrf_used: bool = False
    fallback_used: bool = False
    fallback_reason: Optional[str] = None

    @property
    def chunk_ids(self) -> List[str]:
        return [c["chunk_id"] for c in self.contexts]

    @property
    def chunks(self) -> List[Dict[str, Any]]:
        """Alias used by the staged implementation plan."""
        return self.contexts


def _tokenize_query(query: str) -> List[str]:
    # verbatim from bm25_search (L1097-1101): CJK range U+4E00..U+9FFF
    if any("一" <= char <= "鿿" for char in query):
        return list(jieba.cut(query))
    return query.lower().split()


def build_bm25(chunks: List[Dict[str, Any]]):
    """Mirror of create_bm25_index (build_data_v2_source.py L342)."""
    tokenized_corpus = []
    for c in chunks:
        if c.get("language") == "zh":
            tokens = list(jieba.cut(c["text"]))
        else:
            tokens = c["text"].lower().split()
        tokenized_corpus.append(tokens)
    return BM25Okapi(tokenized_corpus), tokenized_corpus


class Retriever:
    """Holds in-memory indexes over the loaded chunks."""

    def __init__(
        self,
        artifacts: ArtifactState,
        enable_dense: bool = True,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
    ):
        if not artifacts.loaded:
            raise ValueError("Retriever requires loaded artifacts.")
        self.chunks = artifacts.chunks
        self.bm25, self.tokenized_corpus = build_bm25(self.chunks)

        self.dense_available = False
        self.dense_unavailable_reason: Optional[str] = None
        self.embedding_model = None
        self.faiss_index = None

        if not enable_dense:
            self.dense_unavailable_reason = "dense backend disabled by configuration"
        elif not _DENSE_IMPORTS_OK:
            self.dense_unavailable_reason = (
                "dense backend unavailable — install requirements.txt (full profile); "
                f"import error: {_DENSE_IMPORT_ERROR}"
            )
        else:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
                texts = [c["text"] for c in self.chunks]
                embeddings = self.embedding_model.encode(
                    texts, normalize_embeddings=True
                )
                embeddings = np.array(embeddings, dtype="float32")
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                self.faiss_index = index
                self.dense_available = True
            except Exception as exc:  # model download/load failure → degrade
                logger.warning("Dense backend init failed: %s", exc)
                self.dense_unavailable_reason = (
                    f"dense backend failed to initialize: {exc}"
                )

    # --- search primitives (ported verbatim in logic) ---

    def _dense_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        # from dense_search (L1035)
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )
        scores, indices = self.faiss_index.search(
            np.array(query_embedding, dtype="float32"), k
        )
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.chunks):
                results.append({"chunk": self.chunks[idx], "score": float(score)})
        return results

    def _bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        # from bm25_search (L1077)
        query_tokens = _tokenize_query(query)
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [
            {"chunk": self.chunks[idx], "score": float(scores[idx])}
            for idx in top_k_indices
        ]

    def _hybrid_search(
        self,
        query: str,
        k: int,
        rrf_k: int = DEFAULT_RRF_K,
        use_priority_weighting: bool = DEFAULT_USE_PRIORITY_WEIGHTING,
    ) -> List[Dict[str, Any]]:
        # from hybrid_search (L1119) — RRF math verbatim
        dense_results = self._dense_search(query, k * 2)
        bm25_results = self._bm25_search(query, k * 2)

        dense_ranks = {
            r["chunk"]["chunk_id"]: rank
            for rank, r in enumerate(dense_results, start=1)
        }
        bm25_ranks = {
            r["chunk"]["chunk_id"]: rank
            for rank, r in enumerate(bm25_results, start=1)
        }

        all_chunk_ids = set(dense_ranks) | set(bm25_ranks)
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in dense_ranks:
                score += 1 / (rrf_k + dense_ranks[chunk_id])
            if chunk_id in bm25_ranks:
                score += 1 / (rrf_k + bm25_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        chunk_lookup = {c["chunk_id"]: c for c in self.chunks}
        if use_priority_weighting:
            weighted_scores = {
                chunk_id: rrf_score
                * chunk_lookup[chunk_id].get("retrieval_priority", 1.0)
                for chunk_id, rrf_score in rrf_scores.items()
            }
        else:
            weighted_scores = rrf_scores

        sorted_chunk_ids = sorted(
            weighted_scores, key=lambda x: weighted_scores[x], reverse=True
        )[:k]

        return [
            {
                "chunk": chunk_lookup[chunk_id],
                "score": weighted_scores[chunk_id],
                "raw_rrf_score": rrf_scores[chunk_id],
                "priority_weight": chunk_lookup[chunk_id].get(
                    "retrieval_priority", 1.0
                ),
            }
            for chunk_id in sorted_chunk_ids
        ]

    # --- public API ---

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        requested_mode: str = "hybrid",
        use_priority_weighting: bool = DEFAULT_USE_PRIORITY_WEIGHTING,
    ) -> RetrievalResult:
        result = RetrievalResult(requested_mode=requested_mode)

        effective_mode = requested_mode
        if requested_mode in ("hybrid", "dense") and not self.dense_available:
            effective_mode = "bm25"
            result.fallback_used = True
            result.fallback_reason = self.dense_unavailable_reason

        if effective_mode == "dense":
            search_results = self._dense_search(query, top_k)
            result.dense_used = True
        elif effective_mode == "bm25":
            search_results = self._bm25_search(query, top_k)
            result.bm25_used = True
        else:  # hybrid
            search_results = self._hybrid_search(
                query, top_k, use_priority_weighting=use_priority_weighting
            )
            result.dense_used = True
            result.bm25_used = True
            result.rrf_used = True

        result.effective_mode = effective_mode
        # context shape from retrieve_contexts (L1262)
        result.contexts = [
            {
                "chunk_id": r["chunk"].get("chunk_id", ""),
                "source": r["chunk"].get("source", "Unknown"),
                "page": r["chunk"].get("page", 0),
                "text": r["chunk"].get("text", ""),
                "score": r["score"],
                "doc_category": r["chunk"].get("doc_category", "unknown"),
                "explanation_style": r["chunk"].get("explanation_style", "neutral"),
                "related_flags": r["chunk"].get("related_flags", []),
            }
            for r in search_results
        ]
        return result
