"""Build AML retrieval artifacts from private PDFs.

This is an offline port of ``build_data_v2_source.ipynb``. It requires the
full dependency profile and PDFs supplied by the operator; raw PDFs and the
generated FAISS/pickle artifacts are intentionally gitignored.
"""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 80
SEPARATORS = ["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " "]


def get_pdf_metadata(pdf_name: str) -> Dict[str, Any]:
    """Return the notebook's source-layer metadata for a PDF filename."""
    name = pdf_name.lower()
    metadata: Dict[str, Any] = {
        "source": "Unknown",
        "jurisdiction": "Unknown",
        "doc_type": "Unknown",
        "language": "en",
        "doc_category": "unknown",
        "retrieval_priority": 1.0,
        "explanation_style": "neutral",
    }
    if "fatf_tbm_laundering_red_flags" in name:
        metadata.update(
            source="FATF",
            jurisdiction="International",
            doc_type="red_flag",
            language="en",
            doc_category="core",
            retrieval_priority=1.0,
            explanation_style="authoritative",
        )
    elif "fatf_virtual_assets_red_flags" in name:
        metadata.update(
            source="FATF",
            jurisdiction="International",
            doc_type="red_flag",
            language="en",
            doc_category="sector_specific",
            retrieval_priority=0.9,
            explanation_style="authoritative",
        )
    elif "tw_aml_training_slides" in name:
        metadata.update(
            source="TW_Gov",
            jurisdiction="Taiwan",
            doc_type="training",
            language="zh",
            doc_category="knowledge_bridge",
            retrieval_priority=0.8,
            explanation_style="simplified",
        )
    return metadata


def load_pdfs(folder_path: str) -> List[Dict[str, Any]]:
    """Extract non-empty PDF pages and attach source metadata."""
    from pypdf import PdfReader

    folder = Path(folder_path)
    pdf_paths = sorted(path for path in folder.iterdir() if path.suffix.lower() == ".pdf")
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under {folder}.")

    pages = []
    for pdf_path in pdf_paths:
        metadata = get_pdf_metadata(pdf_path.name)
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append(
                    {
                        "pdf_name": pdf_path.name,
                        "page": page_number,
                        "text": text.strip(),
                        **metadata,
                    }
                )
    return pages


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split text with overlap, preferring the notebook's bilingual separators."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end
        if hard_end < len(text):
            for separator in SEPARATORS:
                position = text.rfind(separator, start + 1, hard_end + 1)
                if position > start:
                    end = position + len(separator)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(start + 1, end - chunk_overlap)
        start = next_start
    return chunks


def create_chunks(
    pages: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Convert extracted PDF pages into service-compatible chunks."""
    chunks = []
    for page in pages:
        for index, text in enumerate(split_text(page["text"], chunk_size, chunk_overlap)):
            chunks.append(
                {
                    "text": text,
                    "page": page["page"],
                    "chunk_id": f"{page['pdf_name']}_p{page['page']}_c{index}",
                    "source": page["source"],
                    "language": page["language"],
                    "doc_type": page["doc_type"],
                    "retrieval_priority": page.get("retrieval_priority", 1.0),
                    "doc_category": page.get("doc_category", "unknown"),
                    "explanation_style": page.get("explanation_style", "neutral"),
                }
            )
    return chunks


def create_faiss_index(chunks: List[Dict[str, Any]], embedding_model_name: str):
    """Create the notebook's normalized-vector ``faiss.IndexFlatIP``."""
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return model, index


def create_bm25_index(chunks: List[Dict[str, Any]]):
    """Build BM25 with notebook-compatible Chinese/English tokenization."""
    import jieba
    from rank_bm25 import BM25Okapi

    tokenized_corpus = []
    for chunk in chunks:
        if chunk.get("language") == "zh":
            tokens = list(jieba.cut(chunk["text"]))
        else:
            tokens = chunk["text"].lower().split()
        tokenized_corpus.append(tokens)
    return BM25Okapi(tokenized_corpus), tokenized_corpus


def _source_summaries(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts = Counter(
        (
            chunk.get("source", "Unknown"),
            chunk.get("language", "unknown"),
            chunk.get("doc_category", "unknown"),
        )
        for chunk in chunks
    )
    return [
        {
            "source_name": source,
            "language": language,
            "layer": layer,
            "chunk_count": count,
        }
        for (source, language, layer), count in sorted(counts.items())
    ]


def save_all_indexes(
    out_dir: str,
    faiss_index,
    chunks: List[Dict[str, Any]],
    bm25_index,
    tokenized_corpus: List[List[str]],
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    version_name: str,
) -> None:
    """Write local indexes plus JSON artifacts consumed by the API."""
    import faiss
    import pickle

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(output / "faiss_index.bin"))
    with (output / "chunks.json").open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)
    with (output / "bm25_index.pkl").open("wb") as file:
        pickle.dump(bm25_index, file)
    with (output / "tokenized_corpus.pkl").open("wb") as file:
        pickle.dump(tokenized_corpus, file)

    metadata = {
        "version": version_name,
        "artifact_type": "private-corpus-build",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "embedding_model": embedding_model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "stats": {
            "total_chunks": len(chunks),
            "total_vectors": int(faiss_index.ntotal),
            "vector_dimension": int(faiss_index.d),
        },
        "sources": _source_summaries(chunks),
    }
    for filename in ("metadata.json", "manifest.json"):
        with (output / filename).open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS, BM25, chunks, and manifest artifacts from private PDFs."
    )
    parser.add_argument("--pdf-dir", required=True, help="Directory containing input PDFs.")
    parser.add_argument("--out-dir", required=True, help="Artifact output directory.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--version", default="private-build-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages = load_pdfs(args.pdf_dir)
    chunks = create_chunks(pages, args.chunk_size, args.chunk_overlap)
    if not chunks:
        raise ValueError("No chunks were produced from the supplied PDFs.")
    _, faiss_index = create_faiss_index(chunks, args.embedding_model)
    bm25_index, tokenized_corpus = create_bm25_index(chunks)
    save_all_indexes(
        out_dir=args.out_dir,
        faiss_index=faiss_index,
        chunks=chunks,
        bm25_index=bm25_index,
        tokenized_corpus=tokenized_corpus,
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        version_name=args.version,
    )
    print(f"Built {len(chunks)} chunks under {Path(args.out_dir).resolve()}.")


if __name__ == "__main__":
    main()
