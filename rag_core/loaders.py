"""Artifact loading with graceful degradation.

Loads `chunks.json` and `manifest.json` from the artifact directory.
Missing or malformed artifacts produce a degraded ``ArtifactState``
(``loaded=False`` plus a human-readable message) — never an exception,
so the API can start and report `degraded` on /health.

Reshaped from ``load_all_indexes()`` in the experiment_rag_v4 notebook;
binary FAISS/BM25 artifacts are intentionally NOT loaded from disk here —
both indexes are rebuilt in memory at startup (see rag_core/retrieval.py).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

MISSING_ARTIFACTS_MESSAGE = (
    "Artifacts not found. Please run indexing/build_data_v2.py "
    "or mount artifacts/index."
)


@dataclass
class ArtifactState:
    loaded: bool = False
    artifact_dir: str = ""
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    manifest: Optional[Dict[str, Any]] = None
    message: str = ""

    @property
    def index_version(self) -> Optional[str]:
        if self.manifest:
            return self.manifest.get("version")
        return None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_artifacts(artifact_dir: str) -> ArtifactState:
    """Load chunks.json (required) and manifest.json (optional)."""
    base = Path(artifact_dir)
    state = ArtifactState(artifact_dir=str(base))

    if not base.is_dir():
        state.message = MISSING_ARTIFACTS_MESSAGE
        return state

    chunks_path = base / "chunks.json"
    if not chunks_path.is_file():
        state.message = MISSING_ARTIFACTS_MESSAGE
        return state

    try:
        chunks = _read_json(chunks_path)
    except (json.JSONDecodeError, OSError) as exc:
        state.message = f"Failed to read chunks.json: {exc}"
        return state

    if not isinstance(chunks, list) or not chunks:
        state.message = "chunks.json is empty or not a JSON array."
        return state

    bad = [i for i, c in enumerate(chunks)
           if not isinstance(c, dict) or "chunk_id" not in c or "text" not in c]
    if bad:
        state.message = (
            f"chunks.json has {len(bad)} entries missing chunk_id/text "
            f"(first bad index: {bad[0]})."
        )
        return state

    manifest = None
    manifest_path = base / "manifest.json"
    if manifest_path.is_file():
        try:
            loaded = _read_json(manifest_path)
            if isinstance(loaded, dict):
                manifest = loaded
        except (json.JSONDecodeError, OSError):
            # manifest is optional; a broken one degrades /sources, not /query
            manifest = None

    state.loaded = True
    state.chunks = chunks
    state.manifest = manifest
    state.message = f"Loaded {len(chunks)} chunks from {chunks_path}."
    return state
