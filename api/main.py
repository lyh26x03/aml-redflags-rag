"""FastAPI application for the AML red-flag RAG demo."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from rag_core.config import Settings, get_settings
from rag_core.gate import SemanticScopeClassifier
from rag_core.loaders import ArtifactState, MISSING_ARTIFACTS_MESSAGE, load_artifacts
from rag_core.pipeline import RAGPipeline
from rag_core.retrieval import Retriever
from rag_core.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourcesResponse,
)


def create_app(
    settings: Optional[Settings] = None,
    enable_dense: Optional[bool] = None,
) -> FastAPI:
    configured = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        artifacts = load_artifacts(
            configured.resolved_artifact_dir,
            corpus_profile=configured.corpus_profile,
        )
        application.state.artifacts = artifacts
        application.state.pipeline = None
        if artifacts.loaded:
            retriever = Retriever(
                artifacts,
                enable_dense=True if enable_dense is None else enable_dense,
            )
            scope_classifier = None
            if configured.enable_semantic_gate and retriever.dense_available:
                scope_classifier = SemanticScopeClassifier(retriever.embedding_model)
            application.state.pipeline = RAGPipeline(
                settings=configured,
                retriever=retriever,
                scope_classifier=scope_classifier,
            )
        yield

    application = FastAPI(
        title="AML Red Flag RAG API",
        version="demo-sample-v1",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        artifacts: ArtifactState = application.state.artifacts
        return HealthResponse(
            status="ok" if artifacts.loaded else "degraded",
            corpus_profile=artifacts.corpus_profile,
            artifacts_loaded=artifacts.loaded,
            llm_mode=configured.llm_mode,
            model_name=configured.model_name,
            index_version=artifacts.index_version,
            chunk_count=len(artifacts.chunks),
            source_count=len(artifacts.source_summaries),
            source_names=artifacts.source_names,
            message=None if artifacts.loaded else artifacts.message,
        )

    @application.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest):
        pipeline: Optional[RAGPipeline] = application.state.pipeline
        if pipeline is None:
            artifacts: ArtifactState = application.state.artifacts
            return JSONResponse(
                status_code=503,
                content={
                    "error": "ARTIFACTS_NOT_FOUND",
                    "message": artifacts.message or MISSING_ARTIFACTS_MESSAGE,
                },
            )
        return pipeline.analyze(request)

    @application.get("/sources", response_model=SourcesResponse)
    def sources() -> SourcesResponse:
        artifacts: ArtifactState = application.state.artifacts
        return SourcesResponse(
            corpus_profile=artifacts.corpus_profile,
            index_version=artifacts.index_version,
            chunk_count=len(artifacts.chunks),
            total_chunks=len(artifacts.chunks),
            source_count=len(artifacts.source_summaries),
            source_names=artifacts.source_names,
            sources=artifacts.source_summaries,
            message=None if artifacts.loaded else artifacts.message,
        )

    return application


app = create_app()
