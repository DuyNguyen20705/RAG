"""Version-aware knowledge ingestion and publication endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.rag.from_scratch.embeddings import TfidfVectorizer
from app.rag.versioning import KnowledgeVersionStore, VersioningError, ingest_index
from app.schemas.knowledge import IngestKnowledgeRequest, IngestKnowledgeResponse, KnowledgeVersionResponse
from app.routes.chat import require_api_token


router = APIRouter(dependencies=[Depends(require_api_token)])
store = KnowledgeVersionStore()


def _response(version) -> KnowledgeVersionResponse:
    return KnowledgeVersionResponse(**version.__dict__)


@router.post("/ingest", response_model=IngestKnowledgeResponse, status_code=status.HTTP_201_CREATED)
def ingest_knowledge(request: IngestKnowledgeRequest) -> IngestKnowledgeResponse:
    """Create or reuse a complete version; publication is an explicit operation."""
    try:
        chunks = [{"text": chunk.text, **chunk.metadata} for chunk in request.chunks]
        vectorizer = TfidfVectorizer()
        vectorizer.fit([chunk["text"] for chunk in chunks])
        vectors = vectorizer.transform([chunk["text"] for chunk in chunks])
        source = store.register_source(request.source_key, request.display_name)
        version = ingest_index(
            store, request.source_key, request.content, chunks, vectorizer, vectors,
            processing_config=request.processing_config,
            provenance=request.provenance,
        )
        return IngestKnowledgeResponse(source_id=source.source_id, version=_response(version))
    except (ValueError, TypeError, VersioningError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/sources/{source_id}/versions", response_model=list[KnowledgeVersionResponse])
def list_versions(source_id: str) -> list[KnowledgeVersionResponse]:
    return [_response(version) for version in store.history(source_id)]


@router.post("/versions/{version_id}/activate", response_model=KnowledgeVersionResponse)
def activate_version(version_id: str) -> KnowledgeVersionResponse:
    try:
        version = store.activate(version_id)
        from app.routes.chat import get_bm25_index, get_rag_index
        get_rag_index.cache_clear()
        get_bm25_index.cache_clear()
        return _response(version)
    except (ValueError, VersioningError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/versions/{version_id}/rollback", response_model=KnowledgeVersionResponse)
def rollback_version(version_id: str) -> KnowledgeVersionResponse:
    return activate_version(version_id)
