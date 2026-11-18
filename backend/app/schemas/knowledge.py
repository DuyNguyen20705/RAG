from typing import Any

from pydantic import BaseModel, Field


class KnowledgeChunk(BaseModel):
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestKnowledgeRequest(BaseModel):
    source_key: str = Field(min_length=1)
    display_name: str | None = None
    content: str = Field(min_length=1)
    chunks: list[KnowledgeChunk] = Field(min_length=1)
    processing_config: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class KnowledgeVersionResponse(BaseModel):
    version_id: str
    source_id: str
    sequence: int
    content_fingerprint: str
    processing_fingerprint: str
    status: str
    created_at: str
    ingestion_run_id: str
    parser_version: str
    normalizer_version: str
    chunking_version: str
    embedding_model: str
    processing_config: dict[str, Any]
    provenance: dict[str, Any]
    activated_at: str | None = None
    retired_at: str | None = None
    failure_reason: str | None = None


class IngestKnowledgeResponse(BaseModel):
    source_id: str
    version: KnowledgeVersionResponse
