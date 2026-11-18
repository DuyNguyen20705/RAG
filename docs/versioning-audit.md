# Knowledge ingestion versioning audit

## Existing behavior

- `app.rag.loader` discovers one PDF and returns page text.
- `app.rag.preprocessing` normalizes page text, creates chunks, and writes JSONL.
- `app.rag.from_scratch.pipeline.build_index` builds a TF-IDF index in memory.
- `app.rag.from_scratch.index_store` persists one global `data/processed/tfidf_index.json`.
- `app.routes.chat` caches that global index for retrieval; there is no source, version,
  ingestion-run, publication, or rollback concept.

## Missing behavior and risks

- A filename is used as provenance and there is no stable source identity.
- No deterministic content fingerprint or normalization contract exists.
- Repeated/concurrent ingestion can overwrite the same global index and cannot be
  distinguished from a failed or partial run.
- Retrieval has no eligibility or active-version filter.
- There are no lifecycle constraints, activation history, rollback semantics, or API
  contracts for knowledge versions.
- The existing index format must remain readable for backward compatibility.

## Proposed model

`Source` (stable identity) has many `KnowledgeVersion` records. A version records the
normalized content fingerprint, processing configuration, parser/normalizer/chunking
versions, and its lifecycle status. An `IngestionRun` identifies one execution and is
kept separate from the source version. Versioned index artifacts are stored under the
version id. Only one successfully processed version per source is active; activation
updates the catalog atomically and retrieval loads only active artifacts.

The catalog is JSON-backed because this repository has no database layer. Updates use
an exclusive lock file plus write-to-temporary-and-replace. The uniqueness key is
`source_id + content_fingerprint + processing_fingerprint`, so identical retries are
idempotent while parser/chunker configuration changes remain traceable.

The design follows the useful boundary from VersionRAG (Huwiler, Stockinger, and
Fürst, arXiv:2510.08109): document evolution must be represented explicitly and
retrieval must apply version validity before semantic ranking. This implementation
adopts the version sequence and eligibility boundary, while intentionally not adding
the paper's query-intent graph because this repository's scope is ingestion
versioning and its existing retriever is algorithm-independent.

## Implementation plan

1. Add deterministic normalization/fingerprinting and explicit lifecycle transitions.
2. Add a locked, atomic catalog with source/version/run/activation provenance.
3. Add version-aware ingestion, activation, rollback, and retrieval helpers while
   preserving the legacy index loader.
4. Expose source/version management endpoints following existing FastAPI conventions.
5. Add deterministic unit/integration tests and document recovery/compatibility limits.
