import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.rag.from_scratch.embeddings import TfidfVectorizer
from app.rag.versioning import (
    KnowledgeVersionStore,
    VersionStatus,
    VersioningError,
    content_fingerprint,
    ingest_index,
    normalize_content,
)


def prepared_index(text: str):
    chunks = [{"text": text, "source": "handbook.pdf"}]
    vectorizer = TfidfVectorizer()
    vectorizer.fit([text])
    return chunks, vectorizer, vectorizer.transform([text])


def make_store(tmp_path: Path) -> KnowledgeVersionStore:
    return KnowledgeVersionStore(tmp_path / "catalog.json", tmp_path / "versions")


def test_fingerprint_normalizes_unicode_whitespace_and_line_endings():
    assert normalize_content("  A\r\nB  ") == "A\nB"
    assert content_fingerprint("A\r\nB") == content_fingerprint(" A\nB ")


def test_first_ingestion_is_ready_and_reingestion_is_idempotent(tmp_path: Path):
    store = make_store(tmp_path)
    chunks, vectorizer, vectors = prepared_index("Quy chế học tập")

    first = ingest_index(store, "source://student-handbook", "Quy chế học tập", chunks, vectorizer, vectors)
    second = ingest_index(store, "source://student-handbook", "Quy chế học tập", chunks, vectorizer, vectors)

    assert first.version_id == second.version_id
    assert first.status == VersionStatus.READY
    assert len(store.history(first.source_id)) == 1


def test_different_sources_same_content_remain_distinct(tmp_path: Path):
    store = make_store(tmp_path)
    chunks, vectorizer, vectors = prepared_index("same")
    first = ingest_index(store, "source://one", "same", chunks, vectorizer, vectors)
    second = ingest_index(store, "source://two", "same", chunks, vectorizer, vectors)
    assert first.source_id != second.source_id
    assert first.content_fingerprint == second.content_fingerprint


def test_configuration_change_creates_traceable_version(tmp_path: Path):
    store = make_store(tmp_path)
    chunks, vectorizer, vectors = prepared_index("same")
    first = ingest_index(store, "handbook", "same", chunks, vectorizer, vectors, processing_config={"chunk_size": 500})
    second = ingest_index(store, "handbook", "same", chunks, vectorizer, vectors, processing_config={"chunk_size": 700})
    assert first.version_id != second.version_id
    assert second.sequence == 2


def test_failed_version_cannot_be_activated(tmp_path: Path):
    store = make_store(tmp_path)
    source = store.register_source("handbook")
    version = store.create_or_reuse_version(source, "text")
    store.transition(version.version_id, VersionStatus.FAILED, failure_reason="parser failed")
    with pytest.raises(VersioningError):
        store.activate(version.version_id)


def test_invalid_lifecycle_transition_is_rejected(tmp_path: Path):
    store = make_store(tmp_path)
    version = store.create_or_reuse_version(store.register_source("handbook"), "text")
    with pytest.raises(VersioningError, match="Invalid version transition"):
        store.transition(version.version_id, VersionStatus.ACTIVE)


def test_activation_and_rollback_keep_history_and_artifacts(tmp_path: Path):
    store = make_store(tmp_path)
    chunks, vectorizer, vectors = prepared_index("one")
    first = ingest_index(store, "handbook", "one", chunks, vectorizer, vectors)
    second = ingest_index(store, "handbook", "two", *prepared_index("two"))
    assert store.activate(first.version_id).status == VersionStatus.ACTIVE
    assert store.activate(second.version_id).status == VersionStatus.ACTIVE
    assert store.active_version(first.source_id).version_id == second.version_id
    assert store.activate(first.version_id).version_id == first.version_id
    history = store.history(first.source_id)
    assert [version.status for version in history] == [VersionStatus.ACTIVE, VersionStatus.RETIRED]
    assert (tmp_path / "versions" / first.version_id / "index.json").exists()
    assert store.load_active_index(first.source_id)[1][0][0]["knowledge_version_id"] == first.version_id


def test_concurrent_identical_ingestion_creates_one_version(tmp_path: Path):
    store = make_store(tmp_path)
    results = []
    errors = []

    def run():
        try:
            results.append(ingest_index(store, "handbook", "same", *prepared_index("same")))
        except Exception as error:  # pragma: no cover - assertion reports unexpected race
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len({version.version_id for version in results}) == 1
    assert len(store.history(results[0].source_id)) == 1
