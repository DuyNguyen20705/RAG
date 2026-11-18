"""File-backed knowledge version registry and ingestion lifecycle.

The registry deliberately keeps source identity, source version, ingestion run, and
derived index artifacts separate.  It is suitable for the single-node deployment
used by this project; a multi-node deployment should move the same invariants to a
transactional database.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator
import unicodedata

from .from_scratch.index_store import load_index, save_index


BASE_DIR = Path(__file__).resolve().parents[3]
VERSIONING_DIR = BASE_DIR / "data" / "versions"
CATALOG_PATH = VERSIONING_DIR / "catalog.json"
LOCK_PATH = VERSIONING_DIR / "catalog.lock"
CATALOG_SCHEMA_VERSION = 1
NORMALIZER_VERSION = "nfc-whitespace-v1"
PARSER_VERSION = "pdf-text-v1"
CHUNKING_VERSION = "chunker-v1"


class VersionStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"


_TRANSITIONS: dict[VersionStatus, set[VersionStatus]] = {
    VersionStatus.PROCESSING: {VersionStatus.READY, VersionStatus.FAILED},
    VersionStatus.READY: {VersionStatus.ACTIVE, VersionStatus.FAILED},
    VersionStatus.ACTIVE: {VersionStatus.RETIRED},
    VersionStatus.RETIRED: {VersionStatus.ACTIVE},
    VersionStatus.FAILED: set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_content(content: str) -> str:
    """Return the canonical text used for fingerprints.

    NFC is applied, line endings become LF, trailing whitespace is removed per line,
    runs of horizontal whitespace collapse to one space, and surrounding blank lines
    are removed. Paragraph boundaries are retained.
    """
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    value = unicodedata.normalize("NFC", content).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in value.split("\n")]
    return "\n".join(lines).strip()


def content_fingerprint(content: str) -> str:
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


def processing_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class Source:
    source_id: str
    source_key: str
    display_name: str
    created_at: str


@dataclass
class KnowledgeVersion:
    version_id: str
    source_id: str
    sequence: int
    content_fingerprint: str
    processing_fingerprint: str
    status: str
    created_at: str
    ingestion_run_id: str
    parser_version: str = PARSER_VERSION
    normalizer_version: str = NORMALIZER_VERSION
    chunking_version: str = CHUNKING_VERSION
    embedding_model: str = "tfidf"
    processing_config: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    activated_at: str | None = None
    retired_at: str | None = None
    failure_reason: str | None = None


@dataclass
class IngestionRun:
    run_id: str
    source_id: str
    version_id: str
    status: str
    started_at: str
    completed_at: str | None = None
    error: str | None = None


def _empty_catalog() -> dict[str, Any]:
    return {"schema_version": CATALOG_SCHEMA_VERSION, "sources": {}, "versions": {}, "runs": {}, "activation_history": []}


@contextmanager
def _catalog_lock(lock_path: Path = LOCK_PATH, timeout: float = 10.0) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    handle: int | None = None
    while handle is None:
        try:
            handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for knowledge catalog lock")
            time.sleep(0.01)
    try:
        yield
    finally:
        os.close(handle)
        lock_path.unlink(missing_ok=True)


def _read_catalog() -> dict[str, Any]:
    if not CATALOG_PATH.exists():
        return _empty_catalog()
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if data.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError("Unsupported knowledge catalog schema version")
    return data


def _write_catalog(data: dict[str, Any]) -> None:
    VERSIONING_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CATALOG_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, CATALOG_PATH)


def _source_id(source_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"uniguide-source:{source_key.strip()}"))


def _active_version(data: dict[str, Any], source_id: str) -> dict[str, Any] | None:
    return next((v for v in data["versions"].values() if v["source_id"] == source_id and v["status"] == VersionStatus.ACTIVE), None)


class VersioningError(ValueError):
    """Raised for invalid lifecycle or integrity operations."""


class KnowledgeVersionStore:
    def __init__(self, catalog_path: Path = CATALOG_PATH, artifacts_dir: Path = VERSIONING_DIR) -> None:
        self.catalog_path = catalog_path
        self.artifacts_dir = artifacts_dir

    def _read(self) -> dict[str, Any]:
        if self.catalog_path != CATALOG_PATH:
            if not self.catalog_path.exists():
                return _empty_catalog()
            return json.loads(self.catalog_path.read_text(encoding="utf-8"))
        return _read_catalog()

    def _write(self, data: dict[str, Any]) -> None:
        if self.catalog_path != CATALOG_PATH:
            self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.catalog_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self.catalog_path)
        else:
            _write_catalog(data)

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        lock_path = LOCK_PATH if self.catalog_path == CATALOG_PATH else self.catalog_path.with_suffix(".lock")
        with _catalog_lock(lock_path):
            data = self._read()
            yield data
            self._write(data)

    def register_source(self, source_key: str, display_name: str | None = None) -> Source:
        source_key = source_key.strip()
        if not source_key:
            raise VersioningError("source_key must not be empty")
        source_id = _source_id(source_key)
        with self._locked() as data:
            source = data["sources"].get(source_id)
            if source is None:
                source = asdict(Source(source_id, source_key, display_name or source_key, utc_now()))
                data["sources"][source_id] = source
            elif display_name:
                source["display_name"] = display_name
        return Source(**source)

    def create_or_reuse_version(
        self, source: Source, content: str, *, processing_config: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> KnowledgeVersion:
        config = dict(processing_config or {})
        content_hash = content_fingerprint(content)
        config_hash = processing_fingerprint(config)
        with self._locked() as data:
            existing = next((v for v in data["versions"].values() if v["source_id"] == source.source_id and v["content_fingerprint"] == content_hash and v["processing_fingerprint"] == config_hash and v["status"] != VersionStatus.FAILED), None)
            if existing:
                return KnowledgeVersion(**existing)
            sequence = max((v["sequence"] for v in data["versions"].values() if v["source_id"] == source.source_id), default=0) + 1
            version_id = str(uuid.uuid4())
            run_id = str(uuid.uuid4())
            version = KnowledgeVersion(version_id, source.source_id, sequence, content_hash, config_hash, VersionStatus.PROCESSING, utc_now(), run_id, processing_config=config, provenance=dict(provenance or {}))
            run = IngestionRun(run_id, source.source_id, version_id, VersionStatus.PROCESSING, version.created_at)
            data["versions"][version_id] = asdict(version)
            data["runs"][run_id] = asdict(run)
            return version

    def transition(self, version_id: str, target: VersionStatus, *, failure_reason: str | None = None) -> KnowledgeVersion:
        with self._locked() as data:
            raw = data["versions"].get(version_id)
            if raw is None:
                raise VersioningError(f"Unknown knowledge version: {version_id}")
            current = VersionStatus(raw["status"])
            if target not in _TRANSITIONS[current]:
                raise VersioningError(f"Invalid version transition: {current} -> {target}")
            raw["status"] = target
            if target == VersionStatus.FAILED:
                raw["failure_reason"] = failure_reason or "ingestion failed"
            if target == VersionStatus.RETIRED:
                raw["retired_at"] = utc_now()
            run = data["runs"].get(raw["ingestion_run_id"])
            if run and target in {VersionStatus.READY, VersionStatus.FAILED}:
                run["status"] = target
                run["completed_at"] = utc_now()
                run["error"] = raw.get("failure_reason")
            return KnowledgeVersion(**raw)

    def save_artifact(self, version_id: str, chunks: list[dict], vectorizer: Any, vectors: list[list[float]]) -> Path:
        path = self.artifacts_dir / version_id / "index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        save_index(chunks, vectorizer, vectors, temporary)
        os.replace(temporary, path)
        return path

    def activate(self, version_id: str) -> KnowledgeVersion:
        with self._locked() as data:
            raw = data["versions"].get(version_id)
            if raw is None:
                raise VersioningError(f"Unknown knowledge version: {version_id}")
            if raw["status"] not in {VersionStatus.READY, VersionStatus.RETIRED}:
                raise VersioningError("Only ready or retired versions can be activated")
            artifact = self.artifacts_dir / version_id / "index.json"
            if not artifact.exists():
                raise VersioningError("Cannot activate a version without an index artifact")
            previous = _active_version(data, raw["source_id"])
            now = utc_now()
            if previous and previous["version_id"] != version_id:
                previous["status"] = VersionStatus.RETIRED
                previous["retired_at"] = now
            raw["status"] = VersionStatus.ACTIVE
            raw["activated_at"] = now
            data["activation_history"].append({"source_id": raw["source_id"], "from_version_id": previous["version_id"] if previous else None, "to_version_id": version_id, "activated_at": now})
            return KnowledgeVersion(**raw)

    def active_version(self, source_id: str | None = None) -> KnowledgeVersion | None:
        data = self._read()
        raw = next((v for v in data["versions"].values() if v["status"] == VersionStatus.ACTIVE and (source_id is None or v["source_id"] == source_id)), None)
        return KnowledgeVersion(**raw) if raw else None

    def history(self, source_id: str) -> list[KnowledgeVersion]:
        data = self._read()
        return [KnowledgeVersion(**v) for v in sorted(data["versions"].values(), key=lambda item: item["sequence"]) if v["source_id"] == source_id]

    def load_active_index(self, source_id: str | None = None):
        version = self.active_version(source_id)
        if version is None:
            raise FileNotFoundError("No active knowledge version")
        return version, load_index(self.artifacts_dir / version.version_id / "index.json")


def ingest_index(store: KnowledgeVersionStore, source_key: str, content: str, chunks: list[dict], vectorizer: Any, vectors: list[list[float]], *, processing_config: dict[str, Any] | None = None, provenance: dict[str, Any] | None = None) -> KnowledgeVersion:
    """Persist one complete version and make it eligible, but not active, atomically."""
    source = store.register_source(source_key)
    version = store.create_or_reuse_version(source, content, processing_config=processing_config, provenance=provenance)
    if version.status == VersionStatus.READY or version.status == VersionStatus.ACTIVE:
        return version
    try:
        versioned_chunks = [
            {
                **chunk,
                "knowledge_version_id": version.version_id,
                "source_id": version.source_id,
            }
            for chunk in chunks
        ]
        store.save_artifact(version.version_id, versioned_chunks, vectorizer, vectors)
        return store.transition(version.version_id, VersionStatus.READY)
    except Exception as exc:
        store.transition(version.version_id, VersionStatus.FAILED, failure_reason=str(exc))
        raise
