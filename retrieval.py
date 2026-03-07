"""Chunking and vector-index helpers for local-first retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def stable_hash(value: str) -> str:
    """Return a stable content hash for IDs and dedupe."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_document_id(source_ref: str) -> str:
    """Return a stable document ID derived from the source ref."""
    return f"doc_{stable_hash(source_ref)[:16]}"


def make_chunk_id(document_id: str, chunk_index: int, content_hash: str) -> str:
    """Return a stable chunk ID derived from document identity and content."""
    return f"{document_id}_chunk_{chunk_index}_{content_hash[:12]}"


def tokenize_text(text: str) -> set[str]:
    """Tokenize text into a small lexical set for hybrid retrieval."""
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {token for token in tokens if len(token) > 1 and token not in _STOP_WORDS}


def build_fts_query(text: str) -> str:
    """Build a simple FTS query from normalized lexical tokens."""
    tokens = sorted(tokenize_text(text))
    return " OR ".join(tokens)


def lexical_overlap_score(query_text: str, candidate_text: str) -> float:
    """Return a simple lexical-overlap score for deterministic reranking."""
    query_tokens = tokenize_text(query_text)
    candidate_tokens = tokenize_text(candidate_text)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    return overlap / len(query_tokens)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return cosine similarity for two vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)))


def metadata_matches(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Return true when metadata matches the provided exact-match filters."""
    if not filters:
        return True
    for key, expected in filters.items():
        if metadata.get(key) != expected:
            return False
    return True


def metadata_excludes(metadata: dict[str, Any], exclusions: dict[str, Any] | None) -> bool:
    """Return true when metadata matches one of the exact-match exclusion rules."""
    if not exclusions:
        return False
    for key, forbidden in exclusions.items():
        if metadata.get(key) == forbidden:
            return True
    return False


def chunk_text(
    text: str,
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    max_chunks: int,
) -> list[str]:
    """Split text into bounded overlapping chunks."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized) and len(chunks) < max_chunks:
        end = min(len(normalized), start + chunk_size_chars)
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + max(16, chunk_size_chars // 2), end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        next_start = max(end - chunk_overlap_chars, start + 1)
        while next_start < len(normalized) and normalized[next_start].isspace():
            next_start += 1
        start = next_start
    return chunks


@dataclass(slots=True, frozen=True)
class SourceDocumentRecord:
    """Stored source document."""

    document_id: str
    source_ref: str
    title: str
    content: str
    content_hash: str
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(slots=True, frozen=True)
class DocumentChunkRecord:
    """Stored document chunk and vector payload."""

    chunk_id: str
    document_id: str
    source_ref: str
    chunk_index: int
    content: str
    content_hash: str
    metadata: dict[str, Any]
    embedding_model: str
    vector: tuple[float, ...]
    created_at: str


@dataclass(slots=True, frozen=True)
class VectorSearchHit:
    """Vector-index search hit."""

    chunk_id: str
    score: float


@dataclass(slots=True, frozen=True)
class LexicalSearchHit:
    """Lexical candidate hit returned from the bounded text stage."""

    chunk_id: str
    score: float


@dataclass(slots=True, frozen=True)
class SearchResult:
    """Merged lexical/vector retrieval hit."""

    chunk: DocumentChunkRecord
    lexical_score: float
    vector_score: float
    combined_score: float
    rerank_score: float = 0.0


class VectorIndexAdapter(Protocol):
    """Behavior required by local vector index adapters."""

    backend_name: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def requires_startup_reload(self, *, stored_chunk_count: int) -> bool: ...

    def upsert_chunks(self, chunks: tuple[DocumentChunkRecord, ...]) -> None: ...

    def delete_document(self, document_id: str) -> None: ...

    def search(
        self,
        query_vector: tuple[float, ...],
        *,
        top_k: int,
        metadata_filters: dict[str, Any] | None = None,
    ) -> tuple[VectorSearchHit, ...]: ...


class SimpleInMemoryVectorIndex:
    """Deterministic zero-dependency vector search adapter."""

    backend_name = "simple_inmemory"

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunkRecord] = {}

    def start(self) -> None:
        return

    def stop(self) -> None:
        self._chunks.clear()

    def requires_startup_reload(self, *, stored_chunk_count: int) -> bool:
        return stored_chunk_count > 0

    def upsert_chunks(self, chunks: tuple[DocumentChunkRecord, ...]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def delete_document(self, document_id: str) -> None:
        stale_chunk_ids = [
            chunk_id
            for chunk_id, chunk in self._chunks.items()
            if chunk.document_id == document_id
        ]
        for chunk_id in stale_chunk_ids:
            self._chunks.pop(chunk_id, None)

    def search(
        self,
        query_vector: tuple[float, ...],
        *,
        top_k: int,
        metadata_filters: dict[str, Any] | None = None,
    ) -> tuple[VectorSearchHit, ...]:
        hits: list[VectorSearchHit] = []
        for chunk in self._chunks.values():
            if not metadata_matches(chunk.metadata, metadata_filters):
                continue
            score = cosine_similarity(query_vector, chunk.vector)
            hits.append(VectorSearchHit(chunk_id=chunk.chunk_id, score=score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return tuple(hits[:top_k])


class ChromaVectorIndex:
    """Optional Chroma-backed vector index adapter."""

    backend_name = "chromadb"

    def __init__(self, *, persist_path: Path, collection_name: str) -> None:
        self.persist_path = persist_path
        self.collection_name = collection_name
        self._client: Any | None = None
        self._collection: Any | None = None

    def start(self) -> None:
        import chromadb

        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_path))
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def stop(self) -> None:
        self._collection = None
        self._client = None

    def requires_startup_reload(self, *, stored_chunk_count: int) -> bool:
        if stored_chunk_count <= 0:
            return False
        collection = self._require_collection()
        try:
            return int(collection.count()) != stored_chunk_count
        except Exception:
            return True

    def upsert_chunks(self, chunks: tuple[DocumentChunkRecord, ...]) -> None:
        collection = self._require_collection()
        ids = [chunk.chunk_id for chunk in chunks]
        embeddings = [list(chunk.vector) for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "source_ref": chunk.source_ref,
                **{key: value for key, value in chunk.metadata.items() if isinstance(value, (str, int, float, bool))},
            }
            for chunk in chunks
        ]
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def delete_document(self, document_id: str) -> None:
        collection = self._require_collection()
        existing = collection.get(where={"document_id": document_id})
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)

    def search(
        self,
        query_vector: tuple[float, ...],
        *,
        top_k: int,
        metadata_filters: dict[str, Any] | None = None,
    ) -> tuple[VectorSearchHit, ...]:
        collection = self._require_collection()
        query_result = collection.query(
            query_embeddings=[list(query_vector)],
            n_results=top_k,
            where=metadata_filters or None,
        )
        ids = query_result.get("ids", [[]])[0]
        distances = query_result.get("distances", [[]])[0]
        hits = [
            VectorSearchHit(chunk_id=chunk_id, score=1.0 / (1.0 + float(distance)))
            for chunk_id, distance in zip(ids, distances)
        ]
        return tuple(hits)

    def _require_collection(self) -> Any:
        if self._collection is None:
            raise RuntimeError("ChromaVectorIndex must be started before use.")
        return self._collection


def ordered_token_score(query_text: str, candidate_text: str) -> float:
    """Return a lightweight ordered-token match score for bounded reranking."""
    query_tokens = [token for token in re.findall(r"[a-z0-9]+", query_text.lower()) if token not in _STOP_WORDS]
    candidate_tokens = [token for token in re.findall(r"[a-z0-9]+", candidate_text.lower()) if token not in _STOP_WORDS]
    if not query_tokens or not candidate_tokens:
        return 0.0
    next_position = 0
    matches = 0
    for token in query_tokens:
        try:
            found_index = candidate_tokens.index(token, next_position)
        except ValueError:
            continue
        matches += 1
        next_position = found_index + 1
    return matches / len(query_tokens)


def exact_phrase_score(query_text: str, candidate_text: str) -> float:
    """Return 1.0 when the normalized query phrase appears contiguously."""
    normalized_query = " ".join(re.findall(r"[a-z0-9]+", query_text.lower()))
    normalized_candidate = " ".join(re.findall(r"[a-z0-9]+", candidate_text.lower()))
    if not normalized_query or not normalized_candidate:
        return 0.0
    return 1.0 if normalized_query in normalized_candidate else 0.0


def rerank_search_results(
    query_text: str,
    results: tuple[SearchResult, ...],
    *,
    max_candidates: int,
    combined_weight: float,
    lexical_weight: float,
    order_weight: float,
    exact_phrase_weight: float,
    title_weight: float,
) -> tuple[SearchResult, ...]:
    """Rerank only a bounded top-N candidate set using deterministic heuristics."""
    rerank_limit = min(max(0, max_candidates), len(results))
    if rerank_limit <= 1:
        return results

    reranked_prefix: list[SearchResult] = []
    for result in results[:rerank_limit]:
        title_text = str(result.chunk.metadata.get("title", ""))
        ordered_score = max(
            ordered_token_score(query_text, result.chunk.content),
            ordered_token_score(query_text, title_text),
        )
        phrase_score = max(
            exact_phrase_score(query_text, result.chunk.content),
            exact_phrase_score(query_text, title_text),
        )
        title_overlap = lexical_overlap_score(query_text, title_text)
        rerank_score = min(
            1.0,
            (result.combined_score * combined_weight)
            + (result.lexical_score * lexical_weight)
            + (ordered_score * order_weight)
            + (phrase_score * exact_phrase_weight)
            + (title_overlap * title_weight),
        )
        reranked_prefix.append(
            SearchResult(
                chunk=result.chunk,
                lexical_score=result.lexical_score,
                vector_score=result.vector_score,
                combined_score=result.combined_score,
                rerank_score=rerank_score,
            )
        )

    reranked_prefix.sort(
        key=lambda result: (
            result.rerank_score,
            result.combined_score,
            result.lexical_score,
            result.vector_score,
        ),
        reverse=True,
    )
    return tuple(reranked_prefix + list(results[rerank_limit:]))
