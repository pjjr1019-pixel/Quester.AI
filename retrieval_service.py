"""Dedicated local retrieval service for bounded hybrid search."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from config import RetrievalSettings
from retrieval import (
    DocumentChunkRecord,
    LexicalSearchHit,
    SearchResult,
    VectorIndexAdapter,
    metadata_excludes,
    rerank_search_results,
)


class LexicalCandidateStore(Protocol):
    """Repository boundary for bounded lexical candidate selection."""

    def search_lexical_hits(
        self,
        query_text: str,
        *,
        limit: int,
        metadata_filters: dict[str, Any] | None = None,
        metadata_exclusions: dict[str, Any] | None = None,
    ) -> tuple[LexicalSearchHit, ...]: ...


class ChunkRecordStore(Protocol):
    """Repository boundary for loading persisted chunk/vector records by id."""

    def get_chunk_records_by_ids(
        self,
        chunk_ids: Sequence[str],
    ) -> tuple[DocumentChunkRecord, ...]: ...


class LocalRetrievalService:
    """Bounded hybrid retrieval with lexical/vector merge and optional reranking."""

    def __init__(
        self,
        *,
        settings: RetrievalSettings,
        lexical_store: LexicalCandidateStore,
        chunk_store: ChunkRecordStore,
    ) -> None:
        self._settings = settings
        self._lexical_store = lexical_store
        self._chunk_store = chunk_store

    def search(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int,
        vector_index: VectorIndexAdapter,
        metadata_filters: dict[str, Any] | None = None,
        metadata_exclusions: dict[str, Any] | None = None,
        allow_rerank: bool = False,
    ) -> tuple[SearchResult, ...]:
        """Return bounded local retrieval results without scanning the full corpus."""
        effective_top_k = max(1, top_k)
        candidate_limit = max(
            1,
            min(self._settings.max_vector_candidates, effective_top_k * 4),
        )
        lexical_hits = self._lexical_store.search_lexical_hits(
            query_text,
            limit=candidate_limit,
            metadata_filters=metadata_filters,
            metadata_exclusions=metadata_exclusions,
        )
        vector_hits = vector_index.search(
            tuple(float(value) for value in query_vector),
            top_k=candidate_limit,
            metadata_filters=metadata_filters,
        )

        candidates: dict[str, dict[str, float]] = {}
        for hit in lexical_hits:
            candidates[hit.chunk_id] = {
                "lexical_score": float(hit.score),
                "vector_score": 0.0,
            }
        for hit in vector_hits:
            candidate = candidates.setdefault(
                hit.chunk_id,
                {"lexical_score": 0.0, "vector_score": 0.0},
            )
            candidate["vector_score"] = float(hit.score)

        if not candidates:
            return ()

        records_by_id = {
            record.chunk_id: record
            for record in self._chunk_store.get_chunk_records_by_ids(tuple(candidates))
        }
        results: list[SearchResult] = []
        for chunk_id, candidate in candidates.items():
            chunk = records_by_id.get(chunk_id)
            if chunk is None:
                continue
            if metadata_excludes(chunk.metadata, metadata_exclusions):
                continue
            lexical_score = float(candidate["lexical_score"])
            vector_score = float(candidate["vector_score"])
            combined_score = (
                (lexical_score * self._settings.lexical_weight)
                + (vector_score * self._settings.vector_weight)
            )
            if (
                combined_score < self._settings.minimum_combined_score
                and vector_score < self._settings.vector_only_score_threshold
                and lexical_score == 0.0
            ):
                continue
            results.append(
                SearchResult(
                    chunk=chunk,
                    lexical_score=lexical_score,
                    vector_score=vector_score,
                    combined_score=min(1.0, combined_score),
                )
            )

        results.sort(
            key=lambda result: (
                result.combined_score,
                result.lexical_score,
                result.vector_score,
            ),
            reverse=True,
        )
        if allow_rerank and self._settings.enable_reranking:
            results = list(
                rerank_search_results(
                    query_text,
                    tuple(results),
                    max_candidates=min(
                        effective_top_k,
                        self._settings.max_rerank_candidates,
                    ),
                    combined_weight=self._settings.rerank_combined_weight,
                    lexical_weight=self._settings.rerank_lexical_weight,
                    order_weight=self._settings.rerank_order_weight,
                    exact_phrase_weight=self._settings.rerank_exact_phrase_weight,
                    title_weight=self._settings.rerank_title_weight,
                )
            )
        return tuple(results[:effective_top_k])
