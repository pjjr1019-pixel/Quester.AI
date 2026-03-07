"""SQLite-backed storage, repository boundaries, and local retrieval helpers."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from config import APP_CONFIG, AppConfig
from data_structures import (
    AgentStatus,
    CompressedTrace,
    CompressionRuntimeSubset,
    CritiqueResult,
    DecoderEntry,
    Macro,
    OpcodeEntry,
    OptimizerActivationRecord,
    OptimizerProposalRecord,
    OptimizerReplayEvaluation,
    OptimizerReplaySample,
    OptimizerRollbackRecord,
    PerformanceMetric,
    ProofHashRecord,
    ReasoningLog,
    RuntimeEvent,
    SymbolTableSnapshot,
    TaskResult,
    VerifiedDeepTraceExport,
    WebEvidenceRecord,
    coerce_agent_status,
    coerce_compressed_trace,
    coerce_decoder_entry,
    coerce_optimizer_activation_record,
    coerce_optimizer_proposal_record,
    coerce_optimizer_replay_evaluation,
    coerce_optimizer_replay_sample,
    coerce_optimizer_rollback_record,
    coerce_opcode_entry,
    coerce_proof_hash_record,
    coerce_runtime_event,
    coerce_symbol_table_snapshot,
    coerce_task_result,
    coerce_verified_deep_trace_export,
    coerce_web_evidence_record,
)
from retrieval import (
    ChromaVectorIndex,
    DocumentChunkRecord,
    LexicalSearchHit,
    SearchResult,
    SimpleInMemoryVectorIndex,
    SourceDocumentRecord,
    VectorIndexAdapter,
    build_fts_query,
    chunk_text,
    metadata_excludes,
    lexical_overlap_score,
    make_chunk_id,
    make_document_id,
    metadata_matches,
    stable_hash,
)
from retrieval_service import LocalRetrievalService
from utils import ensure_directory, utc_now_iso


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _json_loads(value: str) -> Any:
    return json.loads(value)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_json_dumps(payload) + "\n")


class _SQLiteRepository:
    """Small helper base class for repositories sharing one SQLite connection."""

    def __init__(self, storage: StorageManager):
        self._storage = storage

    def _conn(self) -> sqlite3.Connection:
        return self._storage._require_conn()

    @property
    def _lock(self) -> threading.Lock:
        return self._storage._lock


class EventLogRepository(_SQLiteRepository):
    """Structured runtime event persistence."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                stage TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        conn.commit()

    def append(self, event: RuntimeEvent, *, events_path: Path) -> None:
        conn = self._conn()
        payload_json = _json_dumps(event.payload)
        timestamp = event.timestamp.isoformat()
        with self._lock:
            conn.execute(
                "INSERT INTO runtime_events (timestamp, stage, payload_json) VALUES (?, ?, ?)",
                (timestamp, event.stage, payload_json),
            )
            conn.commit()
            _append_jsonl(events_path, event.to_dict())

    def list(self, *, stage: str | None = None) -> tuple[RuntimeEvent, ...]:
        conn = self._conn()
        if stage is None:
            query = "SELECT timestamp, stage, payload_json FROM runtime_events ORDER BY id"
            values: tuple[Any, ...] = ()
        else:
            query = (
                "SELECT timestamp, stage, payload_json FROM runtime_events "
                "WHERE stage = ? ORDER BY id"
            )
            values = (stage,)
        with self._lock:
            cursor = conn.execute(query, values)
            rows = cursor.fetchall()
        return tuple(
            RuntimeEvent.from_dict(
                {
                    "timestamp": row["timestamp"],
                    "stage": row["stage"],
                    "payload": _json_loads(row["payload_json"]),
                }
            )
            for row in rows
        )


class KeyValueRepository(_SQLiteRepository):
    """Simple JSON key-value persistence."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL
            );
            """
        )
        conn.commit()

    def set(self, key: str, value: Any) -> None:
        conn = self._conn()
        value_json = _json_dumps(value)
        with self._lock:
            conn.execute(
                "INSERT INTO kv_store (key, value_json) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                (key, value_json),
            )
            conn.commit()

    def get(self, key: str) -> Any | None:
        conn = self._conn()
        with self._lock:
            cursor = conn.execute("SELECT value_json FROM kv_store WHERE key = ?", (key,))
            row = cursor.fetchone()
        if row is None:
            return None
        return _json_loads(row["value_json"])


class TaskRepository(_SQLiteRepository):
    """Task-level persistence for final orchestrator output."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS task_runs (
                task_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.commit()

    def upsert(self, result: TaskResult) -> None:
        conn = self._conn()
        payload_json = _json_dumps(result.to_dict())
        created_at = result.plan.created_at.isoformat()
        updated_at = result.completed_at.isoformat()
        with self._lock:
            conn.execute(
                """
                INSERT INTO task_runs (task_id, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (result.task_id, payload_json, created_at, updated_at),
            )
            conn.commit()

    def get(self, task_id: str) -> TaskResult | None:
        conn = self._conn()
        with self._lock:
            cursor = conn.execute(
                "SELECT payload_json FROM task_runs WHERE task_id = ?",
                (task_id,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return TaskResult.from_dict(_json_loads(row["payload_json"]))

    def count(self) -> int:
        conn = self._conn()
        with self._lock:
            cursor = conn.execute("SELECT COUNT(*) AS count FROM task_runs")
            row = cursor.fetchone()
        return int(row["count"])

    def list(self, *, limit: int | None = None) -> tuple[TaskResult, ...]:
        conn = self._conn()
        query = "SELECT payload_json FROM task_runs ORDER BY updated_at, task_id"
        values: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            values = (limit,)
        with self._lock:
            cursor = conn.execute(query, values)
            rows = cursor.fetchall()
        return tuple(TaskResult.from_dict(_json_loads(row["payload_json"])) for row in rows)


class AgentStatusRepository(_SQLiteRepository):
    """Append-only status history shared by runtime, storage, and dashboard consumers."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_status_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT NOT NULL,
                task_id TEXT,
                state TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_status_history_task ON agent_status_history (task_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_status_history_component ON agent_status_history (component);"
        )
        conn.commit()

    def append(self, status: AgentStatus, *, status_path: Path) -> None:
        conn = self._conn()
        payload = status.to_dict()
        with self._lock:
            conn.execute(
                """
                INSERT INTO agent_status_history (
                    component,
                    task_id,
                    state,
                    severity,
                    message,
                    updated_at,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    status.component,
                    status.task_id,
                    status.state.value,
                    status.severity.value,
                    status.message,
                    status.updated_at.isoformat(),
                    _json_dumps(payload),
                ),
            )
            conn.commit()
            _append_jsonl(status_path, payload)

    def list(
        self,
        *,
        task_id: str | None = None,
        component: str | None = None,
    ) -> tuple[AgentStatus, ...]:
        conn = self._conn()
        clauses: list[str] = []
        values: list[str] = []
        if task_id is not None:
            clauses.append("task_id = ?")
            values.append(task_id)
        if component is not None:
            clauses.append("component = ?")
            values.append(component)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            cursor = conn.execute(
                f"""
                SELECT payload_json
                FROM agent_status_history
                {where_clause}
                ORDER BY id
                """,
                tuple(values),
            )
            rows = cursor.fetchall()
        return tuple(AgentStatus.from_dict(_json_loads(row["payload_json"])) for row in rows)


class WebEvidenceRepository(_SQLiteRepository):
    """Persisted fetched web evidence with lookup provenance."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS web_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                provider TEXT NOT NULL,
                reason TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                degraded INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(task_id, evidence_id)
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_web_evidence_task ON web_evidence (task_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_web_evidence_provider ON web_evidence (provider);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_web_evidence_source ON web_evidence (source_ref);"
        )
        conn.commit()

    def upsert_many(
        self,
        records: Sequence[WebEvidenceRecord],
        *,
        web_path: Path,
    ) -> None:
        if not records:
            return
        conn = self._conn()
        serialized_records = [record.to_dict() for record in records]
        with self._lock:
            conn.executemany(
                """
                INSERT INTO web_evidence (
                    task_id,
                    evidence_id,
                    query_text,
                    provider,
                    reason,
                    source_ref,
                    degraded,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id, evidence_id) DO UPDATE SET
                    query_text = excluded.query_text,
                    provider = excluded.provider,
                    reason = excluded.reason,
                    source_ref = excluded.source_ref,
                    degraded = excluded.degraded,
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at
                """,
                [
                    (
                        record.task_id,
                        record.evidence.id,
                        record.query,
                        record.provider,
                        record.reason,
                        record.evidence.source_ref,
                        1 if record.degraded else 0,
                        _json_dumps(payload),
                        record.created_at.isoformat(),
                    )
                    for record, payload in zip(records, serialized_records)
                ],
            )
            conn.commit()
            for payload in serialized_records:
                _append_jsonl(
                    web_path,
                    {
                        "kind": "web_evidence",
                        **payload,
                    },
                )

    def list(
        self,
        *,
        task_id: str | None = None,
        provider: str | None = None,
        source_ref: str | None = None,
    ) -> tuple[WebEvidenceRecord, ...]:
        conn = self._conn()
        clauses: list[str] = []
        values: list[str] = []
        if task_id is not None:
            clauses.append("task_id = ?")
            values.append(task_id)
        if provider is not None:
            clauses.append("provider = ?")
            values.append(provider)
        if source_ref is not None:
            clauses.append("source_ref = ?")
            values.append(source_ref)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            cursor = conn.execute(
                f"""
                SELECT payload_json
                FROM web_evidence
                {where_clause}
                ORDER BY id
                """,
                tuple(values),
            )
            rows = cursor.fetchall()
        return tuple(WebEvidenceRecord.from_dict(_json_loads(row["payload_json"])) for row in rows)


class SourceDocumentRepository(_SQLiteRepository):
    """Original document persistence independent from embeddings or vector indexes."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS source_documents (
                document_id TEXT PRIMARY KEY,
                source_ref TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.commit()

    def upsert(self, record: SourceDocumentRecord) -> None:
        conn = self._conn()
        with self._lock:
            conn.execute(
                """
                INSERT INTO source_documents (
                    document_id,
                    source_ref,
                    title,
                    content,
                    content_hash,
                    metadata_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    source_ref = excluded.source_ref,
                    title = excluded.title,
                    content = excluded.content,
                    content_hash = excluded.content_hash,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.document_id,
                    record.source_ref,
                    record.title,
                    record.content,
                    record.content_hash,
                    _json_dumps(record.metadata),
                    record.created_at,
                    record.updated_at,
                ),
            )
            conn.commit()

    def get_by_source_ref(self, source_ref: str) -> SourceDocumentRecord | None:
        conn = self._conn()
        with self._lock:
            cursor = conn.execute(
                """
                SELECT
                    document_id,
                    source_ref,
                    title,
                    content,
                    content_hash,
                    metadata_json,
                    created_at,
                    updated_at
                FROM source_documents
                WHERE source_ref = ?
                """,
                (source_ref,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return SourceDocumentRecord(
            document_id=row["document_id"],
            source_ref=row["source_ref"],
            title=row["title"],
            content=row["content"],
            content_hash=row["content_hash"],
            metadata=dict(_json_loads(row["metadata_json"])),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def count(self) -> int:
        conn = self._conn()
        with self._lock:
            cursor = conn.execute("SELECT COUNT(*) AS count FROM source_documents")
            row = cursor.fetchone()
        return int(row["count"])


class ChunkRepository(_SQLiteRepository):
    """Chunk metadata and source text persistence."""

    def create_tables(self) -> None:
        conn = self._conn()
        with self._lock:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_chunks_document ON document_chunks (document_id);"
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    source_ref,
                    title,
                    content,
                    tokenize='unicode61'
                );
                """
            )
            chunk_count = int(
                conn.execute("SELECT COUNT(*) AS count FROM document_chunks").fetchone()["count"]
            )
            fts_count = int(
                conn.execute("SELECT COUNT(*) AS count FROM document_chunks_fts").fetchone()["count"]
            )
            if chunk_count > 0 and fts_count != chunk_count:
                self._rebuild_fts_index_locked(conn)
            conn.commit()

    def replace_document_chunks(self, document_id: str, chunks: Sequence[DocumentChunkRecord]) -> None:
        conn = self._conn()
        keep_ids = tuple(chunk.chunk_id for chunk in chunks)
        with self._lock:
            conn.execute("DELETE FROM document_chunks_fts WHERE document_id = ?", (document_id,))
            if keep_ids:
                placeholders = ", ".join("?" for _ in keep_ids)
                conn.execute(
                    f"DELETE FROM document_chunks WHERE document_id = ? AND chunk_id NOT IN ({placeholders})",
                    (document_id, *keep_ids),
                )
            else:
                conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO document_chunks (
                    chunk_id,
                    document_id,
                    source_ref,
                    chunk_index,
                    content,
                    content_hash,
                    metadata_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    source_ref = excluded.source_ref,
                    chunk_index = excluded.chunk_index,
                    content = excluded.content,
                    content_hash = excluded.content_hash,
                    metadata_json = excluded.metadata_json
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.source_ref,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.content_hash,
                        _json_dumps(chunk.metadata),
                        chunk.created_at,
                    )
                    for chunk in chunks
                ],
            )
            conn.executemany(
                """
                INSERT INTO document_chunks_fts (
                    chunk_id,
                    document_id,
                    source_ref,
                    title,
                    content
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.source_ref,
                        str(chunk.metadata.get("title", "")),
                        chunk.content,
                    )
                    for chunk in chunks
                ],
            )
            conn.commit()

    def search_lexical_hits(
        self,
        query_text: str,
        *,
        limit: int,
        metadata_filters: dict[str, Any] | None = None,
        metadata_exclusions: dict[str, Any] | None = None,
    ) -> tuple[LexicalSearchHit, ...]:
        effective_limit = max(1, limit)
        fts_query = build_fts_query(query_text)
        if not fts_query:
            return ()

        conn = self._conn()
        raw_limit = max(16, effective_limit * 8)
        with self._lock:
            cursor = conn.execute(
                """
                SELECT
                    c.chunk_id,
                    c.content,
                    c.metadata_json
                FROM document_chunks_fts
                INNER JOIN document_chunks AS c
                    ON c.chunk_id = document_chunks_fts.chunk_id
                WHERE document_chunks_fts MATCH ?
                ORDER BY bm25(document_chunks_fts), c.chunk_id
                LIMIT ?
                """,
                (fts_query, raw_limit),
            )
            rows = cursor.fetchall()

        hits: list[LexicalSearchHit] = []
        for row in rows:
            metadata = dict(_json_loads(row["metadata_json"]))
            if not metadata_matches(metadata, metadata_filters):
                continue
            if metadata_excludes(metadata, metadata_exclusions):
                continue
            score = lexical_overlap_score(query_text, row["content"])
            if score <= 0.0:
                continue
            hits.append(LexicalSearchHit(chunk_id=row["chunk_id"], score=score))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return tuple(hits[:effective_limit])

    def count(self) -> int:
        conn = self._conn()
        with self._lock:
            cursor = conn.execute("SELECT COUNT(*) AS count FROM document_chunks")
            row = cursor.fetchone()
        return int(row["count"])

    def _rebuild_fts_index_locked(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute(
            """
            SELECT chunk_id, document_id, source_ref, content, metadata_json
            FROM document_chunks
            ORDER BY document_id, chunk_index
            """
        )
        rows = cursor.fetchall()
        conn.execute("DELETE FROM document_chunks_fts")
        conn.executemany(
            """
            INSERT INTO document_chunks_fts (
                chunk_id,
                document_id,
                source_ref,
                title,
                content
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    row["chunk_id"],
                    row["document_id"],
                    row["source_ref"],
                    str(dict(_json_loads(row["metadata_json"])).get("title", "")),
                    row["content"],
                )
                for row in rows
            ],
        )


class VectorEntryRepository(_SQLiteRepository):
    """Vector payloads and embedding metadata independent from source documents."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_entries (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                embedding_backend TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_version TEXT NOT NULL,
                vector_store_backend TEXT NOT NULL,
                vector_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vector_entries_document ON vector_entries (document_id);"
        )
        conn.commit()

    def replace_document_vectors(
        self,
        document_id: str,
        chunks: Sequence[DocumentChunkRecord],
        *,
        embedding_backend: str,
        embedding_model: str,
        embedding_version: str,
        vector_store_backend: str,
    ) -> None:
        conn = self._conn()
        keep_ids = tuple(chunk.chunk_id for chunk in chunks)
        with self._lock:
            if keep_ids:
                placeholders = ", ".join("?" for _ in keep_ids)
                conn.execute(
                    f"DELETE FROM vector_entries WHERE document_id = ? AND chunk_id NOT IN ({placeholders})",
                    (document_id, *keep_ids),
                )
            else:
                conn.execute("DELETE FROM vector_entries WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO vector_entries (
                    chunk_id,
                    document_id,
                    vector_json,
                    embedding_backend,
                    embedding_model,
                    embedding_version,
                    vector_store_backend,
                    vector_id,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    vector_json = excluded.vector_json,
                    embedding_backend = excluded.embedding_backend,
                    embedding_model = excluded.embedding_model,
                    embedding_version = excluded.embedding_version,
                    vector_store_backend = excluded.vector_store_backend,
                    vector_id = excluded.vector_id,
                    updated_at = excluded.updated_at
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        _json_dumps(chunk.vector),
                        embedding_backend,
                        embedding_model,
                        embedding_version,
                        vector_store_backend,
                        chunk.chunk_id,
                        chunk.created_at,
                    )
                    for chunk in chunks
                ],
            )
            conn.commit()

    def list_chunk_records(self) -> tuple[DocumentChunkRecord, ...]:
        conn = self._conn()
        with self._lock:
            cursor = conn.execute(
                """
                SELECT
                    c.chunk_id,
                    c.document_id,
                    c.source_ref,
                    c.chunk_index,
                    c.content,
                    c.content_hash,
                    c.metadata_json,
                    c.created_at,
                    v.embedding_model,
                    v.vector_json
                FROM document_chunks AS c
                INNER JOIN vector_entries AS v
                    ON v.chunk_id = c.chunk_id
                ORDER BY c.document_id, c.chunk_index
                """
            )
            rows = cursor.fetchall()
        return tuple(self._row_to_chunk_record(row) for row in rows)

    def get_chunk_records_by_ids(self, chunk_ids: Sequence[str]) -> tuple[DocumentChunkRecord, ...]:
        if not chunk_ids:
            return ()
        conn = self._conn()
        placeholders = ", ".join("?" for _ in chunk_ids)
        with self._lock:
            cursor = conn.execute(
                f"""
                SELECT
                    c.chunk_id,
                    c.document_id,
                    c.source_ref,
                    c.chunk_index,
                    c.content,
                    c.content_hash,
                    c.metadata_json,
                    c.created_at,
                    v.embedding_model,
                    v.vector_json
                FROM document_chunks AS c
                INNER JOIN vector_entries AS v
                    ON v.chunk_id = c.chunk_id
                WHERE c.chunk_id IN ({placeholders})
                """,
                tuple(chunk_ids),
            )
            rows = cursor.fetchall()
        records_by_id = {
            row["chunk_id"]: self._row_to_chunk_record(row)
            for row in rows
        }
        return tuple(records_by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in records_by_id)

    def count(self) -> int:
        conn = self._conn()
        with self._lock:
            cursor = conn.execute("SELECT COUNT(*) AS count FROM vector_entries")
            row = cursor.fetchone()
        return int(row["count"])

    def _row_to_chunk_record(self, row: sqlite3.Row) -> DocumentChunkRecord:
        return DocumentChunkRecord(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            source_ref=row["source_ref"],
            chunk_index=int(row["chunk_index"]),
            content=row["content"],
            content_hash=row["content_hash"],
            metadata=dict(_json_loads(row["metadata_json"])),
            embedding_model=row["embedding_model"],
            vector=tuple(float(value) for value in _json_loads(row["vector_json"])),
            created_at=row["created_at"],
        )


class MacroRegistryRepository(_SQLiteRepository):
    """Versioned macro registry persistence."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS macro_registry (
                macro_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (macro_name, version)
            );
            """
        )
        conn.commit()

    def upsert(self, macro: Macro, *, created_at: str | None = None) -> None:
        conn = self._conn()
        created_at = created_at or utc_now_iso()
        with self._lock:
            conn.execute(
                """
                INSERT INTO macro_registry (macro_name, version, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(macro_name, version) DO UPDATE SET
                    payload_json = excluded.payload_json
                """,
                (
                    macro.macro_name,
                    macro.version,
                    _json_dumps(macro.to_dict()),
                    created_at,
                ),
            )
            conn.commit()

    def get(self, macro_name: str, *, version: int = 1) -> Macro | None:
        conn = self._conn()
        with self._lock:
            cursor = conn.execute(
                """
                SELECT payload_json
                FROM macro_registry
                WHERE macro_name = ? AND version = ?
                """,
                (macro_name, version),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return Macro.from_dict(_json_loads(row["payload_json"]))

    def list(
        self,
        *,
        macro_names: Sequence[str] | None = None,
        active_only: bool = True,
    ) -> tuple[Macro, ...]:
        conn = self._conn()
        if macro_names:
            placeholders = ", ".join("?" for _ in macro_names)
            query = f"""
                SELECT payload_json
                FROM macro_registry
                WHERE (macro_name, version) IN (
                    SELECT macro_name, MAX(version)
                    FROM macro_registry
                    WHERE macro_name IN ({placeholders})
                    GROUP BY macro_name
                )
                ORDER BY macro_name
            """
            values: tuple[Any, ...] = tuple(macro_names)
        else:
            query = """
                SELECT payload_json
                FROM macro_registry
                WHERE (macro_name, version) IN (
                    SELECT macro_name, MAX(version)
                    FROM macro_registry
                    GROUP BY macro_name
                )
                ORDER BY macro_name
            """
            values = ()
        with self._lock:
            cursor = conn.execute(query, values)
            rows = cursor.fetchall()
        macros = tuple(Macro.from_dict(_json_loads(row["payload_json"])) for row in rows)
        if active_only:
            return tuple(macro for macro in macros if macro.is_active)
        return macros


class OpcodeRegistryRepository(_SQLiteRepository):
    """Versioned opcode lexicon persistence."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS opcode_registry (
                opcode_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                is_active INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (opcode_name, version)
            );
            """
        )
        conn.commit()

    def upsert(self, opcode: OpcodeEntry, *, created_at: str | None = None) -> None:
        conn = self._conn()
        created_at = created_at or utc_now_iso()
        payload = opcode.to_dict()
        with self._lock:
            conn.execute(
                """
                INSERT INTO opcode_registry (
                    opcode_name,
                    version,
                    is_active,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(opcode_name, version) DO UPDATE SET
                    is_active = excluded.is_active,
                    payload_json = excluded.payload_json
                """,
                (
                    opcode.opcode_name,
                    opcode.version,
                    1 if opcode.is_active else 0,
                    _json_dumps(payload),
                    created_at,
                ),
            )
            conn.commit()

    def get(self, opcode_name: str, *, version: int = 1) -> OpcodeEntry | None:
        conn = self._conn()
        with self._lock:
            cursor = conn.execute(
                """
                SELECT payload_json
                FROM opcode_registry
                WHERE opcode_name = ? AND version = ?
                """,
                (opcode_name, version),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return OpcodeEntry.from_dict(_json_loads(row["payload_json"]))

    def list(
        self,
        *,
        opcode_names: Sequence[str] | None = None,
        active_only: bool = True,
    ) -> tuple[OpcodeEntry, ...]:
        conn = self._conn()
        clauses: list[str] = []
        values: list[Any] = []
        if active_only:
            clauses.append("is_active = 1")
        if opcode_names:
            placeholders = ", ".join("?" for _ in opcode_names)
            clauses.append(f"opcode_name IN ({placeholders})")
            values.extend(opcode_names)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT payload_json
            FROM opcode_registry
            {where_clause}
            AND (opcode_name, version) IN (
                SELECT opcode_name, MAX(version)
                FROM opcode_registry
                {'WHERE is_active = 1' if active_only else ''}
                GROUP BY opcode_name
            )
            ORDER BY opcode_name
        """
        if not clauses:
            query = """
                SELECT payload_json
                FROM opcode_registry
                WHERE (opcode_name, version) IN (
                    SELECT opcode_name, MAX(version)
                    FROM opcode_registry
                    GROUP BY opcode_name
                )
                ORDER BY opcode_name
            """
        with self._lock:
            cursor = conn.execute(query, tuple(values))
            rows = cursor.fetchall()
        return tuple(OpcodeEntry.from_dict(_json_loads(row["payload_json"])) for row in rows)


class DecoderRegistryRepository(_SQLiteRepository):
    """Versioned decoder lexicon persistence."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decoder_registry (
                decoder_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                is_active INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (decoder_name, version)
            );
            """
        )
        conn.commit()

    def upsert(self, decoder: DecoderEntry, *, created_at: str | None = None) -> None:
        conn = self._conn()
        created_at = created_at or utc_now_iso()
        payload = decoder.to_dict()
        with self._lock:
            conn.execute(
                """
                INSERT INTO decoder_registry (
                    decoder_name,
                    version,
                    is_active,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(decoder_name, version) DO UPDATE SET
                    is_active = excluded.is_active,
                    payload_json = excluded.payload_json
                """,
                (
                    decoder.decoder_name,
                    decoder.version,
                    1 if decoder.is_active else 0,
                    _json_dumps(payload),
                    created_at,
                ),
            )
            conn.commit()

    def get(self, decoder_name: str, *, version: int = 1) -> DecoderEntry | None:
        conn = self._conn()
        with self._lock:
            cursor = conn.execute(
                """
                SELECT payload_json
                FROM decoder_registry
                WHERE decoder_name = ? AND version = ?
                """,
                (decoder_name, version),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return DecoderEntry.from_dict(_json_loads(row["payload_json"]))

    def list(
        self,
        *,
        decoder_names: Sequence[str] | None = None,
        active_only: bool = True,
    ) -> tuple[DecoderEntry, ...]:
        conn = self._conn()
        clauses: list[str] = []
        values: list[Any] = []
        if active_only:
            clauses.append("is_active = 1")
        if decoder_names:
            placeholders = ", ".join("?" for _ in decoder_names)
            clauses.append(f"decoder_name IN ({placeholders})")
            values.extend(decoder_names)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT payload_json
            FROM decoder_registry
            {where_clause}
            AND (decoder_name, version) IN (
                SELECT decoder_name, MAX(version)
                FROM decoder_registry
                {'WHERE is_active = 1' if active_only else ''}
                GROUP BY decoder_name
            )
            ORDER BY decoder_name
        """
        if not clauses:
            query = """
                SELECT payload_json
                FROM decoder_registry
                WHERE (decoder_name, version) IN (
                    SELECT decoder_name, MAX(version)
                    FROM decoder_registry
                    GROUP BY decoder_name
                )
                ORDER BY decoder_name
            """
        with self._lock:
            cursor = conn.execute(query, tuple(values))
            rows = cursor.fetchall()
        return tuple(DecoderEntry.from_dict(_json_loads(row["payload_json"])) for row in rows)


class SymbolTableRepository(_SQLiteRepository):
    """Append-only symbol-table snapshot persistence."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS symbol_table_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                snapshot_name TEXT NOT NULL,
                is_active INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_symbol_table_task ON symbol_table_snapshots (task_id);"
        )
        conn.commit()

    def append(self, snapshot: SymbolTableSnapshot) -> None:
        conn = self._conn()
        payload = snapshot.to_dict()
        with self._lock:
            conn.execute(
                """
                INSERT INTO symbol_table_snapshots (
                    task_id,
                    snapshot_name,
                    is_active,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    snapshot.task_id,
                    snapshot.snapshot_name,
                    1 if snapshot.is_active else 0,
                    _json_dumps(payload),
                    snapshot.created_at.isoformat(),
                ),
            )
            conn.commit()

    def get_latest(
        self,
        task_id: str,
        *,
        active_only: bool = True,
    ) -> SymbolTableSnapshot | None:
        conn = self._conn()
        clauses = ["task_id = ?"]
        values: list[Any] = [task_id]
        if active_only:
            clauses.append("is_active = 1")
        where_clause = f"WHERE {' AND '.join(clauses)}"
        with self._lock:
            cursor = conn.execute(
                f"""
                SELECT payload_json
                FROM symbol_table_snapshots
                {where_clause}
                ORDER BY id DESC
                LIMIT 1
                """,
                tuple(values),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return SymbolTableSnapshot.from_dict(_json_loads(row["payload_json"]))


class ProofHashRepository(_SQLiteRepository):
    """Append-only proof-hash history persistence."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS proof_hash_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                proof_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proof_hash_task ON proof_hash_history (task_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proof_hash_artifact ON proof_hash_history (artifact_id, artifact_type);"
        )
        conn.commit()

    def append(self, record: ProofHashRecord) -> None:
        conn = self._conn()
        payload = record.to_dict()
        with self._lock:
            conn.execute(
                """
                INSERT INTO proof_hash_history (
                    task_id,
                    artifact_id,
                    artifact_type,
                    proof_hash,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.task_id,
                    record.artifact_id,
                    record.artifact_type,
                    record.proof_hash,
                    _json_dumps(payload),
                    record.created_at.isoformat(),
                ),
            )
            conn.commit()

    def list(
        self,
        *,
        task_id: str | None = None,
        artifact_id: str | None = None,
        artifact_type: str | None = None,
    ) -> tuple[ProofHashRecord, ...]:
        conn = self._conn()
        clauses: list[str] = []
        values: list[Any] = []
        if task_id is not None:
            clauses.append("task_id = ?")
            values.append(task_id)
        if artifact_id is not None:
            clauses.append("artifact_id = ?")
            values.append(artifact_id)
        if artifact_type is not None:
            clauses.append("artifact_type = ?")
            values.append(artifact_type)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            cursor = conn.execute(
                f"""
                SELECT payload_json
                FROM proof_hash_history
                {where_clause}
                ORDER BY id
                """,
                tuple(values),
            )
            rows = cursor.fetchall()
        return tuple(ProofHashRecord.from_dict(_json_loads(row["payload_json"])) for row in rows)


class ReasoningHistoryRepository(_SQLiteRepository):
    """Append-only reasoning and trace history."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reasoning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()

    def append(
        self,
        *,
        task_id: str,
        payload: dict[str, Any],
        created_at: str | None = None,
    ) -> None:
        conn = self._conn()
        created_at = created_at or utc_now_iso()
        with self._lock:
            conn.execute(
                """
                INSERT INTO reasoning_history (task_id, payload_json, created_at)
                VALUES (?, ?, ?)
                """,
                (task_id, _json_dumps(payload), created_at),
            )
            conn.commit()

    def list(self, *, task_id: str | None = None) -> tuple[dict[str, Any], ...]:
        conn = self._conn()
        if task_id is None:
            query = "SELECT payload_json FROM reasoning_history ORDER BY id"
            values: tuple[Any, ...] = ()
        else:
            query = "SELECT payload_json FROM reasoning_history WHERE task_id = ? ORDER BY id"
            values = (task_id,)
        with self._lock:
            cursor = conn.execute(query, values)
            rows = cursor.fetchall()
        return tuple(dict(_json_loads(row["payload_json"])) for row in rows)

    def list_traces(self, *, task_id: str | None = None) -> tuple[CompressedTrace, ...]:
        payloads = self.list(task_id=task_id)
        traces: list[CompressedTrace] = []
        for payload in payloads:
            if "confidence" not in payload:
                continue
            try:
                traces.append(coerce_compressed_trace(payload))
            except (KeyError, TypeError, ValueError):
                continue
        return tuple(traces)


class PerformanceHistoryRepository(_SQLiteRepository):
    """Append-only performance metric history."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()

    def append(self, metric: PerformanceMetric) -> None:
        conn = self._conn()
        with self._lock:
            conn.execute(
                """
                INSERT INTO performance_history (task_id, payload_json, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    metric.task_id,
                    _json_dumps(metric.to_dict()),
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def list(self, *, task_id: str | None = None) -> tuple[PerformanceMetric, ...]:
        conn = self._conn()
        if task_id is None:
            query = "SELECT payload_json FROM performance_history ORDER BY id"
            values: tuple[Any, ...] = ()
        else:
            query = "SELECT payload_json FROM performance_history WHERE task_id = ? ORDER BY id"
            values = (task_id,)
        with self._lock:
            cursor = conn.execute(query, values)
            rows = cursor.fetchall()
        return tuple(PerformanceMetric.from_dict(_json_loads(row["payload_json"])) for row in rows)


class OptimizerReplaySampleRepository(_SQLiteRepository):
    """Replay-grade optimizer input records derived from completed task results."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS optimizer_replay_samples (
                task_id TEXT PRIMARY KEY,
                trace_proof_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_optimizer_replay_samples_proof_hash "
            "ON optimizer_replay_samples (trace_proof_hash);"
        )
        conn.commit()

    def upsert(self, sample: OptimizerReplaySample) -> None:
        conn = self._conn()
        payload_json = _json_dumps(sample.to_dict())
        with self._lock:
            conn.execute(
                """
                INSERT INTO optimizer_replay_samples (
                    task_id,
                    trace_proof_hash,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    trace_proof_hash = excluded.trace_proof_hash,
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at
                """,
                (
                    sample.task_id,
                    sample.trace_proof_hash,
                    payload_json,
                    sample.created_at.isoformat(),
                ),
            )
            conn.commit()

    def list(self, *, task_id: str | None = None) -> tuple[OptimizerReplaySample, ...]:
        conn = self._conn()
        if task_id is None:
            query = "SELECT payload_json FROM optimizer_replay_samples ORDER BY created_at, task_id"
            values: tuple[Any, ...] = ()
        else:
            query = "SELECT payload_json FROM optimizer_replay_samples WHERE task_id = ? ORDER BY created_at, task_id"
            values = (task_id,)
        with self._lock:
            cursor = conn.execute(query, values)
            rows = cursor.fetchall()
        return tuple(OptimizerReplaySample.from_dict(_json_loads(row["payload_json"])) for row in rows)


class OptimizerReplayEvaluationRepository(_SQLiteRepository):
    """Persisted offline replay evaluations for proposal-only optimizer cycles."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS optimizer_replay_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                trace_proof_hash TEXT NOT NULL,
                aggregate_score REAL NOT NULL,
                accepted INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_optimizer_replay_evaluations_proposal "
            "ON optimizer_replay_evaluations (proposal_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_optimizer_replay_evaluations_task "
            "ON optimizer_replay_evaluations (task_id);"
        )
        conn.commit()

    def append_many(self, evaluations: Sequence[OptimizerReplayEvaluation]) -> None:
        if not evaluations:
            return
        conn = self._conn()
        with self._lock:
            conn.executemany(
                """
                INSERT INTO optimizer_replay_evaluations (
                    proposal_id,
                    task_id,
                    trace_proof_hash,
                    aggregate_score,
                    accepted,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        evaluation.proposal_id,
                        evaluation.task_id,
                        evaluation.trace_proof_hash,
                        evaluation.aggregate_score,
                        1 if evaluation.accepted else 0,
                        _json_dumps(evaluation.to_dict()),
                        evaluation.created_at.isoformat(),
                    )
                    for evaluation in evaluations
                ],
            )
            conn.commit()

    def list(
        self,
        *,
        proposal_id: str | None = None,
        task_id: str | None = None,
    ) -> tuple[OptimizerReplayEvaluation, ...]:
        conn = self._conn()
        clauses: list[str] = []
        values: list[str] = []
        if proposal_id is not None:
            clauses.append("proposal_id = ?")
            values.append(proposal_id)
        if task_id is not None:
            clauses.append("task_id = ?")
            values.append(task_id)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            cursor = conn.execute(
                f"""
                SELECT payload_json
                FROM optimizer_replay_evaluations
                {where_clause}
                ORDER BY id
                """,
                tuple(values),
            )
            rows = cursor.fetchall()
        return tuple(OptimizerReplayEvaluation.from_dict(_json_loads(row["payload_json"])) for row in rows)


class OptimizerProposalRecordRepository(_SQLiteRepository):
    """Append-only lifecycle records for optimizer proposals."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS optimizer_proposal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id TEXT NOT NULL,
                proposal_id TEXT NOT NULL,
                lifecycle_stage TEXT NOT NULL,
                activation_eligible INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_optimizer_proposal_records_cycle "
            "ON optimizer_proposal_records (cycle_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_optimizer_proposal_records_proposal "
            "ON optimizer_proposal_records (proposal_id);"
        )
        conn.commit()

    def append_many(self, records: Sequence[OptimizerProposalRecord]) -> None:
        if not records:
            return
        conn = self._conn()
        with self._lock:
            conn.executemany(
                """
                INSERT INTO optimizer_proposal_records (
                    cycle_id,
                    proposal_id,
                    lifecycle_stage,
                    activation_eligible,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.cycle_id,
                        record.proposal_id,
                        record.lifecycle_stage.value,
                        1 if record.activation_eligible else 0,
                        _json_dumps(record.to_dict()),
                        record.created_at.isoformat(),
                    )
                    for record in records
                ],
            )
            conn.commit()

    def list(
        self,
        *,
        cycle_id: str | None = None,
        proposal_id: str | None = None,
    ) -> tuple[OptimizerProposalRecord, ...]:
        conn = self._conn()
        clauses: list[str] = []
        values: list[str] = []
        if cycle_id is not None:
            clauses.append("cycle_id = ?")
            values.append(cycle_id)
        if proposal_id is not None:
            clauses.append("proposal_id = ?")
            values.append(proposal_id)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            cursor = conn.execute(
                f"""
                SELECT payload_json
                FROM optimizer_proposal_records
                {where_clause}
                ORDER BY id
                """,
                tuple(values),
            )
            rows = cursor.fetchall()
        return tuple(OptimizerProposalRecord.from_dict(_json_loads(row["payload_json"])) for row in rows)


class OptimizerActivationRecordRepository(_SQLiteRepository):
    """Append-only activation decisions for optimizer proposals."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS optimizer_activation_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id TEXT NOT NULL,
                proposal_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                lifecycle_stage TEXT NOT NULL,
                activation_applied INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_optimizer_activation_records_cycle "
            "ON optimizer_activation_records (cycle_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_optimizer_activation_records_proposal "
            "ON optimizer_activation_records (proposal_id);"
        )
        conn.commit()

    def append_many(self, records: Sequence[OptimizerActivationRecord]) -> None:
        if not records:
            return
        conn = self._conn()
        with self._lock:
            conn.executemany(
                """
                INSERT INTO optimizer_activation_records (
                    cycle_id,
                    proposal_id,
                    decision,
                    lifecycle_stage,
                    activation_applied,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.cycle_id,
                        record.proposal_id,
                        record.decision.value,
                        record.lifecycle_stage.value,
                        1 if record.activation_applied else 0,
                        _json_dumps(record.to_dict()),
                        record.created_at.isoformat(),
                    )
                    for record in records
                ],
            )
            conn.commit()

    def list(
        self,
        *,
        cycle_id: str | None = None,
        proposal_id: str | None = None,
    ) -> tuple[OptimizerActivationRecord, ...]:
        conn = self._conn()
        clauses: list[str] = []
        values: list[str] = []
        if cycle_id is not None:
            clauses.append("cycle_id = ?")
            values.append(cycle_id)
        if proposal_id is not None:
            clauses.append("proposal_id = ?")
            values.append(proposal_id)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            cursor = conn.execute(
                f"""
                SELECT payload_json
                FROM optimizer_activation_records
                {where_clause}
                ORDER BY id
                """,
                tuple(values),
            )
            rows = cursor.fetchall()
        return tuple(OptimizerActivationRecord.from_dict(_json_loads(row["payload_json"])) for row in rows)


class OptimizerRollbackRecordRepository(_SQLiteRepository):
    """Rollback snapshots prepared before any future optimizer activation."""

    def create_tables(self) -> None:
        conn = self._conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS optimizer_rollback_records (
                rollback_record_id TEXT PRIMARY KEY,
                cycle_id TEXT NOT NULL,
                proposal_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_optimizer_rollback_records_cycle "
            "ON optimizer_rollback_records (cycle_id);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_optimizer_rollback_records_proposal "
            "ON optimizer_rollback_records (proposal_id);"
        )
        conn.commit()

    def append_many(self, records: Sequence[OptimizerRollbackRecord]) -> None:
        if not records:
            return
        conn = self._conn()
        with self._lock:
            conn.executemany(
                """
                INSERT INTO optimizer_rollback_records (
                    rollback_record_id,
                    cycle_id,
                    proposal_id,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(rollback_record_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    created_at = excluded.created_at
                """,
                [
                    (
                        record.rollback_record_id,
                        record.cycle_id,
                        record.proposal_id,
                        _json_dumps(record.to_dict()),
                        record.created_at.isoformat(),
                    )
                    for record in records
                ],
            )
            conn.commit()

    def list(
        self,
        *,
        cycle_id: str | None = None,
        proposal_id: str | None = None,
    ) -> tuple[OptimizerRollbackRecord, ...]:
        conn = self._conn()
        clauses: list[str] = []
        values: list[str] = []
        if cycle_id is not None:
            clauses.append("cycle_id = ?")
            values.append(cycle_id)
        if proposal_id is not None:
            clauses.append("proposal_id = ?")
            values.append(proposal_id)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            cursor = conn.execute(
                f"""
                SELECT payload_json
                FROM optimizer_rollback_records
                {where_clause}
                ORDER BY created_at, rollback_record_id
                """,
                tuple(values),
            )
            rows = cursor.fetchall()
        return tuple(OptimizerRollbackRecord.from_dict(_json_loads(row["payload_json"])) for row in rows)


class StorageManager:
    """Owns local persistence lifecycle and additively exposes specialized repositories."""

    def __init__(self, config: AppConfig = APP_CONFIG, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger("quester.storage")
        self._db_path: Path = config.storage.sqlite_path
        self._logs_dir: Path = config.storage.logs_dir
        self._events_path: Path = self._logs_dir / config.storage.events_log_name
        self._trace_path: Path = self._logs_dir / config.storage.trace_log_name
        self._web_path: Path = self._logs_dir / config.storage.web_log_name
        self._status_path: Path = self._logs_dir / config.storage.status_log_name
        self._vector_store_dir: Path = self._db_path.parent / f"{self._db_path.stem}_chroma"
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._vector_index: VectorIndexAdapter | None = None
        self._foreground_task_count = 0

        self.events = EventLogRepository(self)
        self.kv = KeyValueRepository(self)
        self.tasks = TaskRepository(self)
        self.agent_statuses = AgentStatusRepository(self)
        self.web_evidence = WebEvidenceRepository(self)
        self.documents = SourceDocumentRepository(self)
        self.chunks = ChunkRepository(self)
        self.vectors = VectorEntryRepository(self)
        self.macros = MacroRegistryRepository(self)
        self.opcodes = OpcodeRegistryRepository(self)
        self.decoders = DecoderRegistryRepository(self)
        self.symbol_tables = SymbolTableRepository(self)
        self.proof_hashes = ProofHashRepository(self)
        self.reasoning_history = ReasoningHistoryRepository(self)
        self.performance_history = PerformanceHistoryRepository(self)
        self.optimizer_replay_samples = OptimizerReplaySampleRepository(self)
        self.optimizer_replay_evaluations = OptimizerReplayEvaluationRepository(self)
        self.optimizer_proposal_records = OptimizerProposalRecordRepository(self)
        self.optimizer_activation_records = OptimizerActivationRecordRepository(self)
        self.optimizer_rollback_records = OptimizerRollbackRecordRepository(self)
        self.retrieval = LocalRetrievalService(
            settings=self.config.retrieval,
            lexical_store=self.chunks,
            chunk_store=self.vectors,
        )

    @property
    def vector_index_backend_name(self) -> str:
        if self._vector_index is None:
            return "uninitialized"
        return self._vector_index.backend_name

    async def start(self) -> None:
        """Open database, initialize repositories, and prepare the vector index."""
        if self._conn is not None:
            return
        ensure_directory(self._logs_dir)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

        for repository in (
            self.events,
            self.kv,
            self.tasks,
            self.agent_statuses,
            self.web_evidence,
            self.documents,
            self.chunks,
            self.vectors,
            self.macros,
            self.opcodes,
            self.decoders,
            self.symbol_tables,
            self.proof_hashes,
            self.reasoning_history,
            self.performance_history,
            self.optimizer_replay_samples,
            self.optimizer_replay_evaluations,
            self.optimizer_proposal_records,
            self.optimizer_activation_records,
            self.optimizer_rollback_records,
        ):
            repository.create_tables()

        self._verify_schema_version()
        self._run_integrity_check()
        await self._ensure_default_runtime_lexicon()

        self._vector_index = self._start_vector_index()
        stored_chunk_count = self.vectors.count()
        if self._vector_index.requires_startup_reload(stored_chunk_count=stored_chunk_count):
            stored_chunks = self.vectors.list_chunk_records()
            if stored_chunks:
                self._vector_index.upsert_chunks(stored_chunks)
        self.logger.info(
            "StorageManager started (db=%s, vector_backend=%s).",
            self._db_path,
            self.vector_index_backend_name,
        )

    def _verify_schema_version(self) -> None:
        expected_version = int(self.config.storage.schema_version)
        stored_version = self.kv.get("storage.schema_version")
        if stored_version is None:
            self.kv.set("storage.schema_version", expected_version)
            return
        if int(stored_version) != expected_version:
            raise RuntimeError(
                f"Unsupported storage schema version {stored_version}; expected {expected_version}."
            )

    def _run_integrity_check(self) -> None:
        if not self.config.storage.integrity_check_on_start:
            return
        conn = self._require_conn()
        with self._lock:
            row = conn.execute("PRAGMA integrity_check;").fetchone()
        result = str(row[0]) if row is not None else ""
        if result.lower() != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {result or 'unknown_error'}")

    async def _ensure_default_runtime_lexicon(self) -> None:
        """Bootstrap the minimal built-in opcode and decoder registry for fresh runtimes."""
        default_opcodes = (
            OpcodeEntry(
                opcode_name="lookup",
                description="Resolve one or more evidence handles for the current task.",
                category="retrieval",
                metadata={
                    "arity": 1,
                    "builtin": True,
                    "stable_semantics": True,
                    "emits": "evidence_handle_set",
                },
            ),
            OpcodeEntry(
                opcode_name="bind",
                description="Bind the active evidence set and task symbols into an answer handle.",
                category="state",
                metadata={
                    "arity": "n",
                    "builtin": True,
                    "stable_semantics": True,
                    "commutative_args": True,
                    "emits": "symbol_binding",
                },
            ),
            OpcodeEntry(
                opcode_name="compare",
                description="Compare two or more bound values without changing provenance ownership.",
                category="reasoning",
                metadata={
                    "arity": "n",
                    "builtin": True,
                    "stable_semantics": True,
                    "commutative_args": True,
                },
            ),
            OpcodeEntry(
                opcode_name="infer",
                description="Derive one intermediate claim from the active bindings and evidence set.",
                category="reasoning",
                metadata={"arity": "n", "builtin": True, "stable_semantics": True},
            ),
            OpcodeEntry(
                opcode_name="aggregate",
                description="Merge compatible claims or evidence into one canonical aggregate result.",
                category="reasoning",
                metadata={
                    "arity": "n",
                    "builtin": True,
                    "stable_semantics": True,
                    "commutative_args": True,
                },
            ),
            OpcodeEntry(
                opcode_name="check",
                description="Run a deterministic verification or consistency check over active state.",
                category="verification",
                metadata={"arity": "n", "builtin": True, "stable_semantics": True},
            ),
            OpcodeEntry(
                opcode_name="emit",
                description="Emit the final answer handle through the active decoder.",
                category="output",
                metadata={
                    "arity": 1,
                    "builtin": True,
                    "stable_semantics": True,
                    "emits": "decoder_projection",
                },
            ),
            OpcodeEntry(
                opcode_name="cite",
                description="Attach stable evidence references to the active answer fragment.",
                category="verification",
                metadata={
                    "arity": "n",
                    "builtin": True,
                    "stable_semantics": True,
                    "commutative_args": True,
                },
            ),
            OpcodeEntry(
                opcode_name="confidence_update",
                description="Update confidence metadata for the active claim without changing its meaning.",
                category="verification",
                metadata={
                    "arity": "n",
                    "builtin": True,
                    "stable_semantics": True,
                    "commutative_args": True,
                },
            ),
        )
        default_decoders = (
            DecoderEntry(
                decoder_name="verified_answer",
                template="Answer: {value}",
                metadata={"channel": "final", "builtin": True, "kind": "answer"},
            ),
            DecoderEntry(
                decoder_name="compressed_trace_summary",
                template="Trace summary: {value}",
                metadata={"channel": "debug", "builtin": True, "kind": "summary"},
            ),
        )
        for opcode in default_opcodes:
            if self.opcodes.get(opcode.opcode_name, version=opcode.version) is None:
                self.opcodes.upsert(opcode)
        for decoder in default_decoders:
            if self.decoders.get(decoder.decoder_name, version=decoder.version) is None:
                self.decoders.upsert(decoder)

    async def stop(self) -> None:
        """Close database and vector index cleanly."""
        if self._conn is None:
            return
        with self._lock:
            self._conn.commit()
            self._conn.close()
            self._conn = None
        if self._vector_index is not None:
            try:
                self._vector_index.stop()
            finally:
                self._vector_index = None
        self.logger.info("StorageManager stopped.")

    async def log_event(self, stage: str, payload: dict[str, Any]) -> None:
        """Persist a structured runtime event to SQLite and JSONL."""
        event_payload = dict(payload)
        timestamp = event_payload.pop("timestamp", utc_now_iso())
        event = coerce_runtime_event(
            {
                "stage": stage,
                "payload": event_payload,
                "timestamp": timestamp,
            }
        )
        await self.record_runtime_event(event)

    async def record_runtime_event(self, event: RuntimeEvent | dict[str, Any]) -> None:
        """Persist a typed runtime event and route special-purpose JSONL mirrors."""
        runtime_event = coerce_runtime_event(event)
        self.events.append(runtime_event, events_path=self._events_path)
        if runtime_event.stage.startswith("researcher.web_"):
            _append_jsonl(self._web_path, runtime_event.to_dict())

    async def list_runtime_events(self, *, stage: str | None = None) -> tuple[RuntimeEvent, ...]:
        """Return persisted runtime events in append order."""
        self._require_conn()
        return self.events.list(stage=stage)

    async def record_agent_status(self, status: AgentStatus | dict[str, Any]) -> None:
        """Persist one typed agent-status update to SQLite and JSONL."""
        agent_status = coerce_agent_status(status)
        self.agent_statuses.append(agent_status, status_path=self._status_path)

    async def list_agent_statuses(
        self,
        *,
        task_id: str | None = None,
        component: str | None = None,
    ) -> tuple[AgentStatus, ...]:
        """Return persisted status updates in append order."""
        self._require_conn()
        return self.agent_statuses.list(task_id=task_id, component=component)

    async def record_web_evidence(
        self,
        record: WebEvidenceRecord | dict[str, Any],
    ) -> None:
        """Persist one fetched web-evidence record to SQLite and JSONL."""
        await self.record_web_evidence_batch((record,))

    async def record_web_evidence_batch(
        self,
        records: Sequence[WebEvidenceRecord | dict[str, Any]],
    ) -> None:
        """Persist fetched web evidence with provenance to SQLite and JSONL."""
        normalized = tuple(coerce_web_evidence_record(record) for record in records)
        self.web_evidence.upsert_many(normalized, web_path=self._web_path)

    async def list_web_evidence(
        self,
        *,
        task_id: str | None = None,
        provider: str | None = None,
        source_ref: str | None = None,
    ) -> tuple[WebEvidenceRecord, ...]:
        """Return persisted web evidence in append order."""
        self._require_conn()
        return self.web_evidence.list(task_id=task_id, provider=provider, source_ref=source_ref)

    async def set_kv(self, key: str, value: Any) -> None:
        """Set or replace a JSON-serializable value in kv_store."""
        self.kv.set(key, value)

    async def get_kv(self, key: str) -> Any | None:
        """Read and decode a value from kv_store."""
        return self.kv.get(key)

    async def record_task_result(self, result: TaskResult | dict[str, Any]) -> None:
        """Persist the final typed task result snapshot."""
        task_result = coerce_task_result(result)
        self.tasks.upsert(task_result)
        self.optimizer_replay_samples.upsert(self._build_optimizer_replay_sample(task_result))

    async def get_task_result(self, task_id: str) -> TaskResult | None:
        """Load one persisted task result by task ID."""
        self._require_conn()
        return self.tasks.get(task_id)

    async def count_tasks(self) -> int:
        """Return the current persisted task-run count."""
        self._require_conn()
        return self.tasks.count()

    async def list_task_results(self, *, limit: int | None = None) -> tuple[TaskResult, ...]:
        """Return persisted task results ordered by update time."""
        self._require_conn()
        return self.tasks.list(limit=limit)

    async def enter_foreground_task(self) -> int:
        """Increment the foreground task counter used to freeze activation state."""
        self._require_conn()
        with self._lock:
            self._foreground_task_count += 1
            return self._foreground_task_count

    async def exit_foreground_task(self) -> int:
        """Decrement the foreground task counter without dropping below zero."""
        self._require_conn()
        with self._lock:
            self._foreground_task_count = max(0, self._foreground_task_count - 1)
            return self._foreground_task_count

    async def get_foreground_task_count(self) -> int:
        """Return the number of foreground tasks currently running."""
        self._require_conn()
        with self._lock:
            return self._foreground_task_count

    def _build_optimizer_replay_sample(self, task_result: TaskResult) -> OptimizerReplaySample:
        """Derive one replay-grade optimizer sample from a completed task result."""
        reasoning = task_result.reasoning
        critique = task_result.critique
        context_metadata = dict(reasoning.context_frames[0].metadata) if reasoning.context_frames else {}
        candidate_by_id = {
            candidate_trace.candidate_id: candidate_trace
            for candidate_trace in reasoning.candidate_traces
        }
        selected_candidate_id = self._select_replay_candidate_id(
            reasoning=reasoning,
            context_metadata=context_metadata,
        )
        selected_candidate = candidate_by_id.get(selected_candidate_id)
        candidate_ids = tuple(candidate_by_id) or ((selected_candidate_id,) if selected_candidate_id else ())
        metric = task_result.metrics[-1] if task_result.metrics else None
        return OptimizerReplaySample(
            task_id=task_result.task_id,
            trace_proof_hash=reasoning.proof_hash,
            selected_candidate_id=selected_candidate_id,
            candidate_ids=candidate_ids,
            candidate_trace_count=len(reasoning.candidate_traces),
            selected_strategy=self._select_replay_strategy(
                context_metadata=context_metadata,
                candidate=selected_candidate,
            ),
            selected_verifier=self._select_replay_verifier(
                critique_verifier=critique.verifier_type,
                context_metadata=context_metadata,
                candidate=selected_candidate,
            ),
            selected_candidate_score=self._clamp_score(
                self._select_replay_score(
                    critique_score=critique.candidate_score,
                    context_metadata=context_metadata,
                    candidate=selected_candidate,
                )
            ),
            selected_agreement_score=self._clamp_score(
                selected_candidate.agreement_score if selected_candidate is not None else 0.0
            ),
            selected_evidence_support_score=self._clamp_score(
                selected_candidate.evidence_support_score if selected_candidate is not None else critique.evidence_coverage
            ),
            provenance_coverage=self._clamp_score(critique.provenance_coverage),
            evidence_coverage=self._clamp_score(critique.evidence_coverage),
            final_adjudication=critique.result,
            is_valid=critique.is_valid,
            proof_hash_match=critique.proof_hash_match,
            failure_categories=tuple(dict.fromkeys(str(item) for item in critique.failure_categories if str(item).strip())),
            suggested_repair_actions=tuple(
                dict.fromkeys(str(item) for item in critique.repair_actions if str(item).strip())
            ),
            applied_repair_actions=self._extract_applied_repair_actions(task_result.warnings),
            answer_text=task_result.answer_text or self._select_replay_answer(reasoning=reasoning, context_metadata=context_metadata),
            latency_s=float(metric.time) if metric is not None else 0.0,
            vram_usage_gb=float(metric.vram_usage) if metric is not None else 0.0,
            iterations=int(metric.iterations) if metric is not None else 0,
            created_at=task_result.completed_at,
        )

    def _select_replay_candidate_id(
        self,
        *,
        reasoning: CompressedTrace,
        context_metadata: dict[str, Any],
    ) -> str:
        selected_candidate_id = str(context_metadata.get("cid", "")).strip()
        if selected_candidate_id:
            return selected_candidate_id
        for step in reversed(reasoning.operation_stream):
            candidate_id = str(step.metadata.get("candidate_id", "")).strip()
            if candidate_id:
                return candidate_id
        if reasoning.candidate_traces:
            return reasoning.candidate_traces[0].candidate_id
        return ""

    def _select_replay_answer(
        self,
        *,
        reasoning: CompressedTrace,
        context_metadata: dict[str, Any],
    ) -> str:
        answer_text = str(context_metadata.get("ta", "")).strip()
        if answer_text:
            return answer_text
        for step in reversed(reasoning.operation_stream):
            emitted = str(step.metadata.get("answer_text", "")).strip()
            if emitted:
                return emitted
        if reasoning.candidate_traces:
            return reasoning.candidate_traces[0].answer_text
        return ""

    def _select_replay_strategy(
        self,
        *,
        context_metadata: dict[str, Any],
        candidate,
    ) -> str:
        strategy = str(context_metadata.get("sa", "")).strip()
        if strategy:
            return strategy
        if candidate is not None and candidate.strategy.strip():
            return candidate.strategy
        return ""

    def _select_replay_verifier(
        self,
        *,
        critique_verifier: str,
        context_metadata: dict[str, Any],
        candidate,
    ) -> str:
        verifier = str(critique_verifier).strip()
        if verifier:
            return verifier
        verifier = str(context_metadata.get("sv", "")).strip()
        if verifier:
            return verifier
        if candidate is not None and candidate.verifier_type.strip():
            return candidate.verifier_type
        return ""

    def _select_replay_score(
        self,
        *,
        critique_score: float,
        context_metadata: dict[str, Any],
        candidate,
    ) -> float:
        if candidate is not None:
            return float(candidate.total_score)
        try:
            context_score = float(context_metadata.get("ss", critique_score))
        except (TypeError, ValueError):
            context_score = critique_score
        return float(context_score)

    def _extract_applied_repair_actions(self, warnings: Sequence[str]) -> tuple[str, ...]:
        actions: list[str] = []
        for warning in warnings:
            if not str(warning).startswith("repair_applied:"):
                continue
            payload = str(warning).split(":", 1)[1]
            actions.extend(
                item.strip()
                for item in payload.split(",")
                if item.strip()
            )
        return tuple(dict.fromkeys(actions))

    def _clamp_score(self, value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        return max(0.0, min(1.0, numeric))

    async def record_optimizer_replay_sample(
        self,
        sample: OptimizerReplaySample | dict[str, Any],
    ) -> None:
        """Persist one optimizer replay sample."""
        self.optimizer_replay_samples.upsert(coerce_optimizer_replay_sample(sample))

    async def list_optimizer_replay_samples(
        self,
        *,
        task_id: str | None = None,
    ) -> tuple[OptimizerReplaySample, ...]:
        """Return persisted replay samples in append order."""
        self._require_conn()
        return self.optimizer_replay_samples.list(task_id=task_id)

    async def record_optimizer_replay_evaluations(
        self,
        evaluations: Sequence[OptimizerReplayEvaluation | dict[str, Any]],
    ) -> None:
        """Persist one or more optimizer replay evaluations."""
        normalized = tuple(coerce_optimizer_replay_evaluation(item) for item in evaluations)
        self.optimizer_replay_evaluations.append_many(normalized)

    async def list_optimizer_replay_evaluations(
        self,
        *,
        proposal_id: str | None = None,
        task_id: str | None = None,
    ) -> tuple[OptimizerReplayEvaluation, ...]:
        """Return persisted replay evaluations in append order."""
        self._require_conn()
        return self.optimizer_replay_evaluations.list(proposal_id=proposal_id, task_id=task_id)

    async def record_optimizer_proposal_records(
        self,
        records: Sequence[OptimizerProposalRecord | dict[str, Any]],
    ) -> None:
        """Persist append-only proposal lifecycle records."""
        normalized = tuple(coerce_optimizer_proposal_record(item) for item in records)
        self.optimizer_proposal_records.append_many(normalized)

    async def list_optimizer_proposal_records(
        self,
        *,
        cycle_id: str | None = None,
        proposal_id: str | None = None,
    ) -> tuple[OptimizerProposalRecord, ...]:
        """Return persisted proposal lifecycle records in append order."""
        self._require_conn()
        return self.optimizer_proposal_records.list(cycle_id=cycle_id, proposal_id=proposal_id)

    async def record_optimizer_activation_records(
        self,
        records: Sequence[OptimizerActivationRecord | dict[str, Any]],
    ) -> None:
        """Persist append-only activation decisions."""
        normalized = tuple(coerce_optimizer_activation_record(item) for item in records)
        self.optimizer_activation_records.append_many(normalized)

    async def list_optimizer_activation_records(
        self,
        *,
        cycle_id: str | None = None,
        proposal_id: str | None = None,
    ) -> tuple[OptimizerActivationRecord, ...]:
        """Return persisted activation decisions in append order."""
        self._require_conn()
        return self.optimizer_activation_records.list(cycle_id=cycle_id, proposal_id=proposal_id)

    async def record_optimizer_rollback_records(
        self,
        records: Sequence[OptimizerRollbackRecord | dict[str, Any]],
    ) -> None:
        """Persist rollback snapshots prepared for future live activation."""
        normalized = tuple(coerce_optimizer_rollback_record(item) for item in records)
        self.optimizer_rollback_records.append_many(normalized)

    async def list_optimizer_rollback_records(
        self,
        *,
        cycle_id: str | None = None,
        proposal_id: str | None = None,
    ) -> tuple[OptimizerRollbackRecord, ...]:
        """Return persisted rollback records."""
        self._require_conn()
        return self.optimizer_rollback_records.list(cycle_id=cycle_id, proposal_id=proposal_id)

    async def register_macro(self, macro: Macro) -> None:
        """Persist a versioned macro definition."""
        self.macros.upsert(macro)

    async def get_macro(self, macro_name: str, *, version: int = 1) -> Macro | None:
        """Load one versioned macro definition from the registry."""
        self._require_conn()
        return self.macros.get(macro_name, version=version)

    async def list_macros(
        self,
        *,
        macro_names: Sequence[str] | None = None,
        active_only: bool = True,
    ) -> tuple[Macro, ...]:
        """List latest macro definitions, optionally filtered to a subset of names."""
        self._require_conn()
        return self.macros.list(macro_names=macro_names, active_only=active_only)

    async def register_opcode(self, opcode: OpcodeEntry | dict[str, Any]) -> None:
        """Persist a versioned opcode lexicon entry."""
        self.opcodes.upsert(coerce_opcode_entry(opcode))

    async def get_opcode(self, opcode_name: str, *, version: int = 1) -> OpcodeEntry | None:
        """Load one opcode registry entry by name/version."""
        self._require_conn()
        return self.opcodes.get(opcode_name, version=version)

    async def list_opcodes(
        self,
        *,
        opcode_names: Sequence[str] | None = None,
        active_only: bool = True,
    ) -> tuple[OpcodeEntry, ...]:
        """List latest opcode entries, optionally filtered to an active subset."""
        self._require_conn()
        return self.opcodes.list(opcode_names=opcode_names, active_only=active_only)

    async def register_decoder(self, decoder: DecoderEntry | dict[str, Any]) -> None:
        """Persist a versioned decoder lexicon entry."""
        self.decoders.upsert(coerce_decoder_entry(decoder))

    async def get_decoder(self, decoder_name: str, *, version: int = 1) -> DecoderEntry | None:
        """Load one decoder registry entry by name/version."""
        self._require_conn()
        return self.decoders.get(decoder_name, version=version)

    async def list_decoders(
        self,
        *,
        decoder_names: Sequence[str] | None = None,
        active_only: bool = True,
    ) -> tuple[DecoderEntry, ...]:
        """List latest decoder entries, optionally filtered to an active subset."""
        self._require_conn()
        return self.decoders.list(decoder_names=decoder_names, active_only=active_only)

    async def record_symbol_table_snapshot(
        self,
        snapshot: SymbolTableSnapshot | dict[str, Any],
    ) -> None:
        """Append one symbol-table snapshot for a task."""
        self.symbol_tables.append(coerce_symbol_table_snapshot(snapshot))

    async def get_latest_symbol_table_snapshot(
        self,
        task_id: str,
        *,
        active_only: bool = True,
    ) -> SymbolTableSnapshot | None:
        """Return the latest symbol-table snapshot for a task."""
        self._require_conn()
        return self.symbol_tables.get_latest(task_id, active_only=active_only)

    async def record_proof_hash(self, record: ProofHashRecord | dict[str, Any]) -> None:
        """Append one proof-hash history record."""
        self.proof_hashes.append(coerce_proof_hash_record(record))

    async def list_proof_hashes(
        self,
        *,
        task_id: str | None = None,
        artifact_id: str | None = None,
        artifact_type: str | None = None,
    ) -> tuple[ProofHashRecord, ...]:
        """List persisted proof-hash history records."""
        self._require_conn()
        return self.proof_hashes.list(
            task_id=task_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
        )

    async def load_active_compression_runtime(
        self,
        task_id: str,
        *,
        macro_names: Sequence[str] | None = None,
        opcode_names: Sequence[str] | None = None,
        decoder_names: Sequence[str] | None = None,
    ) -> CompressionRuntimeSubset:
        """Load only the active runtime subset needed for one task."""
        self._require_conn()
        active_macro_names = (
            tuple(self._infer_active_macro_names(task_id))
            if macro_names is None
            else tuple(str(name).strip().lstrip("@") for name in macro_names if str(name).strip())
        )
        return CompressionRuntimeSubset(
            task_id=task_id,
            macros=(
                self.macros.list(macro_names=active_macro_names or None, active_only=True)
                if active_macro_names
                else ()
            ),
            opcodes=self.opcodes.list(opcode_names=opcode_names, active_only=True) if opcode_names else (),
            decoders=self.decoders.list(decoder_names=decoder_names, active_only=True) if decoder_names else (),
            symbol_table=self.symbol_tables.get_latest(task_id, active_only=True),
            proof_hashes=self.proof_hashes.list(task_id=task_id),
        )

    async def export_verified_deep_traces(
        self,
        *,
        export_path: Path | None = None,
        limit: int | None = None,
    ) -> tuple[VerifiedDeepTraceExport, ...]:
        """Return and optionally persist only verified deep-mode traces."""
        self._require_conn()
        exports = tuple(
            self._build_verified_deep_trace_export(result)
            for result in self.tasks.list(limit=limit)
            if self._is_verified_deep_task_result(result)
        )
        if export_path is not None:
            ensure_directory(export_path.parent)
            with export_path.open("w", encoding="utf-8") as handle:
                for export_row in exports:
                    handle.write(_json_dumps(coerce_verified_deep_trace_export(export_row).to_dict()) + "\n")
        return exports

    def _is_verified_deep_task_result(self, result: TaskResult) -> bool:
        reasoning_mode = ""
        if result.reasoning.context_frames:
            reasoning_mode = str(result.reasoning.context_frames[0].metadata.get("rm", "")).strip()
        return (
            reasoning_mode == "deep"
            and bool(result.reasoning.candidate_traces)
            and result.critique.is_valid
            and result.critique.result == CritiqueResult.VALID
            and result.critique.proof_hash_match
        )

    def _build_verified_deep_trace_export(self, result: TaskResult) -> VerifiedDeepTraceExport:
        evidence_items = {
            item.id: item
            for item in (result.evidence.local_results + result.evidence.web_results)
        }
        source_refs = tuple(
            dict.fromkeys(
                evidence_items[item_id].source_ref
                for item_id in result.reasoning.evidence_handles
                if item_id in evidence_items and evidence_items[item_id].source_ref
            )
        )
        answer_text = result.answer_text.strip()
        if not answer_text and result.reasoning.decode_hints:
            answer_text = str(result.reasoning.decode_hints[0].metadata.get("answer_text", "")).strip()
        return VerifiedDeepTraceExport(
            export_id=f"deep_export:{stable_hash(f'{result.task_id}:{result.reasoning.proof_hash}')[:16]}",
            task_id=result.task_id,
            question=result.plan.question,
            answer_text=answer_text or result.plan.question,
            trace_proof_hash=result.reasoning.proof_hash,
            reasoning=result.reasoning,
            critique=result.critique,
            evidence_source_refs=source_refs,
            created_at=result.completed_at,
        )

    async def export_compression_lexicon(self, export_path: Path) -> Path:
        """Write a human-readable registry export without changing the machine-readable source of truth."""
        self._require_conn()
        macros = self.macros.list(active_only=False)
        opcodes = self.opcodes.list(active_only=True)
        decoders = self.decoders.list(active_only=True)
        lines = [
            "# Compression Lexicon Export",
            "",
            f"Generated at: {utc_now_iso()}",
            "",
            "## Macros",
        ]
        if macros:
            for macro in macros:
                parameters = f"({', '.join(macro.parameters)})" if macro.parameters else ""
                status = "active" if macro.is_active else "inactive"
                lines.append(
                    f"- `{macro.macro_name}` v{macro.version} [{status}, {macro.semantic_kind}]{parameters}: {' '.join(macro.expansion)}"
                )
                if macro.opcode_pattern:
                    lines.append(f"  opcodes: {' -> '.join(macro.opcode_pattern)}")
                if macro.invariants:
                    lines.append(f"  invariants: {', '.join(macro.invariants)}")
                if macro.proof_fingerprint:
                    lines.append(f"  proof_fingerprint: `{macro.proof_fingerprint}`")
                if macro.decoder_template:
                    lines.append(f"  decoder: `{macro.decoder_template}`")
        else:
            lines.append("- none")
        lines.extend(("", "## Opcodes"))
        if opcodes:
            for opcode in opcodes:
                lines.append(
                    f"- `{opcode.opcode_name}` v{opcode.version} [{opcode.category}]: {opcode.description}"
                )
                if opcode.metadata:
                    lines.append(f"  metadata: `{_json_dumps(opcode.metadata)}`")
        else:
            lines.append("- none")
        lines.extend(("", "## Decoders"))
        if decoders:
            for decoder in decoders:
                lines.append(
                    f"- `{decoder.decoder_name}` v{decoder.version}: {decoder.template}"
                )
                if decoder.metadata:
                    lines.append(f"  metadata: `{_json_dumps(decoder.metadata)}`")
        else:
            lines.append("- none")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return export_path

    async def export_trace_debug_view(self, task_id: str, export_path: Path) -> Path:
        """Write a compact operator-facing view of the latest persisted trace for one task."""
        self._require_conn()
        traces = self.reasoning_history.list_traces(task_id=task_id)
        if not traces:
            raise ValueError(f"No persisted compressed trace found for task '{task_id}'.")
        trace = traces[-1]
        active_decoders = {
            decoder.decoder_name: decoder
            for decoder in self.decoders.list(active_only=True)
        }
        lines = [
            "# Trace Debug Export",
            "",
            f"Task: `{trace.task_id}`",
            f"Generated at: {utc_now_iso()}",
            f"IR version: `{trace.ir_version or 'legacy'}`",
            f"Proof hash: `{trace.proof_hash or 'none'}`",
            f"Macros used: {', '.join(trace.macros_used) if trace.macros_used else 'none'}",
            "",
            "## Tokens",
            f"- compressed: {' '.join(trace.tokens)}",
            f"- expanded: {' '.join(trace.expanded_preview) if trace.expanded_preview else 'none'}",
            "",
            "## Operation Stream",
        ]
        if trace.operation_stream:
            for step in trace.operation_stream:
                args = ", ".join(step.args) if step.args else "-"
                output_ref = step.output_ref or "-"
                lines.append(f"- `{step.op_id}` `{step.opcode}` args=[{args}] -> `{output_ref}`")
        else:
            lines.append("- none")
        lines.extend(("", "## Decode Hints"))
        if trace.decode_hints:
            for hint in trace.decode_hints:
                decoder_template = active_decoders.get(hint.template)
                rendered = decoder_template.template if decoder_template is not None else hint.template
                lines.append(
                    f"- `{hint.hint_id}` template=`{hint.template}` rendered=`{rendered}` entities={', '.join(hint.entity_ids) or 'none'}"
                )
        else:
            lines.append("- none")
        lines.extend(("", "## Graph Summary"))
        if trace.canonical_graph is not None:
            lines.append(f"- entities: {len(trace.canonical_graph.entities)}")
            lines.append(f"- activities: {len(trace.canonical_graph.activities)}")
            lines.append(f"- agents: {len(trace.canonical_graph.agents)}")
            lines.append(f"- bundles: {len(trace.canonical_graph.bundles)}")
        else:
            lines.append("- none")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return export_path

    async def record_reasoning_trace(self, trace: CompressedTrace) -> None:
        """Persist the full compressed trace and append it to trace JSONL."""
        payload = trace.to_storage_dict()
        self.reasoning_history.append(
            task_id=trace.task_id,
            payload=payload,
            created_at=trace.created_at.isoformat(),
        )
        _append_jsonl(self._trace_path, payload)
        if trace.proof_hash:
            artifact_suffix = stable_hash(
                f"{trace.task_id}|{trace.created_at.isoformat()}|{trace.proof_hash}"
            )[:16]
            self.proof_hashes.append(
                ProofHashRecord(
                    task_id=trace.task_id,
                    artifact_id=f"trace_{artifact_suffix}",
                    artifact_type="compressed_trace",
                    proof_hash=trace.proof_hash,
                    metadata={
                        "ir_version": trace.ir_version,
                        "operation_count": len(trace.operation_stream),
                    },
                    created_at=trace.created_at,
                )
            )

    async def record_reasoning_log(self, log: ReasoningLog) -> None:
        """Persist a smaller optimizer-facing reasoning projection."""
        self.reasoning_history.append(
            task_id=log.task_id,
            payload=log.to_dict(),
            created_at=log.timestamp.isoformat(),
        )

    async def list_reasoning_history(self, *, task_id: str | None = None) -> tuple[dict[str, Any], ...]:
        """Return persisted reasoning payloads in append order."""
        self._require_conn()
        return self.reasoning_history.list(task_id=task_id)

    async def list_reasoning_traces(
        self,
        *,
        task_id: str | None = None,
    ) -> tuple[CompressedTrace, ...]:
        """Return persisted compressed traces in append order."""
        self._require_conn()
        return self.reasoning_history.list_traces(task_id=task_id)

    async def record_performance_metric(self, metric: PerformanceMetric) -> None:
        """Persist one performance metric for a task run."""
        self.performance_history.append(metric)

    async def list_performance_metrics(
        self,
        *,
        task_id: str | None = None,
    ) -> tuple[PerformanceMetric, ...]:
        """Return persisted performance metrics in append order."""
        self._require_conn()
        return self.performance_history.list(task_id=task_id)

    async def ingest_document(
        self,
        *,
        source_ref: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None,
        embed_text: Callable[[str], Awaitable[list[float]]] | None = None,
        embed_document: Callable[[str], Awaitable[list[float]]] | None = None,
        embedding_model_name: str | None = None,
    ) -> SourceDocumentRecord:
        """Persist one source document, its chunks, and current vector payloads."""
        self._require_conn()
        vector_index = self._require_vector_index()
        document_embedder = embed_document or embed_text
        if document_embedder is None:
            raise ValueError("embed_text or embed_document must be provided.")

        normalized_source_ref = source_ref.strip()
        normalized_title = title.strip() or normalized_source_ref
        normalized_content = " ".join(content.split())
        if not normalized_source_ref:
            raise ValueError("source_ref must not be empty.")
        if not normalized_content:
            raise ValueError("content must not be empty.")

        timestamp = utc_now_iso()
        document_id = make_document_id(normalized_source_ref)
        document_metadata = dict(metadata or {})
        effective_embedding_model = (
            (embedding_model_name or self.config.preflight.backends.embedding_model).strip()
            or self.config.preflight.backends.embedding_model
        )
        content_hash = stable_hash(normalized_content)
        document = SourceDocumentRecord(
            document_id=document_id,
            source_ref=normalized_source_ref,
            title=normalized_title,
            content=normalized_content,
            content_hash=content_hash,
            metadata=document_metadata,
            created_at=timestamp,
            updated_at=timestamp,
        )

        chunk_contents = chunk_text(
            normalized_content,
            chunk_size_chars=self.config.retrieval.chunk_size_chars,
            chunk_overlap_chars=self.config.retrieval.chunk_overlap_chars,
            max_chunks=self.config.retrieval.max_chunks_per_document,
        )
        if not chunk_contents:
            raise ValueError("chunk_text returned no chunks for a non-empty document.")

        chunk_records: list[DocumentChunkRecord] = []
        for chunk_index, chunk_content in enumerate(chunk_contents):
            chunk_hash = stable_hash(chunk_content)
            chunk_metadata = {"title": normalized_title, **document_metadata}
            chunk_metadata.setdefault("embedding_role", "document")
            vector = tuple(float(value) for value in await document_embedder(chunk_content))
            chunk_records.append(
                DocumentChunkRecord(
                    chunk_id=make_chunk_id(document_id, chunk_index, chunk_hash),
                    document_id=document_id,
                    source_ref=normalized_source_ref,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    content_hash=chunk_hash,
                    metadata=chunk_metadata,
                    embedding_model=effective_embedding_model,
                    vector=vector,
                    created_at=timestamp,
                )
            )

        self.documents.upsert(document)
        self.chunks.replace_document_chunks(document_id, chunk_records)
        self.vectors.replace_document_vectors(
            document_id,
            chunk_records,
            embedding_backend=self.config.preflight.backends.embedding_backend,
            embedding_model=effective_embedding_model,
            embedding_version="1",
            vector_store_backend=vector_index.backend_name,
        )
        vector_index.delete_document(document_id)
        vector_index.upsert_chunks(tuple(chunk_records))
        return document

    async def search_local_chunks(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        top_k: int,
        metadata_filters: dict[str, Any] | None = None,
        metadata_exclusions: dict[str, Any] | None = None,
        allow_rerank: bool = False,
    ) -> tuple[SearchResult, ...]:
        """Search local chunks using the dedicated bounded hybrid retrieval service."""
        self._require_conn()
        return self.retrieval.search(
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
            vector_index=self._require_vector_index(),
            metadata_filters=metadata_filters,
            metadata_exclusions=metadata_exclusions,
            allow_rerank=allow_rerank,
        )

    async def count_documents(self) -> int:
        """Return the current source-document count."""
        self._require_conn()
        return self.documents.count()

    async def count_chunks(self) -> int:
        """Return the current persisted chunk count."""
        self._require_conn()
        return self.chunks.count()

    def _start_vector_index(self) -> VectorIndexAdapter:
        candidates = [
            self.config.preflight.backends.vector_store_backend,
            self.config.preflight.backends.vector_store_fallback_backend,
        ]
        seen: set[str] = set()
        for backend_name in candidates:
            if backend_name in seen:
                continue
            seen.add(backend_name)
            adapter = self._make_vector_index(backend_name)
            try:
                adapter.start()
                return adapter
            except Exception as exc:  # pragma: no cover - exercised indirectly in tests
                self.logger.warning(
                    "Vector index backend %s failed to start; trying fallback: %s",
                    backend_name,
                    exc,
                )
        raise RuntimeError("No usable vector index backend could be started.")

    def _make_vector_index(self, backend_name: str) -> VectorIndexAdapter:
        if backend_name == "chromadb":
            return ChromaVectorIndex(
                persist_path=self._vector_store_dir,
                collection_name=self.config.preflight.backends.vector_collection_name,
            )
        if backend_name == "simple_inmemory":
            return SimpleInMemoryVectorIndex()
        raise ValueError(f"Unsupported vector index backend: {backend_name}")

    def _require_vector_index(self) -> VectorIndexAdapter:
        if self._vector_index is None:
            raise RuntimeError("StorageManager must be started before retrieval use.")
        return self._vector_index

    def _infer_active_macro_names(self, task_id: str) -> tuple[str, ...]:
        history = self.reasoning_history.list(task_id=task_id)
        for payload in reversed(history):
            raw_macros = payload.get("macros_used")
            if isinstance(raw_macros, list):
                names = tuple(
                    str(item).strip().lstrip("@")
                    for item in raw_macros
                    if str(item).strip()
                )
                if names:
                    return names
        return ()

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("StorageManager must be started before use.")
        return self._conn
