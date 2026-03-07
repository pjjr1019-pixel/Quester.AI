"""Phase 5 retrieval foundation tests."""

from __future__ import annotations

import json
import asyncio
import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from config import APP_CONFIG, BudgetPolicy
from data_structures import Plan, PlanStep
from model_manager import ModelManager
from researcher import ResearcherAgent
from runtime_errors import WebLookupTimeoutError
from storage import StorageManager
from web_adapter import WebDocument, WebSearchResponse, WikipediaWebSearchAdapter


async def _keyword_embed(text: str) -> list[float]:
    normalized = text.lower()
    terms = (
        "retrieval",
        "chunk",
        "local",
        "web",
        "compatibility",
        "budget",
        "alpha",
        "beta",
    )
    return [float(normalized.count(term)) for term in terms]


async def _rerank_embed(text: str) -> list[float]:
    normalized = text.lower()
    if "runtime healthy status signals" in normalized:
        return [1.0, 1.0, 1.0, 0.0, 1.0]
    if "current runtime status" in normalized:
        return [1.0, 1.0, 0.0, 1.0, 0.0]
    if "resource budget details" in normalized:
        return [0.0, 0.0, 0.0, 1.0, 1.0]
    return [0.0, 0.0, 0.0, 0.0, 0.0]


def _build_plan(question: str) -> Plan:
    budget = BudgetPolicy.from_minutes(30)
    return Plan(
        task_id="phase5-task",
        question=question,
        steps=(PlanStep(step_id="step_1", description="Gather evidence"),),
        required_evidence=("local documents",),
        success_criteria=("return local evidence",),
        budget=budget,
    )


class FakeWebAdapter:
    """Small async adapter used to drive researcher fallback tests deterministically."""

    provider_name = "fake_web"

    def __init__(
        self,
        *,
        response: WebSearchResponse | None = None,
        error: Exception | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.response = response
        self.error = error
        self.delay_s = delay_s
        self.calls = 0

    async def search(self, query: str, *, max_results: int) -> WebSearchResponse:
        self.calls += 1
        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
        return WebSearchResponse(provider=self.provider_name, query=query, results=())


class FakeModelManager:
    """Small async model-manager substitute for policy-focused researcher tests."""

    def __init__(self, *, embedder) -> None:
        self._embedder = embedder

    async def embed_query(self, text: str) -> list[float]:
        return await self._embedder(text)

    async def embed_document(self, text: str) -> list[float]:
        return await self._embedder(text)


class FakePersistentVectorIndex:
    """Small persistent-style vector adapter used to exercise startup reconciliation."""

    backend_name = "fake_persistent"

    def __init__(self, *, indexed_chunk_count: int = 0) -> None:
        self.indexed_chunk_count = indexed_chunk_count
        self.upsert_calls = 0
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def requires_startup_reload(self, *, stored_chunk_count: int) -> bool:
        return stored_chunk_count > 0 and self.indexed_chunk_count != stored_chunk_count

    def upsert_chunks(self, chunks) -> None:
        self.upsert_calls += 1
        self.indexed_chunk_count = len(tuple(chunks))

    def delete_document(self, document_id: str) -> None:
        _ = document_id

    def search(self, query_vector, *, top_k: int, metadata_filters=None):
        _ = (query_vector, top_k, metadata_filters)
        return ()


class StorageManagerRetrievalTests(unittest.IsolatedAsyncioTestCase):
    """Verify retrieval persistence, dedupe, and metadata filtering."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path(f"test_phase5_{self._testMethodName}.sqlite3")
        self.test_logs = Path(f"test_phase5_{self._testMethodName}_logs")
        self.vector_dir = self.test_db.parent / f"{self.test_db.stem}_chroma"
        backends = replace(
            APP_CONFIG.preflight.backends,
            vector_store_backend="simple_inmemory",
            vector_store_fallback_backend="simple_inmemory",
        )
        preflight = replace(
            APP_CONFIG.preflight,
            backends=backends,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
                allow_web_fallback=True,
            ),
        )
        storage = replace(APP_CONFIG.storage, sqlite_path=self.test_db, logs_dir=self.test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        self.test_config = replace(
            APP_CONFIG,
            preflight=preflight,
            storage=storage,
            dashboard=dashboard,
        )
        self.storage = StorageManager(config=self.test_config)
        await self.storage.start()

    async def asyncTearDown(self) -> None:
        await self.storage.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)
        if self.vector_dir.exists():
            shutil.rmtree(self.vector_dir)

    async def test_ingest_document_deduplicates_by_source_identity(self) -> None:
        first = await self.storage.ingest_document(
            source_ref="local://docs/alpha",
            title="Alpha Retrieval Notes",
            content=(
                "Alpha retrieval notes explain how local chunks are embedded and "
                "searched before any optional web lookup."
            ),
            metadata={"topic": "retrieval"},
            embed_text=_keyword_embed,
        )
        initial_chunk_count = await self.storage.count_chunks()
        second = await self.storage.ingest_document(
            source_ref="local://docs/alpha",
            title="Alpha Retrieval Notes",
            content=(
                "Alpha retrieval notes explain how local chunks are embedded and "
                "searched before any optional web lookup."
            ),
            metadata={"topic": "retrieval"},
            embed_text=_keyword_embed,
        )

        self.assertEqual(first.document_id, second.document_id)
        self.assertEqual(await self.storage.count_documents(), 1)
        self.assertEqual(await self.storage.count_chunks(), initial_chunk_count)

    async def test_search_local_chunks_supports_metadata_filters(self) -> None:
        await self.storage.ingest_document(
            source_ref="local://docs/retrieval",
            title="Retrieval Notes",
            content="Retrieval chunks keep local evidence searchable with metadata filters.",
            metadata={"topic": "retrieval"},
            embed_text=_keyword_embed,
        )
        await self.storage.ingest_document(
            source_ref="local://docs/compatibility",
            title="Compatibility Notes",
            content="Compatibility rules preserve signatures and serialized payloads.",
            metadata={"topic": "compatibility"},
            embed_text=_keyword_embed,
        )

        query_vector = await _keyword_embed("retrieval chunks and metadata")
        filtered = await self.storage.search_local_chunks(
            query_text="retrieval chunks and metadata",
            query_vector=query_vector,
            top_k=5,
            metadata_filters={"topic": "retrieval"},
        )
        other = await self.storage.search_local_chunks(
            query_text="compatibility payloads",
            query_vector=await _keyword_embed("compatibility payloads"),
            top_k=5,
            metadata_filters={"topic": "compatibility"},
        )

        self.assertGreater(len(filtered), 0)
        self.assertTrue(all(result.chunk.metadata["topic"] == "retrieval" for result in filtered))
        self.assertGreater(len(other), 0)
        self.assertTrue(all(result.chunk.metadata["topic"] == "compatibility" for result in other))

    async def test_search_local_chunks_avoids_full_vector_record_scans(self) -> None:
        await self.storage.ingest_document(
            source_ref="local://docs/runtime",
            title="Runtime Retrieval",
            content="Runtime retrieval keeps bounded local evidence searchable.",
            metadata={"topic": "runtime"},
            embed_text=_keyword_embed,
        )

        query_vector = await _keyword_embed("runtime retrieval evidence")
        with patch.object(
            self.storage.vectors,
            "list_chunk_records",
            side_effect=AssertionError("full chunk scan should not run during search"),
        ):
            results = await self.storage.search_local_chunks(
                query_text="runtime retrieval evidence",
                query_vector=query_vector,
                top_k=3,
                metadata_filters={"topic": "runtime"},
            )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk.source_ref, "local://docs/runtime")

    async def test_bounded_reranking_reorders_only_top_candidates(self) -> None:
        await self.storage.ingest_document(
            source_ref="local://docs/runtime-signals",
            title="Runtime Signals",
            content="runtime healthy status signals",
            metadata={"topic": "runtime"},
            embed_text=_rerank_embed,
        )
        await self.storage.ingest_document(
            source_ref="local://docs/runtime-status",
            title="Runtime Status",
            content="current runtime status",
            metadata={"topic": "runtime"},
            embed_text=_rerank_embed,
        )
        await self.storage.ingest_document(
            source_ref="local://docs/budget",
            title="Resource Budget",
            content="resource budget details",
            metadata={"topic": "runtime"},
            embed_text=_rerank_embed,
        )

        query_vector = [1.0, 1.0, 1.0, 1.0, 1.0]
        without_rerank = await self.storage.search_local_chunks(
            query_text="runtime status",
            query_vector=query_vector,
            top_k=3,
            metadata_filters={"topic": "runtime"},
            allow_rerank=False,
        )
        with_rerank = await self.storage.search_local_chunks(
            query_text="runtime status",
            query_vector=query_vector,
            top_k=3,
            metadata_filters={"topic": "runtime"},
            allow_rerank=True,
        )

        self.assertEqual(without_rerank[0].chunk.source_ref, "local://docs/runtime-signals")
        self.assertEqual(with_rerank[0].chunk.source_ref, "local://docs/runtime-status")
        self.assertGreater(with_rerank[0].rerank_score, 0.0)
        self.assertEqual(with_rerank[2].chunk.source_ref, without_rerank[2].chunk.source_ref)

    async def test_storage_backfills_fts_index_for_existing_chunks_on_restart(self) -> None:
        await self.storage.ingest_document(
            source_ref="local://docs/backfill",
            title="Lexical Backfill",
            content="bounded lexical candidate generation survives restarts",
            metadata={"topic": "retrieval"},
            embed_text=_keyword_embed,
        )

        conn = self.storage._require_conn()
        conn.execute("DELETE FROM document_chunks_fts")
        conn.commit()

        await self.storage.stop()
        await self.storage.start()

        results = await self.storage.search_local_chunks(
            query_text="lexical candidate generation",
            query_vector=[0.0 for _ in range(8)],
            top_k=3,
            metadata_filters={"topic": "retrieval"},
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk.source_ref, "local://docs/backfill")
        self.assertGreater(results[0].lexical_score, 0.0)

    async def test_storage_reloads_inmemory_vectors_on_restart_for_vector_only_search(self) -> None:
        await self.storage.ingest_document(
            source_ref="local://docs/vector-restart",
            title="Vector Restart",
            content="retrieval chunk local",
            metadata={"topic": "retrieval"},
            embed_text=_keyword_embed,
        )

        await self.storage.stop()
        await self.storage.start()

        results = await self.storage.search_local_chunks(
            query_text="unrelated tokens",
            query_vector=await _keyword_embed("retrieval chunk local"),
            top_k=3,
            metadata_filters={"topic": "retrieval"},
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk.source_ref, "local://docs/vector-restart")
        self.assertGreater(results[0].vector_score, 0.0)


class VectorBackendFallbackTests(unittest.IsolatedAsyncioTestCase):
    """Verify Chroma remains primary but startup falls back safely."""

    async def test_storage_falls_back_when_chroma_startup_fails(self) -> None:
        test_db = Path("test_phase5_fallback.sqlite3")
        test_logs = Path("test_phase5_fallback_logs")
        vector_dir = test_db.parent / f"{test_db.stem}_chroma"
        backends = replace(
            APP_CONFIG.preflight.backends,
            vector_store_backend="chromadb",
            vector_store_fallback_backend="simple_inmemory",
        )
        preflight = replace(
            APP_CONFIG.preflight,
            backends=backends,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
            ),
        )
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=test_db, logs_dir=test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        config = replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)
        storage = StorageManager(config=config)

        try:
            with patch("storage.ChromaVectorIndex.start", side_effect=RuntimeError("chroma unavailable")):
                await storage.start()
            self.assertEqual(storage.vector_index_backend_name, "simple_inmemory")
        finally:
            await storage.stop()
            if test_db.exists():
                test_db.unlink()
            if test_logs.exists():
                shutil.rmtree(test_logs)
            if vector_dir.exists():
                shutil.rmtree(vector_dir)

    async def test_persistent_vector_backend_skips_full_reupsert_when_collection_is_warm(self) -> None:
        test_db = Path("test_phase5_persistent_warm.sqlite3")
        test_logs = Path("test_phase5_persistent_warm_logs")
        vector_dir = test_db.parent / f"{test_db.stem}_chroma"
        backends = replace(
            APP_CONFIG.preflight.backends,
            vector_store_backend="simple_inmemory",
            vector_store_fallback_backend="simple_inmemory",
        )
        preflight = replace(
            APP_CONFIG.preflight,
            backends=backends,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
            ),
        )
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=test_db, logs_dir=test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        config = replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)
        initial_storage = StorageManager(config=config)
        persistent_storage = StorageManager(config=config)

        try:
            await initial_storage.start()
            await initial_storage.ingest_document(
                source_ref="local://docs/persistent",
                title="Persistent Startup",
                content="retrieval startup persistence document",
                metadata={"topic": "retrieval"},
                embed_text=_keyword_embed,
            )
            stored_chunk_count = await initial_storage.count_chunks()
            await initial_storage.stop()

            fake_adapter = FakePersistentVectorIndex(indexed_chunk_count=stored_chunk_count)
            with patch.object(
                persistent_storage,
                "_make_vector_index",
                return_value=fake_adapter,
            ), patch.object(
                persistent_storage.vectors,
                "list_chunk_records",
                side_effect=AssertionError("warm persistent collection should skip full reload"),
            ):
                await persistent_storage.start()

            self.assertEqual(fake_adapter.upsert_calls, 0)
            self.assertEqual(persistent_storage.vector_index_backend_name, "fake_persistent")
        finally:
            await initial_storage.stop()
            await persistent_storage.stop()
            if test_db.exists():
                test_db.unlink()
            if test_logs.exists():
                shutil.rmtree(test_logs)
            if vector_dir.exists():
                shutil.rmtree(vector_dir)

    async def test_persistent_vector_backend_reconciles_when_collection_is_empty(self) -> None:
        test_db = Path("test_phase5_persistent_reconcile.sqlite3")
        test_logs = Path("test_phase5_persistent_reconcile_logs")
        vector_dir = test_db.parent / f"{test_db.stem}_chroma"
        backends = replace(
            APP_CONFIG.preflight.backends,
            vector_store_backend="simple_inmemory",
            vector_store_fallback_backend="simple_inmemory",
        )
        preflight = replace(
            APP_CONFIG.preflight,
            backends=backends,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
            ),
        )
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=test_db, logs_dir=test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        config = replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)
        initial_storage = StorageManager(config=config)
        persistent_storage = StorageManager(config=config)

        try:
            await initial_storage.start()
            await initial_storage.ingest_document(
                source_ref="local://docs/persistent-empty",
                title="Persistent Reconcile",
                content="retrieval startup reconciliation document",
                metadata={"topic": "retrieval"},
                embed_text=_keyword_embed,
            )
            stored_chunk_count = await initial_storage.count_chunks()
            await initial_storage.stop()

            fake_adapter = FakePersistentVectorIndex(indexed_chunk_count=0)
            with patch.object(
                persistent_storage,
                "_make_vector_index",
                return_value=fake_adapter,
            ), patch.object(
                persistent_storage.vectors,
                "list_chunk_records",
                wraps=persistent_storage.vectors.list_chunk_records,
            ) as list_mock:
                await persistent_storage.start()

            self.assertEqual(list_mock.call_count, 1)
            self.assertEqual(fake_adapter.upsert_calls, 1)
            self.assertEqual(fake_adapter.indexed_chunk_count, stored_chunk_count)
        finally:
            await initial_storage.stop()
            await persistent_storage.stop()
            if test_db.exists():
                test_db.unlink()
            if test_logs.exists():
                shutil.rmtree(test_logs)
            if vector_dir.exists():
                shutil.rmtree(vector_dir)


class ResearcherLocalFirstTests(unittest.IsolatedAsyncioTestCase):
    """Verify the researcher uses local retrieval before bounded web fallback."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path(f"test_phase5_researcher_{self._testMethodName}.sqlite3")
        self.test_logs = Path(f"test_phase5_researcher_{self._testMethodName}_logs")
        self.vector_dir = self.test_db.parent / f"{self.test_db.stem}_chroma"
        backends = replace(
            APP_CONFIG.preflight.backends,
            vector_store_backend="simple_inmemory",
            vector_store_fallback_backend="simple_inmemory",
        )
        preflight = replace(
            APP_CONFIG.preflight,
            backends=backends,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
                allow_web_fallback=True,
            ),
        )
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=self.test_db, logs_dir=self.test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        self.test_config = replace(
            APP_CONFIG,
            preflight=preflight,
            storage=storage_cfg,
            dashboard=dashboard,
        )
        self.storage = StorageManager(config=self.test_config)
        self.model_manager = ModelManager(config=self.test_config)
        self.researcher = ResearcherAgent(
            model_manager=self.model_manager,
            storage=self.storage,
            config=self.test_config,
        )
        await self.storage.start()
        await self.model_manager.start()
        await self.researcher.start()

    async def asyncTearDown(self) -> None:
        await self.researcher.stop()
        await self.model_manager.stop()
        await self.storage.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)
        if self.vector_dir.exists():
            shutil.rmtree(self.vector_dir)

    async def test_researcher_uses_seeded_local_corpus_before_web(self) -> None:
        budget = BudgetPolicy.from_minutes(30)
        plan = _build_plan("How does local-first retrieval use chunking and metadata filters?")

        evidence = await self.researcher.research(plan, budget)

        self.assertGreater(len(evidence.local_results), 0)
        self.assertFalse(evidence.used_web_fallback)
        self.assertEqual(len(evidence.web_results), 0)
        self.assertGreaterEqual(await self.storage.count_documents(), 3)
        self.assertTrue(any(item.metadata.get("topic") == "retrieval" for item in evidence.local_results))
        self.assertTrue(
            all(item.metadata.get("corpus_origin") == "seed_demo" for item in evidence.local_results)
        )

    async def test_researcher_uses_query_and_document_embedding_paths(self) -> None:
        budget = BudgetPolicy.from_minutes(30)
        plan = _build_plan("How does local-first retrieval use chunking and metadata filters?")

        with patch.object(
            self.model_manager,
            "embed_query",
            new=AsyncMock(wraps=self.model_manager.embed_query),
        ) as embed_query_mock, patch.object(
            self.model_manager,
            "embed_document",
            new=AsyncMock(wraps=self.model_manager.embed_document),
        ) as embed_document_mock, patch.object(
            self.model_manager,
            "embed",
            new=AsyncMock(wraps=self.model_manager.embed),
        ) as generic_embed_mock:
            evidence = await self.researcher.research(plan, budget)

        self.assertGreater(len(evidence.local_results), 0)
        self.assertGreater(embed_query_mock.await_count, 0)
        self.assertGreater(embed_document_mock.await_count, 0)
        self.assertEqual(generic_embed_mock.await_count, 0)

    async def test_researcher_only_enables_reranking_for_larger_budgets(self) -> None:
        low_budget = BudgetPolicy.from_minutes(5)
        high_budget = BudgetPolicy.from_minutes(30)

        with patch.object(
            self.storage,
            "search_local_chunks",
            new=AsyncMock(wraps=self.storage.search_local_chunks),
        ) as search_mock:
            await self.researcher.research(
                _build_plan("How does local retrieval work?"),
                low_budget,
            )
            await self.researcher.research(
                _build_plan("How does local retrieval work?"),
                high_budget,
            )

        self.assertFalse(search_mock.await_args_list[0].kwargs["allow_rerank"])
        self.assertTrue(search_mock.await_args_list[1].kwargs["allow_rerank"])

    async def test_empty_corpus_returns_empty_local_results_without_crashing(self) -> None:
        await self.researcher.stop()
        await self.model_manager.stop()
        await self.storage.stop()

        empty_db = Path("test_phase5_empty.sqlite3")
        empty_logs = Path("test_phase5_empty_logs")
        empty_vector_dir = empty_db.parent / f"{empty_db.stem}_chroma"
        retrieval_cfg = replace(APP_CONFIG.retrieval, seed_default_corpus=False)
        backends = replace(
            APP_CONFIG.preflight.backends,
            vector_store_backend="simple_inmemory",
            vector_store_fallback_backend="simple_inmemory",
        )
        preflight = replace(
            APP_CONFIG.preflight,
            backends=backends,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
                allow_web_fallback=True,
            ),
        )
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=empty_db, logs_dir=empty_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        config = replace(
            APP_CONFIG,
            preflight=preflight,
            retrieval=retrieval_cfg,
            storage=storage_cfg,
            dashboard=dashboard,
        )
        storage = StorageManager(config=config)
        model_manager = ModelManager(config=config)
        researcher = ResearcherAgent(model_manager=model_manager, storage=storage, config=config)

        try:
            await storage.start()
            await model_manager.start()
            await researcher.start()
            evidence = await researcher.research(
                _build_plan("Explain compatibility surfaces."),
                BudgetPolicy.from_minutes(5),
            )
            self.assertEqual(evidence.local_results, ())
            self.assertEqual(evidence.web_results, ())
            self.assertFalse(evidence.used_web_fallback)
        finally:
            await researcher.stop()
            await model_manager.stop()
            await storage.stop()
            if empty_db.exists():
                empty_db.unlink()
            if empty_logs.exists():
                shutil.rmtree(empty_logs)
            if empty_vector_dir.exists():
                shutil.rmtree(empty_vector_dir)

    async def test_researcher_excludes_seed_corpus_by_default_outside_stub_mode(self) -> None:
        await self.researcher.stop()
        await self.model_manager.stop()
        await self.storage.stop()

        live_db = Path("test_phase5_live_seed_policy.sqlite3")
        live_logs = Path("test_phase5_live_seed_policy_logs")
        live_vector_dir = live_db.parent / f"{live_db.stem}_chroma"
        retrieval_cfg = replace(
            APP_CONFIG.retrieval,
            seed_default_corpus=True,
            seed_corpus_mode="stub_only",
            exclude_seed_corpus_from_live_queries=True,
        )
        backends = replace(
            APP_CONFIG.preflight.backends,
            vector_store_backend="simple_inmemory",
            vector_store_fallback_backend="simple_inmemory",
        )
        preflight = replace(
            APP_CONFIG.preflight,
            backends=backends,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=False,
                enable_self_optimizer=False,
                allow_web_fallback=False,
            ),
        )
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=live_db, logs_dir=live_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        config = replace(
            APP_CONFIG,
            preflight=preflight,
            retrieval=retrieval_cfg,
            storage=storage_cfg,
            dashboard=dashboard,
        )
        storage = StorageManager(config=config)
        researcher = ResearcherAgent(
            model_manager=FakeModelManager(embedder=_keyword_embed),
            storage=storage,
            config=config,
            web_adapter=FakeWebAdapter(),
        )

        try:
            await storage.start()
            await storage.ingest_document(
                source_ref="local://seed/runtime-status",
                title="Seed Runtime Status",
                content="current runtime status from demo seed corpus",
                metadata={
                    "topic": "runtime",
                    "tier": "seed",
                    "corpus_tier": "seed_demo",
                    "corpus_origin": "seed_demo",
                },
                embed_document=_keyword_embed,
            )
            await storage.ingest_document(
                source_ref="local://docs/runtime-status",
                title="User Runtime Status",
                content="current runtime status from user document",
                metadata={"topic": "runtime", "corpus_origin": "user_ingested"},
                embed_document=_keyword_embed,
            )
            await researcher.start()

            evidence = await researcher.research(
                _build_plan("What is the current runtime status?"),
                BudgetPolicy.from_minutes(5),
            )

            self.assertGreater(len(evidence.local_results), 0)
            self.assertTrue(
                all(item.metadata.get("corpus_origin") != "seed_demo" for item in evidence.local_results)
            )
            self.assertTrue(
                any(item.source_ref == "local://docs/runtime-status" for item in evidence.local_results)
            )
            self.assertFalse(
                any(item.source_ref == "local://seed/runtime-status" for item in evidence.local_results)
            )
        finally:
            await researcher.stop()
            await storage.stop()
            if live_db.exists():
                live_db.unlink()
            if live_logs.exists():
                shutil.rmtree(live_logs)
            if live_vector_dir.exists():
                shutil.rmtree(live_vector_dir)


class ResearcherWebFallbackTests(unittest.IsolatedAsyncioTestCase):
    """Verify real web-adapter behavior, degraded mode, and logging."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path(f"test_phase5_web_{self._testMethodName}.sqlite3")
        self.test_logs = Path(f"test_phase5_web_{self._testMethodName}_logs")
        self.vector_dir = self.test_db.parent / f"{self.test_db.stem}_chroma"
        backends = replace(
            APP_CONFIG.preflight.backends,
            vector_store_backend="simple_inmemory",
            vector_store_fallback_backend="simple_inmemory",
        )
        preflight = replace(
            APP_CONFIG.preflight,
            backends=backends,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
                allow_web_fallback=True,
            ),
        )
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=self.test_db, logs_dir=self.test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        self.test_config = replace(
            APP_CONFIG,
            preflight=preflight,
            storage=storage_cfg,
            dashboard=dashboard,
        )
        self.storage = StorageManager(config=self.test_config)
        self.model_manager = ModelManager(config=self.test_config)
        await self.storage.start()
        await self.model_manager.start()

    async def asyncTearDown(self) -> None:
        await self.model_manager.stop()
        await self.storage.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)
        if self.vector_dir.exists():
            shutil.rmtree(self.vector_dir)

    async def test_researcher_uses_web_adapter_results_when_fallback_triggers(self) -> None:
        response = WebSearchResponse(
            provider="fake_web",
            query="What is the latest runtime status?",
            results=(
                WebDocument(
                    title="Runtime status",
                    url="https://example.com/runtime",
                    content="Current runtime status is healthy and bounded.",
                    score=0.77,
                    metadata={"provider": "fake_web", "pageid": "runtime-status"},
                ),
            ),
            metadata={"attempt_count": 1},
        )
        researcher = ResearcherAgent(
            model_manager=self.model_manager,
            storage=self.storage,
            config=self.test_config,
            web_adapter=FakeWebAdapter(response=response),
        )
        await researcher.start()
        self.addAsyncCleanup(researcher.stop)
        plan = _build_plan("What is the latest runtime status?")

        evidence = await researcher.research(
            plan,
            BudgetPolicy.from_minutes(121),
        )

        self.assertTrue(evidence.used_web_fallback)
        self.assertEqual(len(evidence.web_results), 1)
        self.assertEqual(evidence.web_results[0].source_ref, "https://example.com/runtime")
        self.assertEqual(evidence.web_results[0].metadata["provider"], "fake_web")
        self.assertEqual(
            evidence.web_results[0].metadata["reason"],
            "freshness_or_recentness_requested",
        )
        persisted_web = await self.storage.list_web_evidence(task_id=plan.task_id)
        self.assertEqual(len(persisted_web), 1)
        self.assertEqual(persisted_web[0].evidence, evidence.web_results[0])
        self.assertEqual(persisted_web[0].provider, "fake_web")
        self.assertEqual(persisted_web[0].lookup_metadata["attempt_count"], 1)

    async def test_web_timeout_degrades_without_breaking_local_results(self) -> None:
        researcher = ResearcherAgent(
            model_manager=self.model_manager,
            storage=self.storage,
            config=self.test_config,
            web_adapter=FakeWebAdapter(error=WebLookupTimeoutError("timed out")),
        )
        await researcher.start()
        self.addAsyncCleanup(researcher.stop)

        evidence = await researcher.research(
            _build_plan("What is the latest local runtime status?"),
            BudgetPolicy.from_minutes(121),
        )

        self.assertTrue(evidence.used_web_fallback)
        self.assertGreater(len(evidence.local_results), 0)
        self.assertEqual(evidence.web_results, ())
        self.assertEqual(await self.storage.list_web_evidence(task_id="phase5-task"), ())

        events_path = self.test_logs / self.test_config.storage.events_log_name
        self.assertTrue(events_path.exists())
        with events_path.open("r", encoding="utf-8") as handle:
            web_events = [
                json.loads(line)
                for line in handle
                if '"stage": "researcher.web_lookup"' in line
            ]
        self.assertGreaterEqual(len(web_events), 1)
        self.assertTrue(web_events[-1]["payload"]["degraded"])
        self.assertEqual(web_events[-1]["payload"]["returned_results"], 0)


class WikipediaWebAdapterTests(unittest.IsolatedAsyncioTestCase):
    """Verify bounded MediaWiki lookup behavior stays deterministic under mocks."""

    async def test_wikipedia_adapter_deduplicates_duplicate_hits(self) -> None:
        adapter = WikipediaWebSearchAdapter(config=APP_CONFIG)
        search_payload = {
            "query": {
                "search": [
                    {"pageid": 1, "title": "Runtime status", "snippet": "Current <span>runtime</span>"},
                    {"pageid": 1, "title": "Runtime status", "snippet": "Duplicate result"},
                    {"pageid": 2, "title": "Resource budget", "snippet": "Budget details"},
                ]
            }
        }
        extract_payload = {
            "query": {
                "pages": {
                    "1": {
                        "pageid": 1,
                        "fullurl": "https://en.wikipedia.org/wiki/Runtime_status",
                        "extract": "Runtime status extract",
                    },
                    "2": {
                        "pageid": 2,
                        "fullurl": "https://en.wikipedia.org/wiki/Resource_budget",
                        "extract": "Resource budget extract",
                    },
                }
            }
        }

        with patch.object(adapter, "_request_json", side_effect=[search_payload, extract_payload]):
            response = await adapter.search("runtime", max_results=5)

        self.assertFalse(response.degraded)
        self.assertEqual(len(response.results), 2)
        self.assertEqual(
            [result.url for result in response.results],
            [
                "https://en.wikipedia.org/wiki/Runtime_status",
                "https://en.wikipedia.org/wiki/Resource_budget",
            ],
        )

    async def test_wikipedia_adapter_returns_degraded_response_on_timeout(self) -> None:
        adapter = WikipediaWebSearchAdapter(config=APP_CONFIG)

        with patch.object(adapter, "_request_json", side_effect=WebLookupTimeoutError("timed out")):
            response = await adapter.search("runtime", max_results=3)

        self.assertTrue(response.degraded)
        self.assertEqual(response.results, ())
        self.assertTrue(any("timed out" in warning for warning in response.warnings))


if __name__ == "__main__":
    unittest.main()
