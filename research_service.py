"""Shared retrieval and web-fallback logic used by ResearcherAgent."""

from __future__ import annotations

import asyncio

from config import APP_CONFIG, AppConfig
from data_structures import (
    EvidenceBundle,
    EvidenceItem,
    Plan,
    ResourceBudget,
    SourceType,
    WebEvidenceRecord,
)
from model_manager import ModelManager
from prompts import RESEARCHER_PROMPT
from retrieval import stable_hash
from runtime_errors import WebLookupError
from storage import StorageManager
from web_adapter import (
    StubWebSearchAdapter,
    WebSearchAdapter,
    WebSearchResponse,
    WikipediaWebSearchAdapter,
)


class ResearchService:
    """Retrieve evidence while keeping ResearcherAgent thin."""

    _SEED_TIER_LEGACY = "seed"
    output_contract = "evidence_bundle_v1"
    handoff_contract = "research_reasoner_handoff_v1"
    implementation_mode = "deterministic_stub"
    final_text_policy = "post_verification"

    _DEFAULT_SOURCE_DOCUMENTS = (
        {
            "source_ref": "local://seed/retrieval-foundation",
            "title": "Local-first Retrieval Foundation",
            "metadata": {"topic": "retrieval"},
            "content": (
                "Quester.AI uses local-first retrieval. Source documents are chunked into "
                "bounded overlapping passages before embedding so evidence handles stay "
                "stable across re-indexing. Search should favor local chunks, support "
                "metadata filters, and only use web fallback when local evidence quality "
                "is insufficient."
            ),
        },
        {
            "source_ref": "local://seed/runtime-budgeting",
            "title": "Runtime Budgeting Notes",
            "metadata": {"topic": "runtime"},
            "content": (
                "Thinking-time budgets drive retrieval depth, web-query limits, reasoning "
                "passes, and critic depth. Low-resource development targets assume about "
                "four gigabytes of VRAM and eight gigabytes of RAM, so retrieval stays "
                "bounded and reranking is kept small."
            ),
        },
        {
            "source_ref": "local://seed/compatibility",
            "title": "Compatibility Contract",
            "metadata": {"topic": "compatibility"},
            "content": (
                "Public signatures and serialized payload projections must remain stable "
                "while internal storage and retrieval layers evolve. Event logging, key-value "
                "storage, and legacy run_pipeline behavior remain compatibility surfaces."
            ),
        },
        {
            "source_ref": "local://seed/runtime-status",
            "title": "Runtime Status Signals",
            "metadata": {"topic": "runtime"},
            "content": (
                "Runtime status snapshots should report local model health, active jobs, "
                "fallback state, and recent pipeline activity so the dashboard can explain "
                "current behavior without reading raw logs."
            ),
        },
        {
            "source_ref": "local://seed/vector-adapters",
            "title": "Vector Adapter Notes",
            "metadata": {"topic": "retrieval"},
            "content": (
                "The vector adapter should keep Chroma as the primary local index, allow a "
                "simple fallback backend, and preserve chunk IDs across re-embedding so "
                "retrieval results remain auditable."
            ),
        },
        {
            "source_ref": "local://seed/local-telemetry",
            "title": "Local Telemetry Reference",
            "metadata": {"topic": "runtime"},
            "content": (
                "Local telemetry includes runtime events, status updates, and storage-backed "
                "history. Recent status changes should stay queryable from the local store "
                "before any optional web fetch is attempted."
            ),
        },
    )

    def __init__(
        self,
        model_manager: ModelManager,
        storage: StorageManager,
        config: AppConfig = APP_CONFIG,
        web_adapter: WebSearchAdapter | None = None,
    ):
        self.model_manager = model_manager
        self.storage = storage
        self.config = config
        self.web_adapter = web_adapter or self._build_web_adapter()
        self._seeded = False

    async def reset(self) -> None:
        self._seeded = False

    async def research(self, plan: Plan, budget: ResourceBudget) -> EvidenceBundle:
        _ = RESEARCHER_PROMPT
        await self._ensure_seed_corpus()

        question = plan.question
        query_vector = await self.model_manager.embed_query(question)
        local_results = await self._build_local_results(question, budget, query_vector)
        web_results: tuple[EvidenceItem, ...] = ()
        used_web_fallback = False
        if self._should_use_web_fallback(question, budget, local_results):
            used_web_fallback = True
            web_results = await self._build_web_results(
                plan.task_id,
                question,
                budget,
                query_vector,
                local_results,
            )
        evidence = EvidenceBundle(
            task_id=plan.task_id,
            local_results=local_results,
            web_results=web_results,
            used_web_fallback=used_web_fallback,
        )
        await self.storage.log_event(
            "researcher.local_lookup",
            {
                "task_id": plan.task_id,
                "result_count": len(evidence.local_results),
                "web_result_count": len(evidence.web_results),
                "used_web_fallback": evidence.used_web_fallback,
                "retrieval_top_k": budget.retrieval_top_k,
                "max_web_queries": budget.max_web_queries,
                "handoff_contract": self.handoff_contract,
                "final_text_policy": self.final_text_policy,
            },
        )
        return evidence

    async def _build_local_results(
        self,
        question: str,
        budget: ResourceBudget,
        query_vector: list[float],
    ) -> tuple[EvidenceItem, ...]:
        results = await self.storage.search_local_chunks(
            query_text=question,
            query_vector=query_vector,
            top_k=budget.retrieval_top_k,
            metadata_exclusions=self._seed_corpus_exclusions(),
            allow_rerank=self._should_rerank(budget),
        )
        evidence_items: list[EvidenceItem] = []
        for rank, result in enumerate(results, start=1):
            evidence_items.append(
                EvidenceItem(
                    id=result.chunk.chunk_id,
                    content=result.chunk.content,
                    source_type=SourceType.LOCAL,
                    source_ref=result.chunk.source_ref,
                    score=round(result.combined_score, 3),
                    metadata={
                        **result.chunk.metadata,
                        "document_id": result.chunk.document_id,
                        "chunk_index": result.chunk.chunk_index,
                        "rank": rank,
                        "lexical_score": round(result.lexical_score, 3),
                        "vector_score": round(result.vector_score, 3),
                        "combined_score": round(result.combined_score, 3),
                        "rerank_score": round(result.rerank_score, 3),
                        "reranked": result.rerank_score > 0.0,
                        "embedding_model": result.chunk.embedding_model,
                    },
                    vector_preview=result.chunk.vector[:8],
                )
            )
        return tuple(evidence_items)

    async def _build_web_results(
        self,
        task_id: str,
        question: str,
        budget: ResourceBudget,
        query_vector: list[float],
        local_results: tuple[EvidenceItem, ...],
    ) -> tuple[EvidenceItem, ...]:
        result_limit = max(0, min(budget.max_web_queries, self.config.web.max_results_per_query))
        if result_limit <= 0:
            return ()
        reason = self._web_reason(question, local_results)
        response = await self._safe_web_search(question, result_limit)
        vector_preview = tuple(query_vector[-8:]) if len(query_vector) >= 8 else tuple(query_vector)
        local_refs = {item.source_ref for item in local_results}
        evidence_items: list[EvidenceItem] = []
        seen_refs: set[str] = set()
        for rank, result in enumerate(response.results, start=1):
            if result.url in local_refs or result.url in seen_refs:
                continue
            seen_refs.add(result.url)
            evidence_items.append(
                EvidenceItem(
                    id=f"web_{stable_hash(result.url)[:20]}",
                    content=result.content,
                    source_type=SourceType.WEB,
                    source_ref=result.url,
                    score=result.score,
                    metadata={
                        **result.metadata,
                        "title": result.title,
                        "rank": rank,
                        "reason": reason,
                        "provider": response.provider,
                        "degraded": response.degraded,
                        "warnings": list(response.warnings),
                    },
                    vector_preview=vector_preview,
                )
            )
        if evidence_items:
            await self.storage.record_web_evidence_batch(
                tuple(
                    WebEvidenceRecord(
                        task_id=task_id,
                        query=question,
                        provider=response.provider,
                        reason=reason,
                        evidence=item,
                        degraded=response.degraded,
                        warnings=response.warnings,
                        lookup_metadata=response.metadata,
                    )
                    for item in evidence_items
                )
            )
        await self.storage.log_event(
            "researcher.web_lookup",
            {
                "task_id": task_id,
                "query": question,
                "provider": response.provider,
                "reason": reason,
                "requested_results": result_limit,
                "returned_results": len(response.results),
                "persisted_results": len(evidence_items),
                "degraded": response.degraded,
                "warnings": list(response.warnings),
                "source_refs": [result.url for result in response.results],
                "handoff_contract": self.handoff_contract,
                **response.metadata,
            },
        )
        return tuple(evidence_items)

    async def _ensure_seed_corpus(self) -> None:
        if self._seeded or not self._should_seed_default_corpus():
            return
        if await self.storage.count_documents() > 0:
            self._seeded = True
            return
        for document in self._DEFAULT_SOURCE_DOCUMENTS:
            metadata = {
                **document["metadata"],
                "tier": self._SEED_TIER_LEGACY,
                "corpus_tier": self.config.retrieval.seed_corpus_tier,
                "corpus_origin": "seed_demo",
            }
            await self.storage.ingest_document(
                source_ref=document["source_ref"],
                title=document["title"],
                content=document["content"],
                metadata=metadata,
                embed_document=self.model_manager.embed_document,
                embedding_model_name=self.config.preflight.backends.embedding_model,
            )
        self._seeded = True

    def _should_use_web_fallback(
        self,
        question: str,
        budget: ResourceBudget,
        local_results: tuple[EvidenceItem, ...],
    ) -> bool:
        if not self.config.preflight.flags.allow_web_fallback:
            return False
        if budget.max_web_queries <= 1:
            return False
        normalized = question.lower()
        freshness_keywords = ("latest", "current", "today", "recent", "news", "web")
        if any(keyword in normalized for keyword in freshness_keywords):
            return True
        if not local_results:
            return True
        best_score = max(item.score for item in local_results)
        minimum_local_hits = min(2, budget.retrieval_top_k)
        return len(local_results) < minimum_local_hits or best_score < 0.2

    def _web_reason(self, question: str, local_results: tuple[EvidenceItem, ...]) -> str:
        normalized = question.lower()
        freshness_keywords = ("latest", "current", "today", "recent", "news", "web")
        if any(keyword in normalized for keyword in freshness_keywords):
            return "freshness_or_recentness_requested"
        if not local_results:
            return "no_local_evidence_found"
        return "local_evidence_quality_below_threshold"

    def _should_rerank(self, budget: ResourceBudget) -> bool:
        return (
            self.config.retrieval.enable_reranking
            and budget.retrieval_top_k >= self.config.retrieval.rerank_min_budget_top_k
        )

    def _should_seed_default_corpus(self) -> bool:
        if not self.config.retrieval.seed_default_corpus:
            return False
        mode = self.config.retrieval.seed_corpus_mode
        if mode == "disabled":
            return False
        if mode == "always":
            return True
        return self.config.preflight.flags.stub_mode

    def _seed_corpus_exclusions(self) -> dict[str, str] | None:
        if (
            self.config.preflight.flags.stub_mode
            or not self.config.retrieval.exclude_seed_corpus_from_live_queries
        ):
            return None
        return {
            "corpus_tier": self.config.retrieval.seed_corpus_tier,
            "tier": self._SEED_TIER_LEGACY,
        }

    def _build_web_adapter(self) -> WebSearchAdapter:
        if self.config.preflight.flags.stub_mode and not self.config.web.live_web_in_stub_mode:
            return StubWebSearchAdapter(config=self.config)
        if self.config.web.provider == "stub":
            return StubWebSearchAdapter(config=self.config)
        return WikipediaWebSearchAdapter(config=self.config)

    async def _safe_web_search(self, query: str, result_limit: int) -> WebSearchResponse:
        try:
            return await asyncio.wait_for(
                self.web_adapter.search(query, max_results=result_limit),
                timeout=self.config.web.request_timeout_s * max(1, self.config.web.max_retries + 1),
            )
        except asyncio.TimeoutError:
            return WebSearchResponse(
                provider=getattr(self.web_adapter, "provider_name", "unknown_web"),
                query=query,
                results=(),
                degraded=True,
                warnings=("web lookup timed out",),
                metadata={"attempt_count": self.config.web.max_retries + 1},
            )
        except WebLookupError as exc:
            return WebSearchResponse(
                provider=getattr(self.web_adapter, "provider_name", "unknown_web"),
                query=query,
                results=(),
                degraded=True,
                warnings=(str(exc),),
                metadata={"attempt_count": self.config.web.max_retries + 1},
            )
