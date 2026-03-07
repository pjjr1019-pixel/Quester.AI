"""Researcher agent scaffold."""

from __future__ import annotations

import logging

from config import APP_CONFIG, AppConfig
from data_structures import EvidenceBundle, EvidenceItem, Plan, ResourceBudget, SourceType
from model_manager import ModelManager
from prompts import RESEARCHER_PROMPT
from storage import StorageManager


class ResearcherAgent:
    """Retrieves evidence with local-first behavior."""

    def __init__(
        self,
        model_manager: ModelManager,
        storage: StorageManager,
        config: AppConfig = APP_CONFIG,
    ):
        self.model_manager = model_manager
        self.storage = storage
        self.config = config
        self.logger = logging.getLogger("quester.researcher")
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def research(self, plan: Plan, budget: ResourceBudget) -> EvidenceBundle:
        """Return a typed local-first evidence bundle."""
        if not self._started:
            raise RuntimeError("ResearcherAgent must be started before use.")
        question = plan.question
        _ = RESEARCHER_PROMPT
        query_vector = await self.model_manager.embed(question)
        local_item = EvidenceItem(
            id="local_stub_1",
            content=f"Stub local evidence for question: {question}",
            source_type=SourceType.LOCAL,
            source_ref="local://stub",
            score=0.7,
            metadata={
                "phase": 2,
                "retrieval_top_k": budget.retrieval_top_k,
                "max_web_queries": budget.max_web_queries,
            },
            vector_preview=tuple(query_vector[:8]),
        )
        evidence = EvidenceBundle(
            task_id=plan.task_id,
            local_results=(local_item,),
            web_results=(),
            used_web_fallback=False,
        )
        await self.storage.log_event(
            "researcher.local_lookup",
            {"task_id": plan.task_id, "result_count": len(evidence.local_results)},
        )
        return evidence
