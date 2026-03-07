"""Researcher agent with local-first retrieval and bounded web fallback."""

from __future__ import annotations

import logging

from config import APP_CONFIG, AppConfig
from data_structures import EvidenceBundle, Plan, ResourceBudget
from model_manager import ModelManager
from research_service import ResearchService
from storage import StorageManager
from web_adapter import WebSearchAdapter


class ResearcherAgent:
    """Retrieves evidence from local storage before optional web fallback."""

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
        self.logger = logging.getLogger("quester.researcher")
        self.service = ResearchService(
            model_manager=model_manager,
            storage=storage,
            config=config,
            web_adapter=web_adapter,
        )
        self._started = False

    @property
    def web_adapter(self) -> WebSearchAdapter:
        """Expose the active web adapter for compatibility with tests and callers."""
        return self.service.web_adapter

    @web_adapter.setter
    def web_adapter(self, value: WebSearchAdapter) -> None:
        self.service.web_adapter = value

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

    async def stop(self) -> None:
        await self.service.reset()
        self._started = False

    async def research(self, plan: Plan, budget: ResourceBudget) -> EvidenceBundle:
        """Return a typed local-first evidence bundle."""
        if not self._started:
            raise RuntimeError("ResearcherAgent must be started before use.")
        return await self.service.research(plan, budget)
