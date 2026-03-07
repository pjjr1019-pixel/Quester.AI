"""Planner agent scaffold."""

from __future__ import annotations

import logging

from config import APP_CONFIG, AppConfig
from data_structures import Plan, ResourceBudget
from model_manager import ModelManager
from planner_service import PlannerService


class PlannerAgent:
    """Builds a task plan from the user question."""

    def __init__(self, model_manager: ModelManager, config: AppConfig = APP_CONFIG):
        self.model_manager = model_manager
        self.config = config
        self.logger = logging.getLogger("quester.planner")
        self.service = PlannerService(model_manager=model_manager, config=config)
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def plan(self, question: str, budget: ResourceBudget) -> Plan:
        """Return a typed plan for downstream pipeline stages."""
        if not self._started:
            raise RuntimeError("PlannerAgent must be started before use.")
        return await self.service.plan(question, budget)
