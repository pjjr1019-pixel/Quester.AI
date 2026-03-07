"""Reasoner agent scaffold."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import APP_CONFIG, AppConfig
from data_structures import (
    CompressedTrace,
    CompressionRuntimeSubset,
    EvidenceBundle,
    Plan,
    ResearchReasonerHandoff,
    ResourceBudget,
)
from model_manager import ModelManager
from reasoning_service import ReasoningService

if TYPE_CHECKING:
    from storage import StorageManager


class ReasonerAgent:
    """Produces a compressed-style reasoning trace placeholder."""

    def __init__(
        self,
        model_manager: ModelManager,
        storage: StorageManager | None = None,
        config: AppConfig = APP_CONFIG,
    ):
        self.model_manager = model_manager
        self.storage = storage
        self.config = config
        self.logger = logging.getLogger("quester.reasoner")
        self.service = ReasoningService(model_manager=model_manager, storage=storage, config=config)
        self._started = False

    @property
    def last_runtime_subset(self) -> CompressionRuntimeSubset | None:
        """Expose the last loaded active runtime subset for tests and diagnostics."""
        return self.service.last_runtime_subset

    @property
    def last_handoff(self) -> ResearchReasonerHandoff | None:
        """Expose the last typed handoff used by the reasoner."""
        return self.service.last_handoff

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def reason(
        self,
        plan: Plan,
        evidence: EvidenceBundle,
        budget: ResourceBudget,
    ) -> CompressedTrace:
        """Compatibility wrapper that builds the typed Researcher -> Reasoner handoff."""
        handoff = ResearchReasonerHandoff.from_inputs(
            plan=plan,
            evidence=evidence,
            budget=budget,
            final_text_policy=self.service.final_text_policy,
            implementation_mode=self.service.implementation_mode,
        )
        return await self.reason_from_handoff(handoff)

    async def reason_from_handoff(self, handoff: ResearchReasonerHandoff) -> CompressedTrace:
        """Build a typed baseline reasoning trace for critic validation."""
        if not self._started:
            raise RuntimeError("ReasonerAgent must be started before use.")
        return await self.service.reason(handoff)
