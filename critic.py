"""Critic agent wrapper around the typed critique service."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import APP_CONFIG, AppConfig
from critique_service import CritiqueService
from data_structures import (
    CompressionRuntimeSubset,
    CompressedTrace,
    CritiqueReport,
    EvidenceBundle,
    Plan,
    ReasonerCriticHandoff,
    ResourceBudget,
)
from model_manager import ModelManager

if TYPE_CHECKING:
    from storage import StorageManager


class CriticAgent:
    """Validates reasoning output against basic consistency checks."""

    def __init__(
        self,
        model_manager: ModelManager,
        storage: StorageManager | None = None,
        config: AppConfig = APP_CONFIG,
        service: CritiqueService | None = None,
    ):
        self.model_manager = model_manager
        self.storage = storage
        self.config = config
        self.logger = logging.getLogger("quester.critic")
        self.service = service or CritiqueService(model_manager=model_manager, storage=storage, config=config)
        self._started = False

    @property
    def last_runtime_subset(self) -> CompressionRuntimeSubset | None:
        """Expose the last loaded active runtime subset for tests and diagnostics."""
        return self.service.last_runtime_subset

    @property
    def last_handoff(self) -> ReasonerCriticHandoff | None:
        """Expose the last typed handoff used by the critic."""
        return self.service.last_handoff

    async def start(self) -> None:
        """Mark the critic ready to validate typed reasoning traces."""
        if self._started:
            return
        self._started = True

    async def stop(self) -> None:
        """Reject future review requests until the critic is restarted."""
        self._started = False

    async def review(
        self,
        plan: Plan,
        evidence: EvidenceBundle,
        trace: CompressedTrace,
        budget: ResourceBudget,
    ) -> CritiqueReport:
        """Compatibility wrapper that builds the typed Reasoner -> Critic handoff."""
        handoff = ReasonerCriticHandoff.from_inputs(
            plan=plan,
            evidence=evidence,
            trace=trace,
            budget=budget,
            final_text_policy=self.service.final_text_policy,
            implementation_mode=self.service.implementation_mode,
        )
        return await self.review_from_handoff(handoff)

    async def review_from_handoff(self, handoff: ReasonerCriticHandoff) -> CritiqueReport:
        """Return a typed baseline critique report."""
        if not self._started:
            raise RuntimeError("CriticAgent must be started before use.")
        return await self.service.review(handoff)
