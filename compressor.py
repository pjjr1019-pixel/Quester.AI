"""Compressor agent wrapper around the macro-proposal service."""

from __future__ import annotations

import logging

from compression_service import CompressionService
from config import APP_CONFIG, AppConfig
from data_structures import CompressedTrace, MacroProposal, ReasoningLog
from model_manager import ModelManager


class CompressorAgent:
    """Suggests macro candidates from repeated reasoning tokens."""

    def __init__(
        self,
        model_manager: ModelManager,
        config: AppConfig = APP_CONFIG,
        service: CompressionService | None = None,
    ):
        self.model_manager = model_manager
        self.config = config
        self.logger = logging.getLogger("quester.compressor")
        self.service = service or CompressionService(model_manager=model_manager, config=config)
        self._started = False

    async def start(self) -> None:
        """Mark the compressor ready to build macro proposals."""
        if self._started:
            return
        self._started = True

    async def stop(self) -> None:
        """Reject future compression requests until the compressor is restarted."""
        self._started = False

    async def propose(
        self,
        trace: CompressedTrace,
        logs: list[ReasoningLog],
    ) -> list[MacroProposal]:
        """Return typed macro suggestions from repeated chain tokens."""
        if not self._started:
            raise RuntimeError("CompressorAgent must be started before use.")
        return await self.service.propose(trace, logs)
