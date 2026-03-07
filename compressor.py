"""Compressor agent scaffold."""

from __future__ import annotations

import logging
from collections import Counter

from config import APP_CONFIG, AppConfig
from data_structures import CompressedTrace, Macro, MacroProposal, ReasoningLog
from model_manager import ModelManager
from prompts import COMPRESSOR_PROMPT


class CompressorAgent:
    """Suggests macro candidates from repeated reasoning tokens."""

    def __init__(self, model_manager: ModelManager, config: AppConfig = APP_CONFIG):
        self.model_manager = model_manager
        self.config = config
        self.logger = logging.getLogger("quester.compressor")
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def propose(
        self,
        trace: CompressedTrace,
        logs: list[ReasoningLog],
    ) -> list[MacroProposal]:
        """Return typed macro suggestions from repeated chain tokens."""
        if not self._started:
            raise RuntimeError("CompressorAgent must be started before use.")
        chain = trace.tokens
        _ = logs
        prompt = f"{COMPRESSOR_PROMPT}\nChainLength: {len(chain)}"
        await self.model_manager.generate(prompt)
        token_counts = Counter(chain)
        suggestions: list[MacroProposal] = []
        for token, count in token_counts.items():
            if count <= 1:
                continue
            sanitized = token.lstrip("@") or "macro_token"
            suggestions.append(
                MacroProposal(
                    proposal_id=f"{trace.task_id}:{sanitized}",
                    macro=Macro(macro_name=sanitized, expansion=(token,), version=1),
                    reason=f"Token '{token}' repeated {count} times in compressed trace.",
                    examples=(token,),
                    simulation_score=min(1.0, count / 3.0),
                    approved=False,
                )
            )
        return suggestions
