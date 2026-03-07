"""Background self-optimizer scaffold."""

from __future__ import annotations

import asyncio
import logging

from config import APP_CONFIG, AppConfig
from data_structures import Macro, MacroProposal
from storage import StorageManager
from utils import cancel_task, utc_now_iso


class SelfOptimizer:
    """Runs non-blocking optimization cycles while the app is live."""

    def __init__(self, storage: StorageManager, config: AppConfig = APP_CONFIG):
        self.storage = storage
        self.config = config
        self.logger = logging.getLogger("quester.self_optimizer")
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start optimizer loop if enabled."""
        if self._started:
            return
        if not self.config.preflight.flags.enable_self_optimizer:
            self.logger.info("SelfOptimizer disabled by config.")
            return
        self._started = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="self-optimizer-loop")
        self.logger.info("SelfOptimizer started.")

    async def stop(self) -> None:
        """Stop optimizer loop cleanly."""
        self._stop_event.set()
        await cancel_task(self._task, timeout_s=self.config.preflight.flags.shutdown_timeout_s)
        self._task = None
        self._started = False
        self.logger.info("SelfOptimizer stopped.")

    async def run_cycle(self) -> list[MacroProposal]:
        """Execute one optimizer cycle."""
        proposal = MacroProposal(
            proposal_id=f"proposal-{utc_now_iso()}",
            macro=Macro(macro_name="optimizer_stub_macro", expansion=("@compose_answer",), version=1),
            reason="Phase 2 placeholder cycle completed.",
            examples=("@compose_answer",),
            simulation_score=0.25,
            approved=False,
        )
        await self.storage.log_event("self_optimizer.cycle", proposal.to_dict())
        return [proposal]

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.run_cycle()
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                self.logger.exception("SelfOptimizer cycle failed: %s", exc)
            await asyncio.sleep(self.config.self_optimizer.cycle_interval_s)
