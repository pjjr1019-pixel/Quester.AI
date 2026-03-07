"""Shared utility helpers used across runtime modules."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import LoggingSettings


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def ensure_directory(path: Path) -> None:
    """Create a directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def configure_logging(settings: LoggingSettings) -> logging.Logger:
    """Configure process-wide logging and return the app logger."""
    logging.basicConfig(level=getattr(logging, settings.level), format=settings.fmt)
    return logging.getLogger("quester")


async def cancel_task(task: asyncio.Task[Any] | None, timeout_s: float) -> None:
    """Cancel and await a task without leaking cancellation warnings."""
    if task is None or task.done():
        return
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=timeout_s)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        return

