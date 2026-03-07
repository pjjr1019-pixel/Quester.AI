"""Local dashboard service with optional Tkinter UI."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

from config import APP_CONFIG, AppConfig
from utils import utc_now_iso

try:
    import tkinter as tk
except Exception:  # pragma: no cover - environment-specific
    tk = None


class DashboardService:
    """Consumes events and optionally renders them in a local Tkinter window."""

    def __init__(self, config: AppConfig = APP_CONFIG):
        self.config = config
        self.logger = logging.getLogger("quester.dashboard")
        self._events: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=config.concurrency.dashboard_queue_maxsize
        )
        self._started = False
        self._headless = not config.dashboard.enable_ui
        self._ui_thread: threading.Thread | None = None
        self._root: tk.Tk | None = None
        self._text: tk.Text | None = None
        self._stop_flag = threading.Event()
        self._dropped_events = 0

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    async def start(self) -> None:
        """Start the dashboard service."""
        if self._started:
            return
        self._started = True
        if self._headless or tk is None:
            self.logger.info("Dashboard running in headless mode.")
            return
        self._ui_thread = threading.Thread(target=self._run_ui, name="dashboard-ui", daemon=True)
        self._ui_thread.start()
        self.logger.info("Dashboard UI thread started.")

    async def stop(self) -> None:
        """Stop the dashboard service."""
        if not self._started:
            return
        self._started = False
        self._stop_flag.set()
        if self._root is not None:
            try:
                self._root.after(0, self._root.quit)
            except Exception:  # pragma: no cover - defensive
                pass
        if self._ui_thread and self._ui_thread.is_alive():
            self._ui_thread.join(timeout=self.config.preflight.flags.shutdown_timeout_s)
        self.logger.info("Dashboard stopped.")

    def publish_event(self, event: dict[str, Any]) -> None:
        """Publish event to queue for UI and diagnostics."""
        if "timestamp" not in event:
            event = dict(event)
            event["timestamp"] = utc_now_iso()
        try:
            self._events.put_nowait(event)
        except queue.Full:
            try:
                _ = self._events.get_nowait()
            except queue.Empty:
                pass
            self._dropped_events += 1
            overflow_event = dict(event)
            overflow_event["dropped_events"] = self._dropped_events
            overflow_event["queue_overflow"] = "evicted_oldest"
            self._events.put_nowait(overflow_event)

    def _run_ui(self) -> None:
        try:
            self._root = tk.Tk()
        except Exception as exc:  # pragma: no cover - depends on display availability
            self.logger.warning("Falling back to headless dashboard mode: %s", exc)
            self._headless = True
            return
        self._root.title(self.config.dashboard.window_title)
        self._text = tk.Text(self._root, height=30, width=120, wrap="word")
        self._text.pack(fill=tk.BOTH, expand=True)
        self._schedule_poll()
        self._root.mainloop()

    def _schedule_poll(self) -> None:
        if self._root is None:
            return
        self._drain_queue()
        if not self._stop_flag.is_set():
            self._root.after(self.config.dashboard.refresh_interval_ms, self._schedule_poll)

    def _drain_queue(self) -> None:
        if self._text is None:
            return
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            self._text.insert("end", f"{event.get('timestamp')} | {event}\n")
            self._text.see("end")
