"""Phase 13 async-loop and bounded-queue safety tests."""

from __future__ import annotations

import queue
import shutil
import unittest
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from config import APP_CONFIG
from dashboard import DashboardService
from orchestrator import Orchestrator


class _FakeTextWidget:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def insert(self, index: str, text: str) -> None:
        _ = index
        self.lines.append(text)

    def see(self, index: str) -> None:
        _ = index


class _AlwaysFullQueue:
    def put_nowait(self, item: object) -> None:
        _ = item
        raise queue.Full

    def get_nowait(self) -> object:
        raise queue.Empty


def _build_test_config(*, sqlite_name: str, logs_name: str):
    backends = replace(
        APP_CONFIG.preflight.backends,
        vector_store_backend="simple_inmemory",
        vector_store_fallback_backend="simple_inmemory",
    )
    preflight = replace(
        APP_CONFIG.preflight,
        backends=backends,
        flags=replace(
            APP_CONFIG.preflight.flags,
            stub_mode=True,
            enable_self_optimizer=False,
        ),
    )
    storage_cfg = replace(APP_CONFIG.storage, sqlite_path=Path(sqlite_name), logs_dir=Path(logs_name))
    dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
    return replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)


class DashboardAsyncSafetyTests(unittest.TestCase):
    def test_dashboard_drain_queue_is_bounded_per_poll(self) -> None:
        config = replace(
            APP_CONFIG,
            dashboard=replace(APP_CONFIG.dashboard, enable_ui=False),
            concurrency=replace(APP_CONFIG.concurrency, dashboard_queue_maxsize=80),
        )
        dashboard = DashboardService(config=config)
        debug = _FakeTextWidget()
        dashboard._debug_text = debug

        for index in range(80):
            dashboard.publish_event({"stage": "pipeline.started", "task_id": f"task-{index}"})

        dashboard._drain_queue()

        self.assertEqual(len(debug.lines), dashboard._max_debug_events_per_poll)
        self.assertEqual(dashboard._events.qsize(), 80 - dashboard._max_debug_events_per_poll)

    def test_dashboard_overflow_race_drops_current_event_without_raising(self) -> None:
        config = replace(APP_CONFIG, dashboard=replace(APP_CONFIG.dashboard, enable_ui=False))
        dashboard = DashboardService(config=config)
        dashboard._events = _AlwaysFullQueue()  # type: ignore[assignment]

        dashboard.publish_event(
            {
                "stage": "pipeline.completed",
                "task_id": "task-1",
                "answer_text": "Verified answer: 4.",
                "citation_refs": ["local://ev-1"],
                "warning_count": 0,
                "candidate_trace_count": 1,
                "critique_result": "valid",
            }
        )

        state = dashboard.app_state_snapshot()

        self.assertEqual(dashboard._dropped_events, 1)
        self.assertEqual(state.active_task.task_id, "task-1")
        self.assertEqual(state.active_task.answer_text, "Verified answer: 4.")
        self.assertEqual(state.active_task.citation_refs, ("local://ev-1",))


class OrchestratorDashboardFutureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase13_async.sqlite3")
        self.test_logs = Path("test_phase13_async_logs")
        self.orchestrator = Orchestrator(
            config=_build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        )
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_stop_cancels_pending_dashboard_futures_and_detaches_controller(self) -> None:
        future: ConcurrentFuture[object] = ConcurrentFuture()

        def fake_run_coroutine_threadsafe(coroutine, loop):
            _ = loop
            coroutine.close()
            return future

        with patch("orchestrator.asyncio.run_coroutine_threadsafe", side_effect=fake_run_coroutine_threadsafe):
            self.orchestrator._submit_dashboard_task_request("What is 2 + 2?", 30)

        self.assertIn(future, self.orchestrator._dashboard_futures)

        with patch.object(self.orchestrator.logger, "warning") as warning_mock:
            await self.orchestrator.stop()

        self.assertTrue(future.cancelled())
        self.assertEqual(self.orchestrator._dashboard_futures, set())
        self.assertFalse(self.orchestrator.dashboard.request_task_submission("Should fail", 30))
        warning_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
