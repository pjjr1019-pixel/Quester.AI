"""Focused Phase 16 subsystem gate regressions."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from config import APP_CONFIG
from dashboard import DashboardService
from orchestrator import Orchestrator
from runtime_errors import WebLookupTimeoutError


class _FailingWebAdapter:
    provider_name = "failing_web"

    async def search(self, query: str, *, max_results: int):
        _ = (query, max_results)
        raise WebLookupTimeoutError("phase16 forced timeout")


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
            allow_web_fallback=True,
        ),
    )
    storage_cfg = replace(APP_CONFIG.storage, sqlite_path=Path(sqlite_name), logs_dir=Path(logs_name))
    dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
    return replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)


class Phase16OrchestratorGateTests(unittest.IsolatedAsyncioTestCase):
    """Lock the orchestrator gate to explicit stage and status behavior."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase16_orchestrator.sqlite3")
        self.test_logs = Path("test_phase16_orchestrator_logs")
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

    async def test_pipeline_stage_order_and_status_updates_remain_stable(self) -> None:
        result = await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=30)

        events = await self.orchestrator.storage.list_runtime_events()
        stages = [event.stage for event in events]
        required_stage_order = (
            "pipeline.received",
            "pipeline.planner_started",
            "pipeline.planner_done",
            "pipeline.researcher_started",
            "pipeline.researcher_done",
            "pipeline.reasoner_started",
            "pipeline.reasoner_done",
            "pipeline.critic_started",
            "pipeline.critic_done",
            "pipeline.compressor_started",
            "pipeline.compressor_done",
            "pipeline.completed",
        )
        indices = [stages.index(stage) for stage in required_stage_order]

        self.assertEqual(indices, sorted(indices))
        self.assertEqual(result.task_id, next(event.payload["task_id"] for event in events if event.stage == "pipeline.completed"))

        statuses = await self.orchestrator.storage.list_agent_statuses()
        component_states: dict[str, set[str]] = {}
        for status in statuses:
            component_states.setdefault(status.component, set()).add(status.state.value)

        for component in ("planner", "researcher", "reasoner", "critic", "compressor", "orchestrator"):
            self.assertIn("running", component_states.get(component, set()), msg=f"Missing running state for {component}")
            self.assertIn("idle", component_states.get(component, set()), msg=f"Missing idle state for {component}")


class Phase16DashboardGateTests(unittest.IsolatedAsyncioTestCase):
    """Lock the dashboard gate to typed state instead of raw event inspection."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase16_dashboard.sqlite3")
        self.test_logs = Path("test_phase16_dashboard_logs")
        self.orchestrator = Orchestrator(
            config=_build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        )
        await self.orchestrator.start()
        await self.orchestrator._run_dashboard_action(action="examples.load_demo_pack", payload={})

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_dashboard_typed_state_surfaces_status_history_readiness_and_degraded_reason(self) -> None:
        sample = self.orchestrator.phase11_content.get_sample_task("web_runtime_status")
        assert sample is not None

        original_adapter = self.orchestrator.researcher.web_adapter
        self.orchestrator.researcher.web_adapter = _FailingWebAdapter()
        try:
            result = await self.orchestrator.run_task(sample.question, sample.recommended_thinking_minutes)
        finally:
            self.orchestrator.researcher.web_adapter = original_adapter

        await self.orchestrator._run_dashboard_action(action="history.refresh", payload={})
        await self.orchestrator._run_dashboard_action(action="readiness.refresh", payload={})
        state = self.orchestrator.dashboard.app_state_snapshot()

        self.assertFalse(isinstance(state, dict))
        self.assertFalse(isinstance(state.active_task, dict))
        self.assertFalse(isinstance(state.readiness_report, dict))
        self.assertEqual(state.active_task.task_id, result.task_id)
        self.assertTrue(state.active_task.used_web_fallback)
        self.assertTrue(state.statuses)
        self.assertTrue(state.task_history)
        self.assertTrue(state.readiness_report.checks)
        self.assertTrue(
            any(condition.reason == "web_fallback_returned_no_results" for condition in state.recent_conditions)
        )
        self.assertEqual(state.task_history[0].task_id, result.task_id)
        self.assertEqual(state.statuses["researcher"].component, "researcher")
        self.assertIsInstance(self.orchestrator.dashboard, DashboardService)


if __name__ == "__main__":
    unittest.main()
