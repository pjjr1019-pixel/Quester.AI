"""Phase 12 acceptance tests for dashboard state, history, and knowledge UX."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from config import APP_CONFIG
from data_structures import UserSettingsProfile
from dashboard import DashboardService
from orchestrator import Orchestrator


def _build_test_config(sqlite_name: str, logs_name: str):
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


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


def _attach_fake_settings_form(dashboard: DashboardService) -> None:
    dashboard._profile_name_var = _FakeVar("default")
    dashboard._generation_backend_var = _FakeVar("ollama")
    dashboard._embedding_backend_var = _FakeVar("sentence_transformers")
    dashboard._vector_store_var = _FakeVar("chromadb")
    dashboard._web_provider_var = _FakeVar("wikipedia")
    dashboard._reasoning_mode_var = _FakeVar("auto")
    dashboard._thinking_minutes_var = _FakeVar(30)
    dashboard._thinking_label_var = _FakeVar("30 minutes")
    dashboard._allow_web_fallback_var = _FakeVar(True)
    dashboard._enable_self_optimizer_var = _FakeVar(False)
    dashboard._reranking_var = _FakeVar(True)
    dashboard._reranker_role_enabled_var = _FakeVar(False)
    dashboard._speech_to_text_role_enabled_var = _FakeVar(False)
    dashboard._vad_role_enabled_var = _FakeVar(False)
    dashboard._long_horizon_enabled_var = _FakeVar(False)
    dashboard._long_horizon_minutes_var = _FakeVar("120")
    dashboard._optimizer_policy_var = _FakeVar("proposal_only")
    dashboard._optimizer_replay_limit_var = _FakeVar("64")
    dashboard._show_debug_pane_var = _FakeVar(True)
    dashboard._desktop_enabled_var = _FakeVar(False)
    dashboard._desktop_approval_policy_var = _FakeVar("approve_risky_only")
    dashboard._observation_tier_var = _FakeVar("screenshot_on_demand")
    dashboard._cloud_mode_var = _FakeVar("auxiliary_only")
    dashboard._log_runtime_events_var = _FakeVar(True)
    dashboard._allow_cloud_content_var = _FakeVar(False)
    dashboard._log_level_var = _FakeVar("INFO")


class Phase12GuiAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    """Validate the typed dashboard shell in headless acceptance scenarios."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase12_gui.sqlite3")
        self.test_logs = Path("test_phase12_gui_logs")
        self.config = _build_test_config(str(self.test_db), str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_settings_form_round_trip_persists_and_reloads_typed_profile(self) -> None:
        dashboard = self.orchestrator.dashboard
        current = dashboard.app_state_snapshot().user_settings
        _attach_fake_settings_form(dashboard)
        dashboard.apply_user_settings(current)

        dashboard._profile_name_var.set("acceptance")
        dashboard._vector_store_var.set("simple_inmemory")
        dashboard._reasoning_mode_var.set("deep")
        dashboard._thinking_minutes_var.set(45)
        dashboard._allow_web_fallback_var.set(False)
        dashboard._reranking_var.set(False)
        dashboard._reranker_role_enabled_var.set(True)
        dashboard._speech_to_text_role_enabled_var.set(True)
        dashboard._vad_role_enabled_var.set(True)
        dashboard._long_horizon_enabled_var.set(True)
        dashboard._long_horizon_minutes_var.set("180")
        dashboard._show_debug_pane_var.set(False)
        dashboard._desktop_enabled_var.set(True)
        dashboard._desktop_approval_policy_var.set("manual_only")
        dashboard._observation_tier_var.set("vision_on_step")
        dashboard._cloud_mode_var.set("disabled")
        dashboard._log_level_var.set("DEBUG")

        gathered = dashboard._gather_settings_from_form()
        await self.orchestrator.storage.save_user_settings_profile(gathered)
        loaded = await self.orchestrator.storage.load_user_settings_profile("acceptance")

        assert loaded is not None
        dashboard.apply_user_settings(loaded)
        state = dashboard.app_state_snapshot()

        self.assertEqual(loaded, gathered)
        self.assertEqual(state.user_settings.profile_name, "acceptance")
        self.assertEqual(state.user_settings.reasoning["mode"], "deep")
        self.assertEqual(state.user_settings.reasoning["thinking_minutes"], 45)
        self.assertFalse(state.user_settings.retrieval["allow_web_fallback"])
        self.assertFalse(state.user_settings.retrieval["reranking"])
        self.assertIn("reranker", state.user_settings.models["enabled_roles"])
        self.assertIn("speech_to_text", state.user_settings.models["enabled_roles"])
        self.assertIn("vad", state.user_settings.models["enabled_roles"])
        self.assertTrue(state.user_settings.long_horizon["enabled"])
        self.assertTrue(state.user_settings.desktop["enabled"])
        self.assertEqual(state.user_settings.observation["tier"], "vision_on_step")
        self.assertTrue(state.user_settings.observation["ocr_on_step"])
        self.assertTrue(state.user_settings.observation["vision_on_step"])
        self.assertEqual(state.user_settings.cloud["mode"], "disabled")
        self.assertFalse(state.user_settings.ui["show_debug_pane"])

    async def test_headless_readiness_projection_preserves_capability_reasons(self) -> None:
        profile = UserSettingsProfile(
            profile_name="future-flags",
            desktop={"enabled": True, "approval_policy": "approve_risky_only"},
            observation={
                "tier": "vision_on_step",
                "continuous_capture": False,
                "ocr_on_step": True,
                "vision_on_step": True,
            },
            cloud={"enabled": True, "mode": "auxiliary_only"},
        )

        report = self.orchestrator._build_dashboard_readiness_report(active_profile=profile)
        self.orchestrator.dashboard.publish_event(
            {
                "stage": "dashboard.readiness_loaded",
                "report": report.to_dict(),
            }
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        checks = {item.check_id: item for item in state.readiness_report.checks}
        capabilities = {item.capability_name: item for item in state.readiness_report.capabilities}

        self.assertEqual(state.readiness_report, report)
        self.assertEqual(checks["specialist_roles"].status, "disabled")
        self.assertEqual(capabilities["desktop_control"].reason, "phase_20_21_not_implemented")
        self.assertEqual(capabilities["observation_tiers"].reason, "phase_22_not_implemented")
        self.assertEqual(capabilities["cloud_offload"].reason, "phase_23_not_implemented")

    async def test_knowledge_library_actions_cover_ingest_archive_rebuild_remove_and_demo_separation(self) -> None:
        await self.orchestrator._run_dashboard_action(action="examples.load_demo_pack", payload={})
        await self.orchestrator._run_dashboard_action(
            action="knowledge.ingest_text",
            payload={
                "source_ref": "local://phase12/user-note",
                "title": "Phase 12 User Note",
                "content": "User-authored acceptance note for dashboard knowledge actions.",
            },
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        demo_sources = [source for source in state.knowledge_sources if source.corpus_origin == "demo_phase11"]
        user_sources = [source for source in state.knowledge_sources if source.corpus_origin == "user_local"]

        self.assertEqual(len(demo_sources), 4)
        self.assertEqual(len(user_sources), 1)
        self.assertFalse(user_sources[0].archived)

        await self.orchestrator._run_dashboard_action(
            action="knowledge.archive_source",
            payload={"source_ref": "local://phase12/user-note"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        archived_source = next(
            source for source in state.knowledge_sources if source.source_ref == "local://phase12/user-note"
        )
        self.assertTrue(archived_source.archived)

        await self.orchestrator._run_dashboard_action(
            action="knowledge.unarchive_source",
            payload={"source_ref": "local://phase12/user-note"},
        )
        await self.orchestrator._run_dashboard_action(
            action="knowledge.rebuild_source",
            payload={"source_ref": "local://phase12/user-note"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        rebuilt_source = next(
            source for source in state.knowledge_sources if source.source_ref == "local://phase12/user-note"
        )
        self.assertFalse(rebuilt_source.archived)
        self.assertEqual(rebuilt_source.embedding_model, self.config.preflight.backends.embedding_model)

        await self.orchestrator._run_dashboard_action(
            action="knowledge.remove_source",
            payload={"source_ref": "local://phase12/user-note"},
        )
        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertFalse(
            any(source.source_ref == "local://phase12/user-note" for source in state.knowledge_sources)
        )

    async def test_history_browsing_and_trace_debug_export_update_dashboard_state(self) -> None:
        first = await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=5)
        _ = await self.orchestrator.run_task("What does `sum([1, 2, 3])` return?", thinking_minutes=30)
        export_path = self.test_logs / "phase12_trace_debug.txt"

        await self.orchestrator._run_dashboard_action(action="history.refresh", payload={})
        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertGreaterEqual(len(state.task_history), 2)

        await self.orchestrator._run_dashboard_action(
            action="history.inspect_task",
            payload={"task_id": first.task_id},
        )
        await self.orchestrator._run_dashboard_action(
            action="history.export_task_debug",
            payload={"task_id": first.task_id, "path": str(export_path)},
        )

        state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertEqual(state.selected_task.task_id, first.task_id)
        self.assertTrue(export_path.exists())
        self.assertEqual(state.selected_task.trace_debug_export_path, str(export_path))


if __name__ == "__main__":
    unittest.main()
