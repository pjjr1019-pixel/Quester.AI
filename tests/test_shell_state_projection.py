"""Shell-state projection regressions for the premium desktop shell migration."""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest import mock

from config import APP_CONFIG
from dashboard import DashboardService
from data_structures import UserSettingsProfile
from pyside_shell import PySideShellUnavailableError, pyside6_available


def _build_headless_config():
    return replace(APP_CONFIG, dashboard=replace(APP_CONFIG.dashboard, enable_ui=False))


def _cleanup_qt() -> None:
    if not pyside6_available():
        return
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    app.closeAllWindows()
    QtCore.QCoreApplication.sendPostedEvents(None, 0)
    app.processEvents()
    app.quit()
    QtCore.QCoreApplication.sendPostedEvents(None, 0)
    app.processEvents()


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


class ShellStateProjectionTests(unittest.TestCase):
    def _new_shell_dashboard(self, *, coding_mode: bool = False) -> DashboardService:
        dashboard = DashboardService(config=_build_headless_config())
        dashboard.apply_user_settings(
            UserSettingsProfile(
                profile_name="pyside-shell",
                ui={"app_shell": "pyside6"},
                coding={"enabled": coding_mode, "mode": "coding_workspace" if coding_mode else "assistant"},
            )
        )
        return dashboard

    def test_user_settings_profile_accepts_pyside6_shell_preferences(self) -> None:
        profile = UserSettingsProfile(
            profile_name="desktop-shell",
            ui={
                "app_shell": "pyside6",
                "lightweight_mode": True,
                "shell_preset": "immersive",
                "low_resource_mode": True,
                "reduced_effects_mode": True,
                "animation_frame_cap": 24,
                "orb_size": 90,
                "status_text_scale": 1.2,
            },
        )
        profile.validate()

        self.assertEqual(profile.ui["app_shell"], "pyside6")
        self.assertTrue(profile.ui["lightweight_mode"])
        self.assertEqual(profile.ui["shell_preset"], "immersive")
        self.assertTrue(profile.ui["low_resource_mode"])
        self.assertTrue(profile.ui["reduced_effects_mode"])
        self.assertEqual(profile.ui["animation_frame_cap"], 24)
        self.assertEqual(profile.ui["orb_size"], 90)
        self.assertEqual(profile.ui["status_text_scale"], 1.2)
        self.assertTrue(profile.ui["activity_strip_visible"])
        self.assertTrue(profile.ui["resource_ribbon_visible"])

    def test_gather_settings_preserves_selected_app_shell(self) -> None:
        dashboard = DashboardService(config=_build_headless_config())
        dashboard.apply_user_settings(
            UserSettingsProfile(
                profile_name="desktop-shell",
                ui={"app_shell": "pyside6"},
            )
        )
        dashboard._profile_name_var = _FakeVar("desktop-shell")
        dashboard._generation_backend_var = _FakeVar("ollama")
        dashboard._embedding_backend_var = _FakeVar("sentence_transformers")
        dashboard._vector_store_var = _FakeVar("chromadb")
        dashboard._web_provider_var = _FakeVar("wikipedia")
        dashboard._reasoning_mode_var = _FakeVar("auto")
        dashboard._thinking_minutes_var = _FakeVar(30)
        dashboard._allow_web_fallback_var = _FakeVar(True)
        dashboard._enable_self_optimizer_var = _FakeVar(False)
        dashboard._reranking_var = _FakeVar(True)
        dashboard._reranker_role_enabled_var = _FakeVar(False)
        dashboard._speech_to_text_role_enabled_var = _FakeVar(False)
        dashboard._text_to_speech_role_enabled_var = _FakeVar(False)
        dashboard._vad_role_enabled_var = _FakeVar(False)
        dashboard._translation_role_enabled_var = _FakeVar(False)
        dashboard._code_specialist_role_enabled_var = _FakeVar(False)
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

        gathered = dashboard._gather_settings_from_form()

        self.assertEqual(gathered.ui["app_shell"], "pyside6")

    def test_shell_state_projects_deep_reasoning_with_approval_overlay(self) -> None:
        dashboard = DashboardService(config=_build_headless_config())
        dashboard.apply_user_settings(
            UserSettingsProfile(
                profile_name="deep-shell",
                reasoning={"mode": "deep", "thinking_minutes": 180},
                observation={"tier": "vision_on_step", "ocr_on_step": True, "vision_on_step": True},
                ui={"app_shell": "pyside6"},
            )
        )
        dashboard.publish_event(
            {
                "stage": "runtime.health_snapshot",
                "started": True,
                "generation_backend": "ollama",
                "embedding_backend": "sentence_transformers",
                "governor_active": True,
                "governor_summary": "memory pressure",
            }
        )
        dashboard.publish_event(
            {
                "stage": "pipeline.received",
                "task_id": "task-deep",
                "question": "Analyze multiple candidates.",
                "thinking_minutes": 180,
                "budget": {"planned_cycles": 3, "cycle_budget_minutes": 60},
            }
        )
        dashboard.publish_event(
            {
                "stage": "pipeline.long_horizon_started",
                "task_id": "task-deep",
                "session_id": "lh-1",
                "planned_cycles": 3,
            }
        )
        dashboard.publish_event(
            {
                "stage": "dashboard.local_task_session_loaded",
                "local_task_session": {
                    "session_id": "session-1",
                    "status": "running",
                    "pending_approval_summaries": ("Approve browser click",),
                    "requested_observation_tier": "vision_on_step",
                    "effective_observation_tier": "ocr_on_step",
                    "last_action_summary": "Waiting for approval",
                },
            }
        )
        dashboard.publish_event({"stage": "pipeline.reasoner_started", "task_id": "task-deep"})

        shell_state = dashboard.shell_state_snapshot()

        self.assertEqual(shell_state.orb_mode, "reasoner_deep")
        self.assertEqual(shell_state.active_agent, "reasoner")
        self.assertTrue(shell_state.approval_pending)
        self.assertEqual(shell_state.long_horizon_state, "running")
        self.assertEqual(shell_state.resource_pressure_level, "elevated")
        self.assertEqual(shell_state.observation_tier, "ocr_on_step")
        self.assertEqual(shell_state.workspace_mode, "assistant")
        self.assertEqual(shell_state.approval_prompt_summary, "Approve browser click")
        self.assertIn("pressure:elevated", shell_state.resource_ribbon_flags)
        self.assertTrue(shell_state.panel_visibility_state["resource_ribbon"])
        self.assertIn("desktop", shell_state.active_tools)
        self.assertTrue(any(chip.label == "Approval Needed" for chip in shell_state.activity_chips))

    def test_shell_state_records_conversation_notifications_and_completion_effects(self) -> None:
        dashboard = DashboardService(config=_build_headless_config())
        dashboard.apply_user_settings(
            UserSettingsProfile(
                profile_name="reply-shell",
                ui={"app_shell": "pyside6"},
            )
        )
        dashboard.publish_event(
            {
                "stage": "runtime.health_snapshot",
                "started": True,
                "generation_backend": "ollama",
                "embedding_backend": "sentence_transformers",
            }
        )
        dashboard.publish_event(
            {
                "stage": "pipeline.received",
                "task_id": "task-1",
                "question": "What is the answer?",
                "thinking_minutes": 30,
                "budget": {"planned_cycles": 1, "cycle_budget_minutes": 30},
            }
        )
        dashboard.publish_event(
            {
                "stage": "dashboard.notice",
                "message": "Web fallback armed for freshness.",
                "severity": "warning",
            }
        )
        dashboard.publish_event(
            {
                "stage": "pipeline.reasoner_done",
                "task_id": "task-1",
                "candidate_trace_count": 3,
                "candidate_score": 0.91,
                "selected_verifier": "critic",
            }
        )
        dashboard.publish_event(
            {
                "stage": "pipeline.completed",
                "task_id": "task-1",
                "answer_text": "The answer is 42.",
                "citation_refs": ("local://math/answer",),
                "critique_result": "valid",
                "candidate_score": 0.91,
            }
        )

        shell_state = dashboard.shell_state_snapshot()

        self.assertEqual(shell_state.orb_mode, "responding")
        self.assertEqual(shell_state.confidence_band, "high")
        self.assertTrue(shell_state.orb_effects.insight_flash_pending)
        self.assertTrue(shell_state.orb_effects.consensus_shimmer_pending)
        self.assertTrue(shell_state.orb_effects.verification_lock_pending)
        self.assertEqual(len(shell_state.conversation_items), 2)
        self.assertEqual(shell_state.conversation_items[0].role, "user")
        self.assertEqual(shell_state.conversation_items[1].role, "assistant")
        self.assertEqual(shell_state.conversation_items[1].body, "The answer is 42.")
        self.assertEqual(shell_state.shell_notifications[-1].message, "Web fallback armed for freshness.")
        self.assertTrue(any(entry.label == "Answer rendered" for entry in shell_state.timeline_entries))

    def test_shell_state_projects_coding_mode_testing_and_learning_states(self) -> None:
        dashboard = DashboardService(config=_build_headless_config())
        dashboard.apply_user_settings(
            UserSettingsProfile(
                profile_name="coding-shell",
                ui={"app_shell": "pyside6"},
                coding={"enabled": True, "mode": "coding_workspace"},
            )
        )
        dashboard.publish_event(
            {
                "stage": "runtime.health_snapshot",
                "started": True,
                "generation_backend": "ollama",
                "embedding_backend": "sentence_transformers",
            }
        )
        dashboard.publish_event(
            {
                "stage": "coding.testing",
                "request_id": "coding-1",
                "task_type": "feature_generation",
                "language": "python",
                "has_tests": True,
            }
        )
        testing_state = dashboard.shell_state_snapshot()
        self.assertEqual(testing_state.orb_mode, "code_testing")
        self.assertEqual(testing_state.coding_state, "testing")
        self.assertEqual(testing_state.workspace_mode, "coding_workspace")
        self.assertEqual(testing_state.sandbox_state, "running")
        self.assertEqual(testing_state.quality_gate_state, "running")
        self.assertTrue(any(chip.label == "Coding Mode" for chip in testing_state.activity_chips))

        dashboard.publish_event(
            {
                "stage": "dashboard.coding_output_loaded",
                "coding_output": {
                    "request_id": "coding-1",
                    "task_type": "feature_generation",
                    "status": "completed",
                    "active_phase": "indexing",
                    "prompt": "Generate a helper.",
                    "summary": "Completed a bounded coding task.",
                    "language": "python",
                    "role_assignments": {"generator": "stub"},
                    "route_summary": ("generator:stub-model", "reviewer:stub-reviewer"),
                    "artifacts": (
                        {
                            "artifact_id": "artifact-code",
                            "artifact_type": "code",
                            "title": "Generated Code",
                            "language": "python",
                            "path": "sandbox:solution.py",
                            "content_preview": "def helper():\n    return 1\n",
                        },
                    ),
                    "quality_report": {
                        "tests_passed": True,
                        "lint_passed": True,
                        "complexity_passed": True,
                        "security_passed": True,
                        "maintainability_passed": True,
                        "critique_passed": True,
                        "regression_passed": True,
                        "overall_passed": True,
                        "quality_score": 0.9,
                    },
                    "verified_patterns": ("pattern-1",),
                },
            }
        )
        dashboard.publish_event(
            {
                "stage": "coding.completed",
                "request_id": "coding-1",
                "task_type": "feature_generation",
                "language": "python",
                "quality_score": 0.9,
                "summary": "Completed a bounded coding task.",
            }
        )
        learning_state = dashboard.shell_state_snapshot()
        self.assertEqual(learning_state.orb_mode, "code_learning")
        self.assertEqual(learning_state.verifier_state, "verified")
        self.assertEqual(learning_state.coding_state, "completed")
        self.assertEqual(learning_state.workspace_mode, "coding_workspace")
        self.assertEqual(learning_state.sandbox_state, "completed")
        self.assertEqual(learning_state.quality_gate_state, "passed")
        self.assertEqual(learning_state.current_file, "sandbox:solution.py")
        self.assertEqual(learning_state.current_project, "sandbox")
        self.assertEqual(learning_state.active_route_summary, ("generator:stub-model", "reviewer:stub-reviewer"))
        self.assertEqual(learning_state.pattern_tier_counts["verified"], 1)
        self.assertIn("Quality 0.90", learning_state.hero_metric_strip)
        self.assertTrue(learning_state.orb_effects.verification_lock_pending)

    def test_shell_state_activity_strip_covers_long_horizon_cloud_and_route_changes(self) -> None:
        dashboard = self._new_shell_dashboard(coding_mode=True)
        dashboard.apply_user_settings(
            UserSettingsProfile(
                profile_name="activity-audit",
                ui={"app_shell": "pyside6"},
                cloud={"mode": "auxiliary_only"},
                coding={"enabled": True, "mode": "coding_workspace"},
            )
        )
        dashboard.publish_event(
            {
                "stage": "runtime.health_snapshot",
                "started": True,
                "generation_backend": "ollama",
                "embedding_backend": "sentence_transformers",
                "governor_active": True,
                "governor_summary": "memory pressure",
            }
        )
        dashboard.publish_event(
            {
                "stage": "dashboard.model_registry_loaded",
                "model_registry_view": {
                    "registrations": (
                        {
                            "registration_id": "code-a",
                            "role": "code_specialist",
                            "backend": "ollama",
                            "model_identifier": "qwen-coder",
                            "metadata": {"coding_roles": ("generator",)},
                        },
                    ),
                    "last_route_decisions": (
                        {
                            "requested_role": "code_specialist",
                            "selected_registration_id": "code-a",
                            "selected_model_identifier": "qwen-coder",
                            "allowed": True,
                            "used_fallback": True,
                            "fallback_reason": "heavy_slot_pressure",
                            "metadata": {"coding_role": "generator"},
                        },
                    ),
                },
            }
        )
        dashboard.publish_event(
            {
                "stage": "pipeline.received",
                "task_id": "task-activity",
                "question": "Run a deeper coding audit.",
                "thinking_minutes": 180,
                "budget": {"planned_cycles": 3, "cycle_budget_minutes": 60},
            }
        )
        dashboard.publish_event(
            {
                "stage": "pipeline.long_horizon_started",
                "task_id": "task-activity",
                "session_id": "lh-activity",
                "planned_cycles": 3,
            }
        )
        dashboard.publish_event(
            {
                "stage": "dashboard.coding_output_loaded",
                "coding_output": {
                    "request_id": "coding-activity",
                    "task_type": "code_review",
                    "status": "running",
                    "active_phase": "reviewing",
                },
            }
        )
        dashboard.publish_event(
            {
                "stage": "coding.reviewing",
                "request_id": "coding-activity",
                "task_type": "code_review",
            }
        )

        labels = {chip.label for chip in dashboard.shell_state_snapshot().activity_chips}

        self.assertIn("Long Horizon", labels)
        self.assertIn("Cloud Helper", labels)
        self.assertIn("Route Change", labels)
        self.assertIn("Resource Pressure", labels)
        self.assertIn("Coding Mode", labels)

    def test_shell_state_flow_audit_covers_operator_critical_paths(self) -> None:
        scenarios = (
            (
                "fast_task",
                False,
                (
                    {
                        "stage": "runtime.health_snapshot",
                        "started": True,
                        "generation_backend": "ollama",
                        "embedding_backend": "sentence_transformers",
                    },
                    {
                        "stage": "pipeline.received",
                        "task_id": "task-fast-flow",
                        "question": "Answer quickly.",
                        "thinking_minutes": 5,
                    },
                    {"stage": "pipeline.reasoner_started", "task_id": "task-fast-flow"},
                ),
                lambda state: (
                    self.assertEqual(state.orb_mode, "reasoner_fast"),
                    self.assertEqual(state.conversation_items[0].title, "Task Submitted"),
                ),
            ),
            (
                "fallback_and_web",
                False,
                (
                    {
                        "stage": "runtime.health_snapshot",
                        "started": True,
                        "generation_backend": "ollama",
                        "embedding_backend": "sentence_transformers",
                        "fallback_active": True,
                        "fallback_reason": "stale_primary",
                    },
                    {
                        "stage": "pipeline.received",
                        "task_id": "task-fallback-flow",
                        "question": "Check the newest source.",
                        "thinking_minutes": 30,
                    },
                    {
                        "stage": "researcher.web_lookup",
                        "task_id": "task-fallback-flow",
                        "query": "latest update",
                    },
                ),
                lambda state: (
                    self.assertEqual(state.fallback_reason, "stale_primary"),
                    self.assertTrue(any(chip.label == "Fallback Active" for chip in state.activity_chips)),
                    self.assertTrue(any(entry.label == "Web fallback triggered" for entry in state.timeline_entries)),
                ),
            ),
            (
                "approval_flow",
                False,
                (
                    {
                        "stage": "runtime.health_snapshot",
                        "started": True,
                        "generation_backend": "ollama",
                        "embedding_backend": "sentence_transformers",
                    },
                    {
                        "stage": "dashboard.local_task_session_loaded",
                        "local_task_session": {
                            "session_id": "session-approval",
                            "status": "running",
                            "pending_approval_summaries": ("Approve browser click",),
                            "effective_observation_tier": "vision_on_step",
                        },
                    },
                ),
                lambda state: (
                    self.assertTrue(state.approval_pending),
                    self.assertEqual(state.capability_session_state, "running"),
                    self.assertTrue(any(chip.label == "Approval Needed" for chip in state.activity_chips)),
                ),
            ),
            (
                "coding_dojo",
                True,
                (
                    {
                        "stage": "runtime.health_snapshot",
                        "started": True,
                        "generation_backend": "ollama",
                        "embedding_backend": "sentence_transformers",
                    },
                    {
                        "stage": "dashboard.coding_practice_loaded",
                        "coding_practice": {
                            "session_id": "practice-flow",
                            "status": "running",
                            "prompt": "Repair the kata.",
                            "quality_score": 0.5,
                        },
                    },
                    {"stage": "coding.practicing", "request_id": "practice-flow", "task_type": "practice"},
                    {
                        "stage": "coding.practice_completed",
                        "request_id": "practice-flow",
                        "summary": "Captured a reusable practice pattern.",
                    },
                ),
                lambda state: (
                    self.assertEqual(state.workspace_mode, "coding_workspace"),
                    self.assertTrue(any(item.title == "Coding Dojo" for item in state.conversation_items)),
                ),
            ),
        )

        for name, coding_mode, events, assertions in scenarios:
            with self.subTest(name=name):
                dashboard = self._new_shell_dashboard(coding_mode=coding_mode)
                for event in events:
                    dashboard.publish_event(event)
                assertions(dashboard.shell_state_snapshot())

    def test_shell_state_maps_general_pipeline_and_audio_modes(self) -> None:
        offline_dashboard = self._new_shell_dashboard()
        self.assertEqual(offline_dashboard.shell_state_snapshot().orb_mode, "offline")

        scenarios = (
            (
                "idle",
                (),
                {"orb_mode": "idle"},
            ),
            (
                "planner",
                (
                    {"stage": "pipeline.received", "task_id": "task-planner", "question": "Plan the task.", "thinking_minutes": 30},
                    {"stage": "pipeline.planner_started", "task_id": "task-planner"},
                ),
                {"orb_mode": "planner", "active_agent": "planner"},
            ),
            (
                "local_research",
                (
                    {"stage": "pipeline.received", "task_id": "task-local", "question": "Search local notes.", "thinking_minutes": 30},
                    {"stage": "pipeline.researcher_done", "task_id": "task-local", "local_result_count": 3, "web_result_count": 0},
                ),
                {"orb_mode": "researcher_local", "retrieval_state": "local"},
            ),
            (
                "web_lookup",
                (
                    {"stage": "pipeline.received", "task_id": "task-web", "question": "Check freshness.", "thinking_minutes": 30},
                    {"stage": "pipeline.researcher_done", "task_id": "task-web", "local_result_count": 2, "web_result_count": 0},
                    {"stage": "researcher.web_lookup", "task_id": "task-web", "query": "latest freshness", "persisted_results": 1},
                ),
                {"orb_mode": "researcher_web", "retrieval_state": "web_assisted"},
            ),
            (
                "fast_reasoning",
                (
                    {"stage": "pipeline.received", "task_id": "task-fast", "question": "Think quickly.", "thinking_minutes": 30},
                    {"stage": "pipeline.reasoner_started", "task_id": "task-fast"},
                ),
                {"orb_mode": "reasoner_fast", "active_agent": "reasoner"},
            ),
            (
                "deep_reasoning",
                (
                    {"stage": "pipeline.received", "task_id": "task-deep", "question": "Think deeply.", "thinking_minutes": 180, "budget": {"planned_cycles": 3, "cycle_budget_minutes": 60}},
                    {"stage": "pipeline.long_horizon_started", "task_id": "task-deep", "session_id": "lh-9", "planned_cycles": 3},
                    {"stage": "pipeline.reasoner_started", "task_id": "task-deep"},
                ),
                {"orb_mode": "reasoner_deep", "active_agent": "reasoner"},
            ),
            (
                "critic",
                (
                    {"stage": "pipeline.received", "task_id": "task-critic", "question": "Verify the answer.", "thinking_minutes": 30},
                    {"stage": "pipeline.critic_started", "task_id": "task-critic"},
                ),
                {"orb_mode": "critic", "verifier_state": "running"},
            ),
            (
                "compressor",
                (
                    {"stage": "pipeline.received", "task_id": "task-compress", "question": "Compress the trace.", "thinking_minutes": 30},
                    {"stage": "pipeline.compressor_started", "task_id": "task-compress"},
                ),
                {"orb_mode": "compressor", "compression_state": "active"},
            ),
            (
                "responding",
                (
                    {"stage": "pipeline.received", "task_id": "task-answer", "question": "Finish the answer.", "thinking_minutes": 30},
                    {"stage": "pipeline.reasoner_done", "task_id": "task-answer", "candidate_trace_count": 2, "candidate_score": 0.91, "selected_verifier": "critic"},
                    {"stage": "pipeline.completed", "task_id": "task-answer", "answer_text": "Completed answer.", "critique_result": "valid", "candidate_score": 0.91},
                ),
                {"orb_mode": "responding", "verifier_state": "verified"},
            ),
            (
                "listening",
                (
                    {"stage": "dashboard.audio_input_loaded", "audio_input": {"status": "completed", "transcript_text": "hello"}},
                ),
                {"orb_mode": "listening"},
            ),
            (
                "speaking",
                (
                    {"stage": "dashboard.audio_output_loaded", "audio_output": {"status": "completed", "source_text": "hello"}},
                ),
                {"orb_mode": "speaking", "speaking_state": "active"},
            ),
        )

        for name, events, expected in scenarios:
            with self.subTest(name=name):
                dashboard = self._new_shell_dashboard()
                dashboard.publish_event(
                    {
                        "stage": "runtime.health_snapshot",
                        "started": True,
                        "generation_backend": "ollama",
                        "embedding_backend": "sentence_transformers",
                    }
                )
                for event in events:
                    dashboard.publish_event(event)
                shell_state = dashboard.shell_state_snapshot()
                for field_name, expected_value in expected.items():
                    self.assertEqual(getattr(shell_state, field_name), expected_value)

    def test_shell_state_maps_capability_error_and_additional_coding_modes(self) -> None:
        error_dashboard = self._new_shell_dashboard()
        error_dashboard.publish_event(
            {
                "stage": "runtime.health_snapshot",
                "started": True,
                "generation_backend": "ollama",
                "embedding_backend": "sentence_transformers",
                "last_error": "backend unavailable",
            }
        )
        error_state = error_dashboard.shell_state_snapshot()
        self.assertEqual(error_state.orb_mode, "error")

        capability_dashboard = self._new_shell_dashboard()
        capability_dashboard.publish_event(
            {
                "stage": "runtime.health_snapshot",
                "started": True,
                "generation_backend": "ollama",
                "embedding_backend": "sentence_transformers",
            }
        )
        capability_dashboard.publish_event(
            {
                "stage": "dashboard.local_task_session_loaded",
                "local_task_session": {
                    "session_id": "session-2",
                    "label": "Desktop session",
                    "status": "running",
                    "control_mode": "local_task",
                    "current_target": "Notepad",
                    "effective_observation_tier": "vision_on_step",
                    "last_action_summary": "Waiting on operator",
                },
            }
        )
        capability_state = capability_dashboard.shell_state_snapshot()
        self.assertEqual(capability_state.capability_session_state, "running")
        self.assertEqual(capability_state.observation_tier, "vision_on_step")
        self.assertIn("desktop", capability_state.active_tools)
        self.assertIn("observation", capability_state.active_tools)

        coding_scenarios = (
            (
                "planning",
                (
                    {"stage": "coding.planning", "request_id": "coding-plan", "task_type": "feature_generation"},
                ),
                {"orb_mode": "code_planning", "coding_state": "planning"},
            ),
            (
                "refactoring",
                (
                    {
                        "stage": "dashboard.coding_output_loaded",
                        "coding_output": {
                            "request_id": "coding-refactor",
                            "task_type": "refactoring",
                            "status": "running",
                            "active_phase": "generating",
                        },
                    },
                    {"stage": "coding.generating", "request_id": "coding-refactor", "task_type": "refactoring"},
                ),
                {"orb_mode": "code_refactoring", "coding_state": "generating"},
            ),
            (
                "debugging",
                (
                    {
                        "stage": "dashboard.coding_output_loaded",
                        "coding_output": {
                            "request_id": "coding-debug",
                            "task_type": "bug_fixing",
                            "status": "running",
                            "active_phase": "debugging",
                            "quality_report": {
                                "tests_passed": False,
                                "lint_passed": False,
                                "security_passed": True,
                                "regression_passed": False,
                            },
                        },
                    },
                    {"stage": "coding.debugging", "request_id": "coding-debug", "task_type": "bug_fixing"},
                ),
                {"orb_mode": "code_debugging", "sandbox_state": "failed", "quality_gate_state": "running"},
            ),
            (
                "reviewing",
                (
                    {
                        "stage": "dashboard.coding_output_loaded",
                        "coding_output": {
                            "request_id": "coding-review",
                            "task_type": "code_review",
                            "status": "running",
                            "active_phase": "reviewing",
                        },
                    },
                    {"stage": "coding.reviewing", "request_id": "coding-review", "task_type": "code_review"},
                ),
                {"orb_mode": "code_reviewing", "quality_gate_state": "running"},
            ),
            (
                "indexing",
                (
                    {
                        "stage": "dashboard.coding_output_loaded",
                        "coding_output": {
                            "request_id": "coding-index",
                            "task_type": "feature_generation",
                            "status": "completed",
                            "active_phase": "indexing",
                            "artifacts": (
                                {
                                    "artifact_id": "artifact-index",
                                    "artifact_type": "code",
                                    "title": "Generated Code",
                                    "path": "sandbox:index.py",
                                },
                            ),
                            "quality_report": {
                                "tests_passed": True,
                                "lint_passed": True,
                                "security_passed": True,
                                "regression_passed": True,
                                "maintainability_passed": True,
                                "complexity_passed": True,
                                "overall_passed": True,
                                "quality_score": 0.91,
                            },
                        },
                    },
                    {"stage": "coding.indexing", "request_id": "coding-index", "task_type": "feature_generation"},
                ),
                {"orb_mode": "code_indexing", "coding_state": "indexing"},
            ),
            (
                "practicing",
                (
                    {
                        "stage": "dashboard.coding_practice_loaded",
                        "coding_practice": {
                            "session_id": "practice-1",
                            "status": "running",
                            "task_type": "practice",
                            "prompt": "Repair the failing kata.",
                            "quality_score": 0.5,
                        },
                    },
                    {"stage": "coding.practicing", "request_id": "practice-1", "task_type": "practice"},
                ),
                {"orb_mode": "code_practicing", "coding_state": "practicing"},
            ),
            (
                "regression_detected",
                (
                    {
                        "stage": "dashboard.coding_output_loaded",
                        "coding_output": {
                            "request_id": "coding-regression",
                            "task_type": "bug_fixing",
                            "status": "running",
                            "active_phase": "debugging",
                            "warnings": ("regression_detected",),
                        },
                    },
                    {"stage": "coding.regression_detected", "request_id": "coding-regression", "task_type": "bug_fixing"},
                ),
                {"orb_mode": "code_regression", "sandbox_state": "failed", "quality_gate_state": "failed"},
            ),
        )

        for name, events, expected in coding_scenarios:
            with self.subTest(name=name):
                dashboard = self._new_shell_dashboard(coding_mode=True)
                dashboard.publish_event(
                    {
                        "stage": "runtime.health_snapshot",
                        "started": True,
                        "generation_backend": "ollama",
                        "embedding_backend": "sentence_transformers",
                    }
                )
                for event in events:
                    dashboard.publish_event(event)
                shell_state = dashboard.shell_state_snapshot()
                self.assertEqual(shell_state.workspace_mode, "coding_workspace")
                for field_name, expected_value in expected.items():
                    self.assertEqual(getattr(shell_state, field_name), expected_value)


class PySideDashboardServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        _cleanup_qt()

    async def test_dashboard_service_reports_clear_error_when_pyside_host_is_unavailable(self) -> None:
        config = replace(APP_CONFIG, dashboard=replace(APP_CONFIG.dashboard, enable_ui=True))
        dashboard = DashboardService(config=config)
        dashboard.apply_user_settings(
            UserSettingsProfile(
                profile_name="desktop-shell",
                ui={"app_shell": "pyside6"},
            )
        )

        with mock.patch(
            "pyside_shell.PySideShellHost",
            side_effect=PySideShellUnavailableError("PySide6 is not installed."),
        ):
            with self.assertRaisesRegex(RuntimeError, "PySide6 shell requested"):
                await dashboard.start()

    async def test_dashboard_service_passes_settings_bridge_to_pyside_host(self) -> None:
        config = replace(APP_CONFIG, dashboard=replace(APP_CONFIG.dashboard, enable_ui=True))
        dashboard = DashboardService(config=config)
        dashboard.apply_user_settings(
            UserSettingsProfile(
                profile_name="desktop-shell",
                ui={"app_shell": "pyside6"},
            )
        )

        with mock.patch("pyside_shell.PySideShellHost") as host_cls:
            host = host_cls.return_value
            await dashboard.start()
            try:
                self.assertIs(host_cls.call_args.kwargs["save_settings"].__self__, dashboard)
                self.assertEqual(host_cls.call_args.kwargs["save_settings"].__name__, "request_settings_save")
                host.start.assert_called_once()
            finally:
                await dashboard.stop()

    async def test_dashboard_service_can_start_and_stop_pyside_shell_when_requested(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtWidgets

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        config = replace(APP_CONFIG, dashboard=replace(APP_CONFIG.dashboard, enable_ui=True))
        dashboard = DashboardService(config=config)
        dashboard.apply_user_settings(
            UserSettingsProfile(
                profile_name="desktop-shell",
                ui={"app_shell": "pyside6"},
            )
        )

        try:
            await dashboard.start()
            self.assertTrue(dashboard.ui_running)
        finally:
            await dashboard.stop()
            _cleanup_qt()
        self.assertFalse(dashboard.ui_running)


if __name__ == "__main__":
    unittest.main()
