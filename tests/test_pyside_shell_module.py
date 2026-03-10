"""Dependency-gated regressions for the optional PySide6 shell runtime."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from data_structures import DashboardAppState, ShellState
from pyside_shell import (
    PySideShellHost,
    PySideShellUnavailableError,
    PySideShellWindow,
    pyside6_available,
)


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


class PySideShellModuleTests(unittest.TestCase):
    def tearDown(self) -> None:
        _cleanup_qt()

    def test_pyside6_availability_flag_is_boolean(self) -> None:
        self.assertIsInstance(pyside6_available(), bool)

    def test_shell_window_hides_center_cards_while_idle(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PySideShellWindow(shell_state=ShellState(), app_state=DashboardAppState())
        window.show()
        app.processEvents()

        self.assertTrue(window._active_task_card.isHidden())
        self.assertTrue(window._final_answer_card.isHidden())
        self.assertEqual(window._input.placeholderText(), "Type a message...")
        window.close()
        _cleanup_qt()

    def test_shell_window_renders_live_shell_state_and_accepts_submission_callbacks(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        submissions: list[tuple[str, int]] = []
        window = PySideShellWindow(
            shell_state=ShellState(
                status_text="Deep Thought",
                sub_status_text="Ranking options",
                hero_metric_strip=("Candidates 3", "Evidence 6", "Verifier verified"),
                active_route_summary=("generation:qwen", "embedding:e5-small"),
                current_project="sandbox",
                current_file="sandbox:solution.py",
                current_task_summary="Analyze multiple candidates.",
            ),
            app_state=DashboardAppState(),
            submit_task=lambda question, minutes: submissions.append((question, minutes)) or True,
        )
        window._input.setText("Test the shell")
        window._slider.setValue(120)
        window._submit_from_input()

        self.assertEqual(window.windowTitle(), "Quester.AI")
        self.assertEqual(window._status.text(), "Deep Thought")
        self.assertIn("Candidates 3", window._hero_metrics.text())
        self.assertIn("generation:qwen", window._route_summary.text())
        self.assertEqual(submissions, [("Test the shell", 120)])
        window.close()
        _cleanup_qt()

    def test_shell_window_renders_assistant_task_and_final_answer_cards(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PySideShellWindow(
            shell_state=ShellState(
                status_text="Reasoning",
                sub_status_text="Comparing evidence and preparing the final response.",
                active_agent="reasoner",
                verifier_state="verified",
                confidence_band="medium",
                candidate_count=3,
                evidence_count=6,
                elapsed_seconds=125,
                active_route_summary=("generation:qwen", "critic:llama"),
                fallback_reason="web_freshness",
                current_task_summary="Resolve the discrepancy between the local and web sources.",
            ),
            app_state=DashboardAppState.from_dict(
                {
                    "last_stage": "pipeline.reasoner_started",
                    "active_task": {
                        "task_id": "task-42",
                        "question": "Resolve the discrepancy between the local and web sources.",
                        "running_stage": "pipeline.reasoner_started",
                        "answer_text": "The answer is 42.",
                        "citation_refs": ("local://math/answer", "web://freshness/update"),
                        "selected_verifier": "critic",
                        "critique_result": "valid",
                        "candidate_trace_count": 3,
                        "local_result_count": 4,
                        "web_result_count": 2,
                        "failure_categories": ("stale_source",),
                    },
                }
            ),
        )
        window.show()
        app.processEvents()

        self.assertFalse(window._active_task_card.isHidden())
        self.assertEqual(window._active_task_title.text(), "Active Task")
        self.assertIn("Pipeline / Reasoner Started", window._active_task_phase.text())
        self.assertIn("Resolve the discrepancy", window._active_task_summary.text())
        self.assertIn("Elapsed 2m 5s", window._active_task_metrics.text())
        self.assertIn("Candidates 3", window._active_task_metrics.text())
        self.assertIn("Evidence 6", window._active_task_metrics.text())
        self.assertIn("Routes: generation:qwen | critic:llama", window._active_task_routes.text())
        self.assertIn("Fallback: web_freshness", window._active_task_warnings.text())

        self.assertFalse(window._final_answer_card.isHidden())
        self.assertEqual(window._final_answer_title.text(), "Final Answer")
        self.assertEqual(window._final_answer_status.text(), "Valid")
        self.assertEqual(window._final_answer_body.text(), "The answer is 42.")
        self.assertIn("Evidence Summary: 6 sources (4 local / 2 web)", window._final_answer_evidence.text())
        self.assertIn("local://math/answer", window._final_answer_refs.text())
        self.assertIn("Uncertainty medium confidence", window._final_answer_warnings.text())
        self.assertIn("Selected verifier: critic", window._why_answer_text.toPlainText())
        self.assertIn("Verification result: valid", window._how_verified_text.toPlainText())
        self.assertIn("Compared 3 candidates.", window._deep_mode_text.toPlainText())
        self.assertEqual(window._copy_answer_button.text(), "Copy Answer")
        self.assertEqual(window._copy_citations_button.text(), "Copy Citations")
        window.close()
        _cleanup_qt()

    def test_shell_window_renders_coding_center_cards_and_artifacts(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PySideShellWindow(
            shell_state=ShellState(
                workspace_mode="coding_workspace",
                status_text="Coding",
                sub_status_text="Running validation gates.",
                active_agent="generator",
                coding_state="testing",
                verifier_state="verified",
                confidence_band="high",
                candidate_count=2,
                evidence_count=1,
                elapsed_seconds=98,
                current_file="sandbox:solution.py",
                current_project="sandbox",
                sandbox_state="completed",
                quality_gate_state="passed",
                active_route_summary=("generator:qwen-coder", "reviewer:llama-review"),
                pattern_tier_counts={"verified": 2, "candidate": 1, "rejected": 0},
                practice_session_state="reviewing",
                current_task_summary="Implement a helper and validate it.",
            ),
            app_state=DashboardAppState.from_dict(
                {
                    "last_stage": "coding.testing",
                    "coding_output": {
                        "request_id": "coding-7",
                        "task_type": "feature_generation",
                        "status": "completed",
                        "active_phase": "testing",
                        "prompt": "Implement a helper and validate it.",
                        "summary": "",
                        "artifacts": (
                            {
                                "artifact_id": "artifact-1",
                                "artifact_type": "patch",
                                "title": "Validated Patch",
                                "path": "sandbox:solution.py",
                                "content_preview": "diff --git a/solution.py b/solution.py",
                            },
                        ),
                        "quality_report": {
                            "tests_passed": True,
                            "lint_passed": True,
                            "complexity_passed": True,
                            "security_passed": True,
                            "maintainability_passed": True,
                            "regression_passed": True,
                            "overall_passed": True,
                            "quality_score": 0.92,
                        },
                    },
                }
            ),
        )
        window.show()
        app.processEvents()

        self.assertFalse(window._active_task_card.isHidden())
        self.assertEqual(window._active_task_title.text(), "Active Coding Task")
        self.assertIn("Testing", window._active_task_phase.text())
        self.assertIn("Sandbox completed", window._active_task_metrics.text())
        self.assertIn("Gates passed", window._active_task_metrics.text())
        self.assertIn("File sandbox:solution.py", window._active_task_metrics.text())
        self.assertIn("Project sandbox", window._active_task_metrics.text())

        self.assertFalse(window._final_answer_card.isHidden())
        self.assertEqual(window._final_answer_title.text(), "Validated Coding Output")
        self.assertEqual(window._final_answer_status.text(), "Passed")
        self.assertEqual(window._final_answer_body.text(), "")
        self.assertIn("quality 0.92", window._final_answer_evidence.text())
        self.assertIn("Artifacts: Validated Patch: sandbox:solution.py", window._final_answer_refs.text())
        self.assertIn("Task type: feature_generation", window._why_answer_text.toPlainText())
        self.assertIn("Sandbox: completed", window._how_verified_text.toPlainText())
        self.assertIn("Pattern memory: V2 C1 R0", window._deep_mode_text.toPlainText())
        self.assertEqual(window._copy_answer_button.text(), "Copy Summary")
        self.assertEqual(window._copy_citations_button.text(), "Copy Artifacts")
        self.assertEqual(window._input.placeholderText(), "Describe the coding task...")
        window.close()
        _cleanup_qt()

    def test_shell_window_renders_policy_context_and_approval_overlay(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PySideShellWindow(
            shell_state=ShellState(
                status_text="Deep Thought",
                sub_status_text="Waiting for approval before continuing.",
                approval_pending=True,
                approval_prompt_summary="Approve browser click",
                observation_tier="vision_on_step",
                active_tools=("desktop", "observation"),
                current_task_summary="Continue the bounded browser workflow after operator approval.",
            ),
            app_state=DashboardAppState.from_dict(
                {
                    "local_task_session": {
                        "session_id": "session-approval",
                        "label": "Browser task",
                        "status": "running",
                        "control_mode": "local_task",
                        "current_target": "notes.txt - Notepad",
                        "last_action_summary": "Click Save",
                        "pending_approval_summaries": ("Approve browser click",),
                        "effective_observation_tier": "vision_on_step",
                    },
                    "user_settings": {
                        "profile_name": "desktop-shell",
                        "desktop": {"approval_policy": "approve_risky_only"},
                        "cloud": {"mode": "auxiliary_only"},
                    },
                }
            ),
        )
        window.show()
        app.processEvents()

        self.assertFalse(window._approval_overlay.isHidden())
        self.assertEqual(window._approval_title.text(), "Approval Required")
        self.assertIn("Approve browser click", window._approval_summary.text())
        self.assertIn("notes.txt - Notepad", window._approval_target.text())
        self.assertIn("Risk", window._approval_risk.text())
        self.assertTrue(window._approval_pause_button.isEnabled())
        self.assertTrue(window._approval_stop_button.isEnabled())

        context_labels = [
            label.text()
            for label in window._policy_context_bar.findChildren(QtWidgets.QLabel)
            if label.text().strip()
        ]
        self.assertTrue(any("Session running" in text for text in context_labels))
        self.assertTrue(any("Target notes.txt - Notepad" in text for text in context_labels))
        self.assertTrue(any("Observation vision_on_step" in text for text in context_labels))
        self.assertTrue(any("Policy approve risky only" in text for text in context_labels))
        window.close()
        _cleanup_qt()

    def test_shell_window_drawers_render_structured_operator_surfaces_and_toggle_cleanly(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PySideShellWindow(
            shell_state=ShellState.from_dict(
                {
                    "status_text": "Responding",
                    "sub_status_text": "Preparing the verified answer.",
                    "current_task_summary": "Audit the routing and runtime surfaces.",
                    "active_route_summary": ("generation:qwen", "critic:llama"),
                    "timeline_entries": (
                        {"label": "Planner started", "detail": "Structuring the task"},
                        {"label": "Answer rendered", "detail": "The shell is rendering final output"},
                    ),
                }
            ),
            app_state=DashboardAppState.from_dict(
                {
                    "statuses": {
                        "planner": {"component": "planner", "state": "running", "severity": "low", "message": "Structuring the task"},
                        "critic": {"component": "critic", "state": "idle", "severity": "medium", "message": "Verifier waiting"},
                    },
                    "active_task": {
                        "task_id": "task-operator",
                        "question": "Audit the routing and runtime surfaces.",
                        "answer_text": "The shell now shows operator details.",
                        "local_result_count": 3,
                        "web_result_count": 1,
                        "citation_refs": ("local://docs/1",),
                        "supporting_evidence_ids": ("doc-1", "doc-2"),
                        "selected_verifier": "critic",
                        "selected_strategy": "consensus",
                        "candidate_score": 0.88,
                        "repair_actions": ("normalized citations",),
                    },
                    "selected_task": {
                        "task_id": "task-operator",
                        "question": "Audit the routing and runtime surfaces.",
                        "answer_text": "The shell now shows operator details.",
                        "critique_result": "valid",
                        "citation_refs": ("local://docs/1",),
                        "repair_actions": ("normalized citations",),
                        "optimizer_lifecycle": ("requested", "accepted"),
                        "advisor_summaries": ("Keep routing visible.",),
                    },
                    "task_history": (
                        {
                            "task_id": "task-operator",
                            "question": "Audit the routing and runtime surfaces.",
                            "answer_preview": "The shell now shows operator details.",
                            "critique_result": "valid",
                            "candidate_trace_count": 2,
                            "citation_count": 1,
                        },
                    ),
                    "knowledge_sources": (
                        {
                            "document_id": "doc-1",
                            "source_ref": "local://docs/1",
                            "title": "Operator Surface Spec",
                            "chunk_count": 4,
                            "corpus_origin": "notes",
                        },
                    ),
                    "model_registry_view": {
                        "registrations": (
                            {
                                "registration_id": "gen-qwen",
                                "role": "generation",
                                "backend": "ollama",
                                "model_identifier": "qwen",
                            },
                        ),
                        "active_heavy_roles": ("generation",),
                        "last_route_decisions": (
                            {
                                "requested_role": "generation",
                                "selected_registration_id": "gen-qwen",
                                "selected_model_identifier": "qwen",
                                "allowed": True,
                            },
                        ),
                        "compression_insights": (
                            {
                                "proposal_id": "macro-1",
                                "macro_name": "consensus_bundle",
                                "estimated_gain": 0.3,
                                "validation_pass_rate": 1.0,
                                "validation_state": "validated",
                                "accepted": True,
                            },
                        ),
                    },
                    "runtime_health": {
                        "started": True,
                        "generation_backend": "ollama",
                        "embedding_backend": "sentence_transformers",
                        "active_heavy_roles": ("generation",),
                        "governor_active": True,
                        "governor_summary": "memory pressure",
                    },
                }
            ),
        )
        window.show()
        app.processEvents()

        self.assertGreaterEqual(window._agent_list.count(), 2)
        self.assertEqual(window._timeline.count(), 2)
        self.assertIn("Local retrieval", window._evidence_list.item(0).text())
        self.assertGreaterEqual(window._control_plane_list.count(), 2)
        self.assertGreaterEqual(window._runtime_list.count(), 4)
        self.assertEqual(window._history_list.count(), 1)
        self.assertIn("Optimizer lifecycle: requested, accepted", window._history_detail.toPlainText())
        self.assertEqual(window._knowledge_list.count(), 1)

        window._left_dock.hide()
        app.processEvents()
        self.assertFalse(window._left_dock.isVisible())
        window._left_dock.show()
        window._right_dock.hide()
        app.processEvents()
        self.assertFalse(window._right_dock.isVisible())
        window._right_dock.show()
        window._bottom_dock.hide()
        app.processEvents()
        self.assertFalse(window._bottom_dock.isVisible())
        window._bottom_dock.show()
        app.processEvents()
        self.assertTrue(window._left_dock.isVisible())
        self.assertTrue(window._right_dock.isVisible())
        self.assertTrue(window._bottom_dock.isVisible())
        window.close()
        _cleanup_qt()

    def test_shell_window_mode_switch_clears_stale_surfaces_between_assistant_and_coding(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PySideShellWindow(
            shell_state=ShellState(
                status_text="Reasoning",
                sub_status_text="Waiting for approval.",
                approval_pending=True,
                approval_prompt_summary="Approve browser click",
                current_task_summary="Finish the assistant workflow.",
            ),
            app_state=DashboardAppState.from_dict(
                {
                    "local_task_session": {
                        "session_id": "session-mode",
                        "status": "running",
                        "current_target": "Browser",
                        "pending_approval_summaries": ("Approve browser click",),
                    },
                    "active_task": {
                        "question": "Finish the assistant workflow.",
                        "answer_text": "Assistant answer.",
                    },
                }
            ),
        )
        window.show()
        app.processEvents()
        self.assertFalse(window._approval_overlay.isHidden())
        self.assertTrue(window._coding_dock_bar.isHidden())

        window.apply_dashboard_state(
            DashboardAppState.from_dict(
                {
                    "coding_output": {
                        "request_id": "coding-switch",
                        "task_type": "bug_fixing",
                        "status": "completed",
                        "active_phase": "reviewing",
                        "prompt": "Fix the regression and validate it.",
                        "artifacts": (
                            {
                                "artifact_id": "patch-1",
                                "artifact_type": "patch",
                                "title": "Bug Fix Patch",
                                "path": "sandbox:bugfix.py",
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
                            "quality_score": 0.95,
                        },
                    },
                }
            )
        )
        window.apply_shell_state(
            ShellState(
                workspace_mode="coding_workspace",
                status_text="Reviewing",
                sub_status_text="Checking the patch and quality gates.",
                coding_state="reviewing",
                current_task_summary="Fix the regression and validate it.",
                active_route_summary=("generator:qwen-coder",),
                current_file="sandbox:bugfix.py",
                current_project="sandbox",
                sandbox_state="completed",
                quality_gate_state="passed",
            )
        )
        app.processEvents()

        self.assertTrue(window._approval_overlay.isHidden())
        self.assertFalse(window._coding_workspace_card.isHidden())
        self.assertFalse(window._coding_dock_bar.isHidden())
        self.assertEqual(window._input.placeholderText(), "Describe the coding task...")
        self.assertEqual(window._active_task_title.text(), "Active Coding Task")
        coding_dock_labels = [
            label.text()
            for label in window._coding_dock_bar.findChildren(QtWidgets.QLabel)
            if label.text().strip()
        ]
        self.assertTrue(any("File sandbox:bugfix.py" in text for text in coding_dock_labels))

        window.apply_dashboard_state(DashboardAppState())
        window.apply_shell_state(ShellState())
        app.processEvents()

        self.assertTrue(window._coding_workspace_card.isHidden())
        self.assertTrue(window._coding_dock_bar.isHidden())
        self.assertTrue(window._approval_overlay.isHidden())
        self.assertEqual(window._input.placeholderText(), "Type a message...")
        window.close()
        _cleanup_qt()

    def test_shell_window_renders_long_horizon_tray_and_control_dock_state(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PySideShellWindow(
            shell_state=ShellState(
                status_text="Deep Thought",
                sub_status_text="Running a checkpointed long-horizon cycle.",
                long_horizon_state="running",
                cloud_helper_state="available",
                panel_visibility_state={"long_horizon_tray": True},
                current_task_summary="Use more time to improve confidence and evidence coverage.",
            ),
            app_state=DashboardAppState.from_dict(
                {
                    "user_settings": {
                        "profile_name": "long-run",
                        "ui": {"app_shell": "pyside6"},
                        "reasoning": {"mode": "deep", "thinking_minutes": 180},
                        "long_horizon": {
                            "enabled": True,
                            "wall_clock_minutes": 180,
                            "cycle_budget_minutes": 60,
                            "checkpoint_interval_minutes": 60,
                        },
                        "cloud": {"mode": "auxiliary_only"},
                        "desktop": {"enabled": True},
                    },
                    "active_task": {
                        "execution_mode": "long_horizon",
                        "long_horizon_session_id": "lh-77",
                        "long_horizon_status": "running",
                        "long_horizon_current_phase": "candidate_refresh",
                        "long_horizon_cycle_budget_minutes": 60,
                        "long_horizon_checkpoint_interval_minutes": 60,
                        "long_horizon_elapsed_seconds": 7260,
                        "long_horizon_completed_cycles": 2,
                        "long_horizon_total_cycles": 4,
                        "long_horizon_initial_candidate_count": 2,
                        "long_horizon_peak_candidate_count": 5,
                        "long_horizon_additional_candidate_count": 3,
                        "long_horizon_initial_supporting_evidence_count": 4,
                        "long_horizon_additional_supporting_evidence_count": 2,
                        "long_horizon_total_verification_passes": 3,
                        "long_horizon_total_repairs": 1,
                        "long_horizon_confidence_gain": 0.12,
                        "long_horizon_validity_improved": True,
                        "long_horizon_duty_cycle_ratio": 0.72,
                        "long_horizon_advisory_requested_count": 2,
                        "long_horizon_advisory_accepted_count": 1,
                        "long_horizon_advisory_entries": ("accepted broader evidence refresh",),
                    },
                    "local_task_session": {
                        "session_id": "session-long",
                        "status": "running",
                    },
                }
            ),
        )
        window.show()
        app.processEvents()

        self.assertFalse(window._long_horizon_tray.isHidden())
        self.assertIn("cycle budget 60 min", window._long_horizon_summary.text())
        self.assertIn("Checkpoints 2/4", window._long_horizon_metrics.text())
        self.assertIn("Confidence gain +0.12", window._long_horizon_delta.text())
        self.assertTrue(window._long.isChecked())
        self.assertTrue(window._capability_toggle.isChecked())
        self.assertIn("Mode Long Horizon", window._dock_status.text())
        window.close()
        _cleanup_qt()

    def test_shell_window_persists_dock_preferences_and_coding_model_selection(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        saved_profiles = []
        window = PySideShellWindow(
            shell_state=ShellState(
                workspace_mode="coding_workspace",
                status_text="Coding",
                sub_status_text="Routing code-specialist roles.",
                coding_state="generating",
                active_route_summary=("generator:qwen-coder", "reviewer:qwen-review"),
                current_task_summary="Implement and validate the helper.",
            ),
            app_state=DashboardAppState.from_dict(
                {
                    "user_settings": {
                        "profile_name": "coding-shell",
                        "ui": {"app_shell": "pyside6", "show_utility_drawer": True},
                        "runtime": {"allow_web_fallback": True},
                        "retrieval": {"allow_web_fallback": True},
                        "cloud": {"mode": "auxiliary_only"},
                        "coding": {
                            "enabled": True,
                            "mode": "coding_workspace",
                            "local_only": False,
                            "preferred_models_by_role": {},
                        },
                    },
                    "model_registry_view": {
                        "registrations": (
                            {
                                "registration_id": "code-a",
                                "role": "code_specialist",
                                "backend": "ollama",
                                "model_identifier": "qwen-coder",
                                "metadata": {"coding_roles": ("generator", "reviewer")},
                            },
                            {
                                "registration_id": "code-b",
                                "role": "code_specialist",
                                "backend": "ollama",
                                "model_identifier": "deepseek-coder",
                                "metadata": {"coding_roles": ("generator",)},
                            },
                        ),
                        "last_route_decisions": (
                            {
                                "requested_role": "code_specialist",
                                "selected_registration_id": "code-a",
                                "selected_backend": "ollama",
                                "selected_model_identifier": "qwen-coder",
                                "allowed": True,
                                "metadata": {"coding_role": "generator"},
                            },
                        ),
                        "cache_snapshots": (
                            {
                                "namespace": "routing",
                                "max_entries": 8,
                                "warm_keys": ("code-a",),
                            },
                        ),
                    },
                }
            ),
            save_settings=lambda profile: saved_profiles.append(profile) or True,
        )
        window.show()
        app.processEvents()

        self.assertGreaterEqual(window._coding_role_combo.count(), 1)
        self.assertGreaterEqual(window._coding_model_combo.count(), 2)
        self.assertIn("Warm yes", window._coding_route_summary.text())

        window._local_only_toggle.click()
        app.processEvents()
        self.assertTrue(any(profile.coding["local_only"] for profile in saved_profiles))
        self.assertTrue(any(not profile.runtime["allow_web_fallback"] for profile in saved_profiles))
        self.assertTrue(any(profile.cloud["mode"] == "disabled" for profile in saved_profiles))

        window._coding_model_combo.setCurrentIndex(1)
        app.processEvents()
        self.assertEqual(
            saved_profiles[-1].coding["preferred_models_by_role"]["generator"],
            "code-b",
        )
        window.close()
        _cleanup_qt()

    def test_shell_window_renders_coding_memory_validation_and_shortcut_actions(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        actions = []
        window = PySideShellWindow(
            shell_state=ShellState(
                workspace_mode="coding_workspace",
                status_text="Reviewing",
                sub_status_text="Inspecting validation history and pattern memory.",
                coding_state="reviewing",
                quality_gate_state="partial",
                sandbox_state="skipped",
                active_route_summary=("generator:qwen-coder",),
                current_task_summary="Review the patch and explain the validation caveats.",
            ),
            app_state=DashboardAppState.from_dict(
                {
                    "user_settings": {
                        "profile_name": "coding-audit",
                        "ui": {"app_shell": "pyside6", "show_utility_drawer": True},
                    },
                    "selected_task": {"task_id": "task-export"},
                    "coding_output": {
                        "request_id": "coding-review",
                        "task_type": "code_review",
                        "status": "completed",
                        "active_phase": "reviewing",
                        "prompt": "Review the patch and explain the validation caveats.",
                        "quality_report": {
                            "tests_passed": True,
                            "lint_passed": False,
                            "security_passed": True,
                            "regression_passed": True,
                            "overall_passed": False,
                            "quality_score": 0.61,
                        },
                        "warnings": ("sandbox_execution_skipped", "security_tool_unavailable"),
                    },
                    "coding_practice": {
                        "session_id": "practice-1",
                        "status": "completed",
                        "prompt": "Refactor the kata and capture reusable patterns.",
                        "quality_score": 0.77,
                        "validated_patterns": ("pattern-1",),
                        "rejected_patterns": ("pattern-2",),
                    },
                    "coding_patterns": (
                        {
                            "pattern_id": "pattern-1",
                            "title": "Guarded subprocess helper",
                            "tier": "verified",
                            "language": "python",
                            "quality_score": 0.91,
                            "reuse_count": 3,
                            "validation_history": (
                                {
                                    "validation_id": "validation-1",
                                    "checks_passed": ("tests", "security"),
                                    "checks_failed": (),
                                },
                            ),
                        },
                        {
                            "pattern_id": "pattern-2",
                            "title": "Over-broad filesystem writes",
                            "tier": "rejected",
                            "language": "python",
                            "quality_score": 0.22,
                            "reuse_count": 0,
                            "validation_history": (
                                {
                                    "validation_id": "validation-2",
                                    "checks_passed": (),
                                    "checks_failed": ("security",),
                                },
                            ),
                        },
                    ),
                }
            ),
            request_action=lambda action, payload=None: actions.append((action, payload or {})) or True,
        )
        window.show()
        app.processEvents()

        self.assertGreaterEqual(window._practice_log_list.count(), 1)
        self.assertGreaterEqual(window._coding_patterns_list.count(), 2)
        self.assertGreaterEqual(window._coding_validation_list.count(), 2)
        self.assertGreaterEqual(window._coding_routes_list.count(), 1)
        self.assertTrue(
            any(
                "security_tool_unavailable" in window._coding_validation_list.item(index).text()
                for index in range(window._coding_validation_list.count())
            )
        )

        window._readiness_shortcut.click()
        window._support_bundle_button.click()
        window._trace_export_button.click()
        app.processEvents()

        self.assertIn(("readiness.refresh", {}), actions)
        self.assertIn(("capabilities.refresh", {}), actions)
        self.assertIn(("models.refresh", {}), actions)
        self.assertTrue(any(action == "support.export_bundle" for action, _ in actions))
        self.assertTrue(any(action == "history.export_task_debug" for action, _ in actions))
        window.close()
        _cleanup_qt()

    def test_shell_window_honors_low_resource_preferences_and_survives_resize_churn(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = PySideShellWindow(
            shell_state=ShellState(
                status_text="Ready",
                sub_status_text="Running in a reduced-effects shell profile.",
                current_task_summary="Stay responsive under resize churn.",
            ),
            app_state=DashboardAppState.from_dict(
                {
                    "user_settings": {
                        "profile_name": "low-resource",
                        "ui": {
                            "app_shell": "pyside6",
                            "lightweight_mode": True,
                            "low_resource_mode": True,
                            "reduced_motion": True,
                            "reduced_effects_mode": True,
                            "animation_frame_cap": 15,
                            "orb_size": 80,
                            "status_text_scale": 1.2,
                            "left_drawer_visible": False,
                            "right_drawer_visible": False,
                            "show_utility_drawer": False,
                        },
                    },
                }
            ),
        )
        window.show()
        app.processEvents()

        self.assertTrue(window._orb._minimal)
        self.assertTrue(window._orb._reduced_effects)
        self.assertGreaterEqual(window._orb._timer.interval(), 66)
        self.assertFalse(window._left_dock.isVisible())
        self.assertFalse(window._right_dock.isVisible())
        self.assertFalse(window._bottom_dock.isVisible())

        for index in range(8):
            window.resize(1180 + (index * 15), 760 + (index * 9))
            window.apply_shell_state(
                ShellState(
                    status_text=f"Ready {index}",
                    sub_status_text="Handling resize churn.",
                    current_task_summary="Stay responsive under resize churn.",
                )
            )
            app.processEvents()

        self.assertTrue(window._orb._timer.isActive())
        window.shutdown()
        app.processEvents()
        self.assertFalse(window._orb._timer.isActive())
        _cleanup_qt()

    def test_shell_window_keeps_resource_ribbon_and_operator_views_bounded_under_state_bursts(self) -> None:
        if not pyside6_available():
            self.skipTest("PySide6 is not installed in this environment.")

        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        timeline_entries = tuple(
            {
                "entry_id": f"timeline-{index}",
                "label": f"Event {index}",
                "detail": f"detail {index}",
            }
            for index in range(32)
        )
        window = PySideShellWindow(
            shell_state=ShellState.from_dict(
                {
                    "status_text": "Responding",
                    "sub_status_text": "Absorbing a burst of runtime updates.",
                    "resource_pressure_level": "elevated",
                    "cloud_helper_state": "available",
                    "observation_tier": "vision_on_step",
                    "resource_ribbon_flags": (
                        "pressure:elevated",
                        "cloud_helper",
                        "observation:vision_on_step",
                    ),
                    "panel_visibility_state": {"resource_ribbon": True},
                    "timeline_entries": timeline_entries,
                }
            ),
            app_state=DashboardAppState.from_dict(
                {
                    "user_settings": {
                        "profile_name": "burst-shell",
                        "ui": {"app_shell": "pyside6", "show_utility_drawer": True},
                    },
                    "runtime_health": {
                        "started": True,
                        "generation_backend": "ollama",
                        "embedding_backend": "sentence_transformers",
                        "available_ram_gb": 6.0,
                        "total_ram_gb": 8.0,
                        "active_heavy_roles": ("generation",),
                        "heavy_slot_limit": 2,
                        "governor_active": True,
                        "governor_summary": "memory pressure",
                    },
                    "readiness_report": {
                        "stub_mode_ready": True,
                        "real_mode_ready": False,
                        "checks": (
                            {
                                "check_id": "ollama",
                                "title": "Ollama",
                                "status": "warning",
                                "detail": "offline",
                            },
                        ),
                        "capabilities": (
                            {"capability_name": "desktop_input", "status": "guarded", "reason": "approval required"},
                        ),
                    },
                }
            ),
        )
        window.show()
        app.processEvents()

        self.assertIn("Governor elevated", window._ribbon.text())
        self.assertIn("Cloud helper available", window._ribbon.text())
        self.assertIn("Observation vision_on_step", window._ribbon.text())
        self.assertEqual(window._timeline.count(), 24)
        self.assertIn("Ollama", window._readiness.toPlainText())
        self.assertIn("desktop_input", window._capability.toPlainText())
        window.close()
        _cleanup_qt()

    def test_shell_host_starts_and_stops_cleanly(self) -> None:
        if not pyside6_available():
            with self.assertRaises(PySideShellUnavailableError):
                PySideShellHost(
                    shell_state_provider=ShellState,
                    app_state_provider=DashboardAppState,
                )
            return

        from PySide6 import QtWidgets

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        host = PySideShellHost(
            shell_state_provider=ShellState,
            app_state_provider=DashboardAppState,
            startup_timeout_s=5.0,
        )
        host.start()
        self.assertTrue(host.is_running)
        host.stop(timeout_s=5.0)
        self.assertFalse(host.is_running)
        _cleanup_qt()


if __name__ == "__main__":
    unittest.main()
