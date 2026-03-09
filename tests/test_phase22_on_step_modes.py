"""Phase 22.4 explicit OCR-on-step and vision-on-step regressions."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from capability_runtime import CapabilityExecutor, WindowSnapshot
from data_structures import AppFocusSpec, CapabilityRequest, CapabilityType
from orchestrator import Orchestrator
from tests.test_phase20_capability_foundation import _build_test_config, _desktop_profile, _snapshot


class Phase22OnStepObservationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase22_on_step.sqlite3")
        self.test_logs = Path("test_phase22_on_step_logs")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_ocr_on_step_captures_and_ocrs_after_focus_change(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("app_window_focus",),
            observation_tier="ocr_on_step",
            allowlisted_apps=("notepad",),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "OCR on step",
            active_profile=profile,
        )
        matched_window = WindowSnapshot(
            hwnd=301,
            title="notes.txt - Notepad",
            process_name="notepad",
            pid=5555,
        )

        def fake_capture(
            _executor: CapabilityExecutor,
            destination_path: Path,
            *,
            region: str,
            max_width: int,
            max_height: int,
            image_format: str,
        ) -> None:
            self.assertEqual(region, "full_screen")
            self.assertEqual(image_format, "Png")
            self.assertEqual((max_width, max_height), (960, 540))
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_bytes(b"step-image")

        def fake_ocr(
            _executor: CapabilityExecutor,
            source_path: Path,
            *,
            region: str,
            languages: tuple[str, ...],
        ) -> tuple[str, str, tuple[str, ...], int, bool]:
            self.assertTrue(source_path.exists())
            self.assertEqual(region, "full_image")
            self.assertEqual(languages, ())
            return "File Edit View", "windows_ocr", (), 14, False

        with (
            patch.object(CapabilityExecutor, "_wait_for_window_match", return_value=matched_window),
            patch.object(CapabilityExecutor, "_focus_window", return_value=True),
            patch.object(CapabilityExecutor, "_foreground_window_title", return_value="notes.txt - Notepad"),
            patch.object(CapabilityExecutor, "capture_observation_frame", new=fake_capture),
            patch.object(CapabilityExecutor, "extract_bounded_ocr_text", new=fake_ocr),
        ):
            result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-ocr-on-step-focus",
                    capability_type=CapabilityType.APP_WINDOW_FOCUS,
                    summary="Focus the allowlisted editor",
                    app_focus=AppFocusSpec(
                        app_name="notepad",
                        window_title="notes.txt",
                        require_visible_match=True,
                    ),
                )
            )

        final_session = await self.orchestrator.storage.load_local_task_session(session.session_id)

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(final_session.last_observation_tier, "ocr_on_step")
        self.assertEqual(final_session.last_observation_status, "succeeded")
        self.assertEqual(final_session.last_observation_backend, "windows_ocr")
        self.assertEqual(final_session.last_observation_text_preview, "File Edit View")
        self.assertIn("OCR-on-step captured", final_session.last_observation_summary)
        self.assertTrue(Path(final_session.last_observation_output_ref).exists())

    async def test_vision_on_step_falls_back_to_cpu_ocr_when_vision_role_is_disabled(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("app_window_focus",),
            observation_tier="vision_on_step",
            allowlisted_apps=("notepad",),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "Vision on step",
            active_profile=profile,
        )
        matched_window = WindowSnapshot(
            hwnd=302,
            title="notes.txt - Notepad",
            process_name="notepad",
            pid=6666,
        )

        def fake_capture(
            _executor: CapabilityExecutor,
            destination_path: Path,
            *,
            region: str,
            max_width: int,
            max_height: int,
            image_format: str,
        ) -> None:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_bytes(b"vision-step-image")

        def fake_ocr(
            _executor: CapabilityExecutor,
            source_path: Path,
            *,
            region: str,
            languages: tuple[str, ...],
        ) -> tuple[str, str, tuple[str, ...], int, bool]:
            self.assertTrue(source_path.exists())
            return "Toolbar", "windows_ocr", (), 7, False

        with (
            patch.object(CapabilityExecutor, "_wait_for_window_match", return_value=matched_window),
            patch.object(CapabilityExecutor, "_focus_window", return_value=True),
            patch.object(CapabilityExecutor, "_foreground_window_title", return_value="notes.txt - Notepad"),
            patch.object(CapabilityExecutor, "capture_observation_frame", new=fake_capture),
            patch.object(CapabilityExecutor, "extract_bounded_ocr_text", new=fake_ocr),
        ):
            result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-vision-on-step-focus",
                    capability_type=CapabilityType.APP_WINDOW_FOCUS,
                    summary="Focus the allowlisted editor",
                    app_focus=AppFocusSpec(
                        app_name="notepad",
                        window_title="notes.txt",
                        require_visible_match=True,
                    ),
                )
            )

        final_session = await self.orchestrator.storage.load_local_task_session(session.session_id)
        route_decision = self.orchestrator.model_manager.registry_view().last_route_decisions[-1]

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(final_session.last_observation_tier, "vision_on_step")
        self.assertEqual(final_session.last_observation_status, "degraded")
        self.assertEqual(final_session.last_observation_backend, "windows_ocr")
        self.assertEqual(final_session.last_observation_text_preview, "Toolbar")
        self.assertIn("fell back to CPU OCR", final_session.last_observation_summary)
        self.assertTrue(any(item.startswith("vision_route_") for item in final_session.last_observation_warnings))
        self.assertEqual(route_decision.requested_role.value, "vision")
        self.assertFalse(route_decision.allowed)

    async def test_vision_on_step_uses_routed_stub_vision_when_role_is_enabled(self) -> None:
        base_profile = _desktop_profile(
            enabled_capabilities=("app_window_focus",),
            observation_tier="vision_on_step",
            allowlisted_apps=("notepad",),
        )
        profile = replace(
            base_profile,
            models={
                **base_profile.models,
                "enabled_roles": tuple(
                    sorted(
                        {
                            *(str(item) for item in base_profile.models.get("enabled_roles", ())),
                            "generation",
                            "embedding",
                            "vision",
                        }
                    )
                ),
            },
        )
        profile.validate()
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "Vision on step routed",
            active_profile=profile,
        )
        matched_window = WindowSnapshot(
            hwnd=312,
            title="notes.txt - Notepad",
            process_name="notepad",
            pid=6767,
        )

        def fake_capture(
            _executor: CapabilityExecutor,
            destination_path: Path,
            *,
            region: str,
            max_width: int,
            max_height: int,
            image_format: str,
        ) -> None:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.write_bytes(b"vision-step-image")

        def fake_ocr(
            _executor: CapabilityExecutor,
            source_path: Path,
            *,
            region: str,
            languages: tuple[str, ...],
        ) -> tuple[str, str, tuple[str, ...], int, bool]:
            self.assertTrue(source_path.exists())
            return "Toolbar", "windows_ocr", (), 7, False

        with (
            patch.object(CapabilityExecutor, "_wait_for_window_match", return_value=matched_window),
            patch.object(CapabilityExecutor, "_focus_window", return_value=True),
            patch.object(CapabilityExecutor, "_foreground_window_title", return_value="notes.txt - Notepad"),
            patch.object(CapabilityExecutor, "capture_observation_frame", new=fake_capture),
            patch.object(CapabilityExecutor, "extract_bounded_ocr_text", new=fake_ocr),
        ):
            result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-vision-on-step-routed",
                    capability_type=CapabilityType.APP_WINDOW_FOCUS,
                    summary="Focus the allowlisted editor",
                    app_focus=AppFocusSpec(
                        app_name="notepad",
                        window_title="notes.txt",
                        require_visible_match=True,
                    ),
                )
            )

        final_session = await self.orchestrator.storage.load_local_task_session(session.session_id)
        route_decision = self.orchestrator.model_manager.registry_view().last_route_decisions[-1]

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(final_session.last_observation_tier, "vision_on_step")
        self.assertEqual(final_session.last_observation_status, "succeeded")
        self.assertEqual(final_session.last_observation_backend, "stub_vision")
        self.assertEqual(final_session.last_observation_text_preview, "Toolbar")
        self.assertIn("Stub vision review", final_session.last_observation_summary)
        self.assertEqual(route_decision.requested_role.value, "vision")
        self.assertTrue(route_decision.allowed)
        self.assertIn("vision", route_decision.active_heavy_roles)
        self.assertEqual(len(route_decision.active_heavy_roles), 2)

    async def test_on_step_observation_skips_under_resource_pressure(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("app_window_focus",),
            observation_tier="ocr_on_step",
            allowlisted_apps=("notepad",),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        session = await self.orchestrator.start_local_task_session(
            "OCR on step pressure",
            active_profile=profile,
        )
        matched_window = WindowSnapshot(
            hwnd=303,
            title="notes.txt - Notepad",
            process_name="notepad",
            pid=7777,
        )

        with (
            patch.object(CapabilityExecutor, "_wait_for_window_match", return_value=matched_window),
            patch.object(CapabilityExecutor, "_focus_window", return_value=True),
            patch.object(CapabilityExecutor, "_foreground_window_title", return_value="notes.txt - Notepad"),
            patch.object(
                self.orchestrator.model_manager,
                "health_snapshot",
                return_value=_snapshot(available_ram_gb=0.005),
            ),
            patch.object(CapabilityExecutor, "capture_observation_frame") as capture_frame,
        ):
            result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-ocr-on-step-pressure",
                    capability_type=CapabilityType.APP_WINDOW_FOCUS,
                    summary="Focus the allowlisted editor",
                    app_focus=AppFocusSpec(
                        app_name="notepad",
                        window_title="notes.txt",
                        require_visible_match=True,
                    ),
                )
            )

        final_session = await self.orchestrator.storage.load_local_task_session(session.session_id)

        self.assertEqual(result.status.value, "succeeded")
        capture_frame.assert_not_called()
        self.assertEqual(final_session.last_observation_tier, "ocr_on_step")
        self.assertEqual(final_session.last_observation_status, "degraded")
        self.assertIn("skipped", final_session.last_observation_summary)
        self.assertIn("low_available_ram", final_session.last_observation_warnings)
        self.assertEqual(final_session.last_observation_output_ref, "")


if __name__ == "__main__":
    unittest.main()
