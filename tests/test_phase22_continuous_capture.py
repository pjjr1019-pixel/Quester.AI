"""Phase 22 continuous-capture cap and lifecycle regressions."""

from __future__ import annotations

import asyncio
import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from capability_runtime import CapabilityExecutor
from config import APP_CONFIG
from orchestrator import Orchestrator
from tests.test_phase20_capability_foundation import _build_test_config, _desktop_profile, _snapshot


class Phase22ContinuousCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase22_continuous_capture.sqlite3")
        self.test_logs = Path("test_phase22_continuous_capture_logs")
        observation_runtime = replace(
            APP_CONFIG.observation_runtime,
            default_capture_fps=2.0,
            max_capture_fps=2.0,
            default_capture_width=320,
            default_capture_height=180,
            max_capture_width=320,
            max_capture_height=180,
            default_frame_history=2,
            max_frame_history=2,
            default_diff_threshold=0.25,
            min_diff_threshold=0.01,
            max_diff_threshold=0.50,
        )
        self.config = replace(
            _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs)),
            observation_runtime=observation_runtime,
        )
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def _wait_for_frame_count(self, session_id: str, minimum: int) -> None:
        deadline = asyncio.get_running_loop().time() + 3.0
        while asyncio.get_running_loop().time() <= deadline:
            session = await self.orchestrator.storage.load_local_task_session(session_id)
            if session is not None and session.continuous_capture_frame_count >= minimum:
                return
            await asyncio.sleep(0.02)
        self.fail(f"Timed out waiting for continuous capture to reach {minimum} frames.")

    async def test_continuous_capture_retains_only_changed_frames_with_history_cap(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("screenshot",),
            observation_tier="continuous_capture",
            observation_overrides={
                "capture_fps": 2.0,
                "capture_max_width": 320,
                "capture_max_height": 180,
                "capture_frame_history": 2,
                "capture_diff_threshold": 0.25,
            },
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)

        frames = [b"AAAA", b"AAAA", b"BBBB", b"CCCC", b"CCCC"]
        call_index = 0

        def fake_capture(
            _executor: CapabilityExecutor,
            destination_path: Path,
            *,
            region: str,
            max_width: int,
            max_height: int,
        ) -> None:
            nonlocal call_index
            payload = frames[min(call_index, len(frames) - 1)]
            call_index += 1
            self.assertEqual(region, "full_screen")
            self.assertEqual((max_width, max_height), (320, 180))
            destination_path.write_bytes(payload)

        with patch.object(CapabilityExecutor, "capture_continuous_frame", new=fake_capture):
            session = await self.orchestrator.start_local_task_session(
                "Continuous capture retention",
                active_profile=profile,
            )
            await self._wait_for_frame_count(session.session_id, 4)
            await self.orchestrator.pause_local_task_session(session.session_id, reason="test_complete")

        paused = await self.orchestrator.storage.load_local_task_session(session.session_id)
        assert paused is not None
        capture_dir = Path(paused.continuous_capture_directory)

        self.assertEqual(paused.status.value, "paused")
        self.assertFalse(paused.continuous_capture_active)
        self.assertGreaterEqual(paused.continuous_capture_frame_count, 4)
        self.assertEqual(paused.continuous_capture_retained_frame_count, 2)
        self.assertTrue(capture_dir.exists())
        self.assertLessEqual(len(tuple(capture_dir.glob("frame_*.jpg"))), 2)
        self.assertTrue(paused.continuous_capture_last_frame_path.endswith(".jpg"))

    async def test_continuous_capture_clamps_runtime_caps_and_reports_ready_tier(self) -> None:
        capped_runtime = replace(
            self.config.observation_runtime,
            default_capture_fps=0.5,
            max_capture_fps=0.5,
            default_capture_width=320,
            default_capture_height=180,
            max_capture_width=320,
            max_capture_height=180,
            default_frame_history=2,
            max_frame_history=2,
        )
        capped_config = replace(self.config, observation_runtime=capped_runtime)
        capped_orchestrator = Orchestrator(config=capped_config)
        await capped_orchestrator.start()
        try:
            profile = _desktop_profile(
                enabled_capabilities=("screenshot",),
                observation_tier="continuous_capture",
                observation_overrides={
                    "capture_fps": 1.0,
                    "capture_max_width": 960,
                    "capture_max_height": 540,
                    "capture_frame_history": 4,
                    "region_of_interest": "10,12,800,700",
                },
            )
            await capped_orchestrator.storage.save_user_settings_profile(profile)
            await capped_orchestrator._apply_runtime_settings_profile(profile)
            session = await capped_orchestrator.start_local_task_session(
                "Continuous capture caps",
                active_profile=profile,
            )
            persisted = await capped_orchestrator.storage.load_local_task_session(session.session_id)
            report = capped_orchestrator._build_dashboard_readiness_report(active_profile=profile)
            capabilities = {item.capability_name: item for item in report.capabilities}

            assert persisted is not None
            self.assertTrue(persisted.continuous_capture_active)
            self.assertEqual(persisted.continuous_capture_fps, 0.5)
            self.assertEqual(persisted.continuous_capture_max_width, 320)
            self.assertEqual(persisted.continuous_capture_max_height, 180)
            self.assertEqual(persisted.continuous_capture_region, "10,12,320,180")
            self.assertIn("continuous_capture_fps_capped", persisted.continuous_capture_warnings)
            self.assertIn("continuous_capture_width_capped", persisted.continuous_capture_warnings)
            self.assertIn("continuous_capture_height_capped", persisted.continuous_capture_warnings)
            self.assertIn("continuous_capture_history_capped", persisted.continuous_capture_warnings)
            self.assertIn("continuous_capture_region_clamped", persisted.continuous_capture_warnings)
            self.assertEqual(capabilities["observation_tiers"].reason, "continuous_capture_live")
        finally:
            await capped_orchestrator.stop()

    async def test_continuous_capture_degrades_to_screenshot_on_demand_under_pressure(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("screenshot",),
            observation_tier="continuous_capture",
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        with patch.object(
            self.orchestrator.model_manager,
            "health_snapshot",
            return_value=_snapshot(available_ram_gb=0.005),
        ):
            session = await self.orchestrator.start_local_task_session(
                "Continuous capture pressure",
                active_profile=profile,
            )
            await self.orchestrator._emit_health_snapshot()
            report = self.orchestrator._build_dashboard_readiness_report(active_profile=profile)

        persisted = await self.orchestrator.storage.load_local_task_session(session.session_id)
        state = self.orchestrator.dashboard.app_state_snapshot()
        capabilities = {item.capability_name: item for item in report.capabilities}

        assert persisted is not None
        self.assertFalse(persisted.continuous_capture_active)
        self.assertEqual(persisted.requested_observation_tier, "continuous_capture")
        self.assertEqual(persisted.effective_observation_tier, "screenshot_on_demand")
        self.assertIn("low_available_ram", persisted.observation_degraded_reason)
        self.assertIn("continuous_capture", persisted.observation_degraded_features)
        self.assertEqual(capabilities["observation_tiers"].status, "degraded")
        self.assertEqual(
            capabilities["observation_tiers"].reason,
            "hardware_governor_continuous_capture_degraded",
        )
        self.assertEqual(state.local_task_session.effective_observation_tier, "screenshot_on_demand")
        self.assertIn("low_available_ram", state.local_task_session.observation_degraded_reason)
        self.assertTrue(state.runtime_health.governor_active)
        self.assertIn("continuous_capture", state.runtime_health.governor_degraded_features)


if __name__ == "__main__":
    unittest.main()
