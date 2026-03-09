"""Phase 22 screenshot-on-demand and CPU-first OCR regressions."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from capability_runtime import CapabilityExecutor
from data_structures import CapabilityRequest, CapabilityType, OCRRequestSpec, ScreenshotSpec
from orchestrator import Orchestrator
from tests.test_phase20_capability_foundation import _build_test_config, _desktop_profile


class Phase22ObservationExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase22_observation.sqlite3")
        self.test_logs = Path("test_phase22_observation_logs")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()
        self.workspace = (self.test_logs / "workspace").resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_live_screenshot_capture_runs_inside_allowlisted_root(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("screenshot",),
            allowlisted_roots=(str(self.workspace),),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        await self.orchestrator.start_local_task_session(
            "Live screenshot execution",
            active_profile=profile,
        )
        destination = self.workspace / "capture.png"

        def fake_capture(
            _executor: CapabilityExecutor,
            destination_path: Path,
            *,
            region: tuple[int, int, int, int] | None,
            image_format: str,
        ) -> None:
            self.assertEqual(destination_path, destination)
            self.assertIsNone(region)
            self.assertEqual(image_format, "Png")
            destination_path.write_bytes(b"fake-png")

        with patch.object(CapabilityExecutor, "_capture_screenshot_file", new=fake_capture):
            result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-live-screenshot",
                    capability_type=CapabilityType.SCREENSHOT,
                    summary="Capture a bounded screenshot",
                    screenshot=ScreenshotSpec(save_path=str(destination)),
                )
            )

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(result.executor_kind, "live_screenshot")
        self.assertEqual(Path(result.output_ref), destination)
        self.assertTrue(destination.exists())
        self.assertEqual(result.metadata["file_size_bytes"], 8)

    async def test_live_ocr_uses_cpu_first_backend_for_full_image(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("ocr_request",),
            observation_tier="ocr_on_step",
            allowlisted_roots=(str(self.workspace),),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        await self.orchestrator.start_local_task_session(
            "Live OCR execution",
            active_profile=profile,
        )
        source_image = self.workspace / "note.png"
        source_image.write_bytes(b"fake-image")

        def fake_extract(
            _executor: CapabilityExecutor,
            image_path: Path,
            *,
            languages: tuple[str, ...],
        ) -> tuple[str, str]:
            self.assertEqual(image_path, source_image)
            self.assertEqual(languages, ("en-US",))
            return "Bounded OCR text", "windows_ocr"

        with patch.object(CapabilityExecutor, "_extract_ocr_text", new=fake_extract):
            result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-live-ocr",
                    capability_type=CapabilityType.OCR_REQUEST,
                    summary="Read text from a local image",
                    ocr_request=OCRRequestSpec(
                        source_image_path=str(source_image),
                        languages=("en-US",),
                    ),
                )
            )

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(result.executor_kind, "live_ocr")
        self.assertEqual(result.metadata["ocr_backend"], "windows_ocr")
        self.assertEqual(result.metadata["recognized_text"], "Bounded OCR text")
        self.assertFalse(result.warnings)

    async def test_live_ocr_crops_selected_region_before_recognition(self) -> None:
        profile = _desktop_profile(
            enabled_capabilities=("ocr_request",),
            observation_tier="ocr_on_step",
            allowlisted_roots=(str(self.workspace),),
        )
        await self.orchestrator.storage.save_user_settings_profile(profile)
        await self.orchestrator._apply_runtime_settings_profile(profile)
        await self.orchestrator.start_local_task_session(
            "Live OCR crop execution",
            active_profile=profile,
        )
        source_image = self.workspace / "region-source.png"
        cropped_image = self.workspace / "region-cropped.png"
        source_image.write_bytes(b"fake-image")
        cropped_image.write_bytes(b"cropped-image")

        def fake_crop(
            _executor: CapabilityExecutor,
            source_path: Path,
            region: tuple[int, int, int, int],
        ) -> tuple[Path, Path]:
            self.assertEqual(source_path, source_image)
            self.assertEqual(region, (10, 12, 50, 24))
            return cropped_image, cropped_image

        def fake_extract(
            _executor: CapabilityExecutor,
            image_path: Path,
            *,
            languages: tuple[str, ...],
        ) -> tuple[str, str]:
            self.assertEqual(image_path, cropped_image)
            self.assertEqual(languages, ())
            return "Region text", "windows_ocr"

        with (
            patch.object(CapabilityExecutor, "_crop_image_for_ocr", new=fake_crop),
            patch.object(CapabilityExecutor, "_extract_ocr_text", new=fake_extract),
        ):
            result = await self.orchestrator.run_capability_request(
                CapabilityRequest(
                    request_id="cap-live-ocr-region",
                    capability_type=CapabilityType.OCR_REQUEST,
                    summary="Read text from a selected region",
                    ocr_request=OCRRequestSpec(
                        source_image_path=str(source_image),
                        region="10,12,50,24",
                    ),
                )
            )

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(result.metadata["recognized_text"], "Region text")
        self.assertEqual(result.metadata["region"], "10,12,50,24")
        self.assertFalse(cropped_image.exists())


if __name__ == "__main__":
    unittest.main()
