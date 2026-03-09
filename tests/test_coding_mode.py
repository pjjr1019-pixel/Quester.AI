"""Coding Mode regressions for bounded local coding tasks and practice flows."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from coding_mode import CodingModeService
from config import APP_CONFIG
from data_structures import CodingTaskRequest, CodingTaskType, UserSettingsProfile
from model_manager import ModelManager
from storage import StorageManager


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
            allow_web_fallback=False,
        ),
    )
    storage_cfg = replace(APP_CONFIG.storage, sqlite_path=Path(sqlite_name), logs_dir=Path(logs_name))
    dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
    return replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)


def _coding_profile() -> UserSettingsProfile:
    return UserSettingsProfile.from_dict(
        {
            "profile_name": "coding-mode",
            "models": {
                "preferred_by_role": {
                    "code_specialist": "stub_code_specialist:stub-qwen-coder-1.5b",
                },
                "enabled_roles": ("generation", "embedding", "code_specialist"),
            },
            "coding": {
                "enabled": True,
                "mode": "coding_workspace",
                "practice_when_idle": True,
                "default_language": "python",
            },
        }
    )


class CodingModeServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_coding_mode.sqlite3")
        self.test_logs = Path("test_coding_mode_logs")
        self.config = _build_test_config(sqlite_name=str(self.test_db), logs_name=str(self.test_logs))
        self.storage = StorageManager(config=self.config)
        self.model_manager = ModelManager(config=self.config)
        await self.storage.start()
        await self.model_manager.start()
        self.profile = _coding_profile()
        self.model_manager.apply_user_settings_profile(self.profile)
        self.service = CodingModeService(
            model_manager=self.model_manager,
            storage=self.storage,
            config=self.config,
        )

    async def asyncTearDown(self) -> None:
        await self.model_manager.stop()
        await self.storage.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_code_review_creates_candidate_pattern_and_persists_result(self) -> None:
        result = await self.service.run_task(
            CodingTaskRequest(
                task_type=CodingTaskType.CODE_REVIEW,
                prompt="Review this helper and summarize bounded improvements.",
                language="python",
                source_text=(
                    "def add(left: int, right: int) -> int:\n"
                    "    return left + right\n"
                ),
            ),
            user_settings=self.profile,
        )

        self.assertEqual(result.status, "completed")
        self.assertTrue(result.quality_report.critique_passed)
        self.assertGreaterEqual(result.quality_report.quality_score, 0.45)
        self.assertTrue(result.candidate_patterns)
        persisted = await self.storage.get_coding_task_result(result.request_id)
        self.assertIsNotNone(persisted)
        patterns = await self.storage.list_coding_patterns(limit=8)
        self.assertTrue(any(pattern.pattern_id in result.candidate_patterns for pattern in patterns))

    async def test_idle_practice_cycle_promotes_verified_patterns(self) -> None:
        practice = await self.service.run_idle_practice_cycle(user_settings=self.profile)

        self.assertEqual(practice.status, "completed")
        self.assertTrue(practice.validated_patterns)
        self.assertTrue(practice.task_result.quality_report.tests_passed)
        self.assertTrue(practice.task_result.quality_report.overall_passed)
        patterns = await self.storage.list_coding_patterns(tier="verified", limit=8)
        self.assertTrue(any(pattern.pattern_id in practice.validated_patterns for pattern in patterns))
        sessions = await self.storage.list_coding_practice_sessions(limit=4)
        self.assertEqual(len(sessions), 1)


if __name__ == "__main__":
    unittest.main()
