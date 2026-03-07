"""Phase 4 budgeting behavior tests."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from config import APP_CONFIG, BudgetPolicy
from orchestrator import Orchestrator


class BudgetPolicyCalibrationTests(unittest.TestCase):
    """Validate bounded presets and calibration caps."""

    def test_large_minutes_are_bounded_by_calibration_caps(self) -> None:
        budget = BudgetPolicy.from_minutes(10_000)
        calibration = APP_CONFIG.budget_calibration

        self.assertLessEqual(budget.retrieval_top_k, calibration.max_retrieval_top_k)
        self.assertLessEqual(budget.max_web_queries, calibration.max_web_queries)
        self.assertLessEqual(budget.reasoner_passes, calibration.max_reasoner_passes)
        self.assertLessEqual(budget.critic_passes, calibration.max_critic_passes)
        self.assertLessEqual(budget.macro_depth, calibration.max_macro_depth)

    def test_calibration_tracks_baseline_and_dev_profiles(self) -> None:
        calibration = APP_CONFIG.budget_calibration

        self.assertEqual(APP_CONFIG.preflight.hardware.max_vram_gb, 6.0)
        self.assertEqual(APP_CONFIG.preflight.hardware.max_ram_gb, 8.0)
        self.assertEqual(calibration.development_vram_gb, 4.0)
        self.assertEqual(calibration.development_ram_gb, 8.0)
        self.assertLessEqual(calibration.development_vram_gb, APP_CONFIG.preflight.hardware.max_vram_gb)
        self.assertLessEqual(calibration.development_ram_gb, APP_CONFIG.preflight.hardware.max_ram_gb)


class Phase4BudgetRuntimeTests(unittest.IsolatedAsyncioTestCase):
    """Validate thinking time changes real downstream work in stub mode."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase4.sqlite3")
        self.test_logs = Path("test_phase4_logs")
        preflight = replace(
            APP_CONFIG.preflight,
            flags=replace(
                APP_CONFIG.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
                allow_web_fallback=True,
            ),
        )
        storage = replace(APP_CONFIG.storage, sqlite_path=self.test_db, logs_dir=self.test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        self.test_config = replace(
            APP_CONFIG,
            preflight=preflight,
            storage=storage,
            dashboard=dashboard,
        )
        self.orchestrator = Orchestrator(config=self.test_config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_thinking_minutes_change_runtime_work_depth(self) -> None:
        question = "What is the latest local runtime status?"

        low_budget_result = await self.orchestrator.run_task(question, thinking_minutes=1)
        high_budget_result = await self.orchestrator.run_task(question, thinking_minutes=121)

        self.assertLess(len(low_budget_result.plan.steps), len(high_budget_result.plan.steps))
        self.assertLess(
            len(low_budget_result.evidence.local_results),
            len(high_budget_result.evidence.local_results),
        )
        self.assertFalse(low_budget_result.evidence.used_web_fallback)
        self.assertTrue(high_budget_result.evidence.used_web_fallback)
        self.assertEqual(len(low_budget_result.evidence.web_results), 0)
        self.assertGreater(len(high_budget_result.evidence.web_results), 0)
        self.assertLess(len(low_budget_result.reasoning.tokens), len(high_budget_result.reasoning.tokens))
        self.assertIn("checks_run=check.evidence_presence", low_budget_result.critique.critic_notes)
        self.assertIn("check.plan_budget_alignment", high_budget_result.critique.critic_notes)


if __name__ == "__main__":
    unittest.main()
