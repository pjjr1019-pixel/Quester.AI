"""Phase 6 density harness and clean-start runtime bootstrap tests."""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from config import APP_CONFIG
from orchestrator import Orchestrator
from phase6_benchmark import format_trace_density_report, run_trace_density_benchmark


class Phase6TraceDensityBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    """Keep the Phase 6 density benchmark deterministic and runnable."""

    async def test_trace_density_benchmark_is_repeatable(self) -> None:
        first = await run_trace_density_benchmark()
        second = await run_trace_density_benchmark()

        self.assertEqual(tuple(item.scenario_id for item in first), ("baseline", "medium", "wide"))
        self.assertEqual(
            [item.to_dict() for item in first],
            [item.to_dict() for item in second],
        )
        for item in first:
            self.assertGreater(item.token_count, 0)
            self.assertGreater(item.operation_count, 0)
            self.assertGreater(item.entity_count, 0)
            self.assertGreater(item.legacy_json_bytes, 0)
            self.assertGreater(item.ir_json_bytes, 0)
            self.assertGreater(item.legacy_memory_bytes, 0)
            self.assertGreater(item.ir_memory_bytes, 0)
            self.assertGreater(item.json_growth_ratio, 0.0)
            self.assertGreater(item.memory_growth_ratio, 0.0)
        self.assertLess(max(item.json_growth_ratio for item in first), 1.7)
        self.assertLess(max(item.memory_growth_ratio for item in first), 2.7)

        report = format_trace_density_report(first)
        self.assertIn("Phase 6 Trace Density Benchmark", report)
        self.assertIn("baseline", report)
        self.assertIn("wide", report)


class Phase6FreshStartRuntimeBootstrapTests(unittest.IsolatedAsyncioTestCase):
    """Verify clean-start orchestrator runs stay green with the built-in runtime lexicon."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path(f"test_phase6_bootstrap_{self._testMethodName}.sqlite3")
        self.test_logs = Path(f"test_phase6_bootstrap_{self._testMethodName}_logs")
        self.vector_dir = self.test_db.parent / f"{self.test_db.stem}_chroma"
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
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=self.test_db, logs_dir=self.test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        self.test_config = replace(
            APP_CONFIG,
            preflight=preflight,
            storage=storage_cfg,
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
        if self.vector_dir.exists():
            shutil.rmtree(self.vector_dir)

    async def test_clean_start_bootstraps_runtime_lexicon_for_critic(self) -> None:
        lookup_opcode = await self.orchestrator.storage.get_opcode("lookup")
        bind_opcode = await self.orchestrator.storage.get_opcode("bind")
        compare_opcode = await self.orchestrator.storage.get_opcode("compare")
        infer_opcode = await self.orchestrator.storage.get_opcode("infer")
        aggregate_opcode = await self.orchestrator.storage.get_opcode("aggregate")
        check_opcode = await self.orchestrator.storage.get_opcode("check")
        emit_opcode = await self.orchestrator.storage.get_opcode("emit")
        cite_opcode = await self.orchestrator.storage.get_opcode("cite")
        confidence_opcode = await self.orchestrator.storage.get_opcode("confidence_update")
        verified_decoder = await self.orchestrator.storage.get_decoder("verified_answer")
        summary_decoder = await self.orchestrator.storage.get_decoder("compressed_trace_summary")

        self.assertIsNotNone(lookup_opcode)
        self.assertIsNotNone(bind_opcode)
        self.assertIsNotNone(compare_opcode)
        self.assertIsNotNone(infer_opcode)
        self.assertIsNotNone(aggregate_opcode)
        self.assertIsNotNone(check_opcode)
        self.assertIsNotNone(emit_opcode)
        self.assertIsNotNone(cite_opcode)
        self.assertIsNotNone(confidence_opcode)
        self.assertIsNotNone(verified_decoder)
        self.assertIsNotNone(summary_decoder)

        result = await self.orchestrator.run_task(
            "How does local-first retrieval work?",
            thinking_minutes=30,
        )

        self.assertTrue(result.critique.is_valid)
        self.assertNotIn("critique_reported_issues", result.warnings)
        self.assertTrue(result.reasoning.proof_hash)
        self.assertIn("loaded_opcodes=", result.critique.critic_notes)
        self.assertIn("loaded_decoders=", result.critique.critic_notes)
        for name in ("lookup", "bind", "emit", "verified_answer"):
            self.assertIn(name, result.critique.critic_notes)


if __name__ == "__main__":
    unittest.main()
