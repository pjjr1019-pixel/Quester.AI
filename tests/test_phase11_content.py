"""Phase 11 demo-pack fixtures, loading, and sample-task coverage."""

from __future__ import annotations

import json
import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from config import APP_CONFIG
from data_structures import CritiqueResult, VerifiedDeepTraceExport
from orchestrator import Orchestrator
from phase11_content import Phase11ContentLoader
from storage import StorageManager


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


async def _fake_embed_document(_text: str) -> list[float]:
    return [0.05] * APP_CONFIG.model_tuning.embedding_dimensions


class Phase11FixtureValidationTests(unittest.TestCase):
    """Validate the repo-owned Phase 11 fixture pack."""

    def test_phase11_loader_parses_all_fixture_files(self) -> None:
        loader = Phase11ContentLoader(config=APP_CONFIG)

        documents = loader.load_demo_documents()
        sample_tasks = loader.load_sample_tasks()
        macros = loader.load_starter_macros()
        opcodes, decoders, runtime_pack = loader.load_starter_runtime_pack()
        exports = loader.load_packaged_verified_trace_exports()

        self.assertEqual(loader._package_version(), "v1")
        self.assertEqual(len(documents), 4)
        self.assertEqual(len(sample_tasks), 5)
        self.assertEqual(len(macros), 3)
        self.assertEqual(len(opcodes), 2)
        self.assertEqual(len(decoders), 2)
        self.assertEqual(runtime_pack.pack_version, "v1")
        self.assertIsInstance(exports[0], VerifiedDeepTraceExport)
        self.assertEqual(exports[0].task_id, "phase11-demo-deep-trace")
        self.assertEqual(exports[0].critique.result, CritiqueResult.VALID)


class Phase11DemoPackPersistenceTests(unittest.IsolatedAsyncioTestCase):
    """Verify the Phase 11 demo pack loads into typed local persistence surfaces."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase11_storage.sqlite3")
        self.test_logs = Path("test_phase11_storage_logs")
        self.config = _build_test_config(str(self.test_db), str(self.test_logs))
        self.storage = StorageManager(config=self.config)
        self.loader = Phase11ContentLoader(config=self.config)
        await self.storage.start()

    async def asyncTearDown(self) -> None:
        await self.storage.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_demo_pack_loads_with_demo_only_metadata_and_runtime_pack(self) -> None:
        status = await self.loader.load_demo_pack(
            storage=self.storage,
            embed_document=_fake_embed_document,
            embedding_model_name="stub-embed",
        )

        documents = await self.storage.list_source_documents()
        demo_documents = [
            document
            for document in documents
            if str(document.metadata.get("corpus_origin", "")) == "demo_phase11"
        ]
        macros = await self.storage.list_macros(active_only=False)
        opcodes = await self.storage.list_opcodes(active_only=False)
        decoders = await self.storage.list_decoders(active_only=False)

        self.assertTrue(status.loaded)
        self.assertEqual(status.pack_version, "v1")
        self.assertEqual(len(demo_documents), 4)
        self.assertTrue(
            all(document.metadata.get("corpus_tier") == "seed_demo" for document in demo_documents)
        )
        self.assertTrue(
            all(document.metadata.get("demo_pack_version") == "v1" for document in demo_documents)
        )
        self.assertTrue(
            {"summarize_demo_storage_layers", "compare_aurora_conflict", "emit_verified_python_result"}.issubset(
                {macro.macro_name for macro in macros}
            )
        )
        self.assertTrue({"contrast", "qualify"}.issubset({opcode.opcode_name for opcode in opcodes}))
        self.assertTrue(
            {"demo_storage_summary", "demo_conflict_notice"}.issubset(
                {decoder.decoder_name for decoder in decoders}
            )
        )


class Phase11OrchestratorExampleTests(unittest.IsolatedAsyncioTestCase):
    """Verify Phase 11 examples are first-class dashboard/orchestrator actions."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase11_orchestrator.sqlite3")
        self.test_logs = Path("test_phase11_orchestrator_logs")
        self.config = _build_test_config(str(self.test_db), str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        self.loader = Phase11ContentLoader(config=self.config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_dashboard_examples_publish_select_run_and_export(self) -> None:
        initial_state = self.orchestrator.dashboard.app_state_snapshot()
        export_path = self.test_logs / "phase11_packaged_export.jsonl"

        self.assertEqual(initial_state.demo_pack_status.pack_version, "v1")
        self.assertEqual(len(initial_state.sample_tasks), 5)

        await self.orchestrator._run_dashboard_action(
            action="examples.select_sample",
            payload={"sample_id": "arithmetic_exact"},
        )
        selected_state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertEqual(selected_state.selected_sample_task.sample_id, "arithmetic_exact")

        await self.orchestrator._run_dashboard_action(
            action="examples.run_sample_task",
            payload={"sample_id": "arithmetic_exact"},
        )
        run_state = self.orchestrator.dashboard.app_state_snapshot()
        self.assertIn("76", run_state.active_task.answer_text)

        await self.orchestrator._run_dashboard_action(
            action="examples.export_verified_trace_example",
            payload={"path": str(export_path)},
        )
        self.assertTrue(export_path.exists())
        with export_path.open("r", encoding="utf-8") as handle:
            exported_rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(exported_rows), 1)
        self.assertEqual(exported_rows[0]["task_id"], "phase11-demo-deep-trace")

    async def test_demo_pack_load_action_refreshes_examples_and_knowledge_library(self) -> None:
        await self.orchestrator._run_dashboard_action(action="examples.load_demo_pack", payload={})

        state = self.orchestrator.dashboard.app_state_snapshot()
        demo_sources = [
            source for source in state.knowledge_sources if source.corpus_origin == "demo_phase11"
        ]

        self.assertTrue(state.demo_pack_status.loaded)
        self.assertEqual(len(demo_sources), 4)


class Phase11SampleTaskOutcomeTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the deterministic Phase 11 sample-task outcomes."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase11_samples.sqlite3")
        self.test_logs = Path("test_phase11_samples_logs")
        self.config = _build_test_config(str(self.test_db), str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        self.loader = Phase11ContentLoader(config=self.config)
        await self.orchestrator.start()
        await self.loader.load_demo_pack(
            storage=self.orchestrator.storage,
            embed_document=self.orchestrator.model_manager.embed_document,
            embedding_model_name=self.config.preflight.backends.embedding_model,
        )

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_storage_layers_sample_shows_fast_vs_deep_difference(self) -> None:
        sample = self.loader.get_sample_task("storage_layers_comparison")
        assert sample is not None

        fast_result = await self.orchestrator.run_task(sample.question, sample.comparison_fast_minutes)
        deep_result = await self.orchestrator.run_task(sample.question, sample.comparison_deep_minutes)

        fast_answer = fast_result.answer_text.lower()
        deep_answer = deep_result.answer_text.lower()

        self.assertNotEqual(fast_answer, deep_answer)
        self.assertIn("sqlite", deep_answer)
        self.assertIn("jsonl", deep_answer)
        self.assertFalse("sqlite" in fast_answer and "jsonl" in fast_answer)

    async def test_web_fallback_sample_uses_stub_web_search(self) -> None:
        sample = self.loader.get_sample_task("web_runtime_status")
        assert sample is not None

        result = await self.orchestrator.run_task(sample.question, sample.recommended_thinking_minutes)

        self.assertTrue(result.evidence.used_web_fallback)
        self.assertTrue(result.evidence.web_results)
        self.assertTrue(all(item.source_ref.startswith("https://stub.example/") for item in result.evidence.web_results))

    async def test_python_code_sample_is_tool_verified(self) -> None:
        sample = self.loader.get_sample_task("python_code_result")
        assert sample is not None

        result = await self.orchestrator.run_task(sample.question, sample.recommended_thinking_minutes)

        self.assertEqual(result.critique.verifier_type, "tool.python_code_execution")
        self.assertIn("8", result.answer_text)

    async def test_conflicting_evidence_sample_degrades_and_abstains(self) -> None:
        sample = self.loader.get_sample_task("project_aurora_conflict")
        assert sample is not None

        result = await self.orchestrator.run_task(sample.question, sample.recommended_thinking_minutes)

        self.assertEqual(result.critique.result, CritiqueResult.DEGRADED)
        self.assertEqual(result.critique.degraded_reason, "conflicting_evidence")
        self.assertIn("insufficient evidence", result.answer_text.lower())
        self.assertIn("repair_applied:abstain_due_to_low_grounding", result.warnings)
