"""Phase 5 persistence and typed status/event tests."""

from __future__ import annotations

import json
import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from config import APP_CONFIG
from critic import CriticAgent
from data_structures import (
    AgentState,
    AgentStatus,
    CanonicalReasoningGraph,
    CompressedTrace,
    ContextFrame,
    DecoderEntry,
    CritiqueReport,
    CritiqueResult,
    DecodeHint,
    EvidenceBundle,
    EvidenceItem,
    Macro,
    MacroProposal,
    OperationStep,
    OpcodeEntry,
    OptimizerReplaySample,
    PerformanceMetric,
    Plan,
    PlanStep,
    ProofHashRecord,
    ProvenanceBundle,
    ResourceBudget,
    ReasoningLog,
    RuntimeEvent,
    SemanticActivity,
    SemanticAgent,
    SemanticEntity,
    SeverityLevel,
    SourceType,
    SymbolTableSnapshot,
    TaskResult,
    WebEvidenceRecord,
)
from model_manager import ModelManager
from orchestrator import Orchestrator
from reasoner import ReasonerAgent
from runtime_errors import WebLookupTimeoutError
from storage import StorageManager


def _build_task_result(task_id: str) -> TaskResult:
    plan = Plan(
        task_id=task_id,
        question="What should be persisted?",
        steps=(PlanStep(step_id="step_1", description="Persist task output"),),
        required_evidence=("local docs",),
        success_criteria=("task result is stored",),
    )
    evidence_item = EvidenceItem(
        id="ev-1",
        content="Local evidence sample",
        source_type=SourceType.LOCAL,
        source_ref="local://sample",
        score=0.9,
        metadata={"topic": "persistence"},
    )
    evidence = EvidenceBundle(
        task_id=task_id,
        local_results=(evidence_item,),
        web_results=(),
        used_web_fallback=False,
    )
    graph = CanonicalReasoningGraph(
        entities=(
            SemanticEntity(
                entity_id="ent_question",
                entity_type="question",
                value=plan.question,
            ),
            SemanticEntity(
                entity_id="ent_answer",
                entity_type="answer_fragment",
                value="Persistence answer text",
                evidence_handles=(evidence_item.id,),
                confidence=0.8,
            ),
        ),
        activities=(
            SemanticActivity(
                activity_id="act_reason",
                activity_type="reason",
                input_entity_ids=("ent_question",),
                output_entity_ids=("ent_answer",),
                agent_id="agent_reasoner",
                evidence_handles=(evidence_item.id,),
            ),
        ),
        agents=(
            SemanticAgent(
                agent_id="agent_reasoner",
                component="reasoner",
                backend="stub_generation",
                role="foreground_reasoning",
            ),
        ),
        bundles=(
            ProvenanceBundle(
                bundle_id="bundle_primary",
                entity_ids=("ent_question", "ent_answer"),
                activity_ids=("act_reason",),
                agent_ids=("agent_reasoner",),
            ),
        ),
    )
    context_frame = ContextFrame(
        frame_id="ctx_primary",
        scope="task",
        confidence=0.8,
        provenance_bundle_id="bundle_primary",
    )
    trace = CompressedTrace(
        task_id=task_id,
        tokens=("@read_question", "@compose_answer"),
        expanded_preview=("Read question", "Compose answer"),
        macros_used=("@compose_answer",),
        confidence=0.8,
        ir_version="1",
        canonical_graph=graph,
        operation_stream=(
            OperationStep(
                op_id="op_emit",
                opcode="emit",
                args=("sym_answer",),
                context_frame_id=context_frame.frame_id,
                evidence_handles=(evidence_item.id,),
            ),
        ),
        symbol_table_refs=("sym_question", "sym_answer"),
        evidence_handles=(evidence_item.id,),
        context_frames=(context_frame,),
        proof_hash="proof123",
        decode_hints=(
            DecodeHint(
                hint_id="hint_answer",
                template="verified_answer",
                entity_ids=("ent_answer",),
            ),
        ),
    )
    critique = CritiqueReport(
        task_id=task_id,
        is_valid=True,
        issues=(),
        fixed_trace=trace,
        evidence_coverage=1.0,
        result=CritiqueResult.VALID,
    )
    proposal = MacroProposal(
        proposal_id=f"{task_id}:compose",
        macro=Macro(macro_name="compose_answer", expansion=("@compose_answer",), version=1),
        reason="Repeated compose token.",
        examples=("@compose_answer",),
        simulation_score=0.5,
        approved=False,
    )
    metric = PerformanceMetric(
        task_id=task_id,
        time=0.25,
        vram_usage=0.0,
        iterations=3,
    )
    return TaskResult(
        task_id=task_id,
        plan=plan,
        evidence=evidence,
        reasoning=trace,
        critique=critique,
        compression=(proposal,),
        answer_text="Persistence answer text",
        warnings=("warning.one",),
        metrics=(metric,),
    )


class _FailingWebAdapter:
    provider_name = "failing_web"

    async def search(self, query: str, *, max_results: int):
        _ = (query, max_results)
        raise WebLookupTimeoutError("timed out")


class StorageManagerTypedPersistenceTests(unittest.IsolatedAsyncioTestCase):
    """Validate direct typed persistence surfaces in storage."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase5_persistence.sqlite3")
        self.test_logs = Path("test_phase5_persistence_logs")
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
            ),
        )
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=self.test_db, logs_dir=self.test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        self.test_config = replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)
        self.storage = StorageManager(config=self.test_config)
        await self.storage.start()

    async def asyncTearDown(self) -> None:
        await self.storage.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_storage_persists_task_results_statuses_events_traces_and_metrics(self) -> None:
        result = _build_task_result("persist-task")
        status = AgentStatus(
            component="planner",
            state=AgentState.IDLE,
            task_id=result.task_id,
            severity=SeverityLevel.LOW,
            message="planning completed",
        )
        event = RuntimeEvent(stage="pipeline.completed", payload={"task_id": result.task_id})
        reasoning_log = ReasoningLog(
            task_id=result.task_id,
            compressed_chain=result.reasoning.tokens,
            macros_used=result.reasoning.macros_used,
        )

        await self.storage.record_task_result(result)
        await self.storage.record_agent_status(status)
        await self.storage.record_runtime_event(event)
        await self.storage.record_reasoning_trace(result.reasoning)
        await self.storage.record_reasoning_log(reasoning_log)
        await self.storage.record_performance_metric(result.metrics[0])
        await self.storage.register_macro(result.compression[0].macro)

        stored_result = await self.storage.get_task_result(result.task_id)
        statuses = await self.storage.list_agent_statuses(task_id=result.task_id)
        events = await self.storage.list_runtime_events(stage="pipeline.completed")
        reasoning_history = await self.storage.list_reasoning_history(task_id=result.task_id)
        traces = await self.storage.list_reasoning_traces(task_id=result.task_id)
        metrics = await self.storage.list_performance_metrics(task_id=result.task_id)
        proof_hashes = await self.storage.list_proof_hashes(task_id=result.task_id)
        replay_samples = await self.storage.list_optimizer_replay_samples(task_id=result.task_id)
        stored_macro = await self.storage.get_macro("compose_answer")

        self.assertEqual(stored_result, result)
        self.assertEqual(statuses, (status,))
        self.assertEqual(events[-1].payload["task_id"], result.task_id)
        self.assertEqual(len(reasoning_history), 2)
        self.assertEqual(traces, (result.reasoning,))
        self.assertEqual(metrics, result.metrics)
        self.assertEqual(proof_hashes[-1].proof_hash, result.reasoning.proof_hash)
        self.assertEqual(len(replay_samples), 1)
        self.assertIsInstance(replay_samples[0], OptimizerReplaySample)
        self.assertEqual(replay_samples[0].task_id, result.task_id)
        self.assertEqual(replay_samples[0].trace_proof_hash, result.reasoning.proof_hash)
        self.assertEqual(replay_samples[0].final_adjudication, result.critique.result)
        self.assertEqual(replay_samples[0].latency_s, result.metrics[0].time)
        self.assertEqual(replay_samples[0].iterations, result.metrics[0].iterations)
        self.assertEqual(stored_macro, result.compression[0].macro)
        self.assertTrue((self.test_logs / self.test_config.storage.trace_log_name).exists())
        self.assertTrue((self.test_logs / self.test_config.storage.status_log_name).exists())
        self.assertEqual(await self.storage.count_tasks(), 1)

    async def test_storage_persists_web_evidence_records(self) -> None:
        record = WebEvidenceRecord(
            task_id="web-task",
            query="What is the current runtime status?",
            provider="fake_web",
            reason="freshness_or_recentness_requested",
            evidence=EvidenceItem(
                id="web_123",
                content="Current runtime status is healthy and bounded.",
                source_type=SourceType.WEB,
                source_ref="https://example.com/runtime",
                score=0.77,
                metadata={"provider": "fake_web", "title": "Runtime status"},
            ),
            lookup_metadata={"attempt_count": 1},
        )

        await self.storage.record_web_evidence(record)

        stored_records = await self.storage.list_web_evidence(task_id="web-task")

        self.assertEqual(stored_records, (record,))
        web_log_path = self.test_logs / self.test_config.storage.web_log_name
        self.assertTrue(web_log_path.exists())
        with web_log_path.open("r", encoding="utf-8") as handle:
            web_entries = [json.loads(line) for line in handle]
        self.assertEqual(web_entries[-1]["kind"], "web_evidence")
        self.assertEqual(web_entries[-1]["task_id"], "web-task")

    async def test_storage_persists_runtime_registries_and_loads_active_subset(self) -> None:
        compose_macro = Macro(
            macro_name="compose_answer",
            expansion=("@compose_answer",),
            version=1,
            opcode_pattern=("emit",),
            invariants=(
                "deterministic_round_trip",
                "provenance_preserving",
                "uncertainty_preserving",
            ),
            proof_fingerprint="fingerprint-compose",
            semantic_kind="token_macro",
            decoder_template="Answer: {value}",
        )
        unused_macro = Macro(
            macro_name="unused_macro",
            expansion=("@unused",),
            version=1,
            is_active=False,
        )
        opcode = OpcodeEntry(
            opcode_name="lookup",
            description="Lookup evidence by handle.",
            metadata={"arity": 1},
        )
        decoder = DecoderEntry(
            decoder_name="emit_answer",
            template="Answer: {value}",
            metadata={"channel": "final"},
        )
        snapshot = SymbolTableSnapshot(
            task_id="runtime-task",
            symbols={"ev_1": "local://docs/runtime", "ans": "answer://1"},
            metadata={"scope": "test"},
        )
        proof_hash = ProofHashRecord(
            task_id="runtime-task",
            artifact_id="trace-1",
            artifact_type="compressed_trace",
            proof_hash="abc123",
            metadata={"confidence": 0.8},
        )

        await self.storage.register_macro(compose_macro)
        await self.storage.register_macro(unused_macro)
        await self.storage.register_opcode(opcode)
        await self.storage.register_decoder(decoder)
        await self.storage.record_symbol_table_snapshot(snapshot)
        await self.storage.record_proof_hash(proof_hash)
        await self.storage.record_reasoning_log(
            ReasoningLog(
                task_id="runtime-task",
                compressed_chain=("@lookup", "@compose_answer"),
                macros_used=("compose_answer",),
            )
        )

        all_macros = await self.storage.list_macros(active_only=False)
        subset = await self.storage.load_active_compression_runtime(
            "runtime-task",
            opcode_names=("lookup",),
            decoder_names=("emit_answer",),
        )

        self.assertEqual(await self.storage.get_opcode("lookup"), opcode)
        self.assertEqual(await self.storage.get_decoder("emit_answer"), decoder)
        self.assertEqual(await self.storage.get_latest_symbol_table_snapshot("runtime-task"), snapshot)
        self.assertEqual(await self.storage.list_proof_hashes(task_id="runtime-task"), (proof_hash,))
        self.assertEqual(tuple(macro.macro_name for macro in all_macros), ("compose_answer", "unused_macro"))
        self.assertEqual(subset.macros, (compose_macro,))
        self.assertEqual(subset.opcodes, (opcode,))
        self.assertEqual(subset.decoders, (decoder,))
        self.assertEqual(subset.symbol_table, snapshot)
        self.assertEqual(subset.proof_hashes, (proof_hash,))

        export_path = await self.storage.export_compression_lexicon(
            self.test_logs / "compression_lexicon.md"
        )
        export_text = export_path.read_text(encoding="utf-8")
        self.assertIn("compose_answer", export_text)
        self.assertIn("fingerprint-compose", export_text)
        self.assertIn("uncertainty_preserving", export_text)
        self.assertIn("lookup", export_text)
        self.assertIn("emit_answer", export_text)

    async def test_storage_exports_trace_debug_view_and_bootstraps_core_runtime_lexicon(self) -> None:
        result = _build_task_result("trace-debug-task")
        await self.storage.record_reasoning_trace(result.reasoning)

        export_path = await self.storage.export_trace_debug_view(
            result.task_id,
            self.test_logs / "trace_debug.md",
        )
        export_text = export_path.read_text(encoding="utf-8")
        built_in_opcodes = await self.storage.list_opcodes(
            opcode_names=(
                "lookup",
                "bind",
                "compare",
                "infer",
                "aggregate",
                "check",
                "emit",
                "cite",
                "confidence_update",
            ),
        )
        built_in_decoders = await self.storage.list_decoders(
            decoder_names=("verified_answer", "compressed_trace_summary"),
        )

        self.assertIn("Trace Debug Export", export_text)
        self.assertIn("proof123", export_text)
        self.assertIn("verified_answer", export_text)
        self.assertIn("Operation Stream", export_text)
        self.assertEqual(
            {opcode.opcode_name for opcode in built_in_opcodes},
            {
                "aggregate",
                "bind",
                "check",
                "cite",
                "compare",
                "confidence_update",
                "emit",
                "infer",
                "lookup",
            },
        )
        self.assertEqual(
            {decoder.decoder_name for decoder in built_in_decoders},
            {"compressed_trace_summary", "verified_answer"},
        )

    async def test_reasoner_loads_only_active_task_subset_and_persists_symbol_table(self) -> None:
        task_result = _build_task_result("reasoner-runtime-subset")
        for opcode_name in ("lookup", "bind", "emit", "unused_opcode"):
            await self.storage.register_opcode(
                OpcodeEntry(
                    opcode_name=opcode_name,
                    description=f"{opcode_name} opcode",
                )
            )
        await self.storage.register_decoder(
            DecoderEntry(
                decoder_name="verified_answer",
                template="Answer: {value}",
            )
        )
        await self.storage.register_decoder(
            DecoderEntry(
                decoder_name="unused_decoder",
                template="Unused: {value}",
            )
        )
        await self.storage.register_macro(
            Macro(macro_name="historical_macro", expansion=("@old_macro",), version=1)
        )
        await self.storage.record_reasoning_log(
            ReasoningLog(
                task_id=task_result.task_id,
                compressed_chain=("@old_macro",),
                macros_used=("historical_macro",),
            )
        )

        model_manager = ModelManager(config=self.test_config)
        await model_manager.start()
        self.addAsyncCleanup(model_manager.stop)
        reasoner = ReasonerAgent(
            model_manager=model_manager,
            storage=self.storage,
            config=self.test_config,
        )
        await reasoner.start()
        self.addAsyncCleanup(reasoner.stop)

        trace = await reasoner.reason(
            task_result.plan,
            task_result.evidence,
            task_result.plan.budget,
        )
        payload = trace.to_dict()

        subset = reasoner.last_runtime_subset
        self.assertIsNotNone(subset)
        assert subset is not None
        self.assertEqual(trace.canonical_graph_builder, "reasoner_stub_v1")
        self.assertNotIn("canonical_graph", payload)
        self.assertEqual(CompressedTrace.from_dict(payload), trace)
        assert trace.canonical_graph is not None
        self.assertTrue(
            {"question", "evidence_item", "evidence_set", "intermediate_binding", "answer_fragment"}.issubset(
                {entity.entity_type for entity in trace.canonical_graph.entities}
            )
        )
        self.assertEqual(
            {activity.activity_type for activity in trace.canonical_graph.activities},
            {"retrieve", "bind", "emit"},
        )
        self.assertEqual(trace.canonical_graph.agents[0].backend, "stub_generation")
        self.assertEqual(
            {opcode.opcode_name for opcode in subset.opcodes},
            {"lookup", "bind", "emit"},
        )
        self.assertEqual({decoder.decoder_name for decoder in subset.decoders}, {"verified_answer"})
        self.assertEqual(subset.macros, ())
        self.assertNotIn("unused_opcode", {opcode.opcode_name for opcode in subset.opcodes})
        self.assertNotIn("unused_decoder", {decoder.decoder_name for decoder in subset.decoders})
        self.assertIsNotNone(subset.symbol_table)
        assert subset.symbol_table is not None
        self.assertIn("sym_question", subset.symbol_table.symbols)
        self.assertIn("sym_evidence_1", subset.symbol_table.symbols)
        self.assertIn("sym_evidence_set", subset.symbol_table.symbols)
        self.assertIn("sym_answer", subset.symbol_table.symbols)
        self.assertEqual(
            await self.storage.get_latest_symbol_table_snapshot(task_result.task_id),
            subset.symbol_table,
        )
        self.assertIn("loaded_opcodes=", trace.reasoner_notes)
        self.assertIn("loaded_macros=", trace.reasoner_notes)
        self.assertIn("loaded_decoders=", trace.reasoner_notes)
        for name in ("lookup", "bind", "emit", "verified_answer"):
            self.assertIn(name, trace.reasoner_notes)

    async def test_critic_loads_only_trace_scoped_runtime_subset(self) -> None:
        task_result = _build_task_result("critic-runtime-subset")
        await self.storage.register_macro(
            Macro(macro_name="compose_answer", expansion=("@compose_answer",), version=1)
        )
        await self.storage.register_macro(
            Macro(macro_name="unused_macro", expansion=("@unused_macro",), version=1)
        )
        for opcode_name in ("lookup", "emit", "unused_opcode"):
            await self.storage.register_opcode(
                OpcodeEntry(
                    opcode_name=opcode_name,
                    description=f"{opcode_name} opcode",
                )
            )
        await self.storage.register_decoder(
            DecoderEntry(
                decoder_name="verified_answer",
                template="Answer: {value}",
            )
        )
        await self.storage.register_decoder(
            DecoderEntry(
                decoder_name="unused_decoder",
                template="Unused: {value}",
            )
        )
        snapshot = SymbolTableSnapshot(
            task_id=task_result.task_id,
            symbols={
                "sym_question": f"question://{task_result.task_id}",
                "sym_evidence_1": task_result.evidence.local_results[0].id,
                "sym_evidence_set": task_result.evidence.local_results[0].id,
                "sym_answer": f"answer://{task_result.task_id}",
            },
            metadata={"scope": "task", "owner": "test"},
        )
        await self.storage.record_symbol_table_snapshot(snapshot)
        trace = replace(
            task_result.reasoning,
            tokens=("@read_question", "@reason_pass_1", "@compose_answer"),
            expanded_preview=("Read question", "Reasoning pass 1 of 1"),
            macros_used=("compose_answer",),
            operation_stream=(
                OperationStep(
                    op_id="op_lookup",
                    opcode="lookup",
                    args=("sym_question",),
                    output_ref="sym_evidence_set",
                    context_frame_id="ctx_primary",
                    evidence_handles=(task_result.evidence.local_results[0].id,),
                ),
                OperationStep(
                    op_id="op_emit",
                    opcode="emit",
                    args=("sym_answer",),
                    context_frame_id="ctx_primary",
                    evidence_handles=(task_result.evidence.local_results[0].id,),
                ),
            ),
            symbol_table_refs=("sym_question", "sym_evidence_set", "sym_answer"),
            decode_hints=(
                DecodeHint(
                    hint_id="hint_answer",
                    template="verified_answer",
                    entity_ids=("ent_answer",),
                ),
            ),
        )

        model_manager = ModelManager(config=self.test_config)
        await model_manager.start()
        self.addAsyncCleanup(model_manager.stop)
        critic = CriticAgent(
            model_manager=model_manager,
            storage=self.storage,
            config=self.test_config,
        )
        await critic.start()
        self.addAsyncCleanup(critic.stop)

        report = await critic.review(
            task_result.plan,
            task_result.evidence,
            trace,
            ResourceBudget(reasoner_passes=1, critic_passes=2),
        )

        subset = critic.last_runtime_subset
        self.assertIsNotNone(subset)
        assert subset is not None
        self.assertEqual(tuple(macro.macro_name for macro in subset.macros), ("compose_answer",))
        self.assertEqual({opcode.opcode_name for opcode in subset.opcodes}, {"lookup", "emit"})
        self.assertEqual({decoder.decoder_name for decoder in subset.decoders}, {"verified_answer"})
        self.assertNotIn("unused_macro", {macro.macro_name for macro in subset.macros})
        self.assertNotIn("unused_opcode", {opcode.opcode_name for opcode in subset.opcodes})
        self.assertNotIn("unused_decoder", {decoder.decoder_name for decoder in subset.decoders})
        self.assertEqual(subset.symbol_table, snapshot)
        self.assertTrue(report.is_valid)
        self.assertIn("check.runtime_subset_alignment", report.critic_notes)
        self.assertIn("loaded_macros=compose_answer", report.critic_notes)
        self.assertIn("loaded_opcodes=", report.critic_notes)
        self.assertIn("loaded_decoders=", report.critic_notes)
        for name in ("lookup", "emit", "verified_answer"):
            self.assertIn(name, report.critic_notes)


class OrchestratorTypedPersistenceTests(unittest.IsolatedAsyncioTestCase):
    """Validate orchestrator writes typed task/status/event persistence during runs."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase5_orchestrator.sqlite3")
        self.test_logs = Path("test_phase5_orchestrator_logs")
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
        storage_cfg = replace(APP_CONFIG.storage, sqlite_path=self.test_db, logs_dir=self.test_logs)
        dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
        self.test_config = replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)
        self.orchestrator = Orchestrator(config=self.test_config)
        await self.orchestrator.start()

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_orchestrator_persists_task_result_statuses_and_events(self) -> None:
        result = await self.orchestrator.run_task(
            "How does local-first retrieval work?",
            thinking_minutes=30,
        )

        stored_result = await self.orchestrator.storage.get_task_result(result.task_id)
        statuses = await self.orchestrator.storage.list_agent_statuses(task_id=result.task_id)
        completed_events = await self.orchestrator.storage.list_runtime_events(stage="pipeline.completed")
        reasoning_history = await self.orchestrator.storage.list_reasoning_history(task_id=result.task_id)
        traces = await self.orchestrator.storage.list_reasoning_traces(task_id=result.task_id)
        metrics = await self.orchestrator.storage.list_performance_metrics(task_id=result.task_id)
        proof_hashes = await self.orchestrator.storage.list_proof_hashes(task_id=result.task_id)

        self.assertEqual(stored_result, result)
        self.assertTrue(result.answer_text)
        self.assertEqual(result.warnings, ())
        self.assertEqual(len(result.metrics), 1)
        self.assertEqual(metrics, result.metrics)
        self.assertGreaterEqual(len(reasoning_history), 2)
        self.assertEqual(traces[-1].proof_hash, result.reasoning.proof_hash)
        self.assertEqual(proof_hashes[-1].artifact_type, "compressed_trace")
        self.assertTrue(
            {"planner", "researcher", "reasoner", "critic", "compressor", "orchestrator"}.issubset(
                {status.component for status in statuses}
            )
        )
        self.assertTrue(any(event.payload.get("task_id") == result.task_id for event in completed_events))

    async def test_degraded_web_fallback_surfaces_warning_and_status(self) -> None:
        self.orchestrator.researcher.web_adapter = _FailingWebAdapter()

        result = await self.orchestrator.run_task(
            "What is the latest local runtime status?",
            thinking_minutes=121,
        )

        self.assertIn("web_fallback_returned_no_results", result.warnings)

        researcher_statuses = await self.orchestrator.storage.list_agent_statuses(
            task_id=result.task_id,
            component="researcher",
        )
        self.assertTrue(
            any(
                status.severity == SeverityLevel.HIGH
                and "degraded" in status.message
                for status in researcher_statuses
            )
        )

        web_log_path = self.test_logs / self.test_config.storage.web_log_name
        self.assertTrue(web_log_path.exists())
        with web_log_path.open("r", encoding="utf-8") as handle:
            web_entries = [json.loads(line) for line in handle]
        self.assertEqual(web_entries[-1]["stage"], "researcher.web_lookup")
        self.assertTrue(web_entries[-1]["payload"]["degraded"])
        self.assertEqual(
            await self.orchestrator.storage.list_web_evidence(task_id=result.task_id),
            (),
        )


if __name__ == "__main__":
    unittest.main()
