"""Phase 12 acceptance tests for end-to-end runs, failures, and invariants."""

from __future__ import annotations

import json
import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from compressor import CompressorAgent
from config import APP_CONFIG
from critic import CriticAgent
from critique_service import CritiqueService
from data_structures import (
    CandidateTrace,
    CompressedTrace,
    CompressionRuntimeSubset,
    ContextFrame,
    CritiqueReport,
    CritiqueResult,
    DecodeHint,
    DecoderEntry,
    EvidenceBundle,
    EvidenceItem,
    Macro,
    MacroProposal,
    OpcodeEntry,
    OperationStep,
    Plan,
    PlanStep,
    ReasonerCriticHandoff,
    ResearchReasonerHandoff,
    ResourceBudget,
    SourceType,
    SymbolTableSnapshot,
)
from macro_engine import MacroEngine
from orchestrator import Orchestrator
from planner import PlannerAgent
from planner_service import PlannerService
from reasoner import ReasonerAgent
from reasoner import ReasonerAgent
from reasoning_service import ReasoningService
from researcher import ResearcherAgent
from retrieval import stable_hash
from runtime_errors import WebLookupTimeoutError
from self_optimizer import SelfOptimizer


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


class _QueuedGenerationManager:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        _ = (prompt, max_tokens)
        if not self._responses:
            raise AssertionError("No fake responses remaining.")
        return self._responses.pop(0)


class _FailingWebAdapter:
    provider_name = "failing_web"

    async def search(self, query: str, *, max_results: int):
        _ = (query, max_results)
        raise WebLookupTimeoutError("acceptance timeout")


class _StaticCritiqueStorage:
    def __init__(self, subset: CompressionRuntimeSubset) -> None:
        self.subset = subset

    async def load_active_compression_runtime(
        self,
        task_id: str,
        *,
        macro_names: tuple[str, ...],
        opcode_names: tuple[str, ...],
        decoder_names: tuple[str, ...],
    ) -> CompressionRuntimeSubset:
        _ = (task_id, macro_names, opcode_names, decoder_names)
        return self.subset


def _candidate_hash(candidate_id: str, tokens: tuple[str, ...], steps: tuple[OperationStep, ...], evidence_ids: tuple[str, ...]) -> str:
    payload = {
        "task_id": candidate_id,
        "tokens": list(tokens),
        "operation_stream": [step.to_dict() for step in steps],
        "evidence_handles": list(evidence_ids),
    }
    return stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _build_handoff(
    *,
    question: str = "What evidence supports the answer?",
    evidence: EvidenceBundle | None = None,
    selected_answer: str = "Selected answer",
    selected_candidate_id: str = "cand_selected",
    selected_verifier: str = "tool.evidence_grounding",
    candidate_traces: tuple[CandidateTrace, ...] = (),
    macros_used: tuple[str, ...] = (),
    proof_hash: str = "proof-acceptance",
) -> ReasonerCriticHandoff:
    budget = ResourceBudget(
        retrieval_top_k=6,
        max_web_queries=2,
        reasoner_passes=2,
        critic_passes=2,
        macro_depth=3,
    )
    plan = Plan(
        task_id="phase12-handoff",
        question=question,
        steps=(PlanStep(step_id="step_1", description="Inspect evidence"),),
        required_evidence=("local docs",),
        success_criteria=("return typed trace",),
        budget=budget,
    )
    evidence_bundle = evidence or EvidenceBundle(
        task_id=plan.task_id,
        local_results=(
            EvidenceItem(
                id="ev-1",
                content="Supported answer text appears here.",
                source_type=SourceType.LOCAL,
                source_ref="local://ev-1",
                score=0.9,
            ),
        ),
        web_results=(),
        used_web_fallback=False,
    )
    emit_step = OperationStep(
        op_id="op_emit",
        opcode="emit",
        args=("sym_answer",),
        context_frame_id="ctx_phase12",
        evidence_handles=("ev-1",) if evidence_bundle.local_results or evidence_bundle.web_results else (),
        metadata={
            "candidate_id": selected_candidate_id,
            "answer_text": selected_answer,
            "selected_strategy": "acceptance",
            "selected_verifier": selected_verifier,
            "candidate_score": 0.3,
            "supporting_evidence_ids": ("ev-1",) if evidence_bundle.local_results or evidence_bundle.web_results else (),
        },
    )
    decode_hint = DecodeHint(
        hint_id="hint_phase12",
        template="verified_answer",
        entity_ids=("answer",),
        metadata={
            "candidate_id": selected_candidate_id,
            "answer_text": selected_answer,
        },
    )
    trace = CompressedTrace(
        task_id=plan.task_id,
        tokens=("@read_question", "@compose_answer"),
        expanded_preview=("Read question", "Compose answer"),
        macros_used=macros_used,
        confidence=0.9,
        ir_version="1",
        operation_stream=(emit_step,),
        evidence_handles=("ev-1",) if evidence_bundle.local_results or evidence_bundle.web_results else (),
        context_frames=(
            ContextFrame(
                frame_id="ctx_phase12",
                scope="task",
                confidence=0.9,
                provenance_bundle_id="bundle_phase12",
                metadata={
                    "rm": "deep",
                    "cid": selected_candidate_id,
                    "ta": selected_answer,
                    "sv": selected_verifier,
                    "si": ("ev-1",) if evidence_bundle.local_results or evidence_bundle.web_results else (),
                    "ss": 0.3,
                },
            ),
        ),
        candidate_traces=candidate_traces,
        proof_hash=proof_hash,
        decode_hints=(decode_hint,),
    )
    return ReasonerCriticHandoff.from_inputs(
        plan=plan,
        evidence=evidence_bundle,
        trace=trace,
        budget=budget,
    )


def _build_runtime_subset(
    handoff: ReasonerCriticHandoff,
    *,
    include_macros: bool = True,
) -> CompressionRuntimeSubset:
    return CompressionRuntimeSubset(
        task_id=handoff.plan.task_id,
        macros=(
            tuple(
                Macro(
                    macro_name=name,
                    expansion=("@compose_answer",),
                    version=1,
                    opcode_pattern=("emit",),
                    invariants=(
                        "deterministic_round_trip",
                        "provenance_preserving",
                        "uncertainty_preserving",
                    ),
                )
                for name in handoff.required_macro_names
            )
            if include_macros
            else ()
        ),
        opcodes=tuple(
            OpcodeEntry(opcode_name=name, description=f"{name} opcode")
            for name in handoff.required_opcode_names
        ),
        decoders=tuple(
            DecoderEntry(decoder_name=name, template=f"{name}: {{value}}")
            for name in handoff.required_decoder_names
        ),
        symbol_table=SymbolTableSnapshot(
            task_id=handoff.plan.task_id,
            symbols={"sym_answer": "sym_answer"},
            metadata={"scope": "task"},
        ),
        proof_hashes=(),
    )


class PublicApiAcceptanceTests(unittest.TestCase):
    """Validate the published top-level API surface remains present."""

    def test_public_api_methods_exist(self) -> None:
        self.assertTrue(callable(getattr(PlannerAgent, "plan", None)))
        self.assertTrue(callable(getattr(ResearcherAgent, "research", None)))
        self.assertTrue(callable(getattr(ReasonerAgent, "reason", None)))
        self.assertTrue(callable(getattr(CriticAgent, "review", None)))
        self.assertTrue(callable(getattr(CompressorAgent, "propose", None)))
        self.assertTrue(callable(getattr(SelfOptimizer, "run_cycle", None)))
        self.assertTrue(callable(getattr(Orchestrator, "run_task", None)))
        self.assertTrue(callable(getattr(Orchestrator, "run_pipeline", None)))
        self.assertTrue(callable(getattr(MacroEngine, "compress", None)))
        self.assertTrue(callable(getattr(MacroEngine, "expand", None)))
        self.assertTrue(callable(getattr(MacroEngine, "verify_round_trip", None)))


class Phase12PipelineAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the current stub pipeline end to end."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase12_end_to_end.sqlite3")
        self.test_logs = Path("test_phase12_end_to_end_logs")
        self.config = _build_test_config(str(self.test_db), str(self.test_logs))
        self.orchestrator = Orchestrator(config=self.config)
        await self.orchestrator.start()
        await self.orchestrator._run_dashboard_action(action="examples.load_demo_pack", payload={})

    async def asyncTearDown(self) -> None:
        await self.orchestrator.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_public_entrypoints_run_task_and_run_pipeline_return_task_results(self) -> None:
        via_run_task = await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=5)
        via_run_pipeline = await self.orchestrator.run_pipeline("What is 2 + 2?")

        self.assertIn("4", via_run_task.answer_text)
        self.assertIn("4", via_run_pipeline.answer_text)
        self.assertTrue(via_run_task.task_id)
        self.assertTrue(via_run_pipeline.task_id)

    async def test_end_to_end_retrieval_backed_question_returns_local_answer(self) -> None:
        sample = self.orchestrator.phase11_content.get_sample_task("storage_layers_comparison")
        assert sample is not None

        result = await self.orchestrator.run_task(sample.question, sample.comparison_deep_minutes)

        self.assertTrue(result.evidence.local_results)
        self.assertFalse(result.evidence.used_web_fallback)
        self.assertIn("sqlite", result.answer_text.lower())
        self.assertIn("jsonl", result.answer_text.lower())

    async def test_end_to_end_tool_verified_question_routes_through_tool_checks(self) -> None:
        sample = self.orchestrator.phase11_content.get_sample_task("python_code_result")
        assert sample is not None

        result = await self.orchestrator.run_task(sample.question, sample.recommended_thinking_minutes)

        self.assertEqual(result.critique.verifier_type, "tool.python_code_execution")
        self.assertIn("8", result.answer_text)

    async def test_fast_vs_deep_pipeline_budget_can_improve_selected_answer(self) -> None:
        sample = self.orchestrator.phase11_content.get_sample_task("storage_layers_comparison")
        assert sample is not None

        fast_result = await self.orchestrator.run_task(sample.question, sample.comparison_fast_minutes)
        deep_result = await self.orchestrator.run_task(sample.question, sample.comparison_deep_minutes)

        self.assertNotEqual(fast_result.answer_text, deep_result.answer_text)
        self.assertIn("sqlite", deep_result.answer_text.lower())
        self.assertIn("jsonl", deep_result.answer_text.lower())
        self.assertGreater(len(deep_result.reasoning.candidate_traces), len(fast_result.reasoning.candidate_traces))

    async def test_web_timeout_degrades_to_local_only_warning(self) -> None:
        original_adapter = self.orchestrator.researcher.web_adapter
        self.orchestrator.researcher.web_adapter = _FailingWebAdapter()
        try:
            sample = self.orchestrator.phase11_content.get_sample_task("web_runtime_status")
            assert sample is not None
            result = await self.orchestrator.run_task(sample.question, sample.recommended_thinking_minutes)
        finally:
            self.orchestrator.researcher.web_adapter = original_adapter

        self.assertTrue(result.evidence.used_web_fallback)
        self.assertFalse(result.evidence.web_results)
        self.assertIn("web_fallback_returned_no_results", result.warnings)

    async def test_repeated_critic_rejection_returns_structured_invalid_result_after_repair_bound(self) -> None:
        original_review = self.orchestrator.critic.review_from_handoff

        async def always_invalid(handoff: ReasonerCriticHandoff) -> CritiqueReport:
            return CritiqueReport(
                task_id=handoff.plan.task_id,
                is_valid=False,
                issues=("forced critic rejection",),
                fixed_trace=None,
                evidence_coverage=0.0,
                result=CritiqueResult.INVALID,
                verifier_type="acceptance.invalid",
                proof_hash_match=True,
                candidate_score=0.0,
                repair_actions=("reload_runtime_subset", "rebuild_trace_projection"),
                degraded_reason="",
                failure_categories=("schema",),
                provenance_coverage=0.0,
            )

        self.orchestrator.critic.review_from_handoff = always_invalid
        try:
            result = await self.orchestrator.run_task("What is 2 + 2?", thinking_minutes=30)
        finally:
            self.orchestrator.critic.review_from_handoff = original_review

        self.assertEqual(result.critique.result, CritiqueResult.INVALID)
        self.assertIn("critique_reported_issues", result.warnings)
        self.assertTrue(any(warning.startswith("repair_applied:") for warning in result.warnings))


class PlannerFallbackAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    """Validate failure-path fallback behavior for invalid model planner output."""

    async def test_invalid_planner_model_output_falls_back_to_deterministic_plan(self) -> None:
        budget = ResourceBudget(
            retrieval_top_k=4,
            max_web_queries=1,
            reasoner_passes=1,
            critic_passes=1,
            macro_depth=2,
        )
        service = PlannerService(model_manager=_QueuedGenerationManager(["not json", "still not json"]), config=APP_CONFIG)

        plan = await service.plan("Acceptance planner fallback?", budget)

        self.assertEqual(plan.question, "Acceptance planner fallback?")
        self.assertIn("planner_output_mode=deterministic_fallback", plan.planner_notes)


class CritiqueInvariantAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    """Validate agreement, provenance, and empty-evidence failure categories."""

    async def test_candidate_agreement_and_verifier_score_flag_weaker_selected_candidate(self) -> None:
        strong_step = OperationStep(
            op_id="emit_strong",
            opcode="emit",
            args=("sym_answer",),
            context_frame_id="ctx_phase12",
            evidence_handles=("ev-1",),
            metadata={"candidate_id": "cand_strong", "answer_text": "Supported answer text"},
        )
        weak_step = OperationStep(
            op_id="emit_weak",
            opcode="emit",
            args=("sym_answer",),
            context_frame_id="ctx_phase12",
            evidence_handles=("ev-1",),
            metadata={"candidate_id": "cand_weak", "answer_text": "Unsupported answer text"},
        )
        strong_candidate = CandidateTrace(
            candidate_id="cand_strong",
            answer_text="Supported answer text",
            strategy="tool_select",
            verifier_type="tool.evidence_grounding",
            verified=True,
            total_score=0.95,
            agreement_score=1.0,
            evidence_support_score=1.0,
            proof_hash_stability=1.0,
            supporting_evidence_ids=("ev-1",),
            tokens=("@candidate_strong",),
            expanded_preview=("Strong candidate",),
            operation_stream=(strong_step,),
            decode_hints=(DecodeHint(hint_id="h1", template="verified_answer", entity_ids=("a",)),),
            proof_hash=_candidate_hash("cand_strong", ("@candidate_strong",), (strong_step,), ("ev-1",)),
        )
        weak_candidate = CandidateTrace(
            candidate_id="cand_weak",
            answer_text="Unsupported answer text",
            strategy="freeform",
            verifier_type="model_only",
            verified=False,
            total_score=0.2,
            agreement_score=0.1,
            evidence_support_score=0.0,
            proof_hash_stability=1.0,
            supporting_evidence_ids=("ev-1",),
            tokens=("@candidate_weak",),
            expanded_preview=("Weak candidate",),
            operation_stream=(weak_step,),
            decode_hints=(DecodeHint(hint_id="h2", template="verified_answer", entity_ids=("a",)),),
            proof_hash=_candidate_hash("cand_weak", ("@candidate_weak",), (weak_step,), ("ev-1",)),
        )
        handoff = _build_handoff(
            selected_answer="Unsupported answer text",
            selected_candidate_id="cand_weak",
            selected_verifier="model_only",
            candidate_traces=(weak_candidate, strong_candidate),
        )
        service = CritiqueService(
            model_manager=_QueuedGenerationManager([]),
            storage=_StaticCritiqueStorage(_build_runtime_subset(handoff)),
            config=APP_CONFIG,
        )

        report = await service.review(handoff)

        self.assertFalse(report.is_valid)
        self.assertIn("candidate_selection", report.failure_categories)
        self.assertIn("rerun_reasoner", report.repair_actions)

    async def test_empty_evidence_and_provenance_loss_are_categorized(self) -> None:
        empty_evidence = EvidenceBundle(task_id="phase12-handoff", local_results=(), web_results=(), used_web_fallback=False)
        handoff = _build_handoff(evidence=empty_evidence)
        service = CritiqueService(
            model_manager=_QueuedGenerationManager([]),
            storage=_StaticCritiqueStorage(_build_runtime_subset(handoff)),
            config=APP_CONFIG,
        )

        report = await service.review(handoff)

        self.assertFalse(report.is_valid)
        self.assertIn("evidence_coverage", report.failure_categories)
        self.assertIn("No evidence found in local or web sources.", report.issues)

    async def test_macro_signature_mismatch_and_provenance_loss_are_reported(self) -> None:
        evidence = EvidenceBundle(
            task_id="phase12-handoff",
            local_results=(
                EvidenceItem(
                    id="ev-1",
                    content="Supported answer text appears here.",
                    source_type=SourceType.LOCAL,
                    source_ref="local://ev-1",
                    score=0.9,
                ),
            ),
            web_results=(),
            used_web_fallback=False,
        )
        emit_step = OperationStep(
            op_id="emit_phase12",
            opcode="emit",
            args=("sym_answer",),
            context_frame_id="ctx_phase12",
            evidence_handles=("ev-missing",),
            metadata={"candidate_id": "cand_selected", "answer_text": "Selected answer"},
        )
        trace = CompressedTrace(
            task_id="phase12-handoff",
            tokens=("@missing_macro", "@compose_answer"),
            expanded_preview=("Use missing macro", "Compose answer"),
            macros_used=("@missing_macro",),
            confidence=0.8,
            ir_version="1",
            operation_stream=(emit_step,),
            evidence_handles=("ev-1",),
            context_frames=(
                ContextFrame(
                    frame_id="ctx_phase12",
                    scope="task",
                    confidence=0.8,
                    provenance_bundle_id="bundle_phase12",
                ),
            ),
            proof_hash="proof-acceptance",
            decode_hints=(DecodeHint(hint_id="hint", template="verified_answer", entity_ids=("a",)),),
        )
        budget = ResourceBudget(
            retrieval_top_k=6,
            max_web_queries=2,
            reasoner_passes=2,
            critic_passes=2,
            macro_depth=3,
        )
        plan = Plan(
            task_id="phase12-handoff",
            question="What went wrong?",
            steps=(PlanStep(step_id="step_1", description="Inspect evidence"),),
            required_evidence=("local docs",),
            success_criteria=("report failure categories",),
            budget=budget,
        )
        handoff = ReasonerCriticHandoff.from_inputs(plan=plan, evidence=evidence, trace=trace, budget=budget)
        service = CritiqueService(
            model_manager=_QueuedGenerationManager([]),
            storage=_StaticCritiqueStorage(_build_runtime_subset(handoff, include_macros=False)),
            config=APP_CONFIG,
        )

        report = await service.review(handoff)

        self.assertFalse(report.is_valid)
        self.assertIn("provenance", report.failure_categories)
        self.assertIn("macro_signature", report.failure_categories)
        self.assertTrue(any("typed handoff" in issue.lower() for issue in report.issues))


class MacroEngineAcceptanceTests(unittest.TestCase):
    """Validate proof-hash and macro-invariant acceptance behavior."""

    def test_proof_hash_stability_for_semantically_equivalent_sequences(self) -> None:
        engine = MacroEngine()
        engine.register_macro("@macro_step", ("expand 1", "expand 2"))

        trace_with_macro = engine.compress(["@macro_step", "@compose_answer"], task_id="phase12-proof")
        trace_literal = engine.compress(["expand 1", "expand 2", "@compose_answer"], task_id="phase12-proof")

        self.assertEqual(trace_with_macro.proof_hash, trace_literal.proof_hash)

    def test_malformed_macro_definition_fails_fast(self) -> None:
        with self.assertRaises(ValueError):
            Macro(macro_name="", expansion=("@compose_answer",), version=1)

    def test_uncertainty_preserving_invariant_is_required_for_macro_proposals(self) -> None:
        engine = MacroEngine()
        proposal = MacroProposal(
            proposal_id="phase12-proposal",
            macro=Macro(
                macro_name="proposal_without_uncertainty",
                expansion=("@compose_answer",),
                version=1,
                opcode_pattern=("emit",),
                invariants=("deterministic_round_trip", "provenance_preserving"),
            ),
            reason="Acceptance invalid proposal",
            examples=("@compose_answer",),
            simulation_score=0.5,
            approved=False,
        )

        validated = engine.validate_macro_proposal(proposal)

        self.assertFalse(validated.validation_passed)
        self.assertIn("missing proposal invariants", " ".join(validated.validation_issues))

    def test_noncanonical_encoding_is_rejected(self) -> None:
        engine = MacroEngine()
        trace = CompressedTrace(
            task_id="phase12-noncanonical",
            tokens=("bind sym_z sym_a",),
            expanded_preview=("bind sym_z sym_a",),
            macros_used=(),
            confidence=1.0,
            ir_version="1",
            operation_stream=(
                OperationStep(
                    op_id="bind_noncanonical",
                    opcode="bind",
                    args=("sym_z", "sym_a"),
                    output_ref="sym_answer",
                    context_frame_id="ctx_phase12",
                    metadata={"source_token": "bind sym_z sym_a"},
                ),
            ),
            context_frames=(
                ContextFrame(
                    frame_id="ctx_phase12",
                    scope="macro_engine",
                    confidence=1.0,
                    provenance_bundle_id="bundle_phase12",
                ),
            ),
            proof_hash="phase12-noncanonical-proof",
        )

        self.assertFalse(engine.verify_round_trip(trace))


if __name__ == "__main__":
    unittest.main()
