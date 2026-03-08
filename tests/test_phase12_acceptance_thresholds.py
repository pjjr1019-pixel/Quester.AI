"""Phase 12 acceptance-threshold regression tests."""

from __future__ import annotations

import unittest

from acceptance_thresholds import PHASE12_ACCEPTANCE_THRESHOLDS
from compression_service import CompressionService
from config import APP_CONFIG
from data_structures import CandidateTrace, CompressedTrace, ContextFrame, DecodeHint, OperationStep, ReasoningLog
from orchestrator import Orchestrator


class _QueuedGenerationManager:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        _ = (prompt, max_tokens)
        if not self._responses:
            raise AssertionError("No fake responses remaining.")
        return self._responses.pop(0)


def _candidate(candidate_id: str, answer_text: str, opcodes: tuple[str, ...]) -> CandidateTrace:
    steps = tuple(
        OperationStep(
            op_id=f"{candidate_id}_{index}",
            opcode=opcode,
            args=("sym_question",),
            context_frame_id="cf_candidate",
            evidence_handles=("ev-1",),
        )
        for index, opcode in enumerate(opcodes, start=1)
    )
    return CandidateTrace(
        candidate_id=candidate_id,
        answer_text=answer_text,
        strategy="threshold_test",
        verifier_type="tool.evidence_grounding",
        verified=True,
        total_score=0.9,
        agreement_score=0.75,
        evidence_support_score=0.9,
        proof_hash_stability=1.0,
        supporting_evidence_ids=("ev-1",),
        tokens=tuple(f"@{opcode}" for opcode in opcodes),
        expanded_preview=opcodes,
        operation_stream=steps,
        decode_hints=(
            DecodeHint(
                hint_id=f"hint_{candidate_id}",
                template="verified_answer",
                metadata={"candidate_id": candidate_id, "answer_text": answer_text},
            ),
        ),
        proof_hash=f"proof-{candidate_id}",
    )


def _trace() -> CompressedTrace:
    candidates = (
        _candidate("cand_1", "first", ("lookup", "bind", "emit")),
        _candidate("cand_2", "second", ("lookup", "bind", "emit")),
    )
    return CompressedTrace(
        task_id="phase12-threshold-trace",
        tokens=("@lookup", "@bind", "@emit", "@lookup", "@bind", "@emit"),
        expanded_preview=("lookup", "bind", "emit", "lookup", "bind", "emit"),
        macros_used=(),
        confidence=0.92,
        operation_stream=(
            OperationStep(
                op_id="op_1",
                opcode="lookup",
                args=("sym_question",),
                output_ref="sym_evidence",
                context_frame_id="cf_main",
                evidence_handles=("ev-1",),
            ),
            OperationStep(
                op_id="op_2",
                opcode="bind",
                args=("sym_evidence",),
                output_ref="sym_binding",
                context_frame_id="cf_main",
                evidence_handles=("ev-1",),
            ),
            OperationStep(
                op_id="op_3",
                opcode="emit",
                args=("sym_binding",),
                context_frame_id="cf_main",
                evidence_handles=("ev-1",),
            ),
            OperationStep(
                op_id="op_4",
                opcode="lookup",
                args=("sym_question",),
                output_ref="sym_evidence_2",
                context_frame_id="cf_main",
                evidence_handles=("ev-2",),
            ),
            OperationStep(
                op_id="op_5",
                opcode="bind",
                args=("sym_evidence_2",),
                output_ref="sym_binding_2",
                context_frame_id="cf_main",
                evidence_handles=("ev-2",),
            ),
            OperationStep(
                op_id="op_6",
                opcode="emit",
                args=("sym_binding_2",),
                context_frame_id="cf_main",
                evidence_handles=("ev-2",),
            ),
        ),
        symbol_table_refs=("sym_question", "sym_evidence", "sym_answer"),
        evidence_handles=("ev-1", "ev-2"),
        context_frames=(
            ContextFrame(
                frame_id="cf_main",
                scope="task",
                confidence=0.92,
                metadata={"vv": True},
            ),
        ),
        candidate_traces=candidates,
        proof_hash="trace-proof-thresholds",
    )


class Phase12AcceptanceThresholdTests(unittest.IsolatedAsyncioTestCase):
    def test_thresholds_define_concrete_validity_compression_and_resource_limits(self) -> None:
        thresholds = PHASE12_ACCEPTANCE_THRESHOLDS

        self.assertEqual(thresholds.validity.minimum_deep_mode_improvement_examples, 1)
        self.assertTrue(thresholds.validity.require_verifier_backed_final_selection)
        self.assertTrue(thresholds.validity.allow_structured_degraded_results)
        self.assertTrue(thresholds.validity.require_actionable_real_mode_failures)

        self.assertEqual(thresholds.compression.max_foreground_proposals, 5)
        self.assertEqual(thresholds.compression.max_recent_reasoning_logs, 6)
        self.assertGreater(thresholds.compression.minimum_realized_savings_ratio, 0.0)
        self.assertTrue(thresholds.compression.require_validated_proposals)
        self.assertTrue(thresholds.compression.require_proof_hash_stability)
        self.assertTrue(thresholds.compression.require_critic_validity_non_regression)

        self.assertEqual(thresholds.resources.generation_slots, APP_CONFIG.concurrency.generation_slots)
        self.assertEqual(thresholds.resources.embedding_slots, APP_CONFIG.concurrency.embedding_slots)
        self.assertEqual(thresholds.resources.dev_vram_gb, 4)
        self.assertEqual(thresholds.resources.baseline_vram_gb, 6)
        self.assertEqual(thresholds.resources.baseline_ram_gb, 8)
        self.assertTrue(thresholds.resources.require_bounded_queues)

    async def test_compression_service_caps_foreground_proposals_to_threshold(self) -> None:
        trace = _trace()
        logs = [
            ReasoningLog(
                task_id=f"log-{index}",
                compressed_chain=trace.tokens,
                macros_used=(),
            )
            for index in range(8)
        ]
        service = CompressionService(
            model_manager=_QueuedGenerationManager(["not json", "still not json"]),
            config=APP_CONFIG,
        )

        proposals = await service.propose(trace, logs=logs)

        self.assertLessEqual(
            len(proposals),
            PHASE12_ACCEPTANCE_THRESHOLDS.compression.max_foreground_proposals,
        )
        self.assertTrue(proposals)
        self.assertTrue(all(proposal.validation_passed for proposal in proposals))
        self.assertEqual(
            CompressionService.MAX_FOREGROUND_PROPOSALS,
            PHASE12_ACCEPTANCE_THRESHOLDS.compression.max_foreground_proposals,
        )

    def test_orchestrator_uses_threshold_for_recent_compression_history_scan(self) -> None:
        self.assertEqual(
            Orchestrator.COMPRESSION_HISTORY_SCAN_LIMIT,
            PHASE12_ACCEPTANCE_THRESHOLDS.compression.max_recent_reasoning_logs,
        )


if __name__ == "__main__":
    unittest.main()
