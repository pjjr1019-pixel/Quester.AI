"""Phase 12 acceptance tests for translation output and verified export eligibility."""

from __future__ import annotations

import json
import shutil
import unittest
from dataclasses import replace
from pathlib import Path

from config import APP_CONFIG
from data_structures import (
    CandidateTrace,
    CompressedTrace,
    ContextFrame,
    CritiqueReport,
    CritiqueResult,
    DecodeHint,
    EvidenceBundle,
    EvidenceItem,
    OperationStep,
    Plan,
    PlanStep,
    ResourceBudget,
    SourceType,
    TaskResult,
)
from retrieval import stable_hash
from storage import StorageManager
from translation_service import TranslationService


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
        ),
    )
    storage_cfg = replace(APP_CONFIG.storage, sqlite_path=Path(sqlite_name), logs_dir=Path(logs_name))
    dashboard = replace(APP_CONFIG.dashboard, enable_ui=False)
    return replace(APP_CONFIG, preflight=preflight, storage=storage_cfg, dashboard=dashboard)


def _candidate_hash(candidate_id: str, tokens: tuple[str, ...], steps: tuple[OperationStep, ...], evidence_ids: tuple[str, ...]) -> str:
    payload = {
        "task_id": candidate_id,
        "tokens": list(tokens),
        "operation_stream": [step.to_dict() for step in steps],
        "evidence_handles": list(evidence_ids),
    }
    return stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _build_trace(
    *,
    task_id: str,
    answer_text: str,
    supporting_evidence_ids: tuple[str, ...],
    reasoning_mode: str = "deep",
    verifier_type: str = "tool.evidence_grounding",
    candidate_count: int = 1,
) -> CompressedTrace:
    emit_step = OperationStep(
        op_id="emit_1",
        opcode="emit",
        args=("sym_answer",),
        context_frame_id="ctx_acceptance",
        evidence_handles=supporting_evidence_ids,
        metadata={
            "candidate_id": "cand_selected",
            "answer_text": answer_text,
            "selected_strategy": "acceptance",
            "selected_verifier": verifier_type,
            "candidate_score": 0.93,
            "supporting_evidence_ids": supporting_evidence_ids,
            "verified": True,
        },
    )
    decode_hint = DecodeHint(
        hint_id="hint_acceptance",
        template="verified_answer",
        entity_ids=("answer",),
        metadata={
            "candidate_id": "cand_selected",
            "answer_text": answer_text,
            "supporting_evidence_ids": supporting_evidence_ids,
        },
    )
    candidate_traces = []
    for index in range(candidate_count):
        candidate_id = f"cand_{index + 1}"
        candidate_answer = answer_text if index == 0 else f"{answer_text} variant {index + 1}"
        candidate_steps = (
            replace(
                emit_step,
                op_id=f"emit_{index + 1}",
                metadata={
                    **emit_step.metadata,
                    "candidate_id": candidate_id,
                    "answer_text": candidate_answer,
                },
            ),
        )
        tokens = (f"@candidate_{index + 1}", "@compose_answer")
        candidate_traces.append(
            CandidateTrace(
                candidate_id=candidate_id,
                answer_text=candidate_answer,
                strategy="acceptance",
                verifier_type=verifier_type,
                verified=True,
                total_score=0.93 - (index * 0.1),
                agreement_score=1.0 if index == 0 else 0.4,
                evidence_support_score=1.0 if index == 0 else 0.5,
                proof_hash_stability=1.0,
                supporting_evidence_ids=supporting_evidence_ids,
                tokens=tokens,
                expanded_preview=("Candidate reasoning", "Compose answer"),
                operation_stream=candidate_steps,
                decode_hints=(decode_hint,),
                proof_hash=_candidate_hash(candidate_id, tokens, candidate_steps, supporting_evidence_ids),
            )
        )
    return CompressedTrace(
        task_id=task_id,
        tokens=("@read_question", "@compose_answer"),
        expanded_preview=("Read question", "Compose answer"),
        macros_used=("@compose_answer",),
        confidence=0.94,
        ir_version="1",
        operation_stream=(emit_step,),
        evidence_handles=supporting_evidence_ids,
        context_frames=(
            ContextFrame(
                frame_id="ctx_acceptance",
                scope="task",
                confidence=0.94,
                provenance_bundle_id="bundle_acceptance",
                metadata={
                    "rm": reasoning_mode,
                    "cid": "cand_selected",
                    "ta": answer_text,
                    "si": supporting_evidence_ids,
                    "sv": verifier_type,
                    "ss": 0.93,
                },
            ),
        ),
        candidate_traces=tuple(candidate_traces),
        proof_hash="proof-acceptance",
        decode_hints=(decode_hint,),
    )


def _build_result(
    task_id: str,
    *,
    reasoning_mode: str,
    is_valid: bool,
    critique_result: CritiqueResult,
    proof_hash_match: bool,
    candidate_count: int,
) -> TaskResult:
    budget = ResourceBudget(
        retrieval_top_k=6,
        max_web_queries=2,
        reasoner_passes=2,
        critic_passes=2,
        macro_depth=3,
    )
    plan = Plan(
        task_id=task_id,
        question="What should the export keep?",
        steps=(PlanStep(step_id="step_1", description="Answer the question"),),
        required_evidence=("local docs",),
        success_criteria=("produce a verified answer",),
        budget=budget,
    )
    evidence = EvidenceBundle(
        task_id=task_id,
        local_results=(
            EvidenceItem(
                id="ev-1",
                content="SQLite persists structured task results.",
                source_type=SourceType.LOCAL,
                source_ref="local://sqlite",
                score=0.9,
            ),
        ),
        web_results=(),
        used_web_fallback=False,
    )
    reasoning = _build_trace(
        task_id=task_id,
        answer_text="SQLite persists structured task results.",
        supporting_evidence_ids=("ev-1",),
        reasoning_mode=reasoning_mode,
        candidate_count=candidate_count,
    )
    critique = CritiqueReport(
        task_id=task_id,
        is_valid=is_valid,
        issues=() if is_valid else ("acceptance failure",),
        fixed_trace=reasoning if is_valid else None,
        evidence_coverage=1.0 if is_valid else 0.4,
        result=critique_result,
        verifier_type="tool.evidence_grounding",
        proof_hash_match=proof_hash_match,
        candidate_score=0.93 if is_valid else 0.2,
        repair_actions=("preserve_trace",) if is_valid else ("abstain_due_to_low_grounding",),
        degraded_reason="" if critique_result == CritiqueResult.VALID else "low_evidence_support",
        provenance_coverage=1.0 if is_valid else 0.4,
    )
    return TaskResult(
        task_id=task_id,
        plan=plan,
        evidence=evidence,
        reasoning=reasoning,
        critique=critique,
        compression=(),
        answer_text="SQLite persists structured task results.",
    )


class TranslationAcceptanceTests(unittest.TestCase):
    """Verify final translation preserves claim, evidence basis, and uncertainty markers."""

    def test_translation_verified_answer_preserves_claim_and_supporting_citations(self) -> None:
        translator = TranslationService()
        evidence = EvidenceBundle(
            task_id="translation-verified",
            local_results=(
                EvidenceItem(
                    id="ev-1",
                    content="SQLite stores task runs.",
                    source_type=SourceType.LOCAL,
                    source_ref="local://sqlite",
                    score=0.9,
                ),
                EvidenceItem(
                    id="ev-2",
                    content="JSONL mirrors append-only events.",
                    source_type=SourceType.LOCAL,
                    source_ref="local://jsonl",
                    score=0.85,
                ),
                EvidenceItem(
                    id="ev-3",
                    content="Unused evidence should not be cited.",
                    source_type=SourceType.LOCAL,
                    source_ref="local://unused",
                    score=0.2,
                ),
            ),
            web_results=(),
            used_web_fallback=False,
        )
        reasoning = _build_trace(
            task_id="translation-verified",
            answer_text="SQLite and JSONL persist different runtime artifacts",
            supporting_evidence_ids=("ev-1", "ev-2"),
        )
        critique = CritiqueReport(
            task_id="translation-verified",
            is_valid=True,
            issues=(),
            fixed_trace=reasoning,
            evidence_coverage=1.0,
            result=CritiqueResult.VALID,
            verifier_type="tool.evidence_grounding",
            proof_hash_match=True,
            candidate_score=0.93,
            repair_actions=("preserve_trace",),
            provenance_coverage=1.0,
        )

        rendered = translator.render_answer(evidence=evidence, reasoning=reasoning, critique=critique)
        metadata = translator.summarize_answer_metadata(evidence=evidence, reasoning=reasoning)

        self.assertIn("Verified answer: SQLite and JSONL persist different runtime artifacts.", rendered)
        self.assertIn("Verification: tool.evidence_grounding.", rendered)
        self.assertIn("local://sqlite", rendered)
        self.assertIn("local://jsonl", rendered)
        self.assertNotIn("local://unused", rendered)
        self.assertEqual(metadata["citation_refs"], ["local://sqlite", "local://jsonl"])

    def test_translation_degraded_answer_preserves_uncertainty_reason(self) -> None:
        translator = TranslationService()
        evidence = EvidenceBundle(task_id="translation-degraded", local_results=(), web_results=(), used_web_fallback=False)
        reasoning = _build_trace(
            task_id="translation-degraded",
            answer_text="The outcome is uncertain",
            supporting_evidence_ids=(),
        )
        critique = CritiqueReport(
            task_id="translation-degraded",
            is_valid=False,
            issues=("uncertainty must be preserved",),
            fixed_trace=None,
            evidence_coverage=0.0,
            result=CritiqueResult.DEGRADED,
            verifier_type="tool.evidence_grounding",
            proof_hash_match=True,
            candidate_score=0.1,
            repair_actions=("abstain_due_to_low_grounding",),
            degraded_reason="uncertain_due_to_conflicting_evidence",
            provenance_coverage=0.0,
        )

        rendered = translator.render_answer(evidence=evidence, reasoning=reasoning, critique=critique)

        self.assertIn("Degraded answer: The outcome is uncertain.", rendered)
        self.assertIn("Reason: uncertain_due_to_conflicting_evidence.", rendered)


class VerifiedDeepExportAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    """Verify only eligible verified deep traces can be exported."""

    async def asyncSetUp(self) -> None:
        self.test_db = Path("test_phase12_translation_exports.sqlite3")
        self.test_logs = Path("test_phase12_translation_exports_logs")
        self.config = _build_test_config(str(self.test_db), str(self.test_logs))
        self.storage = StorageManager(config=self.config)
        await self.storage.start()

    async def asyncTearDown(self) -> None:
        await self.storage.stop()
        if self.test_db.exists():
            self.test_db.unlink()
        if self.test_logs.exists():
            shutil.rmtree(self.test_logs)

    async def test_export_verified_deep_traces_filters_out_ineligible_results(self) -> None:
        export_path = self.test_logs / "verified_deep_acceptance.jsonl"
        valid_deep = _build_result(
            "deep-valid",
            reasoning_mode="deep",
            is_valid=True,
            critique_result=CritiqueResult.VALID,
            proof_hash_match=True,
            candidate_count=2,
        )
        fast_valid = _build_result(
            "fast-valid",
            reasoning_mode="fast",
            is_valid=True,
            critique_result=CritiqueResult.VALID,
            proof_hash_match=True,
            candidate_count=2,
        )
        deep_invalid = _build_result(
            "deep-invalid",
            reasoning_mode="deep",
            is_valid=False,
            critique_result=CritiqueResult.INVALID,
            proof_hash_match=True,
            candidate_count=2,
        )
        deep_mismatch = _build_result(
            "deep-mismatch",
            reasoning_mode="deep",
            is_valid=True,
            critique_result=CritiqueResult.VALID,
            proof_hash_match=False,
            candidate_count=2,
        )
        deep_without_candidates = _build_result(
            "deep-no-candidates",
            reasoning_mode="deep",
            is_valid=True,
            critique_result=CritiqueResult.VALID,
            proof_hash_match=True,
            candidate_count=0,
        )

        for result in (valid_deep, fast_valid, deep_invalid, deep_mismatch, deep_without_candidates):
            await self.storage.record_task_result(result)

        exports = await self.storage.export_verified_deep_traces(export_path=export_path)

        self.assertEqual(len(exports), 1)
        self.assertEqual(exports[0].task_id, "deep-valid")
        self.assertEqual(exports[0].trace_proof_hash, "proof-acceptance")
        self.assertTrue(export_path.exists())


if __name__ == "__main__":
    unittest.main()
