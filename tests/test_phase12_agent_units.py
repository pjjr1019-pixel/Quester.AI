"""Phase 12 agent and service unit tests with injected collaborators."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from compressor import CompressorAgent
from compression_service import CompressionService
from config import APP_CONFIG
from critic import CriticAgent
from critique_service import CritiqueService
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
    ReasoningLog,
    ResourceBudget,
    SourceType,
    utc_now,
)
from planner import PlannerAgent
from reasoner import ReasonerAgent
from reasoning_service import ReasoningService
from researcher import ResearcherAgent
from retrieval import stable_hash


def _budget() -> ResourceBudget:
    return ResourceBudget(retrieval_top_k=4, max_web_queries=2, reasoner_passes=3, critic_passes=2, macro_depth=3)


def _plan(question: str = "What is 2 + 2?") -> Plan:
    return Plan(
        task_id="task-unit",
        question=question,
        steps=(PlanStep(step_id="step_1", description="Interpret"),),
        required_evidence=("support",),
        success_criteria=("be correct",),
        budget=_budget(),
    )


def _evidence() -> EvidenceBundle:
    return EvidenceBundle(
        task_id="task-unit",
        local_results=(
            EvidenceItem(
                id="ev-local-1",
                content="The arithmetic result of 2 + 2 is 4.",
                source_type=SourceType.LOCAL,
                source_ref="local://math",
                score=0.95,
            ),
            EvidenceItem(
                id="ev-local-2",
                content="Verified arithmetic examples show that 2 + 2 equals 4.",
                source_type=SourceType.LOCAL,
                source_ref="local://math-2",
                score=0.91,
            ),
        ),
        web_results=(),
        used_web_fallback=False,
    )


def _candidate_proof_hash(candidate_id: str, operation_stream: tuple[OperationStep, ...], evidence_ids: tuple[str, ...]) -> str:
    payload = {
        "task_id": candidate_id,
        "tokens": ["@candidate_prepare", "@candidate_reason", "@candidate_verify", "@candidate_emit"],
        "operation_stream": [step.to_dict() for step in operation_stream],
        "evidence_handles": list(evidence_ids),
    }
    return stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _candidate(
    candidate_id: str,
    answer_text: str,
    *,
    strategy: str,
    verifier_type: str,
    verified: bool,
    total_score: float,
    evidence_ids: tuple[str, ...] = ("ev-local-1",),
    degraded_reason: str = "",
) -> CandidateTrace:
    operation_stream = (
        OperationStep(
            op_id=f"{candidate_id}_lookup",
            opcode="lookup",
            args=("sym_question",),
            output_ref="sym_evidence_set",
            context_frame_id="cf_candidate",
            evidence_handles=evidence_ids,
        ),
        OperationStep(
            op_id=f"{candidate_id}_emit",
            opcode="emit",
            args=("sym_answer",),
            context_frame_id="cf_candidate",
            evidence_handles=evidence_ids,
            metadata={"candidate_id": candidate_id, "answer_text": answer_text},
        ),
    )
    return CandidateTrace(
        candidate_id=candidate_id,
        answer_text=answer_text,
        strategy=strategy,
        verifier_type=verifier_type,
        verified=verified,
        total_score=total_score,
        evidence_support_score=0.9 if verified else 0.3,
        proof_hash_stability=1.0,
        degraded_reason=degraded_reason,
        supporting_evidence_ids=evidence_ids,
        tokens=("@candidate_prepare", "@candidate_reason", "@candidate_verify", "@candidate_emit"),
        expanded_preview=("Prepare", "Reason", "Verify", "Emit"),
        operation_stream=operation_stream,
        decode_hints=(
            DecodeHint(
                hint_id=f"{candidate_id}_hint",
                template="verified_answer",
                entity_ids=("a",),
                metadata={"candidate_id": candidate_id, "answer_text": answer_text},
            ),
        ),
        proof_hash=_candidate_proof_hash(candidate_id, operation_stream, evidence_ids),
    )


def _trace(*, candidates: tuple[CandidateTrace, ...], answer_text: str = "4", candidate_id: str = "cand_selected") -> CompressedTrace:
    return CompressedTrace(
        task_id="task-unit",
        tokens=("@read_question", "@compare_candidates", "@compose_answer"),
        expanded_preview=("Read question", "Compare candidates", "Compose answer"),
        macros_used=(),
        confidence=0.88,
        operation_stream=(
            OperationStep(
                op_id="op_lookup",
                opcode="lookup",
                args=("sym_question",),
                output_ref="sym_evidence_set",
                context_frame_id="cf_main",
                evidence_handles=("ev-local-1", "ev-local-2"),
            ),
            OperationStep(
                op_id="op_emit",
                opcode="emit",
                args=("sym_answer",),
                context_frame_id="cf_main",
                evidence_handles=("ev-local-1", "ev-local-2"),
                metadata={
                    "candidate_id": candidate_id,
                    "answer_text": answer_text,
                    "selected_strategy": "top_evidence",
                    "selected_verifier": "tool.evidence_grounding",
                    "candidate_score": 0.68,
                    "verified": False,
                    "candidate_count": len(candidates),
                    "supporting_evidence_ids": ["ev-local-1"],
                },
            ),
        ),
        evidence_handles=("ev-local-1", "ev-local-2"),
        context_frames=(
            ContextFrame(
                frame_id="cf_main",
                scope="task",
                confidence=0.88,
                provenance_bundle_id="bundle_task",
                assumptions=(),
                metadata={"rm": "deep", "cid": candidate_id, "ta": answer_text, "cc": len(candidates)},
                created_at=utc_now(),
            ),
        ),
        candidate_traces=candidates,
        proof_hash="trace-proof",
        decode_hints=(
            DecodeHint(
                hint_id="hint_main",
                template="verified_answer",
                entity_ids=("a",),
                metadata={"candidate_id": candidate_id, "answer_text": answer_text},
            ),
        ),
    )


class _FakePlannerService:
    def __init__(self, plan: Plan) -> None:
        self.plan_result = plan
        self.calls: list[tuple[str, ResourceBudget]] = []

    async def plan(self, question: str, budget: ResourceBudget) -> Plan:
        self.calls.append((question, budget))
        return self.plan_result


class _FakeResearchService:
    def __init__(self, evidence: EvidenceBundle) -> None:
        self.evidence_result = evidence
        self.calls: list[tuple[Plan, ResourceBudget]] = []
        self.web_adapter = "stub-web"

    async def research(self, plan: Plan, budget: ResourceBudget) -> EvidenceBundle:
        self.calls.append((plan, budget))
        return self.evidence_result

    async def reset(self) -> None:
        return None


class _FakeReasoningService:
    final_text_policy = "post_verification"
    implementation_mode = "deterministic_stub"

    def __init__(self, trace: CompressedTrace) -> None:
        self.trace_result = trace
        self.received_handoff = None
        self.last_runtime_subset = None
        self.last_handoff = None

    async def reason(self, handoff) -> CompressedTrace:
        self.received_handoff = handoff
        self.last_handoff = handoff
        return self.trace_result


class _FakeCritiqueService:
    final_text_policy = "post_verification"
    implementation_mode = "deterministic_stub"

    def __init__(self, report: CritiqueReport) -> None:
        self.report_result = report
        self.received_handoff = None
        self.last_runtime_subset = None
        self.last_handoff = None

    async def review(self, handoff) -> CritiqueReport:
        self.received_handoff = handoff
        self.last_handoff = handoff
        return self.report_result


class _FakeCompressionService:
    def __init__(self, proposals: list) -> None:
        self.proposals = proposals
        self.calls: list[tuple[CompressedTrace, list[ReasoningLog]]] = []

    async def propose(self, trace: CompressedTrace, logs: list[ReasoningLog]):
        self.calls.append((trace, logs))
        return self.proposals


class Phase12AgentUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_planner_agent_uses_injected_service(self) -> None:
        service = _FakePlannerService(_plan())
        agent = PlannerAgent(model_manager=SimpleNamespace(), service=service)

        with self.assertRaises(RuntimeError):
            await agent.plan("What is 2 + 2?", _budget())

        await agent.start()
        result = await agent.plan("What is 2 + 2?", _budget())

        self.assertEqual(result.task_id, "task-unit")
        self.assertEqual(service.calls[0][0], "What is 2 + 2?")

    async def test_researcher_agent_uses_injected_service_and_proxies_web_adapter(self) -> None:
        service = _FakeResearchService(_evidence())
        agent = ResearcherAgent(model_manager=SimpleNamespace(), storage=SimpleNamespace(), service=service)
        await agent.start()

        result = await agent.research(_plan(), _budget())
        self.assertEqual(result.task_id, "task-unit")
        self.assertEqual(agent.web_adapter, "stub-web")
        agent.web_adapter = "updated-web"
        self.assertEqual(service.web_adapter, "updated-web")

    async def test_reasoner_agent_builds_typed_handoff_for_service(self) -> None:
        candidates = (_candidate("cand_1", "4", strategy="tool_arithmetic", verifier_type="tool.python_ast_arithmetic", verified=True, total_score=0.99),)
        service = _FakeReasoningService(_trace(candidates=candidates))
        agent = ReasonerAgent(model_manager=SimpleNamespace(), service=service)
        await agent.start()

        result = await agent.reason(_plan(), _evidence(), _budget())

        self.assertEqual(result.task_id, "task-unit")
        self.assertEqual(service.received_handoff.plan.task_id, "task-unit")
        self.assertEqual(service.received_handoff.final_text_policy, "post_verification")
        self.assertEqual(service.received_handoff.implementation_mode, "deterministic_stub")
        self.assertEqual(service.received_handoff.evidence_handles, ("ev-local-1", "ev-local-2"))

    async def test_critic_agent_builds_typed_handoff_for_service(self) -> None:
        candidates = (_candidate("cand_1", "4", strategy="tool_arithmetic", verifier_type="tool.python_ast_arithmetic", verified=True, total_score=0.99),)
        trace = _trace(candidates=candidates)
        report = CritiqueReport(task_id="task-unit", is_valid=True, issues=(), fixed_trace=trace, evidence_coverage=1.0, result=CritiqueResult.VALID)
        service = _FakeCritiqueService(report)
        agent = CriticAgent(model_manager=SimpleNamespace(), service=service)
        await agent.start()

        result = await agent.review(_plan(), _evidence(), trace, _budget())

        self.assertTrue(result.is_valid)
        self.assertEqual(service.received_handoff.proof_hash, "trace-proof")
        self.assertEqual(service.received_handoff.required_opcode_names, ("lookup", "emit"))

    async def test_compressor_agent_uses_injected_service(self) -> None:
        candidates = (_candidate("cand_1", "4", strategy="tool_arithmetic", verifier_type="tool.python_ast_arithmetic", verified=True, total_score=0.99),)
        trace = _trace(candidates=candidates)
        logs = [ReasoningLog(task_id="task-unit", compressed_chain=trace.tokens, macros_used=())]
        service = _FakeCompressionService(["proposal"])
        agent = CompressorAgent(model_manager=SimpleNamespace(), service=service)
        await agent.start()

        proposals = await agent.propose(trace, logs)

        self.assertEqual(proposals, ["proposal"])
        self.assertEqual(service.calls[0][0], trace)
        self.assertEqual(service.calls[0][1], logs)


class Phase12ReasonerCriticFocusedUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reasoning_service = ReasoningService(model_manager=SimpleNamespace())
        self.critique_service = CritiqueService(model_manager=SimpleNamespace())

    def test_reasoning_service_builds_tool_and_abstain_candidates(self) -> None:
        handoff = type("Handoff", (), {})()
        handoff.plan = _plan("What is 2 + 2?")
        handoff.evidence = _evidence()
        handoff.budget = _budget()
        handoff.reasoning_mode = "deep"
        handoff.evidence_handles = ("ev-local-1", "ev-local-2")

        candidates = self.reasoning_service._build_answer_candidates(handoff)

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["answer_text"], "4")
        self.assertTrue(any(candidate["strategy"] == "abstain" for candidate in candidates))

    def test_reasoning_service_prefers_abstain_in_deep_mode_when_no_candidate_is_verified(self) -> None:
        selected = self.reasoning_service._select_answer_candidate(
            (
                {
                    "candidate_id": "cand_1",
                    "answer_text": "weak answer",
                    "strategy": "top_evidence",
                    "verified": False,
                    "total_score": 0.61,
                    "evidence_support_score": 0.52,
                },
                {
                    "candidate_id": "cand_2",
                    "answer_text": "I cannot verify this yet.",
                    "strategy": "abstain",
                    "verified": False,
                    "total_score": 0.3,
                    "evidence_support_score": 0.0,
                },
            ),
            reasoning_mode="deep",
        )

        self.assertEqual(selected["strategy"], "abstain")

    def test_critique_service_flags_when_trace_selected_weaker_candidate_than_best_verified_candidate(self) -> None:
        better = _candidate(
            "cand_best",
            "4",
            strategy="tool_arithmetic",
            verifier_type="tool.python_ast_arithmetic",
            verified=True,
            total_score=0.99,
        )
        weaker = _candidate(
            "cand_selected",
            "four-ish",
            strategy="top_evidence",
            verifier_type="tool.evidence_grounding",
            verified=False,
            total_score=0.62,
            degraded_reason="no_candidate_met_verification_threshold",
        )
        trace = _trace(candidates=(weaker, better), answer_text="four-ish", candidate_id="cand_selected")
        handoff = type("Handoff", (), {})()
        handoff.plan = _plan("What is 2 + 2?")
        handoff.evidence = _evidence()
        handoff.trace = trace
        handoff.budget = _budget()
        handoff.evidence_handles = ("ev-local-1", "ev-local-2")
        handoff.proof_hash = "trace-proof"

        review = self.critique_service._review_candidate_traces(handoff)

        self.assertEqual(review["verifier_type"], "tool.evidence_grounding")
        self.assertIn("weaker candidate", " ".join(review["issues"]))
        self.assertEqual(review["degraded_reason"], "no_candidate_met_verification_threshold")

    def test_critique_service_tool_checks_return_arithmetic_failure_details(self) -> None:
        candidate = _candidate(
            "cand_selected",
            "5",
            strategy="tool_arithmetic",
            verifier_type="tool.python_ast_arithmetic",
            verified=True,
            total_score=0.99,
        )
        trace = _trace(candidates=(candidate,), answer_text="5", candidate_id="cand_selected")
        handoff = type("Handoff", (), {})()
        handoff.plan = _plan("What is 2 + 2?")
        handoff.evidence = _evidence()
        handoff.trace = trace
        handoff.budget = _budget()

        issues, checks_run, details = self.critique_service._run_tool_checks(handoff)

        self.assertEqual(checks_run, ["tool.python_ast_arithmetic"])
        self.assertEqual(details["verifier_type"], "tool.python_ast_arithmetic")
        self.assertIn("expected '4' but trace emitted '5'", issues[0])


if __name__ == "__main__":
    unittest.main()
