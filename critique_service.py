"""Shared critique logic used by CriticAgent."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from agent_schema import critic_output_schema, parse_critic_output
from config import APP_CONFIG, AppConfig
from data_structures import (
    CandidateTrace,
    CompressionRuntimeSubset,
    CompressedTrace,
    CritiqueReport,
    CritiqueResult,
    DecodeHint,
    ReasonerCriticHandoff,
)
from prompts import CRITIC_PROMPT
from retrieval import stable_hash
from structured_generation import StructuredGenerationService
from verification_tools import (
    evaluate_arithmetic_question,
    evaluate_python_code_question,
    evaluate_python_expression_question,
    evaluate_python_unit_test_question,
    expected_evidence_count,
    measure_evidence_support,
    measure_candidate_agreement,
    verify_expected_answer,
)

if TYPE_CHECKING:
    from model_manager import ModelManager
    from storage import StorageManager


class CritiqueService:
    """Validate a reasoning trace from a typed handoff."""

    output_contract = "critique_report_v1"
    handoff_contract = "reasoner_critic_handoff_v1"
    implementation_mode = "deterministic_stub"
    final_text_policy = "post_verification"

    def __init__(
        self,
        model_manager: ModelManager,
        storage: StorageManager | None = None,
        config: AppConfig = APP_CONFIG,
    ):
        self.model_manager = model_manager
        self.storage = storage
        self.config = config
        self.structured_generation = StructuredGenerationService(model_manager=model_manager, config=config)
        self._last_runtime_subset: CompressionRuntimeSubset | None = None
        self._last_handoff: ReasonerCriticHandoff | None = None

    @property
    def last_runtime_subset(self) -> CompressionRuntimeSubset | None:
        return self._last_runtime_subset

    @property
    def last_handoff(self) -> ReasonerCriticHandoff | None:
        return self._last_handoff

    async def review(self, handoff: ReasonerCriticHandoff) -> CritiqueReport:
        self._last_handoff = handoff
        runtime_subset = await self._load_runtime_subset(handoff.plan.task_id, handoff)
        self._last_runtime_subset = runtime_subset
        has_evidence = bool(handoff.evidence.local_results) or bool(handoff.evidence.web_results)
        issues, checks_run, deterministic_details = self._run_deterministic_checks(handoff, runtime_subset)
        base_report = self._build_deterministic_report(
            handoff=handoff,
            runtime_subset=runtime_subset,
            has_evidence=has_evidence,
            issues=issues,
            checks_run=checks_run,
            deterministic_details=deterministic_details,
        )
        if issues:
            return replace(
                base_report,
                critic_notes=self._deterministic_critic_notes(base_report=base_report),
            )

        prompt = self._build_structured_prompt(handoff, runtime_subset, checks_run)
        decode_result = await self.structured_generation.decode_json_output(
            prompt=prompt,
            schema=critic_output_schema(),
            parser=parse_critic_output,
            fallback_factory=lambda _error_message: base_report,
            max_tokens=self.config.model_tuning.default_max_tokens,
        )
        if decode_result.used_fallback:
            return replace(
                base_report,
                critic_notes=self._fallback_critic_notes(
                    base_report=base_report,
                    error_message=decode_result.error_message,
                    raw_output=decode_result.raw_text,
                    repaired_output=decode_result.repaired_text,
                ),
            )
        return self._normalize_structured_report(
            parsed_report=decode_result.value,
            base_report=base_report,
            handoff=handoff,
            raw_output=decode_result.raw_text,
            repaired_output=decode_result.repaired_text,
            used_repair=decode_result.used_repair,
        )

    async def _load_runtime_subset(
        self,
        task_id: str,
        handoff: ReasonerCriticHandoff,
    ) -> CompressionRuntimeSubset:
        if self.storage is None:
            return CompressionRuntimeSubset(task_id=task_id)
        return await self.storage.load_active_compression_runtime(
            task_id,
            macro_names=handoff.required_macro_names,
            opcode_names=handoff.required_opcode_names,
            decoder_names=handoff.required_decoder_names,
        )

    def _run_deterministic_checks(
        self,
        handoff: ReasonerCriticHandoff,
        runtime_subset: CompressionRuntimeSubset,
    ) -> tuple[list[str], list[str], dict[str, Any]]:
        evidence = handoff.evidence
        trace = handoff.trace
        budget = handoff.budget
        has_evidence = bool(evidence.local_results) or bool(evidence.web_results)
        candidate_metadata = self._extract_trace_candidate_metadata(trace)
        candidate_review = self._review_candidate_traces(handoff)
        issues: list[str] = []
        checks_run: list[str] = []
        details: dict[str, Any] = {
            "verifier_type": str(
                candidate_review.get("verifier_type") or candidate_metadata.get("selected_verifier", "")
            ),
            "proof_hash_match": trace.proof_hash == handoff.proof_hash,
            "candidate_score": round(
                max(
                    float(candidate_metadata.get("candidate_score", 0.0)),
                    float(candidate_review.get("candidate_score", 0.0)),
                ),
                3,
            ),
            "repair_actions": [],
            "degraded_reason": str(
                candidate_review.get("degraded_reason") or candidate_metadata.get("degraded_reason", "")
            ),
            "failure_categories": (),
            "provenance_coverage": round(float(candidate_review.get("provenance_coverage", 1.0)), 3),
            "macro_violations": (),
            "drift_score": 0.0,
        }
        if not details["proof_hash_match"]:
            issues.append("Trace proof hash drifted between reasoner and critic handoff.")
        if budget.critic_passes >= 1:
            checks_run.extend(
                (
                    "check.evidence_presence",
                    "check.trace_structure",
                    "check.candidate_review",
                )
            )
            if not has_evidence:
                issues.append("No evidence found in local or web sources.")
            issues.extend(self._validate_trace_structure(handoff))
            issues.extend(str(item) for item in candidate_review.get("issues", ()))
            tool_issues, tool_checks, tool_details = self._run_tool_checks(handoff)
            issues.extend(tool_issues)
            checks_run.extend(tool_checks)
            if tool_details.get("verifier_type"):
                details["verifier_type"] = str(tool_details["verifier_type"])
            details["candidate_score"] = round(
                max(float(details["candidate_score"]), float(tool_details.get("candidate_score", 0.0))),
                3,
            )
            if tool_details.get("degraded_reason"):
                details["degraded_reason"] = str(tool_details["degraded_reason"])
            if "provenance_coverage" in tool_details:
                details["provenance_coverage"] = round(
                    max(float(details["provenance_coverage"]), float(tool_details["provenance_coverage"])),
                    3,
                )
        if budget.critic_passes >= 2:
            checks_run.extend(
                (
                    "check.trace_nonempty",
                    "check.reasoning_pass_tokens",
                    "check.runtime_subset_alignment",
                )
            )
            if not trace.tokens or not trace.expanded_preview:
                issues.append("Reasoning trace is missing required projections.")
            expected_pass_tokens = {
                f"@reason_pass_{pass_index}" for pass_index in range(1, budget.reasoner_passes + 1)
            }
            if not expected_pass_tokens.issubset(set(trace.tokens)):
                issues.append("Reasoning trace does not include all budgeted reasoning passes.")
            issues.extend(self._validate_runtime_subset(handoff, runtime_subset))
        if budget.critic_passes >= 3:
            checks_run.extend(("check.plan_budget_alignment", "check.web_fallback_alignment"))
            if not any("Reasoning pass" in preview for preview in trace.expanded_preview):
                issues.append("Expanded reasoning preview does not expose pass-level debug information.")
            if evidence.web_results and not evidence.used_web_fallback:
                issues.append("Web evidence is present but used_web_fallback is false.")
        issues = list(dict.fromkeys(issues))
        details["macro_violations"] = self._extract_macro_violations(issues)
        details["failure_categories"] = self._derive_failure_categories(
            issues=issues,
            proof_hash_match=bool(details["proof_hash_match"]),
            degraded_reason=str(details["degraded_reason"]),
            provenance_coverage=float(details["provenance_coverage"]),
            macro_violations=tuple(details["macro_violations"]),
        )
        details["drift_score"] = self._derive_drift_score(
            proof_hash_match=bool(details["proof_hash_match"]),
            provenance_coverage=float(details["provenance_coverage"]),
            macro_violations=tuple(details["macro_violations"]),
        )
        details["repair_actions"] = self._derive_repair_actions(
            issues=issues,
            proof_hash_match=bool(details["proof_hash_match"]),
            degraded_reason=str(details["degraded_reason"]),
            failure_categories=tuple(str(item) for item in details["failure_categories"]),
        )
        return issues, checks_run, details

    def _review_candidate_traces(self, handoff: ReasonerCriticHandoff) -> dict[str, Any]:
        evidence_items = tuple(
            (item.id, item.content) for item in (handoff.evidence.local_results + handoff.evidence.web_results)
        )
        trace = handoff.trace
        selected_metadata = self._extract_trace_candidate_metadata(trace)
        raw_candidates = trace.candidate_traces
        if not raw_candidates and selected_metadata.get("answer_text"):
            raw_candidates = (
                self._candidate_trace_from_selected_trace(
                    handoff=handoff,
                    selected_metadata=selected_metadata,
                ),
            )
        if not raw_candidates:
            return {
                "verifier_type": str(selected_metadata.get("selected_verifier", "")),
                "candidate_score": float(selected_metadata.get("candidate_score", 0.0) or 0.0),
                "degraded_reason": str(selected_metadata.get("degraded_reason", "")),
                "provenance_coverage": 0.0 if evidence_items else 1.0,
                "issues": (),
            }

        reviewed_candidates: list[dict[str, Any]] = []
        for candidate in raw_candidates:
            peer_answers = [
                peer.answer_text
                for peer in raw_candidates
                if peer.candidate_id != candidate.candidate_id and peer.strategy != "abstain"
            ]
            support_result = measure_evidence_support(candidate.answer_text, evidence_items)
            agreement_score = measure_candidate_agreement(candidate.answer_text, peer_answers)
            proof_hash_stability = self._candidate_proof_hash_stability(candidate)
            if candidate.strategy == "abstain":
                verifier_score = 0.3
            elif candidate.verifier_type in {
                "tool.python_ast_arithmetic",
                "tool.python_expression",
                "tool.python_code_execution",
                "tool.python_unit_test",
                "tool.evidence_count",
            }:
                verifier_score = 1.0
            elif candidate.verifier_type == "tool.evidence_grounding":
                verifier_score = 0.72 if candidate.verified else 0.5
            elif candidate.verifier_type.startswith("tool."):
                verifier_score = 1.0
            elif candidate.verified:
                verifier_score = 0.9
            else:
                verifier_score = max(0.2, min(0.8, candidate.total_score))
            total_score = round(
                min(
                    0.99,
                    (verifier_score * 0.45)
                    + (support_result.score * 0.25)
                    + (proof_hash_stability * 0.15)
                    + (agreement_score * 0.15),
                ),
                3,
            )
            provenance_coverage = self._calculate_provenance_coverage(
                verifier_type=candidate.verifier_type,
                supporting_evidence_ids=candidate.supporting_evidence_ids or support_result.supporting_evidence_ids,
                evidence_handles=handoff.evidence_handles,
                has_evidence=bool(evidence_items),
            )
            reviewed_candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "answer_text": candidate.answer_text,
                    "strategy": candidate.strategy,
                    "verifier_type": candidate.verifier_type,
                    "verified": candidate.verified,
                    "total_score": total_score,
                    "provenance_coverage": provenance_coverage,
                    "degraded_reason": candidate.degraded_reason,
                }
            )
        reviewed_candidates.sort(
            key=lambda candidate: (
                bool(candidate["verified"]),
                float(candidate["total_score"]),
                float(candidate["provenance_coverage"]),
                str(candidate["strategy"]).startswith("tool_"),
            ),
            reverse=True,
        )
        selected_candidate = self._select_reviewed_candidate(
            reviewed_candidates=tuple(reviewed_candidates),
            selected_metadata=selected_metadata,
        )
        issues: list[str] = []
        best_candidate = reviewed_candidates[0]
        if (
            best_candidate["candidate_id"] != selected_candidate["candidate_id"]
            and best_candidate["verified"]
            and selected_candidate["strategy"] != "abstain"
            and self._verifier_priority(str(best_candidate.get("verifier_type", "")))
            >= self._verifier_priority(str(selected_candidate.get("verifier_type", "")))
            and float(best_candidate.get("total_score", 0.0))
            > float(selected_candidate.get("total_score", 0.0)) + 0.05
        ):
            issues.append("Trace selected a weaker candidate than the critic-ranked candidate set.")
        threshold = self._candidate_review_threshold(trace)
        if all(
            float(candidate["total_score"]) < threshold
            for candidate in reviewed_candidates
            if str(candidate["strategy"]) != "abstain"
        ):
            if selected_candidate["strategy"] != "abstain":
                issues.append("No candidate reached the configured verification threshold.")
            degraded_reason = "no_candidate_met_verification_threshold"
        else:
            degraded_reason = str(selected_candidate.get("degraded_reason", "")) or str(
                selected_metadata.get("degraded_reason", "")
            )
        return {
            "verifier_type": str(selected_candidate.get("verifier_type", "")),
            "candidate_score": round(float(selected_candidate.get("total_score", 0.0)), 3),
            "degraded_reason": degraded_reason,
            "provenance_coverage": round(float(selected_candidate.get("provenance_coverage", 1.0)), 3),
            "issues": tuple(issues),
        }

    def _candidate_trace_from_selected_trace(
        self,
        *,
        handoff: ReasonerCriticHandoff,
        selected_metadata: dict[str, Any],
    ):
        evidence_handles = tuple(
            str(item) for item in selected_metadata.get("supporting_evidence_ids", ()) if str(item).strip()
        ) or handoff.evidence_handles
        return CandidateTrace(
            candidate_id=str(selected_metadata.get("candidate_id", "cand_selected")),
            answer_text=str(selected_metadata.get("answer_text", "")).strip(),
            strategy=str(selected_metadata.get("selected_strategy", "")),
            verifier_type=str(selected_metadata.get("selected_verifier", "")),
            verified=bool(selected_metadata.get("verified", False)),
            total_score=float(selected_metadata.get("candidate_score", 0.0) or 0.0),
            evidence_support_score=0.0,
            proof_hash_stability=1.0 if handoff.trace.proof_hash == handoff.proof_hash else 0.0,
            degraded_reason=str(selected_metadata.get("degraded_reason", "")),
            supporting_evidence_ids=evidence_handles,
            tokens=handoff.trace.tokens,
            expanded_preview=handoff.trace.expanded_preview,
            operation_stream=handoff.trace.operation_stream,
            decode_hints=(
                DecodeHint(
                    hint_id="selected_candidate_hint",
                    template="verified_answer",
                    entity_ids=("a",),
                    metadata={
                        "candidate_id": str(selected_metadata.get("candidate_id", "cand_selected")),
                        "answer_text": str(selected_metadata.get("answer_text", "")).strip(),
                    },
                ),
            ),
            proof_hash=handoff.trace.proof_hash,
        )

    def _candidate_proof_hash_stability(self, candidate) -> float:
        proof_payload = {
            "task_id": candidate.candidate_id,
            "tokens": list(candidate.tokens),
            "operation_stream": [step.to_dict() for step in candidate.operation_stream],
            "evidence_handles": list(candidate.supporting_evidence_ids),
        }
        expected_hash = stable_hash(json.dumps(proof_payload, sort_keys=True, separators=(",", ":")))
        return 1.0 if bool(candidate.proof_hash) and expected_hash == candidate.proof_hash else 0.0

    def _select_reviewed_candidate(
        self,
        *,
        reviewed_candidates: tuple[dict[str, Any], ...],
        selected_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        selected_candidate_id = str(selected_metadata.get("candidate_id", "")).strip()
        selected_answer = str(selected_metadata.get("answer_text", "")).strip()
        for candidate in reviewed_candidates:
            if selected_candidate_id and candidate["candidate_id"] == selected_candidate_id:
                return candidate
        for candidate in reviewed_candidates:
            if selected_answer and candidate["answer_text"] == selected_answer:
                return candidate
        return reviewed_candidates[0]

    def _candidate_review_threshold(self, trace: CompressedTrace) -> float:
        if trace.context_frames and trace.context_frames[0].metadata.get("rm") == "deep":
            return 0.72
        return 0.65

    def _verifier_priority(self, verifier_type: str) -> int:
        if verifier_type in {
            "tool.python_ast_arithmetic",
            "tool.python_expression",
            "tool.python_code_execution",
            "tool.python_unit_test",
            "tool.evidence_count",
        }:
            return 3
        if verifier_type == "tool.evidence_grounding":
            return 2
        if verifier_type.startswith("tool."):
            return 1
        return 0

    def _calculate_provenance_coverage(
        self,
        *,
        verifier_type: str,
        supporting_evidence_ids: tuple[str, ...],
        evidence_handles: tuple[str, ...],
        has_evidence: bool,
    ) -> float:
        if verifier_type in {
            "tool.python_ast_arithmetic",
            "tool.python_expression",
            "tool.python_code_execution",
            "tool.python_unit_test",
            "tool.evidence_count",
        }:
            return 1.0
        if not has_evidence:
            return 1.0
        if not supporting_evidence_ids or not evidence_handles:
            return 0.0
        return round(
            min(1.0, len(set(supporting_evidence_ids) & set(evidence_handles)) / max(1, len(set(evidence_handles)))),
            3,
        )

    def _validate_trace_structure(self, handoff: ReasonerCriticHandoff) -> list[str]:
        trace = handoff.trace
        issues: list[str] = []
        if not trace.operation_stream:
            issues.append("Reasoning trace is missing required projections.")
            return issues
        if trace.operation_stream[-1].opcode != "emit":
            issues.append("Reasoning trace must end with an emit operation.")
        if (
            any(step.opcode in {"compare", "infer", "check", "cite", "confidence_update"} for step in trace.operation_stream)
            or (trace.context_frames and trace.context_frames[0].metadata.get("rm") == "deep")
        ) and not any(step.opcode == "bind" for step in trace.operation_stream):
            issues.append("Reasoning trace is missing a bind operation before emit.")
        if any(
            handle and handle not in handoff.evidence_handles
            for step in trace.operation_stream
            for handle in step.evidence_handles
        ):
            issues.append("Trace references evidence handles outside the typed handoff.")
        if trace.context_frames and trace.context_frames[0].metadata.get("rm") == "deep":
            expected_opcodes = ("lookup", "compare", "infer", "bind", "check", "cite", "confidence_update", "emit")
            actual_opcodes = tuple(step.opcode for step in trace.operation_stream)
            if actual_opcodes != expected_opcodes:
                issues.append("Deep-mode trace is missing required candidate-selection opcodes or ordering.")
        return issues

    def _extract_macro_violations(self, issues: list[str]) -> tuple[str, ...]:
        violations: list[str] = []
        for issue in issues:
            if not issue.startswith("Active macro subset is missing required entries:"):
                continue
            missing = issue.partition(":")[2].strip().rstrip(".")
            violations.extend(part.strip() for part in missing.split(",") if part.strip())
        return tuple(dict.fromkeys(violations))

    def _derive_failure_categories(
        self,
        *,
        issues: list[str],
        proof_hash_match: bool,
        degraded_reason: str,
        provenance_coverage: float,
        macro_violations: tuple[str, ...],
    ) -> tuple[str, ...]:
        categories: list[str] = []
        lowered_issues = [issue.lower() for issue in issues]
        if not proof_hash_match:
            categories.append("proof_hash")
        if any(
            token in issue
            for issue in lowered_issues
            for token in (
                "required projections",
                "must end with an emit",
                "reasoning passes",
                "candidate-selection opcodes",
            )
        ):
            categories.append("schema")
        if any(token in issue for issue in lowered_issues for token in ("symbol table", "typed handoff")):
            categories.append("provenance")
        if any("evidence" in issue for issue in lowered_issues) or degraded_reason in {
            "low_evidence_support",
            "no_candidate_met_verification_threshold",
            "no_retrieved_evidence",
        } or provenance_coverage < 0.8:
            categories.append("evidence_coverage")
        if any("tool verification failed" in issue for issue in lowered_issues):
            categories.append("tool_verification")
        if any("weaker candidate" in issue for issue in lowered_issues):
            categories.append("candidate_selection")
        if macro_violations:
            categories.append("macro_signature")
        return tuple(dict.fromkeys(categories))

    def _derive_drift_score(
        self,
        *,
        proof_hash_match: bool,
        provenance_coverage: float,
        macro_violations: tuple[str, ...],
    ) -> float:
        return round(
            min(
                1.0,
                (0.45 if not proof_hash_match else 0.0)
                + (0.35 * (1.0 - max(0.0, min(1.0, provenance_coverage))))
                + (0.2 if macro_violations else 0.0),
            ),
            3,
        )

    def _build_deterministic_report(
        self,
        *,
        handoff: ReasonerCriticHandoff,
        runtime_subset: CompressionRuntimeSubset,
        has_evidence: bool,
        issues: list[str],
        checks_run: list[str],
        deterministic_details: dict[str, Any],
    ) -> CritiqueReport:
        is_valid = len(issues) == 0
        degraded_reason = str(deterministic_details.get("degraded_reason", ""))
        provenance_coverage = min(
            1.0,
            max(0.0, float(deterministic_details.get("provenance_coverage", 1.0 if has_evidence else 0.0))),
        )
        evidence_coverage = provenance_coverage if has_evidence else max(0.0, provenance_coverage)
        return CritiqueReport(
            task_id=handoff.plan.task_id,
            is_valid=is_valid,
            issues=tuple(issues),
            fixed_trace=handoff.trace if is_valid else None,
            evidence_coverage=evidence_coverage,
            critic_notes=(
                f"checks_run={','.join(checks_run)}\n"
                f"final_text_policy={handoff.final_text_policy}\n"
                f"repair_attempt_count={handoff.repair_attempt_count}\n"
                f"verifier_type={deterministic_details.get('verifier_type', '')}\n"
                f"proof_hash_match={'yes' if deterministic_details.get('proof_hash_match', True) else 'no'}\n"
                f"candidate_score={deterministic_details.get('candidate_score', 0.0)}\n"
                f"repair_actions={','.join(deterministic_details.get('repair_actions', ())) or 'none'}\n"
                f"degraded_reason={degraded_reason or 'none'}\n"
                f"failure_categories={','.join(deterministic_details.get('failure_categories', ())) or 'none'}\n"
                f"provenance_coverage={provenance_coverage}\n"
                f"macro_violations={','.join(deterministic_details.get('macro_violations', ())) or 'none'}\n"
                f"drift_score={deterministic_details.get('drift_score', 0.0)}\n"
                f"loaded_opcodes={','.join(opcode.opcode_name for opcode in runtime_subset.opcodes)}\n"
                f"loaded_macros={','.join(macro.macro_name for macro in runtime_subset.macros)}\n"
                f"loaded_decoders={','.join(decoder.decoder_name for decoder in runtime_subset.decoders)}\n"
                f"symbol_table_present={'yes' if runtime_subset.symbol_table is not None else 'no'}"
            ),
            result=self._select_report_result(issues=issues, degraded_reason=degraded_reason),
            verifier_type=str(deterministic_details.get("verifier_type", "")),
            proof_hash_match=bool(deterministic_details.get("proof_hash_match", True)),
            candidate_score=min(1.0, max(0.0, float(deterministic_details.get("candidate_score", 0.0)))),
            repair_actions=tuple(str(item) for item in deterministic_details.get("repair_actions", ())),
            degraded_reason=degraded_reason,
            failure_categories=tuple(str(item) for item in deterministic_details.get("failure_categories", ())),
            provenance_coverage=provenance_coverage,
            macro_violations=tuple(str(item) for item in deterministic_details.get("macro_violations", ())),
            drift_score=min(1.0, max(0.0, float(deterministic_details.get("drift_score", 0.0)))),
        )

    def _build_structured_prompt(
        self,
        handoff: ReasonerCriticHandoff,
        runtime_subset: CompressionRuntimeSubset,
        checks_run: list[str],
    ) -> str:
        trace = handoff.trace
        candidate_metadata = self._extract_trace_candidate_metadata(trace)
        return (
            f"{CRITIC_PROMPT}\n"
            f"Task: {handoff.plan.question}\n"
            f"TraceLength: {len(trace.tokens)}\n"
            f"CriticPasses: {handoff.budget.critic_passes}\n"
            f"ChecksRun: {checks_run}\n"
            f"AnswerCandidate: {self._extract_trace_answer(trace)}\n"
            f"ToolVerificationHint: {self._tool_verification_hint(handoff)}\n"
            f"VerifierTypeHint: {candidate_metadata.get('selected_verifier', '')}\n"
            f"CandidateScoreHint: {candidate_metadata.get('candidate_score', 0.0)}\n"
            f"CandidateTraceCountHint: {len(trace.candidate_traces)}\n"
            f"ProofHashMatchHint: {'yes' if trace.proof_hash == handoff.proof_hash else 'no'}\n"
            f"AllowedOpcodes: {[opcode.opcode_name for opcode in runtime_subset.opcodes]}\n"
            f"AllowedDecoders: {[decoder.decoder_name for decoder in runtime_subset.decoders]}\n"
            f"EvidenceCoverageHint: {1.0 if (handoff.evidence.local_results or handoff.evidence.web_results) else 0.0}\n"
            f"TraceTokens: {list(trace.tokens)}\n"
            f"OutputContract: {handoff.output_contract}"
        )

    def _normalize_structured_report(
        self,
        *,
        parsed_report: CritiqueReport,
        base_report: CritiqueReport,
        handoff: ReasonerCriticHandoff,
        raw_output: str,
        repaired_output: str | None,
        used_repair: bool,
    ) -> CritiqueReport:
        merged_issues = tuple(dict.fromkeys((*base_report.issues, *parsed_report.issues)))
        proof_hash_match = base_report.proof_hash_match and parsed_report.proof_hash_match
        is_valid = (
            parsed_report.is_valid
            and not merged_issues
            and parsed_report.result == CritiqueResult.VALID
            and proof_hash_match
        )
        result = parsed_report.result
        if is_valid and result != CritiqueResult.VALID:
            result = CritiqueResult.VALID
        if not is_valid and result == CritiqueResult.VALID:
            result = self._select_report_result(
                issues=list(merged_issues),
                degraded_reason=parsed_report.degraded_reason or base_report.degraded_reason,
            )
        return CritiqueReport(
            task_id=handoff.plan.task_id,
            is_valid=is_valid,
            issues=merged_issues,
            fixed_trace=handoff.trace if is_valid else None,
            evidence_coverage=min(1.0, max(0.0, parsed_report.evidence_coverage)),
            critic_notes=self._structured_critic_notes(
                base_report=base_report,
                model_notes=parsed_report.critic_notes,
                raw_output=raw_output,
                repaired_output=repaired_output,
                used_repair=used_repair,
            ),
            result=result,
            verifier_type=parsed_report.verifier_type or base_report.verifier_type,
            proof_hash_match=proof_hash_match,
            candidate_score=round(
                max(base_report.candidate_score, min(1.0, max(0.0, parsed_report.candidate_score))),
                3,
            ),
            repair_actions=parsed_report.repair_actions or base_report.repair_actions,
            degraded_reason=parsed_report.degraded_reason or base_report.degraded_reason,
            failure_categories=parsed_report.failure_categories or base_report.failure_categories,
            provenance_coverage=round(
                min(1.0, max(base_report.provenance_coverage, parsed_report.provenance_coverage)),
                3,
            ),
            macro_violations=parsed_report.macro_violations or base_report.macro_violations,
            drift_score=round(
                max(base_report.drift_score, min(1.0, max(0.0, parsed_report.drift_score))),
                3,
            ),
        )

    def _deterministic_critic_notes(
        self,
        *,
        base_report: CritiqueReport,
    ) -> str:
        return "\n".join(
            (
                base_report.critic_notes,
                "critic_output_mode=deterministic_verifier",
                f"output_contract={self.output_contract}",
                f"implementation_mode={self.implementation_mode}",
            )
        )

    def _structured_critic_notes(
        self,
        *,
        base_report: CritiqueReport,
        model_notes: str,
        raw_output: str,
        repaired_output: str | None,
        used_repair: bool,
    ) -> str:
        note_parts = [
            base_report.critic_notes,
            "critic_output_mode=structured_json",
            f"output_contract={self.output_contract}",
            f"implementation_mode={self.implementation_mode}",
            f"used_repair={'yes' if used_repair else 'no'}",
            "used_fallback=no",
            f"model_notes={model_notes}",
            f"raw_output={raw_output}",
        ]
        if repaired_output is not None:
            note_parts.append(f"repaired_output={repaired_output}")
        return "\n".join(part for part in note_parts if part)

    def _fallback_critic_notes(
        self,
        *,
        base_report: CritiqueReport,
        error_message: str | None,
        raw_output: str,
        repaired_output: str | None,
    ) -> str:
        note_parts = [
            base_report.critic_notes,
            "critic_output_mode=deterministic_fallback",
            f"output_contract={self.output_contract}",
            f"implementation_mode={self.implementation_mode}",
            f"parse_error={error_message or 'unknown'}",
            f"raw_output={raw_output}",
        ]
        if repaired_output is not None:
            note_parts.append(f"repaired_output={repaired_output}")
        return "\n".join(part for part in note_parts if part)

    def _run_tool_checks(
        self,
        handoff: ReasonerCriticHandoff,
    ) -> tuple[list[str], list[str], dict[str, Any]]:
        issues: list[str] = []
        checks_run: list[str] = []
        details: dict[str, Any] = {
            "verifier_type": "",
            "candidate_score": 0.0,
            "degraded_reason": "",
            "provenance_coverage": 0.0,
        }
        answer_text = self._extract_trace_answer(handoff.trace)
        arithmetic_answer = evaluate_arithmetic_question(handoff.plan.question)
        if arithmetic_answer is not None:
            checks_run.append("tool.python_ast_arithmetic")
            verification = verify_expected_answer(
                verifier_type="tool.python_ast_arithmetic",
                expected_answer=arithmetic_answer,
                actual_answer=answer_text,
            )
            details["verifier_type"] = verification.verifier_type
            details["candidate_score"] = verification.score
            details["provenance_coverage"] = 1.0
            if not answer_text:
                issues.append("Trace is missing an answer candidate for arithmetic tool verification.")
            elif not verification.matched:
                issues.append(
                    "Arithmetic tool verification failed: "
                    f"expected '{arithmetic_answer}' but trace emitted '{answer_text}'."
                )
            return issues, checks_run, details
        python_expression_answer = evaluate_python_expression_question(handoff.plan.question)
        if python_expression_answer is not None:
            checks_run.append("tool.python_expression")
            verification = verify_expected_answer(
                verifier_type="tool.python_expression",
                expected_answer=python_expression_answer,
                actual_answer=answer_text,
            )
            details["verifier_type"] = verification.verifier_type
            details["candidate_score"] = verification.score
            details["provenance_coverage"] = 1.0
            if not answer_text:
                issues.append("Trace is missing an answer candidate for Python-expression verification.")
            elif not verification.matched:
                issues.append(
                    "Python-expression tool verification failed: "
                    f"expected '{python_expression_answer}' but trace emitted '{answer_text}'."
                )
            return issues, checks_run, details
        python_code_answer = evaluate_python_code_question(handoff.plan.question)
        if python_code_answer is not None:
            checks_run.append("tool.python_code_execution")
            verification = verify_expected_answer(
                verifier_type="tool.python_code_execution",
                expected_answer=python_code_answer,
                actual_answer=answer_text,
            )
            details["verifier_type"] = verification.verifier_type
            details["candidate_score"] = verification.score
            details["provenance_coverage"] = 1.0
            if not answer_text:
                issues.append("Trace is missing an answer candidate for Python-code execution verification.")
            elif not verification.matched:
                issues.append(
                    "Python-code execution verification failed: "
                    f"expected '{python_code_answer}' but trace emitted '{answer_text}'."
                )
            return issues, checks_run, details
        python_unit_test_answer = evaluate_python_unit_test_question(handoff.plan.question)
        if python_unit_test_answer is not None:
            checks_run.append("tool.python_unit_test")
            verification = verify_expected_answer(
                verifier_type="tool.python_unit_test",
                expected_answer=python_unit_test_answer,
                actual_answer=answer_text,
            )
            details["verifier_type"] = verification.verifier_type
            details["candidate_score"] = verification.score
            details["provenance_coverage"] = 1.0
            if not answer_text:
                issues.append("Trace is missing an answer candidate for Python unit-test verification.")
            elif not verification.matched:
                issues.append(
                    "Python unit-test verification failed: "
                    f"expected '{python_unit_test_answer}' but trace emitted '{answer_text}'."
                )
            return issues, checks_run, details
        evidence_count_answer = expected_evidence_count(
            handoff.plan.question,
            len(handoff.evidence.local_results) + len(handoff.evidence.web_results),
        )
        if evidence_count_answer is not None:
            checks_run.append("tool.evidence_count")
            verification = verify_expected_answer(
                verifier_type="tool.evidence_count",
                expected_answer=evidence_count_answer,
                actual_answer=answer_text,
            )
            details["verifier_type"] = verification.verifier_type
            details["candidate_score"] = verification.score
            details["provenance_coverage"] = 1.0
            if not answer_text:
                issues.append("Trace is missing an answer candidate for evidence-count verification.")
            elif not verification.matched:
                issues.append(
                    "Evidence-count tool verification failed: "
                    f"expected '{evidence_count_answer}' but trace emitted '{answer_text}'."
                )
            return issues, checks_run, details
        if answer_text:
            checks_run.append("tool.evidence_grounding")
            support_result = measure_evidence_support(
                answer_text,
                tuple(
                    (item.id, item.content)
                    for item in (handoff.evidence.local_results + handoff.evidence.web_results)
                ),
            )
            details["verifier_type"] = "tool.evidence_grounding"
            details["candidate_score"] = support_result.score
            details["provenance_coverage"] = self._calculate_provenance_coverage(
                verifier_type="tool.evidence_grounding",
                supporting_evidence_ids=support_result.supporting_evidence_ids,
                evidence_handles=handoff.evidence_handles,
                has_evidence=bool(handoff.evidence.local_results or handoff.evidence.web_results),
            )
            if support_result.score < 0.35:
                details["degraded_reason"] = "low_evidence_support"
                issues.append(
                    "Evidence-grounding verification found weak support for the emitted answer candidate."
                )
        return issues, checks_run, details

    def _extract_trace_answer(self, trace: CompressedTrace) -> str:
        return str(self._extract_trace_candidate_metadata(trace).get("answer_text", ""))

    def _extract_trace_candidate_metadata(self, trace: CompressedTrace) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for step in reversed(trace.operation_stream):
            if step.opcode != "emit":
                continue
            answer_text = str(step.metadata.get("answer_text", "")).strip()
            if answer_text:
                metadata.update(step.metadata)
                break
        if not metadata:
            for hint in trace.decode_hints:
                answer_text = str(hint.metadata.get("answer_text", "")).strip()
                if answer_text:
                    metadata.update(hint.metadata)
                    break
        if trace.context_frames:
            context_metadata = dict(trace.context_frames[0].metadata)
            if "answer_text" not in metadata and context_metadata.get("ta"):
                metadata["answer_text"] = context_metadata.get("ta")
            if "candidate_id" not in metadata and context_metadata.get("cid"):
                metadata["candidate_id"] = context_metadata.get("cid")
            if "selected_strategy" not in metadata and context_metadata.get("sa"):
                metadata["selected_strategy"] = context_metadata.get("sa")
            if "selected_verifier" not in metadata and context_metadata.get("sv"):
                metadata["selected_verifier"] = context_metadata.get("sv")
            if "candidate_score" not in metadata and "ss" in context_metadata:
                metadata["candidate_score"] = context_metadata.get("ss")
            if "verified" not in metadata and "vv" in context_metadata:
                metadata["verified"] = context_metadata.get("vv")
            if "degraded_reason" not in metadata and context_metadata.get("dr"):
                metadata["degraded_reason"] = context_metadata.get("dr")
            if "candidate_count" not in metadata and "cc" in context_metadata:
                metadata["candidate_count"] = context_metadata.get("cc")
            if "supporting_evidence_ids" not in metadata and context_metadata.get("si"):
                metadata["supporting_evidence_ids"] = tuple(context_metadata.get("si", ()))
        try:
            candidate_count = max(1, int(metadata.get("candidate_count", 1) or 1))
        except (TypeError, ValueError):
            candidate_count = 1
        return {
            "candidate_id": str(metadata.get("candidate_id", "")).strip(),
            "answer_text": str(metadata.get("answer_text", "")).strip(),
            "selected_strategy": str(metadata.get("selected_strategy", "")).strip(),
            "selected_verifier": str(metadata.get("selected_verifier", "")).strip(),
            "candidate_score": min(1.0, max(0.0, float(metadata.get("candidate_score", 0.0) or 0.0))),
            "verified": bool(metadata.get("verified", False)),
            "degraded_reason": str(metadata.get("degraded_reason", "")).strip(),
            "candidate_count": candidate_count,
            "supporting_evidence_ids": tuple(
                str(item) for item in metadata.get("supporting_evidence_ids", ()) if str(item).strip()
            ),
        }

    def _tool_verification_hint(self, handoff: ReasonerCriticHandoff) -> str:
        arithmetic_answer = evaluate_arithmetic_question(handoff.plan.question)
        if arithmetic_answer is not None:
            return f"arithmetic={arithmetic_answer}"
        python_expression_answer = evaluate_python_expression_question(handoff.plan.question)
        if python_expression_answer is not None:
            return f"python_expression={python_expression_answer}"
        python_code_answer = evaluate_python_code_question(handoff.plan.question)
        if python_code_answer is not None:
            return f"python_code_execution={python_code_answer}"
        python_unit_test_answer = evaluate_python_unit_test_question(handoff.plan.question)
        if python_unit_test_answer is not None:
            return f"python_unit_test={python_unit_test_answer}"
        evidence_count_answer = expected_evidence_count(
            handoff.plan.question,
            len(handoff.evidence.local_results) + len(handoff.evidence.web_results),
        )
        if evidence_count_answer is not None:
            return f"evidence_count={evidence_count_answer}"
        return "none"

    def _derive_repair_actions(
        self,
        *,
        issues: list[str],
        proof_hash_match: bool,
        degraded_reason: str,
        failure_categories: tuple[str, ...],
    ) -> tuple[str, ...]:
        actions: list[str] = []
        if not proof_hash_match:
            actions.append("rerun_reasoner")
        if any("tool verification failed" in issue.lower() for issue in issues):
            actions.append("replace_answer_with_tool_result")
        if "schema" in failure_categories:
            actions.append("rebuild_trace_projection")
        if any(category in failure_categories for category in ("provenance", "macro_signature")):
            actions.append("reload_runtime_subset")
        if "candidate_selection" in failure_categories:
            actions.append("rerun_reasoner")
        if degraded_reason in {"low_evidence_support", "no_candidate_met_verification_threshold", "no_retrieved_evidence"}:
            actions.append("abstain_due_to_low_grounding")
        if not issues and not actions:
            actions.append("preserve_trace")
        return tuple(dict.fromkeys(actions))

    def _select_report_result(
        self,
        *,
        issues: list[str],
        degraded_reason: str,
    ) -> CritiqueResult:
        if not issues:
            return CritiqueResult.VALID
        if degraded_reason and not any("tool verification failed" in issue.lower() for issue in issues):
            return CritiqueResult.DEGRADED
        return CritiqueResult.INVALID

    def _validate_runtime_subset(
        self,
        handoff: ReasonerCriticHandoff,
        runtime_subset: CompressionRuntimeSubset,
    ) -> list[str]:
        trace = handoff.trace
        issues: list[str] = []
        loaded_opcode_names = {opcode.opcode_name for opcode in runtime_subset.opcodes}
        loaded_macro_names = {macro.macro_name for macro in runtime_subset.macros}
        loaded_decoder_names = {decoder.decoder_name for decoder in runtime_subset.decoders}
        missing_opcodes = sorted(set(handoff.required_opcode_names) - loaded_opcode_names)
        if missing_opcodes:
            issues.append(f"Active opcode subset is missing required entries: {', '.join(missing_opcodes)}.")
        missing_macros = sorted(set(handoff.required_macro_names) - loaded_macro_names)
        if missing_macros:
            issues.append(f"Active macro subset is missing required entries: {', '.join(missing_macros)}.")
        missing_decoders = sorted(set(handoff.required_decoder_names) - loaded_decoder_names)
        if missing_decoders:
            issues.append(
                f"Active decoder subset is missing required entries: {', '.join(missing_decoders)}."
            )
        if trace.proof_hash != handoff.proof_hash:
            issues.append("Trace proof hash drifted between reasoner and critic handoff.")
        if trace.symbol_table_refs:
            snapshot = runtime_subset.symbol_table
            if snapshot is None:
                issues.append("Task-scoped symbol table is missing for the current trace.")
            else:
                available_refs = set(snapshot.symbols)
                required_refs = set(trace.symbol_table_refs)
                required_refs.update(
                    argument
                    for step in trace.operation_stream
                    for argument in step.args
                    if argument.startswith("sym_")
                )
                required_refs.update(
                    step.output_ref
                    for step in trace.operation_stream
                    if step.output_ref.startswith("sym_")
                )
                missing_refs = sorted(ref for ref in required_refs if ref not in available_refs)
                if missing_refs:
                    issues.append(
                        f"Task-scoped symbol table is missing required refs: {', '.join(missing_refs)}."
                    )
        return issues
