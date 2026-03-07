"""Render final user-facing answers from verified trace state."""

from __future__ import annotations

from typing import Any

from data_structures import CompressedTrace, CritiqueReport, CritiqueResult, EvidenceBundle
from verification_tools import measure_evidence_support


class TranslationService:
    """Turn machine-readable reasoning state into a concise final answer."""

    def render_answer(
        self,
        *,
        evidence: EvidenceBundle,
        reasoning: CompressedTrace,
        critique: CritiqueReport,
    ) -> str:
        candidate = self._extract_trace_candidate_metadata(reasoning)
        answer_text = str(candidate.get("answer_text", "")).strip()
        support_result = (
            measure_evidence_support(
                answer_text,
                tuple((item.id, item.content) for item in (evidence.local_results + evidence.web_results)),
            )
            if answer_text
            else measure_evidence_support("", ())
        )
        citation_refs = self._resolve_citation_refs(
            evidence=evidence,
            supporting_evidence_ids=tuple(
                str(item) for item in candidate.get("supporting_evidence_ids", ()) if str(item).strip()
            )
            or support_result.supporting_evidence_ids,
        )
        citation_text = ""
        if citation_refs:
            citation_text = f" Sources: {', '.join(citation_refs[:2])}."
        if answer_text and critique.is_valid:
            verification_text = ""
            if critique.verifier_type:
                verification_text = f" Verification: {critique.verifier_type}."
            return f"Verified answer: {self._sentence(answer_text)}{verification_text}{citation_text}".strip()
        if answer_text and critique.result == CritiqueResult.DEGRADED:
            reason = critique.degraded_reason or str(candidate.get("degraded_reason", "")) or "critic_reported_issues"
            return (
                f"Degraded answer: {self._sentence(answer_text)} Reason: {reason}."
                f"{citation_text}"
            ).strip()
        if answer_text:
            reason = critique.issues[0] if critique.issues else "verification_failed"
            return (
                f"Unverified candidate: {self._sentence(answer_text)} Reason: {reason}."
                f"{citation_text}"
            ).strip()
        return (
            f"Collected {len(evidence.local_results)} local evidence item(s) and "
            f"{len(evidence.web_results)} web evidence item(s). "
            f"Critique result: {critique.result.value}."
        )

    def summarize_answer_metadata(
        self,
        *,
        evidence: EvidenceBundle,
        reasoning: CompressedTrace,
    ) -> dict[str, Any]:
        candidate = self._extract_trace_candidate_metadata(reasoning)
        answer_text = str(candidate.get("answer_text", "")).strip()
        if answer_text:
            support_result = measure_evidence_support(
                answer_text,
                tuple((item.id, item.content) for item in (evidence.local_results + evidence.web_results)),
            )
        else:
            support_result = measure_evidence_support("", ())
        supporting_ids = tuple(
            str(item) for item in candidate.get("supporting_evidence_ids", ()) if str(item).strip()
        ) or support_result.supporting_evidence_ids
        return {
            "answer_text": answer_text,
            "supporting_evidence_ids": list(supporting_ids),
            "citation_refs": list(
                self._resolve_citation_refs(evidence=evidence, supporting_evidence_ids=supporting_ids)
            ),
        }

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
            if "candidate_id" not in metadata and context_metadata.get("cid"):
                metadata["candidate_id"] = context_metadata.get("cid")
            if "answer_text" not in metadata and context_metadata.get("ta"):
                metadata["answer_text"] = context_metadata.get("ta")
            if "degraded_reason" not in metadata and context_metadata.get("dr"):
                metadata["degraded_reason"] = context_metadata.get("dr")
            if "supporting_evidence_ids" not in metadata and context_metadata.get("si"):
                metadata["supporting_evidence_ids"] = tuple(context_metadata.get("si", ()))
        return metadata

    def _resolve_citation_refs(
        self,
        *,
        evidence: EvidenceBundle,
        supporting_evidence_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        evidence_index = {
            item.id: (item.source_ref or item.id)
            for item in (evidence.local_results + evidence.web_results)
        }
        refs = [str(evidence_index[evidence_id]) for evidence_id in supporting_evidence_ids if evidence_id in evidence_index]
        return tuple(dict.fromkeys(refs))

    def _sentence(self, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            return stripped
        if stripped.endswith((".", "!", "?")):
            return stripped
        return f"{stripped}."
