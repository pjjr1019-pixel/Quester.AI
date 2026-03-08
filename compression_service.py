"""Shared proposal-building logic used by CompressorAgent."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Callable

from acceptance_thresholds import PHASE12_ACCEPTANCE_THRESHOLDS
from agent_schema import compressor_output_schema, parse_compressor_output
from config import APP_CONFIG, AppConfig
from data_structures import CompressedTrace, Macro, MacroProposal, ReasoningLog
from macro_engine import MacroEngine
from model_manager import ModelManager
from prompts import COMPRESSOR_PROMPT
from retrieval import stable_hash
from structured_generation import StructuredGenerationService


class CompressionService:
    """Suggest macro candidates while keeping CompressorAgent thin."""

    output_contract = "macro_proposal_list_v1"
    implementation_mode = "deterministic_stub"
    MAX_FOREGROUND_PROPOSALS = PHASE12_ACCEPTANCE_THRESHOLDS.compression.max_foreground_proposals

    def __init__(
        self,
        model_manager: ModelManager,
        config: AppConfig = APP_CONFIG,
        structured_generation: StructuredGenerationService | None = None,
        macro_engine_factory: Callable[[], MacroEngine] | None = None,
    ):
        self.model_manager = model_manager
        self.config = config
        self.structured_generation = structured_generation or StructuredGenerationService(
            model_manager=model_manager,
            config=config,
        )
        self._macro_engine_factory: Callable[[], MacroEngine] = macro_engine_factory or MacroEngine

    async def propose(
        self,
        trace: CompressedTrace,
        logs: list[ReasoningLog],
    ) -> list[MacroProposal]:
        """Return bounded macro proposals, preferring structured output but preserving deterministic fallback."""
        chain = trace.expanded_preview or trace.tokens
        engine = self._macro_engine_factory()
        fallback_proposals = self._build_deterministic_proposals(
            trace=trace,
            chain=tuple(str(item) for item in chain),
            logs=tuple(logs),
            engine=engine,
        )
        prompt = self._build_structured_prompt(
            trace=trace,
            chain=tuple(str(item) for item in chain),
            logs=tuple(logs),
            fallback_proposals=tuple(fallback_proposals),
        )
        decode_result = await self.structured_generation.decode_json_output(
            prompt=prompt,
            schema=compressor_output_schema(),
            parser=parse_compressor_output,
            fallback_factory=lambda _error_message: tuple(fallback_proposals),
            max_tokens=self.config.model_tuning.default_max_tokens,
        )
        proposals = (
            fallback_proposals
            if decode_result.used_fallback
            else self._normalize_structured_proposals(
                proposals=decode_result.value,
                engine=engine,
                fallback_proposals=fallback_proposals,
            )
        )
        return self._apply_prior_scoring(proposals)

    def _build_deterministic_proposals(
        self,
        *,
        trace: CompressedTrace,
        chain: tuple[str, ...],
        logs: tuple[ReasoningLog, ...],
        engine: MacroEngine,
    ) -> list[MacroProposal]:
        suggestions: list[MacroProposal] = []
        seen_signatures: set[tuple[str, ...]] = set()
        context_metadata = dict(trace.context_frames[0].metadata) if trace.context_frames else {}
        verification_bonus = 0.1 if context_metadata.get("vv", False) else 0.0
        for span in (3, 2):
            motifs = Counter(
                tuple(chain[index : index + span])
                for index in range(0, max(0, len(chain) - span + 1))
            )
            for motif, count in motifs.items():
                if count <= 1 or motif in seen_signatures:
                    continue
                seen_signatures.add(motif)
                self._append_validated_proposal(
                    suggestions=suggestions,
                    engine=engine,
                    trace=trace,
                    prefix="motif",
                    expansion=motif,
                    opcode_pattern=tuple(engine._opcode_for_step(step) for step in motif),
                    semantic_kind="motif_macro",
                    reason=f"Repeated {span}-step motif observed {count} times in expanded trace.",
                    example=" | ".join(motif),
                    simulation_score=min(1.0, 0.35 + ((count * span) / 8.0) + verification_bonus),
                )
        opcode_chain = tuple(step.opcode for step in trace.operation_stream if step.opcode)
        for span in (3, 2):
            opcode_motifs = Counter(
                tuple(opcode_chain[index : index + span])
                for index in range(0, max(0, len(opcode_chain) - span + 1))
            )
            for motif, count in opcode_motifs.items():
                signature = tuple(f"opcode:{opcode}" for opcode in motif)
                if count <= 1 or signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                self._append_validated_proposal(
                    suggestions=suggestions,
                    engine=engine,
                    trace=trace,
                    prefix="opcode",
                    expansion=signature,
                    opcode_pattern=motif,
                    semantic_kind="opcode_motif_macro",
                    reason=f"Repeated {span}-opcode runtime motif observed {count} times in the operation stream.",
                    example=" | ".join(signature),
                    simulation_score=min(1.0, 0.32 + ((count * span) / 9.0) + verification_bonus),
                )
        cross_log_counts: Counter[tuple[str, ...]] = Counter()
        for log in logs:
            compressed_chain = tuple(str(item) for item in log.compressed_chain if str(item).strip())
            for span in (3, 2):
                cross_log_counts.update(
                    {
                        tuple(compressed_chain[index : index + span])
                        for index in range(0, max(0, len(compressed_chain) - span + 1))
                    }
                )
        for motif, count in cross_log_counts.items():
            if len(motif) < 2 or count <= 1 or motif in seen_signatures:
                continue
            seen_signatures.add(motif)
            self._append_validated_proposal(
                suggestions=suggestions,
                engine=engine,
                trace=trace,
                prefix="crosslog",
                expansion=motif,
                opcode_pattern=tuple(engine._opcode_for_step(step) for step in motif),
                semantic_kind="cross_log_motif_macro",
                reason=f"Motif observed across {count} reasoning logs, suggesting reusable compression.",
                example=" | ".join(motif),
                simulation_score=min(1.0, 0.3 + (count / 6.0) + verification_bonus),
            )
        self._append_candidate_subproof_proposals(
            suggestions=suggestions,
            trace=trace,
            engine=engine,
            seen_signatures=seen_signatures,
            verification_bonus=verification_bonus,
        )
        self._append_graph_path_proposals(
            suggestions=suggestions,
            trace=trace,
            engine=engine,
            seen_signatures=seen_signatures,
            verification_bonus=verification_bonus,
        )
        self._append_symbol_bundle_proposals(
            suggestions=suggestions,
            trace=trace,
            engine=engine,
            seen_signatures=seen_signatures,
            verification_bonus=verification_bonus,
        )
        suggestions.sort(key=lambda proposal: proposal.simulation_score, reverse=True)
        if suggestions:
            return suggestions[: self.MAX_FOREGROUND_PROPOSALS]

        token_counts = Counter(trace.tokens)
        for token, count in token_counts.items():
            if count <= 1:
                continue
            self._append_validated_proposal(
                suggestions=suggestions,
                engine=engine,
                trace=trace,
                prefix="token",
                expansion=(token,),
                opcode_pattern=(engine._opcode_for_step(token),),
                semantic_kind="token_macro",
                reason=f"Token '{token}' repeated {count} times in compressed trace.",
                example=token,
                simulation_score=min(1.0, 0.25 + (count / 4.0) + verification_bonus),
            )
        return suggestions[: self.MAX_FOREGROUND_PROPOSALS]

    def _append_validated_proposal(
        self,
        *,
        suggestions: list[MacroProposal],
        engine: MacroEngine,
        trace: CompressedTrace,
        prefix: str,
        expansion: tuple[str, ...],
        opcode_pattern: tuple[str, ...],
        semantic_kind: str,
        reason: str,
        example: str,
        simulation_score: float,
    ) -> None:
        sanitized = stable_hash("|".join(expansion))[:10]
        proposal = MacroProposal(
            proposal_id=f"{trace.task_id}:{prefix}:{sanitized}",
            macro=Macro(
                macro_name=f"{prefix}_{sanitized}",
                expansion=expansion,
                version=1,
                opcode_pattern=opcode_pattern,
                invariants=(
                    "deterministic_round_trip",
                    "provenance_preserving",
                    "uncertainty_preserving",
                ),
                semantic_kind=semantic_kind,
            ),
            reason=reason,
            examples=(example,),
            simulation_score=min(1.0, max(0.0, simulation_score)),
            approved=False,
        )
        validated = engine.validate_macro_proposal(proposal)
        if any(existing.proposal_id == validated.proposal_id for existing in suggestions):
            return
        suggestions.append(validated)

    def _append_candidate_subproof_proposals(
        self,
        *,
        suggestions: list[MacroProposal],
        trace: CompressedTrace,
        engine: MacroEngine,
        seen_signatures: set[tuple[str, ...]],
        verification_bonus: float,
    ) -> None:
        candidate_traces = tuple(candidate for candidate in trace.candidate_traces if candidate.operation_stream)
        if len(candidate_traces) < 2:
            return
        opcode_chains = [tuple(step.opcode for step in candidate.operation_stream) for candidate in candidate_traces]
        shared_prefix = self._shared_prefix(opcode_chains)
        if len(shared_prefix) < 2:
            return
        signature = tuple(f"subproof:{opcode}" for opcode in shared_prefix)
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        self._append_validated_proposal(
            suggestions=suggestions,
            engine=engine,
            trace=trace,
            prefix="subproof",
            expansion=signature,
            opcode_pattern=shared_prefix,
            semantic_kind="candidate_subproof_macro",
            reason=(
                f"Deep candidates share a {len(shared_prefix)}-step proof prefix, suggesting reusable "
                "subproof compression."
            ),
            example=" | ".join(signature),
            simulation_score=min(1.0, 0.5 + (0.03 * len(candidate_traces)) + verification_bonus),
        )

    def _append_graph_path_proposals(
        self,
        *,
        suggestions: list[MacroProposal],
        trace: CompressedTrace,
        engine: MacroEngine,
        seen_signatures: set[tuple[str, ...]],
        verification_bonus: float,
    ) -> None:
        graph = trace.canonical_graph
        if graph is None or len(graph.activities) < 2:
            return
        activity_path = tuple(activity.activity_type for activity in graph.activities)
        signature = tuple(f"graph:{activity}" for activity in activity_path)
        if signature in seen_signatures:
            return
        seen_signatures.add(signature)
        self._append_validated_proposal(
            suggestions=suggestions,
            engine=engine,
            trace=trace,
            prefix="graph",
            expansion=signature,
            opcode_pattern=tuple(step.opcode for step in trace.operation_stream[: len(activity_path)]),
            semantic_kind="graph_path_macro",
            reason="Canonical proof graph exposes a reusable task-to-evidence-to-answer path.",
            example=" | ".join(signature),
            simulation_score=min(1.0, 0.46 + (0.02 * len(activity_path)) + verification_bonus),
        )

    def _append_symbol_bundle_proposals(
        self,
        *,
        suggestions: list[MacroProposal],
        trace: CompressedTrace,
        engine: MacroEngine,
        seen_signatures: set[tuple[str, ...]],
        verification_bonus: float,
    ) -> None:
        if len(trace.symbol_table_refs) < 3:
            return
        bundle = tuple(f"symbol:{symbol_ref}" for symbol_ref in trace.symbol_table_refs[:3])
        if bundle in seen_signatures:
            return
        seen_signatures.add(bundle)
        bind_like_opcodes = tuple(
            step.opcode for step in trace.operation_stream if step.opcode in {"lookup", "bind", "cite"}
        )[: max(1, len(bundle) - 1)]
        self._append_validated_proposal(
            suggestions=suggestions,
            engine=engine,
            trace=trace,
            prefix="symbol",
            expansion=bundle,
            opcode_pattern=bind_like_opcodes,
            semantic_kind="symbol_bundle_macro",
            reason="Symbol-table references repeatedly travel together through the verified proof path.",
            example=" | ".join(bundle),
            simulation_score=min(1.0, 0.42 + verification_bonus),
        )

    def _shared_prefix(self, opcode_chains: list[tuple[str, ...]]) -> tuple[str, ...]:
        if not opcode_chains:
            return ()
        prefix: list[str] = list(opcode_chains[0])
        for chain in opcode_chains[1:]:
            max_prefix = min(len(prefix), len(chain))
            next_prefix: list[str] = []
            for index in range(max_prefix):
                if prefix[index] != chain[index]:
                    break
                next_prefix.append(prefix[index])
            prefix = next_prefix
            if not prefix:
                break
        return tuple(prefix)

    def _build_structured_prompt(
        self,
        *,
        trace: CompressedTrace,
        chain: tuple[str, ...],
        logs: tuple[ReasoningLog, ...],
        fallback_proposals: tuple[MacroProposal, ...],
    ) -> str:
        log_summaries = [
            {
                "task_id": log.task_id,
                "compressed_chain": list(log.compressed_chain[:8]),
                "macros_used": list(log.macros_used),
            }
            for log in logs[:2]
        ]
        seed_examples = [proposal.to_dict() for proposal in fallback_proposals[:3]]
        context_metadata = dict(trace.context_frames[0].metadata) if trace.context_frames else {}
        return (
            f"{COMPRESSOR_PROMPT}\n"
            f"TaskId: {trace.task_id}\n"
            f"TraceTokens: {list(trace.tokens)}\n"
            f"ExpandedChain: {list(chain)}\n"
            f"TraceOpcodes: {[step.opcode for step in trace.operation_stream]}\n"
            f"CandidateSummaryHint: {context_metadata.get('candidate_summary', [])}\n"
            f"ExistingMacros: {list(trace.macros_used)}\n"
            f"RecentLogs: {log_summaries}\n"
            f"DeterministicSeedExamples: {seed_examples}\n"
            "Prefer motif-style proposals over one-token aliases when both are plausible.\n"
            f"OutputContract: {self.output_contract}"
        )

    def _normalize_structured_proposals(
        self,
        *,
        proposals: tuple[MacroProposal, ...],
        engine: MacroEngine,
        fallback_proposals: list[MacroProposal],
    ) -> list[MacroProposal]:
        normalized: list[MacroProposal] = []
        seen_ids: set[str] = set()
        for proposal in proposals:
            if proposal.proposal_id in seen_ids:
                continue
            seen_ids.add(proposal.proposal_id)
            normalized.append(engine.validate_macro_proposal(proposal))
        return normalized or fallback_proposals

    def _apply_prior_scoring(self, proposals: list[MacroProposal]) -> list[MacroProposal]:
        lookup_cache = getattr(self.model_manager, "lookup_cache", None)
        rescored: list[MacroProposal] = []
        for proposal in proposals:
            cached_summary = (
                lookup_cache("compression_artifacts", proposal.proof_fingerprint or proposal.proposal_id)
                if callable(lookup_cache)
                else None
            )
            score_bonus = 0.0
            if isinstance(cached_summary, dict):
                try:
                    effectiveness_score = float(cached_summary.get("effectiveness_score", 0.0))
                    validation_pass_rate = float(cached_summary.get("validation_pass_rate", 0.0))
                except (TypeError, ValueError):
                    effectiveness_score = 0.0
                    validation_pass_rate = 0.0
                score_bonus = min(
                    0.12,
                    max(0.0, effectiveness_score) * 0.08 + max(0.0, validation_pass_rate) * 0.04,
                )
            rescored.append(
                replace(
                    proposal,
                    simulation_score=min(1.0, proposal.simulation_score + score_bonus),
                )
            )
        rescored.sort(key=lambda item: item.simulation_score, reverse=True)
        self._publish_compression_artifact_summaries(rescored)
        return rescored[: self.MAX_FOREGROUND_PROPOSALS]

    def _publish_compression_artifact_summaries(self, proposals: list[MacroProposal]) -> None:
        warm_cache = getattr(self.model_manager, "warm_cache", None)
        if not callable(warm_cache):
            return
        for proposal in proposals[: self.MAX_FOREGROUND_PROPOSALS]:
            warm_cache(
                "compression_artifacts",
                proposal.proof_fingerprint or proposal.proposal_id,
                {
                    "proposal_id": proposal.proposal_id,
                    "proof_fingerprint": proposal.proof_fingerprint,
                    "macro_name": proposal.macro.macro_name,
                    "effectiveness_score": round(proposal.simulation_score, 3),
                    "validation_pass_rate": round(float(proposal.validation_passed), 3),
                    "compression_gain": round(proposal.simulation_score, 3),
                    "validation_state": "validated" if proposal.validation_passed else "blocked",
                    "blocked_reason": (
                        proposal.validation_issues[0] if proposal.validation_issues else "validation_failed"
                    ) if not proposal.validation_passed else "",
                    "accepted": bool(proposal.validation_passed),
                    "evidence_basis": "deterministic_analysis",
                    "origin_component": "compression_service",
                    "source": "compression_service",
                },
            )
