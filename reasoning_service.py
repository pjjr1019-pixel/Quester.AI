"""Shared reasoning logic used by ReasonerAgent."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from agent_schema import parse_reasoner_output, reasoner_output_schema
from config import APP_CONFIG, AppConfig
from data_structures import (
    CandidateTrace,
    CanonicalReasoningGraph,
    CompressedTrace,
    CompressionRuntimeSubset,
    ContextFrame,
    DecodeHint,
    EvidenceBundle,
    OperationStep,
    Plan,
    ProvenanceBundle,
    ReasonerCriticHandoff,
    ResearchReasonerHandoff,
    SemanticActivity,
    SemanticAgent,
    SemanticEntity,
    SymbolTableSnapshot,
    _compact_payload,
    _derive_decode_hints,
    _derive_symbol_table_refs,
    _format_reasoner_stub_notes,
    utc_now,
)
from prompts import REASONER_PROMPT
from retrieval import stable_hash
from structured_generation import StructuredGenerationService
from verification_tools import (
    evaluate_arithmetic_question,
    evaluate_python_code_question,
    evaluate_python_expression_question,
    evaluate_python_unit_test_question,
    expected_evidence_count,
    measure_candidate_agreement,
    measure_evidence_support,
    normalize_answer_text,
)

if TYPE_CHECKING:
    from model_manager import ModelManager
    from storage import StorageManager


class ReasoningService:
    """Build a typed trace from a typed handoff with bounded structured fallback."""

    output_contract = "compressed_trace_v1"
    handoff_contract = "research_reasoner_handoff_v1"
    implementation_mode = "deterministic_stub"
    final_text_policy = "post_verification"

    def __init__(
        self,
        model_manager: ModelManager,
        storage: StorageManager | None = None,
        config: AppConfig = APP_CONFIG,
        structured_generation: StructuredGenerationService | None = None,
    ):
        self.model_manager = model_manager
        self.storage = storage
        self.config = config
        self.structured_generation = structured_generation or StructuredGenerationService(
            model_manager=model_manager,
            config=config,
        )
        self._last_runtime_subset: CompressionRuntimeSubset | None = None
        self._last_handoff: ResearchReasonerHandoff | None = None

    @property
    def last_runtime_subset(self) -> CompressionRuntimeSubset | None:
        """Return the last active runtime subset loaded for reasoning, if any."""
        return self._last_runtime_subset

    @property
    def last_handoff(self) -> ResearchReasonerHandoff | None:
        """Return the last typed Researcher -> Reasoner handoff, if any."""
        return self._last_handoff

    async def reason(self, handoff: ResearchReasonerHandoff) -> CompressedTrace:
        """Return a typed reasoning trace from one validated handoff.

        Input:
        - `handoff`: the typed Researcher -> Reasoner boundary payload.

        Output:
        - A `CompressedTrace` normalized against the active runtime subset and
          budget limits.

        Failure behavior:
        - Invalid structured model output gets one repair attempt and then falls
          back to the deterministic trace builder instead of retrying
          indefinitely.
        """
        self._last_handoff = handoff
        runtime_subset = await self._prepare_runtime_subset(handoff)
        self._last_runtime_subset = runtime_subset
        base_trace = self._build_deterministic_trace(handoff, runtime_subset)
        prompt = self._build_structured_prompt(handoff, runtime_subset, base_trace)
        decode_result = await self.structured_generation.decode_json_output(
            prompt=prompt,
            schema=reasoner_output_schema(),
            parser=parse_reasoner_output,
            fallback_factory=lambda _error_message: base_trace,
            max_tokens=self.config.model_tuning.default_max_tokens,
        )
        if decode_result.used_fallback:
            return replace(
                base_trace,
                reasoner_notes=self._fallback_reasoner_notes(
                    base_trace=base_trace,
                    error_message=decode_result.error_message,
                    raw_output=decode_result.raw_text,
                    repaired_output=decode_result.repaired_text,
                ),
            )
        return self._normalize_structured_trace(
            parsed_trace=decode_result.value,
            base_trace=base_trace,
            handoff=handoff,
            runtime_subset=runtime_subset,
            raw_output=decode_result.raw_text,
            repaired_output=decode_result.repaired_text,
            used_repair=decode_result.used_repair,
        )

    def build_critic_handoff(
        self,
        *,
        plan: Plan,
        evidence: EvidenceBundle,
        trace: CompressedTrace,
        budget,
    ) -> ReasonerCriticHandoff:
        """Build the typed Reasoner -> Critic handoff for downstream review."""
        return ReasonerCriticHandoff.from_inputs(
            plan=plan,
            evidence=evidence,
            trace=trace,
            budget=budget,
            final_text_policy=self.final_text_policy,
            implementation_mode=self.implementation_mode,
        )

    async def _prepare_runtime_subset(
        self,
        handoff: ResearchReasonerHandoff,
    ) -> CompressionRuntimeSubset:
        draft_symbol_table = self._build_default_symbol_table(
            plan=handoff.plan,
            evidence=handoff.evidence,
            evidence_handles=handoff.evidence_handles,
        )
        opcode_names = ("lookup", "bind", "emit")
        if handoff.reasoning_mode == "deep":
            opcode_names = (
                "lookup",
                "compare",
                "infer",
                "bind",
                "check",
                "cite",
                "confidence_update",
                "emit",
            )
        return await self._load_runtime_subset(
            task_id=handoff.plan.task_id,
            macro_names=(),
            opcode_names=opcode_names,
            decoder_names=("verified_answer",),
            draft_symbol_table=draft_symbol_table,
        )

    def _build_structured_prompt(
        self,
        handoff: ResearchReasonerHandoff,
        runtime_subset: CompressionRuntimeSubset,
        base_trace: CompressedTrace,
    ) -> str:
        evidence = handoff.evidence
        evidence_lines = [
            f"- {item.id}: {item.content[:160]}"
            for item in (evidence.local_results + evidence.web_results)[: handoff.budget.retrieval_top_k]
        ]
        symbol_refs = ()
        if runtime_subset.symbol_table is not None:
            symbol_refs = tuple(runtime_subset.symbol_table.symbols)
        return (
            f"{REASONER_PROMPT}\n"
            f"Question: {handoff.plan.question}\n"
            f"EvidenceHandles: {list(handoff.evidence_handles)}\n"
            f"ReasoningMode: {handoff.reasoning_mode}\n"
            f"ReasonerPasses: {handoff.budget.reasoner_passes}\n"
            f"MacroDepth: {handoff.budget.macro_depth}\n"
            f"AllowedOpcodes: {[opcode.opcode_name for opcode in runtime_subset.opcodes]}\n"
            f"AllowedDecoders: {[decoder.decoder_name for decoder in runtime_subset.decoders]}\n"
            f"AvailableSymbolRefs: {list(symbol_refs)}\n"
            f"BaseTokens: {list(base_trace.tokens)}\n"
            f"BaseAnswer: {self._extract_trace_answer(base_trace)}\n"
            f"EvidenceSummary:\n" + "\n".join(evidence_lines)
        )

    def _normalize_structured_trace(
        self,
        *,
        parsed_trace: CompressedTrace,
        base_trace: CompressedTrace,
        handoff: ResearchReasonerHandoff,
        runtime_subset: CompressionRuntimeSubset,
        raw_output: str,
        repaired_output: str | None,
        used_repair: bool,
    ) -> CompressedTrace:
        available_refs = set(base_trace.symbol_table_refs)
        if runtime_subset.symbol_table is not None:
            available_refs.update(runtime_subset.symbol_table.symbols)
        context_frame_ids = {frame.frame_id for frame in base_trace.context_frames}
        allowed_opcodes = {opcode.opcode_name for opcode in runtime_subset.opcodes}
        allowed_decoders = {decoder.decoder_name for decoder in runtime_subset.decoders}
        allowed_macros = {macro.macro_name for macro in runtime_subset.macros}

        tokens = parsed_trace.tokens if self._tokens_fit_budget(parsed_trace.tokens, handoff) else base_trace.tokens
        expanded_preview = (
            parsed_trace.expanded_preview
            if self._expanded_preview_fits_budget(parsed_trace.expanded_preview, handoff)
            else base_trace.expanded_preview
        )
        confidence = round(min(1.0, max(0.0, parsed_trace.confidence)), 2)
        operation_stream = (
            parsed_trace.operation_stream
            if self._operation_stream_is_valid(
                parsed_trace.operation_stream,
                base_trace=base_trace,
                available_refs=available_refs,
                context_frame_ids=context_frame_ids,
                allowed_opcodes=allowed_opcodes,
                evidence_handles=handoff.evidence_handles,
            )
            else base_trace.operation_stream
        )
        decode_hints = (
            parsed_trace.decode_hints
            if self._decode_hints_are_valid(parsed_trace.decode_hints, allowed_decoders)
            else base_trace.decode_hints
        )
        macros_used = tuple(macro for macro in parsed_trace.macros_used if macro in allowed_macros)
        if parsed_trace.macros_used and not macros_used:
            macros_used = base_trace.macros_used
        symbol_table_refs = _derive_symbol_table_refs(
            builder="reasoner_stub_v1",
            operation_stream=operation_stream,
        ) or base_trace.symbol_table_refs
        proof_hash = self._compute_reasoner_proof_hash(
            task_id=handoff.plan.task_id,
            tokens=tokens,
            operation_stream=operation_stream,
            evidence_handles=handoff.evidence_handles,
        )
        payload = {
            "task_id": handoff.plan.task_id,
            "tokens": list(tokens),
            "expanded_preview": list(expanded_preview),
            "macros_used": list(macros_used),
            "confidence": confidence,
            "reasoner_notes": self._structured_reasoner_notes(
                base_trace=base_trace,
                raw_output=raw_output,
                repaired_output=repaired_output,
                used_repair=used_repair,
            ),
            "ir_version": "1",
            "canonical_graph_builder": "reasoner_stub_v1",
            "operation_stream": [step.to_dict() for step in operation_stream],
            "symbol_table_refs": list(symbol_table_refs),
            "evidence_handles": list(handoff.evidence_handles),
            "context_frames": [frame.to_dict() for frame in base_trace.context_frames],
            "candidate_traces": [candidate_trace.to_dict() for candidate_trace in base_trace.candidate_traces],
            "proof_hash": proof_hash,
            "decode_hints": [
                hint.to_dict()
                for hint in (
                    decode_hints
                    or _derive_decode_hints(
                        builder="reasoner_stub_v1",
                        operation_stream=operation_stream,
                    )
                )
            ],
            "created_at": base_trace.created_at.isoformat(),
        }
        return CompressedTrace.from_dict(payload)

    def _build_deterministic_trace(
        self,
        handoff: ResearchReasonerHandoff,
        runtime_subset: CompressionRuntimeSubset,
    ) -> CompressedTrace:
        plan = handoff.plan
        evidence = handoff.evidence
        budget = handoff.budget
        evidence_items = evidence.local_results + evidence.web_results
        evidence_count = len(evidence_items)
        review_depth = max(1, min(evidence_count, budget.macro_depth))
        candidates = self._build_answer_candidates(handoff)
        candidate_traces = self._build_candidate_traces(
            handoff=handoff,
            candidates=candidates,
            evidence_count=evidence_count,
        )
        candidates = self._merge_candidate_trace_scores(
            candidates=candidates,
            candidate_traces=candidate_traces,
        )
        selected_candidate = self._select_answer_candidate(candidates, reasoning_mode=handoff.reasoning_mode)
        selected_candidate_id = str(selected_candidate.get("candidate_id", "cand_selected"))
        selected_answer = str(selected_candidate["answer_text"])
        selected_strategy = str(selected_candidate["strategy"])
        selected_verifier = str(selected_candidate.get("verifier_type", ""))
        selected_score = round(float(selected_candidate.get("total_score", 0.0)), 3)
        selected_verified = bool(selected_candidate.get("verified", False))
        degraded_reason = str(selected_candidate.get("degraded_reason", ""))
        supporting_evidence_ids = tuple(
            str(item) for item in selected_candidate.get("supporting_evidence_ids", ()) if str(item).strip()
        )
        candidate_count = len(candidates)
        candidate_summary = self._candidate_summary_metadata(candidates)

        chain: list[str] = ["@read_question", "@extract_constraints", "@match_local_evidence"]
        for evidence_index in range(1, review_depth + 1):
            chain.append(f"@review_evidence_{evidence_index}")
        for pass_index in range(1, budget.reasoner_passes + 1):
            chain.extend((f"@reason_pass_{pass_index}", f"@synthesize_pass_{pass_index}"))
            if pass_index < budget.reasoner_passes:
                chain.append(f"@refine_answer_{pass_index}")
        if handoff.reasoning_mode == "deep" and candidate_count > 1:
            chain.extend(("@compare_candidates", "@select_verified_candidate"))
        chain.append("@compose_answer")

        expanded_preview = [
            "Read question and constraints",
            f"Review {review_depth} evidence item(s)",
        ]
        for pass_index in range(1, budget.reasoner_passes + 1):
            expanded_preview.append(f"Reasoning pass {pass_index} of {budget.reasoner_passes}")
        if handoff.reasoning_mode == "deep":
            expanded_preview.append(f"Evaluate {candidate_count} candidate answer(s)")
            expanded_preview.append("Select best verified candidate")

        confidence = min(
            0.95,
            0.55 + (0.05 * budget.reasoner_passes) + (0.02 * min(evidence_count, budget.retrieval_top_k)),
        )
        if selected_verifier in {
            "tool.python_ast_arithmetic",
            "tool.python_expression",
            "tool.python_code_execution",
            "tool.python_unit_test",
        }:
            confidence = min(0.99, confidence + 0.08)
        elif selected_verified:
            confidence = min(0.98, confidence + 0.05)
        elif handoff.reasoning_mode == "deep":
            confidence = min(0.97, confidence + 0.04)

        evidence_handles = handoff.evidence_handles
        trace_created_at = utc_now()
        question_entity_id = "q"
        evidence_set_entity_id = "es"
        binding_entity_id = "b0"
        answer_entity_id = "a"
        retrieve_activity_id = "ac0"
        bind_activity_id = "ac1"
        emit_activity_id = "ac2"
        context_frame_id = "cf0"
        question_symbol_ref = "sym_question"
        answer_symbol_ref = "sym_answer"
        evidence_set_symbol_ref = "sym_evidence_set"

        entity_ids: list[str] = [question_entity_id]
        entities: list[SemanticEntity] = [
            SemanticEntity(
                entity_id=question_entity_id,
                entity_type="question",
                value=question_symbol_ref,
                confidence=1.0,
                created_at=trace_created_at,
            )
        ]
        symbol_table_refs: list[str] = [question_symbol_ref]
        evidence_entity_ids: list[str] = []
        for index, evidence_item in enumerate(evidence_items, start=1):
            evidence_symbol_ref = f"sym_evidence_{index}"
            symbol_table_refs.append(evidence_symbol_ref)
            evidence_entity_id = f"ev{index}"
            evidence_entity_ids.append(evidence_entity_id)
            entities.append(
                SemanticEntity(
                    entity_id=evidence_entity_id,
                    entity_type="evidence_item",
                    value=evidence_item.id,
                    evidence_handles=(evidence_item.id,),
                    attributes={"symbol_ref": evidence_symbol_ref},
                    created_at=trace_created_at,
                )
            )
            entity_ids.append(evidence_entity_id)
        entities.append(
            SemanticEntity(
                entity_id=evidence_set_entity_id,
                entity_type="evidence_set",
                value=evidence_set_symbol_ref,
                evidence_handles=evidence_handles,
                attributes={"count": len(evidence_handles)},
                created_at=trace_created_at,
            )
        )
        entity_ids.append(evidence_set_entity_id)
        entities.append(
            SemanticEntity(
                entity_id=binding_entity_id,
                entity_type="intermediate_binding",
                value=answer_symbol_ref,
                evidence_handles=evidence_handles,
                attributes=_compact_payload(
                    {
                        "symbol_refs": list(symbol_table_refs),
                        "candidate_count": candidate_count,
                        "selected_strategy": selected_strategy,
                        "selected_verifier": selected_verifier,
                        "candidate_score": selected_score,
                        "verified": selected_verified,
                    },
                    drop_empty=("selected_strategy", "selected_verifier"),
                ),
                created_at=trace_created_at,
            )
        )
        entity_ids.append(binding_entity_id)
        entities.append(
            SemanticEntity(
                entity_id=answer_entity_id,
                entity_type="answer_fragment",
                value=selected_answer,
                evidence_handles=evidence_handles,
                confidence=round(confidence, 2),
                attributes=_compact_payload(
                    {
                        "symbol_ref": answer_symbol_ref,
                        "strategy": selected_strategy,
                        "verifier": selected_verifier,
                        "candidate_score": selected_score,
                        "verified": selected_verified,
                        "degraded_reason": degraded_reason,
                        "candidate_count": candidate_count,
                    },
                    drop_empty=("strategy", "verifier", "degraded_reason"),
                ),
                created_at=trace_created_at,
            )
        )
        symbol_table_refs.append(evidence_set_symbol_ref)
        symbol_table_refs.append(answer_symbol_ref)
        entity_ids.append(answer_entity_id)

        model_snapshot = self.model_manager.health_snapshot()
        semantic_agent = SemanticAgent(
            agent_id="ag0",
            component="reasoner",
            backend=model_snapshot.generation_backend,
            role="foreground_reasoning",
            metadata={"builder": "reasoner_stub_v1"},
            created_at=trace_created_at,
        )
        activities = (
            SemanticActivity(
                activity_id=retrieve_activity_id,
                activity_type="retrieve",
                input_entity_ids=(question_entity_id,),
                output_entity_ids=tuple((*evidence_entity_ids, evidence_set_entity_id)),
                agent_id=semantic_agent.agent_id,
                evidence_handles=evidence_handles,
                created_at=trace_created_at,
            ),
            SemanticActivity(
                activity_id=bind_activity_id,
                activity_type="bind",
                input_entity_ids=(question_entity_id, evidence_set_entity_id),
                output_entity_ids=(binding_entity_id,),
                agent_id=semantic_agent.agent_id,
                evidence_handles=evidence_handles,
                created_at=trace_created_at,
            ),
            SemanticActivity(
                activity_id=emit_activity_id,
                activity_type="emit",
                input_entity_ids=(binding_entity_id,),
                output_entity_ids=(answer_entity_id,),
                agent_id=semantic_agent.agent_id,
                evidence_handles=evidence_handles,
                created_at=trace_created_at,
            ),
        )
        provenance_bundle = ProvenanceBundle(
            bundle_id="pb0",
            entity_ids=tuple(entity_ids),
            activity_ids=tuple(activity.activity_id for activity in activities),
            agent_ids=(semantic_agent.agent_id,),
            created_at=trace_created_at,
        )
        context_frame = ContextFrame(
            frame_id=context_frame_id,
            scope="task",
            confidence=round(confidence, 2),
            provenance_bundle_id=provenance_bundle.bundle_id,
            assumptions=(),
            metadata={
                "mb": model_snapshot.generation_backend,
                "ec": evidence_count,
                "rd": review_depth,
                "rp": budget.reasoner_passes,
                "rm": handoff.reasoning_mode,
                "cid": selected_candidate_id,
                "cc": candidate_count,
                "ta": selected_answer,
                "sa": selected_strategy,
                "sv": selected_verifier,
                "ss": selected_score,
                "vv": selected_verified,
                "dr": degraded_reason,
                "si": list(supporting_evidence_ids),
                "candidate_summary": candidate_summary,
                "op": ",".join(opcode.opcode_name for opcode in runtime_subset.opcodes),
                "mc": ",".join(macro.macro_name for macro in runtime_subset.macros),
                "dc": ",".join(decoder.decoder_name for decoder in runtime_subset.decoders),
            },
            created_at=trace_created_at,
        )

        bind_args = tuple(symbol_table_refs[:-1])
        operation_steps: list[OperationStep] = [
            OperationStep(
                op_id="o0",
                opcode="lookup",
                args=(question_symbol_ref,),
                output_ref=evidence_set_symbol_ref,
                context_frame_id=context_frame.frame_id,
                evidence_handles=evidence_handles,
                metadata={"source_token": "@match_local_evidence"},
            )
        ]
        if handoff.reasoning_mode == "deep":
            operation_steps.extend(
                (
                    OperationStep(
                        op_id="o1",
                        opcode="compare",
                        args=(question_symbol_ref, evidence_set_symbol_ref),
                        context_frame_id=context_frame.frame_id,
                        evidence_handles=evidence_handles,
                        metadata={
                            "source_token": "@compare_candidates",
                            "candidate_count": candidate_count,
                        },
                    ),
                    OperationStep(
                        op_id="o2",
                        opcode="infer",
                        args=(question_symbol_ref, evidence_set_symbol_ref),
                        context_frame_id=context_frame.frame_id,
                        evidence_handles=evidence_handles,
                        metadata={
                            "source_token": "@select_verified_candidate",
                            "selected_strategy": selected_strategy,
                            "selected_verifier": selected_verifier,
                            "candidate_score": selected_score,
                            "verified": selected_verified,
                            "candidate_id": selected_candidate_id,
                        },
                    ),
                    OperationStep(
                        op_id="o3",
                        opcode="bind",
                        args=bind_args,
                        output_ref=answer_symbol_ref,
                        context_frame_id=context_frame.frame_id,
                        evidence_handles=evidence_handles,
                        metadata={
                            "source_token": "@compose_answer",
                            "selected_strategy": selected_strategy,
                            "selected_verifier": selected_verifier,
                            "candidate_score": selected_score,
                            "verified": selected_verified,
                            "candidate_id": selected_candidate_id,
                        },
                    ),
                    OperationStep(
                        op_id="o4",
                        opcode="check",
                        args=(answer_symbol_ref,),
                        context_frame_id=context_frame.frame_id,
                        evidence_handles=evidence_handles,
                        metadata={
                            "source_token": "@select_verified_candidate",
                            "tool_check": selected_verifier or selected_strategy,
                            "candidate_count": candidate_count,
                            "candidate_id": selected_candidate_id,
                        },
                    ),
                    OperationStep(
                        op_id="o5",
                        opcode="cite",
                        args=(answer_symbol_ref, evidence_set_symbol_ref),
                        context_frame_id=context_frame.frame_id,
                        evidence_handles=evidence_handles,
                        metadata={
                            "source_token": "@compose_answer",
                            "evidence_count": evidence_count,
                        },
                    ),
                    OperationStep(
                        op_id="o6",
                        opcode="confidence_update",
                        args=(answer_symbol_ref,),
                        context_frame_id=context_frame.frame_id,
                        evidence_handles=evidence_handles,
                        metadata={
                            "source_token": "@compose_answer",
                            "confidence": round(confidence, 2),
                        },
                    ),
                    OperationStep(
                        op_id="o7",
                        opcode="emit",
                        args=(answer_symbol_ref,),
                        context_frame_id=context_frame.frame_id,
                        evidence_handles=evidence_handles,
                        metadata={
                            "source_token": "@compose_answer",
                            "answer_text": selected_answer,
                            "selected_strategy": selected_strategy,
                            "selected_verifier": selected_verifier,
                            "candidate_score": selected_score,
                            "verified": selected_verified,
                            "degraded_reason": degraded_reason,
                            "candidate_count": candidate_count,
                            "supporting_evidence_ids": list(supporting_evidence_ids),
                            "candidate_id": selected_candidate_id,
                        },
                    ),
                )
            )
        else:
            operation_steps.extend(
                (
                    OperationStep(
                        op_id="o1",
                        opcode="bind",
                        args=bind_args,
                        output_ref=answer_symbol_ref,
                        context_frame_id=context_frame.frame_id,
                        evidence_handles=evidence_handles,
                        metadata={
                            "source_token": "@compose_answer",
                            "selected_strategy": selected_strategy,
                            "selected_verifier": selected_verifier,
                            "candidate_score": selected_score,
                            "verified": selected_verified,
                            "candidate_id": selected_candidate_id,
                        },
                    ),
                    OperationStep(
                        op_id="o2",
                        opcode="emit",
                        args=(answer_symbol_ref,),
                        context_frame_id=context_frame.frame_id,
                        evidence_handles=evidence_handles,
                        metadata={
                            "source_token": "@compose_answer",
                            "answer_text": selected_answer,
                            "selected_strategy": selected_strategy,
                            "selected_verifier": selected_verifier,
                            "candidate_score": selected_score,
                            "verified": selected_verified,
                            "degraded_reason": degraded_reason,
                            "candidate_count": candidate_count,
                            "supporting_evidence_ids": list(supporting_evidence_ids),
                            "candidate_id": selected_candidate_id,
                        },
                    ),
                )
            )
        operation_stream = tuple(operation_steps)
        decode_hints = (
            DecodeHint(
                hint_id="d0",
                template="verified_answer",
                entity_ids=(answer_entity_id,),
                metadata={
                    "answer_text": selected_answer,
                    "selected_strategy": selected_strategy,
                    "selected_verifier": selected_verifier,
                    "candidate_score": selected_score,
                    "verified": selected_verified,
                    "degraded_reason": degraded_reason,
                    "candidate_count": candidate_count,
                    "supporting_evidence_ids": list(supporting_evidence_ids),
                    "candidate_id": selected_candidate_id,
                },
            ),
        )
        proof_hash = self._compute_reasoner_proof_hash(
            task_id=plan.task_id,
            tokens=tuple(chain),
            operation_stream=operation_stream,
            evidence_handles=evidence_handles,
        )
        return CompressedTrace(
            task_id=plan.task_id,
            tokens=tuple(chain),
            expanded_preview=tuple(expanded_preview),
            macros_used=(),
            confidence=round(confidence, 2),
            reasoner_notes="\n".join(
                (
                    _format_reasoner_stub_notes(
                        model_backend=model_snapshot.generation_backend,
                        evidence_count=evidence_count,
                        review_depth=review_depth,
                        reasoner_passes=budget.reasoner_passes,
                        loaded_opcodes=tuple(opcode.opcode_name for opcode in runtime_subset.opcodes),
                        loaded_macros=tuple(macro.macro_name for macro in runtime_subset.macros),
                        loaded_decoders=tuple(decoder.decoder_name for decoder in runtime_subset.decoders),
                        symbol_table_refs=tuple(symbol_table_refs),
                    ),
                    f"reasoning_mode={handoff.reasoning_mode}",
                    f"candidate_count={candidate_count}",
                    f"selected_strategy={selected_strategy}",
                    f"selected_verifier={selected_verifier}",
                    f"selected_score={selected_score}",
                    f"selected_verified={'yes' if selected_verified else 'no'}",
                    f"selected_answer={selected_answer}",
                    f"selected_candidate_id={selected_candidate_id}",
                    f"candidate_summary={candidate_summary}",
                    f"candidate_trace_count={len(candidate_traces)}",
                )
            ),
            ir_version="1",
            canonical_graph=CanonicalReasoningGraph(
                entities=tuple(entities),
                activities=activities,
                agents=(semantic_agent,),
                bundles=(provenance_bundle,),
                created_at=trace_created_at,
            ),
            canonical_graph_builder="reasoner_stub_v1",
            operation_stream=operation_stream,
            symbol_table_refs=tuple(symbol_table_refs),
            evidence_handles=evidence_handles,
            context_frames=(context_frame,),
            candidate_traces=candidate_traces if handoff.reasoning_mode == "deep" else candidate_traces[:1],
            proof_hash=proof_hash,
            decode_hints=decode_hints,
            created_at=trace_created_at,
        )

    def _tokens_fit_budget(
        self,
        tokens: tuple[str, ...],
        handoff: ResearchReasonerHandoff,
    ) -> bool:
        if not tokens or tokens[0] != "@read_question" or "@compose_answer" not in tokens:
            return False
        expected_passes = {
            f"@reason_pass_{pass_index}" for pass_index in range(1, handoff.budget.reasoner_passes + 1)
        }
        return expected_passes.issubset(set(tokens))

    def _expanded_preview_fits_budget(
        self,
        expanded_preview: tuple[str, ...],
        handoff: ResearchReasonerHandoff,
    ) -> bool:
        if len(expanded_preview) < 2:
            return False
        preview_reasoning_passes = sum(1 for item in expanded_preview if item.startswith("Reasoning pass "))
        return preview_reasoning_passes == handoff.budget.reasoner_passes

    def _operation_stream_is_valid(
        self,
        operation_stream: tuple[OperationStep, ...],
        *,
        base_trace: CompressedTrace,
        available_refs: set[str],
        context_frame_ids: set[str],
        allowed_opcodes: set[str],
        evidence_handles: tuple[str, ...],
    ) -> bool:
        if not operation_stream or not allowed_opcodes:
            return False
        if tuple(step.opcode for step in operation_stream) != tuple(
            step.opcode for step in base_trace.operation_stream
        ):
            return False
        for step in operation_stream:
            if step.opcode not in allowed_opcodes:
                return False
            if step.context_frame_id and step.context_frame_id not in context_frame_ids:
                return False
            if any(
                argument.startswith("sym_") and argument not in available_refs for argument in step.args
            ):
                return False
            if step.output_ref.startswith("sym_") and step.output_ref not in available_refs:
                return False
            if any(handle not in evidence_handles for handle in step.evidence_handles):
                return False
        return True

    def _decode_hints_are_valid(
        self,
        decode_hints: tuple[DecodeHint, ...],
        allowed_decoders: set[str],
    ) -> bool:
        if not decode_hints or not allowed_decoders:
            return False
        return all(hint.template in allowed_decoders for hint in decode_hints)

    def _structured_reasoner_notes(
        self,
        *,
        base_trace: CompressedTrace,
        raw_output: str,
        repaired_output: str | None,
        used_repair: bool,
    ) -> str:
        note_parts = [
            base_trace.reasoner_notes,
            "reasoner_output_mode=structured_json",
            f"output_contract={self.output_contract}",
            f"implementation_mode={self.implementation_mode}",
            f"used_repair={'yes' if used_repair else 'no'}",
            "used_fallback=no",
            f"raw_output={raw_output}",
        ]
        if repaired_output is not None:
            note_parts.append(f"repaired_output={repaired_output}")
        return "\n".join(part for part in note_parts if part)

    def _fallback_reasoner_notes(
        self,
        *,
        base_trace: CompressedTrace,
        error_message: str | None,
        raw_output: str,
        repaired_output: str | None,
    ) -> str:
        note_parts = [
            base_trace.reasoner_notes,
            "reasoner_output_mode=deterministic_fallback",
            f"output_contract={self.output_contract}",
            f"implementation_mode={self.implementation_mode}",
            f"parse_error={error_message or 'unknown'}",
            f"raw_output={raw_output}",
        ]
        if repaired_output is not None:
            note_parts.append(f"repaired_output={repaired_output}")
        return "\n".join(part for part in note_parts if part)

    def _compute_reasoner_proof_hash(
        self,
        *,
        task_id: str,
        tokens: tuple[str, ...],
        operation_stream: tuple[OperationStep, ...],
        evidence_handles: tuple[str, ...],
    ) -> str:
        proof_payload = {
            "task_id": task_id,
            "tokens": list(tokens),
            "operation_stream": [step.to_dict() for step in operation_stream],
            "evidence_handles": list(evidence_handles),
        }
        return stable_hash(json.dumps(proof_payload, sort_keys=True, separators=(",", ":")))

    def _build_answer_candidates(
        self,
        handoff: ResearchReasonerHandoff,
    ) -> tuple[dict[str, Any], ...]:
        evidence_items = handoff.evidence.local_results + handoff.evidence.web_results
        candidates: list[dict[str, Any]] = []
        arithmetic_answer = evaluate_arithmetic_question(handoff.plan.question)
        if arithmetic_answer is not None:
            candidates.append(
                {
                    "answer_text": arithmetic_answer,
                    "strategy": "tool_arithmetic",
                    "verifier_type": "tool.python_ast_arithmetic",
                    "expected_answer": arithmetic_answer,
                    "base_score": 0.98,
                }
            )
        python_answer = evaluate_python_expression_question(handoff.plan.question)
        if python_answer is not None and normalize_answer_text(python_answer) != normalize_answer_text(
            arithmetic_answer or ""
        ):
            candidates.append(
                {
                    "answer_text": python_answer,
                    "strategy": "tool_python_expression",
                    "verifier_type": "tool.python_expression",
                    "expected_answer": python_answer,
                    "base_score": 0.97,
                }
            )
        python_code_answer = evaluate_python_code_question(handoff.plan.question)
        if python_code_answer is not None and normalize_answer_text(python_code_answer) not in {
            normalize_answer_text(arithmetic_answer or ""),
            normalize_answer_text(python_answer or ""),
        }:
            candidates.append(
                {
                    "answer_text": python_code_answer,
                    "strategy": "tool_python_code_execution",
                    "verifier_type": "tool.python_code_execution",
                    "expected_answer": python_code_answer,
                    "base_score": 0.975,
                }
            )
        python_unit_test_answer = evaluate_python_unit_test_question(handoff.plan.question)
        if python_unit_test_answer is not None:
            candidates.append(
                {
                    "answer_text": python_unit_test_answer,
                    "strategy": "tool_python_unit_test",
                    "verifier_type": "tool.python_unit_test",
                    "expected_answer": python_unit_test_answer,
                    "base_score": 0.972,
                }
            )
        evidence_count_answer = expected_evidence_count(
            handoff.plan.question,
            len(evidence_items),
        )
        if evidence_count_answer is not None:
            candidates.append(
                {
                    "answer_text": evidence_count_answer,
                    "strategy": "tool_evidence_count",
                    "verifier_type": "tool.evidence_count",
                    "expected_answer": evidence_count_answer,
                    "base_score": 0.96,
                }
            )
        if evidence_items:
            primary_item = evidence_items[0]
            primary_summary = self._summarize_evidence_items((primary_item,))
            candidates.append(
                {
                    "answer_text": primary_summary,
                    "strategy": "top_evidence",
                    "verifier_type": "tool.evidence_grounding",
                    "base_score": 0.62 if primary_item.source_type.value == "local" else 0.58,
                    "supporting_evidence_ids": (primary_item.id,),
                }
            )
            aggregate_summary = self._summarize_evidence_items(
                evidence_items[: min(2, max(1, handoff.budget.reasoner_passes))]
            )
            if aggregate_summary != primary_summary:
                candidates.append(
                    {
                        "answer_text": aggregate_summary,
                        "strategy": "evidence_aggregate",
                        "verifier_type": "tool.evidence_grounding",
                        "base_score": 0.72 if handoff.reasoning_mode == "deep" else 0.62,
                        "supporting_evidence_ids": tuple(item.id for item in evidence_items[:2]),
                    }
                )
        candidates.append(
            self._abstain_candidate(
                "no_retrieved_evidence" if not evidence_items else "no_candidate_met_verification_threshold"
            )
        )
        deduped_by_answer: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            answer_text = str(candidate["answer_text"]).strip()
            normalized_answer = normalize_answer_text(answer_text)
            if not normalized_answer:
                continue
            existing = deduped_by_answer.get(normalized_answer)
            if existing is None or float(candidate.get("base_score", 0.0)) > float(
                existing.get("base_score", 0.0)
            ):
                deduped_by_answer[normalized_answer] = candidate
        scored_candidates = self._score_answer_candidates(
            handoff=handoff,
            candidates=tuple(deduped_by_answer.values()),
        )
        if not scored_candidates:
            return (self._abstain_candidate("no_candidates_built"),)
        return tuple(scored_candidates[: self._candidate_limit(handoff)])

    def _select_answer_candidate(
        self,
        candidates: tuple[dict[str, Any], ...],
        *,
        reasoning_mode: str,
    ) -> dict[str, Any]:
        if not candidates:
            return self._abstain_candidate("no_candidates_built")
        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: (
                bool(candidate.get("verified", False)),
                float(candidate.get("total_score", 0.0)),
                float(candidate.get("evidence_support_score", 0.0)),
                str(candidate.get("strategy", "")).startswith("tool_"),
            ),
            reverse=True,
        )
        verified_candidates = [candidate for candidate in ranked_candidates if candidate.get("verified", False)]
        if verified_candidates:
            return verified_candidates[0]
        if reasoning_mode == "deep":
            for candidate in ranked_candidates:
                if str(candidate.get("strategy", "")) == "abstain":
                    return candidate
            return self._abstain_candidate("no_candidate_met_verification_threshold")
        return ranked_candidates[0]

    def _score_answer_candidates(
        self,
        *,
        handoff: ResearchReasonerHandoff,
        candidates: tuple[dict[str, Any], ...],
    ) -> tuple[dict[str, Any], ...]:
        evidence_items = tuple(
            (item.id, item.content) for item in (handoff.evidence.local_results + handoff.evidence.web_results)
        )
        scored_candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates, start=1):
            answer_text = str(candidate.get("answer_text", "")).strip()
            if not answer_text:
                continue
            strategy = str(candidate.get("strategy", "")).strip() or "candidate"
            verifier_type = str(candidate.get("verifier_type", "")).strip() or "heuristic"
            base_score = round(min(1.0, max(0.0, float(candidate.get("base_score", 0.0)))), 3)
            peer_answers = [
                str(peer.get("answer_text", "")).strip()
                for peer_index, peer in enumerate(candidates)
                if peer_index != index - 1 and str(peer.get("strategy", "")) != "abstain"
            ]
            agreement_score = measure_candidate_agreement(answer_text, peer_answers)
            support_result = measure_evidence_support(answer_text, evidence_items)
            supporting_evidence_ids = tuple(
                str(item) for item in candidate.get("supporting_evidence_ids", ()) if str(item).strip()
            )
            if support_result.supporting_evidence_ids:
                supporting_evidence_ids = support_result.supporting_evidence_ids
            evidence_support_score = support_result.score
            degraded_reason = str(candidate.get("degraded_reason", "")).strip()
            verified = False
            if strategy == "abstain":
                agreement_score = 0.0
                evidence_support_score = 0.0
                total_score = round(max(base_score, 0.3 if not evidence_items else 0.24), 3)
                degraded_reason = degraded_reason or (
                    "no_retrieved_evidence"
                    if not evidence_items
                    else "no_candidate_met_verification_threshold"
                )
            elif verifier_type.startswith("tool."):
                total_score = round(min(0.99, max(base_score, 0.9) + (0.03 * evidence_support_score)), 3)
                verified = True
            else:
                total_score = round(
                    min(0.97, (base_score * 0.4) + (evidence_support_score * 0.45) + (agreement_score * 0.15)),
                    3,
                )
                verified = evidence_support_score >= 0.85 or total_score >= 0.78
                if not verified:
                    degraded_reason = degraded_reason or "no_candidate_met_verification_threshold"
            if (
                handoff.reasoning_mode == "deep"
                and verifier_type == "tool.evidence_grounding"
                and self._question_prefers_multi_evidence_answer(handoff.plan.question)
            ):
                if len(supporting_evidence_ids) > 1:
                    total_score = round(min(0.97, total_score + 0.08), 3)
                elif strategy == "top_evidence":
                    total_score = round(max(0.0, total_score - 0.05), 3)
            scored_candidates.append(
                {
                    **candidate,
                    "candidate_id": f"cand_{index}",
                    "strategy": strategy,
                    "verifier_type": verifier_type,
                    "agreement_score": agreement_score,
                    "evidence_support_score": evidence_support_score,
                    "supporting_evidence_ids": supporting_evidence_ids,
                    "verified": verified,
                    "total_score": total_score,
                    "degraded_reason": degraded_reason,
                }
            )
        scored_candidates.sort(
            key=lambda candidate: (
                bool(candidate.get("verified", False)),
                float(candidate.get("total_score", 0.0)),
                float(candidate.get("evidence_support_score", 0.0)),
                str(candidate.get("strategy", "")).startswith("tool_"),
            ),
            reverse=True,
        )
        return tuple(scored_candidates)

    def _candidate_limit(self, handoff: ResearchReasonerHandoff) -> int:
        if handoff.reasoning_mode == "fast":
            return 1
        return max(2, min(5, handoff.budget.reasoner_passes + handoff.budget.critic_passes - 1))

    def _build_candidate_traces(
        self,
        *,
        handoff: ResearchReasonerHandoff,
        candidates: tuple[dict[str, Any], ...],
        evidence_count: int,
    ) -> tuple[CandidateTrace, ...]:
        candidate_traces: list[CandidateTrace] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", "")).strip()
            answer_text = str(candidate.get("answer_text", "")).strip()
            if not candidate_id or not answer_text:
                continue
            evidence_handles = tuple(
                str(item) for item in candidate.get("supporting_evidence_ids", ()) if str(item).strip()
            ) or handoff.evidence_handles
            tokens = (
                "@candidate_prepare",
                "@candidate_reason",
                "@candidate_verify",
                "@candidate_emit",
            )
            preview = (
                "Prepare candidate trace",
                f"Reason over {max(1, evidence_count)} evidence item(s)",
                "Verify candidate answer",
                "Emit candidate answer",
            )
            operation_stream = (
                OperationStep(
                    op_id=f"{candidate_id}_lookup",
                    opcode="lookup",
                    args=("sym_question",),
                    output_ref="sym_evidence_set",
                    context_frame_id="cf_candidate",
                    evidence_handles=evidence_handles,
                    metadata={"candidate_id": candidate_id},
                ),
                OperationStep(
                    op_id=f"{candidate_id}_bind",
                    opcode="bind",
                    args=("sym_question", "sym_evidence_set"),
                    output_ref="sym_answer",
                    context_frame_id="cf_candidate",
                    evidence_handles=evidence_handles,
                    metadata={
                        "candidate_id": candidate_id,
                        "selected_strategy": str(candidate.get("strategy", "")),
                        "selected_verifier": str(candidate.get("verifier_type", "")),
                        "candidate_score": round(float(candidate.get("total_score", 0.0)), 3),
                        "verified": bool(candidate.get("verified", False)),
                    },
                ),
                OperationStep(
                    op_id=f"{candidate_id}_check",
                    opcode="check",
                    args=("sym_answer",),
                    context_frame_id="cf_candidate",
                    evidence_handles=evidence_handles,
                    metadata={
                        "candidate_id": candidate_id,
                        "tool_check": str(candidate.get("verifier_type", "") or candidate.get("strategy", "")),
                    },
                ),
                OperationStep(
                    op_id=f"{candidate_id}_emit",
                    opcode="emit",
                    args=("sym_answer",),
                    context_frame_id="cf_candidate",
                    evidence_handles=evidence_handles,
                    metadata={
                        "candidate_id": candidate_id,
                        "answer_text": answer_text,
                        "selected_strategy": str(candidate.get("strategy", "")),
                        "selected_verifier": str(candidate.get("verifier_type", "")),
                        "candidate_score": round(float(candidate.get("total_score", 0.0)), 3),
                        "verified": bool(candidate.get("verified", False)),
                        "degraded_reason": str(candidate.get("degraded_reason", "")),
                        "supporting_evidence_ids": list(evidence_handles),
                    },
                ),
            )
            proof_hash = self._compute_reasoner_proof_hash(
                task_id=candidate_id,
                tokens=tokens,
                operation_stream=operation_stream,
                evidence_handles=evidence_handles,
            )
            candidate_traces.append(
                CandidateTrace(
                    candidate_id=candidate_id,
                    answer_text=answer_text,
                    strategy=str(candidate.get("strategy", "")),
                    verifier_type=str(candidate.get("verifier_type", "")),
                    verified=bool(candidate.get("verified", False)),
                    total_score=round(float(candidate.get("total_score", 0.0)), 3),
                    agreement_score=round(float(candidate.get("agreement_score", 0.0)), 3),
                    evidence_support_score=round(float(candidate.get("evidence_support_score", 0.0)), 3),
                    proof_hash_stability=1.0,
                    degraded_reason=str(candidate.get("degraded_reason", "")),
                    supporting_evidence_ids=evidence_handles,
                    tokens=tokens,
                    expanded_preview=preview,
                    operation_stream=operation_stream,
                    decode_hints=(
                        DecodeHint(
                            hint_id=f"{candidate_id}_hint",
                            template="verified_answer",
                            entity_ids=("a",),
                            metadata={
                                "candidate_id": candidate_id,
                                "answer_text": answer_text,
                                "selected_strategy": str(candidate.get("strategy", "")),
                                "selected_verifier": str(candidate.get("verifier_type", "")),
                                "candidate_score": round(float(candidate.get("total_score", 0.0)), 3),
                                "verified": bool(candidate.get("verified", False)),
                                "degraded_reason": str(candidate.get("degraded_reason", "")),
                                "supporting_evidence_ids": list(evidence_handles),
                            },
                        ),
                    ),
                    proof_hash=proof_hash,
                )
            )
        return tuple(candidate_traces)

    def _merge_candidate_trace_scores(
        self,
        *,
        candidates: tuple[dict[str, Any], ...],
        candidate_traces: tuple[CandidateTrace, ...],
    ) -> tuple[dict[str, Any], ...]:
        trace_index = {candidate_trace.candidate_id: candidate_trace for candidate_trace in candidate_traces}
        merged: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", "")).strip()
            candidate_trace = trace_index.get(candidate_id)
            proof_hash_stability = candidate_trace.proof_hash_stability if candidate_trace is not None else 0.0
            merged.append(
                {
                    **candidate,
                    "proof_hash_stability": proof_hash_stability,
                    "total_score": round(
                        min(
                            0.99,
                            max(0.0, float(candidate.get("total_score", 0.0)) + (0.04 * proof_hash_stability)),
                        ),
                        3,
                    ),
                }
            )
        merged.sort(
            key=lambda candidate: (
                bool(candidate.get("verified", False)),
                float(candidate.get("total_score", 0.0)),
                float(candidate.get("evidence_support_score", 0.0)),
                str(candidate.get("strategy", "")).startswith("tool_"),
            ),
            reverse=True,
        )
        return tuple(merged)

    def _candidate_summary_metadata(
        self,
        candidates: tuple[dict[str, Any], ...],
    ) -> list[dict[str, Any]]:
        return [
            {
                "candidate_id": str(candidate.get("candidate_id", f"cand_{index}")),
                "strategy": str(candidate.get("strategy", "")),
                "verifier_type": str(candidate.get("verifier_type", "")),
                "verified": bool(candidate.get("verified", False)),
                "total_score": round(float(candidate.get("total_score", 0.0)), 3),
                "proof_hash_stability": round(float(candidate.get("proof_hash_stability", 1.0)), 3),
            }
            for index, candidate in enumerate(candidates[:3], start=1)
        ]

    def _abstain_candidate(self, reason: str) -> dict[str, Any]:
        return {
            "candidate_id": "cand_abstain",
            "answer_text": "Insufficient evidence to produce a verified answer.",
            "strategy": "abstain",
            "verifier_type": "none",
            "base_score": 0.24,
            "verified": False,
            "total_score": 0.24,
            "degraded_reason": reason,
        }

    def _summarize_evidence_items(
        self,
        evidence_items,
    ) -> str:
        snippets = [self._shorten_text(item.content) for item in evidence_items if item.content.strip()]
        if not snippets:
            return "Evidence exists but does not yet contain a concise answer candidate."
        if len(snippets) == 1:
            return snippets[0]
        return " ".join(snippets[:2])

    def _shorten_text(self, text: str, *, limit: int = 160) -> str:
        sentence = " ".join(text.strip().split())
        if len(sentence) <= limit:
            return sentence
        return sentence[: limit - 3].rstrip() + "..."

    def _question_prefers_multi_evidence_answer(self, question: str) -> bool:
        lowered = question.lower()
        return any(
            token in lowered
            for token in (
                " both ",
                " two ",
                " compare ",
                " difference ",
                " versus ",
                " pair ",
                " layers",
            )
        )

    def _extract_trace_answer(self, trace: CompressedTrace) -> str:
        for step in reversed(trace.operation_stream):
            answer_text = str(step.metadata.get("answer_text", "")).strip()
            if step.opcode == "emit" and answer_text:
                return answer_text
        for hint in trace.decode_hints:
            answer_text = str(hint.metadata.get("answer_text", "")).strip()
            if answer_text:
                return answer_text
        if trace.context_frames:
            answer_text = str(trace.context_frames[0].metadata.get("ta", "")).strip()
            if answer_text:
                return answer_text
        return ""

    def _build_default_symbol_table(
        self,
        *,
        plan: Plan,
        evidence: EvidenceBundle,
        evidence_handles: tuple[str, ...],
    ) -> dict[str, str]:
        symbols: dict[str, str] = {
            "sym_question": f"question://{plan.task_id}",
            "sym_evidence_set": "|".join(evidence_handles) if evidence_handles else "none",
            "sym_answer": f"answer://{plan.task_id}",
        }
        for index, item in enumerate(evidence.local_results + evidence.web_results, start=1):
            symbols[f"sym_evidence_{index}"] = item.id
        return symbols

    async def _load_runtime_subset(
        self,
        *,
        task_id: str,
        macro_names: tuple[str, ...],
        opcode_names: tuple[str, ...],
        decoder_names: tuple[str, ...],
        draft_symbol_table: dict[str, str],
    ) -> CompressionRuntimeSubset:
        if self.storage is None:
            snapshot = SymbolTableSnapshot(
                task_id=task_id,
                symbols=draft_symbol_table,
                metadata={"owner": "reasoner", "scope": "task", "generated": True},
            )
            return CompressionRuntimeSubset(
                task_id=task_id,
                macros=(),
                opcodes=(),
                decoders=(),
                symbol_table=snapshot,
                proof_hashes=(),
            )

        subset = await self.storage.load_active_compression_runtime(
            task_id,
            macro_names=macro_names,
            opcode_names=opcode_names,
            decoder_names=decoder_names,
        )
        snapshot = subset.symbol_table
        if snapshot is None or not set(draft_symbol_table).issubset(snapshot.symbols):
            merged_symbols = dict(snapshot.symbols) if snapshot is not None else {}
            merged_symbols.update(draft_symbol_table)
            snapshot = SymbolTableSnapshot(
                task_id=task_id,
                symbols=merged_symbols,
                metadata={"owner": "reasoner", "scope": "task", "generated": True},
            )
            await self.storage.record_symbol_table_snapshot(snapshot)
            subset = CompressionRuntimeSubset(
                task_id=subset.task_id,
                macros=subset.macros,
                opcodes=subset.opcodes,
                decoders=subset.decoders,
                symbol_table=snapshot,
                proof_hashes=subset.proof_hashes,
                created_at=subset.created_at,
            )
        return subset
