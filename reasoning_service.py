"""Shared reasoning logic used by ReasonerAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import APP_CONFIG, AppConfig
from data_structures import (
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
    _format_reasoner_stub_notes,
    utc_now,
)
from prompts import REASONER_PROMPT
from retrieval import stable_hash

if TYPE_CHECKING:
    from model_manager import ModelManager
    from storage import StorageManager


class ReasoningService:
    """Build a deterministic typed trace from a typed handoff."""

    output_contract = "compressed_trace_v1"
    handoff_contract = "research_reasoner_handoff_v1"
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
        self._last_runtime_subset: CompressionRuntimeSubset | None = None
        self._last_handoff: ResearchReasonerHandoff | None = None

    @property
    def last_runtime_subset(self) -> CompressionRuntimeSubset | None:
        return self._last_runtime_subset

    @property
    def last_handoff(self) -> ResearchReasonerHandoff | None:
        return self._last_handoff

    async def reason(self, handoff: ResearchReasonerHandoff) -> CompressedTrace:
        self._last_handoff = handoff
        plan = handoff.plan
        evidence = handoff.evidence
        budget = handoff.budget
        evidence_count = len(evidence.local_results) + len(evidence.web_results)
        review_depth = max(1, min(evidence_count, budget.macro_depth))
        prompt = (
            f"{REASONER_PROMPT}\n"
            f"Task: {plan.question}\n"
            f"EvidenceCount: {evidence_count}\n"
            f"ReasonerPasses: {budget.reasoner_passes}\n"
            f"MacroDepth: {budget.macro_depth}\n"
            f"OutputContract: {handoff.output_contract}"
        )
        await self.model_manager.generate(prompt)
        chain: list[str] = [
            "@read_question",
            "@extract_constraints",
            "@match_local_evidence",
        ]
        for evidence_index in range(1, review_depth + 1):
            chain.append(f"@review_evidence_{evidence_index}")
        for pass_index in range(1, budget.reasoner_passes + 1):
            chain.extend((f"@reason_pass_{pass_index}", f"@synthesize_pass_{pass_index}"))
            if pass_index < budget.reasoner_passes:
                chain.append(f"@refine_answer_{pass_index}")
        chain.append("@compose_answer")
        expanded_preview = [
            "Read question and constraints",
            f"Review {review_depth} evidence item(s)",
        ]
        for pass_index in range(1, budget.reasoner_passes + 1):
            expanded_preview.append(f"Reasoning pass {pass_index} of {budget.reasoner_passes}")
        confidence = min(
            0.95,
            0.55 + (0.05 * budget.reasoner_passes) + (0.02 * min(evidence_count, budget.retrieval_top_k)),
        )
        evidence_items = evidence.local_results + evidence.web_results
        evidence_handles = handoff.evidence_handles
        draft_symbol_table = self._build_default_symbol_table(
            plan=plan,
            evidence=evidence,
            evidence_handles=evidence_handles,
        )
        runtime_subset = await self._load_runtime_subset(
            task_id=plan.task_id,
            macro_names=(),
            opcode_names=("lookup", "bind", "emit"),
            decoder_names=("verified_answer",),
            draft_symbol_table=draft_symbol_table,
        )
        self._last_runtime_subset = runtime_subset
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
                attributes={"symbol_refs": list(symbol_table_refs)},
                created_at=trace_created_at,
            )
        )
        entity_ids.append(binding_entity_id)
        entities.append(
            SemanticEntity(
                entity_id=answer_entity_id,
                entity_type="answer_fragment",
                value=answer_symbol_ref,
                evidence_handles=evidence_handles,
                confidence=round(confidence, 2),
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
                "op": ",".join(opcode.opcode_name for opcode in runtime_subset.opcodes),
                "mc": ",".join(macro.macro_name for macro in runtime_subset.macros),
                "dc": ",".join(decoder.decoder_name for decoder in runtime_subset.decoders),
            },
            created_at=trace_created_at,
        )
        bind_args = tuple(symbol_table_refs[:-1])
        operation_stream = (
            OperationStep(
                op_id="o0",
                opcode="lookup",
                args=(question_symbol_ref,),
                output_ref=evidence_set_symbol_ref,
                context_frame_id=context_frame.frame_id,
                evidence_handles=evidence_handles,
                metadata={"source_token": "@match_local_evidence"},
            ),
            OperationStep(
                op_id="o1",
                opcode="bind",
                args=bind_args,
                output_ref=answer_symbol_ref,
                context_frame_id=context_frame.frame_id,
                evidence_handles=evidence_handles,
                metadata={"source_token": "@compose_answer"},
            ),
            OperationStep(
                op_id="o2",
                opcode="emit",
                args=(answer_symbol_ref,),
                context_frame_id=context_frame.frame_id,
                evidence_handles=evidence_handles,
                metadata={"source_token": "@compose_answer"},
            ),
        )
        decode_hints = (
            DecodeHint(
                hint_id="d0",
                template="verified_answer",
                entity_ids=(answer_entity_id,),
            ),
        )
        proof_hash = stable_hash(
            "|".join(
                (
                    plan.task_id,
                    ",".join(chain),
                    ",".join(entity_ids),
                    ",".join(step.opcode for step in operation_stream),
                    ",".join(evidence_handles),
                )
            )
        )
        return CompressedTrace(
            task_id=plan.task_id,
            tokens=tuple(chain),
            expanded_preview=tuple(expanded_preview),
            macros_used=(),
            confidence=round(confidence, 2),
            reasoner_notes=_format_reasoner_stub_notes(
                model_backend=model_snapshot.generation_backend,
                evidence_count=evidence_count,
                review_depth=review_depth,
                reasoner_passes=budget.reasoner_passes,
                loaded_opcodes=tuple(opcode.opcode_name for opcode in runtime_subset.opcodes),
                loaded_macros=tuple(macro.macro_name for macro in runtime_subset.macros),
                loaded_decoders=tuple(decoder.decoder_name for decoder in runtime_subset.decoders),
                symbol_table_refs=tuple(symbol_table_refs),
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
            proof_hash=proof_hash,
            decode_hints=decode_hints,
            created_at=trace_created_at,
        )

    def build_critic_handoff(
        self,
        *,
        plan: Plan,
        evidence: EvidenceBundle,
        trace: CompressedTrace,
        budget,
    ) -> ReasonerCriticHandoff:
        return ReasonerCriticHandoff.from_inputs(
            plan=plan,
            evidence=evidence,
            trace=trace,
            budget=budget,
            final_text_policy=self.final_text_policy,
            implementation_mode=self.implementation_mode,
        )

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
