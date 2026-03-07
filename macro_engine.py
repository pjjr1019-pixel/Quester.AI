"""Deterministic IR-backed macro engine with round-trip verification hooks."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from collections.abc import Iterable, Sequence
from typing import Any

from data_structures import (
    CanonicalReasoningGraph,
    CompressedTrace,
    ContextFrame,
    DecodeHint,
    Macro,
    MacroProposal,
    OperationStep,
    ProvenanceBundle,
    SemanticActivity,
    SemanticAgent,
    SemanticEntity,
    utc_now,
)
from retrieval import stable_hash

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_MACRO_TOKEN_RE = re.compile(r"^@?(?P<name>[a-zA-Z0-9_]+)(?:\((?P<args>.*)\))?$")
_CANONICAL_ARG_OPCODES = frozenset({"aggregate", "bind", "cite", "compare", "confidence_update"})


def _normalize_step_token(token: str) -> str:
    cleaned = token.strip()
    if cleaned.startswith("@"):
        cleaned = cleaned[1:]
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned.lower()).strip("_")
    return cleaned or "step"


def _dedupe_preserve_order(values: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _normalize_macro_key(name: str) -> str:
    key = str(name).strip()
    if key.startswith("@"):
        key = key[1:]
    if not key:
        raise ValueError("Macro name must not be empty.")
    return key


def _macro_token(name: str) -> str:
    return f"@{_normalize_macro_key(name)}"


class MacroEngine:
    """Compile macro-expanded token traces into the canonical IR."""

    def __init__(self, *, max_recursion_depth: int = 8) -> None:
        if max_recursion_depth < 1:
            raise ValueError("max_recursion_depth must be positive.")
        self._macros: dict[str, Macro] = {}
        self._max_recursion_depth = max_recursion_depth

    def register_macro(
        self,
        macro: str | Macro,
        expansion: Iterable[str] | None = None,
    ) -> Macro:
        if isinstance(macro, Macro):
            if expansion is not None:
                raise ValueError("expansion must not be provided when registering a Macro object.")
            normalized = replace(macro, macro_name=_normalize_macro_key(macro.macro_name))
        else:
            if expansion is None:
                raise ValueError("expansion must be provided when registering a macro name.")
            normalized = Macro(
                macro_name=_normalize_macro_key(macro),
                expansion=tuple(str(item).strip() for item in expansion),
                version=1,
            )
        validated_macro, issues, _ = self.validate_macro(normalized)
        if issues:
            raise ValueError("; ".join(issues))
        self._macros[validated_macro.macro_name] = validated_macro
        return validated_macro

    def list_macros(self) -> tuple[Macro, ...]:
        return tuple(self._macros[name] for name in sorted(self._macros))

    def validate_macro(self, macro: Macro) -> tuple[Macro, tuple[str, ...], str]:
        normalized = replace(
            macro,
            macro_name=_normalize_macro_key(macro.macro_name),
            expansion=tuple(str(item).strip() for item in macro.expansion if str(item).strip()),
            parameters=tuple(str(item).strip() for item in macro.parameters if str(item).strip()),
            opcode_pattern=tuple(str(item).strip() for item in macro.opcode_pattern if str(item).strip()),
            invariants=tuple(str(item).strip() for item in macro.invariants if str(item).strip()),
            semantic_kind=str(macro.semantic_kind).strip() or "token_macro",
            decoder_template=str(macro.decoder_template).strip(),
        )
        issues: list[str] = []
        if not normalized.expansion:
            issues.append("macro expansion must not be empty")
        placeholders = self._macro_placeholders(normalized.expansion)
        declared = set(normalized.parameters)
        undeclared = sorted(placeholders - declared)
        unused = sorted(declared - placeholders)
        if undeclared:
            issues.append(f"undeclared macro parameters: {', '.join(undeclared)}")
        if unused:
            issues.append(f"unused macro parameters: {', '.join(unused)}")
        self_references = [
            token
            for token in normalized.expansion
            if self._resolve_macro_invocation(token, allow_missing=True) == normalized.macro_name
        ]
        if self_references:
            issues.append("direct self-recursive macro definitions are not allowed")
        if normalized.opcode_pattern and len(normalized.opcode_pattern) != len(normalized.expansion):
            issues.append("opcode_pattern length must match expansion length")
        proof_fingerprint = self.fingerprint_macro(normalized)
        if normalized.proof_fingerprint and normalized.proof_fingerprint != proof_fingerprint:
            issues.append("proof fingerprint does not match canonical macro fingerprint")
        return replace(normalized, proof_fingerprint=proof_fingerprint), tuple(issues), proof_fingerprint

    def validate_macro_proposal(self, proposal: MacroProposal) -> MacroProposal:
        validated_macro, issues, proof_fingerprint = self.validate_macro(proposal.macro)
        issue_list = list(issues)
        required_invariants = {
            "deterministic_round_trip",
            "provenance_preserving",
            "uncertainty_preserving",
        }
        missing_invariants = sorted(required_invariants - set(validated_macro.invariants))
        if missing_invariants:
            issue_list.append(
                f"missing proposal invariants: {', '.join(missing_invariants)}"
            )
        if not validated_macro.opcode_pattern:
            issue_list.append("proposal macro must declare opcode_pattern")
        if not proposal.examples:
            issue_list.append("proposal must include at least one example")
        return replace(
            proposal,
            macro=validated_macro,
            validation_passed=not issue_list,
            validation_issues=tuple(issue_list),
            proof_fingerprint=proof_fingerprint,
            approved=proposal.approved and not issue_list,
        )

    def fingerprint_macro(self, macro: Macro) -> str:
        payload = {
            "macro_name": _normalize_macro_key(macro.macro_name),
            "version": macro.version,
            "parameters": list(macro.parameters),
            "expansion": list(macro.expansion),
            "opcode_pattern": list(macro.opcode_pattern),
            "invariants": list(macro.invariants),
            "semantic_kind": macro.semantic_kind,
            "decoder_template": macro.decoder_template,
        }
        return stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def compress(self, steps: list[str], task_id: str = "macro_engine_task") -> CompressedTrace:
        """Compile token steps into a deterministic IR-backed compressed trace."""
        normalized_steps = [str(step).strip() for step in steps if str(step).strip()]
        if not normalized_steps:
            raise ValueError("steps must not be empty.")

        expanded_steps, expanded_macros = self._expand_tokens(normalized_steps)
        compressed_tokens, compressed_macros = self._compress_expanded_steps(expanded_steps)
        uses_registered_input_macros = any(
            self._resolve_macro_invocation(step, allow_missing=False) is not None
            for step in normalized_steps
        )
        if uses_registered_input_macros and len(normalized_steps) <= len(compressed_tokens):
            final_tokens = tuple(normalized_steps)
            macros_used = _dedupe_preserve_order(
                (
                    *expanded_macros,
                    *(
                        _macro_token(self._resolve_macro_invocation(step, allow_missing=False) or "")
                        for step in normalized_steps
                        if self._resolve_macro_invocation(step, allow_missing=False) is not None
                    ),
                )
            )
        else:
            final_tokens = compressed_tokens
            macros_used = _dedupe_preserve_order((*expanded_macros, *compressed_macros))
        trace_created_at = utc_now()
        operation_stream = self._build_operation_stream(expanded_steps)
        canonical_graph = self._build_canonical_graph(
            task_id,
            tokens=final_tokens,
            macros_used=macros_used,
            operation_stream=operation_stream,
            created_at=trace_created_at,
        )
        context_frames = self._build_context_frames(
            task_id,
            len(operation_stream),
            created_at=trace_created_at,
        )
        decode_hints = self._build_decode_hints(operation_stream)
        proof_hash = self._compute_proof_hash(
            canonical_graph=canonical_graph,
            operation_stream=operation_stream,
            context_frames=context_frames,
            decode_hints=decode_hints,
        )

        return CompressedTrace(
            task_id=task_id,
            tokens=final_tokens,
            expanded_preview=tuple(expanded_steps),
            macros_used=macros_used,
            confidence=1.0,
            reasoner_notes=(
                f"MacroEngine deterministic IR compression.\n"
                f"expanded_steps={len(expanded_steps)}\n"
                f"macros_used={len(macros_used)}"
            ),
            ir_version="1",
            canonical_graph=canonical_graph,
            canonical_graph_builder="macro_engine_v1",
            operation_stream=operation_stream,
            symbol_table_refs=tuple(step.output_ref for step in operation_stream if step.output_ref),
            evidence_handles=(),
            context_frames=context_frames,
            proof_hash=proof_hash,
            decode_hints=decode_hints,
            created_at=trace_created_at,
        )

    def expand(self, tokens: list[str]) -> list[str]:
        """Expand known macros recursively and preserve unknown tokens."""
        expanded, _ = self._expand_tokens(tuple(tokens))
        return list(expanded)

    def expand_to_graph(self, trace: CompressedTrace) -> CanonicalReasoningGraph:
        """Replay the operation stream into a deterministic canonical graph."""
        if trace.operation_stream:
            return self._build_canonical_graph(
                trace.task_id,
                tokens=trace.tokens,
                macros_used=trace.macros_used,
                operation_stream=self._canonicalize_operation_stream(trace.operation_stream),
                created_at=trace.created_at,
            )
        expanded_steps, _ = self._expand_tokens(trace.tokens)
        operation_stream = self._build_operation_stream(expanded_steps)
        return self._build_canonical_graph(
            trace.task_id,
            tokens=trace.tokens,
            macros_used=trace.macros_used,
            operation_stream=operation_stream,
            created_at=trace.created_at,
        )

    def verify_round_trip(self, trace: CompressedTrace) -> bool:
        """Validate that replay, normalization, and recompression are stable."""
        try:
            replayed_graph = self.expand_to_graph(trace)
            replayed_ops = (
                self._canonicalize_operation_stream(trace.operation_stream)
                if trace.operation_stream
                else self._build_operation_stream(self.expand(list(trace.tokens)))
            )
            replayed_contexts = (
                self._canonicalize_context_frames(trace.context_frames)
                if trace.context_frames
                else self._build_context_frames(trace.task_id, len(replayed_ops))
            )
            replayed_hints = (
                self._canonicalize_decode_hints(trace.decode_hints)
                if trace.decode_hints
                else self._build_decode_hints(replayed_ops)
            )
            replayed_hash = self._compute_proof_hash(
                canonical_graph=replayed_graph,
                operation_stream=replayed_ops,
                context_frames=replayed_contexts,
                decode_hints=replayed_hints,
            )
            if trace.operation_stream and trace.operation_stream != replayed_ops:
                return False
            if trace.context_frames and trace.context_frames != replayed_contexts:
                return False
            if trace.decode_hints and trace.decode_hints != replayed_hints:
                return False
            if trace.proof_hash and trace.proof_hash != replayed_hash:
                return False
            if trace.canonical_graph is not None and self._normalize_graph(trace.canonical_graph) != self._normalize_graph(replayed_graph):
                return False

            recompressed = self.compress(self.expand(list(trace.tokens)), task_id=trace.task_id)
            if recompressed.proof_hash != replayed_hash:
                return False
            return self._normalize_graph(self.expand_to_graph(recompressed)) == self._normalize_graph(replayed_graph)
        except ValueError:
            return False

    def _expand_tokens(
        self,
        tokens: Sequence[str],
        *,
        depth: int = 0,
        visited: tuple[str, ...] = (),
    ) -> tuple[list[str], tuple[str, ...]]:
        if depth > self._max_recursion_depth:
            raise ValueError(
                f"Macro expansion exceeded max recursion depth {self._max_recursion_depth}."
            )
        expanded: list[str] = []
        macros_used: list[str] = []
        for raw_token in tokens:
            token = str(raw_token).strip()
            if not token:
                raise ValueError("Token values must not be empty.")
            invocation = self._parse_macro_invocation(token)
            if invocation is None:
                expanded.append(token)
                continue
            macro = self._macros.get(invocation["macro_name"])
            if macro is None or not macro.is_active:
                expanded.append(token)
                continue
            if invocation["macro_name"] in visited:
                cycle = " -> ".join(_macro_token(name) for name in (*visited, invocation["macro_name"]))
                raise ValueError(f"Macro recursion detected: {cycle}")
            macros_used.append(_macro_token(invocation["macro_name"]))
            substituted = [
                self._apply_macro_arguments(item, invocation["arguments"])
                for item in macro.expansion
            ]
            nested_expanded, nested_used = self._expand_tokens(
                substituted,
                depth=depth + 1,
                visited=(*visited, invocation["macro_name"]),
            )
            expanded.extend(nested_expanded)
            macros_used.extend(nested_used)
        return expanded, _dedupe_preserve_order(macros_used)

    def _compress_expanded_steps(
        self,
        expanded_steps: Sequence[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not self._macros:
            return tuple(expanded_steps), ()
        compressed: list[str] = []
        macros_used: list[str] = []
        index = 0
        macros = self._compression_macros()
        while index < len(expanded_steps):
            best_macro: Macro | None = None
            best_arguments: dict[str, str] | None = None
            best_span = 0
            for macro in macros:
                arguments = self._match_macro_at(expanded_steps, index, macro)
                if arguments is None:
                    continue
                span = len(macro.expansion)
                if span > best_span:
                    best_macro = macro
                    best_arguments = arguments
                    best_span = span
            if best_macro is None or best_arguments is None:
                compressed.append(expanded_steps[index])
                index += 1
                continue
            compressed.append(self._render_macro_invocation(best_macro, best_arguments))
            macros_used.append(_macro_token(best_macro.macro_name))
            index += best_span
        return tuple(compressed), _dedupe_preserve_order(macros_used)

    def _compression_macros(self) -> tuple[Macro, ...]:
        return tuple(
            sorted(
                (macro for macro in self._macros.values() if macro.is_active),
                key=lambda macro: (-len(macro.expansion), macro.macro_name),
            )
        )

    def _match_macro_at(
        self,
        expanded_steps: Sequence[str],
        start_index: int,
        macro: Macro,
    ) -> dict[str, str] | None:
        if start_index + len(macro.expansion) > len(expanded_steps):
            return None
        collected: dict[str, str] = {}
        for offset, template in enumerate(macro.expansion):
            actual = expanded_steps[start_index + offset]
            match = self._match_macro_template(
                template=template,
                actual=actual,
                parameter_names=macro.parameters,
                collected=collected,
            )
            if match is None:
                return None
            collected = match
            if macro.opcode_pattern:
                actual_opcode = self._opcode_for_step(actual)
                if actual_opcode != macro.opcode_pattern[offset]:
                    return None
        for parameter in macro.parameters:
            if parameter not in collected:
                return None
        return collected

    def _match_macro_template(
        self,
        *,
        template: str,
        actual: str,
        parameter_names: Sequence[str],
        collected: dict[str, str],
    ) -> dict[str, str] | None:
        placeholders = list(_PLACEHOLDER_RE.finditer(template))
        if not placeholders:
            return collected if template == actual else None
        pattern_parts: list[str] = []
        placeholder_order: list[str] = []
        cursor = 0
        for match in placeholders:
            pattern_parts.append(re.escape(template[cursor:match.start()]))
            pattern_parts.append("(.+?)")
            placeholder_order.append(match.group(1))
            cursor = match.end()
        pattern_parts.append(re.escape(template[cursor:]))
        rendered = re.fullmatch("".join(pattern_parts), actual)
        if rendered is None:
            return None
        updated = dict(collected)
        for name, value in zip(placeholder_order, rendered.groups(), strict=False):
            if parameter_names and name not in parameter_names:
                return None
            if name in updated and updated[name] != value:
                return None
            updated[name] = value
        return updated

    def _build_operation_stream(self, expanded_steps: Sequence[str]) -> tuple[OperationStep, ...]:
        operations: list[OperationStep] = []
        for index, step in enumerate(expanded_steps, start=1):
            normalized = _normalize_step_token(step)
            opcode = self._opcode_for_step(step)
            operations.append(
                OperationStep(
                    op_id=f"op_{index:03d}",
                    opcode=opcode,
                    args=self._canonicalize_args(opcode, self._args_for_step(step)),
                    output_ref=f"sym_{index:03d}_{normalized}",
                    context_frame_id="ctx_primary",
                    evidence_handles=(),
                    metadata={
                        "source_token": step,
                    },
                )
            )
        return tuple(operations)

    def _build_canonical_graph(
        self,
        task_id: str,
        operation_stream: Sequence[OperationStep],
        *,
        tokens: Sequence[str] = (),
        macros_used: Sequence[str] = (),
        created_at,
    ) -> CanonicalReasoningGraph:
        agent = SemanticAgent(
            agent_id="agent_macro_engine",
            component="macro_engine",
            backend="deterministic_ir",
            role="compression_runtime",
            metadata={"task_id": task_id},
            created_at=created_at,
        )
        entities: list[SemanticEntity] = [
            SemanticEntity(
                entity_id="ent_input",
                entity_type="trace_input",
                value=task_id,
                attributes={"kind": "macro_engine_input"},
                created_at=created_at,
            )
        ]
        activities: list[SemanticActivity] = []
        macro_entity_ids: list[str] = []
        for index, macro_name in enumerate(macros_used, start=1):
            entity_id = f"macro_{index:03d}"
            macro_entity_ids.append(entity_id)
            entities.append(
                SemanticEntity(
                    entity_id=entity_id,
                    entity_type="macro_definition",
                    value=macro_name.lstrip("@"),
                    attributes={"token": macro_name},
                    created_at=created_at,
                )
            )
        if macro_entity_ids:
            activities.append(
                SemanticActivity(
                    activity_id="act_000_expand",
                    activity_type="macro_expand",
                    input_entity_ids=("ent_input", *macro_entity_ids),
                    output_entity_ids=("ent_input",),
                    agent_id=agent.agent_id,
                    metadata={"token_count": len(tokens)},
                    created_at=created_at,
                )
            )
        previous_entity_id = "ent_input"
        for index, step in enumerate(operation_stream, start=1):
            entity_id = f"ent_{step.output_ref or f'step_{index:03d}'}"
            source_token = str(step.metadata.get("source_token", step.opcode))
            entities.append(
                SemanticEntity(
                    entity_id=entity_id,
                    entity_type="answer_fragment" if step.opcode == "emit" else "intermediate_binding",
                    value=source_token,
                    evidence_handles=step.evidence_handles,
                    attributes={
                        "opcode": step.opcode,
                        "args": list(self._canonicalize_args(step.opcode, step.args)),
                    },
                    created_at=created_at,
                )
            )
            activities.append(
                SemanticActivity(
                    activity_id=f"act_{index:03d}_{step.opcode}",
                    activity_type=step.opcode,
                    input_entity_ids=(previous_entity_id,),
                    output_entity_ids=(entity_id,),
                    agent_id=agent.agent_id,
                    evidence_handles=step.evidence_handles,
                    metadata={"op_id": step.op_id},
                    created_at=created_at,
                )
            )
            previous_entity_id = entity_id
        bundle = ProvenanceBundle(
            bundle_id="bundle_macro_engine",
            entity_ids=tuple(entity.entity_id for entity in entities),
            activity_ids=tuple(activity.activity_id for activity in activities),
            agent_ids=(agent.agent_id,),
            metadata={"task_id": task_id},
            created_at=created_at,
        )
        return CanonicalReasoningGraph(
            entities=tuple(entities),
            activities=tuple(activities),
            agents=(agent,),
            bundles=(bundle,),
            created_at=created_at,
        )

    def _build_context_frames(
        self,
        task_id: str,
        operation_count: int,
        *,
        created_at=None,
    ) -> tuple[ContextFrame, ...]:
        created_at = created_at or utc_now()
        return (
            ContextFrame(
                frame_id="ctx_primary",
                scope="macro_engine",
                confidence=1.0,
                provenance_bundle_id="bundle_macro_engine",
                assumptions=(),
                metadata={
                    "task_id": task_id,
                    "operation_count": operation_count,
                },
                created_at=created_at,
            ),
        )

    def _build_decode_hints(self, operation_stream: Sequence[OperationStep]) -> tuple[DecodeHint, ...]:
        emit_refs = tuple(
            f"ent_{step.output_ref}"
            for step in operation_stream
            if step.opcode == "emit" and step.output_ref
        )
        if emit_refs:
            return (
                DecodeHint(
                    hint_id="hint_verified_answer",
                    template="verified_answer",
                    entity_ids=tuple(sorted(emit_refs)),
                    metadata={"kind": "emit_projection"},
                ),
            )
        if not operation_stream:
            return ()
        final_ref = operation_stream[-1].output_ref
        if not final_ref:
            return ()
        return (
            DecodeHint(
                hint_id="hint_trace_summary",
                template="compressed_trace_summary",
                entity_ids=(f"ent_{final_ref}",),
                metadata={"kind": "summary_projection"},
            ),
        )

    def _compute_proof_hash(
        self,
        *,
        canonical_graph: CanonicalReasoningGraph,
        operation_stream: Sequence[OperationStep],
        context_frames: Sequence[ContextFrame],
        decode_hints: Sequence[DecodeHint],
    ) -> str:
        payload = {
            "ir_version": "1",
            "graph": self._normalize_graph(canonical_graph),
            "operations": [
                self._normalize_operation(step)
                for step in self._canonicalize_operation_stream(operation_stream)
            ],
            "contexts": [
                self._normalize_context(frame)
                for frame in self._canonicalize_context_frames(context_frames)
            ],
            "decode_hints": [
                self._normalize_decode_hint(hint)
                for hint in self._canonicalize_decode_hints(decode_hints)
            ],
        }
        return stable_hash(json.dumps(payload, sort_keys=True, separators=(",", ":")))

    def _normalize_graph(self, graph: CanonicalReasoningGraph) -> dict[str, Any]:
        normalized_entities = sorted(
            (
                {
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type,
                    "value": entity.value,
                    "evidence_handles": sorted(entity.evidence_handles),
                    "attributes": self._clean_mapping(entity.attributes),
                    "confidence": entity.confidence,
                    "uncertainty": entity.uncertainty,
                }
                for entity in graph.entities
            ),
            key=lambda entity: json.dumps(entity, sort_keys=True, separators=(",", ":")),
        )
        normalized_agents = sorted(
            (
                {
                    "agent_id": agent.agent_id,
                    "component": agent.component,
                    "backend": agent.backend,
                    "role": agent.role,
                    "metadata": self._clean_mapping(agent.metadata),
                }
                for agent in graph.agents
            ),
            key=lambda agent: json.dumps(agent, sort_keys=True, separators=(",", ":")),
        )
        entity_positions = {
            entity["entity_id"]: index
            for index, entity in enumerate(normalized_entities)
        }
        agent_positions = {
            agent["agent_id"]: index
            for index, agent in enumerate(normalized_agents)
        }
        normalized_activities = sorted(
            (
                {
                    "activity_id": activity.activity_id,
                    "activity_type": activity.activity_type,
                    "input_positions": sorted(
                        entity_positions[entity_id]
                        for entity_id in activity.input_entity_ids
                        if entity_id in entity_positions
                    ),
                    "output_positions": sorted(
                        entity_positions[entity_id]
                        for entity_id in activity.output_entity_ids
                        if entity_id in entity_positions
                    ),
                    "agent_position": agent_positions.get(activity.agent_id, -1),
                    "evidence_handles": sorted(activity.evidence_handles),
                    "metadata": self._clean_mapping(activity.metadata),
                }
                for activity in graph.activities
            ),
            key=lambda activity: json.dumps(activity, sort_keys=True, separators=(",", ":")),
        )
        activity_positions = {
            activity["activity_id"]: index
            for index, activity in enumerate(normalized_activities)
        }
        normalized_bundles = sorted(
            (
                {
                    "entity_positions": sorted(
                        entity_positions[entity_id]
                        for entity_id in bundle.entity_ids
                        if entity_id in entity_positions
                    ),
                    "activity_positions": sorted(
                        activity_positions[activity_id]
                        for activity_id in bundle.activity_ids
                        if activity_id in activity_positions
                    ),
                    "agent_positions": sorted(
                        agent_positions[agent_id]
                        for agent_id in bundle.agent_ids
                        if agent_id in agent_positions
                    ),
                    "metadata": self._clean_mapping(bundle.metadata),
                }
                for bundle in graph.bundles
            ),
            key=lambda bundle: json.dumps(bundle, sort_keys=True, separators=(",", ":")),
        )
        return {
            "entities": [
                {
                    "entity_type": entity["entity_type"],
                    "value": entity["value"],
                    "evidence_handles": entity["evidence_handles"],
                    "attributes": entity["attributes"],
                    "confidence": entity["confidence"],
                    "uncertainty": entity["uncertainty"],
                }
                for entity in normalized_entities
            ],
            "activities": [
                {
                    "activity_type": activity["activity_type"],
                    "input_positions": activity["input_positions"],
                    "output_positions": activity["output_positions"],
                    "agent_position": activity["agent_position"],
                    "evidence_handles": activity["evidence_handles"],
                    "metadata": activity["metadata"],
                }
                for activity in normalized_activities
            ],
            "agents": [
                {
                    "component": agent["component"],
                    "backend": agent["backend"],
                    "role": agent["role"],
                    "metadata": agent["metadata"],
                }
                for agent in normalized_agents
            ],
            "bundles": normalized_bundles,
        }

    def _normalize_operation(self, step: OperationStep) -> dict[str, Any]:
        source_token = str(step.metadata.get("source_token", ""))
        return {
            "opcode": step.opcode,
            "args": list(self._canonicalize_args(step.opcode, step.args)),
            "evidence_handles": sorted(step.evidence_handles),
            "source_token": source_token,
            "normalized_token": _normalize_step_token(source_token),
        }

    def _normalize_context(self, frame: ContextFrame) -> dict[str, Any]:
        return {
            "scope": frame.scope,
            "confidence": frame.confidence,
            "provenance_bundle_id": frame.provenance_bundle_id,
            "assumptions": sorted(frame.assumptions),
            "metadata": self._clean_mapping(frame.metadata),
        }

    def _normalize_decode_hint(self, hint: DecodeHint) -> dict[str, Any]:
        return {
            "template": hint.template,
            "entity_ids": sorted(hint.entity_ids),
            "metadata": self._clean_mapping(hint.metadata),
        }

    def _canonicalize_operation_stream(
        self,
        operation_stream: Sequence[OperationStep],
    ) -> tuple[OperationStep, ...]:
        return tuple(
            replace(
                step,
                args=self._canonicalize_args(step.opcode, step.args),
                evidence_handles=tuple(sorted(step.evidence_handles)),
                metadata=self._clean_mapping(step.metadata),
            )
            for step in operation_stream
        )

    def _canonicalize_context_frames(
        self,
        context_frames: Sequence[ContextFrame],
    ) -> tuple[ContextFrame, ...]:
        return tuple(
            replace(
                frame,
                assumptions=tuple(sorted(frame.assumptions)),
                metadata=self._clean_mapping(frame.metadata),
            )
            for frame in context_frames
        )

    def _canonicalize_decode_hints(
        self,
        decode_hints: Sequence[DecodeHint],
    ) -> tuple[DecodeHint, ...]:
        return tuple(
            replace(
                hint,
                entity_ids=tuple(sorted(hint.entity_ids)),
                metadata=self._clean_mapping(hint.metadata),
            )
            for hint in decode_hints
        )

    def _canonicalize_args(self, opcode: str, args: Sequence[str]) -> tuple[str, ...]:
        cleaned = tuple(str(arg) for arg in args if str(arg))
        if opcode in _CANONICAL_ARG_OPCODES:
            return tuple(sorted(cleaned))
        return cleaned

    def _clean_mapping(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): self._clean_mapping(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                if key != "created_at"
            }
        if isinstance(value, (list, tuple)):
            return [self._clean_mapping(item) for item in value]
        return value

    def _opcode_for_step(self, step: str) -> str:
        normalized = _normalize_step_token(step)
        if normalized.startswith(("lookup", "match_local_evidence", "review_evidence", "read_question")):
            return "lookup"
        if normalized.startswith(("extract_constraints", "bind")):
            return "bind"
        if normalized.startswith(("compare",)):
            return "compare"
        if normalized.startswith(("reason_pass", "infer")):
            return "infer"
        if normalized.startswith(("synthesize", "aggregate")):
            return "aggregate"
        if normalized.startswith(("check", "refine_answer", "validate")):
            return "check"
        if normalized.startswith(("compose_answer", "emit")):
            return "emit"
        if normalized.startswith(("cite",)):
            return "cite"
        if normalized.startswith(("confidence", "score")):
            return "confidence_update"
        return "bind"

    def _args_for_step(self, step: str) -> tuple[str, ...]:
        normalized = _normalize_step_token(step)
        suffix_match = re.match(r"(.+?)_(\d+)$", normalized)
        if suffix_match:
            return (suffix_match.group(1), suffix_match.group(2))
        return (normalized,)

    def _resolve_macro_invocation(self, token: str, *, allow_missing: bool) -> str | None:
        match = _MACRO_TOKEN_RE.fullmatch(token.strip())
        if match is None:
            return None
        macro_name = _normalize_macro_key(match.group("name"))
        if allow_missing or macro_name in self._macros:
            return macro_name
        return None

    def _parse_macro_invocation(self, token: str) -> dict[str, Any] | None:
        match = _MACRO_TOKEN_RE.fullmatch(token.strip())
        if match is None:
            return None
        macro_name = _normalize_macro_key(match.group("name"))
        if macro_name not in self._macros:
            return None
        macro = self._macros[macro_name]
        arguments = self._parse_macro_arguments(raw_args=match.group("args") or "", macro=macro)
        return {"macro_name": macro_name, "arguments": arguments}

    def _parse_macro_arguments(self, *, raw_args: str, macro: Macro) -> dict[str, str]:
        if not raw_args.strip():
            if macro.parameters:
                raise ValueError(
                    f"Macro {_macro_token(macro.macro_name)} requires parameters: {', '.join(macro.parameters)}"
                )
            return {}
        if not macro.parameters:
            raise ValueError(f"Macro {_macro_token(macro.macro_name)} does not accept parameters.")
        tokens = [part.strip() for part in raw_args.split(",") if part.strip()]
        positional: list[str] = []
        named: dict[str, str] = {}
        for token in tokens:
            if "=" in token:
                name, value = token.split("=", 1)
                key = name.strip()
                if key not in macro.parameters:
                    raise ValueError(
                        f"Unknown parameter '{key}' for macro {_macro_token(macro.macro_name)}."
                    )
                named[key] = value.strip()
            else:
                positional.append(token)
        if len(positional) > len(macro.parameters):
            raise ValueError(
                f"Too many positional arguments for macro {_macro_token(macro.macro_name)}."
            )
        resolved: dict[str, str] = {}
        for index, value in enumerate(positional):
            resolved[macro.parameters[index]] = value
        resolved.update(named)
        missing = [parameter for parameter in macro.parameters if parameter not in resolved]
        if missing:
            raise ValueError(
                f"Missing parameters for macro {_macro_token(macro.macro_name)}: {', '.join(missing)}"
            )
        return resolved

    def _apply_macro_arguments(self, template: str, arguments: dict[str, str]) -> str:
        def replace_match(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in arguments:
                raise ValueError(f"Missing macro argument for placeholder '{key}'.")
            return arguments[key]

        return _PLACEHOLDER_RE.sub(replace_match, template)

    def _render_macro_invocation(self, macro: Macro, arguments: dict[str, str]) -> str:
        token = _macro_token(macro.macro_name)
        if not macro.parameters:
            return token
        rendered_args = ",".join(
            f"{parameter}={arguments[parameter]}"
            for parameter in macro.parameters
        )
        return f"{token}({rendered_args})"

    def _macro_placeholders(self, expansion: Sequence[str]) -> set[str]:
        placeholders: set[str] = set()
        for item in expansion:
            placeholders.update(match.group(1) for match in _PLACEHOLDER_RE.finditer(item))
        return placeholders
