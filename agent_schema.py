"""Boundary-only schema helpers for structured agent I/O."""

from __future__ import annotations

from typing import Any, Mapping

from data_structures import (
    CompressedTrace,
    CritiqueReport,
    MacroProposal,
    Plan,
    ReasonerCriticHandoff,
    ResearchReasonerHandoff,
)


def _object_schema(*, title: str, required: tuple[str, ...], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": title,
        "type": "object",
        "required": list(required),
        "additionalProperties": True,
        "properties": properties,
    }


def research_reasoner_handoff_schema() -> dict[str, Any]:
    return _object_schema(
        title="research_reasoner_handoff_v1",
        required=("plan", "evidence", "budget", "evidence_handles"),
        properties={
            "plan": {"type": "object"},
            "evidence": {"type": "object"},
            "budget": {"type": "object"},
            "evidence_handles": {"type": "array", "items": {"type": "string"}},
            "reasoning_mode": {"type": "string"},
            "final_text_policy": {"type": "string"},
            "output_contract": {"type": "string"},
            "implementation_mode": {"type": "string"},
        },
    )


def reasoner_critic_handoff_schema() -> dict[str, Any]:
    return _object_schema(
        title="reasoner_critic_handoff_v1",
        required=(
            "plan",
            "evidence",
            "trace",
            "budget",
            "evidence_handles",
            "proof_hash",
            "required_opcode_names",
            "required_macro_names",
            "required_decoder_names",
        ),
        properties={
            "plan": {"type": "object"},
            "evidence": {"type": "object"},
            "trace": {"type": "object"},
            "budget": {"type": "object"},
            "evidence_handles": {"type": "array", "items": {"type": "string"}},
            "proof_hash": {"type": "string"},
            "required_opcode_names": {"type": "array", "items": {"type": "string"}},
            "required_macro_names": {"type": "array", "items": {"type": "string"}},
            "required_decoder_names": {"type": "array", "items": {"type": "string"}},
            "repair_attempt_count": {"type": "integer"},
            "repair_history": {"type": "array", "items": {"type": "string"}},
            "final_text_policy": {"type": "string"},
            "output_contract": {"type": "string"},
            "implementation_mode": {"type": "string"},
        },
    )


def planner_output_schema() -> dict[str, Any]:
    return _object_schema(
        title="planner_plan_v1",
        required=("task_id", "question", "steps"),
        properties={
            "task_id": {"type": "string"},
            "question": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "object"}},
            "required_evidence": {"type": "array", "items": {"type": "string"}},
            "success_criteria": {"type": "array", "items": {"type": "string"}},
            "budget": {"type": "object"},
            "planner_notes": {"type": "string"},
        },
    )


def reasoner_output_schema() -> dict[str, Any]:
    return _object_schema(
        title="compressed_trace_v1",
        required=("task_id", "tokens", "confidence"),
        properties={
            "task_id": {"type": "string"},
            "tokens": {"type": "array", "items": {"type": "string"}},
            "expanded_preview": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "operation_stream": {"type": "array", "items": {"type": "object"}},
            "decode_hints": {"type": "array", "items": {"type": "object"}},
            "proof_hash": {"type": "string"},
        },
    )


def critic_output_schema() -> dict[str, Any]:
    return _object_schema(
        title="critique_report_v1",
        required=("task_id", "is_valid", "issues", "evidence_coverage", "result"),
        properties={
            "task_id": {"type": "string"},
            "is_valid": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "fixed_trace": {"type": "object"},
            "evidence_coverage": {"type": "number"},
            "critic_notes": {"type": "string"},
            "result": {"type": "string"},
        },
    )


def compressor_output_schema() -> dict[str, Any]:
    return {
        "title": "macro_proposal_list_v1",
        "type": "array",
        "items": _object_schema(
            title="macro_proposal_v1",
            required=("proposal_id", "macro", "reason", "simulation_score", "approved"),
            properties={
                "proposal_id": {"type": "string"},
                "macro": {"type": "object"},
                "reason": {"type": "string"},
                "examples": {"type": "array", "items": {"type": "string"}},
                "simulation_score": {"type": "number"},
                "approved": {"type": "boolean"},
                "validation_passed": {"type": "boolean"},
                "validation_issues": {"type": "array", "items": {"type": "string"}},
                "proof_fingerprint": {"type": "string"},
            },
        ),
    }


def parse_research_reasoner_handoff(payload: Mapping[str, Any]) -> ResearchReasonerHandoff:
    return ResearchReasonerHandoff.from_dict(payload)


def parse_reasoner_critic_handoff(payload: Mapping[str, Any]) -> ReasonerCriticHandoff:
    return ReasonerCriticHandoff.from_dict(payload)


def parse_planner_output(payload: Mapping[str, Any]) -> Plan:
    return Plan.from_dict(payload)


def parse_reasoner_output(payload: Mapping[str, Any]) -> CompressedTrace:
    return CompressedTrace.from_dict(payload)


def parse_critic_output(payload: Mapping[str, Any]) -> CritiqueReport:
    return CritiqueReport.from_dict(payload)


def parse_compressor_output(payload: list[Mapping[str, Any]]) -> tuple[MacroProposal, ...]:
    return tuple(MacroProposal.from_dict(item) for item in payload)
