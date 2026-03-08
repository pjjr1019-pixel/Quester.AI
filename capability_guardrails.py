"""Phase 20 local-first guardrails for future desktop and cloud helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocalFirstCapabilityPlan:
    """Proposed integration shape for future capability work."""

    plan_id: str
    summary: str
    extends_existing_orchestrator: bool = True
    keeps_public_task_entrypoint: bool = True
    preserves_local_generation_embedding_base: bool = True
    preserves_local_storage_and_audit: bool = True
    local_helpers_are_optional: bool = True
    local_execution_remains_primary: bool = True
    local_fallback_available: bool = True
    cloud_helper_mode: str = "disabled"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.plan_id.strip():
            raise ValueError("plan_id must not be empty.")
        if not self.summary.strip():
            raise ValueError("summary must not be empty.")
        if not str(self.cloud_helper_mode).strip():
            raise ValueError("cloud_helper_mode must not be empty.")


@dataclass(frozen=True, slots=True)
class LocalFirstCapabilityAssessment:
    """Result of evaluating one capability plan against the local-first rule."""

    guardrail_id: str
    summary: str
    allowed: bool
    reasons: tuple[str, ...] = ()

    def require_allowed(self) -> None:
        """Raise when the assessed plan violates the local-first guardrail."""
        if not self.allowed:
            raise ValueError(
                f"{self.guardrail_id} rejected the capability plan: {', '.join(self.reasons) or 'unknown_reason'}"
            )


@dataclass(frozen=True, slots=True)
class LocalFirstCapabilityGuardrail:
    """Codifies the rule that future capability helpers extend, not replace, the runtime."""

    guardrail_id: str = "phase20_local_first_extension"
    summary: str = (
        "Desktop and cloud helpers must extend the current local-first runtime rather than replace "
        "the orchestrator, base generation or embedding pair, or local storage and audit path."
    )
    required_public_entrypoint: str = "Orchestrator.run_task(question, thinking_minutes)"
    required_runtime_components: tuple[str, ...] = ("orchestrator", "model_manager", "storage", "dashboard")
    allowed_cloud_modes: tuple[str, ...] = ("disabled", "auxiliary_only")

    def evaluate(self, plan: LocalFirstCapabilityPlan) -> LocalFirstCapabilityAssessment:
        """Return whether the plan preserves the required local-first extension shape."""
        reasons: list[str] = []
        if not plan.extends_existing_orchestrator:
            reasons.append("must_extend_existing_orchestrator")
        if not plan.keeps_public_task_entrypoint:
            reasons.append("must_keep_public_task_entrypoint")
        if not plan.preserves_local_generation_embedding_base:
            reasons.append("must_preserve_generation_embedding_base")
        if not plan.preserves_local_storage_and_audit:
            reasons.append("must_preserve_local_storage_and_audit")
        if not plan.local_helpers_are_optional:
            reasons.append("helpers_must_remain_opt_in")
        if not plan.local_execution_remains_primary:
            reasons.append("local_execution_must_remain_primary")
        if not plan.local_fallback_available:
            reasons.append("helpers_require_local_fallback")
        if plan.cloud_helper_mode not in self.allowed_cloud_modes:
            reasons.append("cloud_helpers_must_remain_auxiliary")
        return LocalFirstCapabilityAssessment(
            guardrail_id=self.guardrail_id,
            summary=self.summary,
            allowed=not reasons,
            reasons=tuple(reasons),
        )


PHASE20_LOCAL_FIRST_GUARDRAIL = LocalFirstCapabilityGuardrail()
