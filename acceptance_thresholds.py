"""Explicit acceptance thresholds for bounded local runtime behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidityAcceptanceThresholds:
    """Minimum correctness and failure-handling expectations for acceptance."""

    minimum_deep_mode_improvement_examples: int
    require_verifier_backed_final_selection: bool
    allow_structured_degraded_results: bool
    require_actionable_real_mode_failures: bool


@dataclass(frozen=True, slots=True)
class CompressionAcceptanceThresholds:
    """Foreground compression limits that keep the feature lightweight."""

    max_foreground_proposals: int
    max_recent_reasoning_logs: int
    minimum_realized_savings_ratio: float
    require_validated_proposals: bool
    require_proof_hash_stability: bool
    require_critic_validity_non_regression: bool


@dataclass(frozen=True, slots=True)
class ResourceAcceptanceThresholds:
    """Baseline runtime caps for local-first operation."""

    dev_vram_gb: int
    baseline_vram_gb: int
    baseline_ram_gb: int
    generation_slots: int
    embedding_slots: int
    require_bounded_queues: bool


@dataclass(frozen=True, slots=True)
class Phase12AcceptanceThresholds:
    """Phase 12 acceptance threshold bundle."""

    validity: ValidityAcceptanceThresholds
    compression: CompressionAcceptanceThresholds
    resources: ResourceAcceptanceThresholds


PHASE12_ACCEPTANCE_THRESHOLDS = Phase12AcceptanceThresholds(
    validity=ValidityAcceptanceThresholds(
        minimum_deep_mode_improvement_examples=1,
        require_verifier_backed_final_selection=True,
        allow_structured_degraded_results=True,
        require_actionable_real_mode_failures=True,
    ),
    compression=CompressionAcceptanceThresholds(
        max_foreground_proposals=5,
        max_recent_reasoning_logs=6,
        minimum_realized_savings_ratio=0.05,
        require_validated_proposals=True,
        require_proof_hash_stability=True,
        require_critic_validity_non_regression=True,
    ),
    resources=ResourceAcceptanceThresholds(
        dev_vram_gb=4,
        baseline_vram_gb=6,
        baseline_ram_gb=8,
        generation_slots=1,
        embedding_slots=1,
        require_bounded_queues=True,
    ),
)
