"""Background self-optimizer scaffold."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from config import APP_CONFIG, AppConfig
from data_structures import (
    CompressedTrace,
    CritiqueResult,
    LongHorizonCheckpoint,
    LongHorizonSession,
    Macro,
    MacroProposal,
    OptimizerActivationDecision,
    OptimizerActivationRecord,
    OptimizerLifecycleStage,
    OptimizerProposalRecord,
    OptimizerReplayEvaluation,
    OptimizerReplaySample,
    OptimizerRollbackRecord,
    OptimizerSuggestionKind,
    OptimizerSuggestionRecord,
    PerformanceMetric,
    ResourceBudget,
    ReasoningLog,
    TaskResult,
    VerifiedDeepTraceExport,
)
from macro_engine import MacroEngine
from retrieval import stable_hash
from storage import StorageManager
from utils import cancel_task, utc_now_iso


class SelfOptimizer:
    """Runs non-blocking optimization cycles while the app is live."""

    def __init__(
        self,
        storage: StorageManager,
        config: AppConfig = APP_CONFIG,
        *,
        cache_lookup: Callable[[str, str], Any | None] | None = None,
        cache_warm: Callable[[str, str, Any], None] | None = None,
    ):
        self.storage = storage
        self.config = config
        self.logger = logging.getLogger("quester.self_optimizer")
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._stop_event = asyncio.Event()
        self._last_evaluations: tuple[OptimizerReplayEvaluation, ...] = ()
        self._cache_lookup = cache_lookup
        self._cache_warm = cache_warm

    async def start(self) -> None:
        """Start optimizer loop if enabled."""
        if self._started:
            return
        if not self.config.preflight.flags.enable_self_optimizer:
            self.logger.info("SelfOptimizer disabled by config.")
            return
        self._started = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop(), name="self-optimizer-loop")
        self.logger.info("SelfOptimizer started.")

    async def stop(self) -> None:
        """Stop optimizer loop cleanly."""
        self._stop_event.set()
        await cancel_task(self._task, timeout_s=self.config.preflight.flags.shutdown_timeout_s)
        self._task = None
        self._started = False
        self.logger.info("SelfOptimizer stopped.")

    @property
    def last_evaluations(self) -> tuple[OptimizerReplayEvaluation, ...]:
        return self._last_evaluations

    async def run_cycle(self) -> list[MacroProposal]:
        """Execute one optimizer cycle."""
        foreground_task_count = await self.storage.get_foreground_task_count()
        if foreground_task_count > 0:
            await self.storage.log_event(
                "self_optimizer.cycle_deferred",
                {
                    "reason": "foreground_task_active",
                    "foreground_task_count": foreground_task_count,
                },
            )
            return []

        engine = MacroEngine()
        reasoning_logs = await self._load_reasoning_logs()
        traces = await self.storage.list_reasoning_traces()
        metrics = await self.storage.list_performance_metrics()
        replay_samples = await self._load_replay_samples()
        cycle_id = self._build_cycle_id(
            reasoning_logs=reasoning_logs,
            traces=traces,
            metrics=metrics,
            replay_samples=replay_samples,
        )
        proposals = self._build_trace_driven_proposals(
            engine=engine,
            reasoning_logs=reasoning_logs,
            traces=traces,
            metrics=metrics,
        )
        evaluations = self._evaluate_proposals(
            engine=engine,
            proposals=tuple(proposals),
            replay_samples=replay_samples,
            traces=traces,
        )
        self._last_evaluations = evaluations
        if evaluations:
            await self.storage.record_optimizer_replay_evaluations(evaluations)
            proposals = self._apply_replay_scores(proposals, evaluations)
        self._publish_compression_artifact_summaries(
            proposals=tuple(proposals),
            evaluations=evaluations,
        )
        proposal_summaries = self._summarize_proposal_states(
            proposals=tuple(proposals),
            evaluations=evaluations,
        )
        proposal_records = self._build_proposal_records(
            cycle_id=cycle_id,
            proposals=tuple(proposals),
            replay_samples=replay_samples,
            proposal_summaries=proposal_summaries,
        )
        activation_records, rollback_records = await self._build_activation_and_rollback_records(
            cycle_id=cycle_id,
            proposals=tuple(proposals),
            proposal_summaries=proposal_summaries,
        )
        if proposal_records:
            await self.storage.record_optimizer_proposal_records(proposal_records)
        if activation_records:
            await self.storage.record_optimizer_activation_records(activation_records)
        if rollback_records:
            await self.storage.record_optimizer_rollback_records(rollback_records)
        await self.storage.log_event(
            "self_optimizer.cycle",
            {
                "cycle_id": cycle_id,
                "proposal_count": len(proposals),
                "proposal_ids": [proposal.proposal_id for proposal in proposals],
                "reasoning_log_count": len(reasoning_logs),
                "trace_count": len(traces),
                "metric_count": len(metrics),
                "replay_sample_count": len(replay_samples),
                "evaluation_count": len(evaluations),
                "accepted_evaluation_count": sum(1 for evaluation in evaluations if evaluation.accepted),
                "proposal_scores": self._summarize_evaluations(evaluations),
                "activation_decisions": [
                    {
                        "proposal_id": record.proposal_id,
                        "decision": record.decision.value,
                        "reason": record.reason,
                    }
                    for record in activation_records
                ],
                "rollback_record_count": len(rollback_records),
            },
        )
        return proposals

    async def export_verified_deep_traces(
        self,
        *,
        export_path: Path | None = None,
    ) -> tuple[VerifiedDeepTraceExport, ...]:
        """Export the bounded verified deep-trace dataset for future replay or distillation."""
        return await self.storage.export_verified_deep_traces(
            export_path=export_path,
            limit=self.config.self_optimizer.replay_history_limit,
        )

    async def suggest_for_long_horizon(
        self,
        *,
        session: LongHorizonSession,
        cycle_index: int,
        latest_result: TaskResult,
        checkpoints: tuple[LongHorizonCheckpoint, ...],
        budget: ResourceBudget,
    ) -> tuple[OptimizerSuggestionRecord, ...]:
        """Return bounded typed advice for the next long-horizon cycle."""
        suggestions: list[OptimizerSuggestionRecord] = []
        recent_proposals = (await self.storage.list_optimizer_proposal_records())[-8:]
        candidate_count = len(latest_result.reasoning.candidate_traces)
        evidence_count = len(latest_result.evidence.local_results) + len(latest_result.evidence.web_results)
        repair_count = len(latest_result.critique.repair_actions)
        max_budget = self.config.budget_calibration
        source_task_ids = tuple(
            dict.fromkeys(
                item
                for item in (
                    *(checkpoint.task_id for checkpoint in checkpoints[-3:]),
                    latest_result.task_id,
                )
                if str(item).strip()
            )
        )

        if latest_result.critique.result != CritiqueResult.VALID and (
            budget.max_web_queries < max_budget.max_web_queries
            or budget.retrieval_top_k < max_budget.max_retrieval_top_k
        ):
            suggestions.append(
                self._make_suggestion_record(
                    cycle_id=f"{session.session_id}:cycle:{cycle_index}",
                    kind=OptimizerSuggestionKind.RETRIEVAL_STRATEGY,
                    summary="Refresh retrieval more aggressively on the next bounded cycle.",
                    rationale=(
                        "The current cycle ended without a fully valid critique result, so bounded retrieval depth "
                        "and optional web refresh can increase evidence coverage."
                    ),
                    target_components=("researcher", "reasoner"),
                    confidence=0.74,
                    source_task_ids=source_task_ids,
                    metadata={
                        "budget_delta": {
                            "retrieval_top_k": 1 if budget.retrieval_top_k < max_budget.max_retrieval_top_k else 0,
                            "max_web_queries": 1 if budget.max_web_queries < max_budget.max_web_queries else 0,
                        }
                    },
                )
            )
        if repair_count > 0 and budget.critic_passes < max_budget.max_critic_passes:
            suggestions.append(
                self._make_suggestion_record(
                    cycle_id=f"{session.session_id}:cycle:{cycle_index}",
                    kind=OptimizerSuggestionKind.CRITIQUE_HEURISTIC,
                    summary="Spend one more bounded verifier pass on the next cycle.",
                    rationale=(
                        "Repair actions were needed in the current cycle, which is a strong signal that another "
                        "critic pass is more useful than a larger single prompt."
                    ),
                    target_components=("critic",),
                    confidence=0.71,
                    source_task_ids=source_task_ids,
                    metadata={"budget_delta": {"critic_passes": 1}},
                )
            )
        if candidate_count <= 1 and (
            budget.reasoner_passes < max_budget.max_reasoner_passes or budget.macro_depth < max_budget.max_macro_depth
        ):
            suggestions.append(
                self._make_suggestion_record(
                    cycle_id=f"{session.session_id}:cycle:{cycle_index}",
                    kind=OptimizerSuggestionKind.PLANNING_TEMPLATE,
                    summary="Broaden the next bounded reasoning batch instead of waiting longer.",
                    rationale=(
                        "Only one candidate survived this cycle, so adding a small bounded reasoning expansion is "
                        "more useful than preserving the same narrow search."
                    ),
                    target_components=("planner", "reasoner"),
                    confidence=0.69,
                    source_task_ids=source_task_ids,
                    metadata={
                        "budget_delta": {
                            "reasoner_passes": 1 if budget.reasoner_passes < max_budget.max_reasoner_passes else 0,
                            "macro_depth": 1 if budget.macro_depth < max_budget.max_macro_depth else 0,
                        }
                    },
                )
            )
        eligible_macro_records = [record for record in recent_proposals if record.activation_eligible]
        if eligible_macro_records:
            top_records = sorted(
                eligible_macro_records,
                key=lambda record: (
                    self._compression_effectiveness_score(record.proposal),
                    record.pass_rate,
                    record.mean_simulation_score,
                ),
                reverse=True,
            )[:2]
            suggestions.append(
                self._make_suggestion_record(
                    cycle_id=f"{session.session_id}:cycle:{cycle_index}",
                    kind=OptimizerSuggestionKind.MACRO_ADVICE,
                    summary="Carry forward the latest replay-approved macro candidates as advisory context.",
                    rationale=(
                        "Recent optimizer proposals passed replay gating, but live activation remains policy-blocked, "
                        "so the next cycle should treat them as explainable deferred advice only."
                    ),
                    target_components=("compressor", "reasoner", "critic"),
                    confidence=0.66,
                    source_task_ids=source_task_ids,
                    metadata={
                        "proposal_ids": tuple(record.proposal_id for record in top_records),
                        "proof_fingerprints": tuple(record.proposal.proof_fingerprint for record in top_records),
                        "budget_delta": {},
                        "advisory_only_reason": "proposal_only_policy",
                    },
                )
            )
        if not suggestions and evidence_count > 0:
            suggestions.append(
                self._make_suggestion_record(
                    cycle_id=f"{session.session_id}:cycle:{cycle_index}",
                    kind=OptimizerSuggestionKind.DASHBOARD_HINT,
                    summary="Keep the next cycle bounded; the current evidence set is already materially populated.",
                    rationale=(
                        "No stronger optimizer hint was discovered, so the safest advisory action is to preserve the "
                        "current bounded schedule and surface that choice explicitly."
                    ),
                    target_components=("dashboard",),
                    confidence=0.6,
                    source_task_ids=source_task_ids,
                    metadata={"budget_delta": {}},
                )
            )
        return tuple(suggestions[: min(4, self.config.self_optimizer.proposal_limit)])

    def _compression_effectiveness_score(self, proposal: MacroProposal) -> float:
        if self._cache_lookup is None:
            return 0.0
        cache_key = proposal.proof_fingerprint or proposal.proposal_id
        cached_summary = self._cache_lookup("compression_artifacts", cache_key)
        if not isinstance(cached_summary, dict):
            return 0.0
        try:
            return float(cached_summary.get("effectiveness_score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _publish_compression_artifact_summaries(
        self,
        *,
        proposals: tuple[MacroProposal, ...],
        evaluations: tuple[OptimizerReplayEvaluation, ...],
    ) -> None:
        if self._cache_warm is None:
            return
        evaluation_by_proposal_id = {item.proposal_id: item for item in evaluations}
        for proposal in proposals[: self.config.self_optimizer.proposal_limit]:
            evaluation = evaluation_by_proposal_id.get(proposal.proposal_id)
            summary = {
                "proposal_id": proposal.proposal_id,
                "proof_fingerprint": proposal.proof_fingerprint,
                "macro_name": proposal.macro.macro_name,
                "effectiveness_score": round(
                    evaluation.aggregate_score if evaluation is not None else proposal.simulation_score,
                    3,
                ),
                "validation_pass_rate": round(
                    evaluation.critique_validity if evaluation is not None else float(proposal.validation_passed),
                    3,
                ),
                "compression_gain": round(
                    evaluation.compression_gain if evaluation is not None else proposal.simulation_score,
                    3,
                ),
                "accepted": bool(evaluation.accepted) if evaluation is not None else bool(proposal.validation_passed),
                "validation_state": (
                    "validated"
                    if (evaluation.accepted if evaluation is not None else proposal.validation_passed)
                    else "blocked"
                ),
                "blocked_reason": (
                    evaluation.rejection_reason
                    if evaluation is not None and evaluation.rejection_reason
                    else (
                        proposal.validation_issues[0]
                        if proposal.validation_issues and not proposal.validation_passed
                        else ""
                    )
                ),
                "evidence_basis": "replay_evidence",
                "origin_component": "self_optimizer",
                "source": "self_optimizer",
            }
            cache_key = proposal.proof_fingerprint or proposal.proposal_id
            self._cache_warm("compression_artifacts", cache_key, summary)

    async def _load_reasoning_logs(self) -> tuple[ReasoningLog, ...]:
        payloads = await self.storage.list_reasoning_history()
        logs: list[ReasoningLog] = []
        for payload in payloads:
            if "compressed_chain" not in payload:
                continue
            try:
                logs.append(ReasoningLog.from_dict(payload))
            except (KeyError, TypeError, ValueError):
                continue
        return tuple(logs)

    async def _load_replay_samples(self) -> tuple[OptimizerReplaySample, ...]:
        samples = await self.storage.list_optimizer_replay_samples()
        limit = self.config.self_optimizer.replay_history_limit
        if len(samples) <= limit:
            return samples
        return samples[-limit:]

    def _make_suggestion_record(
        self,
        *,
        cycle_id: str,
        kind: OptimizerSuggestionKind,
        summary: str,
        rationale: str,
        target_components: tuple[str, ...],
        confidence: float,
        source_task_ids: tuple[str, ...],
        metadata: dict[str, object],
    ) -> OptimizerSuggestionRecord:
        seed = "|".join(
            (
                cycle_id,
                kind.value,
                summary,
                ",".join(target_components),
                ",".join(source_task_ids),
                repr(sorted(metadata.items())),
            )
        )
        return OptimizerSuggestionRecord(
            suggestion_id=f"suggestion:{stable_hash(seed)[:16]}",
            cycle_id=cycle_id,
            kind=kind,
            summary=summary,
            rationale=rationale,
            target_components=target_components,
            source_task_ids=source_task_ids,
            confidence=confidence,
            advisory_only=True,
            metadata=metadata,
        )

    def _build_cycle_id(
        self,
        *,
        reasoning_logs: tuple[ReasoningLog, ...],
        traces: tuple[CompressedTrace, ...],
        metrics: tuple[PerformanceMetric, ...],
        replay_samples: tuple[OptimizerReplaySample, ...],
    ) -> str:
        seed = "|".join(
            [
                utc_now_iso(),
                str(len(reasoning_logs)),
                str(len(traces)),
                str(len(metrics)),
                str(len(replay_samples)),
                ",".join(sample.task_id for sample in replay_samples[-4:]),
            ]
        )
        return f"cycle:{stable_hash(seed)[:16]}"

    def _summarize_proposal_states(
        self,
        *,
        proposals: tuple[MacroProposal, ...],
        evaluations: tuple[OptimizerReplayEvaluation, ...],
    ) -> dict[str, dict[str, float | int | tuple[str, ...] | bool]]:
        grouped: dict[str, list[OptimizerReplayEvaluation]] = {}
        for evaluation in evaluations:
            grouped.setdefault(evaluation.proposal_id, []).append(evaluation)
        summaries: dict[str, dict[str, float | int | tuple[str, ...] | bool]] = {}
        for proposal in proposals:
            items = grouped.get(proposal.proposal_id, [])
            sample_count = len(items)
            accepted_count = sum(1 for item in items if item.accepted)
            mean_score = (
                round(sum(item.aggregate_score for item in items) / sample_count, 3)
                if sample_count
                else proposal.simulation_score
            )
            pass_rate = round((accepted_count / sample_count), 3) if sample_count else 0.0
            rejection_reasons = tuple(
                sorted(
                    {
                        reason.strip()
                        for item in items
                        for reason in item.rejection_reason.split(",")
                        if reason.strip()
                    }
                )
            )
            contradiction_risk = self._estimate_contradiction_risk(
                proposal=proposal,
                pass_rate=pass_rate,
                rejection_reasons=rejection_reasons,
            )
            validation_ready = (
                proposal.validation_passed
                and sample_count > 0
                and mean_score >= self.config.self_optimizer.minimum_simulation_score
            )
            activation_eligible = (
                validation_ready
                and pass_rate >= 0.5
                and contradiction_risk <= 0.35
            )
            summaries[proposal.proposal_id] = {
                "sample_count": sample_count,
                "accepted_count": accepted_count,
                "mean_score": mean_score,
                "pass_rate": pass_rate,
                "rejection_reasons": rejection_reasons,
                "contradiction_risk": contradiction_risk,
                "validation_ready": validation_ready,
                "activation_eligible": activation_eligible,
            }
        return summaries

    def _build_proposal_records(
        self,
        *,
        cycle_id: str,
        proposals: tuple[MacroProposal, ...],
        replay_samples: tuple[OptimizerReplaySample, ...],
        proposal_summaries: dict[str, dict[str, float | int | tuple[str, ...] | bool]],
    ) -> tuple[OptimizerProposalRecord, ...]:
        source_task_ids = tuple(sample.task_id for sample in replay_samples)
        records: list[OptimizerProposalRecord] = []
        for proposal in proposals:
            summary = proposal_summaries.get(proposal.proposal_id, {})
            replay_sample_count = int(summary.get("sample_count", 0))
            accepted_count = int(summary.get("accepted_count", 0))
            mean_score = float(summary.get("mean_score", proposal.simulation_score))
            pass_rate = float(summary.get("pass_rate", 0.0))
            contradiction_risk = float(summary.get("contradiction_risk", 1.0))
            activation_eligible = bool(summary.get("activation_eligible", False))
            validation_ready = bool(summary.get("validation_ready", False))
            records.append(
                OptimizerProposalRecord(
                    cycle_id=cycle_id,
                    proposal_id=proposal.proposal_id,
                    proposal=proposal,
                    lifecycle_stage=OptimizerLifecycleStage.PROPOSED,
                    source_task_ids=source_task_ids,
                    replay_sample_count=replay_sample_count,
                    accepted_simulation_count=0,
                    mean_simulation_score=proposal.simulation_score,
                    pass_rate=0.0,
                    contradiction_risk=1.0,
                    activation_eligible=False,
                )
            )
            records.append(
                OptimizerProposalRecord(
                    cycle_id=cycle_id,
                    proposal_id=proposal.proposal_id,
                    proposal=proposal,
                    lifecycle_stage=OptimizerLifecycleStage.SIMULATED,
                    source_task_ids=source_task_ids,
                    replay_sample_count=replay_sample_count,
                    accepted_simulation_count=accepted_count,
                    mean_simulation_score=mean_score,
                    pass_rate=pass_rate,
                    contradiction_risk=contradiction_risk,
                    activation_eligible=False,
                )
            )
            records.append(
                OptimizerProposalRecord(
                    cycle_id=cycle_id,
                    proposal_id=proposal.proposal_id,
                    proposal=proposal,
                    lifecycle_stage=(
                        OptimizerLifecycleStage.VALIDATED
                        if validation_ready
                        else OptimizerLifecycleStage.REJECTED
                    ),
                    source_task_ids=source_task_ids,
                    replay_sample_count=replay_sample_count,
                    accepted_simulation_count=accepted_count,
                    mean_simulation_score=mean_score,
                    pass_rate=pass_rate,
                    contradiction_risk=contradiction_risk,
                    activation_eligible=activation_eligible,
                )
            )
        return tuple(records)

    async def _build_activation_and_rollback_records(
        self,
        *,
        cycle_id: str,
        proposals: tuple[MacroProposal, ...],
        proposal_summaries: dict[str, dict[str, float | int | tuple[str, ...] | bool]],
    ) -> tuple[tuple[OptimizerActivationRecord, ...], tuple[OptimizerRollbackRecord, ...]]:
        active_macros = await self.storage.list_macros(active_only=True)
        active_macro_versions = {
            macro.macro_name: macro.version
            for macro in active_macros
        }
        activation_records: list[OptimizerActivationRecord] = []
        rollback_records: list[OptimizerRollbackRecord] = []
        for proposal in proposals:
            summary = proposal_summaries.get(proposal.proposal_id, {})
            decision, lifecycle_stage, reason = self._evaluate_activation_outcome(
                proposal=proposal,
                summary=summary,
            )
            rollback_record_id = ""
            activation_eligible = bool(summary.get("activation_eligible", False))
            if activation_eligible:
                rollback_record_id = f"rollback:{stable_hash(f'{cycle_id}:{proposal.proposal_id}')[:16]}"
                rollback_records.append(
                    OptimizerRollbackRecord(
                        rollback_record_id=rollback_record_id,
                        cycle_id=cycle_id,
                        proposal_id=proposal.proposal_id,
                        proposal_macro_name=proposal.macro.macro_name,
                        active_macro_versions=active_macro_versions,
                        reason=(
                            "Prepared snapshot before future activation attempt."
                            if decision != OptimizerActivationDecision.REJECTED
                            else "Prepared snapshot for auditable rejection path."
                        ),
                        applied=False,
                    )
                )
            activation_records.append(
                OptimizerActivationRecord(
                    cycle_id=cycle_id,
                    proposal_id=proposal.proposal_id,
                    decision=decision,
                    lifecycle_stage=lifecycle_stage,
                    reason=reason,
                    validation_passed=bool(summary.get("validation_ready", False)),
                    mean_simulation_score=float(summary.get("mean_score", proposal.simulation_score)),
                    pass_rate=float(summary.get("pass_rate", 0.0)),
                    contradiction_risk=float(summary.get("contradiction_risk", 1.0)),
                    activation_applied=False,
                    rollback_record_id=rollback_record_id,
                )
            )
        return tuple(activation_records), tuple(rollback_records)

    def _evaluate_activation_outcome(
        self,
        *,
        proposal: MacroProposal,
        summary: dict[str, float | int | tuple[str, ...] | bool],
    ) -> tuple[OptimizerActivationDecision, OptimizerLifecycleStage, str]:
        sample_count = int(summary.get("sample_count", 0))
        validation_ready = bool(summary.get("validation_ready", False))
        activation_eligible = bool(summary.get("activation_eligible", False))
        rejection_reasons = tuple(str(item) for item in summary.get("rejection_reasons", ()))
        if sample_count == 0:
            return (
                OptimizerActivationDecision.DEFERRED,
                OptimizerLifecycleStage.SIMULATED,
                "replay_samples_unavailable",
            )
        if not validation_ready:
            reason = "proposal_validation_failed" if not proposal.validation_passed else "simulation_gate_failed"
            return (
                OptimizerActivationDecision.REJECTED,
                OptimizerLifecycleStage.REJECTED,
                reason,
            )
        if not activation_eligible:
            return (
                OptimizerActivationDecision.REJECTED,
                OptimizerLifecycleStage.REJECTED,
                ",".join(rejection_reasons) or "contradiction_risk_too_high",
            )
        return (
            OptimizerActivationDecision.BLOCKED,
            OptimizerLifecycleStage.ACTIVATION_BLOCKED,
            "proposal_only_policy",
        )

    def _estimate_contradiction_risk(
        self,
        *,
        proposal: MacroProposal,
        pass_rate: float,
        rejection_reasons: tuple[str, ...],
    ) -> float:
        risk = 1.0 - max(0.0, min(1.0, pass_rate))
        if not proposal.validation_passed:
            risk += 0.2
        if any("proof_hash" in reason for reason in rejection_reasons):
            risk += 0.2
        if any("invalid" in reason for reason in rejection_reasons):
            risk += 0.15
        if any("memory" in reason or "latency" in reason for reason in rejection_reasons):
            risk += 0.05
        return self._clamp_score(risk)

    def _build_trace_driven_proposals(
        self,
        *,
        engine: MacroEngine,
        reasoning_logs: tuple[ReasoningLog, ...],
        traces: tuple[CompressedTrace, ...],
        metrics: tuple[PerformanceMetric, ...],
    ) -> list[MacroProposal]:
        if not reasoning_logs and not traces:
            return []
        proposal_limit = self.config.self_optimizer.proposal_limit
        average_iterations = (
            sum(metric.iterations for metric in metrics) / len(metrics) if metrics else 0.0
        )
        average_time = sum(metric.time for metric in metrics) / len(metrics) if metrics else 0.0
        verification_bonus = self._verified_trace_bonus(traces)
        proposals: list[MacroProposal] = []

        cross_log_counts: Counter[tuple[str, ...]] = Counter()
        for log in reasoning_logs:
            compressed_chain = tuple(str(item) for item in log.compressed_chain if str(item).strip())
            for span in (3, 2):
                cross_log_counts.update(
                    {
                        tuple(compressed_chain[index : index + span])
                        for index in range(0, max(0, len(compressed_chain) - span + 1))
                    }
                )
        for motif, count in cross_log_counts.most_common():
            if len(motif) < 2 or count <= 1:
                continue
            self._append_validated_proposal(
                proposals=proposals,
                engine=engine,
                prefix="optimizer_crosslog",
                expansion=motif,
                opcode_pattern=tuple(engine._opcode_for_step(step) for step in motif),
                semantic_kind="optimizer_cross_log_macro",
                reason=(
                    f"Observed in {count} reasoning logs with avg_iterations={average_iterations:.2f} "
                    f"and avg_time={average_time:.3f}s."
                ),
                example=" | ".join(motif),
                simulation_score=min(1.0, 0.28 + (count / 6.0) + verification_bonus),
            )
            if len(proposals) >= proposal_limit:
                return proposals

        opcode_counts: Counter[tuple[str, ...]] = Counter()
        for trace in traces:
            opcode_chain = tuple(step.opcode for step in trace.operation_stream if step.opcode)
            for span in (3, 2):
                opcode_counts.update(
                    {
                        tuple(opcode_chain[index : index + span])
                        for index in range(0, max(0, len(opcode_chain) - span + 1))
                    }
                )
        for motif, count in opcode_counts.most_common():
            if len(motif) < 2 or count <= 1:
                continue
            expansion = tuple(f"opcode:{opcode}" for opcode in motif)
            self._append_validated_proposal(
                proposals=proposals,
                engine=engine,
                prefix="optimizer_opcode",
                expansion=expansion,
                opcode_pattern=motif,
                semantic_kind="optimizer_opcode_macro",
                reason=(
                    f"Repeated runtime opcode motif across {count} persisted traces with "
                    f"avg_iterations={average_iterations:.2f}."
                ),
                example=" | ".join(expansion),
                simulation_score=min(1.0, 0.3 + (count / 7.0) + verification_bonus),
            )
            if len(proposals) >= proposal_limit:
                return proposals

        for trace in reversed(traces):
            if not trace.operation_stream:
                continue
            expansion = tuple(step.opcode for step in trace.operation_stream[: min(3, len(trace.operation_stream))])
            if len(expansion) < 2:
                continue
            self._append_validated_proposal(
                proposals=proposals,
                engine=engine,
                prefix="optimizer_trace",
                expansion=tuple(f"opcode:{opcode}" for opcode in expansion),
                opcode_pattern=expansion,
                semantic_kind="optimizer_trace_slice_macro",
                reason=(
                    f"Derived from persisted trace {trace.task_id} with confidence={trace.confidence:.2f} "
                    f"and avg_time={average_time:.3f}s."
                ),
                example=" | ".join(expansion),
                simulation_score=min(1.0, 0.24 + trace.confidence + verification_bonus),
            )
            if proposals:
                break
        return proposals[:proposal_limit]

    def _append_validated_proposal(
        self,
        *,
        proposals: list[MacroProposal],
        engine: MacroEngine,
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
            proposal_id=f"{prefix}:{sanitized}",
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
        if any(existing.proposal_id == validated.proposal_id for existing in proposals):
            return
        proposals.append(validated)

    def _verified_trace_bonus(self, traces: tuple[CompressedTrace, ...]) -> float:
        if not traces:
            return 0.0
        verified_count = 0
        for trace in traces:
            if trace.context_frames and trace.context_frames[0].metadata.get("vv", False):
                verified_count += 1
        return min(0.15, verified_count / max(1, len(traces) * 10))

    def _evaluate_proposals(
        self,
        *,
        engine: MacroEngine,
        proposals: tuple[MacroProposal, ...],
        replay_samples: tuple[OptimizerReplaySample, ...],
        traces: tuple[CompressedTrace, ...],
    ) -> tuple[OptimizerReplayEvaluation, ...]:
        if not proposals or not replay_samples:
            return ()
        trace_by_task_id = {trace.task_id: trace for trace in traces}
        evaluations: list[OptimizerReplayEvaluation] = []
        for proposal in proposals:
            validated_proposal = engine.validate_macro_proposal(proposal)
            for sample in replay_samples:
                evaluations.append(
                    self._evaluate_proposal_against_sample(
                        proposal=validated_proposal,
                        sample=sample,
                        trace=trace_by_task_id.get(sample.task_id),
                    )
                )
        return tuple(evaluations)

    def _evaluate_proposal_against_sample(
        self,
        *,
        proposal: MacroProposal,
        sample: OptimizerReplaySample,
        trace: CompressedTrace | None,
    ) -> OptimizerReplayEvaluation:
        settings = self.config.self_optimizer
        compression_gain = self._estimate_compression_gain(
            proposal=proposal,
            sample=sample,
            trace=trace,
        )
        proof_hash_stability = self._estimate_proof_hash_stability(proposal=proposal, sample=sample)
        critique_validity = self._estimate_critique_validity(sample)
        latency_ratio = self._estimate_latency_ratio(
            proposal=proposal,
            sample=sample,
            compression_gain=compression_gain,
        )
        memory_ratio = self._estimate_memory_ratio(
            proposal=proposal,
            sample=sample,
            compression_gain=compression_gain,
        )
        latency_score = self._ratio_score(latency_ratio, settings.max_latency_ratio)
        memory_score = self._ratio_score(memory_ratio, settings.max_memory_ratio)
        validation_penalty = 0.0 if proposal.validation_passed else 0.35
        aggregate_score = self._clamp_score(
            (
                settings.compression_gain_weight * compression_gain
                + settings.proof_hash_stability_weight * proof_hash_stability
                + settings.critique_validity_weight * critique_validity
                + settings.latency_weight * latency_score
                + settings.memory_weight * memory_score
            )
            - validation_penalty
        )
        accepted, rejection_reason = self._simulation_gate(
            proposal=proposal,
            sample=sample,
            aggregate_score=aggregate_score,
            latency_ratio=latency_ratio,
            memory_ratio=memory_ratio,
        )
        return OptimizerReplayEvaluation(
            proposal_id=proposal.proposal_id,
            task_id=sample.task_id,
            trace_proof_hash=sample.trace_proof_hash,
            compression_gain=compression_gain,
            proof_hash_stability=proof_hash_stability,
            critique_validity=critique_validity,
            latency_ratio=latency_ratio,
            memory_ratio=memory_ratio,
            aggregate_score=aggregate_score,
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

    def _estimate_compression_gain(
        self,
        *,
        proposal: MacroProposal,
        sample: OptimizerReplaySample,
        trace: CompressedTrace | None,
    ) -> float:
        opcode_pattern = tuple(str(item) for item in proposal.macro.opcode_pattern if str(item).strip())
        if not opcode_pattern:
            return 0.0
        opcodes = tuple(step.opcode for step in trace.operation_stream if step.opcode) if trace is not None else ()
        if not opcodes:
            baseline = sample.selected_candidate_score * 0.2
            if sample.candidate_trace_count > 1:
                baseline += 0.1
            return self._clamp_score(baseline)
        occurrence_count = self._count_pattern_occurrences(opcodes, opcode_pattern)
        if occurrence_count == 0:
            return 0.0
        density = (occurrence_count * len(opcode_pattern)) / max(1, len(opcodes))
        candidate_bonus = min(0.15, sample.candidate_trace_count / 20.0)
        score_bonus = sample.selected_candidate_score * 0.1
        return self._clamp_score(density + candidate_bonus + score_bonus)

    def _estimate_proof_hash_stability(
        self,
        *,
        proposal: MacroProposal,
        sample: OptimizerReplaySample,
    ) -> float:
        stability = 1.0 if sample.proof_hash_match else 0.55
        if proposal.validation_passed:
            stability += 0.05
        else:
            stability -= 0.25
        if proposal.proof_fingerprint:
            stability += 0.05
        if "proof_hash" in sample.failure_categories:
            stability -= 0.2
        if "provenance" in sample.failure_categories:
            stability -= 0.1
        return self._clamp_score(stability)

    def _estimate_critique_validity(self, sample: OptimizerReplaySample) -> float:
        adjudication = sample.final_adjudication.value
        if sample.is_valid or adjudication == "valid":
            score = 1.0
        elif adjudication == "degraded":
            score = 0.55
        else:
            score = 0.2
        score = max(score, sample.selected_candidate_score * 0.8)
        score = max(score, sample.provenance_coverage * 0.5)
        score -= min(0.3, len(sample.failure_categories) * 0.08)
        score -= min(0.15, len(sample.applied_repair_actions) * 0.03)
        return self._clamp_score(score)

    def _estimate_latency_ratio(
        self,
        *,
        proposal: MacroProposal,
        sample: OptimizerReplaySample,
        compression_gain: float,
    ) -> float:
        ratio = 1.0 - (compression_gain * 0.25)
        if sample.iterations >= 4:
            ratio -= 0.02
        if len(proposal.macro.opcode_pattern) >= 3:
            ratio -= 0.01
        if not proposal.validation_passed:
            ratio += 0.25
        if sample.applied_repair_actions:
            ratio += min(0.08, len(sample.applied_repair_actions) * 0.02)
        return round(max(0.65, ratio), 3)

    def _estimate_memory_ratio(
        self,
        *,
        proposal: MacroProposal,
        sample: OptimizerReplaySample,
        compression_gain: float,
    ) -> float:
        ratio = 1.0 - (compression_gain * 0.18)
        if sample.candidate_trace_count >= 3:
            ratio -= 0.02
        if sample.vram_usage_gb >= 1.0:
            ratio -= 0.01
        if not proposal.validation_passed:
            ratio += 0.2
        return round(max(0.7, ratio), 3)

    def _simulation_gate(
        self,
        *,
        proposal: MacroProposal,
        sample: OptimizerReplaySample,
        aggregate_score: float,
        latency_ratio: float,
        memory_ratio: float,
    ) -> tuple[bool, str]:
        settings = self.config.self_optimizer
        reasons: list[str] = []
        if not proposal.validation_passed:
            reasons.append("proposal_validation_failed")
        if aggregate_score < settings.minimum_simulation_score:
            reasons.append("aggregate_score_below_threshold")
        if latency_ratio > settings.max_latency_ratio:
            reasons.append("latency_ratio_exceeded")
        if memory_ratio > settings.max_memory_ratio:
            reasons.append("memory_ratio_exceeded")
        if not sample.proof_hash_match:
            reasons.append("sample_proof_hash_unstable")
        if sample.final_adjudication.value == "invalid":
            reasons.append("sample_invalid")
        return (not reasons, ",".join(reasons))

    def _apply_replay_scores(
        self,
        proposals: list[MacroProposal],
        evaluations: tuple[OptimizerReplayEvaluation, ...],
    ) -> list[MacroProposal]:
        if not evaluations:
            return proposals
        score_map: dict[str, list[float]] = {}
        for evaluation in evaluations:
            score_map.setdefault(evaluation.proposal_id, []).append(evaluation.aggregate_score)
        rescored: list[MacroProposal] = []
        for proposal in proposals:
            scores = score_map.get(proposal.proposal_id)
            if not scores:
                rescored.append(proposal)
                continue
            rescored.append(
                replace(
                    proposal,
                    simulation_score=round(sum(scores) / len(scores), 3),
                    approved=False,
                )
            )
        return rescored

    def _summarize_evaluations(
        self,
        evaluations: tuple[OptimizerReplayEvaluation, ...],
    ) -> list[dict[str, float | int | str]]:
        if not evaluations:
            return []
        grouped: dict[str, list[OptimizerReplayEvaluation]] = {}
        for evaluation in evaluations:
            grouped.setdefault(evaluation.proposal_id, []).append(evaluation)
        summary: list[dict[str, float | int | str]] = []
        for proposal_id, proposal_evaluations in sorted(grouped.items()):
            mean_score = sum(item.aggregate_score for item in proposal_evaluations) / len(proposal_evaluations)
            pass_rate = sum(1 for item in proposal_evaluations if item.accepted) / len(proposal_evaluations)
            summary.append(
                {
                    "proposal_id": proposal_id,
                    "mean_score": round(mean_score, 3),
                    "pass_rate": round(pass_rate, 3),
                    "sample_count": len(proposal_evaluations),
                }
            )
        return summary

    def _count_pattern_occurrences(
        self,
        sequence: tuple[str, ...],
        pattern: tuple[str, ...],
    ) -> int:
        if not sequence or not pattern or len(pattern) > len(sequence):
            return 0
        count = 0
        width = len(pattern)
        for index in range(0, len(sequence) - width + 1):
            if sequence[index : index + width] == pattern:
                count += 1
        return count

    def _ratio_score(self, ratio: float, maximum: float) -> float:
        if ratio <= 1.0:
            return 1.0
        if maximum <= 1.0:
            return 0.0
        if ratio >= maximum:
            return 0.0
        return self._clamp_score(1.0 - ((ratio - 1.0) / (maximum - 1.0)))

    def _clamp_score(self, value: float) -> float:
        return max(0.0, min(1.0, round(float(value), 3)))

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.run_cycle()
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                self.logger.exception("SelfOptimizer cycle failed: %s", exc)
            await asyncio.sleep(self.config.self_optimizer.cycle_interval_s)
