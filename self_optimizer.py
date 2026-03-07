"""Background self-optimizer scaffold."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import replace

from config import APP_CONFIG, AppConfig
from data_structures import (
    CompressedTrace,
    Macro,
    MacroProposal,
    OptimizerReplayEvaluation,
    OptimizerReplaySample,
    PerformanceMetric,
    ReasoningLog,
)
from macro_engine import MacroEngine
from retrieval import stable_hash
from storage import StorageManager
from utils import cancel_task


class SelfOptimizer:
    """Runs non-blocking optimization cycles while the app is live."""

    def __init__(self, storage: StorageManager, config: AppConfig = APP_CONFIG):
        self.storage = storage
        self.config = config
        self.logger = logging.getLogger("quester.self_optimizer")
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._stop_event = asyncio.Event()
        self._last_evaluations: tuple[OptimizerReplayEvaluation, ...] = ()

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
        engine = MacroEngine()
        reasoning_logs = await self._load_reasoning_logs()
        traces = await self.storage.list_reasoning_traces()
        metrics = await self.storage.list_performance_metrics()
        replay_samples = await self._load_replay_samples()
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
        await self.storage.log_event(
            "self_optimizer.cycle",
            {
                "proposal_count": len(proposals),
                "proposal_ids": [proposal.proposal_id for proposal in proposals],
                "reasoning_log_count": len(reasoning_logs),
                "trace_count": len(traces),
                "metric_count": len(metrics),
                "replay_sample_count": len(replay_samples),
                "evaluation_count": len(evaluations),
                "accepted_evaluation_count": sum(1 for evaluation in evaluations if evaluation.accepted),
                "proposal_scores": self._summarize_evaluations(evaluations),
            },
        )
        return proposals

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
