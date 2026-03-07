"""Shared critique logic used by CriticAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import APP_CONFIG, AppConfig
from data_structures import (
    CompressionRuntimeSubset,
    CritiqueReport,
    CritiqueResult,
    ReasonerCriticHandoff,
)
from prompts import CRITIC_PROMPT

if TYPE_CHECKING:
    from model_manager import ModelManager
    from storage import StorageManager


class CritiqueService:
    """Validate a reasoning trace from a typed handoff."""

    output_contract = "critique_report_v1"
    handoff_contract = "reasoner_critic_handoff_v1"
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
        self._last_handoff: ReasonerCriticHandoff | None = None

    @property
    def last_runtime_subset(self) -> CompressionRuntimeSubset | None:
        return self._last_runtime_subset

    @property
    def last_handoff(self) -> ReasonerCriticHandoff | None:
        return self._last_handoff

    async def review(self, handoff: ReasonerCriticHandoff) -> CritiqueReport:
        self._last_handoff = handoff
        plan = handoff.plan
        evidence = handoff.evidence
        trace = handoff.trace
        budget = handoff.budget
        prompt = (
            f"{CRITIC_PROMPT}\n"
            f"Task: {plan.question}\n"
            f"TraceLength: {len(trace.tokens)}\n"
            f"CriticPasses: {budget.critic_passes}\n"
            f"OutputContract: {handoff.output_contract}"
        )
        model_notes = await self.model_manager.generate(prompt)
        has_evidence = bool(evidence.local_results) or bool(evidence.web_results)
        runtime_subset = await self._load_runtime_subset(plan.task_id, handoff)
        self._last_runtime_subset = runtime_subset
        issues: list[str] = []
        checks_run: list[str] = []
        if budget.critic_passes >= 1:
            checks_run.append("check.evidence_presence")
            if not has_evidence:
                issues.append("No evidence found in local or web sources.")
        if budget.critic_passes >= 2:
            checks_run.extend(
                (
                    "check.trace_nonempty",
                    "check.reasoning_pass_tokens",
                    "check.runtime_subset_alignment",
                )
            )
            if not trace.tokens or not trace.expanded_preview:
                issues.append("Reasoning trace is missing required projections.")
            expected_pass_tokens = {
                f"@reason_pass_{pass_index}" for pass_index in range(1, budget.reasoner_passes + 1)
            }
            if not expected_pass_tokens.issubset(set(trace.tokens)):
                issues.append("Reasoning trace does not include all budgeted reasoning passes.")
            issues.extend(self._validate_runtime_subset(handoff, runtime_subset))
        if budget.critic_passes >= 3:
            checks_run.extend(("check.plan_budget_alignment", "check.web_fallback_alignment"))
            if not any("Reasoning pass" in preview for preview in trace.expanded_preview):
                issues.append("Expanded reasoning preview does not expose pass-level debug information.")
            if evidence.web_results and not evidence.used_web_fallback:
                issues.append("Web evidence is present but used_web_fallback is false.")
        is_valid = len(issues) == 0
        return CritiqueReport(
            task_id=plan.task_id,
            is_valid=is_valid,
            issues=tuple(issues),
            fixed_trace=trace if is_valid else None,
            evidence_coverage=1.0 if has_evidence else 0.0,
            critic_notes=(
                f"{model_notes}\n"
                f"checks_run={','.join(checks_run)}\n"
                f"final_text_policy={handoff.final_text_policy}\n"
                f"repair_attempt_count={handoff.repair_attempt_count}\n"
                f"loaded_opcodes={','.join(opcode.opcode_name for opcode in runtime_subset.opcodes)}\n"
                f"loaded_macros={','.join(macro.macro_name for macro in runtime_subset.macros)}\n"
                f"loaded_decoders={','.join(decoder.decoder_name for decoder in runtime_subset.decoders)}\n"
                f"symbol_table_present={'yes' if runtime_subset.symbol_table is not None else 'no'}"
            ),
            result=CritiqueResult.VALID if is_valid else CritiqueResult.INVALID,
        )

    async def _load_runtime_subset(
        self,
        task_id: str,
        handoff: ReasonerCriticHandoff,
    ) -> CompressionRuntimeSubset:
        if self.storage is None:
            return CompressionRuntimeSubset(task_id=task_id)
        return await self.storage.load_active_compression_runtime(
            task_id,
            macro_names=handoff.required_macro_names,
            opcode_names=handoff.required_opcode_names,
            decoder_names=handoff.required_decoder_names,
        )

    def _validate_runtime_subset(
        self,
        handoff: ReasonerCriticHandoff,
        runtime_subset: CompressionRuntimeSubset,
    ) -> list[str]:
        trace = handoff.trace
        issues: list[str] = []
        loaded_opcode_names = {opcode.opcode_name for opcode in runtime_subset.opcodes}
        loaded_macro_names = {macro.macro_name for macro in runtime_subset.macros}
        loaded_decoder_names = {decoder.decoder_name for decoder in runtime_subset.decoders}
        missing_opcodes = sorted(set(handoff.required_opcode_names) - loaded_opcode_names)
        if missing_opcodes:
            issues.append(f"Active opcode subset is missing required entries: {', '.join(missing_opcodes)}.")
        missing_macros = sorted(set(handoff.required_macro_names) - loaded_macro_names)
        if missing_macros:
            issues.append(f"Active macro subset is missing required entries: {', '.join(missing_macros)}.")
        missing_decoders = sorted(set(handoff.required_decoder_names) - loaded_decoder_names)
        if missing_decoders:
            issues.append(
                f"Active decoder subset is missing required entries: {', '.join(missing_decoders)}."
            )
        if trace.proof_hash != handoff.proof_hash:
            issues.append("Trace proof hash drifted between reasoner and critic handoff.")
        if trace.symbol_table_refs:
            snapshot = runtime_subset.symbol_table
            if snapshot is None:
                issues.append("Task-scoped symbol table is missing for the current trace.")
            else:
                available_refs = set(snapshot.symbols)
                required_refs = set(trace.symbol_table_refs)
                required_refs.update(
                    argument
                    for step in trace.operation_stream
                    for argument in step.args
                    if argument.startswith("sym_")
                )
                required_refs.update(
                    step.output_ref
                    for step in trace.operation_stream
                    if step.output_ref.startswith("sym_")
                )
                missing_refs = sorted(ref for ref in required_refs if ref not in available_refs)
                if missing_refs:
                    issues.append(
                        f"Task-scoped symbol table is missing required refs: {', '.join(missing_refs)}."
                    )
        return issues
