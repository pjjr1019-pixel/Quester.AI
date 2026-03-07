"""Shared proposal-building logic used by CompressorAgent."""

from __future__ import annotations

from collections import Counter

from config import APP_CONFIG, AppConfig
from data_structures import CompressedTrace, Macro, MacroProposal, ReasoningLog
from macro_engine import MacroEngine
from model_manager import ModelManager
from prompts import COMPRESSOR_PROMPT
from retrieval import stable_hash


class CompressionService:
    """Suggest macro candidates while keeping CompressorAgent thin."""

    output_contract = "macro_proposal_list_v1"
    implementation_mode = "deterministic_stub"

    def __init__(self, model_manager: ModelManager, config: AppConfig = APP_CONFIG):
        self.model_manager = model_manager
        self.config = config

    async def propose(
        self,
        trace: CompressedTrace,
        logs: list[ReasoningLog],
    ) -> list[MacroProposal]:
        chain = trace.expanded_preview or trace.tokens
        _ = logs
        prompt = f"{COMPRESSOR_PROMPT}\nChainLength: {len(chain)}\nOutputContract: {self.output_contract}"
        await self.model_manager.generate(prompt)
        engine = MacroEngine()
        suggestions: list[MacroProposal] = []
        seen_signatures: set[tuple[str, ...]] = set()
        for span in (3, 2):
            motifs = Counter(
                tuple(chain[index : index + span])
                for index in range(0, max(0, len(chain) - span + 1))
            )
            for motif, count in motifs.items():
                if count <= 1 or motif in seen_signatures:
                    continue
                seen_signatures.add(motif)
                sanitized = stable_hash("|".join(motif))[:10]
                proposal = MacroProposal(
                    proposal_id=f"{trace.task_id}:motif:{sanitized}",
                    macro=Macro(
                        macro_name=f"motif_{sanitized}",
                        expansion=motif,
                        version=1,
                        opcode_pattern=tuple(engine._opcode_for_step(step) for step in motif),
                        invariants=(
                            "deterministic_round_trip",
                            "provenance_preserving",
                            "uncertainty_preserving",
                        ),
                        semantic_kind="motif_macro",
                    ),
                    reason=f"Repeated {span}-step motif observed {count} times in expanded trace.",
                    examples=(" | ".join(motif),),
                    simulation_score=min(1.0, (count * span) / 6.0),
                    approved=False,
                )
                suggestions.append(engine.validate_macro_proposal(proposal))
        if suggestions:
            return suggestions

        token_counts = Counter(trace.tokens)
        for token, count in token_counts.items():
            if count <= 1:
                continue
            sanitized = token.lstrip("@") or "macro_token"
            proposal = MacroProposal(
                proposal_id=f"{trace.task_id}:{sanitized}",
                macro=Macro(
                    macro_name=sanitized,
                    expansion=(token,),
                    version=1,
                    opcode_pattern=(engine._opcode_for_step(token),),
                    invariants=(
                        "deterministic_round_trip",
                        "provenance_preserving",
                        "uncertainty_preserving",
                    ),
                    semantic_kind="token_macro",
                ),
                reason=f"Token '{token}' repeated {count} times in compressed trace.",
                examples=(token,),
                simulation_score=min(1.0, count / 3.0),
                approved=False,
            )
            suggestions.append(engine.validate_macro_proposal(proposal))
        return suggestions
