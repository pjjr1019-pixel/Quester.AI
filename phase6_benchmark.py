"""Repeatable benchmark harness for Phase 6 trace payload size and memory shape."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

from config import APP_CONFIG, AppConfig
from data_structures import EvidenceBundle, EvidenceItem, Plan, PlanStep, ResourceBudget, SourceType
from model_manager import ModelManager
from reasoner import ReasonerAgent


@dataclass(slots=True, frozen=True)
class TraceDensityScenario:
    """Single deterministic trace scenario for density benchmarking."""

    scenario_id: str
    evidence_count: int
    reasoner_passes: int
    retrieval_top_k: int
    macro_depth: int


@dataclass(slots=True, frozen=True)
class TraceDensityMeasurement:
    """Measured payload and structural sizes for one trace scenario."""

    scenario_id: str
    evidence_count: int
    reasoner_passes: int
    token_count: int
    operation_count: int
    entity_count: int
    activity_count: int
    context_frame_count: int
    decode_hint_count: int
    legacy_json_bytes: int
    ir_json_bytes: int
    legacy_memory_bytes: int
    ir_memory_bytes: int
    json_growth_ratio: float
    memory_growth_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "evidence_count": self.evidence_count,
            "reasoner_passes": self.reasoner_passes,
            "token_count": self.token_count,
            "operation_count": self.operation_count,
            "entity_count": self.entity_count,
            "activity_count": self.activity_count,
            "context_frame_count": self.context_frame_count,
            "decode_hint_count": self.decode_hint_count,
            "legacy_json_bytes": self.legacy_json_bytes,
            "ir_json_bytes": self.ir_json_bytes,
            "legacy_memory_bytes": self.legacy_memory_bytes,
            "ir_memory_bytes": self.ir_memory_bytes,
            "json_growth_ratio": self.json_growth_ratio,
            "memory_growth_ratio": self.memory_growth_ratio,
        }


DEFAULT_SCENARIOS: tuple[TraceDensityScenario, ...] = (
    TraceDensityScenario(
        scenario_id="baseline",
        evidence_count=1,
        reasoner_passes=1,
        retrieval_top_k=4,
        macro_depth=2,
    ),
    TraceDensityScenario(
        scenario_id="medium",
        evidence_count=4,
        reasoner_passes=2,
        retrieval_top_k=4,
        macro_depth=4,
    ),
    TraceDensityScenario(
        scenario_id="wide",
        evidence_count=6,
        reasoner_passes=3,
        retrieval_top_k=6,
        macro_depth=6,
    ),
)


def _benchmark_config(config: AppConfig) -> AppConfig:
    """Use deterministic stub-mode settings for the benchmark harness."""
    return replace(
        config,
        preflight=replace(
            config.preflight,
            flags=replace(
                config.preflight.flags,
                stub_mode=True,
                enable_self_optimizer=False,
                allow_web_fallback=False,
            ),
        ),
    )


def _build_plan(scenario: TraceDensityScenario) -> Plan:
    task_id = f"phase6-density-{scenario.scenario_id}"
    budget = ResourceBudget(
        retrieval_top_k=scenario.retrieval_top_k,
        max_web_queries=0,
        reasoner_passes=scenario.reasoner_passes,
        critic_passes=2,
        macro_depth=scenario.macro_depth,
    )
    return Plan(
        task_id=task_id,
        question=f"Summarize the verified evidence for scenario {scenario.scenario_id}.",
        steps=(
            PlanStep(step_id="step_1", description="Read the question"),
            PlanStep(step_id="step_2", description="Synthesize the evidence"),
        ),
        required_evidence=("local evidence",),
        success_criteria=("return a bounded synthesized answer",),
        budget=budget,
    )


def _build_evidence_bundle(plan: Plan, scenario: TraceDensityScenario) -> EvidenceBundle:
    local_results = tuple(
        EvidenceItem(
            id=f"{scenario.scenario_id}-ev-{index}",
            content=(
                f"Evidence item {index} for {scenario.scenario_id} "
                f"supports bounded local-first reasoning."
            ),
            source_type=SourceType.LOCAL,
            source_ref=f"local://{scenario.scenario_id}/{index}",
            score=max(0.1, 0.95 - (index * 0.05)),
            metadata={"scenario": scenario.scenario_id, "rank": index},
        )
        for index in range(1, scenario.evidence_count + 1)
    )
    return EvidenceBundle(
        task_id=plan.task_id,
        local_results=local_results,
        web_results=(),
        used_web_fallback=False,
    )


def _compact_json_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _deep_sizeof(value: Any, *, _seen_ids: set[int] | None = None) -> int:
    """Approximate recursive in-memory footprint for serialized payloads."""
    seen_ids = _seen_ids if _seen_ids is not None else set()
    object_id = id(value)
    if object_id in seen_ids:
        return 0
    seen_ids.add(object_id)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(
            _deep_sizeof(key, _seen_ids=seen_ids) + _deep_sizeof(item, _seen_ids=seen_ids)
            for key, item in value.items()
        )
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_deep_sizeof(item, _seen_ids=seen_ids) for item in value)
    return size


def _legacy_trace_payload(full_trace_payload: dict[str, Any]) -> dict[str, Any]:
    """Project the current trace back to the legacy pre-IR payload shape."""
    legacy_payload = dict(full_trace_payload)
    for field_name in (
        "ir_version",
        "canonical_graph",
        "operation_stream",
        "symbol_table_refs",
        "evidence_handles",
        "context_frames",
        "proof_hash",
        "decode_hints",
    ):
        legacy_payload.pop(field_name, None)
    return legacy_payload


async def run_trace_density_benchmark(
    *,
    config: AppConfig = APP_CONFIG,
    scenarios: Iterable[TraceDensityScenario] = DEFAULT_SCENARIOS,
) -> tuple[TraceDensityMeasurement, ...]:
    """Build deterministic stub traces and compare legacy vs IR payload size."""
    benchmark_config = _benchmark_config(config)
    model_manager = ModelManager(config=benchmark_config)
    reasoner = ReasonerAgent(model_manager=model_manager, config=benchmark_config)
    await model_manager.start()
    await reasoner.start()
    measurements: list[TraceDensityMeasurement] = []
    try:
        for scenario in scenarios:
            plan = _build_plan(scenario)
            evidence = _build_evidence_bundle(plan, scenario)
            trace = await reasoner.reason(plan, evidence, plan.budget)
            legacy_payload = _legacy_trace_payload(trace.to_dict())
            full_payload = trace.to_storage_dict()
            legacy_json_bytes = _compact_json_bytes(legacy_payload)
            ir_json_bytes = _compact_json_bytes(full_payload)
            legacy_memory_bytes = _deep_sizeof(legacy_payload)
            ir_memory_bytes = _deep_sizeof(full_payload)
            canonical_graph = trace.canonical_graph
            measurements.append(
                TraceDensityMeasurement(
                    scenario_id=scenario.scenario_id,
                    evidence_count=scenario.evidence_count,
                    reasoner_passes=scenario.reasoner_passes,
                    token_count=len(trace.tokens),
                    operation_count=len(trace.operation_stream),
                    entity_count=len(canonical_graph.entities) if canonical_graph is not None else 0,
                    activity_count=len(canonical_graph.activities) if canonical_graph is not None else 0,
                    context_frame_count=len(trace.context_frames),
                    decode_hint_count=len(trace.decode_hints),
                    legacy_json_bytes=legacy_json_bytes,
                    ir_json_bytes=ir_json_bytes,
                    legacy_memory_bytes=legacy_memory_bytes,
                    ir_memory_bytes=ir_memory_bytes,
                    json_growth_ratio=round(ir_json_bytes / legacy_json_bytes, 3),
                    memory_growth_ratio=round(ir_memory_bytes / legacy_memory_bytes, 3),
                )
            )
        return tuple(measurements)
    finally:
        await reasoner.stop()
        await model_manager.stop()


def format_trace_density_report(measurements: Iterable[TraceDensityMeasurement]) -> str:
    """Render a readable plain-text benchmark report."""
    rows = list(measurements)
    header = (
        "scenario  evidence  passes  legacy_json  ir_json  json_ratio  "
        "legacy_mem  ir_mem  mem_ratio  ops  entities"
    )
    body = [
        (
            f"{row.scenario_id:<8}  {row.evidence_count:>8}  {row.reasoner_passes:>6}  "
            f"{row.legacy_json_bytes:>11}  {row.ir_json_bytes:>7}  {row.json_growth_ratio:>10.3f}  "
            f"{row.legacy_memory_bytes:>10}  {row.ir_memory_bytes:>6}  {row.memory_growth_ratio:>9.3f}  "
            f"{row.operation_count:>3}  {row.entity_count:>8}"
        )
        for row in rows
    ]
    lines = [
        "Phase 6 Trace Density Benchmark",
        "",
        header,
        "-" * len(header),
        *body,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Phase 6 density benchmark harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of the plain-text report.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional file path to write the report to.",
    )
    args = parser.parse_args(argv)

    measurements = asyncio.run(run_trace_density_benchmark())
    if args.json:
        output = json.dumps([item.to_dict() for item in measurements], indent=2, sort_keys=True)
    else:
        output = format_trace_density_report(measurements)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
