# Dead Code Log

Audit date: March 7, 2026

Scope:
- Current local repository state only.
- High-confidence dead code only.
- An item is listed here only if it has no inbound repo call sites or it is an unused import with no runtime or test effect.

Method:
- Searched the repo with `rg` for direct call/import sites.
- Cross-checked with a small AST pass for top-level defs and uniquely named methods.
- Used `pyflakes` where available to confirm unused imports.

## Confirmed Dead Runtime Code

1. `coerce_plan` in `data_structures.py:2185`
   - No imports or call sites anywhere in the repo.
   - The surrounding coercion helpers are used selectively; this one is not.

2. `coerce_evidence_bundle` in `data_structures.py:2192`
   - No imports or call sites anywhere in the repo.

3. `coerce_compression_runtime_subset` in `data_structures.py:2229`
   - No imports or call sites anywhere in the repo.

4. `coerce_research_reasoner_handoff` in `data_structures.py:2245`
   - No imports or call sites anywhere in the repo.

5. `coerce_reasoner_critic_handoff` in `data_structures.py:2254`
   - No imports or call sites anywhere in the repo.

6. `coerce_critique_report` in `data_structures.py:2270`
   - No imports or call sites anywhere in the repo.

7. `ReasoningService.build_critic_handoff` in `reasoning_service.py:115`
   - No repo call sites.
   - The live path builds `ReasonerCriticHandoff` directly in `orchestrator.py`, so this wrapper currently has no use.

8. `SourceDocumentRepository.get_by_source_ref` in `storage.py:504`
   - No repo call sites.
   - No `StorageManager` surface delegates to it.

9. `StorageManager.record_optimizer_replay_sample` in `storage.py:2141`
   - No repo call sites.
   - Replay samples are currently written through `record_task_result(...)`, not through this explicit wrapper.

## Confirmed Dead Imports

10. Unused `Iterable` import in `storage.py:9`
   - Confirmed by `pyflakes`.
   - The module uses `Awaitable`, `Callable`, and `Sequence`, but not `Iterable`.

11. Unused `PerformanceMetric` import in `tests/test_phase7_boundaries.py:22`
   - Confirmed by `pyflakes`.
   - Test-only dead code; no effect on runtime behavior.

12. Unused `ReasoningLog` import in `tests/test_phase7_boundaries.py:22`
   - Confirmed by `pyflakes`.
   - Test-only dead code; no effect on runtime behavior.

## Checked And Not Marked Dead

- `AppOrchestrator`, `run_pipeline`, and `run_once` in `orchestrator.py`
  - Compatibility or CLI-entrypoint role.
  - `run_pipeline(...)` is exercised by compatibility tests.

- `phase6_benchmark.py`
  - Manual benchmark entrypoint plus direct test coverage in `tests/test_phase6_density.py`.

- Handoff schema helpers in `agent_schema.py`
  - Used by contract tests even if they are not on the live runtime hot path.

- Storage/export helpers such as `export_compression_lexicon(...)`, `export_trace_debug_view(...)`, `get_task_result(...)`, and `count_chunks(...)`
  - Used in tests and/or explicit debug/export flows.

## Summary

Confirmed dead code count: 12 items

Breakdown:
- Runtime functions/methods: 9
- Dead imports: 3
- Whole dead modules found: 0
