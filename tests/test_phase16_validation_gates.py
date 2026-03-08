"""Phase 16 validation-gate regression tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from validation_gates import (
    AGENT_LAYER_GATE,
    DATA_LAYER_GATE,
    DASHBOARD_GATE,
    MACRO_LAYER_GATE,
    MODEL_LAYER_GATE,
    OPTIMIZER_GATE,
    ORCHESTRATOR_GATE,
    STORAGE_LAYER_GATE,
    SUBSYSTEM_VALIDATION_GATES,
    get_validation_gate,
)


class Phase16ValidationGateTests(unittest.TestCase):
    """Lock the explicit Phase 16 gate definitions into repo state."""

    def test_defined_gates_require_local_and_compatibility_commands(self) -> None:
        for subsystem, gate in SUBSYSTEM_VALIDATION_GATES.items():
            self.assertEqual(gate.subsystem, subsystem)
            self.assertTrue(gate.local_commands, msg=f"{subsystem} gate is missing local commands")
            self.assertTrue(
                gate.compatibility_commands,
                msg=f"{subsystem} gate is missing compatibility commands",
            )

    def test_model_gate_targets_runtime_readiness_and_compatibility(self) -> None:
        local_paths = {
            path
            for command in MODEL_LAYER_GATE.local_commands
            for path in command.required_paths
        }
        compatibility_paths = {
            path
            for command in MODEL_LAYER_GATE.compatibility_commands
            for path in command.required_paths
        }

        self.assertIn("tests/test_phase3_runtime.py", local_paths)
        self.assertIn("tests/test_phase12_preflight.py", local_paths)
        self.assertIn("tests/test_compatibility_contract.py", compatibility_paths)

        criteria = " ".join(MODEL_LAYER_GATE.done_criteria).lower()
        self.assertIn("semaphore", criteria)
        self.assertIn("generation", criteria)
        self.assertIn("embedding", criteria)

    def test_data_gate_targets_contract_round_trip_validation_and_compatibility(self) -> None:
        local_paths = {
            path
            for command in DATA_LAYER_GATE.local_commands
            for path in command.required_paths
        }
        compatibility_paths = {
            path
            for command in DATA_LAYER_GATE.compatibility_commands
            for path in command.required_paths
        }

        self.assertIn("tests/test_phase2_contracts.py", local_paths)
        self.assertIn("data_structures.py", local_paths)
        self.assertIn("tests/test_compatibility_contract.py", compatibility_paths)

        criteria = " ".join(DATA_LAYER_GATE.done_criteria).lower()
        self.assertIn("round-trip", criteria)
        self.assertIn("malformed", criteria)
        self.assertIn("serialized", criteria)

    def test_storage_gate_targets_sqlite_jsonl_restart_and_compatibility(self) -> None:
        local_paths = {
            path
            for command in STORAGE_LAYER_GATE.local_commands
            for path in command.required_paths
        }
        compatibility_paths = {
            path
            for command in STORAGE_LAYER_GATE.compatibility_commands
            for path in command.required_paths
        }

        self.assertIn("tests/test_phase5_persistence.py", local_paths)
        self.assertIn("storage.py", local_paths)
        self.assertIn("tests/test_compatibility_contract.py", compatibility_paths)

        criteria = " ".join(STORAGE_LAYER_GATE.done_criteria).lower()
        self.assertIn("sqlite", criteria)
        self.assertIn("jsonl", criteria)
        self.assertIn("schema", criteria)

    def test_agent_gate_targets_unit_acceptance_and_compatibility(self) -> None:
        local_paths = {
            path
            for command in AGENT_LAYER_GATE.local_commands
            for path in command.required_paths
        }
        compatibility_paths = {
            path
            for command in AGENT_LAYER_GATE.compatibility_commands
            for path in command.required_paths
        }

        self.assertIn("tests/test_phase12_agent_units.py", local_paths)
        self.assertIn("tests/test_phase12_end_to_end.py", local_paths)
        self.assertIn("tests/test_phase12_preflight.py", local_paths)
        self.assertIn("tests/test_compatibility_contract.py", compatibility_paths)

        criteria = " ".join(AGENT_LAYER_GATE.done_criteria).lower()
        self.assertIn("planner", criteria)
        self.assertIn("reasoner", criteria)
        self.assertIn("stub-mode", criteria)

    def test_macro_gate_targets_expansion_round_trip_and_compatibility(self) -> None:
        local_paths = {
            path
            for command in MACRO_LAYER_GATE.local_commands
            for path in command.required_paths
        }
        compatibility_paths = {
            path
            for command in MACRO_LAYER_GATE.compatibility_commands
            for path in command.required_paths
        }

        self.assertIn("tests/test_phase6_macro_engine.py", local_paths)
        self.assertIn("macro_engine.py", local_paths)
        self.assertIn("tests/test_compatibility_contract.py", compatibility_paths)

        criteria = " ".join(MACRO_LAYER_GATE.done_criteria).lower()
        self.assertIn("nested", criteria)
        self.assertIn("recursion", criteria)
        self.assertIn("proof", criteria)

    def test_orchestrator_gate_targets_stage_order_and_status_events(self) -> None:
        local_paths = {
            path
            for command in ORCHESTRATOR_GATE.local_commands
            for path in command.required_paths
        }

        self.assertIn("tests/test_phase16_subsystem_gates.py", local_paths)
        self.assertIn("tests/test_phase12_end_to_end.py", local_paths)
        self.assertIn("tests/test_phase13_async_safety.py", local_paths)

        criteria = " ".join(ORCHESTRATOR_GATE.done_criteria).lower()
        self.assertIn("stage", criteria)
        self.assertIn("status", criteria)
        self.assertIn("cancellation", criteria)

    def test_optimizer_gate_targets_simulation_validation_and_compatibility(self) -> None:
        local_paths = {
            path
            for command in OPTIMIZER_GATE.local_commands
            for path in command.required_paths
        }
        compatibility_paths = {
            path
            for command in OPTIMIZER_GATE.compatibility_commands
            for path in command.required_paths
        }

        self.assertIn("tests/test_phase12_resource_optimizer.py", local_paths)
        self.assertIn("tests/test_phase5_persistence.py", local_paths)
        self.assertIn("self_optimizer.py", local_paths)
        self.assertIn("tests/test_compatibility_contract.py", compatibility_paths)

        criteria = " ".join(OPTIMIZER_GATE.done_criteria).lower()
        self.assertIn("simulation", criteria)
        self.assertIn("validation", criteria)
        self.assertIn("blocked", criteria)

    def test_dashboard_gate_targets_typed_state_and_async_safety(self) -> None:
        local_paths = {
            path
            for command in DASHBOARD_GATE.local_commands
            for path in command.required_paths
        }

        self.assertIn("tests/test_phase16_subsystem_gates.py", local_paths)
        self.assertIn("tests/test_phase12_gui_acceptance.py", local_paths)
        self.assertIn("tests/test_phase13_async_safety.py", local_paths)
        self.assertIn("tests/test_phase7_boundaries.py", local_paths)

        criteria = " ".join(DASHBOARD_GATE.done_criteria).lower()
        self.assertIn("typed", criteria)
        self.assertIn("degraded", criteria)
        self.assertIn("responsive", criteria)

    def test_all_gates_reference_existing_repo_files(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        for gate in SUBSYSTEM_VALIDATION_GATES.values():
            self.assertEqual(gate.missing_paths(repo_root), (), msg=f"Missing paths for {gate.subsystem}")

    def test_get_validation_gate_returns_model_gate(self) -> None:
        self.assertIs(get_validation_gate("data"), DATA_LAYER_GATE)
        self.assertIs(get_validation_gate("storage"), STORAGE_LAYER_GATE)
        self.assertIs(get_validation_gate("model"), MODEL_LAYER_GATE)
        self.assertIs(get_validation_gate("macro"), MACRO_LAYER_GATE)
        self.assertIs(get_validation_gate("agent"), AGENT_LAYER_GATE)
        self.assertIs(get_validation_gate("orchestrator"), ORCHESTRATOR_GATE)
        self.assertIs(get_validation_gate("optimizer"), OPTIMIZER_GATE)
        self.assertIs(get_validation_gate("dashboard"), DASHBOARD_GATE)


if __name__ == "__main__":
    unittest.main()
