"""Phase 16 project-wide validation-gate regression tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from validation_gates import (
    PRE_RELEASE_SMOKE_GATE,
    PROJECT_COMPLETION_GATE,
    PROJECT_VALIDATION_GATES,
    RELEASE_GATE,
    RESOURCE_GATE,
    SUBSYSTEM_VALIDATION_GATES,
    get_project_validation_gate,
)


class Phase16ProjectValidationGateTests(unittest.TestCase):
    """Lock the explicit project-wide Phase 16 gate definitions into repo state."""

    def test_defined_project_gates_require_commands(self) -> None:
        for gate_id, gate in PROJECT_VALIDATION_GATES.items():
            self.assertEqual(gate.gate_id, gate_id)
            self.assertTrue(gate.commands, msg=f"{gate_id} gate is missing commands")

    def test_resource_gate_targets_thresholds_runtime_bounds_and_long_horizon(self) -> None:
        paths = {path for command in RESOURCE_GATE.commands for path in command.required_paths}

        self.assertIn("tests/test_phase12_acceptance_thresholds.py", paths)
        self.assertIn("tests/test_phase3_runtime.py", paths)
        self.assertIn("tests/test_phase12_resource_optimizer.py", paths)
        self.assertIn("tests/test_phase17_long_horizon.py", paths)
        self.assertEqual(RESOURCE_GATE.required_subsystems, ("model", "optimizer", "orchestrator"))

        criteria = " ".join(RESOURCE_GATE.done_criteria).lower()
        self.assertIn("6 gb / 8 gb", criteria)
        self.assertIn("bounded", criteria)
        self.assertIn("long-horizon", criteria)

    def test_pre_release_smoke_gate_targets_stub_and_real_launch_paths(self) -> None:
        paths = {path for command in PRE_RELEASE_SMOKE_GATE.commands for path in command.required_paths}

        self.assertIn("tests/test_phase12_end_to_end.py", paths)
        self.assertIn("tests/test_phase12_packaged_smoke.py", paths)
        self.assertIn("tests/test_phase12_preflight.py", paths)
        self.assertEqual(
            PRE_RELEASE_SMOKE_GATE.required_subsystems,
            ("model", "agent", "orchestrator", "dashboard"),
        )

        criteria = " ".join(PRE_RELEASE_SMOKE_GATE.done_criteria).lower()
        self.assertIn("stub-mode", criteria)
        self.assertIn("real-backend", criteria)
        self.assertIn("packaged", criteria)

    def test_release_gate_requires_resource_and_pre_release_smoke(self) -> None:
        paths = {path for command in RELEASE_GATE.commands for path in command.required_paths}

        self.assertIn("tests/test_phase12_acceptance_thresholds.py", paths)
        self.assertIn("tests/test_phase12_translation_exports.py", paths)
        self.assertIn("tests/test_phase12_gui_acceptance.py", paths)
        self.assertIn("tests/test_phase12_preflight.py", paths)
        self.assertEqual(RELEASE_GATE.required_project_gates, ("resource", "pre_release_smoke"))
        self.assertEqual(RELEASE_GATE.required_subsystems, tuple(SUBSYSTEM_VALIDATION_GATES))

        criteria = " ".join(RELEASE_GATE.done_criteria).lower()
        self.assertIn("packaged windows path", criteria)
        self.assertIn("acceptance checklist", criteria)
        self.assertIn("actionable", criteria)

    def test_project_completion_gate_requires_all_subsystems_release_and_compatibility(self) -> None:
        paths = {path for command in PROJECT_COMPLETION_GATE.commands for path in command.required_paths}

        self.assertIn("tests/test_compatibility_contract.py", paths)
        self.assertIn("tests/test_phase12_packaged_smoke.py", paths)
        self.assertIn("tests/test_phase12_gui_acceptance.py", paths)
        self.assertEqual(PROJECT_COMPLETION_GATE.required_subsystems, tuple(SUBSYSTEM_VALIDATION_GATES))
        self.assertEqual(
            PROJECT_COMPLETION_GATE.required_project_gates,
            ("resource", "pre_release_smoke", "release"),
        )

        criteria = " ".join(PROJECT_COMPLETION_GATE.done_criteria).lower()
        self.assertIn("compatibility contract", criteria)
        self.assertIn("subsystem gates", criteria)
        self.assertIn("acceptance checklist", criteria)

    def test_project_gates_reference_existing_repo_files_and_known_dependencies(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        known_gate_ids = set(PROJECT_VALIDATION_GATES)
        known_subsystems = set(SUBSYSTEM_VALIDATION_GATES)

        for gate in PROJECT_VALIDATION_GATES.values():
            self.assertEqual(gate.missing_paths(repo_root), (), msg=f"Missing paths for {gate.gate_id}")
            self.assertTrue(set(gate.required_subsystems).issubset(known_subsystems))
            self.assertTrue(set(gate.required_project_gates).issubset(known_gate_ids))

    def test_get_project_validation_gate_returns_expected_gates(self) -> None:
        self.assertIs(get_project_validation_gate("resource"), RESOURCE_GATE)
        self.assertIs(get_project_validation_gate("pre_release_smoke"), PRE_RELEASE_SMOKE_GATE)
        self.assertIs(get_project_validation_gate("release"), RELEASE_GATE)
        self.assertIs(get_project_validation_gate("project_completion"), PROJECT_COMPLETION_GATE)


if __name__ == "__main__":
    unittest.main()
