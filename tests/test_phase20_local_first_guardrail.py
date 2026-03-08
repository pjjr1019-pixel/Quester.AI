"""Phase 20 local-first capability-foundation guardrails."""

from __future__ import annotations

import unittest

from capability_guardrails import (
    PHASE20_LOCAL_FIRST_GUARDRAIL,
    LocalFirstCapabilityPlan,
)


class Phase20LocalFirstGuardrailTests(unittest.TestCase):
    def test_additive_local_plan_is_allowed(self) -> None:
        plan = LocalFirstCapabilityPlan(
            plan_id="phase20_additive_desktop_foundation",
            summary="Add typed desktop helpers behind the current orchestrator without replacing local execution.",
            notes=("future desktop tools stay behind the current controller bridge",),
        )

        assessment = PHASE20_LOCAL_FIRST_GUARDRAIL.evaluate(plan)

        self.assertTrue(assessment.allowed)
        self.assertEqual(assessment.reasons, ())
        self.assertIn("Orchestrator.run_task", PHASE20_LOCAL_FIRST_GUARDRAIL.required_public_entrypoint)
        self.assertIn("orchestrator", PHASE20_LOCAL_FIRST_GUARDRAIL.required_runtime_components)

    def test_runtime_replacement_plan_is_rejected(self) -> None:
        plan = LocalFirstCapabilityPlan(
            plan_id="phase20_runtime_replacement",
            summary="Replace the current runtime with a separate desktop agent.",
            extends_existing_orchestrator=False,
            keeps_public_task_entrypoint=False,
            preserves_local_generation_embedding_base=False,
            preserves_local_storage_and_audit=False,
            local_helpers_are_optional=False,
        )

        assessment = PHASE20_LOCAL_FIRST_GUARDRAIL.evaluate(plan)

        self.assertFalse(assessment.allowed)
        self.assertIn("must_extend_existing_orchestrator", assessment.reasons)
        self.assertIn("must_keep_public_task_entrypoint", assessment.reasons)
        self.assertIn("must_preserve_generation_embedding_base", assessment.reasons)
        self.assertIn("must_preserve_local_storage_and_audit", assessment.reasons)
        self.assertIn("helpers_must_remain_opt_in", assessment.reasons)
        with self.assertRaises(ValueError):
            assessment.require_allowed()

    def test_primary_cloud_mode_without_local_fallback_is_rejected(self) -> None:
        plan = LocalFirstCapabilityPlan(
            plan_id="phase20_primary_cloud_helper",
            summary="Let a cloud desktop helper become the primary execution path.",
            local_execution_remains_primary=False,
            local_fallback_available=False,
            cloud_helper_mode="primary",
        )

        assessment = PHASE20_LOCAL_FIRST_GUARDRAIL.evaluate(plan)

        self.assertFalse(assessment.allowed)
        self.assertIn("local_execution_must_remain_primary", assessment.reasons)
        self.assertIn("helpers_require_local_fallback", assessment.reasons)
        self.assertIn("cloud_helpers_must_remain_auxiliary", assessment.reasons)


if __name__ == "__main__":
    unittest.main()
