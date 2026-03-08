"""Phase 13 implementation-guardrail tests."""

from __future__ import annotations

import inspect
from pathlib import Path
import unittest

from compression_service import CompressionService
from critique_service import CritiqueService
from dashboard import DashboardService
from orchestrator import Orchestrator
from planner_service import PlannerService
from reasoning_service import ReasoningService
from research_service import ResearchService
from structured_generation import StructuredGenerationService


class _QueuedModelManager:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        _ = max_tokens
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("No queued responses remaining.")
        return self._responses.pop(0)


class Phase13TaskTemplateTests(unittest.TestCase):
    def test_implementation_task_template_covers_phase13_guardrails(self) -> None:
        template_path = Path("IMPLEMENTATION_TASK_TEMPLATE.md")
        template_text = template_path.read_text(encoding="utf-8")
        readme_text = Path("README.md").read_text(encoding="utf-8")

        self.assertIn("Standing Phase 13 Checklist", template_text)
        self.assertIn("Review the current phase before coding", template_text)
        self.assertIn("Keep the change scoped to one subsystem", template_text)
        self.assertIn("Target files:", template_text)
        self.assertIn("Required contracts or schemas:", template_text)
        self.assertIn("Primary tests to add or update:", template_text)
        self.assertIn("IMPLEMENTATION_TASK_TEMPLATE.md", readme_text)


class Phase13StructuredGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_generation_stops_after_one_repair_attempt(self) -> None:
        manager = _QueuedModelManager(["not json", "still not json", "{\"value\": 3}"])
        service = StructuredGenerationService(model_manager=manager)
        schema = {
            "type": "object",
            "required": ["value"],
            "properties": {"value": {"type": "integer"}},
            "additionalProperties": False,
        }

        result = await service.decode_json_output(
            prompt="Return a value",
            schema=schema,
            parser=lambda payload: payload,
            fallback_factory=lambda error_message: {"fallback": True, "error": error_message or "unknown"},
            max_tokens=32,
        )

        self.assertEqual(len(manager.prompts), 2)
        self.assertTrue(result.used_repair)
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.value["fallback"], True)
        self.assertIn("Return JSON only", manager.prompts[0])
        self.assertIn("Repair the previous response", manager.prompts[1])


class Phase13DocstringTests(unittest.TestCase):
    def test_selected_public_runtime_methods_have_docstrings(self) -> None:
        targets: dict[type[object], tuple[str, ...]] = {
            DashboardService: (
                "dropped_events",
                "ui_running",
                "app_state_snapshot",
                "attach_controller",
                "apply_user_settings",
                "request_task_submission",
                "request_settings_save",
                "request_action",
                "start",
                "stop",
                "publish_event",
            ),
            Orchestrator: (
                "start",
                "stop",
                "run_task",
                "run_pipeline",
                "build_packaged_launch_report",
                "export_packaged_support_bundle",
            ),
            StructuredGenerationService: ("decode_json_output",),
            PlannerService: ("plan",),
            ResearchService: ("reset", "research"),
            ReasoningService: ("last_runtime_subset", "last_handoff", "reason", "build_critic_handoff"),
            CritiqueService: ("last_runtime_subset", "last_handoff", "review"),
            CompressionService: ("propose",),
        }

        for cls, member_names in targets.items():
            for member_name in member_names:
                member = getattr(cls, member_name)
                self.assertTrue(
                    inspect.getdoc(member),
                    f"{cls.__name__}.{member_name} is missing a public docstring.",
                )


if __name__ == "__main__":
    unittest.main()
