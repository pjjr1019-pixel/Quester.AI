"""Bounded local Coding Mode services, routing, sandboxing, and practice flows."""

from __future__ import annotations

import ast
import asyncio
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from config import APP_CONFIG, AppConfig
from data_structures import (
    CodeQualityReport,
    CodeSpecialistResult,
    CodingPattern,
    CodingPatternTier,
    CodingPatternValidation,
    CodingRole,
    CodingTaskArtifact,
    CodingTaskRequest,
    CodingTaskResult,
    CodingTaskType,
    ModelRegistration,
    ModelResourceClass,
    ModelRole,
    ModelRouteDecision,
    PracticeSessionResult,
    UserSettingsProfile,
    utc_now,
)
from model_manager import ModelManager
from retrieval import stable_hash
from storage import StorageManager

_CodingEventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

_PRACTICE_TASKS: tuple[dict[str, str], ...] = (
    {
        "task_id": "practice-factorial",
        "task_type": CodingTaskType.PRACTICE.value,
        "prompt": "Implement factorial(n) for non-negative integers and raise ValueError for negatives.",
        "language": "python",
        "source_text": (
            "def factorial(n: int) -> int:\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    result = 1\n"
            "    for value in range(2, n + 1):\n"
            "        result *= value\n"
            "    return result\n"
        ),
        "tests_text": (
            "import unittest\n"
            "from solution import factorial\n\n"
            "class FactorialTests(unittest.TestCase):\n"
            "    def test_base_cases(self):\n"
            "        self.assertEqual(factorial(0), 1)\n"
            "        self.assertEqual(factorial(1), 1)\n\n"
            "    def test_regular_cases(self):\n"
            "        self.assertEqual(factorial(5), 120)\n"
            "        self.assertEqual(factorial(7), 5040)\n\n"
            "    def test_negative(self):\n"
            "        with self.assertRaises(ValueError):\n"
            "            factorial(-1)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
    },
    {
        "task_id": "practice-slugify",
        "task_type": CodingTaskType.PRACTICE.value,
        "prompt": "Implement slugify(text) that lowercases text, trims whitespace, and joins alphanumeric words with hyphens.",
        "language": "python",
        "source_text": (
            "import re\n\n"
            "def slugify(text: str) -> str:\n"
            "    normalized = re.sub(r'[^a-zA-Z0-9]+', ' ', text).strip().lower()\n"
            "    if not normalized:\n"
            "        return ''\n"
            "    return '-'.join(normalized.split())\n"
        ),
        "tests_text": (
            "import unittest\n"
            "from solution import slugify\n\n"
            "class SlugifyTests(unittest.TestCase):\n"
            "    def test_words(self):\n"
            "        self.assertEqual(slugify('Hello World'), 'hello-world')\n"
            "        self.assertEqual(slugify(' local   AI  shell '), 'local-ai-shell')\n\n"
            "    def test_symbols(self):\n"
            "        self.assertEqual(slugify('A/B Testing!'), 'a-b-testing')\n"
            "        self.assertEqual(slugify('***'), '')\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
    },
)


class SandboxRunner:
    """Execute bounded local code checks without using an unrestricted shell."""

    _DANGEROUS_PATTERNS: tuple[str, ...] = (
        "os.system",
        "subprocess.",
        "socket.",
        "ctypes.",
        "winreg.",
        "eval(",
        "exec(",
        "__import__(",
        "shutil.rmtree",
    )

    def __init__(self, config: AppConfig = APP_CONFIG):
        self.config = config

    async def evaluate_submission(
        self,
        *,
        source_text: str,
        tests_text: str = "",
        language: str = "python",
    ) -> tuple[CodeQualityReport, tuple[CodingTaskArtifact, ...]]:
        """Run bounded syntax, lint, complexity, security, and test checks."""
        return await asyncio.to_thread(
            self._evaluate_submission_sync,
            source_text=source_text,
            tests_text=tests_text,
            language=language,
        )

    def _evaluate_submission_sync(
        self,
        *,
        source_text: str,
        tests_text: str = "",
        language: str = "python",
    ) -> tuple[CodeQualityReport, tuple[CodingTaskArtifact, ...]]:
        clipped_source = self._clip_text(source_text)
        clipped_tests = self._clip_text(tests_text)
        findings: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, float] = {}
        tests_passed = False
        regression_passed = False
        critique_passed = False
        syntax_ok = True

        lint_passed, lint_findings, lint_metrics = self._lint_checks(clipped_source)
        findings.extend(lint_findings)
        metrics.update(lint_metrics)

        complexity_passed, complexity_findings, complexity_metrics = self._complexity_checks(clipped_source)
        findings.extend(complexity_findings)
        metrics.update(complexity_metrics)

        security_passed, security_findings = self._security_checks(clipped_source)
        findings.extend(security_findings)

        maintainability_passed, maintainability_findings = self._maintainability_checks(clipped_source)
        findings.extend(maintainability_findings)

        if language.lower() == "python":
            try:
                compile(clipped_source, "<solution>", "exec")
            except SyntaxError as exc:
                syntax_ok = False
                lint_passed = False
                findings.append(f"syntax_error:{exc.msg}")
                warnings.append("syntax_error")
            if clipped_tests:
                try:
                    compile(clipped_tests, "<tests>", "exec")
                except SyntaxError as exc:
                    findings.append(f"test_syntax_error:{exc.msg}")
                    warnings.append("test_syntax_error")
        else:
            warnings.append("non_python_execution_skipped")

        if language.lower() == "python" and syntax_ok and clipped_tests and security_passed:
            tests_passed, test_findings, test_metrics = self._run_python_tests(
                source_text=clipped_source,
                tests_text=clipped_tests,
            )
            findings.extend(test_findings)
            metrics.update(test_metrics)
            regression_passed = tests_passed
        elif syntax_ok and not clipped_tests:
            findings.append("tests_missing_for_regression_gate")

        quality_score = self._quality_score(
            tests_passed=tests_passed,
            lint_passed=lint_passed,
            complexity_passed=complexity_passed,
            security_passed=security_passed,
            maintainability_passed=maintainability_passed,
            critique_passed=critique_passed,
            regression_passed=regression_passed,
        )
        report = CodeQualityReport(
            tests_passed=tests_passed,
            lint_passed=lint_passed and syntax_ok,
            complexity_passed=complexity_passed,
            security_passed=security_passed,
            maintainability_passed=maintainability_passed,
            critique_passed=critique_passed,
            regression_passed=regression_passed,
            overall_passed=False,
            quality_score=quality_score,
            findings=tuple(dict.fromkeys(findings)),
            warnings=tuple(dict.fromkeys(warnings)),
            metrics=metrics,
        )
        artifacts = (
            CodingTaskArtifact(
                artifact_id=f"artifact:{stable_hash(clipped_source)[:12]}",
                artifact_type="code",
                title="Generated Code",
                language=language,
                path="sandbox:solution.py" if language.lower() == "python" else "",
                content_preview=clipped_source[:400],
                metadata={"line_count": len(clipped_source.splitlines())},
            ),
            CodingTaskArtifact(
                artifact_id=f"artifact:{stable_hash(clipped_tests or 'no-tests')[:12]}",
                artifact_type="tests" if clipped_tests else "report",
                title="Generated Tests" if clipped_tests else "Sandbox Summary",
                language=language,
                path="sandbox:test_solution.py" if clipped_tests and language.lower() == "python" else "",
                content_preview=(clipped_tests[:400] if clipped_tests else "\n".join(report.findings[:6])),
                metadata={"line_count": len(clipped_tests.splitlines()) if clipped_tests else 0},
            ),
        )
        return report, artifacts

    def _clip_text(self, text: str) -> str:
        max_chars = max(1, int(self.config.coding_mode.max_source_chars))
        max_lines = max(1, int(self.config.coding_mode.max_source_lines))
        clipped = str(text or "")[:max_chars]
        return "\n".join(clipped.splitlines()[:max_lines])

    def _lint_checks(self, source_text: str) -> tuple[bool, list[str], dict[str, float]]:
        lines = source_text.splitlines() or [""]
        trailing_whitespace = sum(1 for line in lines if line.rstrip() != line)
        tabs = sum(1 for line in lines if "\t" in line)
        max_line_length = max((len(line) for line in lines), default=0)
        findings: list[str] = []
        if trailing_whitespace:
            findings.append("lint:trailing_whitespace")
        if tabs:
            findings.append("lint:tabs_detected")
        if max_line_length > 120:
            findings.append("lint:line_too_long")
        return (
            trailing_whitespace == 0 and tabs == 0 and max_line_length <= 120,
            findings,
            {
                "line_count": float(len(lines)),
                "max_line_length": float(max_line_length),
                "trailing_whitespace_count": float(trailing_whitespace),
                "tab_line_count": float(tabs),
            },
        )

    def _complexity_checks(self, source_text: str) -> tuple[bool, list[str], dict[str, float]]:
        try:
            tree = ast.parse(source_text or "\n")
        except SyntaxError:
            return False, ["complexity:parse_failed"], {}
        function_count = 0
        branch_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_count += 1
            if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match)):
                branch_count += 1
        findings: list[str] = []
        if function_count > 12:
            findings.append("complexity:function_count_high")
        if branch_count > 24:
            findings.append("complexity:branch_count_high")
        return (
            function_count <= 12 and branch_count <= 24,
            findings,
            {
                "function_count": float(function_count),
                "branch_count": float(branch_count),
            },
        )

    def _security_checks(self, source_text: str) -> tuple[bool, list[str]]:
        findings = [
            f"security:{pattern.replace('.', '_').replace('(', '').replace(')', '')}"
            for pattern in self._DANGEROUS_PATTERNS
            if pattern in source_text
        ]
        return not findings, findings

    def _maintainability_checks(self, source_text: str) -> tuple[bool, list[str]]:
        todo_count = len(re.findall(r"TODO|FIXME|XXX", source_text))
        duplicate_blank_blocks = len(re.findall(r"\n{3,}", source_text))
        findings: list[str] = []
        if todo_count:
            findings.append("maintainability:todo_markers")
        if duplicate_blank_blocks:
            findings.append("maintainability:spacing_noise")
        return todo_count == 0 and duplicate_blank_blocks == 0, findings

    def _run_python_tests(
        self,
        *,
        source_text: str,
        tests_text: str,
    ) -> tuple[bool, list[str], dict[str, float]]:
        findings: list[str] = []
        with tempfile.TemporaryDirectory(prefix="quester-coding-") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            (temp_dir / "solution.py").write_text(source_text, encoding="utf-8")
            (temp_dir / "test_solution.py").write_text(tests_text, encoding="utf-8")
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=float(self.config.coding_mode.sandbox_timeout_s),
                    check=False,
                    shell=False,
                )
            except subprocess.TimeoutExpired:
                return False, ["tests:timeout"], {"test_duration_timeout": float(self.config.coding_mode.sandbox_timeout_s)}
        if completed.returncode != 0:
            findings.append("tests:failed")
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if stderr:
            findings.append("tests:stderr")
        return (
            completed.returncode == 0,
            findings,
            {
                "test_return_code": float(completed.returncode),
                "test_stdout_chars": float(len(stdout)),
                "test_stderr_chars": float(len(stderr)),
            },
        )

    def _quality_score(
        self,
        *,
        tests_passed: bool,
        lint_passed: bool,
        complexity_passed: bool,
        security_passed: bool,
        maintainability_passed: bool,
        critique_passed: bool,
        regression_passed: bool,
    ) -> float:
        weighted = (
            (0.24 if tests_passed else 0.0)
            + (0.12 if lint_passed else 0.0)
            + (0.10 if complexity_passed else 0.0)
            + (0.18 if security_passed else 0.0)
            + (0.10 if maintainability_passed else 0.0)
            + (0.14 if critique_passed else 0.0)
            + (0.12 if regression_passed else 0.0)
        )
        return max(0.0, min(1.0, weighted))


class CodingRouter:
    """Resolve Coding Mode roles onto the existing local code-specialist registry."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def route_request(
        self,
        request: CodingTaskRequest,
        *,
        user_settings: UserSettingsProfile | None = None,
    ) -> dict[CodingRole, ModelRouteDecision]:
        preferred = {}
        if user_settings is not None:
            preferred = dict(user_settings.coding.get("preferred_models_by_role", {}))
        registrations = self.model_manager.list_registered_models(role=ModelRole.CODE_SPECIALIST)
        health = self.model_manager.health_snapshot()
        routes: dict[CodingRole, ModelRouteDecision] = {}
        for role in self.roles_for_task_type(request.task_type):
            selected = self._select_registration(
                registrations,
                role=role,
                preferred_registration=str(preferred.get(role.value, "")),
            )
            if selected is None:
                routes[role] = ModelRouteDecision(
                    requested_role=ModelRole.CODE_SPECIALIST,
                    capability=f"code:{role.value}",
                    allowed=False,
                    fallback_reason="no_coding_registration",
                    active_heavy_roles=tuple(health.active_heavy_roles),
                    heavy_slot_limit=int(health.heavy_slot_limit),
                )
                continue
            routes[role] = ModelRouteDecision(
                requested_role=ModelRole.CODE_SPECIALIST,
                selected_registration_id=selected.registration_id,
                selected_backend=selected.backend,
                selected_model_identifier=selected.model_identifier,
                resource_class=selected.resource_class,
                capability=f"code:{role.value}",
                allowed=bool(selected.enabled and not selected.missing_dependencies),
                fallback_reason=(
                    "role_disabled"
                    if not selected.enabled
                    else "missing_dependencies"
                    if selected.missing_dependencies
                    else ""
                ),
                active_heavy_roles=tuple(health.active_heavy_roles),
                heavy_slot_limit=int(health.heavy_slot_limit),
                metadata={"coding_role": role.value},
            )
        return routes

    def _select_registration(
        self,
        registrations: Sequence[ModelRegistration],
        *,
        role: CodingRole,
        preferred_registration: str,
    ) -> ModelRegistration | None:
        if preferred_registration:
            for registration in registrations:
                if preferred_registration in {
                    registration.registration_id,
                    f"{registration.backend}:{registration.model_identifier}",
                }:
                    return registration
        for registration in registrations:
            coding_roles = {
                str(item)
                for item in registration.metadata.get("coding_roles", ())
            }
            supported = set(registration.supported_capabilities)
            if role.value in coding_roles or f"code:{role.value}" in supported:
                return registration
        for registration in registrations:
            if registration.enabled and not registration.missing_dependencies:
                return registration
        return registrations[0] if registrations else None

    @staticmethod
    def roles_for_task_type(task_type: CodingTaskType) -> tuple[CodingRole, ...]:
        mapping = {
            CodingTaskType.FEATURE_GENERATION: (
                CodingRole.PLANNER,
                CodingRole.GENERATOR,
                CodingRole.TEST_WRITER,
                CodingRole.REVIEWER,
                CodingRole.SUMMARIZER,
            ),
            CodingTaskType.BUG_FIXING: (
                CodingRole.PLANNER,
                CodingRole.DEBUGGER,
                CodingRole.GENERATOR,
                CodingRole.TEST_WRITER,
                CodingRole.REVIEWER,
                CodingRole.SUMMARIZER,
            ),
            CodingTaskType.REFACTORING: (
                CodingRole.PLANNER,
                CodingRole.REFACTORER,
                CodingRole.TEST_WRITER,
                CodingRole.REVIEWER,
                CodingRole.SUMMARIZER,
            ),
            CodingTaskType.TEST_GENERATION: (
                CodingRole.PLANNER,
                CodingRole.TEST_WRITER,
                CodingRole.REVIEWER,
                CodingRole.SUMMARIZER,
            ),
            CodingTaskType.CODE_REVIEW: (
                CodingRole.REVIEWER,
                CodingRole.SUMMARIZER,
            ),
            CodingTaskType.EXPLANATION: (
                CodingRole.SUMMARIZER,
                CodingRole.REVIEWER,
            ),
            CodingTaskType.PROJECT_SCAFFOLDING: (
                CodingRole.PLANNER,
                CodingRole.GENERATOR,
                CodingRole.REVIEWER,
                CodingRole.SUMMARIZER,
            ),
            CodingTaskType.ARCHITECTURE_PLANNING: (
                CodingRole.PLANNER,
                CodingRole.SUMMARIZER,
                CodingRole.REVIEWER,
            ),
            CodingTaskType.PRACTICE: (
                CodingRole.PLANNER,
                CodingRole.GENERATOR,
                CodingRole.TEST_WRITER,
                CodingRole.REVIEWER,
            ),
        }
        return mapping.get(task_type, (CodingRole.REVIEWER, CodingRole.SUMMARIZER))


class CodingModeService:
    """Bounded Coding Mode orchestration layered over the existing local runtime."""

    def __init__(
        self,
        *,
        model_manager: ModelManager,
        storage: StorageManager,
        config: AppConfig = APP_CONFIG,
    ):
        self.model_manager = model_manager
        self.storage = storage
        self.config = config
        self.logger = logging.getLogger("quester.coding_mode")
        self.sandbox = SandboxRunner(config=config)
        self.router = CodingRouter(model_manager=model_manager)

    async def run_task(
        self,
        request: CodingTaskRequest | dict[str, Any],
        *,
        user_settings: UserSettingsProfile | None = None,
        event_callback: _CodingEventCallback | None = None,
    ) -> CodingTaskResult:
        """Execute one bounded Coding Mode request."""
        normalized = request if isinstance(request, CodingTaskRequest) else CodingTaskRequest.from_dict(request)
        if not normalized.request_id:
            normalized = replace(
                normalized,
                request_id=(
                    f"coding-"
                    f"{stable_hash((normalized.prompt or normalized.source_path or normalized.source_text) + utc_now().isoformat())[:12]}"
                ),
            )
        routes = self.router.route_request(normalized, user_settings=user_settings)
        role_assignments = {
            role.value: decision.selected_registration_id or decision.fallback_reason or "unrouted"
            for role, decision in routes.items()
        }
        route_summary = tuple(
            f"{role.value}:{decision.selected_model_identifier or decision.fallback_reason or 'unavailable'}"
            for role, decision in routes.items()
        )
        await self._emit(
            event_callback,
            "coding.planning",
            {
                "request_id": normalized.request_id,
                "task_type": normalized.task_type.value,
                "language": normalized.language,
                "prompt": normalized.prompt,
                "route_summary": list(route_summary),
            },
        )

        source_text = normalized.source_text
        tests_text = normalized.tests_text
        summary_parts: list[str] = []
        warnings: list[str] = []
        artifacts: tuple[CodingTaskArtifact, ...] = ()

        if not source_text and normalized.source_path:
            source_path = Path(normalized.source_path)
            try:
                source_text = source_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                source_text = source_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                warnings.append(f"source_read_failed:{exc}")

        generated = False
        if not source_text and normalized.task_type in {
            CodingTaskType.FEATURE_GENERATION,
            CodingTaskType.BUG_FIXING,
            CodingTaskType.REFACTORING,
            CodingTaskType.TEST_GENERATION,
            CodingTaskType.PROJECT_SCAFFOLDING,
            CodingTaskType.ARCHITECTURE_PLANNING,
        }:
            await self._emit(
                event_callback,
                "coding.generating",
                {
                    "request_id": normalized.request_id,
                    "task_type": normalized.task_type.value,
                    "language": normalized.language,
                },
            )
            source_text, tests_text = self._generate_stub_solution(normalized)
            generated = True
            summary_parts.append("Prepared a bounded local coding scaffold.")

        review_output = CodeSpecialistResult()
        if source_text or normalized.source_path:
            await self._emit(
                event_callback,
                "coding.reviewing",
                {
                    "request_id": normalized.request_id,
                    "task_type": normalized.task_type.value,
                    "language": normalized.language,
                },
            )
            review_output = await self._review_code(
                normalized,
                source_text=source_text,
            )
            if review_output.summary:
                summary_parts.append(review_output.summary)
            warnings.extend(review_output.warnings)

        quality_report = CodeQualityReport()
        if source_text and normalized.language.lower() == "python":
            await self._emit(
                event_callback,
                "coding.testing",
                {
                    "request_id": normalized.request_id,
                    "task_type": normalized.task_type.value,
                    "language": normalized.language,
                    "has_tests": bool(tests_text),
                },
            )
            sandbox_report, sandbox_artifacts = await self.sandbox.evaluate_submission(
                source_text=source_text,
                tests_text=tests_text,
                language=normalized.language,
            )
            quality_report = replace(
                sandbox_report,
                critique_passed=review_output.status == "analyzed",
            )
            quality_report = replace(
                quality_report,
                overall_passed=bool(
                    quality_report.tests_passed
                    and quality_report.lint_passed
                    and quality_report.complexity_passed
                    and quality_report.security_passed
                    and quality_report.maintainability_passed
                    and quality_report.critique_passed
                    and quality_report.regression_passed
                ),
                quality_score=self._updated_quality_score(quality_report),
            )
            artifacts = sandbox_artifacts
            if not quality_report.tests_passed and tests_text:
                await self._emit(
                    event_callback,
                    "coding.debugging",
                    {
                        "request_id": normalized.request_id,
                        "task_type": normalized.task_type.value,
                        "language": normalized.language,
                        "regression_detected": True,
                    },
                )
            if not quality_report.regression_passed and tests_text:
                await self._emit(
                    event_callback,
                    "coding.regression_detected",
                    {
                        "request_id": normalized.request_id,
                        "task_type": normalized.task_type.value,
                        "language": normalized.language,
                    },
                )
        elif source_text:
            summary_parts.append("Skipped sandbox execution because non-Python execution is not enabled.")
            warnings.append("sandbox_execution_skipped")

        await self._emit(
            event_callback,
            "coding.indexing",
            {
                "request_id": normalized.request_id,
                "task_type": normalized.task_type.value,
                "language": normalized.language,
            },
        )
        patterns = self._build_patterns(
            request=normalized,
            source_text=source_text,
            tests_text=tests_text,
            quality_report=quality_report,
            review_output=review_output,
            generated=generated,
        )
        if patterns:
            await self.storage.save_coding_patterns(patterns)

        result = CodingTaskResult(
            request_id=normalized.request_id,
            task_type=normalized.task_type,
            status="completed",
            active_phase="indexing",
            prompt=normalized.prompt,
            summary=" ".join(part for part in summary_parts if part).strip() or "Completed a bounded coding-mode task.",
            language=normalized.language,
            framework=normalized.framework,
            source_scope=normalized.source_scope,
            role_assignments=role_assignments,
            route_summary=route_summary,
            artifacts=artifacts,
            quality_report=quality_report,
            verified_patterns=tuple(pattern.pattern_id for pattern in patterns if pattern.tier == CodingPatternTier.VERIFIED),
            candidate_patterns=tuple(pattern.pattern_id for pattern in patterns if pattern.tier == CodingPatternTier.CANDIDATE),
            rejected_patterns=tuple(pattern.pattern_id for pattern in patterns if pattern.tier == CodingPatternTier.REJECTED),
            warnings=tuple(dict.fromkeys(str(item) for item in warnings if str(item).strip())),
            practice_session_id=str(normalized.metadata.get("practice_session_id", "")),
        )
        await self.storage.record_coding_task_result(result)
        await self._emit(
            event_callback,
            "coding.completed",
            {
                "request_id": result.request_id,
                "task_type": result.task_type.value,
                "language": result.language,
                "status": result.status,
                "summary": result.summary,
                "quality_score": result.quality_report.quality_score,
                "verified_pattern_count": len(result.verified_patterns),
                "candidate_pattern_count": len(result.candidate_patterns),
                "rejected_pattern_count": len(result.rejected_patterns),
            },
        )
        return result

    async def run_idle_practice_cycle(
        self,
        *,
        user_settings: UserSettingsProfile | None = None,
        event_callback: _CodingEventCallback | None = None,
    ) -> PracticeSessionResult:
        """Run one bounded Coding Dojo practice session."""
        existing = await self.storage.list_coding_practice_sessions(limit=self.config.coding_mode.practice_history_limit)
        selected = _PRACTICE_TASKS[len(existing) % len(_PRACTICE_TASKS)]
        session_id = f"practice-{stable_hash(selected['task_id'] + utc_now().isoformat())[:12]}"
        await self._emit(
            event_callback,
            "coding.practicing",
            {
                "session_id": session_id,
                "task_id": selected["task_id"],
                "prompt": selected["prompt"],
                "language": selected["language"],
            },
        )
        request = CodingTaskRequest(
            request_id=f"{session_id}-task",
            task_type=CodingTaskType.PRACTICE,
            prompt=selected["prompt"],
            language=selected["language"],
            source_scope="practice",
            source_text=selected["source_text"],
            tests_text=selected["tests_text"],
            idle_practice=True,
            metadata={
                "practice_session_id": session_id,
                "practice_task_id": selected["task_id"],
                "source": "coding_dojo",
            },
        )
        result = await self.run_task(
            request,
            user_settings=user_settings,
            event_callback=event_callback,
        )
        practice = PracticeSessionResult(
            session_id=session_id,
            status="completed",
            task_type=CodingTaskType.PRACTICE,
            prompt=selected["prompt"],
            language=selected["language"],
            summary=result.summary or "Completed one bounded coding practice task.",
            quality_score=result.quality_report.quality_score,
            validated_patterns=result.verified_patterns,
            rejected_patterns=result.rejected_patterns,
            warnings=result.warnings,
            task_result=result,
            started_at=request.created_at,
            completed_at=utc_now(),
        )
        await self.storage.record_coding_practice_session(practice)
        await self._emit(
            event_callback,
            "coding.practice_completed",
            {
                "session_id": practice.session_id,
                "task_id": selected["task_id"],
                "summary": practice.summary,
                "quality_score": practice.quality_score,
                "validated_pattern_count": len(practice.validated_patterns),
                "rejected_pattern_count": len(practice.rejected_patterns),
            },
        )
        return practice

    async def _review_code(
        self,
        request: CodingTaskRequest,
        *,
        source_text: str,
    ) -> CodeSpecialistResult:
        request_text = request.prompt.strip() or "Review the code for bounded maintenance, verification, and reuse insights."
        if request.source_path:
            return await self.model_manager.analyze_code_file(
                Path(request.source_path),
                request_text=request_text,
            )
        return await self.model_manager.analyze_code(
            text=source_text,
            request_text=request_text,
            source_scope=request.source_scope or "snippet",
        )

    def _build_patterns(
        self,
        *,
        request: CodingTaskRequest,
        source_text: str,
        tests_text: str,
        quality_report: CodeQualityReport,
        review_output: CodeSpecialistResult,
        generated: bool,
    ) -> tuple[CodingPattern, ...]:
        title_base = request.prompt.strip() or review_output.summary or request.task_type.value.replace("_", " ")
        summary = review_output.summary or "Bounded local coding pattern captured from Coding Mode."
        tier = self._pattern_tier_for_report(quality_report)
        validation = CodingPatternValidation(
            validation_id=f"validation-{stable_hash(title_base + request.request_id)[:12]}",
            checks_passed=tuple(
                check_name
                for check_name, passed in (
                    ("tests", quality_report.tests_passed),
                    ("lint", quality_report.lint_passed),
                    ("complexity", quality_report.complexity_passed),
                    ("security", quality_report.security_passed),
                    ("maintainability", quality_report.maintainability_passed),
                    ("critique", quality_report.critique_passed),
                    ("regression", quality_report.regression_passed),
                )
                if passed
            ),
            checks_failed=tuple(
                check_name
                for check_name, passed in (
                    ("tests", quality_report.tests_passed),
                    ("lint", quality_report.lint_passed),
                    ("complexity", quality_report.complexity_passed),
                    ("security", quality_report.security_passed),
                    ("maintainability", quality_report.maintainability_passed),
                    ("critique", quality_report.critique_passed),
                    ("regression", quality_report.regression_passed),
                )
                if not passed
            ),
            reviewer_summary=summary,
            quality_report=quality_report,
        )
        main_pattern = CodingPattern(
            pattern_id=f"coding-pattern-{stable_hash(request.request_id + title_base)[:12]}",
            title=title_base[:120],
            summary=summary,
            tier=tier,
            category=(
                "test_strategy"
                if request.task_type == CodingTaskType.TEST_GENERATION
                else "architecture_template"
                if request.task_type == CodingTaskType.ARCHITECTURE_PLANNING
                else "bug_fix"
                if request.task_type == CodingTaskType.BUG_FIXING
                else "refactor_strategy"
                if request.task_type == CodingTaskType.REFACTORING
                else "good_practice"
            ),
            language=request.language,
            framework=request.framework,
            task_type=request.task_type,
            source=("coding_practice" if request.idle_practice else "coding_mode"),
            quality_score=quality_report.quality_score,
            reuse_count=0,
            code_snippet=source_text[: min(len(source_text), self.config.coding_mode.max_source_chars // 4)],
            metadata={
                "generated": generated,
                "practice_session_id": str(request.metadata.get("practice_session_id", "")),
                "practice_task_id": str(request.metadata.get("practice_task_id", "")),
                "warnings": tuple(review_output.warnings),
            },
            validation_history=(validation,),
        )
        patterns: list[CodingPattern] = [main_pattern]
        if tests_text and tier == CodingPatternTier.VERIFIED:
            patterns.append(
                CodingPattern(
                    pattern_id=f"coding-pattern-{stable_hash(request.request_id + 'tests')[:12]}",
                    title=f"Tests for {title_base[:90]}",
                    summary="Verified reusable test pattern captured from a bounded coding task.",
                    tier=CodingPatternTier.VERIFIED,
                    category="test_strategy",
                    language=request.language,
                    framework=request.framework,
                    task_type=request.task_type,
                    source=("coding_practice" if request.idle_practice else "coding_mode"),
                    quality_score=quality_report.quality_score,
                    reuse_count=0,
                    code_snippet=tests_text[: min(len(tests_text), self.config.coding_mode.max_source_chars // 5)],
                    metadata={
                        "pattern_kind": "tests",
                        "practice_session_id": str(request.metadata.get("practice_session_id", "")),
                    },
                    validation_history=(validation,),
                )
            )
        limits = {
            CodingPatternTier.VERIFIED: self.config.coding_mode.max_verified_patterns,
            CodingPatternTier.CANDIDATE: self.config.coding_mode.max_candidate_patterns,
            CodingPatternTier.REJECTED: self.config.coding_mode.max_rejected_patterns,
        }
        filtered = [pattern for pattern in patterns if pattern.tier == tier]
        return tuple(filtered[: limits[tier]])

    def _pattern_tier_for_report(self, report: CodeQualityReport) -> CodingPatternTier:
        if report.overall_passed:
            return CodingPatternTier.VERIFIED
        if report.quality_score >= 0.45:
            return CodingPatternTier.CANDIDATE
        return CodingPatternTier.REJECTED

    def _updated_quality_score(self, report: CodeQualityReport) -> float:
        weighted = (
            (0.24 if report.tests_passed else 0.0)
            + (0.12 if report.lint_passed else 0.0)
            + (0.10 if report.complexity_passed else 0.0)
            + (0.18 if report.security_passed else 0.0)
            + (0.10 if report.maintainability_passed else 0.0)
            + (0.14 if report.critique_passed else 0.0)
            + (0.12 if report.regression_passed else 0.0)
        )
        return max(0.0, min(1.0, weighted))

    def _generate_stub_solution(self, request: CodingTaskRequest) -> tuple[str, str]:
        function_name = self._derive_function_name(request.prompt or request.task_type.value)
        if request.task_type == CodingTaskType.TEST_GENERATION:
            source_text = (
                f"def {function_name}(value):\n"
                "    return value\n"
            )
            tests_text = (
                "import unittest\n"
                f"from solution import {function_name}\n\n"
                "class GeneratedTests(unittest.TestCase):\n"
                "    def test_identity(self):\n"
                f"        self.assertEqual({function_name}(3), 3)\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            )
            return source_text, tests_text
        if request.task_type == CodingTaskType.ARCHITECTURE_PLANNING:
            return (
                (
                    "class PlannedComponent:\n"
                    "    \"\"\"Bounded architecture sketch generated by local Coding Mode.\"\"\"\n\n"
                    "    def __init__(self, dependency):\n"
                    "        self._dependency = dependency\n\n"
                    "    def execute(self, payload):\n"
                    "        return self._dependency(payload)\n"
                ),
                "",
            )
        source_text = (
            f"def {function_name}(value):\n"
            '    """Bounded local coding scaffold."""\n'
            "    return value\n"
        )
        tests_text = (
            "import unittest\n"
            f"from solution import {function_name}\n\n"
            "class ScaffoldTests(unittest.TestCase):\n"
            "    def test_round_trip(self):\n"
            f"        self.assertEqual({function_name}(5), 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        return source_text, tests_text

    def _derive_function_name(self, prompt: str) -> str:
        tokens = re.findall(r"[a-zA-Z0-9]+", prompt.lower())
        if not tokens:
            return "generated_task"
        filtered = [token for token in tokens if token not in {"the", "a", "an", "and", "for", "with"}]
        collapsed = "_".join(filtered[:4] or tokens[:4])
        return collapsed[:40] or "generated_task"

    async def _emit(
        self,
        callback: _CodingEventCallback | None,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        if callback is None:
            return
        await callback(stage, dict(payload))
